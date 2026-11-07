#!/usr/bin/env python3
"""Convolution reverb on the NPU: a 65536-tap impulse response (1.4 s at 48 kHz), 2048 taps per core.

  ../env.sh python fun/reverb.py in.wav                 offline: NPU vs numpy FFT convolution, writes in.reverb.wav
  ../env.sh python fun/reverb.py in.wav --play          live: block by block through the NPU into the speakers
  ../env.sh python fun/reverb.py in.wav --ir hall.wav   use a real impulse response (first channel, resampled? no: same rate)

Direct-form FIR, int16 Q15 samples and taps, 32-bit partial sums. Core k owns taps [kT, (k+1)T) of the IR;
per block of B outputs it gets a window of B+T-1 input samples plus its (reversed) tap segment, multiplies
32 outputs at a time against each tap (one unaligned load + one mac per tap), and returns B int32 partials.
The host sums the 32 partials (exact, int64) and scales back to int16. The IR is normalised to sum|h| <= 0.9
so no partial can overflow 32 bits. Work: L*fs MACs per second per channel = 3.1 GMAC/s at 48 kHz; one core
at 32 MAC/cycle does 55 GMAC/s, so this is >100x real time on a single channel. No gathers, no FFT, just MACs.
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4
B = 2048                  # output samples per block per dispatch
T = 2048                  # taps per core
CORES = 32
L = CORES * T             # IR length
WIN = B + T - 1           # input window per core
OBJ = WIN + 1 + T         # per-core input object: window, one pad, reversed taps (keeps hr 64-byte aligned: 4096+2048)

KSRC = f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define B {B}
#define T {T}
#define WIN {WIN}
extern "C" {{
/* obj = [window (WIN) | pad | hr (T)]; out[n] = sum_u hr[u] * window[n + u] */
void fir(int16_t *__restrict obj, int32_t *__restrict out) {{
    const int16_t *__restrict w = obj;
    const int16_t *__restrict hr = obj + WIN + 1;
    for (int n = 0; n < B; n += 32) {{
        aie::accum<acc32, 32> acc = aie::mul(aie::load_unaligned_v<32>(w + n), hr[0]);
        for (int u = 1; u < T; u++)
            acc = aie::mac(acc, aie::load_unaligned_v<32>(w + n + u), hr[u]);
        aie::store_v(out + n, acc.to_vector<int32_t>(0));
    }}
}}
}}
"""


KSRC_SHIFT = f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define B {B}
#define T {T}
#define WIN {WIN}
extern "C" {{
/* aligned loads only: the window for tap u is shuffle_down_fill(lo, hi, u%32) of two aligned vectors */
void fir(int16_t *__restrict obj, int32_t *__restrict out) {{
    const int16_t *__restrict w = obj;
    const int16_t *__restrict hr = obj + WIN + 1;
    for (int n = 0; n < B; n += 32) {{
        aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
        for (int u0 = 0; u0 < T; u0 += 32) {{
            aie::vector<int16_t, 32> lo = aie::load_v<32>(w + n + u0);
            aie::vector<int16_t, 32> hi = aie::load_v<32>(w + n + u0 + 32);
            aie::vector<int16_t, 32> hv = aie::load_v<32>(hr + u0);
            acc = aie::mac(acc, lo, (int16_t)hv.get(0));
            for (int j = 1; j < 32; j++)
                acc = aie::mac(acc, aie::shuffle_down_fill(lo, hi, j), (int16_t)hv.get(j));
        }}
        aie::store_v(out + n, acc.to_vector<int32_t>(0));
    }}
}}
}}
"""

KSRC_SLIDING = f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define B {B}
#define T {T}
#define WIN {WIN}
extern "C" {{
/* hardware FIR primitive: 32 outputs x 8 taps per sliding_mac, data window of 64 lanes */
void fir(int16_t *__restrict obj, int32_t *__restrict out) {{
    const int16_t *__restrict w = obj;
    const int16_t *__restrict hr = obj + WIN + 1;
    for (int n = 0; n < B; n += 32) {{
        aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
        for (int u0 = 0; u0 < T; u0 += 32) {{
            aie::vector<int16_t, 64> d = aie::concat(aie::load_v<32>(w + n + u0), aie::load_v<32>(w + n + u0 + 32));
            aie::vector<int16_t, 32> hv = aie::load_v<32>(hr + u0);
            acc = aie::sliding_mac<32, 8>(acc, hv, 0, d, 0);
            acc = aie::sliding_mac<32, 8>(acc, hv, 8, d, 8);
            acc = aie::sliding_mac<32, 8>(acc, hv, 16, d, 16);
            acc = aie::sliding_mac<32, 8>(acc, hv, 24, d, 24);
        }}
        aie::store_v(out + n, acc.to_vector<int32_t>(0));
    }}
}}
}}
"""

SOURCES = {"naive": KSRC, "shift": KSRC_SHIFT, "sliding": KSRC_SLIDING}


@iron.jit
def fir_npu(inp: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS
    in_ty = np.ndarray[(cores * OBJ,), np.dtype[np.int16]]
    out_ty = np.ndarray[(cores * B,), np.dtype[np.int32]]
    col_in_ty = np.ndarray[(ROWS * OBJ,), np.dtype[np.int16]]
    col_out_ty = np.ndarray[(ROWS * B,), np.dtype[np.int32]]
    obj_ty = np.ndarray[(OBJ,), np.dtype[np.int16]]
    part_ty = np.ndarray[(B,), np.dtype[np.int32]]
    k = ExternalFunction("fir", source_string=SOURCES[variant], arg_types=[obj_ty, part_ty])
    col_in = [ObjectFifo(col_in_ty, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(col_out_ty, name=f"out{c}", depth=2) for c in range(cols)]
    ci, co = [], []
    for c in range(cols):
        ci += col_in[c].cons().split([i * OBJ for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[obj_ty] * ROWS, names=[f"ci{c}_{i}" for i in range(ROWS)])
        co += col_out[c].prod().join([i * B for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[part_ty] * ROWS, names=[f"co{c}_{i}" for i in range(ROWS)])

    def core(fi, fo, k):
        x = fi.acquire(1); y = fo.acquire(1)
        k(x, y)
        fo.release(1); fi.release(1)

    workers = [Worker(core, [ci[j].cons(), co[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    tin = TensorAccessPattern.full((cores * OBJ,)).partition(cols)
    tout = TensorAccessPattern.full((cores * B,)).partition(cols)
    ip, oc = [f.prod() for f in col_in], [f.cons() for f in col_out]

    def seq(inp, out, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            ip[c].fill(inp, tin[c], group=tg)
        for c in range(cols):
            oc[c].drain(out, tout[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [in_ty, out_ty, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class Reverb:
    """One channel. feed(block of B float samples) -> B float samples of wet signal."""

    def __init__(self, ir, variant="sliding"):
        self.variant = variant
        ir = np.asarray(ir, np.float64)[:L]
        ir = np.pad(ir, (0, L - len(ir)))
        ir *= 0.9 / np.abs(ir).sum()                       # no 32-bit partial can overflow
        self.h = np.clip(np.round(ir * 32768), -32768, 32767).astype(np.int16)
        self.cols = iron.get_current_device().cols
        self.hist = np.zeros(L - 1 + B, np.int16)         # L-1 past samples then the current block
        self.packed = np.zeros((CORES, OBJ), np.int16)
        for k in range(CORES):                            # core k: taps [kT, (k+1)T) reversed, window starts at (31-k)T
            self.packed[k, WIN + 1:] = self.h[k * T:(k + 1) * T][::-1]
        # staging buffers shared by every channel: alternating between buffer sets per call costs ~7 ms a call
        if not hasattr(Reverb, "_in"):
            Reverb._in = iron.zeros((CORES * OBJ,), dtype=np.int16, device="npu")
            Reverb._out = iron.zeros((CORES * B,), dtype=np.int32, device="npu")
        self.npu_ms = 0.0

    def feed(self, x):
        xi = np.clip(np.round(np.asarray(x, np.float64) * 32767), -32768, 32767).astype(np.int16)
        self.hist[:L - 1] = self.hist[B:]
        self.hist[L - 1:] = xi
        for k in range(CORES):
            s = (CORES - 1 - k) * T
            self.packed[k, :WIN] = self.hist[s:s + WIN]
        with self._in.overwrite() as a:
            a[...] = self.packed.ravel()
        t0 = time.perf_counter()
        fir_npu(self._in, self._out, cols=self.cols, variant=self.variant)
        self.npu_ms = (time.perf_counter() - t0) * 1e3
        parts = self._out.numpy().reshape(CORES, B).astype(np.int64)
        return parts.sum(0) / (32768.0 * 32768.0)


def synth_ir(fs, rt60=1.3, seed=3):
    """stereo-ish hall: early reflections + exponentially decaying noise, darker over time"""
    rng = np.random.default_rng(seed)
    n = L
    t = np.arange(n) / fs
    env = np.exp(-6.91 * t / rt60)
    noise = rng.normal(size=n)
    # darker tail: average with a running mean that grows with time
    k = 1 + (t * 12).astype(int)
    cs = np.cumsum(np.pad(noise, (16, 0)))
    smooth = np.array([(cs[i + 16] - cs[max(i + 16 - k[i], 0)]) / max(k[i], 1) for i in range(n)])
    ir = env * (0.3 * noise + 0.7 * smooth)
    for d, g in ((0.011, 0.5), (0.019, 0.4), (0.031, 0.35), (0.047, 0.25)):
        ir[int(d * fs)] += g
    ir[0] = 0.0                                             # no direct sound in the wet signal
    return ir


def reference(x, h):
    n = len(x) + len(h) - 1
    nf = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(h, nf), nf)[:len(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ir")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--wet", type=float, default=0.6)
    ap.add_argument("--seconds", type=float, default=0, help="limit the input length (offline)")
    ap.add_argument("--kernel", choices=tuple(SOURCES), default="sliding")
    a = ap.parse_args()
    import soundfile as sf
    x, fs = sf.read(a.wav, dtype="float32", always_2d=True)
    if a.seconds:
        x = x[:int(a.seconds * fs)]
    nch = x.shape[1]
    ir = sf.read(a.ir, dtype="float64", always_2d=True)[0][:, 0] if a.ir else synth_ir(fs)
    rv = [Reverb(ir, a.kernel) for _ in range(nch)]
    print(f"{a.wav}: {nch} ch, {fs} Hz, {len(x) / fs:.1f} s; IR {L} taps = {L / fs:.2f} s; block {B} = {1e3 * B / fs:.1f} ms")
    nb = -(-len(x) // B)
    xp = np.pad(x, ((0, nb * B - len(x)), (0, 0)))
    if a.play:
        import sounddevice as sd
        pos = 0; stats = []

        def cb(outdata, frames, tinfo, status):
            nonlocal pos
            t0 = time.perf_counter()
            if pos >= nb:
                raise sd.CallbackStop
            blk = xp[pos * B:(pos + 1) * B]
            wet = np.stack([rv[c].feed(blk[:, c]) for c in range(nch)], 1)
            outdata[:] = np.clip((1 - a.wet) * blk + a.wet * 3 * wet, -1, 1)
            stats.append(((time.perf_counter() - t0) * 1e3, sum(r.npu_ms for r in rv)))
            pos += 1

        rv[0].feed(np.zeros(B)); [r.feed(np.zeros(B)) for r in rv]  # compile before the stream starts
        for r in rv: r.hist[:] = 0
        with sd.OutputStream(samplerate=fs, channels=nch, blocksize=B, dtype="float32", callback=cb) as s:
            while s.active:
                time.sleep(0.1)
        st = np.array(stats[1:])
        print(f"played {len(stats)} blocks; per block: callback med {np.median(st[:, 0]):.2f} ms (NPU {np.median(st[:, 1]):.2f} ms) "
              f"of {1e3 * B / fs:.1f} ms available -> {np.median(st[:, 0]) / (1e3 * B / fs) * 100:.0f}% load")
        return
    for r in rv:                                      # warm up: the first call loads/compiles the design
        r.feed(np.zeros(B)); r.hist[:] = 0
    t0 = time.perf_counter(); npu = 0.0
    wet = np.zeros_like(xp)
    for i in range(nb):
        for c in range(nch):
            wet[i * B:(i + 1) * B, c] = rv[c].feed(xp[i * B:(i + 1) * B, c])
            npu += rv[c].npu_ms
    dt = time.perf_counter() - t0
    hq = rv[0].h.astype(np.float64) / 32768
    ref = reference(np.round(xp[:, 0] * 32767) / 32768, hq)
    err = np.abs(wet[:, 0] - ref)
    print(f"NPU vs numpy FFT: max |err| {err.max():.2e} ({err.max() * 32768:.1f} LSB of int16), rms {np.sqrt((err ** 2).mean()):.2e}")
    macs = nch * len(xp) * L
    print(f"{nb * nch} dispatches: total {dt:.2f} s wall (NPU calls {npu / 1e3:.2f} s) for {len(xp) / fs:.1f} s of audio "
          f"-> {len(xp) / fs / dt:.0f}x real time, {macs / dt / 1e9:.0f} GMAC/s wall, {macs / (npu / 1e3) / 1e9:.0f} GMAC/s in the NPU calls")
    out = np.clip((1 - a.wet) * xp + a.wet * 3 * wet, -1, 1)[:len(x) + L]
    name = a.wav.rsplit(".", 1)[0].rsplit("/", 1)[-1] + ".reverb.wav"
    sf.write(name, out, fs)
    print("wrote", name)


if __name__ == "__main__":
    main()
