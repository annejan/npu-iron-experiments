#!/usr/bin/env python3
"""Binaural rendering on the NPU: 32 sound sources around your head, one per core, MIT KEMAR HRTFs.

  ../env.sh python fun/binaural.py             render a circular arpeggio, write binaural_arpeggio.wav (headphones!)
  ../env.sh python fun/binaural.py --play      play it live, block by block through the NPU
  ../env.sh python fun/binaural.py --seconds 20

Each core owns one source: per block of B samples it gets the source's history window and the left and right
head-related impulse responses for the source's current direction (MIT KEMAR "compact" set, 128 taps at 44.1 kHz,
resampled to 48 kHz, 160-tap slots, interaural delay included), runs two FIRs with sliding_mac<32, 8>, and returns
two int32 partials; the host sums 32 of each ear. Sources: 32 plucked notes (Karplus-Strong on the host) placed every
11.25 degrees, triggered in turn, the whole ring slowly rotating. The data set is downloaded to data/kemar/ on first run.
"""
import argparse, io, os, sys, time, urllib.request, zipfile
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4
CORES = 32
FS = 48000
B = 1024                  # samples per block (21 ms)
T = 160                   # HRIR taps per ear (128 at 44.1 kHz -> 139 at 48 kHz, zero padded)
WIN = B + T - 1
OBJ = WIN + 1 + 2 * T     # window, pad, hrL reversed, hrR reversed  (WIN + 1 = 1184 keeps the taps 64-byte aligned)
KEMAR_URL = "https://sound.media.mit.edu/resources/KEMAR/compact.zip"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kemar")

KSRC = f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define B {B}
#define T {T}
#define WIN {WIN}
extern "C" {{
/* obj = [window (WIN) | pad | hrL (T, reversed) | hrR (T, reversed)]; out = [L (B) | R (B)] int32 */
void hrir(int16_t *__restrict obj, int32_t *__restrict out) {{
    const int16_t *__restrict w = obj;
    for (int ear = 0; ear < 2; ear++) {{
        const int16_t *__restrict hr = obj + WIN + 1 + ear * T;
        int32_t *__restrict o = out + ear * B;
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
            aie::store_v(o + n, acc.to_vector<int32_t>(0));
        }}
    }}
}}
}}
"""


@iron.jit
def hrir_npu(inp: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS
    in_ty = np.ndarray[(cores * OBJ,), np.dtype[np.int16]]
    out_ty = np.ndarray[(cores * 2 * B,), np.dtype[np.int32]]
    col_in_ty = np.ndarray[(ROWS * OBJ,), np.dtype[np.int16]]
    col_out_ty = np.ndarray[(ROWS * 2 * B,), np.dtype[np.int32]]
    obj_ty = np.ndarray[(OBJ,), np.dtype[np.int16]]
    part_ty = np.ndarray[(2 * B,), np.dtype[np.int32]]
    k = ExternalFunction("hrir", source_string=KSRC, arg_types=[obj_ty, part_ty])
    col_in = [ObjectFifo(col_in_ty, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(col_out_ty, name=f"out{c}", depth=2) for c in range(cols)]
    ci, co = [], []
    for c in range(cols):
        ci += col_in[c].cons().split([i * OBJ for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[obj_ty] * ROWS, names=[f"ci{c}_{i}" for i in range(ROWS)])
        co += col_out[c].prod().join([i * 2 * B for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[part_ty] * ROWS, names=[f"co{c}_{i}" for i in range(ROWS)])

    def core(fi, fo, k):
        x = fi.acquire(1); y = fo.acquire(1)
        k(x, y)
        fo.release(1); fi.release(1)

    workers = [Worker(core, [ci[j].cons(), co[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    tin = TensorAccessPattern.full((cores * OBJ,)).partition(cols)
    tout = TensorAccessPattern.full((cores * 2 * B,)).partition(cols)
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


def load_kemar():
    """azimuth (deg, 0 = front, clockwise) -> (hrL, hrR) at 48 kHz, T taps, float"""
    import soundfile as sf
    os.makedirs(DATA, exist_ok=True)
    zpath = os.path.join(DATA, "compact.zip")
    if not os.path.exists(zpath):
        print("downloading", KEMAR_URL)
        urllib.request.urlretrieve(KEMAR_URL, zpath)
    z = zipfile.ZipFile(zpath)
    t44 = np.arange(128) / 44100
    t48 = np.arange(T) / FS
    table = {}
    for az in range(0, 181, 5):
        with z.open(f"elev0/H0e{az:03d}a.wav") as f:
            h, _ = sf.read(io.BytesIO(f.read()), dtype="float64", always_2d=True)     # (128, 2): left ear, right ear
        h48 = np.stack([np.interp(t48, t44, h[:, c], right=0.0) for c in range(2)], 1)
        table[az] = h48
        if 0 < az < 180:
            table[360 - az] = h48[:, ::-1].copy()                                         # mirror: swap ears
    return table


def pluck(freq, seconds, fs=FS, seed=0):
    """Karplus-Strong plucked string"""
    rng = np.random.default_rng(seed)
    n = int(fs / freq)
    buf = rng.uniform(-1, 1, n)
    out = np.empty(int(seconds * fs))
    for i in range(len(out)):
        j = i % n
        out[i] = buf[j]
        buf[j] = 0.5 * (buf[j] + buf[(j + 1) % n]) * 0.996
    return out * 0.5


class Binaural:
    def __init__(self, sources, hrtf):
        """sources: (32, N) float, one mono signal per core; hrtf: azimuth table"""
        self.sources = np.clip(np.round(sources * 32767), -32768, 32767).astype(np.int16)
        self.hrtf = hrtf
        self.cols = iron.get_current_device().cols
        self.hist = np.zeros((CORES, T - 1), np.int16)
        self.packed = np.zeros((CORES, OBJ), np.int16)
        self._in = iron.zeros((CORES * OBJ,), dtype=np.int16, device="npu")
        self._out = iron.zeros((CORES * 2 * B,), dtype=np.int32, device="npu")
        self.pos = 0
        self.npu_ms = 0.0

    def hr_q(self, az):
        az = int(round(az / 5.0) * 5) % 360
        h = self.hrtf[az]
        q = np.clip(np.round(h * 32767 * 0.5), -32768, 32767).astype(np.int16)     # half scale: 32 sources will add up
        return q[::-1, 0], q[::-1, 1]

    def block(self, azimuths):
        """render the next B samples; azimuths: 32 directions (deg) for this block"""
        blk = self.sources[:, self.pos:self.pos + B]
        if blk.shape[1] < B:
            blk = np.pad(blk, ((0, 0), (0, B - blk.shape[1])))
        self.pos += B
        for k in range(CORES):
            self.packed[k, :T - 1] = self.hist[k]
            self.packed[k, T - 1:WIN] = blk[k]
            hl, hr = self.hr_q(azimuths[k])
            self.packed[k, WIN + 1:WIN + 1 + T] = hl
            self.packed[k, WIN + 1 + T:WIN + 1 + 2 * T] = hr
            self.hist[k] = blk[k, B - (T - 1):]
        with self._in.overwrite() as a:
            a[...] = self.packed.ravel()
        t0 = time.perf_counter()
        hrir_npu(self._in, self._out, cols=self.cols, variant="hrir1")
        self.npu_ms = (time.perf_counter() - t0) * 1e3
        parts = self._out.numpy().reshape(CORES, 2, B).astype(np.int64)
        return parts.sum(0).T / (32768.0 * 32768.0)        # (B, 2) float, ears


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=16)
    ap.add_argument("--play", action="store_true")
    a = ap.parse_args()
    hrtf = load_kemar()
    n = int(a.seconds * FS)
    nb = -(-n // B)
    # 32 plucked notes around a pentatonic ladder, each source triggered in turn every 0.25 s, repeating
    scale = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26, 28, 31, 33, 36]
    sources = np.zeros((CORES, nb * B), np.float32)
    period = 0.25
    for k in range(CORES):
        f = 110 * 2 ** (scale[k % 16] / 12) * (2 if k >= 16 else 1)
        note = pluck(f, 1.2, seed=k)
        t = k * period
        while t < a.seconds:
            s = int(t * FS); e = min(s + len(note), sources.shape[1])
            sources[k, s:e] += note[:e - s]
            t += CORES * period
    bin_ = Binaural(sources, hrtf)
    # verification of block 0 against numpy (full-precision int16 convolution)
    az0 = [(k * 360 / CORES) % 360 for k in range(CORES)]
    out0 = bin_.block(az0)
    ref = np.zeros((B, 2))
    for k in range(CORES):
        hl, hr = bin_.hr_q(az0[k])
        x = bin_.sources[k, :B].astype(np.int64)
        for ear, h in enumerate((hl[::-1].astype(np.int64), hr[::-1].astype(np.int64))):
            ref[:, ear] += np.convolve(x, h)[:B]
    ref /= 32768.0 * 32768.0
    print(f"block 0 vs numpy: max |err| {np.abs(out0 - ref).max():.2e}")
    bin_.pos = 0; bin_.hist[:] = 0
    blocks = []
    t0 = time.perf_counter(); npu = 0.0
    if a.play:
        import sounddevice as sd
        def cb(outdata, frames, tinfo, status):
            i = len(blocks)
            if i >= nb:
                raise sd.CallbackStop
            az = [(k * 360 / CORES + i * B / FS * 12) % 360 for k in range(CORES)]        # ring turns 12 deg/s
            y = bin_.block(az); blocks.append(y)
            outdata[:] = np.clip(y, -1, 1)
        with sd.OutputStream(samplerate=FS, channels=2, blocksize=B, dtype="float32", callback=cb) as s:
            while s.active:
                time.sleep(0.1)
    else:
        for i in range(nb):
            az = [(k * 360 / CORES + i * B / FS * 12) % 360 for k in range(CORES)]
            blocks.append(bin_.block(az)); npu += bin_.npu_ms
        dt = time.perf_counter() - t0
        print(f"{nb} blocks: {dt:.2f} s for {a.seconds:.0f} s of audio ({a.seconds / dt:.0f}x real time); NPU {npu / nb:.2f} ms per block "
              f"= {CORES * 2 * T * B / (npu / nb * 1e-3) / 1e9:.1f} GMAC/s")
    import soundfile as sf
    y = np.concatenate(blocks)[:n]
    y /= max(1e-9, np.abs(y).max()) / 0.9
    sf.write("binaural_arpeggio.wav", y.astype(np.float32), FS)
    print("wrote binaural_arpeggio.wav (headphones)")


if __name__ == "__main__":
    main()
