#!/usr/bin/env python3
"""Additive synthesis on the NPU: 3072 sinusoidal partials (96 per core) as magic-circle oscillators, time in the lanes.

  ./env.sh python fun/additive.py [drumloop.wav]    check vs a numpy replica, write additive_bells.wav and (from a 48 kHz
                                                    drum loop, any wav you have) additive_resynth.wav

One partial keeps 32 consecutive samples of its sinusoid in a 32-lane int16 vector pair (u = cos lanes, v = the
Gordon-Smith companion). Advancing all 32 lanes by one sub-block of 32 samples is one rotation by theta = 32 w with the
same constant in every lane: u' = u + e v; v' = v - e u', e = 2 sin(theta/2). The update has determinant 1 for any
quantised e, so the amplitude cannot drift. Per 1024-sample block a core rotates its P partials 32 times and
accumulates amplitude-ramped partials into the audio; the state (u, v) round-trips through DDR with the block.
No sin/cos on the chip, no tables: the host sets the initial lanes once. Demo 1: bells (inharmonic partial sets,
per-partial exponential decay). Demo 2: resynthesis of a drum loop from its STFT magnitudes (2048 partials, one per bin).

Oscillator: J. W. Gordon and J. O. Smith, "A sine generation algorithm for VLSI applications", ICMC 1985; the "magic
circle" name follows J. O. Smith, Physical Audio Signal Processing (ccrma.stanford.edu/~jos/pasp/Digital_Sinusoid_Generators.html).
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4; CORES = 32
P = 96                      # partials per core
B = 1024                    # samples per block
NSUB = B // 32
FS = 48000
IN_N = 2 * P * 32 + 4 * P   # u, v, a0, a1, eps, dec   (int16)
OUT_N = 2 * P * 32 + 2 * B  # u', v', audio (B int32 as 2B int16)
AMP = 31130                 # 0.95 in Q15

KSRC = f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define P {P}
#define NSUB {NSUB}
using V = aie::vector<int16_t, 32>;
extern "C" {{
void add_block(int16_t *__restrict in, int16_t *__restrict out) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    const int16_t *a0 = in + 2 * P * 32, *a1 = a0 + P, *eps = a1 + P, *dec = eps + P;
    int16_t *uo = out, *vo = out + P * 32;
    int32_t *audio = (int32_t *)(out + 2 * P * 32);
    for (int i = 0; i < 2 * P * 32; i += 32) aie::store_v(out + i, aie::load_v<32>(in + i));
    for (int k = 0; k < NSUB; k++) {{
        aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
        for (int p = 0; p < P; p++) {{
            V u = aie::load_v<32>(uo + p * 32), v = aie::load_v<32>(vo + p * 32);
            const int16_t e = eps[p], d = dec[p];
            aie::accum<acc32, 32> t = aie::mul(u, (int16_t)8192); t = aie::mac(t, v, e);      // Q15*Q13 -> Q28
            V un = t.to_vector<int16_t>(13);
            t = aie::mul(v, (int16_t)8192); t = aie::msc(t, un, e);
            V vn = t.to_vector<int16_t>(13);
            if (d != 32767) {{                                  // 32767 means no decay (Q15 cannot hold 1.0)
                un = aie::mul(un, d).to_vector<int16_t>(15);
                vn = aie::mul(vn, d).to_vector<int16_t>(15);
            }}
            aie::store_v(uo + p * 32, un); aie::store_v(vo + p * 32, vn);
            const int16_t a = (int16_t)(a0[p] + (((int32_t)(a1[p] - a0[p]) * k) >> 5));
            acc = aie::mac(acc, un, a);
        }}
        aie::store_v(audio + k * 32, acc.to_vector<int32_t>(0));
    }}
}}
}}
"""


@iron.jit
def add_npu(inp: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS
    in_ty = np.ndarray[(cores * IN_N,), np.dtype[np.int16]]
    out_ty = np.ndarray[(cores * OUT_N,), np.dtype[np.int16]]
    cin_ty = np.ndarray[(ROWS * IN_N,), np.dtype[np.int16]]
    cout_ty = np.ndarray[(ROWS * OUT_N,), np.dtype[np.int16]]
    obj_in = np.ndarray[(IN_N,), np.dtype[np.int16]]
    obj_out = np.ndarray[(OUT_N,), np.dtype[np.int16]]
    k = ExternalFunction("add_block", source_string=KSRC, arg_types=[obj_in, obj_out])
    col_in = [ObjectFifo(cin_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(cout_ty, name=f"out{c}", depth=1) for c in range(cols)]
    ci, co = [], []
    for c in range(cols):
        ci += col_in[c].cons().split([i * IN_N for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS, obj_types=[obj_in] * ROWS, names=[f"ci{c}_{i}" for i in range(ROWS)])
        co += col_out[c].prod().join([i * OUT_N for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS, obj_types=[obj_out] * ROWS, names=[f"co{c}_{i}" for i in range(ROWS)])

    def core(fi, fo, k):
        x = fi.acquire(1); y = fo.acquire(1); k(x, y); fo.release(1); fi.release(1)

    workers = [Worker(core, [ci[j].cons(), co[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    tin = TensorAccessPattern.full((cores * IN_N,)).partition(cols)
    tout = TensorAccessPattern.full((cores * OUT_N,)).partition(cols)
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


class Additive:
    """3072 partials. set(freqs) initialises the oscillators; block(a0, a1, dec) renders 1024 samples."""

    def __init__(self):
        self.cols = iron.get_current_device().cols
        self.N = CORES * P
        self.state = np.zeros((CORES, IN_N), np.int16)
        self._in = iron.zeros((CORES * IN_N,), dtype=np.int16, device="npu")
        self._out = iron.zeros((CORES * OUT_N,), dtype=np.int16, device="npu")
        self.ms = 0.0

    def set(self, freqs, phases=None):
        freqs = np.asarray(freqs, np.float64); assert len(freqs) == self.N
        phases = np.zeros(self.N) if phases is None else np.asarray(phases)
        w = 2 * np.pi * freqs / FS
        theta = 32 * w
        eps = np.clip(np.round(2 * np.sin(theta / 2) * 8192), -32767, 32767).astype(np.int16)
        # the block rotation angle the quantised e produces, on the same branch as theta mod 2pi (cos is even,
        # so e alone cannot tell theta from 2pi - theta); the lanes then run at exactly that frequency
        tm = np.mod(theta, 2 * np.pi); k = np.floor(theta / (2 * np.pi))
        base = 2 * np.arcsin(np.clip(eps / 16384, -1, 1))
        theta_q = np.where(tm <= np.pi, base, 2 * np.pi - base) + 2 * np.pi * k
        wq = theta_q / 32
        j = np.arange(32)
        u = np.round(AMP * np.cos(np.outer(wq, j) + phases[:, None]))
        v = np.round(-AMP * np.sin(np.outer(wq, j) + phases[:, None] + theta_q[:, None] / 2))
        s = self.state.reshape(CORES, IN_N)
        s[:, :P * 32] = u.reshape(CORES, P * 32); s[:, P * 32:2 * P * 32] = v.reshape(CORES, P * 32)
        s[:, 2 * P * 32 + 2 * P:2 * P * 32 + 3 * P] = eps.reshape(CORES, P)
        self.fq = wq * FS / (2 * np.pi)

    def fill(self, a0, a1, dec):
        s = self.state
        s[:, 2 * P * 32:2 * P * 32 + P] = np.asarray(a0).reshape(CORES, P)
        s[:, 2 * P * 32 + P:2 * P * 32 + 2 * P] = np.asarray(a1).reshape(CORES, P)
        s[:, 2 * P * 32 + 3 * P:] = np.asarray(dec).reshape(CORES, P)

    def block(self, a0, a1, dec):
        """a0, a1: amplitudes (Q15 int, sum over all partials <= 32767) at block start/end; dec: per-sub-block decay Q15 (32767 = none)"""
        self.fill(a0, a1, dec)
        s = self.state
        with self._in.overwrite() as a:
            a[...] = s.ravel()
        t0 = time.perf_counter()
        add_npu(self._in, self._out, cols=self.cols, variant="add2")
        self.ms = (time.perf_counter() - t0) * 1e3
        o = self._out.numpy().reshape(CORES, OUT_N)
        s[:, :2 * P * 32] = o[:, :2 * P * 32]
        audio = o[:, 2 * P * 32:].copy().view(np.int32).astype(np.int64).sum(0)
        return audio / (32768.0 * 32768.0)


def replica(state_in):
    """numpy int replica of add_block for one core object (bit-exact: conv_even shifts, saturation)"""
    s = state_in.astype(np.int64)
    u = s[:P * 32].reshape(P, 32).copy(); v = s[P * 32:2 * P * 32].reshape(P, 32).copy()
    a0, a1, eps, dec = (s[2 * P * 32 + i * P:2 * P * 32 + (i + 1) * P] for i in range(4))
    def srs(x, sh):
        q, r = np.divmod(x, 1 << sh); half = 1 << (sh - 1)
        q = np.where(r > half, q + 1, np.where(r == half, q + (q & 1), q))      # round to nearest even
        return np.clip(q, -32768, 32767)
    audio = np.zeros((NSUB, 32), np.int64)
    for k in range(NSUB):
        acc = np.zeros(32, np.int64)
        for p in range(P):
            un = srs(u[p] * 8192 + v[p] * eps[p], 13)
            vn = srs(v[p] * 8192 - un * eps[p], 13)
            if dec[p] != 32767:
                un = srs(un * dec[p], 15); vn = srs(vn * dec[p], 15)
            u[p], v[p] = un, vn
            a = a0[p] + (((a1[p] - a0[p]) * k) >> 5)
            acc += un * a
        audio[k] = np.clip(acc, -2 ** 31, 2 ** 31 - 1)
    return u, v, audio.ravel()


def main():
    import soundfile as sf
    syn = Additive()
    rng = np.random.default_rng(0)
    # --- check: random partials, two blocks, NPU vs replica on core 0
    syn.set(rng.uniform(50, 12000, syn.N), rng.uniform(0, 2 * np.pi, syn.N))
    a = rng.integers(0, 300, syn.N); dec = rng.integers(32000, 32768, syn.N)
    syn.fill(a, a, dec); s0 = syn.state[0].copy(); syn.block(a, a, dec); rep_u, rep_v, rep_audio = replica(s0)
    o = syn._out.numpy().reshape(CORES, OUT_N)[0]
    ok = np.array_equal(o[:P * 32], rep_u.ravel()) and np.array_equal(o[P * 32:2 * P * 32], rep_v.ravel()) and np.array_equal(o[2 * P * 32:].copy().view(np.int32), rep_audio.astype(np.int32))
    if not ok:
        du = np.where(o[:P * 32] != rep_u.ravel())[0]; dv = np.where(o[P * 32:2 * P * 32] != rep_v.ravel())[0]
        da = np.where(o[2 * P * 32:].copy().view(np.int32) != rep_audio.astype(np.int32))[0]
        print(f"  diffs: u {len(du)} (first {du[:3]}: npu {o[du[:3]]} rep {rep_u.ravel()[du[:3]]}), v {len(dv)} (first {dv[:3]}), audio {len(da)} (first {da[:3]}: npu {o[2 * P * 32:].copy().view(np.int32)[da[:3]]} rep {rep_audio[da[:3]]})")
    print(f"core 0 vs numpy replica after one block: {'bit-exact' if ok else 'MISMATCH'}; NPU {syn.ms:.2f} ms per 1024-sample block of {syn.N} partials "
          f"= {syn.N * B / (syn.ms * 1e-3) / 1e9:.1f} G partial-samples/s")
    # --- frequency accuracy and amplitude stability: one partial at 1000 Hz for 10 s
    f = np.zeros(syn.N); f[0] = 1000.0; syn.set(np.where(f > 0, f, 100.0))
    amp = np.zeros(syn.N, np.int64); amp[0] = 20000; decn = np.full(syn.N, 32767)
    y = np.concatenate([syn.block(amp, amp, decn) for _ in range(469)])
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))); pk = np.argmax(spec) * FS / len(y)
    print(f"1000 Hz partial over 10 s: peak at {pk:.3f} Hz (quantised target {syn.fq[0]:.3f}); amplitude first/last second {np.abs(y[:FS]).max():.4f} / {np.abs(y[-FS:]).max():.4f}")
    # --- demo 1: bells. 12 strikes of inharmonic partial sets with per-partial decays
    f = rng.uniform(100, 8000, syn.N); syn.set(f, rng.uniform(0, 2 * np.pi, syn.N))
    nb = int(14 * FS / B); out = np.zeros(nb * B)
    amps = np.zeros(syn.N); decs = np.full(syn.N, 32767.0)
    notes = [220, 261.6, 329.6, 392, 440, 523.3, 587.3, 659.3, 783.9, 880, 1046.5, 1174.7]
    strikes = {int(i * 1.0 * FS / B): notes[i] for i in range(12)}
    ratios = np.array([1, 2.76, 5.40, 8.93, 13.34, 18.64, 1.5, 3.2, 6.1]); rel = np.array([1, .6, .4, .25, .15, .1, .3, .2, .1])
    pdec = 32767 * np.exp(-B / FS / np.array([2.5, 1.8, 1.2, 0.8, 0.5, 0.35, 1.0, 0.7, 0.4]))
    used = 0
    for i in range(nb):
        if i in strikes:
            base = strikes[i]
            for r, g, d in zip(ratios, rel, pdec):
                p = used % syn.N; used += 1
                f[p] = base * r; amps[p] = 2500 * g; decs[p] = d
                # re-seed that partial's lanes at the new frequency (one partial): recompute its state rows
                w = 2 * np.pi * f[p] / FS; th = 32 * w; e = int(np.clip(round(2 * np.sin(th / 2) * 8192), -32767, 32767))
                thq = 2 * np.arcsin(e / 8192 / 2); wq = thq / 32; j = np.arange(32)
                c, l = divmod(p, P); st = syn.state[c]
                st[l * 32:(l + 1) * 32] = np.round(AMP * np.cos(wq * j)); st[P * 32 + l * 32:P * 32 + (l + 1) * 32] = np.round(-AMP * np.sin(wq * j + thq / 2))
                st[2 * P * 32 + 2 * P + l] = e
        a0 = amps.copy(); amps *= (decs / 32767) ** 32      # amplitude envelope follows the decay too (approx), partial state decays on chip
        out[i * B:(i + 1) * B] = syn.block(np.round(a0).astype(np.int64), np.round(amps).astype(np.int64), np.round(decs).astype(np.int64))
    out /= max(1e-9, np.abs(out).max()) / 0.8
    sf.write("additive_bells.wav", out.astype(np.float32), FS)
    # --- demo 2: resynthesis of a drum loop from STFT magnitudes: partial i = bin i of a 4096-point STFT, hop 1024
    try:
        loop = sys.argv[1] if len(sys.argv) > 1 else "drumloop.wav"
        x, fs = sf.read(loop, dtype="float64", always_2d=True)
        x = x.mean(1); assert fs == FS
    except Exception as e:
        print("no drum loop for the resynthesis demo:", e); return
    NFFT = 4096; nbins = NFFT // 2
    freqs = np.arange(syn.N) * FS / NFFT; freqs[nbins:] = 100.0
    syn.set(freqs, rng.uniform(0, 2 * np.pi, syn.N))
    nb = (len(x) - NFFT) // B
    out = np.zeros(nb * B); win = np.hanning(NFFT)
    mags = np.zeros(syn.N); decn = np.full(syn.N, 32767, np.int64)
    prev = np.zeros(syn.N)
    for i in range(nb):
        frame = x[i * B:i * B + NFFT] * win
        m = np.abs(np.fft.rfft(frame))[:nbins] / NFFT * 4
        cur = np.zeros(syn.N); cur[:nbins] = m
        q = np.minimum(cur * 32767 * 0.6, 3000)
        a0 = np.round(prev).astype(np.int64); a1 = np.round(q).astype(np.int64); prev = q
        if a1.sum() > 32000: a1 = (a1 * 32000 // max(1, a1.sum())).astype(np.int64)
        out[i * B:(i + 1) * B] = syn.block(a0, a1, decn)
    out /= max(1e-9, np.abs(out).max()) / 0.8
    sf.write("additive_resynth.wav", out.astype(np.float32), FS)
    print(f"wrote additive_bells.wav (14 s) and additive_resynth.wav ({nb * B / FS:.1f} s, {nbins} partials from the STFT); NPU {syn.ms:.2f} ms per block")


if __name__ == "__main__":
    main()
