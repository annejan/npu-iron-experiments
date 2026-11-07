#!/usr/bin/env python3
"""1024-point complex FFTs on all 32 AIE cores with the aie_api radix-4 DIT stages.

  ../env.sh python fun/fft.py            check against numpy, measure FFTs per second
  ../env.sh python fun/fft.py --check    static compile report only

cint16 (int16 re, im) in Q15, twiddles cint16 Q15 (shift_tw 15). Each radix-4 stage shifts by 17: 15 for the
twiddle scaling and 2 so the stage gain of 4 never overflows; the output is fft(x)/1024. Five stages:
vectorization 256, 64, 16, 4, 1 (the first stage is 256 independent 4-point FFTs). Twiddles for stage s,
rotation r: exp(-2j pi r i / n_stage), n_stage = 4^(s+1), i < 4^s; radix-4 stages take them as (r=2, r=1, r=3).
Per dispatch every core transforms FRAMES frames (ObjectFifo loop); twiddles are broadcast once per dispatch.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4
N = 1024
FRAMES = 32               # frames per core per dispatch -> 1024 FFTs per dispatch
TWSTRIDE = 256            # cint16 per twiddle table slot; 15 slots
TWN = 15 * TWSTRIDE
SHIFT_TW, SHIFT = 15, 17

KSRC = f"""
#include <stdint.h>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <aie_api/aie.hpp>
#define N {N}
#define TWSTRIDE {TWSTRIDE}
extern "C" {{
void fft1024(cint16 *__restrict x, cint16 *__restrict tw, cint16 *__restrict y) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    #define T(s, r) (tw + ((s) * 3 + (r)) * TWSTRIDE)
    // AIE2P has no fft_dit_r4_stage(); the (deprecated) class has run() with the same arguments.
    // Twiddle args in the order (r=2, r=1, r=3); x and y alternate, the result lands in y.
    aie::fft_dit<256, 4, cint16>().run(x, T(0, 1), T(0, 0), T(0, 2), N, {SHIFT_TW}, {SHIFT}, false, y);
    aie::fft_dit<64, 4, cint16>().run(y, T(1, 1), T(1, 0), T(1, 2), N, {SHIFT_TW}, {SHIFT}, false, x);
    aie::fft_dit<16, 4, cint16>().run(x, T(2, 1), T(2, 0), T(2, 2), N, {SHIFT_TW}, {SHIFT}, false, y);
    aie::fft_dit<4, 4, cint16>().run(y, T(3, 1), T(3, 0), T(3, 2), N, {SHIFT_TW}, {SHIFT}, false, x);
    aie::fft_dit<1, 4, cint16>().run(x, T(4, 1), T(4, 0), T(4, 2), N, {SHIFT_TW}, {SHIFT}, false, y);
}}
}}
"""


def twiddles():
    tw = np.zeros((15, TWSTRIDE), np.complex128)
    for s in range(5):
        n_stage = 4 ** (s + 1)
        for r in (1, 2, 3):
            i = np.arange(4 ** s)
            tw[s * 3 + (r - 1), :len(i)] = np.exp(-2j * np.pi * r * i / n_stage)
    q = np.clip(np.round(tw * (1 << SHIFT_TW)), -32768, 32767)
    out = np.empty((15, TWSTRIDE, 2), np.int16)
    out[..., 0] = q.real; out[..., 1] = q.imag
    return out.ravel()


def ext():
    frame_ty = np.ndarray[(2 * N,), np.dtype[np.int16]]
    tw_ty = np.ndarray[(2 * TWN,), np.dtype[np.int16]]
    return ExternalFunction("fft1024", source_string=KSRC, arg_types=[frame_ty, tw_ty, frame_ty])


@iron.jit
def fft_npu(x: In, tw: In, y: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS
    n = cores * FRAMES * 2 * N
    data_ty = np.ndarray[(n,), np.dtype[np.int16]]
    tw_ty = np.ndarray[(2 * TWN,), np.dtype[np.int16]]
    frame_ty = np.ndarray[(2 * N,), np.dtype[np.int16]]
    quad_ty = np.ndarray[(ROWS * 2 * N,), np.dtype[np.int16]]
    k = ext()
    tw_in = [ObjectFifo(tw_ty, name=f"tw{c}", depth=1) for c in range(cols)]
    col_in = [ObjectFifo(quad_ty, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    fi, fo = [], []
    for c in range(cols):
        fi += col_in[c].cons().split([i * 2 * N for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[frame_ty] * ROWS, names=[f"fi{c}_{i}" for i in range(ROWS)])
        fo += col_out[c].prod().join([i * 2 * N for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[frame_ty] * ROWS, names=[f"fo{c}_{i}" for i in range(ROWS)])

    def core(tw, fi, fo, k):
        t = tw.acquire(1)
        for _ in range_(FRAMES):
            a = fi.acquire(1); b = fo.acquire(1)
            k(a, t, b)
            fo.release(1); fi.release(1)
        tw.release(1)

    workers = [Worker(core, [tw_in[c].cons(), fi[c * ROWS + r].cons(), fo[c * ROWS + r].prod(), k], tile=Tile(c, 2 + r))
               for c in range(cols) for r in range(ROWS)]
    # column object j = frame j of each of the column's 4 cores: core (c, r) gets frames [(4c+r)*FRAMES, +FRAMES)
    taps = [TensorAccessPattern((n,), c * ROWS * FRAMES * 2 * N, [FRAMES, ROWS, 2 * N], [2 * N, FRAMES * 2 * N, 1]) for c in range(cols)]
    tp, ip, oc = [f.prod() for f in tw_in], [f.prod() for f in col_in], [f.cons() for f in col_out]
    full = TensorAccessPattern.full((2 * TWN,))

    def seq(x, tw, y, tp, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            tp[c].fill(tw, full, group=tg)
            ip[c].fill(x, taps[c], group=tg)
        for c in range(cols):
            oc[c].drain(y, taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [data_ty, tw_ty, data_ty, tp, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def check():
    from aie.utils.compile import remarks
    d = Path(".remarks/rem_fft"); d.mkdir(parents=True, exist_ok=True)
    rep, why = remarks.analyze(ext(), "aie2p", d)
    if rep is None:
        print(why); sys.exit(1)
    print("libcalls", rep.libcalls, "stack", rep.kernel_stack_bytes, "notes", rep.schedule_notes)
    for key, l in rep.loops.items():
        print("  loop", key, "ii", l.ii, "ns", l.ns, "zol", l.zol, "trips", l.trips, "bundles", l.bundle_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    if a.check:
        return check()
    cols = iron.get_current_device().cols
    cores = cols * ROWS
    nfr = cores * FRAMES
    rng = np.random.default_rng(1)
    # test frames: random complex noise at 1/4 scale, plus a few pure tones
    xc = (rng.normal(size=(nfr, N)) + 1j * rng.normal(size=(nfr, N))) * 0.25
    for i in range(8):
        xc[i] = 0.8 * np.exp(2j * np.pi * (7 + 31 * i) * np.arange(N) / N)
    xq = np.empty((nfr, N, 2), np.int16)
    xq[..., 0] = np.clip(np.round(xc.real * 32767), -32768, 32767); xq[..., 1] = np.clip(np.round(xc.imag * 32767), -32768, 32767)
    x = iron.tensor(xq.ravel(), dtype=np.int16, device="npu")
    tw = iron.tensor(twiddles(), dtype=np.int16, device="npu")
    y = iron.zeros((nfr * 2 * N,), dtype=np.int16, device="npu")
    t0 = time.perf_counter(); fft_npu(x, tw, y, cols=cols, variant="fft1"); print(f"first call {1e3 * (time.perf_counter() - t0):.0f} ms")
    got = y.numpy().reshape(nfr, N, 2).astype(np.float64) / 32768
    got = got[..., 0] + 1j * got[..., 1]
    ref = np.fft.fft(xq[..., 0] / 32768 + 1j * xq[..., 1] / 32768, axis=1) / N
    err = np.abs(got - ref)
    print(f"vs numpy: max |err| {err.max():.2e} ({err.max() * 32768:.1f} LSB), rms {np.sqrt((err ** 2).mean()):.2e}; "
          f"tone frame 0 peak bin NPU {np.argmax(np.abs(got[0]))} ref {np.argmax(np.abs(ref[0]))}")
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter(); fft_npu(x, tw, y, cols=cols, variant="fft1"); ts.append(time.perf_counter() - t0)
    t = min(ts)
    print(f"{nfr} x {N}-point cint16 FFTs per dispatch: {t * 1e3:.2f} ms -> {nfr / t / 1e6:.2f} M FFT/s, "
          f"{nfr * N / t / 1e9:.2f} Gsample/s ({nfr * 5 * N * np.log2(N) / t / 1e9:.0f} GFLOP/s-equivalent), "
          f"{nfr * 2 * N * 2 * 2 / t / 1e9:.1f} GB/s in+out")


if __name__ == "__main__":
    main()
