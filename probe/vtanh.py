#!/usr/bin/env python3
"""The AIE2P hardware tanh (::tanh on a v16accfloat, bf16 out): accuracy against numpy over [-8, 8] and the
cost per 16 lanes. One column, four cores, 1024 floats each."""
import sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4; N = 1024


def ksrc(reps):
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void vtanh(float *in, float *out) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int r = 0; r < {reps}; r++)
    for (int x = 0; x < {N}; x += 16) {{
        aie::vector<float, 16> v = aie::load_v<16>(in + x);
        aie::vector<bfloat16, 16> t = aie::tanh<bfloat16>(v);
        aie::accum<accfloat, 16> a(t);
        aie::store_v(out + x, a.to_vector<float>());
    }}
}}
"""


@iron.jit
def probe(inp: In, out: Out, *, reps: CompileTime[int], variant: CompileTime[str]):
    ty = np.ndarray[(ROWS * N,), np.dtype[np.float32]]
    obj = np.ndarray[(N,), np.dtype[np.float32]]
    k = ExternalFunction("vtanh", source_string=ksrc(reps), arg_types=[obj, obj])
    fin = ObjectFifo(ty, name="in", depth=1)
    fout = ObjectFifo(ty, name="out", depth=1)
    ci = fin.cons().split([i * N for i in range(ROWS)], tile=Tile(0, 1), depths=[1] * ROWS, obj_types=[obj] * ROWS, names=[f"ci{i}" for i in range(ROWS)])
    co = fout.prod().join([i * N for i in range(ROWS)], tile=Tile(0, 1), depths=[1] * ROWS, obj_types=[obj] * ROWS, names=[f"co{i}" for i in range(ROWS)])

    def core(fi, fo, k):
        x = fi.acquire(1); y = fo.acquire(1); k(x, y); fo.release(1); fi.release(1)

    workers = [Worker(core, [ci[j].cons(), co[j].prod(), k], tile=Tile(0, 2 + j)) for j in range(ROWS)]

    def seq(inp, out, ip, oc):
        tg = TaskGroup()
        ip.fill(inp, TensorAccessPattern.full((ROWS * N,)), group=tg)
        oc.drain(out, TensorAccessPattern.full((ROWS * N,)), wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [ty, ty, fin.prod(), fout.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    x = np.linspace(-8, 8, ROWS * N, dtype=np.float32)
    a = iron.zeros((ROWS * N,), dtype=np.float32, device="npu")
    b = iron.zeros((ROWS * N,), dtype=np.float32, device="npu")
    with a.overwrite() as v: v[...] = x
    probe(a, b, reps=1, variant="vtanh1")
    y = b.numpy().copy()
    ref = np.tanh(x.astype(np.float64))
    err = y - ref
    print(f"max |err| {np.abs(err).max():.3e} at x={x[np.abs(err).argmax()]:.3f}; rms {np.sqrt((err**2).mean()):.3e}")
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1), (1, 2), (2, 4), (4, 8)]:
        m = (np.abs(x) >= lo) & (np.abs(x) < hi)
        print(f"  |x| in [{lo},{hi}): max err {np.abs(err[m]).max():.3e}  rel-to-bf16-ulp {np.abs(err[m]).max() / (2**-8 * max(np.abs(ref[m]).max(), 1e-9)):.1f}")
    # a few spot values
    for xv in [0.1, 0.3, 0.49, 0.5, 0.51, 0.7, 1.0, 2.0, 3.0]:
        i = np.abs(x - xv).argmin()
        print(f"  x={x[i]:+.4f} npu={y[i]:+.6f} tanh={ref[i]:+.6f} err={y[i]-ref[i]:+.2e}")
    # timing: reps loop
    for reps in (1, 64):
        probe(a, b, reps=reps, variant="vtanh1")
        t0 = time.perf_counter(); n = 20
        for _ in range(n): probe(a, b, reps=reps, variant="vtanh1")
        dt = (time.perf_counter() - t0) / n
        print(f"reps={reps}: {dt*1e3:.3f} ms per call")
