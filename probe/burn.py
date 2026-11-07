#!/usr/bin/env python3
"""Compute ceiling: 32 cores each loop REPS times over a 16 KB int16 buffer with a vector mul (II 1 ideal)."""
import sys, time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.helpers.taplib import TensorAccessPattern

ROWS = 4
N = 8192  # int16 per core chunk = 16 KB

KSRC = """
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" {
void burn(int16_t *__restrict in, int16_t *__restrict out, int32_t reps) {
    for (int r = 0; r < reps; r++) {
        const int16_t *__restrict p = in;
        int16_t *__restrict q = out;
        for (unsigned i = 0; i < %d / 32; i++) {
            aie::vector<int16_t, 32> v = aie::load_v<32>(p); p += 32;
            aie::accum<acc32, 32> a = aie::mul(v, (int16_t)3);
            aie::store_v(q, a.to_vector<int16_t>(1)); q += 32;
        }
        in = out;  // chain: next rep reads what we wrote (keeps the loop honest)
    }
}
}
""" % N

SOURCES = {'chain': KSRC}

@iron.jit
def burn_all(a: In, c: Out, *, cols: CompileTime[int], reps: CompileTime[int], variant: CompileTime[str] = 'chain'):
    cores = cols * ROWS
    n = cores * N
    ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS * N,), np.dtype[np.int16]]
    chunk_ty = np.ndarray[(N,), np.dtype[np.int16]]
    k = ExternalFunction("burn", source_string=SOURCES[variant], arg_types=[chunk_ty, chunk_ty, np.dtype(np.int32)])
    col_in = [ObjectFifo(col_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    cin, cout = [], []
    for c in range(cols):
        cin += col_in[c].cons().split([i * N for i in range(ROWS)], tile=Tile(c, 1), depths=[1]*ROWS,
                                      obj_types=[chunk_ty]*ROWS, names=[f"ci{c}_{i}" for i in range(ROWS)])
        cout += col_out[c].prod().join([i * N for i in range(ROWS)], tile=Tile(c, 1), depths=[1]*ROWS,
                                       obj_types=[chunk_ty]*ROWS, names=[f"co{c}_{i}" for i in range(ROWS)])
    def core(i, o, k):
        x = i.acquire(1); y = o.acquire(1)
        k(x, y, reps)
        i.release(1); o.release(1)
    workers = [Worker(core, [cin[j].cons(), cout[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    taps = TensorAccessPattern.full((n,)).partition(cols)
    ip = [f.prod() for f in col_in]; oc = [f.cons() for f in col_out]
    def seq(a, c, ip, oc):
        tg = TaskGroup()
        for col in range(cols): ip[col].fill(a, taps[col], group=tg)
        for col in range(cols): oc[col].drain(c, taps[col], wait=True, group=tg)
        tg.finish()
    rt = Runtime(seq, [ty, ty, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

def bench(reps, cols, iters=10, variant='chain'):
    cores = cols * ROWS; n = cores * N
    a = iron.tensor(np.ones(n, dtype=np.int16), dtype=np.int16, device="npu")
    c = iron.zeros((n,), dtype=np.int16, device="npu")
    burn_all(a, c, cols=cols, reps=reps, variant=variant)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); burn_all(a, c, cols=cols, reps=reps, variant=variant); ts.append(time.perf_counter() - t0)
    t = min(ts)
    ops = cores * N * reps
    if iters > 1: print(f"{variant:14s} cols={cols} cores={cores} reps={reps:5d}: {t*1e6:9.1f} us  {ops/t/1e9:8.1f} G int16-mul/s  "
          f"(ideal II1: {cores*32*1.76:8.1f})  per-core {ops/cores/t/1e9:6.2f} G/s")
    return t

if __name__ == "__main__":
    for cols in (1, 8):
        for reps in (1, 16, 256, 2048):
            bench(reps, cols)
