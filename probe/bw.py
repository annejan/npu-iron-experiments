#!/usr/bin/env python3
"""Host<->NPU DMA bandwidth + dispatch floor: shim -> memtile -> shim on every column, no cores."""
import sys, time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.helpers.taplib import TensorAccessPattern

LINE = 4096  # int32 per DMA chunk (16 KB)

@iron.jit
def passthru(a: In, c: Out, *, n: CompileTime[int], cols: CompileTime[int]):
    ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(LINE,), np.dtype[np.int32]]
    ins, outs = [], []
    for col in range(cols):
        fi = ObjectFifo(line_ty, name=f"in{col}", depth=2)
        fo = fi.cons().forward(tile=Tile(col, 1), name=f"out{col}", depth=2)
        ins.append(fi.prod(tile=Tile(col, 0)))
        outs.append(fo.cons(tile=Tile(col, 0)))
    taps = TensorAccessPattern.full((n,)).partition(cols)
    def seq(a, c, ins, outs):
        tg = TaskGroup()
        for col in range(cols):
            ins[col].fill(a, taps[col], group=tg)
        for col in range(cols):
            outs[col].drain(c, taps[col], wait=True, group=tg)
        tg.finish()
    rt = Runtime(seq, [ty, ty, ins, outs])
    return Program(iron.get_current_device(), rt).resolve_program()

def bench(n, cols, iters=30):
    a = iron.arange(0, n, dtype=np.int32, device="npu")
    c = iron.zeros((n,), dtype=np.int32, device="npu")
    passthru(a, c, n=n, cols=cols)  # compile + warm
    assert np.array_equal(a.numpy(), c.numpy()), "mismatch"
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        passthru(a, c, n=n, cols=cols)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e6
    mb = n * 4 / 1e6
    print(f"cols={cols} {mb:8.2f} MB in+out: min {ts.min():8.1f} us  med {np.median(ts):8.1f} us  "
          f"-> {2*mb/ (ts.min()/1e6) / 1e3:6.2f} GB/s (in+out, min)  {mb/(ts.min()/1e6)/1e3:6.2f} GB/s one way")

if __name__ == "__main__":
    dev = iron.get_current_device()
    for cols in (1, 8):
        for n in (LINE*cols, LINE*cols*4, LINE*cols*16, LINE*cols*64, LINE*cols*256):
            bench(n, cols)
