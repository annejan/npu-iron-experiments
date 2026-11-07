#!/usr/bin/env python3
"""Host<->NPU DMA bandwidth, second attempt: both shim channels per column in each direction (2 in + 2 out),
big objects, no compute. bw.py used one fifo per direction per column and 16 KB objects and got 26 GB/s;
others report 52-98 GB/s on this chip."""
import sys, time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.helpers.taplib import TensorAccessPattern

@iron.jit
def passthru2(a: In, c: Out, *, n: CompileTime[int], cols: CompileTime[int], chans: CompileTime[int], line: CompileTime[int]):
    ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(line,), np.dtype[np.int32]]
    ins, outs = [], []
    for col in range(cols):
        for ch in range(chans):
            fi = ObjectFifo(line_ty, name=f"in{col}_{ch}", depth=2)
            fo = fi.cons().forward(tile=Tile(col, 1), name=f"out{col}_{ch}", depth=2)
            ins.append(fi.prod(tile=Tile(col, 0)))
            outs.append(fo.cons(tile=Tile(col, 0)))
    taps = TensorAccessPattern.full((n,)).partition(cols * chans)
    def seq(a, c, ins, outs):
        tg = TaskGroup()
        for i in range(cols * chans):
            ins[i].fill(a, taps[i], group=tg)
        for i in range(cols * chans):
            outs[i].drain(c, taps[i], wait=True, group=tg)
        tg.finish()
    rt = Runtime(seq, [ty, ty, ins, outs])
    return Program(iron.get_current_device(), rt).resolve_program()

def bench(n, cols, chans, line, iters=20):
    a = iron.arange(0, n, dtype=np.int32, device="npu")
    c = iron.zeros((n,), dtype=np.int32, device="npu")
    passthru2(a, c, n=n, cols=cols, chans=chans, line=line)
    assert np.array_equal(a.numpy(), c.numpy())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); passthru2(a, c, n=n, cols=cols, chans=chans, line=line); ts.append(time.perf_counter() - t0)
    t = min(ts); mb = n * 4 / 1e6
    print(f"cols={cols} chans/col={chans} line={line * 4 // 1024:3d} KB  {mb:7.1f} MB: min {t * 1e6:8.1f} us -> {mb / t / 1e3:6.2f} GB/s one way ({2 * mb / t / 1e3:6.2f} in+out)")

if __name__ == "__main__":
    for chans in (1, 2):
        for line in (4096, 16384):
            for n_mb in (8, 32, 64):
                n = n_mb * 1024 * 1024 // 4
                n -= n % (8 * chans * line)
                try:
                    bench(n, 8, chans, line)
                except Exception as e:
                    print(f"chans={chans} line={line} n={n}: {str(e).splitlines()[0][:120]}")
