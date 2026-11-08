#!/usr/bin/env python3
"""aie::mmul<8,8,8,int8,int8> on aie2p (the dense full-rate shape; <8,16,8> wants a sparse B): operand layouts (A 8 px x 8 ch
row-major from an 8-byte-aligned pointer, B 8x8 row-major, C 8x8 int32 row-major) checked against numpy as a 3-tap conv over 16
channel blocks with two output blocks, and the cost per mmul in that loop. One core."""
import sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

PX = 40; RS = 48; KB = 16; T = 3; NB = 2            # RS: ring row stride in px (48*8 = 384 B, a multiple of 64)
AW = KB * RS * 8 // 4; BW = KB * T * NB * 64 // 4; INW = AW + BW; OUTW = PX // 8 * NB * 64


def ksrc(reps):
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void probe(int32_t *in, int32_t *out) {{
    const int8_t *A = (const int8_t *)in; const int8_t *B = (const int8_t *)(in + {AW});
    using MM = aie::mmul<8, 8, 8, int8, int8>;
    for (int r = 0; r < {reps}; r++)
    for (int m = 0; m < {PX // 8}; m++) {{
        MM c0, c1;
        const int8_t *bp = B;
        const int8_t *ap = A + m * 64;
        _Pragma("clang loop unroll_count(2)")
        for (int kb = 0; kb < {KB}; kb++) {{
            aie::vector<int8, 64> v0 = aie::load_v<64>(ap), v1 = aie::load_v<64>(ap + 64); ap += {RS * 8};
            aie::vector<int8, 64> a1 = aie::shuffle_down_fill(v0, v1, 8), a2 = aie::shuffle_down_fill(v0, v1, 16);
            c0.mac(v0, aie::load_v<64>(bp)); c1.mac(v0, aie::load_v<64>(bp + 64));
            c0.mac(a1, aie::load_v<64>(bp + 128)); c1.mac(a1, aie::load_v<64>(bp + 192));
            c0.mac(a2, aie::load_v<64>(bp + 256)); c1.mac(a2, aie::load_v<64>(bp + 320)); bp += 384;
        }}
        aie::store_v(out + m * 128, c0.template to_vector<int32>());
        aie::store_v(out + m * 128 + 64, c1.template to_vector<int32>());
    }}
}}
"""


@iron.jit
def probe(inp: In, out: Out, *, reps: CompileTime[int], variant: CompileTime[str]):
    ity = np.ndarray[(INW,), np.dtype[np.int32]]; oty = np.ndarray[(OUTW,), np.dtype[np.int32]]
    k = ExternalFunction("probe", source_string=ksrc(reps), arg_types=[ity, oty])
    fin = ObjectFifo(ity, name="in", depth=1); fout = ObjectFifo(oty, name="out", depth=1)

    def core(fi, fo, k):
        x = fi.acquire(1); y = fo.acquire(1); k(x, y); fo.release(1); fi.release(1)

    w = Worker(core, [fin.cons(), fout.prod(), k], tile=Tile(0, 2))

    def seq(inp, out, ip, oc):
        tg = TaskGroup()
        ip.fill(inp, TensorAccessPattern.full((INW,)), group=tg)
        oc.drain(out, TensorAccessPattern.full((OUTW,)), wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [ity, oty, fin.prod(), fout.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    A = rng.integers(-128, 128, (KB, RS, 8), dtype=np.int8)
    B = rng.integers(-128, 128, (KB, T, NB, 8, 8), dtype=np.int8)
    buf = np.zeros(INW * 4, np.int8); buf[:A.size] = A.reshape(-1); buf[AW * 4:AW * 4 + B.size] = B.reshape(-1)
    a = iron.zeros((INW,), dtype=np.int32, device="npu"); b = iron.zeros((OUTW,), dtype=np.int32, device="npu")
    with a.overwrite() as v: v[...] = buf.view(np.int32)
    probe(a, b, reps=1, variant="mmul2")
    C = b.numpy().copy().reshape(PX // 8, NB, 8, 8).transpose(0, 2, 1, 3).reshape(PX, NB * 8)
    ref = np.zeros((PX, NB * 8), np.int64)
    for kb in range(KB):
        for t in range(T):
            ref += A[kb, t:t + PX].astype(np.int64) @ B[kb, t].transpose(1, 0, 2).reshape(8, NB * 8).astype(np.int64)
    print("max |C - ref| =", np.abs(C - ref).max(), " (C[0,:4]", C[0, :4], "ref", ref[0, :4], ")")
    for reps in (1, 2000):
        probe(a, b, reps=reps, variant="mmul2")
        t0 = time.perf_counter(); n = 10
        for _ in range(n): probe(a, b, reps=reps, variant="mmul2")
        dt = (time.perf_counter() - t0) / n
        print(f"reps={reps}: {dt*1e3:.3f} ms per call", end="")
        if reps > 1:
            per = (dt - base) / (reps - 1) / (PX // 8 * KB * T * NB)
            print(f"  -> {per*1e9:.2f} ns per mmul = {per*1.76e9:.2f} cycles at 1.76 GHz ({512/(per*1.76e9):.0f} MAC/cycle)", end="")
        else:
            base = dt
        print()
