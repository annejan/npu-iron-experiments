#!/usr/bin/env python3
"""Can a core run a data-driven schedule? One column: a broadcast fifo (rows), a split fifo (big) and a join (out); each
core loops over a phase table held in a local Buffer: kind (0 rows-only, 1 big-only, 2 both) and step count, read at run
time, with if_ selecting the body and dynamic objectfifo lowering. Checks the object order and times a step."""
import sys, time
import numpy as np
import aie.iron as iron
from aie.dialects import arith
from aie.extras import types as T
from aie.ir import IndexType
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import else_, if_, range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4; ROWB = 5632; NB = 2560; OUTB = 1408

SRC = r"""
#include <stdint.h>
static int cnt;
extern "C" void step_r(int32_t *rows, int32_t *out, int32_t kind) { if (kind < 0) cnt = 0; out[0] = cnt++; out[1] = rows[0]; out[2] = -1; out[3] = kind; }
extern "C" void step_b(int32_t *big, int32_t *out, int32_t kind) { out[0] = cnt++; out[1] = -1; out[2] = big[0]; out[3] = kind; }
extern "C" void step_rb(int32_t *rows, int32_t *big, int32_t *out, int32_t kind) { out[0] = cnt++; out[1] = rows[0]; out[2] = big[0]; out[3] = kind; }
"""
DECL = r"""
#include <stdint.h>
extern "C" void step_b(int32_t *big, int32_t *out, int32_t kind);
"""


@iron.jit
def probe(rows_in: In, big_in: In, out: Out, *, phases: CompileTime[str], variant: CompileTime[str]):
    ph = [tuple(map(int, p.split(":"))) for p in phases.split(",")]
    nrow = sum(n for k, n in ph if k in (0, 2)); nbig = sum(n for k, n in ph if k in (1, 2)); nout = sum(n for k, n in ph)
    nrow_b, nbig_b = max(nrow, 1), max(nbig, 1)
    rows_ty = np.ndarray[(ROWB // 4,), np.dtype[np.int32]]; big_ty = np.ndarray[(NB // 4,), np.dtype[np.int32]]
    big4_ty = np.ndarray[(ROWS * NB // 4,), np.dtype[np.int32]]; out_ty = np.ndarray[(OUTB // 4,), np.dtype[np.int32]]
    out4_ty = np.ndarray[(ROWS * OUTB // 4,), np.dtype[np.int32]]
    k_r = ExternalFunction("step_r", source_string=SRC, arg_types=[rows_ty, out_ty, np.dtype(np.int32)])
    k_b = ExternalFunction("step_b", source_string=DECL, arg_types=[big_ty, out_ty, np.dtype(np.int32)])
    k_rb = ExternalFunction("step_rb", source_string=DECL.replace("step_b(int32_t *big,", "step_rb(int32_t *rows, int32_t *big,"), arg_types=[rows_ty, big_ty, out_ty, np.dtype(np.int32)])
    ri = ObjectFifo(rows_ty, name="rin", depth=2)
    rb = ri.cons().forward(tile=Tile(0, 1), depth=2, name="rbc")
    bi = ObjectFifo(big4_ty, name="bin", depth=2)
    bs = bi.cons().split([r * NB // 4 for r in range(ROWS)], tile=Tile(0, 1), depths=[2] * ROWS, obj_types=[big_ty] * ROWS, names=[f"big{r}" for r in range(ROWS)])
    oj = ObjectFifo(out4_ty, name="out", depth=2)
    os_ = oj.prod().join([r * OUTB // 4 for r in range(ROWS)], tile=Tile(0, 1), depths=[2] * ROWS, obj_types=[out_ty] * ROWS, names=[f"o{r}" for r in range(ROWS)])
    kinds = np.array([k for k, n in ph], np.int32); counts = np.array([n for k, n in ph], np.int32)

    def body(rf, bf, of, kr, kb, krb, kt, ct):
        idx = IndexType.get()
        for p in range_(len(ph)):
            kind = kt[p]; n = arith.index_cast(idx, ct[p])
            with if_(arith.cmpi(arith.CmpIPredicate.eq, kind, arith.constant(T.i32(), 0))) as b0:
                for _ in range_(0, n, 1):
                    r = rf.acquire(1); o = of.acquire(1); kr(r, o, kind); of.release(1); rf.release(1)
            with else_(b0):
                with if_(arith.cmpi(arith.CmpIPredicate.eq, kind, arith.constant(T.i32(), 1))) as b1:
                    for _ in range_(0, n, 1):
                        b = bf.acquire(1); o = of.acquire(1); kb(b, o, kind); of.release(1); bf.release(1)
                with else_(b1):
                    for _ in range_(0, n, 1):
                        r = rf.acquire(1); b = bf.acquire(1); o = of.acquire(1); krb(r, b, o, kind); of.release(1); bf.release(1); rf.release(1)

    workers = []
    for r in range(ROWS):
        kt = Buffer(initial_value=kinds, name=f"kinds{r}"); ct = Buffer(initial_value=counts, name=f"counts{r}")
        workers.append(Worker(body, [rb.cons(depth=2), bs[r].cons(), os_[r].prod(), k_r, k_b, k_rb, kt, ct], tile=Tile(0, 2 + r), dynamic_objfifo_lowering=True))

    def seq(rows_in, big_in, out, rp, bp, oc):
        tg = TaskGroup()
        oc.drain(out, TensorAccessPattern.full((nout * ROWS * OUTB // 4,)), wait=True, group=tg)
        if nrow: rp.fill(rows_in, TensorAccessPattern.full((nrow * ROWB // 4,)), group=tg)
        if nbig: bp.fill(big_in, TensorAccessPattern.full((nbig * ROWS * NB // 4,)), group=tg)
        tg.finish()

    rt = Runtime(seq, [np.ndarray[(nrow_b * ROWB // 4,), np.dtype[np.int32]], np.ndarray[(nbig_b * ROWS * NB // 4,), np.dtype[np.int32]],
                       np.ndarray[(nout * ROWS * OUTB // 4,), np.dtype[np.int32]], ri.prod(), bi.prod(), oj.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def run(ph, reps=0):
    spec = ",".join(f"{k}:{n}" for k, n in ph)
    nrow = sum(n for k, n in ph if k in (0, 2)); nbig = sum(n for k, n in ph if k in (1, 2)); nout = sum(n for k, n in ph)
    a = iron.zeros((max(nrow, 1) * ROWB // 4,), dtype=np.int32, device="npu"); b = iron.zeros((max(nbig, 1) * ROWS * NB // 4,), dtype=np.int32, device="npu")
    o = iron.zeros((nout * ROWS * OUTB // 4,), dtype=np.int32, device="npu")
    with a.overwrite() as v: v.reshape(max(nrow, 1), -1)[:, 0] = 1000 + np.arange(max(nrow, 1))
    with b.overwrite() as v: v.reshape(max(nbig, 1), ROWS, -1)[:, :, 0] = 100000 * (1 + np.arange(ROWS))[None, :] + np.arange(max(nbig, 1))[:, None]
    probe(a, b, o, phases=spec, variant="loop1")
    got = o.numpy().reshape(nout, ROWS, -1)[:, :, :4]
    ok = True; ir = ib = 0
    for step, (k, n) in [(s, kn) for s, kn in enumerate([kn for kn in ph for _ in range(kn[1])])]:
        for r in range(ROWS):
            want = [step, 1000 + ir if k in (0, 2) else -1, 100000 * (r + 1) + ib if k in (1, 2) else -1, k]
            if list(got[step, r]) != want:
                if ok: print("mismatch at step", step, "core", r, "got", list(got[step, r]), "want", want)
                ok = False
        ir += k in (0, 2); ib += k in (1, 2)
    t = None
    if reps:
        t0 = time.perf_counter()
        for _ in range(reps): probe(a, b, o, phases=spec, variant="loop1")
        t = (time.perf_counter() - t0) / reps
    return ok, t


if __name__ == "__main__":
    ok, _ = run([(0, 5), (1, 3), (2, 4), (0, 2)])
    print("order check:", "PASS" if ok else "FAIL")
    for kind in (0, 1):
        _, t0 = run([(kind, 8)], reps=20)
        _, t1 = run([(kind, 1008)], reps=20)
        print(f"kind {kind}: {(t1 - t0) / 1000 * 1e6:.2f} us per step ({'rows broadcast 5632 B' if kind == 0 else 'split 4x2560 B'} in, 4x{OUTB} B out)")
