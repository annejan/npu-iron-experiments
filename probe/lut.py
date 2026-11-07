#!/usr/bin/env python3
"""The parallel lookup unit: a 256-entry RGBA palette applied to a 512x512 index image, aie::parallel_lookup
(16 lookups per fetch from a run-duplicated table) against a scalar gather loop. Also a 64x64 RGB565 texture
(4096 entries) as the rotozoom case."""
import sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4; CORES = 32
W = 512; H = 512; R = H // CORES


def ksrc(variant, entries, bits, reps=1):
    """variant: 'lut' or 'scalar'; entries: LUT size; bits: 32 (int32 values, run of 8) or 16 (uint16 values, run of 16)"""
    run = 8 if bits == 32 else 16
    vt = "int32_t" if bits == 32 else "uint16_t"
    at = "int32" if bits == 32 else "uint16"
    body_lut = f"""
    using lut_t = aie::lut<4, {at}, {at}>;
    lut_t table({entries}, lut_ab, lut_cd);
    aie::parallel_lookup<uint16, lut_t, aie::lut_oor_policy::truncate> look(table, 0);
    for (int r = 0; r < {reps}; r++)
    for (int x = 0; x < W; x += 16) {{
        aie::vector<uint16_t, 16> idx = aie::load_v<16>(in + x);
        aie::vector<{vt}, 16> v = look.fetch(idx);
        aie::store_v(out + x, v);
    }}"""
    body_scalar = f"""
    for (int r = 0; r < {reps}; r++)
    for (int x = 0; x < W; x++) out[x] = pal[in[x]];"""
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define ENTRIES {entries}
#define RUN {run}
alignas(64) static {vt} lut_ab[2 * ENTRIES];
alignas(64) static {vt} lut_cd[2 * ENTRIES];
alignas(64) static {vt} pal[ENTRIES];
extern "C" {{
/* mode -1: build the run-duplicated tables from the palette object; else map one row */
void palette_row(uint16_t *__restrict in, {vt} *__restrict palette, {vt} *__restrict out, int32_t mode) {{
    if (mode < 0) {{
        for (unsigned d = 0; d < ENTRIES; d++) {{
            unsigned i = (d / RUN) * (2 * RUN) + d % RUN;
            lut_ab[i] = lut_cd[i] = palette[d];
            lut_ab[i + RUN] = lut_cd[i + RUN] = palette[d];
            pal[d] = palette[d];
        }}
        return;
    }}
    {body_lut if variant == "lut" else body_scalar}
}}
}}
"""


@iron.jit
def palette_npu(idx: In, pal: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str], entries: CompileTime[int], bits: CompileTime[int], reps: CompileTime[int]):
    cores = cols * ROWS
    vdt = np.int32 if bits == 32 else np.uint16
    idx_ty = np.ndarray[(H * W,), np.dtype[np.uint16]]
    out_ty = np.ndarray[(H * W,), np.dtype[vdt]]
    pal_ty = np.ndarray[(entries,), np.dtype[vdt]]
    irow = np.ndarray[(W,), np.dtype[np.uint16]]
    orow = np.ndarray[(W,), np.dtype[vdt]]
    iquad = np.ndarray[(ROWS * W,), np.dtype[np.uint16]]
    oquad = np.ndarray[(ROWS * W,), np.dtype[vdt]]
    k = ExternalFunction("palette_row", source_string=ksrc(variant, entries, bits, reps), arg_types=[irow, pal_ty, orow, np.dtype(np.int32)])
    pal_in = [ObjectFifo(pal_ty, name=f"pal{c}", depth=1) for c in range(cols)]
    col_in = [ObjectFifo(iquad, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(oquad, name=f"out{c}", depth=2) for c in range(cols)]
    ri, ro = [], []
    for c in range(cols):
        ri += col_in[c].cons().split([i * W for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS, obj_types=[irow] * ROWS, names=[f"ri{c}_{i}" for i in range(ROWS)])
        ro += col_out[c].prod().join([i * W for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS, obj_types=[orow] * ROWS, names=[f"ro{c}_{i}" for i in range(ROWS)])

    def core(pf, fi, fo, k):
        p = pf.acquire(1)
        r = fi.acquire(1); o = fo.acquire(1)
        k(r, p, o, -1)
        k(r, p, o, 1)
        fo.release(1); fi.release(1)
        for _ in range_(R - 1):
            r = fi.acquire(1); o = fo.acquire(1)
            k(r, p, o, 1)
            fo.release(1); fi.release(1)
        pf.release(1)

    workers = [Worker(core, [pal_in[j // ROWS].cons(), ri[j].cons(), ro[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    tin = [TensorAccessPattern((H * W,), c * ROWS * R * W, [R, ROWS, W], [W, R * W, 1]) for c in range(cols)]
    pp, ip, oc = [f.prod() for f in pal_in], [f.prod() for f in col_in], [f.cons() for f in col_out]
    pfull = TensorAccessPattern.full((entries,))

    def seq(idx, pal, out, pp, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            pp[c].fill(pal, pfull, group=tg)
            ip[c].fill(idx, tin[c], group=tg)
        for c in range(cols):
            oc[c].drain(out, tin[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [idx_ty, pal_ty, out_ty, pp, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def run(entries, bits, variant, reps=1, iters=15):
    cols = iron.get_current_device().cols
    rng = np.random.default_rng(1)
    vdt = np.int32 if bits == 32 else np.uint16
    idx = rng.integers(0, entries, (H * W,)).astype(np.uint16)
    pal = rng.integers(0, 2 ** 31 - 1 if bits == 32 else 65535, (entries,)).astype(vdt)
    ti, tp = iron.tensor(idx, dtype=np.uint16, device="npu"), iron.tensor(pal, dtype=vdt, device="npu")
    to = iron.zeros((H * W,), dtype=vdt, device="npu")
    palette_npu(ti, tp, to, cols=cols, variant=variant, entries=entries, bits=bits, reps=reps)
    ok = np.array_equal(to.numpy(), pal[idx])
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); palette_npu(ti, tp, to, cols=cols, variant=variant, entries=entries, bits=bits, reps=reps); ts.append(time.perf_counter() - t0)
    t = min(ts)
    print(f"{variant:7s} {entries:5d} x {bits}-bit, x{reps}: {'exact' if ok else 'WRONG'}; {t * 1e3:.2f} ms ({H * W * reps / t / 1e9:.2f} G lookups/s)")


if __name__ == "__main__":
    for entries, bits in ((256, 32), (4096, 16)):
        for variant in ("scalar", "lut"):
            try:
                run(entries, bits, variant, reps=200)
            except Exception as e:
                print(f"{variant} {entries}x{bits}: FAILED: {str(e).splitlines()[0][:200]}")
