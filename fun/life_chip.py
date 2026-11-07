#!/usr/bin/env python3
"""Game of Life that stays on the chip: 32 AIE cores in a snake, halos over shared memory.

  ../env.sh python life_chip.py                      live in the terminal, 1 generation per frame
  ../env.sh python life_chip.py --bench --gens 1001  1001 generations per NPU call, race numpy

life.py sends every band with a halo of G rows and the host rebuilds the halos between calls.
Here the world goes onto the chip once per call and stays there for GENS generations:

  - core k holds band k (R rows of W cells) in its own memory;
  - every generation it hands its top row to core k-1 and its bottom row to core k+1, through
    ObjectFifos of depth 2, and waits for theirs: those are its halo rows for this generation;
  - the cores form a snake (down column 0, up column 1, ...), so chain neighbours are always
    memory neighbours (north/south, or east/west on AIE2) and the rows go through shared memory;
    only the wrap from the last core back to the first (the torus) goes over a DMA stream;
  - the core ping-pongs between its input and its output buffer; GENS is odd, so the last
    generation lands in the output buffer.
"""

import argparse
import shutil
import sys
import time

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

from life import cpu, draw, levels

ROWS = 4  # core rows per column on NPU2 (rows 2..5)
BAND_MAX = 24 * 1024  # bytes per band: in + out buffers (48 KB) + the edge rows must fit in 64 KB
PAD = 64  # an edge row sits at offset 64 in a W+128 buffer: the c-1 / c+1 loads stay inside it

CELL = """
#include <stdint.h>
#include <aie_api/aie.hpp>
#define CELL(c) do {{ \\
    int cc = (c), l = cc ? cc - 1 : {w} - 1, rt = cc == {w} - 1 ? 0 : cc + 1; \\
    int s = up[l] + up[cc] + up[rt] + mid[l] + mid[rt] + dn[l] + dn[cc] + dn[rt]; \\
    o[cc] = s == 3 || (s == 2 && mid[cc]); \\
}} while (0)
"""

STEP = """
extern "C" {{
/* one generation of a band: src -> dst, with the neighbours' rows above and below */
void life_step(uint8_t *src, uint8_t *dst, uint8_t *above, uint8_t *below) {{
    const auto zero = aie::zeros<uint8_t, 64>();
    const auto one = aie::broadcast<uint8_t, 64>(1);
    const auto three = aie::broadcast<uint8_t, 64>(3);
    for (int r = 0; r < {rows}; r++) {{
        const uint8_t *up = r ? src + (r - 1) * {w} : above + {pad};
        const uint8_t *mid = src + r * {w};
        const uint8_t *dn = r < {rows} - 1 ? src + (r + 1) * {w} : below + {pad};
        uint8_t *o = dst + r * {w};
        for (int c = 0; c < {w}; c += 64) {{
            auto s = aie::add(aie::load_unaligned_v<64>(up + c - 1), aie::load_unaligned_v<64>(up + c));
            s = aie::add(s, aie::load_unaligned_v<64>(up + c + 1));
            s = aie::add(s, aie::load_unaligned_v<64>(mid + c - 1));
            s = aie::add(s, aie::load_unaligned_v<64>(mid + c + 1));
            s = aie::add(s, aie::load_unaligned_v<64>(dn + c - 1));
            s = aie::add(s, aie::load_unaligned_v<64>(dn + c));
            s = aie::add(s, aie::load_unaligned_v<64>(dn + c + 1));
            auto alive = aie::eq(aie::bit_or(s, aie::load_unaligned_v<64>(mid + c)), three);
            aie::store_unaligned_v(o + c, aie::select(zero, one, alive));
        }}
        CELL(0);
        CELL({w} - 1);
    }}
}}
}}
"""

ROW = """
extern "C" {{
/* copy row r of a band into an edge buffer (at offset PAD) */
void life_row(uint8_t *band, uint8_t *edge, int32_t r) {{
    for (int c = 0; c < {w}; c += 64)
        aie::store_unaligned_v(edge + {pad} + c, aie::load_unaligned_v<64>(band + r * {w} + c));
}}
}}
"""


def snake(k):
    """Chain position -> (column, row) of the core: down column 0, up column 1, ..."""
    col, i = divmod(k, ROWS)
    return col, (2 + i) if col % 2 == 0 else (2 + ROWS - 1 - i)


@iron.jit
def life_chip(world: In, out: Out, *, w: CompileTime[int], rows: CompileTime[int], gens: CompileTime[int],
              cols: CompileTime[int]):
    cores = cols * ROWS
    n = cores * rows * w
    band = rows * w
    tensor_ty = np.ndarray[(n,), np.dtype[np.uint8]]
    col_ty = np.ndarray[(ROWS * band,), np.dtype[np.uint8]]
    band_ty = np.ndarray[(band,), np.dtype[np.uint8]]
    edge_ty = np.ndarray[(w + 2 * PAD,), np.dtype[np.uint8]]

    fmt = {"w": w, "rows": rows, "pad": PAD}
    step = ExternalFunction("life_step", source_string=CELL.format(**fmt) + STEP.format(**fmt),
                            arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    row = ExternalFunction("life_row", source_string=CELL.format(**fmt) + ROW.format(**fmt),
                           arg_types=[band_ty, edge_ty, np.dtype(np.int32)])

    # host <-> memtile per column, memtile <-> the column's 4 cores (in chain order)
    col_in = [ObjectFifo(col_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    band_in, band_out = [], []
    for c in range(cols):
        band_in += col_in[c].cons().split([i * band for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS,
                                          obj_types=[band_ty] * ROWS, names=[f"b_in{c}_{i}" for i in range(ROWS)])
        band_out += col_out[c].prod().join([i * band for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS,
                                           obj_types=[band_ty] * ROWS, names=[f"b_out{c}_{i}" for i in range(ROWS)])

    # edge rows: down[k] carries core k's bottom row to core k+1 (its row above),
    # up[k] carries core k's top row to core k-1 (its row below)
    down = [ObjectFifo(edge_ty, name=f"down{k}", depth=2) for k in range(cores)]
    up = [ObjectFifo(edge_ty, name=f"up{k}", depth=2) for k in range(cores)]

    def core(b_in, b_out, to_up, to_down, from_above, from_below, step, row):
        a = b_in.acquire(1)
        b = b_out.acquire(1)

        def generation(src, dst):
            e = to_up.acquire(1)
            row(src, e, 0)
            to_up.release(1)
            e = to_down.acquire(1)
            row(src, e, rows - 1)
            to_down.release(1)
            above = from_above.acquire(1)
            below = from_below.acquire(1)
            step(src, dst, above, below)
            from_above.release(1)
            from_below.release(1)

        for _ in range_(gens // 2):
            generation(a, b)
            generation(b, a)
        generation(a, b)
        b_in.release(1)
        b_out.release(1)

    workers = []
    for k in range(cores):
        workers.append(Worker(core, [band_in[k].cons(), band_out[k].prod(), up[k].prod(), down[k].prod(),
                                     down[(k - 1) % cores].cons(), up[(k + 1) % cores].cons(), step, row],
                              tile=Tile(*snake(k))))

    taps = TensorAccessPattern.full((n,)).partition(cols)
    in_prods = [f.prod() for f in col_in]
    out_conses = [f.cons() for f in col_out]

    def sequence(world, out, in_prods, out_conses):
        # one group: the fills are only freed once the drains (which wait) are done
        tg = TaskGroup()
        for c in range(cols):
            in_prods[c].fill(world, taps[c], group=tg)
        for c in range(cols):
            out_conses[c].drain(out, taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [tensor_ty, tensor_ty, in_prods, out_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class Chip:
    def __init__(self, w, h, gens, cols):
        self.w = max(64, w - w % 64)
        self.cores = cols * ROWS
        self.rows = max(1, min(-(-h // self.cores), BAND_MAX // self.w))
        self.h = self.rows * self.cores
        self.gens, self.cols = gens, cols

    def run(self, cells):
        # the same two XRT buffers every call: allocating fresh ones each frame made the window stutter
        if not hasattr(self, "_in"):
            self._in = iron.zeros((cells.size,), dtype=np.uint8, device="npu")
            self._out = iron.zeros((cells.size,), dtype=np.uint8, device="npu")
        with self._in.overwrite() as a:
            a[...] = cells.ravel()
        t0 = time.perf_counter()
        life_chip(self._in, self._out, w=self.w, rows=self.rows, gens=self.gens, cols=self.cols)
        ms = (time.perf_counter() - t0) * 1000
        return self._out.numpy().reshape(self.h, self.w).copy(), ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=1, help="generations per NPU call, odd")
    p.add_argument("--bench", action="store_true", help="no drawing: the biggest world, race numpy")
    p.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), help="world size for --bench")
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--fill", type=float, default=0.3)
    p.add_argument("--fps", type=float, default=30)
    args = p.parse_args()
    if args.gens % 2 == 0:
        sys.exit("--gens must be odd")

    dev = iron.get_current_device()
    rng = np.random.default_rng(1)
    if args.bench:
        w, h = args.size or (512, 10**6)
        chip = Chip(w, h, args.gens, dev.cols)
        cells = (rng.random((chip.h, chip.w)) < args.fill).astype(np.uint8)
        got, first = chip.run(cells)  # compile + warm up, and the check
        same = np.array_equal(got, cpu(cells, args.gens))
        t0 = time.perf_counter()
        c = cells
        for _ in range(args.calls):
            c, _ = chip.run(c)
        dt = time.perf_counter() - t0
        n = args.gens * args.calls
        t1 = time.perf_counter()
        want = cpu(cells, min(n, 200))
        cpu_dt = (time.perf_counter() - t1) * n / min(n, 200)
        if n <= 1000:  # all calls, so a reused buffer that hands back stale data shows up too
            same = same and np.array_equal(c, want if n <= 200 else cpu(cells, n))
        cups = chip.w * chip.h * n
        print(f"on-chip life: world {chip.w}x{chip.h}, {chip.cores} cores x {chip.rows} rows, "
              f"{args.gens} gens per call, {args.calls} calls")
        print(f"NPU  : {n / dt:9.0f} gen/s  {cups / dt / 1e6:9.1f} Mcell/s")
        print(f"numpy: {n / cpu_dt:9.0f} gen/s  {cups / cpu_dt / 1e6:9.1f} Mcell/s")
        what = "first call and all calls" if n <= 1000 else "first call"
        print(f"{what} match numpy" if same else f"DIFFERENT FROM NUMPY ({what})")
        return

    cols, trows = shutil.get_terminal_size()
    chip = Chip(cols, (trows - 2) * 2, args.gens, dev.cols)
    cells = (rng.random((chip.h, chip.w)) < args.fill).astype(np.uint8)
    heat = np.zeros(cells.shape)
    shown = (trows - 2) * 2
    sys.stdout.write("\x1b[2J\x1b[?25l")
    gen, t_start = 0, time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            new, ms = chip.run(cells)
            if gen < 3 * args.gens and not np.array_equal(new, cpu(cells, args.gens)):
                sys.stdout.write("\x1b[0m\x1b[?25h\n")
                sys.exit(f"NPU and numpy differ at generation {gen}")
            cells, gen = new, gen + args.gens
            rate = gen / (time.perf_counter() - t_start)
            draw(levels(cells, heat)[:shown],
                 f"gen {gen:6d} | alive {int(cells.sum()):5d} | {chip.cores} cores on chip: {ms:5.1f} ms / "
                 f"{args.gens} gens | {rate:5.0f} gen/s | {chip.w}x{chip.h}")
            if not cells.any():
                cells = (rng.random(cells.shape) < args.fill).astype(np.uint8)
            time.sleep(max(0.0, 1 / args.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")


if __name__ == "__main__":
    main()
