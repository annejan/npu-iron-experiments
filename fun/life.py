#!/usr/bin/env python3
"""Game of Life on the NPU: the world is cut in bands, one AIE core per column works its bands.

  ../env.sh python life.py              random soup, as big as the terminal
  ../env.sh python life.py --gens 7     7 generations per NPU call (odd numbers only)
  ../env.sh python life.py --check      compare every NPU step with numpy

Each band travels to its core with GENS extra rows (a halo) above and below. The core runs GENS
generations on its own: every generation the valid part shrinks by one row at each side, so after
GENS generations exactly the band itself is right. The host rebuilds the halos between calls.
The world wraps around (a torus): left-right inside the kernel, top-bottom in the halos.
"""

import argparse
import shutil
import sys
import time

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, Out

MAX_TILE = 12 * 1024  # bytes per band; 4 buffers (in, out, double buffered) must fit in 64 KB


SCALAR_ROW = """
            for (int c = 0; c < {w}; c++) {{
                CELL(c);
            }}"""

# 64 cells per step: 8 shifted loads, 7 adds, and the rule without branches:
# (neighbours | self) == 3  <=>  3 neighbours, or 2 neighbours and alive.
# The loads at c-1 and c+1 run into the next row or the previous one at the edges, so
# column 0 and column W-1 (which wrap around) are redone in scalar code after each row.
VECTOR_ROW = """
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
            CELL({w} - 1);"""


def source(w, rows, gens, vector=True):
    row = (VECTOR_ROW if vector else SCALAR_ROW).format(w=w)
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define CELL(c) do {{ \\
    int cc = (c), l = cc ? cc - 1 : {w} - 1, rt = cc == {w} - 1 ? 0 : cc + 1; \\
    int s = up[l] + up[cc] + up[rt] + mid[l] + mid[rt] + dn[l] + dn[cc] + dn[rt]; \\
    o[cc] = s == 3 || (s == 2 && mid[cc]); \\
}} while (0)
extern "C" {{
void life(uint8_t *a, uint8_t *b, int32_t n) {{
    const auto zero = aie::zeros<uint8_t, 64>();
    const auto one = aie::broadcast<uint8_t, 64>(1);
    const auto three = aie::broadcast<uint8_t, 64>(3);
    (void)zero; (void)one; (void)three;
    uint8_t *src = a, *dst = b;
    for (int g = 1; g <= {gens}; g++) {{
        for (int r = g; r < {rows} - g; r++) {{
            const uint8_t *up = src + (r - 1) * {w}, *mid = src + r * {w}, *dn = src + (r + 1) * {w};
            uint8_t *o = dst + r * {w};{row}
        }}
        uint8_t *t = src; src = dst; dst = t;
    }}
    /* GENS is odd, so the last generation was written to b: the output buffer */
}}
}}
"""


@iron.jit
def life_npu(world: In, out: Out, *, func: CompileTime[ExternalFunction], num_elements: CompileTime[int],
             tile: CompileTime[int]):
    ty = np.ndarray[(num_elements,), np.dtype[np.uint8]]
    return iron.algorithms.transform_parallel(func, ty, tile_size=tile)


class World:
    def __init__(self, w, h, gens, cols, vector=True):
        if vector:
            w -= w % 64  # the vector kernel does 64 cells per step
        self.w, self.gens = w, gens
        # bands: a multiple of the NPU columns, each band small enough for a core's memory
        bands = cols
        while (-(-h // bands) + 2 * gens) * w > MAX_TILE:
            bands += cols
        self.band = -(-h // bands)
        self.bands, self.h = bands, self.band * bands
        self.rows = self.band + 2 * gens
        self.tile = self.rows * w
        t = np.ndarray[(self.tile,), np.dtype[np.uint8]]
        self.func = ExternalFunction("life", source_string=source(w, self.rows, gens, vector),
                                     arg_types=[t, t, np.dtype(np.int32)])
        # which world row goes where in the halo'd bands, computed once
        self.index = np.concatenate([np.arange(i * self.band - gens, (i + 1) * self.band + gens) % self.h
                                     for i in range(bands)])

    def step(self, cells):
        """GENS generations on the NPU. Returns the new world and the NPU call time in ms."""
        a = iron.tensor(np.ascontiguousarray(cells[self.index]).ravel(), dtype=np.uint8, device="npu")
        out = iron.zeros((self.bands * self.tile,), dtype=np.uint8, device="npu")
        t0 = time.perf_counter()
        life_npu(a, out, func=self.func, num_elements=self.bands * self.tile, tile=self.tile)
        ms = (time.perf_counter() - t0) * 1000
        res = out.numpy().reshape(self.bands, self.rows, self.w)[:, self.gens:self.gens + self.band]
        return res.reshape(self.h, self.w).copy(), ms


def cpu(cells, gens):
    for _ in range(gens):
        s = sum(np.roll(np.roll(cells, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if dy or dx)
        cells = ((s == 3) | ((s == 2) & (cells == 1))).astype(np.uint8)
    return cells


# xterm 256-colour codes (short): background, a 6-step trail from dark green to teal, alive.
# Foreground (top pixel) and background (bottom pixel) are sent only when they change.
PALETTE = [16, 22, 23, 29, 30, 36, 37, 156]
LEVELS = len(PALETTE)
FG = np.array([f"\x1b[38;5;{c}m" for c in PALETTE], dtype=object)
BG = np.array([f"\x1b[48;5;{c}m" for c in PALETTE], dtype=object)


def draw(level, status):
    """level: rows x cols of 0..LEVELS-1, two rows per terminal line (half blocks)."""
    out = ["\x1b[H"]
    for top, bot in zip(level[0:-1:2], level[1::2], strict=False):
        tch = np.empty(len(top), bool)
        bch = np.empty(len(top), bool)
        tch[0] = bch[0] = True
        tch[1:] = top[1:] != top[:-1]
        bch[1:] = bot[1:] != bot[:-1]
        starts = np.flatnonzero(tch | bch)
        runs = np.diff(np.append(starts, len(top)))
        codes = np.where(tch[starts], FG[top[starts]], "") + np.where(bch[starts], BG[bot[starts]], "")
        out.extend(c + "▀" * n for c, n in zip(codes.tolist(), runs.tolist(), strict=True))
        out.append("\x1b[0m\n")
    out.append(f"\x1b[0m\x1b[K{status}")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def levels(cells, heat):
    """Living cells bright, the dead leave a fading trail."""
    heat *= 0.82
    heat[cells == 1] = 1.0
    lv = np.rint(heat * (LEVELS - 2)).astype(np.int64)
    lv[cells == 1] = LEVELS - 1
    return lv


def bench(dev, w, h, gens, calls, vector):
    world = World(w, h, gens, dev.cols, vector)
    cells = (np.random.default_rng(1).random((world.h, world.w)) < 0.3).astype(np.uint8)
    world.step(cells)  # compile + warm up
    t0, npu_ms = time.perf_counter(), 0.0
    c = cells
    for _ in range(calls):
        c, ms = world.step(c)
        npu_ms += ms
    total = time.perf_counter() - t0
    t1 = time.perf_counter()
    want = cpu(cells, gens * calls)
    cpu_s = time.perf_counter() - t1
    n = gens * calls
    cups = world.w * world.h * n
    print(f"{'vector' if vector else 'scalar'} kernel, world {world.w}x{world.h}, {gens} gens per call, {world.bands} bands of {world.band} rows "
          f"(+{2 * gens} halo) on {dev.cols} cores")
    print(f"NPU  : {n / total:8.0f} gen/s  {cups / total / 1e6:8.1f} Mcell/s   (in the NPU call: "
          f"{npu_ms / calls:.2f} ms per call; the rest is halos and copies on the host)")
    print(f"numpy: {n / cpu_s:8.0f} gen/s  {cups / cpu_s / 1e6:8.1f} Mcell/s")
    print("same result" if np.array_equal(c, want) else "DIFFERENT RESULT")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=3, help="generations per NPU call, odd (default 3)")
    p.add_argument("--fill", type=float, default=0.3, help="share of living cells at the start")
    p.add_argument("--check", action="store_true", help="check every step against numpy")
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--bench", nargs=2, type=int, metavar=("W", "H"), help="no drawing: race a W x H world")
    p.add_argument("--calls", type=int, default=50, help="NPU calls in --bench")
    p.add_argument("--scalar", action="store_true", help="the plain C kernel instead of the vector one")
    args = p.parse_args()
    if args.gens % 2 == 0:
        sys.exit("--gens must be odd: the kernel ping-pongs and ends in the output buffer on odd counts")

    dev = iron.get_current_device()
    if args.bench:
        return bench(dev, *args.bench, args.gens, args.calls, not args.scalar)
    cols, rows = shutil.get_terminal_size()
    world = World(cols - cols % 4, (rows - 2) * 2, args.gens, dev.cols, not args.scalar)
    rng = np.random.default_rng()
    cells = (rng.random((world.h, world.w)) < args.fill).astype(np.uint8)
    heat = np.zeros(cells.shape)
    shown = (rows - 2) * 2

    sys.stdout.write("\x1b[2J\x1b[?25l")
    gen, t_start = 0, time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            new, ms = world.step(cells)
            if args.check or gen < 3 * args.gens:
                want = cpu(cells, args.gens)
                if not np.array_equal(new, want):
                    bad = np.argwhere(new != want)
                    sys.stdout.write("\x1b[0m\x1b[?25h\n")
                    sys.exit(f"NPU and numpy differ at generation {gen}: {len(bad)} cells, first at {bad[0]}")
            cells, gen = new, gen + args.gens
            if gen == args.gens:
                c0 = time.perf_counter()
                cpu(cells, args.gens)
                cpu_ms = (time.perf_counter() - c0) * 1000
            alive = int(cells.sum())
            rate = gen / (time.perf_counter() - t_start)
            draw(levels(cells, heat)[:shown],
                 f"gen {gen:6d} | alive {alive:5d} | NPU {dev.cols} cores x {world.bands // dev.cols} bands: "
                 f"{ms:5.1f} ms / {args.gens} gens | numpy {cpu_ms:5.1f} ms | {rate:5.0f} gen/s | "
                 f"{world.w}x{world.h}")
            if alive == 0:
                cells = (rng.random(cells.shape) < args.fill).astype(np.uint8)
            time.sleep(max(0.0, 1 / args.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")


if __name__ == "__main__":
    main()
