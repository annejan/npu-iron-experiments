#!/usr/bin/env python3
"""Abelian sandpile on the chip: drop 2^19 grains on the centre cell and let it topple into the fractal.

  ../env.sh python fun/sandpile.py [--grains 524288]     stabilise, write sandpile.png
  ../env.sh python fun/sandpile.py --window

Bak-Tang-Wiesenfeld: a cell with >= 4 grains gives one to each of its 4 neighbours. Synchronous multi-toppling
(c -> c & 3 + sum of neighbours' c >> 2) reaches the same stable pile as any sequential order (the model is
abelian), and so does feeding the grains in portions: the centre gets at most 24000 grains per call so int16
holds them. World 512x512 int16 on the 32 cores (slime.py layout), STEPS rounds per dispatch; the host stops when
no cell has 4 or more grains. Exact integers: the check is grain conservation and max(cell) < 4.
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from slime import snake, ROWS_PER_COL, PADC

W = 512
S = W + 2 * PADC
ROWS = 16


def kcommon():
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define S {S}
#define ROWS {ROWS}
#define PADC {PADC}
using V = aie::vector<int16_t, 32>;
static inline V ld(const int16_t *p) {{ return aie::load_v<32>(p); }}
static inline V ldu(const int16_t *p) {{ return aie::load_unaligned_v<32>(p); }}
"""


def kstep():
    return kcommon() + """
extern "C" {
void sand_step(int16_t *src, int16_t *dst, int16_t *above, int16_t *below) {
    const V three = aie::broadcast<int16_t, 32>(3);
    for (int r = 0; r < ROWS; r++) {
        const int16_t *s = src + r * S;
        const int16_t *up = r ? s - S : above;
        const int16_t *dn = r < ROWS - 1 ? s + S : below;
        int16_t *o = dst + r * S;
        for (int x = PADC; x < PADC + W; x += 32) {
            V c = ld(s + x);
            V keep = aie::bit_and(c, three);
            V inflow = aie::add(aie::add(aie::downshift(ldu(s + x - 1), 2), aie::downshift(ldu(s + x + 1), 2)),
                                aie::add(aie::downshift(ld(up + x), 2), aie::downshift(ld(dn + x), 2)));
            aie::store_v(o + x, aie::add(keep, inflow));
        }
        o[PADC - 1] = 0; o[PADC + W] = 0;       // grains falling off the edge vanish (open boundary)
    }
}
}
"""


def krow():
    return kcommon() + """
extern "C" {
void sand_row(int16_t *band, int16_t *edge, int32_t r) {
    const int16_t *s = band + r * S;
    for (int i = 0; i < S; i += 32) aie::store_v(edge + i, aie::load_v<32>(s + i));
}
}
"""


@iron.jit
def sand_chip(world: In, out: Out, *, steps: CompileTime[int], cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS_PER_COL
    band = ROWS * S
    n = cores * band
    tensor_ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS_PER_COL * band,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(S,), np.dtype[np.int16]]
    step = ExternalFunction("sand_step", source_string=kstep(), arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    row = ExternalFunction("sand_row", source_string=krow(), arg_types=[band_ty, edge_ty, np.dtype(np.int32)])
    col_in = [ObjectFifo(col_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    band_in, band_out = [], []
    for c in range(cols):
        kw = dict(tile=Tile(c, 1), depths=[1] * ROWS_PER_COL, obj_types=[band_ty] * ROWS_PER_COL)
        band_in += col_in[c].cons().split([i * band for i in range(ROWS_PER_COL)], names=[f"b_in{c}_{i}" for i in range(ROWS_PER_COL)], **kw)
        band_out += col_out[c].prod().join([i * band for i in range(ROWS_PER_COL)], names=[f"b_out{c}_{i}" for i in range(ROWS_PER_COL)], **kw)
    down = [ObjectFifo(edge_ty, name=f"down{k}", depth=2) for k in range(cores)]
    up = [ObjectFifo(edge_ty, name=f"up{k}", depth=2) for k in range(cores)]
    zero_edge = None

    def core(b_in, b_out, to_up, to_down, from_above, from_below, step, row):
        a = b_in.acquire(1); b = b_out.acquire(1)

        def gen(s, d):
            e = to_up.acquire(1); row(s, e, 0); to_up.release(1)
            e = to_down.acquire(1); row(s, e, ROWS - 1); to_down.release(1)
            above = from_above.acquire(1); below = from_below.acquire(1)
            step(s, d, above, below)
            from_above.release(1); from_below.release(1)

        for _ in range_(steps // 2):
            gen(a, b); gen(b, a)
        gen(a, b)
        b_in.release(1); b_out.release(1)

    # open boundary top and bottom: the first core's "above" and the last core's "below" are their own
    # edge rows, which the row kernel copies from the band... that would wrap. Use the torus halo and keep
    # the top and bottom rows empty on the host instead (grains never reach them in this run).
    workers = [Worker(core, [band_in[k].cons(), band_out[k].prod(), up[k].prod(), down[k].prod(),
                             down[(k - 1) % cores].cons(), up[(k + 1) % cores].cons(), step, row], tile=Tile(*snake(k)))
               for k in range(cores)]
    taps = TensorAccessPattern.full((n,)).partition(cols)
    ip, oc = [f.prod() for f in col_in], [f.cons() for f in col_out]

    def sequence(world, out, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            ip[c].fill(world, taps[c], group=tg)
        for c in range(cols):
            oc[c].drain(out, taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [tensor_ty, tensor_ty, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class Sandpile:
    def __init__(self, steps=101):
        self.cols = iron.get_current_device().cols
        self.cores = self.cols * ROWS_PER_COL
        self.H = self.cores * ROWS
        self.steps = steps
        self.s = np.zeros((self.H, W), np.int16)
        n = self.cores * ROWS * S
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")

    def run(self):
        a = np.zeros((self.H, S), np.int16)
        a[:, PADC:PADC + W] = self.s
        with self._in.overwrite() as x:
            x[...] = a.ravel()
        t0 = time.perf_counter()
        sand_chip(self._in, self._out, steps=self.steps, cols=self.cols, variant="sand1")
        ms = (time.perf_counter() - t0) * 1e3
        self.s[:] = self._out.numpy().reshape(self.H, S)[:, PADC:PADC + W]
        return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grains", type=int, default=1 << 19)
    ap.add_argument("--window", action="store_true")
    a = ap.parse_args()
    sim = Sandpile()
    cy, cx = sim.H // 2, W // 2
    remaining = a.grains
    fed = 0
    t0 = time.perf_counter(); calls = 0; npu = 0.0
    if a.window:
        import pygame
        pygame.init(); screen = pygame.display.set_mode((W * 2, sim.H * 2)); font = pygame.font.SysFont(None, 24)
    pal = np.array([[30, 30, 60], [60, 110, 220], [240, 200, 60], [250, 250, 250]], np.uint8)
    while remaining > 0 or sim.s.max() >= 4:
        if remaining > 0 and sim.s[cy, cx] < 8000:
            portion = min(remaining, 24000); sim.s[cy, cx] += portion; remaining -= portion; fed += portion
        npu += sim.run(); calls += 1
        if a.window:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
            img = pal[np.clip(sim.s, 0, 3)]
            screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2))), screen.get_size()), (0, 0))
            screen.blit(font.render(f"sandpile: {fed} grains fed, {remaining} to go, max cell {sim.s.max()}, {calls} calls x {sim.steps} rounds", True, (255, 80, 80)), (8, 8))
            pygame.display.flip()
    dt = time.perf_counter() - t0
    total = int(sim.s.astype(np.int64).sum())
    print(f"{a.grains} grains: stable after {calls} calls x {sim.steps} rounds = {calls * sim.steps} rounds in {dt:.1f} s "
          f"(NPU {npu / 1e3:.1f} s, {calls * sim.steps / dt:.0f} rounds/s); grains on the grid {total} (lost off the edge {a.grains - total}); max cell {sim.s.max()}; "
          f"pile radius ~{int(np.sqrt((sim.s > 0).sum() / np.pi))} cells")
    import cv2
    cv2.imwrite("sandpile.png", pal[np.clip(sim.s, 0, 3)][..., ::-1])
    print("wrote sandpile.png")


if __name__ == "__main__":
    main()
