#!/usr/bin/env python3
"""Integer excitable-media automata on the chip: cyclic CA (demons) and Greenberg-Hastings (spiral waves).

  ../env.sh python fun/spiral.py [--rule cyclic|gh] [--steps 2000]     write spiral_<rule>.png, check 20 steps vs numpy
  ../env.sh python fun/spiral.py --window [--rule gh]

cyclic: n = 16 states, a cell in state s becomes (s + 1) mod n when a 4-neighbour is already in that state.
gh: n = 8 states: 0 rest, 1 excited, 2..7 refractory; rest cells get excited when an 8-neighbour is excited,
all other states advance by one (7 -> 0). Exact integers in int16 lanes, world 512x512 on chip (slime.py layout).
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
RULES = {"cyclic": dict(n=16), "gh": dict(n=8)}


def kcommon():
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define S {S}
#define ROWS {ROWS}
#define PADC {PADC}
using V = aie::vector<int16_t, 32>;
using M = aie::mask<32>;
static inline V ld(const int16_t *p) {{ return aie::load_v<32>(p); }}
static inline V ldu(const int16_t *p) {{ return aie::load_unaligned_v<32>(p); }}
"""


def kstep(rule):
    n = RULES[rule]["n"]
    body = {
        "cyclic": f"""
            V nxt = aie::add(c, one);
            nxt = aie::select(nxt, zero, aie::eq(nxt, nv));
            M hit = aie::eq(l, nxt) | aie::eq(rt, nxt) | aie::eq(u, nxt) | aie::eq(d, nxt);
            aie::store_v(o + x, aie::select(c, nxt, hit));""",
        "gh": f"""
            V adv = aie::add(c, one);
            adv = aie::select(adv, zero, aie::eq(adv, nv));
            adv = aie::select(adv, zero, aie::eq(c, zero));            // rest stays at rest unless excited
            M ex = aie::eq(l, one) | aie::eq(rt, one) | aie::eq(u, one) | aie::eq(d, one)
                 | aie::eq(ldu(up + x - 1), one) | aie::eq(ldu(up + x + 1), one) | aie::eq(ldu(dn + x - 1), one) | aie::eq(ldu(dn + x + 1), one);
            M fire = aie::eq(c, zero) & ex;
            aie::store_v(o + x, aie::select(adv, one, fire));""",
    }[rule]
    return kcommon() + f"""
extern "C" {{
void ca_step(int16_t *src, int16_t *dst, int16_t *above, int16_t *below) {{
    const V one = aie::broadcast<int16_t, 32>(1), zero = aie::zeros<int16_t, 32>(), nv = aie::broadcast<int16_t, 32>({n});
    for (int r = 0; r < ROWS; r++) {{
        const int16_t *s = src + r * S;
        const int16_t *up = r ? s - S : above, *dn = r < ROWS - 1 ? s + S : below;
        int16_t *o = dst + r * S;
        for (int x = PADC; x < PADC + W; x += 32) {{
            V c = ld(s + x), l = ldu(s + x - 1), rt = ldu(s + x + 1), u = ld(up + x), d = ld(dn + x);
            {body}
        }}
        o[PADC - 1] = o[PADC + W - 1]; o[PADC + W] = o[PADC];
    }}
}}
}}
"""


def krow():
    return kcommon() + """
extern "C" {
void ca_row(int16_t *band, int16_t *edge, int32_t r) {
    const int16_t *s = band + r * S;
    for (int i = 0; i < S; i += 32) aie::store_v(edge + i, aie::load_v<32>(s + i));
}
}
"""


@iron.jit
def ca_chip(world: In, out: Out, *, steps: CompileTime[int], cols: CompileTime[int], rule: CompileTime[str], variant: CompileTime[str]):
    cores = cols * ROWS_PER_COL
    band = ROWS * S
    n = cores * band
    tensor_ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS_PER_COL * band,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(S,), np.dtype[np.int16]]
    step = ExternalFunction("ca_step", source_string=kstep(rule), arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    row = ExternalFunction("ca_row", source_string=krow(), arg_types=[band_ty, edge_ty, np.dtype(np.int32)])
    col_in = [ObjectFifo(col_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    band_in, band_out = [], []
    for c in range(cols):
        kw = dict(tile=Tile(c, 1), depths=[1] * ROWS_PER_COL, obj_types=[band_ty] * ROWS_PER_COL)
        band_in += col_in[c].cons().split([i * band for i in range(ROWS_PER_COL)], names=[f"b_in{c}_{i}" for i in range(ROWS_PER_COL)], **kw)
        band_out += col_out[c].prod().join([i * band for i in range(ROWS_PER_COL)], names=[f"b_out{c}_{i}" for i in range(ROWS_PER_COL)], **kw)
    down = [ObjectFifo(edge_ty, name=f"down{j}", depth=2) for j in range(cores)]
    up = [ObjectFifo(edge_ty, name=f"up{j}", depth=2) for j in range(cores)]

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

    workers = [Worker(core, [band_in[j].cons(), band_out[j].prod(), up[j].prod(), down[j].prod(),
                             down[(j - 1) % cores].cons(), up[(j + 1) % cores].cons(), step, row], tile=Tile(*snake(j))) for j in range(cores)]
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


class CA:
    def __init__(self, rule="cyclic", steps=21, seed=1):
        self.cols = iron.get_current_device().cols
        self.cores = self.cols * ROWS_PER_COL
        self.H = self.cores * ROWS
        self.rule, self.steps = rule, steps
        self.n = RULES[rule]["n"]
        rng = np.random.default_rng(seed)
        self.s = rng.integers(0, self.n, (self.H, W)).astype(np.int16)
        if rule == "gh":
            # a broken wavefront: half a column excited with a refractory column behind it; the free end curls into a spiral
            self.s[:] = 0
            self.s[:self.H // 2, 100] = 1; self.s[:self.H // 2, 99] = 2
            self.s[self.H // 2:, 350] = 1; self.s[self.H // 2:, 351] = 2
        n = self.cores * ROWS * S
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")

    def run(self):
        a = np.zeros((self.H, S), np.int16)
        a[:, PADC:PADC + W] = self.s; a[:, PADC - 1] = self.s[:, -1]; a[:, PADC + W] = self.s[:, 0]
        with self._in.overwrite() as x:
            x[...] = a.ravel()
        t0 = time.perf_counter()
        ca_chip(self._in, self._out, steps=self.steps, cols=self.cols, rule=self.rule, variant="ca1")
        ms = (time.perf_counter() - t0) * 1e3
        self.s[:] = self._out.numpy().reshape(self.H, S)[:, PADC:PADC + W]
        return ms

    def rgb(self):
        t = self.s.astype(np.float32) / (self.n - 1)
        import colorsys
        pal = np.array([colorsys.hsv_to_rgb(i / self.n, 0.85, 1.0) for i in range(self.n)], np.float32)
        if self.rule == "gh":
            pal = np.array([[0.05, 0.05, 0.12]] + [[1, 0.95, 0.6]] + [[0.1 + 0.5 * (1 - k / 6), 0.1, 0.5 * (1 - k / 6) + 0.2] for k in range(self.n - 2)], np.float32)
        return (pal[self.s] * 255).astype(np.uint8)


def reference(s, rule, n):
    sh = lambda a, dy, dx: np.roll(np.roll(a, dy, 0), dx, 1)
    l, r, u, d = sh(s, 0, 1), sh(s, 0, -1), sh(s, 1, 0), sh(s, -1, 0)
    if rule == "cyclic":
        nxt = (s + 1) % n
        hit = (l == nxt) | (r == nxt) | (u == nxt) | (d == nxt)
        return np.where(hit, nxt, s).astype(np.int16)
    ex = (l == 1) | (r == 1) | (u == 1) | (d == 1) | (sh(s, 1, 1) == 1) | (sh(s, 1, -1) == 1) | (sh(s, -1, 1) == 1) | (sh(s, -1, -1) == 1)
    adv = np.where(s == 0, 0, (s + 1) % n)
    return np.where((s == 0) & ex, 1, adv).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=tuple(RULES), default="cyclic")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--window", action="store_true")
    a = ap.parse_args()
    sim = CA(a.rule)
    ref = sim.s.copy()
    sim.run()
    for _ in range(sim.steps): ref = reference(ref, a.rule, sim.n)
    print(f"{a.rule}: after {sim.steps} steps NPU vs numpy: {'bit-exact' if np.array_equal(sim.s, ref) else 'MISMATCH ' + str(int((sim.s != ref).sum()))}")
    if a.window:
        import pygame
        pygame.init(); screen = pygame.display.set_mode((W * 2, sim.H * 2)); font = pygame.font.SysFont(None, 24)
        n, t0 = 0, time.perf_counter()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
            ms = sim.run(); n += sim.steps
            screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(sim.rgb(), (1, 0, 2))), screen.get_size()), (0, 0))
            screen.blit(font.render(f"{a.rule} CA  step {n}  {sim.steps} steps {ms:.1f} ms  {n / (time.perf_counter() - t0):.0f} steps/s", True, (255, 255, 255)), (8, 8))
            pygame.display.flip()
    t0 = time.perf_counter(); calls = a.steps // sim.steps
    for _ in range(calls): sim.run()
    dt = time.perf_counter() - t0
    print(f"{W}x{sim.H}: {calls * sim.steps / dt:.0f} steps/s ({W * sim.H * calls * sim.steps / dt / 1e9:.1f} Gcell-steps/s)")
    import cv2
    cv2.imwrite(f"spiral_{a.rule}.png", sim.rgb()[..., ::-1])
    print(f"wrote spiral_{a.rule}.png")


if __name__ == "__main__":
    main()
