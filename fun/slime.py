#!/usr/bin/env python3
"""Slime mould / ant trails as a grid chemotaxis sim on the chip: 32 AIE cores in a snake, halos over shared memory.

  ./env.sh python fun/slime.py              bench, write slime.png
  ./env.sh python fun/slime.py --window     pygame (click = food, N = reseed, q/a w/s e/d r/f t/g tune parameters, ESC)

Per cell: a = organism density, p = pheromone/trail, f = food (constant). Every step (all bf16, 4-neighbour stencil):
  flux from c to neighbour n:  F = upwind(a) * clamp(p_n - p_c) + D * (a_c - a_n)     (conservative)
  a' = max(a - sum F, 0)
  p' = min(decay * (p + diff * laplacian(p)) + depo * a + f, 1)
Density flows up the pheromone gradient, deposits more pheromone, pheromone diffuses: trails form and
sharpen between food sources (slime-mould routing). No gathers, no scatters: a 3x3 stencil, like Life.

World stays on chip for STEPS steps per dispatch, as in life_chip.py: core k holds band k (ROWS rows),
hands its top/bottom rows to its chain neighbours each step, snake placement keeps those in shared memory.
Band row = [a | p | f], each with a 32-cell pad on both sides so the x-1 / x+1 loads stay in the buffer;
pads hold the wrapped edge cells (torus). bf16 travels in int16 containers.
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS_PER_COL = 4
PADC = 32
W = 320
S = W + 2 * PADC          # row stride (cells)
ROWS = 8                  # band rows per core -> 32 cores = 256 rows
FIELDS = 3                # a, p, f
PARAMS = dict(D=0.08, L=0.12, decay=0.995, diff=0.24, depo=0.02)


def ksrc(P):
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define S {S}
#define ROWS {ROWS}
#define PADC {PADC}
using bf = bfloat16;
using V = aie::vector<bf, 32>;
using A = aie::accum<accfloat, 32>;
static inline V vb(float f) {{ return aie::broadcast<bf, 32>((bf)f); }}
static inline V tobf(const A &a) {{ return a.template to_vector<bf>(); }}
static inline V ld(const bf *p) {{ return aie::load_v<32>(p); }}
static inline V ldu(const bf *p) {{ return aie::load_unaligned_v<32>(p); }}

/* one neighbour's contribution: flux out of c towards n (negative = in) */
static inline A flux(A acc_a, const V &ac, const V &pc, const V &an, const V &pn, const V &L, const V &negL, const V &D) {{
    V dp = aie::min(aie::max(aie::sub(pn, pc), negL), L);
    V upw = aie::select(an, ac, aie::gt(dp, vb(0.0f)));   // density on the upwind side
    acc_a = aie::msc(acc_a, upw, dp);
    acc_a = aie::msc(acc_a, ac, D);
    acc_a = aie::mac(acc_a, an, D);
    return acc_a;
}}

"""


def kstep(P):
    return ksrc(P) + f"""
extern "C" {{
void slime_step(int16_t *src_, int16_t *dst_, int16_t *above_, int16_t *below_) {{
    const bf *src = (const bf *)src_; bf *dst = (bf *)dst_;
    const bf *above = (const bf *)above_, *below = (const bf *)below_;
    const V L = vb({P['L']}f), negL = vb(-{P['L']}f), D = vb({P['D']}f);
    const V pcen = vb({P['decay'] * (1 - 4 * P['diff'])}f), pnb = vb({P['decay'] * P['diff']}f);
    const V depo = vb({P['depo']}f), one = vb(1.0f), zero = vb(0.0f);
    aie::set_rounding(aie::rounding_mode::conv_even);   // default truncates: 0.3% of the mass vanished per step
    for (int r = 0; r < ROWS; r++) {{
        const bf *a = src + r * 3 * S, *p = a + S, *f = a + 2 * S;
        const bf *ua = r ? a - 3 * S : above;            // edge buffers hold [a | p] of one row
        const bf *da = r < ROWS - 1 ? a + 3 * S : below;
        const bf *up = ua + S, *dp_ = da + S;
        bf *oa = dst + r * 3 * S, *op = oa + S;
        for (int x = PADC; x < PADC + W; x += 32) {{
            V ac = ld(a + x), pc = ld(p + x);
            A acc_a = aie::mul(ac, one);
            acc_a = flux(acc_a, ac, pc, ldu(a + x - 1), ldu(p + x - 1), L, negL, D);
            acc_a = flux(acc_a, ac, pc, ldu(a + x + 1), ldu(p + x + 1), L, negL, D);
            V au = ld(ua + x), pu = ld(up + x), ad = ld(da + x), pd = ld(dp_ + x);
            acc_a = flux(acc_a, ac, pc, au, pu, L, negL, D);
            acc_a = flux(acc_a, ac, pc, ad, pd, L, negL, D);
            A acc_p = aie::mul(pc, pcen);
            acc_p = aie::mac(acc_p, ldu(p + x - 1), pnb);
            acc_p = aie::mac(acc_p, ldu(p + x + 1), pnb);
            acc_p = aie::mac(acc_p, pu, pnb);
            acc_p = aie::mac(acc_p, pd, pnb);
            acc_p = aie::mac(acc_p, ac, depo);
            acc_p = aie::mac(acc_p, ld(f + x), one);
            aie::store_v(oa + x, aie::max(tobf(acc_a), zero));
            aie::store_v(op + x, aie::min(tobf(acc_p), one));
        }}
        // food rides along in the band and must survive the ping-pong: copy its row
        for (int x = 0; x < S; x += 32) aie::store_v(dst_ + r * 3 * S + 2 * S + x, aie::load_v<32>(src_ + r * 3 * S + 2 * S + x));
        // wrap pads: x = -1 is cell W-1, x = W is cell 0 (plain 16-bit moves, no float math)
        int16_t *oa16 = dst_ + r * 3 * S, *op16 = oa16 + S;
        oa16[PADC - 1] = oa16[PADC + W - 1]; oa16[PADC + W] = oa16[PADC];
        op16[PADC - 1] = op16[PADC + W - 1]; op16[PADC + W] = op16[PADC];
    }}
}}
}}
"""


def krow(P):
    return ksrc(P) + f"""
extern "C" {{
/* copy row r's [a | p] (2S cells) into an edge buffer */
void slime_row(int16_t *band, int16_t *edge, int32_t r) {{
    const int16_t *s = band + r * 3 * S;
    for (int i = 0; i < 2 * S; i += 32)
        aie::store_v(edge + i, aie::load_v<32>(s + i));
}}
}}
"""


KEYS = {"q": ("D", 1.25), "a": ("D", 0.8), "w": ("decay", 1.002), "s": ("decay", 0.998), "e": ("diff", 1.1), "d": ("diff", 0.9),
        "r": ("depo", 1.25), "f": ("depo", 0.8), "t": ("L", 1.25), "g": ("L", 0.8)}


def snake(k):
    col, i = divmod(k, ROWS_PER_COL)
    return col, (2 + i) if col % 2 == 0 else (2 + ROWS_PER_COL - 1 - i)


@iron.jit
def slime_chip(world: In, out: Out, *, steps: CompileTime[int], cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS_PER_COL
    band = ROWS * FIELDS * S
    n = cores * band
    tensor_ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS_PER_COL * band,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(2 * S,), np.dtype[np.int16]]
    step = ExternalFunction("slime_step", source_string=kstep(PARAMS), arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    row = ExternalFunction("slime_row", source_string=krow(PARAMS), arg_types=[band_ty, edge_ty, np.dtype(np.int32)])

    col_in = [ObjectFifo(col_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    band_in, band_out = [], []
    for c in range(cols):
        kw = dict(tile=Tile(c, 1), depths=[1] * ROWS_PER_COL, obj_types=[band_ty] * ROWS_PER_COL)
        band_in += col_in[c].cons().split([i * band for i in range(ROWS_PER_COL)], names=[f"b_in{c}_{i}" for i in range(ROWS_PER_COL)], **kw)
        band_out += col_out[c].prod().join([i * band for i in range(ROWS_PER_COL)], names=[f"b_out{c}_{i}" for i in range(ROWS_PER_COL)], **kw)
    down = [ObjectFifo(edge_ty, name=f"down{k}", depth=2) for k in range(cores)]
    up = [ObjectFifo(edge_ty, name=f"up{k}", depth=2) for k in range(cores)]

    def core(b_in, b_out, to_up, to_down, from_above, from_below, step, row):
        a = b_in.acquire(1)
        b = b_out.acquire(1)

        def generation(s, d):
            e = to_up.acquire(1); row(s, e, 0); to_up.release(1)
            e = to_down.acquire(1); row(s, e, ROWS - 1); to_down.release(1)
            above = from_above.acquire(1); below = from_below.acquire(1)
            step(s, d, above, below)
            from_above.release(1); from_below.release(1)

        for _ in range_(steps // 2):
            generation(a, b)
            generation(b, a)
        generation(a, b)
        b_in.release(1); b_out.release(1)

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


def f2bf(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x8000) >> 16).astype(np.uint16).view(np.int16)


def bf2f(x):
    return (np.asarray(x).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


class Slime:
    def __init__(self, steps=15, seed=1):
        self.cols = iron.get_current_device().cols
        self.cores = self.cols * ROWS_PER_COL
        self.H = self.cores * ROWS
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.a = np.zeros((self.H, W), np.float32)
        self.p = np.zeros((self.H, W), np.float32)
        self.food = np.zeros((self.H, W), np.float32)
        self.reseed()
        n = self.cores * ROWS * FIELDS * S
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")
        self.dirty = True

    def reseed(self):
        self.a[:] = self.rng.random((self.H, W), dtype=np.float32) * 0.02
        self.p[:] = 0
        self.food[:] = 0
        yy, xx = np.ogrid[0:self.H, 0:W]
        for _ in range(7):
            x, y = self.rng.integers(20, W - 20), self.rng.integers(20, self.H - 20)
            self.add_food(x, y)
            self.a[(xx - x) ** 2 + (yy - y) ** 2 <= 100] = 1.0     # the organism starts on the food
        self.dirty = True

    def add_food(self, x, y, r=5, v=0.3):
        yy, xx = np.ogrid[0:self.H, 0:W]
        self.food[(xx - x) ** 2 + (yy - y) ** 2 <= r * r] = v
        self.dirty = True

    def pack(self):
        buf = np.zeros((self.H, FIELDS, S), np.float32)
        for i, fld in enumerate((self.a, self.p, self.food)):
            buf[:, i, PADC:PADC + W] = fld
            buf[:, i, PADC - 1] = fld[:, -1]
            buf[:, i, PADC + W] = fld[:, 0]
        return f2bf(buf.ravel())

    def unpack(self, raw):
        buf = bf2f(raw).reshape(self.H, FIELDS, S)
        self.a[:] = buf[:, 0, PADC:PADC + W]
        self.p[:] = buf[:, 1, PADC:PADC + W]

    def run(self):
        """steps generations; the output buffer becomes next call's input (food rides along in the band)."""
        if self.dirty:
            with self._in.overwrite() as x:
                x[...] = self.pack()
            self.dirty = False
        t0 = time.perf_counter()
        slime_chip(self._in, self._out, steps=self.steps, cols=self.cols, variant="slime2-" + "-".join(f"{k}{v}" for k, v in PARAMS.items()))
        ms = (time.perf_counter() - t0) * 1e3
        self.unpack(self._out.numpy())
        self._in, self._out = self._out, self._in
        return ms

    def rgb(self):
        p = np.clip(self.p, 0, 1); a = np.clip(self.a * 3, 0, 1)
        img = np.zeros((self.H, W, 3), np.float32)
        img[..., 0] = 0.1 * p + 0.9 * a            # slime: warm white
        img[..., 1] = 0.5 * p + 0.8 * a
        img[..., 2] = 0.9 * p ** 0.5 + 0.3 * a     # trail: blue glow
        img[self.food > 0] = (1.0, 0.3, 0.2)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def reference_step(a, p, f, P):
    """numpy, float32, same scheme, for a sanity check of one step (torus)."""
    def sh(x, dy, dx): return np.roll(np.roll(x, dy, 0), dx, 1)
    acc = a.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        an, pn = sh(a, dy, dx), sh(p, dy, dx)
        dp = np.clip(pn - p, -P["L"], P["L"])
        upw = np.where(dp > 0, a, an)
        acc -= upw * dp + P["D"] * (a - an)
    lap = sh(p, 0, 1) + sh(p, 0, -1) + sh(p, 1, 0) + sh(p, -1, 0) - 4 * p
    pn = np.minimum(P["decay"] * (p + P["diff"] * lap) + P["depo"] * a + f, 1)
    return np.maximum(acc, 0), pn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--steps", type=int, default=15, help="steps per NPU call, odd")
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--scale", type=int, default=3)
    a = ap.parse_args()
    if a.steps % 2 == 0:
        sys.exit("--steps must be odd")
    sim = Slime(a.steps)
    # one-step check against numpy (bf16 vs fp32: expect small differences, no blow-up)
    chk = Slime(1, seed=1)
    a0, p0, f0 = chk.a.copy(), chk.p.copy(), chk.food.copy()
    ms = chk.run()
    ra, rp = reference_step(a0, p0, f0, PARAMS)
    print(f"one step: {ms:.1f} ms (first call, compile); max |a - numpy| {np.abs(chk.a - ra).max():.4f}, "
          f"max |p - numpy| {np.abs(chk.p - rp).max():.4f}, mass {a0.sum():.1f} -> {chk.a.sum():.1f} (numpy {ra.sum():.1f})")
    if a.window:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((W * a.scale, sim.H * a.scale))
        font = pygame.font.SysFont(None, 24)
        gen, t0 = 0, time.perf_counter()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_n:
                    sim.reseed()
                if e.type == pygame.MOUSEBUTTONDOWN:
                    sim.add_food(e.pos[0] // a.scale, e.pos[1] // a.scale)
                if e.type == pygame.KEYDOWN and e.unicode in KEYS:      # live parameter tuning: recompiles (~2 s)
                    k, f = KEYS[e.unicode]
                    PARAMS[k] = round(PARAMS[k] * f, 5)
            ms = sim.run(); gen += a.steps
            surf = pygame.surfarray.make_surface(np.transpose(sim.rgb(), (1, 0, 2)))
            screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
            screen.blit(font.render(f"step {gen}  {a.steps} steps/call {ms:.1f} ms  {gen / (time.perf_counter() - t0):.0f} steps/s  "
                                    f"click: food  N: reseed", True, (255, 255, 255)), (8, 8))
            screen.blit(font.render("  ".join(f"{k} {v}" for k, v in PARAMS.items()) + "   keys: q/a D  w/s decay  e/d diff  r/f depo  t/g L",
                                    True, (255, 255, 0)), (8, 30))
            pygame.display.flip()
    sim.run()
    t0 = time.perf_counter()
    for _ in range(a.calls):
        ms = sim.run()
    dt = time.perf_counter() - t0
    print(f"{W}x{sim.H} on {sim.cores} cores: {a.steps} steps/call {ms:.1f} ms, {a.calls * a.steps / dt:.0f} steps/s, "
          f"{W * sim.H * a.calls * a.steps / dt / 1e6:.0f} Mcell-steps/s")
    import cv2
    cv2.imwrite("slime.png", cv2.resize(sim.rgb()[..., ::-1], (W * a.scale, sim.H * a.scale), interpolation=cv2.INTER_NEAREST))
    print("wrote slime.png")


if __name__ == "__main__":
    main()
