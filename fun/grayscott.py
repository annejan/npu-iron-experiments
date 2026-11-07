#!/usr/bin/env python3
"""Gray-Scott reaction-diffusion on the chip (Pearson's patterns), int16 Q14, world in L1 like slime.py.

  ./env.sh python fun/grayscott.py [--F 0.037 --k 0.06] [--steps 8000]    write grayscott.png
  ./env.sh python fun/grayscott.py --window                                keys: f/F and k/K move through Pearson's map

u_t = Du lap u - u v^2 + F (1 - u);  v_t = Dv lap v + u v^2 - (F + k) v,  Du = 0.16, Dv = 0.08, dt = 1 (the classic
Karl Sims / Pearson scaling with the 5-point Laplacian). Two int16 Q14 fields per band row [u | v]; the reaction term
is two int16 x int16 products with shifts. Default (F, k) = (0.037, 0.060): worms and coral; (0.0545, 0.062): mitosis;
(0.014, 0.054): chaos; (0.030, 0.062): solitons.

P. Gray and S. K. Scott, Chem. Eng. Sci. 39:1087, 1984; J. E. Pearson, "Complex Patterns in a Simple System", Science
261:189-192, 1993; the (F, k) presets follow Pearson's map as charted by R. Munafo (mrob.com/pub/comp/xmorphia) and
K. Sims (karlsims.com/rd.html).
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
ROWS = 8                  # band rows per core (2 fields) -> 256 rows
FIELDS = 2
Q = 14
DU, DV = 0.16, 0.08


def kcommon():
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define S {S}
#define ROWS {ROWS}
#define PADC {PADC}
using V = aie::vector<int16_t, 32>;
using A = aie::accum<acc32, 32>;
static inline V ld(const int16_t *p) {{ return aie::load_v<32>(p); }}
static inline V ldu(const int16_t *p) {{ return aie::load_unaligned_v<32>(p); }}
"""


def kstep(F, k):
    du, dv = round(DU * 16384), round(DV * 16384)
    f_q, fk_q = round(F * 16384), round((F + k) * 16384)
    return kcommon() + f"""
extern "C" {{
void gs_step(int16_t *src, int16_t *dst, int16_t *above, int16_t *below) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    const V one = aie::broadcast<int16_t, 32>(16384), zero = aie::zeros<int16_t, 32>();
    for (int r = 0; r < ROWS; r++) {{
        const int16_t *u = src + r * 2 * S, *v = u + S;
        const int16_t *uu = r ? u - 2 * S : above, *ud = r < ROWS - 1 ? u + 2 * S : below;
        const int16_t *vu = uu + S, *vd = ud + S;
        int16_t *ou = dst + r * 2 * S, *ov = ou + S;
        for (int x = PADC; x < PADC + W; x += 32) {{
            V uc = ld(u + x), vc = ld(v + x);
            // Laplacians in Q14 (sum of 4 neighbours - 4 c), kept in the accumulator
            A lu = aie::mul(ldu(u + x - 1), (int16_t){du}); lu = aie::mac(lu, ldu(u + x + 1), (int16_t){du});
            lu = aie::mac(lu, ld(uu + x), (int16_t){du}); lu = aie::mac(lu, ld(ud + x), (int16_t){du}); lu = aie::msc(lu, uc, (int16_t)(4 * {du}));
            A lv = aie::mul(ldu(v + x - 1), (int16_t){dv}); lv = aie::mac(lv, ldu(v + x + 1), (int16_t){dv});
            lv = aie::mac(lv, ld(vu + x), (int16_t){dv}); lv = aie::mac(lv, ld(vd + x), (int16_t){dv}); lv = aie::msc(lv, vc, (int16_t)(4 * {dv}));
            // reaction u v^2 in Q14: (v*v)>>14 then *u >>14
            V vv = aie::mul(vc, vc).to_vector<int16_t>(14);
            V uvv = aie::mul(uc, vv).to_vector<int16_t>(14);
            // u' = u + Du lap u - uvv + F (1 - u)
            A au = aie::mac(lu, uc, (int16_t)16384); au = aie::msc(au, uvv, (int16_t)16384);
            au = aie::mac(au, aie::sub(one, uc), (int16_t){f_q});
            // v' = v + Dv lap v + uvv - (F + k) v
            A av = aie::mac(lv, vc, (int16_t)16384); av = aie::mac(av, uvv, (int16_t)16384);
            av = aie::msc(av, vc, (int16_t){fk_q});
            aie::store_v(ou + x, aie::min(aie::max(au.to_vector<int16_t>(14), zero), one));
            aie::store_v(ov + x, aie::min(aie::max(av.to_vector<int16_t>(14), zero), one));
        }}
        int16_t *ou16 = dst + r * 2 * S, *ov16 = ou16 + S;
        ou16[PADC - 1] = ou16[PADC + W - 1]; ou16[PADC + W] = ou16[PADC];
        ov16[PADC - 1] = ov16[PADC + W - 1]; ov16[PADC + W] = ov16[PADC];
    }}
}}
}}
"""


def krow():
    return kcommon() + """
extern "C" {
void gs_row(int16_t *band, int16_t *edge, int32_t r) {
    const int16_t *s = band + r * 2 * S;
    for (int i = 0; i < 2 * S; i += 32) aie::store_v(edge + i, aie::load_v<32>(s + i));
}
}
"""


@iron.jit
def gs_chip(world: In, out: Out, *, steps: CompileTime[int], cols: CompileTime[int], F: CompileTime[float], k: CompileTime[float], variant: CompileTime[str]):
    cores = cols * ROWS_PER_COL
    band = ROWS * FIELDS * S
    n = cores * band
    tensor_ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS_PER_COL * band,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(2 * S,), np.dtype[np.int16]]
    step = ExternalFunction("gs_step", source_string=kstep(F, k), arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    row = ExternalFunction("gs_row", source_string=krow(), arg_types=[band_ty, edge_ty, np.dtype(np.int32)])
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


class GrayScott:
    def __init__(self, F=0.037, k=0.060, steps=101, seed=1):
        self.cols = iron.get_current_device().cols
        self.cores = self.cols * ROWS_PER_COL
        self.H = self.cores * ROWS
        self.F, self.k, self.steps = F, k, steps
        rng = np.random.default_rng(seed)
        self.u = np.ones((self.H, W), np.float32); self.v = np.zeros((self.H, W), np.float32)
        yy, xx = np.mgrid[0:self.H, 0:W]
        for _ in range(12):
            cx, cy = rng.integers(0, W), rng.integers(0, self.H)
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < 36
            self.u[m] = 0.5; self.v[m] = 0.25
        self.v += rng.uniform(0, 0.02, self.v.shape).astype(np.float32)
        n = self.cores * ROWS * FIELDS * S
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")
        self.dirty = True

    def pack(self):
        a = np.zeros((self.H, FIELDS, S), np.float32)
        for i, f in enumerate((self.u, self.v)):
            a[:, i, PADC:PADC + W] = f; a[:, i, PADC - 1] = f[:, -1]; a[:, i, PADC + W] = f[:, 0]
        return np.clip(np.round(a.ravel() * (1 << Q)), 0, 32767).astype(np.int16)

    def run(self):
        if self.dirty:
            with self._in.overwrite() as x:
                x[...] = self.pack()
            self.dirty = False
        t0 = time.perf_counter()
        gs_chip(self._in, self._out, steps=self.steps, cols=self.cols, F=float(self.F), k=float(self.k), variant=f"gs1-{self.F:.4f}-{self.k:.4f}")
        ms = (time.perf_counter() - t0) * 1e3
        a = self._out.numpy().reshape(self.H, FIELDS, S).astype(np.float32) / (1 << Q)
        self.u[:] = a[:, 0, PADC:PADC + W]; self.v[:] = a[:, 1, PADC:PADC + W]
        self._in, self._out = self._out, self._in
        return ms

    def rgb(self):
        t = np.clip(self.v * 2.5, 0, 1)
        img = np.empty((self.H, W, 3), np.float32)
        img[..., 0] = 0.1 + 0.9 * t; img[..., 1] = 0.15 + 0.6 * t ** 1.5; img[..., 2] = 0.35 * (1 - t) + 0.2 * t
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", type=float, default=0.037); ap.add_argument("--k", type=float, default=0.060)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--window", action="store_true")
    a = ap.parse_args()
    sim = GrayScott(a.F, a.k)
    if a.window:
        import pygame
        pygame.init(); screen = pygame.display.set_mode((W * 2, sim.H * 2)); font = pygame.font.SysFont(None, 24)
        n, t0 = 0, time.perf_counter()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
                if e.type == pygame.KEYDOWN and e.unicode in "fFkK":
                    d = 0.001 if e.unicode in "FK" else -0.001
                    if e.unicode in "fF": sim.F = round(sim.F + d, 4)
                    else: sim.k = round(sim.k + d, 4)
            ms = sim.run(); n += sim.steps
            screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(sim.rgb(), (1, 0, 2))), screen.get_size()), (0, 0))
            screen.blit(font.render(f"Gray-Scott F={sim.F:.4f} k={sim.k:.4f}  step {n}  {sim.steps} steps {ms:.1f} ms  {n / (time.perf_counter() - t0):.0f} steps/s   f/F k/K", True, (255, 255, 255)), (8, 8))
            pygame.display.flip()
    t0 = time.perf_counter(); calls = a.steps // sim.steps
    for _ in range(calls): ms = sim.run()
    dt = time.perf_counter() - t0
    print(f"{W}x{sim.H}, F={a.F} k={a.k}: {calls * sim.steps} steps in {dt:.1f} s = {calls * sim.steps / dt:.0f} steps/s ({W * sim.H * calls * sim.steps / dt / 1e9:.1f} Gcell-steps/s); v mean {sim.v.mean():.4f}")
    import cv2
    cv2.imwrite("grayscott.png", cv2.resize(sim.rgb()[..., ::-1], (W * 2, sim.H * 2), interpolation=cv2.INTER_LINEAR))
    print("wrote grayscott.png")


if __name__ == "__main__":
    main()
