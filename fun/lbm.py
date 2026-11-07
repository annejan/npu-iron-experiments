#!/usr/bin/env python3
"""Lattice-Boltzmann (D2Q9, BGK) fluid on all 32 AIE cores: flow past a cylinder, vortex street.

  ./env.sh python fun/lbm.py              bench, write lbm.png
  ./env.sh python fun/lbm.py --window     pygame: click/drag = draw obstacles, C = clear, ESC

Nine distributions per cell do not fit the cores' 64 KB with a world worth looking at, so the world lives
in DDR and streams through the chip once per step: each core takes a band of R rows plus one halo row
above and below (the shim DMA gathers that with one 3-D access pattern per column: no host repacking),
keeps a sliding window of three rows in L1 (ObjectFifo depth 3) and writes its R rows back.
Per cell: pull streaming (f_i from x - c_i, out of the three rows), density/velocity, BGK collision,
full-way bounce-back on obstacle cells; inlet on the left, free outlet on the right, periodic in y.
Distributions are stored as deviations g_i = f_i - w_i so bf16 keeps its precision near rest.
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from slime import f2bf, bf2f, ROWS_PER_COL, PADC

W = 384
S = W + 2 * PADC
R = 8                     # rows per core -> 32 cores = 256 rows
H = 32 * R
NF = 10                   # 9 distributions + obstacle mask
ROW = NF * S              # int16 per row object
CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
WT = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]
PARAMS = dict(tau=0.58, fx=0.0, u_in=0.07)
RAMP = 1500               # steps to ramp the inlet from rest (an impulsive start shocks rho by 20% and bf16 noise then blows up)


def ksrc(P):
    omega = 1 / P["tau"]
    # pull source for direction i: row (prev if cy=+1 ... note: y grows downwards in memory, "prev" is y-1)
    def src(i):
        row = {0: "cur", 1: "prev", -1: "next"}[CY[i]]
        dx = -CX[i]
        off = f" + {dx}" if dx else ""
        fn = "ldu" if dx else "ld"
        return f"{fn}({row} + {i} * S + x{off})"
    pulls = "\n".join(f"        V g{i} = {src(i)};" for i in range(9))
    # equilibrium (deviation form): eq_i = w_i * (rd + rho * (3 cu + 4.5 cu^2 - 1.5 u2)), cu = c_i . u
    def cu(i):
        return {(0, 0): None, (1, 0): "ux", (-1, 0): "nux", (0, 1): "uy", (0, -1): "nuy",
                (1, 1): "sp", (-1, -1): "nsp", (-1, 1): "dm", (1, -1): "ndm"}[(CX[i], CY[i])]
    coll = []
    for i in range(9):
        c = cu(i)
        if c is None:
            coll.append(f"        A e{i} = aie::mul(rhob, t0);")
        else:
            coll.append(f"        A e{i} = aie::mul({c}, three); e{i} = aie::mac(e{i}, tobf(aie::mul({c}, {c})), fourhalf); "
                        f"e{i} = aie::msc(e{i}, u2, onehalf); e{i} = aie::mul(rhob, tobf(e{i}));")
        coll.append(f"        e{i} = aie::mac(e{i}, rd, one); A n{i} = aie::mul(g{i}, oneminus); n{i} = aie::mac(n{i}, tobf(e{i}), vb({omega * WT[i]}f));")
        coll.append(f"        aie::store_v(out + {i} * S + x, aie::select(tobf(n{i}), g{OPP[i]}, solid));")
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define S {S}
#define PADC {PADC}
#define W {W}
using bf = bfloat16;
using V = aie::vector<bf, 32>;
using A = aie::accum<accfloat, 32>;
static inline V vb(float f) {{ return aie::broadcast<bf, 32>((bf)f); }}
static inline V tobf(const A &a) {{ return a.template to_vector<bf>(); }}
static inline V ld(const bf *p) {{ return aie::load_v<32>(p); }}
static inline V ldu(const bf *p) {{ return aie::load_unaligned_v<32>(p); }}
static inline V negb(const V &v) {{ return tobf(aie::mul(v, vb(-1.0f))); }}   // aie::neg on bf16 is not legalised by Peano
extern "C" {{
void lbm_step(int16_t *prev_, int16_t *cur_, int16_t *next_, int16_t *out_) {{
    const bf *prev = (const bf *)prev_, *cur = (const bf *)cur_, *next = (const bf *)next_;
    bf *out = (bf *)out_;
    aie::set_rounding(aie::rounding_mode::conv_even);
    const V one = vb(1.0f), three = vb(3.0f), fourhalf = vb(4.5f), onehalf = vb(1.5f), oneminus = vb({1 - omega}f);
    const V fx = vb({P['fx'] * P['tau']}f);
    for (int x = PADC; x < PADC + W; x += 32) {{
{pulls}
        aie::mask<32> solid = aie::gt(ld(cur + 9 * S + x), vb(0.5f));
        A r = aie::mul(g0, one);
        r = aie::mac(r, g1, one); r = aie::mac(r, g2, one); r = aie::mac(r, g3, one); r = aie::mac(r, g4, one);
        r = aie::mac(r, g5, one); r = aie::mac(r, g6, one); r = aie::mac(r, g7, one); r = aie::mac(r, g8, one);
        V rd = tobf(r);                                   // rho - 1
        V rhob = tobf(aie::mac(r, one, one));             // rho
        V invr = aie::inv(rhob);
        A jx = aie::mul(g1, one); jx = aie::msc(jx, g3, one); jx = aie::mac(jx, g5, one); jx = aie::msc(jx, g6, one);
        jx = aie::msc(jx, g7, one); jx = aie::mac(jx, g8, one);
        A jy = aie::mul(g2, one); jy = aie::msc(jy, g4, one); jy = aie::mac(jy, g5, one); jy = aie::mac(jy, g6, one);
        jy = aie::msc(jy, g7, one); jy = aie::msc(jy, g8, one);
        A uxa = aie::mul(tobf(jx), invr); uxa = aie::mac(uxa, one, fx);      // body force: shifted velocity
        V ux = tobf(uxa), uy = tobf(aie::mul(tobf(jy), invr));
        V nux = negb(ux), nuy = negb(uy);
        V sp = tobf(aie::mac(aie::mul(ux, one), uy, one)), nsp = negb(sp);      // ux + uy
        V dm = tobf(aie::msc(aie::mul(uy, one), ux, one)), ndm = negb(dm);      // uy - ux
        A u2a = aie::mul(ux, ux); u2a = aie::mac(u2a, uy, uy);
        V u2 = tobf(u2a);
        A t0a = aie::mul(u2, vb(-1.5f)); V t0 = tobf(t0a);
{chr(10).join(coll)}
        aie::store_v(out + 9 * S + x, ld(cur + 9 * S + x));
    }}
}}
}}
"""


@iron.jit
def lbm_npu(world: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS_PER_COL
    n_in = (H + 2) * ROW
    in_ty = np.ndarray[(n_in,), np.dtype[np.int16]]
    row_ty = np.ndarray[(ROW,), np.dtype[np.int16]]
    quad_ty = np.ndarray[(ROWS_PER_COL * ROW,), np.dtype[np.int16]]
    k = ExternalFunction("lbm_step", source_string=ksrc(PARAMS), arg_types=[row_ty, row_ty, row_ty, row_ty])
    col_in = [ObjectFifo(quad_ty, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    rows_in, rows_out = [], []
    for c in range(cols):
        rows_in += col_in[c].cons().split([i * ROW for i in range(ROWS_PER_COL)], tile=Tile(c, 1), depths=[3] * ROWS_PER_COL,
                                          obj_types=[row_ty] * ROWS_PER_COL, names=[f"ri{c}_{i}" for i in range(ROWS_PER_COL)])
        rows_out += col_out[c].prod().join([i * ROW for i in range(ROWS_PER_COL)], tile=Tile(c, 1), depths=[2] * ROWS_PER_COL,
                                           obj_types=[row_ty] * ROWS_PER_COL, names=[f"ro{c}_{i}" for i in range(ROWS_PER_COL)])

    def core(fi, fo, k):
        for _ in range_(R):
            win = fi.acquire(3)
            o = fo.acquire(1)
            k(win[0], win[1], win[2], o)
            fo.release(1)
            fi.release(1)
        fi.release(2)

    workers = [Worker(core, [rows_in[j].cons(), rows_out[j].prod(), k], tile=Tile(j // ROWS_PER_COL, 2 + j % ROWS_PER_COL))
               for j in range(cores)]
    ip, oc = [f.prod() for f in col_in], [f.cons() for f in col_out]
    # column c: for j in 0..R+1: for core kk in 0..3: padded-world row (c*4+kk)*R + j   (halo rows included)
    taps_in = [TensorAccessPattern((n_in,), c * ROWS_PER_COL * R * ROW, [R + 2, ROWS_PER_COL, ROW], [ROW, R * ROW, 1]) for c in range(cols)]
    taps_out = [TensorAccessPattern((n_in,), (1 + c * ROWS_PER_COL * R) * ROW, [R, ROWS_PER_COL, ROW], [ROW, R * ROW, 1]) for c in range(cols)]

    def seq(world, out, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            ip[c].fill(world, taps_in[c], group=tg)
        for c in range(cols):
            oc[c].drain(out, taps_out[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [in_ty, in_ty, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class LBM:
    def __init__(self):
        self.cols = iron.get_current_device().cols
        self.world = np.zeros((H + 2, NF, S), np.float32)       # row 0 and H+1 are wrap halos
        self.mask = np.zeros((H, W), bool)
        yy, xx = np.mgrid[0:H, 0:W]
        self.mask[(xx - W // 4) ** 2 + (yy - H // 2 + 3) ** 2 < 14 ** 2] = True
        self.reset()
        n = (H + 2) * ROW
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")
        self.upload = True

    def inlet(self, u):
        cu = np.array(CX, np.float32) * u
        self.eq_in = f2bf(np.array(WT, np.float32) * (3 * cu + 4.5 * cu * cu - 1.5 * u * u))     # inlet equilibrium, deviation form

    def reset(self):
        rng = np.random.default_rng(1)
        self.n = 0
        self.inlet(0.0)
        self.world[1:H + 1, :9, PADC:PADC + W] = rng.normal(0, 2e-4, (H, 9, W)).astype(np.float32) * np.array(WT, np.float32)[None, :, None]
        self.set_mask()

    def set_mask(self):
        self.world[1:H + 1, 9, PADC:PADC + W] = self.mask
        self.upload = True

    def fix_halos(self, a):
        """wrap rows (periodic y); inlet at the left (fixed equilibrium), zero-gradient outlet at the right.
        Done on the packed int16 array (H+2, NF, S) between steps. A body force would vanish in bf16 and a
        periodic domain would accelerate without bound, so the flow is driven by the boundary instead."""
        a[:, :9, PADC - 1:PADC + 2] = self.eq_in[None, :, None]
        a[:, :9, PADC + W - 1] = a[:, :9, PADC + W - 2]; a[:, :9, PADC + W] = a[:, :9, PADC + W - 2]
        a[0] = a[H]; a[H + 1] = a[1]

    def step(self):
        if self.upload:
            a = f2bf(self.world.ravel()).reshape(H + 2, NF, S)
            self.fix_halos(a)
            with self._in.overwrite() as x:
                x[...] = a.ravel()
            self.upload = False
        t0 = time.perf_counter()
        lbm_npu(self._in, self._out, cols=self.cols, variant="lbm1-" + "-".join(f"{k}{v}" for k, v in PARAMS.items()))
        ms = (time.perf_counter() - t0) * 1e3
        self.n += 1
        self.inlet(PARAMS["u_in"] * min(1.0, self.n / RAMP))
        with self._out.overwrite() as x:          # the kernel wrote rows 1..H; patch the halos in place
            self.fix_halos(x.reshape(H + 2, NF, S))
        self._in, self._out = self._out, self._in
        return ms

    def fields(self):
        g = bf2f(self._in.numpy()).reshape(H + 2, NF, S)[1:H + 1, :9, PADC:PADC + W]
        rho = 1 + g.sum(1)
        ux = (g * np.array(CX, np.float32)[None, :, None]).sum(1) / rho
        uy = (g * np.array(CY, np.float32)[None, :, None]).sum(1) / rho
        return rho, ux, uy

    def rgb(self):
        rho, ux, uy = self.fields()
        curl = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
        c = np.clip(curl * 60, -1, 1)
        img = np.empty((H, W, 3), np.float32)
        img[..., 0] = 0.5 + 0.5 * c; img[..., 1] = 0.5 - 0.5 * np.abs(c) * 0.6; img[..., 2] = 0.5 - 0.5 * c
        speed = np.clip(np.hypot(ux, uy) * 6, 0, 1)
        img = img * (0.5 + 0.5 * speed)[..., None]
        img[self.mask] = 0
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--per-frame", type=int, default=10)
    a = ap.parse_args()
    sim = LBM()
    ms = sim.step()
    rho, ux, uy = sim.fields()
    print(f"first step (compile): {ms:.1f} ms; rho {rho.min():.4f}..{rho.max():.4f}, |u| max {np.hypot(ux, uy).max():.5f}")
    if a.window:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((W * a.scale, H * a.scale))
        font = pygame.font.SysFont(None, 24)
        n, t0 = 0, time.perf_counter()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                    sim.mask[:] = False; sim.world[1:H + 1, :9] = 0; sim.reset()
                if e.type == pygame.MOUSEBUTTONDOWN or (e.type == pygame.MOUSEMOTION and e.buttons[0]):
                    x, y = e.pos[0] // a.scale, e.pos[1] // a.scale
                    yy, xx = np.ogrid[0:H, 0:W]
                    sim.mask[(xx - x) ** 2 + (yy - y) ** 2 < 36] = True
                    sim.world[1:H + 1, :9] = bf2f(sim._in.numpy()).reshape(H + 2, NF, S)[1:H + 1, :9]
                    sim.set_mask()
            for _ in range(a.per_frame):
                ms = sim.step()
            n += a.per_frame
            surf = pygame.surfarray.make_surface(np.transpose(sim.rgb(), (1, 0, 2)))
            screen.blit(pygame.transform.smoothscale(surf, screen.get_size()), (0, 0))
            rho, ux, uy = sim.fields()
            screen.blit(font.render(f"LBM D2Q9 {W}x{H}  step {n}  {ms:.2f} ms/step  {n / (time.perf_counter() - t0):.0f} steps/s  "
                                    f"|u| max {np.hypot(ux, uy).max():.3f}  rho {rho.min():.3f}..{rho.max():.3f}   drag: obstacles  C: clear",
                                    True, (255, 255, 255)), (8, 8))
            pygame.display.flip()
    t0 = time.perf_counter()
    for i in range(a.steps):
        ms = sim.step()
        if i % 500 == 499:
            rho, ux, uy = sim.fields()
            print(f"step {i + 1}: rho {rho.min():.4f}..{rho.max():.4f}, |u| max {np.hypot(ux, uy).max():.4f}, nan {np.isnan(rho).sum()}")
    dt = time.perf_counter() - t0
    print(f"{W}x{H} D2Q9 on 32 cores: {ms:.2f} ms/step, {a.steps / dt:.0f} steps/s, {W * H * a.steps / dt / 1e6:.0f} Mcell-steps/s")
    import cv2
    cv2.imwrite("lbm.png", cv2.resize(sim.rgb()[..., ::-1], (W * a.scale, H * a.scale), interpolation=cv2.INTER_LINEAR))
    print("wrote lbm.png")


if __name__ == "__main__":
    main()
