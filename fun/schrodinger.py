#!/usr/bin/env python3
"""2-D Schrodinger equation on the NPU: a wave packet through a double slit (Visscher's staggered scheme).

  ../env.sh python fun/schrodinger.py                 run, write schrodinger.png (|psi|^2 and the screen)
  ../env.sh python fun/schrodinger.py --check         30 steps against a numpy float64 reference
  ../env.sh python fun/schrodinger.py --window        pygame

psi = R + i I on a 512x512 grid (hbar = m = dx = 1), H psi = -1/2 lap(psi) + V psi, V a wall with two slits.
Visscher: R(t+dt) = R + dt H I(t+dt/2); I(t+3dt/2) = I(t+dt/2) - dt H R(t+dt). Both fields and V are int32 (R, I
in Q28; V in Q20) streamed through the chip like lbm.py with a 5-row sliding window: a core computes the new R for
rows y-1, y, y+1 from the I window, then the new I for row y from those, so one dispatch is one full step.
dt = 0.1 (stable for dt < 0.5 at dx = 1 without potential). The screen (a column behind the slits) accumulates |psi|^2.
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4; CORES = 32
W = 512; H = 512; R = H // CORES
NF = 3                      # R, I, V
ROW = NF * W                # int32 per row object
Q = 28; QV = 20
DT = 0.1
PARAMS = dict(dt=DT)


def ksrc(P):
    c_lap = round(P["dt"] / 2 * 16384)          # dt/2 in Q14 (Laplacian coefficient of the -1/2 lap term)
    c_v = round(P["dt"] * 16384)                # dt in Q14 for the V psi term; V is Q20 -> total shift 14 + 20 - 28 = 6 extra
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
using V16 = aie::vector<int32_t, 16>;
static inline V16 ld(const int32_t *p) {{ return aie::load_v<16>(p); }}
static inline V16 ldu(const int32_t *p) {{ return aie::load_unaligned_v<16>(p); }}
/* H f at row (f, fm, fp) = -1/2 (f[x-1] + f[x+1] + fm[x] + fp[x] - 4 f[x]) + V f ;  returns dt * H f in Q28 */
static inline V16 dtH(const int32_t *f, const int32_t *fm, const int32_t *fp, const int32_t *v, int x) {{
    V16 c = ld(f + x);
    aie::accum<acc64, 16> acc = aie::mul(c, (int16_t)(4 * {c_lap}));        // +2 dt c
    acc = aie::msc(acc, ldu(f + x - 1), (int16_t){c_lap});
    acc = aie::msc(acc, ldu(f + x + 1), (int16_t){c_lap});
    acc = aie::msc(acc, ld(fm + x), (int16_t){c_lap});
    acc = aie::msc(acc, ld(fp + x), (int16_t){c_lap});
    V16 lap = acc.to_vector<int32_t>(14);
    // V psi: V (Q20) * c (Q28) -> Q48; dt in Q14 -> Q62; want Q28: shift 34. do it in two multiplies: (V * dt_q14) >> 6 -> Q28 scalar-vector mul
    aie::accum<acc64, 16> vv = aie::mul(ld(v + x), (int16_t){c_v});          // Q34
    V16 vdt = vv.to_vector<int32_t>(6);                                        // dt V in Q28
    aie::accum<acc64, 16> pv = aie::mul(c, vdt);                               // Q56 (int32 x int32)
    return aie::add(lap, pv.to_vector<int32_t>(28));
}}
extern "C" {{
/* rows r0..r4 = y-2..y+2, each [R | I | V]; out = row y [R' | I' | V] */
void schro_row(int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3, int32_t *r4, int32_t *out) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    alignas(64) static int32_t Rn[3 * W];            // new R for rows y-1, y, y+1
    const int32_t *rows[5] = {{r0, r1, r2, r3, r4}};
    for (int j = 0; j < 3; j++) {{                    // R' = R + dt H I
        const int32_t *rm = rows[j], *rc = rows[j + 1], *rp = rows[j + 2];
        for (int x = 0; x < W; x += 16) {{
            V16 h = dtH(rc + W, rm + W, rp + W, rc + 2 * W, x);
            aie::store_v(Rn + j * W + x, aie::add(ld(rc + x), h));
        }}
    }}
    for (int x = 0; x < W; x += 16) {{                 // I' = I - dt H R'
        V16 h = dtH(Rn + W, Rn, Rn + 2 * W, r2 + 2 * W, x);
        aie::store_v(out + x, ld(Rn + W + x));
        aie::store_v(out + W + x, aie::sub(ld(r2 + W + x), h));
        aie::store_v(out + 2 * W + x, ld(r2 + 2 * W + x));
    }}
    // walls at x = 0 and x = W-1: psi = 0
    out[0] = out[W - 1] = out[W] = out[2 * W - 1] = 0;
}}
}}
"""


def ext():
    row_ty = np.ndarray[(ROW,), np.dtype[np.int32]]
    return ExternalFunction("schro_row", source_string=ksrc(PARAMS), arg_types=[row_ty] * 6)


@iron.jit
def schro_npu(field: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS
    n = (H + 4) * ROW
    f_ty = np.ndarray[(n,), np.dtype[np.int32]]
    row_ty = np.ndarray[(ROW,), np.dtype[np.int32]]
    quad_ty = np.ndarray[(ROWS * ROW,), np.dtype[np.int32]]
    k = ext()
    col_in = [ObjectFifo(quad_ty, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    ri, ro = [], []
    for c in range(cols):
        ri += col_in[c].cons().split([i * ROW for i in range(ROWS)], tile=Tile(c, 1), depths=[5] * ROWS, obj_types=[row_ty] * ROWS, names=[f"ri{c}_{i}" for i in range(ROWS)])
        ro += col_out[c].prod().join([i * ROW for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS, obj_types=[row_ty] * ROWS, names=[f"ro{c}_{i}" for i in range(ROWS)])

    def core(fi, fo, k):
        for _ in range_(R):
            win = fi.acquire(5)
            o = fo.acquire(1)
            k(win[0], win[1], win[2], win[3], win[4], o)
            fo.release(1); fi.release(1)
        fi.release(4)

    workers = [Worker(core, [ri[j].cons(), ro[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    tin = [TensorAccessPattern((n,), c * ROWS * R * ROW, [R + 4, ROWS, ROW], [ROW, R * ROW, 1]) for c in range(cols)]
    tout = [TensorAccessPattern((n,), (2 + c * ROWS * R) * ROW, [R, ROWS, ROW], [ROW, R * ROW, 1]) for c in range(cols)]
    ip, oc = [f.prod() for f in col_in], [f.cons() for f in col_out]

    def seq(field, out, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            ip[c].fill(field, tin[c], group=tg)
        for c in range(cols):
            oc[c].drain(out, tout[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [f_ty, f_ty, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class Schrodinger:
    def __init__(self):
        self.cols = iron.get_current_device().cols
        self.V = np.zeros((H, W), np.float64)
        self.V[:, W // 2 - 2:W // 2 + 2] = 4.0                       # wall
        for cy in (H // 2 - 24, H // 2 + 24):
            self.V[cy - 5:cy + 5, W // 2 - 2:W // 2 + 2] = 0          # two slits, 10 wide, 48 apart
        self.screen = np.zeros(H)
        self.sx = W // 2 + 150
        y, x = np.mgrid[0:H, 0:W]
        k0, sig = 0.6, 28.0
        env = np.exp(-((x - 150) ** 2 + (y - H / 2) ** 2) / (2 * sig ** 2))
        psi = env * np.exp(1j * k0 * x)
        psi /= np.sqrt((np.abs(psi) ** 2).sum()) / 25          # peak amplitude ~0.3 so Q28 has headroom
        self.Rf, self.If = psi.real.copy(), psi.imag.copy()
        # Visscher offset: I is stored at t + dt/2: one explicit half step backwards is small here; keep I as is
        n = (H + 4) * ROW
        self._in = iron.zeros((n,), dtype=np.int32, device="npu")
        self._out = iron.zeros((n,), dtype=np.int32, device="npu")
        buf = np.zeros((H + 4, NF, W), np.int32)
        buf[2:H + 2, 0] = np.round(self.Rf * (1 << Q)); buf[2:H + 2, 1] = np.round(self.If * (1 << Q))
        buf[:, 2] = np.round(self.V[0] * (1 << QV))                 # halo rows get the same V (wall), psi 0
        buf[2:H + 2, 2] = np.round(self.V * (1 << QV))
        with self._in.overwrite() as a:
            a[...] = buf.ravel()
        self.t = 0

    def step(self):
        t0 = time.perf_counter()
        schro_npu(self._in, self._out, cols=self.cols, variant="schro1-" + str(PARAMS))
        ms = (time.perf_counter() - t0) * 1e3
        self._in, self._out = self._out, self._in                 # halo rows 0,1,H+2,H+3 stay zero (psi) and never read for V
        self.t += 1
        return ms

    def fields(self):
        a = self._in.numpy().reshape(H + 4, NF, W)[2:H + 2]
        return a[:, 0].astype(np.float64) / (1 << Q), a[:, 1].astype(np.float64) / (1 << Q)

    def density(self):
        r, i = self.fields()
        return r * r + i * i


def reference_step(Rf, If, V, dt):
    def Hop(f):
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = f[1:-1, :-2] + f[1:-1, 2:] + f[:-2, 1:-1] + f[2:, 1:-1] - 4 * f[1:-1, 1:-1]
        return -0.5 * lap + V * f
    Rn = Rf + dt * Hop(If)
    In = If - dt * Hop(Rn)
    Rn[:, 0] = Rn[:, -1] = In[:, 0] = In[:, -1] = 0
    return Rn, In


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--steps", type=int, default=1400)
    a = ap.parse_args()
    sim = Schrodinger()
    if a.check:
        Rf, If = sim.fields()
        Rr, Ir = Rf.copy(), If.copy()
        n0 = (Rr ** 2 + Ir ** 2).sum()
        for i in range(30):
            sim.step(); Rr, Ir = reference_step(Rr, Ir, sim.V, DT)
            if i % 10 == 9:
                Rg, Ig = sim.fields()
                err = max(np.abs(Rg - Rr).max(), np.abs(Ig - Ir).max())
                print(f"step {i + 1}: max|err| {err:.2e} ({err * (1 << Q):.0f} LSB); norm NPU {((Rg ** 2 + Ig ** 2).sum() / n0):.6f} ref {((Rr ** 2 + Ir ** 2).sum() / n0):.6f}")
        return
    if a.window:
        import pygame
        pygame.init(); screen = pygame.display.set_mode((W * 2, H * 2)); font = pygame.font.SysFont(None, 24)
        n, t0 = 0, time.perf_counter()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
            for _ in range(4): ms = sim.step()
            n += 4
            d = sim.density(); sim.screen += d[:, sim.sx]
            img = np.zeros((H, W, 3), np.float32); dd = np.clip(d * 12, 0, 1)
            img[..., 0] = dd ** 0.5; img[..., 1] = dd; img[..., 2] = np.clip(sim.V / 4, 0, 1) * 0.6 + dd ** 2 * 0.5
            img[:, sim.sx, :] = np.stack([np.clip(sim.screen / max(sim.screen.max(), 1e-9), 0, 1)] * 3, 1)
            screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.transpose((img * 255).astype(np.uint8), (1, 0, 2))), screen.get_size()), (0, 0))
            screen.blit(font.render(f"2-D Schrodinger, double slit  step {n}  {ms:.2f} ms/step  {n / (time.perf_counter() - t0):.0f} steps/s", True, (255, 255, 255)), (8, 8))
            pygame.display.flip()
    ts = []
    for i in range(a.steps):
        ts.append(sim.step())
        if i % 20 == 19:
            sim.screen += sim.density()[:, sim.sx]
    d = sim.density()
    print(f"{a.steps} steps: {np.median(ts):.2f} ms/step median ({1e3 / np.median(ts):.0f} steps/s); norm {d.sum() / 625:.4f} (1 = conserved)")
    img = np.zeros((H, W, 3), np.float32); dd = np.clip(d * 12, 0, 1)
    img[..., 0] = dd ** 0.5; img[..., 1] = dd; img[..., 2] = np.clip(sim.V / 4, 0, 1) * 0.6 + dd ** 2 * 0.5
    img[:, sim.sx, :] = np.stack([np.clip(sim.screen / max(sim.screen.max(), 1e-9), 0, 1)] * 3, 1)
    import cv2
    cv2.imwrite("schrodinger.png", cv2.resize((img[..., ::-1] * 255).astype(np.uint8), (W * 2, H * 2)))
    print("wrote schrodinger.png")


if __name__ == "__main__":
    main()
