#!/usr/bin/env python3
"""Ripple tank on the chip: height-field water, 32 AIE cores in a snake, halos over shared memory.

  ../env.sh python fun/water.py              bench, write water.png
  ../env.sh python fun/water.py --window     pygame: click/drag = drops, R = rain, ESC

Per cell h (height) and v (velocity), int16 Q12 (bf16 cannot integrate small v into h):
  v' = damp * (v + c2 * (h_l + h_r + h_u + h_d - 4 h))
  h' = h + v'
World stays on chip for STEPS steps per dispatch (same plumbing as slime.py / life_chip.py). The host
renders it as water: normals from the height gradient refract a tiled background, plus a specular sun.
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

W = 320
S = W + 2 * PADC
ROWS = 12                 # band rows per core -> 32 cores = 384 rows
FIELDS = 2                # h, v
PARAMS = dict(c2=0.22, damp=0.996)


Q = 12                    # h, v in int16 Q12; coefficients in Q14
C2_Q14 = round(PARAMS["c2"] * 16384)
DAMP_Q14 = round(PARAMS["damp"] * 16384)


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


def kstep(P):
    return kcommon() + f"""
extern "C" {{
/* int16 Q12 heights: bf16 cannot integrate h += v when v is far smaller than h (the ripples freeze) */
void water_step(int16_t *src, int16_t *dst, int16_t *above, int16_t *below) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int r = 0; r < ROWS; r++) {{
        const int16_t *h = src + r * 2 * S, *v = h + S;
        const int16_t *hu = r ? h - 2 * S : above;
        const int16_t *hd = r < ROWS - 1 ? h + 2 * S : below;
        int16_t *oh = dst + r * 2 * S, *ov = oh + S;
        for (int x = PADC; x < PADC + W; x += 32) {{
            V hc = ld(h + x);
            V lap = aie::sub(aie::add(aie::add(ldu(h + x - 1), ldu(h + x + 1)), aie::add(ld(hu + x), ld(hd + x))),
                             aie::add(hc, aie::add(hc, aie::add(hc, hc))));
            A acc = aie::mul(ld(v + x), (int16_t)16384);          // v in Q14 of itself
            acc = aie::mac(acc, lap, (int16_t){C2_Q14});
            V vn = acc.to_vector<int16_t>(14);
            vn = aie::mul(vn, (int16_t){DAMP_Q14}).to_vector<int16_t>(14);
            aie::store_v(ov + x, vn);
            aie::store_v(oh + x, aie::add(hc, vn));
        }}
        int16_t *oh16 = dst + r * 2 * S;
        oh16[PADC - 1] = oh16[PADC]; oh16[PADC + W] = oh16[PADC + W - 1];   // mirror walls
    }}
}}
}}
"""


def krow():
    return kcommon() + f"""
extern "C" {{
void water_row(int16_t *band, int16_t *edge, int32_t r) {{
    const int16_t *s = band + r * 2 * S;
    for (int i = 0; i < S; i += 32)
        aie::store_v(edge + i, aie::load_v<32>(s + i));
}}
}}
"""


@iron.jit
def water_chip(world: In, out: Out, *, steps: CompileTime[int], cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS_PER_COL
    band = ROWS * FIELDS * S
    n = cores * band
    tensor_ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS_PER_COL * band,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(S,), np.dtype[np.int16]]
    step = ExternalFunction("water_step", source_string=kstep(PARAMS), arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    row = ExternalFunction("water_row", source_string=krow(), arg_types=[band_ty, edge_ty, np.dtype(np.int32)])

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

    # top and bottom walls: core 0's "above" is its own top row, the last core's "below" its own bottom row.
    # (its own fifo: it produces the row, then consumes it back; depth 2 keeps that from blocking)
    def src_above(k): return up[k].cons() if k == 0 else down[k - 1].cons()
    def src_below(k): return down[k].cons() if k == cores - 1 else up[k + 1].cons()
    workers = [Worker(core, [band_in[k].cons(), band_out[k].prod(), up[k].prod(), down[k].prod(),
                             src_above(k), src_below(k), step, row], tile=Tile(*snake(k))) for k in range(cores)]
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


class Water:
    def __init__(self, steps=5):
        self.cols = iron.get_current_device().cols
        self.cores = self.cols * ROWS_PER_COL
        self.H = self.cores * ROWS
        self.steps = steps
        self.h = np.zeros((self.H, W), np.float32)
        self.v = np.zeros((self.H, W), np.float32)
        n = self.cores * ROWS * FIELDS * S
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")
        self.dirty = True
        yy, xx = np.mgrid[0:self.H, 0:W]
        self.tiles = (((xx // 20 + yy // 20) & 1) * 0.5 + 0.4).astype(np.float32)   # background
        self.drop(W // 2, self.H // 2)

    def drop(self, x, y, r=6, amp=1.0):
        yy, xx = np.ogrid[0:self.H, 0:W]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        self.h += amp * np.exp(-d2 / (r * r)).astype(np.float32)
        self.dirty = True

    def pack(self):
        buf = np.zeros((self.H, FIELDS, S), np.float32)
        for i, fld in enumerate((self.h, self.v)):
            buf[:, i, PADC:PADC + W] = fld
            buf[:, i, PADC - 1] = fld[:, 0]
            buf[:, i, PADC + W] = fld[:, -1]
        return np.clip(np.round(buf.ravel() * (1 << Q)), -32768, 32767).astype(np.int16)

    def unpack(self, raw):
        buf = raw.astype(np.float32).reshape(self.H, FIELDS, S) / (1 << Q)
        self.h[:] = buf[:, 0, PADC:PADC + W]
        self.v[:] = buf[:, 1, PADC:PADC + W]

    def run(self):
        if self.dirty:
            with self._in.overwrite() as x:
                x[...] = self.pack()
            self.dirty = False
        t0 = time.perf_counter()
        water_chip(self._in, self._out, steps=self.steps, cols=self.cols, variant="water2-int16-" + "-".join(f"{k}{v}" for k, v in PARAMS.items()))
        ms = (time.perf_counter() - t0) * 1e3
        self.unpack(self._out.numpy())
        self._in, self._out = self._out, self._in
        return ms

    def rgb(self):
        """refract the tiled floor through the surface, slope lighting, specular sun."""
        gx = np.gradient(self.h, axis=1); gy = np.gradient(self.h, axis=0)
        yy, xx = np.mgrid[0:self.H, 0:W]
        sx = np.clip(xx + gx * 200, 0, W - 1).astype(np.int32)
        sy = np.clip(yy + gy * 200, 0, self.H - 1).astype(np.int32)
        floor = self.tiles[sy, sx]
        light = np.clip(0.75 + 12 * (0.4 * gx + 0.9 * gy), 0.3, 1.6)      # slope towards the light = brighter
        nz = 1 / np.sqrt(1 + 400 * (gx * gx + gy * gy))
        spec = np.clip(1 - 30 * ((gx * 20 * nz + 0.25) ** 2 + (gy * 20 * nz - 0.3) ** 2), 0, 1) ** 2
        img = np.empty((self.H, W, 3), np.float32)
        img[..., 0] = (0.15 * floor) * light + 0.9 * spec
        img[..., 1] = (0.45 * floor) * light + 0.9 * spec
        img[..., 2] = (0.75 * floor + 0.15) * light + 0.9 * spec
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--steps", type=int, default=5, help="steps per NPU call, odd")
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--scale", type=int, default=3)
    a = ap.parse_args()
    if a.steps % 2 == 0:
        sys.exit("--steps must be odd")
    sim = Water(a.steps)
    e0 = (sim.h ** 2).sum()
    ms = sim.run()
    print(f"first call (compile): {ms:.1f} ms; energy {e0:.1f} -> {(sim.h ** 2).sum() + (sim.v ** 2).sum():.1f}")
    if a.window:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((W * a.scale, sim.H * a.scale))
        font = pygame.font.SysFont(None, 24)
        rng = np.random.default_rng(); rain = False
        gen, t0 = 0, time.perf_counter()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    rain = not rain
                if e.type == pygame.MOUSEBUTTONDOWN or (e.type == pygame.MOUSEMOTION and e.buttons[0]):
                    sim.drop(e.pos[0] // a.scale, e.pos[1] // a.scale, r=7, amp=0.5)
            if rain and rng.random() < 0.3:
                sim.drop(rng.integers(0, W), rng.integers(0, sim.H), r=5, amp=0.4)
            ms = sim.run(); gen += a.steps
            surf = pygame.surfarray.make_surface(np.transpose(sim.rgb(), (1, 0, 2)))
            screen.blit(pygame.transform.smoothscale(surf, screen.get_size()), (0, 0))
            screen.blit(font.render(f"{W}x{sim.H} on 32 cores  {a.steps} steps/call {ms:.1f} ms  {gen / (time.perf_counter() - t0):.0f} steps/s  "
                                    f"click/drag: drops  R: rain", True, (255, 255, 255)), (8, 8))
            pygame.display.flip()
    t0 = time.perf_counter()
    for i in range(a.calls):
        ms = sim.run()
    dt = time.perf_counter() - t0
    sim.drop(W // 3, sim.H // 3, r=9, amp=0.6); sim.drop(2 * W // 3, 2 * sim.H // 3, r=9, amp=0.6)
    for _ in range(60 // a.steps):
        sim.run()
    print(f"{W}x{sim.H} on {sim.cores} cores: {a.steps} steps/call {ms:.1f} ms, {a.calls * a.steps / dt:.0f} steps/s, "
          f"{W * sim.H * a.calls * a.steps / dt / 1e6:.0f} Mcell-steps/s (incl. host repack on drops)")
    import cv2
    cv2.imwrite("water.png", cv2.resize(sim.rgb()[..., ::-1], (W * a.scale, sim.H * a.scale), interpolation=cv2.INTER_LINEAR))
    print("wrote water.png")


if __name__ == "__main__":
    main()
