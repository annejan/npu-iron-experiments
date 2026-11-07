#!/usr/bin/env python3
"""Rotozoomer and tunnel at 1080p on all 32 AIE cores: texture gathers on the scalar units.

  ./env.sh python fun/rotozoom.py                     bench both effects, write rotozoom.png / tunnel.png
  ./env.sh python fun/rotozoom.py --window            pygame window, TAB switches effect, ESC quits

There is no per-lane vector gather on AIE, so a texture lookup is a scalar load. A 1080p frame is
2M pixels / 32 cores = 65k pixels per core, a few cycles each: well under a millisecond.

Frame layout is the plasma one: core (col, row) renders rows y = 32*j + 4*col + row; the four cores
of a column are joined in the memtile into 4-row stripes, the shim DMA scatters them into the frame.

  rotozoom: 64x64 RGBA texture (16 KB) + 6 affine parameters per frame; u,v computed on the core.
  tunnel:   per-pixel (u,v) bytes precomputed once on the host (4 MB), streamed in row by row;
            the core adds the scroll offset and gathers. 4 MB in + 8 MB out per frame.
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.helpers.taplib import TensorAccessPattern

W, H = 1920, 1088
ROWS = 4
TEX = 64
NPARAM = 8
TEXBUF = TEX * TEX + NPARAM  # int32: texture then parameters

KSRC = f"""
#include <stdint.h>
#define W {W}
#define TEX {TEX}
#define MASK ({TEX} - 1)
extern "C" {{
/* params after the texture: ox, oy, dxx, dxy, dyx, dyy in Q16 */
void rotozoom(int32_t *__restrict tex, uint16_t *__restrict uv_unused, int32_t *__restrict out, int32_t y) {{
    const int32_t *p = tex + TEX * TEX;
    int32_t u = p[0] + y * p[4], v = p[1] + y * p[5];
    const int32_t dxx = p[2], dxy = p[3];
    for (int x = 0; x < W; x++) {{
        out[x] = tex[(((v >> 16) & MASK) << 6) | ((u >> 16) & MASK)];
        u += dxx; v += dxy;
    }}
}}
/* params: tu, tv scroll offsets in texels; uv row: u in the low byte, v in the high byte */
void tunnel(int32_t *__restrict tex, uint16_t *__restrict uv, int32_t *__restrict out, int32_t y) {{
    const int32_t *p = tex + TEX * TEX;
    const uint32_t tu = p[0], tv = p[1];
    for (int x = 0; x < W; x++) {{
        const uint32_t e = uv[x];
        out[x] = tex[((((e >> 8) + tv) & MASK) << 6) | ((e + tu) & MASK)];
    }}
}}
}}
"""


@iron.jit
def effect_npu(tex: In, uv: In, frame: Out, *, cols: CompileTime[int], effect: CompileTime[str]):
    cores = cols * ROWS
    per_core = H // cores
    frame_ty = np.ndarray[(H * W,), np.dtype[np.int32]]
    uv_ty = np.ndarray[(H * W,), np.dtype[np.uint16]]
    tex_ty = np.ndarray[(TEXBUF,), np.dtype[np.int32]]
    quad_ty = np.ndarray[(ROWS * W,), np.dtype[np.int32]]
    row_ty = np.ndarray[(W,), np.dtype[np.int32]]
    uvquad_ty = np.ndarray[(ROWS * W,), np.dtype[np.uint16]]
    uvrow_ty = np.ndarray[(W,), np.dtype[np.uint16]]
    k = ExternalFunction(effect, source_string=KSRC, arg_types=[tex_ty, uvrow_ty, row_ty, np.dtype(np.int32)])

    tex_in = [ObjectFifo(tex_ty, name=f"tex{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    col_uv = [ObjectFifo(uvquad_ty, name=f"uv{c}", depth=2) for c in range(cols)]
    rows_out, rows_uv = [], []
    for c in range(cols):
        rows_out += col_out[c].prod().join([i * W for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                           obj_types=[row_ty] * ROWS, names=[f"row{c}_{i}" for i in range(ROWS)])
        rows_uv += col_uv[c].cons().split([i * W for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                          obj_types=[uvrow_ty] * ROWS, names=[f"uvrow{c}_{i}" for i in range(ROWS)])

    def core(tex, uv, out, k, base):
        t = tex.acquire(1)
        for j in range_(per_core):
            e = uv.acquire(1)
            o = out.acquire(1)
            k(t, e, o, j * cores + base)
            out.release(1)
            uv.release(1)
        tex.release(1)

    workers = [Worker(core, [tex_in[c].cons(), rows_uv[c * ROWS + r].cons(), rows_out[c * ROWS + r].prod(), k,
                             c * ROWS + r], tile=Tile(c, 2 + r)) for c in range(cols) for r in range(ROWS)]

    tp = [f.prod() for f in tex_in]
    up = [f.prod() for f in col_uv]
    oc = [f.cons() for f in col_out]
    full = TensorAccessPattern.full((TEXBUF,))
    stripes = [TensorAccessPattern((H * W,), ROWS * c * W, [per_core, ROWS * W], [cores * W, 1]) for c in range(cols)]

    def seq(tex, uv, frame, tp, up, oc):
        tg = TaskGroup()
        for c in range(cols):
            tp[c].fill(tex, full, group=tg)
            up[c].fill(uv, stripes[c], group=tg)
        for c in range(cols):
            oc[c].drain(frame, stripes[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [tex_ty, uv_ty, frame_ty, tp, up, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def texture():
    """64x64 RGBA: a plasma-ish tile that wraps, packed r | g<<8 | b<<16 | 0xFF<<24."""
    y, x = np.mgrid[0:TEX, 0:TEX]
    a = np.sin(x * 2 * np.pi / TEX * 2) + np.sin(y * 2 * np.pi / TEX * 3)
    b = np.sin((x + y) * 2 * np.pi / TEX * 2)
    checker = ((x // 8 + y // 8) & 1) * 60
    r = np.clip(128 + 90 * a + checker, 0, 255).astype(np.uint32)
    g = np.clip(128 + 120 * b, 0, 255).astype(np.uint32)
    bl = np.clip(90 + 80 * a * b + checker, 0, 255).astype(np.uint32)
    return (r | (g << 8) | (bl << 16) | (0xFF << 24)).astype(np.uint32).view(np.int32).ravel()


def tunnel_uv():
    """Per pixel: u = angle around the centre, v = depth (texels), as bytes u | v<<8."""
    y, x = np.mgrid[0:H, 0:W]
    dx, dy = x - W / 2, y - 1080 / 2
    dist = np.sqrt(dx * dx + dy * dy) + 1e-3
    v = (32 * 180 / dist).astype(np.int64)
    u = ((np.arctan2(dy, dx) / np.pi + 1) * 0.5 * TEX * 2).astype(np.int64)
    return ((u & 0xFF) | ((v & 0xFF) << 8)).astype(np.uint16).ravel()


class Effects:
    def __init__(self):
        self.cols = iron.get_current_device().cols
        self.tex = iron.zeros((TEXBUF,), dtype=np.int32, device="npu")
        self.uv = iron.tensor(tunnel_uv(), dtype=np.uint16, device="npu")
        self.frame = iron.zeros((H * W,), dtype=np.int32, device="npu")
        self.texture = texture()

    def params(self, effect, t):
        p = np.zeros(NPARAM, dtype=np.int32)
        if effect == "rotozoom":
            zoom = 1.5 + 1.2 * np.sin(t * 0.7)
            ang = t * 0.5
            c, s = np.cos(ang) / zoom, np.sin(ang) / zoom
            p[2], p[3] = int(c * 65536), int(s * 65536)  # du/dx, dv/dx
            p[4], p[5] = int(-s * 65536), int(c * 65536)  # du/dy, dv/dy
            p[0] = int((TEX * 8 * np.sin(t * 0.3)) * 65536) - (W // 2) * p[2] - 540 * p[4]
            p[1] = int((TEX * 8 * np.cos(t * 0.4)) * 65536) - (W // 2) * p[3] - 540 * p[5]
        else:
            p[0], p[1] = int(t * 20) & 0xFF, int(t * 30) & 0xFF
        return p

    def render(self, effect, t):
        with self.tex.overwrite() as a:
            a[:TEX * TEX] = self.texture
            a[TEX * TEX:] = self.params(effect, t)
        t0 = time.perf_counter()
        effect_npu(self.tex, self.uv, self.frame, cols=self.cols, effect=effect)
        ms = (time.perf_counter() - t0) * 1e3
        return self.frame.numpy(), ms


def reference(fx, effect, t):
    tex = fx.texture
    p = fx.params(effect, t)
    y, x = np.mgrid[0:H, 0:W]
    if effect == "rotozoom":
        u = p[0] + x * p[2] + y * p[4]
        v = p[1] + x * p[3] + y * p[5]
        idx = (((v >> 16) & (TEX - 1)) << 6) | ((u >> 16) & (TEX - 1))
    else:
        e = fx.uv.numpy().reshape(H, W).astype(np.int64)
        idx = ((((e >> 8) + p[1]) & (TEX - 1)) << 6) | ((e + p[0]) & (TEX - 1))
    return tex[idx].ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--effect", choices=("rotozoom", "tunnel"), default="rotozoom")
    a = ap.parse_args()
    fx = Effects()
    if a.window:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        font = pygame.font.SysFont(None, 28)
        effect, t, n, t_start = a.effect, 0.0, 0, time.perf_counter()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_TAB:
                    effect = "tunnel" if effect == "rotozoom" else "rotozoom"
            f, ms = fx.render(effect, t)
            surf = pygame.image.frombuffer(f.view(np.uint8)[: 1080 * W * 4], (W, 1080), "RGBA")
            if screen.get_size() != (W, 1080):
                surf = pygame.transform.smoothscale(surf, screen.get_size())
            screen.blit(surf, (0, 0))
            n += 1
            screen.blit(font.render(f"{effect}: NPU 32 cores {ms:.2f} ms/frame  {n / (time.perf_counter() - t_start):.0f} fps (pygame-bound)  TAB switches",
                                    True, (255, 255, 255)), (10, 10))
            pygame.display.flip()
            t += 0.02
    import cv2
    for effect in ("rotozoom", "tunnel"):
        f, ms = fx.render(effect, 1.0)
        ref = reference(fx, effect, 1.0)
        bad = int((f != ref).sum())
        ts = [fx.render(effect, i * 0.02)[1] for i in range(a.frames)]
        print(f"{effect:8s}: {bad} of {f.size} pixels differ from numpy; NPU call min {min(ts):.2f} ms "
              f"med {np.median(ts):.2f} ms ({1e3 / np.median(ts):.0f} fps NPU-only)")
        cv2.imwrite(f"{effect}.png", f.view(np.uint8).reshape(H, W, 4)[:1080, :, [2, 1, 0]])


if __name__ == "__main__":
    main()
