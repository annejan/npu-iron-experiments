#!/usr/bin/env python3
"""1080p plasma rendered by all 32 AIE cores, packed RGBA straight from the NPU.

  ./env.sh python probe/plasma.py            bench: frames/s, writes plasma.png
  ./env.sh python probe/plasma.py --window   pygame window

Host sends per-frame sine tables (3 per-x, 3 per-y, int32 Q12, 36 KB); each core renders 34 rows
(row y = 32*j + 4*col + row), a row at a time through a depth-2 ObjectFifo; the 4 cores of a column
are joined in the memtile into 4-row stripes and the shim DMA scatters the stripes into the frame.
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.helpers.taplib import TensorAccessPattern

W, H = 1920, 1088  # 1088 = 32 cores x 34 rows; the window shows 1080
ROWS = 4
TBL = 3 * W + 3 * H

KSRC = f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define H {H}
extern "C" {{
void plasma(int32_t *__restrict tbl, int32_t *__restrict out, int32_t y) {{
    const int32_t *__restrict T1 = tbl, *__restrict S3 = tbl + W, *__restrict C3 = tbl + 2 * W;
    const int32_t v2 = tbl[3 * W + y], s = tbl[3 * W + H + y], c = tbl[3 * W + 2 * H + y];
    const auto lo = aie::zeros<int32_t, 16>();
    const auto hi = aie::broadcast<int32_t, 16>(255);
    const auto alpha = aie::broadcast<int32_t, 16>((int32_t)0xFF000000);
    const auto off = aie::broadcast<int32_t, 16>(128);
    const auto v2v = aie::broadcast<int32_t, 16>(v2);
    for (unsigned i = 0; i < W / 16; i++) {{
        auto t1 = aie::load_v<16>(T1 + 16 * i);
        auto s3 = aie::load_v<16>(S3 + 16 * i);
        auto c3 = aie::load_v<16>(C3 + 16 * i);
        aie::accum<acc64, 16> a = aie::mul<acc64>(s3, c);
        a = aie::mac(a, c3, s);
        auto v3 = a.to_vector<int32_t>(12);               // sin(c(x+y)+t) in Q12
        auto r = aie::add(aie::downshift(aie::add(t1, v2v), 6), off);
        auto g = aie::add(aie::downshift(aie::add(v2v, v3), 6), off);
        auto b = aie::add(aie::downshift(aie::add(t1, v3), 6), off);
        r = aie::min(aie::max(r, lo), hi);
        g = aie::min(aie::max(g, lo), hi);
        b = aie::min(aie::max(b, lo), hi);
        auto px = aie::bit_or(aie::bit_or(r, aie::upshift(g, 8)), aie::bit_or(aie::upshift(b, 16), alpha));
        aie::store_v(out + 16 * i, px);
    }}
}}
}}
"""


@iron.jit
def plasma_npu(tbl: In, frame: Out, *, cols: CompileTime[int]):
    cores = cols * ROWS
    per_core = H // cores
    frame_ty = np.ndarray[(H * W,), np.dtype[np.int32]]
    tbl_ty = np.ndarray[(TBL,), np.dtype[np.int32]]
    quad_ty = np.ndarray[(ROWS * W,), np.dtype[np.int32]]
    row_ty = np.ndarray[(W,), np.dtype[np.int32]]
    k = ExternalFunction("plasma", source_string=KSRC, arg_types=[tbl_ty, row_ty, np.dtype(np.int32)])

    tbl_in = [ObjectFifo(tbl_ty, name=f"tbl{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    rows_out = []
    for c in range(cols):
        rows_out += col_out[c].prod().join([i * W for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                           obj_types=[row_ty] * ROWS, names=[f"row{c}_{i}" for i in range(ROWS)])

    def core(tbl, out, k, base):
        t = tbl.acquire(1)
        for j in range_(per_core):
            o = out.acquire(1)
            k(t, o, j * cores + base)
            out.release(1)
        tbl.release(1)

    workers = [Worker(core, [tbl_in[c].cons(), rows_out[c * ROWS + r].prod(), k, c * ROWS + r],
                      tile=Tile(c, 2 + r)) for c in range(cols) for r in range(ROWS)]

    tp = [f.prod() for f in tbl_in]
    oc = [f.cons() for f in col_out]
    full = TensorAccessPattern.full((TBL,))
    stripes = [TensorAccessPattern((H * W,), ROWS * c * W, [per_core, ROWS * W], [cores * W, 1]) for c in range(cols)]

    def seq(tbl, frame, tp, oc):
        tg = TaskGroup()
        for c in range(cols):
            tp[c].fill(tbl, full, group=tg)
        for c in range(cols):
            oc[c].drain(frame, stripes[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [tbl_ty, frame_ty, tp, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def tables(t):
    x = np.arange(W); y = np.arange(H)
    q = 4096.0
    out = np.empty(TBL, dtype=np.int32)
    out[0:W] = q * np.sin(x * 0.021 + t * 1.3)
    out[W:2 * W] = q * np.sin(x * 0.013 + t * 0.7)
    out[2 * W:3 * W] = q * np.cos(x * 0.013 + t * 0.7)
    out[3 * W:3 * W + H] = q * np.sin(y * 0.033 + t * 0.9)
    out[3 * W + H:3 * W + 2 * H] = q * np.sin(y * 0.013 + t * 0.5)
    out[3 * W + 2 * H:] = q * np.cos(y * 0.013 + t * 0.5)
    return out


def reference(tb):
    T1, S3, C3 = tb[:W], tb[W:2 * W], tb[2 * W:3 * W]
    v2 = tb[3 * W:3 * W + H][:, None]; s = tb[3 * W + H:3 * W + 2 * H][:, None]; c = tb[3 * W + 2 * H:][:, None]
    v3 = (S3[None, :].astype(np.int64) * c + C3[None, :].astype(np.int64) * s) >> 12
    t1 = T1[None, :].astype(np.int64)
    r = np.clip(((t1 + v2) >> 6) + 128, 0, 255); g = np.clip(((v2 + v3) >> 6) + 128, 0, 255)
    b = np.clip(((t1 + v3) >> 6) + 128, 0, 255)
    return (r | (g << 8) | (b << 16) | (0xFF << 24)).astype(np.uint32).view(np.int32).ravel()


class Plasma:
    def __init__(self):
        self.cols = iron.get_current_device().cols
        self.tbl = iron.zeros((TBL,), dtype=np.int32, device="npu")
        self.frame = iron.zeros((H * W,), dtype=np.int32, device="npu")

    def render(self, t):
        tb = tables(t)
        with self.tbl.overwrite() as a:
            a[...] = tb
        t0 = time.perf_counter()
        plasma_npu(self.tbl, self.frame, cols=self.cols)
        npu_ms = (time.perf_counter() - t0) * 1e3
        return self.frame.numpy(), npu_ms, tb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", action="store_true")
    p.add_argument("--frames", type=int, default=200)
    a = p.parse_args()
    pl = Plasma()
    f, ms, tb = pl.render(0.0)  # compile + warm up
    ref = reference(tb)
    bad = int((f != ref).sum())
    print(f"first frame: {ms:.2f} ms; {bad} of {f.size} pixels differ from numpy reference")
    if a.window:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        clock = pygame.time.Clock(); t = 0.0; n = 0; t_start = time.perf_counter()
        font = pygame.font.SysFont(None, 28)
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
            f, ms, _ = pl.render(t)
            surf = pygame.image.frombuffer(f.view(np.uint8)[: 1080 * W * 4], (W, 1080), "RGBA")
            if screen.get_size() != (W, 1080):
                surf = pygame.transform.smoothscale(surf, screen.get_size())
            screen.blit(surf, (0, 0))
            n += 1; fps = n / (time.perf_counter() - t_start)
            screen.blit(font.render(f"NPU 32 cores: {ms:.2f} ms/frame   {fps:.0f} fps (pygame-bound)", True, (255, 255, 255)), (10, 10))
            pygame.display.flip()
            t += 0.03
        return
    ts = []; t_all = time.perf_counter()
    for i in range(a.frames):
        _, ms, _ = pl.render(i * 0.03); ts.append(ms)
    wall = time.perf_counter() - t_all
    ts = np.array(ts)
    print(f"{a.frames} frames 1920x1088 RGBA: NPU call min {ts.min():.2f} ms  med {np.median(ts):.2f} ms "
          f"({1e3/np.median(ts):.0f} fps NPU-only); whole loop incl. numpy tables + readback: {a.frames/wall:.0f} fps")
    import cv2
    cv2.imwrite("plasma.png", f.view(np.uint8).reshape(H, W, 4)[:1080, :, [2, 1, 0]])
    print("wrote plasma.png")


if __name__ == "__main__":
    main()
