#!/usr/bin/env python3
"""Mandelbrot on the NPU: all 32 AIE cores, 16 pixels per vector, live zoom in the terminal.

  ./env.sh python fun/mandel.py                 zoom into Seahorse Valley
  ./env.sh python fun/mandel.py --still         one frame, then exit
  ./env.sh python fun/mandel.py --png out.png   one big frame as a PNG (no PIL needed)
  ./env.sh python fun/mandel.py --bench         scalar kernel vs 16-lane vector kernel vs numpy, 1920x1080

The host computes c for each pixel (fixed point, Q24 in an int32), the NPU runs the
iterations and gives back the count per pixel. The coordinates are data, not compile-time
values, so the kernel compiles once and every zoom frame reuses it.
"""

import argparse
import shutil
import struct
import sys
import time
import zlib

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

FRAC = 24  # Q24: precision ~6e-8, so the zoom ends near 1e5x
TILE = 512  # pixels per kernel call
MAXIT = 400
CORES_PER_COL = 4  # workers per column: 8 columns x 4 = all 32 cores

KERNEL_SCALAR = f"""
#include <stdint.h>
extern "C" {{
void mandel(int32_t *c, int32_t *out, int32_t n) {{
    const int32_t *cx = c, *cy = c + n;
    const int64_t four = 4LL << {FRAC};
    for (int i = 0; i < n; i++) {{
        int64_t x = 0, y = 0;
        const int64_t a = cx[i], b = cy[i];
        int k = 0;
        for (; k < {MAXIT}; k++) {{
            int64_t xx = (x * x) >> {FRAC};
            int64_t yy = (y * y) >> {FRAC};
            if (xx + yy > four) break;
            int64_t xy = (x * y) >> {FRAC - 1};
            x = xx - yy + a;
            y = xy + b;
        }}
        out[i] = k;
    }}
}}
}}
"""


# 16 pixels per vector: x*x is int32 x int32 -> acc64 (two 32x16 multiplies on AIE2P), shifted back to Q24.
# Lanes that escaped stay dead in the mask; the loop leaves once every lane is dead.
KERNEL_VECTOR = f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" {{
void mandel(int32_t *__restrict c, int32_t *__restrict out, int32_t n) {{
    const int32_t *__restrict cx = c, *__restrict cy = c + n;
    const auto four = aie::broadcast<int32_t, 16>(4 << {FRAC});
    const auto one = aie::broadcast<int32_t, 16>(1);
    const auto zero = aie::zeros<int32_t, 16>();
    for (int i = 0; i < n; i += 16) {{
        const auto a = aie::load_v<16>(cx + i);
        const auto b = aie::load_v<16>(cy + i);
        auto x = zero, y = zero, k = zero;
        aie::mask<16> alive(true);
        for (int it = 0; it < {MAXIT}; it++) {{
            const auto xx = aie::mul<acc64>(x, x).to_vector<int32_t>({FRAC});
            const auto yy = aie::mul<acc64>(y, y).to_vector<int32_t>({FRAC});
            const auto xy = aie::mul<acc64>(x, y).to_vector<int32_t>({FRAC - 1});
            alive &= ~aie::gt(aie::add(xx, yy), four);
            if (alive.empty()) break;
            k = aie::add(k, aie::select(zero, one, alive));
            x = aie::add(aie::sub(xx, yy), a);
            y = aie::add(xy, b);
        }}
        aie::store_v(out + i, k);
    }}
}}
}}
"""

KERNELS = {"scalar": KERNEL_SCALAR, "vector": KERNEL_VECTOR}


@iron.jit
def mandel_npu(c: In, out: Out, *, func: CompileTime[ExternalFunction], num_elements: CompileTime[int],
               variant: CompileTime[str]):
    """All 32 cores. c holds, per tile of TILE pixels, TILE cx values then TILE cy values. Column c owns a
    contiguous 1/8 of the tiles and streams them through its memtile in objects of 4 tiles, of which core r
    takes tile r (split on the way in, join on the way out). One shim channel in, one out, per column.
    variant is only here so the JIT cache tells the kernels apart (it does not hash the C source)."""
    dev = iron.get_current_device()
    cols, cores = dev.cols, dev.cols * CORES_PER_COL
    n = num_elements
    objs = n // (cols * CORES_PER_COL * TILE)  # objects per column per frame
    assert objs * CORES_PER_COL * TILE * cols == n, "num_elements must be a multiple of TILE * cores"
    in_ty = np.ndarray[(2 * n,), np.dtype[np.int32]]
    out_ty = np.ndarray[(n,), np.dtype[np.int32]]
    col_in_ty = np.ndarray[(CORES_PER_COL * 2 * TILE,), np.dtype[np.int32]]
    col_out_ty = np.ndarray[(CORES_PER_COL * TILE,), np.dtype[np.int32]]
    tile_in_ty = np.ndarray[(2 * TILE,), np.dtype[np.int32]]
    tile_out_ty = np.ndarray[(TILE,), np.dtype[np.int32]]

    ins = [ObjectFifo(col_in_ty, name=f"c{col}", depth=2) for col in range(cols)]
    outs = [ObjectFifo(col_out_ty, name=f"k{col}", depth=2) for col in range(cols)]
    ti, to = [], []
    for col in range(cols):
        ti += ins[col].cons().split([r * 2 * TILE for r in range(CORES_PER_COL)], tile=Tile(col, 1),
                                    depths=[2] * CORES_PER_COL, obj_types=[tile_in_ty] * CORES_PER_COL,
                                    names=[f"ti{col}_{r}" for r in range(CORES_PER_COL)])
        to += outs[col].prod().join([r * TILE for r in range(CORES_PER_COL)], tile=Tile(col, 1),
                                    depths=[2] * CORES_PER_COL, obj_types=[tile_out_ty] * CORES_PER_COL,
                                    names=[f"to{col}_{r}" for r in range(CORES_PER_COL)])

    def core(fi, fo, func):
        for _ in range_(objs):
            a = fi.acquire(1)
            o = fo.acquire(1)
            func(a, o, TILE)
            fo.release(1)
            fi.release(1)

    workers = [Worker(core, [ti[i].cons(), to[i].prod(), func], tile=Tile(i // CORES_PER_COL, 2 + i % CORES_PER_COL))
               for i in range(cores)]
    taps_in = TensorAccessPattern.full((2 * n,)).partition(cols)
    taps_out = TensorAccessPattern.full((n,)).partition(cols)
    pi, po = [f.prod() for f in ins], [f.cons() for f in outs]

    def sequence(c, out, pi, po):
        tg = TaskGroup()
        for col in range(cols):
            pi[col].fill(c, taps_in[col], group=tg)
        for col in range(cols):
            po[col].drain(out, taps_out[col], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [in_ty, out_ty, pi, po])
    return Program(dev, rt, workers=workers).resolve_program()


def kernel(variant="vector"):
    t = np.ndarray[(TILE,), np.dtype[np.int32]]
    t2 = np.ndarray[(2 * TILE,), np.dtype[np.int32]]
    return ExternalFunction("mandel", source_string=KERNELS[variant], arg_types=[t2, t, np.dtype(np.int32)])


def grid(w, h, cx, cy, width, pad_to):
    """c for every pixel in Q24, flattened and padded with points far outside the set.

    Pixels are square: in the terminal each cell is two half-block pixels on top of each other."""
    xs = cx + (np.arange(w) / w - 0.5) * width
    ys = cy + (np.arange(h) / h - 0.5) * width * h / w
    gx, gy = np.meshgrid(xs, ys)
    n = w * h
    total = -(-n // pad_to) * pad_to
    out = []
    for g in (gx, gy):
        a = np.full(total, 4 << FRAC, dtype=np.int32)
        a[:n] = np.round(g.ravel() * (1 << FRAC)).astype(np.int32)
        out.append(a)
    return out, n


def cpu(gx, gy):
    """The same fixed-point maths in numpy, for the race."""
    a, b = gx.astype(np.int64), gy.astype(np.int64)
    x, y = np.zeros_like(a), np.zeros_like(a)
    k = np.full(a.shape, MAXIT, dtype=np.int32)
    alive = np.ones(a.shape, bool)
    four = 4 << FRAC
    for i in range(MAXIT):
        xx, yy = (x * x) >> FRAC, (y * y) >> FRAC
        esc = alive & (xx + yy > four)
        k[esc] = i
        alive &= ~esc
        if not alive.any():
            break
        xy = (x * y) >> (FRAC - 1)
        x = np.where(alive, xx - yy + a, x)
        y = np.where(alive, xy + b, y)
    return k


def colors(k):
    t = np.log1p(k) / np.log1p(MAXIT)  # log scale: most points escape in a few steps
    rgb = np.stack([9 * (1 - t) * t**3, 15 * (1 - t) ** 2 * t**2, 8.5 * (1 - t) ** 3 * t], -1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgb[k >= MAXIT] = 0
    return rgb


def draw(rgb, status):
    out = ["\x1b[H"]
    for r in range(0, rgb.shape[0] - 1, 2):
        top, bot = rgb[r], rgb[r + 1]
        out.append("".join(f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m▀"
                           for a, b in zip(top, bot, strict=True)))
        out.append("\x1b[0m\n")
    out.append(f"\x1b[0m\x1b[K{status}")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\0" + rgb[r].tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--still", action="store_true")
    p.add_argument("--png")
    p.add_argument("--no-cpu", action="store_true", help="skip the CPU race")
    p.add_argument("--scalar", action="store_true", help="the old scalar int64 kernel")
    p.add_argument("--bench", action="store_true", help="one 1920x1080 frame with both kernels and numpy, then exit")
    p.add_argument("--x", type=float, default=-0.743643887037151)
    p.add_argument("--y", type=float, default=0.131825904205330)
    args = p.parse_args()

    dev = iron.get_current_device()
    pad_to = TILE * CORES_PER_COL * dev.cols
    variant = "scalar" if args.scalar else "vector"
    func = kernel(variant)

    if args.bench:
        # the scalar kernel is too slow for a 1080p frame (XRT times the dispatch out), so both run at 480x270
        # and the vector kernel again at 1920x1080
        for (w, h), variants in (((480, 270), ("scalar", "vector")), ((1920, 1080), ("vector",))):
            (gx, gy), n = grid(w, h, -0.6, 0.0, 3.2, pad_to)
            t0 = time.perf_counter()
            ref = cpu(gx[:n], gy[:n])
            cpu_ms = (time.perf_counter() - t0) * 1000
            print(f"{w}x{h}, {MAXIT} max iterations: numpy CPU {cpu_ms:8.1f} ms")
            for v in variants:
                f = kernel(v)
                run(f, gx, gy, n, v)  # compile + warm up
                best = None
                for _ in range(3):
                    k, ms = run(f, gx, gy, n, v)
                    best = ms if best is None else min(best, ms)
                ok = "matches numpy" if np.array_equal(k[:n], ref) else f"DIFFERS from numpy in {int((k[:n] != ref).sum())} pixels"
                print(f"{'':>{len(f'{w}x{h}')}}  NPU {v:6s} {best:8.1f} ms  ({cpu_ms / best:6.1f}x numpy)  {ok}")
        return

    if args.png:
        w, h = 1920, 1080
        (gx, gy), n = grid(w, h, -0.6, 0.0, 3.2, pad_to)
        k = run(func, gx, gy, n, variant)[0]
        png(args.png, colors(k[:n].reshape(h, w)))
        print(f"{args.png}: {w}x{h} on {dev.cols * CORES_PER_COL} NPU cores")
        return

    cols, rows = shutil.get_terminal_size()
    w, h = cols, (rows - 2) * 2
    width, zoom = 3.2, 0.94
    cx, cy = -0.6, 0.0
    sys.stdout.write("\x1b[2J\x1b[?25l")
    try:
        frame = 0
        while width > 2e-5:
            # glide from the whole set to the target, then zoom in
            f = min(1.0, frame / 40)
            px, py = cx + (args.x - cx) * f, cy + (args.y - cy) * f
            (gx, gy), n = grid(w, h, px, py, width, pad_to)
            k, npu_ms = run(func, gx, gy, n, variant)
            status = f"NPU {dev.cols * CORES_PER_COL} cores: {npu_ms:6.1f} ms"
            if not args.no_cpu and frame % 10 == 0:
                t0 = time.perf_counter()
                cpu(gx[:n], gy[:n])
                cpu_ms = (time.perf_counter() - t0) * 1000
                status += f" | numpy CPU: {cpu_ms:6.1f} ms ({cpu_ms / npu_ms:.1f}x)"
            status += f" | zoom {3.2 / width:,.0f}x | {w}x{h}"
            draw(colors(k[:n].reshape(h, w)), status)
            if args.still:
                break
            width *= zoom if frame >= 20 else 1
            frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")


def run(func, gx, gy, n, variant="vector"):
    packed = np.stack([gx.reshape(-1, TILE), gy.reshape(-1, TILE)], axis=1).ravel()  # per tile: cx[TILE], cy[TILE]
    a = iron.tensor(packed, dtype=np.int32, device="npu")
    out = iron.zeros(gx.shape, dtype=np.int32, device="npu")
    t0 = time.perf_counter()
    mandel_npu(a, out, func=func, num_elements=int(gx.shape[0]), variant=variant)
    ms = (time.perf_counter() - t0) * 1000
    return out.numpy().copy(), ms


if __name__ == "__main__":
    main()
