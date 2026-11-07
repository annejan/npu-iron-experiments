#!/usr/bin/env python3
"""Generic K x K convolution engine over a DDR world on all 32 cores (the Lenia machinery, parameterised).

  ./env.sh python fun/conv2d.py        self-test: 3x3 and 7x7 Gaussian blurs against numpy

Conv2D(K, taps, post, params): taps is a K x K float kernel; post is the elementwise pass applied to the int16 Q15
convolution result uq before storing (C++ snippet; "decay" multiplies by params['decay']). World and result are
int16 Q15 (H, W) numpy arrays; the halos are a torus. See lenia.py for the dataflow (3-D DMA gather of band + halo
rows, K-row ring buffer in L1, sliding_mac over the taps).
"""
import sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4
CORES = 32
KSHIFT = 16

POSTS = {
    "decay": "aie::store_unaligned_v(out + x + PADL, aie::mul(uq, (int16_t){decay_q15}).to_vector<int16_t>(15));",
    "copy": "aie::store_unaligned_v(out + x + PADL, uq);",
}


def ksrc(K, W, S, PADL, post):
    H2 = K // 2
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define K {K}
#define S {S}
#define PADL {PADL}
#define H2 {H2}
#define KSHIFT {KSHIFT}
alignas(64) static int16_t ring[K * S];
static int head = 0;
static int filled = 0;
extern "C" {{
void conv_row(int16_t *__restrict kern, int16_t *__restrict row, int16_t *__restrict out, int32_t produce) {{
    if (produce < 0) {{ head = 0; filled = 0; return; }}
    int w = head + filled; if (w >= K) w -= K;
    int16_t *dst = ring + w * S;
    for (int x = 0; x < S; x += 32) aie::store_v(dst + x, aie::load_v<32>(row + x));
    if (filled < K) filled++; else {{ head++; if (head >= K) head = 0; }}
    if (!produce) return;
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    for (int x = 0; x < W; x += 32) {{
        aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
        int ridx = head;
        for (int kr = 0; kr < K; kr++) {{
            const int16_t *d = ring + ridx * S + x + PADL - H2;
            ridx++; if (ridx >= K) ridx = 0;
            aie::vector<int16_t, 64> dv = aie::concat(aie::load_v<32>(d), aie::load_v<32>(d + 32));
            aie::vector<int16_t, 32> kv = aie::load_v<32>(kern + kr * 32);
            acc = aie::sliding_mac<32, 8>(acc, kv, 0, dv, 0);
            acc = aie::sliding_mac<32, 8>(acc, kv, 8, dv, 8);
            acc = aie::sliding_mac<32, 8>(acc, kv, 16, dv, 16);
            acc = aie::sliding_mac<32, 8>(acc, kv, 24, dv, 24);
        }}
        aie::vector<int16_t, 32> uq = acc.to_vector<int16_t>(KSHIFT);
        {post}
    }}
}}
}}
"""


@iron.jit
def conv_npu(world: In, kern: In, out: Out, *, cols: CompileTime[int], K: CompileTime[int], W: CompileTime[int],
             R: CompileTime[int], S: CompileTime[int], PADL: CompileTime[int], post: CompileTime[str], variant: CompileTime[str]):
    cores = cols * ROWS
    H = cores * R
    H2 = K // 2
    n = (H + K - 1) * S
    world_ty = np.ndarray[(n,), np.dtype[np.int16]]
    k_ty = np.ndarray[(K * 32,), np.dtype[np.int16]]
    row_ty = np.ndarray[(S,), np.dtype[np.int16]]
    quad_ty = np.ndarray[(ROWS * S,), np.dtype[np.int16]]
    k = ExternalFunction("conv_row", source_string=ksrc(K, W, S, PADL, post), arg_types=[k_ty, row_ty, row_ty, np.dtype(np.int32)])
    k_in = [ObjectFifo(k_ty, name=f"k{c}", depth=1) for c in range(cols)]
    col_in = [ObjectFifo(quad_ty, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    ri, ro = [], []
    for c in range(cols):
        ri += col_in[c].cons().split([i * S for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[row_ty] * ROWS, names=[f"ri{c}_{i}" for i in range(ROWS)])
        ro += col_out[c].prod().join([i * S for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[row_ty] * ROWS, names=[f"ro{c}_{i}" for i in range(ROWS)])

    def core(kf, fi, fo, k):
        kk = kf.acquire(1)
        r = fi.acquire(1)
        k(kk, r, r, -1)
        k(kk, r, r, 0)
        fi.release(1)
        for _ in range_(K - 2):
            r = fi.acquire(1); k(kk, r, r, 0); fi.release(1)
        for _ in range_(R):
            r = fi.acquire(1)
            o = fo.acquire(1)
            k(kk, r, o, 1)
            fo.release(1); fi.release(1)
        kf.release(1)

    workers = [Worker(core, [k_in[j // ROWS].cons(), ri[j].cons(), ro[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    taps_in = [TensorAccessPattern((n,), c * ROWS * R * S, [R + K - 1, ROWS, S], [S, R * S, 1]) for c in range(cols)]
    taps_out = [TensorAccessPattern((n,), (H2 + c * ROWS * R) * S, [R, ROWS, S], [S, R * S, 1]) for c in range(cols)]
    kp, ip, oc = [f.prod() for f in k_in], [f.prod() for f in col_in], [f.cons() for f in col_out]
    kfull = TensorAccessPattern.full((K * 32,))

    def seq(world, kern, out, kp, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            kp[c].fill(kern, kfull, group=tg)
            ip[c].fill(world, taps_in[c], group=tg)
        for c in range(cols):
            oc[c].drain(out, taps_out[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [world_ty, k_ty, world_ty, kp, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class Conv2D:
    def __init__(self, K, taps, W=512, H=512, post="copy", params=None):
        assert K % 2 == 1 and K <= 32 and H % CORES == 0 and W % 32 == 0
        self.K, self.W, self.H = K, W, H
        self.H2 = K // 2
        self.PADL = 32 + self.H2
        self.S = ((W + 2 * self.PADL + 31) // 32) * 32
        self.R = H // CORES
        self.cols = iron.get_current_device().cols
        params = dict(params or {})
        if post == "decay":
            params["decay_q15"] = int(round(params.get("decay", 0.9) * 32767))
        self.post_code = POSTS[post].format(**params)
        self.variant = f"conv2d-{K}-{W}-{H}-{post}-{sorted(params.items())}"
        kq = np.zeros((K, 32), np.int16)
        t = np.asarray(taps, np.float64)
        assert t.shape == (K, K) and abs(t.sum()) <= 1.0001, "kernel taps must sum to <= 1 (int32 accumulator)"
        kq[:, :K] = np.clip(np.round(t * (1 << KSHIFT)), -32768, 32767)
        self.kq = kq
        n = (H + K - 1) * self.S
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")
        self._kern = iron.tensor(kq.ravel(), dtype=np.int16, device="npu")
        self.buf = np.zeros((H + K - 1, self.S), np.int16)
        self.npu_ms = 0.0

    def run(self, world_q15):
        """world_q15: (H, W) int16 -> (H, W) int16 after one convolution + post pass (torus halos)"""
        H, W, H2, PADL, S = self.H, self.W, self.H2, self.PADL, self.S
        a = self.buf
        a[H2:H2 + H, PADL:PADL + W] = world_q15
        a[:H2] = a[H:H + H2]; a[H + H2:] = a[H2:2 * H2]
        a[:, PADL - H2:PADL] = a[:, PADL + W - H2:PADL + W]
        a[:, PADL + W:PADL + W + H2] = a[:, PADL:PADL + H2]
        with self._in.overwrite() as x:
            x[...] = a.ravel()
        t0 = time.perf_counter()
        conv_npu(self._in, self._kern, self._out, cols=self.cols, K=self.K, W=W, R=self.R, S=S, PADL=PADL,
                 post=self.post_code, variant=self.variant)
        self.npu_ms = (time.perf_counter() - t0) * 1e3
        return self._out.numpy().reshape(H + self.K - 1, S)[H2:H2 + H, PADL:PADL + W].copy()


def reference(world, taps):
    """circular convolution, float"""
    K = taps.shape[0]; H2 = K // 2
    kp = np.zeros(world.shape); kp[:K, :K] = taps
    kp = np.roll(np.roll(kp, -H2, 0), -H2, 1)
    return np.real(np.fft.ifft2(np.fft.fft2(world) * np.fft.fft2(kp)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 1, (512, 512))
    wq = np.round(w * 32767).astype(np.int16)
    for K, sigma in ((3, 0.8), (7, 1.5), (15, 3.0)):
        y, x = np.mgrid[-(K // 2):K // 2 + 1, -(K // 2):K // 2 + 1]
        g = np.exp(-(x * x + y * y) / (2 * sigma * sigma)); g /= g.sum()
        c = Conv2D(K, g, post="copy")
        got = c.run(wq).astype(np.float64) / 32767
        c.run(wq)
        ts = []
        for _ in range(10):
            c.run(wq); ts.append(c.npu_ms)
        ref = reference(wq / 32767, g)
        print(f"K={K:2d}: max|err| {np.abs(got - ref).max() * 32767:.1f} LSB, rms {np.sqrt(((got - ref) ** 2).mean()) * 32767:.2f} LSB; "
              f"NPU {np.median(ts):.2f} ms per 512x512 ({512 * 512 * K * K / (np.median(ts) * 1e-3) / 1e9:.0f} GMAC/s)")
