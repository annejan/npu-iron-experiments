#!/usr/bin/env python3
"""Lenia (continuous cellular automaton) on all 32 AIE cores: a 27x27 ring kernel convolution per step.

  ../env.sh python fun/lenia.py             check the convolution against numpy, measure steps/s, write lenia.png
  ../env.sh python fun/lenia.py --window    pygame
  ../env.sh python fun/lenia.py --check     static compile report

World 512x512 in int16 Q15 (values 0..1) in DDR, streamed through the chip every step like lbm.py: each core
takes a band of R rows plus K-1 halo rows (one 3-D shim DMA pattern per column), keeps the last K rows in a
circular buffer in L1 and emits one output row per input row once the window is full. The convolution is the
FIR primitive: for every kernel row, 32 outputs x 32 taps (27 used) with four sliding_mac<32,8> on a 64-lane
data window, summed over the K rows in one int32 accumulator. Kernel taps are int16 scaled by 2^KSHIFT.
Growth and update are a second, elementwise pass in bf16: u = conv; G = 2 exp(-(u-mu)^2 / (2 sigma^2)) - 1;
a' = clip(a + dt G, 0, 1).  Default species: Orbium (R=13, mu=0.15, sigma=0.015, dt=0.1), ring kernel
exp(4 - 1/(r(1-r))) on r = |x|/R, normalised to sum 1.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4
CORES = 32
W = 512
K = 27                     # kernel size (2R+1), <= 32
H2 = K // 2                # 13
PADL = 32 + H2             # left pad: the data window for output x starts at x + PADL - H2 = x + 32, aligned
S = ((W + 2 * PADL + 31) // 32) * 32   # row stride, multiple of 32
R = 16                     # output rows per core -> 512 rows
H = CORES * R
KSHIFT = 19                # kernel taps scaled by 2^19 (max tap ~ 0.01 -> ~5000), output >> 19
PARAMS = dict(R=13, mu=0.15, sigma=0.015, dt=0.1)


def ksrc(P):
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define K {K}
#define S {S}
#define PADL {PADL}
#define H2 {H2}
#define KSHIFT {KSHIFT}
using bf = bfloat16;
using V = aie::vector<bf, 32>;
using A = aie::accum<accfloat, 32>;
static inline V vb(float f) {{ return aie::broadcast<bf, 32>((bf)f); }}
static inline V tobf(const A &a) {{ return a.template to_vector<bf>(); }}

/* circular buffer of the last K input rows, in this core's data memory */
alignas(64) static int16_t ring[K * S];
static int head = 0;       // index of the oldest row
static int filled = 0;

extern "C" {{
void lenia_push(int16_t *__restrict row) {{
    int16_t *dst = ring + ((head + filled) % K) * S;
    for (int x = 0; x < S; x += 32) aie::store_v(dst + x, aie::load_v<32>(row + x));
    if (filled < K) filled++; else head = (head + 1) % K;
}}
/* one output row = the row at window centre, convolved with the K x K kernel, then growth */
void lenia_step(int16_t *__restrict kern, int16_t *__restrict centre_in, int16_t *__restrict out) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    const V mu = vb({P['mu']}f), inv2s2 = vb({-1.0 / (2 * P['sigma'] ** 2) * 1.4426950408889634}f);   // * log2(e): exp2
    const V dt = vb({P['dt']}f), one = vb(1.0f), zero = vb(0.0f), two = vb(2.0f);
    for (int x = 0; x < W; x += 32) {{
        aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
        for (int kr = 0; kr < K; kr++) {{
            const int16_t *d = ring + ((head + kr) % K) * S + x + PADL - H2;     // aligned (PADL - H2 = 32)
            aie::vector<int16_t, 64> dv = aie::concat(aie::load_v<32>(d), aie::load_v<32>(d + 32));
            aie::vector<int16_t, 32> kv = aie::load_v<32>(kern + kr * 32);
            acc = aie::sliding_mac<32, 8>(acc, kv, 0, dv, 0);
            acc = aie::sliding_mac<32, 8>(acc, kv, 8, dv, 8);
            acc = aie::sliding_mac<32, 8>(acc, kv, 16, dv, 16);
            acc = aie::sliding_mac<32, 8>(acc, kv, 24, dv, 24);
        }}
        // u in Q15 -> bf16 in [0, 1]
        V u = aie::to_float<bf>(acc.to_vector<int16_t>(KSHIFT), 15);
        V a = aie::to_float<bf>(aie::load_v<32>(centre_in + x + PADL), 15);
        V d = tobf(aie::mac(aie::mul(mu, vb(-1.0f)), u, one));             // u - mu
        V g = tobf(aie::mac(aie::mul(aie::exp2(tobf(aie::mul(tobf(aie::mul(d, d)), inv2s2))), two), one, vb(-1.0f)));   // 2 exp(..) - 1
        V an = aie::min(aie::max(tobf(aie::mac(aie::mul(a, one), g, dt)), zero), one);
        aie::store_v(out + x + PADL, aie::to_fixed<int16_t>(an, 15));
    }}
}}
}}
"""


def lenia_kernel(P):
    r = P["R"]
    y, x = np.mgrid[-H2:H2 + 1, -H2:H2 + 1]
    d = np.sqrt(x * x + y * y) / r
    k = np.where((d > 0) & (d < 1), np.exp(4 - 1 / np.clip(d * (1 - d), 1e-9, None)), 0.0)
    k /= k.sum()
    kq = np.zeros((K, 32), np.int16)
    kq[:, :K] = np.clip(np.round(k * (1 << KSHIFT)), -32768, 32767)
    return k, kq


def ext_push():
    row_ty = np.ndarray[(S,), np.dtype[np.int16]]
    return ExternalFunction("lenia_push", source_string=ksrc(PARAMS), arg_types=[row_ty])


def ext_step():
    row_ty = np.ndarray[(S,), np.dtype[np.int16]]
    k_ty = np.ndarray[(K * 32,), np.dtype[np.int16]]
    return ExternalFunction("lenia_step", source_string=ksrc(PARAMS), arg_types=[k_ty, row_ty, row_ty])


@iron.jit
def lenia_npu(world: In, kern: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS
    n = (H + K - 1) * S
    world_ty = np.ndarray[(n,), np.dtype[np.int16]]
    k_ty = np.ndarray[(K * 32,), np.dtype[np.int16]]
    row_ty = np.ndarray[(S,), np.dtype[np.int16]]
    quad_ty = np.ndarray[(ROWS * S,), np.dtype[np.int16]]
    push, step = ext_push(), ext_step()
    k_in = [ObjectFifo(k_ty, name=f"k{c}", depth=1) for c in range(cols)]
    col_in = [ObjectFifo(quad_ty, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    ri, ro = [], []
    for c in range(cols):
        ri += col_in[c].cons().split([i * S for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[row_ty] * ROWS, names=[f"ri{c}_{i}" for i in range(ROWS)])
        ro += col_out[c].prod().join([i * S for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[row_ty] * ROWS, names=[f"ro{c}_{i}" for i in range(ROWS)])

    def core(kf, fi, fo, push, step):
        kk = kf.acquire(1)
        for _ in range_(K - 1):              # fill the window
            r = fi.acquire(1); push(r); fi.release(1)
        for _ in range_(R):                  # one output row per further input row
            r = fi.acquire(1); push(r)
            o = fo.acquire(1)
            step(kk, r, o)                   # r is now the row H2 below the window centre? no: see note in push
            fo.release(1); fi.release(1)
        kf.release(1)

    workers = [Worker(core, [k_in[j // ROWS].cons(), ri[j].cons(), ro[j].prod(), push, step],
                      tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
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


class Lenia:
    def __init__(self, seed=1):
        self.cols = iron.get_current_device().cols
        self.k, self.kq = lenia_kernel(PARAMS)
        self.world = np.zeros((H, W), np.float32)
        self.seed(seed)
        n = (H + K - 1) * S
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")
        self._kern = iron.tensor(self.kq.ravel(), dtype=np.int16, device="npu")
        self.upload = True

    def seed(self, seed):
        rng = np.random.default_rng(seed)
        self.world[:] = 0
        yy, xx = np.mgrid[0:H, 0:W]
        for _ in range(40):                       # random soft blobs
            cx, cy, r = rng.integers(0, W), rng.integers(0, H), rng.uniform(8, 20)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            self.world += rng.uniform(0.3, 1.0) * np.exp(-d2 / (2 * r * r))
        self.world[:] = np.clip(self.world + rng.uniform(0, 0.05, self.world.shape), 0, 1)
        self.upload = True

    @staticmethod
    def fix_halos(a):
        """a: int16 (H + K - 1, S): wrap rows and pad columns (torus)"""
        a[:H2] = a[H:H + H2]; a[H + H2:] = a[H2:2 * H2]
        a[:, PADL - H2:PADL] = a[:, PADL + W - H2:PADL + W]
        a[:, PADL + W:PADL + W + H2] = a[:, PADL:PADL + H2]

    def pack(self):
        a = np.zeros((H + K - 1, S), np.int16)
        a[H2:H2 + H, PADL:PADL + W] = np.clip(np.round(self.world * 32767), 0, 32767)
        self.fix_halos(a)
        return a

    def step(self):
        if self.upload:
            with self._in.overwrite() as x:
                x[...] = self.pack().ravel()
            self.upload = False
        t0 = time.perf_counter()
        lenia_npu(self._in, self._kern, self._out, cols=self.cols, variant="lenia1-" + "-".join(f"{k}{v}" for k, v in PARAMS.items()))
        ms = (time.perf_counter() - t0) * 1e3
        with self._out.overwrite() as x:
            self.fix_halos(x.reshape(H + K - 1, S))
        self._in, self._out = self._out, self._in
        return ms

    def read(self):
        a = self._in.numpy().reshape(H + K - 1, S)
        self.world[:] = a[H2:H2 + H, PADL:PADL + W].astype(np.float32) / 32767
        return self.world

    def rgb(self):
        w = self.read()
        img = np.empty((H, W, 3), np.float32)
        img[..., 0] = np.clip(w * 1.6 - 0.3, 0, 1); img[..., 1] = np.clip(w * 1.2, 0, 1); img[..., 2] = np.clip(0.2 + w * 0.6, 0, 1)
        return (img * 255).astype(np.uint8)


def reference_step(world, k, P):
    """numpy float32 reference: circular convolution (torus) + growth"""
    n = world.shape
    kp = np.zeros(n, np.float64); kp[:K, :K] = k
    kp = np.roll(np.roll(kp, -H2, 0), -H2, 1)
    u = np.real(np.fft.ifft2(np.fft.fft2(world) * np.fft.fft2(kp)))
    g = 2 * np.exp(-((u - P["mu"]) ** 2) / (2 * P["sigma"] ** 2)) - 1
    return u, np.clip(world + P["dt"] * g, 0, 1)


def check():
    from aie.utils.compile import remarks
    d = Path(".remarks/rem_lenia"); d.mkdir(parents=True, exist_ok=True)
    for ef in (ext_push(), ext_step()):
        rep, why = remarks.analyze(ef, "aie2p", d)
        if rep is None:
            print(why); sys.exit(1)
        print(ef.name, "libcalls", rep.libcalls, "stack", rep.kernel_stack_bytes, "notes", rep.schedule_notes)
        for key, l in rep.loops.items():
            print("  loop", key, "ii", l.ii, "ns", l.ns, "zol", l.zol, "trips", l.trips, "bundles", l.bundle_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args()
    if a.check:
        return check()
    sim = Lenia()
    w0 = sim.world.copy()
    ms = sim.step()
    got = sim.read().copy()
    u_ref, w_ref = reference_step(w0.astype(np.float64), sim.k, PARAMS)
    err = np.abs(got - w_ref)
    print(f"first step (compile) {ms:.0f} ms; vs numpy: max |err| {err.max():.4f} rms {np.sqrt((err ** 2).mean()):.5f} "
          f"(world mean {got.mean():.4f} ref {w_ref.mean():.4f})")
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
                if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    sim.seed(int(time.time()))
            for _ in range(2):
                ms = sim.step()
            n += 2
            surf = pygame.surfarray.make_surface(np.transpose(sim.rgb(), (1, 0, 2)))
            screen.blit(pygame.transform.smoothscale(surf, screen.get_size()), (0, 0))
            screen.blit(font.render(f"Lenia {W}x{H} K={K}  step {n}  {ms:.2f} ms/step  {n / (time.perf_counter() - t0):.0f} steps/s  R: reseed", True, (255, 255, 255)), (8, 8))
            pygame.display.flip()
    ts = []
    for i in range(a.steps):
        ts.append(sim.step())
    print(f"{W}x{H}, {K}x{K} kernel: {np.median(ts):.2f} ms/step median (min {min(ts):.2f}) -> {1e3 / np.median(ts):.0f} steps/s, "
          f"{W * H * K * K / (np.median(ts) * 1e-3) / 1e9:.0f} GMAC/s; world mean {sim.read().mean():.4f}")
    import cv2
    cv2.imwrite("lenia.png", cv2.resize(sim.rgb()[..., ::-1], (W * a.scale, H * a.scale), interpolation=cv2.INTER_LINEAR))
    print("wrote lenia.png")


if __name__ == "__main__":
    main()
