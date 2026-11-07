#!/usr/bin/env python3
"""2-D Ising model on the chip: checkerboard Metropolis with an xorshift RNG in the vector lanes, checked
against Onsager's exact magnetisation.

  ../env.sh python fun/ising.py                 sweep temperatures, print |M| vs Onsager, write ising.png (lattice at Tc)
  ../env.sh python fun/ising.py --window        pygame: temperature with up/down keys

Spins s = +-1 as int16, world 512x512 in the 32 cores' L1 (slime.py layout: bands, halo rows over shared memory,
ping-pong). One call = STEPS half-sweeps; a half-sweep updates one checkerboard colour: dE = 2 s (sum of the 4
neighbours) in {-8..8}; flip if dE <= 0, else with probability exp(-dE/T), decided per lane against a 16-bit random
number from a per-core xorshift32 generator (static state in L1, seeded per core). Thresholds for dE = 4 and 8 are
compile-time constants, so the temperature is a variant of the design (recompile ~2 s, cached per T).
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
ROWS = 16                  # rows per core -> 512
TC = 2 / np.log(1 + np.sqrt(2))


def kcommon():
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define S {S}
#define ROWS {ROWS}
#define PADC {PADC}
using V = aie::vector<int16_t, 32>;
using U = aie::vector<int32_t, 16>;   // xorshift in int32 lanes; logical right shift = arithmetic shift + mask
static inline V ld(const int16_t *p) {{ return aie::load_v<32>(p); }}
static inline V ldu(const int16_t *p) {{ return aie::load_unaligned_v<32>(p); }}
"""


def kstep(T):
    t4 = int(min(1.0, np.exp(-4.0 / T)) * 32767)       # 15-bit random numbers
    t8 = int(min(1.0, np.exp(-8.0 / T)) * 32767)
    return kcommon() + f"""
alignas(64) static int32_t rng_state[32];
static int seeded = 0;
extern "C" {{
/* colour: which checkerboard colour flips this half-sweep (0/1); seed: per-core RNG seed */
void ising_step(int16_t *src, int16_t *dst, int16_t *above, int16_t *below, int32_t colour, int32_t seed) {{
    if (!seeded) {{
        for (int i = 0; i < 32; i++) rng_state[i] = (int32_t)(0x9E3779B9u * (uint32_t)(seed * 32 + i + 1) | 1u);
        seeded = 1;
    }}
    U r0 = aie::load_v<16>(rng_state), r1 = aie::load_v<16>(rng_state + 16);
    const V t4 = aie::broadcast<int16_t, 32>({t4});
    const V t8 = aie::broadcast<int16_t, 32>({t8});
    const U m15 = aie::broadcast<int32_t, 16>(0x7FFF), m17 = aie::broadcast<int32_t, 16>(0x7FFF);   // masks after >>17 (15 bits left)
    const V four = aie::broadcast<int16_t, 32>(4), eight = aie::broadcast<int16_t, 32>(8), zero = aie::zeros<int16_t, 32>();
    // lane parity mask: lanes with (x + y + colour) even are updated
    alignas(64) static const int16_t par[32] = {{0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1}};
    const V lanepar = aie::load_v<32>(par);
    for (int r = 0; r < ROWS; r++) {{
        const int16_t *s = src + r * S;
        const int16_t *up = r ? s - S : above;
        const int16_t *dn = r < ROWS - 1 ? s + S : below;
        int16_t *o = dst + r * S;
        const int rowpar = (r + colour) & 1;
        const aie::mask<32> update = aie::eq(lanepar, aie::broadcast<int16_t, 32>((int16_t)rowpar));   // (lane + r + colour) even
        for (int x = PADC; x < PADC + W; x += 32) {{
            V sc = ld(s + x);
            V nb = aie::add(aie::add(ldu(s + x - 1), ldu(s + x + 1)), aie::add(ld(up + x), ld(dn + x)));
            V dE = aie::mul(sc, nb).to_vector<int16_t>(0);          // s * nbsum; real dE is twice that: 2 s nb in {{-8..8}} -> compare with 2 and 4
            // xorshift32 on two 16-lane vectors -> 32 random 16-bit numbers
            r0 = aie::bit_xor(r0, aie::upshift(r0, 13)); r0 = aie::bit_xor(r0, aie::bit_and(aie::downshift(r0, 17), m17)); r0 = aie::bit_xor(r0, aie::upshift(r0, 5));
            r1 = aie::bit_xor(r1, aie::upshift(r1, 13)); r1 = aie::bit_xor(r1, aie::bit_and(aie::downshift(r1, 17), m17)); r1 = aie::bit_xor(r1, aie::upshift(r1, 5));
            V rnd = aie::concat(aie::bit_and(r0, m15).pack(), aie::bit_and(r1, m15).pack());   // 32 random 15-bit numbers
            aie::mask<32> flip = aie::le(dE, zero);
            flip = flip | (aie::eq(dE, aie::broadcast<int16_t, 32>(2)) & aie::lt(rnd, t4));
            flip = flip | (aie::eq(dE, four) & aie::lt(rnd, t8));
            flip = flip & update;
            aie::store_v(o + x, aie::select(sc, aie::sub(zero, sc), flip));
        }}
        o[PADC - 1] = o[PADC + W - 1]; o[PADC + W] = o[PADC];        // wrap pads
    }}
    aie::store_v(rng_state, r0); aie::store_v(rng_state + 16, r1);
}}
}}
"""


def krow():
    return kcommon() + f"""
extern "C" {{
void ising_row(int16_t *band, int16_t *edge, int32_t r) {{
    const int16_t *s = band + r * S;
    for (int i = 0; i < S; i += 32) aie::store_v(edge + i, aie::load_v<32>(s + i));
}}
}}
"""


@iron.jit
def ising_chip(world: In, out: Out, *, steps: CompileTime[int], cols: CompileTime[int], T: CompileTime[float], variant: CompileTime[str]):
    cores = cols * ROWS_PER_COL
    band = ROWS * S
    n = cores * band
    tensor_ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS_PER_COL * band,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(S,), np.dtype[np.int16]]
    step = ExternalFunction("ising_step", source_string=kstep(T), arg_types=[band_ty, band_ty, edge_ty, edge_ty, np.dtype(np.int32), np.dtype(np.int32)])
    row = ExternalFunction("ising_row", source_string=krow(), arg_types=[band_ty, edge_ty, np.dtype(np.int32)])
    col_in = [ObjectFifo(col_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    band_in, band_out = [], []
    for c in range(cols):
        kw = dict(tile=Tile(c, 1), depths=[1] * ROWS_PER_COL, obj_types=[band_ty] * ROWS_PER_COL)
        band_in += col_in[c].cons().split([i * band for i in range(ROWS_PER_COL)], names=[f"b_in{c}_{i}" for i in range(ROWS_PER_COL)], **kw)
        band_out += col_out[c].prod().join([i * band for i in range(ROWS_PER_COL)], names=[f"b_out{c}_{i}" for i in range(ROWS_PER_COL)], **kw)
    down = [ObjectFifo(edge_ty, name=f"down{k}", depth=2) for k in range(cores)]
    up = [ObjectFifo(edge_ty, name=f"up{k}", depth=2) for k in range(cores)]

    def core(b_in, b_out, to_up, to_down, from_above, from_below, step, row, seed):
        a = b_in.acquire(1)
        b = b_out.acquire(1)

        def half(s, d, colour):
            e = to_up.acquire(1); row(s, e, 0); to_up.release(1)
            e = to_down.acquire(1); row(s, e, ROWS - 1); to_down.release(1)
            above = from_above.acquire(1); below = from_below.acquire(1)
            step(s, d, above, below, colour, seed)
            from_above.release(1); from_below.release(1)

        for _ in range_(steps // 2):           # steps half-sweeps, pairs keep the ping-pong straight
            half(a, b, 0)
            half(b, a, 1)
        half(a, b, 0)
        half(b, b, 1)                          # last half-sweep in place? no: colours do not overlap, so b -> b is safe
        b_in.release(1); b_out.release(1)

    workers = [Worker(core, [band_in[k].cons(), band_out[k].prod(), up[k].prod(), down[k].prod(),
                             down[(k - 1) % cores].cons(), up[(k + 1) % cores].cons(), step, row, k], tile=Tile(*snake(k)))
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


class Ising:
    def __init__(self, T=TC, steps=20, seed=1):
        self.cols = iron.get_current_device().cols
        self.cores = self.cols * ROWS_PER_COL
        self.H = self.cores * ROWS
        self.T, self.steps = T, steps
        rng = np.random.default_rng(seed)
        self.s = rng.choice([-1, 1], (self.H, W)).astype(np.int16)
        n = self.cores * ROWS * S
        self._in = iron.zeros((n,), dtype=np.int16, device="npu")
        self._out = iron.zeros((n,), dtype=np.int16, device="npu")
        self.dirty = True

    def pack(self):
        a = np.zeros((self.H, S), np.int16)
        a[:, PADC:PADC + W] = self.s
        a[:, PADC - 1] = self.s[:, -1]; a[:, PADC + W] = self.s[:, 0]
        return a.ravel()

    def run(self):
        if self.dirty:
            with self._in.overwrite() as x:
                x[...] = self.pack()
            self.dirty = False
        t0 = time.perf_counter()
        ising_chip(self._in, self._out, steps=self.steps, cols=self.cols, T=float(self.T), variant=f"ising1-T{self.T:.4f}")
        ms = (time.perf_counter() - t0) * 1e3
        self.s[:] = self._out.numpy().reshape(self.H, S)[:, PADC:PADC + W]
        self._in, self._out = self._out, self._in
        return ms

    def observables(self):
        s = self.s.astype(np.float64)
        m = abs(s.mean())
        e = -(s * (np.roll(s, 1, 0) + np.roll(s, 1, 1))).mean()
        return m, e


def onsager_m(T):
    return (1 - np.sinh(2 / T) ** -4) ** 0.125 if T < TC else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--sweeps", type=int, default=3000, help="full sweeps per temperature (2 half-sweeps each)")
    a = ap.parse_args()
    if a.window:
        import pygame
        sim = Ising(T=2.0, steps=10)
        pygame.init()
        screen = pygame.display.set_mode((W * 2, sim.H * 2)); font = pygame.font.SysFont(None, 24)
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_UP: sim.T = round(sim.T + 0.05, 3)
                if e.type == pygame.KEYDOWN and e.key == pygame.K_DOWN: sim.T = round(max(0.5, sim.T - 0.05), 3)
            ms = sim.run(); m, en = sim.observables()
            img = np.repeat(((sim.s + 1) * 127).astype(np.uint8)[:, :, None], 3, 2)
            screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2))), screen.get_size()), (0, 0))
            screen.blit(font.render(f"T = {sim.T:.2f} (Tc {TC:.3f})  |M| {m:.3f} (Onsager {onsager_m(sim.T):.3f})  E {en:.3f}  {sim.steps} half-sweeps {ms:.1f} ms   up/down: T", True, (255, 60, 60)), (8, 8))
            pygame.display.flip()
    import cv2
    print(f"{W}x{32 * ROWS} lattice, {a.sweeps} sweeps per T after a quench from random")
    for T in (1.5, 2.0, 2.2, TC, 2.4, 2.7, 3.5):
        sim = Ising(T=T, steps=20)
        sim.run()
        t0 = time.perf_counter()
        calls = a.sweeps * 2 // 20
        for _ in range(calls): sim.run()
        dt = time.perf_counter() - t0
        m, en = sim.observables()
        print(f"T = {T:.3f}: |M| = {m:.4f}  Onsager {onsager_m(T):.4f}   E/N = {en:.4f}   {calls * 20 / 2 / dt:.0f} sweeps/s ({W * 32 * ROWS * calls * 10 / dt / 1e9:.1f} Gspin-updates/s)")
        if abs(T - TC) < 1e-6:
            cv2.imwrite("ising.png", ((sim.s + 1) * 127).astype(np.uint8))
    print("wrote ising.png (lattice at Tc)")


if __name__ == "__main__":
    main()
