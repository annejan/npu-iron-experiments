#!/usr/bin/env python3
"""Room acoustics: a 3-D FDTD wave simulation on all 32 AIE cores, recording an impulse response you can
then play through fun/reverb.py.

  ./env.sh python fun/room.py                 simulate 1 s at 8 kHz, write room_ir_8k.wav, room_ir_48k.wav, room.png
  ./env.sh python fun/room.py --steps 2000    shorter

Grid NX x NY x NZ = 32 x 32 x 128 cells, dx = c*dt*sqrt(3)/... we use lambda = c dt / dx = 0.5 (< 1/sqrt(3), stable),
so at fs = 8 kHz, dx = 343 / 8000 / 0.5 = 8.6 cm: a 2.7 m x 2.7 m x 11 m corridor. Pressure p in int32 Q28
(int16 Q14 was tried first: the leapfrog update loses its energy within 100 ms once amplitudes are tens of LSB,
and a numpy emulation with exact round-to-nearest does the same, so it is the format, not the chip):
  p_next = g * (2 p - p_prev + lambda^2 (sum of 6 neighbours - 6 p)),  lambda^2 = 1/4, g = damping (per step)
Walls: p = 0 one cell outside the box (pressure-release, phase-inverting; fine for an IR demo). The field is
streamed through the chip once per step like lbm.py: each core owns R = 4 z-planes and sees one halo plane on
each side via a 3-D shim DMA pattern; an object is one z-plane holding [p | p_prev]; the core keeps a 3-plane
sliding window. The host injects a short pulse at the source for the first steps and records the pressure at
two receiver cells (left/right ear, 20 cm apart) every step: that is the stereo impulse response.
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
NX, NY = 32, 32
R = 4
NZ = CORES * R
PL = NX * NY                # cells per plane
OBJ = 2 * PL                # [p | p_prev] int32
FS = 8000
LAMBDA = 0.5
C = 343.0
DX = C / FS / LAMBDA
Q = 28
PARAMS = dict(damp=0.999)


def ksrc(P):
    g = P["damp"]; l2 = LAMBDA ** 2
    c_p = round(g * (2 - 6 * l2) * 16384)          # coefficient of p        (Q14)
    c_pp = round(-g * 16384)                        # coefficient of p_prev
    c_n = round(g * l2 * 16384)                     # coefficient of each neighbour
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define NX {NX}
#define NY {NY}
#define PL {PL}
using V = aie::vector<int32_t, 16>;
static inline V ld(const int32_t *p) {{ return aie::load_v<16>(p); }}
static inline V ldu(const int32_t *p) {{ return aie::load_unaligned_v<16>(p); }}
extern "C" {{
void fdtd_plane(int32_t *__restrict zm, int32_t *__restrict cur, int32_t *__restrict zp, int32_t *__restrict out) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::saturate);
    const int32_t *p = cur, *pp = cur + PL, *pm = zm, *pz = zp;
    int32_t *po = out, *ppo = out + PL;
    const V zero = aie::zeros<int32_t, 16>();
    for (int y = 0; y < NY; y++) {{
        for (int x = 0; x < NX; x += 16) {{
            const int i = y * NX + x;
            if (y == 0 || y == NY - 1) {{ aie::store_v(po + i, zero); aie::store_v(ppo + i, zero); continue; }}
            V pc = ld(p + i);
            aie::accum<acc64, 16> acc = aie::mul(pc, (int16_t){c_p});
            acc = aie::mac(acc, ld(pp + i), (int16_t){c_pp});
            acc = aie::mac(acc, ldu(p + i - 1), (int16_t){c_n});
            acc = aie::mac(acc, ldu(p + i + 1), (int16_t){c_n});
            acc = aie::mac(acc, ld(p + i - NX), (int16_t){c_n});
            acc = aie::mac(acc, ld(p + i + NX), (int16_t){c_n});
            acc = aie::mac(acc, ld(pm + i), (int16_t){c_n});
            acc = aie::mac(acc, ld(pz + i), (int16_t){c_n});
            aie::store_v(po + i, acc.to_vector<int32_t>(14));
            aie::store_v(ppo + i, pc);                       // this step's p becomes next step's p_prev
        }}
        // side walls: x = 0 and x = NX-1 stay 0 (pressure release)
        po[y * NX] = 0; po[y * NX + NX - 1] = 0;
    }}
}}
}}
"""


def ext():
    obj_ty = np.ndarray[(OBJ,), np.dtype[np.int32]]
    return ExternalFunction("fdtd_plane", source_string=ksrc(PARAMS), arg_types=[obj_ty] * 4)


@iron.jit
def room_npu(field: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS
    n = (NZ + 2) * OBJ
    f_ty = np.ndarray[(n,), np.dtype[np.int32]]
    obj_ty = np.ndarray[(OBJ,), np.dtype[np.int32]]
    quad_ty = np.ndarray[(ROWS * OBJ,), np.dtype[np.int32]]
    k = ext()
    col_in = [ObjectFifo(quad_ty, name=f"in{c}", depth=2) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    pi, po = [], []
    for c in range(cols):
        pi += col_in[c].cons().split([i * OBJ for i in range(ROWS)], tile=Tile(c, 1), depths=[3] * ROWS,
                                     obj_types=[obj_ty] * ROWS, names=[f"pi{c}_{i}" for i in range(ROWS)])
        po += col_out[c].prod().join([i * OBJ for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                     obj_types=[obj_ty] * ROWS, names=[f"po{c}_{i}" for i in range(ROWS)])

    def core(fi, fo, k):
        for _ in range_(R):
            win = fi.acquire(3)
            o = fo.acquire(1)
            k(win[0], win[1], win[2], o)
            fo.release(1)
            fi.release(1)
        fi.release(2)

    workers = [Worker(core, [pi[j].cons(), po[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    taps_in = [TensorAccessPattern((n,), c * ROWS * R * OBJ, [R + 2, ROWS, OBJ], [OBJ, R * OBJ, 1]) for c in range(cols)]
    taps_out = [TensorAccessPattern((n,), (1 + c * ROWS * R) * OBJ, [R, ROWS, OBJ], [OBJ, R * OBJ, 1]) for c in range(cols)]
    ip, oc = [f.prod() for f in col_in], [f.cons() for f in col_out]

    def seq(field, out, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            ip[c].fill(field, taps_in[c], group=tg)
        for c in range(cols):
            oc[c].drain(out, taps_out[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [f_ty, f_ty, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class Room:
    def __init__(self):
        self.cols = iron.get_current_device().cols
        n = (NZ + 2) * OBJ
        self._in = iron.zeros((n,), dtype=np.int32, device="npu")
        self._out = iron.zeros((n,), dtype=np.int32, device="npu")
        self.src = (NX // 2 - 6, NY // 2, 20)              # source 1.7 m into the corridor, 50 cm off-axis (gives the ears a time difference)
        self.rx = [(NX // 2 - 1, NY // 2 + 2, 96), (NX // 2 + 1, NY // 2 + 2, 96)]   # two ears, 2 cells (17 cm) apart, 8 m away
        self.t = 0

    def idx(self, x, y, z, field=0):
        return (z + 1) * OBJ + field * PL + y * NX + x

    def step(self, inject=0.0):
        if inject:
            with self._in.overwrite() as a:
                x, y, z = self.src
                a[self.idx(x, y, z)] = int(np.clip(a[self.idx(x, y, z)] + inject * (1 << Q), -(1 << 31) + 1, (1 << 31) - 1))
        t0 = time.perf_counter()
        room_npu(self._in, self._out, cols=self.cols, variant="room2-int32-" + "-".join(f"{k}{v}" for k, v in PARAMS.items()))
        ms = (time.perf_counter() - t0) * 1e3
        self._in, self._out = self._out, self._in          # halo planes 0 and NZ+1 are never written: stay zero
        self.t += 1
        return ms

    def field(self):
        return self._in.numpy().reshape(NZ + 2, 2, NY, NX)[1:NZ + 1, 0].astype(np.float32) / (1 << Q)

    def receivers(self, a):
        return [a[self.idx(x, y, z)] / (1 << Q) for (x, y, z) in self.rx]


def reference_step(p, pp, P):
    g = P["damp"]; l2 = LAMBDA ** 2
    s = np.zeros_like(p)
    s[1:-1, 1:-1, 1:-1] = (p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] + p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1]
                           + p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:])
    pn = g * ((2 - 6 * l2) * p - pp + l2 * s)
    pn[0] = pn[-1] = 0; pn[:, 0] = pn[:, -1] = 0; pn[:, :, 0] = pn[:, :, -1] = 0
    return pn, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=FS)
    ap.add_argument("--check", action="store_true", help="compare 30 steps with numpy")
    a = ap.parse_args()
    room = Room()
    print(f"room {NX * DX:.1f} x {NY * DX:.1f} x {NZ * DX:.1f} m, dx {DX * 100:.1f} cm, fs {FS} Hz, {NZ} planes on {CORES} cores")
    if a.check:
        p = np.zeros((NZ, NY, NX)); pp = np.zeros_like(p)
        x, y, z = room.src
        for i in range(30):
            inj = np.exp(-((i - 6) / 2.5) ** 2) * 0.5 if i < 14 else 0.0
            p[z, y, x] += inj
            room.step(inject=inj)
            p, pp = reference_step(p, pp, PARAMS)
            got = room.field()
            if i % 5 == 4:
                err = np.abs(got - p)
                print(f"step {i + 1}: max|err| {err.max():.5f} ({err.max() * (1 << Q):.1f} LSB) rms {np.sqrt((err ** 2).mean()):.2e}; max|p| {np.abs(p).max():.4f}")
        return
    ir = np.zeros((a.steps, 2), np.float32)
    ts = []
    t0 = time.perf_counter()
    for i in range(a.steps):
        inj = np.exp(-((i - 6) / 2.5) ** 2) * 0.5 if i < 14 else 0.0
        ms = room.step(inject=inj); ts.append(ms)
        raw = room._in.numpy()
        ir[i] = room.receivers(raw)
        if i == 150:
            snap = room.field()
    dt = time.perf_counter() - t0
    print(f"{a.steps} steps in {dt:.1f} s: NPU call med {np.median(ts):.2f} ms, loop {dt / a.steps * 1e3:.2f} ms/step incl. readback; "
          f"IR peak {np.abs(ir).max():.4f} at {np.abs(ir[:, 0]).argmax() / FS * 1e3:.1f} ms (direct path {np.hypot(*(np.array(room.rx[0][:3]) - np.array(room.src)) * DX).sum() if False else 0:.0f})")
    # Schroeder backward integration: energy decay curve, RT60 from the -5..-25 dB slope
    edc = np.cumsum((ir[::-1, 0] ** 2))[::-1]; edc /= edc[0]
    db = 10 * np.log10(edc + 1e-12)
    i5, i25 = np.argmax(db < -5), np.argmax(db < -25)
    t60 = (i25 - i5) / FS * 3 if i25 > i5 else float("nan")
    print(f"RT60 (Schroeder, -5..-25 dB slope) ~ {t60:.2f} s; energy -30 dB at {np.argmax(db < -30) / FS:.2f} s")
    import soundfile as sf
    irn = ir / np.abs(ir).max()
    sf.write("room_ir_8k.wav", irn, FS)
    t48 = np.arange(0, len(ir) / FS, 1 / 48000)
    ir48 = np.stack([np.interp(t48, np.arange(len(ir)) / FS, irn[:, c]) for c in range(2)], 1)
    sf.write("room_ir_48k.wav", ir48, 48000)
    import cv2
    img = np.clip(np.abs(snap[:, NY // 2, :]).T * 80, 0, 1) ** 0.6   # |p| in the horizontal mid-plane at t = 19 ms: x down, z across
    cv2.imwrite("room.png", cv2.resize((img * 255).astype(np.uint8), (NZ * 6, NX * 6), interpolation=cv2.INTER_NEAREST))
    print("wrote room_ir_8k.wav room_ir_48k.wav room.png")


if __name__ == "__main__":
    main()
