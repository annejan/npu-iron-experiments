#!/usr/bin/env python3
"""The resident-world probe: the ripple tank's state lives in each core's L1 across dispatches. Per call the host
sends one 32-int command (drop x, y, amplitude) and reads the height field back; the world never goes to DDR.

  ./env.sh python fun/water_resident.py        compare with water.py (bit-identical?), time per call, autosuspend test

Mechanism: `Buffer`s for the two band buffers (h and v rows, int16 Q12) per Worker, Workers in their default
`while_true` loop where one loop iteration is one dispatch: the iteration starts by acquiring the command object
from a broadcast ObjectFifo (so it blocks until the host sends one), applies the drop, runs STEPS (even) generations
with the halo exchange of water.py, and pushes its h rows into the output fifo. amdxdna suspends the device after
5 s idle; whether the cores come back with their memory is what the last test measures.
"""
import sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
import water as Wt
from water import W, S, ROWS, FIELDS, PARAMS, Q, kstep, krow
from slime import snake, ROWS_PER_COL, PADC

NCMD = 32


def kcmd():
    return Wt.kcommon() + f"""
extern "C" {{
/* cmd = [x, y, amp, ...]: add a drop if row y falls in this band (row0 = the band's first world row) */
void water_cmd(int16_t *band, int32_t *cmd, int32_t row0) {{
    if (cmd[2] == 0) return;
    int y = cmd[1] - row0;
    if (y < 0 || y >= ROWS) return;
    int16_t *h = band + y * 2 * S + PADC + cmd[0];
    int v = (int)*h + cmd[2];
    *h = (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
}}
}}
"""


def khout():
    return Wt.kcommon() + f"""
extern "C" {{
/* copy the h rows of a band into the output object */
void water_hout(int16_t *band, int16_t *out) {{
    for (int r = 0; r < ROWS; r++)
        for (int x = 0; x < S; x += 32) aie::store_v(out + r * S + x, aie::load_v<32>(band + r * 2 * S + x));
}}
}}
"""


@iron.jit
def water_resident(cmd: In, out: Out, *, steps: CompileTime[int], cols: CompileTime[int], variant: CompileTime[str]):
    assert steps % 2 == 0
    cores = cols * ROWS_PER_COL
    band = ROWS * FIELDS * S
    hband = ROWS * S
    n_out = cores * hband
    cmd_ty = np.ndarray[(NCMD,), np.dtype[np.int32]]
    out_ty = np.ndarray[(n_out,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS_PER_COL * hband,), np.dtype[np.int16]]
    hband_ty = np.ndarray[(hband,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(S,), np.dtype[np.int16]]
    step = ExternalFunction("water_step", source_string=kstep(PARAMS), arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    row = ExternalFunction("water_row", source_string=krow(), arg_types=[band_ty, edge_ty, np.dtype(np.int32)])
    cmdk = ExternalFunction("water_cmd", source_string=kcmd(), arg_types=[band_ty, cmd_ty, np.dtype(np.int32)])
    hout = ExternalFunction("water_hout", source_string=khout(), arg_types=[band_ty, hband_ty])
    cmd_in = [ObjectFifo(cmd_ty, name=f"cmd{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    band_out = []
    for c in range(cols):
        band_out += col_out[c].prod().join([i * hband for i in range(ROWS_PER_COL)], tile=Tile(c, 1), depths=[1] * ROWS_PER_COL,
                                           obj_types=[hband_ty] * ROWS_PER_COL, names=[f"b_out{c}_{i}" for i in range(ROWS_PER_COL)])
    down = [ObjectFifo(edge_ty, name=f"down{k}", depth=2) for k in range(cores)]
    up = [ObjectFifo(edge_ty, name=f"up{k}", depth=2) for k in range(cores)]
    zeros = np.zeros(band, np.int16)
    bufs_a = [Buffer(band_ty, initial_value=zeros, name=f"wa{k}") for k in range(cores)]
    bufs_b = [Buffer(band_ty, initial_value=zeros, name=f"wb{k}") for k in range(cores)]

    def core(cf, fo, to_up, to_down, from_above, from_below, wa, wb, step, row, cmdk, hout, row0):
        c = cf.acquire(1)                      # one iteration per dispatch: blocks until the host sends a command
        cmdk(wa, c, row0)
        cf.release(1)

        def generation(s, d):
            e = to_up.acquire(1); row(s, e, 0); to_up.release(1)
            e = to_down.acquire(1); row(s, e, ROWS - 1); to_down.release(1)
            above = from_above.acquire(1); below = from_below.acquire(1)
            step(s, d, above, below)
            from_above.release(1); from_below.release(1)

        for _ in range_(steps // 2):
            generation(wa, wb)
            generation(wb, wa)
        o = fo.acquire(1)
        hout(wa, o)
        fo.release(1)

    def src_above(k): return up[k].cons() if k == 0 else down[k - 1].cons()
    def src_below(k): return down[k].cons() if k == cores - 1 else up[k + 1].cons()
    workers = [Worker(core, [cmd_in[snake(k)[0]].cons(), band_out[k].prod(), up[k].prod(), down[k].prod(),
                             src_above(k), src_below(k), bufs_a[k], bufs_b[k], step, row, cmdk, hout, k * ROWS],
                      tile=Tile(*snake(k))) for k in range(cores)]
    taps = TensorAccessPattern.full((n_out,)).partition(cols)
    cp, oc = [f.prod() for f in cmd_in], [f.cons() for f in col_out]
    cfull = TensorAccessPattern.full((NCMD,))

    def sequence(cmd, out, cp, oc):
        tg = TaskGroup()
        for c in range(cols):
            cp[c].fill(cmd, cfull, group=tg)
        for c in range(cols):
            oc[c].drain(out, taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [cmd_ty, out_ty, cp, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class ResidentWater:
    def __init__(self, steps=4):
        self.cols = iron.get_current_device().cols
        self.cores = self.cols * ROWS_PER_COL
        self.H = self.cores * ROWS
        self.steps = steps
        self._cmd = iron.zeros((NCMD,), dtype=np.int32, device="npu")
        self._out = iron.zeros((self.cores * ROWS * S,), dtype=np.int16, device="npu")

    def call(self, drop=None):
        with self._cmd.overwrite() as c:
            c[...] = 0
            if drop: c[0], c[1], c[2] = drop
        t0 = time.perf_counter()
        water_resident(self._cmd, self._out, steps=self.steps, cols=self.cols, variant="resident1")
        ms = (time.perf_counter() - t0) * 1e3
        return ms

    def h(self):
        # output bands are in snake order k (joined per column in placement order): core k wrote band k; the join
        # order within a column follows the split index i = position in the column list, so unscramble by snake
        raw = self._out.numpy().reshape(self.cols, ROWS_PER_COL, ROWS, S)
        out = np.zeros((self.H, W), np.float32)
        for k in range(self.cores):
            c, r = snake(k)
            out[k * ROWS:(k + 1) * ROWS] = raw[c, r - 2, :, PADC:PADC + W].astype(np.float32) / (1 << Q)
        return out


def main():
    # 1. reference: water.py with the same drop and step count (its world goes through DDR every call)
    ref = Wt.Water(steps=5)                 # water.py needs odd steps; use 5 and compare after matching totals
    ref.h[:] = 0; ref.v[:] = 0; ref.dirty = True
    res = ResidentWater(steps=4)
    res.call()                               # compile (the world is still all zero)
    # the resident sim: a drop of 0.8 at (160, 192) then 20 calls x 4 steps = 80 steps
    amp = int(round(0.8 * (1 << Q)))
    t_first = res.call(drop=(160, 192, amp))
    ts = [res.call() for _ in range(19)]
    hr = res.h()
    # reference: same drop, 80 steps (16 x 5)
    ref.h[192, 160] += 0.8; ref.dirty = True
    for _ in range(16): ref.run()
    diff = np.abs(hr - ref.h).max()
    print(f"resident vs water.py after 80 steps from one drop: max |diff| {diff * (1 << Q):.1f} LSB (rms {np.sqrt(((hr - ref.h) ** 2).mean()) * (1 << Q):.2f} LSB); "
          f"energy resident {(hr ** 2).sum():.3f} ref {(ref.h ** 2).sum():.3f}")
    print(f"per call, 4 steps, 320x384: resident {np.median(ts):.3f} ms (no world upload/download) vs water.py {np.median([ref.run() for _ in range(10)]):.3f} ms (5 steps, world through DDR)")
    # 2. autosuspend: idle 6 s, then call again: does the world continue or come back reset?
    e_before = (res.h() ** 2).sum()
    time.sleep(6.5)
    res.call()
    e_after = (res.h() ** 2).sum()
    print(f"after 6.5 s idle: energy before {e_before:.3f}, after one more call {e_after:.3f} -> {'STATE SURVIVED the autosuspend' if e_after > 0.5 * e_before else 'state LOST (cores reset)'}")
    import cv2
    img = np.clip(np.abs(res.h()) * 4, 0, 1)
    cv2.imwrite("water_resident.png", (img * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
