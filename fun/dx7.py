#!/usr/bin/env python3
"""A DX7-style FM synthesizer on the NPU: 512 six-operator voices, sixteen voices in the lanes of every core.

  ./env.sh python fun/dx7.py                 check the NPU against a numpy replica (bit-exact), render dx7_demo.wav
  ./env.sh python fun/dx7.py --window        play it: keys z s x d c v g b h n j m , l . (lower octave) q 2 w 3 e r 5 t 6 y 7 u i
                                             (upper), 1-5 pick a patch, up/down transpose, ESC quits (needs pygame, sounddevice)
  ./env.sh python fun/dx7.py --patch bells --seconds 20 --out x.wav

The model is the DX7's: six sine operators per voice, each with a phase accumulator, a frequency ratio (or a
fixed frequency), an output level and a 4-rate / 4-level envelope; an algorithm (the 32 of the DX7, after the
chart in the manual) says which operator modulates which operator's phase and which operators are carriers that
reach the output; one operator feeds back on itself. Phase modulation of a sine is the whole trick.

On the chip: the voices are the lanes (16 int32 lanes per core, 32 cores: 512 voices), time is the loop, so the
per-sample recursion of the feedback operator is natural. Per sample and operator: the phase (Q32 turns) advances
by the lane's increment, the modulators' outputs (Q15) times their modulation depth are added (wrap-around
arithmetic is exactly sin(x + 2 pi) = sin(x)), the top 12 bits of the phase index a 4096-entry sine table through
the lookup unit (`aie::parallel_lookup`, 16 lanes per fetch), the sine is scaled by the operator's level (a
per-sample Q31 ramp between the envelope values the host computes per 256-sample block), carriers are summed,
and the 16 voices are mixed with `reduce_add` into one int32 sample. The host does envelopes, voice allocation
and the keyboard; the chip does 6 x 512 operators at 48 kHz for a few percent of one core per block. Mapping of
the DX7's numbers to sound (output level in 0.75 dB steps, rates to seconds, modulation depth) is approximate.

Yamaha's DX7 patches are not included; the five here are made up for the demo (electric piano, bass, bells,
brass, pad), but any patch is six operators of (ratio, level, envelope) plus an algorithm number.
"""
import argparse, sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

ROWS = 4; CORES = 32
L = 16                      # voices per core (lanes)
OPS = 6
B = 256                     # samples per block (5.3 ms at 48 kHz)
FS = 48000
TAB = 4096                  # sine table entries (int16 Q15)
NVOICES = CORES * L
# per-core input object (int32): ph[6][16], dph[6][16], lvl[6][16] (Q31), dlvl[6][16], fb1[16], fb2[16], patch[48], table[TAB] (as int32 pairs? no: int16 half)
PATCH_N = OPS * 8           # per op: nsrc, s0, s1, s2, K0, K1, K2, fbK|carrier<<16
IN_N = 4 * OPS * L + 2 * L + PATCH_N + TAB // 2     # the table rides along as 2048 int32 = 4096 int16
OUT_N = OPS * L + 2 * L + B                         # ph out, fb state, mix

KSRC = f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define L {L}
#define OPS {OPS}
#define B {B}
#define TAB {TAB}
using V32 = aie::vector<int32_t, L>;
using V16 = aie::vector<int16_t, L>;
alignas(64) static int16_t lut_ab[2 * TAB];
alignas(64) static int16_t lut_cd[2 * TAB];
static int built = 0;
extern "C" {{
void dx7_block(int32_t *__restrict in, int32_t *__restrict out) {{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::set_saturation(aie::saturation_mode::none);           // phases wrap: sin(x + 2 pi) = sin(x)
    int32_t *ph = in, *dph = in + OPS * L, *lvl = in + 2 * OPS * L, *dlvl = in + 3 * OPS * L;
    int32_t *fb = in + 4 * OPS * L, *patch = fb + 2 * L;
    const int16_t *table = (const int16_t *)(patch + OPS * 8);
    if (!built) {{                                              // run-duplicated copies for the lookup unit (16 x 16-bit runs)
        for (unsigned d = 0; d < TAB; d++) {{
            unsigned i = (d / 16) * 32 + d % 16;
            lut_ab[i] = lut_cd[i] = table[d]; lut_ab[i + 16] = lut_cd[i + 16] = table[d];
        }}
        built = 1;
    }}
    using lut_t = aie::lut<4, int16, int16>;
    lut_t sine(TAB, lut_ab, lut_cd);
    aie::parallel_lookup<uint16, lut_t, aie::lut_oor_policy::truncate> look(sine, 0);
    V32 P[OPS], D[OPS], A[OPS], DA[OPS];
    V16 O[OPS];
    for (int o = 0; o < OPS; o++) {{
        P[o] = aie::load_v<L>(ph + o * L); D[o] = aie::load_v<L>(dph + o * L);
        A[o] = aie::load_v<L>(lvl + o * L); DA[o] = aie::load_v<L>(dlvl + o * L);
        O[o] = aie::zeros<int16_t, L>();
    }}
    V16 fb1 = aie::load_v<L>(fb).pack(), fb2 = aie::load_v<L>(fb + L).pack();
    int fbop = -1;
    for (int o = 0; o < OPS; o++) if (patch[o * 8 + 7] & 0x8000) fbop = o;
    for (int n = 0; n < B; n++) {{
        V32 acc = aie::zeros<int32_t, L>();
        for (int o = OPS - 1; o >= 0; o--) {{                    // operator 6 first: modulators before carriers
            const int32_t *pp = patch + o * 8;
            P[o] = aie::add(P[o], D[o]);
            V32 m = P[o];
            for (int s = 0; s < pp[0]; s++) {{
                V32 src = O[pp[1 + s]].unpack();
                m = aie::add(m, aie::mul(src, pp[4 + s]).to_vector<int32_t>(0));
            }}
            if (o == fbop) {{
                V32 avg = aie::downshift(aie::add(fb1.unpack(), fb2.unpack()), 1);
                m = aie::add(m, aie::mul(avg, pp[7] & 0x7fff).to_vector<int32_t>(0));
            }}
            V32 idx32 = aie::bit_and((int32_t)(TAB - 1), aie::logical_downshift(m, 20));
            aie::vector<uint16_t, L> idx = idx32.pack().cast_to<uint16_t>();
            V16 sn = look.fetch(idx);
            V16 lv = aie::downshift(A[o], 16).pack();            // Q31 level -> Q15
            O[o] = aie::mul(sn, lv).to_vector<int16_t>(15);
            A[o] = aie::add(A[o], DA[o]);
            if (pp[7] & 0x10000) acc = aie::add(acc, O[o].unpack());
        }}
        if (fbop >= 0) {{ fb2 = fb1; fb1 = O[fbop]; }}
        out[OPS * L + 2 * L + n] = aie::reduce_add(acc);
    }}
    for (int o = 0; o < OPS; o++) aie::store_v(out + o * L, P[o]);
    aie::store_v(out + OPS * L, fb1.unpack()); aie::store_v(out + OPS * L + L, fb2.unpack());
}}
}}
"""


@iron.jit
def dx7_npu(inp: In, out: Out, *, cols: CompileTime[int], variant: CompileTime[str]):
    cores = cols * ROWS
    in_ty = np.ndarray[(cores * IN_N,), np.dtype[np.int32]]
    out_ty = np.ndarray[(cores * OUT_N,), np.dtype[np.int32]]
    cin_ty = np.ndarray[(ROWS * IN_N,), np.dtype[np.int32]]
    cout_ty = np.ndarray[(ROWS * OUT_N,), np.dtype[np.int32]]
    obj_in = np.ndarray[(IN_N,), np.dtype[np.int32]]
    obj_out = np.ndarray[(OUT_N,), np.dtype[np.int32]]
    k = ExternalFunction("dx7_block", source_string=KSRC, arg_types=[obj_in, obj_out])
    col_in = [ObjectFifo(cin_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(cout_ty, name=f"out{c}", depth=1) for c in range(cols)]
    ci, co = [], []
    for c in range(cols):
        ci += col_in[c].cons().split([i * IN_N for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS, obj_types=[obj_in] * ROWS, names=[f"ci{c}_{i}" for i in range(ROWS)])
        co += col_out[c].prod().join([i * OUT_N for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS, obj_types=[obj_out] * ROWS, names=[f"co{c}_{i}" for i in range(ROWS)])

    def core(fi, fo, k):
        x = fi.acquire(1); y = fo.acquire(1); k(x, y); fo.release(1); fi.release(1)

    workers = [Worker(core, [ci[j].cons(), co[j].prod(), k], tile=Tile(j // ROWS, 2 + j % ROWS)) for j in range(cores)]
    tin = TensorAccessPattern.full((cores * IN_N,)).partition(cols)
    tout = TensorAccessPattern.full((cores * OUT_N,)).partition(cols)
    ip, oc = [f.prod() for f in col_in], [f.cons() for f in col_out]

    def seq(inp, out, ip, oc):
        tg = TaskGroup()
        for c in range(cols):
            ip[c].fill(inp, tin[c], group=tg)
        for c in range(cols):
            oc[c].drain(out, tout[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [in_ty, out_ty, ip, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# ----------------------------------------------------------------------------------------------- the DX7 model
# algorithm -> (modulators of each op 1..6, feedback op), after the DX7 algorithm chart; carriers = ops nobody modulates
# into except the output, i.e. the ops listed in `carriers`
ALGOS = {
    1: ({1: [2], 3: [4], 4: [5], 5: [6]}, [1, 3], 6),
    2: ({1: [2], 3: [4], 4: [5], 5: [6]}, [1, 3], 2),
    3: ({1: [2], 2: [3], 4: [5], 5: [6]}, [1, 4], 6),
    4: ({1: [2], 2: [3], 4: [5], 5: [6]}, [1, 4], 4),
    5: ({1: [2], 3: [4], 5: [6]}, [1, 3, 5], 6),
    6: ({1: [2], 3: [4], 5: [6]}, [1, 3, 5], 5),
    7: ({1: [2], 3: [4, 5], 5: [6]}, [1, 3], 6),
    8: ({1: [2], 3: [4, 5], 5: [6]}, [1, 3], 4),
    9: ({1: [2], 3: [4, 5], 5: [6]}, [1, 3], 2),
    10: ({1: [2], 2: [3], 4: [5, 6]}, [1, 4], 3),
    11: ({1: [2], 2: [3], 4: [5, 6]}, [1, 4], 6),
    12: ({1: [2], 3: [4, 5, 6]}, [1, 3], 2),
    13: ({1: [2], 3: [4, 5, 6]}, [1, 3], 6),
    14: ({1: [2], 3: [4], 4: [5, 6]}, [1, 3], 6),
    15: ({1: [2], 3: [4], 4: [5, 6]}, [1, 3], 2),
    16: ({1: [2, 3, 5], 3: [4], 5: [6]}, [1], 6),
    17: ({1: [2, 3, 5], 3: [4], 5: [6]}, [1], 2),
    18: ({1: [2, 3, 4], 4: [5], 5: [6]}, [1], 3),
    19: ({1: [2], 2: [3], 4: [6], 5: [6]}, [1, 4, 5], 6),
    20: ({1: [3], 2: [3], 4: [5, 6]}, [1, 2, 4], 3),
    21: ({1: [3], 2: [3], 4: [6], 5: [6]}, [1, 2, 4, 5], 3),
    22: ({1: [2], 3: [6], 4: [6], 5: [6]}, [1, 3, 4, 5], 6),
    23: ({2: [3], 4: [6], 5: [6]}, [1, 2, 4, 5], 6),
    24: ({3: [6], 4: [6], 5: [6]}, [1, 2, 3, 4, 5], 6),
    25: ({4: [6], 5: [6]}, [1, 2, 3, 4, 5], 6),
    26: ({2: [3], 4: [5, 6]}, [1, 2, 4], 6),
    27: ({2: [3], 4: [5, 6]}, [1, 2, 4], 3),
    28: ({1: [2], 3: [4], 4: [5]}, [1, 3, 6], 5),
    29: ({3: [4], 5: [6]}, [1, 2, 3, 5], 6),
    30: ({3: [4], 4: [5]}, [1, 2, 3, 6], 5),
    31: ({5: [6]}, [1, 2, 3, 4, 5], 6),
    32: ({}, [1, 2, 3, 4, 5, 6], 6),
}
MOD_TURNS = 6.0             # phase deviation (in turns) of a modulator at output level 99 and full envelope
FB_TURNS = {k: 0.0 if k == 0 else 0.06 * 2 ** (k - 1) for k in range(8)}   # feedback level 0..7 -> turns at full level


def level_amp(level):
    """DX7 output level 0..99 -> linear amplitude (0.75 dB per step, 99 = 1.0, 0 = silence)."""
    level = np.asarray(level, np.float64)
    return np.where(level <= 0, 0.0, 10 ** (-0.75 * (99 - level) / 20))


def rate_seconds(rate):
    """Seconds for a full 0..99 swing at DX7 rate 0..99 (99: ~2 ms, 50: ~0.2 s, 0: ~30 s). Approximate."""
    return 30.0 * 2 ** (-np.asarray(rate, np.float64) / 7.0)


def op(coarse=1, fine=0, detune=0, level=99, rates=(99, 50, 30, 50), levels=(99, 80, 0, 0), fixed=False, vel=0.0):
    return dict(coarse=coarse, fine=fine, detune=detune, level=level, rates=rates, levels=levels, fixed=fixed, vel=vel)


PATCHES = {
    # ops listed 1..6; made up, not Yamaha's
    "epiano": dict(algo=5, feedback=5, ops=[
        op(1, 0, 0, 99, (95, 25, 20, 60), (99, 85, 0, 0), vel=0.4),      # 1 carrier: body
        op(1, 0, 0, 80, (95, 30, 20, 60), (99, 80, 0, 0), vel=0.9),      # 2 -> 1: tine edge, velocity
        op(1, 0, 3, 90, (90, 20, 20, 60), (99, 90, 0, 0), vel=0.3),      # 3 carrier
        op(14, 0, 0, 74, (99, 60, 60, 70), (99, 0, 0, 0), vel=1.0),      # 4 -> 3: bell-like attack transient
        op(1, 0, -3, 80, (95, 22, 20, 60), (99, 80, 0, 0), vel=0.3),     # 5 carrier, detuned
        op(1, 0, 0, 76, (95, 22, 20, 60), (99, 85, 0, 0), vel=0.6),      # 6 -> 5
    ]),
    "bass": dict(algo=2, feedback=6, ops=[
        op(1, 0, 0, 99, (99, 40, 20, 60), (99, 90, 0, 0), vel=0.5),
        op(1, 0, 0, 90, (99, 45, 40, 70), (99, 85, 0, 0), vel=0.9),      # 2 -> 1 with feedback: growl
        op(1, 0, 0, 90, (99, 35, 20, 60), (99, 90, 0, 0), vel=0.5),
        op(2, 0, 0, 78, (99, 60, 40, 70), (99, 60, 0, 0), vel=0.8),
        op(1, 0, 0, 0, (99, 50, 20, 50), (99, 0, 0, 0)),
        op(1, 0, 0, 0, (99, 50, 20, 50), (99, 0, 0, 0)),
    ]),
    "bells": dict(algo=5, feedback=0, ops=[
        op(1, 0, 0, 99, (99, 20, 15, 40), (99, 90, 0, 0)),
        op(3, 50, 0, 80, (99, 25, 15, 40), (99, 85, 0, 0)),              # 2 -> 1: 3.5 ratio, inharmonic
        op(2, 0, 4, 85, (99, 18, 15, 40), (99, 90, 0, 0)),
        op(7, 0, 0, 72, (99, 30, 15, 40), (99, 80, 0, 0)),
        op(5, 0, -4, 70, (99, 15, 15, 40), (99, 90, 0, 0)),
        op(11, 0, 0, 66, (99, 35, 20, 40), (99, 70, 0, 0)),
    ]),
    "brass": dict(algo=18, feedback=6, ops=[
        op(1, 0, 0, 99, (70, 50, 40, 55), (99, 85, 85, 0), vel=0.4),
        op(1, 0, 2, 86, (65, 50, 40, 55), (99, 88, 88, 0), vel=0.8),
        op(1, 0, -2, 84, (68, 50, 40, 55), (99, 88, 88, 0), vel=0.8),    # 3 -> 1, feedback: buzz
        op(1, 0, 0, 80, (72, 50, 40, 55), (99, 85, 85, 0), vel=0.6),
        op(2, 0, 0, 72, (72, 50, 40, 55), (99, 80, 80, 0), vel=0.6),
        op(1, 0, 0, 0, (99, 50, 40, 50), (99, 0, 0, 0)),
    ]),
    "pad": dict(algo=32, feedback=6, ops=[
        op(1, 0, 0, 90, (40, 30, 30, 35), (99, 99, 99, 0)),
        op(2, 0, 3, 80, (38, 30, 30, 35), (99, 99, 99, 0)),
        op(3, 0, -3, 70, (36, 30, 30, 35), (99, 99, 99, 0)),
        op(1, 0, 5, 85, (42, 30, 30, 35), (99, 99, 99, 0)),
        op(2, 0, -5, 75, (40, 30, 30, 35), (99, 99, 99, 0)),
        op(4, 0, 0, 55, (34, 30, 30, 35), (99, 99, 99, 0)),
    ]),
}


def patch_table(patch):
    """The per-core algorithm table: per op (nsrc, s0, s1, s2, K0, K1, K2, fbK | carrier << 16). K scales a
    modulator's Q15 output to a Q32 phase offset: out * K with K = turns * 2^32 / 2^15 at the modulator's
    full level; the modulator's own envelope and output level are in its output already."""
    mods, carriers, fbop = ALGOS[patch["algo"]]
    t = np.zeros(PATCH_N, np.int64)
    kfull = int(round(MOD_TURNS * 2 ** 17))
    for o in range(1, OPS + 1):
        srcs = mods.get(o, [])
        row = t[(o - 1) * 8:(o - 1) * 8 + 8]
        row[0] = len(srcs)
        for i, s in enumerate(srcs):
            row[1 + i] = s - 1; row[4 + i] = kfull
        fbk = int(round(FB_TURNS[patch.get("feedback", 0)] * 2 ** 17)) if o == fbop and patch.get("feedback", 0) > 0 else 0
        row[7] = (fbk & 0x7fff) | (0x8000 if o == fbop else 0) | (0x10000 if o in carriers else 0)
    return t


def op_freq(o, note_hz):
    if o["fixed"]:
        return 10.0 ** (o["coarse"] % 4) * (1 + o["fine"] / 100.0)
    ratio = (0.5 if o["coarse"] == 0 else float(o["coarse"])) * (1 + o["fine"] / 100.0)
    return note_hz * ratio + o["detune"] * 0.3


# ----------------------------------------------------------------------------------------------- the synth
class DX7:
    """512 voices. note_on(voice, patch, midi, velocity), note_off(voice); block() renders 256 samples."""

    def __init__(self):
        self.cols = iron.get_current_device().cols
        self.inp = np.zeros((CORES, IN_N), np.int32)
        self._in = iron.zeros((CORES * IN_N,), dtype=np.int32, device="npu")
        self._out = iron.zeros((CORES * OUT_N,), dtype=np.int32, device="npu")
        tab = np.round(32767 * np.sin(2 * np.pi * np.arange(TAB) / TAB)).astype(np.int16)
        self.table = tab
        self.inp[:, -TAB // 2:] = tab.view(np.int32)[None, :]
        self.core_patch = [None] * CORES
        # envelope state per voice per op: segment (0..3 attack/decay1/decay2, 4 release, 5 idle), level in dB (0 = full)
        self.seg = np.full((NVOICES, OPS), 5, np.int32)
        self.db = np.full((NVOICES, OPS), -120.0)
        self.target = np.zeros((NVOICES, OPS)); self.step = np.zeros((NVOICES, OPS))
        self.rates = np.zeros((NVOICES, OPS, 4)); self.levels = np.zeros((NVOICES, OPS, 4))
        self.scale = np.zeros((NVOICES, OPS))             # output level x velocity, linear
        self.lvl_prev = np.zeros((NVOICES, OPS))           # amplitude at the end of the last block
        self.age = np.zeros(NVOICES, np.int64); self.t = 0
        self.ms = 0.0

    def set_patch(self, core, name):
        self.core_patch[core] = name
        self.inp[core, 4 * OPS * L + 2 * L:4 * OPS * L + 2 * L + PATCH_N] = patch_table(PATCHES[name])

    def alloc(self, name):
        """A free voice on a core that holds `name` (a free core gets the patch); else the oldest voice."""
        for c in range(CORES):
            if self.core_patch[c] == name or self.core_patch[c] is None:
                if self.core_patch[c] is None:
                    self.set_patch(c, name)
                for v in range(c * L, (c + 1) * L):
                    if self.seg[v, 0] == 5:
                        return v
        cands = [c for c in range(CORES) if self.core_patch[c] == name]
        vs = np.concatenate([np.arange(c * L, (c + 1) * L) for c in cands])
        return int(vs[np.argmin(self.age[vs])])

    def note_on(self, name, midi, velocity=1.0):
        v = self.alloc(name); c, lane = divmod(v, L)
        p = PATCHES[name]; hz = 440.0 * 2 ** ((midi - 69) / 12)
        for o, od in enumerate(p["ops"]):
            f = op_freq(od, hz)
            self.inp[c, OPS * L + o * L + lane] = np.int64(round(f / FS * 2 ** 32)) & 0xffffffff   # dph
            self.inp[c, o * L + lane] = 0                                                         # phase
            self.rates[v, o] = od["rates"]; self.levels[v, o] = od["levels"]
            self.scale[v, o] = level_amp(od["level"]) * (1 - od["vel"] * (1 - velocity))
            self.seg[v, o] = 0; self.db[v, o] = -120.0
            self._aim(v, o)
        self.inp[c, 4 * OPS * L + lane] = 0; self.inp[c, 4 * OPS * L + L + lane] = 0               # feedback state
        self.age[v] = self.t
        return v

    def note_off(self, v):
        for o in range(OPS):
            if self.seg[v, o] < 4:
                self.seg[v, o] = 4; self._aim(v, o)

    def _aim(self, v, o):
        s = self.seg[v, o]
        if s >= 5:
            return
        lv = self.levels[v, o, min(s, 3)]; rate = self.rates[v, o, min(s, 3)]
        self.target[v, o] = -120.0 if lv <= 0 else -0.75 * (99 - lv)
        # dB per block: the full swing (99 levels = 74 dB) takes rate_seconds; attacks run faster
        per_block = 74.25 / rate_seconds(rate) * (B / FS)
        self.step[v, o] = per_block * (3.0 if self.target[v, o] > self.db[v, o] else 1.0)

    def _envelopes(self):
        """Advance every voice's operator envelopes by one block; returns amplitudes at the block end."""
        active = self.seg < 5
        d = self.target - self.db
        move = np.clip(d, -self.step, self.step)
        self.db = np.where(active, self.db + move, -120.0)
        arrived = active & (np.abs(self.target - self.db) < 1e-9)
        # next segment: 0->1->2 (hold at 2 until note off), 4 -> 5 (idle)
        nxt = np.where(self.seg == 4, 5, np.where(self.seg < 2, self.seg + 1, self.seg))
        changed = arrived & (nxt != self.seg)
        self.seg = np.where(changed, nxt, self.seg)
        for v, o in zip(*np.nonzero(changed)):
            self._aim(v, o)
        amp = np.where(self.seg < 5, 10 ** (self.db / 20), 0.0) * self.scale
        amp[self.db <= -119.0] = 0.0
        return amp

    def block(self):
        """Render 256 samples of all 512 voices; returns float mono in [-1, 1] (sum of 512 voices / 64)."""
        amp = self._envelopes()
        l0 = np.round(self.lvl_prev * 32767).astype(np.int64); l1 = np.round(amp * 32767).astype(np.int64)
        self.lvl_prev = amp
        lv = (l0 << 16).astype(np.int64); dl = ((l1 - l0) << 16) // B
        x = self.inp
        x[:, 2 * OPS * L:3 * OPS * L] = lv.reshape(CORES, L, OPS).transpose(0, 2, 1).reshape(CORES, OPS * L)
        x[:, 3 * OPS * L:4 * OPS * L] = dl.reshape(CORES, L, OPS).transpose(0, 2, 1).reshape(CORES, OPS * L)
        self.last_in = x.copy()
        with self._in.overwrite() as a:
            a[...] = x.ravel()
        t0 = time.perf_counter()
        dx7_npu(self._in, self._out, cols=self.cols, variant="dx7a")
        self.ms = (time.perf_counter() - t0) * 1e3
        o = self._out.numpy().reshape(CORES, OUT_N)
        x[:, :OPS * L] = o[:, :OPS * L]                                  # phases
        x[:, 4 * OPS * L:4 * OPS * L + 2 * L] = o[:, OPS * L:OPS * L + 2 * L]
        self.t += 1
        return o[:, OPS * L + 2 * L:].astype(np.int64).sum(0) / (32768.0 * 64)


def replica(inp, table):
    """numpy int replica of dx7_block for one core input object; returns (phases, fb1, fb2, mix)."""
    x = inp.astype(np.int64)
    ph = x[:OPS * L].reshape(OPS, L).copy(); dph = x[OPS * L:2 * OPS * L].reshape(OPS, L)
    lvl = x[2 * OPS * L:3 * OPS * L].reshape(OPS, L).copy(); dlv = x[3 * OPS * L:4 * OPS * L].reshape(OPS, L)
    fb1 = x[4 * OPS * L:4 * OPS * L + L].copy(); fb2 = x[4 * OPS * L + L:4 * OPS * L + 2 * L].copy()
    patch = x[4 * OPS * L + 2 * L:4 * OPS * L + 2 * L + PATCH_N].reshape(OPS, 8)
    tab = table.astype(np.int64)
    fbop = [o for o in range(OPS) if patch[o, 7] & 0x8000]; fbop = fbop[0] if fbop else -1
    def wrap32(a):
        return ((a + 2 ** 31) % 2 ** 32) - 2 ** 31
    def srs15(a):
        q, r = np.divmod(a, 1 << 15); half = 1 << 14
        q = np.where(r > half, q + 1, np.where(r == half, q + (q & 1), q))
        return np.clip(q, -32768, 32767)
    O = np.zeros((OPS, L), np.int64); mix = np.zeros(B, np.int64)
    for n in range(B):
        acc = np.zeros(L, np.int64)
        for o in range(OPS - 1, -1, -1):
            ph[o] = wrap32(ph[o] + dph[o]); m = ph[o].copy()
            for s in range(patch[o, 0]):
                m = wrap32(m + O[patch[o, 1 + s]] * patch[o, 4 + s])
            if o == fbop:
                avg = (fb1 + fb2) >> 1
                m = wrap32(m + avg * (patch[o, 7] & 0x7fff))
            idx = ((m % 2 ** 32) >> 20) & (TAB - 1)
            sn = tab[idx]
            lv = lvl[o] >> 16
            O[o] = srs15(sn * lv)
            lvl[o] = wrap32(lvl[o] + dlv[o])
            if patch[o, 7] & 0x10000:
                acc += O[o]
        if fbop >= 0:
            fb2 = fb1.copy(); fb1 = O[fbop].copy()
        mix[n] = acc.sum()
    return ph, fb1, fb2, mix


def selftest(syn):
    rng = np.random.default_rng(2)
    names = list(PATCHES)
    for i in range(40):
        syn.note_on(names[i % len(names)], int(rng.integers(36, 84)), float(rng.uniform(0.3, 1.0)))
    for _ in range(3):
        syn.block()
    syn.block()
    inp = syn.last_in
    o = syn._out.numpy().reshape(CORES, OUT_N)
    ok = True
    for c in range(CORES):
        if syn.core_patch[c] is None:
            continue
        ph, fb1, fb2, mix = replica(inp[c], syn.table)
        got = o[c]
        same = (np.array_equal(got[:OPS * L], ph.ravel().astype(np.int32)) and np.array_equal(got[OPS * L:OPS * L + L], fb1)
                and np.array_equal(got[OPS * L + 2 * L:], mix))
        if not same:
            d = np.abs(got[OPS * L + 2 * L:] - mix)
            print(f"  core {c} ({syn.core_patch[c]}): mix max diff {d.max()} at {np.argmax(d)}, phases equal {np.array_equal(got[:OPS * L], ph.ravel().astype(np.int32))}")
            ok = False
    print(f"selftest: NPU vs numpy replica on {sum(p is not None for p in syn.core_patch)} cores: {'bit-exact' if ok else 'MISMATCH'}; "
          f"{syn.ms:.2f} ms per {B}-sample block of {NVOICES} voices ({NVOICES * OPS * B / (syn.ms * 1e-3) / 1e6:.0f} M operator-samples/s)")
    return ok


def demo(syn, seconds=24):
    """A little tune: e-piano chords, bass, bells on top, a brass stab, then the pad."""
    import soundfile as sf
    nblocks = int(seconds * FS / B)
    out = np.zeros(nblocks * B)
    chords = [[60, 64, 67, 71], [57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 65]]
    bass = [36, 33, 29, 31]
    bell_seq = [84, 88, 91, 95, 91, 88]
    events = []                                      # (time s, kind, name, midi, vel, duration)
    for bar in range(8):
        t = bar * 2.0
        for n in chords[bar % 4]:
            events.append((t, "epiano", n, 0.8, 1.6))
        for k in range(4):
            events.append((t + k * 0.5, "bass", bass[bar % 4] + (12 if k == 3 else 0), 0.9, 0.4))
        if bar >= 2:
            for k, n in enumerate(bell_seq):
                events.append((t + k * 0.33, "bells", n, 0.5, 1.2))
        if bar in (3, 7):
            for n in chords[bar % 4]:
                events.append((t + 1.0, "brass", n - 12, 1.0, 0.8))
    for n in [48, 55, 60, 64]:
        events.append((16.0, "pad", n, 0.7, 7.0))
    events.sort()
    ons = {}; pending = sorted(events)
    offs = []
    t0 = time.perf_counter(); npu_ms = 0.0
    for b in range(nblocks):
        t = b * B / FS
        while pending and pending[0][0] <= t:
            tt, name, midi, vel, dur = pending.pop(0)
            v = syn.note_on(name, midi, vel); offs.append((tt + dur, v))
        offs.sort()
        while offs and offs[0][0] <= t:
            syn.note_off(offs.pop(0)[1])
        out[b * B:(b + 1) * B] = syn.block(); npu_ms += syn.ms
    out /= max(1e-9, np.abs(out).max()) / 0.9
    sf.write("dx7_demo.wav", out.astype(np.float32), FS)
    print(f"wrote dx7_demo.wav: {seconds} s in {time.perf_counter() - t0:.1f} s wall, NPU {npu_ms / nblocks:.2f} ms per block "
          f"({seconds / (npu_ms / 1e3):.0f}x real time in the NPU calls)")


def play(syn, patch="epiano"):
    import pygame, sounddevice as sd, queue, threading
    keys_lo = "zsxdcvgbhnjm,l."; keys_hi = "q2w3er5t6y7ui9o0p"
    names = list(PATCHES); cur = [names.index(patch) if patch in PATCHES else 0]; transpose = [0]
    q = queue.Queue(maxsize=4); held = {}
    lock = threading.Lock()

    def producer():
        while True:
            with lock:
                blk = syn.block()
            q.put(np.clip(blk * 4, -1, 1).astype(np.float32))

    threading.Thread(target=producer, daemon=True).start()
    last = [np.zeros(B, np.float32)]

    def cb(outdata, frames, t, status):
        try:
            last[0] = q.get_nowait()
        except queue.Empty:
            pass
        outdata[:, 0] = last[0]

    pygame.init(); screen = pygame.display.set_mode((800, 300)); font = pygame.font.SysFont(None, 28)
    stream = sd.OutputStream(samplerate=FS, blocksize=B, channels=1, dtype="float32", callback=cb); stream.start()
    clock = pygame.time.Clock(); running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                ch = e.unicode
                if ch in "12345":
                    cur[0] = int(ch) - 1
                elif e.key == pygame.K_UP:
                    transpose[0] += 12
                elif e.key == pygame.K_DOWN:
                    transpose[0] -= 12
                elif ch in keys_lo or ch in keys_hi:
                    midi = (48 + keys_lo.index(ch)) if ch in keys_lo else (60 + keys_hi.index(ch))
                    midi += transpose[0]
                    with lock:
                        held[e.key] = syn.note_on(names[cur[0]], midi, 0.9)
            elif e.type == pygame.KEYUP and e.key in held:
                with lock:
                    syn.note_off(held.pop(e.key))
        screen.fill((20, 20, 30))
        screen.blit(font.render(f"DX7 on the NPU: patch {cur[0] + 1} {names[cur[0]]}   transpose {transpose[0]:+d}   NPU {syn.ms:.2f} ms/block   1-5 patch, up/down octave, ESC", True, (220, 220, 220)), (10, 10))
        s = last[0]
        pts = [(10 + i * 780 / B, 170 - s[i] * 110) for i in range(B)]
        pygame.draw.lines(screen, (90, 220, 120), False, pts, 2)
        pygame.display.flip(); clock.tick(60)
    stream.stop(); pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--patch", default="epiano")
    ap.add_argument("--seconds", type=float, default=24)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    syn = DX7()
    syn.block()                                        # compile / load
    if a.window:
        play(syn, a.patch); return
    ok = selftest(syn)
    syn2 = DX7(); syn2.block()
    if a.out:
        import soundfile as sf
        v = syn2.note_on(a.patch, 60, 0.9); blocks = []
        for b in range(int(a.seconds * FS / B)):
            if b == int(a.seconds * 0.6 * FS / B):
                syn2.note_off(v)
            blocks.append(syn2.block())
        x = np.concatenate(blocks); x /= max(1e-9, np.abs(x).max()) / 0.9
        sf.write(a.out, x.astype(np.float32), FS); print("wrote", a.out)
    else:
        demo(syn2, a.seconds)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
