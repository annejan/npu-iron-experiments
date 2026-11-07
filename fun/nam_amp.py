#!/usr/bin/env python3
"""Neural Amp Modeler (WaveNet) guitar amps on the NPU.

A NAM 'standard' capture is a WaveNet: two layer arrays (16 then 8 channels) of ten dilated causal convolutions
each (kernel 3, dilations 1..512), tanh, a 1x1 back into the residual stream, the activations summed into a head
stream; 13320 multiply-adds per sample. Here one amp is a pipeline of cores (four per column): time is in the
lanes (64 samples per block), every weight is a scalar broadcast into a bf16 multiply-accumulate, the residual and
head streams stay fp32. The net amplifies its operands by ~25 dB per layer, so plain bf16 operands (8-bit mantissa)
leave 19 dB of SNR; every operand is therefore split into a hi + lo bf16 pair (two macs, fp32-exact), and the 2*d
sample history of every layer lives in core memory as such pairs. Weights are single bf16 (a fixed perturbation of
the model, about 46 dB, --bf16-weights) or pairs too (default, three macs, about 66 dB: the limit of the 256-segment
piecewise-linear tanh that replaces the hardware vtanh, which is the identity for |x| <= 0.5). Layers too big for one core are split
over cores by input channel with a partial accumulator as payload. Two amps run at once on the 32 cores.

    ./env.sh python fun/nam_amp.py --selftest                      # vs numpy fp32 reference and a bf16 emulation
    ./env.sh python fun/nam_amp.py --stages                        # every core's output against the fp32 trace
    ./env.sh python fun/nam_amp.py --wav di.wav --model m.nam [--ir cab.wav] --out out.wav
    ./env.sh python fun/nam_amp.py --live --model m.nam [--ir cab.wav]   # mic/line in -> amp -> speakers
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

T = 64                 # samples per block (one 64-lane bf16 vector per channel)
FS = 48000
NSEG = 256             # tanh segments over [-4, 4)
ROWS = 4
CTL = 32               # inp[0] reset flag, inp[1] generation counter
XOFF = CTL             # raw input block (first stage only)
TABOFF = CTL + T       # NSEG float offsets, then NSEG bf16 slopes packed in NSEG/2 floats
WOFF = TABOFF + NSEG + NSEG // 2
OUT_N = T
DEBUG_HEAD = False     # debug channel of every core: x[0] (False) or head[0] (True) after its items
BUDGET = 57000         # bytes of core data memory a stage may plan for (64 KB minus stack and bookkeeping)
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.expanduser("~/models/nam")


# ----------------------------------------------------------------------------------------------- bf16 helpers
def bf16_bits(a):
    a = np.ascontiguousarray(np.asarray(a, np.float32))
    u = a.view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def bf16(a):
    """round float32 to the nearest-even bfloat16, returned as float32"""
    return (bf16_bits(a).astype(np.uint32) << 16).view(np.float32).reshape(np.shape(a))


def limbs(a):
    hi = bf16(a); return hi, bf16(np.asarray(a, np.float32) - hi)


def pack_bf16(bits):
    bits = np.asarray(bits, np.uint16)
    if len(bits) % 2: bits = np.concatenate([bits, np.zeros(1, np.uint16)])
    return np.frombuffer(bits.tobytes(), np.float32)


# ----------------------------------------------------------------------------------------------- the model file
class NAM:
    """A .nam WaveNet: weights unpacked per layer array in the order the file stores them."""

    def __init__(self, path):
        d = json.load(open(path))
        assert d["architecture"] == "WaveNet", d["architecture"]
        self.path = path; self.meta = d.get("metadata") or {}; self.sample_rate = d.get("sample_rate") or FS
        cfg = d["config"]; w = np.array(d["weights"], np.float32)
        assert cfg.get("head") is None and cfg.get("condition_dsp") is None
        self.arrays = []; i = 0
        for lc in cfg["layers"]:
            A = {}
            Cin = lc["input_size"]; C = lc["channels"]
            assert lc.get("condition_size", 1) == 1 and not lc.get("gated", False)
            acts = lc["activation"] if isinstance(lc["activation"], list) else [lc["activation"]]
            for a in acts:
                assert (a if isinstance(a, str) else a.get("type", a.get("name"))) == "Tanh", a
            assert lc.get("layer1x1", {}).get("active", True) and not lc.get("head1x1", {}).get("active", False)
            if "head" in lc:
                Hout, Hb = lc["head"]["out_channels"], lc["head"]["bias"]; assert lc["head"]["kernel_size"] == 1
            else:
                Hout, Hb = lc["head_size"], lc["head_bias"]
            ks = lc.get("kernel_size", lc.get("kernel_sizes", 3))
            assert ks == 3 or ks == [3] * len(lc["dilations"])
            A["Cin"], A["C"], A["Hout"], A["Hb"] = Cin, C, Hout, Hb
            A["Wr"] = w[i:i + C * Cin].reshape(C, Cin); i += C * Cin
            A["layers"] = []
            for D in lc["dilations"]:
                L = {"D": D}
                L["W"] = w[i:i + C * C * 3].reshape(C, C, 3); i += C * C * 3      # [out][in][tap]; tap 2 = now
                L["b"] = w[i:i + C]; i += C
                L["mix"] = w[i:i + C]; i += C
                L["W1"] = w[i:i + C * C].reshape(C, C); i += C * C                 # [out][in]
                L["b1"] = w[i:i + C]; i += C
                A["layers"].append(L)
            A["Wh"] = w[i:i + Hout * C].reshape(Hout, C); i += Hout * C
            A["bh"] = w[i:i + Hout] if Hb else np.zeros(Hout, np.float32)
            if Hb: i += Hout
            self.arrays.append(A)
        self.head_scale = np.float32(cfg["head_scale"])
        assert i == len(w) - 1 and abs(w[-1] - cfg["head_scale"]) < 1e-5, (i, len(w))
        self.receptive = 1 + sum(2 * sum(L["D"] for L in A["layers"]) for A in self.arrays)

    def describe(self):
        return " | ".join(f"{A['Cin']}->{A['C']}ch x{len(A['layers'])} head{A['Hout']}" for A in self.arrays)

    def macs_per_sample(self):
        return sum(A["Wr"].size + A["Wh"].size + sum(L["W"].size + L["W1"].size + L["mix"].size for L in A["layers"]) for A in self.arrays)

    def signature(self):
        return tuple((A["Cin"], A["C"], A["Hout"], tuple(L["D"] for L in A["layers"])) for A in self.arrays)


class RefFP32:
    """float32 reference, block-based with explicit history; block(x, trace) appends (array, layer, x, head)."""

    def __init__(self, m):
        self.m = m; self.reset()

    def reset(self):
        self.hist = [[np.zeros((2 * L["D"], A["C"]), np.float32) for L in A["layers"]] for A in self.m.arrays]

    def block(self, x, trace=None):
        x = np.asarray(x, np.float32); c = x; r = x[:, None]; h = None; Tn = len(x)
        for ai, A in enumerate(self.m.arrays):
            r = r @ A["Wr"].T
            for li, L in enumerate(A["layers"]):
                D = L["D"]; xp = np.concatenate([self.hist[ai][li], r], 0); self.hist[ai][li] = xp[-2 * D:].copy()
                z = xp[0:Tn] @ L["W"][:, :, 0].T + xp[D:Tn + D] @ L["W"][:, :, 1].T + xp[2 * D:Tn + 2 * D] @ L["W"][:, :, 2].T
                z += L["b"] + c[:, None] * L["mix"]
                a = np.tanh(z)
                h = a if h is None else h + a
                r = r + a @ L["W1"].T + L["b1"]
                if trace is not None: trace.append((ai, li, r.copy(), h.copy()))
            h = h @ A["Wh"].T + A["bh"]
        return self.m.head_scale * h[:, 0]

    def process(self, x, block=T):
        self.reset(); x = np.asarray(x, np.float32); n = len(x) - len(x) % block; y = np.zeros(n, np.float32)
        for s in range(0, n, block): y[s:s + block] = self.block(x[s:s + block])
        return y


# ----------------------------------------------------------------------------------------------- tanh table
def tanh_table():
    """segment j covers x in [(j-128-0.5)/32, (j-128+0.5)/32); the line through (xc, tanh xc) with the chord
    slope, slope rounded to bf16 first so the fp32 offset absorbs that rounding"""
    j = np.arange(NSEG); xc = (j - NSEG // 2) / 32.0; h = 1 / 64
    slope = bf16((np.tanh(xc + h) - np.tanh(xc - h)) / (2 * h))
    off = (np.tanh(xc) - slope.astype(np.float64) * xc).astype(np.float32)
    return off, slope


_TAB = None


def pwl_tanh(z):
    """the kernel's tanh: hi/lo bf16 input limbs, clamp, nearest segment, offset + slope*(hi + lo) in fp32"""
    global _TAB
    if _TAB is None: _TAB = tanh_table()
    off, slope = _TAB
    hi, lo = limbs(z); xc = np.clip(hi, -4.0, 3.96875).astype(np.float32)
    idx = np.rint(xc * 32).astype(np.int64) + NSEG // 2
    return (off[idx] + slope[idx] * xc + slope[idx] * lo).astype(np.float32)


# ----------------------------------------------------------------------------------------------- the plan
def roundup(n, k): return -(-n // k) * k


def hist_len(D): return 2 * D + T if D >= T else roundup(2 * D + T, 32)


def weight_layout(it, C_in, C_out, wl):
    """(name, count) of bf16 values a stage item reads, in order; 'lo' twins only with --hifi weight pairs"""
    seg = []
    if it["k"] == "rech_in":
        seg += [("Wr", C_out)] + ([("Wr_lo", C_out)] if wl else [])
    elif it["k"] == "layer":
        C, NI = it["C"], it["i1"] - it["i0"]
        if it["begin"]: seg += [("b_hi", C), ("b_lo", C), ("mix", C)] + ([("mix_lo", C)] if wl else [])
        seg += [("W", C * 3 * NI)] + ([("W_lo", C * 3 * NI)] if wl else [])
        if it["end"]: seg += [("W1", C * C)] + ([("W1_lo", C * C)] if wl else []) + [("b1_hi", C), ("b1_lo", C)]
    elif it["k"] == "trans":
        seg += [("Wh", C_out * C_in)] + ([("Wh_lo", C_out * C_in)] if wl else []) + [("bh_hi", C_out), ("bh_lo", C_out), ("Wr2", C_out * C_in)] + ([("Wr2_lo", C_out * C_in)] if wl else [])
    elif it["k"] == "final":
        seg += [("Wh2", C_in)] + ([("Wh2_lo", C_in)] if wl else []) + [("bh2_hi", 1), ("bh2_lo", 1), ("s_hi", 1), ("s_lo", 1)]
    return seg


def stage_weights(m, st, wl):
    """bf16 bits in the order the stage kernel reads them"""
    out = []
    for it in st["items"]:
        vals = {}
        if it["k"] == "rech_in":
            vals["Wr"], vals["Wr_lo"] = limbs(m.arrays[0]["Wr"][:, 0])
        elif it["k"] == "layer":
            A = m.arrays[it["a"]]; L = A["layers"][[l["D"] for l in A["layers"]].index(it["D"])]
            vals["b_hi"], vals["b_lo"] = limbs(L["b"]); vals["mix"], vals["mix_lo"] = limbs(L["mix"])
            vals["W"], vals["W_lo"] = limbs(np.ascontiguousarray(L["W"][:, it["i0"]:it["i1"], :].transpose(0, 2, 1)).ravel())   # [o][k][i]
            vals["W1"], vals["W1_lo"] = limbs(L["W1"].ravel()); vals["b1_hi"], vals["b1_lo"] = limbs(L["b1"])
        elif it["k"] == "trans":
            A0, A1 = m.arrays
            vals["Wh"], vals["Wh_lo"] = limbs(A0["Wh"].ravel()); vals["bh_hi"], vals["bh_lo"] = limbs(A0["bh"]); vals["Wr2"], vals["Wr2_lo"] = limbs(A1["Wr"].ravel())
        elif it["k"] == "final":
            A1 = m.arrays[1]
            vals["Wh2"], vals["Wh2_lo"] = limbs(A1["Wh"][0]); vals["bh2_hi"], vals["bh2_lo"] = limbs(A1["bh"][:1]); vals["s_hi"], vals["s_lo"] = limbs(np.array([m.head_scale], np.float32))
        for name, n in weight_layout(it, st["C_in"], st["C_out"], wl):
            v = np.asarray(vals[name], np.float32).ravel(); assert len(v) == n, (name, len(v), n)
            out.append(v)
    return bf16_bits(np.concatenate(out)) if out else np.zeros(0, np.uint16)


def stage_bytes(st):
    """core data memory a stage needs: histories (hi+lo bf16), tables, limb buffers, fifo objects"""
    layers = [it for it in st["items"] if it["k"] == "layer"]
    b = sum(4 * (it["i1"] - it["i0"]) * hist_len(it["D"]) for it in layers)
    Cmax = max([st["C_in"], st["C_out"]] + [it["C"] for it in layers])
    b += 4 * Cmax * T + 4 * T                                   # a hi/lo, c hi/lo
    if any(it["end"] for it in layers): b += 2 * 2 * NSEG * 4 + 2 * 2 * NSEG * 2   # offsets (float) and slopes (bf16), two tables of two runs
    if any(it["k"] == "trans" for it in st["items"]): b += 4 * st["C_in"] * T
    if st.get("first"): b += 8 * st["C_out"] * T
    if st.get("scratch"): b += 4 * Cmax * T
    # fifo objects: the host input, the outgoing payload (adjacent fifos live on the producer tile), the debug
    # output; the incoming payload only at the top of a column, where it arrives by DMA into a local copy
    b += 4 * st["in_n"] + 4 * st["pay_out_n"] + 4 * OUT_N + (4 * st["pay_in_n"] if st.get("si", 0) % ROWS == 0 else 0)
    b += 5120   # stack, descriptors, slack
    return b


def finish_stage(m, st, C_in, wl):
    st["C_in"] = C_in
    C = C_in
    for it in st["items"]:
        if it["k"] == "rech_in": C = m.arrays[0]["C"]
        if it["k"] == "trans": C = m.arrays[1]["C"]
    st["C_out"] = C
    its = st["items"]
    if its:
        st["part_in"] = its[0]["k"] == "layer" and not its[0]["begin"]      # first piece continues a layer from the previous core
        st["part_out"] = its[-1]["k"] == "layer" and not its[-1]["end"]     # last piece hands a partial sum on
    else:
        st["part_in"] = st["part_out"] = st.get("carry", False)             # pass-through: forward whatever arrives
    st["scratch"] = any(a["k"] == "layer" and b["k"] == "layer" and a["a"] == b["a"] and a["D"] == b["D"] for a, b in zip(its, its[1:]))
    st["pay_in_n"] = 0 if st.get("first") else (2 * C_in + 1) * T + (C_in * T if st["part_in"] else 0)
    st["pay_out_n"] = 0 if st.get("last") else (2 * C + 1) * T + (C * T if st["part_out"] else 0)
    st["nw"] = int(len(stage_weights(m, st, wl)))
    st["in_n"] = roundup(WOFF + (st["nw"] + 1) // 2, 32)
    return st


def make_plan(m, wl, budget=BUDGET, max_stages=8 * ROWS):
    """Pack rech_in, the layers (split by input channel when one does not fit a core), trans and final into
    stages of at most `budget` bytes, greedily in order; pad to a multiple of four with pass-through stages."""
    sig = m.signature()
    assert len(sig) == 2 and sig[0][0] == 1 and sig[1][0] == sig[0][1] and sig[1][2] == 1, f"two layer arrays, 1 -> C0 -> C1 -> 1 expected; got {sig}"
    items = [dict(k="rech_in")]
    for ai, A in enumerate(m.arrays):
        if ai == 1: items.append(dict(k="trans"))
        C = A["C"]
        for L in A["layers"]:
            D = L["D"]; pieces = 1
            while pieces < C:
                probes = [dict(k="layer", a=ai, D=D, C=C, i0=0, i1=C // pieces, begin=pieces == 1 or b, end=pieces == 1 or not b) for b in (True, False)]
                if all(stage_bytes(finish_stage(m, dict(items=[pr], first=False, last=False), C, wl)) <= budget for pr in probes): break
                pieces *= 2
            n = C // pieces
            for p in range(pieces):
                items.append(dict(k="layer", a=ai, D=D, C=C, i0=p * n, i1=(p + 1) * n, begin=p == 0, end=p == pieces - 1))
    items.append(dict(k="final"))
    stages = []; cur = []; C_in = m.arrays[0]["C"]

    def fits(its, first):
        st = finish_stage(m, dict(items=list(its), first=first, last=its[-1]["k"] == "final", si=len(stages)), C_in, wl)
        return stage_bytes(st) <= budget, st
    for it in items:
        ok, st = fits(cur + [it], not stages)
        if ok: cur.append(it); continue
        assert cur, f"item does not fit a core alone: {it}"
        stages.append(fits(cur, not stages)[1]); C_in = stages[-1]["C_out"]; cur = [it]
        assert fits(cur, False)[0], f"item does not fit a core alone: {it}"
    stages.append(fits(cur, not stages)[1])
    # pad with pass-through stages so a column holds four: at a boundary with no partial sum in flight, chosen
    # from the end, such that every shifted stage still fits its new row
    def refinish(sts):
        for si, st in enumerate(sts):
            st["first"], st["last"], st["si"] = si == 0, si == len(sts) - 1, si
            finish_stage(m, st, st["C_in"], wl)
        return all(stage_bytes(st) <= budget for st in sts)
    while len(stages) % ROWS:
        for j in sorted((j for j in range(1, len(stages)) if not stages[j]["part_in"]), reverse=True):
            st = finish_stage(m, dict(items=[], first=False, last=False, carry=stages[j - 1]["part_out"]), stages[j]["C_in"], wl)
            trial = stages[:j] + [st] + stages[j:]
            if refinish(trial): stages = trial; break
        else:
            raise AssertionError("no place for a pass-through stage")
    assert refinish(stages) and len(stages) <= max_stages, len(stages)
    return stages


def plan_text(stages):
    lines = []
    for si, st in enumerate(stages):
        d = []
        for it in st["items"]:
            if it["k"] == "layer": d.append(f"a{it['a']}d{it['D']}" + ("" if it["i1"] - it["i0"] == it["C"] else f"[{it['i0']}:{it['i1']}]"))
            else: d.append(it["k"])
        lines.append(f"  core {si} (col {si // ROWS} row {si % ROWS}): {' '.join(d) or 'pass'}; {stage_bytes(st) / 1024:.1f} KB")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- the kernels
TANH_SRC = """
using lutf = aie::lut<4, float, float>;
using lutb = aie::lut<4, bf, bf>;
static inline A tanh_pwl(const A &acc, const lutf &lo, const lutb &ls) {
    aie::parallel_lookup<int32, lutf, aie::lut_oor_policy::truncate> look_o(lo, 0);
    aie::parallel_lookup<int32, lutb, aie::lut_oor_policy::truncate> look_s(ls, 0);
    V zh = acc.to_vector<bf>(); V zl = aie::sub(acc, A(zh)).to_vector<bf>();
    V xc = aie::max(aie::min(zh, aie::broadcast<bf, T>((bf)3.96875f)), aie::broadcast<bf, T>((bf)-4.0f));
    A r;
    for (int q = 0; q < 4; q++) {
        V16 xq = xc.extract<16>(q); V16 lq = zl.extract<16>(q);
        aie::vector<int32, 16> idx = aie::add(aie::to_fixed<int32>(xq, 5), (int32)(NSEG / 2));
        aie::accum<accfloat, 16> t(look_o.fetch(idx)); V16 sl = look_s.fetch(idx);
        t = aie::mac(t, sl, xq); t = aie::mac(t, sl, lq);
        r.insert(q, t);
    }
    return r;
}
"""


def kernel_source(si, st, wl):
    """C++ for one core: a generic layer() over a descriptor table (code size does not grow with the layer
    count), plus rech_in / trans / final straight-line code."""
    items = st["items"]; first, last = st["first"], st["last"]
    C_in, C_out = st["C_in"], st["C_out"]
    layers = [it for it in items if it["k"] == "layer"]
    Cmax = max([C_in, C_out] + [it["C"] for it in layers])
    need_tanh = any(it["end"] for it in layers)
    need_trans = any(it["k"] == "trans" for it in items)
    WL = 1 if wl else 0
    L = []; w = L.append
    w(f"""#include <stdint.h>
#include <aie_api/aie.hpp>
#define T {T}
#define NSEG {NSEG}
#define WL {WL}
using bf = bfloat16;
using V = aie::vector<bf, T>;
using V16 = aie::vector<bf, 16>;
using A = aie::accum<accfloat, T>;
using A32 = aie::accum<accfloat, 32>;
static inline A ldf(const float *p) {{ A r; r.insert(0, A32(aie::load_v<32>(p))); r.insert(1, A32(aie::load_v<32>(p + 32))); return r; }}
static inline void stf(float *p, const A &a) {{ aie::store_v(p, a.extract<32>(0).to_vector<float>()); aie::store_v(p + 32, a.extract<32>(1).to_vector<float>()); }}
static inline void split(const A &v, bf *ph, bf *pl) {{ V h = v.to_vector<bf>(); aie::store_v(ph, h); aie::store_v(pl, aie::sub(v, A(h)).to_vector<bf>()); }}
static inline void splitu(const A &v, bf *ph, bf *pl) {{ V h = v.to_vector<bf>(); aie::store_unaligned_v(ph, h); aie::store_unaligned_v(pl, aie::sub(v, A(h)).to_vector<bf>()); }}
static inline void cpf(float *d, const float *s, int n) {{ for (int i = 0; i < n; i += 32) aie::store_v(d + i, aie::load_v<32>(s + i)); }}
// operand hi+lo pair times a weight (single bf16, or hi+lo pair with WL)
static inline void mac2(A &acc, const V &vh, const V &vl, bf whi, bf wlo) {{
    acc = aie::mac(acc, vh, whi); acc = aie::mac(acc, vl, whi);
#if WL
    acc = aie::mac(acc, vh, wlo);
#endif
}}
alignas(64) static bf ch[T], cl[T];
alignas(64) static bf ah[{Cmax}][T], al[{Cmax}][T];
static int gen_last = -1;""")
    if first: w(f"alignas(64) static float xs[{C_out}][T]; alignas(64) static float hs[{C_out}][T];")
    if need_tanh: w("alignas(64) static float tab_ab[2 * NSEG], tab_cd[2 * NSEG]; alignas(64) static bf sl_ab[2 * NSEG], sl_cd[2 * NSEG];")
    if need_trans: w(f"alignas(64) static bf xh[{C_in}][T], xl[{C_in}][T];")
    if st["scratch"]: w(f"alignas(64) static float ps[{Cmax} * T];   // partial sums between pieces of one layer on this core")
    for li, it in enumerate(layers):
        D, NI = it["D"], it["i1"] - it["i0"]
        w(f"alignas(64) static bf hh{li}[{NI * hist_len(D)}], hl{li}[{NI * hist_len(D)}];")
    if layers: w(f"static int pos[{len(layers)}];")
    if need_tanh: w(TANH_SRC)
    else: w("struct lutf {}; struct lutb {}; static inline A tanh_pwl(const A &a, const lutf &, const lutb &) { return a; }   // never called: no layer ends here")
    if layers:
        w("""struct Lay { bf *hh; bf *hl; int NI, i0, C, D, len, ring, begin, end; int Ob_hi, Ob_lo, Omix, Omix_lo, OW, OW_lo, OW1, OW1_lo, Ob1_hi, Ob1_lo; };
template <bool END>
static __attribute__((noinline)) void layer(const Lay &L, int li, const bf *w, float *xw, float *hw, const float *part_in, float *part_out, const lutf &lo, const lutb &ls) {
    const V ones = aie::broadcast<bf, T>((bf)1.0f);
    const int NI = L.NI, C = L.C, D = L.D, len = L.len;
    int o0, o1, o2;
    if (L.ring) { int p = pos[li]; o0 = p - 2 * D; if (o0 < 0) o0 += len; o1 = p - D; if (o1 < 0) o1 += len; o2 = p; }
    else { o0 = 0; o1 = D; o2 = 2 * D; }
    for (int i = 0; i < NI; i++) splitu(ldf(xw + (L.i0 + i) * T), L.hh + i * len + o2, L.hl + i * len + o2);
    for (int o = 0; o < C; o++) {
        A acc;
        if (L.begin) {
            acc = A(aie::broadcast<bf, T>(w[L.Ob_hi + o])); acc = aie::mac(acc, ones, w[L.Ob_lo + o]);
            mac2(acc, aie::load_v<T>(ch), aie::load_v<T>(cl), w[L.Omix + o], w[L.Omix_lo + o]);
        } else {
            acc = ldf(part_in + o * T);
        }
        const bf *wo = w + L.OW + o * 3 * NI; const bf *wlo = w + L.OW_lo + o * 3 * NI;
        for (int i = 0; i < NI; i++) { const bf *h = L.hh + i * len, *l = L.hl + i * len;
            mac2(acc, aie::load_unaligned_v<T>(h + o0), aie::load_unaligned_v<T>(l + o0), wo[i], wlo[i]);
            mac2(acc, aie::load_unaligned_v<T>(h + o1), aie::load_unaligned_v<T>(l + o1), wo[NI + i], wlo[NI + i]);
            mac2(acc, aie::load_unaligned_v<T>(h + o2), aie::load_unaligned_v<T>(l + o2), wo[2 * NI + i], wlo[2 * NI + i]);
        }
        if constexpr (END) split(tanh_pwl(acc, lo, ls), ah[o], al[o]);
        else stf(part_out + o * T, acc);
    }
    if constexpr (END) {
        for (int o = 0; o < C; o++) {
            A xa = ldf(xw + o * T); xa = aie::mac(xa, ones, w[L.Ob1_hi + o]); xa = aie::mac(xa, ones, w[L.Ob1_lo + o]);
            for (int i = 0; i < C; i++) mac2(xa, aie::load_v<T>(ah[i]), aie::load_v<T>(al[i]), w[L.OW1 + o * C + i], w[L.OW1_lo + o * C + i]);
            stf(xw + o * T, xa);
            A ha = ldf(hw + o * T); ha = aie::add(ha, A(aie::load_v<T>(ah[o]))); ha = aie::add(ha, A(aie::load_v<T>(al[o]))); stf(hw + o * T, ha);
        }
    }
    if (L.ring) { int p = pos[li] + T; if (p >= len) p = 0; pos[li] = p; }
    else if (2 * D >= 32) { for (int i = 0; i < NI; i++) for (int t = 0; t < 2 * D; t += 32) { bf *h = L.hh + i * len, *l = L.hl + i * len;
        aie::store_unaligned_v(h + t, aie::load_unaligned_v<32>(h + T + t)); aie::store_unaligned_v(l + t, aie::load_unaligned_v<32>(l + T + t)); } }
    else { for (int i = 0; i < NI; i++) for (int t = 0; t < 2 * D; t++) { L.hh[i * len + t] = L.hh[i * len + T + t]; L.hl[i * len + t] = L.hl[i * len + T + t]; } }
}""")
    # descriptor table with weight offsets
    off = 0; descs = []; oitems = []
    for it in items:
        O = {}; o = off
        for name, n in weight_layout(it, C_in, C_out, wl): O[name] = o; o += n
        off = o; oitems.append((it, O))
        if it["k"] == "layer":
            D, NI = it["D"], it["i1"] - it["i0"]; li = len(descs)
            g = lambda k: O.get(k, 0)
            descs.append(f"{{ hh{li}, hl{li}, {NI}, {it['i0']}, {it['C']}, {D}, {hist_len(D)}, {1 if D >= T else 0}, {1 if it['begin'] else 0}, {1 if it['end'] else 0}, "
                         f"{g('b_hi')}, {g('b_lo')}, {g('mix')}, {g('mix_lo')}, {g('W')}, {g('W_lo')}, {g('W1')}, {g('W1_lo')}, {g('b1_hi')}, {g('b1_lo')} }}")
    if layers: w("static const Lay LAYS[] = {\n  " + ",\n  ".join(descs) + "\n};")
    args = ["float *inp"] + ([] if first else ["float *pin"]) + ([] if last else ["float *pout"]) + ["float *out"]
    w(f'extern "C" void stage{si}({", ".join(args)}) {{')
    w("  aie::set_rounding(aie::rounding_mode::conv_even);")
    w("  int gen = (int)inp[1]; bool reset = inp[0] != 0.0f || gen != gen_last + 1; gen_last = gen;")
    if layers:
        w(f"  if (reset) for (int li = 0; li < {len(layers)}; li++) {{ const Lay &L = LAYS[li]; pos[li] = 0;")
        w("    for (int t = 0; t < L.NI * L.len; t += 32) { aie::store_v(L.hh + t, aie::zeros<bf, 32>()); aie::store_v(L.hl + t, aie::zeros<bf, 32>()); } }")
    if need_tanh:
        w(f"""  {{ const float *to = inp + {TABOFF}; const bf *ts = (const bf *)(inp + {TABOFF + NSEG});
    for (int d = 0; d < NSEG; d++) {{ int i = (d >> 3) * 16 + (d & 7); tab_ab[i] = tab_cd[i] = tab_ab[i + 8] = tab_cd[i + 8] = to[d];
      int j = (d >> 4) * 32 + (d & 15); sl_ab[j] = sl_cd[j] = sl_ab[j + 16] = sl_cd[j + 16] = ts[d]; }} }}
  lutf lo(NSEG, tab_ab, tab_cd); lutb ls(NSEG, sl_ab, sl_cd);""")
    else:
        w("  lutf lo; lutb ls;")
    w(f"  const bf *w = (const bf *)(inp + {WOFF});")
    w("  const V ones = aie::broadcast<bf, T>((bf)1.0f); (void)ones;")
    if first:
        w(f"  float *xw = &xs[0][0]; float *hw = &hs[0][0]; const float *cf = inp + {XOFF};")
        w(f"  for (int t = 0; t < {C_out} * T; t += 32) aie::store_v(hw + t, aie::zeros<float, 32>());")
    else:
        w(f"  float *xw = pin; float *hw = pin + {C_in} * T; const float *cf = pin + {2 * C_in} * T;")
    w(f"  const float *part_in = {('pin + ' + str((2 * C_in + 1) * T)) if (not first and st['part_in']) else 'nullptr'};")
    w(f"  float *part_out = {('pout + ' + str((2 * C_out + 1) * T)) if (not last and st['part_out']) else 'nullptr'};")
    w("  (void)part_in; (void)part_out;")
    w("  split(ldf(cf), ch, cl);")
    li = 0
    for it, O in oitems:
        g = lambda k: O.get(k, 0)
        if it["k"] == "rech_in":
            C = C_out
            w(f"  for (int o = 0; o < {C}; o++) {{ A t(aie::zeros<bf, T>()); mac2(t, aie::load_v<T>(ch), aie::load_v<T>(cl), w[{g('Wr')} + o], w[{g('Wr_lo')} + o]); stf(xw + o * T, t); }}")
        elif it["k"] == "layer":
            j = items.index(it); same = lambda o: o["k"] == "layer" and o["a"] == it["a"] and o["D"] == it["D"]
            pin_ = "ps" if (not it["begin"] and j > 0 and same(items[j - 1])) else "part_in"
            pout_ = "ps" if (not it["end"] and j + 1 < len(items) and same(items[j + 1])) else "part_out"
            w(f"  layer<{'true' if it['end'] else 'false'}>(LAYS[{li}], {li}, w, xw, hw, {pin_}, {pout_}, lo, ls);")
            li += 1
        elif it["k"] == "trans":
            w(f"  for (int c = 0; c < {C_in}; c++) {{ split(ldf(hw + c * T), ah[c], al[c]); split(ldf(xw + c * T), xh[c], xl[c]); }}")
            w(f"  for (int j = 0; j < {C_out}; j++) {{ A h(aie::broadcast<bf, T>(w[{g('bh_hi')} + j])); h = aie::mac(h, ones, w[{g('bh_lo')} + j]); A x(aie::zeros<bf, T>());")
            w(f"    for (int c = 0; c < {C_in}; c++) {{ mac2(h, aie::load_v<T>(ah[c]), aie::load_v<T>(al[c]), w[{g('Wh')} + j * {C_in} + c], w[{g('Wh_lo')} + j * {C_in} + c]);"
              f" mac2(x, aie::load_v<T>(xh[c]), aie::load_v<T>(xl[c]), w[{g('Wr2')} + j * {C_in} + c], w[{g('Wr2_lo')} + j * {C_in} + c]); }}")
            w("    stf(hw + j * T, h); stf(xw + j * T, x); }")
        elif it["k"] == "final":
            C = C_in
            w(f"  {{ A y(aie::broadcast<bf, T>(w[{g('bh2_hi')}])); y = aie::mac(y, ones, w[{g('bh2_lo')}]);")
            w(f"    for (int c = 0; c < {C}; c++) {{ split(ldf(hw + c * T), ah[0], al[0]); mac2(y, aie::load_v<T>(ah[0]), aie::load_v<T>(al[0]), w[{g('Wh2')} + c], w[{g('Wh2_lo')} + c]); }}")
            w(f"    V hi = y.to_vector<bf>(); V lo2 = aie::sub(y, A(hi)).to_vector<bf>(); A r = aie::mul(hi, w[{g('s_hi')}]); r = aie::mac(r, lo2, w[{g('s_hi')}]); r = aie::mac(r, hi, w[{g('s_lo')}]); stf(out, r); }}")
    if not last and not items:
        w(f"  cpf(pout, pin, {st['pay_in_n']});")
    elif not last:
        w(f"  cpf(pout, xw, {C_out} * T); cpf(pout + {C_out} * T, hw, {C_out} * T); cpf(pout + {2 * C_out} * T, cf, T);")
    if not any(it["k"] == "final" for it in items):
        w(f"  cpf(out, {'hw' if DEBUG_HEAD else 'xw'}, T);")
    w("}")
    assert off == st["nw"], (si, off, st["nw"])
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------- the design
@iron.jit
def nam_npu(inp: In, out: Out, *, amps: CompileTime[int], plan: CompileTime[str], variant: CompileTime[str]):
    P = json.loads(plan); stages = P["stages"]; S = len(stages); cpa = S // ROWS
    cols = amps * cpa
    in_n = [st["in_n"] for st in stages]
    col_n = [sum(in_n[cc * ROWS:(cc + 1) * ROWS]) for cc in range(cpa)]          # host floats per column (per amp)
    amp_n = sum(col_n); total_in = amps * amp_n
    in_ty = np.ndarray[(total_in,), np.dtype[np.float32]]
    out_ty = np.ndarray[(cols * ROWS * OUT_N,), np.dtype[np.float32]]
    cin_ty = [np.ndarray[(n,), np.dtype[np.float32]] for n in col_n]
    cout_ty = np.ndarray[(ROWS * OUT_N,), np.dtype[np.float32]]
    obj_in = [np.ndarray[(n,), np.dtype[np.float32]] for n in in_n]
    obj_out = np.ndarray[(OUT_N,), np.dtype[np.float32]]
    pay_ty = [np.ndarray[(st["pay_out_n"],), np.dtype[np.float32]] for st in stages[:-1]]
    kern = []
    for si, st in enumerate(stages):
        at = [obj_in[si]] + ([] if st["first"] else [pay_ty[si - 1]]) + ([] if st["last"] else [pay_ty[si]]) + [obj_out]
        kern.append(ExternalFunction(f"stage{si}", source_string=P["src"][si], arg_types=at))

    def core_first(fi, fpo, fo, k):
        x = fi.acquire(1); po = fpo.acquire(1); y = fo.acquire(1); k(x, po, y); fo.release(1); fpo.release(1); fi.release(1)

    def core_mid(fi, fpi, fpo, fo, k):
        x = fi.acquire(1); pi = fpi.acquire(1); po = fpo.acquire(1); y = fo.acquire(1); k(x, pi, po, y); fo.release(1); fpo.release(1); fpi.release(1); fi.release(1)

    def core_last(fi, fpi, fo, k):
        x = fi.acquire(1); pi = fpi.acquire(1); y = fo.acquire(1); k(x, pi, y); fo.release(1); fpi.release(1); fi.release(1)

    col_in = [ObjectFifo(cin_ty[c % cpa], name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(cout_ty, name=f"out{c}", depth=1) for c in range(cols)]
    workers = []
    for a in range(amps):
        ci, co = [], []
        for cc in range(cpa):
            c = a * cpa + cc; s0 = cc * ROWS
            offs = [sum(in_n[s0:s0 + r]) for r in range(ROWS)]
            ci += col_in[c].cons().split(offs, tile=Tile(c, 1), depths=[1] * ROWS, obj_types=obj_in[s0:s0 + ROWS], names=[f"ci{c}_{r}" for r in range(ROWS)])
            co += col_out[c].prod().join([r * OUT_N for r in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS, obj_types=[obj_out] * ROWS, names=[f"co{c}_{r}" for r in range(ROWS)])
        pay = [ObjectFifo(pay_ty[s], name=f"p{a}_{s}", depth=1) for s in range(S - 1)]
        for s in range(S):
            tile = Tile(a * cpa + s // ROWS, 2 + s % ROWS)
            if s == 0: workers.append(Worker(core_first, [ci[s].cons(), pay[s].prod(), co[s].prod(), kern[s]], tile=tile))
            elif s == S - 1: workers.append(Worker(core_last, [ci[s].cons(), pay[s - 1].cons(), co[s].prod(), kern[s]], tile=tile))
            else: workers.append(Worker(core_mid, [ci[s].cons(), pay[s - 1].cons(), pay[s].prod(), co[s].prod(), kern[s]], tile=tile))
    tin = [TensorAccessPattern((total_in,), (c // cpa) * amp_n + sum(col_n[:c % cpa]), [1, 1, 1, col_n[c % cpa]], [0, 0, 0, 1]) for c in range(cols)]
    tout = TensorAccessPattern.full((cols * ROWS * OUT_N,)).partition(cols)
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


# ----------------------------------------------------------------------------------------------- host side
class NAMNPU:
    """Amps of one architecture on the NPU; process() takes one 64-sample block per amp."""

    def __init__(self, models, hifi=True, verbose=True):
        self.models = [NAM(p) if isinstance(p, str) else p for p in models]
        self.hifi = hifi
        assert all(m.signature() == self.models[0].signature() for m in self.models)
        self.stages = make_plan(self.models[0], hifi); S = len(self.stages)
        self.cpa = S // ROWS
        ncols = iron.get_current_device().cols
        self.amps = len(self.models); assert 1 <= self.amps <= ncols // self.cpa, f"at most {ncols // self.cpa} amps of {self.cpa} columns each"
        self.src = [kernel_source(si, st, hifi) for si, st in enumerate(self.stages)]
        plan = json.dumps(dict(stages=[{k: v for k, v in st.items() if k != "items"} for st in self.stages], src=self.src))
        self.plan = plan
        self.variant = "nam-" + hashlib.md5((plan + f"{T}{NSEG}{DEBUG_HEAD}").encode()).hexdigest()[:10]
        self.cols = self.amps * self.cpa
        self.in_n = [st["in_n"] for st in self.stages]; self.amp_n = sum(self.in_n)
        self.inp = np.zeros((self.amps, self.amp_n), np.float32)
        off, slope = tanh_table()
        tab = np.concatenate([off, pack_bf16(bf16_bits(slope))])
        for a, m in enumerate(self.models):
            for s, st in enumerate(self.stages):
                o = self.obj_offset(s)
                self.inp[a, o + TABOFF:o + TABOFF + len(tab)] = tab
                wb = pack_bf16(stage_weights(m, st, hifi))
                self.inp[a, o + WOFF:o + WOFF + len(wb)] = wb
        self._in = iron.zeros((self.amps * self.amp_n,), dtype=np.float32, device="npu")
        self._out = iron.zeros((self.cols * ROWS * OUT_N,), dtype=np.float32, device="npu")
        self.gen = 0; self.first = True
        if verbose:
            print(f"NAM on NPU: {self.amps} amp(s) x {S} cores, {self.models[0].describe()}, {self.models[0].macs_per_sample()} MAC/sample, "
                  f"block {T} samples, weights {'hi+lo' if hifi else 'bf16'}, host input {self.amps * self.amp_n * 4 / 1024:.0f} KB/call", flush=True)
            print(plan_text(self.stages), flush=True)
        t0 = time.perf_counter()
        self.process(np.zeros((self.amps, T), np.float32), reset=True)
        if verbose: print(f"compiled/loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    def obj_offset(self, s):
        """float offset of stage s's host object inside one amp's input"""
        return sum(self.in_n[:s])

    def process(self, blocks, reset=False):
        """blocks: (amps, T) float32 -> (amps, T) float32; self.debug (amps, cores, T) holds x[0] after every core"""
        self.gen += 1
        for a in range(self.amps):
            for s in range(len(self.stages)):
                o = self.obj_offset(s)
                self.inp[a, o] = 1.0 if (reset or self.first) else 0.0
                self.inp[a, o + 1] = float(self.gen)
            self.inp[a, XOFF:XOFF + T] = blocks[a]
        self.first = False
        with self._in.overwrite() as v: v[...] = self.inp.ravel()
        nam_npu(self._in, self._out, amps=self.amps, plan=self.plan, variant=self.variant)
        o = self._out.numpy().reshape(self.amps, len(self.stages), OUT_N)
        self.debug = o
        return o[:, -1, :].copy()

    def run(self, x, reset=True):
        """x: (amps, N) or (N,) float32 -> same shape, N truncated to a multiple of T; returns (y, seconds in NPU calls)"""
        x = np.asarray(x, np.float32); one = x.ndim == 1
        if one: x = np.tile(x[None, :], (self.amps, 1))
        n = x.shape[1] - x.shape[1] % T; y = np.zeros((self.amps, n), np.float32); tn = 0.0
        for s in range(0, n, T):
            t0 = time.perf_counter(); y[:, s:s + T] = self.process(x[:, s:s + T], reset=(reset and s == 0)); tn += time.perf_counter() - t0
        return (y[0] if one and self.amps == 1 else y), tn


class Emu:
    """numpy model of the kernel's arithmetic: operands fp32-exact (hi+lo limbs), weights bf16 unless hifi, the
    PWL tanh, fp32 accumulation; separates quantisation from bugs when compared with the NPU"""

    def __init__(self, m, hifi=True, rw=None):
        rw = rw or ((lambda a: np.asarray(a, np.float32)) if hifi else bf16)
        self.m = m; self.reset()
        self.W = [dict(Wr=rw(A["Wr"]), Wh=rw(A["Wh"]), bh=A["bh"],
                       layers=[dict(W=rw(L["W"]), b=L["b"], mix=rw(L["mix"]), W1=rw(L["W1"]), b1=L["b1"], D=L["D"]) for L in A["layers"]]) for A in m.arrays]
        self.scale = m.head_scale

    def reset(self):
        self.hist = [[np.zeros((2 * L["D"], A["C"]), np.float32) for L in A["layers"]] for A in self.m.arrays]

    def block(self, x):
        x = np.asarray(x, np.float32); c = x; Tn = len(x)
        r = (c[:, None] * self.W[0]["Wr"].T).astype(np.float32); h = np.zeros((Tn, self.m.arrays[0]["C"]), np.float32)
        for ai, (A, Wa) in enumerate(zip(self.m.arrays, self.W)):
            if ai > 0:
                h = (h @ self.W[ai - 1]["Wh"].T + self.W[ai - 1]["bh"]).astype(np.float32)
                r = (r @ Wa["Wr"].T).astype(np.float32)
            for li, L in enumerate(Wa["layers"]):
                D = L["D"]; xp = np.concatenate([self.hist[ai][li], r], 0); self.hist[ai][li] = xp[-2 * D:].copy()
                z = (L["b"] + c[:, None] * L["mix"]).astype(np.float32)
                z += xp[0:Tn] @ L["W"][:, :, 0].T + xp[D:Tn + D] @ L["W"][:, :, 1].T + xp[2 * D:Tn + 2 * D] @ L["W"][:, :, 2].T
                a = pwl_tanh(z)
                h = h + a
                r = r + (a @ L["W1"].T + L["b1"]).astype(np.float32)
        y = (h @ Wa["Wh"].T + Wa["bh"]).astype(np.float32)[:, 0]
        return (y * self.scale).astype(np.float32)

    def process(self, x, block=T):
        self.reset(); x = np.asarray(x, np.float32); n = len(x) - len(x) % block; y = np.zeros(n, np.float32)
        for s in range(0, n, block): y[s:s + block] = self.block(x[s:s + block])
        return y


def snr_db(ref, y):
    ref = np.asarray(ref, np.float64); e = np.asarray(y, np.float64) - ref
    return 10 * np.log10(max(np.sum(ref ** 2), 1e-30) / max(np.sum(e ** 2), 1e-30))


def test_signal(seconds=1.0, seed=0, level_db=-12):
    t = np.arange(int(FS * seconds)) / FS
    f0, f1 = 80.0, 5000.0
    sweep = np.sin(2 * np.pi * f0 * seconds / np.log(f1 / f0) * (np.exp(t / seconds * np.log(f1 / f0)) - 1))
    rng = np.random.default_rng(seed)
    bursts = rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 3 * t) > 0.8)
    x = 0.25 * sweep + 0.1 * bursts
    return (x / np.abs(x).max() * 10 ** (level_db / 20)).astype(np.float32)


def load_wav(path, seconds=None):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, fs = sf.read(path, dtype="float32", always_2d=True); x = x[:, 0]
    if fs != FS:
        from math import gcd
        g = gcd(FS, fs); x = resample_poly(x, FS // g, fs // g).astype(np.float32)
    if seconds: x = x[:int(seconds * FS)]
    return x


def default_models():
    cands = ["nam-plugin-reaper/model.nam", "neuralaudio/BossWN-standard.nam", "pelennor2170/Tim_R_Fender_TwinVerb_Norm_Bright.nam",
             "pelennor2170/Tim_R_JCM2000_Crunch.nam", "pelennor2170/Helga_B_5150_BlockLetter_-_NoBoost.nam"]
    return [os.path.join(MODELS, c) for c in cands if os.path.exists(os.path.join(MODELS, c))]


def selftest(model_paths, seconds=1.0, hifi=True):
    models = [NAM(p) for p in model_paths]
    npu = NAMNPU(models, hifi)
    x = test_signal(seconds)
    y, tn = npu.run(x)
    y = y if y.ndim == 2 else y[None, :]
    n = y.shape[1]; nblocks = n // T
    print(f"{nblocks} blocks of {T}: {tn * 1e3 / nblocks:.3f} ms per NPU call ({T / FS * 1e3:.2f} ms of audio), {npu.amps} amp(s) at once")
    for a, m in enumerate(models):
        ref = RefFP32(m).process(x[:n]); emu = Emu(m, hifi).process(x[:n])
        skip = m.receptive  # judge after the receptive field has filled with signal
        print(f"  {os.path.basename(m.path)}: NPU vs fp32 {snr_db(ref[skip:], y[a, skip:]):.1f} dB, NPU vs emu {snr_db(emu[skip:], y[a, skip:]):.1f} dB, "
              f"emu vs fp32 {snr_db(ref[skip:], emu[skip:]):.1f} dB; peak ref {np.abs(ref).max():.3f} npu {np.abs(y[a]).max():.3f}")
    return npu


def stage_check(model_path, nblocks=80, hifi=True):
    """compare every core's debug channel (x[0] after its items) with the fp32 trace, block by block"""
    m = NAM(model_path); npu = NAMNPU([m], hifi); ref = RefFP32(m)
    x = test_signal(nblocks * T / FS + 0.01)
    # what each core's x[0] is: ('c',) before any layer, ('x', ai, li) after a layer, ('r2',) after trans, ('y',) final
    keys = []; key = ("c",)
    for st in npu.stages:
        for it in st["items"]:
            if it["k"] == "rech_in": key = ("r1",)
            elif it["k"] == "layer" and it["end"]: key = ("x", it["a"], [l["D"] for l in m.arrays[it["a"]]["layers"]].index(it["D"]))
            elif it["k"] == "trans": key = ("r2",)
            elif it["k"] == "final": key = ("y",)
        keys.append(key)
    S = len(keys); err = np.zeros(S); pw = np.zeros(S)
    for b in range(nblocks):
        xb = x[b * T:(b + 1) * T]; trace = []
        yref = ref.block(xb, trace); npu.process(xb[None, :], reset=(b == 0)); dbg = npu.debug[0]
        xs = {(ai, li): r for ai, li, r, h in trace}; hs = {(ai, li): h for ai, li, r, h in trace}
        nl0 = len(m.arrays[0]["layers"])
        for s, key in enumerate(keys):
            if DEBUG_HEAD and key[0] != "y":
                if key[0] == "x": r = hs[(key[1], key[2])][:, 0]
                elif key[0] == "r2": r = (hs[(0, nl0 - 1)] @ m.arrays[0]["Wh"].T + m.arrays[0]["bh"])[:, 0]
                else: r = np.zeros(T, np.float32)
            elif key[0] == "x": r = xs[(key[1], key[2])][:, 0]
            elif key[0] == "r1": r = xb * m.arrays[0]["Wr"][0, 0]
            elif key[0] == "r2": r = (xs[(0, nl0 - 1)] @ m.arrays[1]["Wr"].T)[:, 0]
            elif key[0] == "y": r = yref
            else: r = xb
            if b * T >= m.receptive:
                err[s] += np.sum((dbg[s] - r) ** 2); pw[s] += np.sum(r ** 2)
    for s in range(S):
        print(f"  core {s} ({'/'.join(str(k) for k in keys[s])}): SNR {10 * np.log10(max(pw[s], 1e-30) / max(err[s], 1e-30)):.1f} dB")


def render(model_paths, wav, out, ir=None, seconds=None, gain_db=0.0, level_db=-18.0, hifi=True):
    import soundfile as sf
    from scipy.signal import fftconvolve
    x = load_wav(wav, seconds) * 10 ** (gain_db / 20)
    npu = NAMNPU(model_paths, hifi)
    y, tn = npu.run(x)
    y = y if y.ndim == 2 else y[None, :]
    print(f"{len(x) / FS:.1f} s of audio in {tn:.2f} s of NPU calls ({len(x) / FS / tn:.1f}x real time)")
    h = None
    if ir:
        h = load_wav(ir); h = h / np.sqrt(np.sum(h ** 2))
    outs = []
    for a, m in enumerate(npu.models):
        ya = y[a]
        if h is not None: ya = fftconvolve(ya, h)[:len(ya)].astype(np.float32)
        loud = m.meta.get("loudness")
        g = 10 ** ((level_db - loud) / 20) if loud is not None else 10 ** (level_db / 20) / max(np.sqrt(np.mean(ya ** 2)), 1e-9)
        g = min(g, 0.89 / max(np.abs(ya).max(), 1e-9))        # loudness target, but never past -1 dBFS peak
        ya = ya * g
        path = out if npu.amps == 1 else out.replace(".wav", f"_{a}.wav")
        sf.write(path, ya, FS); outs.append(path)
        print(f"wrote {path}: {os.path.basename(m.path)} ({m.meta.get('name', '?')}), peak {np.abs(ya).max():.2f}")
    return outs


def live(model_paths, ir=None, gain_db=0.0, device=None, blocks=1, hifi=True, seconds=None):
    """mic/line in -> NPU amp -> speakers; one 64-sample block per callback (1.3 ms) unless blocks > 1"""
    import sounddevice as sd
    from scipy.signal import fftconvolve
    npu = NAMNPU(model_paths, hifi)
    gain = 10 ** (gain_db / 20)
    h = None; tail = None
    if ir:
        h = load_wav(ir); h = (h / np.sqrt(np.sum(h ** 2))).astype(np.float32); tail = np.zeros(len(h) - 1, np.float32)
    m = npu.models[0]; loud = m.meta.get("loudness")
    og = 10 ** ((-18.0 - loud) / 20) if loud is not None else 1.0
    stats = dict(calls=0, t=0.0, under=0, peak=0.0)
    sel = [0]

    def cb(indata, outdata, frames, tinfo, status):
        if status: stats["under"] += 1
        x = indata[:, 0] * gain
        t0 = time.perf_counter(); y = np.zeros(frames, np.float32)
        for s in range(0, frames, T):
            y[s:s + T] = npu.process(np.tile(x[None, s:s + T], (npu.amps, 1)))[sel[0]]
        stats["t"] += time.perf_counter() - t0; stats["calls"] += frames // T
        if h is not None:
            full = fftconvolve(y, h); full[:len(tail)] += tail; tail[:] = full[frames:frames + len(tail)]; y = full[:frames]
        y = np.clip(y * og, -1, 1); stats["peak"] = max(stats["peak"], float(np.abs(y).max()))
        outdata[:, 0] = y
        if outdata.shape[1] > 1: outdata[:, 1] = y

    print(f"live: {[os.path.basename(p) for p in model_paths]}; keys 1-{npu.amps} pick the amp, q quits; input gain {gain_db:+.0f} dB")
    with sd.Stream(samplerate=FS, blocksize=T * blocks, channels=(1, 2), dtype="float32", device=device, callback=cb, latency="low") as stream:
        print(f"stream latency {stream.latency[0] * 1e3:.1f} ms in, {stream.latency[1] * 1e3:.1f} ms out, blocksize {T * blocks}")
        import select
        tty_in = sys.stdin.isatty() and seconds is None
        if tty_in:
            import termios, tty
            fd = sys.stdin.fileno(); old = termios.tcgetattr(fd); tty.setcbreak(fd)
        t_end = time.time() + (seconds or 1e9)
        try:
            while time.time() < t_end:
                if tty_in and select.select([sys.stdin], [], [], 0.5)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "q": break
                    if ch.isdigit() and 1 <= int(ch) <= npu.amps: sel[0] = int(ch) - 1; print(f"amp {ch}: {os.path.basename(model_paths[sel[0]])}")
                elif not tty_in: time.sleep(0.5)
                if stats["calls"]:
                    print(f"\r{stats['t'] / stats['calls'] * 1e3:.2f} ms/block NPU, underruns {stats['under']}, peak {stats['peak']:.2f}   ", end="", flush=True)
                    stats["peak"] = 0.0
        finally:
            if tty_in: termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", help=".nam file(s)")
    ap.add_argument("--bf16-weights", action="store_true", help="single bf16 weights (two macs per term, ~46 dB) instead of hi+lo pairs (three macs, ~66 dB)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stages", action="store_true", help="per-core comparison with the fp32 trace")
    ap.add_argument("--plan", action="store_true", help="print the core plan and exit")
    ap.add_argument("--head", action="store_true", help="with --stages: compare the head stream instead of x")
    ap.add_argument("--wav"); ap.add_argument("--out", default="nam_out.wav"); ap.add_argument("--ir")
    ap.add_argument("--seconds", type=float); ap.add_argument("--gain", type=float, default=0.0, help="input gain dB")
    ap.add_argument("--live", action="store_true"); ap.add_argument("--device"); ap.add_argument("--blocks", type=int, default=1)
    a = ap.parse_args(); a.hifi = not a.bf16_weights
    if a.head: DEBUG_HEAD = True
    models = a.model or default_models()[:1]
    if a.plan:
        m = NAM(models[0]); print(plan_text(make_plan(m, a.hifi)))
    elif a.stages: stage_check(models[0], hifi=a.hifi)
    elif a.selftest: selftest(models, a.seconds or 1.0, a.hifi)
    elif a.live: live(models, a.ir, a.gain, a.device, a.blocks, a.hifi, a.seconds)
    elif a.wav: render(models, a.wav, a.out, a.ir, a.seconds, a.gain, hifi=a.hifi)
    else: ap.print_help()
