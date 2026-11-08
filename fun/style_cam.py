#!/usr/bin/env python3
"""Fast neural style transfer on the NPU, live from the webcam: Johnson et al.'s TransformerNet (the pytorch/examples
fast_neural_style models, here the ONNX zoo exports) as a 31-phase int8 pipeline on all 32 cores in one dispatch.

  ./env.sh python fun/style_cam.py --selftest                  COCO image: fp32 vs onnxruntime, int8 emulation, NPU
  ./env.sh python fun/style_cam.py --image in.jpg --out x.png  one picture, every style
  ./env.sh python fun/style_cam.py --window                    live webcam (keys 1..5 pick the style, 0 input, ESC)
  ./env.sh python fun/style_cam.py --bench                     NPU time per frame
  ./env.sh python fun/style_cam.py --vcam auto                 virtual webcam for Zoom/Jitsi/the browser (v4l2loopback),
                                                               headless; add --window to watch, --cycle 20 to rotate styles

The network: conv9x9 3->32, two stride-2 conv3x3 (->64, ->128), five residual blocks at 128 channels, two nearest-upsample
conv3x3 (->64, ->32), conv9x9 32->3; reflection padding; InstanceNorm (affine) + ReLU after every conv but the last.
13.8 GMAC per 320x240 frame.

The NPU plan: 16 conv phases and 15 norm phases in one dispatch, every phase on all 32 cores. A column streams one band's
row windows through a memtile broadcast to its 4 cores, which hold different output-channel groups (or, in the last
layer, different tile columns); 40-pixel tiles, 8x8x8 int8 mmul, int32 accumulation, int16 raw outputs plus per-channel
sum and sum of squares. A norm phase reduces those statistics, applies instance norm, ReLU and the residual add, and
requantises to int8 planar activations. Activations are signed int8 throughout: post-ReLU tensors are stored as q - 128
(the constant this adds to a conv output is removed by the next instance norm, and folded into the last layer's bias).
Every core runs one generic loop over a segment table (rows only / rows + out / out + m rows) with run-time trip counts;
the kernel keeps its own phase and step counters. Weights (per style) are a runtime buffer: switching styles costs
nothing. Everything intermediate stays in a DDR scratch buffer.

Models: ~/models/style/onnx-zoo/*-9.onnx (ONNX Model Zoo, BSD-3-Clause; see NOTE.txt there). Not redistributed.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

TILE = 40                      # output tile width (px); a tile-row is the unit of work
RS = 48                        # ring row stride in px per channel block (48 * 8 = 384 B, a multiple of 64)
ROWB = 5632                    # bytes of a `rows` object: a row window, 4 weight-record pieces, consts, stats, raw or residual
OUTB = 1280                    # bytes of an `out` object: a raw tile-row (16 ch x 40 px int16), a norm record or RGBA
WREC = 21120                   # bytes of one core's weight record (4 records = 15 rows objects); tail at WTAIL
WTAIL = WREC - 256
NWOBJ = 4 * WREC // ROWB       # rows objects that carry a column's 4 weight records
CREC = 1408                    # bytes of one norm core's constants (4 = one rows object)
SREC = 176                     # stats bytes read per conv core (128 used; 32 records = one rows object)
COLS, ROWS = 8, 4
NCORE = COLS * ROWS
MODELS = Path.home() / "models" / "style" / "onnx-zoo"
EPS = 1e-5
assert NWOBJ * ROWB == 4 * WREC and 4 * CREC == ROWB and NCORE * SREC == ROWB


# ------------------------------------------------------------------------------------------------- the network
class Net:
    """TransformerNet weights from an ONNX zoo file, as a list of conv layers (+ instance norm, relu, residual)."""

    def __init__(self, path):
        import onnx
        from onnx import numpy_helper
        self.path = str(path); self.name = Path(path).stem.replace("-9", "")
        m = onnx.load(path); self.onnx = m
        w = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
        self.layers = []

        def conv(name, K, stride=1, up=1, norm=None, relu=True, residual=False):
            self.layers.append(dict(name=name, W=w[f"{name}.conv2d.weight"].astype(np.float32), b=w[f"{name}.conv2d.bias"].astype(np.float32),
                                    K=K, stride=stride, up=up, relu=relu, residual=residual,
                                    gamma=None if norm is None else w[f"{norm}.weight"].astype(np.float32),
                                    beta=None if norm is None else w[f"{norm}.bias"].astype(np.float32)))
        conv("conv1", 9, norm="in1"); conv("conv2", 3, stride=2, norm="in2"); conv("conv3", 3, stride=2, norm="in3")
        for r in range(1, 6):
            conv(f"res{r}.conv1", 3, norm=f"res{r}.in1"); conv(f"res{r}.conv2", 3, norm=f"res{r}.in2", relu=False, residual=True)
        conv("deconv1", 3, up=2, norm="in4"); conv("deconv2", 3, up=2, norm="in5"); conv("deconv3", 9, norm=None, relu=False)
        used = sum(l["W"].size + l["b"].size + (0 if l["gamma"] is None else 2 * l["gamma"].size) for l in self.layers)
        assert used == sum(v.size for v in w.values()), "unexpected weights in the file"
        self.params = used

    def spec(self):
        """Architecture only (what the NPU plan needs): shapes and layer kinds, no weights."""
        return [dict(name=l["name"], cout=int(l["W"].shape[0]), K=l["K"], stride=l["stride"], up=l["up"], relu=l["relu"],
                     residual=l["residual"], norm=l["gamma"] is not None) for l in self.layers]

    def forward(self, img, trace=None):
        """img (H,W,3) 0..255 -> (3,H,W) float32, unclipped. trace collects every layer output."""
        x = np.ascontiguousarray(np.transpose(np.asarray(img, np.float32), (2, 0, 1)))
        skip = None
        for l in self.layers:
            if l["name"].endswith(".conv1"): skip = x
            if l["up"] == 2: x = x.repeat(2, axis=1).repeat(2, axis=2)
            y = conv2d(np.pad(x, ((0, 0), (l["K"] // 2,) * 2, (l["K"] // 2,) * 2), mode="reflect"), l["W"], l["b"], l["K"], l["stride"])
            if l["gamma"] is not None:
                mu = y.mean(axis=(1, 2), keepdims=True); var = y.var(axis=(1, 2), keepdims=True)
                y = (y - mu) / np.sqrt(var + EPS) * l["gamma"][:, None, None] + l["beta"][:, None, None]
            if l["relu"]: y = np.maximum(y, 0)
            if l["residual"]: y = y + skip
            if trace is not None: trace.append(y)
            x = y
        return x

    def ort(self, img):
        """The same picture through onnxruntime (input dims made dynamic)."""
        import onnx, onnxruntime as ort
        m = onnx.ModelProto.FromString(self.onnx.SerializeToString())
        for io in (m.graph.input[0], m.graph.output[0]):
            for d, nm in zip(io.type.tensor_type.shape.dim, "nchw"):
                if nm in "hw": d.dim_param = nm
        so = ort.SessionOptions(); so.log_severity_level = 3
        s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
        x = np.transpose(np.asarray(img, np.float32), (2, 0, 1))[None]
        return s.run(None, {s.get_inputs()[0].name: x})[0][0]


def conv2d(x, W, b, K, stride):          # x padded (C,Hp,Wp), W (O,C,K,K) -> (O,H,W)
    C, Hp, Wp = x.shape; O = W.shape[0]
    H = (Hp - K) // stride + 1; Wd = (Wp - K) // stride + 1
    y = np.zeros((O, H * Wd), np.float32)
    for dy in range(K):
        for dx in range(K):
            y += W[:, :, dy, dx] @ x[:, dy:dy + stride * H:stride, dx:dx + stride * Wd:stride].reshape(C, -1)
    return y.reshape(O, H, Wd) + b[:, None, None]


def to_rgb(y):
    return np.clip(np.transpose(y, (1, 2, 0)) + 0.5, 0, 255).astype(np.uint8)


def psnr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    e = np.mean((a - b) ** 2)
    return 99.0 if e == 0 else 10 * np.log10(255.0 ** 2 / e)


# ------------------------------------------------------------------------------------------------- the plan
def refl(i, n):
    """Reflection padding index (PyTorch ReflectionPad / ONNX reflect): -1 -> 1, n -> n-2."""
    if i < 0: i = -i
    if i >= n: i = 2 * n - 2 - i
    return i




class Plan:
    """Phases and per-column streams for a W x H picture (W a multiple of 320, H a multiple of 16)."""

    def __init__(self, spec, W, H):
        if isinstance(spec, Net): spec = spec.spec()
        self.spec = spec
        assert W % 320 == 0 and H % 16 == 0, "W a multiple of 320, H a multiple of 16"
        self.W, self.H = W, H
        self.phases = []
        res_names = ["R0", "R1"]; rsel = 0
        src = "IMG"; shape = (8, H, W)
        uns = {"IMG": False}                     # stored values are signed int8 everywhere; this marks q-128 storage
        for li, l in enumerate(spec):
            C, h, w = shape; O = l["cout"]
            if l["stride"] == 2: ho, wo, mode = h // 2, w // 2, "s2"
            elif l["up"] == 2: ho, wo, mode = h * 2, w * 2, "up"
            else: ho, wo, mode = h, w, "s1"
            final = not l["norm"]
            if final: O = 8
            conv = self.conv_phase(li, l, mode, C, O, h, w, ho, wo, src, final)
            conv["src_uns"] = uns[src]
            self.phases.append(conv)
            if final: break
            if l["residual"]: dst = res_names[1 - rsel]; resbuf = res_names[rsel]; rsel = 1 - rsel
            elif li == 2: dst = res_names[rsel]; resbuf = None
            else: dst = f"X{li}"; resbuf = None
            nrm = self.norm_phase(li, l, conv, O, ho, wo, dst, resbuf)
            nrm["res_uns"] = uns[resbuf] if resbuf else False
            uns[dst] = bool(l["relu"])
            self.phases.append(nrm)
            src = dst; shape = (O, ho, wo)
        self.tensors = {"IMG": (8, H, W)}
        for p in self.phases:
            if p["kind"] == "norm":
                shp = (p["cout"], p["hout"], p["wout"])
                assert self.tensors.get(p["dst"], shp) == shp
                self.tensors[p["dst"]] = shp
        self.unsigned = uns
        self.layout()

    def conv_phase(self, li, l, mode, C, O, h, w, ho, wo, src, final):
        K = l["K"]; TX = wo // TILE
        assert wo % TILE == 0 and C % 8 == 0 and O % 8 == 0
        G = 1 if final else O // 16 if O >= 64 else O // 8      # output-channel groups: 16 channels, 8 for 32-channel layers
        cpc = O // G
        CB = max(1, G // ROWS); B = COLS // CB                     # columns per band, bands
        bh = ho // B
        assert ho % B == 0 and (not final or TX % ROWS == 0), (li, ho, B)
        kb = C // 8
        need = {"s1": TILE + 2 * (K // 2), "s2": 2 * TILE + 2, "up": TILE // 2 + 2}[mode]
        pxo = ROWB // (kb * 8); assert pxo * kb * 8 == ROWB and pxo >= need, (li, pxo, need)
        nlines = {"s1": bh + K - 1, "s2": 2 * bh + 1, "up": bh + 1}[mode]
        q = TX // ROWS if final else 0
        nsteps = q * nlines if final else TX * nlines + 1          # A steps (one per row window) + the stats step; final: Q steps
        return dict(kind="conv", li=li, name=l["name"], K=K, mode=mode, cin=C, cout=O, hin=h, win=w, hout=ho, wout=wo,
                    src=src, final=final, G=G, cpc=cpc, CB=CB, B=B, bh=bh, TX=TX, q=q, kb=kb, nb=cpc // 8, pxo=pxo,
                    nlines=nlines, nsteps=nsteps)

    def norm_phase(self, li, l, conv, O, ho, wo, dst, resbuf):
        cbs = O // 8 // ROWS
        cpc = conv["cpc"]; G = conv["G"]
        TG = 4 // cbs                                              # tiles per out record (4 planes x 1 tile, 2 x 2, 1 x 4)
        TX = conv["TX"]; CB = conv["CB"]
        ntile = TX // CB                                           # tiles per norm column
        assert ntile % TG == 0, (li, ntile, TG)
        nobj = -(-G * OUTB // ROWB)                                # raw objects per tile-row (4 records each)
        return dict(kind="norm", li=li, name=l["name"] + ".norm", cout=O, hout=ho, wout=wo, dst=dst, res=resbuf,
                    relu=l["relu"], cbs=cbs, cpc=cpc, G=G, CB=CB, B=conv["B"], bh=conv["bh"], TX=TX, TG=TG,
                    ngroups=ntile // TG, nobj=nobj, m=TG * nobj, conv=conv)

    # ---- roles
    def band(self, p, c):
        """(y0, y1) output rows of column c in phase p."""
        b = c // p["CB"]
        return b * p["bh"], (b + 1) * p["bh"]

    def conv_group(self, p, c, r):
        return 0 if p["final"] else (c % p["CB"]) * ROWS + r

    def norm_tiles(self, p, c):
        """Tile groups (lists of TG adjacent tiles) of norm column c."""
        tiles = [tx for tx in range(p["TX"]) if tx % p["CB"] == c % p["CB"]]
        assert all(tiles[i + 1] - tiles[i] == p["CB"] for i in range(len(tiles) - 1))
        return [tiles[i:i + p["TG"]] for i in range(0, len(tiles), p["TG"])]

    def norm_core(self, p, r):
        """(object index, byte offset in it, channel offsets) of norm core r's raw records within a tile-row."""
        g0 = r * p["cbs"] * 8 // p["cpc"]
        return g0 // 4, (g0 % 4) * OUTB

    # ---- streams
    @staticmethod
    def lo(u, h):
        return refl(u, 2 * h) // 2

    def line_row(self, p, y0, li):
        """Source row (input tensor) of line li of a tile column whose output band starts at y0."""
        K, h = p["K"], p["hin"]
        if p["mode"] == "s1": return refl(y0 - K // 2 + li, h)
        if p["mode"] == "s2": return refl(2 * y0 - 1 + li, h)
        return self.lo(y0 - 1, h) if li == 0 else self.lo(y0 + li, h)

    def line_out(self, p, li):
        """Output row offset (in the band) produced after line li, or None."""
        if p["mode"] == "s1": return li - (p["K"] - 1) if li >= p["K"] - 1 else None
        if p["mode"] == "s2": return (li - 2) // 2 if li >= 2 and li % 2 == 0 else None
        return li - 1 if li >= 1 else None

    def x_src(self, p, tx):
        """First input px of the row window of tile column tx (may be negative: the region has head slack)."""
        pd = p["K"] // 2
        if p["mode"] == "s1": return tx * TILE - pd
        if p["mode"] == "s2": return 2 * tx * TILE - 1
        return tx * TILE // 2 - 1

    def step_of(self, p, tx, y):
        """A-step index (in its column) that produced output row y (band-relative) of tile tx."""
        off = {"s1": p["K"] - 1, "s2": None, "up": 1}[p["mode"]]
        li = 2 * y + 2 if p["mode"] == "s2" else y + off
        return tx * p["nlines"] + li

    # ---- DDR layout of the scratch buffer and the weights blob
    HEAD = 64

    def layout(self):
        off = 0; self.buf = {}

        def alloc(name, nbytes):
            nonlocal off
            nb = (nbytes + 4095) // 4096 * 4096
            self.buf[name] = (off, nb); off += nb
        regions = {}
        for name, (C, h, w) in self.tensors.items():
            reg = "X" if name.startswith("X") else name
            regions[reg] = max(regions.get(reg, 0), C * h * w)
        for reg, nbytes in regions.items():
            alloc(reg, self.HEAD + nbytes + 16384)
        self.region = lambda name: "X" if name.startswith("X") else name
        raw = max(p["B"] * p["nsteps"] * p["G"] * OUTB for p in self.phases if p["kind"] == "conv" and not p["final"])
        alloc("RAW", raw + 16384)
        fin = self.phases[-1]
        alloc("RGB", COLS * fin["nsteps"] * ROWS * OUTB)
        self.scratch_bytes = off
        woff = 0; self.wrec = {}
        for p in self.phases:
            if p["kind"] == "conv":
                self.wrec[p["li"], "w"] = woff; woff += WREC * (4 if p["final"] else p["G"])
            else:
                self.wrec[p["li"], "n"] = woff; woff += ROWB
        self.weights_bytes = woff

    def signature(self):
        keys = ("kind", "li", "K", "mode", "cin", "cout", "hin", "win", "hout", "wout", "final", "G", "cpc", "CB", "B", "bh",
                "TX", "q", "kb", "nb", "pxo", "nlines", "nsteps", "cbs", "TG", "ngroups", "nobj", "m", "relu", "res", "src",
                "dst", "src_uns", "res_uns")
        return json.dumps([{k: p[k] for k in keys if k in p} for p in self.phases] + [self.W, self.H, TILE, RS, ROWB, OUTB, WREC])


# ------------------------------------------------------------------------------------------------- quantisation
def sat(v, lo, hi):
    return np.clip(np.rint(v), lo, hi)


class Quant:
    """int8 parameters of a Net and a numpy emulation of exactly what the NPU computes (signed int8 storage,
    exact int32 conv, int16 raw with a per-layer shift, centred statistics, fp32 norm, requantisation)."""
    VERSION = 3

    def __init__(self, net, images, headroom=1.25):
        self.net = net
        L = net.layers; n = len(L)
        traces = []
        for img in images:
            tr = []; net.forward(img, tr); traces.append(tr)
        self.s_out = [None] * n; self.uns_out = [False] * n
        for li, l in enumerate(L):
            if l["gamma"] is None: continue
            ys = np.stack([tr[li] for tr in traces])
            flat = np.abs(ys).reshape(ys.shape[0], ys.shape[1], -1)
            mx = np.quantile(flat, 0.9999, axis=(0, 2)) * headroom
            mx = np.maximum(mx, 0.01 * mx.max())
            self.s_out[li] = (mx / (255.0 if l["relu"] else 127.0)).astype(np.float32); self.uns_out[li] = bool(l["relu"])
        self.s_in = [None] * n; self.uns_in = [False] * n; self.s_res = [None] * n; self.uns_res = [False] * n
        for li, l in enumerate(L):
            if li == 0: self.s_in[li] = np.array([2.0] * 3 + [1.0] * 5, np.float32)
            else: self.s_in[li] = self.s_out[li - 1]; self.uns_in[li] = self.uns_out[li - 1]
            if l["residual"]: self.s_res[li] = self.s_out[li - 2]; self.uns_res[li] = self.uns_out[li - 2]
        self.qw = []; self.s_w = []
        for li, l in enumerate(L):
            W = l["W"].astype(np.float64)
            if li == 0: W = np.concatenate([W, np.zeros((W.shape[0], 5) + W.shape[2:])], axis=1)
            Wf = W * self.s_in[li][None, :, None, None].astype(np.float64)
            if l["gamma"] is None: Wf = np.concatenate([Wf, np.zeros((8 - Wf.shape[0],) + Wf.shape[1:])], axis=0)
            sw = np.maximum(np.abs(Wf).reshape(Wf.shape[0], -1).max(axis=1) / 127.0, 1e-12)
            self.qw.append(sat(Wf / sw[:, None, None, None], -127, 127).astype(np.int8)); self.s_w.append(sw.astype(np.float32))
        fl = L[-1]; qw = self.qw[-1].astype(np.float64)
        off = 128.0 * qw.sum(axis=(1, 2, 3)) if self.uns_in[n - 1] else 0.0      # stored q-128 inputs: fold +128 into the bias
        self.fbias = (np.concatenate([fl["b"], np.zeros(5)]) + off * self.s_w[-1]).astype(np.float32)
        self.shift = [0] * n; self.m0 = [None] * n
        accmax = [0.0] * n; m0acc = [[] for _ in range(n)]
        for img in images: self.emulate(img, calib=(accmax, m0acc))
        for li, l in enumerate(L):
            if l["gamma"] is not None: self.shift[li] = max(0, int(np.ceil(np.log2(max(accmax[li], 1.0) * 1.5 / 32767.0))))
        m0acc = [[] for _ in range(n)]
        for img in images: self.emulate(img, calib=(None, m0acc))
        for li, l in enumerate(L):
            if l["gamma"] is not None: self.m0[li] = sat(np.mean(m0acc[li], axis=0), -32767, 32767).astype(np.int16)

    def norm_consts(self, li):
        """Per-channel float32 constants the kernel uses: unit (raw LSB in conv units), gamma, beta, 1/s_out, s_res."""
        l = self.net.layers[li]
        unit = (2.0 ** self.shift[li]) * self.s_w[li].astype(np.float64)
        return unit, l["gamma"].astype(np.float64), l["beta"].astype(np.float64), 1.0 / self.s_out[li].astype(np.float64), \
            (self.s_res[li].astype(np.float64) if self.s_res[li] is not None else np.zeros_like(unit))

    def emulate(self, img, calib=None, trace=None, upto=None):
        """The NPU arithmetic in numpy. Returns the uint8 (H,W,3) picture (or None when stopped early by `upto`)."""
        L = self.net.layers
        x = np.zeros((8,) + np.asarray(img).shape[:2], np.float64)
        x[:3] = np.transpose(np.asarray(img), (2, 0, 1)).astype(np.float64) // 2          # stored image: pixel >> 1
        skip = None
        for li, l in enumerate(L):
            if l["name"].endswith(".conv1"): skip = x
            xi = x.repeat(2, axis=1).repeat(2, axis=2) if l["up"] == 2 else x
            pd = l["K"] // 2
            acc = conv2d(np.pad(xi, ((0, 0), (pd, pd), (pd, pd)), mode="reflect"), self.qw[li].astype(np.float64),
                         np.zeros(self.qw[li].shape[0]), l["K"], l["stride"])
            if l["gamma"] is None:
                y = acc * self.s_w[li][:, None, None].astype(np.float64) + self.fbias[:, None, None]
                rgb = np.clip(np.floor(y[:3] + 0.5), 0, 255).astype(np.uint8)
                if trace is not None: trace.append(dict(acc=acc))
                return np.transpose(rgb, (1, 2, 0))
            if calib is not None and calib[0] is not None: calib[0][li] = max(calib[0][li], float(np.abs(acc).max()))
            raw = sat(acc / 2.0 ** self.shift[li], -32768, 32767)
            if calib is not None: calib[1][li].append(raw.mean(axis=(1, 2)))
            m0 = self.m0[li].astype(np.float64) if self.m0[li] is not None else np.zeros(raw.shape[0])   # calibration passes
            c = raw - m0[:, None, None]; N = raw.shape[1] * raw.shape[2]
            mean_c = c.sum(axis=(1, 2)) / N; var16 = np.maximum((c * c).sum(axis=(1, 2)) / N - mean_c ** 2, 0)
            unit, gam, bet, inv, sres = self.norm_consts(li)
            a = gam / np.sqrt(var16 * unit * unit + EPS)
            sc = unit * a * inv; sh = (bet - (mean_c + m0) * unit * a) * inv; sr = sres * inv
            if l["relu"]: sh = sh - 128.0
            if l["residual"] and self.uns_res[li]: sh = sh + 128.0 * sr
            y = raw * sc[:, None, None] + sh[:, None, None]
            if l["residual"]: y = y + skip * sr[:, None, None]
            if l["relu"]: y = np.maximum(y, -128.0)
            x = sat(y, -128, 127)
            if trace is not None: trace.append(dict(raw=raw, x=x))
            if upto is not None and li >= upto: return None
        raise AssertionError("no final layer")

    def conv_raw(self, li, x):
        """Layer li's int16 raw output for a stored int8 input tensor x (C, h, w)."""
        l = self.net.layers[li]; x = np.asarray(x, np.float64)
        if l["up"] == 2: x = x.repeat(2, axis=1).repeat(2, axis=2)
        pd = l["K"] // 2
        acc = conv2d(np.pad(x, ((0, 0), (pd, pd), (pd, pd)), mode="reflect"), self.qw[li].astype(np.float64), np.zeros(self.qw[li].shape[0]), l["K"], l["stride"])
        return sat(acc / 2.0 ** self.shift[li], -32768, 32767)

    def save(self, path):
        n = len(self.net.layers)
        d = dict(version=self.VERSION, shift=np.array(self.shift), uns_out=np.array(self.uns_out), uns_in=np.array(self.uns_in),
                 uns_res=np.array(self.uns_res), fbias=self.fbias)
        for i in range(n):
            d[f"qw{i}"] = self.qw[i]; d[f"sw{i}"] = self.s_w[i]; d[f"si{i}"] = self.s_in[i]
            for key, v in (("so", self.s_out[i]), ("m0", self.m0[i]), ("sr", self.s_res[i])):
                if v is not None: d[f"{key}{i}"] = v
        np.savez(path, **d)

    @classmethod
    def load(cls, net, path):
        d = np.load(path); q = cls.__new__(cls); q.net = net; n = len(net.layers)
        if int(d["version"]) != cls.VERSION: return None
        q.shift = [int(v) for v in d["shift"]]; q.uns_out = [bool(v) for v in d["uns_out"]]
        q.uns_in = [bool(v) for v in d["uns_in"]]; q.uns_res = [bool(v) for v in d["uns_res"]]; q.fbias = d["fbias"]
        q.qw = [d[f"qw{i}"] for i in range(n)]; q.s_w = [d[f"sw{i}"] for i in range(n)]; q.s_in = [d[f"si{i}"] for i in range(n)]
        get = lambda k: d[k] if k in d.files else None
        q.s_out = [get(f"so{i}") for i in range(n)]; q.m0 = [get(f"m0{i}") for i in range(n)]; q.s_res = [get(f"sr{i}") for i in range(n)]
        return q


def calibration_images(n=5, size=(320, 240)):
    import cv2
    files = sorted((Path.home() / "models" / "yolo26n" / "coco128" / "images" / "train2017").glob("*.jpg"))[:n]
    assert files, "no calibration images (expected ~/models/yolo26n/coco128)"
    return [cv2.resize(cv2.imread(str(f)), size)[:, :, ::-1].copy() for f in files]


def get_quant(net, calib=None):
    cache = Path(net.path).with_suffix(f".quant{Quant.VERSION}.npz")
    if cache.exists():
        q = Quant.load(net, cache)
        if q is not None: return q
    q = Quant(net, calib if calib is not None else calibration_images()); q.save(cache); return q


# ------------------------------------------------------------------------------------------------- the kernel
def kernel_source(plan):
    rows = []
    for p in plan.phases:
        if p["kind"] == "conv":
            kind = 1 if p["final"] else 0
            mode = {"s1": 0, "s2": 1, "up": 2}[p["mode"]]
            rows.append(f"{{{kind},{p['K']},{mode},{p['kb']},{p['nb']},{p['pxo']},{p['win']},{p['hin']},{p['nlines']},{p['nsteps']},"
                        f"{p['TX']},{p['q']},{p['bh']},{p['CB']},0,{p['cpc']},{p['G']},0,0,0,0,0}}")
        else:
            rows.append(f"{{2,0,0,0,0,0,0,0,0,0,{p['TX']},0,{p['bh']},{p['CB']},{p['cbs']},{p['cpc']},{p['G']},{p['TG']},{p['nobj']},"
                        f"{p['m']},{1 if p['res'] else 0},{1 if p['relu'] else 0}}}")
    table = ",\n".join(rows)
    defs = dict(TILE=TILE, RS=RS, ROWB=ROWB, OUTB=OUTB, WREC=WREC, WTAIL=WTAIL, CREC=CREC, SREC=SREC, NWOBJ=NWOBJ)
    head = "".join(f"#define {k} {v}\n" for k, v in defs.items())
    return "#include <stdint.h>\n#include <aie_api/aie.hpp>\n" + head + r"""
#define RS8 (RS * 8)
#define COLD __attribute__((noinline, minsize))
#define HOT __attribute__((noinline))
struct Ph { int16_t kind, K, mode, kb, nb, pxo, win, hin, nlines, nsteps, TX, q, bh, CB, cbs, cpc, G, TG, nobj, m, res, relu; };
static const Ph PH[] = {
""" + table + r"""
};
using VF = aie::vector<float, 32>;
using VB = aie::vector<bfloat16, 32>;
using AF = aie::accum<accfloat, 32>;
using MM = aie::mmul<8, 8, 8, int8, int8>;

alignas(64) static int8_t ring[3 * 16 * RS8];          // conv: row windows; 18432 B
alignas(64) static int8_t wts[WREC];                   // conv: this core's weights; norm: residual rows [bh][4 planes][320]
#define accbuf ((int32_t *)(ring + 16384))              // final layer accumulators (its ring needs 13824 B)
alignas(64) static int16_t m0v[2][32];
alignas(64) static int32_t stS[2][32];
alignas(64) static float stQ[2][32];
alignas(64) static bfloat16 sch[4][32], scl[4][32], rsh[4][32], rsl[4][32];
alignas(64) static float shv[4][32];
alignas(64) static float ncon[8][32];                  // unit, gamma, beta, inv_s_out, s_res, m0, 1/N, extra shift
static int ph = -1, cn, aj, li, txi, iq, yq, tg, rrow, slot, curlo, prevlo, rr, ya, rshift, g0, o_r, off_r;

static inline int refl(int i, int n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; return i; }
static inline int lo2(int u, int h) { return refl(u, 2 * h) >> 1; }
static inline void cp8(int8_t *d, const int8_t *s) { ((int32_t *)d)[0] = ((const int32_t *)s)[0]; ((int32_t *)d)[1] = ((const int32_t *)s)[1]; }
static inline void cp64(int8_t *d, const int8_t *s, int n) { for (int i = 0; i < n; i += 64) aie::store_v(d + i, aie::load_v<64>(s + i)); }

static inline void limbs(VF x, VB &hi, VB &lo) {
    AF a(x); hi = a.template to_vector<bfloat16>(); lo = aie::sub(a, AF(hi)).template to_vector<bfloat16>();
}
static VF COLD mul3(VF a, VF b) {                      // fp32 x fp32 to ~16 bits with three native bf16 macs
    VB ah, al, bh, bl; limbs(a, ah, al); limbs(b, bh, bl);
    AF acc = aie::mul(ah, bh); acc = aie::mac(acc, al, bh); acc = aie::mac(acc, ah, bl);
    return acc.template to_vector<float>();
}
static VF COLD vadd(VF a, VF b) { return aie::add(AF(a), b).template to_vector<float>(); }
static VF COLD vsub(VF a, VF b) { return aie::sub(AF(a), b).template to_vector<float>(); }
static VF COLD rsqrt(VF x) {
    aie::vector<int32, 32> i = x.template cast_to<int32>();
    i = aie::sub(aie::broadcast<int32, 32>(0x5f3759df), aie::downshift(i, 1));
    VF y = i.template cast_to<float>();
    const VF c15 = aie::broadcast<float, 32>(1.5f), c05 = aie::broadcast<float, 32>(0.5f);
    for (int it = 0; it < 3; it++) { VF t = mul3(mul3(x, y), y); t = vsub(c15, mul3(c05, t)); y = mul3(y, t); }
    return y;
}

static inline int ringrow(const Ph &P) { return P.mode == 1 ? P.kb * 2 * RS8 : P.kb * RS8; }

// ---- copy the row window `src` (kb planes of pxo px) of tile column tx into the next ring slot
static void COLD do_row(const Ph &P, const int8_t *src, int tx) {
    const int pd = P.K / 2, win = P.win, rb = ringrow(P);
    int8_t *dst = ring + slot * rb;
    const int slots = P.mode == 2 ? 2 : P.K;
    slot = slot + 1 == slots ? 0 : slot + 1;
    if (P.mode == 0) {
        const int n = TILE + 2 * pd, xs = tx * TILE - pd;
        for (int k = 0; k < P.kb; k++) {
            const int8_t *s = src + k * P.pxo * 8; int8_t *q = dst + k * RS8;
            int j = 0;
            for (; j + 8 <= n; j += 8) aie::store_v(q + j * 8, aie::load_unaligned_v<64>(s + j * 8, 8));
            for (; j < n; j++) cp8(q + j * 8, s + j * 8);
            if (xs < 0) for (int jj = 0; jj < pd; jj++) cp8(q + jj * 8, s + (refl(xs + jj, win) - xs) * 8);
            if (xs + n > win) for (int jj = n - pd; jj < n; jj++) cp8(q + jj * 8, s + (refl(xs + jj, win) - xs) * 8);
        }
    } else if (P.mode == 1) {
        const int xs = 2 * tx * TILE - 1;
        for (int k = 0; k < P.kb; k++) {
            const int8_t *s = src + k * P.pxo * 8; int8_t *E = dst + k * 2 * RS8, *O = E + RS8;
            for (int j = 0; j < TILE; j++) cp8(E + j * 8, s + (2 * (tx * TILE + j) - xs) * 8);
            for (int j = 0; j <= TILE; j++) cp8(O + j * 8, s + (refl(2 * (tx * TILE - 1 + j) + 1, win) - xs) * 8);
        }
    } else {
        const int xs = tx * TILE / 2 - 1;
        for (int k = 0; k < P.kb; k++) {
            const int8_t *s = src + k * P.pxo * 8; int8_t *q = dst + k * RS8;
            for (int j = 0; j < TILE + 2; j++) cp8(q + j * 8, s + (lo2(tx * TILE - 1 + j, win) - xs) * 8);
        }
    }
}

// ---- the mmul loops: one output tile-row; TWO: 2 blocks of 8 output channels
template <bool TWO>
static void HOT conv_rows(const Ph &P, const int8_t *rows[9], int16_t *o16, bool fin) {
    const int K = P.K, kb = P.kb, KT = P.mode == 1 ? 3 : K;
    const int kstride = P.mode == 1 ? 2 * RS8 : RS8;
    constexpr int BS = TWO ? 128 : 64;
    for (int m = 0; m < TILE / 8; m++) {
        MM c0, c1;
        const int8_t *bp = wts;
        for (int dy = 0; dy < K; dy++) {
            const int8_t *rp = rows[dy] + m * 64;
            if (P.mode == 1) {
                for (int k = 0; k < kb; k++) {
                    const int8_t *E = rp + k * kstride, *O = E + RS8;
                    aie::vector<int8, 64> a0 = aie::load_v<64>(O), a1 = aie::load_v<64>(E);
                    aie::vector<int8, 64> a2 = aie::shuffle_down_fill(a0, aie::load_v<64>(O + 64), 8);
                    c0.mac(a0, aie::load_v<64>(bp)); if (TWO) c1.mac(a0, aie::load_v<64>(bp + 64));
                    c0.mac(a1, aie::load_v<64>(bp + BS)); if (TWO) c1.mac(a1, aie::load_v<64>(bp + BS + 64));
                    c0.mac(a2, aie::load_v<64>(bp + 2 * BS)); if (TWO) c1.mac(a2, aie::load_v<64>(bp + 2 * BS + 64)); bp += 3 * BS;
                }
            } else {
                for (int k = 0; k < kb; k++) {
                    aie::vector<int8, 64> v0 = aie::load_v<64>(rp + k * kstride), v1 = aie::load_v<64>(rp + k * kstride + 64);
                    for (int dx = 0; dx < KT; dx++) {
                        aie::vector<int8, 64> a = dx == 8 ? v1 : aie::shuffle_down_fill(v0, v1, 8 * dx);
                        c0.mac(a, aie::load_v<64>(bp)); if (TWO) c1.mac(a, aie::load_v<64>(bp + 64)); bp += BS;
                    }
                }
            }
        }
        if (fin) aie::store_v(accbuf + m * 64, c0.template to_vector<int32>());
        else {
            aie::store_v(o16 + m * 64, c0.template to_vector<int16>(rshift));
            if (TWO) aie::store_v(o16 + TILE * 8 + m * 64, c1.template to_vector<int16>(rshift));
        }
    }
}

// ---- running per-lane statistics of a raw tile-row: sums of (v - m0) exact in int32, squares in fp32
static void COLD stats_rows(const int16_t *o16, int nb) {
    for (int n = 0; n < nb; n++) {
        aie::vector<int16, 32> mz = aie::load_v<32>(m0v[n]);
        aie::accum<acc32, 32> aS(aie::load_v<32>(stS[n])); AF aQ(aie::load_v<32>(stQ[n]));
        for (int i = 0; i < TILE / 4; i++) {
            aie::vector<int16, 32> d = aie::sub(aie::load_v<32>(o16 + n * TILE * 8 + i * 32), mz);
            aS = aie::add(aS, d);
            aQ = aie::add(aQ, aie::to_float<float>(aie::mul(d, d).template to_vector<int32>()));
        }
        aie::store_v(stS[n], aS.template to_vector<int32>()); aie::store_v(stQ[n], aQ.template to_vector<float>());
    }
}

static void COLD stats_out(float *o) {                 // [16 sums][16 sums of squares], per channel from the 4 px lanes
    for (int n = 0; n < 2; n++) {
        aie::vector<int32, 32> s = aie::load_v<32>(stS[n]); VF q = aie::load_v<32>(stQ[n]);
        aie::vector<float, 8> fs = aie::zeros<float, 8>(), fq = aie::zeros<float, 8>();
        for (int g = 0; g < 4; g++) {
            fs = aie::add(aie::accum<accfloat, 8>(fs), aie::to_float<float>(s.template extract<8>(g))).template to_vector<float>();
            fq = aie::add(aie::accum<accfloat, 8>(fq), q.template extract<8>(g)).template to_vector<float>();
        }
        aie::store_v(o + n * 8, fs); aie::store_v(o + 16 + n * 8, fq);
    }
}

// ---- final layer: int32 accumulators -> RGBA bytes of a tile-row (channel 3 is a constant 255 from the bias)
static void COLD final_rows(uint8_t *out) {
    VF vs = aie::load_v<32>((const float *)(wts + WTAIL)), vb = aie::load_v<32>((const float *)(wts + WTAIL + 128));
    VB sh, sl; limbs(vs, sh, sl);
    for (int i = 0; i < TILE / 4; i++) {
        VF x = aie::to_float<float>(aie::load_v<32>(accbuf + i * 32));
        VB xh, xl; limbs(x, xh, xl);
        AF acc(vb); acc = aie::mac(acc, xh, sh); acc = aie::mac(acc, xl, sh); acc = aie::mac(acc, xh, sl);
        aie::vector<uint8, 32> q = aie::to_fixed<uint8>(acc.template to_vector<float>());     // [4 px][8 ch], saturating
        aie::vector<int32, 8> w = q.template cast_to<int32>();
        aie::store_v((int32_t *)out + i * 4, aie::filter_even(w, 1));                          // ch 0..3 of each px
    }
}

// ---- one line of a tile column: bring its row into the ring, compute the output row it completes
static void COLD conv_line(const Ph &P, const int8_t *src, int8_t *out, int tx) {
    if (li == 0) { slot = 0; curlo = prevlo = -1; }
    int y = -1;
    if (P.mode == 0) { do_row(P, src, tx); if (li >= P.K - 1) y = ya + li - (P.K - 1); }
    else if (P.mode == 1) { do_row(P, src, tx); if (li >= 2 && !(li & 1)) y = ya + ((li - 2) >> 1); }
    else {
        int need = li == 0 ? lo2(ya - 1, P.hin) : lo2(ya + li, P.hin);
        if (need != curlo) { do_row(P, src, tx); prevlo = curlo; curlo = need; }
        if (li >= 1) y = ya + li - 1;
    }
    if (y < 0) return;
    const int rb = ringrow(P);
    const int8_t *rows[9];
    if (P.mode == 2) {
        const int8_t *cur = ring + (slot == 0 ? 1 : 0) * rb, *prev = ring + slot * rb;
        for (int t = 0; t < 3; t++) rows[t] = lo2(y - 1 + t, P.hin) == curlo ? cur : prev;
    } else {
        for (int t = 0; t < P.K; t++) { int sl = slot + t; if (sl >= P.K) sl -= P.K; rows[t] = ring + sl * rb; }
    }
    if (P.kind == 1) { conv_rows<false>(P, rows, nullptr, true); final_rows((uint8_t *)out); return; }
    if (P.nb == 2) conv_rows<true>(P, rows, (int16_t *)out, false); else conv_rows<false>(P, rows, (int16_t *)out, false);
    stats_rows((const int16_t *)out, P.nb);
}

// ---- norm: reduce the 32 conv cores' statistics into my channels' scale/shift (bf16 limb lane vectors)
static void COLD nstat_apply(const Ph &P, const int8_t *stat) {
    // records: [band][group] of [16 sums][16 squares] fp32; my channels are groups g0 .. g0+ngr-1 (16 or 8 channels each)
    const int B = P.G == 8 ? 4 : 8, ngr = P.cbs == 4 ? 2 : 1;
    aie::vector<float, 16> s16[2], q16[2];
    for (int gi = 0; gi < ngr; gi++) {
        VF acc = aie::zeros<float, 32>();
        for (int b = 0; b < B; b++) acc = vadd(acc, aie::load_unaligned_v<32>((const float *)(stat + (b * P.G + g0 + gi) * SREC), 4));
        s16[gi] = acc.template extract<16>(0); q16[gi] = acc.template extract<16>(1);
    }
    if (ngr == 1) { s16[1] = s16[0]; q16[1] = q16[0]; }
    VF sum = aie::concat(s16[0], s16[1]), sq = aie::concat(q16[0], q16[1]);
    VF unit = aie::load_v<32>(ncon[0]), gam = aie::load_v<32>(ncon[1]), bet = aie::load_v<32>(ncon[2]), inv = aie::load_v<32>(ncon[3]);
    VF sres = aie::load_v<32>(ncon[4]), m0 = aie::load_v<32>(ncon[5]), invN = aie::load_v<32>(ncon[6]), extra = aie::load_v<32>(ncon[7]);
    VF mean_c = mul3(sum, invN);
    VF var16 = aie::max(vsub(mul3(sq, invN), mul3(mean_c, mean_c)), aie::zeros<float, 32>());
    VF a = mul3(gam, rsqrt(vadd(mul3(var16, mul3(unit, unit)), aie::broadcast<float, 32>(1e-5f))));
    VF ua = mul3(unit, a);
    VF sc = mul3(ua, inv), sh = vadd(mul3(vsub(bet, mul3(vadd(mean_c, m0), ua)), inv), extra), sr = mul3(sres, inv);
    alignas(64) float tsc[32], tsh[32], tsr[32];
    aie::store_v(tsc, sc); aie::store_v(tsh, sh); aie::store_v(tsr, sr);
    for (int k = 0; k < P.cbs; k++) {                    // lane l of block k = channel k*8 + (l & 7)
        VF vs = aie::broadcast<float, 32>(0.f), vh = vs, vr = vs;
        aie::vector<float, 8> e = aie::load_v<8>(tsc + k * 8), f = aie::load_v<8>(tsh + k * 8), g = aie::load_v<8>(tsr + k * 8);
        vs = aie::concat(e, e, e, e); vh = aie::concat(f, f, f, f); vr = aie::concat(g, g, g, g);
        VB hi, lo;
        limbs(vs, hi, lo); aie::store_v(sch[k], hi); aie::store_v(scl[k], lo);
        limbs(vr, hi, lo); aie::store_v(rsh[k], hi); aie::store_v(rsl[k], lo);
        aie::store_v(shv[k], vh);
    }
}

// ---- norm: one tile of my channels: raw (in the rows object) -> int8 into the out record
static void COLD do_norm(const Ph &P, const int8_t *raw, int8_t *out, int t) {
    for (int k = 0; k < P.cbs; k++) {
        const int16_t *rw = (const int16_t *)(raw + off_r + (P.cpc == 16 ? k * 640 : k * OUTB));
        VB hs = aie::load_v<32>(sch[k]), ls = aie::load_v<32>(scl[k]), hr = aie::load_v<32>(rsh[k]), lr = aie::load_v<32>(rsl[k]);
        VF shb = aie::load_v<32>(shv[k]);
        const int8_t *rp = wts + yq * 1280 + k * 320;
        int8_t *op = out + k * P.TG * 320 + t * 320;
        for (int p = 0; p < TILE / 4; p++) {
            VF x = aie::to_float<float>(aie::load_v<32>(rw + p * 32));
            VB xh, xl; limbs(x, xh, xl);
            AF acc(shb);
            acc = aie::mac(acc, xh, hs); acc = aie::mac(acc, xl, hs); acc = aie::mac(acc, xh, ls);
            if (P.res) {
                VF rf = aie::to_float<float>(aie::load_v<32>(rp + p * 32));
                VB rh, rl; limbs(rf, rh, rl);
                acc = aie::mac(acc, rh, hr); acc = aie::mac(acc, rl, hr); acc = aie::mac(acc, rh, lr);
            }
            VF y = acc.template to_vector<float>();
            if (P.relu) y = aie::max(y, aie::broadcast<float, 32>(-128.f));
            aie::store_v(op + p * 32, aie::to_fixed<int8>(y));
        }
    }
}

static void COLD begin(int arg, int cid) {
    aie::set_rounding(aie::rounding_mode::conv_even); aie::set_saturation(aie::saturation_mode::saturate);
    ph = arg; const Ph &P = PH[ph];
    rr = cid & 3; const int c = cid >> 2;
    ya = (P.CB == 2 ? c >> 1 : c) * P.bh;
    cn = aj = li = txi = iq = yq = tg = rrow = slot = 0; curlo = prevlo = -1;
    if (P.kind == 2) { g0 = P.cpc == 16 ? (rr * P.cbs) >> 1 : rr * P.cbs; o_r = g0 >> 2; off_r = (g0 & 3) * OUTB; }
}

// ---- R calls: weights pieces (conv), consts / stats / residual rows (norm)
extern "C" void style_r(int8_t *rows, int32_t arg, int32_t cid) {
    if (arg != ph) begin(arg, cid);
    const Ph &P = PH[ph];
    if (P.kind != 2) {                                     // weights: my record is bytes [rr*WREC, (rr+1)*WREC) of the stream
        const int lo = cn * ROWB, hi = lo + ROWB, a = rr * WREC, b = a + WREC;
        const int s = lo > a ? lo : a, e = hi < b ? hi : b;
        if (s < e) cp64(wts + (s - a), rows + (s - lo), e - s);
        if (e == b) {
            rshift = *(const int32_t *)(wts + WTAIL);
            const int16_t *m0 = (const int16_t *)(wts + WTAIL + 32);
            for (int n = 0; n < 2; n++) for (int l = 0; l < 32; l++) m0v[n][l] = m0[n * 8 + (l & 7)];
            for (int n = 0; n < 2; n++) { aie::store_v(stS[n], aie::zeros<int32, 32>()); aie::store_v(stQ[n], aie::zeros<float, 32>()); }
        }
    } else if (cn == 0) {
        const float *f = (const float *)(rows + rr * CREC);
        for (int c = 0; c < 8; c++) aie::store_v(ncon[c], aie::load_v<32>(f + 32 * c));
    } else if (cn == 1) {
        nstat_apply(P, rows);
    } else {                                               // residual row: planes 4rr..4rr+3 of [16 planes][352 B]
        for (int k = 0; k < 4; k++) {                      // 352-byte planes: only 32-byte aligned
            const int8_t *sp = rows + (rr * 4 + k) * 352; int8_t *dp = wts + rrow * 1280 + k * 320;
            for (int i = 0; i < 320; i += 64) aie::store_v(dp + i, aie::load_unaligned_v<64>(sp + i, 32));
        }
        rrow++;
    }
    cn++;
}

// ---- A / Q calls
extern "C" void style_ro(int8_t *rows, int8_t *out, int32_t arg, int32_t cid) {
    if (arg != ph) begin(arg, cid);
    const Ph &P = PH[ph];
    if (P.kind == 0) {
        if (aj == P.nsteps - 1) stats_out((float *)out);
        else {
            conv_line(P, rows, out, txi);
            if (++li == P.nlines) { li = 0; txi++; }
        }
        aj++;
    } else if (P.kind == 1) {
        if (iq == rr) conv_line(P, rows, out, rr * P.q + txi);
        if (++iq == 4) { iq = 0; if (++li == P.nlines) { li = 0; txi++; } }
    } else {
        const int t = P.nobj == 2 ? iq >> 1 : iq, o = P.nobj == 2 ? iq & 1 : 0;
        if (o == o_r) do_norm(P, rows, out, t);
        if (++iq == P.m) { iq = 0; if (++yq == P.bh) { yq = 0; tg++; rrow = 0; } }
    }
}
"""


# ------------------------------------------------------------------------------------------------- the DMA streams
class Fill:
    """One DMA transfer: a strided walk (bytes, outermost first, innermost contiguous) over a region."""

    def __init__(self, region, offset, sizes, strides=None):
        strides = strides if strides is not None else [1]
        assert len(sizes) == len(strides) <= 4 and strides[-1] == 1, (sizes, strides)
        assert offset % 4 == 0 and all(sz > 0 for sz in sizes) and all(st % 4 == 0 for st in strides[:-1]) and sizes[-1] % 4 == 0, (region, offset, sizes, strides)
        self.region, self.offset, self.sizes, self.strides = region, offset, list(sizes), list(strides)

    @property
    def nbytes(self):
        return int(np.prod(self.sizes))

    def chunks(self):
        out = []; dims = self.sizes[:-1]; strs = self.strides[:-1]; inner = self.sizes[-1]
        for idx in (np.ndindex(*dims) if dims else [()]):
            out.append((self.offset + sum(i * st for i, st in zip(idx, strs)), inner))
        return out

    def pieces(self):
        """(word offset, sizes, strides) walks of at most 4 dims whose innermost size fits a shim BD (<= 1023 words,
        unless the walk is linear); a walk that needs more dims is split on its outermost dim."""
        sizes = [sz for sz in self.sizes[:-1]] + [self.sizes[-1] // 4]
        strides = [st // 4 for st in self.strides[:-1]] + [1]
        keep = [i for i in range(len(sizes) - 1) if sizes[i] > 1] + [len(sizes) - 1]
        sizes = [sizes[i] for i in keep]; strides = [strides[i] for i in keep]
        if len(sizes) > 1 and sizes[-1] > 1023:
            w = sizes[-1]; b = max(d for d in range(1, 1024) if w % d == 0)
            sizes = sizes[:-1] + [w // b, b]; strides = strides[:-1] + [b, 1]
        if len(sizes) <= 4:
            return [(self.offset // 4, sizes, strides)]
        out = []
        for i in range(sizes[0]):
            sub = Fill(self.region, self.offset + i * strides[0] * 4, [sz for sz in sizes[1:-1]] + [sizes[-1] * 4], [st * 4 for st in strides[1:-1]] + [1])
            out += sub.pieces()
        return out

    def taps(self, total_words, base_bytes):
        from aie.helpers.taplib import TensorAccessPattern
        out = []
        for off, sizes, strides in self.pieces():
            sz = [1] * (4 - len(sizes)) + sizes; st = [0] * (4 - len(strides)) + strides
            st = [s if z > 1 else 0 for z, s in zip(sz, st)]; st[-1] = 1
            out.append(TensorAccessPattern((total_words,), base_bytes // 4 + off, sz, st))
        return out


def compress_entries(offsets, kb, ps, inner, region):
    """Row windows (byte offsets, each [kb planes stride ps][inner bytes]) in stream order -> Fills of [n][m][kb][inner]."""
    fills = []; i = 0; N = len(offsets)
    while i < N:
        best = None
        for m in (4, 2, 1):
            if i + m > N: continue
            grp = offsets[i:i + m]; Bs = grp[1] - grp[0] if m > 1 else 0
            if m > 1 and (Bs <= 0 or any(grp[j + 1] - grp[j] != Bs for j in range(m - 1))): continue
            n = 1; A = None
            while i + (n + 1) * m <= N:
                nxt = offsets[i + n * m:i + (n + 1) * m]
                if m > 1 and any(nxt[j + 1] - nxt[j] != Bs for j in range(m - 1)): break
                a = nxt[0] - offsets[i + (n - 1) * m]
                if a <= 0 or (A is not None and a != A): break
                A = a; n += 1
            if best is None or n * m > best[0] * best[1]: best = (n, m, A or 4, Bs or 4)
        n, m, A, Bs = best
        sizes = [n, m, kb, inner]; strides = [A, Bs, ps, 1]
        while len(sizes) > 1 and sizes[0] == 1: sizes.pop(0); strides.pop(0)
        if kb == 1 and len(sizes) >= 2: sizes.pop(-2); strides.pop(-2)
        fills.append(Fill(region, offsets[i], sizes, strides)); i += n * m
    return fills


class Design:
    """Per phase and column: the rows fills (in stream order), the out drains, and the core segment table
    [(kind 0 R / 1 A / 2 Q, count, m, phase)] shared by the column's four cores."""

    def __init__(self, plan, nphases=None):
        self.plan = plan; self.cols = []
        for pi, p in enumerate(plan.phases[:nphases]):
            self.cols.append([(self.conv_col if p["kind"] == "conv" else self.norm_col)(pi, p, c) for c in range(COLS)])
        self.nphases = len(self.cols)

    def segments(self, c):
        segs = [sg for ph in self.cols for sg in ph[c]["segs"]]
        out = []
        for sg in segs:                                       # merge neighbours of the same kind, m and phase
            if out and out[-1][0] == sg[0] and out[-1][2] == sg[2] and out[-1][3] == sg[3]: out[-1] = (sg[0], out[-1][1] + sg[1], sg[2], sg[3])
            else: out.append(sg)
        return out

    def conv_col(self, pi, p, c):
        plan = self.plan; H = Plan.HEAD
        C, h, w = plan.tensors[p["src"]]; PS = h * w * 8; RB = w * 8
        reg = plan.region(p["src"]); y0, y1 = plan.band(p, c)
        offs = []
        if p["final"]:
            for qi in range(p["q"]):
                for li in range(p["nlines"]):
                    row = plan.line_row(p, y0, li)
                    for r in range(ROWS): offs.append(H + row * RB + plan.x_src(p, r * p["q"] + qi) * 8)
        else:
            for tx in range(p["TX"]):
                for li in range(p["nlines"]): offs.append(H + plan.line_row(p, y0, li) * RB + plan.x_src(p, tx) * 8)
        rows = compress_entries(offs, p["kb"], PS, p["pxo"] * 8, reg)
        assert [ch for f in rows for ch in f.chunks()] == [(o + k * PS, p["pxo"] * 8) for o in offs for k in range(p["kb"])], (p["name"], c)
        wbase = plan.wrec[p["li"], "w"] + (0 if p["final"] else (c % p["CB"]) * ROWS * WREC)
        fills = [Fill("W", wbase, [ROWS * WREC])] + rows
        if p["final"]:
            drains = [Fill("RGB", c * p["nsteps"] * ROWS * OUTB, [p["nsteps"] * ROWS * OUTB])]
            segs = [(0, NWOBJ, 0, pi), (2, p["nsteps"], ROWS, pi)]
        else:
            fills.append(Fill(reg, 0, [ROWB]))                # the stats step's (unused) rows object
            b = c // p["CB"]; G = p["G"]
            drains = [Fill("RAW", (b * p["nsteps"] * G + (c % p["CB"]) * ROWS) * OUTB, [p["nsteps"], ROWS, OUTB], [G * OUTB, OUTB, 1])]
            segs = [(0, NWOBJ, 0, pi), (1, p["nsteps"], 0, pi)]
        nobj = sum(f.nbytes for f in fills) // ROWB
        assert nobj * ROWB == sum(f.nbytes for f in fills) and nobj == sum(n * (m if k == 2 else 1) for k, n, m, _ in segs), (p["name"], c)
        assert sum(f.nbytes for f in drains) == ROWS * OUTB * sum(n for k, n, m, _ in segs if k != 0)
        return dict(fills=fills, drains=drains, segs=segs)

    def norm_col(self, pi, p, c):
        plan = self.plan; H = Plan.HEAD; cv = p["conv"]; G = p["G"]; ns = cv["nsteps"]
        b = c // p["CB"]; y0, y1 = plan.band(p, c); bh = p["bh"]
        C, h, w = p["cout"], p["hout"], p["wout"]; PSd = h * w * 8; RBd = w * 8
        fills = [Fill("W", plan.wrec[p["li"], "n"], [ROWB]),
                 Fill("RAW", (ns - 1) * G * OUTB, [cv["B"], G, SREC], [ns * G * OUTB, OUTB, 1])]
        drains = []; segs = [(0, 2, 0, pi)]
        ystride = 2 if cv["mode"] == "s2" else 1
        for tiles in plan.norm_tiles(p, c):
            t0 = tiles[0]
            if p["res"]:
                Cr, hr, wr = plan.tensors[p["res"]]
                fills.append(Fill(plan.region(p["res"]), H + y0 * wr * 8 + t0 * TILE * 8, [bh, Cr // 8, 352], [wr * 8, hr * wr * 8, 1]))
                segs.append((0, bh, 0, pi))
            base = (b * ns * G + plan.step_of(cv, t0, 0) * G) * OUTB
            fills.append(Fill("RAW", base, [bh, p["TG"], p["nobj"], ROWB], [ystride * G * OUTB, p["CB"] * cv["nlines"] * G * OUTB, ROWS * OUTB, 1]))
            segs.append((2, bh, p["m"], pi))
            drains.append(Fill(plan.region(p["dst"]), H + y0 * RBd + t0 * TILE * 8, [bh, ROWS, p["cbs"], p["TG"] * 320], [RBd, p["cbs"] * PSd, PSd, 1]))
        nobj = sum(f.nbytes for f in fills) // ROWB
        assert nobj * ROWB == sum(f.nbytes for f in fills) and nobj == sum(n * (m if k == 2 else 1) for k, n, m, _ in segs), (p["name"], c)
        assert sum(f.nbytes for f in drains) == ROWS * OUTB * sum(n for k, n, m, _ in segs if k != 0)
        return dict(fills=fills, drains=drains, segs=segs)


# ------------------------------------------------------------------------------------------------- weights blob
def pack_blob(plan, Q):
    """The per-style runtime buffer: per conv phase G weight records (int8 B tiles + tail), per norm phase 4 const records."""
    blob = np.zeros(plan.weights_bytes, np.int8)
    L = Q.net.layers
    for p in plan.phases:
        li = p["li"]
        if p["kind"] == "conv":
            qw = Q.qw[li]; cpc = p["cpc"]; nb = p["nb"]
            recs = []
            for g in range(p["G"]):
                rec = np.zeros(WREC, np.int8); idx = 0
                for dy in range(p["K"]):
                    for kb in range(p["kb"]):
                        for dx in range(p["K"]):
                            for n in range(nb):
                                o0 = g * cpc + n * 8
                                rec[idx * 64:(idx + 1) * 64] = qw[o0:o0 + 8, kb * 8:kb * 8 + 8, dy, dx].T.reshape(-1); idx += 1
                assert idx * 64 <= WTAIL, (p["name"], idx * 64)
                tail = rec[WTAIL:]
                if p["final"]:
                    sc = np.zeros(8, np.float32); sc[:3] = Q.s_w[li][:3]
                    bs = np.zeros(8, np.float32); bs[:3] = Q.fbias[:3]; bs[3] = 255.0
                    tail[0:128] = np.tile(sc, 4).view(np.int8); tail[128:256] = np.tile(bs, 4).view(np.int8)
                else:
                    tail[0:4] = np.array([Q.shift[li]], np.int32).view(np.int8)
                    m0 = np.zeros(16, np.int16); m0[:cpc] = Q.m0[li][g * cpc:(g + 1) * cpc]
                    tail[32:64] = m0.view(np.int8)
                recs.append(rec)
            if p["final"]: recs = recs * ROWS
            at = plan.wrec[li, "w"]
            blob[at:at + len(recs) * WREC] = np.concatenate(recs)
        else:
            unit, gam, bet, inv, sres = Q.norm_consts(li)
            m0 = Q.m0[li].astype(np.float64); N = p["hout"] * p["wout"]
            rec = np.zeros((ROWS, CREC), np.int8)
            for r in range(ROWS):
                ch = np.arange(r * p["cbs"] * 8, (r + 1) * p["cbs"] * 8); n = len(ch)
                con = np.zeros((8, 32), np.float32); con[0, :] = 1.0
                con[0, :n] = unit[ch]; con[1, :n] = gam[ch]; con[2, :n] = bet[ch]; con[3, :n] = inv[ch]
                con[4, :n] = sres[ch]; con[5, :n] = m0[ch]; con[6, :] = 1.0 / N
                extra = np.zeros(n)
                if p["relu"]: extra -= 128.0
                if p["res"] and p["res_uns"]: extra += 128.0 * sres[ch] * inv[ch]
                con[7, :n] = extra
                rec[r, :con.nbytes] = con.reshape(-1).view(np.int8)
            at = plan.wrec[li, "n"]; blob[at:at + ROWB] = rec.reshape(-1)
    return blob


# ------------------------------------------------------------------------------------------------- the IRON program
BD_CAP = 6                     # rows fills in flight per column (shim BDs: 16 per tile, drains take the rest)


def build_program(spec_json, nphases):
    import aie.iron as iron
    from aie.dialects import arith
    from aie.extras import types as T
    from aie.ir import IndexType
    from aie.iron import Buffer, ExternalFunction, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import else_, if_, range_
    from aie.iron.device import Tile
    spec = json.loads(spec_json); plan = Plan(spec["layers"], spec["W"], spec["H"]); D = Design(plan, nphases)
    rows_ty = np.ndarray[(ROWB,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUTB,), np.dtype[np.int8]]; out4_ty = np.ndarray[(ROWS * OUTB,), np.dtype[np.int8]]
    Ww = plan.weights_bytes // 4; Sw = plan.scratch_bytes // 4
    wts_ty = np.ndarray[(Ww,), np.dtype[np.int32]]; scr_ty = np.ndarray[(Sw,), np.dtype[np.int32]]
    src = kernel_source(plan)
    k_r = ExternalFunction("style_r", source_string=src, arg_types=[rows_ty, np.dtype(np.int32), np.dtype(np.int32)])
    k_ro = ExternalFunction("style_ro", source_string="#include <stdint.h>\nextern \"C\" void style_ro(int8_t *, int8_t *, int32_t, int32_t);\n",
                            arg_types=[rows_ty, out_ty, np.dtype(np.int32), np.dtype(np.int32)])

    def body(rf, of, kr, kro, tb, cid):
        idx = IndexType.get(); i32 = T.i32()
        for s in range_(NSEG):
            w = tb[2 * s]; n = arith.index_cast(idx, tb[2 * s + 1])                     # w = kind | m << 4 | phase << 12
            kind = arith.andi(w, arith.constant(i32, 15))
            arg = arith.shrui(w, arith.constant(i32, 12))
            mval = arith.andi(arith.shrui(w, arith.constant(i32, 4)), arith.constant(i32, 255))
            with if_(arith.cmpi(arith.CmpIPredicate.eq, kind, arith.constant(T.i32(), 0))) as b0:
                for _ in range_(0, n, 1):
                    r = rf.acquire(1); kr(r, arg, cid); rf.release(1)
            with else_(b0):
                with if_(arith.cmpi(arith.CmpIPredicate.eq, kind, arith.constant(T.i32(), 1))) as b1:
                    for _ in range_(0, n, 1):
                        r = rf.acquire(1); o = of.acquire(1); kro(r, o, arg, cid); of.release(1); rf.release(1)
                with else_(b1):
                    m = arith.index_cast(idx, mval)
                    for _ in range_(0, n, 1):
                        o = of.acquire(1)
                        for _ in range_(0, m, 1):
                            r = rf.acquire(1); kro(r, o, arg, cid); rf.release(1)
                        of.release(1)

    rin, oj, workers = [], [], []
    segtabs = [D.segments(c) for c in range(COLS)]
    NSEG = len(segtabs[0]); assert all(len(t) == NSEG for t in segtabs)
    for c in range(COLS):
        ri = ObjectFifo(rows_ty, name=f"rin{c}", depth=2)
        rb = ri.cons().forward(tile=Tile(c, 1), depth=2, name=f"rbc{c}")
        o4 = ObjectFifo(out4_ty, name=f"out{c}", depth=2)
        os_ = o4.prod().join([r * OUTB for r in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS, obj_types=[out_ty] * ROWS, names=[f"o{c}_{r}" for r in range(ROWS)])
        rin.append(ri); oj.append(o4)
        tab = np.array([[k | (m << 4) | (ph << 12), n] for k, n, m, ph in segtabs[c]], np.int32).reshape(-1)
        for r in range(ROWS):
            tb = Buffer(initial_value=tab, name=f"seg{c}_{r}")
            workers.append(Worker(body, [rb.cons(depth=2), os_[r].prod(), k_r, k_ro, tb, c * ROWS + r], tile=Tile(c, 2 + r), dynamic_objfifo_lowering=True))

    def seq(wts, scratch, rp, oc):
        for cols in D.cols:
            pending = []
            for c in range(COLS):
                for f in cols[c]["drains"]:
                    for tap in f.taps(Sw, plan.buf[f.region][0]):
                        pending.append(oc[c].drain(scratch, tap, wait=True, managed=False))
            queues = [[(f, tap) for f in cols[c]["fills"] for tap in f.taps(Ww if f.region == "W" else Sw, 0 if f.region == "W" else plan.buf[f.region][0])]
                      for c in range(COLS)]
            inflight = [[] for _ in range(COLS)]
            while any(queues):
                for c in range(COLS):
                    if not queues[c]: continue
                    if len(inflight[c]) >= BD_CAP: inflight[c].pop(0).await_()
                    f, tap = queues[c].pop(0)
                    inflight[c].append(rp[c].fill(wts if f.region == "W" else scratch, tap, wait=True, managed=False))
            for c in range(COLS):
                for t in inflight[c]: t.await_()
            for t in pending: t.await_()

    rt = Runtime(seq, [wts_ty, scr_ty, [f.prod(tile=Tile(c, 0)) for c, f in enumerate(rin)], [f.cons(tile=Tile(c, 0)) for c, f in enumerate(oj)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _jit():
    import aie.iron as iron
    from aie.iron import CompileTime, In, InOut

    @iron.jit
    def style_npu(wts: In, scratch: InOut, *, spec: CompileTime[str], nphases: CompileTime[int], variant: CompileTime[str]):
        return build_program(spec, nphases)
    return style_npu


class StyleNPU:
    """Styles on the NPU: one program per picture size, one weights buffer per style."""

    def __init__(self, models, W=320, H=240, nphases=None, verbose=True):
        import aie.iron as iron
        self.nets = [Net(m) for m in models]
        spec = self.nets[0].spec()
        assert all(n.spec() == spec for n in self.nets), "all styles must share the architecture"
        self.plan = Plan(spec, W, H); self.design = Design(self.plan, nphases); self.nphases = self.design.nphases
        self.spec_json = json.dumps(dict(layers=spec, W=W, H=H))
        self.variant = "style-" + hashlib.md5((self.plan.signature() + kernel_source(self.plan) + str(self.nphases) + str(BD_CAP)).encode()).hexdigest()[:10]
        self.scratch = iron.zeros((self.plan.scratch_bytes // 4,), dtype=np.int32, device="npu")
        self.quants = [get_quant(n) for n in self.nets]
        self.blobs = []
        for q in self.quants:
            t = iron.zeros((self.plan.weights_bytes // 4,), dtype=np.int32, device="npu")
            with t.overwrite() as v: v[...] = pack_blob(self.plan, q).view(np.int32)
            self.blobs.append(t)
        self.fn = _jit(); self.W, self.H = W, H; self.cur = 0; self.ms = 0.0
        if verbose:
            nf = sum(len(col["fills"]) + len(col["drains"]) for ph in self.design.cols for col in ph)
            print(f"plan {W}x{H}: {self.nphases} phases, {nf} DMA transfers, scratch {self.plan.scratch_bytes / 1e6:.1f} MB, "
                  f"weights {self.plan.weights_bytes / 1e6:.2f} MB per style")

    def region(self, name):
        off, n = self.plan.buf[name]
        return self.scratch.numpy().view(np.int8)[off:off + n]

    def set_image(self, rgb):
        off = self.plan.buf["IMG"][0] + Plan.HEAD
        img8 = np.zeros((self.H, self.W, 8), np.int8); img8[:, :, :3] = (np.asarray(rgb, np.uint8) >> 1).astype(np.int8)
        with self.scratch.overwrite() as v: v.view(np.int8)[off:off + img8.size] = img8.reshape(-1)

    def run(self, rgb, style=None):
        if style is not None: self.cur = style
        self.set_image(rgb)
        t0 = time.perf_counter()
        self.fn(self.blobs[self.cur], self.scratch, spec=self.spec_json, nphases=self.nphases, variant=self.variant)
        self.ms = (time.perf_counter() - t0) * 1e3
        return self.read_rgb() if self.nphases == len(self.plan.phases) else None

    def read_rgb(self):
        p = self.plan.phases[-1]
        recs = self.region("RGB")[:COLS * p["nsteps"] * ROWS * OUTB].view(np.uint8).reshape(COLS, p["q"], p["nlines"], ROWS, OUTB)
        out = np.zeros((self.H, self.W, 3), np.uint8); K = p["K"]
        for c in range(COLS):
            y0, y1 = self.plan.band(p, c)
            for qi in range(p["q"]):
                for r in range(ROWS):
                    tx = r * p["q"] + qi
                    out[y0:y1, tx * TILE:(tx + 1) * TILE] = recs[c, qi, K - 1:, r, :TILE * 4].reshape(y1 - y0, TILE, 4)[:, :, :3]
        return out

    def tensor(self, name):
        """A planar int8 activation tensor from the scratch as (C, h, w)."""
        C, h, w = self.plan.tensors[name]
        raw = self.region(self.plan.region(name))[Plan.HEAD:Plan.HEAD + C * h * w]
        return np.transpose(raw.reshape(C // 8, h, w, 8), (0, 3, 1, 2)).reshape(C, h, w)

    def raw(self, p):
        """Conv phase p's int16 raw outputs (C, h, w) and per-channel (sum, sumsq) from the RAW region."""
        plan = self.plan; G = p["G"]; ns = p["nsteps"]; cpc = p["cpc"]
        recs = self.region("RAW")[:p["B"] * ns * G * OUTB].view(np.int16).reshape(p["B"], ns, G, OUTB // 2)
        out = np.zeros((p["cout"], p["hout"], p["wout"]), np.int16); st = np.zeros((p["cout"], 2))
        for c in range(COLS):
            b = c // p["CB"]; y0, y1 = plan.band(p, c)
            for r in range(ROWS):
                g = plan.conv_group(p, c, r)
                for tx in range(p["TX"]):
                    for li in range(p["nlines"]):
                        y = plan.line_out(p, li)
                        if y is None: continue
                        rec = recs[b, tx * p["nlines"] + li, g, :cpc * TILE].reshape(cpc // 8, TILE, 8)
                        out[g * cpc:(g + 1) * cpc, y0 + y, tx * TILE:(tx + 1) * TILE] = np.transpose(rec, (0, 2, 1)).reshape(cpc, TILE)
                f = recs[b, ns - 1, g].view(np.float32)
                st[g * cpc:(g + 1) * cpc, 0] += f[:cpc]; st[g * cpc:(g + 1) * cpc, 1] += f[16:16 + cpc]
        return out, st


# ------------------------------------------------------------------------------------------------- virtual webcam
def v4l2_devices():
    """[(path, driver, card, device_caps)] for every /dev/video* that answers VIDIOC_QUERYCAP, in numeric order."""
    import fcntl, glob, re, struct
    out = []
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        try: fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        except OSError: continue
        try:
            cap = bytearray(104); fcntl.ioctl(fd, VirtualCam.VIDIOC_QUERYCAP, cap)
            name = lambda a, b: bytes(cap[a:b]).split(b"\0")[0].decode(errors="replace")
            caps, dcaps = struct.unpack_from("II", cap, 84)
            out.append((path, name(0, 16), name(16, 48), dcaps or caps))
        except OSError: pass
        finally: os.close(fd)
    return out


def find_camera(exclude=None):
    """Index of the first real capture device, or None. Never a v4l2loopback node: after a reboot the loopback can be
    /dev/video0, and reading it back would feed the filter its own output."""
    for path, driver, card, caps in v4l2_devices():
        if driver != "v4l2 loopback" and caps & 0x1 and path != exclude:   # V4L2_CAP_VIDEO_CAPTURE
            return int(path.replace("/dev/video", ""))
    return None


class VirtualCam:
    """Frames out to a v4l2loopback device (load the module with exclusive_caps=1 so browsers see a camera), as
    I420 (YU12), the format Zoom, Jitsi, Chrome and Firefox take from v4l2loopback. Plain ioctl + write, no packages.
    Apps see the camera only while frames are being written. `watched()` reports whether an app is streaming from it
    (v4l2loopback's client-usage event, checked on 0.15.4: 1 on a reader's STREAMON, 0 on its STREAMOFF or close)."""
    VIDIOC_QUERYCAP, VIDIOC_S_FMT = 0x80685600, 0xC0D05605          # x86_64 values (checked against linux/videodev2.h)
    VIDIOC_SUBSCRIBE_EVENT, VIDIOC_DQEVENT = 0x4020565A, 0x80885659  # struct sizes 32 and 136 (checked with gcc)
    EVENT_CLIENT_USAGE = 0x08000000 + 0x08E00000 + 1                # V4L2_EVENT_PRIVATE_START + v4l2loopback's offset + 1
    YU12, BUF_OUTPUT, FIELD_NONE, CS_SMPTE170M, CAP_OUTPUT = 0x32315559, 2, 1, 1, 0x2

    @staticmethod
    def find():
        """The first v4l2loopback node, preferring one labelled "NPU"."""
        loops = [(p, card) for p, driver, card, _ in v4l2_devices() if driver == "v4l2 loopback"]
        if not loops:
            sys.exit("no v4l2loopback device: sudo modprobe v4l2loopback video_nr=10 card_label=\"NPU Style\" exclusive_caps=1")
        return next((p for p, card in loops if "NPU" in card), loops[0][0])

    def __init__(self, dev, w, h):
        import fcntl, struct
        assert w % 2 == 0 and h % 2 == 0
        if dev == "auto": dev = self.find()
        self.w, self.h = w, h
        self.fd = os.open(dev, os.O_RDWR)
        cap = bytearray(104); fcntl.ioctl(self.fd, self.VIDIOC_QUERYCAP, cap)
        driver = bytes(cap[:16]).split(b"\0")[0].decode(errors="replace")
        caps, dcaps = struct.unpack_from("II", cap, 84)
        if not ((dcaps or caps) & self.CAP_OUTPUT):
            os.close(self.fd)
            sys.exit(f"{dev} ({driver}) is not an output device: point --vcam at the v4l2loopback node, not the camera")
        fmt = bytearray(208)
        struct.pack_into("I", fmt, 0, self.BUF_OUTPUT)
        struct.pack_into("12I", fmt, 8, w, h, self.YU12, self.FIELD_NONE, w, w * h * 3 // 2, self.CS_SMPTE170M, 0, 0, 0, 0, 0)
        fcntl.ioctl(self.fd, self.VIDIOC_S_FMT, fmt)
        gw, gh, gfmt = struct.unpack_from("III", fmt, 8)
        if (gw, gh, gfmt) != (w, h, self.YU12):
            sys.exit(f"{dev} took {gw}x{gh} fourcc {gfmt:#x} instead of {w}x{h} YU12")
        self.name = f"{dev} ({driver})"; self.dev = dev
        self.usage = None                                         # None: the module has no client-usage event (assume watched)
        try:
            sub = bytearray(32); struct.pack_into("III", sub, 0, self.EVENT_CLIENT_USAGE, 0, 1)   # 1 = SEND_INITIAL
            fcntl.ioctl(self.fd, self.VIDIOC_SUBSCRIBE_EVENT, sub); self.usage = 0
        except OSError:
            pass

    def watched(self, timeout=0.0):
        """True while an app streams from the device. Waits up to `timeout` seconds for a change."""
        import fcntl, select, struct
        if self.usage is None:                                    # cannot tell: wait as promised, say yes
            if timeout: time.sleep(timeout)
            return True
        p = select.poll(); p.register(self.fd, select.POLLPRI); t = int(timeout * 1000)
        while p.poll(t):
            ev = bytearray(136)
            try: fcntl.ioctl(self.fd, self.VIDIOC_DQEVENT, ev)
            except OSError: break
            if struct.unpack_from("I", ev, 0)[0] == self.EVENT_CLIENT_USAGE: self.usage = struct.unpack_from("I", ev, 8)[0]
            t = 0
        return self.usage > 0

    def encode(self, rgb):
        """RGB (any size) -> the device's I420 frame bytes."""
        import cv2
        if rgb.shape[:2] != (self.h, self.w): rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YUV_I420).tobytes()

    def write(self, frame):
        """An RGB array, or bytes from encode() (a still frame converted once: no OpenCV work per write)."""
        os.write(self.fd, frame if isinstance(frame, bytes) else self.encode(frame))

    def close(self):
        os.close(self.fd)


# ------------------------------------------------------------------------------------------------- checks and CLI
def check(npu, img):
    """Compare the NPU's last computed phase with the emulator."""
    plan = npu.plan; p = plan.phases[npu.nphases - 1]; Q = npu.quants[0]; li = p["li"]
    tr = []; emu = Q.emulate(img, trace=tr, upto=li)
    npu.run(img)
    if p["kind"] == "conv" and not p["final"]:
        got, st = npu.raw(p); want = tr[li]["raw"].astype(np.int16)
        d = np.abs(got.astype(np.int32) - want)
        own = Q.conv_raw(li, npu.tensor(p["src"])).astype(np.int32)            # the same conv on the NPU's own input
        do = np.abs(got.astype(np.int32) - own)
        print(f"   vs the emulated conv of the NPU's own input: max|diff| {do.max()} ({np.count_nonzero(do)} values)")
        if do.max():
            bad = np.argwhere(do == do.max())[0]; ys, xs = np.nonzero(do.max(axis=0))
            print("   own-input worst (c,y,x)", bad, "got", got[tuple(bad)], "want", own[tuple(bad)], "rows", sorted(set(ys.tolist()))[:16], "cols", sorted(set(xs.tolist()))[:16])
        cen = want - Q.m0[li].astype(np.float64)[:, None, None]
        ws, wq = cen.sum(axis=(1, 2)), (cen ** 2).sum(axis=(1, 2))
        print(f"phase {npu.nphases - 1} {p['name']}: raw max|diff| {d.max()} mean {d.mean():.4f} (range {want.min()}..{want.max()}); "
              f"stats rel err sum {np.max(np.abs(st[:, 0] - ws) / (np.abs(ws) + 1e3)):.1e} sq {np.max(np.abs(st[:, 1] - wq) / (wq + 1e3)):.1e}; NPU {npu.ms:.2f} ms")
        if d.max():
            bad = np.argwhere(d == d.max())[0]; ys, xs = np.nonzero(d.max(axis=0))
            print("   worst (c,y,x)", bad, "got", got[tuple(bad)], "want", want[tuple(bad)], "rows", sorted(set(ys.tolist()))[:12], "cols", sorted(set(xs.tolist()))[:12])
        return d.max()
    if p["kind"] == "norm":
        got = npu.tensor(p["dst"]).astype(np.int32); want = tr[li]["x"].astype(np.int32)
        d = np.abs(got - want)
        print(f"phase {npu.nphases - 1} {p['name']}: x max|diff| {d.max()} mean {d.mean():.4f} >1: {100 * np.mean(d > 1):.3f}% "
              f"(range {want.min()}..{want.max()}); NPU {npu.ms:.2f} ms")
        if d.max() > 1:
            bad = np.argwhere(d == d.max())[0]; ys, xs = np.nonzero(d.max(axis=0) > 1)
            print("   worst (c,y,x)", bad, "got", got[tuple(bad)], "want", want[tuple(bad)], "rows", sorted(set(ys.tolist()))[:12], "cols", sorted(set(xs.tolist()))[:12])
        return d.max()
    rgb = npu.read_rgb(); emu = Q.emulate(img); ref = to_rgb(Q.net.forward(img))
    print(f"final: NPU vs emu {psnr(rgb, emu):.1f} dB, NPU vs fp32 {psnr(rgb, ref):.1f} dB, emu vs fp32 {psnr(emu, ref):.1f} dB; NPU {npu.ms:.2f} ms")
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", help="style .onnx (repeatable; default: the five in ~/models/style/onnx-zoo)")
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--phases", type=int, help="build only the first N phases (debug)")
    ap.add_argument("--check", action="store_true", help="compare the last built phase with the emulator on the test image")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--image"); ap.add_argument("--out", default="style_out.png")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--camera", type=int, help="camera index (default: the first capture device that is not a v4l2loopback)")
    ap.add_argument("--still", action="store_true", help="one webcam frame through every style -> style_still.png")
    ap.add_argument("--seconds", type=float, help="end the live loop after this many seconds (prints the frame rate)")
    ap.add_argument("--vcam", help="also send the result to this v4l2loopback device (e.g. /dev/video10, or auto); without --window it runs headless")
    ap.add_argument("--vcam-size", default="640x480", help="virtual camera resolution (the 320x240 result is scaled up)")
    ap.add_argument("--style", type=int, default=1, help="style to start with, 1-based")
    ap.add_argument("--cycle", type=float, help="switch to the next style every this many seconds")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--test-image", default=str(Path.home() / "models/yolo26n/coco128/images/train2017/000000000030.jpg"))
    a = ap.parse_args()
    import cv2
    W, H = map(int, a.size.lower().split("x"))
    models = a.model or [str(MODELS / f"{n}-9.onnx") for n in ("mosaic", "candy", "rain-princess", "udnie", "pointilism")]
    if not (a.check or a.selftest or a.bench or a.image or a.window or a.still or a.vcam):
        ap.print_help(); return
    npu = StyleNPU(models[:1] if a.check else models, W, H, nphases=a.phases)
    img = cv2.resize(cv2.imread(a.test_image), (W, H))[:, :, ::-1].copy()
    if a.check:
        check(npu, img); return
    if a.selftest:
        ok = True
        for i, (net, Q) in enumerate(zip(npu.nets, npu.quants)):
            rgb = npu.run(img, style=i); emu = Q.emulate(img); ref = to_rgb(net.forward(img)); ort = to_rgb(net.ort(img))
            print(f"{net.name:14s} NPU vs emu {psnr(rgb, emu):5.1f} dB  NPU vs fp32 {psnr(rgb, ref):5.1f} dB  emu vs fp32 {psnr(emu, ref):5.1f} dB  "
                  f"fp32 vs onnxruntime {psnr(ref, ort):5.1f} dB  NPU {npu.ms:.2f} ms")
            ok &= psnr(rgb, emu) > 40 and psnr(emu, ref) > 25          # the NPU computes the int8 design; the design tracks fp32
            if i == 0: cv2.imwrite("style_selftest.png", np.concatenate([img, ref, rgb], 1)[:, :, ::-1])
        print("selftest", "PASS" if ok else "FAIL"); return
    if a.bench:
        for _ in range(3): npu.run(img)
        t0 = time.perf_counter(); n = 20
        for _ in range(n): npu.run(img)
        dt = (time.perf_counter() - t0) / n
        print(f"{W}x{H}: {dt * 1e3:.2f} ms per frame = {1 / dt:.0f} fps (whole call incl. upload and readback); NPU call {npu.ms:.2f} ms"); return
    if a.image:
        src = cv2.imread(a.image); rgb = cv2.resize(src, (W, H))[:, :, ::-1].copy()
        for i, net in enumerate(npu.nets):
            out = npu.run(rgb, style=i); name = a.out if len(npu.nets) == 1 else a.out.replace(".png", f"_{net.name}.png")
            cv2.imwrite(name, out[:, :, ::-1]); print(f"{net.name}: {npu.ms:.2f} ms -> {name}")
        return
    vcam = None
    if a.vcam:                                                    # before the camera: a missing loopback exits cleanly
        vw, vh = map(int, a.vcam_size.lower().split("x")); vcam = VirtualCam(a.vcam, vw, vh)
        print(f"virtual camera {vcam.name}: {vw}x{vh} I420" + ("" if a.window else "; headless, Ctrl-C stops"))
    cam_index = find_camera(exclude=vcam.dev if vcam else None) if a.camera is None else a.camera
    if cam_index is None: sys.exit("no camera")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened(): sys.exit("no camera")

    def grab():
        ok, bgr = cap.read()
        if not ok: return None
        h, w = bgr.shape[:2]; s = min(w / W, h / H); cw, ch = int(W * s), int(H * s)
        crop = bgr[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
    if a.still:
        for _ in range(10): rgb = grab()                          # let the exposure settle
        tiles = [rgb] + [npu.run(rgb, style=i) for i in range(len(npu.nets))]
        while len(tiles) % 3: tiles.append(np.zeros_like(rgb))
        grid = np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)], 0)
        cv2.imwrite("style_still.png", grid[:, :, ::-1]); print(f"wrote style_still.png ({npu.ms:.1f} ms per frame on the NPU)"); return
    import threading
    latest = [None]; stop = [False]

    def reader():                                                 # the camera paces itself; keep only the newest frame
        while not stop[0]:
            f = grab()
            if f is not None: latest[0] = f
    reader_thread = threading.Thread(target=reader, daemon=True); reader_thread.start()
    if a.window:
        import pygame
        pygame.init(); scale = 2 if W <= 480 else 1
        screen = pygame.display.set_mode((W * scale, H * scale)); font = pygame.font.SysFont(None, 24)
    n, t0, show_in = 0, time.perf_counter(), False
    style = max(0, min(len(npu.nets) - 1, a.style - 1)); t_style = t0; t_print = t0
    import signal                                                 # Ctrl-C only sets a flag: an exception raised inside
    quit_ = [False]                                               # npu.run aborts XRT ("xrt::run destructed while ...")
    signal.signal(signal.SIGINT, lambda *_: quit_.__setitem__(0, True))
    try:
        while not quit_[0]:
            if a.window:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
                    if e.type == pygame.KEYDOWN and pygame.K_1 <= e.key <= pygame.K_9 and e.key - pygame.K_1 < len(npu.nets): style = e.key - pygame.K_1
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_0: show_in = not show_in
            now = time.perf_counter()
            if a.cycle and now - t_style > a.cycle: style = (style + 1) % len(npu.nets); t_style = now
            rgb, latest[0] = latest[0], None
            if rgb is None: time.sleep(0.002); continue
            out = rgb if show_in else npu.run(rgb, style=style); n += 1
            if vcam: vcam.write(out)
            if a.seconds and now - t0 > a.seconds:
                print(f"{n} frames in {now - t0:.1f} s = {n / (now - t0):.1f} fps (camera-bound); NPU {npu.ms:.2f} ms per frame"); return
            if not a.window:
                if now - t_print > 5: print(f"{npu.nets[style].name}: {n / (now - t0):.1f} fps, NPU {npu.ms:.1f} ms per frame"); t_print = now
                continue
            frame = cv2.resize(out, (W * scale, H * scale), interpolation=cv2.INTER_LINEAR) if scale != 1 else out
            screen.blit(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (0, 0))
            screen.blit(font.render(f"{npu.nets[style].name} on the NPU: {npu.ms:.1f} ms  {n / (time.perf_counter() - t0):.1f} fps  (1-{len(npu.nets)} style, 0 input, ESC)", True, (255, 255, 0)), (8, 8))
            pygame.display.flip()
    except KeyboardInterrupt:
        pass
    finally:
        stop[0] = True                                            # stop the reader before the interpreter exits:
        reader_thread.join(timeout=2); cap.release()              # a thread left inside cap.read() aborts the process
        if vcam: vcam.close()


if __name__ == "__main__":
    main()
