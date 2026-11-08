#!/usr/bin/env python3
"""The NPU learns: MNIST training (not just inference) on all 32 cores, one dispatch per epoch.

  ./env.sh python fun/mnist_npu.py --selftest        init bit-exact vs numpy, one step vs a numpy replica, 1 epoch
  ./env.sh python fun/mnist_npu.py --epochs 5        train, test accuracy after every epoch, NPU time per epoch
  ./env.sh python fun/mnist_npu.py --window          watch it learn: weights, accuracy curve, draw a digit (mouse),
                                                     teach it new symbols (draw, then press A-F)
  ./env.sh python fun/mnist_npu.py --bench           one epoch per dispatch, back to back

A 784-256-10 multilayer perceptron (ReLU, softmax cross-entropy, plain SGD, batch 64) trained with bf16 matmuls into
fp32 accumulators and fp32 master weights. Model parallel: core k (along a snake through the 32 cores, so that
neighbours share memory) owns hidden units 8k..8k+7, i.e. 784x8 of W1 and 8x10 of W2, resident in its L1 across the
whole run. Per minibatch:

  * the host stream (one shim, one memtile broadcasting to all 32 cores) brings the labels and the 64 images in
    7 chunks of 112 pixels, as 8x8 tiles transposed for the forward pass (H^T = W1^T X^T, mmul<8,8,8,bf16>);
  * each core adds its partial logits P^T = W2^T relu(H^T) to the running sum handed down the snake (shared memory);
  * the last core adds the bias, takes the softmax (exp2 and inv in bf16), forms G = -lr/B (p - onehot) and
    broadcasts it to all cores;
  * every core computes dh^T = W2 G (masked by ReLU), updates W2, b1 (row sums via a mmul with a ones tile) and,
    while the same images stream in again as untransposed tiles, W1^T += dh^T X.

Weights never leave the chip during training; a dispatch ends with every core dumping its slice.

Data: ~/models/mnist (LeCun, Cortes, Burges; CC BY-SA 3.0; see NOTE.txt there). Not redistributed.
"""
import argparse, gzip, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

B = 64                          # minibatch
NBB = B // 8                    # batch blocks of 8
NPB = 98                        # pixel blocks of 8 (784 pixels)
CPB = 7                         # pixel blocks per chunk
NCH = NPB // CPB                # chunks per pass
XOBJ = CPB * NBB * 128          # bytes of a stream object (7168): bf16 tiles
NHID, UPC = 256, 8              # hidden units, units per core
NOUT = 16                       # classes: 10 digits + 6 symbols the window can teach
DOBJ = 3136                     # dump object bytes (W1 slice = 8 objects)
NDUMP = 9
COLS, ROWS = 8, 4
NCORE = COLS * ROWS
MAGIC = 0x4D4E4953
DATA = Path.home() / "models" / "mnist"
MB_OBJ = 1 + NCH                # labels + chunks
OBJ_PER_MB = 2 * MB_OBJ         # per minibatch the stream carries them twice (forward and backward pass)


def snake(k):
    c = k // ROWS; r = k % ROWS
    return c, (r if c % 2 == 0 else ROWS - 1 - r)


# ------------------------------------------------------------------------------------------------- data
def load_mnist():
    def idx(f):
        b = gzip.open(DATA / f).read(); nd = b[3]
        dims = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
        return np.frombuffer(b, np.uint8, offset=4 + 4 * nd).reshape(dims)
    return (idx("train-images-idx3-ubyte.gz").reshape(-1, 784), idx("train-labels-idx1-ubyte.gz"),
            idx("t10k-images-idx3-ubyte.gz").reshape(-1, 784), idx("t10k-labels-idx1-ubyte.gz"))


def bf16_bits(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16(x):
    return (bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def control(init, seed, lr):
    c = np.zeros(XOBJ // 4, np.int32)
    c[0] = MAGIC; c[1] = int(init); c[2] = seed
    c[3] = bf16(np.array([-lr / B], np.float32)).view(np.int32)[0]
    return c.view(np.uint8)


_BF16_CACHE = {}


def pixels_bf16(X):
    """pixel / 256 as bf16 bits (exact), cached per array."""
    key = (id(X), X.shape)
    if key not in _BF16_CACHE:
        _BF16_CACHE.clear(); _BF16_CACHE[key] = bf16_bits(X.astype(np.float32) / 256)
    return _BF16_CACHE[key]


def stream_into(buf, X, y, idx, init, seed, lr):
    """Write one call's data into buf (uint8): control object, then per minibatch the labels and the images as bf16
    8x8 tiles [pb][bb][8 b][8 px] (the forward pass transposes them in the kernel), twice, for the forward and the
    backward pass (a shim BD cannot re-read with stride 0). Returns the minibatch count."""
    nmb = len(idx) // B
    v = buf[XOBJ:XOBJ + nmb * OBJ_PER_MB * XOBJ].reshape(nmb, 2, MB_OBJ, XOBJ)
    xs = pixels_bf16(X)[idx[:nmb * B]]                                   # [m*B + bb*8 + b][784] bf16 bits
    src = xs.view(np.uint64).reshape(nmb, NBB, 8, NPB, 2).transpose(0, 3, 1, 2, 4)          # [m][pb][bb][b] of 8 px
    for r in range(2):
        np.copyto(v[:, r, 1:].view(np.uint64).reshape(nmb, NPB, NBB, 8, 2), src)
        v[:, r, 0, :B] = y[idx[:nmb * B]].reshape(nmb, B)
    buf[:XOBJ] = control(init, seed, lr)
    return nmb


# ------------------------------------------------------------------------------------------------- reference
def xorshift_init(seed, cid, n, shift):
    """The kernel's initialiser: 32 xorshift32 lanes, w = (int32(s) >> 8) * 2^-shift, lane-interleaved."""
    s = np.array([(seed ^ ((cid * 0x9E3779B9) & 0xFFFFFFFF) ^ (((l + 1) * 0x85EBCA6B) & 0xFFFFFFFF)) | 1 for l in range(32)], np.uint32)
    out = np.zeros(n, np.float32)
    for i in range(n // 32):
        s ^= s << np.uint32(13); s ^= s >> np.uint32(17); s ^= s << np.uint32(5)
        out[i * 32:(i + 1) * 32] = (s.view(np.int32) >> 8).astype(np.float32) * np.float32(2.0 ** -shift)
    return out, s


class Model:
    """Full weights in the usual layout (W1 784x256, b1, W2 256x10, b2), assembled from / split into core slices."""

    def __init__(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

    @classmethod
    def init(cls, seed):
        W1 = np.zeros((784, NHID), np.float32); W2 = np.zeros((NHID, NOUT), np.float32)
        for k in range(NCORE):
            w1, s = xorshift_init(seed, k, NPB * 64, 27)
            W1[:, k * UPC:(k + 1) * UPC] = np.transpose(w1.reshape(NPB, UPC, 8), (0, 2, 1)).reshape(784, UPC)
            w2 = np.zeros(128, np.float32)
            for i in range(4):
                s ^= s << np.uint32(13); s ^= s >> np.uint32(17); s ^= s << np.uint32(5)
                w2[i * 32:(i + 1) * 32] = (s.view(np.int32) >> 8).astype(np.float32) * np.float32(2.0 ** -26)
            W2[k * UPC:(k + 1) * UPC] = w2.reshape(NOUT, UPC).T              # [o][u]
        return cls(W1, np.zeros(NHID, np.float32), W2, np.zeros(NOUT, np.float32))

    @classmethod
    def from_dump(cls, dump):
        """dump: (NCORE, NDUMP * DOBJ) bytes in snake order."""
        W1 = np.zeros((784, NHID), np.float32); W2 = np.zeros((NHID, NOUT), np.float32); b1 = np.zeros(NHID, np.float32)
        b2 = None; stats = None
        for k in range(NCORE):
            d = dump[k]
            w1 = d[:NPB * 64 * 4].view(np.float32).reshape(NPB, UPC, 8)          # [pb][u][p]
            W1[:, k * UPC:(k + 1) * UPC] = np.transpose(w1, (0, 2, 1)).reshape(784, UPC)
            t = d[8 * DOBJ:9 * DOBJ].view(np.float32)
            w2t = t[:128].reshape(16, UPC)                                      # [o][u]
            W2[k * UPC:(k + 1) * UPC] = w2t.T
            b1[k * UPC:(k + 1) * UPC] = t[128:192].reshape(UPC, 8)[:, 0]
            if k == NCORE - 1:
                b2 = t[192:320].reshape(16, 8)[:, 0].copy()
                stats = d[9 * DOBJ - 64:9 * DOBJ].view(np.int32)
        cls.prof = np.stack([dump[k][9 * DOBJ - 144:9 * DOBJ - 64].view(np.uint32) for k in range(NCORE)])
        return cls(W1, b1, W2, b2), stats

    def logits(self, X):
        h = np.maximum((X.astype(np.float32) / 256) @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def accuracy(self, X, y):
        return float((self.logits(X).argmax(1) == y).mean())

    def step(self, x, y, lr, emulate=True):
        """One SGD step in numpy; emulate=True rounds where the NPU rounds to bf16."""
        r = bf16 if emulate else (lambda a: a)
        x = x.astype(np.float32) / 256
        h = np.maximum(x @ r(self.W1) + self.b1, 0)
        z = r(h) @ r(self.W2) + self.b2
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(y)), y] -= 1
        c = float(bf16(np.array([-lr / B], np.float32))[0]) if emulate else -lr / B
        g = r(r(g) * c)
        dh = r((g @ r(self.W2).T) * (h > 0))
        self.W2 += r(h).T @ g; self.b2 += g.sum(0); self.W1 += x.T @ dh; self.b1 += dh.sum(0)
        return float((z.argmax(1) == y).sum())


# ------------------------------------------------------------------------------------------------- the kernel
KERNEL = r"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define NPB 98
#define CPB 7
#define NBB 8
#define DOBJ 3136
#define MAGIC 0x4D4E4953
#define COLD __attribute__((noinline, minsize))
using bf = bfloat16;
using VB = aie::vector<bf, 64>;
using MM = aie::mmul<8, 8, 8, bf, bf>;
using AF = aie::accum<accfloat, 32>;
using VF = aie::vector<float, 32>;
using V16 = aie::vector<float, 16>;

alignas(64) static float W1T[NPB * 64];          // [pb][u][p] fp32 master
alignas(64) static bf W1Tb[CPB * 64];            // the current chunk in bf16
alignas(64) static float H[NBB * 64];            // [bb][u][b]
alignas(64) static bf hT[NBB * 64], dhT[NBB * 64];
alignas(64) static float W2T[2 * 64];            // [half][o][u]
alignas(64) static bf W2Tb[2 * 64], W2b[2 * 64]; // bf16 W2^T tiles [o][u] and W2 tiles [u][o]
alignas(64) static float b1T[64];                // [u][b] = b1[u]
alignas(64) static float b2T[2 * 64];            // [half][o][b] = b2[o]; classes 10..15 are for symbols taught live
alignas(64) static bf GT[NBB * 2 * 64];          // last core: its own copy of G
alignas(64) static bf ONES[64];
alignas(64) static int32_t lab[64];
alignas(64) static int32_t stats[16];
static int chunk, dumpi, cid_;
static int32_t alive;
// profile: per entry point, cycles running and cycles waiting since the previous entry returned
static uint32_t prun[10], pwait[10];
static uint32_t tlast; static bool tvalid;
static inline uint32_t cyc() { return read_tm(0x340F8); }      // core module Timer_Low
#ifdef PROFILE                                                  // hangs the cores on this setup (read_tm of Timer_Low)
struct Prof {
    int id; uint32_t t0;
    Prof(int i) : id(i) { t0 = cyc(); if (tvalid) pwait[id] += t0 - tlast; }
    ~Prof() { tlast = cyc(); tvalid = true; prun[id] += tlast - t0; }
};
#else
struct Prof { Prof(int) {} };
#endif
static bf cscale;

static inline VB f2b(const float *p) {
    return aie::concat(AF(aie::load_v<32>(p)).template to_vector<bf>(), AF(aie::load_v<32>(p + 32)).template to_vector<bf>());
}
static inline VB xt(const int8_t *x, int i) { return aie::load_v<64>((const bf *)x + 64 * i); }   // pixels / 256 as bf16
static inline void st_add(float *dst, const float *add, MM &c) {   // dst = add + c (fp32)
    auto a = c.to_accum();
    for (int h = 0; h < 2; h++) {
        AF x = a.template extract<32>(h);
        x = aie::add(x, aie::load_v<32>(add + 32 * h));
        aie::store_v(dst + 32 * h, x.template to_vector<float>());
    }
}
static inline VF acc_half(MM &c, int h) { return c.to_accum().template extract<32>(h).template to_vector<float>(); }

static void COLD refresh_w2() {
    for (int h = 0; h < 2; h++) {
        VB t = f2b(W2T + 64 * h);
        aie::store_v(W2Tb + 64 * h, t);
        aie::store_v(W2b + 64 * h, aie::transpose(t, 8, 8));
    }
}

static void COLD init_weights(uint32_t seed) {
    aie::vector<uint32, 32> s;
    for (int l = 0; l < 32; l++) s[l] = (seed ^ ((uint32_t)cid_ * 0x9E3779B9u) ^ ((uint32_t)(l + 1) * 0x85EBCA6Bu)) | 1u;
    for (int i = 0; i < NPB * 64 / 32 + 4; i++) {
        s = aie::bit_xor(s, aie::upshift(s, 13)); s = aie::bit_xor(s, aie::downshift(s, 17)); s = aie::bit_xor(s, aie::upshift(s, 5));
        aie::vector<int32, 32> si = aie::downshift(s.template cast_to<int32>(), 8);
        if (i < NPB * 2) aie::store_v(W1T + 32 * i, aie::to_float<float>(si, 27));
        else aie::store_v(W2T + 32 * (i - NPB * 2), aie::to_float<float>(si, 26));
    }
    aie::store_v(b1T, aie::zeros<float, 32>()); aie::store_v(b1T + 32, aie::zeros<float, 32>());
    for (int l = 0; l < 128; l += 32) aie::store_v(b2T + l, aie::zeros<float, 32>());
}

// ---- control object: magic, init flag, seed, -lr/B (fp32 bits, already bf16-exact)
extern "C" void k_ctl(int8_t *x, int32_t cid) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *c = (const int32_t *)x;
    cid_ = cid;
    if (c[1]) { init_weights((uint32_t)c[2]); alive = 0x600D; }
    cscale = (bf)(*(const float *)(c + 3));
    for (int l = 0; l < 64; l++) ONES[l] = (bf)1.0f;
    for (int i = 0; i < 16; i++) stats[i] = 0;
    for (int i = 0; i < 10; i++) { prun[i] = 0; pwait[i] = 0; }
    tvalid = false;
    stats[1] = c[0] == MAGIC; stats[2] = alive == 0x600D;      // stats[2] = 0: the weights were lost (device reset)
    refresh_w2();
    dumpi = 0;
}

extern "C" void k_lab(int8_t *x) {             // the first copy: labels; the second copy only resets the chunk counter
    Prof prof_(1);
    for (int i = 0; i < 64; i++) lab[i] = ((const uint8_t *)x)[i];
    chunk = 0;
}

// ---- forward chunk: H^T[bb] (+)= W1^T[pb] X^T[pb][bb] over the chunk's 14 pixel blocks (tiles arrive as [b][p])
extern "C" void k_fwd(int8_t *x) {
    Prof prof_(2);
#ifdef NOCOMPUTE
    chunk++; return;
#endif
    const int pb0 = chunk * CPB;
    for (int p = 0; p < CPB; p++) aie::store_v(W1Tb + 64 * p, f2b(W1T + 64 * (pb0 + p)));
    for (int bb = 0; bb < NBB; bb++) {
        MM c;
        for (int p = 0; p < CPB; p++) c.mac(aie::load_v<64>(W1Tb + 64 * p), aie::transpose(xt(x, p * NBB + bb), 8, 8));
        st_add(H + 64 * bb, chunk == 0 ? b1T : H + 64 * bb, c);
    }
    chunk++;
}

// ---- partial logits P^T[bb][half] = W2^T[half] relu(H^T[bb]), plus the incoming running sum (o: [2 halves][64])
static void COLD partial_bb(int bb, const float *sin, float *o, bool last) {
    VF a = aie::max(aie::load_v<32>(H + 64 * bb), aie::zeros<float, 32>());
    VF b = aie::max(aie::load_v<32>(H + 64 * bb + 32), aie::zeros<float, 32>());
    aie::store_v(hT + 64 * bb, aie::concat(AF(a).template to_vector<bf>(), AF(b).template to_vector<bf>()));
    for (int h = 0; h < 2; h++) {
        MM c; c.mac(aie::load_v<64>(W2Tb + 64 * h), aie::load_v<64>(hT + 64 * bb));
        if (sin) st_add(o + 64 * h, sin + (bb * 2 + h) * 64, c);
        else { aie::store_v(o + 64 * h, acc_half(c, 0)); aie::store_v(o + 64 * h + 32, acc_half(c, 1)); }
        if (last)
            for (int q = 0; q < 2; q++) aie::store_v(o + 64 * h + 32 * q, aie::add(AF(aie::load_v<32>(o + 64 * h + 32 * q)), aie::load_v<32>(b2T + 64 * h + 32 * q)).template to_vector<float>());
    }
}

extern "C" void k_part(float *so) {
    Prof prof_(3);
#ifdef NOCHAIN
    return;
#endif
 for (int bb = 0; bb < NBB; bb++) partial_bb(bb, nullptr, so + bb * 128, false); }
static void add_to(const float *si, float *so) {
    for (int i = 0; i < NBB * 128; i += 32) aie::store_v(so + i, aie::add(AF(aie::load_v<32>(so + i)), aie::load_v<32>(si + i)).template to_vector<float>());
}
extern "C" void k_add(float *si, float *so) {           // the chain hop: so += si
    Prof prof_(4);
#if defined(NOCHAIN) || defined(NOHOP)
    return;
#endif
    for (int i = 0; i < NBB * 128; i += 32) aie::store_v(so + i, aie::add(AF(aie::load_v<32>(so + i)), aie::load_v<32>(si + i)).template to_vector<float>());
}

// ---- backward, given G^T [bb][half][o][b] (bf16, already scaled by -lr/B)
static void COLD grad(const bf *g, bool last) {
    MM cb;
    for (int bb = 0; bb < NBB; bb++) {
        MM c;
        c.mac(aie::load_v<64>(W2b), aie::load_v<64>(g + (bb * 2) * 64));
        c.mac(aie::load_v<64>(W2b + 64), aie::load_v<64>(g + (bb * 2 + 1) * 64));
        VB d = aie::concat(c.to_accum().template extract<32>(0).template to_vector<bf>(), c.to_accum().template extract<32>(1).template to_vector<bf>());
        VB hv = aie::load_v<64>(hT + 64 * bb);
        d = aie::select(aie::zeros<bf, 64>(), d, aie::gt(hv, aie::zeros<bf, 64>()));
        aie::store_v(dhT + 64 * bb, d);
        cb.mac(d, aie::load_v<64>(ONES));
    }
    st_add(b1T, b1T, cb);
    for (int h = 0; h < 2; h++) {
        MM cw, c2;
        for (int bb = 0; bb < NBB; bb++) {
            VB gt = aie::load_v<64>(g + (bb * 2 + h) * 64);
            cw.mac(gt, aie::transpose(aie::load_v<64>(hT + 64 * bb), 8, 8));
            if (last) c2.mac(gt, aie::load_v<64>(ONES));
        }
        st_add(W2T + 64 * h, W2T + 64 * h, cw);
        if (last) st_add(b2T + 64 * h, b2T + 64 * h, c2);
    }
    refresh_w2();
    chunk = 0;
}

extern "C" void k_grad(bf *g) {
    Prof prof_(7);
#if defined(NOCHAIN) || defined(NOGRAD)
    return;
#endif
    grad(g, false);
}

// ---- last core: logits -> softmax -> G = -lr/B (p - onehot), broadcast and keep a copy
alignas(64) static float SR[16 * 16];             // rows o of 16 lanes (two batch blocks)
alignas(64) static bf SE[16 * 16];

static const float *Lg;
static void COLD softmax_pair(int pr, bf *gout) {
    const aie::vector<bf, 16> l2h = aie::broadcast<bf, 16>((bf)1.4453125f), l2l = aie::broadcast<bf, 16>((bf)-0.0025749207f);
    for (int o = 0; o < 16; o++) {
        const int h = o >> 3, oo = o & 7;
        aie::store_v(SR + 16 * o, aie::concat(aie::load_v<8>(Lg + (2 * pr) * 128 + h * 64 + oo * 8), aie::load_v<8>(Lg + (2 * pr + 1) * 128 + h * 64 + oo * 8)));
    }
    V16 m = aie::load_v<16>(SR);
    for (int o = 1; o < 16; o++) m = aie::max(m, aie::load_v<16>(SR + 16 * o));
    aie::vector<int32, 16> lb = aie::load_v<16>(lab + 16 * pr);
    V16 s = aie::zeros<float, 16>(), ty = aie::zeros<float, 16>();
    for (int o = 0; o < 16; o++) {
        V16 row = aie::load_v<16>(SR + 16 * o);
        aie::accum<accfloat, 16> za = aie::sub(aie::accum<accfloat, 16>(row), m);
        aie::vector<bf, 16> zh = za.template to_vector<bf>(), zl = aie::sub(za, aie::accum<accfloat, 16>(zh)).template to_vector<bf>();
        aie::accum<accfloat, 16> t = aie::mul(zh, l2h); t = aie::mac(t, zl, l2h); t = aie::mac(t, zh, l2l);
        aie::vector<bf, 16> e = aie::exp2<bf>(t.template to_vector<float>());
        aie::store_v(SE + 16 * o, e);
        s = aie::add(aie::accum<accfloat, 16>(s), aie::accum<accfloat, 16>(e).template to_vector<float>()).template to_vector<float>();
        ty = aie::select(ty, row, aie::eq(lb, aie::broadcast<int32, 16>(o)));
    }
    aie::vector<bf, 16> r = aie::inv(aie::accum<accfloat, 16>(s).template to_vector<bf>());
    aie::mask<16> ok = aie::ge(ty, m);
    stats[0] += ok.count();
    for (int o = 0; o < 16; o++) {
        V16 p = aie::mul(aie::load_v<16>(SE + 16 * o), r).template to_vector<float>();
        V16 oh = aie::select(aie::zeros<float, 16>(), aie::broadcast<float, 16>(1.0f), aie::eq(lb, aie::broadcast<int32, 16>(o)));
        aie::vector<bf, 16> d = aie::sub(aie::accum<accfloat, 16>(p), oh).template to_vector<bf>();
        aie::vector<bf, 16> gv = aie::mul(d, cscale).template to_vector<bf>();
        const int h = o >> 3, oo = o & 7;
        for (int q = 0; q < 2; q++) {
            aie::vector<bf, 8> part = gv.template extract<8>(q);
            aie::store_v(GT + ((2 * pr + q) * 2 + h) * 64 + oo * 8, part);
            aie::store_v(gout + ((2 * pr + q) * 2 + h) * 64 + oo * 8, part);
        }
    }
}

extern "C" void k_lpart(float *L) {
    Prof prof_(5);
#ifdef NOCHAIN
    return;
#endif
 for (int bb = 0; bb < NBB; bb++) partial_bb(bb, nullptr, L + bb * 128, true); }
extern "C" void k_last(float *si, float *L, bf *gout) {
    Prof prof_(6);
#ifdef NOCHAIN
    return;
#endif
    add_to(si, L); Lg = L;
#ifdef NOGRAD
    return;
#endif
    for (int pr = 0; pr < NBB / 2; pr++) softmax_pair(pr, gout);
}
extern "C" void k_lgrad() {                             // the last core's own update, after G has been released
    Prof prof_(7);
#if defined(NOCHAIN) || defined(NOGRAD)
    return;
#endif
    grad(GT, true);
}

// ---- backward chunk: W1^T[pb] += dh^T X[pb] (X tiles [bb][b][p])
extern "C" void k_bwd(int8_t *x) {
    Prof prof_(8);
#ifdef NOCOMPUTE
    chunk++; return;
#endif
    const int pb0 = chunk * CPB;
    for (int p = 0; p < CPB; p++) {
        MM c;
        for (int bb = 0; bb < NBB; bb++) c.mac(aie::load_v<64>(dhT + 64 * bb), xt(x, p * NBB + bb));
        st_add(W1T + 64 * (pb0 + p), W1T + 64 * (pb0 + p), c);
    }
    chunk++;
}

// ---- dump: 8 objects of W1^T, then W2^T | b1^T | b2^T | stats (last 64 bytes)
extern "C" void k_dump(int8_t *o) {
    if (dumpi < 8) {
        const int8_t *s = (const int8_t *)W1T + dumpi * DOBJ;
        for (int i = 0; i < DOBJ; i += 64) aie::store_v(o + i, aie::load_v<64>(s + i));
    } else {
        float *f = (float *)o;
        for (int i = 0; i < 128; i += 32) aie::store_v(f + i, aie::load_v<32>(W2T + i));
        for (int i = 0; i < 64; i += 32) aie::store_v(f + 128 + i, aie::load_v<32>(b1T + i));
        for (int i = 0; i < 128; i += 32) aie::store_v(f + 192 + i, aie::load_v<32>(b2T + i));
        int32_t *t = (int32_t *)(o + DOBJ - 64);
        for (int i = 0; i < 16; i++) t[i] = stats[i];
        uint32_t *pr = (uint32_t *)(o + DOBJ - 64 - 80);
        for (int i = 0; i < 10; i++) { pr[i] = prun[i]; pr[10 + i] = pwait[i]; }
    }
    dumpi++;
}
"""


# ------------------------------------------------------------------------------------------------- the IRON program
def build_program(nmb):
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Buffer, ExternalFunction, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
    x_ty = np.ndarray[(XOBJ,), np.dtype[np.int8]]
    s_ty = np.ndarray[(NBB * 2 * 64,), np.dtype[np.float32]]
    g_ty = np.ndarray[(NBB * 2 * 64,), np.dtype[np.int16]]          # bf16 bits
    o_ty = np.ndarray[(DOBJ,), np.dtype[np.int8]]; o4_ty = np.ndarray[(ROWS * DOBJ,), np.dtype[np.int8]]
    i32 = np.dtype(np.int32)
    decl = "#include <stdint.h>\n"
    nc = os.environ.get("MNIST_NOCOMPUTE", "")
    src = ("#define NOCOMPUTE\n" if nc else "") + ("#define NOCHAIN\n" if nc == "2" else "") + \
          ("#define NOGRAD\n" if nc == "A" else "") + ("#define NOHOP\n" if nc == "B" else "") + KERNEL
    kctl = ExternalFunction("k_ctl", source_string=src, arg_types=[x_ty, i32])
    kf = {n: ExternalFunction(n, source_string=decl + f'extern "C" void {n}({sig});\n', arg_types=at) for n, sig, at in [
        ("k_lab", "int8_t *", [x_ty]), ("k_fwd", "int8_t *", [x_ty]), ("k_bwd", "int8_t *", [x_ty]),
        ("k_part", "float *", [s_ty]), ("k_add", "float *, float *", [s_ty, s_ty]), ("k_lpart", "float *", [s_ty]),
        ("k_last", "float *, float *, int16_t *", [s_ty, s_ty, g_ty]), ("k_lgrad", "void", []),
        ("k_grad", "int16_t *", [g_ty]), ("k_dump", "int8_t *", [o_ty])]}

    xi = ObjectFifo(x_ty, name="xin", depth=2)
    xb = xi.cons().forward(tile=Tile(0, 1), depth=2, name="xbc")
    chain = [ObjectFifo(s_ty, name=f"s{k}", depth=1) for k in range(NCORE - 1)]
    gf = ObjectFifo(g_ty, name="g", depth=1)
    outs = [ObjectFifo(o4_ty, name=f"d{c}", depth=1) for c in range(COLS)]
    joins = [outs[c].prod().join([r * DOBJ for r in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS, obj_types=[o_ty] * ROWS,
                                 names=[f"d{c}_{r}" for r in range(ROWS)]) for c in range(COLS)]

    def step_loop(xf, klab, kfwd, kbwd, middle):
        for _ in range_(nmb):
            x = xf.acquire(1); klab(x); xf.release(1)
            for _ in range_(NCH):
                x = xf.acquire(1); kfwd(x); xf.release(1)
            middle()
            x = xf.acquire(1); klab(x); xf.release(1)
            for _ in range_(NCH):
                x = xf.acquire(1); kbwd(x); xf.release(1)

    def tail(of, kdump):
        for _ in range_(NDUMP):
            o = of.acquire(1); kdump(o); of.release(1)

    # every core puts its partial logits into its own outgoing buffer first; the hop down the snake is then just an add
    def body_first(xf, so, gi, of, kctl, klab, kfwd, kbwd, kpart, kgrad, kdump, cid):
        x = xf.acquire(1); kctl(x, cid); xf.release(1)

        def middle():
            s = so.acquire(1); kpart(s); so.release(1)
            g = gi.acquire(1); kgrad(g); gi.release(1)
        step_loop(xf, klab, kfwd, kbwd, middle); tail(of, kdump)

    def body_mid(xf, si, so, gi, of, kctl, klab, kfwd, kbwd, kpart, kadd, kgrad, kdump, cid):
        x = xf.acquire(1); kctl(x, cid); xf.release(1)

        def middle():
            s = so.acquire(1); kpart(s); a = si.acquire(1); kadd(a, s); si.release(1); so.release(1)
            g = gi.acquire(1); kgrad(g); gi.release(1)
        step_loop(xf, klab, kfwd, kbwd, middle); tail(of, kdump)

    def body_last(xf, si, go, of, L, kctl, klab, kfwd, kbwd, klpart, klast, klgrad, kdump, cid):
        x = xf.acquire(1); kctl(x, cid); xf.release(1)

        def middle():
            klpart(L); a = si.acquire(1); g = go.acquire(1); klast(a, L, g); go.release(1); si.release(1); klgrad()
        step_loop(xf, klab, kfwd, kbwd, middle); tail(of, kdump)

    workers = []
    common = [kctl, kf["k_lab"], kf["k_fwd"], kf["k_bwd"]]
    for k in range(NCORE):
        c, r = snake(k); tile = Tile(c, 2 + r); of = joins[c][r].prod()
        if k == 0:
            workers.append(Worker(body_first, [xb.cons(depth=2), chain[0].prod(), gf.cons(), of] + common + [kf["k_part"], kf["k_grad"], kf["k_dump"], k], tile=tile, dynamic_objfifo_lowering=True))
        elif k < NCORE - 1:
            workers.append(Worker(body_mid, [xb.cons(depth=2), chain[k - 1].cons(), chain[k].prod(), gf.cons(), of] + common + [kf["k_part"], kf["k_add"], kf["k_grad"], kf["k_dump"], k], tile=tile, dynamic_objfifo_lowering=True))
        else:
            L = Buffer(s_ty, name="logits")
            workers.append(Worker(body_last, [xb.cons(depth=2), chain[k - 1].cons(), gf.prod(), of, L] + common + [kf["k_lpart"], kf["k_last"], kf["k_lgrad"], kf["k_dump"], k], tile=tile, dynamic_objfifo_lowering=True))

    words = (1 + nmb * OBJ_PER_MB) * XOBJ // 4; dwords = COLS * NDUMP * ROWS * DOBJ // 4
    in_ty = np.ndarray[(words,), np.dtype[np.int32]]; d_ty = np.ndarray[(dwords,), np.dtype[np.int32]]
    per = 256 * XOBJ // 4                                          # words per fill (256 objects)

    def seq(src, dump, xp, oc):
        drains = [oc[c].drain(dump, TensorAccessPattern((dwords,), c * dwords // COLS, [1, 1, 1, dwords // COLS], [0, 0, 0, 1]), wait=True, managed=False) for c in range(COLS)]
        inflight = []
        for off in range(0, words, per):
            n = min(per, words - off)
            if len(inflight) >= 8: inflight.pop(0).await_()
            inflight.append(xp.fill(src, TensorAccessPattern((words,), off, [1, 1, 1, n], [0, 0, 0, 1]), wait=True, managed=False))
        for t in inflight: t.await_()
        for t in drains: t.await_()

    rt = Runtime(seq, [in_ty, d_ty, xi.prod(tile=Tile(0, 0)), [outs[c].cons(tile=Tile(c, 0)) for c in range(COLS)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _jit():
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    @iron.jit
    def mnist_npu(src: In, dump: Out, *, nmb: CompileTime[int], variant: CompileTime[str]):
        return build_program(nmb)
    return mnist_npu


class TrainerNPU:
    """One compiled program per minibatch count; weights stay on the chip between calls."""

    def __init__(self, nmb, nbuf=1):
        import aie.iron as iron
        self.nmb = nmb; self.fn = _jit()
        self.variant = "mnist-" + hashlib.md5((KERNEL + str(nmb) + str(B) + os.environ.get("MNIST_NOCOMPUTE", "")).encode()).hexdigest()[:10]
        self.srcs = [iron.zeros(((1 + nmb * OBJ_PER_MB) * XOBJ // 4,), dtype=np.int32, device="npu") for _ in range(nbuf)]
        self.src = self.srcs[0]
        self.dump = iron.zeros((COLS * NDUMP * ROWS * DOBJ // 4,), dtype=np.int32, device="npu")
        self.ms = 0.0

    def again(self):
        """Run the stream already in the buffer once more (benchmarks: no host work in between)."""
        t0 = time.perf_counter()
        self.fn(self.src, self.dump, nmb=self.nmb, variant=self.variant)
        self.ms = (time.perf_counter() - t0) * 1e3

    def prepare(self, X, y, idx, init=False, seed=1, lr=0.4, buf=0):
        with self.srcs[buf].overwrite() as v:
            nmb = stream_into(v.view(np.uint8), X, y, idx, init, seed, lr)
        assert nmb == self.nmb, (nmb, self.nmb)

    def call(self, buf=0):
        self.src = self.srcs[buf]
        t0 = time.perf_counter()
        self.fn(self.src, self.dump, nmb=self.nmb, variant=self.variant)
        self.ms = (time.perf_counter() - t0) * 1e3
        return self.result()

    def run(self, X, y, idx, init=False, seed=1, lr=0.4):
        self.prepare(X, y, idx, init, seed, lr)
        return self.call()

    def result(self):
        raw = self.dump.numpy().view(np.uint8).reshape(COLS, NDUMP, ROWS, DOBJ)
        return self._parse(raw)

    @staticmethod
    def _parse(raw):
        per_core = np.zeros((NCORE, NDUMP * DOBJ), np.uint8)
        for k in range(NCORE):
            c, r = snake(k); per_core[k] = raw[c, :, r, :].reshape(-1)
        model, stats = Model.from_dump(per_core)
        return model, stats


# ------------------------------------------------------------------------------------------------- CLI
def rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def selftest():
    Xtr, ytr, Xte, yte = load_mnist()
    t1 = TrainerNPU(1)
    idx = np.arange(B)
    m0, st = t1.run(Xtr, ytr, idx, init=True, seed=7, lr=0.0)
    ref = Model.init(7)
    ok_init = all(np.array_equal(a, b) for a, b in ((m0.W1, ref.W1), (m0.W2, ref.W2), (m0.b1, ref.b1), (m0.b2, ref.b2)))
    print(f"init + zero-lr step: weights {'bit-exact' if ok_init else 'DIFFER'} vs the numpy replica (W1 max diff {np.abs(m0.W1 - ref.W1).max():.3g}); magic ok {st[1]}; correct {st[0]}/{B}")
    m1, st = t1.run(Xtr, ytr, idx, init=True, seed=7, lr=0.4)
    em = Model.init(7); corr = em.step(Xtr[:B], ytr[:B], 0.4, emulate=True)
    ex = Model.init(7); ex.step(Xtr[:B], ytr[:B], 0.4, emulate=False)
    for name in ("W1", "b1", "W2", "b2"):
        d_npu = getattr(m1, name) - getattr(ref, name); d_em = getattr(em, name) - getattr(ref, name); d_ex = getattr(ex, name) - getattr(ref, name)
        print(f"   one step, delta {name}: NPU vs bf16 replica {rel(d_npu, d_em):.2e}, NPU vs float32 {rel(d_npu, d_ex):.2e}, replica vs float32 {rel(d_em, d_ex):.2e}")
    print(f"   correct in the minibatch: NPU {st[0]}, numpy {int(corr)}")
    return ok_init


def train(epochs, seed=1, lr=0.4, verbose=True):
    """Epoch e+1's stream is written by a second thread into the other buffer while the NPU runs epoch e."""
    from concurrent.futures import ThreadPoolExecutor
    Xtr, ytr, Xte, yte = load_mnist()
    nmb = len(Xtr) // B
    t = TrainerNPU(nmb, nbuf=2); rng = np.random.default_rng(seed); hist = []
    perms = [rng.permutation(len(Xtr)) for _ in range(epochs)]
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(t.prepare, Xtr, ytr, perms[0], True, seed, lr, 0)
    t_start = time.perf_counter(); npu_total = 0.0
    for ep in range(epochs):
        nxt.result()
        if ep + 1 < epochs: nxt = pool.submit(t.prepare, Xtr, ytr, perms[ep + 1], False, seed, lr, (ep + 1) % 2)
        model, st = t.call(ep % 2); npu_total += t.ms
        if not st[2]: sys.exit("the NPU lost its weights between calls (autosuspend after 5 s idle?)")
        acc = model.accuracy(Xte, yte); hist.append(acc)
        if verbose:
            print(f"epoch {ep + 1}: test {acc * 100:.2f}%  train (on the fly) {st[0] / (nmb * B) * 100:.2f}%  NPU {t.ms:.1f} ms")
    if verbose:
        print(f"{epochs} epochs: {time.perf_counter() - t_start:.2f} s wall clock, {npu_total:.0f} ms of it in NPU calls (the first includes loading the design)")
    return model, hist


CLASS_NAMES = [str(d) for d in range(10)] + list("ABCDEF")


def to_mnist(img):
    """A drawing (float 0..1, any size) the way MNIST was made: the ink's bounding box scaled into 20x20, centred by
    centre of mass in 28x28. Returns uint8 (784,) or None for an empty drawing."""
    import cv2
    ys, xs = np.nonzero(img > 0.1)
    if len(ys) == 0: return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]; h, w = crop.shape; s_ = 20.0 / max(h, w)
    crop = cv2.resize(crop, (max(1, round(w * s_)), max(1, round(h * s_))), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.float32); h, w = crop.shape
    out[(28 - h) // 2:(28 - h) // 2 + h, (28 - w) // 2:(28 - w) // 2 + w] = crop
    cy, cx = np.average(np.indices(out.shape), axis=(1, 2), weights=out + 1e-9)
    out = cv2.warpAffine(out, np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]]), (28, 28))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8).reshape(784)


def augment(x, rng):
    """A taught example, slightly moved, turned and scaled."""
    import cv2
    a = np.deg2rad(rng.uniform(-12, 12)); sc = rng.uniform(0.85, 1.12); tx, ty = rng.uniform(-2, 2, 2)
    M_ = np.float32([[sc * np.cos(a), -sc * np.sin(a), 0], [sc * np.sin(a), sc * np.cos(a), 0]])
    M_[:, 2] = np.array([14, 14]) - M_[:, :2] @ np.array([14, 14]) + [tx, ty]
    return cv2.warpAffine(x.reshape(28, 28), M_, (28, 28)).reshape(784)


def window(lr=0.4, seed=1, nmb=47, seconds=None, shot=None):
    """Watch it learn: first-layer weights, test accuracy, a drawing pad classified with the live weights, and new
    symbols taught by drawing them (keys A-F; 0-9 adds an example of a digit)."""
    import pygame
    Xtr, ytr, Xte, yte = load_mnist()
    t = TrainerNPU(nmb); rng = np.random.default_rng(seed)
    pygame.init(); W, H = 1280, 720
    screen = pygame.display.set_mode((W, H)); pygame.display.set_caption("The NPU learns MNIST")
    font = pygame.font.SysFont(None, 24); small = pygame.font.SysFont(None, 20); big = pygame.font.SysFont(None, 64)
    pad = np.zeros((280, 280), np.float32); pad_rect = pygame.Rect(930, 352, 280, 280)
    perm = rng.permutation(len(Xtr)); pos = 0; seen = 0; hist = []; t0 = time.perf_counter()
    model = None; init = True; paused = False; cur_lr = lr; npu_ms = []; last_call = 0.0
    sub = rng.choice(len(Xte), 2000, replace=False)
    taught = {c: [] for c in range(NOUT)}; flash = ""; flash_t = 0.0

    def weights_surface(m):
        w = m.W1.T.reshape(NHID, 28, 28)
        a = np.abs(w).reshape(NHID, -1).max(1)[:, None, None] + 1e-9
        v = np.clip(w / a * 1.6, -1, 1)[..., None]                    # dark = 0, orange = +, blue = -
        rgb = np.maximum(v, 0) * np.array([255, 150, 40]) + np.maximum(-v, 0) * np.array([60, 150, 255]) + 12
        tiles = np.full((16, 29, 16, 29, 3), 30.0)                      # 1 px gutters
        tiles[:, :28, :, :28] = rgb.reshape(16, 16, 28, 28, 3).transpose(0, 2, 1, 3, 4)
        tiles = np.clip(tiles.reshape(464, 464, 3), 0, 255).astype(np.uint8)
        return pygame.transform.smoothscale(pygame.surfarray.make_surface(np.transpose(tiles, (1, 0, 2))), (672, 672))

    def batch():
        """The next call's images: MNIST in order of a shuffle, with an eighth of them taught examples if any."""
        nonlocal perm, pos
        n = nmb * B
        if pos + n > len(Xtr): perm = rng.permutation(len(Xtr)); pos = 0
        idx = perm[pos:pos + n]; pos += n
        ex = [(x, c) for c in taught for x in taught[c]]
        if not ex: return Xtr, ytr, idx
        X = Xtr[idx].copy(); y = ytr[idx].copy()
        k = n // 8; slots = rng.choice(n, k, replace=False)
        pick = rng.integers(0, len(ex), k)
        X[slots] = np.stack([augment(ex[i][0], rng) for i in pick]); y[slots] = [ex[i][1] for i in pick]
        return X, y, np.arange(n)

    def draw_stroke(points, width=20):
        import cv2
        for a_, b_ in zip(points[:-1], points[1:]):
            cv2.line(pad, (int(a_[0]), int(a_[1])), (int(b_[0]), int(b_[1])), 1.0, width)

    demo = []
    if shot:                                                            # the picture: teach a star as "A", then draw one
        import cv2
        def star(cx, cy, r, rot):
            pts = [(cx + r * (1 if i % 2 == 0 else 0.42) * np.sin(rot + i * np.pi / 5), cy - r * (1 if i % 2 == 0 else 0.42) * np.cos(rot + i * np.pi / 5)) for i in range(11)]
            return pts
        demo = [("teach", 10, star(140 + dx, 140 + dy, r, rot)) for dx, dy, r, rot in
                ((0, 0, 110, 0), (-10, 5, 95, 0.1), (8, -6, 100, -0.08), (0, 10, 85, 0.05), (5, 0, 105, -0.12))]
        demo += [("teach", 7, [(70, 70), (210, 70), (120, 230)])]
        demo += [("show", None, star(145, 135, 100, 0.06))]

    drawing = False; last_xy = None; t_demo = t0
    while True:
        if seconds and time.perf_counter() - t0 > seconds:
            if shot: pygame.image.save(screen, shot)
            print(f"{seen / 60000:.2f} epochs in {time.perf_counter() - t0:.1f} s, test (2000) {hist[-1][1] * 100:.2f}%, NPU {np.mean(npu_ms):.2f} ms per call of {nmb * B} images")
            if model is not None:
                x = to_mnist(pad)
                if x is not None:
                    z = model.logits(x[None])[0]; print("pad classified as", CLASS_NAMES[int(z.argmax())])
            return
        if demo and time.perf_counter() - t_demo > 1.2:                  # scripted teaching for the screenshot
            kind, cls_, pts = demo.pop(0); pad[:] = 0; draw_stroke(pts); t_demo = time.perf_counter()
            if kind == "teach":
                x = to_mnist(pad); taught[cls_].append(x); pad[:] = 0; flash = f"taught {CLASS_NAMES[cls_]}"; flash_t = time.perf_counter()
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: paused = not paused
                elif e.key == pygame.K_r: init = True; hist = []; seen = 0; t0 = time.perf_counter(); taught = {c: [] for c in range(NOUT)}
                elif e.key in (pygame.K_BACKSPACE, pygame.K_DELETE): pad[:] = 0
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS): cur_lr *= 2
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS): cur_lr /= 2
                else:
                    name = pygame.key.name(e.key).upper()
                    if name in CLASS_NAMES:
                        x = to_mnist(pad)
                        if x is not None:
                            c = CLASS_NAMES.index(name); taught[c].append(x); pad[:] = 0
                            flash = f"example {len(taught[c])} of {name} added"; flash_t = time.perf_counter()
            if e.type == pygame.MOUSEBUTTONDOWN and pad_rect.collidepoint(e.pos):
                if e.button == 3: pad[:] = 0
                else: drawing = True; last_xy = None
            if e.type == pygame.MOUSEBUTTONUP: drawing = False
        if drawing:
            mx, my = pygame.mouse.get_pos()
            if pad_rect.collidepoint(mx, my):
                xy = (mx - pad_rect.x, my - pad_rect.y)
                draw_stroke([last_xy or xy, xy]); last_xy = xy
        now = time.perf_counter()
        if not paused or now - last_call > 1.0:                        # paused: lr 0 calls keep the weights alive
            X, y, idx = batch()
            model, st = t.run(X, y, idx, init=init, seed=seed, lr=0.0 if paused else cur_lr); last_call = now
            if not st[2] and not init: init = True; continue
            init = False
            if not paused:
                seen += nmb * B; npu_ms.append(t.ms)
                hist.append((seen, model.accuracy(Xte[sub], yte[sub])))
        screen.fill((16, 16, 20))
        if model is None: pygame.display.flip(); continue
        screen.blit(weights_surface(model), (24, 24))
        screen.blit(small.render("first-layer weights of the 256 hidden units; core k owns units 8k..8k+7", True, (140, 140, 150)), (24, 700))
        gx, gy, gw, gh = 770, 70, 480, 200
        pygame.draw.rect(screen, (60, 60, 70), (gx, gy, gw, gh), 1)
        for a in (0.9, 0.95, 1.0):
            yy = gy + gh - (a - 0.8) / 0.2 * gh
            pygame.draw.line(screen, (45, 45, 55), (gx, yy), (gx + gw, yy)); screen.blit(font.render(f"{a:.0%}", True, (120, 120, 130)), (gx - 50, yy - 8))
        lx = lambda n: gx + (np.log10(max(n, 1000)) - 3) / (np.log10(max(hist[-1][0] if hist else 1, 600000)) - 3) * gw
        for n_, lab_ in ((1000, "1k"), (10000, "10k"), (60000, "1 epoch"), (600000, "10 epochs")):
            if hist and lx(n_) + 8 * len(lab_) <= gx + gw: screen.blit(font.render(lab_, True, (110, 110, 120)), (lx(n_) - 10, gy + gh + 4))
        if len(hist) > 1:
            pts = [(lx(s_), gy + gh - min(max((a_ - 0.8) / 0.2, 0), 1) * gh) for s_, a_ in hist]
            pygame.draw.lines(screen, (90, 200, 255), False, pts, 2)
        acc = hist[-1][1] if hist else 0
        screen.blit(big.render(f"{acc * 100:.1f}%", True, (240, 240, 240)), (gx, 12))
        screen.blit(small.render("MNIST test accuracy, images seen (log)", True, (140, 140, 150)), (gx + 200, 38))
        ms = np.mean(npu_ms[-20:]) if npu_ms else 0
        info = f"{seen / 60000:.1f} epochs in {now - t0:.0f} s   NPU {ms:.1f} ms per call of {nmb * B} images   lr {cur_lr:g}" + ("   PAUSED" if paused else "")
        screen.blit(font.render(info, True, (200, 200, 200)), (724, gy + gh + 30))
        nt = {CLASS_NAMES[c]: len(v) for c, v in taught.items() if v}
        screen.blit(font.render("taught: " + ("  ".join(f"{k} x{v}" for k, v in nt.items()) if nt else "nothing yet (draw, then press A-F or a digit)"), True, (200, 180, 120)), (724, gy + gh + 54))
        if flash and now - flash_t < 2: screen.blit(font.render(flash, True, (255, 220, 120)), (930, 330))
        pygame.draw.rect(screen, (40, 40, 48), pad_rect)
        surf = pygame.surfarray.make_surface(np.transpose(np.repeat((pad * 255).astype(np.uint8)[:, :, None], 3, 2), (1, 0, 2)))
        screen.blit(surf, pad_rect.topleft)
        screen.blit(small.render("draw; A-F teaches a new symbol, 0-9 a digit", True, (150, 150, 160)), (724, 650))
        screen.blit(small.render("right click / backspace clears, SPACE pause, R restart, +/- lr", True, (150, 150, 160)), (724, 670))
        x = to_mnist(pad)
        if x is not None:
            z = model.logits(x[None])[0]; p = np.exp(z - z.max()); p /= p.sum()
            for d in range(NOUT):
                yb = 352 + d * 17
                pygame.draw.rect(screen, (90, 200, 255) if d == p.argmax() else (70, 90, 110), (750, yb + 3, int(p[d] * 150), 12))
                screen.blit(small.render(CLASS_NAMES[d], True, (220, 220, 220)), (730, yb))
            screen.blit(big.render(CLASS_NAMES[int(p.argmax())], True, (255, 220, 120)), (1222, 362))
        pygame.display.flip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bench", action="store_true", help="time one epoch's dispatch 6x back to back (warm clocks)")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--seconds", type=float, help="end the window after this many seconds")
    ap.add_argument("--shot", help="save a screenshot of the window at the end (with --seconds)")
    a = ap.parse_args()
    if a.selftest: selftest()
    if a.epochs:
        train(a.epochs, a.seed, a.lr)

    if a.bench:
        Xtr, ytr, _, _ = load_mnist(); t = TrainerNPU(len(Xtr) // B)
        t.run(Xtr, ytr, np.random.default_rng(0).permutation(len(Xtr)), init=True, lr=a.lr)
        ms = []
        for _ in range(6): t.again(); ms.append(t.ms)
        print(f"one epoch (937 minibatches of 64) per dispatch, back to back: min {min(ms):.1f} ms, median {np.median(ms):.1f} ms  {[round(m, 1) for m in ms]}")
    if a.window: window(a.lr, a.seed, seconds=a.seconds, shot=a.shot)
    if not (a.selftest or a.epochs or a.window or a.bench): ap.print_help()


if __name__ == "__main__":
    main()
