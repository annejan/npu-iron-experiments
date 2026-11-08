#!/usr/bin/env python3
"""The NPU learns: MNIST training (not just inference) on all 32 cores, one dispatch per epoch.

  ./env.sh python fun/mnist_npu.py --selftest        init bit-exact vs numpy, one step vs a numpy replica, 1 epoch
  ./env.sh python fun/mnist_npu.py --epochs 5        train, test accuracy after every epoch, NPU time per epoch
  ./env.sh python fun/mnist_npu.py --window          watch it learn: weights, accuracy curve, draw a digit (mouse)

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
CPB = 14                        # pixel blocks per chunk
NCH = NPB // CPB                # chunks per pass
XOBJ = CPB * NBB * 64           # bytes of a stream object (7168)
NHID, UPC = 256, 8              # hidden units, units per core
NOUT = 16                       # logits padded to 16 (10 used)
DOBJ = 3136                     # dump object bytes (W1 slice = 8 objects)
NDUMP = 9
COLS, ROWS = 8, 4
NCORE = COLS * ROWS
MAGIC = 0x4D4E4953
DATA = Path.home() / "models" / "mnist"
OBJ_PER_MB = 1 + 2 * NCH


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


def stream_into(buf, X, y, idx, init, seed, lr):
    """Write one call's byte stream into buf (uint8): control object, then per minibatch the labels and the images
    twice as 8x8 tiles [pb][bb][8 b][8 px] (the forward pass transposes them in the kernel). Returns the minibatch count."""
    nmb = len(idx) // B
    v = buf[XOBJ:XOBJ + nmb * OBJ_PER_MB * XOBJ].reshape(nmb, OBJ_PER_MB, XOBJ)
    xs = X[idx[:nmb * B]]                                                # [m*B + bb*8 + b][784]
    t = np.ascontiguousarray(xs.view(np.uint64).reshape(nmb, NBB, 8, NPB).transpose(0, 3, 1, 2))   # [m][pb][bb][b] of 8 px
    t = t.view(np.uint8).reshape(nmb, NCH, XOBJ)
    v[:, 1:1 + NCH] = t; v[:, 1 + NCH:] = t
    v[:, 0, :B] = y[idx[:nmb * B]].reshape(nmb, B)
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
        W1 = np.zeros((784, NHID), np.float32); W2 = np.zeros((NHID, 10), np.float32)
        for k in range(NCORE):
            w1, s = xorshift_init(seed, k, NPB * 64, 27)
            W1[:, k * UPC:(k + 1) * UPC] = np.transpose(w1.reshape(NPB, UPC, 8), (0, 2, 1)).reshape(784, UPC)
            w2 = np.zeros(128, np.float32)
            for i in range(4):
                s ^= s << np.uint32(13); s ^= s >> np.uint32(17); s ^= s << np.uint32(5)
                w2[i * 32:(i + 1) * 32] = (s.view(np.int32) >> 8).astype(np.float32) * np.float32(2.0 ** -26)
            w2 = w2.reshape(16, UPC)                                    # [o][u]
            W2[k * UPC:(k + 1) * UPC] = w2[:10].T
        return cls(W1, np.zeros(NHID, np.float32), W2, np.zeros(10, np.float32))

    @classmethod
    def from_dump(cls, dump):
        """dump: (NCORE, NDUMP * DOBJ) bytes in snake order."""
        W1 = np.zeros((784, NHID), np.float32); W2 = np.zeros((NHID, 10), np.float32); b1 = np.zeros(NHID, np.float32)
        b2 = None; stats = None
        for k in range(NCORE):
            d = dump[k]
            w1 = d[:NPB * 64 * 4].view(np.float32).reshape(NPB, UPC, 8)          # [pb][u][p]
            W1[:, k * UPC:(k + 1) * UPC] = np.transpose(w1, (0, 2, 1)).reshape(784, UPC)
            t = d[8 * DOBJ:9 * DOBJ].view(np.float32)
            w2t = t[:128].reshape(16, UPC)                                      # [o][u]
            W2[k * UPC:(k + 1) * UPC] = w2t[:10].T
            b1[k * UPC:(k + 1) * UPC] = t[128:192].reshape(UPC, 8)[:, 0]
            if k == NCORE - 1:
                b2 = t[192:320].reshape(16, 8)[:10, 0].copy()
                stats = d[9 * DOBJ - 64:9 * DOBJ].view(np.int32)
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
#define CPB 14
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
alignas(64) static float b2T[2 * 64];            // [half][o][b] = b2[o]; unused logits sit at -1e4
alignas(64) static bf GT[NBB * 2 * 64];          // last core: its own copy of G
alignas(64) static bf ONES[64];
alignas(64) static int32_t lab[64];
alignas(64) static int32_t stats[16];
static int chunk, dumpi, cid_;
static int32_t alive;
static bf cscale;

static inline VB f2b(const float *p) {
    return aie::concat(AF(aie::load_v<32>(p)).template to_vector<bf>(), AF(aie::load_v<32>(p + 32)).template to_vector<bf>());
}
static inline VB u2b(const uint8_t *p) {                  // pixels / 256, exact in bf16
    aie::vector<uint8, 64> v = aie::load_v<64>(p);
    return aie::concat(AF(aie::to_float<float>(v.template extract<32>(0), 8)).template to_vector<bf>(),
                       AF(aie::to_float<float>(v.template extract<32>(1), 8)).template to_vector<bf>());
}
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
    for (int l = 16; l < 64; l++) W2T[64 + l] = 0.f;      // logits 10..15 unused
    aie::store_v(b1T, aie::zeros<float, 32>()); aie::store_v(b1T + 32, aie::zeros<float, 32>());
    for (int l = 0; l < 128; l++) b2T[l] = (l >= 64 + 16) ? -1.0e4f : 0.f;
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
    stats[1] = c[0] == MAGIC; stats[2] = alive == 0x600D;      // stats[2] = 0: the weights were lost (device reset)
    refresh_w2();
    dumpi = 0;
}

extern "C" void k_lab(int8_t *x) {
    for (int i = 0; i < 64; i++) lab[i] = ((const uint8_t *)x)[i];
    chunk = 0;
}

// ---- forward chunk: H^T[bb] (+)= W1^T[pb] X^T[pb][bb] over the chunk's 14 pixel blocks (tiles arrive as [b][p])
extern "C" void k_fwd(int8_t *x) {
#ifdef NOCOMPUTE
    chunk++; return;
#endif
    const uint8_t *xt = (const uint8_t *)x;
    const int pb0 = chunk * CPB;
    for (int p = 0; p < CPB; p++) aie::store_v(W1Tb + 64 * p, f2b(W1T + 64 * (pb0 + p)));
    for (int bb = 0; bb < NBB; bb++) {
        MM c;
        for (int p = 0; p < CPB; p++) c.mac(aie::load_v<64>(W1Tb + 64 * p), aie::transpose(u2b(xt + (p * NBB + bb) * 64), 8, 8));
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

extern "C" void k_part(float *so) { for (int bb = 0; bb < NBB; bb++) partial_bb(bb, nullptr, so + bb * 128, false); }
extern "C" void k_add(float *si, float *so) {           // the chain hop: so += si
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

extern "C" void k_grad(bf *g) { grad(g, false); }

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

extern "C" void k_lpart(float *L) { for (int bb = 0; bb < NBB; bb++) partial_bb(bb, nullptr, L + bb * 128, true); }
extern "C" void k_last(float *si, float *L, bf *gout) {
    k_add(si, L); Lg = L;
    for (int pr = 0; pr < NBB / 2; pr++) softmax_pair(pr, gout);
    grad(GT, true);
}

// ---- backward chunk: W1^T[pb] += dh^T X[pb] (X tiles [bb][b][p])
extern "C" void k_bwd(int8_t *x) {
#ifdef NOCOMPUTE
    chunk++; return;
#endif
    const uint8_t *xb = (const uint8_t *)x;
    const int pb0 = chunk * CPB;
    for (int p = 0; p < CPB; p++) {
        MM c;
        for (int bb = 0; bb < NBB; bb++) c.mac(aie::load_v<64>(dhT + 64 * bb), u2b(xb + (p * NBB + bb) * 64));
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
    nobj = 1 + nmb * OBJ_PER_MB
    x_ty = np.ndarray[(XOBJ,), np.dtype[np.int8]]
    s_ty = np.ndarray[(NBB * 2 * 64,), np.dtype[np.float32]]
    g_ty = np.ndarray[(NBB * 2 * 64,), np.dtype[np.int16]]          # bf16 bits
    o_ty = np.ndarray[(DOBJ,), np.dtype[np.int8]]; o4_ty = np.ndarray[(ROWS * DOBJ,), np.dtype[np.int8]]
    i32 = np.dtype(np.int32)
    decl = "#include <stdint.h>\n"
    src = ("#define NOCOMPUTE\n" if os.environ.get("MNIST_NOCOMPUTE") else "") + KERNEL
    kctl = ExternalFunction("k_ctl", source_string=src, arg_types=[x_ty, i32])
    kf = {n: ExternalFunction(n, source_string=decl + f'extern "C" void {n}({sig});\n', arg_types=at) for n, sig, at in [
        ("k_lab", "int8_t *", [x_ty]), ("k_fwd", "int8_t *", [x_ty]), ("k_bwd", "int8_t *", [x_ty]),
        ("k_part", "float *", [s_ty]), ("k_add", "float *, float *", [s_ty, s_ty]), ("k_lpart", "float *", [s_ty]),
        ("k_last", "float *, float *, int16_t *", [s_ty, s_ty, g_ty]),
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

    def body_last(xf, si, go, of, L, kctl, klab, kfwd, kbwd, klpart, klast, kdump, cid):
        x = xf.acquire(1); kctl(x, cid); xf.release(1)

        def middle():
            klpart(L); a = si.acquire(1); g = go.acquire(1); klast(a, L, g); go.release(1); si.release(1)
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
            workers.append(Worker(body_last, [xb.cons(depth=2), chain[k - 1].cons(), gf.prod(), of, L] + common + [kf["k_lpart"], kf["k_last"], kf["k_dump"], k], tile=tile, dynamic_objfifo_lowering=True))

    words = nobj * XOBJ // 4; dwords = COLS * NDUMP * ROWS * DOBJ // 4
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

    def __init__(self, nmb):
        import aie.iron as iron
        self.nmb = nmb; self.fn = _jit()
        self.variant = "mnist-" + hashlib.md5((KERNEL + str(nmb) + str(B) + os.environ.get("MNIST_NOCOMPUTE", "")).encode()).hexdigest()[:10]
        self.src = iron.zeros(((1 + nmb * OBJ_PER_MB) * XOBJ // 4,), dtype=np.int32, device="npu")
        self.dump = iron.zeros((COLS * NDUMP * ROWS * DOBJ // 4,), dtype=np.int32, device="npu")
        self.ms = 0.0

    def run(self, X, y, idx, init=False, seed=1, lr=0.4):
        with self.src.overwrite() as v:
            nmb = stream_into(v.view(np.uint8), X, y, idx, init, seed, lr)
        assert nmb == self.nmb, (nmb, self.nmb)
        t0 = time.perf_counter()
        self.fn(self.src, self.dump, nmb=self.nmb, variant=self.variant)
        self.ms = (time.perf_counter() - t0) * 1e3
        raw = self.dump.numpy().view(np.uint8).reshape(COLS, NDUMP, ROWS, DOBJ)
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
    Xtr, ytr, Xte, yte = load_mnist()
    nmb = len(Xtr) // B
    t = TrainerNPU(nmb); rng = np.random.default_rng(seed); hist = []
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        th = time.perf_counter()
        model, st = t.run(Xtr, ytr, idx, init=(ep == 0), seed=seed, lr=lr)
        host = (time.perf_counter() - th) * 1e3 - t.ms
        acc = model.accuracy(Xte, yte); hist.append(acc)
        if verbose:
            print(f"epoch {ep + 1}: test {acc * 100:.2f}%  train (on the fly) {st[0] / (nmb * B) * 100:.2f}%  NPU {t.ms:.1f} ms per epoch (host stream prep {host:.0f} ms)")
    return model, hist


def window(lr=0.4, seed=1, nmb=47, seconds=None, shot=None):
    """Watch it learn: first-layer weights, test accuracy, and a drawing pad classified with the live weights."""
    import pygame
    Xtr, ytr, Xte, yte = load_mnist()
    t = TrainerNPU(nmb); rng = np.random.default_rng(seed)
    pygame.init(); W, H = 1280, 720
    screen = pygame.display.set_mode((W, H)); pygame.display.set_caption("The NPU learns MNIST")
    font = pygame.font.SysFont(None, 24); big = pygame.font.SysFont(None, 64)
    pad = np.zeros((280, 280), np.float32); pad_rect = pygame.Rect(900, 352, 280, 280)
    perm = rng.permutation(len(Xtr)); pos = 0; seen = 0; hist = []; t0 = time.perf_counter()
    model = None; init = True; paused = False; cur_lr = lr; npu_ms = []; last_call = 0.0
    sub = rng.choice(len(Xte), 2000, replace=False)

    def weights_surface(m):
        w = m.W1.T.reshape(NHID, 28, 28)
        a = np.abs(w).reshape(NHID, -1).max(1)[:, None, None] + 1e-9
        v = np.clip(w / a * 1.6, -1, 1)[..., None]                    # dark = 0, orange = +, blue = -
        rgb = np.maximum(v, 0) * np.array([255, 150, 40]) + np.maximum(-v, 0) * np.array([60, 150, 255]) + 12
        tiles = np.full((16, 29, 16, 29, 3), 30.0)                      # 1 px gutters
        tiles[:, :28, :, :28] = rgb.reshape(16, 16, 28, 28, 3).transpose(0, 2, 1, 3, 4)
        tiles = np.clip(tiles.reshape(464, 464, 3), 0, 255).astype(np.uint8)
        return pygame.transform.smoothscale(pygame.surfarray.make_surface(np.transpose(tiles, (1, 0, 2))), (672, 672))

    def pad_input():
        import cv2
        img = cv2.resize(pad, (20, 20), interpolation=cv2.INTER_AREA)
        if img.sum() < 1e-3: return None
        ys, xs = np.nonzero(img > 0.05); cy, cx = (np.average(np.indices(img.shape), axis=(1, 2), weights=img + 1e-9))
        out = np.zeros((28, 28), np.float32); oy, ox = int(round(14 - cy)), int(round(14 - cx))
        oy = min(max(oy, 0), 8); ox = min(max(ox, 0), 8); out[oy:oy + 20, ox:ox + 20] = img
        return (np.clip(out, 0, 1) * 255).astype(np.uint8).reshape(1, 784)

    drawing = False
    if shot:                                                            # a synthetic "7" on the pad for the picture
        import cv2
        cv2.line(pad, (70, 70), (210, 70), 1.0, 22); cv2.line(pad, (210, 70), (120, 230), 1.0, 22); pad[:] = cv2.GaussianBlur(pad, (0, 0), 3) * 1.6
        np.clip(pad, 0, 1, out=pad)
    while True:
        if seconds and time.perf_counter() - t0 > seconds:
            if shot: pygame.image.save(screen, shot)
            print(f"{seen / 60000:.2f} epochs in {time.perf_counter() - t0:.1f} s, test (2000) {hist[-1][1] * 100:.2f}%, NPU {np.mean(npu_ms):.2f} ms per call of {nmb * B} images")
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE: paused = not paused
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r: init = True; hist = []; seen = 0; t0 = time.perf_counter()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_c: pad[:] = 0
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS): cur_lr *= 2
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_MINUS, pygame.K_KP_MINUS): cur_lr /= 2
            if e.type == pygame.MOUSEBUTTONDOWN and pad_rect.collidepoint(e.pos):
                if e.button == 3: pad[:] = 0
                else: drawing = True
            if e.type == pygame.MOUSEBUTTONUP: drawing = False
        if drawing:
            mx, my = pygame.mouse.get_pos()
            if pad_rect.collidepoint(mx, my):
                yy, xx = np.ogrid[:280, :280]
                pad[:] = np.maximum(pad, np.clip(1.5 - np.hypot(yy - (my - pad_rect.y), xx - (mx - pad_rect.x)) / 9, 0, 1))
        now = time.perf_counter()
        if not paused or now - last_call > 1.0:                        # paused: lr 0 calls keep the weights alive
            if pos + nmb * B > len(Xtr): perm = rng.permutation(len(Xtr)); pos = 0
            idx = perm[pos:pos + nmb * B]
            model, st = t.run(Xtr, ytr, idx, init=init, seed=seed, lr=0.0 if paused else cur_lr); last_call = now
            if not st[2] and not init: init = True; continue
            init = False
            if not paused:
                pos += nmb * B; seen += nmb * B; npu_ms.append(t.ms)
                hist.append((seen, model.accuracy(Xte[sub], yte[sub])))
        screen.fill((16, 16, 20))
        if model is not None:
            screen.blit(weights_surface(model), (24, 24))
            gx, gy, gw, gh = 770, 70, 480, 220
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
            screen.blit(font.render("test accuracy, images seen (log scale)", True, (140, 140, 150)), (gx + 200, 34))
            ms = np.mean(npu_ms[-20:]) if npu_ms else 0
            info = f"{seen / 60000:.2f} epochs in {now - t0:.1f} s   NPU {ms:.1f} ms per {nmb * B} images   lr {cur_lr:g}" + ("   PAUSED" if paused else "")
            screen.blit(font.render(info, True, (200, 200, 200)), (724, gy + gh + 30))
            screen.blit(font.render("draw a digit (right click / C clears)   SPACE pause   R restart   +/- lr", True, (150, 150, 160)), (724, 668))
            screen.blit(font.render("first-layer weights of the 256 hidden units, one core owns 8 of them", True, (140, 140, 150)), (24, 700))
            pygame.draw.rect(screen, (40, 40, 48), pad_rect)
            surf = pygame.surfarray.make_surface(np.transpose(np.repeat((pad * 255).astype(np.uint8)[:, :, None], 3, 2), (1, 0, 2)))
            screen.blit(surf, pad_rect.topleft)
            x = pad_input()
            if x is not None:
                z = model.logits(x)[0]; p = np.exp(z - z.max()); p /= p.sum()
                for d in range(10):
                    yb = 352 + d * 28
                    pygame.draw.rect(screen, (90, 200, 255) if d == p.argmax() else (70, 90, 110), (750, yb + 4, int(p[d] * 130), 18))
                    screen.blit(font.render(str(d), True, (220, 220, 220)), (730, yb + 4))
                screen.blit(big.render(str(int(p.argmax())), True, (255, 220, 120)), (1200, 362))
        pygame.display.flip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--seconds", type=float, help="end the window after this many seconds")
    ap.add_argument("--shot", help="save a screenshot of the window at the end (with --seconds)")
    a = ap.parse_args()
    if a.selftest: selftest()
    if a.epochs: train(a.epochs, a.seed, a.lr)
    if a.window: window(a.lr, a.seed, seconds=a.seconds, shot=a.shot)
    if not (a.selftest or a.epochs or a.window): ap.print_help()


if __name__ == "__main__":
    main()
