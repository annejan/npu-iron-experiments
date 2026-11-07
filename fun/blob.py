#!/usr/bin/env python3
"""Blobs: four spheres and a floor joined with smooth-min, sphere traced on all 32 AIE cores.

  ../env.sh python fun/blob.py            bench, write blob.png
  ../env.sh python fun/blob.py --window   pygame window
  ../env.sh python fun/blob.py --check    static loop report

The "bloep": smin(a, b, k) = min(a, b) - h*h*k/4 with h = max(k - |a - b|, 0) / k. Every op is
native in bf16 (sub, abs, max, mul, msc). The normal is a numeric gradient: three extra distance
evaluations at the hit point only, not per step. Frame plumbing, window and bench come from sdf.py.
"""
import argparse, sys
from pathlib import Path
import numpy as np
from aie.iron import ExternalFunction
from sdf import SDF, STEPS, W, bench, window

NPRM = 32  # 0..7 camera + light as in sdf.py, 8.. four spheres (x, y, z, r), 24 smooth k, 25 1/(4k)
NSPH = 4


def ksrc(steps):
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define STEPS {steps}
#define NSPH {NSPH}
using bf = bfloat16;
using V = aie::vector<bf, 32>;
using A = aie::accum<accfloat, 32>;
static inline A acc(float f) {{ A a; a.from_vector(aie::broadcast<float, 32>(f)); return a; }}
static inline V vb(float f) {{ return aie::broadcast<bf, 32>((bf)f); }}
static inline V tobf(const A &a) {{ return a.template to_vector<bf>(); }}
alignas(64) static const float RAMP[32] = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}};

struct Ray {{ A px, py, pz, t; V dxb, dyb, dzb, dxl, dyl, dzl, d; }};

static inline V sphere(const A &px, const A &py, const A &pz, const float *c) {{
    V qx = tobf(aie::sub(px, acc(c[0]))), qy = tobf(aie::sub(py, acc(c[1]))), qz = tobf(aie::sub(pz, acc(c[2])));
    A s2 = aie::mul(qx, qx); s2 = aie::mac(s2, qy, qy); s2 = aie::mac(s2, qz, qz);
    V s2b = tobf(s2);
    return tobf(aie::add(aie::mul(s2b, aie::invsqrt(s2b)), acc(-c[3])));
}}
/* smooth union, polynomial: min(a,b) - h*h*k/4, h = max(k - |a-b|, 0)/k  ==  min(a,b) - hk*hk/(4k), hk = max(k-|a-b|,0) */
static inline A smin(const V &a, const V &b, const V &k, const V &inv4k) {{
    V hk = aie::max(aie::sub(k, aie::abs(aie::sub(a, b))), vb(0.0f));
    A r = aie::mul(aie::min(a, b), vb(1.0f));
    return aie::msc(r, tobf(aie::mul(hk, hk)), inv4k);
}}
static inline A dist(const A &px, const A &py, const A &pz, const float *prm) {{
    const V k = vb(prm[24]), inv4k = vb(prm[25]);
    V d = sphere(px, py, pz, prm + 8);
    d = tobf(smin(d, sphere(px, py, pz, prm + 12), k, inv4k));
    d = tobf(smin(d, sphere(px, py, pz, prm + 16), k, inv4k));
    d = tobf(smin(d, sphere(px, py, pz, prm + 20), k, inv4k));
    return smin(d, tobf(aie::add(py, acc(1.5f))), k, inv4k);          // floor at y = -1.5, also blended
}}
static inline void step(Ray &r, const float *prm) {{
    r.d = tobf(dist(r.px, r.py, r.pz, prm));
    V d = aie::min(r.d, vb(8.0f));
    r.px = aie::mac(aie::mac(r.px, r.dxb, d), r.dxl, d);
    r.py = aie::mac(aie::mac(r.py, r.dyb, d), r.dyl, d);
    r.pz = aie::mac(aie::mac(r.pz, r.dzb, d), r.dzl, d);
    r.t = aie::mac(r.t, d, vb(1.0f));
}}
static inline void init(Ray &r, const A &u, float vf, float ca, float sa, const float *prm) {{
    V ub = tobf(u);
    A dx = aie::mac(acc(sa), ub, vb(ca));
    A dz = aie::mac(acc(ca), ub, vb(-sa));
    V dxb = tobf(dx), dyb = vb(vf), dzb = tobf(dz);
    A s = aie::mul(dxb, dxb); s = aie::mac(s, dyb, dyb); s = aie::mac(s, dzb, dzb);
    V inv = aie::invsqrt(tobf(s));
    const V one = vb(1.0f);
    A nx = aie::mul(dxb, inv), ny = aie::mul(dyb, inv), nz = aie::mul(dzb, inv);
    r.dxb = tobf(nx); r.dyb = tobf(ny); r.dzb = tobf(nz);
    r.dxl = tobf(aie::sub(nx, aie::mul(r.dxb, one)));
    r.dyl = tobf(aie::sub(ny, aie::mul(r.dyb, one)));
    r.dzl = tobf(aie::sub(nz, aie::mul(r.dzb, one)));
    r.px = acc(prm[0]); r.py = acc(prm[1]); r.pz = acc(prm[2]); r.t = acc(0.0f);
}}
static inline void shade(const Ray &r, int16_t *__restrict o16, const float *prm) {{
    const V Lx = vb(prm[5]), Ly = vb(prm[6]), Lz = vb(prm[7]);
    // numeric normal: forward differences in fp32 (bf16 distances quantise to rings), eps 0.08;
    // g2 is kept away from 0 so a sky ray's flat gradient does not become inf * 0 = NaN
    const A ex = acc(0.08f);
    A d0a = dist(r.px, r.py, r.pz, prm);
    V d0 = tobf(d0a);
    V gx = tobf(aie::sub(dist(aie::add(r.px, ex), r.py, r.pz, prm), d0a));
    V gy = tobf(aie::sub(dist(r.px, aie::add(r.py, ex), r.pz, prm), d0a));
    V gz = tobf(aie::sub(dist(r.px, r.py, aie::add(r.pz, ex), prm), d0a));
    A g2 = acc(1e-6f); g2 = aie::mac(g2, gx, gx); g2 = aie::mac(g2, gy, gy); g2 = aie::mac(g2, gz, gz);
    V inv = aie::invsqrt(tobf(g2));
    V nx = tobf(aie::mul(gx, inv)), ny = tobf(aie::mul(gy, inv)), nz = tobf(aie::mul(gz, inv));
    A dots = aie::mul(nx, Lx); dots = aie::mac(dots, ny, Ly); dots = aie::mac(dots, nz, Lz);
    V diff = aie::max(tobf(dots), vb(0.0f));
    // specular: half vector of light and view (-dir); (n.h)^16 by squaring four times
    A hx = aie::mul(Lx, vb(1.0f)); hx = aie::msc(hx, r.dxb, vb(1.0f));
    A hy = aie::mul(Ly, vb(1.0f)); hy = aie::msc(hy, r.dyb, vb(1.0f));
    A hz = aie::mul(Lz, vb(1.0f)); hz = aie::msc(hz, r.dzb, vb(1.0f));
    V hxb = tobf(hx), hyb = tobf(hy), hzb = tobf(hz);
    A h2 = aie::mul(hxb, hxb); h2 = aie::mac(h2, hyb, hyb); h2 = aie::mac(h2, hzb, hzb);
    V hinv = aie::invsqrt(tobf(h2));
    A nh = aie::mul(nx, hxb); nh = aie::mac(nh, ny, hyb); nh = aie::mac(nh, nz, hzb);
    V spec = aie::max(tobf(aie::mul(tobf(nh), hinv)), vb(0.0f));
    spec = tobf(aie::mul(spec, spec)); spec = tobf(aie::mul(spec, spec));
    spec = tobf(aie::mul(spec, spec)); spec = tobf(aie::mul(spec, spec));
    // floor-ish vs blob colour from the normal's y (the floor is blended in, so no mask): white floor, orange blob
    V up = aie::max(ny, vb(0.0f));                         // 1 on the floor, lower on blob sides
    V tb = tobf(r.t);
    V fog = aie::inv(tobf(aie::mac(acc(1.0f), tb, vb(0.06f))));
    V eps = tobf(aie::mac(acc(0.01f), tb, vb(0.004f)));
    V hit = aie::select(vb(0.0f), vb(1.0f), aie::lt(d0, eps) & aie::lt(tb, vb(60.0f)));
    fog = tobf(aie::mul(fog, hit));
    V shade = tobf(aie::mac(acc(0.12f), diff, vb(0.88f)));
    V skyw = tobf(aie::add(aie::mul(fog, vb(-1.0f)), acc(1.0f)));
    // base colour: lerp(orange, light grey, up^4)
    V u2 = tobf(aie::mul(up, up)); u2 = tobf(aie::mul(u2, u2));
    V br = tobf(aie::mac(aie::mul(u2, vb(200.0f - 255.0f)), vb(255.0f), vb(1.0f)));
    V bg = tobf(aie::mac(aie::mul(u2, vb(200.0f - 90.0f)), vb(90.0f), vb(1.0f)));
    V bb = tobf(aie::mac(aie::mul(u2, vb(210.0f - 40.0f)), vb(40.0f), vb(1.0f)));
    V sp = tobf(aie::mul(spec, vb(180.0f)));
    V cr = tobf(aie::mac(aie::mul(tobf(aie::add(aie::mul(br, shade), aie::mul(sp, vb(1.0f)))), fog), skyw, vb(120.0f)));
    V cg = tobf(aie::mac(aie::mul(tobf(aie::add(aie::mul(bg, shade), aie::mul(sp, vb(1.0f)))), fog), skyw, vb(160.0f)));
    V cb = tobf(aie::mac(aie::mul(tobf(aie::add(aie::mul(bb, shade), aie::mul(sp, vb(1.0f)))), fog), skyw, vb(220.0f)));
    auto ri = aie::to_fixed<int16_t>(cr, 0), gi = aie::to_fixed<int16_t>(cg, 0), bi = aie::to_fixed<int16_t>(cb, 0);
    const auto lo = aie::zeros<int16_t, 32>(), hi = aie::broadcast<int16_t, 32>(255);
    ri = aie::min(aie::max(ri, lo), hi); gi = aie::min(aie::max(gi, lo), hi); bi = aie::min(aie::max(bi, lo), hi);
    auto rg = aie::bit_or(ri, aie::upshift(gi, 8));
    auto ba = aie::bit_or(bi, aie::broadcast<int16_t, 32>((int16_t)0xFF00));
    auto [p0, p1] = aie::interleave_zip(rg, ba, 1);
    aie::store_v(o16, p0);
    aie::store_v(o16 + 32, p1);
}}

extern "C" {{
void blob(float *__restrict prm, int32_t *__restrict out, int32_t y) {{
    const float ca = prm[3], sa = prm[4];
    const float vf = (540.0f - (float)y) / 540.0f;
    A rampa; rampa.from_vector(aie::load_v<32>(RAMP));
    const V ramp = tobf(rampa);
    const A ustep = acc(32.0f / 540.0f);
    A ubase = acc(-960.0f / 540.0f);
    int16_t *__restrict o16 = (int16_t *)out;
    for (unsigned i = 0; i < W / 32; i += 2) {{
        Ray a, b;
        init(a, aie::mac(ubase, ramp, vb(1.0f / 540.0f)), vf, ca, sa, prm);
        ubase = aie::add(ubase, ustep);
        init(b, aie::mac(ubase, ramp, vb(1.0f / 540.0f)), vf, ca, sa, prm);
        ubase = aie::add(ubase, ustep);
        for (int k = 0; k < STEPS; k++) {{ step(a, prm); step(b, prm); }}
        shade(a, o16 + 64 * i, prm);
        shade(b, o16 + 64 * i + 64, prm);
    }}
}}
}}
"""


def ext(steps):
    prm_ty = np.ndarray[(NPRM,), np.dtype[np.float32]]
    row_ty = np.ndarray[(W,), np.dtype[np.int32]]
    return ExternalFunction("blob", source_string=ksrc(steps), arg_types=[prm_ty, row_ty, np.dtype(np.int32)])


def params(t):
    p = np.zeros(NPRM, dtype=np.float32)
    p[0], p[1], p[2] = 0.0, 0.6, -7.0         # camera, looking down +z
    p[3], p[4] = 1.0, 0.0
    L = np.array([0.5, 0.8, -0.6]); L /= np.linalg.norm(L)
    p[5:8] = L
    k = 0.9
    p[24], p[25] = k, 1.0 / (4 * k)
    for i in range(NSPH):
        ph = t * (0.7 + 0.2 * i) + i * 1.9
        p[8 + 4 * i] = 1.8 * np.sin(ph) * np.cos(t * 0.3 + i)
        p[9 + 4 * i] = -0.4 + 1.3 * np.sin(ph * 1.3 + i)          # dips into the floor at -1.5
        p[10 + 4 * i] = 1.2 * np.cos(ph * 0.8)
        p[11 + 4 * i] = 0.7 + 0.15 * i
    return p


def check(steps):
    from aie.utils.compile import remarks
    d = Path(".remarks/rem_blob")
    d.mkdir(parents=True, exist_ok=True)
    rep, why = remarks.analyze(ext(steps), "aie2p", d)
    if rep is None:
        print(why); sys.exit(1)
    print("libcalls:", rep.libcalls, "stack:", rep.kernel_stack_bytes, "notes:", rep.schedule_notes)
    for key, l in rep.loops.items():
        print("  loop", key, "ii", l.ii, "ns", l.ns, "zol", l.zol, "trips", l.trips, "bundles", l.bundle_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--steps", type=int, default=48)
    a = ap.parse_args()
    if a.check:
        return check(a.steps)
    scene = SDF(a.steps, func=ext(a.steps), nprm=NPRM, params=params, variant=f"blob3-{a.steps}")
    if a.window:
        return window(scene, f"blobs {a.steps} steps")
    bench(scene, f"blobs {a.steps} steps", f"blob{a.steps}.png", a.frames)


if __name__ == "__main__":
    main()
