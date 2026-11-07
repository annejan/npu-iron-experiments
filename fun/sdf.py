#!/usr/bin/env python3
"""Sphere-traced signed distance field at 1080p on all 32 AIE cores.

  ./env.sh python fun/sdf.py --check      compile the kernel statically, print loop IIs / errors
  ./env.sh python fun/sdf.py              bench, write sdf.png
  ./env.sh python fun/sdf.py --window     pygame window

Why this fits the NPU: every lane runs the same branchless op sequence, no memory lookups, pure ALU.
Numerics: ray positions live in fp32 accumulators (fp32 add is native); every multiply is bf16 x bf16
into an fp32 accumulator (native MAC); length uses bf16 invsqrt; min/max/select in bf16.
Scene: infinite grid of spheres (domain repetition via round(p/4)*4 with the magic-number trick,
fp32 adds only) over a floor plane, lambert shading, depth fog. STEPS fixed steps, no early exit.

Domain repetition and the distance functions after Inigo Quilez (iquilezles.org/articles/distfunctions); sphere tracing:
J. C. Hart, "Sphere tracing", The Visual Computer 12:527-545, 1996.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup
from aie.helpers.taplib import TensorAccessPattern

W, H = 1920, 1088
ROWS = 4
STEPS = 64
NPRM = 16
PERIOD = 4.0

def ksrc(STEPS):
    return f"""
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {W}
#define STEPS {STEPS}
using bf = bfloat16;
using V = aie::vector<bf, 32>;
using A = aie::accum<accfloat, 32>;
static inline A acc(float f) {{ A a; a.from_vector(aie::broadcast<float, 32>(f)); return a; }}
static inline V vb(float f) {{ return aie::broadcast<bf, 32>((bf)f); }}
static inline V tobf(const A &a) {{ return a.template to_vector<bf>(); }}
alignas(64) static const float RAMP[32] = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}};

struct Ray {{ A px, py, pz, t; V dxb, dyb, dzb, dxl, dyl, dzl, qxb, qyb, qzb, invlen, dsb, dfb; }};

/* distance evaluation: repeated unit spheres on a 4-grid in x,z over a floor at y=-1.5.
   round(p/4)*4 = (p + 1.5*2^25) - 1.5*2^25 in fp32 adds only (the ulp at 2^25 is 4) */
static inline void eval(Ray &r, const A &magic4) {{
    A rx = aie::sub(aie::add(r.px, magic4), magic4);
    A rz = aie::sub(aie::add(r.pz, magic4), magic4);
    r.qxb = tobf(aie::sub(r.px, rx)); r.qyb = tobf(r.py); r.qzb = tobf(aie::sub(r.pz, rz));
    A s2 = aie::mul(r.qxb, r.qxb); s2 = aie::mac(s2, r.qyb, r.qyb); s2 = aie::mac(s2, r.qzb, r.qzb);
    V s2b = tobf(s2);
    r.invlen = aie::invsqrt(s2b);
    r.dsb = tobf(aie::add(aie::mul(s2b, r.invlen), acc(-1.0f)));
    r.dfb = tobf(aie::add(r.py, acc(1.5f)));
}}
static inline void step(Ray &r, const V &one) {{
    V d = aie::min(aie::min(r.dsb, r.dfb), vb(8.0f));   // clamp: keeps p under 2^24 so the magic rounding stays exact
    // direction as bf16 hi + lo parts: one bf16 has 8 mantissa bits, too coarse (rows share a direction)
    r.px = aie::mac(aie::mac(r.px, r.dxb, d), r.dxl, d);
    r.py = aie::mac(aie::mac(r.py, r.dyb, d), r.dyl, d);
    r.pz = aie::mac(aie::mac(r.pz, r.dzb, d), r.dzl, d);
    r.t = aie::mac(r.t, d, one);
}}
static inline void init(Ray &r, const A &u, float vf, float ca, float sa, const float *prm) {{
    V ub = tobf(u);
    A dx = aie::mac(acc(sa), ub, vb(ca));
    A dz = aie::mac(acc(ca), ub, vb(-sa));
    V dxb = tobf(dx), dyb = vb(vf), dzb = tobf(dz);
    A s = aie::mul(dxb, dxb); s = aie::mac(s, dyb, dyb); s = aie::mac(s, dzb, dzb);
    A dy = acc(vf);
    V inv = aie::invsqrt(tobf(s));
    const V one = vb(1.0f);
    A nx = aie::mul(dxb, inv), ny = aie::mul(dyb, inv), nz = aie::mul(dzb, inv);
    // hi/lo of the fp32 products; dx, dy, dz themselves are only bf16-exact here (fine: the split keeps the normalised sum exact)
    r.dxb = tobf(nx); r.dyb = tobf(ny); r.dzb = tobf(nz);
    r.dxl = tobf(aie::sub(nx, aie::mul(r.dxb, one)));
    r.dyl = tobf(aie::sub(ny, aie::mul(r.dyb, one)));
    r.dzl = tobf(aie::sub(nz, aie::mul(r.dzb, one)));
    r.px = acc(prm[0]); r.py = acc(prm[1]); r.pz = acc(prm[2]); r.t = acc(0.0f);
}}
static inline void shade(const Ray &r, int16_t *__restrict o16, const V &Lx, const V &Ly, const V &Lz) {{
    V nx = tobf(aie::mul(r.qxb, r.invlen)), ny = tobf(aie::mul(r.qyb, r.invlen)), nz = tobf(aie::mul(r.qzb, r.invlen));
    A dots = aie::mul(nx, Lx); dots = aie::mac(dots, ny, Ly); dots = aie::mac(dots, nz, Lz);
    V diff_s = aie::max(tobf(dots), vb(0.0f));
    aie::mask<32> sphere = aie::lt(r.dsb, r.dfb);
    V diff = aie::select(Ly, diff_s, sphere);
    V tb = tobf(r.t);
    V fog = aie::inv(tobf(aie::mac(acc(1.0f), tb, vb(0.05f))));   // 1 / (1 + 0.05 t)
    // a hit is a final distance under eps (eps grows with t: one pixel is ~t/540 wide); else sky
    V dmin = aie::min(r.dsb, r.dfb);
    V eps = tobf(aie::mac(acc(0.01f), tb, vb(0.004f)));
    V far = aie::select(vb(0.0f), vb(1.0f), aie::lt(dmin, eps) & aie::lt(tb, vb(60.0f)));
    fog = tobf(aie::mul(fog, far));
    V shade = tobf(aie::mac(acc(0.15f), diff, vb(0.85f)));
    V skyw = tobf(aie::add(aie::mul(fog, vb(-1.0f)), acc(1.0f)));
    V br = aie::select(vb(150.0f), vb(255.0f), sphere), bg = aie::select(vb(150.0f), vb(110.0f), sphere),
      bb = aie::select(vb(160.0f), vb(50.0f), sphere);
    V cr = tobf(aie::mac(aie::mul(tobf(aie::mul(br, shade)), fog), skyw, vb(120.0f)));
    V cg = tobf(aie::mac(aie::mul(tobf(aie::mul(bg, shade)), fog), skyw, vb(160.0f)));
    V cb = tobf(aie::mac(aie::mul(tobf(aie::mul(bb, shade)), fog), skyw, vb(220.0f)));
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
/* prm: 0 cam.x 1 cam.y 2 cam.z 3 cos(yaw) 4 sin(yaw) 5 light.x 6 light.y 7 light.z */
void sdf(float *__restrict prm, int32_t *__restrict out, int32_t y) {{
    const float ca = prm[3], sa = prm[4];
    const V Lx = vb(prm[5]), Ly = vb(prm[6]), Lz = vb(prm[7]);
    const float vf = (540.0f - (float)y) / 540.0f;
    A rampa; rampa.from_vector(aie::load_v<32>(RAMP));
    const V ramp = tobf(rampa);
    const A magic4 = acc(50331648.0f);
    const V one = vb(1.0f);
    const A ustep = acc(32.0f / 540.0f);
    A ubase = acc(-960.0f / 540.0f);
    int16_t *__restrict o16 = (int16_t *)out;
    for (unsigned i = 0; i < W / 32; i += 2) {{
        // two independent rays per iteration so the two dependency chains overlap
        Ray a, b;
        init(a, aie::mac(ubase, ramp, vb(1.0f / 540.0f)), vf, ca, sa, prm);
        ubase = aie::add(ubase, ustep);
        init(b, aie::mac(ubase, ramp, vb(1.0f / 540.0f)), vf, ca, sa, prm);
        ubase = aie::add(ubase, ustep);
        for (int k = 0; k < STEPS; k++) {{
            eval(a, magic4); eval(b, magic4);
            step(a, one); step(b, one);
        }}
        eval(a, magic4); eval(b, magic4);
        shade(a, o16 + 64 * i, Lx, Ly, Lz);
        shade(b, o16 + 64 * i + 64, Lx, Ly, Lz);
    }}
}}
}}
"""


def ext(steps):
    prm_ty = np.ndarray[(NPRM,), np.dtype[np.float32]]
    row_ty = np.ndarray[(W,), np.dtype[np.int32]]
    return ExternalFunction("sdf", source_string=ksrc(steps), arg_types=[prm_ty, row_ty, np.dtype(np.int32)])


@iron.jit
def sdf_npu(prm: In, frame: Out, *, cols: CompileTime[int], func: CompileTime[ExternalFunction],
            nprm: CompileTime[int], variant: CompileTime[str]):
    """Frame plumbing shared by the ray marchers: func(prm, row, y) renders one 1920-pixel row.
    variant must change whenever the kernel source changes (the JIT cache does not hash it)."""
    cores = cols * ROWS
    per_core = H // cores
    frame_ty = np.ndarray[(H * W,), np.dtype[np.int32]]
    prm_ty = np.ndarray[(nprm,), np.dtype[np.float32]]
    quad_ty = np.ndarray[(ROWS * W,), np.dtype[np.int32]]
    row_ty = np.ndarray[(W,), np.dtype[np.int32]]
    k = func
    prm_in = [ObjectFifo(prm_ty, name=f"prm{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(quad_ty, name=f"out{c}", depth=2) for c in range(cols)]
    rows_out = []
    for c in range(cols):
        rows_out += col_out[c].prod().join([i * W for i in range(ROWS)], tile=Tile(c, 1), depths=[2] * ROWS,
                                           obj_types=[row_ty] * ROWS, names=[f"row{c}_{i}" for i in range(ROWS)])

    def core(prm, out, k, base):
        p = prm.acquire(1)
        for j in range_(per_core):
            o = out.acquire(1)
            k(p, o, j * cores + base)
            out.release(1)
        prm.release(1)

    workers = [Worker(core, [prm_in[c].cons(), rows_out[c * ROWS + r].prod(), k, c * ROWS + r],
                      tile=Tile(c, 2 + r)) for c in range(cols) for r in range(ROWS)]
    pp = [f.prod() for f in prm_in]
    oc = [f.cons() for f in col_out]
    full = TensorAccessPattern.full((nprm,))
    stripes = [TensorAccessPattern((H * W,), ROWS * c * W, [per_core, ROWS * W], [cores * W, 1]) for c in range(cols)]

    def seq(prm, frame, pp, oc):
        tg = TaskGroup()
        for c in range(cols):
            pp[c].fill(prm, full, group=tg)
        for c in range(cols):
            oc[c].drain(frame, stripes[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [prm_ty, frame_ty, pp, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def params(t):
    p = np.zeros(NPRM, dtype=np.float32)
    yaw = 0.3 * np.sin(t * 0.4)
    p[0], p[1], p[2] = 2.0 + 0.5 * np.sin(t * 0.3), 0.3 + 0.3 * np.sin(t * 0.5), (2.0 + 2.5 * t) % PERIOD  # keep p small
    p[3], p[4] = np.cos(yaw), np.sin(yaw)
    L = np.array([0.4, 0.8, -0.45]); L /= np.linalg.norm(L)
    p[5:8] = L
    return p


class SDF:
    """Host side: params(t) -> float32 buffer, one dispatch per frame, RGBA frame back."""

    def __init__(self, steps=STEPS, func=None, nprm=NPRM, params=params, variant=None):
        self.func = func or ext(steps)
        self.nprm, self.params = nprm, params
        self.variant = variant or f"sdf5-{steps}"
        self.cols = iron.get_current_device().cols
        self.prm = iron.zeros((nprm,), dtype=np.float32, device="npu")
        self.frame = iron.zeros((H * W,), dtype=np.int32, device="npu")

    def render(self, t):
        with self.prm.overwrite() as a:
            a[...] = self.params(t)
        t0 = time.perf_counter()
        sdf_npu(self.prm, self.frame, cols=self.cols, func=self.func, nprm=self.nprm, variant=self.variant)
        return self.frame.numpy(), (time.perf_counter() - t0) * 1e3


def check(steps):
    from aie.utils.compile import remarks
    d = Path(__file__).resolve().parent.parent / ".remarks" / ("rem_sdf")
    d.mkdir(parents=True, exist_ok=True)
    rep, why = remarks.analyze(ext(steps), "aie2p", d)
    if rep is None:
        print(why); sys.exit(1)
    print("libcalls:", rep.libcalls, "stack:", rep.kernel_stack_bytes, "notes:", rep.schedule_notes)
    for key, l in rep.loops.items():
        print("  loop", key, "ii", l.ii, "ns", l.ns, "zol", l.zol, "trips", l.trips, "bundles", l.bundle_count)


def window(scene, title, dt=0.03):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
    font = pygame.font.SysFont(None, 28)
    t, n, t0 = 0.0, 0, time.perf_counter()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
        f, ms = scene.render(t)
        surf = pygame.image.frombuffer(f.view(np.uint8)[: 1080 * W * 4], (W, 1080), "RGBA")
        if screen.get_size() != (W, 1080):
            surf = pygame.transform.smoothscale(surf, screen.get_size())
        screen.blit(surf, (0, 0)); n += 1
        screen.blit(font.render(f"{title}, 32 cores: {ms:.1f} ms/frame  {n / (time.perf_counter() - t0):.0f} fps", True, (255, 255, 255)), (10, 10))
        pygame.display.flip()
        t += dt


def bench(scene, title, png, frames=30):
    f, ms = scene.render(0.0)
    print(f"first frame (compile): {ms:.1f} ms")
    ts = [scene.render(i * 0.03)[1] for i in range(frames)]
    print(f"1920x1080, {title}: NPU min {min(ts):.1f} ms  med {np.median(ts):.1f} ms  ({1e3 / np.median(ts):.1f} fps)")
    import cv2
    cv2.imwrite(png, f.view(np.uint8).reshape(H, W, 4)[:1080, :, [2, 1, 0]])
    print("wrote", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    if a.check:
        return check(a.steps)
    s = SDF(a.steps)
    if a.window:
        return window(s, f"SDF {a.steps} steps")
    bench(s, f"{a.steps} steps", f"sdf{a.steps}.png", a.frames)


if __name__ == "__main__":
    main()
