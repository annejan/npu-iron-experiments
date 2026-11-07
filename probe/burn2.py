#!/usr/bin/env python3
"""burn variants: static II report + hardware run."""
import sys, time
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import ExternalFunction
from aie.utils.compile import remarks
sys.path.insert(0, str(Path(__file__).parent))
import burn

N = burn.N
VARIANTS = {
 "chain": burn.KSRC,
 "plain": """
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" {
void burn(int16_t *__restrict in, int16_t *__restrict out, int32_t reps) {
    for (int r = 0; r < reps; r++) {
        const int16_t *__restrict p = in;
        int16_t *__restrict q = out;
        for (unsigned i = 0; i < %d / 32; i++) {
            aie::vector<int16_t, 32> v = aie::load_v<32>(p); p += 32;
            aie::accum<acc32, 32> a = aie::mul(v, (int16_t)3);
            aie::store_v(q, a.to_vector<int16_t>(1)); q += 32;
        }
    }
}
}
""" % N,
 "mac_chain_reg": """
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" {
/* register-resident: load once, reps x (mul) in registers, store once. pure ALU ceiling */
void burn(int16_t *__restrict in, int16_t *__restrict out, int32_t reps) {
    const int16_t *__restrict p = in;
    int16_t *__restrict q = out;
    for (unsigned i = 0; i < %d / 128; i++) {
        aie::vector<int16_t, 32> v0 = aie::load_v<32>(p); p += 32;
        aie::vector<int16_t, 32> v1 = aie::load_v<32>(p); p += 32;
        aie::vector<int16_t, 32> v2 = aie::load_v<32>(p); p += 32;
        aie::vector<int16_t, 32> v3 = aie::load_v<32>(p); p += 32;
        for (int r = 0; r < reps; r++) {
            v0 = aie::mul(v0, (int16_t)3).to_vector<int16_t>(1);
            v1 = aie::mul(v1, (int16_t)3).to_vector<int16_t>(1);
            v2 = aie::mul(v2, (int16_t)3).to_vector<int16_t>(1);
            v3 = aie::mul(v3, (int16_t)3).to_vector<int16_t>(1);
        }
        aie::store_v(q, v0); q += 32; aie::store_v(q, v1); q += 32;
        aie::store_v(q, v2); q += 32; aie::store_v(q, v3); q += 32;
    }
}
}
""" % N,
}

chunk_ty = np.ndarray[(N,), np.dtype[np.int16]]
for name, src in VARIANTS.items():
    ef = ExternalFunction("burn", source_string=src, arg_types=[chunk_ty, chunk_ty, np.dtype(np.int32)])
    d = Path(__file__).resolve().parent.parent / ".remarks" / ("rem_" + name); d.mkdir(parents=True, exist_ok=True)
    rep, why = remarks.analyze(ef, "aie2p", d)
    if rep is None: print(name, why); continue
    print(f"== {name}: libcalls={rep.libcalls} stack={rep.kernel_stack_bytes} notes={rep.schedule_notes}")
    for key, l in rep.loops.items():
        print("   loop", key, {k: getattr(l, k, None) for k in ("ii", "ns", "zol", "trips", "bundle_count", "pipelined", "line")})
    burn.SOURCES[name] = src
    for cols, reps in ((1, 2048), (8, 2048)):
        burn.bench(reps, cols, iters=5, variant=name)
