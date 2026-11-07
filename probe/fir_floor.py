"""Where do the 7 ms per reverb dispatch go? Same dataflow, kernels of different cost."""
import sys, time; sys.path.insert(0, "fun")
import numpy as np, reverb
reverb.SOURCES["nop"] = reverb.KSRC_SLIDING.replace("for (int u0 = 0; u0 < T; u0 += 32) {", "for (int u0 = 0; u0 < 0; u0 += 32) {")
reverb.SOURCES["sliding64"] = reverb.KSRC_SLIDING.replace("#define T 2048", "#define T 64")
reverb.SOURCES["sliding512"] = reverb.KSRC_SLIDING.replace("#define T 2048", "#define T 512")
ir = reverb.synth_ir(48000)
for v in ("nop", "sliding64", "sliding512", "sliding"):
    r = reverb.Reverb(ir, v)
    r.feed(np.zeros(reverb.B))
    ts = []
    for _ in range(10):
        r.feed(np.random.uniform(-0.5, 0.5, reverb.B)); ts.append(r.npu_ms)
    print(f"{v:12s}: NPU call min {min(ts):6.2f} ms  med {np.median(ts):6.2f} ms")
