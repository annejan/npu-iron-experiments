import sys, time; sys.path.insert(0, "fun")
import numpy as np, reverb
ir = reverb.synth_ir(48000)
a, b = reverb.Reverb(ir, "sliding"), reverb.Reverb(ir, "sliding")
x = np.random.uniform(-0.5, 0.5, reverb.B)
a.feed(x); b.feed(x)
ts = []
for _ in range(10):
    a.feed(x); ts.append(a.npu_ms); b.feed(x); ts.append(b.npu_ms)
print(f"alternating two instances (own buffers): med {np.median(ts):.2f} ms per call")
ts = []
for _ in range(20):
    a.feed(x); ts.append(a.npu_ms)
print(f"same instance repeatedly: med {np.median(ts):.2f} ms per call")
