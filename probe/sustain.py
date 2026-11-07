"""Does a long, heavy vector load hold the performance clock? Same II-1 loop, short vs long calls, back to back."""
import sys, time; sys.path.insert(0, "probe")
import numpy as np, burn, burn2  # burn2 registers the 'plain' variant source
burn.SOURCES["plain"] = burn2.VARIANTS["plain"]
for reps, calls in ((2048, 10), (32768, 20), (2048, 10)):
    ts = []
    for _ in range(calls):
        a = iron_a if 'iron_a' in dir() else None
        t = burn.bench(reps, 8, iters=1, variant="plain")
        ts.append(t)
    ts = np.array(ts) * 1e6
    per_iter = (ts - 180) / (reps * 256) * 1e3   # ns per inner iteration (II 1 ideal: 1/1.81 GHz = 0.55 ns, 1/1.40 = 0.71)
    print(f"reps={reps:6d}: per call first {ts[0]:8.0f} us  last {ts[-1]:8.0f} us  -> ~{per_iter[-1]:.2f} ns/iter -> ~{1/per_iter[-1]:.2f} GHz implied")
