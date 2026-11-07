"""Run the II-1 burn loop continuously for ~8 s on all 32 cores while npu_top samples the H clock."""
import sys, time, subprocess; sys.path.insert(0, "probe")
import numpy as np, burn, burn2
burn.SOURCES["plain"] = burn2.VARIANTS["plain"]
burn.bench(32768, 8, iters=1, variant="plain")          # compile + warm
mon = subprocess.Popen([sys.executable, "probe/npu_top.py", "--watch", "0.5", "--count", "18"], stdout=subprocess.PIPE, text=True)
t0 = time.perf_counter(); calls = 0; ts = []
while time.perf_counter() - t0 < 8:
    t = burn.bench(32768, 8, iters=1, variant="plain"); ts.append(t); calls += 1
out, _ = mon.communicate()
print(out)
ts = np.array(ts) * 1e6
print(f"{calls} sustained 32768-rep calls: per call min {ts.min():.0f} us med {np.median(ts):.0f} us -> implied {(32768 * 256) / ((np.median(ts) - 180) * 1e-6) / 1e9:.2f} GHz at II 1")
