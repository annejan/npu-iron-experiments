import sys; sys.path.insert(0, "fun")
import numpy as np
from slime import Slime
s = Slime(15)
for calls in (1, 5, 20, 50, 100, 200):
    for _ in range(calls): s.run()
    a, p = s.a, s.p
    print(f"after +{calls:3d} calls: a min {np.nanmin(a):.4f} max {np.nanmax(a):.3f} mean {np.nanmean(a):.4f} nan {np.isnan(a).sum()} | "
          f"p max {np.nanmax(p):.3f} mean {np.nanmean(p):.4f} nan {np.isnan(p).sum()} | cells a>0.01: {(a > 0.01).sum()}")
