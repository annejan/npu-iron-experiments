import sys; sys.path.insert(0, "fun")
import numpy as np, lenia
from lenia import Lenia, K, H2, W, H, KSHIFT

def best_shift(a, b, rng=20):
    """find (dy, dx) with min error between a and roll(b)"""
    best = (1e9, 0, 0)
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            e = np.abs(a - np.roll(np.roll(b, dy, 0), dx, 1)).mean()
            if e < best[0]: best = (e, dy, dx)
    return best

sim = Lenia(seed=3)
rng = np.random.default_rng(0)
sim.world[:] = rng.uniform(0, 1, (H, W)).astype(np.float32); sim.upload = True
w0 = sim.world.copy()

# 1. delta kernel: u must equal the world
D = 32767; sc = D / (1 << KSHIFT)   # a delta of 2^16 does not fit int16: use 32767 and scale the reference
kq = np.zeros((K, 32), np.int16); kq[H2, H2] = D
with sim._kern.overwrite() as a: a[...] = kq.ravel()
sim.step(outmode=2); u = sim.read().copy()
w0s = w0 * sc
print(f"delta kernel: max|u-w| {np.abs(u - w0s).max():.4f} mean {np.abs(u - w0s).mean():.4f}; best shift (err,dy,dx) = {best_shift(u, w0s)}")
rows_bad = np.where(np.abs(u - w0s).max(1) > 0.01)[0]
print(f"  bad rows: {len(rows_bad)} of {H}; first {rows_bad[:12].tolist()}; cols bad in row {rows_bad[0] if len(rows_bad) else -1}: {np.where(np.abs(u - w0s)[rows_bad[0]] > 0.01)[0][:10].tolist() if len(rows_bad) else []}")

# 2. kernel with a single off-centre tap (dy=+3, dx=+5): u must be world shifted
kq[:] = 0; kq[H2 + 3, H2 + 5] = D
with sim._kern.overwrite() as a: a[...] = kq.ravel()
sim.world[:] = w0; sim.upload = True
sim.step(outmode=2); u = sim.read().copy()
print(f"tap (+3,+5): best shift = {best_shift(u, w0s)}  (expect dy,dx = -3,-5 for correlation form)")

# 3. impulse world, ring kernel
sim.kq = lenia.lenia_kernel(lenia.PARAMS)[1]
with sim._kern.overwrite() as a: a[...] = sim.kq.ravel()
sim.world[:] = 0; sim.world[200, 300] = 1.0; sim.upload = True
sim.step(outmode=2); u = sim.read().copy()
ref = np.zeros((H, W)); ref[200 - H2:200 + H2 + 1, 300 - H2:300 + H2 + 1] = sim.kq[:, :K] / (1 << KSHIFT)
print(f"impulse: max|u-ref| {np.abs(u - ref).max():.5f}; u sum {u.sum():.4f} ref sum {ref.sum():.4f}; nonzero u rows {np.where(u.max(1) > 0)[0].min()}..{np.where(u.max(1) > 0)[0].max()} cols {np.where(u.max(0) > 0)[0].min()}..{np.where(u.max(0) > 0)[0].max()}")
