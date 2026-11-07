import sys; sys.path.insert(0, "fun")
import numpy as np, lenia
from lenia import Lenia, K, H2, W, H, S, PADL, KSHIFT
sim = Lenia(seed=3)
rng = np.random.default_rng(0)
sim.world[:] = rng.uniform(0.2, 1, (H, W)).astype(np.float32); sim.upload = True
D = 32767
kq = np.zeros((K, 32), np.int16); kq[H2, H2] = D
with sim._kern.overwrite() as a: a[...] = kq.ravel()
# poison the output buffer so unwritten columns show up
with sim._out.overwrite() as a: a[...] = -12345
sim.step(outmode=2)
raw = sim._in.numpy().reshape(H + K - 1, S)      # after the swap, _in holds the fresh output (halos patched by fix_halos)
row = raw[H2 + 100]
print("padded row 113 (world row 100):")
print("  cols 28..48 :", row[28:49].tolist())
print("  cols 540..575:", row[540:576].tolist())
print("  cols 576..607:", row[576:608].tolist())
nz = np.where(row != -12345)[0]
print(f"  written span: {nz.min()}..{nz.max()}  (kernel intends {PADL}..{PADL + W - 1}; host fix_halos rewrites {PADL - H2}..{PADL - 1} and {PADL + W}..{PADL + W + H2 - 1})")
w0 = sim.world
print("  world row 100 cols 0..4 * 32767/65536:", (w0[100, :5] * D / 65536 * 32767).round().astype(int).tolist())
print("  raw row cols 45..49:", row[45:50].tolist(), " cols 32..36:", row[32:37].tolist())
