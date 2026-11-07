import sys; sys.path.insert(0, "fun")
import numpy as np, room as R
from room import Room, reference_step, PARAMS, FS, NZ, NY, NX
import soundfile as sf
ir, fs = sf.read("room_ir_8k.wav", always_2d=True)
e = ir[:, 0] ** 2
print("written IR, energy per 100 ms (dB rel. first bin):", [round(10 * np.log10(e[i:i + 800].sum() / e[:800].sum() + 1e-12), 1) for i in range(0, len(e), 800)])
rm = Room()
p = np.zeros((NZ, NY, NX)); pp = np.zeros_like(p)
x, y, z = rm.src
N = 1500
rec_npu = np.zeros(N); rec_ref = np.zeros(N)
for i in range(N):
    inj = np.exp(-((i - 6) / 2.5) ** 2) * 0.5 if i < 14 else 0.0
    p[z, y, x] += inj
    rm.step(inject=inj)
    p, pp = reference_step(p, pp, PARAMS)
    rec_npu[i] = rm.receivers(rm._in.numpy())[0]
    rx, ry, rz = rm.rx[0]; rec_ref[i] = p[rz, ry, rx]
    if i in (300, 800, 1499):
        g = rm.field(); err = np.abs(g - p)
        print(f"step {i}: field max|err| {err.max():.5f} rms {np.sqrt((err**2).mean()):.2e}; ref rms {np.sqrt((p**2).mean()):.2e} npu rms {np.sqrt((g**2).mean()):.2e}")
for name, r in (("npu", rec_npu), ("ref", rec_ref)):
    e = r ** 2
    print(f"{name} receiver energy per 100 ms (dB):", [round(10 * np.log10(e[i:i + 800].sum() / (rec_ref[:800] ** 2).sum() + 1e-15), 1) for i in range(0, N, 800)])
