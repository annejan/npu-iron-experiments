"""numpy emulation of the int16 Q14 update with (a) round-to-nearest, (b) truncation toward zero, (c) floor"""
import sys; sys.path.insert(0, "fun")
import numpy as np
from room import NZ, NY, NX, PARAMS, LAMBDA
g = PARAMS["damp"]; l2 = LAMBDA ** 2
c_p = round(g * (2 - 6 * l2) * 16384); c_pp = round(-g * 16384); c_n = round(g * l2 * 16384)
def run(mode, N=800):
    p = np.zeros((NZ, NY, NX), np.int64); pp = np.zeros_like(p)
    rms = []
    for i in range(N):
        inj = np.exp(-((i - 6) / 2.5) ** 2) * 0.5 if i < 14 else 0.0
        p[20, NY // 2, NX // 2] += int(round(inj * 16384))
        s = np.zeros_like(p)
        s[1:-1, 1:-1, 1:-1] = (p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] + p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] + p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:])
        acc = c_p * p + c_pp * pp + c_n * s
        if mode == "nearest": pn = np.floor(acc / 16384 + 0.5)
        elif mode == "trunc": pn = np.trunc(acc / 16384)
        else: pn = np.floor(acc / 16384)
        pn = pn.astype(np.int64)
        pn[:, 0] = pn[:, -1] = 0; pn[:, :, 0] = pn[:, :, -1] = 0
        pp, p = p, pn
        if i in (100, 300, 500, 799): rms.append((i, round(float(np.sqrt((p.astype(float) ** 2).mean())) / 16384, 6)))
    return rms
for m in ("nearest", "trunc", "floor"):
    print(m, run(m))
