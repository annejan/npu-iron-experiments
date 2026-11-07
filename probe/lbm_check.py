import sys; sys.path.insert(0, "fun")
import numpy as np
from lbm import LBM, PARAMS, CX, CY, WT, OPP, H, W, S, PADC, NF, bf2f, f2bf

def ref_step(g, mask, eq_in):
    """g: (H, 9, W) deviations float32, periodic y, inlet/outlet in x like fix_halos. returns new g"""
    tau = PARAMS["tau"]; om = 1 / tau
    # pad in x: inlet cols at -1 (eq), outlet at W (copy of W-2)
    gp = np.empty((H, 9, W + 2), np.float32)
    gp[:, :, 1:W + 1] = g
    gp[:, :, 0] = eq_in[None, :]; gp[:, :, W + 1] = g[:, :, W - 2]
    gp[:, :, 1:3] = eq_in[None, :, None]              # inlet columns 0,1 fixed
    gp[:, :, W] = g[:, :, W - 2]                      # outlet col W-1 = col W-2
    pulled = np.empty_like(g)
    for i in range(9):
        src = np.roll(gp[:, i, :], CY[i], axis=0)     # from y - cy  (roll by +cy brings y-cy to y)
        pulled[:, i, :] = src[:, 1 - CX[i]:1 - CX[i] + W]
    rho = 1 + pulled.sum(1)
    ux = (pulled * np.array(CX, np.float32)[None, :, None]).sum(1) / rho
    uy = (pulled * np.array(CY, np.float32)[None, :, None]).sum(1) / rho
    u2 = ux * ux + uy * uy
    out = np.empty_like(g)
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        eq = WT[i] * ((rho - 1) + rho * (3 * cu + 4.5 * cu * cu - 1.5 * u2))
        out[:, i] = np.where(mask, pulled[:, OPP[i]], pulled[:, i] + om * (eq - pulled[:, i]))
    return out

sim = LBM()
eq_in = bf2f(sim.eq_in)
g = bf2f(sim.world[1:H + 1, :9, PADC:PADC + W].astype(np.float32).view(np.float32).copy().ravel().view(np.float32)) if False else None
# take the exact bf16-rounded initial state the NPU gets
a0 = f2bf(sim.world.ravel()).reshape(H + 2, NF, S); sim.fix_halos(a0)
g = bf2f(a0[1:H + 1, :9, PADC:PADC + W]).copy()
for step in range(1, 61):
    sim.step()
    g = ref_step(g, sim.mask, eq_in)
    npu = bf2f(sim._in.numpy()).reshape(H + 2, NF, S)[1:H + 1, :9, PADC:PADC + W]
    d = np.abs(npu - g); nn = np.isnan(npu)
    if step in (1, 2, 3, 5, 10, 20, 40, 60) or nn.any():
        rho = 1 + npu.sum(1)
        print(f"step {step:3d}: max|npu-ref| {np.nanmax(d):.2e} at {np.unravel_index(np.nanargmax(d), d.shape)}  "
              f"ref rho {1 + g.sum(1).min():.4f}..{1 + g.sum(1).max():.4f}  npu rho {np.nanmin(rho):.4f}..{np.nanmax(rho):.4f}  nan {nn.sum()}")
        if nn.any():
            ys, fs, xs = np.where(nn); print("   first nans at (y, x):", sorted(set(zip(ys.tolist(), xs.tolist())))[:8]); break
