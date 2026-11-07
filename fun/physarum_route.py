#!/usr/bin/env python3
"""Physarum global router for KiCad boards: every net is a slime-mould tube network competing for board space.

  ./env.sh python fun/physarum_route.py board.kicad_pcb                 route, write <board>_physarum.png (+ _usage.png)
  ./env.sh python fun/physarum_route.py board.kicad_pcb --cell 0.5 --steps 300 --max-nets 120
  ./env.sh python fun/physarum_route.py board.kicad_pcb --write-pro      also <board>_physarum.kicad_pro with topor_p1..3 classes
  ./env.sh python fun/physarum_route.py --selftest                       two synthetic nets around an obstacle, prints stretch

Model (Tero, Kobayashi, Nakagaki, "A mathematical model for adaptive transport network in path finding by true
slime mold", J. Theor. Biol. 244:553-564, 2007; the multi-source version of Tero et al., Science 327:439, 2010):
the board's two outer copper layers are a grid; each net n has a conductance field D_n (its tube thickness) on
the free cells and a pressure field p_n. Every step one pad of the net is the sink (p = 0) and another the source
(unit inflow), the pressure obeys div(D grad p) = s (solved by a few red-black SOR sweeps, warm-started), the flow
through a cell q = (sum of |D dp| over its edges) / 2 feeds back into the tube: dD/dt = q - D. Tubes that carry
flow fatten, idle ones vanish; the net converges to a shortest-path tree between its pads (Steiner-like for
multi-pad nets, Tero 2010's random pairs). Vias are vertical edges whose conductance is scaled down by
--via-cost so a layer change costs like a few mm of track. Nets compete: where other nets' flow exceeds the
cell's track capacity, a net's conductance is damped (exp(-kappa * excess)), so crowded corridors push tubes apart.

Output: a picture of the tubes per net (hue per net) on F.Cu and B.Cu, a usage map against capacity, and per net
the flow-weighted length, its stretch over the straight pad distance and how much competition it meets along its
route. --write-pro turns that into topor's priority net classes (topor_p1 = contested or stretched, p2, p3) in
a copy of the project file; the plugin reads them as routing order.

It is a global estimate, not a router: obstacles are pads (plus clearance) and the board edge box, zones and
existing copper are ignored, power nets are skipped (they live on the inner planes). Grid cell 0.5 mm by default.
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np

POWER = re.compile(r"^(\+|-)?\d|^(V|VCC|VDD|VBAT|GND|AGND|PWR)", re.I)

# ----------------------------------------------------------------------------------------------- KiCad file
_TOK = re.compile(r'"(?:[^"\\]|\\.)*"|\(|\)|[^\s()]+')


def sexp(text):
    """Parse an s-expression file into nested lists; strings keep their quotes stripped."""
    toks = _TOK.findall(text)
    stack = [[]]
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            node = stack.pop(); stack[-1].append(node)
        elif t[0] == '"':
            stack[-1].append(t[1:-1])
        else:
            stack[-1].append(t)
    return stack[0][0]


def child(node, key):
    for c in node[1:]:
        if isinstance(c, list) and c and c[0] == key:
            return c
    return None


def children(node, key):
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == key]


class Board:
    """Pads with absolute positions (mm, KiCad frame: y down), nets, edge box."""

    def __init__(self, path):
        text = Path(path).read_text()
        root = sexp(text)
        self.pads = []               # (net, x, y, w, h, rot_deg, front, back)
        self.nets = {}               # name -> [pad index]
        xs, ys = [], []
        for item in root[1:]:
            if not isinstance(item, list):
                continue
            k = item[0]
            if k == "footprint":
                at = child(item, "at"); fx, fy = float(at[1]), float(at[2]); frot = float(at[3]) if len(at) > 3 else 0.0
                flayer = child(item, "layer")[1]
                cr, sr = np.cos(np.radians(frot)), np.sin(np.radians(frot))
                for pad in children(item, "pad"):
                    kind = pad[2]
                    pat = child(pad, "at"); dx, dy = float(pat[1]), float(pat[2]); prot = float(pat[3]) if len(pat) > 3 else frot
                    # pad offsets are stored in the footprint's local frame (already mirrored for back-side parts)
                    x = fx + dx * cr + dy * sr
                    y = fy - dx * sr + dy * cr
                    sz = child(pad, "size"); w, h = float(sz[1]), float(sz[2])
                    layers = child(pad, "layers"); layers = layers[1:] if layers else []
                    th = kind in ("thru_hole", "np_thru_hole")
                    front = th or any(l in ("F.Cu", "*.Cu") for l in layers)
                    back = th or any(l in ("B.Cu", "*.Cu") for l in layers)
                    if not (front or back):
                        continue
                    netn = child(pad, "net")
                    name = (netn[1] if netn and len(netn) == 2 else (netn[2] if netn and len(netn) > 2 else "")) if kind != "np_thru_hole" else ""
                    if kind == "np_thru_hole":
                        name = None
                    self.pads.append((name, x, y, w, h, prot, front, back))
                    if name:
                        self.nets.setdefault(name, []).append(len(self.pads) - 1)
            elif k in ("gr_line", "gr_arc", "gr_rect", "gr_circle", "gr_poly") and child(item, "layer") and child(item, "layer")[1] == "Edge.Cuts":
                for key in ("start", "end", "mid", "center"):
                    c = child(item, key)
                    if c:
                        xs.append(float(c[1])); ys.append(float(c[2]))
                pts = child(item, "pts")
                if pts:
                    for xy in children(pts, "xy"):
                        xs.append(float(xy[1])); ys.append(float(xy[2]))
                if k == "gr_circle":
                    c, e = child(item, "center"), child(item, "end")
                    r = np.hypot(float(e[1]) - float(c[1]), float(e[2]) - float(c[2]))
                    xs += [float(c[1]) - r, float(c[1]) + r]; ys += [float(c[2]) - r, float(c[2]) + r]
        # Edge.Cuts may live inside footprints (outline parts) and be incomplete at the top level: take the union
        # with the pads' extent plus a margin
        px = [p[1] for p in self.pads]; py = [p[2] for p in self.pads]
        xs += [min(px) - 3, max(px) + 3]; ys += [min(py) - 3, max(py) + 3]
        self.box = (min(xs), min(ys), max(xs), max(ys))


def project_rules(pcb):
    """(clearance, track width) in mm of the Default net class from the .kicad_pro next to the board, else None."""
    pro = Path(pcb).with_suffix(".kicad_pro")
    try:
        d = json.loads(pro.read_text())
        for c in d["net_settings"]["classes"]:
            if c.get("name") == "Default":
                return float(c.get("clearance", 0.2)), float(c.get("track_width", 0.2))
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------------------------- raster
def rasterize(board, names, cell, clearance, width=0.2, copper=(), keep=()):
    """owner[layer, y, x]: index into `names` of the pad there (own-net conductive), -1 free, -2 blocked for all.
    Two passes: every pad's clearance ring blocks free cells, then every pad's core (the cells under its copper,
    at least the cell under its centre) is painted over the rings; two cores of different nets in one cell stay
    blocked (grid too coarse there). Pads of nets we do not route are plain obstacles. Returns owner, (x0, y0)."""
    x0, y0, x1, y1 = board.box
    W = int(np.ceil((x1 - x0) / cell)) + 1; H = int(np.ceil((y1 - y0) / cell)) + 1
    owner = np.full((2, H, W), -1, np.int32)
    index = {n: i for i, n in enumerate(names)}
    yy, xx = np.mgrid[0:H, 0:W]
    cx = x0 + xx * cell; cy = y0 + yy * cell
    windows = []
    for name, x, y, w, h, rot, front, back in board.pads:
        r = np.radians(rot); c, sn = np.cos(r), np.sin(r)
        rad = 0.5 * np.hypot(w, h) + clearance + cell
        ix0 = max(0, int((x - rad - x0) / cell)); ix1 = min(W, int((x + rad - x0) / cell) + 2)
        iy0 = max(0, int((y - rad - y0) / cell)); iy1 = min(H, int((y + rad - y0) / cell) + 2)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        dx = cx[iy0:iy1, ix0:ix1] - x; dy = cy[iy0:iy1, ix0:ix1] - y
        u = dx * c - dy * sn; v = dx * sn + dy * c                 # into the pad frame
        ring = (np.abs(u) <= w / 2 + clearance + width / 2) & (np.abs(v) <= h / 2 + clearance + width / 2)
        hard = (np.abs(u) < w / 2 + width / 2) & (np.abs(v) < h / 2 + width / 2)           # copper would overlap
        core = (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)
        dist = np.maximum(np.abs(u) - w / 2, 0) ** 2 + np.maximum(np.abs(v) - h / 2, 0) ** 2   # to the pad's edge
        core[min(iy1 - iy0 - 1, max(0, int(round((y - y0) / cell)) - iy0)), min(ix1 - ix0 - 1, max(0, int(round((x - x0) / cell)) - ix0))] = True
        i = index.get(name, -2) if name else -2
        layers = [l for l, on in ((0, front), (1, back)) if on]
        windows.append((iy0, iy1, ix0, ix1, ring, core, hard, dist, i, layers))
    # the net whose clearance ring covers the cell (free for that net only); where rings of two nets overlap
    # (the gap between fine-pitch pads) the cell goes to the nearer pad, so a pad is never walled in by its
    # neighbours' clearance
    ringof = np.full((2, H, W), -1, np.int32)
    ringd = np.full((2, H, W), np.inf, np.float32)
    hardmask = np.zeros((2, H, W), np.int32)       # -1 free, else the one net whose pad's hard zone it is, -2 if several
    hardmask[:] = -1
    for iy0, iy1, ix0, ix1, ring, core, hard, dist, i, layers in windows:  # pass 1: rings and foreign pads
        for l in layers:
            o = owner[l, iy0:iy1, ix0:ix1]; r = ringof[l, iy0:iy1, ix0:ix1]; rd = ringd[l, iy0:iy1, ix0:ix1]
            hm = hardmask[l, iy0:iy1, ix0:ix1]
            if i < 0:
                rr = ring & (dist < rd)                   # a pad we do not route: its copper and ring block,
                r[rr] = -2; rd[rr] = dist[rr]             # but only where no routed pad is nearer
                hm[hard] = -2
            else:
                rr = ring & ~core & (dist < rd)
                r[rr] = i; rd[rr] = dist[rr]
                hm[hard & (hm == -1)] = i
                hm[hard & (hm >= 0) & (hm != i)] = -2
    owner[ringof == -2] = -2
    owner[hardmask == -2] = -2
    # the hard zone of a routed pad is free for that net only (its own exit), blocked for the rest
    foreign_hard = (hardmask >= 0) & (ringof != hardmask)
    owner[foreign_hard] = -2
    ringof[hardmask >= 0] = np.where(ringof[hardmask >= 0] == hardmask[hardmask >= 0], ringof[hardmask >= 0], -1)
    cores = np.full((2, H, W), -1, np.int32)
    for iy0, iy1, ix0, ix1, ring, core, hard, dist, i, layers in windows:  # pass 2: own cores win over rings
        if i < 0:
            continue
        for l in layers:
            o = cores[l, iy0:iy1, ix0:ix1]
            clash = core & (o >= 0) & (o != i)
            o[core & (o == -1)] = i
            o[clash] = -3
    own = cores >= 0
    owner[own] = cores[own]
    owner[cores == -3] = -2
    if copper:
        import cv2
        for l in (0, 1):
            m = np.zeros((H, W), np.uint8)
            for kind, layer, pts, wd in copper:
                if layer != l:
                    continue
                r = max(1, int(round((wd / 2 + clearance + width / 2) / cell)))
                if kind == "segment":
                    (ax, ay), (bx, by) = pts
                    cv2.line(m, (int(round((ax - x0) / cell)), int(round((ay - y0) / cell))),
                             (int(round((bx - x0) / cell)), int(round((by - y0) / cell))), 1, 2 * r + 1)
                else:
                    (ax, ay), = pts
                    cv2.circle(m, (int(round((ax - x0) / cell)), int(round((ay - y0) / cell))), r, 1, -1)
            blk = (m > 0) & (owner[l] == -1) & (ringof[l] == -1)    # the clearance part spares a pad's own exit ring
            owner[l][blk] = -2
            mh = np.zeros((H, W), np.uint8)                           # the copper itself (+ half a track): always
            for kind, layer, pts, wd in copper:
                if layer != l:
                    continue
                r = max(1, int(round((wd / 2 + width / 2) / cell)))
                if kind == "segment":
                    (ax, ay), (bx, by) = pts
                    cv2.line(mh, (int(round((ax - x0) / cell)), int(round((ay - y0) / cell))),
                             (int(round((bx - x0) / cell)), int(round((by - y0) / cell))), 1, 2 * r + 1)
                else:
                    (ax, ay), = pts
                    cv2.circle(mh, (int(round((ax - x0) / cell)), int(round((ay - y0) / cell))), r, 1, -1)
            owner[l][(mh > 0) & (owner[l] < 0)] = -2
    if keep:
        import cv2
        for layers, pts in keep:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.array([[int(round((x - x0) / cell)), int(round((y - y0) / cell))] for x, y in pts], np.int32)], 1)
            cv2.polylines(m, [np.array([[int(round((x - x0) / cell)), int(round((y - y0) / cell))] for x, y in pts], np.int32)], True, 1,
                          max(1, int(round((clearance + width / 2) / cell)) * 2 + 1))
            for l in layers:
                owner[l][(m > 0) & (owner[l] < 0)] = -2
    ringof[(owner == -2) | own] = -1
    ringof[ringof < -1] = -1
    return owner, ringof, (x0, y0)


# ----------------------------------------------------------------------------------------------- model
def harm(a, b):
    return 2 * a * b / (a + b + 1e-12)


SQ2 = np.float32(np.sqrt(2))


class Net:
    """One net's tube network on its region of interest (the pads' box plus a margin), 2 layers, 8 in-plane
    neighbours (diagonals cost sqrt 2) plus the via edge. Each step: effective conductances (own tubes, damped where
    other nets crowd), the pressure solve as one sparse direct solve (scipy splu; the thin-tube contrast makes the
    system far too stiff for a few relaxation sweeps), the flows, dD/dt = q^mu - D, and pruning: cells whose tube
    fell below `prune` x the thickest leave the system for good, which is what keeps the solves small."""

    def __init__(self, n, owner, pads, box, *, via_g, dmax, mu, dt, dmin, prune, rng, norm=True, ringof=None):
        self.norm = norm
        self.n = n; self.pads = pads
        y0, y1, x0, x1 = box; self.box = box
        own = owner[:, y0:y1, x0:x1] == n
        self.free = (owner[:, y0:y1, x0:x1] == -1) | own
        self.exit = own.copy()
        if ringof is not None:
            self.exit |= ringof[:, y0:y1, x0:x1] == n            # a pad's clearance ring: free for its own net, and
            self.free |= self.exit                               # no crowding penalty there (pins sit close by design)
        self.failed = False; self.hard = False; self.skipped = 0
        self.yield_ = float(rng.uniform(0.5, 1.5))                 # how much this net gives way in a crowd
        self.own = own
        self.active = self.free.copy()
        self.via_g, self.dmax, self.mu, self.dt, self.dmin, self.prune = via_g, dmax, mu, dt, dmin, prune
        self.D = np.where(self.free, 1.0, 0.0).astype(np.float32) * rng.uniform(0.9, 1.1, self.free.shape).astype(np.float32)
        self.D[own] = dmax
        self.q = np.zeros(self.free.shape, np.float32)
        self.length = 0.0; self.vias = 0.0; self.arrived = 0.0; self.resets = 0
        self.p = None; self.E = None; self.why = ""
        # pad cell groups in ROI coordinates
        self.groups = [(l, ys - y0, xs - x0) for l, ys, xs in pads]
        self.sink_i = 0

    def unreachable(self):
        """Indices of terminal groups not connected to group 0 through free cells (8 neighbours + the via)."""
        from scipy.ndimage import label
        lab, _ = label(self.free, structure=np.ones((3, 3, 3), int))
        l, ys, xs = self.groups[0]
        root = lab[l, ys[0], xs[0]]
        return {gi for gi, (l, ys, xs) in enumerate(self.groups) if lab[l, ys[0], xs[0]] != root}

    def exits(self, l, ys, xs):
        """Does this terminal touch any free non-pad cell (8 neighbours on its layer, or the other layer)?"""
        L, H, W = self.free.shape
        for y, x in zip(ys, xs):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and self.free[l, yy, xx] and not self.own[l, yy, xx]:
                        return True
            if self.free[1 - l, y, x] and not self.own[1 - l, y, x]:
                return True
        return False

    def sources(self, step, rotate):
        k = len(self.groups)
        b = 0 if k == 2 else (step // rotate) % k
        s = np.zeros(self.free.shape, np.float32); sink = np.zeros(self.free.shape, bool)
        for a, (l, ys, xs) in enumerate(self.groups):
            if a == b:
                sink[l, ys, xs] = True
            else:
                s[l, ys, xs] = 1.0 / (k - 1) / len(ys)
        return s, sink

    def step(self, step, rotate, others, kappa, crowd=None):
        if self.failed:
            return
        self.last_step, self.rotate = step, rotate
        D = self.D * self.active
        if kappa > 0 and step > 0:
            # crowding damps the tubes, but never below 1e-3 of their thickness: a terminal whose only way out
            # crosses a crowd must stay numerically connected (exp(-40) made the matrix singular: nets gave up)
            D = D * np.maximum(np.exp(-kappa * self.yield_ * np.maximum(others, 0) * ~self.exit), 1e-3).astype(np.float32)
            if self.hard and crowd is not None:
                D = D * ((crowd < 0.5) | self.exit)              # regrown after rip-up: other tubes are walls
            D[self.own] = self.dmax
        s, sink = self.sources(step, rotate)
        # unknowns: active, non-sink cells; sinks are p = 0 neighbours
        unk = self.active & ~sink
        idx = np.full(self.free.shape, -1, np.int64); idx[unk] = np.arange(int(unk.sum()))
        nunk = int(unk.sum())
        rows, cols, vals = [], [], []
        diag = np.zeros(nunk, np.float64)

        def edge(ia, ib, c):
            m = (ia >= 0) & (ib >= 0) & (c > 0)
            a, b_, cc = ia[m], ib[m], c[m].astype(np.float64)
            np.add.at(diag, a, cc); np.add.at(diag, b_, cc)
            rows.append(a); cols.append(b_); vals.append(-cc); rows.append(b_); cols.append(a); vals.append(-cc)
            # one end a sink (idx -1 but active): only the other end's diagonal grows
            for ii, jj in ((ia, ib), (ib, ia)):
                ms = (ii >= 0) & (jj < 0) & (c > 0) & sinkpos(jj)
                np.add.at(diag, ii[ms], c[ms].astype(np.float64))

        sink_idx = np.where(sink, -2, -1)                     # -2 marks a sink position in the "other end" test

        def sinkpos(jj):
            return jj == -2

        def mk(ia, ib, c):
            # replace -1 (blocked/inactive) by -1 and sinks by -2 in the index maps used for the sink test
            return edge(ia, ib, c)

        I = np.where(sink, -2, idx)
        Dx = harm(D[..., :, :-1], D[..., :, 1:]); mk(I[..., :, :-1], I[..., :, 1:], Dx)
        Dy = harm(D[..., :-1, :], D[..., 1:, :]); mk(I[..., :-1, :], I[..., 1:, :], Dy)
        Dse = harm(D[..., :-1, :-1], D[..., 1:, 1:]) / SQ2; mk(I[..., :-1, :-1], I[..., 1:, 1:], Dse)
        Dsw = harm(D[..., :-1, 1:], D[..., 1:, :-1]) / SQ2; mk(I[..., :-1, 1:], I[..., 1:, :-1], Dsw)
        Dz = harm(D[0], D[1]) * self.via_g; mk(I[0], I[1], Dz)
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        # a tiny leak to ground on every unknown keeps enclosed pockets of free cells (no path to any sink) from
        # making the matrix singular; the flow lost through it is 1e-6
        rows.append(np.arange(nunk)); cols.append(np.arange(nunk)); vals.append(diag + 1e-12)
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nunk, nunk))
        b = s[unk].astype(np.float64)
        try:
            pv = spl.splu(A).solve(b)
        except Exception:
            pv = np.full(nunk, np.nan)
        p = np.zeros(self.free.shape, np.float32); p[unk] = pv
        ok = bool(np.isfinite(pv).all())
        self.why = "" if ok else f"solve failed (nunk {nunk})"
        if ok:
            def flow(c, ia_sl, ib_sl):
                return np.abs(c * (p[ia_sl] - p[ib_sl]))
            Qx = flow(Dx, (Ellipsis, slice(None), slice(None, -1)), (Ellipsis, slice(None), slice(1, None)))
            Qy = flow(Dy, (Ellipsis, slice(None, -1), slice(None)), (Ellipsis, slice(1, None), slice(None)))
            Qse = flow(Dse, (Ellipsis, slice(None, -1), slice(None, -1)), (Ellipsis, slice(1, None), slice(1, None)))
            Qsw = flow(Dsw, (Ellipsis, slice(None, -1), slice(1, None)), (Ellipsis, slice(1, None), slice(None, -1)))
            Qz = np.abs(Dz * (p[0] - p[1]))
            q = np.zeros_like(p)
            q[..., :, :-1] += Qx; q[..., :, 1:] += Qx; q[..., :-1, :] += Qy; q[..., 1:, :] += Qy
            q[..., :-1, :-1] += Qse; q[..., 1:, 1:] += Qse; q[..., :-1, 1:] += Qsw; q[..., 1:, :-1] += Qsw
            q[0] += Qz; q[1] += Qz
            q *= 0.5
            # flow into the sink = what arrived; a sink cell has inflow only, so its q (half the edge flows) is half
            self.arrived = float(2 * q[sink].sum())
            self.length = float(Qx.sum() + Qy.sum() + (Qse.sum() + Qsw.sum()) * SQ2)
            self.vias = float(Qz.sum())
            if not (0.5 < self.arrived < 1.5 and np.isfinite(q).all()):
                ok = False
                walled = [gi for gi, (l, ys, xs) in enumerate(self.groups) if not self.exits(l, ys, xs)]
                self.why = (f"arrived {self.arrived:.3f} of 1" + (f", pads {walled} walled in (no free neighbour)" if walled else
                            f" (nunk {nunk}, active {int(self.active.sum())}, |p| max {np.abs(p).max():.3g})"))
        if not ok:
            # the pruned network lost its way (disconnected or singular): grow everything back and start over;
            # a net that cannot connect even on the full region is given up (walled-in pads)
            self.active = self.free.copy(); self.D[self.free] = 1.0; self.D[self.own] = self.dmax; self.resets += 1
            self.q[:] = 0; self.length = self.vias = 0.0
            if self.resets >= 2 and len(self.groups) > 2:
                # drop the terminals that cannot be reached from the sink at all (enclosed pads) and route the rest
                lost = self.unreachable()
                if lost and len(self.groups) - len(lost) >= 2:
                    self.groups = [g for gi, g in enumerate(self.groups) if gi not in lost]
                    self.skipped += len(lost); self.resets = 0
                    return
            if self.resets >= 3:
                self.failed = True
            return
        self.q = q
        self.p, self.E = p, (Dx, Dy, Dse, Dsw, Dz)
        Dn = self.D
        Dn += self.dt * (np.power(q, self.mu, dtype=np.float32) - Dn)
        if self.norm:
            m = float(np.where(self.active & ~self.own, Dn, 0).max()) or 1.0
            Dn /= m
        Dn[self.own] = self.dmax
        Dn[~self.active] = 0
        dead = self.active & ~self.own & (Dn < self.prune)
        self.active &= ~dead
        Dn[~self.active] = 0
        np.clip(Dn, 0, self.dmax, out=Dn)
        self.D = Dn


class Physarum:
    def __init__(self, owner, net_pads, *, via_cost=4.0, cell=0.5, pitch=0.35, kappa=2.0, dt=0.7, mu=1.5,
                 dmin=0.0, dmax=4.0, prune=1e-3, margin=0.3, seed=0, rotate=20, sweeps=0, norm=False, ringof=None,
                 crowd=1.0, walls=False):
        self.owner = owner; L, H, W = owner.shape; self.N = len(net_pads)
        self.cell, self.kappa, self.rotate, self.walls = cell, kappa, rotate, walls
        self.cap_cell = cell / pitch                                # tracks per cell (pictures, statistics)
        self.reach = max(1, int(round(pitch / cell)))             # cells within one track pitch
        # other nets' flow tolerated within one track pitch: a fraction `crowd` of what fits beside one track in
        # the (2r+1)-cell square (each track through it counts 2r+1 units); 0 = nothing, 1 = fill the square
        self.cap = crowd * max(0.0, (2 * self.reach + 1) * cell / pitch - 1) * (2 * self.reach + 1) + 0.25
        self.rng = np.random.default_rng(seed)
        self.net_pads, self.ringof, self.margin = net_pads, ringof, margin
        self.kw = dict(via_g=cell / via_cost, dmax=dmax, mu=mu, dt=dt, dmin=dmin, prune=prune, norm=norm)
        self.nets = [self.make(n, margin) for n in range(self.N)]
        self.usage = np.zeros((L, H, W), np.float32)
        self.history = np.zeros((L, H, W), np.float32)   # crowding seen in earlier rounds (PathFinder's history)
        self.steps = 0; self.last_sweeps = 0

    def ripup(self, frac=0.34):
        """End of a round: the most contested nets (contention > 0, top `frac`) grow back from scratch on their
        whole region; the crowding they met stays in `history`, so they regrow around it. Returns how many."""
        length, vias, cont = self.report()
        order = [i for i in np.argsort(-cont) if cont[i] > 1e-3][:max(1, int(frac * self.N))]
        excess = np.maximum(self.blur(self.usage) - self.cap, 0)
        self.history += excess / max(1, self.reach * 2 + 1)
        for i in order:
            self.nets[i] = self.make(i, None if self.nets[i].grown else self.margin)
            self.nets[i].hard = self.walls
        return len(order)

    def make(self, n, margin):
        """A net on its region of interest: the pads' box plus `margin` of its size plus 5 mm; margin None = the
        whole board (for a net whose pads turned out not to be connected inside the smaller box)."""
        L, H, W = self.owner.shape; pads = self.net_pads[n]
        if margin is None:
            box = (0, H, 0, W)
        else:
            ys = np.concatenate([g[1] for g in pads]); xs = np.concatenate([g[2] for g in pads])
            my = int(margin * (ys.max() - ys.min())) + int(5 / self.cell); mx = int(margin * (xs.max() - xs.min())) + int(5 / self.cell)
            box = (max(0, ys.min() - my), min(H, ys.max() + my + 1), max(0, xs.min() - mx), min(W, xs.max() + mx + 1))
        net = Net(n, self.owner, pads, box, rng=self.rng, ringof=self.ringof, **self.kw)
        net.grown = margin is None
        return net

    def blur(self, a, reach=None):
        """Sum over the neighbourhood within one track pitch (a square of 2r+1 cells per layer): what a track at
        this cell would have to keep clear of."""
        from scipy.ndimage import uniform_filter
        k = 2 * (self.reach if reach is None else reach) + 1
        return uniform_filter(a, size=(1, k, k), mode="constant") * (k * k)

    def step(self):
        usage = self.blur(self.usage) + self.history
        raw = self.blur(self.usage, 1)                                 # the cells right next to other tubes
        new_usage = np.zeros_like(self.usage)
        for i, net in enumerate(self.nets):
            if net.failed and "walled" not in net.why:
                if not net.grown:
                    net = self.nets[i] = self.make(i, None)         # second chance on the whole board
                elif net.hard:
                    net = self.nets[i] = self.make(i, None)         # third: without the walls (connected beats clean)
                    net.hard = False
            y0, y1, x0, x1 = net.box
            bq = self.blur(net.q)
            others = usage[:, y0:y1, x0:x1] - bq - self.cap
            crowd = raw[:, y0:y1, x0:x1] - self.blur(net.q, 1)      # other nets' tubes in the adjacent cells
            net.step(self.steps, self.rotate, others, self.kappa, crowd)
            new_usage[:, y0:y1, x0:x1] += net.q
        self.usage = new_usage
        self.steps += 1

    @property
    def q(self):
        """Full-board flow per net (for the picture); builds the big array on demand."""
        L, H, W = self.owner.shape
        q = np.zeros((self.N, L, H, W), np.float32)
        for n, net in enumerate(self.nets):
            y0, y1, x0, x1 = net.box; q[n, :, y0:y1, x0:x1] = net.q
        return q

    def report(self):
        length = np.array([n.length for n in self.nets]); vias = np.array([n.vias for n in self.nets])
        cont = np.zeros(self.N)
        usage = self.blur(self.usage)
        for i, net in enumerate(self.nets):
            y0, y1, x0, x1 = net.box
            # any other net's tube within a track pitch counts, outside the pad exits (pins are close by design)
            others = np.maximum(usage[:, y0:y1, x0:x1] - self.blur(net.q) - 0.25, 0) * ~net.exit
            cont[i] = float((net.q * others).sum() / max(net.q.sum(), 1e-9))
        return length, vias, cont


# ----------------------------------------------------------------------------------------------- driver
def keepouts(text):
    """Zones that forbid tracks: [(layers set of 0/1, polygon [(x, y)...])], top-level or inside footprints."""
    out = []
    for m in re.finditer(r"^(\t+)\(zone\n(.*?)\n\1\)\n", text, re.S | re.M):   # closes at its own indentation
        body = m.group(2)
        if "(keepout" not in body or "(tracks not_allowed)" not in body:
            continue
        ly = re.search(r'\(layers? ((?:"[^"]+"\s*)+)\)', body)
        if not ly:
            continue
        names = re.findall(r'"([^"]+)"', ly.group(1))
        layers = {l for l, n in ((0, "F.Cu"), (1, "B.Cu")) if n in names or "*.Cu" in names or "F&B.Cu" in names}
        if not layers:
            continue
        poly = re.search(r"\(polygon\s*\(pts(.*?)\n\s*\)", body, re.S)
        if not poly:
            continue
        pts = [(float(a), float(b)) for a, b in re.findall(r"\(xy ([-\d.]+) ([-\d.]+)\)", poly.group(1))]
        if len(pts) >= 3:
            out.append((layers, pts))
    return out


def kept_copper(text, routed):
    """Top-level segments and vias of nets that are not routed: [(kind, layer 0/1, points, width)]."""
    out = []
    for m in re.finditer(r"^\t\((segment|via)\n(.*?)^\t\)\n", text, re.M | re.S):
        kind, body = m.group(1), m.group(2)
        net = re.search(r'\(net "((?:[^"\\]|\\.)*)"\)', body)
        if net and net.group(1) in routed:
            continue
        if kind == "segment":
            st = re.search(r"\(start ([-\d.]+) ([-\d.]+)\)", body); en = re.search(r"\(end ([-\d.]+) ([-\d.]+)\)", body)
            wd = re.search(r"\(width ([-\d.]+)\)", body); ly = re.search(r'\(layer "([^"]+)"\)', body)
            if st and en and wd and ly and ly.group(1) in ("F.Cu", "B.Cu"):
                out.append(("segment", 0 if ly.group(1) == "F.Cu" else 1,
                            ((float(st.group(1)), float(st.group(2))), (float(en.group(1)), float(en.group(2)))), float(wd.group(1))))
        else:
            at = re.search(r"\(at ([-\d.]+) ([-\d.]+)\)", body); sz = re.search(r"\(size ([-\d.]+)\)", body)
            if at and sz:
                for l in (0, 1):
                    out.append(("via", l, ((float(at.group(1)), float(at.group(2))),), float(sz.group(1))))
    return out


def setup(board, names, cell, clearance, width=0.2, copper=(), keep=()):
    owner, ringof, origin = rasterize(board, names, cell, clearance, width, copper, keep)
    net_pads, keep = [], []
    for i, n in enumerate(names):
        groups = []
        for pi in board.nets[n]:
            name, x, y, w, h, rot, front, back = board.pads[pi]
            for l, on in ((0, front), (1, back)):
                if not on:
                    continue
                r = 0.5 * np.hypot(w, h) + cell
                ix0 = max(0, int((x - r - origin[0]) / cell)); ix1 = int((x + r - origin[0]) / cell) + 2
                iy0 = max(0, int((y - r - origin[1]) / cell)); iy1 = int((y + r - origin[1]) / cell) + 2
                sub = owner[l, iy0:iy1, ix0:ix1] == i
                ys, xs = np.nonzero(sub)
                if len(ys):
                    groups.append((l, ys + iy0, xs + ix0))
                    break                                       # one layer group per pad is enough
        # pads of one net that share cells (stacked pads, pad pairs of a connector) are one terminal
        merged = []
        for g in groups:
            cells = set(zip([g[0]] * len(g[1]), g[1].tolist(), g[2].tolist()))
            hit = [m for m in merged if m[0] == g[0] and m[1] & cells]
            for m in hit:
                merged.remove(m); cells |= m[1]
            merged.append((g[0], cells))
        groups = [(l, np.array([c[1] for c in sorted(cs)]), np.array([c[2] for c in sorted(cs)])) for l, cs in merged]
        if len(groups) >= 2:
            net_pads.append(groups); keep.append(i)
    if len(keep) != len(names):
        # drop nets whose pads got swallowed (all cells blocked): renumber the owner map
        remap = np.full(len(names) + 2, -2, np.int32); remap[-1] = -1; remap[-2] = -2
        for new, old in enumerate(keep):
            remap[old] = new
        owner = remap[owner]; ringof = remap[ringof]
        names = [names[i] for i in keep]
    return owner, ringof, origin, names, net_pads


def straight(board, names, nets):
    """Straight pad-to-pad distance per net (mm): MST length over pads, a lower bound on any route."""
    out = []
    for n in names:
        pts = np.array([(board.pads[i][1], board.pads[i][2]) for i in board.nets[n]])
        k = len(pts); inT = np.zeros(k, bool); inT[0] = True; d = np.hypot(*(pts - pts[0]).T); tot = 0.0
        for _ in range(k - 1):
            d[inT] = np.inf; j = int(np.argmin(d)); tot += d[j]; inT[j] = True
            d = np.minimum(d, np.hypot(*(pts - pts[j]).T))
        out.append(tot)
    return np.array(out)


def frame(sim):
    """(tubes image, usage image) as BGR uint8, F.Cu | B.Cu side by side, upscaled to ~1024 px wide per layer."""
    import cv2
    qall = sim.q
    N, L, H, W = qall.shape
    hue = (np.arange(N) * 0.618033988749895) % 1.0
    hsv = np.stack([hue * 179, np.full(N, 230.0), np.full(N, 255.0)], 1).astype(np.uint8).reshape(N, 1, 3)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(N, 3).astype(np.float32)
    panels = []
    for l in range(L):
        w = np.clip(qall[:, l] / max(sim.cap_cell, 1e-6), 0, 1)         # (N, H, W)
        img = np.clip(np.tensordot(w, rgb, axes=(0, 0)), 0, 255)         # (H, W, 3)
        base = np.full((H, W, 3), 28.0, np.float32)
        base[sim.owner[l] == -2] = 70
        pad = sim.owner[l] >= 0
        base[pad] = rgb[sim.owner[l][pad]] * 0.45 + 40
        panels.append(np.where(img.sum(-1, keepdims=True) > 8, np.maximum(img, base * 0.3), base))
    usage = sim.usage
    um = []
    for l in range(L):
        u = usage[l] / max(sim.cap_cell, 1e-6)
        img = np.zeros((H, W, 3), np.float32)
        img[..., 0] = np.clip(255 * (1 - u / 2), 0, 255) * (u > 0.05)
        img[..., 1] = np.clip(255 * (u / 2), 0, 255) * (u > 0.05)
        img[..., 2] = np.clip(255 * (u - 1), 0, 255)
        img[sim.owner[l] == -2] = 60
        img[sim.owner[l] >= 0] = 90
        um.append(img)
    scale = max(1, int(1024 / W))

    def up(a):
        return cv2.resize(np.clip(a, 0, 255).astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    sep = np.full((H * scale, 8, 3), 255, np.uint8)
    top = np.concatenate([up(panels[0]), sep, up(panels[1])], 1)
    bot = np.concatenate([up(um[0]), sep, up(um[1])], 1)
    return top, bot


def render(sim, names, path, origin, cell):
    import cv2
    top, bot = frame(sim)
    cv2.imwrite(str(path), top)
    cv2.imwrite(str(path).replace(".png", "_usage.png"), bot)
    return top


class Animation:
    """Shows every frame in a pygame window and/or pipes it into ffmpeg (mp4, then a gif of the same)."""

    def __init__(self, sim, names, window, video, fps, title=""):
        import cv2
        self.sim, self.window, self.video, self.fps = sim, window, video, fps
        top, _ = frame(sim)
        self.h, self.w = top.shape[:2]
        self.w += self.w % 2; self.h += self.h % 2                     # yuv420p wants even sizes
        self.screen = self.proc = None
        if window:
            import pygame
            pygame.init(); self.pygame = pygame
            self.screen = pygame.display.set_mode((self.w, self.h)); pygame.display.set_caption(f"physarum: {title}")
        if video:
            import subprocess
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", video], stdin=subprocess.PIPE)

    def frame(self, step):
        import cv2
        top, _ = frame(self.sim)
        img = np.zeros((self.h, self.w, 3), np.uint8); img[:top.shape[0], :top.shape[1]] = top
        cv2.putText(img, f"step {step}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if self.proc:
            self.proc.stdin.write(img.tobytes())
        if self.screen:
            pg = self.pygame
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    return False
            surf = pg.image.frombuffer(np.ascontiguousarray(img[..., ::-1]).tobytes(), (self.w, self.h), "RGB")
            self.screen.blit(surf, (0, 0)); pg.display.flip()
        return True

    def close(self):
        if self.proc:
            self.proc.stdin.close(); self.proc.wait()
            import subprocess
            gif = str(Path(self.video).with_suffix(".gif"))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.video, "-vf",
                            f"fps=10,scale={min(960, self.w)}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                            gif])
            print(f"wrote {self.video}, {gif}")
        if self.screen:
            self.pygame.quit()


NB = ((0, 0, 1, "x", 0), (0, 0, -1, "x", 1), (0, 1, 0, "y", 0), (0, -1, 0, "y", 1), (0, 1, 1, "se", 0), (0, -1, -1, "se", 1),
      (0, 1, -1, "sw", 0), (0, -1, 1, "sw", 1), (1, 0, 0, "z", 0), (-1, 0, 0, "z", 1))


def trace(net):
    """Polylines (lists of (layer, y, x) in ROI cells) from every source terminal down the flow to the sink or into
    an already traced branch: at each cell take the neighbour that receives the most flow (p lower, D p-drop largest)."""
    if net.p is None or net.failed:
        return []
    p = net.p; Dx, Dy, Dse, Dsw, Dz = net.E
    L, H, W = p.shape
    s, sink = net.sources(net.last_step, net.rotate)
    taken = np.zeros(p.shape, bool); taken |= sink
    paths = []

    def cond(l, y, x, dl, dy, dx):
        yy, xx = y + dy, x + dx
        if dl:
            return Dz[y, x]
        if dy == 0:
            return Dx[l, y, min(x, xx)]
        if dx == 0:
            return Dy[l, min(y, yy), x]
        if dy == dx:
            return Dse[l, min(y, yy), min(x, xx)]
        return Dsw[l, min(y, yy), min(x, xx)]

    for gi, (l, ys, xs) in enumerate(net.groups):
        if sink[l, ys[0], xs[0]]:
            continue
        # start at the terminal cell with the largest outflow
        best, start = -1.0, None
        for y, x in zip(ys, xs):
            for dl, dy, dx, _, _ in NB:
                ll, yy, xx = l + dl, y + dy, x + dx
                if 0 <= ll < L and 0 <= yy < H and 0 <= xx < W and not net.own[ll, yy, xx]:
                    f = cond(l, y, x, dl, dy, dx) * (p[l, y, x] - p[ll, yy, xx])
                    if f > best:
                        best, start = f, (l, y, x)
        if start is None:
            continue
        path = [start]; cur = start; taken[cur] = True
        for _ in range(4 * (H + W)):
            l, y, x = cur; best, nxt = 0.0, None
            for dl, dy, dx, _, _ in NB:
                ll, yy, xx = l + dl, y + dy, x + dx
                if 0 <= ll < L and 0 <= yy < H and 0 <= xx < W:
                    f = cond(l, y, x, dl, dy, dx) * (p[l, y, x] - p[ll, yy, xx])
                    if f > best:
                        best, nxt = f, (ll, yy, xx)
            if nxt is None:
                break
            path.append(nxt)
            if taken[nxt]:
                break
            taken[nxt] = True
            cur = nxt
        if len(path) > 1:
            paths.append(path)
    return paths


def chaikin(pts, rounds=2):
    for _ in range(rounds):
        if len(pts) < 3:
            break
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        out.append(pts[-1]); pts = out
    return pts


def write_pcb(pcb, out, sim, names, origin, cell, width, via, rip=True):
    """Copy of the board with the tubes as tracks: per net the traced flow paths, smoothed (Chaikin), as segments
    on F.Cu / B.Cu and a via wherever a path changes layer. Existing tracks and vias of the routed nets are removed
    first (`rip`). Returns (segments, vias) written."""
    import uuid
    text = Path(pcb).read_text()
    routed = set(names)
    if rip:
        # drop top-level (segment|arc|via ...) blocks of the routed nets
        out_parts, pos = [], 0
        for m in re.finditer(r"^\t\((segment|arc|via)\n", text, re.M):
            start = m.start()
            if start < pos:
                continue
            end = text.index("\n\t)\n", start) + 4
            block = text[start:end]
            mm = re.search(r'\(net "((?:[^"\\]|\\.)*)"\)', block)
            if mm and mm.group(1) in routed:
                out_parts.append(text[pos:start]); pos = end
        out_parts.append(text[pos:]); text = "".join(out_parts)
    x0, y0 = origin
    layers = ("F.Cu", "B.Cu")
    items = []; nseg = nvia = 0
    for i, net in enumerate(sim.nets):
        name = names[i].replace("\\", "\\\\").replace('"', '\\"')
        yb, _, xb, _ = net.box
        centres = {}                                              # terminal cell -> its pad centre (mm)
        for l, ys, xs in net.groups:
            cx, cy = x0 + (xs.mean() + xb) * cell, y0 + (ys.mean() + yb) * cell
            for y, x in zip(ys, xs):
                centres[(l, y, x)] = (cx, cy)
        for path in trace(net):
            for end in (path[0], path[-1]):
                if end in centres:
                    ex, ey = centres[end]
                    px, py = x0 + (end[2] + xb) * cell, y0 + (end[1] + yb) * cell
                    if abs(ex - px) + abs(ey - py) > 1e-6:
                        items.append((0, end[0], px, py, ex, ey))
            # split at layer changes, smooth each run, emit segments; via at the change
            runs, run = [], [path[0]]
            for a, b in zip(path[:-1], path[1:]):
                if b[0] != a[0]:
                    runs.append(run); run = [b]
                    items.append((1, b[0], x0 + (b[2] + xb) * cell, y0 + (b[1] + yb) * cell, None, None))
                else:
                    run.append(b)
            runs.append(run)
            for run in runs:
                pts = [(x0 + (c[2] + xb) * cell, y0 + (c[1] + yb) * cell) for c in run]
                pts = chaikin(pts) if len(pts) > 2 else pts
                for a, b in zip(pts[:-1], pts[1:]):
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 1e-6:
                        continue
                    items.append((0, run[0][0], a[0], a[1], b[0], b[1]))
        for kind, l, ax, ay, bx, by in items:
            if kind == 0:
                text_item = (f'\t(segment\n\t\t(start {ax:.4f} {ay:.4f})\n\t\t(end {bx:.4f} {by:.4f})\n\t\t(width {width})\n'
                             f'\t\t(layer "{layers[l]}")\n\t\t(net "{name}")\n\t\t(uuid "{uuid.uuid4()}")\n\t)\n')
                nseg += 1
            else:
                text_item = (f'\t(via\n\t\t(at {ax:.4f} {ay:.4f})\n\t\t(size {via[0]})\n\t\t(drill {via[1]})\n'
                             f'\t\t(layers "F.Cu" "B.Cu")\n\t\t(net "{name}")\n\t\t(uuid "{uuid.uuid4()}")\n\t)\n')
                nvia += 1
            text = text[:text.rstrip().rfind(")")] + text_item + ")\n"
        items = []
    Path(out).write_text(text)
    return nseg, nvia


def write_seed(pcb, out, sim, names, origin, cell, board, length, vias, cont, stretch, tiers):
    """Seed file for a topological router: per net the traced tube polylines (cell centres, mm, KiCad frame, one
    per source terminal down to the sink or into an earlier branch), split into runs per layer with the via
    positions between them, plus the scores. Raw cell-centre polylines, not smoothed: the consumer maps them onto
    its own obstacle graph (which side of each obstacle, which layer) and draws its own geometry."""
    x0, y0 = origin
    layers = ("F.Cu", "B.Cu")
    nets = {}
    for i, net in enumerate(sim.nets):
        yb, _, xb, _ = net.box
        paths, vs = [], []
        for path in trace(net):
            run = [path[0]]
            for a, b in zip(path[:-1], path[1:]):
                if b[0] != a[0]:
                    paths.append({"layer": layers[run[0][0]], "points": [[round(x0 + (c[2] + xb) * cell, 4), round(y0 + (c[1] + yb) * cell, 4)] for c in run]})
                    vs.append([round(x0 + (b[2] + xb) * cell, 4), round(y0 + (b[1] + yb) * cell, 4)])
                    run = [b]
                else:
                    run.append(b)
            paths.append({"layer": layers[run[0][0]], "points": [[round(x0 + (c[2] + xb) * cell, 4), round(y0 + (c[1] + yb) * cell, 4)] for c in run]})
        nets[names[i]] = {
            "pads": len(board.nets[names[i]]), "routed": bool(not net.failed and net.p is not None), "skipped_pads": int(net.skipped),
            "length_mm": round(float(length[i]) * cell, 3), "straight_mm": round(float(stretch[i] and length[i] * cell / stretch[i]), 3),
            "stretch": round(float(stretch[i]), 3), "vias": round(float(vias[i]), 2), "contention": round(float(cont[i]), 4),
            "tier": int(tiers[i]), "paths": paths, "via_points": vs,
        }
    doc = {"generator": "physarum_route.py", "board": str(Path(pcb).name), "units": "mm", "frame": "kicad (y down)",
           "cell_mm": cell, "layers": list(layers), "nets": nets}
    Path(out).write_text(json.dumps(doc, indent=1))
    return len(nets), sum(len(n["paths"]) for n in nets.values())


def write_pro(pcb, names, tiers, out):
    """Copy of the project file with topor_p1..3 net classes and, per routed net, a net class assignment
    (KiCad 9+: `netclass_assignments` maps a net to a list of classes; earlier topor_p* entries are replaced)."""
    pro = Path(pcb).with_suffix(".kicad_pro")
    d = json.loads(pro.read_text()) if pro.exists() else {"net_settings": {"classes": [], "meta": {"version": 3}}}
    ns = d.setdefault("net_settings", {})
    classes = ns.setdefault("classes", [])
    have = {c.get("name") for c in classes}
    for t in (1, 2, 3):
        if f"topor_p{t}" not in have:
            classes.append({"name": f"topor_p{t}", "priority": 2147483647})
    asg = ns.get("netclass_assignments") or {}
    ns["netclass_assignments"] = asg
    for n, t in zip(names, tiers):
        old = asg.get(n, [])
        old = old if isinstance(old, list) else [old]
        asg[n] = [c for c in old if not str(c).startswith("topor_p")] + [f"topor_p{t}"]
    Path(out).write_text(json.dumps(d, indent=2))


def selftest(steps=300, sweeps=4, mu=1.5):
    """Two nets, one straight across an obstacle the other must go around; checks tubes form and stretch is right."""
    H, W = 40, 60
    owner = np.full((2, H, W), -1, np.int32)
    owner[:, 10:30, 28:32] = -2                       # a wall
    owner[0, 19:21, 5:7] = 0; owner[0, 19:21, 53:55] = 0      # net 0: through the wall's shadow
    owner[1, 3:5, 5:7] = 1; owner[1, 3:5, 53:55] = 1          # net 1: clear path on B.Cu
    pads = [[(0, *np.nonzero(owner[0, :, :10] == 0)), (0, np.nonzero(owner[0, :, 50:] == 0)[0], np.nonzero(owner[0, :, 50:] == 0)[1] + 50)],
            [(1, *np.nonzero(owner[1, :, :10] == 1)), (1, np.nonzero(owner[1, :, 50:] == 1)[0], np.nonzero(owner[1, :, 50:] == 1)[1] + 50)]]
    sim = Physarum(owner, pads, cell=0.5, via_cost=4.0, kappa=0.0, mu=mu)
    t0 = time.time()
    for _ in range(steps):
        sim.step()
    length, vias, cont = sim.report()
    q = sim.q
    print(f"arrived {[round(n.arrived, 3) for n in sim.nets]}, resets {[n.resets for n in sim.nets]}, active cells "
          f"{[int(n.active.sum()) for n in sim.nets]}, max q {q[0].max():.2f} {q[1].max():.2f}")
    print(f"selftest: {time.time() - t0:.1f} s; net0 length {length[0]:.1f} cells (straight 48, must detour over the wall: "
          f"octile ~58), net1 {length[1]:.1f} (expect ~48); vias {vias[0]:.2f} {vias[1]:.2f}; tube cells > cap/2: "
          f"{(q[0] > sim.cap_cell / 2).sum()} / {(q[1] > sim.cap_cell / 2).sum()}")
    ok = 52 < length[0] < 68 and 44 < length[1] < 54 and (q[0] > sim.cap_cell / 2).sum() > 30
    print("selftest", "OK" if ok else "FAIL")
    try:
        render(sim, ["a", "b"], Path("physarum_selftest.png"), (0, 0), 0.5)
    except Exception as e:
        print("no picture:", e)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcb", nargs="?")
    ap.add_argument("--cell", type=float, default=0.0, help="grid cell, mm (default: longest side / 250, 0.2..0.5)")
    ap.add_argument("--clearance", type=float, default=0.0, help="mm; default: the project's Default net class, else 0.2")
    ap.add_argument("--pitch", type=float, default=0.0, help="track + clearance, mm: capacity per cell; default from the project")
    ap.add_argument("--steps", type=int, default=40, help="steps per round (tubes settle in ~25)")
    ap.add_argument("--rounds", type=int, default=4, help="rip-up rounds: the most contested nets regrow with the others as walls")
    ap.add_argument("--prune", type=float, default=1e-3, help="drop cells whose tube got thinner than this")
    ap.add_argument("--dt", type=float, default=0.7, help="tube adaptation per step (explicit Euler, < 1)")
    ap.add_argument("--window", action="store_true", help="watch the tubes grow (pygame)")
    ap.add_argument("--video", default="", help="write the growth as an mp4 (ffmpeg) and a gif")
    ap.add_argument("--every", type=int, default=1, help="render every k steps")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--mu", type=float, default=1.5, help="tube response exponent (> 1: one path wins)")
    ap.add_argument("--via-cost", type=float, default=4.0, help="a layer change costs this many mm of track")
    ap.add_argument("--kappa", type=float, default=2.0, help="competition strength (0 = nets ignore each other)")
    ap.add_argument("--crowd", type=float, default=1.0, help="other nets' flow tolerated within a track pitch, 0 (none) .. 1")
    ap.add_argument("--walls", action="store_true", help="ripped-up nets regrow with the other tubes as walls (fewer shorts, more failures)")
    ap.add_argument("--max-nets", type=int, default=0)
    ap.add_argument("--max-pads", type=int, default=0, help="skip nets with more pads than this (LED matrices: hours of solves); 0 = route all")
    ap.add_argument("--skip", default="", help="extra net name regex to skip (power nets are skipped by default)")
    ap.add_argument("--route-power", action="store_true")
    ap.add_argument("--write-pro", action="store_true")
    ap.add_argument("--write-pcb", action="store_true", help="copy of the board with the tubes as tracks and vias (opens in KiCad)")
    ap.add_argument("--write-seed", action="store_true", help="<board>_physarum.seed.json: the tube polylines and scores per net, for a topological router")
    ap.add_argument("--keep-copper", action="store_true", help="with --write-pcb: keep the routed nets' existing tracks")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(max(a.steps, 100), 0, a.mu) else 1)
    board = Board(a.pcb)
    rules = project_rules(a.pcb) or (0.2, 0.2)
    a.clearance = a.clearance or rules[0]
    a.pitch = a.pitch or (rules[0] + rules[1])
    names = [n for n, ps in board.nets.items() if len(ps) >= 2]
    if not a.route_power:
        names = [n for n in names if not POWER.match(n.rsplit("/", 1)[-1])]
    if a.skip:
        rx = re.compile(a.skip); names = [n for n in names if not rx.search(n)]
    if a.max_pads:
        names = [n for n in names if len(board.nets[n]) <= a.max_pads]
    names.sort(key=lambda n: -len(board.nets[n]))
    if a.max_nets:
        names = names[:a.max_nets]
    x0, y0, x1, y1 = board.box
    if not a.cell:
        a.cell = float(np.clip(max(x1 - x0, y1 - y0) / 250, 0.2, 0.5))
    print(f"{Path(a.pcb).name}: {len(board.pads)} pads, {len(board.nets)} nets, routing {len(names)} "
          f"({sum(len(board.nets[n]) for n in names)} pads); board {x1 - x0:.0f} x {y1 - y0:.0f} mm")
    names0 = list(names)
    text = Path(a.pcb).read_text()
    copper = kept_copper(text, set(names)); keep = keepouts(text)
    owner, ringof, origin, names, net_pads = setup(board, names, a.cell, a.clearance, rules[1], copper, keep)
    L, H, W = owner.shape
    print(f"grid {W} x {H} x 2 at {a.cell:.2f} mm (clearance {a.clearance} mm, track pitch {a.pitch:.3f} mm = "
          f"{a.cell / a.pitch:.2f} tracks per cell); free {np.mean(owner == -1) * 100:.0f}%; {len(copper)} kept tracks/vias and "
          f"{len(keep)} track keepouts as obstacles")
    sim = Physarum(owner, net_pads, via_cost=a.via_cost, cell=a.cell, pitch=a.pitch, kappa=a.kappa, mu=a.mu, seed=a.seed,
                   prune=a.prune, dt=a.dt, ringof=ringof, crowd=a.crowd, walls=a.walls)
    t0 = time.time()
    anim = Animation(sim, names, a.window, a.video, a.fps, title=Path(a.pcb).stem) if (a.window or a.video) else None
    dropped = [n for n in names0 if n not in names]
    if dropped:
        print(f"dropped (pads swallowed at this cell size): {dropped[:8]}{' ...' if len(dropped) > 8 else ''}")
    total = a.steps * a.rounds; stop = False
    for r in range(a.rounds):
        if r:
            n = sim.ripup()
            print(f"round {r + 1}: {n} contested nets ripped up and regrowing", flush=True)
        for i in range(a.steps):
            sim.step()
            k = r * a.steps + i
            if anim and (k % a.every == 0 or k == total - 1):
                if not anim.frame(k + 1):
                    print("window closed"); stop = True; break
            if (i + 1) % 20 == 0 or i == a.steps - 1 or k == 0:
                length, vias, cont = sim.report()
                act = sum(int(n.active.sum()) for n in sim.nets)
                print(f"step {k + 1}: {time.time() - t0:.0f} s, active cells {act / 1e3:.0f}k, mean length {length.mean() * a.cell:.1f} mm, "
                      f"vias/net {vias.mean():.2f}, cells over capacity {(sim.blur(sim.usage) > 2 * sim.cap).sum()}, contested nets "
                      f"{(cont > 1e-3).sum()}, failed {sum(n.failed for n in sim.nets)}", flush=True)
        if stop:
            break
    length, vias, cont = sim.report()
    part = [(names[i], n.skipped) for i, n in enumerate(sim.nets) if n.skipped and not n.failed]
    if part:
        print(f"{len(part)} nets routed without their enclosed pads: {[f'{n} (-{k})' for n, k in part[:8]]}{' ...' if len(part) > 8 else ''}")
    bad = [(names[i], n.why) for i, n in enumerate(sim.nets) if n.failed]
    if bad:
        print(f"{len(bad)} nets gave up:")
        for name, why in bad[:8]:
            print(f"  {name}: {why}")
    st = straight(board, names, board.nets)
    stretch = length * a.cell / np.maximum(st, 1e-6)
    score = cont + np.maximum(stretch - 1.2, 0)
    order = np.argsort(-score)
    n1 = max(1, len(names) // 10); n2 = max(1, len(names) * 3 // 10)
    tiers = np.full(len(names), 3); tiers[order[:n1 + n2]] = 2; tiers[order[:n1]] = 1
    print("net                              pads  mm   straight stretch vias contention tier")
    for i in order[:25]:
        print(f"{names[i][:32]:32s} {len(board.nets[names[i]]):4d} {length[i] * a.cell:6.1f} {st[i]:7.1f} {stretch[i]:6.2f} "
              f"{vias[i]:5.2f} {cont[i]:8.3f}  p{tiers[i]}")
    stem = Path(a.pcb).stem
    if stem == "project":
        stem = Path(a.pcb).resolve().parent.name
    if anim:
        anim.close()
    render(sim, names, Path(f"{stem}_physarum.png"), origin, a.cell)
    print(f"wrote {stem}_physarum.png, {stem}_physarum_usage.png")
    if a.write_pcb:
        pro = project_rules(a.pcb)
        via = (0.6, 0.3)
        try:
            d = json.loads(Path(a.pcb).with_suffix(".kicad_pro").read_text())
            c = [c for c in d["net_settings"]["classes"] if c.get("name") == "Default"][0]
            via = (float(c.get("via_diameter", 0.6)), float(c.get("via_drill", 0.3)))
        except Exception:
            pass
        nseg, nvia = write_pcb(a.pcb, f"{stem}_physarum.kicad_pcb", sim, names, origin, a.cell, rules[1], via, rip=not a.keep_copper)
        print(f"wrote {stem}_physarum.kicad_pcb: {nseg} segments, {nvia} vias ({'kept' if a.keep_copper else 'replaced'} the routed nets' copper)")
    if a.write_seed:
        nn, npaths = write_seed(a.pcb, f"{stem}_physarum.seed.json", sim, names, origin, a.cell, board, length, vias, cont, stretch, tiers)
        print(f"wrote {stem}_physarum.seed.json: {nn} nets, {npaths} polylines")
    if a.write_pro:
        write_pro(a.pcb, names, tiers, f"{stem}_physarum.kicad_pro")
        print(f"wrote {stem}_physarum.kicad_pro ({n1} nets topor_p1, {n2} topor_p2, {len(names) - n1 - n2} topor_p3)")


if __name__ == "__main__":
    main()
