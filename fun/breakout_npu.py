#!/usr/bin/env python3
"""The NPU learns Breakout from pixels: the Pong learner of pong_npu.py on another game.

  ./env.sh python fun/breakout_npu.py --train 800 --gif breakout.gif

128 games of a 28x28 Breakout run in numpy: a 5-pixel paddle on row 26, four rows of 2-pixel bricks, a 1-pixel ball.
A return of the ball is +0.5, a lost ball -1 (and a new ball); a cleared wall is rebuilt. Bricks are only scored:
rewarding them credited the moves between a return and the brick, when the paddle cannot change anything, and those
large random advantages drowned the signal (0.5 bricks per ball after 1000 updates, random play 0.34). The policy is
mnist_npu.py's perceptron on the NPU again (784-256-3), trained by REINFORCE exactly as in pong_npu.py.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mnist_npu as M
import pong_npu as P

SZ = P.SZ; NG = P.NG; PADW = 5; ROW0 = 3; NROW = 4; NCOL = 14; PY = 26


class Breakout:
    def __init__(self, rng):
        self.rng = rng
        self.bricks = np.ones((NG, NROW, NCOL), bool)
        self.px = np.full(NG, SZ / 2); self.bx = np.zeros(NG); self.by = np.zeros(NG); self.vx = np.zeros(NG); self.vy = np.zeros(NG)
        self.serve(np.ones(NG, bool)); self.prev = self.objects()
        self.lost = 0; self.hit_bricks = 0

    def serve(self, m):
        n = int(m.sum())
        self.bx[m] = self.rng.uniform(6, SZ - 6, n); self.by[m] = 14.0
        ang = self.rng.uniform(-0.7, 0.7, n); self.vx[m] = 0.9 * np.sin(ang); self.vy[m] = 0.9 * np.cos(ang)

    def objects(self):
        return np.round(self.bx).astype(int), np.round(self.by).astype(int), np.round(self.px).astype(int)

    def frame(self):
        f = np.zeros((NG, SZ, SZ), np.uint8); g = np.arange(NG)
        f[:, ROW0:ROW0 + NROW, :] = np.repeat(self.bricks, 2, axis=2).astype(np.uint8) * 150
        for (bx, by, px), val in ((self.prev, 90), (self.objects(), 255)):
            for dx in range(-(PADW // 2), PADW // 2 + 1):
                xx = np.clip(px + dx, 0, SZ - 1); f[g, PY, xx] = np.maximum(f[g, PY, xx], val)
            yy, xx = np.clip(by, 0, SZ - 1), np.clip(bx, 0, SZ - 1); f[g, yy, xx] = np.maximum(f[g, yy, xx], val)
        return f.reshape(NG, SZ * SZ)

    def step(self, act, opp_act=None):
        self.prev = self.objects()
        self.px = np.clip(self.px + (act - 1) * 1.0, PADW // 2, SZ - 1 - PADW // 2)
        nx, ny = self.bx + self.vx, self.by + self.vy
        side = (nx < 0) | (nx > SZ - 1); self.vx[side] *= -1; nx = np.clip(nx, 0, SZ - 1)
        top = ny < 0; self.vy[top] *= -1; ny[top] = -ny[top]
        r = np.zeros(NG); hit = np.zeros(NG, bool)
        row = np.floor(ny).astype(int) - ROW0; col = np.clip(np.floor(nx).astype(int) // 2, 0, NCOL - 1)
        inb = (row >= 0) & (row < NROW)
        g = np.nonzero(inb)[0]
        brick = np.zeros(NG, bool); brick[g] = self.bricks[g, row[g], col[g]]
        if brick.any():
            gb = np.nonzero(brick)[0]; self.bricks[gb, row[gb], col[gb]] = False
            self.vy[gb] *= -1; ny[gb] = self.by[gb]; self.hit_bricks += len(gb)     # scored, not rewarded (see the docstring)
        catch = (self.vy > 0) & (ny >= PY - 0.5) & (self.by < PY - 0.5) & (np.abs(nx - self.px) <= PADW / 2 + 0.6)
        if catch.any():
            gc = np.nonzero(catch)[0]; off = np.clip((nx[gc] - self.px[gc]) / (PADW / 2 + 0.6), -1, 1)
            sp = np.minimum(np.hypot(self.vx[gc], self.vy[gc]) * 1.01, 1.3); ang = off * 1.0
            self.vx[gc] = sp * np.sin(ang); self.vy[gc] = -sp * np.cos(ang); ny[gc] = PY - 0.6; hit[gc] = True
        self.bx, self.by = nx, ny
        out = self.by > SZ - 0.5
        if out.any(): r[out] = -1; self.lost += int(out.sum()); self.serve(out)
        clear = ~self.bricks.reshape(NG, -1).any(1)
        if clear.any(): self.bricks[clear] = True
        return r, hit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=0); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--gif"); ap.add_argument("--save")
    a = ap.parse_args()
    if not a.train: ap.print_help(); return
    P.Pong = Breakout                                                  # the learner builds whatever game P.Pong names
    L = P.Learner(a.lr, a.seed); t0 = time.perf_counter()
    for i in range(a.train):
        L.update()
        if (i + 1) % 50 == 0:
            g = L.game
            print(f"update {i + 1}: bricks per lost ball {g.hit_bricks / max(1, g.lost):.2f} (cumulative), "
                  f"NPU {np.mean(L.ms[-50:]):.1f} ms per update, {time.perf_counter() - t0:.0f} s")
            g.hit_bricks = 0; g.lost = 0
    if a.save: P.save(L.model, a.save)
    if a.gif: record_gif(L, a.gif); print("wrote", a.gif)


def record_gif(L, path, steps=240, games=(0, 1, 2, 3), scale=8):
    from PIL import Image, ImageDraw, ImageFont
    S = SZ * scale; pad = 6; head = 44; font = ImageFont.load_default(size=22)
    cols = [(255, 90, 90), (255, 170, 60), (240, 220, 80), (120, 220, 120)]
    frames = []; trails = {g: [] for g in games}
    for step in range(steps):
        x = L.game.frame(); a, _ = P.policy(L.model, x, L.rng)
        for g in games: trails[g] = (trails[g] + [(L.game.bx[g], L.game.by[g])])[-6:]
        r, _ = L.game.step(a)
        for g in games:
            if r[g] < 0: trails[g] = []
        im = Image.new("RGB", (2 * S + 3 * pad, 2 * S + 3 * pad + head), (10, 10, 16)); d = ImageDraw.Draw(im)
        d.text((pad, 12), f"Breakout, NPU-trained policy ({L.updates} updates)", fill=(235, 235, 235), font=font)
        for k, g in enumerate(games):
            ox, oy = pad + (k % 2) * (S + pad), head + pad + (k // 2) * (S + pad)
            d.rectangle([ox, oy, ox + S - 1, oy + S - 1], fill=(16, 18, 30))
            for rr in range(NROW):
                for cc in range(NCOL):
                    if L.game.bricks[g, rr, cc]:
                        d.rounded_rectangle([ox + cc * 2 * scale + 1, oy + (ROW0 + rr) * scale + 1, ox + (cc + 1) * 2 * scale - 2, oy + (ROW0 + rr + 1) * scale - 2], radius=2, fill=cols[rr])
            for t, (bx, by) in enumerate(trails[g]):
                rad = scale * (0.2 + 0.06 * t); c = 50 + 30 * t
                d.ellipse([ox + (bx + 0.5) * scale - rad, oy + (by + 0.5) * scale - rad, ox + (bx + 0.5) * scale + rad, oy + (by + 0.5) * scale + rad], fill=(c, c, min(255, c + 20)))
            px = L.game.px[g]
            d.rounded_rectangle([ox + (px - PADW / 2 + 0.5) * scale, oy + PY * scale, ox + (px + PADW / 2 + 0.5) * scale, oy + (PY + 1) * scale - 1], radius=3, fill=(90, 180, 255))
            bx, by = L.game.bx[g], L.game.by[g]
            d.ellipse([ox + bx * scale + 1, oy + by * scale + 1, ox + (bx + 1) * scale - 1, oy + (by + 1) * scale - 1], fill=(250, 250, 250))
        frames.append(im.quantize(colors=48, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=45, loop=0, optimize=True)


if __name__ == "__main__":
    main()
