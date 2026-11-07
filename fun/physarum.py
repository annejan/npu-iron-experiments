#!/usr/bin/env python3
"""Physarum (slime mould) agents on the CPU, trail diffusion and decay on the NPU: the real network-forming
algorithm (Jones 2010), as a hybrid.

  ./env.sh python fun/physarum.py --steps 1500        write physarum.png
  ./env.sh python fun/physarum.py --window            pygame (N: reseed, ESC)

Agents sense the trail at three points ahead (distance SO, angles -SA, 0, +SA), turn toward the strongest by RA,
move SS, deposit. That part is gather/scatter (numpy, vectorised with bincount). Every step the trail map is then
blurred (3x3 mean) and decayed on the NPU through fun/conv2d.py. Food sources deposit constantly, so the network
grows between them. The honest division of labour: the NPU does the dense part (0.3 ms per 512x512 step), the CPU
the agents (about 10 ms per 100k agents); the point is the result, not the NPU load.

Agent model: Jeff Jones, "Characteristics of pattern formation and evolution in approximations of Physarum transport
networks", Artificial Life 16(2):127-153, 2010.
"""
import argparse, sys, time
import numpy as np
from conv2d import Conv2D

W = H = 512
PARAMS = dict(SA=22.5, RA=45.0, SO=9.0, SS=1.2, deposit=0.03, decay=0.92, agents=120_000, food=0.08)


class Physarum:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.trail = np.zeros((H, W), np.float32)
        self.food = np.zeros((H, W), np.float32)
        self.conv = Conv2D(3, np.full((3, 3), 1 / 9), W, H, post="decay", params=dict(decay=PARAMS["decay"]))
        self.reseed()

    def reseed(self):
        n = PARAMS["agents"]
        r = self.rng
        # agents start in a disc in the middle, random headings
        ang = r.uniform(0, 2 * np.pi, n); rad = np.sqrt(r.uniform(0, 1, n)) * 60
        self.x = (W / 2 + rad * np.cos(ang)).astype(np.float32)
        self.y = (H / 2 + rad * np.sin(ang)).astype(np.float32)
        self.h = r.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.trail[:] = 0
        self.food[:] = 0
        yy, xx = np.ogrid[0:H, 0:W]
        for _ in range(9):
            cx, cy = r.integers(40, W - 40), r.integers(40, H - 40)
            self.food[(xx - cx) ** 2 + (yy - cy) ** 2 <= 36] = PARAMS["food"]

    def sense(self, dx, dy):
        xi = np.floor(self.x + dx).astype(np.int32) % W          # integer modulo after the cast: float32 x % W can round to W
        yi = np.floor(self.y + dy).astype(np.int32) % H
        return self.trail[yi, xi]

    def step(self):
        P = PARAMS; r = self.rng
        sa, ra, so, ss = np.radians(P["SA"]), np.radians(P["RA"]), P["SO"], P["SS"]
        f = self.sense(so * np.cos(self.h), so * np.sin(self.h))
        fl = self.sense(so * np.cos(self.h - sa), so * np.sin(self.h - sa))
        fr = self.sense(so * np.cos(self.h + sa), so * np.sin(self.h + sa))
        turn = np.zeros_like(self.h)
        both = (f < fl) & (f < fr)
        turn[both] = r.choice([-ra, ra], both.sum())
        left = (fl > fr) & ~both & (fl > f)
        right = (fr > fl) & ~both & (fr > f)
        turn[left] = -ra; turn[right] = ra
        self.h += turn.astype(np.float32)
        self.x = (self.x + ss * np.cos(self.h)) % W
        self.y = (self.y + ss * np.sin(self.h)) % H
        idx = (np.floor(self.y).astype(np.int32) % H) * W + (np.floor(self.x).astype(np.int32) % W)
        dep = np.bincount(idx, minlength=W * H).reshape(H, W).astype(np.float32) * P["deposit"]
        self.trail = np.minimum(self.trail + dep + self.food, 1.0)
        q = np.round(self.trail * 32767).astype(np.int16)
        self.trail = self.conv.run(q).astype(np.float32) / 32767
        return self.conv.npu_ms

    def rgb(self):
        t = np.clip(self.trail * 2.5, 0, 1)
        img = np.empty((H, W, 3), np.float32)
        img[..., 0] = t ** 0.7; img[..., 1] = t ** 1.2 * 0.9; img[..., 2] = 0.15 + t ** 2 * 0.5
        img[self.food > 0] = (1.0, 0.25, 0.2)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args()
    sim = Physarum()
    sim.step()
    if a.window:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((W * a.scale, H * a.scale))
        font = pygame.font.SysFont(None, 24)
        n, t0 = 0, time.perf_counter()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_n:
                    sim.reseed()
            ms = sim.step(); n += 1
            surf = pygame.surfarray.make_surface(np.transpose(sim.rgb(), (1, 0, 2)))
            screen.blit(pygame.transform.smoothscale(surf, screen.get_size()), (0, 0))
            screen.blit(font.render(f"physarum {PARAMS['agents']} agents  step {n}  NPU blur {ms:.2f} ms  {n / (time.perf_counter() - t0):.0f} steps/s  N: reseed", True, (255, 255, 255)), (8, 8))
            pygame.display.flip()
    t0 = time.perf_counter(); npu = 0.0
    for i in range(a.steps):
        npu += sim.step()
    dt = time.perf_counter() - t0
    print(f"{a.steps} steps, {PARAMS['agents']} agents: {dt / a.steps * 1e3:.1f} ms/step total, NPU blur+decay {npu / a.steps:.2f} ms/step; "
          f"trail mean {sim.trail.mean():.4f}")
    import cv2
    cv2.imwrite("physarum.png", cv2.resize(sim.rgb()[..., ::-1], (W * a.scale, H * a.scale), interpolation=cv2.INTER_LINEAR))
    print("wrote physarum.png")


if __name__ == "__main__":
    main()
