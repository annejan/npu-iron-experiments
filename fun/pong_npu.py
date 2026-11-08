#!/usr/bin/env python3
"""The NPU learns to play Pong from pixels: policy gradient (REINFORCE) with the MNIST trainer of mnist_npu.py.

  ./env.sh python fun/pong_npu.py --train 300           300 updates, prints the share of points won
  ./env.sh python fun/pong_npu.py --window              watch four of the 64 games while it learns

The player is the right paddle; the left one is a script that follows the ball a little too slowly. The network is
mnist_npu.py's 784-256-16 perceptron with three live outputs (up, stay, down); its input is the 28x28 screen with the
previous frame's ball and paddles drawn at half brightness, so the direction of the ball is visible. 64 games run in
numpy on the CPU, the actions are sampled from the policy on the CPU with the weights of the latest update, and every
3008 decided moves (the moves of finished points) go to the NPU as one call of 47 minibatches: the gradient of
log pi(action) weighted by the move's advantage (its point's reward discounted back, standardised per batch), which
the trainer applies as G = -lr/B * advantage * (p - onehot(action)). Same kernel, same dispatch as the digits.
A return of the ball counts as a small reward of its own (the moves before it get the credit), and the learning rate
decays as lr0 / (1 + updates/300): without either, the policy rose to 55% of points won and then collapsed.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mnist_npu as M

SZ = 28; PADH = 5; NG = 64; NMB = 47; BATCH = NMB * M.B; NACT = 3; GAMMA = 0.97
HIT_REWARD = 0.5                # a return is an event too: the moves before it get this, discounted back
LR_HALF = 300                   # learning rate lr0 / (1 + updates / LR_HALF)


class Pong:
    """NG games at once. Ball and paddles in float pixel coordinates; the player is at x = SZ-2, the script at x = 1."""

    def __init__(self, rng, opp_speed=0.55):
        self.rng = rng; self.opp_speed = opp_speed
        self.bx = np.zeros(NG); self.by = np.zeros(NG); self.vx = np.zeros(NG); self.vy = np.zeros(NG)
        self.py = np.full(NG, SZ / 2); self.oy = np.full(NG, SZ / 2)
        self.prev = None
        self.serve(np.ones(NG, bool))
        self.prev = self.objects()

    def serve(self, m):
        n = int(m.sum())
        self.bx[m] = SZ / 2; self.by[m] = self.rng.uniform(6, SZ - 6, n)
        ang = self.rng.uniform(-0.6, 0.6, n); side = self.rng.choice([-1.0, 1.0], n)
        self.vx[m] = side * 0.9 * np.cos(ang); self.vy[m] = 0.9 * np.sin(ang)

    def objects(self):
        return (np.round(self.bx).astype(int), np.round(self.by).astype(int), np.round(self.py).astype(int), np.round(self.oy).astype(int))

    def frame(self):
        """uint8 (NG, 784): current objects at 255, the previous frame's at 110."""
        f = np.zeros((NG, SZ, SZ), np.uint8); g = np.arange(NG)
        for (bx, by, py, oy), val in ((self.prev, 110), (self.objects(), 255)):
            for dy in range(-(PADH // 2), PADH // 2 + 1):
                f[g, np.clip(py + dy, 0, SZ - 1), SZ - 2] = np.maximum(f[g, np.clip(py + dy, 0, SZ - 1), SZ - 2], val)
                f[g, np.clip(oy + dy, 0, SZ - 1), 1] = np.maximum(f[g, np.clip(oy + dy, 0, SZ - 1), 1], val)
            f[g, np.clip(by, 0, SZ - 1), np.clip(bx, 0, SZ - 1)] = np.maximum(f[g, np.clip(by, 0, SZ - 1), np.clip(bx, 0, SZ - 1)], val)
        return f.reshape(NG, SZ * SZ)

    def step(self, act):
        """act in {0 up, 1 stay, 2 down}. Returns reward (+1 point won, -1 lost, 0) and whether the player hit the ball."""
        self.prev = self.objects()
        self.py = np.clip(self.py + (act - 1) * 1.0, PADH // 2, SZ - 1 - PADH // 2)
        self.oy = self.oy + np.clip(self.by - self.oy, -self.opp_speed, self.opp_speed)
        self.oy = np.clip(self.oy, PADH // 2, SZ - 1 - PADH // 2)
        self.bx += self.vx; self.by += self.vy
        top = self.by < 0; bot = self.by > SZ - 1
        self.by[top] = -self.by[top]; self.by[bot] = 2 * (SZ - 1) - self.by[bot]; self.vy[top | bot] *= -1
        r = np.zeros(NG); hit = np.zeros(NG, bool)
        for side, px, pad in ((1, SZ - 2.5, self.py), (-1, 1.5, self.oy)):
            cross = (self.vx * side > 0) & ((self.bx - px) * side >= 0)
            catch = cross & (np.abs(self.by - pad) <= PADH / 2 + 0.6)
            off = (self.by - pad) / (PADH / 2 + 0.6)
            sp = np.hypot(self.vx, self.vy) * 1.02
            ang = np.clip(off * 0.9, -1.0, 1.0)
            self.vx[catch] = -side * np.minimum(sp[catch], 1.6) * np.cos(ang[catch]); self.vy[catch] = np.minimum(sp[catch], 1.6) * np.sin(ang[catch])
            self.bx[catch] = px - side * 0.01
            if side == 1: hit |= catch
        lost = self.bx > SZ - 0.5; won = self.bx < -0.5
        r[lost] = -1; r[won] = 1
        if (lost | won).any(): self.serve(lost | won)
        return r, hit


def policy(model, x, rng):
    z = model.logits(x)[:, :NACT]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return (p.cumsum(1) > rng.random((len(x), 1))).argmax(1), p


class Learner:
    def __init__(self, lr=0.05, seed=3):
        self.rng = np.random.default_rng(seed); self.game = Pong(self.rng); self.lr = lr; self.seed = seed
        self.trainer = M.TrainerNPU(NMB)
        self.model = M.Model.init(seed, ncls=NACT)
        self.traj = [[] for _ in range(NG)]                            # per game: (frame, action) since the last point
        self.buf_x, self.buf_a, self.buf_r = [], [], []
        self.updates = 0; self.points = []; self.hits = 0; self.ms = []; self.first = True

    def play_until_batch(self):
        while sum(len(a) for a in self.buf_a) < BATCH:
            x = self.game.frame(); a, _ = policy(self.model, x, self.rng)
            r, hit = self.game.step(a); self.hits += int(hit.sum())
            for g in range(NG):
                self.traj[g].append((x[g], a[g]))
                ev = r[g] if r[g] else (HIT_REWARD if hit[g] else 0.0)
                if ev:                                                  # a point or a return closes the credit segment
                    T = len(self.traj[g]); ret = ev * GAMMA ** np.arange(T - 1, -1, -1)
                    self.buf_x.append(np.stack([t_[0] for t_ in self.traj[g]])); self.buf_a.append(np.array([t_[1] for t_ in self.traj[g]]))
                    self.buf_r.append(ret); self.traj[g] = []
                if r[g]: self.points.append(r[g] > 0)

    def update(self):
        self.play_until_batch()
        X = np.concatenate(self.buf_x); A = np.concatenate(self.buf_a); R = np.concatenate(self.buf_r)
        X, rest_x = X[:BATCH], X[BATCH:]; A, rest_a = A[:BATCH], A[BATCH:]; R, rest_r = R[:BATCH], R[BATCH:]
        self.buf_x, self.buf_a, self.buf_r = ([rest_x], [rest_a], [rest_r]) if len(rest_a) else ([], [], [])
        adv = (R - R.mean()) / (R.std() + 1e-6)
        perm = self.rng.permutation(BATCH)
        lr = self.lr / (1 + self.updates / LR_HALF)
        model, st = self.trainer.run(X, A.astype(np.uint8), perm, init=self.first, seed=self.seed, lr=lr, adv=adv, ncls=NACT)
        if not st[2] and not self.first: sys.exit("the NPU lost its weights between calls")
        self.first = False; self.model = model; self.updates += 1; self.ms.append(self.trainer.ms)

    def win_rate(self, n=500):
        p = self.points[-n:]
        return float(np.mean(p)) if p else 0.0


def train(n, lr, seed):
    L = Learner(lr, seed); t0 = time.perf_counter()
    for i in range(n):
        L.update()
        if (i + 1) % 20 == 0:
            print(f"update {i + 1}: points won {L.win_rate() * 100:5.1f}% (last 500 points), {len(L.points)} points so far, "
                  f"hits per point {L.hits / max(1, len(L.points)):.2f}, NPU {np.mean(L.ms[-20:]):.1f} ms per update, {time.perf_counter() - t0:.0f} s")
    return L


def window(lr, seed, seconds=None, shot=None):
    import pygame
    L = Learner(lr, seed); pygame.init(); W, H = 1100, 620
    screen = pygame.display.set_mode((W, H)); pygame.display.set_caption("The NPU learns Pong")
    font = pygame.font.SysFont(None, 26); big = pygame.font.SysFont(None, 56)
    t0 = time.perf_counter(); curve = []
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
        L.update(); curve.append((L.updates, L.win_rate(300)))
        screen.fill((14, 14, 18))
        f = L.game.frame().reshape(NG, SZ, SZ)
        for i in range(4):
            img = np.repeat(f[i][:, :, None], 3, 2); img[:, :, 0] = np.maximum(img[:, :, 0], 0)
            surf = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2))), (252, 252))
            x0, y0 = 24 + (i % 2) * 268, 24 + (i // 2) * 268
            screen.blit(surf, (x0, y0)); pygame.draw.rect(screen, (60, 60, 70), (x0 - 1, y0 - 1, 254, 254), 1)
        gx, gy, gw, gh = 600, 90, 460, 260
        pygame.draw.rect(screen, (60, 60, 70), (gx, gy, gw, gh), 1)
        for v in (0.25, 0.5, 0.75):
            yy = gy + gh - v * gh; pygame.draw.line(screen, (40, 40, 50), (gx, yy), (gx + gw, yy)); screen.blit(font.render(f"{v:.0%}", True, (120, 120, 130)), (gx - 46, yy - 9))
        if len(curve) > 1:
            n = max(curve[-1][0], 50)
            pygame.draw.lines(screen, (120, 220, 140), False, [(gx + u / n * gw, gy + gh - w_ * gh) for u, w_ in curve], 2)
        screen.blit(big.render(f"{L.win_rate(300) * 100:.0f}% of points won", True, (235, 235, 235)), (gx, 24))
        lines = [f"{L.updates} updates on the NPU, {len(L.points)} points played, {time.perf_counter() - t0:.0f} s",
                 f"hits per point {L.hits / max(1, len(L.points)):.2f}   NPU {np.mean(L.ms[-20:]):.1f} ms per update of {BATCH} moves",
                 "right paddle: the network (784-256-3 on 32 cores); left: a script, a bit too slow",
                 "input: 28x28 pixels, previous frame at half brightness; REINFORCE, 64 games in parallel"]
        for k, s_ in enumerate(lines): screen.blit(font.render(s_, True, (190, 190, 200)), (gx - 40, gy + gh + 24 + 28 * k))
        pygame.display.flip()
        if seconds and time.perf_counter() - t0 > seconds:
            if shot: pygame.image.save(screen, shot)
            print(f"{L.updates} updates, {len(L.points)} points, won {L.win_rate(300) * 100:.1f}% of the last 300"); return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=0); ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--window", action="store_true"); ap.add_argument("--seconds", type=float); ap.add_argument("--shot")
    a = ap.parse_args()
    if a.train: train(a.train, a.lr, a.seed)
    if a.window: window(a.lr, a.seed, a.seconds, a.shot)
    if not (a.train or a.window): ap.print_help()


if __name__ == "__main__":
    main()
