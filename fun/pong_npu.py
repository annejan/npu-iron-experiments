#!/usr/bin/env python3
"""The NPU learns to play Pong from pixels: policy gradient (REINFORCE) with the MNIST trainer of mnist_npu.py.

  ./env.sh python fun/pong_npu.py --train 300           300 updates, prints the share of points won
  ./env.sh python fun/pong_npu.py --window              watch four of the 64 games while it learns
  ./env.sh python fun/pong_npu.py --train 1000 --save pong.npz --gif pong.gif   keep the learned weights
  ./env.sh python fun/pong_npu.py --play pong.npz        you (left paddle, up/down or W/S) against what it learned

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

SZ = 28; PADH = 5; NG = 128; NMB = 47; BATCH = NMB * M.B; NACT = 3; GAMMA = 0.97; TMAX = 512
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

    def frame(self, opponent=False):
        """uint8 (NG, 784): current objects at 255, the previous frame's at 110. The opponent's paddle is left out
        unless asked for: a policy that sees it learns to copy it (the script follows the ball) instead of the ball."""
        f = np.zeros((NG, SZ, SZ), np.uint8); g = np.arange(NG)
        for (bx, by, py, oy), val in ((self.prev, 110), (self.objects(), 255)):
            for dy in range(-(PADH // 2), PADH // 2 + 1):
                f[g, np.clip(py + dy, 0, SZ - 1), SZ - 2] = np.maximum(f[g, np.clip(py + dy, 0, SZ - 1), SZ - 2], val)
                if opponent: f[g, np.clip(oy + dy, 0, SZ - 1), 1] = np.maximum(f[g, np.clip(oy + dy, 0, SZ - 1), 1], val)
            f[g, np.clip(by, 0, SZ - 1), np.clip(bx, 0, SZ - 1)] = np.maximum(f[g, np.clip(by, 0, SZ - 1), np.clip(bx, 0, SZ - 1)], val)
        return f.reshape(NG, SZ * SZ)

    def mirrored(self):
        """The left player's view: the frame flipped left-right, so it also plays from the right."""
        f = self.frame(opponent=True).reshape(NG, SZ, SZ)[:, :, ::-1].copy()
        f[:, :, 1] = 0                                                  # the right player's paddle, now on the left: not shown
        return f.reshape(NG, SZ * SZ)

    def step(self, act, opp_act=None):
        """act in {0 up, 1 stay, 2 down}. Returns reward (+1 point won, -1 lost, 0) and whether the player hit the ball.
        opp_act moves the left paddle instead of the script (self-play)."""
        self.prev = self.objects()
        self.py = np.clip(self.py + (act - 1) * 1.0, PADH // 2, SZ - 1 - PADH // 2)
        if opp_act is None: self.oy = self.oy + np.clip(self.by - self.oy, -self.opp_speed, self.opp_speed)
        else: self.oy = self.oy + (np.asarray(opp_act) - 1) * 1.0
        self.oy = np.clip(self.oy, PADH // 2, SZ - 1 - PADH // 2)
        self.bx += self.vx; self.by += self.vy
        top = self.by < 0; bot = self.by > SZ - 1
        self.by[top] = -self.by[top]; self.by[bot] = 2 * (SZ - 1) - self.by[bot]; self.vy[top | bot] *= -1
        r = np.zeros(NG); hit = np.zeros(NG, bool); self.hit_left = np.zeros(NG, bool)
        for side, px, pad in ((1, SZ - 2.5, self.py), (-1, 1.5, self.oy)):
            cross = (self.vx * side > 0) & ((self.bx - px) * side >= 0)
            catch = cross & (np.abs(self.by - pad) <= PADH / 2 + 0.6)
            off = (self.by - pad) / (PADH / 2 + 0.6)
            sp = np.hypot(self.vx, self.vy) * 1.02
            ang = np.clip(off * 0.9, -1.0, 1.0)
            self.vx[catch] = -side * np.minimum(sp[catch], 1.6) * np.cos(ang[catch]); self.vy[catch] = np.minimum(sp[catch], 1.6) * np.sin(ang[catch])
            self.bx[catch] = px - side * 0.01
            if side == 1: hit |= catch
            else: self.hit_left |= catch
        lost = self.bx > SZ - 0.5; won = self.bx < -0.5
        r[lost] = -1; r[won] = 1
        if (lost | won).any(): self.serve(lost | won)
        return r, hit


def policy(model, x, rng):
    z = model.logits(x)[:, :NACT]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return (p.cumsum(1) > rng.random((len(x), 1))).argmax(1), p


class Learner:
    def __init__(self, lr=0.05, seed=3, selfplay=False, lr_half=LR_HALF):
        self.rng = np.random.default_rng(seed); self.game = Pong(self.rng); self.lr = lr; self.seed = seed; self.selfplay = selfplay
        self.lr_half = lr_half
        self.trainer = M.TrainerNPU(NMB)
        self.model = M.Model.init(seed, ncls=NACT)
        ns = 2 * NG if selfplay else NG                                # self-play: both sides of every game learn
        self.tx = np.zeros((ns, TMAX, SZ * SZ), np.uint8); self.ta = np.zeros((ns, TMAX), np.uint8)   # moves since the last event
        self.tlen = np.zeros(ns, int)
        self.buf_x, self.buf_a, self.buf_r = [], [], []
        self.updates = 0; self.points = []; self.hits = 0; self.ms = []; self.first = True

    def play_until_batch(self):
        have = sum(len(a) for a in self.buf_a); g_all = np.arange(len(self.tlen))
        while have < BATCH:
            x = self.game.frame(); a, _ = policy(self.model, x, self.rng)
            if self.selfplay:
                xl = self.game.mirrored(); al, _ = policy(self.model, xl, self.rng)
                r, hit = self.game.step(a, al); self.hits += int(hit.sum())
                x = np.concatenate([x, xl]); a = np.concatenate([a, al])
                hit = np.concatenate([hit, self.game.hit_left]); r_side = np.concatenate([r, -r])
            else:
                r, hit = self.game.step(a); self.hits += int(hit.sum()); r_side = r
            full = self.tlen >= TMAX                                     # (never in practice) keep the newest moves
            if full.any(): self.tx[full, :-1] = self.tx[full, 1:]; self.ta[full, :-1] = self.ta[full, 1:]; self.tlen[full] = TMAX - 1
            self.tx[g_all, self.tlen] = x; self.ta[g_all, self.tlen] = a; self.tlen += 1
            ev = np.where(r_side != 0, r_side, np.where(hit, HIT_REWARD, 0.0))
            for g in np.nonzero(ev)[0]:                                 # a point or a return closes the credit segment
                T = self.tlen[g]
                self.buf_x.append(self.tx[g, :T].copy()); self.buf_a.append(self.ta[g, :T].copy())
                self.buf_r.append(ev[g] * GAMMA ** np.arange(T - 1, -1, -1)); self.tlen[g] = 0; have += T
            self.points += list(r[r != 0] > 0)

    def update(self):
        self.play_until_batch()
        X = np.concatenate(self.buf_x); A = np.concatenate(self.buf_a); R = np.concatenate(self.buf_r)
        X, rest_x = X[:BATCH], X[BATCH:]; A, rest_a = A[:BATCH], A[BATCH:]; R, rest_r = R[:BATCH], R[BATCH:]
        self.buf_x, self.buf_a, self.buf_r = ([rest_x], [rest_a], [rest_r]) if len(rest_a) else ([], [], [])
        adv = (R - R.mean()) / (R.std() + 1e-6)
        perm = self.rng.permutation(BATCH)
        lr = self.lr / (1 + self.updates / self.lr_half)
        model, st = self.trainer.run(X, A.astype(np.uint8), perm, init=self.first, seed=self.seed, lr=lr, adv=adv, ncls=NACT)
        if not st[2] and not self.first: sys.exit("the NPU lost its weights between calls")
        self.first = False; self.model = model; self.updates += 1; self.ms.append(self.trainer.ms)

    def eval_vs_script(self, steps=1500, seed=99, frozen=False):
        """Points won by the current policy against the scripted paddle (or one frozen in the middle), in fresh games;
        also the returns per point."""
        rng = np.random.default_rng(seed); g = Pong(rng); won = lost = hits = 0
        for _ in range(steps):
            a, _ = policy(self.model, g.frame(), rng); r, h = g.step(a, np.ones(NG, int) if frozen else None)
            won += int((r > 0).sum()); lost += int((r < 0).sum()); hits += int(h.sum())
        return won / max(1, won + lost), hits / max(1, won + lost)

    def win_rate(self, n=500):
        p = self.points[-n:]
        return float(np.mean(p)) if p else 0.0


def train(n, lr, seed, selfplay=False, lr_half=LR_HALF):
    L = Learner(lr, seed, selfplay, lr_half); t0 = time.perf_counter()
    for i in range(n):
        L.update()
        if (i + 1) % 20 == 0:
            vs = ""
            if (i + 1) % 200 == 0:
                w, h = L.eval_vs_script(frozen=True); vs = f"| frozen opponent: returns per point {h:.2f}"
            print(f"update {i + 1}: points won {L.win_rate() * 100:5.1f}% (last 500 points), {len(L.points)} points so far, "
                  f"hits per point {L.hits / max(1, len(L.points)):.2f}, NPU {np.mean(L.ms[-20:]):.1f} ms per update, {time.perf_counter() - t0:.0f} s {vs}")
    return L


def record_gif(L, path, steps=160, games=(0, 1, 2, 3), scale=8, selfplay=False):
    """Four of the games played by the current policy, drawn at 8x with a ball trail, as an animated GIF."""
    from PIL import Image, ImageDraw, ImageFont
    S = SZ * scale; pad = 6; head = 44
    font = ImageFont.load_default(size=18); bigf = ImageFont.load_default(size=22)
    trails = {g: [] for g in games}; score = {g: [0, 0] for g in games}; frames = []
    won_total = [0, 0]
    for step in range(steps):
        x = L.game.frame(); a, _ = policy(L.model, x, L.rng)
        oa = policy(L.model, L.game.mirrored(), L.rng)[0] if selfplay else None
        for g in games: trails[g] = (trails[g] + [(L.game.bx[g], L.game.by[g])])[-7:]
        r, hit = L.game.step(a, oa)
        for g in games:
            if r[g] > 0: score[g][0] += 1
            if r[g] < 0: score[g][1] += 1
            if r[g]: trails[g] = []
        won_total[0] += int((r > 0).sum()); won_total[1] += int((r < 0).sum())
        im = Image.new("RGB", (2 * S + 3 * pad, 2 * S + 3 * pad + head), (10, 10, 16)); d = ImageDraw.Draw(im)
        d.text((pad, 12), "the learned policy against itself" if selfplay else f"NPU (orange) after {L.updates} updates", fill=(235, 235, 235), font=bigf)
        for k, g in enumerate(games):
            ox, oy = pad + (k % 2) * (S + pad), head + pad + (k // 2) * (S + pad)
            d.rectangle([ox, oy, ox + S - 1, oy + S - 1], fill=(18, 20, 34))
            for yy in range(0, S, 2 * scale): d.rectangle([ox + S // 2 - 1, oy + yy, ox + S // 2, oy + yy + scale - 1], fill=(40, 44, 66))
            for t, (bx, by) in enumerate(trails[g]):
                rad = scale * (0.25 + 0.06 * t); c = int(40 + 25 * t)
                d.ellipse([ox + (bx + 0.5) * scale - rad, oy + (by + 0.5) * scale - rad, ox + (bx + 0.5) * scale + rad, oy + (by + 0.5) * scale + rad], fill=(c, c, c + 20))
            for px, py, col in ((SZ - 2, L.game.py[g], (255, 150, 40)), (1, L.game.oy[g], (70, 150, 255))):
                x0 = ox + px * scale; y0 = oy + (py - PADH / 2 + 0.5) * scale
                d.rounded_rectangle([x0 - 2, y0 - 2, x0 + scale + 1, y0 + PADH * scale + 1], radius=4, fill=tuple(v // 3 for v in col))
                d.rounded_rectangle([x0, y0, x0 + scale - 1, y0 + PADH * scale - 1], radius=3, fill=col)
            bx, by = L.game.bx[g], L.game.by[g]
            d.ellipse([ox + bx * scale + 1, oy + by * scale + 1, ox + (bx + 1) * scale - 1, oy + (by + 1) * scale - 1], fill=(250, 250, 250))
            d.text((ox + S // 2 - 40, oy + 6), f"{score[g][1]}", fill=(70, 150, 255), font=font)
            d.text((ox + S // 2 + 28, oy + 6), f"{score[g][0]}", fill=(255, 150, 40), font=font)
        frames.append(im.quantize(colors=48, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=45, loop=0, optimize=True)
    return won_total


def save(model, path):
    np.savez(path, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)


def load(path):
    d = np.load(path); return M.Model(d["W1"], d["b1"], d["W2"], d["b2"])


def play(path, seconds=None):
    """You against the learned policy, one big game; the policy runs on the CPU with the weights the NPU learned."""
    import pygame
    model = load(path); rng = np.random.default_rng()
    g = Pong(rng, opp_speed=0.0); scale = 22; S = SZ * scale
    pygame.init(); screen = pygame.display.set_mode((S, S + 50)); pygame.display.set_caption("You vs the NPU's Pong")
    font = pygame.font.SysFont(None, 40); clock = pygame.time.Clock(); score = [0, 0]; t0 = time.perf_counter(); trail = []
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): return
        k = pygame.key.get_pressed()
        move = (1.0 if (k[pygame.K_DOWN] or k[pygame.K_s]) else 0.0) - (1.0 if (k[pygame.K_UP] or k[pygame.K_w]) else 0.0)
        a, _ = policy(model, g.frame(), rng)
        trail = (trail + [(g.bx[0], g.by[0])])[-8:]
        g.oy = np.clip(g.oy + move * 1.0, PADH // 2, SZ - 1 - PADH // 2) - np.clip(g.by - g.oy, -0.0, 0.0)
        r, _ = g.step(a)
        if r[0] > 0: score[0] += 1; trail = []                          # the NPU won the point
        if r[0] < 0: score[1] += 1; trail = []
        screen.fill((10, 10, 16)); pygame.draw.rect(screen, (18, 20, 34), (0, 50, S, S))
        for yy in range(0, S, 2 * scale): pygame.draw.rect(screen, (40, 44, 66), (S // 2 - 2, 50 + yy, 4, scale))
        for t, (bx, by) in enumerate(trail):
            c = 40 + 22 * t; pygame.draw.circle(screen, (c, c, c + 20), (int((bx + 0.5) * scale), int(50 + (by + 0.5) * scale)), int(scale * (0.2 + 0.04 * t)))
        for px, py, col in ((SZ - 2, g.py[0], (255, 150, 40)), (1, g.oy[0], (70, 150, 255))):
            pygame.draw.rect(screen, col, (px * scale, 50 + (py - PADH / 2 + 0.5) * scale, scale, PADH * scale), border_radius=6)
        pygame.draw.circle(screen, (250, 250, 250), (int((g.bx[0] + 0.5) * scale), int(50 + (g.by[0] + 0.5) * scale)), scale // 2)
        screen.blit(font.render(f"you {score[1]}   NPU {score[0]}", True, (235, 235, 235)), (16, 12))
        pygame.display.flip(); clock.tick(30)
        if seconds and time.perf_counter() - t0 > seconds: return


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
    ap.add_argument("--gif", help="after --train, record four games of the trained policy to this GIF")
    ap.add_argument("--save", help="after --train, save the learned weights (npz)")
    ap.add_argument("--selfplay", action="store_true", help="train against itself (both paddles learn)")
    ap.add_argument("--lr-half", type=float, default=LR_HALF, help="learning rate lr / (1 + updates / this)")
    ap.add_argument("--play", help="play against saved weights (CPU only)")
    ap.add_argument("--selfplay-gif", nargs=2, metavar=("WEIGHTS", "GIF"), help="record the saved policy playing itself")
    a = ap.parse_args()
    if a.train:
        L = train(a.train, a.lr, a.seed, a.selfplay, a.lr_half)
        if a.save: save(L.model, a.save); print(f"saved {a.save}")
        if a.gif:
            w, l_ = record_gif(L, a.gif); print(f"wrote {a.gif}; while recording, all 64 games: {w} points won, {l_} lost")
    if a.window: window(a.lr, a.seed, a.seconds, a.shot)
    if a.play: play(a.play, a.seconds)
    if a.selfplay_gif:
        class _L: pass
        L = _L(); L.rng = np.random.default_rng(a.seed); L.game = Pong(L.rng); L.model = load(a.selfplay_gif[0]); L.updates = 0
        record_gif(L, a.selfplay_gif[1], steps=240, selfplay=True); print("wrote", a.selfplay_gif[1])
    if not (a.train or a.window or a.play or a.selfplay_gif): ap.print_help()


if __name__ == "__main__":
    main()
