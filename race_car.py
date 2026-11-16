"""
RaceCar-v0: a self-driving car learns to race on tracks it has never seen.

    import race_car                                  # registers the env
    env = gymnasium.make("RaceCar-v0", render_mode="rgb_array")

Every reset() builds a brand-new random track (reset(seed=n) always gives track n).
The car only "sees" 9 laser distance readings, its speed and its steering angle,
so it cannot memorise a track: it has to learn how to drive.

SUCCESS (info["success"] == True): one full lap, never touching a wall, inside 25 s.
FAIL: touching a wall ("crashed") or running out of time ("timeout").
info also has "progress" (fraction of the lap done) and "lap_time" (seconds).

Actions (Discrete(9)):   0 brake+left   1 brake   2 brake+right
                         3 coast+left   4 coast   5 coast+right
                         6 gas+left     7 gas     8 gas+right
Observation (11 floats): 9 laser distances (left to right), speed, steering

The tyres have limited grip: take a corner too fast and the car slides wide
into the wall, so a good driver has to brake before corners.

Try it yourself:   python race_car.py            (arrow keys: LEFT/RIGHT steer, UP gas, DOWN brake)
Watch random play: python race_car.py --random
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---- car (metres, seconds) ----
DT = 0.05
V_MIN, V_MAX = 4.0, 28.0      # the car always rolls forward at least V_MIN
ACCEL, BRAKE, DRAG = 9.0, 16.0, 1.0
GRIP = 14.0                   # max sideways acceleration before the tyres slide
SKID_DRAG = 20.0              # sliding tyres scrub off speed
WHEELBASE = 2.6
MAX_STEER = 0.45              # radians
STEER_SPEED = 0.25            # how quickly the wheels follow the command (0..1)
CAR_HALF_W = 1.0

# ---- track ----
TRACK_HALF_W = 5.0
MIN_CORNER_RADIUS = 11.0
MAX_STEPS = 500               # 25 seconds

# ---- reward ----
PROGRESS_REWARD = 0.1         # per metre driven along the track
LAP_BONUS = 10.0
CRASH_PENALTY = 5.0

# ---- sensors ----
LIDAR_ANGLES = np.radians([90, 60, 35, 15, 0, -15, -35, -60, -90])   # left to right
LIDAR_RANGE = 50.0

# ---- rendering ----
W, HUD_H = 640, 80
TITLE_H = 24                  # light strip at the very top: room for a caption drawn in black
H = W + HUD_H
WORLD_HALF = 52.0
PX = W / (2 * WORLD_HALF)

OUTCOME_TEXT = {"lap": "LAP COMPLETE", "crashed": "CRASHED", "timeout": "OUT OF TIME"}


def make_track(rng):
    """Random closed loop. Returns centre-line points spaced ~1 m apart, shape (N, 2)."""
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for attempt in range(200):
        wiggle = 0.45 * (0.985 ** attempt)        # relax slowly if we keep failing
        radius = np.ones_like(theta)
        for k in range(2, 7):
            radius += rng.uniform(0, wiggle) / (k - 1) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
        pts = np.stack([radius * np.cos(theta) * rng.uniform(1.0, 1.35), radius * np.sin(theta)], axis=1)
        spin = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(spin), -np.sin(spin)], [np.sin(spin), np.cos(spin)]])
        pts = pts @ rot.T
        if rng.random() < 0.5:
            pts = pts[::-1]                        # half the tracks run clockwise
        pts = pts - (pts.max(0) + pts.min(0)) / 2
        pts = pts * (WORLD_HALF - TRACK_HALF_W - 3) / np.abs(pts).max()

        # resample so points are evenly spaced along the track
        closed = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        dist = np.concatenate([[0], np.cumsum(seg)])
        n = int(round(dist[-1]))
        s = np.linspace(0, dist[-1], n, endpoint=False)
        centre = np.stack([np.interp(s, dist, closed[:, 0]), np.interp(s, dist, closed[:, 1])], axis=1)

        tangent = np.roll(centre, -1, 0) - np.roll(centre, 1, 0)
        heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        turn = np.abs(np.roll(heading, -3) - np.roll(heading, 3))
        turn = np.minimum(turn, np.abs(turn - 2 * np.pi))
        tightest = (6 * dist[-1] / n) / max(turn.max(), 1e-9)
        if tightest >= MIN_CORNER_RADIUS and dist[-1] <= 290:
            return centre
    t = np.linspace(0, 2 * np.pi, 250, endpoint=False)   # practically unreachable fallback
    return 40 * np.stack([np.cos(t), np.sin(t)], axis=1)


class RaceCarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, show_tally=False):
        self.render_mode = render_mode
        self.show_tally = show_tally
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-1.0, 1.5, shape=(11,), dtype=np.float32)
        self.laps = 0
        self.attempts = 0
        self.outcome = None
        self._surface = self._window = self._clock = self._fonts = self._track_surf = None

    # ------------------------------------------------------------------ core
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.centre = c = make_track(r)
        self.n = len(c)
        self.length = float(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1).sum())
        self.ds = self.length / self.n
        t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
        self.tangent = t / np.linalg.norm(t, axis=1, keepdims=True)
        normal = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)      # points left
        self.walls = (c + normal * TRACK_HALF_W, c - normal * TRACK_HALF_W)
        self._seg_a = np.vstack(self.walls)
        self._seg_e = np.vstack([np.roll(w, -1, 0) - w for w in self.walls])

        self.pos = c[0] + normal[0] * r.uniform(-1.5, 1.5)
        self.heading = math.atan2(self.tangent[0, 1], self.tangent[0, 0]) + r.uniform(-0.1, 0.1)
        self.speed = r.uniform(V_MIN, 8.0)
        self.steer = 0.0                 # -1 (full right) .. +1 (full left)
        self.idx = 0
        self.along = 0.0
        self.lap_s = 0.0                 # metres driven along the track
        self.steps = 0
        self.outcome = None
        self.skidding = False
        self._trail = []
        self._track_surf = None
        self._lidar = self._scan()
        if self.render_mode == "human":
            self.render()
        return self._obs(), {}

    def _scan(self):
        ang = self.heading + LIDAR_ANGLES
        dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
        ao = self._seg_a - self.pos
        ex, ey = self._seg_e[:, 0][None], self._seg_e[:, 1][None]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = dx * ey - dy * ex
            t = (ao[:, 0] * self._seg_e[:, 1] - ao[:, 1] * self._seg_e[:, 0])[None] / denom
            u = (ao[:, 0][None] * dy - ao[:, 1][None] * dx) / denom
        t = np.where((t > 0) & (u >= 0) & (u <= 1), t, LIDAR_RANGE)
        return np.minimum(t.min(axis=1), LIDAR_RANGE)

    def _obs(self):
        return np.concatenate([self._lidar / LIDAR_RANGE, [self.speed / V_MAX, self.steer]]).astype(np.float32)

    def step(self, action):
        action = int(action)
        assert self.outcome is None, "episode is over, call reset()"
        throttle, steer_cmd = action // 3, (1.0, 0.0, -1.0)[action % 3]

        self.steer += (steer_cmd - self.steer) * STEER_SPEED
        wanted_turn = self.speed * math.tan(self.steer * MAX_STEER) / WHEELBASE
        max_turn = GRIP / self.speed
        self.skidding = abs(wanted_turn) > max_turn        # tyres can't hold it: slide wide
        self.heading += max(-max_turn, min(max_turn, wanted_turn)) * DT

        if throttle == 2:
            self.speed += ACCEL * (1 - self.speed / V_MAX) * DT
        else:
            self.speed -= (BRAKE if throttle == 0 else DRAG) * DT
        if self.skidding:
            self.speed -= SKID_DRAG * DT
        self.speed = min(V_MAX, max(V_MIN, self.speed))
        self.pos = self.pos + self.speed * DT * np.array([math.cos(self.heading), math.sin(self.heading)])
        self.steps += 1

        # where are we on the track? (only look near the last known spot)
        near = (self.idx + np.arange(-12, 13)) % self.n
        rel = self.pos - self.centre[near]
        best = int(np.argmin((rel ** 2).sum(axis=1)))
        new_idx, rel = int(near[best]), rel[best]
        tx, ty = self.tangent[new_idx]
        along = rel[0] * tx + rel[1] * ty
        off_centre = abs(rel[0] * ty - rel[1] * tx)
        moved = (best - 12) * self.ds + (along - self.along)
        self.idx, self.along = new_idx, along
        self.lap_s += moved

        reward = PROGRESS_REWARD * moved
        outcome, truncated = None, False
        if off_centre > TRACK_HALF_W - CAR_HALF_W:
            outcome, reward = "crashed", reward - CRASH_PENALTY
        elif self.lap_s >= self.length:
            outcome, reward = "lap", reward + LAP_BONUS
        elif self.steps >= MAX_STEPS:
            outcome, truncated = "timeout", True

        self._lidar = self._scan()
        if self.render_mode is not None:
            self._trail.append((self.pos[0], self.pos[1], self.speed, self.skidding))

        info = {}
        if outcome is not None:
            self.outcome = outcome
            self.attempts += 1
            self.laps += outcome == "lap"
            info = {"success": outcome == "lap", "outcome": outcome,
                    "progress": float(np.clip(self.lap_s / self.length, 0, 1)),
                    "lap_time": self.steps * DT}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), outcome is not None and not truncated, truncated, info

    # ------------------------------------------------------------- rendering
    @staticmethod
    def _px(p):
        return (W / 2 + p[0] * PX, HUD_H + W / 2 - p[1] * PX)

    def _car_px(self, forward, left):
        c, s = math.cos(self.heading), math.sin(self.heading)
        return self._px((self.pos[0] + forward * c - left * s, self.pos[1] + forward * s + left * c))

    @staticmethod
    def _speed_colour(v):
        t = (v - V_MIN) / (V_MAX - V_MIN)
        lo, mid, hi = (56, 189, 248), (250, 204, 21), (248, 113, 113)
        a, b, f = (lo, mid, t * 2) if t < 0.5 else (mid, hi, t * 2 - 1)
        return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))

    def _draw_track(self, pygame):
        surf = pygame.Surface((W, H))
        surf.fill((22, 64, 44))
        rng = np.random.default_rng(1)
        for _ in range(500):                                   # grass texture
            x, y = rng.integers(0, W), rng.integers(HUD_H, H)
            pygame.draw.line(surf, (27, 74, 51), (x, y), (x + 3, y - 4), 1)

        def area(poly):
            return abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
        outer, inner = sorted(self.walls, key=area, reverse=True)
        pygame.draw.polygon(surf, (52, 56, 64), [self._px(p) for p in outer])
        pygame.draw.polygon(surf, (22, 64, 44), [self._px(p) for p in inner])
        for i in range(0, self.n, 6):                          # dashed centre line
            pygame.draw.line(surf, (82, 88, 98), self._px(self.centre[i]),
                             self._px(self.centre[(i + 2) % self.n]), 1)
        for wall in self.walls:                                # red / white kerbs
            for j, i in enumerate(range(0, self.n, 3)):
                pygame.draw.line(surf, (226, 60, 60) if j % 2 else (240, 240, 240),
                                 self._px(wall[i]), self._px(wall[(i + 3) % self.n]), 4)
        c0, t0 = self.centre[0], self.tangent[0]               # chequered start / finish line
        n0 = np.array([-t0[1], t0[0]])
        cell = TRACK_HALF_W / 4
        for j in range(8):
            for row in range(2):
                a = c0 + n0 * (-TRACK_HALF_W + j * cell) + t0 * (row - 1) * cell
                quad = [a, a + n0 * cell, a + n0 * cell + t0 * cell, a + t0 * cell]
                pygame.draw.polygon(surf, (245, 245, 245) if (j + row) % 2 else (20, 20, 20),
                                    [self._px(q) for q in quad])
        return surf

    def render(self):
        if self.render_mode is None:
            return None
        import pygame

        if self._surface is None:
            pygame.font.init()
            self._surface = pygame.Surface((W, H))
            mono = pygame.font.match_font("menlo,dejavusansmono,consolas,couriernew,courier")
            self._fonts = (pygame.font.Font(mono, 13), pygame.font.Font(None, 60), pygame.font.Font(None, 26))
            if self.render_mode == "human":
                pygame.display.init()
                self._window = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Race Car")
                self._clock = pygame.time.Clock()
        if self._track_surf is None:
            self._track_surf = self._draw_track(pygame)

        s = self._surface
        small, big, medium = self._fonts
        if self._trail and self._trail[-1][3]:                 # leave skid marks on the tarmac
            for side in (-0.85, 0.85):
                pygame.draw.circle(self._track_surf, (26, 28, 32), self._car_px(-1.4, side), 2)
        s.blit(self._track_surf, (0, 0))

        for a, b in zip(self._trail, self._trail[1:]):         # driven line, coloured by speed
            pygame.draw.line(s, self._speed_colour(b[2]), self._px(a), self._px(b), 2)

        crashed = self.outcome == "crashed"
        if self.outcome is None:                               # lasers
            for ang, d in zip(LIDAR_ANGLES, self._lidar):
                end = self.pos + d * np.array([math.cos(self.heading + ang), math.sin(self.heading + ang)])
                close = max(0.0, 1 - d / 20)
                col = (int(90 + 165 * close), int(230 - 130 * close), int(120 - 60 * close))
                pygame.draw.line(s, col, self._px(self.pos), self._px(end), 1)
                pygame.draw.circle(s, col, self._px(end), 3)

        if crashed:
            cx, cy = self._car_px(1.5, 0)
            for i in range(12):
                a = i * math.pi / 6
                rad = 26 if i % 2 else 15
                pygame.draw.line(s, (255, 200, 60), (cx, cy), (cx + math.cos(a) * rad, cy + math.sin(a) * rad), 3)
        front = self.heading + self.steer * MAX_STEER
        for fwd, ang in ((1.3, front), (-1.3, self.heading)):  # wheels
            for side in (-1.0, 1.0):
                wx, wy = self._car_px(fwd, side)
                ox, oy = math.cos(ang) * 0.55 * PX, -math.sin(ang) * 0.55 * PX
                pygame.draw.line(s, (12, 12, 14), (wx - ox, wy - oy), (wx + ox, wy + oy), 4)
        body = [(2.3, 0.55), (2.3, -0.55), (1.3, -0.95), (-2.0, -0.95), (-2.0, 0.95), (1.3, 0.95)]
        pygame.draw.polygon(s, (120, 30, 30) if crashed else (239, 68, 68), [self._car_px(*p) for p in body])
        cockpit = [(0.7, 0.62), (0.7, -0.62), (-0.9, -0.62), (-0.9, 0.62)]
        pygame.draw.polygon(s, (30, 34, 44), [self._car_px(*p) for p in cockpit])

        # HUD strip
        pygame.draw.rect(s, (15, 18, 24), (0, 0, W, HUD_H))
        pygame.draw.rect(s, (226, 232, 240), (0, 0, W, TITLE_H))
        white, dim = (235, 238, 245), (150, 158, 172)
        tag = small.render("RaceCar-v0", True, (100, 116, 139))
        s.blit(tag, (W - tag.get_width() - 10, 4))
        top = TITLE_H + 6
        s.blit(small.render("SPEED", True, dim), (16, top))
        s.blit(medium.render(f"{self.speed * 3.6:3.0f} km/h", True, self._speed_colour(self.speed)), (16, top + 20))
        frac = float(np.clip(self.lap_s / self.length, 0, 1))
        s.blit(small.render(f"LAP PROGRESS  {frac * 100:3.0f}%", True, dim), (150, top))
        pygame.draw.rect(s, (48, 54, 66), (150, top + 24, 300, 14), border_radius=4)
        bar = (34, 197, 94) if self.outcome in (None, "lap") else (239, 68, 68)
        pygame.draw.rect(s, bar, (150, top + 24, int(300 * frac), 14), border_radius=4)
        s.blit(small.render("TIME", True, dim), (476, top))
        s.blit(medium.render(f"{self.steps * DT:4.1f} s", True, white), (476, top + 20))
        if self.show_tally:
            s.blit(small.render(f"LAPS {self.laps}/{self.attempts}", True, white), (556, top))

        if self.outcome is not None:
            ok = self.outcome == "lap"
            band = pygame.Surface((W, 96), pygame.SRCALPHA)
            band.fill(((34, 197, 94) if ok else (239, 68, 68)) + (220,))
            s.blit(band, (0, HUD_H + 250))
            txt = big.render(OUTCOME_TEXT[self.outcome], True, (255, 255, 255))
            s.blit(txt, ((W - txt.get_width()) // 2, HUD_H + 258))
            detail = f"SUCCESS   lap time {self.steps * DT:.1f} s" if ok else f"FAIL   {frac * 100:.0f}% of the lap"
            sub = medium.render(detail, True, (255, 255, 255))
            s.blit(sub, ((W - sub.get_width()) // 2, HUD_H + 310))

        if self.render_mode == "human":
            self._window.blit(s, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2)).copy()

    def close(self):
        if self._window is not None:
            import pygame
            pygame.display.quit()
            self._window = None


gym.register(id="RaceCar-v0", entry_point=RaceCarEnv, max_episode_steps=MAX_STEPS)


def evaluate(policy, episodes=100, seed=1_000_000):
    """policy(obs) -> action, driven on `episodes` fixed test tracks.
    Returns (fraction of laps completed, average lap time of the completed laps)."""
    env = RaceCarEnv()
    times = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        while True:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if terminated or truncated:
                if info["success"]:
                    times.append(info["lap_time"])
                break
    return len(times) / episodes, (float(np.mean(times)) if times else float("nan"))


if __name__ == "__main__":
    import sys
    import pygame

    random_play = "--random" in sys.argv
    env = RaceCarEnv(render_mode="human", show_tally=True)
    obs, _ = env.reset()
    while True:
        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            break
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break
        if random_play:
            action = env.action_space.sample()
        else:
            steer = 0 if keys[pygame.K_LEFT] else 2 if keys[pygame.K_RIGHT] else 1
            action = steer + (6 if keys[pygame.K_UP] else 0 if keys[pygame.K_DOWN] else 3)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            pygame.time.wait(1200)
            obs, _ = env.reset()
    env.close()
