# env_wrapper.py
# Minimal DDA environment wrapper (agent controls difficulty)
# No external RL libs required for the env itself.
# Put this file next to your working game.py

import random
import numpy as np
from collections import deque
from dataclasses import dataclass

# Keep constants in sync with game.py
WIDTH, HEIGHT = 480, 640
PLAYER_SIZE = 40
PLAYER_SPEED_PPS = 240.0
BLOCK_WIDTH = 40
BLOCK_HEIGHT = 40

TARGET_FALL_SPEED_INIT = 240.0
TARGET_SPAWN_INTERVAL_INIT = 900.0

@dataclass
class Player:
    x: float
    y: float
    w: int
    h: int

@dataclass
class Block:
    x: float
    y: float
    w: int
    h: int
    spawn_time_ms: int

def rects_collide(a, b):
    return (a.x < b.x + b.w and a.x + a.w > b.x and
            a.y < b.y + b.h and a.y + a.h > b.y)

# Simple simulated human for testing/training with controllable skill
class SimpleSimPlayer:
    def __init__(self, skill=0.6):
        self.skill = float(np.clip(skill, 0.0, 1.0))
        self.base_reaction_ms = 600 - 500 * self.skill
        self.noise = 0.35 * (1.0 - self.skill)

    def choose(self, player, blocks, now_ms):
        if not blocks:
            return 0.0
        # nearest falling block
        b = min(blocks, key=lambda b: b.y)
        block_center = b.x + b.w / 2.0
        player_center = player.x + player.w / 2.0
        # desired direction: move away from block center
        desired = 1.0 if block_center < player_center else -1.0
        # reaction time gating
        if now_ms is not None and b.spawn_time_ms is not None:
            reaction_ms = self.base_reaction_ms + np.random.normal(scale=40*(1-self.skill))
            if (now_ms - b.spawn_time_ms) < reaction_ms:
                return 0.0
        # add small noise and clip
        desired = desired + np.random.normal(scale=self.noise)
        return float(np.clip(desired, -1.0, 1.0))

class DDAEnv:
    """
    Discrete DDA environment:
      - Actions: 0=no change, 1=slightly harder, 2=slightly easier, 3=much harder, 4=much easier
      - Observation: numpy vector of telemetry:
           [accuracy, mistakes_norm, target_speed_norm, target_spawn_norm]
      - step_ms: how many ms of gameplay to simulate per agent step
    """
    def __init__(self, sim_player=None, step_ms=1000, episode_ms=60000, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.sim = sim_player if sim_player else SimpleSimPlayer(skill=0.6)
        self.step_ms = int(step_ms)
        self.episode_ms = int(episode_ms)
        self._reset_state()

    def _reset_state(self):
        self.player = Player(x=(WIDTH-PLAYER_SIZE)/2, y=HEIGHT-PLAYER_SIZE-10, w=PLAYER_SIZE, h=PLAYER_SIZE)
        self.blocks = []
        self.now_ms = 0
        self.last_spawn_time = 0
        self.target_fall_speed = float(TARGET_FALL_SPEED_INIT)
        self.target_spawn_interval = float(TARGET_SPAWN_INTERVAL_INIT)
        self.current_fall_speed = float(self.target_fall_speed)
        self.current_spawn_interval = float(self.target_spawn_interval)
        self.blocks_spawned = 0
        self.blocks_avoided = 0
        self.blocks_collided = 0
        self.mistakes = 0
        self.reaction_times = deque(maxlen=200)
        self.episode_time = 0

    def reset(self):
        self._reset_state()
        return self._obs()

    def _spawn_block(self):
        bx = int(random.gauss(WIDTH/2, WIDTH/6))
        bx = max(0, min(WIDTH - BLOCK_WIDTH, bx))
        b = Block(x=bx, y=-BLOCK_HEIGHT, w=BLOCK_WIDTH, h=BLOCK_HEIGHT, spawn_time_ms=self.now_ms)
        self.blocks.append(b)
        self.blocks_spawned += 1

    def _step_physics(self, dt_ms):
        dt = dt_ms / 1000.0
        # ease current values toward targets
        alpha = 1.0 - pow(0.001, dt_ms / (0.6 * 1000.0 + 1e-9))
        if alpha < 0: alpha = 0.01
        if alpha > 1: alpha = 1.0
        self.current_fall_speed += (self.target_fall_speed - self.current_fall_speed) * alpha
        self.current_spawn_interval += (self.target_spawn_interval - self.current_spawn_interval) * alpha

        # spawn check
        if self.now_ms - self.last_spawn_time >= int(self.current_spawn_interval):
            self._spawn_block()
            self.last_spawn_time = self.now_ms

        # update blocks
        new_blocks = []
        for b in self.blocks:
            b.y += self.current_fall_speed * dt
            if b.y > HEIGHT:
                self.blocks_avoided += 1
            else:
                new_blocks.append(b)
        self.blocks = new_blocks

        # simulated player action and movement
        desired = self.sim.choose(self.player, self.blocks, self.now_ms)
        self.player.x += desired * PLAYER_SPEED_PPS * dt
        self.player.x = max(0, min(WIDTH - self.player.w, self.player.x))

        # collisions
        for b in list(self.blocks):
            if rects_collide(self.player, b):
                self.blocks_collided += 1
                self.mistakes += 1
                try:
                    self.blocks.remove(b)
                except ValueError:
                    pass

    def step(self, action):
        # map discrete action to difficulty delta
        if action == 0:
            ds, dpi = 0.0, 0.0
        elif action == 1:
            ds, dpi = +10.0, -50.0
        elif action == 2:
            ds, dpi = -10.0, +50.0
        elif action == 3:
            ds, dpi = +30.0, -150.0
        elif action == 4:
            ds, dpi = -30.0, +150.0
        else:
            ds, dpi = 0.0, 0.0

        # apply and clamp targets
        self.target_fall_speed = float(np.clip(self.target_fall_speed + ds, 50.0, 1500.0))
        self.target_spawn_interval = float(np.clip(self.target_spawn_interval + dpi, 150.0, 5000.0))

        # simulate step_ms in slices
        slice_ms = 50
        slices = max(1, int(self.step_ms / slice_ms))
        for _ in range(slices):
            self.now_ms += slice_ms
            self.episode_time += slice_ms
            self._step_physics(slice_ms)

        obs = self._obs()
        accuracy = (self.blocks_avoided / self.blocks_spawned) if self.blocks_spawned>0 else 1.0

        # reward: encourage accuracy near 0.8, penalize mistakes and large divergence from base speed
        reward = (accuracy * 10.0) - (self.mistakes * 2.0) - (abs(self.target_fall_speed - 240.0) / 100.0)

        done = self.episode_time >= self.episode_ms

        info = {
            "blocks_spawned": self.blocks_spawned,
            "blocks_avoided": self.blocks_avoided,
            "blocks_collided": self.blocks_collided,
            "mistakes": self.mistakes,
            "accuracy": accuracy,
            "avg_reaction_ms": (sum(self.reaction_times)/len(self.reaction_times)) if self.reaction_times else 0.0,
            "target_fall_speed": self.target_fall_speed,
            "target_spawn_interval": self.target_spawn_interval,
            "episode_time_ms": self.episode_time
        }
        return obs, reward, done, info

    def _obs(self):
        accuracy = (self.blocks_avoided / self.blocks_spawned) if self.blocks_spawned>0 else 1.0
        avg_react = (sum(self.reaction_times)/len(self.reaction_times)) if self.reaction_times else 0.0
        # return compact vector observation (float32)
        # [accuracy, mistakes_norm, speed_norm, spawn_norm]
        mistakes_norm = np.tanh(self.mistakes / 5.0)
        speed_norm = np.clip(self.target_fall_speed / 1500.0, 0.0, 1.0)
        spawn_norm = np.clip(self.target_spawn_interval / 5000.0, 0.0, 1.0)
        return np.array([accuracy, mistakes_norm, speed_norm, spawn_norm], dtype=np.float32)

# quick test when run directly
if __name__ == "__main__":
    env = DDAEnv(SimpleSimPlayer(skill=0.6), step_ms=1000, episode_ms=15000, seed=42)
    obs = env.reset()
    print("reset obs:", obs)
    for t in range(8):
        a = random.randint(0,4)
        obs, r, done, info = env.step(a)
        print(f"step {t:02d} action={a} reward={r:.3f} acc={info['accuracy']:.3f} "
              f"spawned={info['blocks_spawned']} avoided={info['blocks_avoided']} mistakes={info['mistakes']} "
              f"speed={info['target_fall_speed']:.1f} spawn_ms={info['target_spawn_interval']:.0f}")
        if done:
            break
