# env_wrapper_3d.py
# Headless Space Shooter DDA environment for PPO training
# Simulates a 2D space shooter with enemies, asteroids, and player behavior
# The DDA agent observes telemetry and adjusts difficulty knobs

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List

# ---- Constants ----
WIDTH, HEIGHT = 1280, 720
PLAYER_SPEED = 350.0
LASER_SPEED_PLAYER = 600.0
LASER_SPEED_ENEMY = 350.0

# ---- Simulated Player ----
class SimPlayer:
    """Simulated human player with tuneable skill level."""
    def __init__(self, skill=0.6):
        self.skill = float(np.clip(skill, 0.0, 1.0))
        # Higher skill = better dodge, faster reactions, better aim
        self.dodge_ability = 0.3 + 0.65 * self.skill    # 0.3 to 0.95
        self.reaction_ms = 600 - 400 * self.skill         # 200–600ms
        self.aim_spread = 30 * (1 - self.skill)            # 0–30px spread

    def decide_move(self, px, py, enemies, enemy_lasers, dt):
        """Returns (dx, dy) movement direction."""
        dx, dy = 0.0, 0.0

        # DODGE: avoid nearest enemy laser
        nearest_threat = None
        nearest_dist = float('inf')
        for lx, ly in enemy_lasers:
            dist = math.sqrt((lx - px) ** 2 + (ly - py) ** 2)
            if dist < nearest_dist and ly < py:  # only threats coming toward us
                nearest_dist = dist
                nearest_threat = (lx, ly)

        if nearest_threat and nearest_dist < 150:
            # Dodge sideways
            lx, ly = nearest_threat
            if random.random() < self.dodge_ability:
                dx = 1.0 if lx < px else -1.0
                dy = -0.3  # slight upward

        # APPROACH: if no immediate threat, move toward a good firing position
        if not nearest_threat or nearest_dist > 200:
            if enemies:
                # Aim at nearest enemy's X position
                nearest_e = min(enemies, key=lambda e: e['y'])
                target_x = nearest_e['x']
                if abs(target_x - px) > self.aim_spread:
                    dx = 1.0 if target_x > px else -1.0

        # Add small noise
        dx += np.random.normal(0, 0.1 * (1 - self.skill))
        dy += np.random.normal(0, 0.1 * (1 - self.skill))

        length = math.sqrt(dx * dx + dy * dy)
        if length > 1:
            dx /= length
            dy /= length

        return dx, dy

    def should_fire(self, px, py, enemies, dt):
        """Returns True if player should fire this frame."""
        if not enemies:
            return False
        nearest = min(enemies, key=lambda e: abs(e['x'] - px) + abs(e['y'] - py))
        # Fire if roughly aligned
        x_diff = abs(nearest['x'] - px)
        if x_diff < 30 + self.aim_spread:
            return random.random() < (0.5 + 0.4 * self.skill)
        return False


# ---- DDA Environment ----
class DDAEnv3D:
    """
    Headless space shooter DDA environment.
    Obs: 7 floats [aim_acc, kill_rate, dmg_rate, hp, shield, wave, enemies_alive]
    Actions: 0=hold, 1=harder+, 2=easier-, 3=harder++, 4=easier--
    """
    DEFAULT_ENEMY_COUNT = 3
    DEFAULT_ENEMY_SPEED = 80.0
    DEFAULT_ENEMY_HP = 40.0
    DEFAULT_ENEMY_FIRE_RATE = 2.0
    DEFAULT_ASTEROID_COUNT = 3

    def __init__(self, sim_player=None, step_ms=2000, episode_ms=60000, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.sim = sim_player if sim_player else SimPlayer(skill=0.6)
        self.step_ms = int(step_ms)
        self.episode_ms = int(episode_ms)
        self._reset_state()

    def _reset_state(self):
        self.px = WIDTH / 2
        self.py = HEIGHT - 100
        self.player_hp = 100.0
        self.player_shield = 50.0
        self.player_max_hp = 100.0
        self.player_max_shield = 50.0

        self.shots_fired = 0
        self.shots_hit = 0
        self.kills = 0
        self.damage_taken = 0.0
        self.fire_cd = 0.0

        self.enemies = []
        self.player_lasers = []
        self.enemy_lasers = []
        self.asteroids = []

        self.now_ms = 0
        self.episode_time = 0
        self.wave = 0

        self.target_enemy_count = self.DEFAULT_ENEMY_COUNT
        self.target_enemy_speed = self.DEFAULT_ENEMY_SPEED
        self.target_enemy_hp = self.DEFAULT_ENEMY_HP
        self.target_enemy_fire_rate = self.DEFAULT_ENEMY_FIRE_RATE
        self.target_asteroid_count = self.DEFAULT_ASTEROID_COUNT

        self._spawn_wave()

    def _spawn_wave(self):
        self.wave += 1
        count = int(np.clip(self.target_enemy_count, 1, 12))
        for _ in range(count):
            self.enemies.append({
                'x': random.uniform(40, WIDTH - 40),
                'y': random.uniform(-300, -40),
                'hp': float(self.target_enemy_hp),
                'max_hp': float(self.target_enemy_hp),
                'speed': float(self.target_enemy_speed) * random.uniform(0.7, 1.3),
                'fire_rate': float(self.target_enemy_fire_rate),
                'fire_cd': random.uniform(0.5, self.target_enemy_fire_rate),
                'size': 18,
            })
        for _ in range(int(np.clip(self.target_asteroid_count, 0, 10))):
            self.asteroids.append({
                'x': random.uniform(30, WIDTH - 30),
                'y': random.uniform(-500, -50),
                'radius': random.uniform(8, 25),
                'speed': random.uniform(40, 100),
            })

    def _step_physics(self, dt_ms):
        dt = dt_ms / 1000.0

        # Shield regen
        if self.player_shield < self.player_max_shield:
            self.player_shield = min(self.player_shield + 3.0 * dt, self.player_max_shield)

        # Sim player decision
        enemy_laser_pos = [(l['x'], l['y']) for l in self.enemy_lasers]
        dx, dy = self.sim.decide_move(self.px, self.py, self.enemies, enemy_laser_pos, dt)
        self.px += dx * PLAYER_SPEED * dt
        self.py += dy * PLAYER_SPEED * dt
        self.px = max(20, min(WIDTH - 20, self.px))
        self.py = max(20, min(HEIGHT - 20, self.py))

        # Sim player fire
        self.fire_cd -= dt
        if self.fire_cd <= 0 and self.sim.should_fire(self.px, self.py, self.enemies, dt):
            self.player_lasers.append({'x': self.px, 'y': self.py - 15})
            self.shots_fired += 1
            self.fire_cd = 0.12

        # Update player lasers
        new_pl = []
        for l in self.player_lasers:
            l['y'] -= LASER_SPEED_PLAYER * dt
            if l['y'] < -20:
                continue
            hit = False
            for e in self.enemies:
                if abs(l['x'] - e['x']) < e['size'] and abs(l['y'] - e['y']) < e['size']:
                    self.shots_hit += 1
                    e['hp'] -= 15
                    if e['hp'] <= 0:
                        self.kills += 1
                    hit = True
                    break
            if not hit:
                new_pl.append(l)
        self.player_lasers = new_pl
        self.enemies = [e for e in self.enemies if e['hp'] > 0]

        # Update enemies
        new_enemies = []
        for e in self.enemies:
            e['y'] += e['speed'] * dt
            e['fire_cd'] -= dt
            if e['fire_cd'] <= 0 and e['y'] > 0:
                self.enemy_lasers.append({'x': e['x'], 'y': e['y'] + e['size']})
                e['fire_cd'] = e['fire_rate'] + random.uniform(-0.3, 0.3)
            if e['y'] < HEIGHT + 40:
                new_enemies.append(e)
                # Collision with player
                if abs(e['x'] - self.px) < 25 and abs(e['y'] - self.py) < 25:
                    self._player_take_damage(25)
                    continue
        self.enemies = new_enemies

        # Update enemy lasers
        new_el = []
        for l in self.enemy_lasers:
            l['y'] += LASER_SPEED_ENEMY * dt
            if l['y'] > HEIGHT + 20:
                continue
            if abs(l['x'] - self.px) < 15 and abs(l['y'] - self.py) < 20:
                self._player_take_damage(10)
                continue
            new_el.append(l)
        self.enemy_lasers = new_el

        # Update asteroids
        new_ast = []
        for a in self.asteroids:
            a['y'] += a['speed'] * dt
            if a['y'] > HEIGHT + 50:
                continue
            if abs(a['x'] - self.px) < a['radius'] + 15 and abs(a['y'] - self.py) < a['radius'] + 15:
                self._player_take_damage(15)
                continue
            new_ast.append(a)
        self.asteroids = new_ast

        # Check wave clear
        if len(self.enemies) == 0:
            self._spawn_wave()

    def _player_take_damage(self, amount):
        if self.player_shield > 0:
            absorbed = min(self.player_shield, amount)
            self.player_shield -= absorbed
            amount -= absorbed
        self.player_hp -= amount
        self.damage_taken += amount

    def reset(self):
        self._reset_state()
        return self._obs()

    def step(self, action):
        # Apply DDA action
        if action == 1:
            self.target_enemy_count += 1; self.target_enemy_speed += 5.0; self.target_enemy_hp += 8.0
            self.target_enemy_fire_rate = max(0.5, self.target_enemy_fire_rate - 0.1)
        elif action == 2:
            self.target_enemy_count -= 1; self.target_enemy_speed -= 5.0; self.target_enemy_hp -= 8.0
            self.target_enemy_fire_rate = min(4.0, self.target_enemy_fire_rate + 0.1)
        elif action == 3:
            self.target_enemy_count += 2; self.target_enemy_speed += 12.0; self.target_enemy_hp += 20.0
            self.target_enemy_fire_rate = max(0.5, self.target_enemy_fire_rate - 0.3)
        elif action == 4:
            self.target_enemy_count -= 2; self.target_enemy_speed -= 12.0; self.target_enemy_hp -= 20.0
            self.target_enemy_fire_rate = min(4.0, self.target_enemy_fire_rate + 0.3)

        self.target_enemy_count = int(np.clip(self.target_enemy_count, 1, 12))
        self.target_enemy_speed = float(np.clip(self.target_enemy_speed, 30, 200))
        self.target_enemy_hp = float(np.clip(self.target_enemy_hp, 15, 120))
        self.target_enemy_fire_rate = float(np.clip(self.target_enemy_fire_rate, 0.5, 4.0))

        # Simulate
        slice_ms = 50
        slices = max(1, self.step_ms // slice_ms)
        for _ in range(slices):
            self.now_ms += slice_ms
            self.episode_time += slice_ms
            self._step_physics(slice_ms)
            if self.player_hp <= 0:
                break

        obs = self._obs()

        # Reward: keep player in flow state
        aim_acc = self.shots_hit / max(1, self.shots_fired)
        hp_pct = self.player_hp / self.player_max_hp

        # Target accuracy around 0.5-0.7 (flow zone)
        acc_reward = -abs(aim_acc - 0.6) * 8.0

        # Target health: not too high (too easy), not too low (too hard)
        if hp_pct > 0.85:
            health_reward = -3.0  # too easy
        elif hp_pct < 0.2:
            health_reward = -5.0  # too hard, about to die
        elif 0.4 <= hp_pct <= 0.75:
            health_reward = 2.0   # sweet spot
        else:
            health_reward = 0.0

        kill_reward = self.kills * 0.3
        survival = 1.0 if self.player_hp > 0 else -5.0

        reward = acc_reward + health_reward + kill_reward + survival

        done = (self.episode_time >= self.episode_ms) or (self.player_hp <= 0)

        info = {
            "wave": self.wave, "kills": self.kills,
            "shots_fired": self.shots_fired, "shots_hit": self.shots_hit,
            "aim_accuracy": aim_acc, "health": self.player_hp,
            "shield": self.player_shield, "damage_taken": self.damage_taken,
            "enemy_count": self.target_enemy_count,
            "enemy_speed": self.target_enemy_speed,
            "enemy_hp": self.target_enemy_hp,
            "enemy_fire_rate": self.target_enemy_fire_rate,
        }
        return obs, reward, done, info

    def _obs(self):
        elapsed_s = max(1.0, self.episode_time / 1000.0)
        aim_acc = self.shots_hit / max(1, self.shots_fired)
        kill_rate = self.kills / elapsed_s
        dmg_rate = self.damage_taken / elapsed_s

        return np.array([
            aim_acc,
            min(kill_rate / 2.0, 1.0),
            min(dmg_rate / 30.0, 1.0),
            self.player_hp / self.player_max_hp,
            self.player_shield / self.player_max_shield,
            min(self.wave / 10.0, 1.0),
            min(len(self.enemies) / 15.0, 1.0),
        ], dtype=np.float32)


# ---- Quick self-test ----
if __name__ == "__main__":
    env = DDAEnv3D(SimPlayer(skill=0.6), step_ms=2000, episode_ms=30000, seed=42)
    obs = env.reset()
    print("reset obs:", obs)
    for t in range(15):
        a = random.randint(0, 4)
        obs, r, done, info = env.step(a)
        print(f"step {t:02d} action={a} reward={r:.2f} kills={info['kills']} "
              f"acc={info['aim_accuracy']:.2f} hp={info['health']:.0f} "
              f"wave={info['wave']} enemies={info['enemy_count']} "
              f"e_spd={info['enemy_speed']:.0f} e_hp={info['enemy_hp']:.0f}")
        if done:
            print("Episode done!")
            break
