# game_3d.py
# Space Dogfight — Polished Pygame Arcade Shooter with AI DDA
# Works perfectly on macOS (no OpenGL shaders needed)
# Controls: Arrow keys / WASD = Move, Space = Shoot, R = Restart, Q = Quit

import pygame
import sys
import math
import random
import time as pytime
import numpy as np
import torch
import torch.nn as nn
import requests
import uuid

# ──────────────────── INIT ────────────────────
pygame.init()
pygame.mixer.init()

WIDTH, HEIGHT = 1280, 720
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("⋆ SPACE DOGFIGHT — AI Dynamic Difficulty ⋆")
clock = pygame.time.Clock()

# ──────────────────── COLORS ────────────────────
BLACK       = (  0,   0,   0)
SPACE_BG    = (  5,   5,  20)
WHITE       = (255, 255, 255)
CYAN        = (  0, 230, 255)
CYAN_DIM    = (  0, 100, 140)
NEON_GREEN  = (  0, 255, 120)
NEON_RED    = (255,  40,  60)
ORANGE      = (255, 160,  30)
YELLOW      = (255, 220,   0)
PURPLE      = (180,  60, 255)
SHIELD_BLUE = ( 60, 160, 255)
DARK_GRAY   = ( 30,  30,  40)
MID_GRAY    = ( 80,  80, 100)
HP_GREEN    = (  0, 220,  80)
HP_YELLOW   = (255, 200,   0)
HP_RED      = (255,  50,  50)
GRID_COLOR  = ( 20,  40,  60)

# ──────────────────── FONTS ────────────────────
font_lg = pygame.font.SysFont("Menlo", 36, bold=True)
font_md = pygame.font.SysFont("Menlo", 22, bold=True)
font_sm = pygame.font.SysFont("Menlo", 16)
font_hud = pygame.font.SysFont("Menlo", 18, bold=True)

# ──────────────────── NEURAL NET ────────────────────
class MLPPolicy3D(nn.Module):
    def __init__(self, obs_dim=7, act_dim=5, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden // 2, act_dim)
        self.critic = nn.Linear(hidden // 2, 1)
    def forward(self, x):
        h = self.net(x)
        return self.actor(h), self.critic(h).squeeze(-1)

class StatefulLSTMPolicy3D(nn.Module):
    """LSTM-based policy with memory of past 30 seconds."""
    def __init__(self, obs_dim=7, act_dim=5, lstm_hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(obs_dim, lstm_hidden, num_layers=1, batch_first=False)
        self.net = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (batch_size, obs_dim)
            hidden_state: tuple (h, c) of shape (1, batch_size, lstm_hidden)
        Returns:
            logits: (batch_size, act_dim)
            value: (batch_size,)
            hidden_state: new (h, c)
        """
        if hidden_state is None:
            h_t = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
            c_t = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
            hidden_state = (h_t, c_t)

        x_seq = x.unsqueeze(0)  # (1, batch_size, obs_dim)
        lstm_out, (h_new, c_new) = self.lstm(x_seq, hidden_state)
        lstm_out = lstm_out.squeeze(0)  # (batch_size, lstm_hidden)

        h = self.net(lstm_out)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value, (h_new, c_new)

MODEL_PATH = "dda_ppo_3d.pth"
BACKEND_URL = "http://localhost:8000"

# Keep the live game closer to the training environment so the DDA policy
# does not spend the whole session trapped in "make it easier" mode.
DDA_DEFAULT_ENEMY_COUNT = 6
DDA_DEFAULT_ENEMY_SPEED = 100.0
DDA_DEFAULT_ENEMY_HP = 55.0
DDA_DEFAULT_ENEMY_FIRE_RATE = 1.8
DDA_DEFAULT_ASTEROID_COUNT = 6

DDA_MIN_ENEMY_COUNT = 2
DDA_MAX_ENEMY_COUNT = 12
DDA_MIN_ENEMY_SPEED = 60.0
DDA_MAX_ENEMY_SPEED = 180.0
DDA_MIN_ENEMY_HP = 30.0
DDA_MAX_ENEMY_HP = 120.0
DDA_MIN_ENEMY_FIRE_RATE = 0.6
DDA_MAX_ENEMY_FIRE_RATE = 3.5
DDA_MIN_ASTEROID_COUNT = 2
DDA_MAX_ASTEROID_COUNT = 10

# ──────────────────── HELPER: DRAW FUNCTIONS ────────────────────
def draw_gradient_rect(surface, rect, color_top, color_bot):
    """Draw a vertical gradient filled rectangle."""
    x, y, w, h = rect
    for i in range(h):
        ratio = i / max(h - 1, 1)
        r = int(color_top[0] + (color_bot[0] - color_top[0]) * ratio)
        g = int(color_top[1] + (color_bot[1] - color_top[1]) * ratio)
        b = int(color_top[2] + (color_bot[2] - color_top[2]) * ratio)
        pygame.draw.line(surface, (r, g, b), (x, y + i), (x + w, y + i))

def draw_ship(surface, cx, cy, size, col):
    """Draw a sleek triangular ship."""
    pts = [
        (cx, cy - size),               # nose
        (cx - size * 0.7, cy + size * 0.6),   # left wing
        (cx - size * 0.2, cy + size * 0.3),
        (cx + size * 0.2, cy + size * 0.3),
        (cx + size * 0.7, cy + size * 0.6),   # right wing
    ]
    pygame.draw.polygon(surface, col, pts)
    # Engine glow
    pygame.draw.circle(surface, ORANGE,
                       (int(cx), int(cy + size * 0.4)),
                       max(2, int(size * 0.15)))

def draw_enemy_ship(surface, cx, cy, size, col, variant=0):
    """Draw enemy ship variations."""
    if variant == 0:  # standard fighter
        pts = [
            (cx, cy + size),
            (cx - size * 0.8, cy - size * 0.5),
            (cx, cy - size * 0.1),
            (cx + size * 0.8, cy - size * 0.5),
        ]
        pygame.draw.polygon(surface, col, pts)
    elif variant == 1:  # heavy bomber
        pts = [
            (cx, cy + size * 0.8),
            (cx - size, cy - size * 0.3),
            (cx - size * 0.5, cy - size * 0.6),
            (cx + size * 0.5, cy - size * 0.6),
            (cx + size, cy - size * 0.3),
        ]
        pygame.draw.polygon(surface, col, pts)
    else:  # interceptor
        pts = [
            (cx, cy + size * 1.2),
            (cx - size * 0.4, cy),
            (cx - size * 0.6, cy - size * 0.4),
            (cx + size * 0.6, cy - size * 0.4),
            (cx + size * 0.4, cy),
        ]
        pygame.draw.polygon(surface, col, pts)
    # Red eye
    pygame.draw.circle(surface, (255, 0, 0), (int(cx), int(cy)), max(2, int(size * 0.15)))

def draw_asteroid(surface, cx, cy, radius):
    """Draw a rough asteroid shape."""
    num_pts = 8
    pts = []
    for i in range(num_pts):
        angle = (2 * math.pi / num_pts) * i
        r = radius * (0.7 + random.Random(int(cx * 100 + cy * 10 + i)).random() * 0.5)
        pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    pygame.draw.polygon(surface, (120, 80, 50), pts)
    pygame.draw.polygon(surface, (90, 60, 35), pts, 2)

# ──────────────────── PARTICLE SYSTEM ────────────────────
class Particle:
    def __init__(self, x, y, vx, vy, life, col, size=3):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = life
        self.max_life = life
        self.col = col
        self.size = size

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surface):
        alpha = max(0, self.life / self.max_life)
        r = int(self.col[0] * alpha)
        g = int(self.col[1] * alpha)
        b = int(self.col[2] * alpha)
        sz = max(1, int(self.size * alpha))
        pygame.draw.circle(surface, (r, g, b), (int(self.x), int(self.y)), sz)

particles = []

def spawn_explosion(x, y, count=20, col=ORANGE):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(50, 250)
        life = random.uniform(0.3, 0.8)
        size = random.uniform(2, 5)
        particles.append(Particle(
            x, y,
            math.cos(angle) * speed, math.sin(angle) * speed,
            life, col, size
        ))

def spawn_hit_sparks(x, y, count=8):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(30, 120)
        particles.append(Particle(
            x, y,
            math.cos(angle) * speed, math.sin(angle) * speed,
            random.uniform(0.1, 0.3), WHITE, 2
        ))

# ──────────────────── STAR BACKGROUND ────────────────────
class Star:
    def __init__(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.speed = random.uniform(20, 120)
        self.brightness = random.randint(80, 255)
        self.size = 1 if self.speed < 60 else 2

    def update(self, dt):
        self.y += self.speed * dt
        if self.y > HEIGHT:
            self.y = 0
            self.x = random.uniform(0, WIDTH)

    def draw(self, surface):
        col = (self.brightness, self.brightness, self.brightness)
        pygame.draw.circle(surface, col, (int(self.x), int(self.y)), self.size)

bg_stars = [Star() for _ in range(150)]

# ──────────────────── GAME ENTITIES ────────────────────
class Player:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT - 100
        self.size = 22
        self.speed = 350
        self.health = 100.0
        self.max_health = 100.0
        self.shield = 50.0
        self.max_shield = 50.0
        self.fire_cooldown = 0.0
        self.shots_fired = 0
        self.shots_hit = 0
        self.kills = 0
        self.asteroid_hits = 0
        self.damage_taken = 0.0
        self.invuln = 0.0  # invulnerability timer after hit

    def take_damage(self, amount):
        if self.invuln > 0:
            return
        if self.shield > 0:
            absorbed = min(self.shield, amount)
            self.shield -= absorbed
            amount -= absorbed
        self.health -= amount
        self.damage_taken += amount
        self.invuln = 0.3  # brief invulnerability
        spawn_hit_sparks(self.x, self.y, 12)

    def update(self, dt, keys):
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.x -= self.speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.x += self.speed * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.y -= self.speed * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.y += self.speed * dt

        # Clamp
        self.x = max(self.size, min(WIDTH - self.size, self.x))
        self.y = max(self.size, min(HEIGHT - self.size, self.y))

        # Shield regen
        if self.shield < self.max_shield:
            self.shield = min(self.shield + 3.0 * dt, self.max_shield)

        self.fire_cooldown -= dt
        self.invuln -= dt

    def draw(self, surface):
        # Flicker when invulnerable
        if self.invuln > 0 and int(self.invuln * 20) % 2 == 0:
            return
        draw_ship(surface, self.x, self.y, self.size, CYAN)
        # Shield aura
        if self.shield > 10:
            alpha = int(60 * (self.shield / self.max_shield))
            r = int(self.size * 1.5 + 5)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*SHIELD_BLUE, alpha), (r, r), r, 2)
            surface.blit(s, (self.x - r, self.y - r))

class EnemyBase:
    def __init__(self, x, y, hp, speed, fire_rate, variant=0):
        self.x = x
        self.y = y
        self.hp = hp
        self.max_hp = hp
        self.speed = speed
        self.fire_rate = fire_rate
        self.fire_cd = random.uniform(0.5, fire_rate)
        self.alive = True
        self.size = 18 + variant * 4
        self.variant = variant

    def update(self, dt, player):
        self.y += self.speed * dt
        self.fire_cd -= dt
        if self.y > HEIGHT + 40:
            self.alive = False

    def draw(self, surface):
        col = NEON_RED if self.variant == 0 else (PURPLE if self.variant == 2 else (255, 100, 30))
        draw_enemy_ship(surface, self.x, self.y, self.size, col, self.variant)
        # HP bar
        if self.hp < self.max_hp:
            bar_w = self.size * 2
            bar_h = 3
            bx = self.x - bar_w / 2
            by = self.y - self.size - 8
            ratio = max(0, self.hp / self.max_hp)
            pygame.draw.rect(surface, DARK_GRAY, (bx, by, bar_w, bar_h))
            pygame.draw.rect(surface, HP_RED, (bx, by, bar_w * ratio, bar_h))

    def take_hit(self, damage):
        self.hp -= damage
        spawn_hit_sparks(self.x, self.y, 6)
        if self.hp <= 0:
            self.alive = False

class Asteroid:
    def __init__(self, x, y, radius, speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.alive = True
        self.rot = random.uniform(0, 360)

    def update(self, dt):
        self.y += self.speed * dt
        self.rot += 30 * dt
        if self.y > HEIGHT + self.radius * 2:
            self.alive = False

    def draw(self, surface):
        draw_asteroid(surface, self.x, self.y, self.radius)

class Laser:
    def __init__(self, x, y, owner='player'):
        self.x = x
        self.y = y
        self.owner = owner
        self.speed = 600 if owner == 'player' else 350
        self.alive = True
        self.damage = 15.0 if owner == 'player' else 10.0

    def update(self, dt):
        if self.owner == 'player':
            self.y -= self.speed * dt
        else:
            self.y += self.speed * dt
        if self.y < -20 or self.y > HEIGHT + 20:
            self.alive = False

    def draw(self, surface):
        if self.owner == 'player':
            col = NEON_GREEN
            w, h = 3, 16
        else:
            col = NEON_RED
            w, h = 3, 12
        pygame.draw.rect(surface, col, (self.x - w//2, self.y - h//2, w, h))
        # Glow
        glow_s = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
        pygame.draw.rect(glow_s, (*col, 60), (0, 0, w + 6, h + 6))
        surface.blit(glow_s, (self.x - w//2 - 3, self.y - h//2 - 3))

# ──────────────────── GAME STATE ────────────────────
class GameState:
    def __init__(self):
        self.session_id = str(uuid.uuid4())  # Generate unique session ID
        self.wave = 0
        self.score = 0
        self.game_over = False
        self.start_time = pytime.time()
        self.agent_timer = 0.0

        self.enemy_count = DDA_DEFAULT_ENEMY_COUNT
        self.enemy_speed = DDA_DEFAULT_ENEMY_SPEED
        self.enemy_hp = DDA_DEFAULT_ENEMY_HP
        self.enemy_fire_rate = DDA_DEFAULT_ENEMY_FIRE_RATE
        self.asteroid_count = DDA_DEFAULT_ASTEROID_COUNT

        # Player skill tracking for intelligent AI
        self.player_skill = 0.5  # 0.0 (dying) to 1.0 (dominating)
        self.skill_update_timer = 0.0

        # GRADUAL SPAWNING SYSTEM
        self.wave_started = False
        self.enemies_spawned_this_wave = 0
        self.spawn_timer = 0.0
        self.spawn_delay = 1.0  # seconds between enemy spawns (lower = faster)

        self.enemies = []
        self.lasers = []
        self.asteroids_list = []

        # AI
        self.agent = None
        self.agent_type = "MLP"  # Track if using LSTM or MLP
        self.lstm_h = None
        self.lstm_c = None
        self._load_agent()

        print(f"🎮 Game Session: {self.session_id}")
        print(f"📊 Model: {self.agent_type}")

    @staticmethod
    def _enemy_multipliers(variant):
        hp_mult = [1.0, 1.8, 0.7][variant]
        spd_mult = [1.0, 0.6, 1.5][variant]
        return hp_mult, spd_mult

    def _retune_active_wave(self, action):
        # Difficulty changes should help the player immediately, not only on
        # the next wave. We retune enemies that are already alive and cull the
        # farthest-off threats when the DDA is trying to ease pressure.
        for enemy in self.enemies:
            hp_mult, spd_mult = self._enemy_multipliers(enemy.variant)
            target_max_hp = self.enemy_hp * hp_mult
            enemy.max_hp = target_max_hp
            if action in [2, 4]:
                enemy.hp = min(enemy.hp, target_max_hp)
            enemy.speed = self.enemy_speed * spd_mult
            enemy.fire_rate = self.enemy_fire_rate
            enemy.fire_cd = min(enemy.fire_cd, enemy.fire_rate)

        if action in [2, 4]:
            excess_enemies = max(0, len(self.enemies) - self.enemy_count)
            if excess_enemies > 0:
                self.enemies.sort(key=lambda enemy: enemy.y)
                del self.enemies[:excess_enemies]

            excess_asteroids = max(0, len(self.asteroids_list) - self.asteroid_count)
            if excess_asteroids > 0:
                self.asteroids_list.sort(key=lambda asteroid: asteroid.y)
                del self.asteroids_list[:excess_asteroids]

    def _load_agent(self):
        try:
            # Try loading LSTM model first (more advanced)
            self.agent = StatefulLSTMPolicy3D(obs_dim=7, act_dim=5, lstm_hidden=128)
            self.agent.load_state_dict(torch.load("dda_ppo_3d_lstm.pth", map_location='cpu'))
            self.agent.eval()
            self.agent_type = "LSTM"
            print("✓ DDA Agent loaded! (LSTM with Memory)")
        except FileNotFoundError:
            try:
                # Fallback to MLP model
                self.agent = MLPPolicy3D(obs_dim=7, act_dim=5, hidden=256)
                self.agent.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
                self.agent.eval()
                self.agent_type = "MLP"
                print("✓ DDA Agent loaded! (MLP - Standard)")
            except Exception as e:
                print(f"✗ No AI model found: {e}")
                self.agent = None
                self.agent_type = None

    def agent_step(self, player):
        if not self.agent:
            return

        # ─────────────────────────────────────────────────────────
        # GRACE PERIOD: Don't adjust difficulty in first 20 seconds
        # (let player accumulate stats before DDA makes decisions)
        # ─────────────────────────────────────────────────────────
        elapsed = max(1.0, pytime.time() - self.start_time)
        if elapsed < 20.0:
            return  # Skip DDA decisions during grace period
        aim_acc = player.shots_hit / max(1, player.shots_fired)
        kill_rate = player.kills / elapsed
        dmg_rate = player.damage_taken / elapsed

        health_ratio = player.health / player.max_health
        accuracy_ratio = min(aim_acc, 1.0)
        
        # ─────────────────────────────────────────────────────────
        # FIX 3 & 4: Multi-factor Decision with Performance Window
        # ─────────────────────────────────────────────────────────
        current_perf_score = (0.4 * health_ratio) + (0.3 * accuracy_ratio) + (0.3 * min(kill_rate / 2.0, 1.0))
        
        # Safely initialize the tracking variables if they don't exist yet
        if not hasattr(self, 'performance_history'):
            self.performance_history = []
            self.recovery_cooldown = 0
            self.last_damage = 0.0

        # Track the last 5 steps to establish a rolling average (smoothing)
        self.performance_history.append(current_perf_score)
        if len(self.performance_history) > 5:
            self.performance_history.pop(0)
            
        avg_score = sum(self.performance_history) / len(self.performance_history)
        self.player_skill = avg_score  

        # ─────────────────────────────────────────────────────────
        # FIX 6: Recovery Mode 🔥
        # ─────────────────────────────────────────────────────────
        damage_taken_recently = player.damage_taken - self.last_damage
        self.last_damage = player.damage_taken

        if damage_taken_recently > 0:
            self.recovery_cooldown = 5  # Start a 5-step breather period

        # ── Calculate Actual Action ──
        decision_source = "model"
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
            action = 4  # EASIER-- strictly enforced during cooldown
            decision_source = "recovery"
        else:
            # ─────────────────────────────────────────────────────────
            # WORKING HEURISTIC DDA (model is broken, disabled)
            # ─────────────────────────────────────────────────────────
            # Simple rules that actually work:
            # 1. If player healthy & making kills → increase difficulty
            # 2. If player taking damage → give relief
            # 3. If both extremes true → balance

            # Check if player is doing well
            is_dominating = (
                health_ratio > 0.7 and
                kill_rate > 0.3 and
                accuracy_ratio > 0.4
            )

            # Check if player is in trouble
            taking_heavy_damage = (
                health_ratio < 0.5 or
                player.asteroid_hits >= 1 or
                damage_taken_recently > 5.0
            )

            # Check if at difficulty extremes
            at_max_difficulty = (
                self.enemy_count >= DDA_MAX_ENEMY_COUNT - 1 and
                self.enemy_speed >= DDA_MAX_ENEMY_SPEED - 20 and
                self.spawn_delay <= 0.3
            )

            at_min_difficulty = (
                self.enemy_count <= DDA_MIN_ENEMY_COUNT and
                self.enemy_speed <= DDA_MIN_ENEMY_SPEED + 10 and
                self.spawn_delay >= 1.4
            )

            # Make decision
            if taking_heavy_damage and not at_min_difficulty:
                action = 4  # EASIER-- for relief
                decision_source = "heuristic-relief"
            elif is_dominating and not at_max_difficulty:
                action = 3  # HARDER++ to challenge
                decision_source = "heuristic-challenge"
            elif is_dominating:
                action = 1  # HARDER+ (at max, still try to push)
                decision_source = "heuristic-maxed"
            elif at_min_difficulty and health_ratio > 0.5:
                # At minimum but player fine = gradually increase to find sweet spot
                action = 1  # Soft increase
                decision_source = "heuristic-baseline"
            else:
                action = 0  # HOLD - maintain current difficulty
                decision_source = "heuristic-steady"

        # ─────────────────────────────────────────────────────────
        # FIX 1 & 5: Difficulty Smoothing & Soft Adjust
        # ─────────────────────────────────────────────────────────
        if action in [1, 3]:  # HARDER directions
            self.enemy_count += 1       # Soft adjust (+1 instead of +12)
            self.enemy_speed += 5.0     # Soft adjust (+5 instead of +40)
            self.enemy_hp += 5.0
            self.enemy_fire_rate = max(0.5, self.enemy_fire_rate - 0.1)
            self.spawn_delay -= 0.05
            if action == 3:
                self.enemy_count += 1
                self.enemy_speed += 5.0
                self.enemy_hp += 5.0
                self.enemy_fire_rate = max(0.5, self.enemy_fire_rate - 0.1)
                self.spawn_delay -= 0.07
                self.asteroid_count += 1
                
        elif action in [2, 4]:  # EASIER directions
            self.enemy_count -= 1       
            self.enemy_speed -= 5.0
            self.enemy_hp -= 5.0
            self.enemy_fire_rate = min(3.5, self.enemy_fire_rate + 0.1)
            self.spawn_delay += 0.08
            self.asteroid_count -= 1
            if action == 4:
                self.enemy_count -= 1
                self.enemy_speed -= 5.0
                self.enemy_hp -= 5.0
                self.enemy_fire_rate = min(3.5, self.enemy_fire_rate + 0.1)
                self.spawn_delay += 0.12
                self.asteroid_count -= 1

        # ─────────────────────────────────────────────────────────
        # FIX 2: Difficulty Cap Logic (Anti-overkill)
        # ─────────────────────────────────────────────────────────
        self.enemy_count = int(np.clip(self.enemy_count, DDA_MIN_ENEMY_COUNT, DDA_MAX_ENEMY_COUNT))
        self.enemy_speed = float(np.clip(self.enemy_speed, DDA_MIN_ENEMY_SPEED, DDA_MAX_ENEMY_SPEED))
        self.enemy_hp = float(np.clip(self.enemy_hp, DDA_MIN_ENEMY_HP, DDA_MAX_ENEMY_HP))
        self.enemy_fire_rate = float(np.clip(self.enemy_fire_rate, DDA_MIN_ENEMY_FIRE_RATE, DDA_MAX_ENEMY_FIRE_RATE))
        self.asteroid_count = int(np.clip(self.asteroid_count, DDA_MIN_ASTEROID_COUNT, DDA_MAX_ASTEROID_COUNT))
        self.spawn_delay = float(np.clip(self.spawn_delay, 0.25, 1.6))
        self._retune_active_wave(action)

        # ── Post Results to Backend ──
        try:
            decision_data = {
                "session_id": self.session_id,
                "action": action,
                "target_speed": self.enemy_speed,
                "target_spawn_interval": self.spawn_delay, # tracking interval
                "timestamp": pytime.time()
            }
            requests.post(f"{BACKEND_URL}/agent/decision", json=decision_data, timeout=2)
        except Exception:
            pass

        # ── Output Formatting ──
        names = ['HOLD / AI', 'HARDER+', 'EASIER-', 'HARDER++', 'EASIER--']
        skill_str = ['DYING', 'STRUGGLING', 'OK', 'GOOD', 'DOMINATING'][int(np.clip(self.player_skill * 4, 0, 4))]
        damage_indicator = f"🛡️ RECOVERY({self.recovery_cooldown})" if self.recovery_cooldown > 0 else ""

        print(f"[AI/{decision_source.upper()} ({skill_str}) Wave:{self.wave} {damage_indicator}] {names[action]} "
              f"→ enemies={self.enemy_count} spd={self.enemy_speed:.0f} hp={self.enemy_hp:.0f} "
              f"fire={self.enemy_fire_rate:.2f} ast={self.asteroid_count} spawn={self.spawn_delay:.2f}s")

    def send_telemetry(self, player):
        """Send game telemetry to backend for storage and training."""
        try:
            elapsed = pytime.time() - self.start_time
            acc = int(100 * player.shots_hit / max(1, player.shots_fired))
            diff_score = int(self.enemy_count * 12 + self.enemy_speed * 0.35 + self.enemy_hp * 0.25)

            telemetry = {
                "session_id": self.session_id,
                "elapsed_time_s": elapsed,
                "score": self.score,
                "kills": player.kills,
                "asteroids_destroyed": player.asteroid_hits,
                "accuracy": acc,
                "wave": self.wave,
                "health": player.health,
                "shield": player.shield,
                "difficulty_score": diff_score,
                "enemy_count": self.enemy_count,
                "enemy_speed": self.enemy_speed,
                "enemy_hp": self.enemy_hp,
                "enemy_fire_rate": self.enemy_fire_rate,
            }
            response = requests.post(f"{BACKEND_URL}/telemetry", json=telemetry, timeout=2)
        except Exception as e:
            pass  # Silently fail if backend unavailable

    def spawn_wave(self, player):
        """Start a new wave - enemies will spawn gradually"""
        self.wave += 1
        self.wave_started = True
        self.enemies_spawned_this_wave = 0
        self.spawn_timer = 0.0

        # Spawn delay: faster if player struggling, slower if dominating
        # This makes game responsive: hard when winning, easier when losing
        base_spawn_rate = 1.0  # seconds between spawns
        skill_factor = max(0.3, 1.0 - self.player_skill)  # 0.3-1.0
        self.spawn_delay = base_spawn_rate * skill_factor

        # Spawn asteroids immediately (not gradual)
        for _ in range(self.asteroid_count):
            x = random.uniform(30, WIDTH - 30)
            y = random.uniform(-500, -50)
            r = random.uniform(8, 25)
            spd = random.uniform(40, 100)
            self.asteroids_list.append(Asteroid(x, y, r, spd))

    def spawn_enemy_gradual(self):
        """Spawn one enemy if it's time"""
        if not self.wave_started or self.enemies_spawned_this_wave >= self.enemy_count:
            return False

        self.spawn_timer += 1.0 / FPS  # Delta time per frame
        if self.spawn_timer < self.spawn_delay:
            return False

        self.spawn_timer = 0.0
        x = random.uniform(40, WIDTH - 40)
        y = random.uniform(-300, -40)
        variant = random.choices([0, 1, 2], weights=[5, 2, 3])[0]
        hp_mult = [1.0, 1.8, 0.7][variant]
        spd_mult = [1.0, 0.6, 1.5][variant]
        self.enemies.append(EnemyBase(
            x, y,
            hp=self.enemy_hp * hp_mult,
            speed=self.enemy_speed * spd_mult,
            fire_rate=self.enemy_fire_rate,
            variant=variant,
        ))
        self.enemies_spawned_this_wave += 1
        return True

    def reset(self):
        self.wave = 0
        self.score = 0
        self.game_over = False
        self.start_time = pytime.time()
        self.agent_timer = 0.0
        self.enemy_count = DDA_DEFAULT_ENEMY_COUNT
        self.enemy_speed = DDA_DEFAULT_ENEMY_SPEED
        self.enemy_hp = DDA_DEFAULT_ENEMY_HP
        self.enemy_fire_rate = DDA_DEFAULT_ENEMY_FIRE_RATE
        self.asteroid_count = DDA_DEFAULT_ASTEROID_COUNT
        self.enemies.clear()
        self.lasers.clear()
        self.asteroids_list.clear()
        particles.clear()
        self.lstm_h = None  # Reset LSTM hidden state
        self.lstm_c = None
        self.player_skill = 0.5  # Reset skill assessment
        self.performance_history = []
        self.recovery_cooldown = 0
        self.last_damage = 0.0
        self.wave_started = False
        self.enemies_spawned_this_wave = 0
        self.spawn_timer = 0.0
        self.spawn_delay = 1.0

# ──────────────────── HUD DRAWING ────────────────────
def draw_hud(surface, player, gs):
    # Background bar
    hud_h = 50
    hud_surface = pygame.Surface((WIDTH, hud_h), pygame.SRCALPHA)
    hud_surface.fill((0, 0, 0, 160))
    surface.blit(hud_surface, (0, 0))

    # Health bar
    bar_x, bar_y, bar_w, bar_h = 15, 12, 200, 14
    hp_ratio = max(0, player.health / player.max_health)
    hp_col = HP_GREEN if hp_ratio > 0.5 else (HP_YELLOW if hp_ratio > 0.25 else HP_RED)

    pygame.draw.rect(surface, DARK_GRAY, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
    pygame.draw.rect(surface, hp_col, (bar_x, bar_y, int(bar_w * hp_ratio), bar_h), border_radius=3)
    pygame.draw.rect(surface, MID_GRAY, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=3)
    t = font_sm.render(f"HP {int(player.health)}", True, WHITE)
    surface.blit(t, (bar_x + 5, bar_y - 1))

    # Shield bar
    sh_y = bar_y + 18
    sh_ratio = max(0, player.shield / player.max_shield)
    pygame.draw.rect(surface, DARK_GRAY, (bar_x, sh_y, bar_w, bar_h), border_radius=3)
    pygame.draw.rect(surface, SHIELD_BLUE, (bar_x, sh_y, int(bar_w * sh_ratio), bar_h), border_radius=3)
    pygame.draw.rect(surface, MID_GRAY, (bar_x, sh_y, bar_w, bar_h), 1, border_radius=3)
    t = font_sm.render(f"SH {int(player.shield)}", True, WHITE)
    surface.blit(t, (bar_x + 5, sh_y - 1))

    # Score
    t = font_hud.render(f"SCORE  {gs.score}", True, YELLOW)
    surface.blit(t, (WIDTH - 220, 8))

    # Wave
    t = font_hud.render(f"WAVE  {gs.wave}", True, WHITE)
    surface.blit(t, (WIDTH - 220, 28))

    # Kills
    t = font_hud.render(f"KILLS  {player.kills}", True, NEON_RED)
    surface.blit(t, (WIDTH // 2 - 60, 8))

    # Accuracy
    acc = int(100 * player.shots_hit / max(1, player.shots_fired))
    t = font_hud.render(f"ACC  {acc}%", True, NEON_GREEN)
    surface.blit(t, (WIDTH // 2 - 60, 28))

    # Asteroid hits counter
    asteroid_warning_col = NEON_RED if player.asteroid_hits >= 2 else (ORANGE if player.asteroid_hits >= 1 else WHITE)
    t = font_hud.render(f"ASTEROID HITS  {player.asteroid_hits}/3", True, asteroid_warning_col)
    surface.blit(t, (WIDTH // 2 - 200, 8))

    # DDA indicator only
    if gs.agent:
        t = font_sm.render("AI DDA: ACTIVE", True, CYAN)
    else:
        t = font_sm.render("AI DDA: OFF", True, MID_GRAY)
    surface.blit(t, (WIDTH // 2 + 80, 15))
    # Quick difficulty indicator so player can feel AI adjustments.
    diff_score = int(gs.enemy_count * 12 + gs.enemy_speed * 0.35 + gs.enemy_hp * 0.25)
    t = font_sm.render(f"DIFF: {diff_score}", True, ORANGE)
    surface.blit(t, (WIDTH // 2 + 200, 32))

def draw_legend(surface):
    """Draw on-screen legend for what to kill/avoid."""
    legend_surface = pygame.Surface((280, 120), pygame.SRCALPHA)
    legend_surface.fill((0, 0, 0, 200))
    surface.blit(legend_surface, (10, HEIGHT - 130))

    # Legend title
    t = font_sm.render("SURVIVAL:", True, YELLOW)
    surface.blit(t, (15, HEIGHT - 125))

    # Enemy legend
    pygame.draw.circle(surface, NEON_RED, (28, HEIGHT - 100), 5)
    t = font_sm.render("Red = SHOOT", True, WHITE)
    surface.blit(t, (38, HEIGHT - 107))

    # Asteroid legend
    pygame.draw.circle(surface, (120, 80, 50), (28, HEIGHT - 75), 5)
    t = font_sm.render("Brown = AVOID", True, WHITE)
    surface.blit(t, (38, HEIGHT - 82))

    # Deaths
    t = font_sm.render("⚠️ 3 asteroids OR health=0 = END", True, NEON_RED)
    surface.blit(t, (15, HEIGHT - 50))

def draw_level_select(surface):
    surface.fill(SPACE_BG)
    for s in bg_stars:
        s.draw(surface)

    title = font_lg.render("SPACE DOGFIGHT", True, CYAN)
    subtitle = font_md.render("Get Ready!", True, WHITE)
    instructions = [
        "🔴 RED SHIPS = SHOOT!",
        "🟤 BROWN ASTEROIDS = AVOID!",
        f"⚠️ 3 asteroid hits OR health→0 = GAME OVER",
        "📊 AI Difficulty: AUTO-ADJUSTS to your skill",
        "",
        "↑ WASD/Arrows = Move  | SPACE = Shoot",
        "R = Restart  |  Q/ESC = Quit",
    ]

    surface.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 150))
    surface.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, HEIGHT // 2 - 90))
    for i, txt in enumerate(instructions):
        if i == 4: continue  # skip blank line
        color = NEON_RED if "🔴" in txt else (ORANGE if "🟤" in txt else (NEON_RED if "⚠️" in txt else (CYAN if "📊" in txt else WHITE)))
        line = font_sm.render(txt, True, color)
        surface.blit(line, (WIDTH // 2 - line.get_width() // 2, HEIGHT // 2 - 40 + i * 28))

    start_txt = font_md.render("Press SPACE to Start", True, NEON_GREEN)
    surface.blit(start_txt, (WIDTH // 2 - start_txt.get_width() // 2, HEIGHT - 50))

def draw_game_over(surface, player, gs):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    t = font_lg.render("SHIP DESTROYED", True, NEON_RED)
    surface.blit(t, (WIDTH // 2 - t.get_width() // 2, HEIGHT // 2 - 80))

    info = [
        f"Asteroid Hits: {player.asteroid_hits}/3",
        f"Score: {gs.score}    Kills: {player.kills}    Waves: {gs.wave}",
        f"Accuracy: {int(100 * player.shots_hit / max(1, player.shots_fired))}%",
    ]
    for i, line in enumerate(info):
        t = font_md.render(line, True, WHITE)
        surface.blit(t, (WIDTH // 2 - t.get_width() // 2, HEIGHT // 2 - 20 + i * 25))

    t = font_md.render("Press R to Restart  |  Q to Quit", True, CYAN)
    surface.blit(t, (WIDTH // 2 - t.get_width() // 2, HEIGHT // 2 + 80))

# ──────────────────── MAIN GAME LOOP ────────────────────
def main():
    # Start screen
    start_shown = False
    while not start_shown:
        dt = clock.tick(FPS) / 1000.0
        for s in bg_stars:
            s.update(dt)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    start_shown = True
        draw_level_select(screen)
        pygame.display.flip()

    player = Player()
    gs = GameState()
    gs.spawn_wave(player)

    print("\n+---------------------------------------+")
    print("|   SPACE DOGFIGHT - AI DDA Engine      |")
    print("+---------------------------------------+")
    print(f"|  AI Model: {gs.agent_type:<28}|")
    print("|  🔴 RED SHIPS = SHOOT!               |")
    print("|  🟤 BROWN ASTEROIDS = AVOID!         |")
    print("|  ⚠️ 3 asteroid hits OR health 0      |")
    print("|  AI increases difficulty aggressively|")
    print("| Arrow Keys / WASD = Move              |")
    print("| Space = Shoot    R = Restart          |")
    print("| Q / ESC = Quit                        |")
    print("+---------------------------------------+\n")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)  # clamp

        # ─── Events ───
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                if gs.game_over and event.key == pygame.K_r:
                    player = Player()
                    gs.reset()
                    gs.spawn_wave(player)

        if gs.game_over:
            screen.fill(SPACE_BG)
            for s in bg_stars: s.draw(screen)
            draw_game_over(screen, player, gs)
            pygame.display.flip()
            continue

        keys = pygame.key.get_pressed()

        # ─── Update player ───
        player.update(dt, keys)

        # ─── Shoot ───
        if keys[pygame.K_SPACE] and player.fire_cooldown <= 0:
            gs.lasers.append(Laser(player.x - 8, player.y - 15, 'player'))
            gs.lasers.append(Laser(player.x + 8, player.y - 15, 'player'))
            player.shots_fired += 2
            player.fire_cooldown = 0.12

        # ─── Update stars ───
        for s in bg_stars:
            s.update(dt)

        # ─── Update enemies ───
        for e in gs.enemies[:]:
            e.update(dt, player)
            # Enemy shoot
            if e.fire_cd <= 0 and e.alive and e.y > 0:
                gs.lasers.append(Laser(e.x, e.y + e.size, 'enemy'))
                e.fire_cd = e.fire_rate + random.uniform(-0.3, 0.3)

            # Collision with player
            if e.alive and abs(e.x - player.x) < 25 and abs(e.y - player.y) < 25:
                player.take_damage(25)
                e.alive = False
                spawn_explosion(e.x, e.y, 15, NEON_RED)

        gs.enemies = [e for e in gs.enemies if e.alive]

        # ─── Update asteroids ───
        for a in gs.asteroids_list[:]:
            a.update(dt)
            if a.alive and abs(a.x - player.x) < a.radius + 15 and abs(a.y - player.y) < a.radius + 15:
                player.asteroid_hits += 1
                a.alive = False
                spawn_explosion(a.x, a.y, 10, ORANGE)
                if player.asteroid_hits >= 3:
                    gs.game_over = True

        gs.asteroids_list = [a for a in gs.asteroids_list if a.alive]

        # ─── Update lasers ───
        for l in gs.lasers[:]:
            l.update(dt)
            if l.owner == 'player':
                # Check collision with enemies only
                for e in gs.enemies:
                    if e.alive and abs(l.x - e.x) < e.size and abs(l.y - e.y) < e.size:
                        player.shots_hit += 1
                        e.take_hit(l.damage)
                        l.alive = False
                        if not e.alive:
                            player.kills += 1
                            gs.score += 100
                            spawn_explosion(e.x, e.y, 25, ORANGE)
                        break
            elif l.owner == 'enemy':
                if abs(l.x - player.x) < 15 and abs(l.y - player.y) < 20:
                    player.take_damage(l.damage)
                    l.alive = False

        gs.lasers = [l for l in gs.lasers if l.alive]

        # ─── Update particles ───
        for p in particles[:]:
            if not p.update(dt):
                particles.remove(p)

        # ─── Spawn next wave (gradual) ───
        if len(gs.enemies) == 0 and gs.enemies_spawned_this_wave >= gs.enemy_count:
            gs.spawn_wave(player)

        # ─── Gradual enemy spawning ───
        gs.spawn_enemy_gradual()

        # ─── DDA agent ───
        gs.agent_timer += dt
        if gs.agent_timer >= 2.0:
            gs.agent_timer = 0.0
            gs.agent_step(player)
            gs.send_telemetry(player)

        # ─── Check death ───
        if player.health <= 0:
            gs.game_over = True
            spawn_explosion(player.x, player.y, 40, CYAN)

        # ──── DRAW ────
        screen.fill(SPACE_BG)

        # Stars
        for s in bg_stars:
            s.draw(screen)

        # Grid lines (subtle depth)
        for gx in range(0, WIDTH, 80):
            pygame.draw.line(screen, GRID_COLOR, (gx, 0), (gx, HEIGHT), 1)
        for gy in range(0, HEIGHT, 80):
            pygame.draw.line(screen, GRID_COLOR, (0, gy), (WIDTH, gy), 1)

        # Asteroids
        for a in gs.asteroids_list:
            a.draw(screen)

        # Enemies
        for e in gs.enemies:
            e.draw(screen)

        # Lasers
        for l in gs.lasers:
            l.draw(screen)

        # Particles
        for p in particles:
            p.draw(screen)

        # Player
        player.draw(screen)

        # HUD
        draw_hud(screen, player, gs)

        # Legend
        draw_legend(screen)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
