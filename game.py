
# game.py - Telemetry-enabled Block Dodger (RL-ready)
import pygame
import random
import sys
import csv
import time
from dataclasses import dataclass
import torch
import numpy as np
from train_dd_agent import MLPPolicy

from collections import deque

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 480, 640
FPS = 60
PLAYER_SIZE = 40
PLAYER_SPEED_PPS = 240
BLOCK_WIDTH = 40
BLOCK_HEIGHT = 40

# Difficulty parameters
_target_fall_speed = 600.0
_target_spawn_interval = 450
DIFFICULTY_SMOOTH_SEC = 0.4

# Telemetry logging
TELEMETRY_CSV = "telemetry_log.csv"
CSV_FIELDS = [
    "timestamp", "dt_ms", "score", "elapsed_time_s",
    "blocks_spawned", "blocks_avoided", "blocks_collided",
    "reaction_time_latest_ms", "reaction_time_moving_avg_ms",
    "accuracy", "score_rate_per_s", "mistakes", "time_since_last_error_s",
    "current_fall_speed_pps", "current_spawn_interval_ms"
]

SEED = 42
random.seed(SEED)

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

_current_fall_speed = _target_fall_speed
_current_spawn_interval = _target_spawn_interval

def rects_collide(a, b):
    return (a.x < b.x + b.w and a.x + a.w > b.x and
            a.y < b.y + b.h and a.y + a.h > b.y)

def game_over(screen, font, score):
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((10, 10, 10))
    screen.blit(overlay, (0, 0))

    msg1 = font.render("GAME OVER", True, (255, 50, 50))
    msg2 = font.render(f"Score: {score}", True, (255, 255, 255))
    msg3 = font.render("Press R to Restart, Q to Quit", True, (220, 220, 220))

    screen.blit(msg1, ((WIDTH - msg1.get_width()) // 2, HEIGHT // 2 - 40))
    screen.blit(msg2, ((WIDTH - msg2.get_width()) // 2, HEIGHT // 2))
    screen.blit(msg3, ((WIDTH - msg3.get_width()) // 2, HEIGHT // 2 + 40))

    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    waiting = False
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

def build_agent_obs(blocks_spawned, blocks_avoided, mistakes, target_fall_speed, target_spawn_interval):
    """
    Must match env_wrapper._obs() used during training:
    returns np.array([accuracy, mistakes_norm, speed_norm, spawn_norm], dtype=float32)
    """
    accuracy = (blocks_avoided / blocks_spawned) if blocks_spawned > 0 else 1.0
    mistakes_norm = float(np.tanh(mistakes / 5.0))
    speed_norm = float(np.clip(target_fall_speed / 1500.0, 0.0, 1.0))
    spawn_norm = float(np.clip(target_spawn_interval / 5000.0, 0.0, 1.0))
    return np.array([accuracy, mistakes_norm, speed_norm, spawn_norm], dtype=np.float32)


def main():
        
    global _current_fall_speed, _current_spawn_interval, _target_fall_speed, _target_spawn_interval

    

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Block Dodger - Telemetry")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
        # ---- Load DDA agent (trained policy) ----
    AGENT_INTERVAL_MS = 1000   # agent acts every 1000 ms (1s). Tweak as needed.
    time_since_last_agent = 0

    agent_policy = None
    try:
        obs_dim = 4   # [accuracy, mistakes_norm, speed_norm, spawn_norm]
        act_dim = 5
        agent_policy = MLPPolicy(obs_dim, act_dim, hidden=128)
        agent_policy.load_state_dict(torch.load("dda_ppo.pth", map_location="cpu"))
        agent_policy.eval()
        print("DDA agent loaded successfully.")
    except Exception as e:
        agent_policy = None
        print("Could not load DDA agent (continuing without it):", e)


    player = Player(x=(WIDTH - PLAYER_SIZE) / 2, y=HEIGHT - PLAYER_SIZE - 10,
                    w=PLAYER_SIZE, h=PLAYER_SIZE)

    blocks = []
    score = 0
    running = True
    last_spawn_time = pygame.time.get_ticks()
    start_time = pygame.time.get_ticks()

    blocks_spawned = 0
    blocks_avoided = 0
    blocks_collided = 0
    mistakes = 0

    last_move_time_ms = pygame.time.get_ticks()
    reaction_times = deque(maxlen=50)

    with open(TELEMETRY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

    while running:
        dt_ms = clock.tick(FPS)
        dt_sec = dt_ms / 1000.0

        now_ms = pygame.time.get_ticks()

        # Spawn logic
        if now_ms - last_spawn_time >= int(_current_spawn_interval):
    # bias spawn towards center
            bx = int(random.gauss(WIDTH / 2, WIDTH / 6))
            bx = max(0, min(WIDTH - BLOCK_WIDTH, bx))
            spawn_t = now_ms
            blocks.append(Block(x=bx, y=-BLOCK_HEIGHT, w=BLOCK_WIDTH, h=BLOCK_HEIGHT, spawn_time_ms=spawn_t))
            last_spawn_time = spawn_t
            blocks_spawned += 1


        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        moved_this_frame = False
        if keys[pygame.K_LEFT]:
            player.x -= PLAYER_SPEED_PPS * dt_sec
            moved_this_frame = True
        if keys[pygame.K_RIGHT]:
            player.x += PLAYER_SPEED_PPS * dt_sec
            moved_this_frame = True

        player.x = max(0, min(WIDTH - player.w, player.x))

        # Update block positions
        for b in blocks:
            b.y += _current_fall_speed * dt_sec

        # Remove passed blocks
        new_blocks = []
        for b in blocks:
            if b.y > HEIGHT:
                score += 1
                blocks_avoided += 1
            else:
                new_blocks.append(b)
        blocks = new_blocks

        # Collision detection
        dead = False
        for b in blocks:
            if rects_collide(player, b):
                dead = True
                blocks_collided += 1
                mistakes += 1
                break

        screen.fill((30, 30, 40))
        pygame.draw.rect(screen, (80, 200, 120), (player.x, player.y, player.w, player.h))
        for b in blocks:
            pygame.draw.rect(screen, (200, 80, 80), (b.x, b.y, b.w, b.h))
                # ---- Agent periodic action (if loaded) ----
        time_since_last_agent += dt_ms
        if agent_policy is not None and time_since_last_agent >= AGENT_INTERVAL_MS:
            # build observation vector from current telemetry
            obs_vec = build_agent_obs(blocks_spawned, blocks_avoided, mistakes, _target_fall_speed, _target_spawn_interval)
            obs_t = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)  # shape (1,4)
            with torch.no_grad():
                logits, _ = agent_policy(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            # map action to difficulty deltas (same mapping as env_wrapper/train)
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

            # apply to target difficulty (clamped)
            _target_fall_speed = float(max(50.0, min(1500.0, _target_fall_speed + ds)))
            _target_spawn_interval = float(max(150.0, min(5000.0, _target_spawn_interval + dpi)))

            # debug print (optional)
            print(f"[AGENT] action={action} -> target_speed={_target_fall_speed:.1f}, spawn_ms={_target_spawn_interval:.0f}")

            time_since_last_agent = 0


        pygame.display.flip()

        if dead:
            blocks.clear()
            game_over(screen, font, score)
            score = blocks_spawned = blocks_avoided = blocks_collided = mistakes = 0
            reaction_times.clear()
            start_time = pygame.time.get_ticks()
            last_spawn_time = pygame.time.get_ticks()
            continue

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

