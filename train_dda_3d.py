# train_dda_3d.py
# PPO Trainer for the 3D Space Dogfight DDA environment
# Usage: python3 train_dda_3d.py
# Saves model to dda_ppo_3d.pth

import math, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from env_wrapper_3d import DDAEnv3D, SimPlayer

# ---- Hyperparameters ----
SEED = 42
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.02            # Slightly higher entropy for exploration in larger action space
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

NUM_UPDATES = 600           # More updates for complex 3D env
STEPS_PER_UPDATE = 32       # More steps per update for stability
MINI_BATCHES = 4
EPOCHS = 4
EPISODE_MS = 40_000         # 40-second episodes
STEP_MS = 2000              # DDA decision every 2 seconds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dda_ppo_3d.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---- Policy/Value Network ----
class MLPPolicy3D(nn.Module):
    """Shared-backbone actor-critic for 3D DDA."""
    def __init__(self, obs_dim=7, act_dim=5, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden // 2, act_dim)
        self.critic = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.net(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

# ---- GAE ----
def compute_gae(rewards, masks, values, gamma=GAMMA, lam=LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - masks[t]
        nextvalues = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values[:T]
    return advantages, returns

# ---- Training ----
def train():
    # Train across player levels so DDA adapts for beginners to pros.
    skill_levels = [0.2, 0.35, 0.5, 0.65, 0.8, 0.92]
    current_skill_idx = 0

    sim = SimPlayer(skill=skill_levels[current_skill_idx])
    env = DDAEnv3D(sim_player=sim, step_ms=STEP_MS, episode_ms=EPISODE_MS, seed=SEED)

    obs_dim = 7
    act_dim = 5
    policy = MLPPolicy3D(obs_dim, act_dim, hidden=256).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    obs = env.reset()
    best_avg_reward = -float('inf')

    print(f"Training 3D Space Dogfight DDA Agent")
    print(f"Device: {DEVICE}, Updates: {NUM_UPDATES}, Steps/Update: {STEPS_PER_UPDATE}")
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")
    print("=" * 70)

    for update in range(1, NUM_UPDATES + 1):
        # ---- Collect rollout ----
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, mask_buf = [], [], [], [], [], []

        for step in range(STEPS_PER_UPDATE):
            if step == 0:
                # Rotate skill level every update for curriculum training
                current_skill_idx = (update - 1) % len(skill_levels)
                sim = SimPlayer(skill=skill_levels[current_skill_idx])
                env = DDAEnv3D(sim_player=sim, step_ms=STEP_MS, episode_ms=EPISODE_MS,
                               seed=SEED + update)
                obs = env.reset()

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, value = policy(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                v = value.item()

            next_obs, reward, done, info = env.step(action)

            obs_buf.append(obs.copy())
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(float(reward))
            val_buf.append(v)
            mask_buf.append(1.0 if not done else 0.0)

            if done:
                obs = env.reset()
            else:
                obs = next_obs

        # Bootstrap value
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, last_val = policy(obs_tensor)
            last_val = last_val.item()

        # GAE
        values = np.array(val_buf + [last_val], dtype=np.float32)
        advantages, returns = compute_gae(rew_buf, mask_buf, values)

        # To tensors
        obs_arr = torch.as_tensor(np.array(obs_buf, dtype=np.float32), device=DEVICE)
        acts_arr = torch.as_tensor(np.array(act_buf), device=DEVICE)
        old_logp_arr = torch.as_tensor(np.array(logp_buf, dtype=np.float32), device=DEVICE)
        adv_arr = torch.as_tensor(advantages, device=DEVICE)
        ret_arr = torch.as_tensor(returns, device=DEVICE)

        # Normalize advantages
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # ---- PPO update ----
        batch_size = max(1, len(obs_arr) // MINI_BATCHES)
        for epoch in range(EPOCHS):
            idxs = np.arange(len(obs_arr))
            np.random.shuffle(idxs)
            for start in range(0, len(obs_arr), batch_size):
                mb_idx = idxs[start:start + batch_size]
                mb_obs = obs_arr[mb_idx]
                mb_acts = acts_arr[mb_idx]
                mb_oldlogp = old_logp_arr[mb_idx]
                mb_adv = adv_arr[mb_idx]
                mb_ret = ret_arr[mb_idx]

                logits, values_pred = policy(mb_obs)
                dist = Categorical(logits=logits)
                mb_logp = dist.log_prob(mb_acts)
                mb_entropy = dist.entropy().mean()

                ratio = torch.exp(mb_logp - mb_oldlogp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_ret - values_pred).pow(2).mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * mb_entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # ---- Logging ----
        avg_rew = np.mean(rew_buf)
        if update % 10 == 0 or update <= 5:
            skill = skill_levels[current_skill_idx]
            print(f"[{update:3d}/{NUM_UPDATES}] skill={skill:.1f} avg_rew={avg_rew:+.2f} "
                  f"kills={info.get('kills',0)} acc={info.get('aim_accuracy',0):.2f} "
                  f"hp={info.get('health',0):.0f} wave={info.get('wave',0)} "
                  f"e_count={info.get('enemy_count',0)} e_spd={info.get('enemy_speed', 0):.0f} "
                  f"e_hp={info.get('enemy_hp',0):.0f}")

        # Save best + periodic
        if avg_rew > best_avg_reward:
            best_avg_reward = avg_rew
            torch.save(policy.state_dict(), MODEL_PATH)
        if update % 100 == 0 or update == NUM_UPDATES:
            torch.save(policy.state_dict(), MODEL_PATH)
            print(f"  → Saved model to {MODEL_PATH}")

    print(f"\nTraining complete! Best avg_reward: {best_avg_reward:.2f}")
    print(f"Model saved to {MODEL_PATH}")
    return policy


if __name__ == "__main__":
    start = time.time()
    policy = train()
    elapsed = time.time() - start
    print(f"Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
