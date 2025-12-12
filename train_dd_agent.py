# train_dd_agent.py
# Minimal PPO trainer (PyTorch) for the DDAEnv in env_wrapper.py
# Usage: python3 train_dd_agent.py
# Saves model to dda_ppo.pth

import math, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from env_wrapper import DDAEnv, SimpleSimPlayer

# ---- Hyperparameters ----
SEED = 42
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
POLICY_LR = 3e-4
VALUE_LR = 1e-3
ENT_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

NUM_UPDATES = 400         # number of policy updates
STEPS_PER_UPDATE = 16     # how many env steps per collection before update
MINI_BATCHES = 4
EPOCHS = 4
EPISODE_MS = 20_000       # shorter episodes (20s) for faster samples
STEP_MS = 1000            # env step corresponds to 1 second game sim
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dda_ppo.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---- Policy / Value networks ----
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden, act_dim)   # logits
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

# ---- Helpers ----
def compute_gae(rewards, masks, values, gamma=GAMMA, lam=LAMBDA):
    # rewards, masks, values are lists/arrays (len = T)
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - masks[t]
        nextvalues = values[t+1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values[:T]
    return advantages, returns

# ---- Training loop ----
def train():
    # env & dims
    sim = SimpleSimPlayer(skill=0.65)   # train against an "average" player
    env = DDAEnv(sim_player=sim, step_ms=STEP_MS, episode_ms=EPISODE_MS, seed=SEED)
    obs0 = env.reset()
    obs_dim = obs0.shape[0]
    act_dim = 5  # actions: 0..4

    policy = MLPPolicy(obs_dim, act_dim, hidden=128).to(DEVICE)
    optim_policy = torch.optim.Adam([p for p in policy.parameters()], lr=POLICY_LR)

    # storage containers
    for update in range(1, NUM_UPDATES + 1):
        # collect rollout
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        mask_buf = []

        # collect STEPS_PER_UPDATE episodes (not full episodes, but steps)
        for step in range(STEPS_PER_UPDATE):
            obs = env.reset() if step == 0 and (update == 1) else obs  # preserve obs
            # for stability, reset occasionally if done
            if step == 0:
                obs = env.reset()

            # single environment: run one step (we can treat each call as one sample)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, value = policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                m = Categorical(probs)
                action = m.sample().item()
                logp = m.log_prob(torch.tensor(action, device=DEVICE)).item()
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

        # add final value for bootstrap
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, last_val = policy(obs_tensor)
            last_val = last_val.item()

        # compute GAE advantages
        values = np.array(val_buf + [last_val], dtype=np.float32)
        advantages, returns = compute_gae(rew_buf, mask_buf, values)

        # convert buffers to tensors
        obs_arr = torch.as_tensor(np.array(obs_buf, dtype=np.float32), device=DEVICE)
        acts_arr = torch.as_tensor(np.array(act_buf), device=DEVICE)
        old_logp_arr = torch.as_tensor(np.array(logp_buf, dtype=np.float32), device=DEVICE)
        adv_arr = torch.as_tensor(advantages, device=DEVICE)
        ret_arr = torch.as_tensor(returns, device=DEVICE)

        # normalize advantages
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # PPO update epochs
        batch_size = max(1, len(obs_arr) // MINI_BATCHES)
        for epoch in range(EPOCHS):
            # shuffle indices
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
                dist = torch.distributions.Categorical(logits=logits)
                mb_logp = dist.log_prob(mb_acts)
                mb_entropy = dist.entropy().mean()

                ratio = torch.exp(mb_logp - mb_oldlogp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_ret - values_pred).pow(2).mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * mb_entropy

                optim_policy.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optim_policy.step()

        # logging
        avg_rew = np.mean(rew_buf) if len(rew_buf)>0 else 0.0
        avg_acc = info.get("accuracy", 0.0)
        print(f"[{update}/{NUM_UPDATES}] avg_rew={avg_rew:.3f} last_acc={avg_acc:.3f} "
              f"speed={env.target_fall_speed:.1f} spawn_ms={env.target_spawn_interval:.0f}")

        # periodically save
        if update % 50 == 0 or update == NUM_UPDATES:
            torch.save(policy.state_dict(), MODEL_PATH)
            print("Saved model to", MODEL_PATH)

    print("Training complete. Model saved to", MODEL_PATH)
    return policy

if __name__ == "__main__":
    start = time.time()
    policy = train()
    print("Elapsed:", time.time() - start)
