# train_dda_3d_lstm.py
# PPO Trainer with LSTM Memory for 3D Space Dogfight DDA
# LSTM remembers 30-second gameplay window (15 decisions) to detect frustration patterns
# Usage: python3 train_dda_3d_lstm.py
# Saves model to dda_ppo_3d_lstm.pth

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
ENT_COEF = 0.02
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

NUM_UPDATES = 600
STEPS_PER_UPDATE = 32       # 32 steps * 2000ms = 64 seconds of gameplay per update
MINI_BATCHES = 4
EPOCHS = 4
EPISODE_MS = 40_000         # 40-second episodes
STEP_MS = 2000              # DDA decision every 2 seconds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dda_ppo_3d_lstm.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---- LSTM Policy Network (STATEFUL) ----
class StatefulLSTMPolicy3D(nn.Module):
    """Recurrent actor-critic using LSTM for memory of past 30 seconds."""
    def __init__(self, obs_dim=7, act_dim=5, lstm_hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lstm_hidden = lstm_hidden

        # LSTM layer: 7 obs dim -> 128 hidden
        self.lstm = nn.LSTM(obs_dim, lstm_hidden, num_layers=1, batch_first=False)

        # Shared backbone after LSTM
        self.net = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.Tanh(),
        )

        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: shape (batch_size, obs_dim) for single timestep
            hidden_state: tuple (h, c) of shape (1, batch_size, lstm_hidden)

        Returns:
            logits: (batch_size, act_dim)
            value: (batch_size,)
            hidden_state: new (h, c) for next timestep
        """
        if hidden_state is None:
            h_t = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
            c_t = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
            hidden_state = (h_t, c_t)

        # LSTM expects (seq_len, batch_size, input_size)
        x_seq = x.unsqueeze(0)  # (1, batch_size, obs_dim)
        lstm_out, (h_new, c_new) = self.lstm(x_seq, hidden_state)

        # Use LSTM output: (1, batch_size, lstm_hidden) -> (batch_size, lstm_hidden)
        lstm_out = lstm_out.squeeze(0)

        # Pass through backbone
        h = self.net(lstm_out)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value, (h_new, c_new)

    def forward_sequence(self, x_seq, hidden_state=None):
        """
        Process sequence of observations (for training with BPTT).
        Args:
            x_seq: shape (seq_len, batch_size, obs_dim)
            hidden_state: tuple (h, c) of shape (1, batch_size, lstm_hidden)

        Returns:
            logits_seq: (seq_len, batch_size, act_dim)
            values_seq: (seq_len, batch_size)
            hidden_state: final (h, c)
        """
        if hidden_state is None:
            h_t = torch.zeros(1, x_seq.shape[1], self.lstm_hidden, device=x_seq.device)
            c_t = torch.zeros(1, x_seq.shape[1], self.lstm_hidden, device=x_seq.device)
            hidden_state = (h_t, c_t)

        lstm_out, (h_new, c_new) = self.lstm(x_seq, hidden_state)  # (seq_len, batch, lstm_hidden)

        h = self.net(lstm_out)  # (seq_len, batch, 64)
        logits = self.actor(h)  # (seq_len, batch, act_dim)
        values = self.critic(h).squeeze(-1)  # (seq_len, batch)

        return logits, values, (h_new, c_new)

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
    skill_levels = [0.2, 0.35, 0.5, 0.65, 0.8, 0.92]
    current_skill_idx = 0

    sim = SimPlayer(skill=skill_levels[current_skill_idx])
    env = DDAEnv3D(sim_player=sim, step_ms=STEP_MS, episode_ms=EPISODE_MS, seed=SEED)

    obs_dim = 7
    act_dim = 5
    policy = StatefulLSTMPolicy3D(obs_dim, act_dim, lstm_hidden=128).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    obs = env.reset()
    best_avg_reward = -float('inf')

    # Initialize hidden state
    hidden_state = None

    print(f"Training 3D Space Dogfight DDA Agent (LSTM Memory)")
    print(f"Device: {DEVICE}, Updates: {NUM_UPDATES}, Steps/Update: {STEPS_PER_UPDATE}")
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}, LSTM hidden: 128")
    print("=" * 70)

    for update in range(1, NUM_UPDATES + 1):
        # ---- Collect rollout ----
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, mask_buf = [], [], [], [], [], []
        hidden_buf = []  # Store hidden states for BPTT

        for step in range(STEPS_PER_UPDATE):
            if step == 0:
                # Rotate skill level every update
                current_skill_idx = (update - 1) % len(skill_levels)
                sim = SimPlayer(skill=skill_levels[current_skill_idx])
                env = DDAEnv3D(sim_player=sim, step_ms=STEP_MS, episode_ms=EPISODE_MS,
                               seed=SEED + update)
                obs = env.reset()
                hidden_state = None  # Reset hidden state at episode start

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                logits, value, hidden_state_new = policy(obs_tensor, hidden_state)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                v = value.item()

            hidden_buf.append(hidden_state)
            hidden_state = hidden_state_new

            next_obs, reward, done, info = env.step(action)

            obs_buf.append(obs.copy())
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(float(reward))
            val_buf.append(v)
            mask_buf.append(1.0 if not done else 0.0)

            if done:
                obs = env.reset()
                hidden_state = None  # Reset hidden state on episode end
            else:
                obs = next_obs

        # Bootstrap value (use final hidden state)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, last_val, _ = policy(obs_tensor, hidden_state)
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

        # ---- PPO update (NOTE: LSTM training uses full sequence for BPTT) ----
        # For LSTM, we process the full rollout sequence to maintain temporal coherence
        batch_idxs = list(range(len(obs_arr)))

        for epoch in range(EPOCHS):
            random.shuffle(batch_idxs)

            # Process whole sequence per epoch (BPTT)
            obs_seq = obs_arr[batch_idxs].unsqueeze(1)  # (seq_len, 1, obs_dim) for single batch
            acts_seq = acts_arr[batch_idxs].unsqueeze(1)  # (seq_len, 1)
            old_logp_seq = old_logp_arr[batch_idxs].unsqueeze(1)  # (seq_len, 1)
            adv_seq = adv_arr[batch_idxs].unsqueeze(1)  # (seq_len, 1)
            ret_seq = ret_arr[batch_idxs].unsqueeze(1)  # (seq_len, 1)

            logits_seq, values_pred_seq, _ = policy.forward_sequence(obs_seq, hidden_state=None)

            # Flatten sequences for loss computation
            logits_flat = logits_seq.reshape(-1, 5)  # (seq_len, 5)
            values_flat = values_pred_seq.reshape(-1)
            acts_flat = acts_seq.reshape(-1)
            old_logp_flat = old_logp_seq.reshape(-1)
            adv_flat = adv_seq.reshape(-1)
            ret_flat = ret_seq.reshape(-1)

            dist = Categorical(logits=logits_flat)
            new_logp = dist.log_prob(acts_flat)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_flat)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (ret_flat - values_flat).pow(2).mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * entropy

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
            print(f"  → Saved LSTM model to {MODEL_PATH}")

    print(f"\nTraining complete! Best avg_reward: {best_avg_reward:.2f}")
    print(f"LSTM Model saved to {MODEL_PATH}")
    return policy


if __name__ == "__main__":
    start = time.time()
    policy = train()
    elapsed = time.time() - start
    print(f"Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
