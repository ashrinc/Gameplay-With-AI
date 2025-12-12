# eval snippet (python REPL)
import torch, numpy as np
from env_wrapper import DDAEnv, SimpleSimPlayer
from train_dd_agent import MLPPolicy, DEVICE

env = DDAEnv(SimpleSimPlayer(skill=0.6), step_ms=1000, episode_ms=20000)
obs = env.reset()
policy = MLPPolicy(obs.shape[0], 5).to(DEVICE)
policy.load_state_dict(torch.load("dda_ppo.pth", map_location=DEVICE))
policy.eval()

for ep in range(5):
    obs = env.reset()
    total_r = 0.0
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(obs_t)
            a = torch.argmax(logits, dim=-1).item()
        obs, r, done, info = env.step(a)
        total_r += r
    print("Episode", ep, "total reward", total_r, "final accuracy", info["accuracy"])
