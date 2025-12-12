Rein_Block — Adaptive Difficulty Game (Reinforcement Learning + Pygame)

Rein_Block is a Python/Pygame block-dodging game enhanced with Dynamic Difficulty Adjustment (DDA). A reinforcement learning agent (trained using PPO) continuously adjusts the game's difficulty based on player performance. Telemetry is logged for analysis, and the project includes a custom RL environment and training pipeline.

Features

Pygame arcade-style block dodging

RL-driven dynamic difficulty (speed + spawn rate)

PPO agent integrated into live gameplay

Telemetry logging (accuracy, mistakes, reaction trends)

Custom environment for training


Project Structure:- 
game.py              # Main game with RL-based DDA
env_wrapper.py       # Gym-style environment
train_dd_agent.py    # PPO training script
dda_ppo.pth          # Trained model weights
requirements.txt


setup using :-
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 game.py


How It Works

The agent observes accuracy, mistakes, fall speed, and spawn interval, then outputs difficulty-adjusting actions to maintain a balanced gameplay experience.

Future Improvements

More block types and movement patterns

Web version of the game

Analytics dashboard

Additional RL algorithms
