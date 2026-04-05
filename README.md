# Gamee - AI Dynamic Difficulty Space Dogfight

Welcome to **Gamee**, a beautifully polished Pygame arcade shooter that intelligently adapts to your skill level using Reinforcement Learning! The game features a proximal policy optimization (PPO) machine learning agent that acts as a true AI Game Master—tracking your hit accuracy, damage rates, and flow state to adjust enemy parameters on the fly. 

If you are dominating, the AI pushes the limits (faster ships, more health, quicker spawn delays). If you are struggling, the AI pulls back to give you room to breathe. 

---

## 🚀 Features

- **Fluid Space Dogfighting:** Fast-paced action requiring precise maneuvering to dodge enemy fire and asteroids. 
- **AI DDA (Dynamic Difficulty Adjustment):** A locally deployed, pre-trained neural network (MLP Policy) actively tracks 7 telemetry parameters (such as `accuracy_ratio`, `kill_rate`, `health`, etc.) every 2 seconds to scale difficulty accurately in real-time. 
- **Hardcore Rules:** No second chances for crashing! Crashing into an enemy ship or an asteroid instantly results in game over.
- **FastAPI Telemetry Backend:** A modern backend running FastAPI connected to a PostgreSQL database streams your gameplay sessions securely so that you can view your real-time analytics.
- **React Dashboard:** A beautiful React frontend dashboard to analyze your gameplay metrics across sessions, mapping out your difficulty curves visually.
- **Model Training Pipelines:** Want to train your own difficulty scaler? Simply run `train_dda_3d.py` (or the temporal coherent `train_dda_3d_lstm.py`) to run simulated headless environments using PPO.

## 🛠 Project Structure

- `game_3d.py` - The core game engine containing Pygame mechanics and live PPO Inference integration via PyTorch.
- `env_wrapper_3d.py` - The headless reinforcement learning environment that simulates a player testing boundaries for training purposes.
- `train_dda_3d.py` / `train_dda_3d_lstm.py` - Training scripts establishing standard stateful MLP and sequence memory pipelines.
- `dda_ppo_3d.pth` - The active trained PyTorch Multi-layer Perceptron logic weights determining your game.  
- `/backend_fastapi/` - Telemetry APIs linking SQL models together.
- `/dashboard/` - React SPA for analyzing database hits.
- `dda_curve_visualizer.py` / `dda_logger.py` / `dda_plot.py` - Extraneous charting tools for difficulty curving.

## ⚙️ How to Play

### 1. Requirements

Ensure you are using Python 3+ and have installed all requirements:
```bash
pip install -r requirements.txt
pip install -r backend_fastapi/requirements.txt
```

*(You will also need an instance of PostgreSQL configured on `localhost:5432` if you want to use the backend/dashboard).*

### 2. Launch the Game
```bash
python3 game_3d.py
```

### 3. Controls
- **Movement:** `W A S D` or `Arrow Keys`
- **Shoot:** `SPACE`
- **Restart:** `R`
- **Quit:** `ESC` or `Q`

Watch closely: 
- 🔴 **Red Ships** - Shoot these immediately. They shoot back with varied speeds and health intervals dictated by the AI.
- 🟤 **Brown Asteroids** - Avoid! A single asteroid hit is instant death.

### 4. Running the Dashboard (Optional)
In terminal 1: 
```bash
cd backend_fastapi
uvicorn main:app --reload
```
In terminal 2:
```bash
cd dashboard
npm install
npm start
```

## 🧠 Retraining the AI models
Want to manipulate how the agent learns?
1. Modify the rewards system located in `env_wrapper_3d.py`.
2. Run `python3 train_dda_3d.py` (Fast, momentary inference).
3. The new configurations will overwrite your `.pth` parameters, making your DDA behave fully autonomously around your new preferences!

Enjoy the dogfight!
