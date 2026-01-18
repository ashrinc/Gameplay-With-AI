
# Rein_Block 🎮🧠  

# 🎮 Gameplay With AI — Reinforcement Learning Driven Adaptive Game

A full-stack AI project where a Reinforcement Learning agent dynamically adjusts game difficulty based on player performance.  
Includes a live Pygame game, FastAPI backend, and React analytics dashboard.

🔗 **Live Dashboard:**  
https://gameplay-with-ai-ralm.vercel.app

🔗 **Live Backend API:**  
https://gameplay-ai-backend.onrender.com

---

## 📌 Features

- 🎮 Pygame arcade-style dodging game  
- 🧠 Reinforcement Learning (PPO) controls difficulty dynamically  
- 📡 Real-time telemetry streaming to backend  
- ⚙️ FastAPI backend with REST endpoints  
- 📊 React dashboard with live visualizations  
- 🌐 Fully deployed (Vercel + Render)  

---

## 🧱 Architecture

```

Local Machine:

* Game (Pygame + RL Agent)

Cloud:

* Backend (FastAPI on Render)
* Dashboard (React on Vercel)

Game → sends telemetry → Backend
Dashboard → fetches live data → Backend

```

---

## 📁 Project Structure

```

gamee/
├── game.py
├── train_dd_agent.py
├── dda_ppo.pth
├── backend_fastapi/
│   ├── main.py
│   └── requirements.txt
├── dashboard/
│   ├── package.json
│   └── src/

````

---

## ⚙️ Local Setup (Run Game on Your Machine)

### 1. Clone the repository

```bash
git clone https://github.com/ashrinc/Gameplay-With-AI.git
cd Gameplay-With-AI
````

---

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```

On Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install Python dependencies

```bash
pip install -r backend_fastapi/requirements.txt
```

If pygame is missing:

```bash
pip install pygame requests torch numpy
```

---

### 4. Run the game

```bash
python game.py
```

Now play the game using:

* ⬅️ Left Arrow
* ➡️ Right Arrow

The game will automatically:

* Send telemetry to the deployed backend
* Update the deployed dashboard in real time

---

## 🌐 Viewing Live Data (No setup required)

You don’t need to run backend or frontend locally.

Just open:

### 📊 Live Dashboard

```
https://gameplay-with-ai-ralm.vercel.app
```

### 📡 Backend API

```
https://gameplay-ai-backend.onrender.com/telemetry/live
https://gameplay-ai-backend.onrender.com/agent/decisions
```

When the game is running, these endpoints update live.

---

## 🧠 How the AI Works

* The game tracks player performance:

  * Accuracy
  * Mistakes
  * Survival time
  * Block collisions
* These metrics form the **state**
* A trained PPO agent chooses actions like:

  * Increase difficulty
  * Decrease difficulty
  * Keep difficulty stable
* This creates a **Dynamic Difficulty Adjustment (DDA)** system used in real-world games

---

## 🛠 Tech Stack

| Layer      | Technology                          |
| ---------- | ----------------------------------- |
| Game       | Python, Pygame                      |
| AI         | PyTorch, PPO (RL)                   |
| Backend    | FastAPI                             |
| Frontend   | React.js                            |
| Charts     | Recharts                            |
| Deployment | Render (backend), Vercel (frontend) |

---

## 🎯 Use Cases

* AI-powered adaptive systems
* Game analytics platforms
* Human-in-the-loop learning demos
* RL + Full Stack portfolio project
* Interview-ready project (AI + Backend + Frontend)

---

## 👩‍💻 Author

**Ashritha**
Built as an advanced AI + Full Stack portfolio project.

---

## ⭐ If you like this project

Star the repo and feel free to fork and extend it.



