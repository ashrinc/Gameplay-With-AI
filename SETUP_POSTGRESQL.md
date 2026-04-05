# Space Dogfight - PostgreSQL + Multi-User Setup Guide

## 🎯 What's New

### ✅ LSTM Memory AI
- **File**: `train_dda_3d_lstm.py` (NEW)
- **Model**: StatefulLSTMPolicy3D with 128-hidden LSTM layer
- **Memory**: Remembers 30 seconds of gameplay (15 decisions)
- **Benefit**: Detects frustration patterns faster, adapts better

### ✅ PostgreSQL Multi-User Backend
- **Database**: PostgreSQL (instead of SQLite)
- **Multi-user**: Session-based telemetry isolation
- **Session ID**: UUID auto-generated per game
- **Training data**: All telemetry stored for AI training later

### ✅ Enhanced Game
- Sends session_id with every telemetry post
- Stores: score, kills, accuracy, wave, health, shield, difficulty_score, **enemy params**
- Can play 2+ concurrent games, no data overlap

---

## 📋 Setup Instructions

### Step 1: Install PostgreSQL

**macOS (using Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Check if running:**
```bash
psql --version
```

---

### Step 2: Create Database

```bash
# Connect to PostgreSQL default database
psql postgres

# In PostgreSQL shell:
CREATE DATABASE gamee_db;
CREATE USER gamee_user WITH PASSWORD 'gamee_password_123';
ALTER ROLE gamee_user SET client_encoding TO 'utf8';
ALTER ROLE gamee_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE gamee_user SET default_transaction_deferrable TO on;
ALTER ROLE gamee_user SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE gamee_db TO gamee_user;
\q
```

---

### Step 3: Set Environment Variable

**Option A: For this session only:**
```bash
export DATABASE_URL="postgresql://gamee_user:gamee_password_123@localhost:5432/gamee_db"
```

**Option B: Permanent (add to ~/.zshrc or ~/.bash_profile):**
```bash
echo 'export DATABASE_URL="postgresql://gamee_user:gamee_password_123@localhost:5432/gamee_db"' >> ~/.zshrc
source ~/.zshrc
```

---

### Step 4: Start Backend

```bash
cd backend_fastapi
export DATABASE_URL="postgresql://gamee_user:gamee_password_123@localhost:5432/gamee_db"
pip install -r requirements.txt  # Already has psycopg2-binary
uvicorn main:app --reload --port 8000
```

**Expected output:**
```
[DB] Connecting to: localhost:5432/gamee_db
Uvicorn running on http://127.0.0.1:8000
```

---

### Step 5: Train LSTM Model (Optional but Recommended)

**Terminal 2:**
```bash
cd /Users/ashritha/Documents/gamee
source venv/bin/activate
python3 train_dda_3d_lstm.py
```

This trains on CPU/GPU for ~15-30 minutes. Creates `dda_ppo_3d_lstm.pth` when done.

---

### Step 6: Play Game

**Terminal 3:**
```bash
cd /Users/ashritha/Documents/gamee
source venv/bin/activate
python3 game_3d.py
```

**Expected output:**
```
🎮 Game Session: a1b2c3d4-e5f6-7890-abcd-ef1234567890
📊 Model: LSTM
✓ DDA Agent loaded! (LSTM with Memory)
```

---

## 🔍 Testing Multi-User

### Test 1: Verify Session Isolation

**Terminal 1 (Game A):**
```bash
source venv/bin/activate && python3 game_3d.py
# Play for 10 seconds, note session ID
# Press Q to quit
```

**Terminal 2 (Game B):** (While Game A is running)
```bash
source venv/bin/activate && python3 game_3d.py
# Different session ID
# Play simultaneously
```

### Test 2: Query Specific Sessions

```bash
# Get latest data for Session A
curl "http://localhost:8000/telemetry/live?session_id=YOUR_SESSION_ID_A"

# Get latest data for Session B
curl "http://localhost:8000/telemetry/live?session_id=YOUR_SESSION_ID_B"

# Get history for Session A (last 10 records)
curl "http://localhost:8000/telemetry/history?session_id=YOUR_SESSION_ID_A&limit=10"

# Get all active sessions
curl "http://localhost:8000/telemetry/sessions"

# Export full session data (for training)
curl "http://localhost:8000/telemetry/export?session_id=YOUR_SESSION_ID_A"
```

---

## 📊 Backend Endpoints

### Telemetry
- `POST /telemetry` - Send game data
- `GET /telemetry/live?session_id=UUID` - Get latest data for session
- `GET /telemetry/history?session_id=UUID&limit=50` - Get session history
- `GET /telemetry/sessions?limit=50` - Get all active sessions
- `GET /telemetry/export?session_id=UUID` - Export for training

### AI Decisions
- `POST /agent/decision` - Store agent adjustment
- `GET /agent/decisions?session_id=UUID&limit=100` - Get decisions for session

### Health
- `GET /health` - Check backend status

---

## 🗄️ Database Schema

### `telemetry` table
- `id` (Primary Key)
- `session_id` (UUID, indexed)
- `created_at` (Timestamp)
- `elapsed_time_s` - Game runtime
- `score` - Player score
- `kills` - Enemies killed
- `asteroids_destroyed` - Asteroid hits avoided
- `accuracy` - Shot accuracy %
- `wave` - Current wave
- `health` - Player HP
- `shield` - Player shield
- `difficulty_score` - Composite difficulty
- `enemy_count` - Active enemies
- `enemy_speed` - Enemy speed px/sec
- `enemy_hp` - Enemy HP
- `enemy_fire_rate` - Enemy fire frequency

### `agent_decisions` table
- `id` (Primary Key)
- `session_id` (UUID, indexed)
- `action` (0-4: HOLD, HARDER+, EASIER-, HARDER++, EASIER--)
- `timestamp` - When decision was made
- `created_at` (Timestamp)

---

## 📁 File Changes Summary

| File | Change | Purpose |
|------|--------|---------|
| `train_dda_3d_lstm.py` | **NEW** | Train LSTM model with memory |
| `game_3d.py` | Updated | LSTM loading, session_id generation, telemetry |
| `backend_fastapi/main.py` | Complete rewrite | PostgreSQL, multi-user, new endpoints |
| `env_wrapper_3d.py` | (No changes needed) | Existing training env works as-is |

---

## 🚀 Usage Workflow

##1. **First Time Setup**
```bash
# 1. Create PostgreSQL database (one-time)
brew services start postgresql@15
psql postgres < setup.sql  # Run CREATE DATABASE commands

# 2. Train LSTM model (optional, takes 30 min)
python3 train_dda_3d_lstm.py  # Creates dda_ppo_3d_lstm.pth

# 3. Start backend (Terminal 1, persistent)
cd backend_fastapi && uvicorn main:app --reload

# 4. Play game (Terminal 2+)
python3 game_3d.py
```

## 2. **Subsequent Sessions**
```bash
# Terminal 1: Backend
cd backend_fastapi && uvicorn main:app --reload

# Terminal 2: Game
python3 game_3d.py

# Terminal 3: Query data
curl "http://localhost:8000/telemetry/live?session_id=abc-123"
```

## 3. **Use Data for Training**
```python
# Python script to fetch & train on real gameplay data
import requests

session_id = "abc-123"
response = requests.get(f"http://localhost:8000/telemetry/export?session_id={session_id}")
gameplay_data = response.json()["data"]

# Feed into new training with real player patterns
for record in gameplay_data:
    # Use record["accuracy"], record["health"], etc to train
    pass
```

---

## 🐛 Troubleshooting

**Q: Backend says "could not connect to server"**
```bash
# Check PostgreSQL is running
brew services list | grep postgresql

# Restart if needed
brew services restart postgresql@15
```

**Q: "ERROR: database gamee_db does not exist"**
```bash
# Create database again
psql postgres
CREATE DATABASE gamee_db;
```

**Q: Two games show same data**
```bash
# Check session IDs are different
# Should print different UUID each game start
# If same, check game_3d.py session_id generation
```

**Q: Backend won't start on port 8000**
```bash
# Kill process using port
lsof -ti:8000 | xargs kill -9
# Then restart backend
```

---

## ✨ Next Steps

1. ✅ Train LSTM model and verify faster adjustments
2. ✅ Play with 2+ concurrent sessions
3. ✅ Use exported gameplay data to train improved DDA
4. 🔜 Add frontend dashboard for multi-session monitoring
5. 🔜 Implement replay system using stored telemetry

---

## 📞 Quick Commands Reference

```bash
# Start everything
Terminal 1: brew services start postgresql@15 && cd backend_fastapi && uvicorn main:app
Terminal 2: cd /Users/ashritha/Documents/gamee && python3 game_3d.py
Terminal 3: cd /Users/ashritha/Documents/gamee && python3 game_3d.py  # Another game

# Query data
curl "http://localhost:8000/health"
curl "http://localhost:8000/telemetry/sessions"
curl "http://localhost:8000/telemetry/live?session_id=YOUR_ID"

# Train LSTM
python3 train_dda_3d_lstm.py

# Check database
psql gamee_db -U gamee_user
SELECT COUNT(*) FROM telemetry;
SELECT DISTINCT session_id FROM telemetry;
```
