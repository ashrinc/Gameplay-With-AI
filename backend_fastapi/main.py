from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import time
import os

# ──────────────── FASTAPI APP ──────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost", "127.0.0.1", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────── DATABASE SETUP (PostgreSQL) ──────────────
# Connection string: postgresql://user:password@host:port/database
# For local dev: postgresql://postgres:postgres@localhost:5432/gamee_db
# Set via env: export DATABASE_URL="postgresql://..."
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ashritha@localhost:5432/gamee_db"
)

print(f"[DB] Connecting to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'local'}")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ──────────────── DATABASE TABLES ──────────────
class TelemetryDB(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)  # UUID per game session
    created_at = Column(DateTime, default=func.now())  # Timestamp
    elapsed_time_s = Column(Float)
    score = Column(Integer)
    kills = Column(Integer)
    asteroids_destroyed = Column(Integer)
    accuracy = Column(Float)
    wave = Column(Integer)
    health = Column(Float)
    shield = Column(Float)
    difficulty_score = Column(Integer)
    enemy_count = Column(Integer)
    enemy_speed = Column(Float)
    enemy_hp = Column(Float)
    enemy_fire_rate = Column(Float)


class AgentDecisionDB(Base):
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)  # Track per session
    action = Column(Integer)
    target_speed = Column(Float)
    target_spawn_interval = Column(Float)
    timestamp = Column(Float)
    created_at = Column(DateTime, default=func.now())


Base.metadata.create_all(bind=engine)

# ──────────────── SCHEMAS ──────────────
class Telemetry(BaseModel):
    session_id: Optional[str] = None  # Session identifier
    elapsed_time_s: Optional[float] = 0
    score: Optional[int] = 0
    kills: Optional[int] = 0
    asteroids_destroyed: Optional[int] = 0
    accuracy: Optional[float] = 0
    wave: Optional[int] = 0
    health: Optional[float] = 0
    shield: Optional[float] = 0
    difficulty_score: Optional[int] = 0
    enemy_count: Optional[int] = 0
    enemy_speed: Optional[float] = 0
    enemy_hp: Optional[float] = 0
    enemy_fire_rate: Optional[float] = 0


# ──────────────── ENDPOINTS ──────────────
@app.post("/telemetry")
def receive_telemetry(data: Telemetry):
    """Store telemetry for a game session."""
    db = SessionLocal()
    entry = TelemetryDB(
        session_id=data.session_id,
        elapsed_time_s=data.elapsed_time_s,
        score=data.score,
        kills=data.kills,
        asteroids_destroyed=data.asteroids_destroyed,
        accuracy=data.accuracy,
        wave=data.wave,
        health=data.health,
        shield=data.shield,
        difficulty_score=data.difficulty_score,
        enemy_count=data.enemy_count,
        enemy_speed=data.enemy_speed,
        enemy_hp=data.enemy_hp,
        enemy_fire_rate=data.enemy_fire_rate,
    )
    db.add(entry)
    db.commit()
    db.close()

    return {"status": "saved", "session_id": data.session_id}


@app.get("/telemetry/live")
def get_live(session_id: str):
    """Get latest telemetry for a specific session."""
    db = SessionLocal()
    latest = db.query(TelemetryDB)\
        .filter(TelemetryDB.session_id == session_id)\
        .order_by(TelemetryDB.created_at.desc())\
        .first()
    db.close()

    if latest:
        return {
            "session_id": latest.session_id,
            "elapsed_time_s": latest.elapsed_time_s,
            "score": latest.score,
            "kills": latest.kills,
            "asteroids_destroyed": latest.asteroids_destroyed,
            "accuracy": latest.accuracy,
            "wave": latest.wave,
            "health": latest.health,
            "shield": latest.shield,
            "difficulty_score": latest.difficulty_score,
            "enemy_count": latest.enemy_count,
            "enemy_speed": latest.enemy_speed,
            "enemy_hp": latest.enemy_hp,
            "enemy_fire_rate": latest.enemy_fire_rate,
            "created_at": latest.created_at.isoformat(),
        }
    return {"error": "No data for session"}


@app.get("/telemetry/history")
def get_history(session_id: str, limit: int = Query(50, ge=1, le=500)):
    """Get telemetry history for a session (for training/analysis)."""
    db = SessionLocal()
    records = db.query(TelemetryDB)\
        .filter(TelemetryDB.session_id == session_id)\
        .order_by(TelemetryDB.created_at.desc())\
        .limit(limit).all()
    db.close()

    return [
        {
            "id": r.id,
            "session_id": r.session_id,
            "elapsed_time_s": r.elapsed_time_s,
            "score": r.score,
            "kills": r.kills,
            "asteroids_destroyed": r.asteroids_destroyed,
            "accuracy": r.accuracy,
            "wave": r.wave,
            "health": r.health,
            "shield": r.shield,
            "difficulty_score": r.difficulty_score,
            "enemy_count": r.enemy_count,
            "enemy_speed": r.enemy_speed,
            "enemy_hp": r.enemy_hp,
            "enemy_fire_rate": r.enemy_fire_rate,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]


@app.get("/telemetry/sessions")
def get_all_sessions(limit: int = Query(50, ge=1, le=500)):
    """Get list of all active sessions (latest record per session)."""
    db = SessionLocal()
    # Get latest record per session
    from sqlalchemy import distinct
    sessions = db.query(TelemetryDB)\
        .distinct(TelemetryDB.session_id)\
        .order_by(TelemetryDB.session_id.desc(), TelemetryDB.created_at.desc())\
        .limit(limit).all()
    db.close()

    return [
        {
            "session_id": s.session_id,
            "latest_score": s.score,
            "latest_wave": s.wave,
            "latest_kills": s.kills,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sessions
    ]


@app.get("/telemetry/export")
def export_session_data(session_id: str):
    """Export all telemetry data for a session (for training)."""
    db = SessionLocal()
    records = db.query(TelemetryDB)\
        .filter(TelemetryDB.session_id == session_id)\
        .order_by(TelemetryDB.created_at.asc())\
        .all()
    db.close()

    return {
        "session_id": session_id,
        "record_count": len(records),
        "data": [
            {
                "elapsed_time_s": r.elapsed_time_s,
                "score": r.score,
                "kills": r.kills,
                "asteroids_destroyed": r.asteroids_destroyed,
                "accuracy": r.accuracy,
                "wave": r.wave,
                "health": r.health,
                "shield": r.shield,
                "difficulty_score": r.difficulty_score,
                "enemy_count": r.enemy_count,
                "enemy_speed": r.enemy_speed,
                "enemy_hp": r.enemy_hp,
                "enemy_fire_rate": r.enemy_fire_rate,
            }
            for r in records
        ]
    }


@app.post("/agent/decision")
def receive_agent_decision(data: dict):
    """Store AI agent decisions for analysis."""
    db = SessionLocal()
    entry = AgentDecisionDB(
        session_id=data.get("session_id"),
        action=data.get("action"),
        target_speed=data.get("target_speed"),
        target_spawn_interval=data.get("target_spawn_interval"),
        timestamp=data.get("timestamp", time.time()),
    )
    db.add(entry)
    db.commit()
    db.close()

    return {"status": "saved"}


@app.get("/agent/decisions")
def get_agent_decisions(session_id: str, limit: int = Query(100, ge=1, le=1000)):
    """Get agent decisions for a session (for training/analysis)."""
    db = SessionLocal()
    decisions = db.query(AgentDecisionDB)\
        .filter(AgentDecisionDB.session_id == session_id)\
        .order_by(AgentDecisionDB.created_at.desc())\
        .limit(limit).all()
    db.close()

    return [
        {
            "id": d.id,
            "action": d.action,
            "target_speed": d.target_speed,
            "target_spawn_interval": d.target_spawn_interval,
            "timestamp": d.timestamp,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in decisions
    ]


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "database": "postgresql",
        "version": "1.0"
    }
