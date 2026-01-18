from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import time

# ------------------ FASTAPI APP ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gameplay-with-ai-ralm.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ DATABASE SETUP ------------------
DATABASE_URL = "sqlite:///./reinblock.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


# ------------------ DATABASE TABLES ------------------
class TelemetryDB(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True)
    elapsed_time_s = Column(Float)
    score = Column(Integer)
    accuracy = Column(Float)
    mistakes = Column(Integer)
    current_fall_speed_pps = Column(Float)
    current_spawn_interval_ms = Column(Float)


class AgentDecisionDB(Base):
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(Integer)
    target_speed = Column(Float)
    target_spawn_interval = Column(Float)
    timestamp = Column(Float)


Base.metadata.create_all(bind=engine)


# ------------------ SCHEMAS ------------------
class Telemetry(BaseModel):
    elapsed_time_s: Optional[float] = 0
    score: Optional[int] = 0
    blocks_spawned: Optional[int] = 0
    blocks_avoided: Optional[int] = 0
    blocks_collided: Optional[int] = 0
    reaction_time_latest_ms: Optional[float] = 0
    reaction_time_moving_avg_ms: Optional[float] = 0
    accuracy: Optional[float] = 0
    mistakes: Optional[int] = 0
    current_fall_speed_pps: Optional[float] = 0
    current_spawn_interval_ms: Optional[float] = 0


# ------------------ IN-MEMORY FOR DASHBOARD ------------------
latest_telemetry: Telemetry = Telemetry()
agent_decisions: List[dict] = []


# ------------------ ENDPOINTS ------------------
@app.post("/telemetry")
def receive_telemetry(data: Telemetry):
    global latest_telemetry
    latest_telemetry = data

    # Save to database
    db = SessionLocal()
    entry = TelemetryDB(
        elapsed_time_s=data.elapsed_time_s,
        score=data.score,
        accuracy=data.accuracy,
        mistakes=data.mistakes,
        current_fall_speed_pps=data.current_fall_speed_pps,
        current_spawn_interval_ms=data.current_spawn_interval_ms,
    )
    db.add(entry)
    db.commit()
    db.close()

    return {"status": "saved"}


@app.get("/telemetry/live")
def get_live():
    return latest_telemetry


@app.post("/agent/decision")
def receive_agent_decision(data: dict):
    agent_decisions.append(data)

    if len(agent_decisions) > 100:
        agent_decisions.pop(0)

    # Save to database
    db = SessionLocal()
    entry = AgentDecisionDB(
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
def get_agent_decisions():
    return agent_decisions
