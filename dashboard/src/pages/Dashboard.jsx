import { useState, useEffect } from "react";
import StatCard from "../components/StatCard";
import DifficultyChart from "../components/Difficultychart";
import BlocksChart from "../components/BlocksChart";
import ReactionChart from "../components/ReactionChart";
import AgentDecisionChart from "../components/AgentDecisionChart";
import { getLiveTelemetry, getAgentDecisions, getAllSessions } from "../api/backend";

export default function Dashboard() {
  const [timeline, setTimeline] = useState([]);
  const [agentTimeline, setAgentTimeline] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch all active sessions
  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await getAllSessions(10);
        setSessions(res.data || []);
        if (res.data && res.data.length > 0) {
          setSelectedSession(res.data[0].session_id);
        }
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch sessions:", err);
        setLoading(false);
      }
    };

    fetchSessions();
    const interval = setInterval(fetchSessions, 3000); // Refresh sessions every 3s
    return () => clearInterval(interval);
  }, []);

  // Fetch telemetry and agent decisions for selected session
  useEffect(() => {
    if (!selectedSession) return;

    const interval = setInterval(async () => {
      try {
        // Fetch live telemetry
        const telemRes = await getLiveTelemetry(selectedSession);
        const telem = telemRes.data;

        if (telem && telem.session_id) {
          setTimeline((prev) => [
            ...prev.slice(-50),
            { time: prev.length + 1, ...telem },
          ]);
        }

        // Fetch agent decisions
        const agentRes = await getAgentDecisions(selectedSession, 20);
        const agentData = agentRes.data || [];

        setAgentTimeline(
          agentData.map((item, i) => ({
            time: i + 1,
            ...item,
          }))
        );
      } catch (err) {
        console.error("Backend error:", err.message);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [selectedSession]);

  const latest = timeline[timeline.length - 1] || {};

  if (loading) {
    return <div style={{ padding: 20 }}>Loading sessions...</div>;
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>🎮 Space Dogfight Dashboard</h2>

      {/* Session Selector */}
      <div style={{ marginBottom: 20, padding: 15, backgroundColor: "#f0f0f0", borderRadius: 8 }}>
        <label style={{ marginRight: 10 }}>
          <strong>Select Session:</strong>
        </label>
        <select
          value={selectedSession || ""}
          onChange={(e) => setSelectedSession(e.target.value)}
          style={{ padding: 8, borderRadius: 4, fontSize: 14 }}
        >
          {sessions.map((session) => (
            <option key={session.session_id} value={session.session_id}>
              {session.session_id.slice(0, 8)}... | Score: {session.latest_score} | Wave: {session.latest_wave}
            </option>
          ))}
        </select>
        <span style={{ marginLeft: 15, color: "#666" }}>
          Active Sessions: {sessions.length}
        </span>
      </div>

      {/* KPI cards */}
      <div style={{ display: "flex", gap: 15, marginBottom: 25, flexWrap: "wrap" }}>
        <StatCard label="Survival" value={latest.elapsed_time_s?.toFixed(1) || 0} suffix="s" />
        <StatCard label="Score" value={latest.score || 0} />
        <StatCard label="Kills" value={latest.kills || 0} />
        <StatCard
          label="Accuracy"
          value={(latest.accuracy || 0)}
          suffix="%"
        />
        <StatCard label="Wave" value={latest.wave || 0} />
        <StatCard label="Difficulty" value={latest.difficulty_score || 0} />
      </div>

      {/* Difficulty over time */}
      <DifficultyChart data={timeline} />

      <div style={{ display: "flex", gap: 25, marginTop: 25, flexWrap: "wrap" }}>
        <BlocksChart data={timeline} />
        <ReactionChart data={timeline} />
      </div>

      {/* Agent decisions */}
      <AgentDecisionChart data={agentTimeline} />
    </div>
  );
}
