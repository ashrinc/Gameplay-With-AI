import { useState, useEffect } from "react";
import StatCard from "../components/StatCard";
import DifficultyChart from "../components/Difficultychart";
import BlocksChart from "../components/BlocksChart";
import ReactionChart from "../components/ReactionChart";
import AgentDecisionChart from "../components/AgentDecisionChart";
import { getLiveTelemetry, getAgentDecisions } from "../api/backend";

export default function Dashboard() {
  const [timeline, setTimeline] = useState([]);
  const [agentTimeline, setAgentTimeline] = useState([]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        // fetch live telemetry
        const telemRes = await getLiveTelemetry();
        const telem = telemRes.data;

        setTimeline((prev) => [
          ...prev.slice(-50),
          { time: prev.length + 1, ...telem },
        ]);

        // fetch agent decisions
        const agentRes = await getAgentDecisions();
        const agentData = agentRes.data;

        setAgentTimeline(
          agentData.map((item, i) => ({
            time: i + 1,
            ...item,
          }))
        );
      } catch (err) {
        console.error("Backend not reachable yet");
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const latest = timeline[timeline.length - 1] || {};

  return (
    <div style={{ padding: 20 }}>
      <h2>Rein_Block AI Dashboard</h2>

      {/* KPI cards */}
      <div style={{ display: "flex", gap: 15, marginBottom: 25 }}>
        <StatCard label="Survival" value={latest.elapsed_time_s || 0} suffix="s" />
        <StatCard label="Score" value={latest.score || 0} />
        <StatCard
          label="Accuracy"
          value={((latest.accuracy || 0) * 100).toFixed(1)}
          suffix="%"
        />
        <StatCard label="Mistakes" value={latest.mistakes || 0} />
      </div>

      {/* Difficulty over time */}
      <DifficultyChart data={timeline} />

      <div style={{ display: "flex", gap: 25, marginTop: 25 }}>
        <BlocksChart data={timeline} />
        <ReactionChart data={timeline} />
      </div>

      {/* NEW: Agent decisions */}
      <AgentDecisionChart data={agentTimeline} />
    </div>
  );
}
