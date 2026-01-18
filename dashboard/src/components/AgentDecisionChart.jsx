import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export default function AgentDecisionChart({ data }) {
  return (
    <div style={{ marginTop: "40px" }}>
      <h2>AI Difficulty Decisions</h2>

      <LineChart width={700} height={300} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />

        <Line type="monotone" dataKey="target_speed" stroke="#f97316" />
        <Line type="monotone" dataKey="target_spawn_interval" stroke="#7c3aed" />
      </LineChart>
    </div>
  );
}
