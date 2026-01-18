import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export default function ReactionChart({ data }) {
  return (
    <div>
      <h3>Reaction Time Trend</h3>

      <LineChart width={360} height={250} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />

        <Line
          dataKey="reaction_time_latest_ms"
          stroke="#10b981"
          name="Latest"
        />

        <Line
          dataKey="reaction_time_moving_avg_ms"
          stroke="#2563eb"
          name="Moving Avg"
        />
      </LineChart>
    </div>
  );
}
