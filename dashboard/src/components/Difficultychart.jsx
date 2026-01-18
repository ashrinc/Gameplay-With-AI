import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

export default function DifficultyChart({ data }) {
  return (
    <div>
      <h3>Difficulty Over Time</h3>

      <LineChart width={750} height={260} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />

        <Line
          type="monotone"
          dataKey="current_fall_speed_pps"
          stroke="#f87171"
          name="Fall Speed"
        />

        <Line
          type="monotone"
          dataKey="current_spawn_interval_ms"
          stroke="#4f46e5"
          name="Spawn Interval"
        />
      </LineChart>
    </div>
  );
}
