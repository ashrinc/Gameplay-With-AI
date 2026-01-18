import { AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export default function BlocksChart({ data }) {
  return (
    <div>
      <h3>Blocks Performance</h3>

      <AreaChart width={360} height={250} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />

        <Area dataKey="blocks_spawned" fill="#a5b4fc" stroke="#6366f1" />
        <Area dataKey="blocks_avoided" fill="#86efac" stroke="#22c55e" />
        <Area dataKey="blocks_collided" fill="#fecaca" stroke="#ef4444" />
      </AreaChart>
    </div>
  );
}
