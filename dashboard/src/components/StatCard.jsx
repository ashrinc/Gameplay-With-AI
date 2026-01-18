export default function StatCard({ label, value, suffix }) {
  return (
    <div
      style={{
        padding: 15,
        borderRadius: 10,
        background: "#f4f4ff",
        minWidth: 140,
      }}
    >
      <div style={{ fontSize: 12, color: "#666" }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700 }}>
        {value} {suffix || ""}
      </div>
    </div>
  );
}
