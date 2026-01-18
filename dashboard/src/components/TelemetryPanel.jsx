export default function TelemetryPanel({ stats }) {
  return (
    <div style={{ marginTop: "20px" }}>
      <h3>Player Telemetry</h3>

      <div style={{ display: "flex", gap: "25px" }}>
        <div>
          <strong>Survival:</strong> {stats.survival}s
        </div>
        <div>
          <strong>Mistakes:</strong> {stats.mistakes}
        </div>
        <div>
          <strong>Reaction:</strong> {stats.reaction} ms
        </div>
      </div>
    </div>
  );
}

