import axios from "axios";

export const getLiveTelemetry = () =>
  axios.get("http://127.0.0.1:8000/telemetry/live");

export const getAgentDecisions = () =>
  axios.get("http://127.0.0.1:8000/agent/decisions");
