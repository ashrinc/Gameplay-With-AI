import axios from "axios";

export const getLiveTelemetry = () =>
  axios.get("https://gameplay-ai-backend.onrender.com/telemetry/live");

export const getAgentDecisions = () =>
  axios.get("https://gameplay-ai-backend.onrender.com/agent/decisions");
