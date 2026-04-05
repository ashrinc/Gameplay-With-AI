import axios from "axios";

const BACKEND_URL = "http://localhost:8000";

export const getLiveTelemetry = (sessionId) =>
  axios.get(`${BACKEND_URL}/telemetry/live?session_id=${sessionId}`);

export const getTelemetryHistory = (sessionId, limit = 50) =>
  axios.get(`${BACKEND_URL}/telemetry/history?session_id=${sessionId}&limit=${limit}`);

export const getAllSessions = (limit = 50) =>
  axios.get(`${BACKEND_URL}/telemetry/sessions?limit=${limit}`);

export const getAgentDecisions = (sessionId, limit = 100) =>
  axios.get(`${BACKEND_URL}/agent/decisions?session_id=${sessionId}&limit=${limit}`);

export const exportSessionData = (sessionId) =>
  axios.get(`${BACKEND_URL}/telemetry/export?session_id=${sessionId}`);

export const healthCheck = () =>
  axios.get(`${BACKEND_URL}/health`);
