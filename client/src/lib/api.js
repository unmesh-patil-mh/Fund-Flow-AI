import axios from "axios";

const API_URL = import.meta.env.DEV
  ? "/api"
  : "https://fund-flow-ai-do04.onrender.com/api";

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
    "X-Demo-Mode": "true",
  },
  timeout: 60000,
});

// Response interceptor for consistent error handling
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.message || error.message || "Request failed";
    console.error(`API Error: ${message}`);
    return Promise.reject({ message, status: error.response?.status });
  }
);

export default api;
