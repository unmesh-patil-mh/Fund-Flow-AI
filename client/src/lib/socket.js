import { io } from "socket.io-client";

const SOCKET_URL =
  import.meta.env.DEV
    ? "/"
    : "https://fund-flow-ai-do04.onrender.com";

const socket = io(SOCKET_URL, {
  transports: ["websocket", "polling"],
  autoConnect: true,
});

socket.on("connect", () => {
  console.log("[Socket] Connected:", socket.id);
});

socket.on("disconnect", (reason) => {
  console.log("[Socket] Disconnected:", reason);
});

export default socket;
