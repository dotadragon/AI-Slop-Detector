import { useEffect, useState } from "react";

export function useJobSocket(jobId?: string) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!jobId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/api/v1/jobs/ws/${jobId}`);
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setProgress(payload.progress ?? 0);
      } catch {
        setProgress(0);
      }
    };
    return () => {
      socket.close();
    };
  }, [jobId]);

  return progress;
}
