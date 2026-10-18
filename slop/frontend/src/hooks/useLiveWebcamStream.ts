import { useEffect, useRef, useState, useCallback } from "react";
import { liveImageSocketUrl } from "../api/liveImage";
import type { DetectionResult } from "../types";

export type SourceType = "camera" | "screen";

function captureVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  socket: WebSocket,
  onSent: () => void
) {
  const targetWidth = 384;
  const targetHeight = Math.round((video.videoHeight / video.videoWidth) * targetWidth);
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  canvas.getContext("2d")?.drawImage(video, 0, 0, targetWidth, targetHeight);

  canvas.toBlob(
    (blob) => {
      if (blob && socket.readyState === WebSocket.OPEN) {
        socket.send(blob);
      }
      onSent();
    },
    "image/jpeg",
    0.72
  );
}

function parseSocketPayload(
  data: string,
  onScore: (score: number) => void,
  onResult: (result: DetectionResult) => void,
  onStatus: (status: string) => void
) {
  try {
    const payload = JSON.parse(data);
    if (payload.result) {
      onScore(payload.result.overall_score);
      onResult(payload.result);
    }
    if (payload.error) {
      onStatus(payload.error.replaceAll("_", " "));
    }
  } catch {
    // Ignore parse errors
  }
}

function createTelemetrySocket(
  onScore: (score: number) => void,
  onResult: (result: DetectionResult) => void,
  onStatus: (status: string) => void,
  onStreaming: (active: boolean) => void,
  onOpenCallback: () => void
): WebSocket {
  const socket = new WebSocket(liveImageSocketUrl());
  socket.onopen = () => {
    onStreaming(true);
    onStatus("Real-time neural telemetry active");
    onOpenCallback();
  };
  socket.onmessage = (event) => parseSocketPayload(event.data, onScore, onResult, onStatus);
  socket.onerror = () => onStatus("Live telemetry connection error");
  socket.onclose = () => onStreaming(false);
  return socket;
}

async function acquireMediaStream(sourceType: SourceType): Promise<MediaStream> {
  if (sourceType === "screen") {
    return navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
  }
  return navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment", width: { ideal: 1280 } },
    audio: false,
  });
}

function trySendFrame(
  video: HTMLVideoElement | null,
  canvas: HTMLCanvasElement | null,
  socket: WebSocket | null,
  awaitingRef: React.MutableRefObject<boolean>
) {
  if (!video || !canvas || !socket) return;
  if (socket.readyState !== WebSocket.OPEN || awaitingRef.current) return;
  if (!video.videoWidth || !video.videoHeight) return;

  awaitingRef.current = true;
  captureVideoFrame(video, canvas, socket, () => {
    awaitingRef.current = false;
  });
}

export function useLiveWebcamStream(onResult: (result: DetectionResult) => void) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | undefined>();
  const streamRef = useRef<MediaStream | null>(null);
  const awaitingFrameRef = useRef(false);

  const [streaming, setStreaming] = useState(false);
  const [sourceType, setSourceType] = useState<SourceType>("camera");
  const [status, setStatus] = useState("Live ingestion idle");
  const [latestScore, setLatestScore] = useState<number | null>(null);

  const stop = useCallback(() => {
    window.clearInterval(timerRef.current);
    timerRef.current = undefined;
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    awaitingFrameRef.current = false;
    setStreaming(false);
    setStatus("Live ingestion idle");
    setLatestScore(null);
  }, []);

  const sendFrame = useCallback(() => {
    trySendFrame(videoRef.current, canvasRef.current, socketRef.current, awaitingFrameRef);
  }, []);

  const start = async () => {
    setStatus(sourceType === "camera" ? "Opening optical sensor..." : "Requesting display surface...");
    try {
      const stream = await acquireMediaStream(sourceType);
      stream.getVideoTracks()[0].onended = stop;
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      socketRef.current = createTelemetrySocket(
        setLatestScore,
        onResult,
        setStatus,
        setStreaming,
        () => {
          timerRef.current = window.setInterval(sendFrame, 400);
        }
      );
    } catch {
      stop();
      setStatus(sourceType === "screen" ? "Screen share cancelled" : "Camera access denied");
    }
  };

  useEffect(() => stop, [stop]);

  return {
    videoRef,
    canvasRef,
    streaming,
    sourceType,
    setSourceType,
    status,
    latestScore,
    start,
    stop,
  };
}
