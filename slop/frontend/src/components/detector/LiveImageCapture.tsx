import { Camera, Square, Monitor, ShieldAlert, Activity } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { liveImageSocketUrl } from "../../api/liveImage";
import type { DetectionResult } from "../../types";
import { scoreColor } from "../../utils/validators";

type LiveImageCaptureProps = {
  onResult: (result: DetectionResult) => void;
};

export function LiveImageCapture({ onResult }: LiveImageCaptureProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | undefined>();
  const streamRef = useRef<MediaStream | null>(null);
  const awaitingFrameRef = useRef(false);

  const [streaming, setStreaming] = useState(false);
  const [sourceType, setSourceType] = useState<"camera" | "screen">("camera");
  const [status, setStatus] = useState("Live ingestion idle");
  const [latestScore, setLatestScore] = useState<number | null>(null);
  const [fpsCount, setFpsCount] = useState<number>(0);
  const frameCounterRef = useRef<number>(0);

  const stop = () => {
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
  };

  const sendFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const socket = socketRef.current;
    if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || awaitingFrameRef.current) return;
    if (!video.videoWidth || !video.videoHeight) return;

    const targetWidth = 384;
    const targetHeight = Math.round((video.videoHeight / video.videoWidth) * targetWidth);
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0, targetWidth, targetHeight);
    awaitingFrameRef.current = true;

    canvas.toBlob(
      (blob) => {
        if (!blob || socket.readyState !== WebSocket.OPEN) {
          awaitingFrameRef.current = false;
          return;
        }
        socket.send(blob);
        frameCounterRef.current += 1;
      },
      "image/jpeg",
      0.72
    );
  };

  const start = async () => {
    setStatus(sourceType === "camera" ? "Opening optical sensor..." : "Requesting display surface...");
    try {
      let stream: MediaStream;
      if (sourceType === "screen") {
        stream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: false,
        });
      } else {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 } },
          audio: false,
        });
      }

      // Handle user stopping screen share via native browser bar
      stream.getVideoTracks()[0].onended = () => {
        stop();
      };

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const socket = new WebSocket(liveImageSocketUrl());
      socketRef.current = socket;
      socket.onopen = () => {
        setStreaming(true);
        setStatus("Real-time neural telemetry active");
        // 400ms interval = 2.5 FPS (well within the backend 5 FPS ceiling)
        timerRef.current = window.setInterval(sendFrame, 400);
      };
      socket.onmessage = (event) => {
        awaitingFrameRef.current = false;
        try {
          const payload = JSON.parse(event.data);
          if (payload.result) {
            setLatestScore(payload.result.overall_score);
            onResult(payload.result);
          }
          if (payload.error) setStatus(payload.error.replaceAll("_", " "));
        } catch {
          // ignore parsing error
        }
      };
      socket.onerror = () => {
        awaitingFrameRef.current = false;
        setStatus("Live telemetry connection error");
      };
      socket.onclose = () => {
        awaitingFrameRef.current = false;
        setStreaming(false);
      };
    } catch {
      stop();
      setStatus(sourceType === "screen" ? "Screen share cancelled" : "Camera access denied");
    }
  };

  useEffect(() => stop, []);

  return (
    <section className="input-panel live-panel" style={{ background: "var(--bg-card)", border: "1px solid var(--border-card)", borderRadius: "8px", padding: "16px", color: "var(--text-main)" }}>
      {/* Source Selector Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <div style={{ display: "flex", gap: "6px" }}>
          <button
            type="button"
            className={`btn ${sourceType === "camera" ? "btn-primary" : "btn-secondary"}`}
            disabled={streaming}
            onClick={() => setSourceType("camera")}
            style={{ fontSize: "11px", padding: "5px 10px", display: "flex", alignItems: "center", gap: "6px" }}
          >
            <Camera size={13} />
            <span>Webcam Feed</span>
          </button>
          <button
            type="button"
            className={`btn ${sourceType === "screen" ? "btn-primary" : "btn-secondary"}`}
            disabled={streaming}
            onClick={() => setSourceType("screen")}
            style={{ fontSize: "11px", padding: "5px 10px", display: "flex", alignItems: "center", gap: "6px" }}
          >
            <Monitor size={13} />
            <span>Screen / Window Share</span>
          </button>
        </div>

        <span style={{ fontSize: "11px", color: streaming ? "#10b981" : "var(--text-muted)", fontFamily: "monospace", display: "flex", alignItems: "center", gap: "6px" }}>
          {streaming && <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#10b981", boxShadow: "0 0 6px #10b981" }} />}
          {streaming ? "TELEMETRY LIVE (2.5 FPS)" : "IDLE"}
        </span>
      </div>

      {/* Video Viewport with HUD Reticle */}
      <div
        className="live-preview"
        style={{
          position: "relative",
          aspectRatio: "16/9",
          background: "var(--bg-page)",
          borderRadius: "6px",
          overflow: "hidden",
          border: "1px solid var(--border-card)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <video
          ref={videoRef}
          muted
          playsInline
          style={{ width: "100%", height: "100%", objectFit: "contain", display: streaming ? "block" : "none" }}
        />

        {!streaming && (
          <div style={{ textAlign: "center", color: "var(--text-muted)" }}>
            {sourceType === "camera" ? <Camera size={36} style={{ margin: "0 auto 8px", opacity: 0.6 }} /> : <Monitor size={36} style={{ margin: "0 auto 8px", opacity: 0.6 }} />}
            <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-main)" }}>{status}</div>
            <div style={{ fontSize: "11px", marginTop: "4px" }}>Click Start Live to begin real-time stream inspection</div>
          </div>
        )}

        {/* Augmented Cyber HUD Reticle */}
        {streaming && (
          <div
            className="hud-overlay"
            style={{
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              padding: "12px",
            }}
          >
            {/* Top row: source badge + live confidence radar */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div style={{ background: "rgba(7, 9, 14, 0.8)", border: "1px solid #1e293b", padding: "4px 8px", borderRadius: "4px", fontSize: "10px", color: "var(--accent)", fontFamily: "monospace" }}>
                SOURCE: {sourceType.toUpperCase()} | ViT-SWIN + PRNU
              </div>

              {latestScore !== null && (
                <div
                  style={{
                    background: "rgba(7, 9, 14, 0.85)",
                    border: `1px solid ${scoreColor(latestScore)}`,
                    padding: "4px 10px",
                    borderRadius: "4px",
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                  }}
                >
                  <Activity size={12} style={{ color: scoreColor(latestScore) }} />
                  <span style={{ fontSize: "12px", fontWeight: 800, color: scoreColor(latestScore), fontFamily: "monospace" }}>
                    {Math.round(latestScore * 100)}% AI
                  </span>
                </div>
              )}
            </div>

            {/* Corner Crosshairs */}
            <div style={{ position: "absolute", top: 12, left: 12, width: 14, height: 14, borderTop: "2px solid var(--accent)", borderLeft: "2px solid var(--accent)" }} />
            <div style={{ position: "absolute", top: 12, right: 12, width: 14, height: 14, borderTop: "2px solid var(--accent)", borderRight: "2px solid var(--accent)" }} />
            <div style={{ position: "absolute", bottom: 12, left: 12, width: 14, height: 14, borderBottom: "2px solid var(--accent)", borderLeft: "2px solid var(--accent)" }} />
            <div style={{ position: "absolute", bottom: 12, right: 12, width: 14, height: 14, borderBottom: "2px solid var(--accent)", borderRight: "2px solid var(--accent)" }} />

            {/* Bottom Status Ticker */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
              <div style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "monospace", background: "rgba(7, 9, 14, 0.8)", padding: "2px 6px", borderRadius: "2px" }}>
                BUFFER: 384x216 | 64KB PKT
              </div>
            </div>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} hidden />

      {/* Action Footer */}
      <div className="panel-actions" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "12px" }}>
        <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>{status}</span>
        <button
          className={streaming ? "btn btn-danger" : "btn btn-primary"}
          type="button"
          onClick={streaming ? stop : start}
          style={{ display: "flex", alignItems: "center", gap: "6px" }}
        >
          {streaming ? <Square size={15} /> : <Camera size={15} />}
          {streaming ? "Stop Ingestion" : "Start Live Inspection"}
        </button>
      </div>
    </section>
  );
}
