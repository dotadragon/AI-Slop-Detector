import { Camera, Square, Monitor } from "lucide-react";
import type { DetectionResult } from "../../types";
import { scoreColor } from "../../utils/validators";
import { useLiveWebcamStream, type SourceType } from "../../hooks/useLiveWebcamStream";
import { Activity } from "lucide-react";

type LiveImageCaptureProps = {
  onResult: (result: DetectionResult) => void;
};

function LiveSourceSelector({
  sourceType,
  setSourceType,
  streaming,
}: {
  sourceType: SourceType;
  setSourceType: (s: SourceType) => void;
  streaming: boolean;
}) {
  return (
    <div className="live-capture-header">
      <div className="live-capture-source-group">
        <button
          type="button"
          className={`btn ${sourceType === "camera" ? "btn-primary" : "btn-secondary"} live-capture-source-btn`}
          disabled={streaming}
          onClick={() => setSourceType("camera")}
        >
          <Camera size={13} />
          <span>Webcam Feed</span>
        </button>
        <button
          type="button"
          className={`btn ${sourceType === "screen" ? "btn-primary" : "btn-secondary"} live-capture-source-btn`}
          disabled={streaming}
          onClick={() => setSourceType("screen")}
        >
          <Monitor size={13} />
          <span>Screen / Window Share</span>
        </button>
      </div>

      <span className="live-capture-status-badge" style={{ color: streaming ? "#10b981" : "var(--text-muted)" }}>
        {streaming && <span className="live-capture-dot" />}
        {streaming ? "TELEMETRY LIVE (2.5 FPS)" : "IDLE"}
      </span>
    </div>
  );
}

function LiveHudOverlay({
  sourceType,
  latestScore,
}: {
  sourceType: SourceType;
  latestScore: number | null;
}) {
  const toneColor = latestScore !== null ? scoreColor(latestScore) : undefined;

  return (
    <div className="live-capture-hud-overlay">
      <div className="live-capture-hud-top">
        <div className="live-capture-source-tag">
          SOURCE: {sourceType.toUpperCase()} | ViT-SWIN + PRNU
        </div>

        {latestScore !== null && toneColor && (
          <div className="live-capture-score-chip" style={{ border: `1px solid ${toneColor}` }}>
            <Activity size={12} style={{ color: toneColor }} />
            <span style={{ fontSize: "12px", fontWeight: 800, color: toneColor, fontFamily: "monospace" }}>
              {Math.round(latestScore * 100)}% AI
            </span>
          </div>
        )}
      </div>

      <div className="live-capture-corner-tl" />
      <div className="live-capture-corner-tr" />
      <div className="live-capture-corner-bl" />
      <div className="live-capture-corner-br" />

      <div className="live-capture-hud-bottom">
        <div className="live-capture-buffer-tag">BUFFER: 384x216 | 64KB PKT</div>
      </div>
    </div>
  );
}

export function LiveImageCapture({ onResult }: LiveImageCaptureProps) {
  const {
    videoRef,
    canvasRef,
    streaming,
    sourceType,
    setSourceType,
    status,
    latestScore,
    start,
    stop,
  } = useLiveWebcamStream(onResult);

  return (
    <section className="input-panel live-panel live-capture-panel">
      <LiveSourceSelector sourceType={sourceType} setSourceType={setSourceType} streaming={streaming} />

      <div className="live-preview live-capture-viewport">
        <video
          ref={videoRef}
          muted
          playsInline
          className="live-capture-video"
          style={{ display: streaming ? "block" : "none" }}
        />

        {!streaming && (
          <div className="live-capture-idle-prompt">
            {sourceType === "camera" ? <Camera size={36} style={{ margin: "0 auto 8px", opacity: 0.6 }} /> : <Monitor size={36} style={{ margin: "0 auto 8px", opacity: 0.6 }} />}
            <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-main)" }}>{status}</div>
            <div style={{ fontSize: "11px", marginTop: "4px" }}>Click Start Live to begin real-time stream inspection</div>
          </div>
        )}

        {streaming && <LiveHudOverlay sourceType={sourceType} latestScore={latestScore} />}
      </div>

      <canvas ref={canvasRef} hidden />

      <div className="panel-actions live-capture-actions">
        <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>{status}</span>
        <button
          className={streaming ? "btn btn-danger" : "btn btn-primary"}
          type="button"
          onClick={streaming ? stop : start}
          style={{ display: "flex", alignItems: "center", gap: "6px" }}
        >
          {streaming ? <Square size={15} /> : <Camera size={15} />}
          <span>{streaming ? "Stop Ingestion" : "Start Live Inspection"}</span>
        </button>
      </div>
    </section>
  );
}
