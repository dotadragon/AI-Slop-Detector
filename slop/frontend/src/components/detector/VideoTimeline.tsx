import { useState } from "react";
import { AlertTriangle, ChevronLeft, ChevronRight, Zap, Eye, Film } from "lucide-react";
import type { DetectionResult } from "../../types";
import { scoreColor } from "../../utils/validators";

export function VideoTimeline({ result }: { result?: DetectionResult }) {
  const frames = result?.frame_scores || [];
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number>(() => {
    if (!frames.length) return 0;
    // Default to the frame with the highest synthetic anomaly score
    let maxIdx = 0;
    let maxS = -1;
    frames.forEach((f, i) => {
      if (f.score > maxS) {
        maxS = f.score;
        maxIdx = i;
      }
    });
    return maxIdx;
  });

  if (!frames.length) return null;

  const activeIdx = hoveredIdx !== null ? hoveredIdx : selectedIdx;
  const activeFrame = frames[activeIdx] || frames[0];
  const maxScore = Math.max(...frames.map((f) => f.score));
  const minScore = Math.min(...frames.map((f) => f.score));
  const anomalyCount = frames.filter((f) => f.score >= 0.70).length;
  const temporalJitter = result?.signals?.temporal_jitter as number | undefined;

  const jumpToMaxAnomaly = () => {
    let maxI = 0;
    let maxVal = -1;
    frames.forEach((f, i) => {
      if (f.score > maxVal) {
        maxVal = f.score;
        maxI = i;
      }
    });
    setSelectedIdx(maxI);
    setHoveredIdx(null);
  };

  const handlePrev = () => {
    setSelectedIdx((prev) => Math.max(0, prev - 1));
    setHoveredIdx(null);
  };

  const handleNext = () => {
    setSelectedIdx((prev) => Math.min(frames.length - 1, prev + 1));
    setHoveredIdx(null);
  };

  return (
    <div className="video-timeline-container" style={{ marginTop: "16px", background: "var(--bg-card)", border: "1px solid var(--border-card)", borderRadius: "8px", padding: "16px", color: "var(--text-main)" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px", flexWrap: "wrap", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Film size={16} className="text-cyan-400" />
          <span style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-main)" }}>
            Temporal Anomaly Scrubber
          </span>
          {anomalyCount > 0 && (
            <span style={{ fontSize: "10px", fontWeight: 700, padding: "2px 6px", borderRadius: "4px", background: "rgba(244, 63, 94, 0.15)", color: "#f43f5e", border: "1px solid rgba(244, 63, 94, 0.3)" }}>
              {anomalyCount} {anomalyCount === 1 ? "Spike" : "Spikes"} Detected
            </span>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={jumpToMaxAnomaly}
            style={{ fontSize: "11px", padding: "4px 8px", display: "flex", alignItems: "center", gap: "4px" }}
            title="Jump to highest synthetic anomaly frame"
          >
            <Zap size={12} className="text-amber-400" />
            <span>Peak Anomaly</span>
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={selectedIdx <= 0}
            onClick={handlePrev}
            style={{ padding: "4px 6px" }}
            aria-label="Previous frame"
          >
            <ChevronLeft size={14} />
          </button>
          <span style={{ fontSize: "11px", color: "#64748b", fontFamily: "monospace", minWidth: "48px", textAlign: "center" }}>
            {selectedIdx + 1} / {frames.length}
          </span>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={selectedIdx >= frames.length - 1}
            onClick={handleNext}
            style={{ padding: "4px 6px" }}
            aria-label="Next frame"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>

      {/* Interactive Bar Timeline */}
      <div
        className="timeline"
        style={{
          display: "flex",
          alignItems: "flex-end",
          height: "90px",
          gap: "2px",
          background: "var(--bg-page)",
          padding: "8px 6px",
          borderRadius: "6px",
          border: "1px solid var(--border-card)",
          marginBottom: "12px",
        }}
      >
        {frames.map((frame, idx) => {
          const isSelected = selectedIdx === idx;
          const isHovered = hoveredIdx === idx;
          const isSpike = frame.score >= 0.70;
          const barColor = scoreColor(frame.score);

          return (
            <div
              key={frame.timestamp}
              className="timeline-bar-wrapper"
              style={{
                flex: 1,
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "flex-end",
                alignItems: "center",
                cursor: "pointer",
                position: "relative",
              }}
              onClick={() => {
                setSelectedIdx(idx);
                setHoveredIdx(null);
              }}
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {isSpike && (
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    width: "4px",
                    height: "4px",
                    borderRadius: "50%",
                    background: "#f43f5e",
                    boxShadow: "0 0 4px #f43f5e",
                  }}
                />
              )}
              <div
                style={{
                  width: "100%",
                  height: `${Math.max(12, Math.round(frame.score * 70))}px`,
                  background: barColor,
                  borderRadius: "2px",
                  opacity: isSelected || isHovered ? 1 : 0.55,
                  transform: isSelected || isHovered ? "scaleY(1.05)" : "none",
                  border: isSelected ? "1px solid #ffffff" : "none",
                  transition: "opacity 0.15s ease, transform 0.15s ease",
                }}
              />
            </div>
          );
        })}
      </div>

      {/* Frame Telemetry Card */}
      {activeFrame && (
        <div
          style={{
            background: "var(--bg-page)",
            border: "1px solid var(--border-card)",
            borderRadius: "6px",
            padding: "10px 14px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))",
            gap: "10px",
            fontSize: "12px",
            color: "var(--text-main)",
          }}
        >
          <div>
            <span style={{ color: "#64748b", display: "block", fontSize: "10px", textTransform: "uppercase" }}>Frame Index</span>
            <strong style={{ color: "#f8fafc", fontFamily: "monospace" }}>#{activeIdx + 1} ({activeFrame.timestamp}s)</strong>
          </div>
          <div>
            <span style={{ color: "#64748b", display: "block", fontSize: "10px", textTransform: "uppercase" }}>Synthetic Score</span>
            <strong style={{ color: scoreColor(activeFrame.score) }}>
              {Math.round(activeFrame.score * 100)}% {activeFrame.score >= 0.70 ? "AI" : activeFrame.score >= 0.40 ? "Mixed" : "Real"}
            </strong>
          </div>
          <div>
            <span style={{ color: "#64748b", display: "block", fontSize: "10px", textTransform: "uppercase" }}>Biometric Face</span>
            <strong style={{ color: activeFrame.has_face ? "#38bdf8" : "#94a3b8" }}>
              {activeFrame.has_face ? `Face Detected (${activeFrame.face_count || 1})` : "None"}
            </strong>
          </div>
          {temporalJitter !== undefined && (
            <div>
              <span style={{ color: "#64748b", display: "block", fontSize: "10px", textTransform: "uppercase" }}>Temporal Jitter</span>
              <strong style={{ color: temporalJitter > 0.08 ? "#f59e0b" : "#10b981", fontFamily: "monospace" }}>
                {temporalJitter.toFixed(4)}
              </strong>
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "10px", fontSize: "11px", color: "#64748b" }}>
        <div style={{ display: "flex", gap: "12px" }}>
          {frames.length > 1 ? (
            <>
              <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--green, #10b981)" }} />
                Low ({Math.round(minScore * 100)}%)
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent3, #f43f5e)" }} />
                High ({Math.round(maxScore * 100)}%)
              </span>
            </>
          ) : (
            <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: scoreColor(frames[0].score) }} />
              Single Keyframe ({Math.round(frames[0].score * 100)}%)
            </span>
          )}
        </div>
        <span>{frames.length > 1 ? "Click bar to inspect frame" : "Keyframe snapshot"}</span>
      </div>
    </div>
  );
}
