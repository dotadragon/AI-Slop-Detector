import { useState } from "react";
import { ChevronLeft, ChevronRight, Zap, Film } from "lucide-react";
import type { DetectionResult, FrameScore } from "../../types";
import { scoreColor } from "../../utils/validators";

function findPeakAnomaly(frames: FrameScore[]): number {
  let peakIndex = 0;
  let peakVal = -1;
  frames.forEach((frame, idx) => {
    if (frame.score > peakVal) {
      peakVal = frame.score;
      peakIndex = idx;
    }
  });
  return peakIndex;
}

function TimelineHeader({
  anomalyCount,
  selectedIdx,
  frameCount,
  onJumpPeak,
  onPrev,
  onNext,
}: {
  anomalyCount: number;
  selectedIdx: number;
  frameCount: number;
  onJumpPeak: () => void;
  onPrev: () => void;
  onNext: () => void;
}) {
  return (
    <div className="video-timeline-header">
      <div className="video-timeline-title-wrap">
        <Film size={16} className="text-cyan-400" />
        <span className="video-timeline-title">Temporal Anomaly Scrubber</span>
        {anomalyCount > 0 && (
          <span className="video-timeline-spike-badge">
            {anomalyCount} {anomalyCount === 1 ? "Spike" : "Spikes"} Detected
          </span>
        )}
      </div>

      <div className="video-timeline-controls">
        <button
          type="button"
          className="btn btn-secondary video-timeline-btn"
          onClick={onJumpPeak}
          title="Jump to highest synthetic anomaly frame"
        >
          <Zap size={12} className="text-amber-400" />
          <span>Peak Anomaly</span>
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          disabled={selectedIdx <= 0}
          onClick={onPrev}
          aria-label="Previous frame"
        >
          <ChevronLeft size={14} />
        </button>
        <span className="video-timeline-counter">
          {selectedIdx + 1} / {frameCount}
        </span>
        <button
          type="button"
          className="btn btn-secondary"
          disabled={selectedIdx >= frameCount - 1}
          onClick={onNext}
          aria-label="Next frame"
        >
          <ChevronRight size={14} />
        </button>
      </div>
    </div>
  );
}

function TimelineBarTrack({
  frames,
  selectedIdx,
  hoveredIdx,
  onSelect,
  onHover,
  onLeave,
}: {
  frames: FrameScore[];
  selectedIdx: number;
  hoveredIdx: number | null;
  onSelect: (index: number) => void;
  onHover: (index: number) => void;
  onLeave: () => void;
}) {
  return (
    <div className="video-timeline-track">
      {frames.map((frame, idx) => {
        const isSelected = selectedIdx === idx;
        const isHovered = hoveredIdx === idx;
        const isSpike = frame.score >= 0.70;
        const barColor = scoreColor(frame.score);

        return (
          <div
            key={frame.timestamp}
            className="video-timeline-bar-wrapper"
            onClick={() => onSelect(idx)}
            onMouseEnter={() => onHover(idx)}
            onMouseLeave={onLeave}
          >
            {isSpike && <div className="video-timeline-spike-dot" />}
            <div
              className="video-timeline-bar"
              style={{
                height: `${Math.max(12, Math.round(frame.score * 70))}px`,
                background: barColor,
                opacity: isSelected || isHovered ? 1 : 0.55,
                transform: isSelected || isHovered ? "scaleY(1.05)" : "none",
                border: isSelected ? "1px solid #ffffff" : "none",
              }}
            />
          </div>
        );
      })}
    </div>
  );
}

function TimelineTelemetryCard({
  activeFrame,
  activeIdx,
  temporalJitter,
}: {
  activeFrame: FrameScore;
  activeIdx: number;
  temporalJitter?: number;
}) {
  const toneColor = scoreColor(activeFrame.score);
  const statusLabel = activeFrame.score >= 0.70 ? "AI" : activeFrame.score >= 0.40 ? "Mixed" : "Real";

  return (
    <div className="video-timeline-card">
      <div>
        <span className="video-timeline-card-label">Frame Index</span>
        <strong className="video-timeline-card-value">#{activeIdx + 1} ({activeFrame.timestamp}s)</strong>
      </div>
      <div>
        <span className="video-timeline-card-label">Synthetic Score</span>
        <strong style={{ color: toneColor }}>
          {Math.round(activeFrame.score * 100)}% {statusLabel}
        </strong>
      </div>
      <div>
        <span className="video-timeline-card-label">Biometric Face</span>
        <strong style={{ color: activeFrame.has_face ? "var(--accent)" : "var(--text-muted)" }}>
          {activeFrame.has_face ? `Face Detected (${activeFrame.face_count || 1})` : "None"}
        </strong>
      </div>
      {temporalJitter !== undefined && (
        <div>
          <span className="video-timeline-card-label">Temporal Jitter</span>
          <strong style={{ color: temporalJitter > 0.08 ? "#f59e0b" : "#10b981", fontFamily: "monospace" }}>
            {temporalJitter.toFixed(4)}
          </strong>
        </div>
      )}
    </div>
  );
}

function TimelineLegend({ frames }: { frames: FrameScore[] }) {
  const minScore = Math.min(...frames.map((f) => f.score));
  const maxScore = Math.max(...frames.map((f) => f.score));

  return (
    <div className="video-timeline-legend">
      <div className="video-timeline-legend-row">
        {frames.length > 1 ? (
          <>
            <span className="video-timeline-legend-item">
              <span className="video-timeline-legend-dot" style={{ background: "var(--green, #10b981)" }} />
              Low ({Math.round(minScore * 100)}%)
            </span>
            <span className="video-timeline-legend-item">
              <span className="video-timeline-legend-dot" style={{ background: "var(--accent3, #f43f5e)" }} />
              High ({Math.round(maxScore * 100)}%)
            </span>
          </>
        ) : (
          <span className="video-timeline-legend-item">
            <span className="video-timeline-legend-dot" style={{ background: scoreColor(frames[0].score) }} />
            Single Keyframe ({Math.round(frames[0].score * 100)}%)
          </span>
        )}
      </div>
      <span>{frames.length > 1 ? "Click bar to inspect frame" : "Keyframe snapshot"}</span>
    </div>
  );
}

export function VideoTimeline({ videoResult }: { videoResult?: DetectionResult }) {
  const frames = videoResult?.frame_scores || [];
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number>(() => findPeakAnomaly(frames));

  if (!frames.length) return null;

  const activeIdx = hoveredIdx !== null ? hoveredIdx : selectedIdx;
  const activeFrame = frames[activeIdx] || frames[0];
  const anomalyCount = frames.filter((f) => f.score >= 0.70).length;

  return (
    <div className="video-timeline-panel">
      <TimelineHeader
        anomalyCount={anomalyCount}
        selectedIdx={selectedIdx}
        frameCount={frames.length}
        onJumpPeak={() => {
          setSelectedIdx(findPeakAnomaly(frames));
          setHoveredIdx(null);
        }}
        onPrev={() => {
          setSelectedIdx((prev) => Math.max(0, prev - 1));
          setHoveredIdx(null);
        }}
        onNext={() => {
          setSelectedIdx((prev) => Math.min(frames.length - 1, prev + 1));
          setHoveredIdx(null);
        }}
      />

      <TimelineBarTrack
        frames={frames}
        selectedIdx={selectedIdx}
        hoveredIdx={hoveredIdx}
        onSelect={(idx) => {
          setSelectedIdx(idx);
          setHoveredIdx(null);
        }}
        onHover={setHoveredIdx}
        onLeave={() => setHoveredIdx(null)}
      />

      {activeFrame && (
        <TimelineTelemetryCard
          activeFrame={activeFrame}
          activeIdx={activeIdx}
          temporalJitter={videoResult?.signals?.temporal_jitter as number | undefined}
        />
      )}

      <TimelineLegend frames={frames} />
    </div>
  );
}
