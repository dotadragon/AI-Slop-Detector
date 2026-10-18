import { useRef, useState } from "react";
import { Info } from "lucide-react";
import { scoreColor } from "../../utils/validators";
import { SIGNAL_META, SIGNAL_ORDER } from "./signalMeta";

function labelForScore(score: number) {
  if (score >= 0.70) return "High AI Signal";
  if (score >= 0.35) return "Moderate";
  return "Low (Human-Like)";
}

interface SignalCardProps {
  signalKey: string;
  value: number;
  hasOverride: boolean;
}

function SignalCard({ signalKey, value, hasOverride }: SignalCardProps) {
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);
  const hoverTimerRef = useRef<number | null>(null);

  const clamped = Math.max(0, Math.min(1, value));
  const percent = Math.round(clamped * 100);
  const meta = SIGNAL_META[signalKey];
  const isNeuralKey = signalKey === "model_ai_probability" || signalKey === "neural_ai_score";
  const isCalibrated = isNeuralKey && hasOverride;

  const handleMouseEnter = () => {
    if (hoverTimerRef.current) window.clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = window.setTimeout(() => setIsTooltipVisible(true), 400);
  };

  const handleMouseLeave = () => {
    if (hoverTimerRef.current) {
      window.clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    setIsTooltipVisible(false);
  };

  return (
    <div className={`signal signal-metered ${isCalibrated ? "calibrated-signal" : ""}`}>
      <div className="signal-header-top">
        <span className="signal-title-text">{meta.label}</span>
        <div className="signal-header-actions">
          {isCalibrated && (
            <span className="calibrated-tag" title="Adjusted by forensic heuristic reconciliation">
              Calibrated
            </span>
          )}
          <div
            className="signal-info-wrapper"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onFocus={() => setIsTooltipVisible(true)}
            onBlur={handleMouseLeave}
          >
            <button type="button" className="signal-info-btn" aria-label={`What is ${meta.label}?`}>
              <Info size={13} />
            </button>
            {isTooltipVisible && (
              <div className="signal-tooltip-popover" role="tooltip">
                <strong>{meta.label}</strong>
                <p>{meta.tooltip}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="signal-value-row">
        <strong>{percent}%</strong>
        <em>{labelForScore(clamped)}</em>
      </div>

      <div className="signal-meter" aria-hidden="true">
        <div style={{ width: `${percent}%`, background: scoreColor(clamped) }} />
      </div>

      <p className="signal-short-desc">{meta.shortDesc}</p>
    </div>
  );
}

export function SignalGrid({ signals }: { signals?: Record<string, unknown> }) {
  if (!signals) return null;
  const hasOverride = Boolean(signals.heuristic_override_explanation);

  const entries = SIGNAL_ORDER.map((key) => [key, signals[key]] as const).filter(
    ([, val]) => typeof val === "number" && Number.isFinite(val)
  );

  if (entries.length === 0) return null;

  return (
    <div className="signal-grid">
      {entries.map(([key, val]) => (
        <SignalCard
          key={key}
          signalKey={key}
          value={val as number}
          hasOverride={hasOverride}
        />
      ))}
    </div>
  );
}
