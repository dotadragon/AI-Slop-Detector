import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { scoreColor, scoreLabel } from "../../utils/validators";

const RADIUS = 46;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function ScoreGauge({ pct, color }: { pct: number; color: string }) {
  const numRef = useRef<HTMLSpanElement>(null);
  const circleRef = useRef<SVGCircleElement>(null);
  const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;

  useEffect(() => {
    if (numRef.current) {
      const obj = { val: 0 };
      gsap.to(obj, {
        val: pct,
        duration: 1.0,
        ease: "power2.out",
        onUpdate: () => {
          if (numRef.current) {
            numRef.current.textContent = Math.round(obj.val).toString();
          }
        },
      });
    }

    if (circleRef.current) {
      gsap.fromTo(
        circleRef.current,
        { strokeDashoffset: CIRCUMFERENCE },
        { strokeDashoffset: offset, duration: 1.2, ease: "power3.out" }
      );
    }
  }, [pct, offset]);

  return (
    <div className="score-gauge-container">
      <svg className="score-svg" viewBox="0 0 120 120" width="112" height="112">
        <circle cx="60" cy="60" r={RADIUS} fill="none" stroke="var(--bg3)" strokeWidth="8" />
        <circle
          ref={circleRef}
          cx="60"
          cy="60"
          r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 60 60)"
          style={{ transition: "stroke 0.3s ease" }}
        />
      </svg>
      <div className="score-center-val">
        <span className="score-number" ref={numRef}>{pct}</span>
        <span className="score-unit">%</span>
      </div>
    </div>
  );
}

function formatScoreSubtitle(score: number, pct: number): string {
  if (score > 0.70) return `Synthetic signature probability: ${pct}%`;
  if (score >= 0.35) return `Mixed stylometric / spectral signals: ${pct}%`;
  return `Authentic human provenance confidence: ${100 - pct}%`;
}

interface ScoreCardProps {
  score: number;
  confidence_range?: number[];
  confidence_band?: string;
  isEditedCamera?: boolean;
}

export function ScoreCard({ score, confidence_range, isEditedCamera }: ScoreCardProps) {
  const pct = Math.round(score * 100);
  const color = scoreColor(score);
  const label = scoreLabel(score);
  const Icon = score > 0.70 ? ShieldAlert : score >= 0.35 ? AlertTriangle : ShieldCheck;

  const rangeMin = confidence_range && confidence_range.length >= 2 ? Math.round(confidence_range[0] * 100) : null;
  const rangeMax = confidence_range && confidence_range.length >= 2 ? Math.round(confidence_range[1] * 100) : null;

  return (
    <div className="score-card">
      <ScoreGauge pct={pct} color={color} />
      <div className="score-info">
        <div className="score-header-row">
          <span className="score-badge-pill" style={{ color, borderColor: color }}>
            <Icon size={14} />
            <strong>{label}</strong>
          </span>
          {isEditedCamera && (
            <span
              className="score-badge-pill"
              style={{
                color: "#34d399",
                borderColor: "rgba(16, 185, 129, 0.4)",
                background: "rgba(16, 185, 129, 0.12)",
              }}
            >
              📷 Camera Photo · Post-Processed / Edited
            </span>
          )}
        </div>
        <p className="score-subtitle">{formatScoreSubtitle(score, pct)}</p>
        {rangeMin !== null && rangeMax !== null && (
          <div className="score-confidence-range" aria-label="Confidence range">
            <span className="score-confidence-range-label">Confidence range:</span>
            <span className="score-confidence-range-value">{rangeMin}% – {rangeMax}%</span>
          </div>
        )}
        <div className="score-bar-track">
          <div
            className="score-bar-fill"
            style={{
              width: `${pct}%`,
              background: color,
              transition: "width 0.9s cubic-bezier(0.16, 1, 0.3, 1)",
            }}
          />
        </div>
      </div>
    </div>
  );
}
