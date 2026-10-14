import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { scoreColor, scoreLabel } from "../../utils/validators";

export function ScoreCard({
  score,
  confidence_range,
  confidence_band,
}: {
  score: number;
  confidence_range?: number[];
  confidence_band?: string;
}) {
  const pct = Math.round(score * 100);
  const numRef = useRef<HTMLSpanElement>(null);
  const circleRef = useRef<SVGCircleElement>(null);

  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  const color = scoreColor(score);
  const label = scoreLabel(score);

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
        { strokeDashoffset: circumference },
        { strokeDashoffset: offset, duration: 1.2, ease: "power3.out" }
      );
    }
  }, [pct, circumference, offset]);

  const Icon = score > 0.70 ? ShieldAlert : score >= 0.35 ? AlertTriangle : ShieldCheck;

  const bandColorMap: Record<string, string> = {
    "DEFINITELY_AI": "var(--red)",
    "LIKELY_AI": "var(--red)",
    "MIXED": "var(--amber)",
    "LIKELY_HUMAN": "var(--green)",
    "DEFINITELY_HUMAN": "var(--green)",
    "UNCERTAIN": "var(--fg3)",
  };
  const bandColor = confidence_band ? (bandColorMap[confidence_band] ?? "var(--fg3)") : undefined;
  const rangeMin = confidence_range && confidence_range.length >= 2 ? Math.round(confidence_range[0] * 100) : null;
  const rangeMax = confidence_range && confidence_range.length >= 2 ? Math.round(confidence_range[1] * 100) : null;

  return (
    <div className="score-card">
      <div className="score-gauge-container">
        <svg className="score-svg" viewBox="0 0 120 120" width="112" height="112">
          {/* Background track circle */}
          <circle
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke="var(--bg3)"
            strokeWidth="8"
          />
          {/* Clean score indicator without neon drop-shadow */}
          <circle
            ref={circleRef}
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform="rotate(-90 60 60)"
            style={{
              transition: "stroke 0.3s ease",
            }}
          />
        </svg>

        <div className="score-center-val">
          <span className="score-number" ref={numRef}>
            {pct}
          </span>
          <span className="score-unit">%</span>
        </div>
      </div>

      <div className="score-info">
        <div className="score-header-row">
          <span className="score-badge-pill" style={{ color, borderColor: color }}>
            <Icon size={14} />
            <strong>{label}</strong>
          </span>
        </div>
        <p className="score-subtitle">
          {score > 0.70
            ? `${pct}% probability of AI-generated content detected`
            : score >= 0.35
            ? `${pct}% mixed heuristic & neural signals detected`
            : `${100 - pct}% confidence of authentic human photograph / graphic`}
        </p>
        {rangeMin !== null && rangeMax !== null && (
          <div className="score-confidence-range" aria-label="Confidence range">
            <span className="score-confidence-range-label">Confidence range:</span>
            <span className="score-confidence-range-value">{rangeMin}% – {rangeMax}%</span>
          </div>
        )}
        {confidence_band && (
          <div
            className="score-confidence-band"
            style={{ color: bandColor, borderColor: bandColor }}
            aria-label={`Confidence band: ${confidence_band}`}
          >
            {confidence_band.replace(/_/g, " ")}
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
