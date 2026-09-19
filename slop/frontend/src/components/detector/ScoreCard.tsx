import { motion } from "framer-motion";
import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { scoreColor, scoreLabel } from "../../utils/validators";

export function ScoreCard({ score }: { score: number }) {
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

  const Icon = score >= 0.72 ? ShieldAlert : score >= 0.42 ? AlertTriangle : ShieldCheck;

  return (
    <motion.div
      className="score-card"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
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
          {/* Animated score indicator */}
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
              filter: `drop-shadow(0 0 6px ${color})`,
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
          {score >= 0.72
            ? `${pct}% probability of AI-generated content detected`
            : score >= 0.42
            ? `${pct}% mixed heuristic & neural signals detected`
            : `${100 - pct}% confidence of authentic human photograph / graphic`}
        </p>
        <div className="score-bar-track">
          <motion.div
            className="score-bar-fill"
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.9, ease: "easeOut" }}
            style={{ background: color }}
          />
        </div>
      </div>
    </motion.div>
  );
}
