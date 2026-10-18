import { useState, useEffect, useCallback, type JSX } from "react";
import { ExternalLink, RefreshCw, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "../components/ui/Button";
import { useSeo } from "../hooks/useSeo";

interface SubtypeMetrics {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

interface ReliabilityBin {
  mean_confidence: number | null;
  accuracy: number;
  count: number;
}

interface ModalityReport {
  modality: string;
  accuracy_pct: number;
  accuracy_wilson_95: [number, number];
  ece: number;
  total: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  by_subtype: Record<string, SubtypeMetrics>;
  reliability_diagram: ReliabilityBin[];
}

type CalibrationData = Record<string, ModalityReport>;

function ReliabilityCurve({ diagram }: { diagram: ReliabilityBin[] }) {
  const svgWidth = 320;
  const svgHeight = 200;
  const padding = 30;

  const points: JSX.Element[] = [];

  diagram.forEach((bin, idx) => {
    if (bin.mean_confidence === null) return;
    const xCoord = padding + bin.mean_confidence * (svgWidth - 2 * padding);
    const yCoord = svgHeight - padding - bin.accuracy * (svgHeight - 2 * padding);
    const pointRadius = Math.min(8, 2 + bin.count);
    points.push(
      <circle
        key={idx}
        cx={xCoord.toFixed(1)}
        cy={yCoord.toFixed(1)}
        r={pointRadius}
        fill="var(--accent, #f97316)"
        opacity={0.85}
      />
    );
  });

  return (
    <div className="calibration-box">
      <h3>Reliability &amp; Calibration Curve</h3>
      <svg width="100%" height="180" viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ overflow: "visible", maxWidth: "340px", display: "block" }}>
        <line x1={padding} y1={svgHeight - padding} x2={svgWidth - padding} y2={svgHeight - padding} stroke="var(--border-card)" strokeWidth="1.5" />
        <line x1={padding} y1={padding} x2={padding} y2={svgHeight - padding} stroke="var(--border-card)" strokeWidth="1.5" />
        <line
          x1={padding}
          y1={svgHeight - padding}
          x2={svgWidth - padding}
          y2={padding}
          stroke="var(--text-muted)"
          strokeDasharray="4 3"
          strokeWidth="1"
          opacity={0.5}
        />
        {points}
        <text x={svgWidth / 2} y={svgHeight - 6} textAnchor="middle" fontSize="10" fill="var(--text-muted)" fontFamily="var(--mono)">
          confidence &rarr;
        </text>
        <text
          x={12}
          y={svgHeight / 2}
          fontSize="10"
          transform={`rotate(-90 12 ${svgHeight / 2})`}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontFamily="var(--mono)"
        >
          accuracy &rarr;
        </text>
      </svg>
      <div className="calibration-legend">
        <span>Dashed: Ideal Calibration</span>
        <span>Orange: Observed Accuracy</span>
      </div>
    </div>
  );
}

function SubtypeBreakdownTable({ modality }: { modality: ModalityReport }) {
  const subtypes = Object.entries(modality.by_subtype || {});

  return (
    <div className="calibration-box calibration-table-container">
      <h3>Per-Subtype Partition Breakdown</h3>
      <table className="calibration-table">
        <thead>
          <tr>
            <th>Subtype</th>
            <th>n</th>
            <th>TP</th>
            <th>FP</th>
            <th>TN</th>
            <th>FN</th>
            <th>Recall</th>
          </tr>
        </thead>
        <tbody>
          {subtypes.map(([subName, metrics]) => {
            const subTotal = metrics.tp + metrics.fp + metrics.tn + metrics.fn;
            const recallDenom = metrics.tp + metrics.fn;
            const recall = recallDenom > 0 ? ((metrics.tp / recallDenom) * 100).toFixed(1) : "N/A";
            return (
              <tr key={subName}>
                <td>{subName}</td>
                <td>{subTotal}</td>
                <td>{metrics.tp}</td>
                <td>{metrics.fp}</td>
                <td>{metrics.tn}</td>
                <td>{metrics.fn}</td>
                <td>{recall}%</td>
              </tr>
            );
          })}
          <tr className="calibration-table-total">
            <td>TOTAL</td>
            <td>{modality.total}</td>
            <td>{modality.tp}</td>
            <td>{modality.fp}</td>
            <td>{modality.tn}</td>
            <td>{modality.fn}</td>
            <td>{modality.accuracy_pct}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function ModalityCard({ modality }: { modality: ModalityReport }) {
  const acc = modality.accuracy_pct;
  const badgeBg = acc >= 80 ? "rgba(16, 185, 129, 0.15)" : acc >= 70 ? "rgba(249, 115, 22, 0.15)" : "rgba(239, 68, 68, 0.15)";
  const badgeColor = acc >= 80 ? "#10b981" : acc >= 70 ? "#f97316" : "#ef4444";
  const [ciLo, ciHi] = modality.accuracy_wilson_95 || [0, 0];

  return (
    <section className="calibration-card">
      <div className="calibration-card-header">
        <div className="calibration-card-title-group">
          <h2 className="calibration-card-title">{modality.modality} Forensics</h2>
          <span
            className="calibration-modality-badge"
            style={{
              background: badgeBg,
              color: badgeColor,
              border: `1px solid ${badgeColor}40`,
            }}
          >
            {acc}% BAcc
          </span>
        </div>
        <div className="calibration-meta-ci">
          n={modality.total} &bull; Wilson 95% CI: [{ciLo}%, {ciHi}%] &bull; ECE: {modality.ece}
        </div>
      </div>

      <div className="calibration-card-grid">
        <ReliabilityCurve diagram={modality.reliability_diagram || []} />
        <SubtypeBreakdownTable modality={modality} />
      </div>
    </section>
  );
}

function CalibrationHeader({
  isCalibrating,
  onRefresh,
}: {
  isCalibrating: boolean;
  onRefresh: () => void;
}) {
  return (
    <header className="calibration-header-wrap">
      <div>
        <div className="calibration-badge">
          <CheckCircle2 size={13} />
          EMPIRICALLY VERIFIED &bull; ZERO-MOCK PIPELINE
        </div>
        <h1 className="calibration-title">Forensic Calibration &amp; Accuracy</h1>
        <p className="calibration-desc">
          Live empirical balanced accuracy, Wilson 95% confidence intervals, and reliability curves across sealed held-out evaluation benchmarks.
        </p>
      </div>

      <div className="calibration-actions">
        <Button variant="ghost" icon={<RefreshCw size={15} className={isCalibrating ? "spin" : ""} />} onClick={onRefresh}>
          Refresh Data
        </Button>
        <a href="/api/v1/ops/calibration-dashboard" target="_blank" rel="noopener noreferrer" className="calibration-link-clean">
          <Button variant="primary" icon={<ExternalLink size={15} />}>
            Open HTML Dashboard
          </Button>
        </a>
      </div>
    </header>
  );
}

export function Calibration(): JSX.Element {
  useSeo({
    title: "Calibration & Accuracy Dashboard — SLOP Forensics",
    description: "Multi-modal calibration, Wilson 95% confidence intervals, and reliability diagrams across text, image, and video models.",
    path: "/calibration",
  });

  const [data, setData] = useState<CalibrationData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [showIframe, setShowIframe] = useState<boolean>(false);
  const [key, setKey] = useState(0);

  const fetchReport = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/ops/calibration-report");
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchReport();
  }, [fetchReport, key]);

  return (
    <main className="calibration-page">
      <CalibrationHeader isCalibrating={loading} onRefresh={() => setKey((k) => k + 1)} />

      <div className="calibration-card-list">
        {data && Object.entries(data).map(([keyName, modality]) => (
          <ModalityCard key={keyName} modality={modality} />
        ))}
      </div>

      <button type="button" className="calibration-toggle-btn" onClick={() => setShowIframe((s) => !s)}>
        <span>Detailed Heldout Test Harness Dashboard</span>
        {showIframe ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </button>

      {showIframe && (
        <div className="calibration-iframe-box">
          <iframe src="/api/v1/ops/calibration-dashboard" title="Calibration Full Dashboard" />
        </div>
      )}
    </main>
  );
}
