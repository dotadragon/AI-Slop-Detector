import React from "react";
import { ExternalLink, RefreshCw, ShieldCheck } from "lucide-react";
import { Button } from "../components/ui/Button";
import { useSeo } from "../hooks/useSeo";

export function Calibration(): React.JSX.Element {
  useSeo({
    title: "Calibration & Accuracy Dashboard — SLOP Forensics",
    description: "Multi-modal calibration, Wilson 95% confidence intervals, and reliability diagrams across text, image, and video models.",
    path: "/calibration",
  });

  const [key, setKey] = React.useState(0);

  const handleRefresh = () => {
    setKey((prev) => prev + 1);
  };

  return (
    <main className="calibration-page" style={{ maxWidth: "1280px", margin: "0 auto", padding: "2rem 1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.5rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h1 style={{ fontSize: "2rem", fontWeight: 700, letterSpacing: "-0.02em", color: "#f8fafc", margin: 0 }}>
            Forensic Calibration &amp; Accuracy
          </h1>
          <p style={{ color: "#94a3b8", fontSize: "0.95rem", marginTop: "0.5rem", maxWidth: "680px" }}>
            Live empirical accuracy, Wilson 95% confidence intervals, reliability diagrams, and subtype confusion matrices evaluated on sealed held-out benchmarks.
          </p>
        </div>

        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <Button
            variant="ghost"
            icon={<RefreshCw size={15} />}
            onClick={handleRefresh}
          >
            Refresh
          </Button>
          <a
            href="/api/v1/ops/calibration-dashboard"
            target="_blank"
            rel="noopener noreferrer"
            style={{ textDecoration: "none" }}
          >
            <Button variant="primary" icon={<ExternalLink size={15} />}>
              Open Raw Report
            </Button>
          </a>
        </div>
      </div>

      <section style={{ marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "1.25rem", fontWeight: 600, color: "#e2e8f0", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <ShieldCheck size={20} color="#10b981" /> Empirical Calibration Report
        </h2>
        <div
          style={{
            background: "#0d1117",
            border: "1px solid #1e293b",
            borderRadius: "12px",
            overflow: "hidden",
            boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.4)",
          }}
        >
          <iframe
            key={key}
            src="/api/v1/ops/calibration-dashboard"
            title="SLOP Accuracy and Calibration Dashboard"
            style={{
              width: "100%",
              height: "820px",
              border: "none",
              display: "block",
              background: "#ffffff",
            }}
          />
        </div>
      </section>

      <section style={{ background: "var(--bg-card)", border: "1px solid var(--border-card)", borderRadius: "12px", padding: "1.5rem", color: "var(--text-main)" }}>
        <h3 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--text-main)", marginTop: 0, marginBottom: "0.75rem" }}>
          Forensic Methodology &amp; Directives
        </h3>
        <ul style={{ margin: 0, paddingLeft: "1.25rem", lineHeight: 1.7, fontSize: "0.9rem", color: "var(--text-muted)" }}>
          <li>
            <strong style={{ color: "var(--text-main)" }}>Zero-Mock Execution:</strong> All evaluation metrics are generated strictly from live forward passes on genuine model backbones.
          </li>
          <li>
            <strong style={{ color: "var(--text-main)" }}>Wilson Score Intervals:</strong> 95% confidence intervals account for sample size and variance rather than relying on point estimates.
          </li>
          <li>
            <strong style={{ color: "#f8fafc" }}>Sealed Test Split:</strong> Test benchmarks in <code>heldout_set/</code> are evaluated strictly post-promotion and are never accessed during parameter tuning.
          </li>
        </ul>
      </section>
    </main>
  );
}
