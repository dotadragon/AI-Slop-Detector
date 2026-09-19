import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";
import gsap from "gsap";
import { Shield, ShieldCheck, Brain, Eye, Cpu, Sparkles, AlertTriangle, Fingerprint, FileText, Layers } from "lucide-react";
import type { DetectionResult } from "../../types";
import { ScanningHUD } from "./ScanningHUD";
import { HeatmapOverlay } from "./HeatmapOverlay";
import { ScoreCard } from "./ScoreCard";
import { SentenceHighlight } from "./SentenceHighlight";
import { SignalGrid } from "./SignalGrid";
import { VideoTimeline } from "./VideoTimeline";
import { ThreeDForensicScanner } from "../ui/ThreeDForensicScanner";

function ProvenanceBadge({ signals }: { signals?: Record<string, unknown> }) {
  if (!signals) return null;

  const provenance = signals.provenance as string | undefined;
  const inferenceMode = signals.inference_mode as string | undefined;

  return (
    <div className="provenance-badges">
      {provenance === "verified" && (
        <div className="provenance-badge verified">
          <ShieldCheck size={16} />
          <span>Verified Provenance (C2PA)</span>
        </div>
      )}
      {provenance === "none" && (
        <div className="provenance-badge none">
          <Shield size={14} />
          <span>No C2PA provenance header</span>
        </div>
      )}
      {(inferenceMode === "transformers_image_classifier" || inferenceMode === "neural" || inferenceMode === "neural_multi_crop") && (
        <div className="inference-badge neural">
          <Brain size={14} />
          <span>{inferenceMode === "neural_multi_crop" ? "Neural Multi-Crop v3.0" : "Neural Vision Transformer"}</span>
        </div>
      )}
      {inferenceMode === "calibrated_frontier_ensemble" && (
        <div className="inference-badge neural frontier">
          <Brain size={14} className="text-cyan-400" />
          <span>Calibrated Frontier Ensemble v3.4</span>
        </div>
      )}
      {inferenceMode === "roberta_classifier" && (
        <div className="inference-badge neural">
          <Brain size={14} />
          <span>RoBERTa Neural v3.0</span>
        </div>
      )}
      {Boolean(signals?.prnu_sensor_residual_active) && (
        <div className="inference-badge prnu">
          <Eye size={14} className="text-emerald-400" />
          <span>PRNU Sensor Active</span>
        </div>
      )}
      {inferenceMode === "heuristic_fallback" && (
        <div className="inference-badge heuristic">
          <Eye size={14} />
          <span>Heuristic Analysis v3.0</span>
        </div>
      )}
    </div>
  );
}

function AttributionCard({ result }: { result: DetectionResult }) {
  const attribution = result.generator_attribution;
  const signature = result.digital_signature;

  if (!attribution && !signature?.detected) return null;

  const generatorName = attribution?.generator || signature?.generator || "AI Engine";
  const confidence = Math.round((attribution?.confidence || signature?.confidence || 0.8) * 100);

  return (
    <div className="attribution-card">
      <div className="attribution-header">
        <div className="attribution-title">
          <Cpu size={16} className="text-amber-400" />
          <span>AI Engine Attribution & Forensics</span>
        </div>
        <span className="attribution-badge">{confidence}% Match</span>
      </div>

      <div className="attribution-body">
        <div className="attribution-engine">
          <span className="engine-label">Probable Generator:</span>
          <span className="engine-name">{generatorName}</span>
        </div>

        {signature?.detected && (
          <div className="attribution-sig">
            <div className="sig-row">
              <Sparkles size={14} className="text-emerald-400" />
              <span className="sig-label">Digital Signature:</span>
              <span className="sig-value">{signature.signature_type}</span>
            </div>
            {signature.bounding_box && (
              <div className="sig-box-note">
                <span className="sig-reticle-dot" />
                <span>Localized signature highlighted with a <strong>green reticle box</strong> on image preview</span>
              </div>
            )}
          </div>
        )}

        {(attribution?.details || signature?.details) && (
          <div className="attribution-details">
            {signature?.details || attribution?.details}
          </div>
        )}
      </div>
    </div>
  );
}

function LLMAttributionCard({ result }: { result: DetectionResult }) {
  const attr = result.llm_attribution;
  if (!attr || !attr.detected_llm) return null;

  const confidence = Math.round(attr.confidence * 100);
  const colorClass = confidence >= 80 ? "high" : confidence >= 50 ? "medium" : "low";

  return (
    <div className="attribution-card llm-attribution">
      <div className="attribution-header">
        <div className="attribution-title">
          <Fingerprint size={16} className="text-violet-400" />
          <span>LLM Source Attribution</span>
        </div>
        <span className={`attribution-badge ${colorClass}`}>{confidence}% Confidence</span>
      </div>

      <div className="attribution-body">
        <div className="attribution-engine">
          <span className="engine-label">Detected Source:</span>
          <span className="engine-name llm-name">{attr.detected_llm}</span>
        </div>

        <div className="llm-scores">
          {Object.entries(attr.scores).map(([llm, score]) => (
            <div key={llm} className={`llm-score-row ${score > 0 ? "matched" : ""}`}>
              <span>{llm}</span>
              <span className="llm-score-count">{score} pattern{score !== 1 ? "s" : ""}</span>
            </div>
          ))}
        </div>

        {attr.all_patterns && attr.all_patterns.length > 0 && (
          <div className="llm-patterns-list">
            <span className="patterns-label">Matched patterns:</span>
            {attr.all_patterns.slice(0, 5).map((p, i) => (
              <code key={i} className="pattern-tag">{p}</code>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ModelHint({ hint }: { hint?: string }) {
  if (!hint) return null;
  return (
    <p className="hint">
      {hint}
    </p>
  );
}

function DocumentSectionsHeatmap({ sections }: { sections?: DetectionResult["section_scores"] }) {
  if (!sections || sections.length === 0) return null;

  return (
    <div className="document-sections-card" style={{ marginTop: "16px", background: "#0b0f19", border: "1px solid #1e293b", borderRadius: "8px", padding: "16px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Layers size={16} className="text-cyan-400" />
          <span style={{ fontSize: "13px", fontWeight: 700, color: "#f8fafc" }}>Document Section Threat Map</span>
        </div>
        <span style={{ fontSize: "11px", color: "#64748b", fontFamily: "monospace" }}>{sections.length} Sections Analyzed</span>
      </div>

      <div style={{ display: "flex", height: "8px", borderRadius: "4px", overflow: "hidden", marginBottom: "14px", background: "#1e293b", gap: "2px" }}>
        {sections.map((sec) => {
          const color = sec.score >= 0.7 ? "#f43f5e" : sec.score >= 0.4 ? "#f59e0b" : "#10b981";
          return (
            <div
              key={sec.index}
              style={{
                flex: 1,
                background: color,
                transition: "opacity 0.2s ease",
              }}
              title={`Section ${sec.index + 1}: ${Math.round(sec.score * 100)}% AI`}
            />
          );
        })}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "220px", overflowY: "auto" }}>
        {sections.map((sec) => {
          const badgeColor = sec.score >= 0.7 ? "#f43f5e" : sec.score >= 0.4 ? "#f59e0b" : "#10b981";
          const label = sec.score >= 0.7 ? "AI Generated" : sec.score >= 0.4 ? "Mixed" : "Human";
          return (
            <div
              key={sec.index}
              style={{
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "space-between",
                padding: "8px 10px",
                background: "#07090e",
                borderRadius: "4px",
                borderLeft: `3px solid ${badgeColor}`,
                fontSize: "12px",
                gap: "10px",
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
                  <span style={{ fontWeight: 700, color: "#cbd5e1" }}>Sec #{sec.index + 1}</span>
                  <span style={{ fontSize: "10px", color: "#64748b" }}>({sec.word_count} words)</span>
                </div>
                <p style={{ color: "#94a3b8", fontSize: "11px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", margin: 0 }}>
                  {sec.preview}
                </p>
              </div>
              <span
                style={{
                  fontSize: "10px",
                  fontWeight: 700,
                  color: badgeColor,
                  background: `${badgeColor}1a`,
                  padding: "2px 6px",
                  borderRadius: "3px",
                  whiteSpace: "nowrap",
                }}
              >
                {Math.round(sec.score * 100)}% {label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DossierExportAction({ jobId }: { jobId?: string }) {
  if (!jobId) return null;

  return (
    <div style={{ marginTop: "16px" }}>
      <a
        href={`/api/v1/jobs/${jobId}/dossier`}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "8px",
          width: "100%",
          padding: "10px 16px",
          background: "rgba(0, 229, 255, 0.08)",
          border: "1px solid rgba(0, 229, 255, 0.3)",
          color: "#00e5ff",
          borderRadius: "6px",
          fontWeight: 700,
          fontSize: "12px",
          letterSpacing: "0.03em",
          textDecoration: "none",
          textTransform: "uppercase",
          transition: "all 0.2s ease",
        }}
      >
        <FileText size={15} />
        <span>Export Cryptographic Dossier (PDF / HTML)</span>
      </a>
    </div>
  );
}

export function ResultPanel({
  result,
  loading,
  error,
}: {
  result?: DetectionResult;
  loading: boolean;
  error?: string;
}) {
  const resultContainerRef = useRef<HTMLDivElement>(null);

  // GSAP: Cascading animation timeline whenever a new result is mounted
  useEffect(() => {
    if (!result || loading || !resultContainerRef.current) return;

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: "power2.out" } });

      tl.from(".score-card", {
        y: 16,
        opacity: 0,
        duration: 0.4,
      })
        .from(
          ".provenance-badges > div",
          {
            y: 10,
            opacity: 0,
            scale: 0.95,
            duration: 0.3,
            stagger: 0.06,
          },
          "-=0.2"
        )
        .from(
          ".attribution-card",
          {
            y: 18,
            opacity: 0,
            duration: 0.4,
          },
          "-=0.15"
        )
        .from(
          ".signal",
          {
            y: 14,
            opacity: 0,
            duration: 0.3,
            stagger: 0.04,
          },
          "-=0.2"
        )
        .from(
          ".signal-meter > div",
          {
            width: "0%",
            duration: 0.6,
            stagger: 0.04,
          },
          "-=0.2"
        )
        .from(
          ".heatmap, .video-timeline-panel, .sentence-highlights, .hint",
          {
            y: 12,
            opacity: 0,
            duration: 0.35,
            stagger: 0.08,
          },
          "-=0.2"
        );
    }, resultContainerRef);

    return () => ctx.revert();
  }, [result, loading]);

  return (
    <aside className="result-panel">
      <AnimatePresence mode="wait">
        {loading && (
          <motion.div
            key="loading"
            className="scanning-wrapper"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
          >
            <ScanningHUD />
          </motion.div>
        )}

        {!loading && error && (
          <motion.div
            key="error"
            className="empty-state error-state"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <AlertTriangle size={32} className="text-red-400 mb-2" />
            <strong className="text-red-400">Analysis Error</strong>
            <span className="text-gray-400 text-sm">{error}</span>
            <span className="text-xs text-gray-500 mt-2">Please try submitting your sample again.</span>
          </motion.div>
        )}

        {!loading && !error && !result && (
          <motion.div
            key="empty"
            className="empty-state"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <ThreeDForensicScanner size="sm" interactive={true} />
            <strong className="text-gray-300" style={{ marginTop: 12 }}>Awaiting Sample</strong>
            <span className="text-gray-500 text-sm">Upload an image, video, or paste text to initialize real-time neural detection</span>
          </motion.div>
        )}

        {result && !loading && (
          <div key="result" ref={resultContainerRef}>
            <ScoreCard score={result.overall_score} />
            <ProvenanceBadge signals={result.signals} />
            <LLMAttributionCard result={result} />
            <AttributionCard result={result} />
            <SignalGrid signals={result.signals} />
            <DocumentSectionsHeatmap sections={result.section_scores} />
            <HeatmapOverlay heatmap={result.heatmap_b64} gradcamHeatmap={result.gradcam_heatmap_b64} prnuHeatmap={result.prnu_heatmap_b64} />
            <VideoTimeline result={result} />
            <SentenceHighlight sentences={result.sentences} />
            <ModelHint hint={result.model_hint} />
            <DossierExportAction jobId={result.job_id} />
          </div>
        )}
      </AnimatePresence>
    </aside>
  );
}
