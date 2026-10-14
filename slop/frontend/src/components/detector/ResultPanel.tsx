import { motion, AnimatePresence } from "framer-motion";
import { lazy, Suspense, useEffect, useRef } from "react";
import gsap from "gsap";
import { Shield, ShieldCheck, Brain, Eye, Cpu, Sparkles, AlertTriangle, Fingerprint, FileText, Layers, Scale, TrendingDown, TrendingUp, CircleCheck, ChevronDown } from "lucide-react";
import type { DetectionResult, MixedAuthorshipInfo } from "../../types";
import { ScanningHUD } from "./ScanningHUD";
import { HeatmapOverlay } from "./HeatmapOverlay";
import { ScoreCard } from "./ScoreCard";
import { SentenceHighlight } from "./SentenceHighlight";
import { SignalGrid } from "./SignalGrid";
import { VideoTimeline } from "./VideoTimeline";

// Code-split: the animated 3D scanner (and its animejs dependency) only loads
// when the empty state is actually rendered. Styled shimmer fallback keeps
// layout stable — no visual/behavior change, smaller initial bundle.
const ThreeDForensicScanner = lazy(() =>
  import("../ui/ThreeDForensicScanner").then((module) => ({ default: module.ThreeDForensicScanner }))
);

/**
 * Calibrated confidence band chip for text results.
 *
 * The band and its zone edges come exclusively from the API response
 * (`confidence_band` + `zone_low`/`zone_high`, derived server-side from the
 * promoted calibrator artifact). No band decision is made in the frontend.
 */
function ConfidenceBandChip({ result }: { result: DetectionResult }) {
  const band = result.confidence_band ?? (result.signals?.confidence_band as string | undefined);
  if (!band) return null;

  const zoneLow = result.zone_low;
  const zoneHigh = result.zone_high;
  const bandLabels: Record<string, string> = {
    DEFINITELY_HUMAN: "Clear Human",
    LIKELY_HUMAN: "Likely Human",
    UNCERTAIN: "Ambiguous",
    LIKELY_AI: "Likely AI",
    DEFINITELY_AI: "Clear AI",
  };
  const label = bandLabels[band] ?? band;
  const tone =
    band === "DEFINITELY_HUMAN"
      ? "human"
      : band === "LIKELY_HUMAN"
        ? "likely-human"
        : band === "UNCERTAIN"
          ? "ambiguous"
          : band === "LIKELY_AI"
            ? "likely-ai"
            : "ai";

  return (
    <div className="confidence-band-chip-wrap" role="status" aria-label={`Calibrated confidence band: ${label}`}>
      <span className={`confidence-band-chip ${tone}`}>
        <CircleCheck size={14} />
        <span>{label}</span>
      </span>
      {zoneLow !== undefined && zoneHigh !== undefined && (
        <span className="band-zone-edges">
          AI zone begins at {Math.round(zoneLow * 100)}% · Clear AI at {Math.round(zoneHigh * 100)}%
        </span>
      )}
    </div>
  );
}

function ThreeDForensicScannerFallback() {
  return (
    <Suspense fallback={<div className="scanner-suspense-fallback" aria-hidden="true" />}>
      <ThreeDForensicScanner size="sm" interactive={true} />
    </Suspense>
  );
}

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

function HeuristicOverrideNotice({ result }: { result: DetectionResult }) {
  const explanation =
    result.heuristic_override_explanation ||
    (result.signals?.heuristic_override_explanation as string | undefined);

  const rawNeural =
    (result.signals?.model_ai_probability ??
      result.signals?.neural_ai_score) as number | undefined;

  // Material override threshold: |overall - neural| > 0.30 OR explicit explanation exists
  const hasMaterialDelta =
    rawNeural !== undefined && Math.abs(result.overall_score - rawNeural) > 0.30;

  if (!explanation && !hasMaterialDelta) return null;

  const neuralPct = rawNeural !== undefined ? Math.round(rawNeural * 100) : null;
  const finalPct = Math.round(result.overall_score * 100);
  const delta = rawNeural !== undefined ? result.overall_score - rawNeural : 0;
  const isDownward = delta < 0;

  const defaultExplanation = isDownward
    ? `Neural model: ${neuralPct ?? ""}% AI-like, but corrected downward: forensic heuristics identified authentic physical sensor or drawing characteristics.`
    : `Neural model: ${neuralPct ?? ""}% AI-like, but corrected upward: forensic spectral harmonics and compression artifacts indicate synthetic generation.`;

  const displayText = explanation || defaultExplanation;

  return (
    <div
      className={`heuristic-override-card ${isDownward ? "downward" : "upward"}`}
      role="region"
      aria-label="Forensic Heuristic Signal Reconciliation"
    >
      <div className="override-header">
        <div className="override-title-wrap">
          <Scale size={15} className={isDownward ? "text-emerald-400" : "text-amber-400"} />
          <span className="override-heading">Heuristic Signal Reconciliation</span>
        </div>
        <span className={`override-direction-badge ${isDownward ? "downward" : "upward"}`}>
          {isDownward ? (
            <>
              <TrendingDown size={12} />
              <span>Corrected Downward</span>
            </>
          ) : (
            <>
              <TrendingUp size={12} />
              <span>Corrected Upward</span>
            </>
          )}
        </span>
      </div>

      <p className="override-explanation-text">{displayText}</p>

      {neuralPct !== null && (
        <div className="override-comparison-strip">
          <div className="comparison-col">
            <span className="col-label">Neural Model</span>
            <span className="col-value neural">{neuralPct}%</span>
          </div>
          <div className="comparison-arrow">➔</div>
          <div className="comparison-col">
            <span className="col-label">Forensic Delta</span>
            <span className={`col-value ${isDownward ? "downward" : "upward"}`}>
              {delta >= 0 ? "+" : ""}{Math.round(delta * 100)}%
            </span>
          </div>
          <div className="comparison-arrow">➔</div>
          <div className="comparison-col">
            <span className="col-label">Final Consensus</span>
            <span className="col-value final">{finalPct}%</span>
          </div>
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

function ModelHint({ hint, explanation }: { hint?: string; explanation?: string }) {
  if (!hint) return null;
  let cleanHint = hint;
  if (explanation && cleanHint.startsWith(explanation)) {
    cleanHint = cleanHint.slice(explanation.length).trim();
  }
  if (!cleanHint) return null;
  return (
    <p className="hint">
      {cleanHint}
    </p>
  );
}

function DocumentSectionsHeatmap({ sections }: { sections?: DetectionResult["section_scores"] }) {
  if (!sections || sections.length === 0) return null;

  return (
    <div className="document-sections-card" style={{ marginTop: "16px", background: "var(--bg-card)", border: "1px solid var(--border-card)", borderRadius: "8px", padding: "16px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Layers size={16} className="text-cyan-400" />
          <span style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-main)" }}>Document Section Threat Map</span>
        </div>
        <span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "monospace" }}>{sections.length} Sections Analyzed</span>
      </div>

      <div style={{ display: "flex", height: "8px", borderRadius: "4px", overflow: "hidden", marginBottom: "14px", background: "var(--border-card)", gap: "2px" }}>
        {sections.map((sec) => {
          const color = sec.score > 0.70 ? "#f43f5e" : sec.score >= 0.35 ? "#f59e0b" : "#10b981";
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
          const badgeColor = sec.score > 0.70 ? "#f43f5e" : sec.score >= 0.35 ? "#f59e0b" : "#10b981";
          const label = sec.score > 0.70 ? "AI Generated" : sec.score >= 0.35 ? "Mixed" : "Human";
          return (
            <div
              key={sec.index}
              style={{
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "space-between",
                padding: "8px 10px",
                background: "var(--bg-page)",
                borderRadius: "4px",
                borderLeft: `3px solid ${badgeColor}`,
                fontSize: "12px",
                gap: "10px",
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
                  <span style={{ fontWeight: 700, color: "var(--text-main)" }}>Sec #{sec.index + 1}</span>
                  <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>({sec.word_count} words)</span>
                </div>
                <p style={{ color: "var(--text-muted)", fontSize: "11px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", margin: 0 }}>
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

function ConsolidatedVerdictPill({ score }: { score: number }) {
  let label: string;
  let tone: "ai" | "authentic" | "uncertain";
  let Icon = AlertTriangle;

  if (score > 0.70) {
    label = "AI Detected";
    tone = "ai";
    Icon = AlertTriangle;
  } else if (score < 0.35) {
    label = "Authentic";
    tone = "authentic";
    Icon = ShieldCheck;
  } else {
    label = "Uncertain";
    tone = "uncertain";
    Icon = Scale;
  }

  return (
    <div className="consolidated-verdict-wrap" role="status" aria-label={`Consolidated verdict: ${label}`}>
      <span className={`consolidated-pill pill-${tone}`}>
        <Icon size={14} className="consolidated-pill-icon" />
        <span>{label}</span>
      </span>
    </div>
  );
}

function CodeVerdictBadge({ result }: { result: DetectionResult }) {
  const isCode = result.signals?.is_code;
  if (isCode !== 1.0 && isCode !== 1) return null;

  const score = result.overall_score;
  const isHuman = score < 0.20;
  const isAi = score > 0.70;

  const badgeText = isHuman
    ? "MANUAL CODED (HUMAN)"
    : isAi
    ? "VIBE CODED (AI)"
    : "MIXED / UNVERIFIED CODE";

  const badgeClass = isHuman ? "code-badge-human" : isAi ? "code-badge-ai" : "code-badge-mixed";

  return (
    <div className="code-verdict-container" role="status" aria-label={`Code Forensics Verdict: ${badgeText}`}>
      <span className={`code-verdict-badge ${badgeClass}`}>
        <span>{badgeText}</span>
      </span>
      <p className="code-verdict-subtitle">
        Code Forensics Active: Evaluated via AST &amp; Identifier Structure.
      </p>
    </div>
  );
}

function MixedAuthorshipBadge({ result }: { result: DetectionResult }) {
  const info = result.mixed_authorship || (result.signals?.mixed_authorship as MixedAuthorshipInfo | undefined);
  if (!info || !info.is_mixed_authorship) return null;

  const aiPct = Math.round(info.ai_sentence_ratio * 100);
  const humanPct = 100 - aiPct;

  return (
    <div className="mixed-authorship-card" role="status" aria-label="Mixed Authorship Detected">
      <div className="mixed-authorship-header">
        <Scale size={14} className="text-amber-400" />
        <span className="mixed-authorship-title">Mixed Authorship Detected</span>
        <span className="mixed-authorship-badge">{info.transitions} transitions</span>
      </div>
      <p className="mixed-authorship-desc">
        Document exhibits transitions between AI-drafted ({aiPct}%) and human-written ({humanPct}%) content blocks.
      </p>
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
          ".consolidated-verdict-wrap",
          {
            y: 8,
            opacity: 0,
            duration: 0.25,
          },
          "-=0.2"
        )
        .from(
          ".technical-signals-drawer",
          {
            y: 12,
            opacity: 0,
            duration: 0.3,
          },
          "-=0.15"
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
            <ThreeDForensicScannerFallback />
            <strong className="text-gray-300" style={{ marginTop: 12 }}>Awaiting Sample</strong>
            <span className="text-gray-500 text-sm">Upload an image, video, or paste text to initialize real-time neural detection</span>
          </motion.div>
        )}

        {result && !loading && (
          <div key="result" ref={resultContainerRef} className="result-container-flow">
            <ScoreCard score={result.overall_score} />
            <ConsolidatedVerdictPill score={result.overall_score} />
            <CodeVerdictBadge result={result} />
            <MixedAuthorshipBadge result={result} />

            {/* Collapsible Technical Signals & Forensic Drawer */}
            <details className="technical-signals-drawer">
              <summary className="technical-signals-summary">
                <div className="summary-left">
                  <Layers size={14} className="summary-icon" />
                  <span>Technical Signals & Forensic Breakdown</span>
                </div>
                <ChevronDown size={14} className="summary-chevron" />
              </summary>
              <div className="technical-signals-content">
                <ConfidenceBandChip result={result} />
                <ProvenanceBadge signals={result.signals} />
                <HeuristicOverrideNotice result={result} />
                <LLMAttributionCard result={result} />
                <AttributionCard result={result} />
                <SignalGrid signals={result.signals} />
              </div>
            </details>

            <DocumentSectionsHeatmap sections={result.section_scores} />
            <HeatmapOverlay heatmap={result.heatmap_b64} gradcamHeatmap={result.gradcam_heatmap_b64} prnuHeatmap={result.prnu_heatmap_b64} />
            <VideoTimeline result={result} />
            <SentenceHighlight sentences={result.sentences} />
            <ModelHint
              hint={result.model_hint}
              explanation={
                result.heuristic_override_explanation ||
                (result.signals?.heuristic_override_explanation as string | undefined)
              }
            />
            <DossierExportAction jobId={result.job_id} />
          </div>
        )}
      </AnimatePresence>
    </aside>
  );
}
