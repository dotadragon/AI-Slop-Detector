import {
  Shield,
  ShieldCheck,
  Brain,
  Eye,
  Cpu,
  Sparkles,
  Scale,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import type { DetectionResult, GeneratorAttribution, DigitalSignatureInfo } from "../../types";

const INFERENCE_BADGES: Record<string, { label: string; isFrontier?: boolean; isHeuristic?: boolean }> = {
  neural: { label: "Neural Vision Transformer" },
  transformers_image_classifier: { label: "Neural Vision Transformer" },
  neural_multi_crop: { label: "Neural Multi-Crop v3.0" },
  calibrated_frontier_ensemble: { label: "Calibrated Frontier Ensemble v3.4", isFrontier: true },
  roberta_classifier: { label: "RoBERTa Neural v3.0" },
  heuristic_fallback: { label: "Heuristic Analysis v3.0", isHeuristic: true },
};

function ProvenanceHeader({ provenance }: { provenance?: string }) {
  if (provenance === "verified") {
    return (
      <div className="provenance-badge verified">
        <ShieldCheck size={16} />
        <span>Verified Provenance (C2PA)</span>
      </div>
    );
  }
  if (provenance === "none") {
    return (
      <div className="provenance-badge none">
        <Shield size={14} />
        <span>No C2PA provenance header</span>
      </div>
    );
  }
  return null;
}

function InferenceModeBadge({ inferenceMode }: { inferenceMode?: string }) {
  const badgeInfo = inferenceMode ? INFERENCE_BADGES[inferenceMode] : undefined;
  if (!badgeInfo) return null;

  const modeClass = badgeInfo.isHeuristic ? "heuristic" : "neural";
  const frontierClass = badgeInfo.isFrontier ? "frontier" : "";

  return (
    <div className={`inference-badge ${modeClass} ${frontierClass}`.trim()}>
      {badgeInfo.isHeuristic ? (
        <Eye size={14} />
      ) : (
        <Brain size={14} className={badgeInfo.isFrontier ? "text-cyan-400" : undefined} />
      )}
      <span>{badgeInfo.label}</span>
    </div>
  );
}

function PrnuBadge({ active }: { active?: boolean }) {
  if (!active) return null;
  return (
    <div className="inference-badge prnu">
      <Eye size={14} className="text-emerald-400" />
      <span>PRNU Sensor Active</span>
    </div>
  );
}

export function ProvenanceBadge({ signals }: { signals?: Record<string, unknown> }) {
  if (!signals) return null;
  const provenance = typeof signals.provenance === "string" ? signals.provenance : undefined;
  const inferenceMode = typeof signals.inference_mode === "string" ? signals.inference_mode : undefined;
  const isPrnuActive = Boolean(signals.prnu_sensor_residual_active);

  return (
    <div className="provenance-badges">
      <ProvenanceHeader provenance={provenance} />
      <InferenceModeBadge inferenceMode={inferenceMode} />
      <PrnuBadge active={isPrnuActive} />
    </div>
  );
}

function OverrideComparisonStrip({
  neuralPct,
  delta,
  finalPct,
  isDownward,
}: {
  neuralPct: number;
  delta: number;
  finalPct: number;
  isDownward: boolean;
}) {
  const signPrefix = delta >= 0 ? "+" : "";
  const deltaText = `${signPrefix}${Math.round(delta * 100)}%`;
  const toneClass = isDownward ? "downward" : "upward";

  return (
    <div className="override-comparison-strip">
      <div className="comparison-col">
        <span className="col-label">Neural Model</span>
        <span className="col-value neural">{neuralPct}%</span>
      </div>
      <div className="comparison-arrow">➔</div>
      <div className="comparison-col">
        <span className="col-label">Forensic Delta</span>
        <span className={`col-value ${toneClass}`}>{deltaText}</span>
      </div>
      <div className="comparison-arrow">➔</div>
      <div className="comparison-col">
        <span className="col-label">Final Consensus</span>
        <span className="col-value final">{finalPct}%</span>
      </div>
    </div>
  );
}

function resolveExplanation(explanation?: string, isDownward?: boolean, neuralPct?: number | null): string {
  if (explanation) return explanation;
  if (isDownward) {
    return `Neural model: ${neuralPct ?? ""}% AI-like, but corrected downward: forensic heuristics identified authentic physical sensor or drawing characteristics.`;
  }
  return `Neural model: ${neuralPct ?? ""}% AI-like, but corrected upward: forensic spectral harmonics and compression artifacts indicate synthetic generation.`;
}

interface OverrideData {
  explanation: string;
  neuralPct: number | null;
  finalPct: number;
  delta: number;
  isDownward: boolean;
}

function extractOverrideData(result: DetectionResult): OverrideData | null {
  const explanation =
    result.heuristic_override_explanation ||
    (result.signals?.heuristic_override_explanation as string | undefined);

  const rawNeural = (result.signals?.model_ai_probability ??
    result.signals?.neural_ai_score) as number | undefined;

  const delta = rawNeural !== undefined ? result.overall_score - rawNeural : 0;
  const isMaterial = rawNeural !== undefined && Math.abs(delta) > 0.30;
  if (!explanation && !isMaterial) return null;

  const neuralPct = rawNeural !== undefined ? Math.round(rawNeural * 100) : null;
  const isDownward = delta < 0;

  return {
    explanation: resolveExplanation(explanation, isDownward, neuralPct),
    neuralPct,
    finalPct: Math.round(result.overall_score * 100),
    delta,
    isDownward,
  };
}

function OverrideHeader({ isDownward }: { isDownward: boolean }) {
  const toneClass = isDownward ? "downward" : "upward";
  const iconClass = isDownward ? "text-emerald-400" : "text-amber-400";
  const labelText = isDownward ? "Corrected Downward" : "Corrected Upward";

  return (
    <div className="override-header">
      <div className="override-title-wrap">
        <Scale size={15} className={iconClass} />
        <span className="override-heading">Heuristic Signal Reconciliation</span>
      </div>
      <span className={`override-direction-badge ${toneClass}`}>
        {isDownward ? <TrendingDown size={12} /> : <TrendingUp size={12} />}
        <span>{labelText}</span>
      </span>
    </div>
  );
}

export function HeuristicOverrideNotice({ result }: { result: DetectionResult }) {
  const data = extractOverrideData(result);
  if (!data) return null;

  const toneClass = data.isDownward ? "downward" : "upward";

  return (
    <div className={`heuristic-override-card ${toneClass}`} role="region" aria-label="Forensic Heuristic Signal Reconciliation">
      <OverrideHeader isDownward={data.isDownward} />
      <p className="override-explanation-text">{data.explanation}</p>
      {data.neuralPct !== null && (
        <OverrideComparisonStrip
          neuralPct={data.neuralPct}
          delta={data.delta}
          finalPct={data.finalPct}
          isDownward={data.isDownward}
        />
      )}
    </div>
  );
}

function AttributionSignatureBlock({ signature }: { signature?: DigitalSignatureInfo }) {
  if (!signature?.detected) return null;

  return (
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
  );
}

function resolveAttributionDetails(attribution?: GeneratorAttribution, signature?: DigitalSignatureInfo): string | null {
  const detailsStr = signature?.details || (typeof attribution?.details === "string" ? attribution.details : null);
  if (!detailsStr || detailsStr.includes("Vision Transformer classified")) return null;
  return detailsStr;
}

function extractAttributionData(result: DetectionResult) {
  const attribution = result.generator_attribution;
  const signature = result.digital_signature;

  const generator = attribution?.generator;
  const isGeneric = generator ? generator.includes("Standard Photo") || generator.includes("Diffusion Neural Engine") : false;
  const hasAttribution = Boolean(signature?.detected) || Boolean(generator && !isGeneric);
  if (!hasAttribution) return null;

  const generatorName = generator || signature?.generator || "AI Engine";
  const confidence = Math.round((attribution?.confidence || signature?.confidence || 0.8) * 100);
  const details = resolveAttributionDetails(attribution, signature);

  return { generatorName, confidence, details, signature };
}

export function AttributionCard({ result }: { result: DetectionResult }) {
  const data = extractAttributionData(result);
  if (!data) return null;

  return (
    <div className="attribution-card">
      <div className="attribution-header">
        <div className="attribution-title">
          <Cpu size={16} className="text-amber-400" />
          <span>AI Engine Attribution &amp; Forensics</span>
        </div>
        <span className="attribution-badge">{data.confidence}% Match</span>
      </div>

      <div className="attribution-body">
        <div className="attribution-engine">
          <span className="engine-label">Probable Generator:</span>
          <span className="engine-name">{data.generatorName}</span>
        </div>

        <AttributionSignatureBlock signature={data.signature} />

        {data.details && <div className="attribution-details">{data.details}</div>}
      </div>
    </div>
  );
}
