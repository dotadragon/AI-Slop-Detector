import React from "react";
import { scoreColor } from "../../utils/validators";

const PERCENT_SIGNAL_KEYS = new Set([
  "compression_artifact_score",
  "frequency_artifact_score",
  "noise_uniformity_score",
  "metadata_risk",
  "heuristic_score",
  "model_ai_probability",
  "multi_crop_ai_probability",
  // v3.0: new forensic signals
  "cfa_score",
  "dct_block_score",
  // v3.0: text signals
  "neural_ai_score",
  "llm_pattern_score",
  "contraction_avoidance",
  "vocabulary_richness",
  "hedging_ratio",
  "passive_voice_score",
  "zipf_deviation",
  "punctuation_entropy",
  "list_structure_score",
  "sentence_start_entropy",
  "paragraph_uniformity",
  "stylometry_score",
  "uniformity",
]);

const SIGNAL_ORDER = [
  // Primary scores
  "model_ai_probability",
  "neural_ai_score",
  "multi_crop_ai_probability",
  "heuristic_score",
  "llm_pattern_score",
  // Image forensics
  "cfa_score",
  "dct_block_score",
  "compression_artifact_score",
  "frequency_artifact_score",
  "noise_uniformity_score",
  "metadata_risk",
  // Text stylometrics
  "contraction_avoidance",
  "vocabulary_richness",
  "hedging_ratio",
  "passive_voice_score",
  "zipf_deviation",
  "punctuation_entropy",
  "list_structure_score",
  "sentence_start_entropy",
  "paragraph_uniformity",
  "stylometry_score",
  "uniformity",
  // Meta
  "exif_flags",
  "inference_mode",
  "provenance",
  "model_id",
  "model_error",
  "file_name",
];

// v3.0: Human-readable labels for signal keys
const SIGNAL_LABELS: Record<string, string> = {
  "model_ai_probability": "Neural AI Score",
  "neural_ai_score": "Neural AI Score",
  "multi_crop_ai_probability": "Multi-Crop AI Score",
  "heuristic_score": "Heuristic Score",
  "llm_pattern_score": "LLM Pattern Score",
  "cfa_score": "CFA Sensor Analysis",
  "dct_block_score": "DCT Block Analysis",
  "compression_artifact_score": "Compression Artifacts (ELA)",
  "frequency_artifact_score": "Frequency Artifacts (FFT)",
  "noise_uniformity_score": "Noise Uniformity",
  "metadata_risk": "Metadata Risk",
  "contraction_avoidance": "Contraction Avoidance",
  "vocabulary_richness": "Vocabulary Richness",
  "hedging_ratio": "Hedging Language",
  "passive_voice_score": "Passive Voice",
  "zipf_deviation": "Zipf's Law Deviation",
  "punctuation_entropy": "Punctuation Predictability",
  "list_structure_score": "List/Structure Score",
  "sentence_start_entropy": "Sentence Start Entropy",
  "paragraph_uniformity": "Paragraph Uniformity",
  "stylometry_score": "Stylometry Score",
  "uniformity": "Text Uniformity",
};

// v3.0: Hide internal/redundant signals from display
const HIDDEN_SIGNALS = new Set([
  "model_scores",
  "crop_scores",
  "neural_weight",
  "heuristic_weight",
  "disagreement",
  "generator_attribution",
  "generator_confidence",
]);

function labelForScore(score: number) {
  if (score >= 0.70) return "High";
  if (score >= 0.35) return "Medium";
  return "Low";
}

function formatKey(key: string) {
  return SIGNAL_LABELS[key] || key.replaceAll("_", " ");
}

function formatRawValue(value: unknown) {
  if (value === null || value === undefined) return "none";
  if (Array.isArray(value)) return value.join(", ") || "none";
  if (typeof value === "object") {
    const sig = value as { detected?: boolean; signature_type?: string; generator?: string };
    if (sig.detected) return `${sig.signature_type || "Detected"} (${sig.generator || "AI"})`;
    return "None detected";
  }
  return String(value);
}

function orderedEntries(signals: Record<string, unknown>) {
  const priority = new Map(SIGNAL_ORDER.map((key, index) => [key, index]));
  return Object.entries(signals)
    .filter(([key]) => !HIDDEN_SIGNALS.has(key))
    .sort(([a], [b]) => {
      const ai = priority.get(a) ?? Number.MAX_SAFE_INTEGER;
      const bi = priority.get(b) ?? Number.MAX_SAFE_INTEGER;
      return ai === bi ? a.localeCompare(b) : ai - bi;
    });
}

export function SignalGrid({ signals }: { signals?: Record<string, unknown> }) {
  if (!signals) return null;
  return (
    <div className="signal-grid">
      {orderedEntries(signals).map(([key, value]) => {
        const numericValue = typeof value === "number" ? value : Number.NaN;
        const showPercent = PERCENT_SIGNAL_KEYS.has(key) && Number.isFinite(numericValue);
        const percent = Math.round(numericValue * 100);

        return (
          <div key={key} className={`signal ${showPercent ? "signal-metered" : ""}`}>
            <span>{formatKey(key)}</span>
            {showPercent ? (
              <>
                <div className="signal-value-row">
                  <strong>{percent}%</strong>
                  <em>{labelForScore(numericValue)}</em>
                </div>
                <div className="signal-meter" aria-hidden="true">
                  <div style={{ width: `${percent}%`, background: scoreColor(numericValue) }} />
                </div>
              </>
            ) : (
              <strong>{formatRawValue(value)}</strong>
            )}
          </div>
        );
      })}
    </div>
  );
}
