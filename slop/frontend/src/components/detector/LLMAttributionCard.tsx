import { Fingerprint } from "lucide-react";
import type { DetectionResult } from "../../types";

export function LLMAttributionCard({ result }: { result: DetectionResult }) {
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
          {Object.entries(attr.scores).map(([llmModel, modelScore]) => (
            <div key={llmModel} className={`llm-score-row ${modelScore > 0 ? "matched" : ""}`}>
              <span>{llmModel}</span>
              <span className="llm-score-count">{modelScore} pattern{modelScore !== 1 ? "s" : ""}</span>
            </div>
          ))}
        </div>

        {attr.all_patterns && attr.all_patterns.length > 0 && (
          <div className="llm-patterns-list">
            <span className="patterns-label">Matched patterns:</span>
            {attr.all_patterns.slice(0, 5).map((patternText) => (
              <code key={patternText} className="pattern-tag">{patternText}</code>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
