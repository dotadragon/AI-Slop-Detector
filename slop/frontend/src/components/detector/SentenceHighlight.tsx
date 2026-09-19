import { useState } from "react";
import { AlertCircle, CheckCircle, HelpCircle, Sparkles, Filter } from "lucide-react";
import type { SentenceScore } from "../../types";

export function SentenceHighlight({ sentences }: { sentences?: SentenceScore[] }) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [filterMode, setFilterMode] = useState<"all" | "ai" | "warm" | "human">("all");

  if (!sentences?.length) return null;

  const total = sentences.length;
  const aiCount = sentences.filter((s) => s.ai_prob >= 0.70).length;
  const humanCount = sentences.filter((s) => s.ai_prob < 0.40).length;
  const warmCount = total - aiCount - humanCount;

  const selectedSentence = selectedIdx !== null ? sentences[selectedIdx] : null;

  return (
    <div className="forensic-token-inspector">
      <div className="inspector-header">
        <div className="inspector-title">
          <Sparkles size={16} className="text-rose-400" />
          <span>INTERACTIVE SENTENCE FORENSICS</span>
          <span className="inspector-count">{total} SEGMENTS PROCESSED</span>
        </div>

        <div className="inspector-filters" role="group" aria-label="Sentence threat filters">
          <button
            type="button"
            className={`filter-btn ${filterMode === "all" ? "active" : ""}`}
            onClick={() => setFilterMode("all")}
          >
            <Filter size={12} />
            <span>All ({total})</span>
          </button>
          <button
            type="button"
            className={`filter-btn threat-ai ${filterMode === "ai" ? "active" : ""}`}
            onClick={() => setFilterMode("ai")}
          >
            <span className="dot red" />
            <span>AI ({aiCount})</span>
          </button>
          <button
            type="button"
            className={`filter-btn threat-warm ${filterMode === "warm" ? "active" : ""}`}
            onClick={() => setFilterMode("warm")}
          >
            <span className="dot amber" />
            <span>Mixed ({warmCount})</span>
          </button>
          <button
            type="button"
            className={`filter-btn threat-human ${filterMode === "human" ? "active" : ""}`}
            onClick={() => setFilterMode("human")}
          >
            <span className="dot green" />
            <span>Human ({humanCount})</span>
          </button>
        </div>
      </div>

      {/* Interactive Sentence Text Flow */}
      <div className="sentence-stream-container">
        <p className="sentence-stream">
          {sentences.map((sentence, index) => {
            const isAi = sentence.ai_prob >= 0.70;
            const isWarm = sentence.ai_prob >= 0.40 && sentence.ai_prob < 0.70;
            const isHuman = sentence.ai_prob < 0.40;

            const isDimmed =
              (filterMode === "ai" && !isAi) ||
              (filterMode === "warm" && !isWarm) ||
              (filterMode === "human" && !isHuman);

            const isSelected = selectedIdx === index;
            const threatClass = isAi ? "hot" : isWarm ? "warm" : "safe";

            return (
              <span
                key={`${sentence.text.slice(0, 16)}-${index}`}
                className={`sentence-token ${threatClass} ${isSelected ? "selected" : ""} ${isDimmed ? "dimmed" : ""}`}
                onClick={() => setSelectedIdx(isSelected ? null : index)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && setSelectedIdx(isSelected ? null : index)}
                aria-pressed={isSelected}
              >
                {sentence.text}{" "}
              </span>
            );
          })}
        </p>
      </div>

      {/* Selected Diagnostic Telemetry Tray */}
      {selectedSentence && (
        <div className="sentence-telemetry-tray">
          <div className="tray-header">
            <div className="tray-status">
              {selectedSentence.ai_prob >= 0.70 ? (
                <AlertCircle size={16} className="text-rose-400" />
              ) : selectedSentence.ai_prob < 0.40 ? (
                <CheckCircle size={16} className="text-emerald-400" />
              ) : (
                <HelpCircle size={16} className="text-amber-400" />
              )}
              <span className="status-label">
                Segment #{selectedIdx! + 1}:{" "}
                {selectedSentence.ai_prob >= 0.70
                  ? "AI GENERATED"
                  : selectedSentence.ai_prob < 0.40
                  ? "HUMAN WRITING"
                  : "MIXED STYLOMETRIC SIGNALS"}
              </span>
            </div>
            <span className="tray-pct">{Math.round(selectedSentence.ai_prob * 100)}% AI Confidence</span>
          </div>

          <div className="tray-metrics">
            <div className="metric-chip">
              <span className="label">Perplexity:</span>
              <span className="val">{selectedSentence.perplexity.toFixed(1)}</span>
            </div>
            {selectedSentence.model_ai_prob !== undefined && (
              <div className="metric-chip">
                <span className="label">Neural Output:</span>
                <span className="val">{Math.round(selectedSentence.model_ai_prob * 100)}%</span>
              </div>
            )}
            <div className="metric-chip">
              <span className="label">Pattern Matched:</span>
              <span className="val">{selectedSentence.pattern_matched ? "Yes (Frontier LLM)" : "None"}</span>
            </div>
          </div>

          <p className="tray-quote">"{selectedSentence.text}"</p>
        </div>
      )}
    </div>
  );
}
