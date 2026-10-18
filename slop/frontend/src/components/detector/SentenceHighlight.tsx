import { useState } from "react";
import { AlertCircle, CheckCircle, HelpCircle, Sparkles, Filter } from "lucide-react";
import type { SentenceScore } from "../../types";

type FilterMode = "all" | "ai" | "warm" | "human";

function getThreatCategory(prob: number): "ai" | "warm" | "human" {
  if (prob >= 0.70) return "ai";
  if (prob >= 0.40) return "warm";
  return "human";
}

function SentenceFilterBar({
  filterMode,
  setFilterMode,
  total,
  aiCount,
  warmCount,
  humanCount,
}: {
  filterMode: FilterMode;
  setFilterMode: (m: FilterMode) => void;
  total: number;
  aiCount: number;
  warmCount: number;
  humanCount: number;
}) {
  return (
    <div className="token-inspector-filters" role="group" aria-label="Sentence threat filters">
      <button type="button" className={`filter-btn ${filterMode === "all" ? "active" : ""}`} onClick={() => setFilterMode("all")}>
        <Filter size={12} />
        <span>All ({total})</span>
      </button>
      <button type="button" className={`filter-btn threat-ai ${filterMode === "ai" ? "active" : ""}`} onClick={() => setFilterMode("ai")}>
        <span className="dot red" />
        <span>AI ({aiCount})</span>
      </button>
      <button type="button" className={`filter-btn threat-warm ${filterMode === "warm" ? "active" : ""}`} onClick={() => setFilterMode("warm")}>
        <span className="dot amber" />
        <span>Mixed ({warmCount})</span>
      </button>
      <button type="button" className={`filter-btn threat-human ${filterMode === "human" ? "active" : ""}`} onClick={() => setFilterMode("human")}>
        <span className="dot green" />
        <span>Human ({humanCount})</span>
      </button>
    </div>
  );
}

function SentenceTokenSpan({
  sentence,
  index,
  isSelected,
  filterMode,
  onSelect,
}: {
  sentence: SentenceScore;
  index: number;
  isSelected: boolean;
  filterMode: FilterMode;
  onSelect: () => void;
}) {
  const category = getThreatCategory(sentence.ai_prob);
  const isDimmed = filterMode !== "all" && filterMode !== category;
  const threatClass = category === "ai" ? "hot" : category === "warm" ? "warm" : "safe";

  return (
    <span
      className={`sentence-token ${threatClass} ${isSelected ? "selected" : ""} ${isDimmed ? "dimmed" : ""}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter") onSelect();
      }}
      aria-pressed={isSelected}
    >
      {sentence.text}{" "}
    </span>
  );
}

function SentenceTelemetryTray({
  sentence,
  index,
}: {
  sentence: SentenceScore;
  index: number;
}) {
  const category = getThreatCategory(sentence.ai_prob);
  const label = category === "ai" ? "AI GENERATED" : category === "human" ? "HUMAN WRITING" : "MIXED STYLOMETRIC SIGNALS";

  return (
    <div className="sentence-telemetry-tray">
      <div className="tray-header">
        <div className="tray-status">
          {category === "ai" && <AlertCircle size={16} className="text-rose-400" />}
          {category === "human" && <CheckCircle size={16} className="text-emerald-400" />}
          {category === "warm" && <HelpCircle size={16} className="text-amber-400" />}
          <span className="status-label">Segment #{index + 1}: {label}</span>
        </div>
        <span className="tray-pct">{Math.round(sentence.ai_prob * 100)}% AI Confidence</span>
      </div>

      <div className="tray-metrics">
        <div className="metric-chip">
          <span className="label">Perplexity:</span>
          <span className="val">{sentence.perplexity.toFixed(1)}</span>
        </div>
        {sentence.model_ai_prob !== undefined && (
          <div className="metric-chip">
            <span className="label">Neural Output:</span>
            <span className="val">{Math.round(sentence.model_ai_prob * 100)}%</span>
          </div>
        )}
        <div className="metric-chip">
          <span className="label">Pattern Matched:</span>
          <span className="val">{sentence.pattern_matched ? "Yes (Frontier LLM)" : "None"}</span>
        </div>
      </div>

      <p className="tray-quote">"{sentence.text}"</p>
    </div>
  );
}

export function SentenceHighlight({ sentences }: { sentences?: SentenceScore[] }) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [filterMode, setFilterMode] = useState<FilterMode>("all");

  if (!sentences?.length) return null;

  const total = sentences.length;
  const aiCount = sentences.filter((s) => s.ai_prob >= 0.70).length;
  const humanCount = sentences.filter((s) => s.ai_prob < 0.40).length;
  const warmCount = total - aiCount - humanCount;
  const selectedSentence = selectedIdx !== null ? sentences[selectedIdx] : null;

  return (
    <div className="forensic-token-inspector">
      <div className="token-inspector-header">
        <div className="token-inspector-title">
          <Sparkles size={16} className="text-rose-400" />
          <span>INTERACTIVE SENTENCE FORENSICS</span>
          <span className="token-inspector-count">{total} SEGMENTS PROCESSED</span>
        </div>
        <SentenceFilterBar
          filterMode={filterMode}
          setFilterMode={setFilterMode}
          total={total}
          aiCount={aiCount}
          warmCount={warmCount}
          humanCount={humanCount}
        />
      </div>

      <div className="sentence-stream-container">
        <p className="sentence-stream">
          {sentences.map((sentence, index) => (
            <SentenceTokenSpan
              key={`${sentence.text.slice(0, 16)}-${index}`}
              sentence={sentence}
              index={index}
              isSelected={selectedIdx === index}
              filterMode={filterMode}
              onSelect={() => setSelectedIdx(selectedIdx === index ? null : index)}
            />
          ))}
        </p>
      </div>

      {selectedSentence && selectedIdx !== null && (
        <SentenceTelemetryTray sentence={selectedSentence} index={selectedIdx} />
      )}
    </div>
  );
}
