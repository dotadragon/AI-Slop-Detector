import { Layers, FileText, Scale } from "lucide-react";
import type { DetectionResult, MixedAuthorshipInfo } from "../../types";

export function ModelHint({ hint, explanation }: { hint?: string; explanation?: string }) {
  if (!hint) return null;
  let cleanHint = hint;
  if (explanation && cleanHint.startsWith(explanation)) {
    cleanHint = cleanHint.slice(explanation.length).trim();
  }
  if (!cleanHint) return null;
  return <p className="hint">{cleanHint}</p>;
}

export function DocumentSectionsHeatmap({ sections }: { sections?: DetectionResult["section_scores"] }) {
  if (!sections || sections.length === 0) return null;

  return (
    <div className="document-sections-card">
      <div className="document-sections-header">
        <div className="document-sections-title">
          <Layers size={16} className="text-cyan-400" />
          <span>Document Section Threat Map</span>
        </div>
        <span className="document-sections-count">{sections.length} Sections Analyzed</span>
      </div>

      <div className="document-sections-bar">
        {sections.map((sec) => {
          const barColor = sec.score > 0.70 ? "#f43f5e" : sec.score > 0.55 ? "#f59e0b" : "#10b981";
          return (
            <div
              key={sec.index}
              className="document-section-bar-segment"
              style={{ background: barColor }}
              title={`Section ${sec.index + 1}: ${Math.round(sec.score * 100)}% AI`}
            />
          );
        })}
      </div>

      <div className="document-sections-list">
        {sections.map((sec) => {
          const badgeColor = sec.score > 0.70 ? "#f43f5e" : sec.score > 0.55 ? "#f59e0b" : "#10b981";
          const label = sec.score > 0.70 ? "Likely AI" : sec.score > 0.55 ? "Possible AI" : "Likely Human";
          return (
            <div
              key={sec.index}
              className="document-section-row"
              style={{ borderLeft: `3px solid ${badgeColor}` }}
            >
              <div className="document-section-info">
                <div className="document-section-meta">
                  <span className="document-section-name">Sec #{sec.index + 1}</span>
                  <span className="document-section-words">({sec.word_count} words)</span>
                </div>
                <p className="document-section-preview">{sec.preview}</p>
              </div>
              <span
                className="document-section-badge"
                style={{ color: badgeColor, background: `${badgeColor}1a` }}
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

export function DossierExportAction({ jobId }: { jobId?: string }) {
  if (!jobId) return null;

  return (
    <div className="dossier-export-wrapper">
      <a
        href={`/api/v1/jobs/${jobId}/dossier`}
        target="_blank"
        rel="noopener noreferrer"
        className="dossier-export-link"
      >
        <FileText size={15} />
        <span>Export Cryptographic Dossier (PDF / HTML)</span>
      </a>
    </div>
  );
}

export function CodeVerdictBadge({ codeResult }: { codeResult: DetectionResult }) {
  const isCode = codeResult.signals?.is_code;
  if (isCode !== 1.0 && isCode !== 1) return null;

  const score = codeResult.overall_score;
  const isHuman = score < 0.20;
  const isAi = score > 0.70;

  const badgeText = isHuman ? "MANUAL CODED (HUMAN)" : isAi ? "VIBE CODED (AI)" : "MIXED / UNVERIFIED CODE";
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

export function MixedAuthorshipBadge({ docResult }: { docResult: DetectionResult }) {
  const info = docResult.mixed_authorship || (docResult.signals?.mixed_authorship as MixedAuthorshipInfo | undefined);
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
