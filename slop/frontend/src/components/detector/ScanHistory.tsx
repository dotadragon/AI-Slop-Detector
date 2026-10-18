import { Trash2, Image, FileText, Film, Camera, ExternalLink } from "lucide-react";
import { useEffect, useRef } from "react";
import gsap from "gsap";
import type { DetectorMode } from "../../types";
import type { ScanEntry } from "../../hooks/useScanHistory";
import { scoreColor } from "../../utils/validators";

const MODE_ICONS: Partial<Record<DetectorMode, typeof FileText>> = {
  image: Image,
  text: FileText,
  video: Film,
  live_image: Camera,
};

function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function getBadgeClass(score: number): string {
  if (score >= 0.72) return "badge-ai";
  if (score >= 0.42) return "badge-amber";
  return "badge-emerald";
}

function ScanHistoryItemRow({
  entry,
  onSelect,
}: {
  entry: ScanEntry;
  onSelect: (entry: ScanEntry) => void;
}) {
  const Icon = MODE_ICONS[entry.mode] || FileText;
  const pct = Math.round(entry.score * 100);
  const color = scoreColor(entry.score);
  const badgeClass = getBadgeClass(entry.score);

  return (
    <div
      className="scan-history-row"
      onClick={() => onSelect(entry)}
      title="Click to view detailed report for this scan"
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          onSelect(entry);
        }
      }}
    >
      {entry.thumbnail ? (
        <img className="scan-history-thumb" src={entry.thumbnail} alt="" />
      ) : (
        <div className="scan-history-thumb-placeholder">
          <Icon size={16} />
        </div>
      )}

      <div className="scan-history-info">
        <div className="scan-history-topline">
          <span className="scan-history-mode">{entry.mode.replace("_", " ").toUpperCase()}</span>
          {entry.title && (
            <span className="scan-history-item-title" title={entry.title}>
              — {entry.title}
            </span>
          )}
        </div>
        <span className="scan-history-time">{formatTime(entry.timestamp)}</span>
      </div>

      <div className="scan-history-actions">
        <span className="scan-history-score" style={{ color }}>{pct}%</span>
        <span className={`badge ${badgeClass}`}>{entry.label}</span>
        <ExternalLink size={12} className="scan-history-link-icon" />
      </div>
    </div>
  );
}

function ScanHistoryHeader({
  count,
  onClear,
}: {
  count: number;
  onClear: () => void;
}) {
  return (
    <div className="scan-history-header">
      <div className="scan-history-title-row">
        <h3>Scan History</h3>
        <span className="scan-history-count">
          ({count} {count === 1 ? "scan" : "scans"} cached)
        </span>
      </div>
      {count > 0 && (
        <button
          type="button"
          className="btn btn-sm btn-ghost text-red"
          onClick={onClear}
          title="Clear all saved scans from browser cache"
        >
          <Trash2 size={13} />
          <span>Clear History</span>
        </button>
      )}
    </div>
  );
}

export function ScanHistory({
  history,
  onClear,
  onSelect,
}: {
  history: ScanEntry[];
  onClear: () => void;
  onSelect: (entry: ScanEntry) => void;
}) {
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!listRef.current || history.length === 0) return;
    gsap.fromTo(
      listRef.current.querySelectorAll(".scan-history-row"),
      { y: 12, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.35, stagger: 0.04, ease: "power2.out" }
    );
  }, [history.length]);

  const handleAnimatedClear = () => {
    if (!listRef.current) {
      onClear();
      return;
    }
    const rows = listRef.current.querySelectorAll(".scan-history-row");
    if (rows.length === 0) {
      onClear();
      return;
    }
    gsap.to(rows, {
      opacity: 0,
      x: 16,
      duration: 0.2,
      stagger: 0.02,
      ease: "power2.in",
      onComplete: onClear,
    });
  };

  return (
    <section className="scan-history" aria-label="Recent forensic scan history">
      <ScanHistoryHeader count={history.length} onClear={handleAnimatedClear} />

      {history.length === 0 ? (
        <div className="scan-history-empty">
          <p>No recent scans yet. Completed text, image, and video analyses will appear here automatically.</p>
        </div>
      ) : (
        <div className="scan-history-list" ref={listRef}>
          {history.map((entry) => (
            <ScanHistoryItemRow key={entry.id} entry={entry} onSelect={onSelect} />
          ))}
        </div>
      )}
    </section>
  );
}
