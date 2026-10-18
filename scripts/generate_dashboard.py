"""Generate the public calibration/accuracy dashboard (Phase E14).

Reads `heldout_report.json` (produced by `eval_rigorous.py`) and renders a
self-contained HTML page: per-modality accuracy with Wilson CIs, ECE,
reliability diagrams, per-subtype confusion matrices, and the gate/promotion
state. Refresh by re-running after each eval or promoted retrain.

Usage:
  python generate_dashboard.py            # writes dashboard/eval_dashboard.html
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
REPORT = ROOT / "heldout_report.json"
OUT_DIR = ROOT / "dashboard"
OUT = OUT_DIR / "eval_dashboard.html"


def reliability_svg(diagram: list[dict]) -> str:
    """Render a reliability diagram (10 bins) as inline SVG."""
    w, h, pad = 320, 200, 30
    pts = []
    maxc = 0.0
    for b in diagram:
        if b["mean_confidence"] is None:
            continue
        x = pad + b["mean_confidence"] * (w - 2 * pad)
        y = h - pad - b["accuracy"] * (h - 2 * pad)
        r = min(8, 2 + b["count"])
        pts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#f97316"/>')
        maxc = max(maxc, b["mean_confidence"])
    diag = (f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{pad}" '
            f'stroke="#64748b" stroke-dasharray="4 3"/>')
    axes = (f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#64748b"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#64748b"/>')
    cap = (f'<text x="{w / 2}" y="{h - 6}" text-anchor="middle" font-size="10" fill="#94a3b8">predicted confidence</text>'
           f'<text x="10" y="{h / 2}" font-size="10" transform="rotate(-90 10 {h / 2})" fill="#94a3b8">observed acc</text>')
    body = "".join(pts) or '<text x="120" y="100" font-size="11" fill="#888">no data</text>'
    return f'<svg width="{w}" height="{h}">{axes}{diag}{body}{cap}</svg>'


def confusion_table(subtypes: dict) -> str:
    rows = []
    for key, c in sorted(subtypes.items()):
        total = c["tp"] + c["fp"] + c["tn"] + c["fn"]
        recall = c["tp"] / (c["tp"] + c["fn"]) * 100 if c["tp"] + c["fn"] else None
        rows.append(
            f"<tr><td>{html.escape(key)}</td><td>{total}</td><td>{c['tp']}</td>"
            f"<td>{c['fp']}</td><td>{c['tn']}</td><td>{c['fn']}</td>"
            f"<td>{'—' if recall is None else f'{recall:.1f}%'}</td></tr>")
    if not rows:
        return "<p class='muted'>No subtype data.</p>"
    return ("<table><tr><th>subtype</th><th>n</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th><th>recall</th></tr>"
            + "".join(rows) + "</table>")


def modality_card(mod: dict) -> str:
    name = mod["modality"]
    acc = mod["accuracy_pct"]
    lo, hi = mod["accuracy_wilson_95"]
    ece = mod["ece"]
    n = mod["total"]
    badge = "ok" if acc is not None and acc >= 75 else ("warn" if acc is not None and acc >= 50 else "bad")
    return f"""
    <section class="card">
      <h2>{html.escape(name.upper())} <span class="badge {badge}">{acc}%</span></h2>
      <p class="muted">n={n} &middot; Wilson 95% CI [{lo}%, {hi}%] &middot; ECE {ece}</p>
      <div class="cols">
        <div><h3>Reliability Curve</h3>{reliability_svg(mod["reliability_diagram"])}</div>
        <div style="flex: 1; min-width: 280px;"><h3>Per-Subtype Confusion Matrix</h3>{confusion_table(mod["by_subtype"])}</div>
      </div>
      <p class="muted">TP {mod['tp']} &middot; FP {mod['fp']} &middot; TN {mod['tn']} &middot; FN {mod['fn']}</p>
    </section>"""


def main() -> None:
    if not REPORT.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("<html><body><h1>No heldout_report.json yet — run eval_rigorous.py</h1></body></html>",
                       encoding="utf-8")
        print(f"placeholder written: {OUT}")
        return
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = "".join(modality_card(m) for m in report.values() if isinstance(m, dict) and "modality" in m)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SLOP Detection — Accuracy &amp; Calibration</title>
<style>
:root {{
  --bg: #0d121c;
  --text: #f8fafc;
  --muted: #94a3b8;
  --card-bg: #141c2e;
  --border: #1e293b;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg: #f8fafc;
    --text: #0f172a;
    --muted: #64748b;
    --card-bg: #ffffff;
    --border: #e2e8f0;
  }}
}}
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: var(--bg); color: var(--text); line-height: 1.5; }}
.card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
.badge {{ font-family: var(--mono); font-size: 0.85em; font-weight: 700; padding: 3px 10px; border-radius: 6px; color: #fff; }}
.badge.ok {{ background: #10b981; }} .badge.warn {{ background: #f59e0b; }} .badge.bad {{ background: #ef4444; }}
table {{ border-collapse: collapse; font-family: var(--mono); font-size: 0.82em; width: 100%; }}
td, th {{ border: 1px solid var(--border); padding: 6px 10px; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
th {{ background: rgba(125,125,125,0.08); font-weight: 600; }}
.cols {{ display: flex; gap: 2rem; flex-wrap: wrap; margin: 1rem 0; }}
.muted {{ color: var(--muted); font-size: 0.85em; }}
code {{ font-family: var(--mono); background: rgba(125,125,125,0.12); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
</style></head>
<body>
<h1>SLOP Detection — Multi-Modal Calibration &amp; Verification</h1>
<p class="muted">Generated {generated} from heldout_report.json across verified test partitions.
Refresh: <code>python scripts/generate_dashboard.py</code></p>
{cards}
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"dashboard written: {OUT}")


if __name__ == "__main__":
    main()
