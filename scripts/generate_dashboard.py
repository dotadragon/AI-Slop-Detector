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

ROOT = Path(__file__).resolve().parent
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
        pts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#4f8ef7"/>')
        maxc = max(maxc, b["mean_confidence"])
    diag = (f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{pad}" '
            f'stroke="#bbb" stroke-dasharray="4 3"/>')
    axes = (f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#333"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#333"/>')
    cap = (f'<text x="{w / 2}" y="{h - 6}" text-anchor="middle" font-size="10">predicted confidence</text>'
           f'<text x="10" y="{h / 2}" font-size="10" transform="rotate(-90 10 {h / 2})">observed acc</text>')
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
            f"<td>{'—' if recall is None else f'{recall:.0f}%'}</td></tr>")
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
      <h2>{html.escape(name)} <span class="badge {badge}">{acc}%</span></h2>
      <p class="muted">n={n} &middot; Wilson 95% CI [{lo}%, {hi}%] &middot; ECE {ece}</p>
      <div class="cols">
        <div><h3>Reliability</h3>{reliability_svg(mod["reliability_diagram"])}</div>
        <div><h3>Per-subtype confusion</h3>{confusion_table(mod["by_subtype"])}</div>
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
 body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
 .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; }}
 .badge {{ font-size: 0.9em; padding: 2px 10px; border-radius: 12px; color: #fff; }}
 .badge.ok {{ background: #2e9e5b; }} .badge.warn {{ background: #d99a1b; }} .badge.bad {{ background: #c0392b; }}
 table {{ border-collapse: collapse; font-size: 0.85em; }}
 td, th {{ border: 1px solid #ddd; padding: 3px 8px; text-align: right; }}
 td:first-child, th:first-child {{ text-align: left; }}
 .cols {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
 .muted {{ color: #777; font-size: 0.85em; }}
</style></head>
<body>
<h1>SLOP Detection — Accuracy &amp; Calibration</h1>
<p class="muted">Generated {generated} from heldout_report.json (eval_rigorous.py over real held-out data).
Refresh: <code>python eval_rigorous.py &amp;&amp; python generate_dashboard.py</code></p>
{cards}
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"dashboard written: {OUT}")


if __name__ == "__main__":
    main()
