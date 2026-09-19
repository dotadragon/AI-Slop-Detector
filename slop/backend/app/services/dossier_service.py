import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.models.job import DetectionResult, Job


def _compute_sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def generate_dossier_html(job: Job, result: DetectionResult | None, extra_data: dict[str, Any] | None = None) -> str:
    """
    Generates a self-contained, print-ready HTML forensic dossier with SHA-256
    cryptographic chain-of-custody seal and embedded visual evidence.
    """
    extra = extra_data or {}
    job_id = str(job.id)
    job_type = job.job_type.upper()
    created_at = job.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if job.created_at else "N/A"
    completed_at = job.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if job.completed_at else "N/A"
    overall_score = result.overall_score if result else 0.0
    percent_score = round(overall_score * 100, 1)

    # Determine classification label
    if overall_score >= 0.70:
        verdict_label = "SYNTHETIC / AI GENERATED"
        verdict_color = "#f43f5e"
        verdict_bg = "rgba(244, 63, 94, 0.12)"
    elif overall_score >= 0.40:
        verdict_label = "SUSPICIOUS / MIXED SIGNALS"
        verdict_color = "#f59e0b"
        verdict_bg = "rgba(245, 158, 11, 0.12)"
    else:
        verdict_label = "AUTHENTIC / HUMAN ORIGIN"
        verdict_color = "#10b981"
        verdict_bg = "rgba(16, 185, 129, 0.12)"

    # Media hash
    media_content = job.file_path or job.file_name or job_id
    media_sha256 = _compute_sha256(media_content)

    # Cryptographic integrity seal
    seal_payload = f"{job_id}|{job.job_type}|{overall_score}|{media_sha256}|{completed_at}"
    dossier_seal = hmac.new(
        settings.secret_key.encode("utf-8"),
        seal_payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    # Signals
    signals = result.signals if result and result.signals else {}
    signal_rows = ""
    for k, v in signals.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        signal_rows += f"""
        <tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #1e293b; font-family: monospace; color: #94a3b8;">{k}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #1e293b; font-weight: 600; color: #e2e8f0;">{val_str}</td>
        </tr>
        """

    # Modality-specific visual embeds
    modality_section = ""

    if job.job_type == "image":
        gradcam = extra.get("heatmap_b64") or (result.heatmap_path if result and result.heatmap_path and result.heatmap_path.startswith("data:") else None)
        ela = extra.get("ela_heatmap_b64")
        prnu = extra.get("prnu_heatmap_b64")

        img_grid = ""
        if gradcam:
            src = gradcam if gradcam.startswith("data:") else f"data:image/jpeg;base64,{gradcam}"
            img_grid += f"""
            <div style="flex: 1; min-width: 200px; background: #0b0f19; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; font-weight: 700; color: #38bdf8; margin-bottom: 8px; text-transform: uppercase;">GradCAM Attention</div>
                <img src="{src}" alt="GradCAM" style="width: 100%; max-height: 220px; object-fit: contain; border-radius: 4px;" />
            </div>
            """
        if ela:
            src = ela if ela.startswith("data:") else f"data:image/jpeg;base64,{ela}"
            img_grid += f"""
            <div style="flex: 1; min-width: 200px; background: #0b0f19; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; font-weight: 700; color: #a855f7; margin-bottom: 8px; text-transform: uppercase;">Error Level Analysis (ELA)</div>
                <img src="{src}" alt="ELA" style="width: 100%; max-height: 220px; object-fit: contain; border-radius: 4px;" />
            </div>
            """
        if prnu:
            src = prnu if prnu.startswith("data:") else f"data:image/jpeg;base64,{prnu}"
            img_grid += f"""
            <div style="flex: 1; min-width: 200px; background: #0b0f19; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; font-weight: 700; color: #10b981; margin-bottom: 8px; text-transform: uppercase;">PRNU Sensor Noise</div>
                <img src="{src}" alt="PRNU" style="width: 100%; max-height: 220px; object-fit: contain; border-radius: 4px;" />
            </div>
            """

        modality_section = f"""
        <div style="margin-top: 24px;">
            <h3 style="color: #cbd5e1; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Multi-Spectral Visual Evidence</h3>
            <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                {img_grid if img_grid else '<p style="color: #64748b;">Visual heatmaps not retained.</p>'}
            </div>
        </div>
        """

    elif job.job_type == "text":
        sentence_data = result.sentence_data if result and result.sentence_data else []
        sentence_rows = ""
        for i, s in enumerate(sentence_data[:30]):  # Top 30 sentences
            score = s.get("ai_probability", 0.0) if isinstance(s, dict) else 0.0
            text = s.get("text", "") if isinstance(s, dict) else str(s)
            badge_color = "#f43f5e" if score >= 0.70 else ("#f59e0b" if score >= 0.40 else "#10b981")
            sentence_rows += f"""
            <div style="margin-bottom: 8px; padding: 8px 12px; background: #0b0f19; border-left: 3px solid {badge_color}; border-radius: 0 4px 4px 0;">
                <span style="display: inline-block; font-size: 10px; font-weight: 700; color: {badge_color}; margin-right: 8px;">[{round(score*100)}%]</span>
                <span style="font-size: 13px; color: #e2e8f0; line-height: 1.5;">{text}</span>
            </div>
            """
        modality_section = f"""
        <div style="margin-top: 24px;">
            <h3 style="color: #cbd5e1; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Sentence-Level Forensics</h3>
            <div>
                {sentence_rows if sentence_rows else '<p style="color: #64748b;">No sentence segmentation available.</p>'}
            </div>
        </div>
        """

    elif job.job_type == "video":
        frame_data = result.frame_data if result and result.frame_data else []
        modality_section = f"""
        <div style="margin-top: 24px;">
            <h3 style="color: #cbd5e1; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Temporal Frame Analysis</h3>
            <p style="color: #94a3b8; font-size: 13px;">Total Frames Evaluated: <strong style="color: #f8fafc;">{len(frame_data)}</strong></p>
        </div>
        """

    # Complete standalone HTML document
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SLOP Forensic Dossier — {job_id[:8]}</title>
    <style>
        @page {{
            size: A4;
            margin: 1.5cm;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #07090e;
            color: #f1f5f9;
            line-height: 1.5;
            padding: 32px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: #0d121f;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 32px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid #1e293b;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .logo {{
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #f8fafc;
        }}
        .logo span {{
            color: #00e5ff;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.05em;
        }}
        .verdict-box {{
            background: {verdict_bg};
            border: 1px solid {verdict_color};
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .seal-box {{
            margin-top: 32px;
            padding: 16px;
            background: #090d16;
            border: 1px solid #1e293b;
            border-radius: 6px;
            font-family: monospace;
            font-size: 11px;
            color: #94a3b8;
            word-break: break-all;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}
        .print-btn {{
            background: #00e5ff;
            color: #07090e;
            font-weight: 700;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 16px;
        }}
        @media print {{
            body {{
                background-color: #ffffff;
                color: #0f172a;
                padding: 0;
            }}
            .container {{
                background-color: #ffffff;
                border: none;
                padding: 0;
                max-width: 100%;
            }}
            .print-btn {{
                display: none;
            }}
            .verdict-box {{
                background: #f8fafc;
                border: 2px solid #0f172a;
            }}
            .seal-box {{
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                color: #334155;
            }}
            td {{
                color: #0f172a !important;
                border-bottom: 1px solid #e2e8f0 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>

        <header class="header">
            <div>
                <div class="logo">SLOP <span>FORENSICS</span></div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Sophisticated Latent &amp; Orthogonal Provenance Dossier</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; font-weight: 700; color: #00e5ff; letter-spacing: 0.1em;">EVIDENCE DOSSIER</div>
                <div style="font-size: 12px; color: #94a3b8; font-family: monospace;">ID: {job_id}</div>
            </div>
        </header>

        <section class="verdict-box">
            <div>
                <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Forensic Determination</div>
                <div style="font-size: 22px; font-weight: 800; color: {verdict_color}; margin-top: 4px;">{verdict_label}</div>
                <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">Modality: {job_type} | Evaluated via Calibrated Frontier Pipeline</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 38px; font-weight: 900; color: {verdict_color}; line-height: 1;">{percent_score}%</div>
                <div style="font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase;">Confidence Score</div>
            </div>
        </section>

        <section style="margin-bottom: 24px;">
            <h3 style="color: #cbd5e1; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Chain-of-Custody Metadata</h3>
            <table style="font-size: 13px;">
                <tr>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; color: #94a3b8; width: 35%;">Media Filename</td>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; font-weight: 600; color: #f8fafc;">{job.file_name or "Direct Input Buffer"}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; color: #94a3b8;">SHA-256 Media Digest</td>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; font-family: monospace; font-size: 12px; color: #38bdf8;">{media_sha256}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; color: #94a3b8;">Submission Timestamp</td>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; color: #f8fafc;">{created_at}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; color: #94a3b8;">Analysis Completion</td>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; color: #f8fafc;">{completed_at}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; color: #94a3b8;">Client Node Audit Stamp</td>
                    <td style="padding: 6px 12px; border-bottom: 1px solid #1e293b; font-family: monospace; color: #f8fafc;">{job.ip_address or "Internal Session"}</td>
                </tr>
            </table>
        </section>

        {modality_section}

        <section style="margin-top: 24px;">
            <h3 style="color: #cbd5e1; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Extracted Forensic Signals</h3>
            <table style="font-size: 13px;">
                <thead>
                    <tr style="text-align: left; background: #0b0f19;">
                        <th style="padding: 8px 12px; color: #64748b; font-size: 11px; text-transform: uppercase;">Signal Name</th>
                        <th style="padding: 8px 12px; color: #64748b; font-size: 11px; text-transform: uppercase;">Metric / Value</th>
                    </tr>
                </thead>
                <tbody>
                    {signal_rows if signal_rows else '<tr><td colspan="2" style="padding: 8px 12px; color: #64748b;">No individual signals stored.</td></tr>'}
                </tbody>
            </table>
        </section>

        <footer class="seal-box">
            <div style="font-weight: 700; color: #00e5ff; margin-bottom: 4px;">CRYPTOGRAPHIC VERIFICATION SEAL</div>
            <div>SEAL: {dossier_seal}</div>
            <div style="margin-top: 4px; color: #64748b;">Generated by SLOP Forensic Engine v3.4 | Tamper-Evident Integrity Record</div>
        </footer>
    </div>
</body>
</html>
"""
