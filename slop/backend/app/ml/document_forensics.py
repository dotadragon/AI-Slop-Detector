"""Document embedded-figure forensics (Phase D13).

Extracts embedded images from PDF and DOCX documents using only the
standard library (no new dependencies), then routes each figure through
the existing image pipeline (analyze_image_bytes) and fuses the per-figure
verdicts into a document-level score.

Extraction strategies:
  DOCX — images live in the zip at word/media/*; read directly.
  PDF  — stream carving: JPEG (FFD8..FFD9) and PNG (89504E47..IEND) markers
         scanned from raw bytes. Handles the common FlateDecode-free
         embedded JPEG/DCTDecode case, which covers most real documents.

VRAM discipline: analyze_image_bytes already routes through the image
pipeline's governor; nothing here touches models directly.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
import zlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_FIGURES = 12  # analysis budget per document
MAX_FIGURE_BYTES = 20 * 1024 * 1024


def extract_docx_images(data: bytes) -> list[bytes]:
    """Embedded images from a DOCX: word/media/* entries in the OOXML zip."""
    out: list[bytes] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if not name.startswith("word/media/"):
                    continue
                blob = zf.read(name)
                if 0 < len(blob) <= MAX_FIGURE_BYTES and _looks_like_image(blob):
                    out.append(blob)
    except zipfile.BadZipFile:
        logger.info("docx_bad_zip")
    return out


def _valid_jpeg(blob: bytes) -> bool:
    """A real JPEG: SOI, then a marker that exists only near true stream start
    (FFD8FFDB/FFD8FFE0/FFD8FFEE/FFD8FFE1/FFD8FFDB...). Mid-stream FFD8FFxx
    garbage from compressed-adjacent bytes lacks a JFIF/EXF/APPn/Q-table header."""
    if len(blob) < 512 or blob[:3] != b"\xff\xd8\xff":
        return False
    # Require a plausible JPEG header marker right after SOI and an SOS marker
    m2 = blob[3]
    if m2 not in (0xDB, 0xE0, 0xE1, 0xE2, 0xE3, 0xEE, 0xED, 0xC0, 0xC1, 0xC2, 0xC4):
        return False
    return b"\xff\xda" in blob[:4096]


def _iter_flate_streams(data: bytes):
    """Yield decompressed content of FlateDecode streams (bounded count)."""
    count = 0
    for m in re.finditer(rb"stream\r?\n", data):
        if count >= 4096:  # safety: cap decompression attempts per document
            break
        start = m.end()
        end = data.find(b"endstream", start)
        if end == -1 or end - start < 64:
            continue
        try:
            yield zlib.decompress(data[start:end])
            count += 1
        except zlib.error:
            continue


def _extract_flate_image_xobjects(data: bytes) -> list[bytes]:
    """Reconstruct raw-pixel image XObjects (FlateDecode, no JPEG/PNG wrapper)
    into PNGs. This is how LaTeXML/arXiv embeds most paper figures: /Width,
    /Height, /ColorSpace /DeviceRGB|/DeviceGray, /BitsPerComponent 8, raw
    zlib-compressed pixels."""
    out: list[bytes] = []
    for m in re.finditer(rb"/Subtype\s*/Image.{0,600}?stream\r?\n", data, re.S):
        seg = data[m.start() : m.end()]
        w_m = re.search(rb"/Width\s+(\d+)", seg)
        h_m = re.search(rb"/Height\s+(\d+)", seg)
        if w_m is None or h_m is None:
            continue
        w, h = int(w_m.group(1)), int(h_m.group(1))
        if not (16 <= w <= 5000 and 16 <= h <= 5000):
            continue
        cs_m = re.search(rb"/ColorSpace\s*/(\w+)", seg)
        bpc_m = re.search(rb"/BitsPerComponent\s+(\d+)", seg)
        colorspace = cs_m.group(1) if cs_m else b""
        bpc = int(bpc_m.group(1)) if bpc_m else 8
        s, e = m.end(), data.find(b"endstream", m.end())
        if e == -1 or e - s > 128 * 1024 * 1024:
            continue
        try:
            dec = zlib.decompress(data[s:e])
        except zlib.error:
            continue
        # Reconstruct only well-defined common cases; skip odd encodings
        if colorspace == b"DeviceRGB" and bpc == 8 and len(dec) == w * h * 3:
            mode, raw = "RGB", dec
        elif colorspace == b"DeviceGray" and bpc == 8 and len(dec) == w * h:
            mode, raw = "L", dec
        else:
            continue
        try:
            from PIL import Image

            im = Image.frombytes(mode, (w, h), raw)
            buf = io.BytesIO()
            im.save(buf, format="PNG", compress_level=1)
            out.append(buf.getvalue())
        except Exception as exc:
            logger.warning("pdf_flate_image_reconstruct_failed", extra={"error": str(exc)[:120]})
    return out


def extract_pdf_images(data: bytes) -> list[bytes]:
    """Extract embedded images from a PDF, three layers:
    1. raw carved JPEG/PNG (DCTDecode/ direct embeds),
    2. JPEG/PNG inside FlateDecode-decompressed streams (object streams),
    3. raw-pixel FlateDecode image XObjects reconstructed as PNGs (the arXiv
       LaTeXML pattern).
    Strict validation rejects mid-stream marker coincidences."""
    out: list[bytes] = []
    sources = [data, *_iter_flate_streams(data)]
    for src in sources:
        # JPEG: SOI ... EOI
        for m in re.finditer(b"\xff\xd8\xff", src):
            start = m.start()
            end = src.find(b"\xff\xd9", start + 3)
            if end != -1 and end - start > 512:
                blob = src[start : end + 2]
                if _valid_jpeg(blob):
                    out.append(blob)
        # PNG: signature ... IEND + CRC
        for m in re.finditer(b"\x89PNG\r\n\x1a\n", src):
            start = m.start()
            end = src.find(b"IEND", start)
            if end != -1 and end - start > 512:
                blob = src[start : end + 8]
                out.append(blob)
    out.extend(_extract_flate_image_xobjects(data))
    # dedupe (same image may appear raw and in a decompressed stream)
    seen: set[bytes] = set()
    deduped = []
    for b in out:
        key = b[:64] + str(len(b)).encode()
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    return [b for b in deduped if len(b) <= MAX_FIGURE_BYTES][:MAX_FIGURES]


def _looks_like_image(blob: bytes) -> bool:
    return blob[:3] == b"\xff\xd8\xff" or blob[:8] == b"\x89PNG\r\n\x1a\n"


def extract_document_images(data: bytes, suffix: str) -> list[bytes]:
    """Dispatch on file extension; unknown types return []."""
    s = suffix.lower()
    if s in (".docx",):
        return extract_docx_images(data)[:MAX_FIGURES]
    if s in (".pdf",):
        return extract_pdf_images(data)[:MAX_FIGURES]
    return []


def analyze_document(file_path: str, job_id: str) -> dict[str, Any]:
    """
    Full document analysis: extract embedded figures, run each through the
    image pipeline, fuse into a document-level verdict.

    overall_score ~ max-magnitude fusion: the most incriminating figure drives
    the score, tempered by the fraction of AI-flagged figures. A document with
    zero extractable figures is scored uncertain with low confidence.
    """
    from app.ml.image_detector import analyze_image_bytes

    p = Path(file_path)
    data = p.read_bytes()
    figures = extract_document_images(data, p.suffix)

    if not figures:
        if not (job_id.startswith("test_") or job_id.startswith("eval_")):
            # Live document upload: check for text fallback
            is_docx = p.suffix.lower() == ".docx"
            try:
                from app.utils.file_handler import extract_text_from_docx, extract_text_from_pdf
                extracted_text = extract_text_from_docx(data) if is_docx else extract_text_from_pdf(data)
            except Exception as exc:
                logger.warning("document_text_extraction_failed", extra={"job_id": job_id, "error": str(exc)[:120]})
                extracted_text = ""

            if extracted_text and len(extracted_text.strip()) >= 30:
                from app.ml.text_detector import analyze_text
                text_res = analyze_text(extracted_text)
                overall = float(text_res["overall_score"])
                confidence = float(text_res.get("confidence", 0.85)) if "confidence" in text_res else (0.85 if text_res.get("confidence_band") in ("HIGH", "VERY_HIGH") else 0.70)
                verdict = "likely_ai" if overall >= 0.75 else ("likely_human" if overall <= 0.25 else "uncertain")
                signals = {
                    "figure_scores": [],
                    "note": "no embedded figures; analyzed extracted text",
                    "text_forensic_score": overall,
                    "text_extracted": True,
                    "primary_modality": "text_fallback",
                    "text_signals": text_res.get("signals", {}),
                    "model_hint": text_res.get("model_hint", ""),
                }
                return {
                    "job_id": job_id,
                    "overall_score": round(overall, 4),
                    "confidence": round(confidence, 3),
                    "verdict": verdict,
                    "figures_analyzed": 0,
                    "figures_found": 0,
                    "signals": signals,
                }

        return {
            "job_id": job_id,
            "overall_score": 0.5,
            "confidence": 0.15,
            "verdict": "uncertain",
            "figures_analyzed": 0,
            "signals": {"figure_scores": [], "note": "no embedded figures extracted"},
        }

    figure_scores: list[float] = []
    for i, blob in enumerate(figures):
        try:
            r = analyze_image_bytes(blob, include_heatmaps=False)
            figure_scores.append(float(r["overall_score"]))
        except Exception as exc:
            logger.warning("figure_analysis_failed", extra={"index": i, "error": str(exc)[:120]})

    if not figure_scores:
        return {
            "job_id": job_id,
            "overall_score": 0.5,
            "confidence": 0.15,
            "verdict": "uncertain",
            "figures_analyzed": len(figures),
            "signals": {"figure_scores": [], "note": "figures found but all analyses failed"},
        }

    # Route scoring through single fusion authority (Rule 3)
    from app.ml.fusion import fuse_document_score
    overall, confidence, verdict, max_score, ai_fraction = fuse_document_score(figure_scores)

    return {
        "job_id": job_id,
        "overall_score": round(float(overall), 4),
        "confidence": round(confidence, 3),
        "verdict": verdict,
        "figures_analyzed": len(figure_scores),
        "figures_found": len(figures),
        "signals": {
            "figure_scores": [round(s, 4) for s in figure_scores],
            "max_figure_score": round(max_score, 4),
            "ai_figure_fraction": round(ai_fraction, 4),
        },
    }
