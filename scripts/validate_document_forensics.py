"""Validate document forensics on real documents (Phase D13 gate evidence).

Two-part real-world validation:

Part 1 — REAL PDFs: arXiv papers (known provenance, human-made figures).
  arxiv_1706.03762.pdf (Attention Is All You Need), arxiv_2005.14165.pdf
  (GPT-3). These exercise the stream-carver against real-world PDF encodings
  (object streams, FlateDecode) and measure per-figure image-pipeline scores.

Part 2 — CONTAINER GROUND TRUTH: real scanned document pages (DocVQA UCSF
  industry documents, public research use) written into real PDF (DCTDecode
  embeds) and DOCX (OOXML media) containers with known figure provenance:
  human=the scanned page itself, ai=SDXL gallery images embedded in the same
  container construction. A document containing AI figures should score
  higher than one containing only real-page figures.

Usage: python validate_document_forensics.py
"""
from __future__ import annotations

import io
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, "slop/backend")

OUT = Path("document_validation_result.json")
CONTAINER_DIR = Path("heldout_set/document/containers")


def make_png_header(width: int, height: int) -> tuple:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return chunk


def png_from_bytes(raw_rgb: bytes, width: int, height: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_rgb))
        + chunk(b"IEND", b"")
    )


def wrap_jpeg_in_pdf(jpeg: bytes) -> bytes:
    """Minimal valid PDF embedding one JPEG via DCTDecode (real-world pattern)."""
    objs = []
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objs.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n")
    objs.append(
        b"4 0 obj\n<< /Type /XObject /Subtype /Image /Width 612 /Height 792 "
        b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length "
        + str(len(jpeg)).encode() + b" >>\nstream\n" + jpeg + b"\nendstream\nendobj\n"
    )
    content = b"q 612 0 0 792 0 0 cm /Im0 Do Q"
    objs.append(b"5 0 obj\n<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream\nendobj\n")
    header = b"%PDF-1.4\n"
    body = header + b"".join(objs)
    xref_pos = len(body)
    # simple xref (parser-tolerant readers accept this; our carver doesn't need it)
    return body + b"xref\n" + str(xref_pos).encode() + b"\n%%EOF"


def wrap_images_in_docx(images: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="jpeg" ContentType="image/jpeg"/></Types>')
        zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
        zf.writestr("word/document.xml", "<w:document/>")
        for i, blob in enumerate(images):
            ext = "png" if blob[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
            zf.writestr(f"word/media/image{i}.{ext}", blob)
    return buf.getvalue()


def main() -> None:
    from app.ml.document_forensics import analyze_document, extract_pdf_images
    from app.ml.image_detector import analyze_image_bytes

    result: dict = {"real_pdfs": {}, "container_eval": {}}

    # ---- Part 1: real arXiv PDFs (human-made figures, known provenance) ----
    real = Path("heldout_set/document/real")
    for pdf_path in sorted(real.glob("*.pdf")):
        data = pdf_path.read_bytes()
        figures = extract_pdf_images(data)
        scores = []
        errors = 0
        for blob in figures:
            try:
                scores.append(round(float(analyze_image_bytes(blob)["overall_score"]), 4))
            except Exception:
                errors += 1
        result["real_pdfs"][pdf_path.name] = {
            "figures_extracted": len(figures),
            "figures_scored": len(scores),
            "figure_scores": scores,
            "errors": errors,
            "ai_flagged_fraction": round(sum(1 for s in scores if s >= 0.5) / len(scores), 4) if scores else None,
            "max": max(scores) if scores else None,
        }
        print(f"{pdf_path.name}: {len(figures)} extracted, {len(scores)} scored, errors={errors} -> max={max(scores) if scores else None}", flush=True)

    # ---- Part 2: container ground truth ----
    CONTAINER_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_docvqa_pages(n: int) -> list[bytes]:
        """Real scanned document pages (JPEG) from DocVQA validation."""
        import time
        import urllib.parse
        import urllib.request

        pages: list[bytes] = []
        offset = 0
        while len(pages) < n and offset < 100:
            url = ("https://datasets-server.huggingface.co/rows?dataset=lmms-lab-encoder%2FDocVQA"
                   f"&config=DocVQA&split=validation&offset={offset}&length=10")
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.load(resp)
            rows = data.get("rows", [])
            if not rows:
                break
            for r in rows:
                src = r["row"]["image"][0]["src"] if isinstance(r["row"]["image"], list) else r["row"]["image"].get("src")
                try:
                    req = urllib.request.Request(src, headers={"User-Agent": "slop-eval/1.0"})
                    with urllib.request.urlopen(req, timeout=60) as resp2:
                        blob = resp2.read()
                    if blob[:3] == b"\xff\xd8\xff":
                        pages.append(blob)
                except Exception as exc:
                    print(f"  page fetch failed: {str(exc)[:60]}")
                if len(pages) >= n:
                    break
            offset += 10
            time.sleep(0.3)
        return pages

    print("Fetching 12 real document pages (DocVQA)...", flush=True)
    human_pages = fetch_docvqa_pages(12)
    print(f"  got {len(human_pages)} pages", flush=True)

    ai_figs = sorted(Path("heldout_set/image/ai/sdxl_proxy").glob("*.png"))[: len(human_pages)]
    ai_blobs = [p.read_bytes() for p in ai_figs]

    # Build containers: 6 human-docs (2 pages each), 6 ai-docs (2 AI figs each)
    n_docs = min(6, len(human_pages) // 2, len(ai_blobs) // 2)
    entries = []
    for i in range(n_docs):
        hdoc = CONTAINER_DIR / f"human_doc_{i}.pdf"
        hdoc.write_bytes(wrap_jpeg_in_pdf(human_pages[2 * i]))
        entries.append((str(hdoc), 0))
        hdocx = CONTAINER_DIR / f"human_doc_{i}.docx"
        hdocx.write_bytes(wrap_images_in_docx(human_pages[2 * i : 2 * i + 2]))
        entries.append((str(hdocx), 0))

        adoc = CONTAINER_DIR / f"ai_doc_{i}.pdf"
        adoc.write_bytes(wrap_jpeg_in_pdf(_to_jpeg(ai_blobs[2 * i])))
        entries.append((str(adoc), 1))
        adocx = CONTAINER_DIR / f"ai_doc_{i}.docx"
        adocx.write_bytes(wrap_images_in_docx(ai_blobs[2 * i : 2 * i + 2]))
        entries.append((str(adocx), 1))

    scores_h, scores_a = [], []
    per_doc = {}
    for fpath, label in entries:
        r = analyze_document(fpath, "docval")
        per_doc[Path(fpath).name] = {"label": label, "score": r["overall_score"], "figures": r["figures_analyzed"]}
        (scores_a if label == 1 else scores_h).append(r["overall_score"])
        print(f"  {Path(fpath).name}: label={label} score={r['overall_score']} figures={r['figures_analyzed']}", flush=True)

    def auc(pos: list[float], neg: list[float]) -> float:
        if not pos or not neg:
            return 0.5
        wins = sum(1 for p in pos for n in neg if p > n)
        ties = sum(1 for p in pos for n in neg if p == n)
        return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)

    result["container_eval"] = {
        "n_docs": len(entries),
        "human_doc_scores": scores_h,
        "ai_doc_scores": scores_a,
        "auc_ai_vs_human": auc(scores_a, scores_h),
        "per_doc": per_doc,
    }

    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result["container_eval"].items() if k != "per_doc"}, indent=2))


def _to_jpeg(png_bytes: bytes) -> bytes:
    """Re-encode a PNG to JPEG via the image pipeline's PIL (DCTDecode needs JPEG)."""
    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


if __name__ == "__main__":
    main()
