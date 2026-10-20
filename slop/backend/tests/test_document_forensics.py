"""Document forensics tests (Phase D13).

Synthetic DOCX/PDF fixtures — stdlib-built, no external test assets needed.
"""
from __future__ import annotations

import io
import struct
import zipfile
import zlib

import pytest


def _tiny_png(width: int = 256, height: int = 256) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # pseudo-random rows so the blob stays >512 bytes even after compress
    row = b"\x00" + bytes((i * 37 + 11) % 256 for i in range(3 * width))
    raw = row * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _tiny_jpeg(size: int = 2048) -> bytes:
    """Realistic JPEG skeleton: SOI, APP0/JFIF, DQT, SOS, filler, EOI.
    The validator requires a true header chain, not just FFD8..FFD9."""
    jfif = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    dqt = b"\xff\xdb\x00\x43\x00" + bytes(64)
    sos = b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00"
    filler = b"\x12\x34" * max(1, (size - 26) // 2)
    return b"\xff\xd8" + jfif + dqt + sos + filler + b"\xff\xd9"


def _docx(*media: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<doc/>")
        for i, blob in enumerate(media):
            zf.writestr(f"word/media/image{i}.png", blob)
    return buf.getvalue()


def _pdf(*images: bytes) -> bytes:
    body = b"%PDF-1.4\n"
    for img in images:
        body += b"stream\n" + img + b"\nendstream\n"
    return body + b"%%EOF"


from app.ml.document_forensics import (
    analyze_document,
    extract_document_images,
    extract_docx_images,
    extract_pdf_images,
)


def test_docx_extraction():
    imgs = extract_docx_images(_docx(_tiny_png(), _tiny_png()))
    assert len(imgs) == 2
    assert all(b[:8] == b"\x89PNG\r\n\x1a\n" for b in imgs)


def test_docx_ignores_non_media_entries():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<doc/>")
        zf.writestr("word/settings.xml", b"\xff\xd8\xff" + b"\x00" * 600 + b"\xff\xd9")
    assert extract_docx_images(buf.getvalue()) == []


def test_pdf_extraction_jpeg_and_png():
    imgs = extract_pdf_images(_pdf(_tiny_jpeg(), _tiny_png()))
    kinds = {"jpeg" if b[:3] == b"\xff\xd8\xff" else "png" for b in imgs}
    assert kinds == {"jpeg", "png"}


def test_dispatch_by_suffix():
    docx = _docx(_tiny_png())
    pdf = _pdf(_tiny_jpeg())
    assert len(extract_document_images(docx, ".docx")) == 1
    assert len(extract_document_images(pdf, ".pdf")) == 1
    assert extract_document_images(docx, ".txt") == []


def test_pdf_flate_raw_pixel_xobject():
    """The arXiv/LaTeXML pattern: raw-pixel FlateDecode image XObject.
    Regression test for the real-PDF validation finding (raw carving alone
    returns 0 figures on modern arXiv papers)."""
    import zlib

    w, h = 64, 64
    raw = bytes((i * 13 + 7) % 256 for i in range(w * h * 3))  # RGB noise
    compressed = zlib.compress(raw)
    obj = (
        b"4 0 obj\n<< /Type /XObject /Subtype /Image /Width " + str(w).encode()
        + b" /Height " + str(h).encode()
        + b" /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length "
        + str(len(compressed)).encode() + b" >>\nstream\n" + compressed + b"\nendstream\nendobj\n"
    )
    pdf = b"%PDF-1.5\n" + obj + b"%%EOF"
    imgs = extract_pdf_images(pdf)
    assert len(imgs) == 1
    assert imgs[0][:8] == b"\x89PNG\r\n\x1a\n"  # reconstructed as PNG


def test_pdf_carver_rejects_midstream_false_positive():
    """A FFD8..FFD9 span without a valid JPEG header chain must be rejected
    (the mid-stream garbage pattern found in real compressed streams)."""
    bad = b"\xff\xd8\xff\xff\xff\xff\xf4\x4b" + b"\x00" * 600 + b"\xff\xd9"
    assert extract_pdf_images(_pdf(bad)) == []


def test_analyze_document_no_figures(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(_pdf())
    r = analyze_document(str(p), "job_x")
    assert r["figures_analyzed"] == 0
    assert r["verdict"] == "uncertain"
    assert r["confidence"] <= 0.2  # deliberately low without evidence


@pytest.mark.asyncio
async def test_analyze_document_end_to_end(tmp_path):
    """Full route through the image pipeline on synthetic figures."""
    p = tmp_path / "figs.docx"
    p.write_bytes(_docx(_tiny_png(), _tiny_png()))
    r = analyze_document(str(p), "job_y")
    assert r["figures_analyzed"] == 2
    assert 0.0 <= r["overall_score"] <= 1.0
    assert r["verdict"] in {"likely_ai", "likely_human", "uncertain"}
    assert len(r["signals"]["figure_scores"]) == 2


def test_document_feature_gate_in_config():
    from app.core.config import settings

    # Phase D13: Document forensics graduated to production after passing all
    # functional tests (PDF carving, DOCX figure extraction, Flate raw pixel
    # parsing, end-to-end analysis). Feature flag remains env-overridable.
    assert settings.enable_document_forensics is True  # graduated in Phase D13


def test_analyze_document_text_fallback(tmp_path):
    p = tmp_path / "text_only.pdf"
    content = b"BT /F1 12 Tf 72 712 Td (The quick brown fox jumps over the lazy dog and tests text forensics pipeline) Tj ET"
    p.write_bytes(b"%PDF-1.4\nstream\n" + content + b"\nendstream\n%%EOF")
    r = analyze_document(str(p), "live_job_uuid_12345")
    assert r["figures_analyzed"] == 0
    assert r["signals"]["text_extracted"] is True
    assert 0.0 <= r["overall_score"] <= 1.0
    assert r["signals"]["primary_modality"] == "text_fallback"
