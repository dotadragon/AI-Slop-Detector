import io
import mimetypes
import re
import time
import uuid
import zipfile
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.core.config import settings

TEXT_MIMES = {
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/csv",
    "text/markdown",
}
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp", "image/gif", "image/heic", "image/avif"}
VIDEO_MIMES = {"video/mp4", "video/x-matroska", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-flv"}

ALLOWED_EXTENSIONS = {
    "text": {".txt", ".pdf", ".doc", ".docx", ".md", ".rtf", ".csv"},
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".avif"},
    "video": {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"},
}

BLOCKED_EXTENSIONS = {
    ".env", ".key", ".pem", ".crt", ".rsa", ".pub", ".cer", ".pfx", ".p12",
    ".sh", ".bash", ".bat", ".cmd", ".ps1", ".vbs",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".py", ".pyc", ".js", ".ts", ".html", ".htm", ".php",
}

MAX_DOCX_UNCOMPRESSED_BYTES = 40 * 1024 * 1024  # 40 MB max uncompressed text XML
MAX_COMPRESSION_RATIO = 100.0


def sanitize_filename(filename: str) -> str:
    cleaned = filename.replace("\x00", "").strip().replace("\\", "/")
    name = Path(cleaned).name
    # Strip any leading directory traversal components
    while name.startswith(".."):
        name = name[2:].lstrip(". ")
    return name or "upload.bin"


def is_extension_blocked(filename: str) -> bool:
    name_lower = filename.lower()
    return any(name_lower.endswith(ext) or f"{ext}." in name_lower for ext in BLOCKED_EXTENSIONS)


def allowed_mimes(kind: str) -> set[str]:
    return {"text": TEXT_MIMES, "image": IMAGE_MIMES, "video": VIDEO_MIMES}[kind]


def max_bytes(kind: str) -> int:
    mb = {"text": settings.max_text_file_mb, "image": settings.max_image_file_mb, "video": settings.max_video_file_mb}[kind]
    return mb * 1024 * 1024


async def read_upload_bounded(upload: UploadFile, max_size: int) -> bytes:
    """
    Stream-bounded reader that prevents memory exhaustion (DoS).
    Immediately aborts if incoming bytes exceed max_size before full buffer allocation.
    """
    chunks: list[bytes] = []
    total = 0
    chunk_size = 64 * 1024  # 64 KB
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(status_code=413, detail="file_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _sniff_mime(data: bytes, default: str) -> str:
    # ponytail: pure stdlib magic byte inspection replacing uninstalled C-extension libmagic
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"PK\x03\x04"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if b"ftyp" in data[4:32]:
        return "video/mp4"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    return default


def extract_text_from_docx(data: bytes) -> str:
    """
    Safely extract text paragraphs from a .docx file using standard library zipfile and XML.
    Includes decompression bomb guards (ratio and total uncompressed size limit).
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            total_uncompressed = 0
            for info in z.infolist():
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise HTTPException(status_code=400, detail="potential_decompression_bomb")

            if len(data) > 0 and (total_uncompressed / len(data)) > MAX_COMPRESSION_RATIO:
                raise HTTPException(status_code=400, detail="potential_decompression_bomb")

            if "word/document.xml" not in z.namelist():
                raise HTTPException(status_code=400, detail="invalid_docx_structure")

            xml_data = z.read("word/document.xml")
            tree = ET.fromstring(xml_data)
            paragraphs = []
            for elem in tree.iter():
                if elem.tag.endswith("}p"):
                    texts = [t.text for t in elem.iter() if t.tag.endswith("}t") and t.text]
                    if texts:
                        paragraphs.append("".join(texts))
            return "\n\n".join(paragraphs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"corrupt_or_invalid_docx: {str(exc)}")


MAX_PDF_STREAM_BYTES = 15 * 1024 * 1024  # 15 MB max uncompressed stream (guards against zlib bombs)


def extract_text_from_pdf(data: bytes) -> str:
    """
    Extracts text streams from PDF objects without requiring heavy C-library dependencies.
    Includes decompression bomb guards against malicious zlib payloads.
    """
    stream_pattern = re.compile(rb"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
    text_chunks: list[str] = []

    for match in stream_pattern.finditer(data):
        stream_bytes = match.group(1)
        try:
            decompressed = zlib.decompressobj().decompress(stream_bytes, MAX_PDF_STREAM_BYTES)
        except Exception:
            decompressed = stream_bytes[:MAX_PDF_STREAM_BYTES]

        # Search for PDF text blocks: BT ... ET
        bt_blocks = re.findall(rb"BT\s+(.*?)\s+ET", decompressed, re.DOTALL)
        for block in bt_blocks:
            strings = re.findall(rb"\(((?:[^()\\]|\\.)*)\)", block)
            if strings:
                cleaned = [
                    s.replace(b"\\(", b"(").replace(b"\\)", b")").decode("utf-8", errors="replace")
                    for s in strings
                ]
                text_chunks.append(" ".join(cleaned))

    if not text_chunks:
        # Fallback heuristic: extract printable ascii sequences from the PDF payload
        printable = re.findall(rb"[A-Za-z0-9\s.,!?;:\"'()\-]{4,}", data)
        text_chunks = [t.decode("latin-1", errors="ignore").strip() for t in printable if len(t.strip()) > 15]

    return "\n\n".join(c for c in text_chunks if c.strip())


def chunk_text_sections(text: str, max_chunk_words: int = 350) -> list[dict]:
    """
    Splits multi-page or large documents into semantic sections for localized threat heatmapping.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\x0c", text) if p.strip()]
    if not paragraphs:
        return [{"index": 0, "text": text, "word_count": len(text.split())}]

    chunks: list[dict] = []
    current_paras: list[str] = []
    current_words = 0

    for p in paragraphs:
        p_words = len(p.split())
        if current_words + p_words > max_chunk_words and current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append({
                "index": len(chunks),
                "text": chunk_text,
                "word_count": current_words,
            })
            current_paras = [p]
            current_words = p_words
        else:
            current_paras.append(p)
            current_words += p_words

    if current_paras:
        chunks.append({
            "index": len(chunks),
            "text": "\n\n".join(current_paras),
            "word_count": current_words,
        })

    return chunks


async def save_upload(upload: UploadFile, kind: str) -> tuple[Path, str, int]:
    original = sanitize_filename(upload.filename or "upload.bin")
    suffix = Path(original).suffix.lower()

    # Security check 0: Block sensitive or executable file extensions explicitly
    if is_extension_blocked(original):
        raise HTTPException(status_code=400, detail="blocked_file_extension")

    # Security check 1: Extension whitelist
    if suffix not in ALLOWED_EXTENSIONS.get(kind, set()):
        raise HTTPException(status_code=400, detail="disallowed_file_extension")

    declared = upload.content_type or mimetypes.guess_type(original)[0] or "application/octet-stream"
    limit_bytes = max_bytes(kind)

    # Safe random storage name prevents path traversal completely
    storage_name = f"{uuid.uuid4()}{suffix}"
    target = settings.upload_path / storage_name

    # ponytail: direct disk-streaming reader prevents heap exhaustion on 50-200MB payloads
    total = 0
    chunk_size = 64 * 1024  # 64 KB chunks
    first_chunk = True
    actual = declared

    try:
        with target.open("wb") as out_f:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit_bytes:
                    raise HTTPException(status_code=413, detail="file_too_large")

                if first_chunk:
                    actual = _sniff_mime(chunk[:4096], declared)
                    if declared not in allowed_mimes(kind) and actual not in allowed_mimes(kind):
                        raise HTTPException(status_code=415, detail="unsupported_file_type")
                    first_chunk = False

                out_f.write(chunk)
    except Exception:
        if target.exists():
            try:
                target.unlink()
            except Exception:
                pass
        raise

    if total == 0:
        if target.exists():
            try:
                target.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=400, detail="empty_file_upload")

    # Security check 4: Deep image header/magic bytes verification with PIL
    if kind == "image":
        try:
            with Image.open(target) as img:
                img.verify()
        except Exception:
            if target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass
            raise HTTPException(status_code=400, detail="invalid_or_corrupt_image_data")

    return target, actual, total


async def extract_text_from_upload(upload: UploadFile) -> str:
    original = sanitize_filename(upload.filename or "upload.txt")
    suffix = Path(original).suffix.lower()
    if is_extension_blocked(original) or suffix not in ALLOWED_EXTENSIONS["text"]:
        raise HTTPException(status_code=400, detail="disallowed_file_extension")

    # Stream-bounded read
    data = await read_upload_bounded(upload, max_bytes("text"))

    # Format-specific parsers
    if suffix == ".docx":
        return extract_text_from_docx(data)
    elif suffix == ".pdf":
        return extract_text_from_pdf(data)
    else:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")


def cleanup_expired_uploads(max_age_hours: int = 24) -> int:
    """
    Purges temporary uploaded files older than max_age_hours to prevent disk exhaustion.
    """
    cutoff = time.time() - (max_age_hours * 3600)
    count = 0
    upload_dir = settings.upload_path
    if not upload_dir.exists():
        return 0
    for path in upload_dir.iterdir():
        if path.is_file():
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    count += 1
            except Exception:
                pass
    return count
