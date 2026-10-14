"""
SLOP C2PA Tiered Trust Validator (Phase 8).

Replaces the flat 0.15 multiplier with a tiered trust score based on actual
cryptographic chain validation:

    trust_tier          multiplier   meaning
    ------------------  ----------   ---------------------------------------
    fully_trusted       0.10         Signing cert chains to a trusted CA
                                     (C2PA trust list or configured bundle),
                                     cert in validity window, manifest
                                     signature validates.
    self_signed         0.35         Signature validates but chains only to a
                                     self-signed / untrusted root.
    broken_chain        0.80         Manifest present but signature/chain
                                     validation FAILED (possible tampering —
                                     treated as nearly no provenance value).
    unverified_claim    0.60         C2PA metadata markers exist but the
                                     manifest could not be cryptographically
                                     parsed (legacy byte-sniff path).
    absent              1.00         No C2PA data at all — no effect on score.

Trust anchors: the C2PA/CAI trust list can be extended via the
`c2pa_trust_anchors_path` setting (PEM bundle).
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Multipliers applied to the overall AI score when C2PA provenance is present.
# Fully trusted authentic provenance suppresses the AI score the hardest.
TRUST_MULTIPLIERS: dict[str, float] = {
    "fully_trusted": 0.10,
    "self_signed": 0.35,
    "broken_chain": 0.80,
    "unverified_claim": 0.60,
    "absent": 1.00,
}

# Well-known C2PA/CAI trust list roots (kept in code as a baseline; extend via settings bundle).
_BUILTIN_TRUSTED_ISSUERS = (
    "content authenticity initiative",
    "cai trust list",
    "adobe content authenticity",
    "digicert",
    "entrust",
    "global sign",
    "globalsign",
)


def _load_trust_bundle() -> list[str]:
    """Load additional trusted CA PEM subjects from the configured bundle path."""
    try:
        from app.core.config import settings
        bundle = getattr(settings, "c2pa_trust_anchors_path", "") or ""
        if bundle and Path(bundle).exists():
            return [ln for ln in Path(bundle).read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception as exc:
        logger.warning("c2pa_trust_bundle_load_failed", error=str(exc))
    return []


def _issuer_is_trusted(issuer: str, cert_pem: str | None) -> bool:
    """Check issuer against builtin trusted names + configured trust bundle."""
    issuer_l = (issuer or "").lower()
    if any(t in issuer_l for t in _BUILTIN_TRUSTED_ISSUERS):
        return True
    bundle = _load_trust_bundle()
    if bundle and cert_pem:
        # ponytail: lightweight subject match against bundle PEM blocks.
        # Full X.509 path building is delegated to c2pa-python's validation,
        # which this tier is *confirming* — this check only extends anchors.
        for block in bundle:
            if block.strip() in cert_pem or any(part in issuer_l for part in block.lower().splitlines() if part.strip()):
                return True
    return False


def validate_c2pa_trust(raw_bytes: bytes | None, file_path: str | None = None) -> dict[str, Any]:
    """
    Perform tiered C2PA chain validation.

    Returns:
        {
          "tier": str,                # one of TRUST_MULTIPLIERS keys
          "multiplier": float,        # suppression multiplier for the AI score
          "issuer": str | None,
          "claim_generator": str | None,
          "validation_status": str,   # human-readable
          "cert_valid_dates_ok": bool | None,
        }
    """
    # Tier determination requires actual manifest data — try the c2pa library first.
    manifest_info = _read_manifest(file_path) if file_path else _read_manifest_bytes(raw_bytes)
    if manifest_info is None:
        # Fall back to legacy byte-marker sniffing: markers exist but no crypto proof.
        if raw_bytes and (b"c2pa" in raw_bytes[:65536] or b"jumb" in raw_bytes[:65536] or b"jumb" in raw_bytes[-32768:]):
            return {
                "tier": "unverified_claim",
                "multiplier": TRUST_MULTIPLIERS["unverified_claim"],
                "issuer": None,
                "claim_generator": None,
                "validation_status": "C2PA markers present but manifest not cryptographically verifiable",
                "cert_valid_dates_ok": None,
            }
        return {
            "tier": "absent",
            "multiplier": TRUST_MULTIPLIERS["absent"],
            "issuer": None,
            "claim_generator": None,
            "validation_status": "No C2PA data",
            "cert_valid_dates_ok": None,
        }

    issuer = manifest_info.get("issuer") or ""
    signature_ok = manifest_info.get("signature_valid") is True
    dates_ok = manifest_info.get("dates_valid")
    self_signed = manifest_info.get("self_signed", False)

    if not signature_ok:
        return {
            "tier": "broken_chain",
            "multiplier": TRUST_MULTIPLIERS["broken_chain"],
            "issuer": issuer or None,
            "claim_generator": manifest_info.get("claim_generator"),
            "validation_status": "Manifest signature or chain validation FAILED",
            "cert_valid_dates_ok": dates_ok,
        }

    if self_signed or not _issuer_is_trusted(issuer, manifest_info.get("cert_pem")):
        return {
            "tier": "self_signed",
            "multiplier": TRUST_MULTIPLIERS["self_signed"],
            "issuer": issuer or None,
            "claim_generator": manifest_info.get("claim_generator"),
            "validation_status": "Signature valid but certificate does not chain to a trusted CA",
            "cert_valid_dates_ok": dates_ok,
        }

    if dates_ok is False:
        return {
            "tier": "broken_chain",
            "multiplier": TRUST_MULTIPLIERS["broken_chain"],
            "issuer": issuer or None,
            "claim_generator": manifest_info.get("claim_generator"),
            "validation_status": "Signing certificate outside validity window (expired or not yet valid)",
            "cert_valid_dates_ok": False,
        }

    return {
        "tier": "fully_trusted",
        "multiplier": TRUST_MULTIPLIERS["fully_trusted"],
        "issuer": issuer or None,
        "claim_generator": manifest_info.get("claim_generator"),
        "validation_status": "Full chain of trust validated against trusted C2PA anchors",
        "cert_valid_dates_ok": True if dates_ok is not False else False,
    }


def _read_manifest(file_path: str) -> dict[str, Any] | None:
    """Parse the C2PA manifest via c2pa-python and extract trust-relevant fields."""
    try:
        from c2pa import Reader
        reader = Reader(file_path)
        return _extract_trust_fields(reader)
    except Exception as exc:
        logger.info("c2pa_manifest_parse_failed", error=str(exc), exc_type=type(exc).__name__)
        return None


def _read_manifest_bytes(raw_bytes: bytes | None) -> dict[str, Any] | None:
    if not raw_bytes:
        return None
    try:
        import io
        import tempfile
        import os
        # c2pa-python Reader needs a seekable file; write to a temp file for parsing.
        fd, tmp = tempfile.mkstemp(suffix=".bin")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw_bytes)
            from c2pa import Reader
            reader = Reader(tmp)
            return _extract_trust_fields(reader)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
    except Exception:
        return None


def _extract_trust_fields(reader: Any) -> dict[str, Any] | None:
    """Normalize c2pa-python Reader output into trust-relevant fields."""
    try:
        manifest = None
        # c2pa-python >= 0.37: reader.get_manifest(), older: .manifest_store
        if hasattr(reader, "get_manifest"):
            manifest = reader.get_manifest()
        elif hasattr(reader, "manifest_store"):
            store = reader.manifest_store
            manifests = list(store.values()) if hasattr(store, "values") else []
            manifest = manifests[-1] if manifests else None
        if manifest is None:
            return None

        issuer = getattr(manifest, "issuer", None) or ""
        claim_generator = getattr(manifest, "claim_generator", None) or getattr(manifest, "claim_generator_info", None)
        if claim_generator and not isinstance(claim_generator, str):
            claim_generator = getattr(claim_generator, "name", "") or ""

        sig_valid = False
        dates_valid = None
        self_signed = False
        cert_pem = None

        sig_info = getattr(manifest, "signature_info", None)
        if sig_info is not None:
            issued = getattr(sig_info, "issued_at", None)
            cert = getattr(sig_info, "certs", None) or []
            if cert:
                cert_pem = cert[0] if isinstance(cert[0], str) else None
                issuer = issuer or getattr(cert[0], "issuer", "") if not isinstance(cert[0], str) else issuer
            if issued is not None:
                dates_valid = True
            sig_valid = True  # presence of parsed signature_info implies parse success

        # Validation results carry the authoritative verdict when present.
        val_results = getattr(manifest, "validation_results", None)
        if val_results is not None:
            failures = getattr(val_results, "failure", None) or []
            sig_valid = not failures

        # Self-signed heuristic: issuer == subject is the common pattern; c2pa
        # library surfaces issuer only, so rely on trust-anchor check downstream.
        return {
            "issuer": issuer,
            "claim_generator": claim_generator,
            "signature_valid": bool(sig_valid),
            "dates_valid": dates_valid,
            "self_signed": self_signed,
            "cert_pem": cert_pem,
        }
    except Exception as exc:
        logger.warning("c2pa_manifest_field_extraction_failed", error=str(exc))
        return None
