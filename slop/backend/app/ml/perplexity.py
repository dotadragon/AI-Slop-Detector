"""LM perplexity feature for text fusion (Phase B4).

Perplexity under a small causal LM (Qwen2.5-0.5B, already on disk for
Binoculars) is a well-established AI-text signal: machine text sits in
high-likelihood regions of LM space, so its perplexity is systematically
lower than human text. This adds a feature axis the current 10-feature
fusion does not cover.

VRAM discipline: the model is lazily loaded, shared process-wide, and
`evict_perplexity_model()` must be called after batch use so the Swin model
can reclaim the budget (Neural-First rule). CPU works too, just slower.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_PERPLEXITY_MODEL = None
_PERPLEXITY_TOK = None
_ACTIVE_DEVICE: str | None = None


def _find_qwen_dir(variant: str) -> Path | None:
    base = Path(__file__).resolve()
    for parents_idx in (4, 3, 2):
        c = base.parents[parents_idx] / "Model" / "text" / variant
        if c.exists() and ((c / "model.safetensors").exists() or (c / "pytorch_model.bin").exists()):
            return c
    return None


def _load_perplexity_model():
    """Lazily load Qwen2.5-0.5B (base) for perplexity scoring. Returns None if unavailable."""
    global _PERPLEXITY_MODEL, _PERPLEXITY_TOK, _ACTIVE_DEVICE
    if _PERPLEXITY_MODEL is not None:
        return _PERPLEXITY_TOK, _PERPLEXITY_MODEL, _ACTIVE_DEVICE

    try:
        from app.core.config import settings
        import app.ml.binoculars as bino

        if (
            bino._OBSERVER_MODEL is not None
            and bino._OBSERVER_TOK is not None
            and getattr(bino, "_ACTIVE_DEVICE", None) == settings.compute_device
        ):
            _PERPLEXITY_MODEL = bino._OBSERVER_MODEL
            _PERPLEXITY_TOK = bino._OBSERVER_TOK
            _ACTIVE_DEVICE = bino._ACTIVE_DEVICE
            logger.info("perplexity_model_reused_observer", device=settings.compute_device)
            return _PERPLEXITY_TOK, _PERPLEXITY_MODEL, _ACTIVE_DEVICE
    except Exception:
        pass

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.info("perplexity_model_unavailable", reason="torch/transformers not installed")
        return None

    qwen_dir = _find_qwen_dir("qwen2.5_0.5b")
    if qwen_dir is None:
        logger.info("perplexity_model_unavailable", reason="qwen2.5_0.5b not found on disk")
        return None

    try:
        from app.core.config import settings

        device = settings.compute_device
        dtype = torch.float16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(str(qwen_dir), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(qwen_dir), dtype=dtype, local_files_only=True)
        model.to(device).eval()
        _PERPLEXITY_MODEL = model
        _PERPLEXITY_TOK = tok
        _ACTIVE_DEVICE = device
        logger.info("perplexity_model_loaded", device=device, source=str(qwen_dir))
        return tok, model, device
    except Exception as exc:
        logger.warning("perplexity_model_load_failed", error=str(exc), exc_type=type(exc).__name__)
        return None


def evict_perplexity_model() -> None:
    """Evict the perplexity LM from VRAM so other models can reclaim budget."""
    global _PERPLEXITY_MODEL, _PERPLEXITY_TOK, _ACTIVE_DEVICE
    try:
        import torch

        _PERPLEXITY_MODEL = None
        _PERPLEXITY_TOK = None
        _ACTIVE_DEVICE = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


_UNAVAILABLE_PERPLEXITY: dict[str, Any] = {"perplexity": float("nan"), "log_likelihood": float("nan"), "available": False}
_UNAVAILABLE_COHERENCE: dict[str, Any] = {"coherence_mean": float("nan"), "coherence_var": float("nan"), "available": False}


def compute_perplexity(text: str, max_tokens: int = 512) -> dict[str, Any]:
    """Compute causal-LM perplexity (and per-token burstiness proxy) for text.

    Returns {'perplexity': float, 'log_likelihood': float, 'available': bool}.
    Long texts are truncated to `max_tokens` — consistent truncation keeps the
    feature comparable across samples.
    """
    loaded = _load_perplexity_model()
    if loaded is None:
        return dict(_UNAVAILABLE_PERPLEXITY)

    tok, model, device = loaded
    import torch

    try:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
        if ids.shape[1] < 2:
            return dict(_UNAVAILABLE_PERPLEXITY)

        with torch.inference_mode():
            logits = model(ids).logits
            # shift: predict token t from tokens < t
            log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]
            token_ll = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

        mean_ll = float(token_ll.mean())
        ppl = float(math.exp(-mean_ll))
        return {
            "perplexity": round(ppl, 4),
            "log_likelihood": round(mean_ll, 4),
            "available": True,
        }
    except Exception as exc:
        logger.warning("perplexity_compute_failed", error=str(exc), exc_type=type(exc).__name__)
        return dict(_UNAVAILABLE_PERPLEXITY)


def compute_coherence(text: str, max_sentences: int = 12, max_tokens: int = 768) -> dict[str, Any]:
    """Discourse-level coherence feature (Phase B5).

    Splits text into sentences, embeds each with mean-pooled hidden states of
    the same Qwen causal LM, and returns the variance of cosine similarity
    between consecutive sentences. AI text (especially chat-tuned models) keeps
    discourse on-topic: consecutive-sentence similarity is high AND low-variance.
    Human text wanders: lower mean similarity, higher variance.

    Returns {'coherence_mean': float, 'coherence_var': float, 'available': bool}.
    """
    loaded = _load_perplexity_model()
    if loaded is None:
        return dict(_UNAVAILABLE_COHERENCE)

    tok, model, device = loaded
    import re as _re
    import torch

    try:
        # lightweight sentence split (avoid a dependency on nltk)
        sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if len(s.strip().split()) >= 4]
        if len(sents) < 3:
            return dict(_UNAVAILABLE_COHERENCE)
        sents = sents[:max_sentences]

        embs = []
        backbone = getattr(model, "model", None)
        for s in sents:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=128)
            ids = enc.input_ids.to(device)
            with torch.inference_mode():
                if backbone is not None:
                    out = backbone(ids)
                    h = out.last_hidden_state[0].float().cpu()  # [T, H]
                else:
                    out = model(ids, output_hidden_states=True)
                    h = out.hidden_states[-1][0].float().cpu()  # [T, H]
            mask = enc.attention_mask[0].cpu()
            pooled = (h * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp(min=1)
            embs.append(torch.nn.functional.normalize(pooled, dim=0))

        sims = [float(torch.dot(embs[i], embs[i + 1])) for i in range(len(embs) - 1)]
        mean_sim = sum(sims) / len(sims)
        var_sim = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
        return {
            "coherence_mean": round(mean_sim, 4),
            "coherence_var": round(var_sim, 4),
            "available": True,
        }
    except Exception as exc:
        logger.warning("coherence_compute_failed", error=str(exc), exc_type=type(exc).__name__)
        return dict(_UNAVAILABLE_COHERENCE)
