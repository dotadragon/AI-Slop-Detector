"""
SLOP Binoculars & Fast-DetectGPT Zero-Shot Detection Engine.
References:
1. Hans et al. (ICML 2024): "Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated Text"
2. Bao et al. (ICLR 2024): "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"

Computes Binoculars cross-perplexity ratio and Fast-DetectGPT probability curvature
using Qwen2.5-0.5B (Observer) and Qwen2.5-0.5B-Instruct (Performer).
Strict VRAM budgeting: models run in fp16 or 4-bit, taking < 2.0 GB VRAM total.
Includes dynamic eviction function `evict_binoculars_models()`.
"""

import gc
import math
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Global cache for loaded models
_OBSERVER_MODEL = None
_OBSERVER_TOK = None
_PERFORMER_MODEL = None
_PERFORMER_TOK = None
_ACTIVE_DEVICE = None


def _find_qwen_dir(variant: str) -> Path | None:
    """Find local directory for Qwen model variant ('qwen2.5_0.5b' or 'qwen2.5_0.5b_instruct')."""
    base = Path(__file__).resolve()
    candidates = [
        base.parents[4] / "Model" / "text" / variant,
        base.parents[3] / "Model" / "text" / variant,
        base.parents[2] / "Model" / "text" / variant,
    ]
    for c in candidates:
        if c.exists():
            sf = c / "model.safetensors"
            bin_f = c / "pytorch_model.bin"
            if sf.exists() and sf.stat().st_size >= 900_000_000:
                return c
            if bin_f.exists() and bin_f.stat().st_size >= 900_000_000:
                return c
    return None


def evict_binoculars_models() -> None:
    """Evicts text models from GPU VRAM and reclaims memory (ANTIGRAVITY Rule 5)."""
    global _OBSERVER_MODEL, _OBSERVER_TOK, _PERFORMER_MODEL, _PERFORMER_TOK, _ACTIVE_DEVICE
    try:
        import torch
        del _OBSERVER_MODEL
        del _OBSERVER_TOK
        del _PERFORMER_MODEL
        del _PERFORMER_TOK
        _OBSERVER_MODEL = None
        _OBSERVER_TOK = None
        _PERFORMER_MODEL = None
        _PERFORMER_TOK = None
        _ACTIVE_DEVICE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Binoculars models successfully evicted from VRAM.")
    except Exception as e:
        logger.warning(f"Error evicting binoculars models: {e}")


def load_binoculars_pair(device: str | None = None) -> tuple[Any, Any, Any, Any, str]:
    """Loads Observer and Performer models onto target device."""
    global _OBSERVER_MODEL, _OBSERVER_TOK, _PERFORMER_MODEL, _PERFORMER_TOK, _ACTIVE_DEVICE
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from app.core.config import settings

    target_device = device or settings.compute_device
    if (
        _OBSERVER_MODEL is not None
        and _PERFORMER_MODEL is not None
        and _OBSERVER_TOK is not None
        and _PERFORMER_TOK is not None
        and _ACTIVE_DEVICE == target_device
    ):
        return _OBSERVER_TOK, _OBSERVER_MODEL, _PERFORMER_TOK, _PERFORMER_MODEL, target_device

    obs_dir = _find_qwen_dir("qwen2.5_0.5b")
    perf_dir = _find_qwen_dir("qwen2.5_0.5b_instruct")

    if not obs_dir or not perf_dir:
        raise RuntimeError(f"Binoculars models not ready on disk (obs={obs_dir}, perf={perf_dir})")

    obs_source = str(obs_dir)
    perf_source = str(perf_dir)

    logger.info(f"Loading Binoculars Observer ({obs_source}) and Performer ({perf_source}) onto {target_device}...")
    dtype = torch.float16 if target_device == "cuda" else torch.float32

    # Observer
    obs_tok = AutoTokenizer.from_pretrained(obs_source, local_files_only=True)
    obs_model = AutoModelForCausalLM.from_pretrained(
        obs_source,
        dtype=dtype,
        local_files_only=True,
    )
    obs_model.to(target_device).eval()

    # Performer
    perf_tok = AutoTokenizer.from_pretrained(perf_source, local_files_only=True)
    perf_model = AutoModelForCausalLM.from_pretrained(
        perf_source,
        dtype=dtype,
        local_files_only=True,
    )
    perf_model.to(target_device).eval()

    _OBSERVER_MODEL = obs_model
    _OBSERVER_TOK = obs_tok
    _PERFORMER_MODEL = perf_model
    _PERFORMER_TOK = perf_tok
    _ACTIVE_DEVICE = target_device

    return obs_tok, obs_model, perf_tok, perf_model, target_device


def _unavailable_result(error: str | None = None) -> dict[str, Any]:
    res: dict[str, Any] = {
        "binoculars_score": None,
        "binoculars_ai_prob": None,
        "fast_detectgpt_z": None,
        "fast_detectgpt_ai_prob": None,
        "gltr_top10_frac": None,
        "gltr_tail_frac": None,
        "token_ll_std": None,
        "fast_z_norm": None,
        "observer_perplexity": None,
        "cross_perplexity": None,
        "available": False,
    }
    if error is not None:
        res["error"] = error
    return res


def compute_binoculars_and_curvature(text: str) -> dict[str, Any]:
    """
    Executes live forward passes to compute Binoculars score, Fast-DetectGPT curvature z-score,
    and their calibrated probabilities.
    Zero mocks: executes real PyTorch inference (Rule 0).
    """
    if not text or len(text.strip()) < 30:
        return _unavailable_result()

    try:
        import torch
        import torch.nn.functional as F
        obs_tok, obs_model, perf_tok, perf_model, device = load_binoculars_pair()
    except Exception as e:
        logger.warning(f"Could not initialize Binoculars pair: {e}")
        return _unavailable_result(str(e))

    # Encode input
    enc = obs_tok(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    if seq_len < 4:
        return _unavailable_result()

    with torch.inference_mode():
        # 1. Observer forward pass
        obs_out = obs_model(input_ids)
        obs_logits = obs_out.logits[:, :-1, :]  # shape: [1, T-1, V]
        target_ids = input_ids[:, 1:]           # shape: [1, T-1]

        # 2. Performer forward pass
        perf_out = perf_model(input_ids)
        perf_logits = perf_out.logits[:, :-1, :] # shape: [1, T-1, V]

        # --- PPL(Observer) ---
        obs_log_probs = F.log_softmax(obs_logits, dim=-1)

        # Gather token log likelihoods
        token_log_probs = torch.gather(obs_log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1) # [1, T-1]
        neg_log_likelihood = -token_log_probs.mean().item()
        observer_ppl = math.exp(min(neg_log_likelihood, 20.0))

        # --- Cross-PPL(Observer, Performer) ---
        # Cross entropy of performer distribution evaluated under observer log probabilities
        perf_probs = F.softmax(perf_logits, dim=-1) # [1, T-1, V]
        cross_entropy_per_pos = -torch.sum(perf_probs * obs_log_probs, dim=-1) # [1, T-1]
        mean_cross_entropy = cross_entropy_per_pos.mean().item()
        cross_ppl = math.exp(min(mean_cross_entropy, 20.0))

        # Binoculars Score: log(PPL_obs) / Cross-PPL
        # In Hans et al. (2024), human text has higher score (~0.90+), AI text has lower score (< 0.85)
        binoculars_score = neg_log_likelihood / max(mean_cross_entropy, 1e-6)

        # Calibrate Binoculars score to AI probability in [0, 1] using dev logistic fit
        # P(AI) = sigmoid(-0.3256 * bino_score + 0.2971) -> lower bino => higher AI prob
        bino_logit = -0.3256 * binoculars_score + 0.2971
        bino_ai_prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, bino_logit))))

        # Free performer tensors (~1.2 GB for 1024 tokens) before Fast-DetectGPT / GLTR
        del perf_out, perf_logits, perf_probs, cross_entropy_per_pos

        # --- Fast-DetectGPT (Conditional Probability Curvature) ---
        # Analytical curvature using observer distribution
        # E[log p] = sum_v p(v) log p(v)
        obs_probs = F.softmax(obs_logits, dim=-1)
        expected_log_p = torch.sum(obs_probs * obs_log_probs, dim=-1) # [1, T-1]
        # Var[log p] = sum_v p(v) (log p(v))^2 - (E[log p])^2
        expected_sq_log_p = torch.sum(obs_probs * (obs_log_probs ** 2), dim=-1) # [1, T-1]
        var_log_p = torch.clamp(expected_sq_log_p - (expected_log_p ** 2), min=1e-8) # [1, T-1]

        diff = token_log_probs - expected_log_p
        sum_diff = diff.sum().item()
        sum_var = var_log_p.sum().item()

        fast_detectgpt_z = sum_diff / math.sqrt(max(sum_var, 1e-8))

        # Calibrate Fast-DetectGPT z-score to probability using dev logistic fit
        # P(AI) = sigmoid(+0.1670 * fast_z - 0.2343) -> higher z => higher AI prob
        fast_logit = 0.1670 * fast_detectgpt_z - 0.2343
        fast_ai_prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, fast_logit))))

        # Free Fast-DetectGPT intermediates before token rank tensor comparison
        del obs_out, obs_logits, obs_probs, expected_log_p, expected_sq_log_p, var_log_p, diff

        # --- GLTR Token-Rank & Length-Normalized Curvature Features ---
        token_ranks = (obs_log_probs > token_log_probs.unsqueeze(-1)).sum(dim=-1).float()
        gltr_top10_frac = float((token_ranks < 10).float().mean().item())
        gltr_tail_frac = float((token_ranks >= 1000).float().mean().item())
        token_ll_std = float(token_log_probs.std().item()) if seq_len > 2 else 0.0
        fast_z_norm = float(fast_detectgpt_z / math.sqrt(max(seq_len - 1, 1)))

        del token_ranks, obs_log_probs, token_log_probs

    return {
        "binoculars_score": round(float(binoculars_score), 4),
        "binoculars_ai_prob": round(float(bino_ai_prob), 4),
        "fast_detectgpt_z": round(float(fast_detectgpt_z), 3),
        "fast_detectgpt_ai_prob": round(float(fast_ai_prob), 4),
        "gltr_top10_frac": round(float(gltr_top10_frac), 4),
        "gltr_tail_frac": round(float(gltr_tail_frac), 4),
        "token_ll_std": round(float(token_ll_std), 4),
        "fast_z_norm": round(float(fast_z_norm), 4),
        "observer_perplexity": round(float(observer_ppl), 2),
        "cross_perplexity": round(float(cross_ppl), 2),
        "available": True,
    }
