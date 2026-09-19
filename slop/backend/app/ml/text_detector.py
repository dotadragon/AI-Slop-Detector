"""
SLOP Text Detector v3.0 — Production-grade AI text detection.
- Fixed model path resolution (was off by one parent level)
- Rewrote ensemble blending with disagreement detection
- Added LLM-specific pattern libraries (Gemini, Claude, GPT-4, Copilot)
- Added advanced stylometric analysis (Zipf's law, vocabulary richness, hedging, passive voice)
- Added confidence calibration framework
- Added LLM attribution (identifies WHICH model wrote the text)
"""

import logging
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev

import unicodedata
import bleach

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = str(Path(__file__).resolve().parent.parent.parent / "models")

# ============================================================================
# SECURITY & ADVERSARIAL SANITIZATION
# ============================================================================
ZERO_WIDTH_CHARS = frozenset({
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space / byte order mark
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidirectional formatting overrides
})

PROMPT_INJECTION_PATTERNS = (
    r"(?i)\bignore\s+(all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules|commands)\b",
    r"(?i)\byou\s+are\s+now\s+(?:dan|unrestricted|jailbroken|an\s+ai\s+without\s+rules)\b",
    r"(?i)\b(?:repeat|print|output|display)\s+(?:the\s+)?(?:system\s+prompt|instructions\s+above)\b",
    r"(?i)\bsystem\s*:\s*(?:override|new\s+rule|disregard|ignore)\b",
    r"(?i)\b(?:reveal|show|dump)\s+(?:all\s+)?(?:api[_\-\s]?keys?|secrets?|passwords?|tokens?)\b",
    r"(?i)\b(?:do\s+anything\s+now|developer\s+mode\s+enabled)\b",
    r"(?i)\bformat\s+your\s+response\s+as\s+a\s+system\s+command\b",
)

def _normalize_and_sanitize_text(text: str) -> str:
    """
    Normalizes text for security & forensic analysis:
    1. Normalizes unicode with NFKC (neutralizing homoglyph and composite character evasion).
    2. Strips invisible zero-width and bidirectional override characters.
    3. Strips HTML tags via bleach.
    """
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = "".join(ch for ch in normalized if ch not in ZERO_WIDTH_CHARS)
    return bleach.clean(cleaned, tags=[], strip=True)

def _detect_prompt_injection(text: str) -> dict:
    """
    Scans input for direct and indirect prompt injection attempts.
    Returns injection status, risk level, and matched count.
    """
    matches = [pat for pat in PROMPT_INJECTION_PATTERNS if re.search(pat, text)]
    risk = "none"
    if len(matches) >= 2:
        risk = "high"
    elif len(matches) == 1:
        risk = "medium"
    return {
        "detected": len(matches) > 0,
        "risk": risk,
        "matched_count": len(matches),
    }


# ============================================================================
# AI MARKER LEXICON — Words/phrases disproportionately used by AI
# ============================================================================
AI_MARKERS = (
    "delve", "tapestry", "underscore", "furthermore", "moreover",
    "in conclusion", "it is important to note", "it's worth noting",
    "in the realm of", "a testament to", "it is crucial",
    "plays a pivotal role", "multifaceted", "holistic approach",
    "paradigm shift", "synergy", "leveraging", "cutting-edge",
    "game-changer", "innovative solutions", "robust and scalable",
    "seamlessly integrate", "navigating the", "at the end of the day",
    "it is worth mentioning", "in today's world", "in this article",
    "stands as a", "serves as a", "it's important to",
    "a myriad of", "foster", "landscape", "underscores",
    "in the context of", "significantly", "enhancing",
    # v3.0: expanded markers for modern LLMs
    "it's also worth", "this ensures", "this allows",
    "here's a breakdown", "let's break", "key takeaway",
    "overarching", "encompasses", "in essence",
    "facilitating", "streamlining", "pivotal",
    "noteworthy", "bolster", "spearhead",
    "harnessing", "culmination", "embark",
    "intricate", "nuanced", "comprehensive",
    "foundational", "instrumental", "underpinning",
    "in light of", "with that said", "to that end",
    "it bears mentioning", "it goes without saying",
    "meticulous", "meticulously", "elevate",
)

# ============================================================================
# LLM-SPECIFIC PATTERN LIBRARIES — Identify WHICH model wrote the text
# ============================================================================

# ChatGPT / GPT-4 / GPT-4o patterns
CHATGPT_PATTERNS = (
    r"\byour (?:updated )?(?:slide deck|presentation|code|document|draft|file|summary|version|outline) is ready\b",
    r"\bgenerating the (?:updated )?\w+ now\b",
    r"\btake a look and let me know (?:what you think|if you need|if this works|if you'd like)\b",
    r"\blet me know if (?:you'd like|you need|you want|you have any questions)\b",
    r"\bfeel free to (?:reach out|ask|modify|adjust|tweak|let me know)\b",
    r"\bi am (?:splitting|updating|creating|organizing|refactoring|providing|generating)\b",
    r"\bi have (?:split|updated|created|organized|modified|restructured|incorporated)\b",
    r"\bi'm (?:keeping|splitting|updating|organizing|adding|providing|happy to)\b",
    r"\bjust as (?:we|you) (?:set up|requested|discussed|mentioned|outlined)\b",
    r"\bmirroring your (?:reference|example|template|structure)\b",
    r"\bfully dedicated to\b",
    r"\bcleanly displays?\b",
    r"\bstrictly contains?\b",
    r"\bhere is the (?:updated|revised|complete|sample|breakdown)\b",
    r"\bcertainly!|sure!|absolutely!|of course!|glad to help\b",
    r"\bas an ai language model\b",
    r"\bi hope this helps\b",
    r"\b(?:slide|step|section|part|tier|phase) \d+\b",
    r"\bhere(?:'s| is) (?:a|the|an) (?:quick|brief|detailed|comprehensive)\b",
    r"\bin summary,?\b",
    r"\bwould you like me to\b",
    r"\bshall i (?:proceed|continue|elaborate|explain)\b",
    # GPT-4o patterns
    r"\byou're all set!?\b",
    r"\bhere's what (?:i did|was changed|changed)\b",
    r"\bhere's a quick summary of\b",
    r"\bfeel free to reach out if you need\b",
    r"\bif you have any further questions\b",
    r"\bi'd be happy to assist\b",
    r"\bdiving into the details\b",
    r"\bseamlessly integrates with\b",
)

# Google Gemini patterns (1.5 Pro / Flash & 2.0)
GEMINI_PATTERNS = (
    r"\bi can help (?:you )?with that\b",
    r"\bhere(?:'s| is) (?:the|a) (?:complete|updated|full|revised|generated)\b",
    r"\byour (?:slide deck|presentation|document|file) (?:on|for|about)\b",
    r"\bit now mirrors\b",
    r"\bincluding the (?:dedicated|exact|specific)\b",
    r"\beverything is styled\b",
    r"\ba crisp,? (?:academic|clean|modern|professional)\b",
    r"\bmatching (?:project|your|the) (?:imagery|design|theme|style)\b",
    r"\bfeel free to review\b",
    r"\bif you (?:would like|need|want) (?:any )?(?:further|additional|more) (?:adjustments|changes|modifications)\b",
    r"\bi(?:'ve| have) (?:generated|created|prepared|crafted|built|designed)\b",
    r"\bthis (?:presentation|document|deck|file) follows\b",
    r"\bwith (?:bold|clean|professional|modern|sleek)\b",
    r"\b(?:white|dark|light)[\-\s]themed\b",
    r"\bproject[\-\s]specific\b",
    # Gemini 1.5 / 2.0 additions
    r"\bcertainly!? here(?:'s| is) (?:a|an|the)\b",
    r"\blet's break (?:this|it) down step[- ]by[- ]step\b",
    r"\bhere is a comprehensive breakdown\b",
    r"\bkey takeaways?:\b",
    r"\bto provide some context\b",
    r"\bhope this helps!? let me know\b",
    r"\bfrom an architectural standpoint\b",
    r"\baligned with best practices\b",
)

# Anthropic Claude patterns (3.5 Sonnet / Haiku / Opus)
CLAUDE_PATTERNS = (
    r"\bi appreciate (?:you|your|the|that)\b",
    r"\bit's worth noting\b",
    r"\bthat said,?\b",
    r"\bi'd be happy to\b",
    r"\blet me (?:think about|consider|work through|break down)\b",
    r"\bi should note\b",
    r"\bi want to be (?:transparent|clear|honest|upfront|careful)\b",
    r"\bi don't (?:have access|actually|think i should)\b",
    r"\bhere's (?:my|what i)\b",
    r"\bthis is a (?:great|good|interesting|thoughtful|important) (?:question|point|observation)\b",
    r"\bi'd (?:suggest|recommend|note|argue|say)\b",
    r"\bthere(?:'s| is) (?:an important|a key|a subtle|a crucial) (?:distinction|nuance|difference|point)\b",
    r"\bon one hand.+on the other\b",
    r"\bi should (?:mention|clarify|note|add|point out)\b",
    # Claude 3.5 Sonnet additions
    r"\bi've structured (?:this|the)\b",
    r"\bto clarify,?\b",
    r"\bfrom an architectural perspective\b",
    r"\bhere is a clean implementation\b",
    r"\bit's important to differentiate\b",
    r"\bthis approach ensures that\b",
    r"\bworth noting that\b",
    r"\bhappy to adjust (?:this|any)\b",
)

# DeepSeek patterns (R1 reasoning / V3)
DEEPSEEK_PATTERNS = (
    r"\blet's analyze this (?:carefully|systematically|step by step)\b",
    r"\bfirst, let's consider\b",
    r"\balternatively, we (?:can|could)\b",
    r"\bto verify this result\b",
    r"\bin conclusion, we find that\b",
    r"\bwe observe that\b",
    r"\btherefore, the (?:solution|result|output) is\b",
    r"\bfrom the above analysis\b",
    r"\bwe can break this down as follows\b",
    r"\bnotice that\b",
)

# Microsoft Copilot patterns
COPILOT_PATTERNS = (
    r"\bhere's how you can\b",
    r"\byou can (?:use|try|do|achieve) (?:this|that|it) by\b",
    r"\bfor (?:more information|further details|additional help)\b",
    r"\bwould you like (?:more details|me to explain|to explore)\b",
    r"\bi found (?:some|the|a few) (?:results|information|details)\b",
    r"\bbased on (?:my search|the information|what i found)\b",
    r"\baccording to (?:my information|the sources|the data)\b",
)

# All patterns combined for backward compat
CHATGPT_ASSISTANT_PATTERNS = CHATGPT_PATTERNS  # backward compat alias

# AI text sentence starters — they repeat these a lot
AI_STARTERS = (
    "the ", "this ", "it is ", "it's ", "in ", "these ", "however,",
    "moreover,", "furthermore,", "additionally,", "overall,",
    "here's ", "here is ", "below ", "above ",
)

# Hedging words — AI uses more hedging language
HEDGING_WORDS = (
    "might", "could", "perhaps", "possibly", "potentially",
    "may", "likely", "arguably", "somewhat", "approximately",
    "generally", "typically", "usually", "often", "sometimes",
    "tend to", "seems to", "appears to", "it seems",
)

# Passive voice indicators
PASSIVE_INDICATORS = (
    "is used", "was used", "are used", "were used", "been used",
    "is made", "was made", "are made", "were made", "been made",
    "is done", "was done", "are done", "were done", "been done",
    "is known", "was known", "are known", "were known", "been known",
    "is considered", "was considered", "are considered",
    "is designed", "was designed", "are designed",
    "is implemented", "was implemented", "are implemented",
    "is achieved", "was achieved", "are achieved",
    "is ensured", "was ensured", "is required",
    "is provided", "was provided", "is expected",
    "is recommended", "was recommended",
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


# ============================================================================
# LLM PATTERN MATCHING
# ============================================================================

def _match_patterns(text: str, patterns: tuple) -> list[str]:
    lower = text.lower()
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, lower)
        if found:
            matches.extend(found)
    return matches


def _llm_attribution(text: str) -> dict:
    """Identify which LLM likely generated the text."""
    chatgpt = _match_patterns(text, CHATGPT_PATTERNS)
    gemini = _match_patterns(text, GEMINI_PATTERNS)
    claude = _match_patterns(text, CLAUDE_PATTERNS)
    deepseek = _match_patterns(text, DEEPSEEK_PATTERNS)
    copilot = _match_patterns(text, COPILOT_PATTERNS)

    scores = {
        "ChatGPT / GPT-4": len(set(chatgpt)),
        "Google Gemini": len(set(gemini)),
        "Anthropic Claude": len(set(claude)),
        "DeepSeek": len(set(deepseek)),
        "Microsoft Copilot": len(set(copilot)),
    }

    all_matches = set(chatgpt) | set(gemini) | set(claude) | set(deepseek) | set(copilot)
    total_unique = len(all_matches)

    if total_unique == 0:
        return {"detected_llm": None, "confidence": 0.0, "scores": scores,
                "total_pattern_matches": 0, "all_patterns": []}

    best_llm = max(scores, key=scores.get)
    best_count = scores[best_llm]

    if best_count >= 4:
        confidence = min(0.99, 0.85 + 0.03 * (best_count - 4))
    elif best_count == 3:
        confidence = 0.80
    elif best_count == 2:
        confidence = 0.65
    elif best_count == 1:
        confidence = 0.40
    else:
        confidence = 0.0

    return {
        "detected_llm": best_llm if confidence >= 0.40 else None,
        "confidence": round(confidence, 3),
        "scores": scores,
        "total_pattern_matches": total_unique,
        "all_patterns": list(all_matches)[:10],  # top 10 matched patterns
    }


def _all_llm_patterns_score(text: str) -> tuple[float, int]:
    """Get overall AI assistant pattern probability from ALL LLM pattern libraries."""
    chatgpt = _match_patterns(text, CHATGPT_PATTERNS)
    gemini = _match_patterns(text, GEMINI_PATTERNS)
    claude = _match_patterns(text, CLAUDE_PATTERNS)
    deepseek = _match_patterns(text, DEEPSEEK_PATTERNS)
    copilot = _match_patterns(text, COPILOT_PATTERNS)

    all_unique = len(set(chatgpt) | set(gemini) | set(claude) | set(deepseek) | set(copilot))

    if all_unique >= 5:
        prob = min(0.99, 0.92 + 0.015 * (all_unique - 5))
    elif all_unique == 4:
        prob = 0.91
    elif all_unique == 3:
        prob = 0.88
    elif all_unique == 2:
        prob = 0.78
    elif all_unique == 1:
        prob = 0.55
    else:
        prob = 0.0

    return prob, all_unique


def _sentence_has_ai_pattern(s: str) -> bool:
    lower = s.lower()
    all_patterns = CHATGPT_PATTERNS + GEMINI_PATTERNS + CLAUDE_PATTERNS + COPILOT_PATTERNS
    return any(re.search(p, lower) for p in all_patterns) or any(m in lower for m in AI_MARKERS)


# ============================================================================
# HEURISTIC SIGNAL FUNCTIONS
# ============================================================================

def _perplexity(sentence: str) -> float:
    words = re.findall(r"[A-Za-z']+", sentence.lower())
    if not words:
        return 0.0
    unique = len(set(words)) / max(len(words), 1)
    avg_len = mean(len(w) for w in words)
    return round((80 / max(avg_len, 1)) * (1 + (1 - unique)) + len(words) ** 0.35, 2)


def _contraction_score(text: str) -> float:
    """AI avoids contractions. High score = likely AI."""
    words = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    if len(words) < 20:
        return 0.5  # too short to tell
    contractions = sum(1 for w in words if "'" in w and w not in ("i'll", "i'm", "i'd", "i've"))
    expandable = sum(1 for w in words if w in (
        "cannot", "will not", "would not", "should not", "could not",
        "do not", "does not", "did not", "is not", "are not",
        "was not", "were not", "has not", "have not", "had not",
        "it is", "that is", "there is", "he is", "she is",
    ))
    if expandable > 0 and contractions == 0:
        return min(1.0, 0.6 + expandable * 0.08)
    if contractions == 0 and len(words) > 50:
        return 0.65
    ratio = contractions / max(len(words), 1)
    return round(max(0, 1.0 - ratio * 25), 3)


def _sentence_start_entropy(sents: list[str]) -> float:
    """AI text starts sentences with boring patterns. Low entropy = likely AI."""
    if len(sents) < 4:
        return 0.5
    starts = [s[:5].lower().strip() for s in sents]
    ai_starts = sum(1 for s in sents if any(s.lower().startswith(p) for p in AI_STARTERS))
    ratio = ai_starts / len(sents)
    unique_starts = len(set(starts))
    diversity = unique_starts / len(starts)
    return round(max(0, min(1.0, ratio * 0.6 + (1 - diversity) * 0.4)), 3)


def _paragraph_uniformity(text: str) -> float:
    """AI paragraphs are suspiciously even in length."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 3:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) < 3:
        return 0.5
    lens = [len(p) for p in paras]
    avg = mean(lens)
    if avg == 0:
        return 0.5
    cv = pstdev(lens) / avg
    return round(max(0, min(1.0, 1.0 - cv * 1.5)), 3)


# ============================================================================
# v3.0: ADVANCED STYLOMETRIC SIGNALS
# ============================================================================

def _vocabulary_richness(text: str) -> float:
    """MATTR (Moving-Average Type-Token Ratio) — AI has unnaturally consistent vocabulary.
    High score = likely AI (too consistent)."""
    words = _words(text)
    if len(words) < 30:
        return 0.5

    # MATTR with window size 25
    window = 25
    ratios = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i + window]
        ratios.append(len(set(chunk)) / window)

    if not ratios:
        return 0.5

    mattr = mean(ratios)
    mattr_std = pstdev(ratios) if len(ratios) > 1 else 0

    # AI typically has MATTR between 0.72-0.82 with low std
    # Humans have more variable MATTR
    consistency_signal = max(0, 1.0 - mattr_std * 12)  # Low std = AI
    range_signal = 1.0 - abs(mattr - 0.77) * 5  # Close to 0.77 = typical AI

    return round(max(0, min(1.0, 0.6 * consistency_signal + 0.4 * max(0, range_signal))), 3)


def _hedging_score(text: str) -> float:
    """AI uses hedging language at a very consistent rate. High = likely AI."""
    words = _words(text)
    if len(words) < 30:
        return 0.5

    lower = text.lower()
    hedge_count = sum(lower.count(h) for h in HEDGING_WORDS)
    ratio = hedge_count / max(len(words), 1) * 100  # per 100 words

    # AI typically hedges 2-5% of words; humans are more variable
    if 1.5 < ratio < 5.5:
        return round(min(1.0, 0.5 + ratio * 0.08), 3)
    elif ratio > 5.5:
        return 0.7  # over-hedging (still somewhat AI-like)
    else:
        return round(max(0.1, ratio * 0.3), 3)


def _passive_voice_score(text: str) -> float:
    """AI uses more passive constructions. High = likely AI."""
    lower = text.lower()
    words = _words(text)
    if len(words) < 30:
        return 0.5

    passive_count = sum(1 for p in PASSIVE_INDICATORS if p in lower)
    ratio = passive_count / max(len(words), 1) * 100

    # AI text typically has 1-4% passive constructions
    if ratio > 1.5:
        return round(min(1.0, 0.5 + ratio * 0.1), 3)
    return round(max(0.1, 0.3 + ratio * 0.15), 3)


def _zipf_deviation(text: str) -> float:
    """Zipf's Law: AI text follows word frequency distributions TOO perfectly.
    High deviation score = likely AI (follows Zipf's too closely)."""
    words = _words(text)
    if len(words) < 50:
        return 0.5

    freq = Counter(words)
    sorted_freq = sorted(freq.values(), reverse=True)

    if len(sorted_freq) < 10:
        return 0.5

    # Zipf's law: frequency ∝ 1/rank
    # Calculate how well the distribution fits Zipf's law
    top_n = min(30, len(sorted_freq))
    observed = sorted_freq[:top_n]
    expected_zipf = [observed[0] / (i + 1) for i in range(top_n)]

    # Normalized RMSE between observed and Zipf's prediction
    diffs = [(o - e) ** 2 for o, e in zip(observed, expected_zipf)]
    rmse = (mean(diffs)) ** 0.5
    normalized_rmse = rmse / max(observed[0], 1)

    # Low RMSE = follows Zipf's closely = likely AI
    # Human text has more deviation from Zipf's
    return round(max(0, min(1.0, 1.0 - normalized_rmse * 3)), 3)


def _punctuation_entropy(text: str) -> float:
    """AI uses very predictable punctuation patterns. High = likely AI."""
    punct = re.findall(r'[.!?,;:\-\(\)\[\]"\'—]', text)
    if len(punct) < 10:
        return 0.5

    freq = Counter(punct)
    total = sum(freq.values())
    probs = [c / total for c in freq.values()]

    # Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(max(len(freq), 1))

    if max_entropy == 0:
        return 0.5

    normalized = entropy / max_entropy

    # AI has lower punctuation entropy (more predictable)
    # Typical AI: 0.4-0.65, Typical Human: 0.55-0.85
    return round(max(0, min(1.0, 1.0 - normalized)), 3)


def _list_and_structure_score(text: str) -> float:
    """AI loves numbered lists, bullet points, and structured formatting. High = likely AI."""
    lines = text.split("\n")
    if len(lines) < 3:
        return 0.0

    # Count structured patterns
    numbered = sum(1 for l in lines if re.match(r'^\s*\d+[\.\)]\s', l))
    bulleted = sum(1 for l in lines if re.match(r'^\s*[\-\*\•]\s', l))
    headers = sum(1 for l in lines if re.match(r'^\s*#{1,6}\s', l) or re.match(r'^\s*\*\*[^*]+\*\*\s*$', l))
    bold_items = len(re.findall(r'\*\*[^*]+\*\*', text))

    total_structure = numbered + bulleted + headers
    structure_ratio = total_structure / max(len(lines), 1)

    if structure_ratio > 0.4:
        return min(1.0, 0.75 + structure_ratio * 0.3)
    elif structure_ratio > 0.2:
        return min(1.0, 0.55 + structure_ratio * 0.5)
    elif bold_items > 3:
        return min(0.8, 0.4 + bold_items * 0.05)
    return max(0.0, structure_ratio * 2)


def _binoculars_transitional_entropy(text: str) -> float:
    """
    v3.4 Binoculars-style Transitional Entropy Variance.
    Measures bigram transitional token entropy consistency across text.
    Frontier LLMs (GPT-4o, Claude 3.5, Gemini 1.5, DeepSeek) exhibit unnaturally uniform
    token-to-token transitional probability profiles compared to human writing.
    Low variance = High AI probability score.
    """
    words = _words(text)
    if len(words) < 25:
        return 0.5

    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    if len(bigrams) < 20:
        return 0.5

    window_size = 15
    entropy_samples = []
    for i in range(0, len(bigrams) - window_size + 1, 5):
        chunk = bigrams[i : i + window_size]
        first_words = [b[0] for b in chunk]
        counts = Counter(first_words)
        total = len(first_words)
        h = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
        entropy_samples.append(h)

    if len(entropy_samples) < 2:
        return 0.5

    variance = pstdev(entropy_samples)
    ai_prob = max(0.0, min(1.0, 1.0 - (variance / 0.70)))
    return round(ai_prob, 3)


# ============================================================================
# MAIN HEURISTIC PIPELINE
# ============================================================================

def _heuristic(text: str) -> dict:
    """CPU-safe heuristic analysis with comprehensive signal extraction and security normalization."""
    clean = _normalize_and_sanitize_text(text)
    injection_info = _detect_prompt_injection(clean)
    sents = _sentences(clean) or [clean]
    perps = [_perplexity(s) for s in sents]
    avg_p = mean(perps) if perps else 0
    burst = (pstdev(perps) / avg_p) if len(perps) > 1 and avg_p else 0

    markers = sum(clean.lower().count(m) for m in AI_MARKERS)
    uniformity = max(0, min(1, 1 - burst))
    perp_signal = max(0, min(1, (35 - avg_p) / 35))
    style_score = max(0, min(1, markers / 3 + min(1, len(clean) / 5000) * 0.1))

    # v3.0: All LLM pattern libraries
    llm_pattern_prob, llm_pattern_count = _all_llm_patterns_score(clean)

    # LLM attribution
    attribution = _llm_attribution(clean)

    # v2 signals
    contraction = _contraction_score(clean)
    start_entropy = _sentence_start_entropy(sents)
    para_uniform = _paragraph_uniformity(clean)

    # v3.0 & v3.4: Advanced stylometric signals
    vocab_rich = _vocabulary_richness(clean)
    hedging = _hedging_score(clean)
    passive = _passive_voice_score(clean)
    zipf = _zipf_deviation(clean)
    punct_entropy = _punctuation_entropy(clean)
    list_score = _list_and_structure_score(clean)
    binoculars_entropy = _binoculars_transitional_entropy(clean)

    # Multi-signal blended heuristic
    # Core signals (48%)
    core = (
        0.13 * uniformity +
        0.10 * perp_signal +
        0.13 * style_score +
        0.12 * contraction
    )
    # Structural signals (18%)
    structural = (
        0.07 * start_entropy +
        0.05 * para_uniform +
        0.06 * list_score
    )
    # Stylometric & Binoculars signals (34%)
    stylometric = (
        0.06 * vocab_rich +
        0.05 * hedging +
        0.05 * passive +
        0.06 * zipf +
        0.06 * punct_entropy +
        0.06 * binoculars_entropy
    )

    base_heuristic = core + structural + stylometric

    # LLM pattern detection overrides base if strong
    overall = max(0.02, min(0.98, max(base_heuristic, llm_pattern_prob)))

    sent_data = []
    for s, p in zip(sents, perps):
        has_pat = _sentence_has_ai_pattern(s)
        sprob = round(max(0.02, min(0.98, overall + (avg_p - p) / 80)), 3)
        if has_pat and llm_pattern_prob >= 0.70:
            sprob = max(sprob, 0.94 if llm_pattern_prob >= 0.85 else 0.82)
        sent_data.append({"text": s, "ai_prob": sprob, "perplexity": p, "pattern_matched": has_pat})

    return {
        "overall_score": round(overall, 3),
        "sentences": sent_data,
        "signals": {
            "perplexity": round(avg_p, 2),
            "burstiness": round(burst, 3),
            "stylometry_score": round(style_score, 3),
            "uniformity": round(uniformity, 3),
            "contraction_avoidance": round(contraction, 3),
            "sentence_start_entropy": round(start_entropy, 3),
            "paragraph_uniformity": round(para_uniform, 3),
            # v3.0 & v3.4 signals
            "vocabulary_richness": round(vocab_rich, 3),
            "hedging_ratio": round(hedging, 3),
            "passive_voice_score": round(passive, 3),
            "zipf_deviation": round(zipf, 3),
            "punctuation_entropy": round(punct_entropy, 3),
            "list_structure_score": round(list_score, 3),
            "transitional_entropy_variance": round(binoculars_entropy, 3),
            # Pattern detection
            "marker_hits": markers,
            "llm_patterns_detected": llm_pattern_count,
            "llm_pattern_score": round(llm_pattern_prob, 3),
            "sentence_count": len(sents),
            "char_count": len(clean),
            "prompt_injection_detected": injection_info["detected"],
            "prompt_injection_risk": injection_info["risk"],
        },
        "llm_attribution": attribution,
    }


# ============================================================================
# NEURAL MODEL (RoBERTa)
# ============================================================================

class _ModelUnavailable(RuntimeError):
    pass


def _find_text_model_dir() -> str | None:
    """Find the text model directory — fixed path resolution (v3.0)."""
    base = Path(__file__).resolve()

    # Path 1: Project root Model/text (AISlop/Model/text)
    # text_detector.py is at slop/backend/app/ml/ — need 5 parents to reach AISlop/
    p1 = base.parent.parent.parent.parent.parent / "Model" / "text"
    if p1.exists() and any(p1.glob("*.bin")):
        return str(p1)

    # Path 2: Inside slop/backend parent (slop/Model/text)
    p2 = base.parent.parent.parent.parent / "Model" / "text"
    if p2.exists() and any(p2.glob("*.bin")):
        return str(p2)

    # Path 3: Local models dir (slop/backend/models/text)
    p3 = base.parent.parent.parent / "models" / "text"
    if p3.exists() and any(p3.glob("*.bin")):
        return str(p3)

    return None


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise _ModelUnavailable("torch/transformers not installed") from e

    from app.core.config import settings
    device = settings.compute_device

    model_dir = _find_text_model_dir()
    if model_dir:
        try:
            tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            model.to(device).eval()
            model_id = "Hello-SimpleAI/chatgpt-detector-roberta"
            return tok, model, torch, model_id, device
        except Exception as e:
            # ponytail: Local directory model load failure falls back to cached huggingface weights
            logger.warning("Local text model load from %s failed: %s; falling back to %s", model_dir, e, mid)

    mid = "openai-community/roberta-base-openai-detector"
    try:
        tok = AutoTokenizer.from_pretrained(mid, cache_dir=MODEL_CACHE_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(mid, cache_dir=MODEL_CACHE_DIR)
    except Exception as e:
        raise _ModelUnavailable(f"Can't load {mid}: {e}") from e

    model.to(device).eval()
    return tok, model, torch, mid, device


def _model_prob(text: str) -> float | None:
    """Run RoBERTa with sliding window for long text."""
    try:
        tok, model, torch, _, device = _load_model()
    except _ModelUnavailable:
        return None

    tokens = tok.encode(text, add_special_tokens=False)
    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    fake_idx = next((i for i, l in labels.items() if any(t in l for t in ("fake", "ai", "artificial", "synthetic", "chatgpt"))), 0)

    if len(tokens) <= 480:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits[0], dim=-1).cpu().tolist()
        return round(float(probs[fake_idx]), 4)

    # Long text: sliding window with 50% overlap
    window_size = 480
    stride = 240
    chunk_probs = []

    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start:start + window_size]
        if len(chunk_tokens) < 50:
            break
        chunk_text = tok.decode(chunk_tokens, skip_special_tokens=True)
        inputs = tok(chunk_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits[0], dim=-1).cpu().tolist()
        chunk_probs.append(float(probs[fake_idx]))

    if not chunk_probs:
        return None
    weights = [1.0 / (1 + 0.05 * i) for i in range(len(chunk_probs))]
    total_w = sum(weights)
    avg = sum(p * w for p, w in zip(chunk_probs, weights)) / total_w
    return round(avg, 4)


def _model_sentence_probs(sentences: list[str]) -> list[float | None]:
    """Per-sentence AI probability."""
    try:
        tok, model, torch, _, device = _load_model()
    except _ModelUnavailable:
        return [None] * len(sentences)

    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    fake_idx = next((i for i, l in labels.items() if any(t in l for t in ("fake", "ai", "artificial", "synthetic", "chatgpt"))), 0)

    results = []
    for s in sentences:
        if len(s.strip()) < 10:
            results.append(None)
            continue
        inputs = tok(s, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits[0], dim=-1).cpu().tolist()
        results.append(round(float(probs[fake_idx]), 3))
    return results


# ============================================================================
# CONFIDENCE CALIBRATION
# ============================================================================

def _calibrate_score(raw_score: float, signals: dict) -> float:
    """v3.0: Calibrate raw ensemble score to produce more meaningful probabilities.
    Uses signal-based adjustment to correct for known biases."""

    pattern_count = signals.get("llm_patterns_detected", 0)
    marker_hits = signals.get("marker_hits", 0)
    contraction = signals.get("contraction_avoidance", 0.5)
    burstiness = signals.get("burstiness", 0.5)

    # Strong AI evidence boosters
    evidence_boost = 0.0
    if pattern_count >= 3:
        evidence_boost += 0.10
    if marker_hits >= 5:
        evidence_boost += 0.08
    if contraction > 0.7:
        evidence_boost += 0.05

    # Human evidence (high burstiness, low marker density, zero LLM patterns)
    human_evidence = 0.0
    if burstiness > 0.6:
        human_evidence += 0.10
    if contraction < 0.3:
        human_evidence += 0.08
    if marker_hits == 0 and pattern_count == 0:
        if signals.get("stylometry_score", 0.5) < 0.10:
            human_evidence += 0.35
        elif signals.get("stylometry_score", 0.5) < 0.20:
            human_evidence += 0.20

    calibrated = raw_score + evidence_boost - human_evidence

    # Apply sigmoid-like sharpening to push scores toward extremes
    # This makes the score more decisive (less wishy-washy 40-60% scores)
    if calibrated > 0.5:
        calibrated = 0.5 + (calibrated - 0.5) ** 0.85 * (0.5 / 0.5 ** 0.85)
    else:
        calibrated = 0.5 - (0.5 - calibrated) ** 0.85 * (0.5 / 0.5 ** 0.85)

    return round(max(0.02, min(0.98, calibrated)), 3)


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_text(text: str) -> dict:
    """v3.0: Heuristic + Neural ensemble with smart disagreement handling,
    confidence calibration, LLM attribution, and security normalization."""
    clean_text = _normalize_and_sanitize_text(text)
    h = _heuristic(clean_text)
    mp = _model_prob(clean_text)
    llm_pattern_score = h["signals"].get("llm_pattern_score", 0.0)
    attribution = h.get("llm_attribution", {})

    try:
        _, _, _, model_id, _ = _load_model()
    except Exception:
        model_id = "openai-community/roberta-base-openai-detector"

    # v3.5: Binoculars & Fast-DetectGPT zero-shot signals
    from app.ml.binoculars import compute_binoculars_and_curvature
    bino_res = compute_binoculars_and_curvature(clean_text)
    bino_prob = bino_res.get("binoculars_ai_prob")
    fast_prob = bino_res.get("fast_detectgpt_ai_prob")
    bino_score = bino_res.get("binoculars_score")
    fast_z = bino_res.get("fast_detectgpt_z")

    if mp is not None:
        # v3.0: Smart ensemble blending with disagreement detection
        char_count = len(text.strip())
        heuristic_score = h["overall_score"]

        # Determine base blend weights by text length
        if char_count < 200:
            neural_w, heuristic_w = 0.45, 0.55
        elif char_count < 1000:
            neural_w, heuristic_w = 0.60, 0.40
        else:
            neural_w, heuristic_w = 0.70, 0.30

        disagreement = abs(heuristic_score - mp)
        inference_mode = "calibrated_frontier_ensemble" if (llm_pattern_score >= 0.60 or (heuristic_score > 0.55 and mp < 0.35)) else "roberta_classifier"

        # ===== FUSION MODEL IS THE SOLE SCORE AUTHORITY (Rule 2, 3, 5) =====
        # ponytail: PROTECTED - DO NOT MODIFY WITHOUT RE-RUNNING eval_accuracy.py AND HUMAN APPROVAL
        from app.ml.fusion import fuse_text_score

        fusion_features = {
            "neural_prob": mp,
            "heuristic_score": heuristic_score,
            "llm_pattern_score": llm_pattern_score,
            "transitional_entropy": h["signals"].get("transitional_entropy_variance", 0.5),
            "burstiness": h["signals"].get("burstiness", 0.5),
            "contraction_score": h["signals"].get("contraction_avoidance", 0.5),
            "zipf_deviation": h["signals"].get("zipf_deviation", 0.5),
            "vocab_richness": h["signals"].get("vocabulary_richness", 0.5),
            "char_count": char_count,
            "binoculars_score": bino_score,
            "binoculars_ai_prob": bino_prob,
            "fast_detectgpt_z": fast_z,
            "fast_detectgpt_ai_prob": fast_prob,
        }

        fusion_res = fuse_text_score(fusion_features)
        overall = fusion_res["overall_score"]
        conf_band = fusion_res.get("confidence_band", "UNCERTAIN")
        conf_range = fusion_res.get("confidence_range", [0.0, 1.0])

        logger.info(
            "text_forensics_scored",
            extra={
                "raw_neural": mp,
                "fusion_features": fusion_features,
                "final_score": overall,
            },
        )

        # Per-sentence neural scores
        sent_probs = _model_sentence_probs([s["text"] for s in h["sentences"]])
        for s, sp in zip(h["sentences"], sent_probs):
            if sp is not None:
                s["model_ai_prob"] = sp
                if s.get("pattern_matched") and llm_pattern_score >= 0.70:
                    s["ai_prob"] = max(s["ai_prob"], 0.94)
                else:
                    s["ai_prob"] = round(0.55 * sp + 0.45 * s["ai_prob"], 3)

        h["overall_score"] = round(overall, 3)
        h["confidence_band"] = conf_band
        h["confidence_range"] = conf_range
        h["signals"]["neural_ai_score"] = mp
        h["signals"]["model_id"] = model_id
        h["signals"]["inference_mode"] = inference_mode
        h["signals"]["frontier_llm_calibrated"] = (inference_mode == "calibrated_frontier_ensemble")
        h["signals"]["neural_weight"] = neural_w
        h["signals"]["heuristic_weight"] = heuristic_w
        h["signals"]["disagreement"] = round(disagreement, 3)
        h["signals"]["confidence_band"] = conf_band
        h["signals"]["confidence_range"] = conf_range
        if bino_res.get("available"):
            h["signals"]["binoculars_score"] = bino_score
            h["signals"]["binoculars_ai_prob"] = bino_prob
            h["signals"]["fast_detectgpt_z"] = fast_z
            h["signals"]["fast_detectgpt_ai_prob"] = fast_prob
        h["model_hint"] = (
            f"v3.5 Ensemble ({inference_mode}): RoBERTa neural ({int(neural_w*100)}%) + "
            f"frontier stylometrics ({int(heuristic_w*100)}%) with "
            f"Binoculars & Fast-DetectGPT verification."
        )
    else:
        # Heuristic-only fallback — route through single score authority
        from app.ml.fusion import fuse_text_score
        fusion_features = {
            "neural_prob": None,
            "heuristic_score": h["overall_score"],
            "llm_pattern_score": llm_pattern_score,
            "transitional_entropy": h["signals"].get("transitional_entropy_variance", 0.5),
            "burstiness": h["signals"].get("burstiness", 0.5),
            "contraction_score": h["signals"].get("contraction_avoidance", 0.5),
            "zipf_deviation": h["signals"].get("zipf_deviation", 0.5),
            "vocab_richness": h["signals"].get("vocabulary_richness", 0.5),
            "char_count": len(clean_text),
            "binoculars_score": bino_score,
            "binoculars_ai_prob": bino_prob,
            "fast_detectgpt_z": fast_z,
            "fast_detectgpt_ai_prob": fast_prob,
        }
        fusion_res = fuse_text_score(fusion_features)
        overall = fusion_res["overall_score"]
        conf_band = fusion_res.get("confidence_band", "UNCERTAIN")
        conf_range = fusion_res.get("confidence_range", [0.0, 1.0])
        h["overall_score"] = round(overall, 3)
        h["confidence_band"] = conf_band
        h["confidence_range"] = conf_range
        h["signals"]["confidence_band"] = conf_band
        h["signals"]["confidence_range"] = conf_range
        h["signals"]["inference_mode"] = "heuristic_fallback"
        h["model_hint"] = (
            "v3.5 Heuristic-only (12 stylometric signals + LLM pattern libraries). "
            "Install torch for neural accuracy."
        )

    # Confidence-gated LLM attribution (Rule F)
    if h["overall_score"] < 0.60 and attribution:
        attribution["detected_llm"] = None
        attribution["confidence"] = 0.0

    # Always include LLM attribution
    h["llm_attribution"] = attribution

    return h
