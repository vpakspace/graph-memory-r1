"""Evaluation metrics for Graph-Memory-R1.

Metrics:
- F1 Score (token-level)
- Exact Match (normalized)
- BLEU-1 (unigram precision)
"""

from __future__ import annotations

import re
from collections import Counter


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 score."""
    pred_tokens = _normalize_and_tokenize(prediction)
    ref_tokens = _normalize_and_tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """Normalized exact match."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def bleu1(prediction: str, reference: str) -> float:
    """BLEU-1 (unigram precision with brevity penalty)."""
    pred_tokens = _normalize_and_tokenize(prediction)
    ref_tokens = _normalize_and_tokenize(reference)

    if not pred_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)
    pred_counter = Counter(pred_tokens)

    clipped = sum(min(count, ref_counter.get(token, 0)) for token, count in pred_counter.items())
    precision = clipped / len(pred_tokens)

    # Brevity penalty
    bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0.0

    return bp * precision


def compute_all_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Compute all metrics at once."""
    return {
        "f1": f1_score(prediction, reference),
        "em": exact_match(prediction, reference),
        "bleu1": bleu1(prediction, reference),
    }


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_and_tokenize(text: str) -> list[str]:
    return _normalize(text).split()
