"""LoCoMo benchmark runner.

Evaluates memory-augmented QA performance on LoCoMo test set.
Supports multiple configurations: no memory, flat memory, graph memory, graph+GRPO.
"""

from __future__ import annotations

from typing import Any, Callable

from evaluation.metrics import compute_all_metrics
from training.dataset import QAPair


def run_benchmark(
    qa_pairs: list[QAPair],
    answer_fn: Callable[[str], str],
    judge_fn: Callable[[str, str, str], dict] | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Run benchmark on a list of QA pairs.

    Args:
        qa_pairs: List of QA pairs to evaluate.
        answer_fn: Function that takes a question and returns an answer string.
        judge_fn: Optional LLM judge function(question, gold, predicted) -> {score, reasoning}.
        max_items: Maximum number of items to evaluate.

    Returns:
        Benchmark results dict.
    """
    items = qa_pairs[:max_items] if max_items else qa_pairs
    results = []

    for i, qa in enumerate(items):
        predicted = answer_fn(qa.question)
        metrics = compute_all_metrics(predicted, qa.answer)

        result = {
            "index": i,
            "question": qa.question,
            "gold_answer": qa.answer,
            "predicted_answer": predicted,
            "category": qa.category,
            **metrics,
        }

        if judge_fn:
            judge_result = judge_fn(qa.question, qa.answer, predicted)
            result["judge_score"] = judge_result.get("score", 0)
            result["judge_reasoning"] = judge_result.get("reasoning", "")

        results.append(result)

    # Aggregate metrics
    n = len(results)
    if n == 0:
        return {"results": [], "aggregate": {}}

    avg_f1 = sum(r["f1"] for r in results) / n
    avg_em = sum(r["em"] for r in results) / n
    avg_bleu1 = sum(r["bleu1"] for r in results) / n

    aggregate: dict[str, Any] = {
        "total": n,
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "avg_bleu1": avg_bleu1,
    }

    if any("judge_score" in r for r in results):
        scores = [r["judge_score"] for r in results if "judge_score" in r]
        aggregate["avg_judge_score"] = sum(scores) / len(scores) if scores else 0

    # Per-category breakdown
    categories: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    category_metrics = {}
    for cat, cat_results in categories.items():
        cn = len(cat_results)
        category_metrics[cat] = {
            "count": cn,
            "avg_f1": sum(r["f1"] for r in cat_results) / cn,
            "avg_em": sum(r["em"] for r in cat_results) / cn,
        }

    aggregate["categories"] = category_metrics

    return {
        "results": results,
        "aggregate": aggregate,
    }


def compare_benchmarks(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare multiple benchmark results.

    Args:
        results: Dict of {config_name: benchmark_result}.

    Returns:
        Comparison table.
    """
    table = []
    for name, result in results.items():
        agg = result.get("aggregate", {})
        table.append({
            "config": name,
            "total": agg.get("total", 0),
            "avg_f1": agg.get("avg_f1", 0),
            "avg_em": agg.get("avg_em", 0),
            "avg_bleu1": agg.get("avg_bleu1", 0),
            "avg_judge": agg.get("avg_judge_score", "N/A"),
        })

    return {"comparison": table}
