#!/usr/bin/env python3
"""Full LoCoMo benchmark runner for Graph-Memory-R1.

Runs the complete pipeline: load conversations → build graph memory → answer QA pairs.
Supports multiple configurations for comparison:
  - no_memory:     Answer Agent without any graph memory (baseline)
  - graph_base:    Memory Manager (base Qwen, no LoRA) → graph → Answer Agent
  - graph_grpo:    Memory Manager (Qwen + LoRA) → graph → Answer Agent

Usage:
    python scripts/run_benchmark.py                          # all configs
    python scripts/run_benchmark.py --config graph_grpo      # single config
    python scripts/run_benchmark.py --max-qa 50 --max-conv 3 # limit scope
    python scripts/run_benchmark.py --use-judge              # + LLM judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from config import get_settings
from evaluation.benchmark import run_benchmark, compare_benchmarks
from evaluation.llm_judge import judge_answer
from training.dataset import load_locomo, chunk_conversation


def build_graph_memory():
    """Create a fresh GraphMemory instance with embedder."""
    from memory.graph_memory import GraphMemory
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai.api_key)

    def embedder(text: str) -> list[float]:
        resp = client.embeddings.create(
            model=settings.openai.embedding_model,
            input=text,
            dimensions=settings.openai.embedding_dimensions,
        )
        return resp.data[0].embedding

    graph = GraphMemory(embedder=embedder)
    graph.init_schema()
    return graph


def process_conversations(
    conversations: list[dict],
    memory_manager,
    max_conversations: int | None = None,
) -> int:
    """Process conversations through Memory Manager to build graph.

    Returns number of operations executed.
    """
    items = conversations[:max_conversations] if max_conversations else conversations
    total_ops = 0

    for i, conv in enumerate(items):
        turns = conv["turns"]
        conv_id = conv["id"]
        chunks = chunk_conversation(turns, conversation_id=conv_id)

        print(f"  Conversation {i + 1}/{len(items)} (id={conv_id}): {len(chunks)} chunks")

        for chunk in chunks:
            result = memory_manager.process_chunk(chunk.text)
            total_ops += result.total
            ok = result.successful
            fail = result.failed
            if fail > 0:
                print(f"    Chunk {chunk.chunk_index}: {ok} ok, {fail} failed")

    return total_ops


def run_config_no_memory(
    qa_pairs,
    use_judge: bool = False,
    max_qa: int | None = None,
) -> dict[str, Any]:
    """Baseline: answer questions without any graph memory."""
    print("\n--- Config: no_memory (baseline) ---")

    from agents.answer_agent import AnswerAgent

    graph = build_graph_memory()
    graph.clear()  # ensure empty
    agent = AnswerAgent(graph=graph)

    answer_fn = lambda q: agent.answer(q)["answer"]
    judge_fn = judge_answer if use_judge else None

    return run_benchmark(qa_pairs, answer_fn, judge_fn, max_items=max_qa)


def run_config_graph(
    conversations,
    qa_pairs,
    lora_path: str | None = None,
    config_name: str = "graph",
    use_judge: bool = False,
    max_conversations: int | None = None,
    max_qa: int | None = None,
) -> dict[str, Any]:
    """Run with graph memory built by Memory Manager."""
    lora_label = f" + LoRA ({lora_path})" if lora_path else " (base Qwen)"
    print(f"\n--- Config: {config_name}{lora_label} ---")

    from agents.memory_manager import MemoryManager
    from agents.answer_agent import AnswerAgent

    # Fresh graph
    graph = build_graph_memory()
    graph.clear()

    # Memory Manager
    mm = MemoryManager(graph=graph)
    print("  Loading Qwen model...")
    mm.load_model()

    if lora_path:
        print(f"  Loading LoRA from {lora_path}...")
        mm.load_lora(lora_path)

    # Process conversations → build graph
    print("  Processing conversations...")
    t0 = time.time()
    total_ops = process_conversations(conversations, mm, max_conversations)
    build_time = time.time() - t0

    stats = graph.get_stats()
    print(f"  Graph built: {stats['total_nodes']} nodes, {stats['total_edges']} edges "
          f"({total_ops} ops in {build_time:.1f}s)")

    # Answer questions
    agent = AnswerAgent(graph=graph)
    answer_fn = lambda q: agent.answer(q)["answer"]
    judge_fn = judge_answer if use_judge else None

    print(f"  Answering {max_qa or len(qa_pairs)} questions...")
    t0 = time.time()
    result = run_benchmark(qa_pairs, answer_fn, judge_fn, max_items=max_qa)
    answer_time = time.time() - t0

    result["build_time"] = build_time
    result["answer_time"] = answer_time
    result["graph_stats"] = stats
    result["total_ops"] = total_ops

    return result


def print_results(name: str, result: dict[str, Any]) -> None:
    """Print benchmark results for a single config."""
    agg = result.get("aggregate", {})
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total QA:     {agg.get('total', 0)}")
    print(f"  Avg F1:       {agg.get('avg_f1', 0):.4f}")
    print(f"  Avg EM:       {agg.get('avg_em', 0):.4f}")
    print(f"  Avg BLEU-1:   {agg.get('avg_bleu1', 0):.4f}")
    if "avg_judge_score" in agg:
        print(f"  Avg Judge:    {agg['avg_judge_score']:.1f}/100")
    if "build_time" in result:
        print(f"  Build time:   {result['build_time']:.1f}s")
    if "answer_time" in result:
        print(f"  Answer time:  {result['answer_time']:.1f}s")
    if "graph_stats" in result:
        gs = result["graph_stats"]
        print(f"  Graph:        {gs.get('total_nodes', 0)} nodes, {gs.get('total_edges', 0)} edges")

    # Per-category breakdown
    cats = agg.get("categories", {})
    if cats:
        print(f"\n  Per-category breakdown:")
        for cat, cm in sorted(cats.items()):
            print(f"    {cat:20s}  n={cm['count']:3d}  F1={cm['avg_f1']:.4f}  EM={cm['avg_em']:.4f}")


def print_comparison(all_results: dict[str, dict]) -> None:
    """Print comparison table across configs."""
    comp = compare_benchmarks(all_results)
    print(f"\n{'='*80}")
    print("  COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"  {'Config':<20s} {'Total':>6s} {'F1':>8s} {'EM':>8s} {'BLEU1':>8s} {'Judge':>8s}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for row in comp["comparison"]:
        judge = f"{row['avg_judge']:.1f}" if isinstance(row["avg_judge"], (int, float)) else row["avg_judge"]
        print(f"  {row['config']:<20s} {row['total']:>6d} "
              f"{row['avg_f1']:>8.4f} {row['avg_em']:>8.4f} "
              f"{row['avg_bleu1']:>8.4f} {judge:>8s}")


def main():
    parser = argparse.ArgumentParser(description="Graph-Memory-R1 LoCoMo Benchmark")
    parser.add_argument("--dataset", default="data/locomo/locomo10.json", help="LoCoMo dataset path")
    parser.add_argument("--config", choices=["no_memory", "graph_base", "graph_grpo", "all"],
                        default="all", help="Which config to run")
    parser.add_argument("--lora-path", default="checkpoints/final_lora", help="Path to LoRA checkpoint")
    parser.add_argument("--max-qa", type=int, default=None, help="Max QA pairs to evaluate")
    parser.add_argument("--max-conv", type=int, default=None, help="Max conversations to process")
    parser.add_argument("--use-judge", action="store_true", help="Use LLM judge (GPT-4o-mini)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_locomo(args.dataset)
    print(f"  {data['num_conversations']} conversations, {data['num_qa_pairs']} QA pairs")

    conversations = data["conversations"]
    qa_pairs = data["qa_pairs"]

    all_results = {}

    # Run selected configs
    configs_to_run = (
        ["no_memory", "graph_base", "graph_grpo"]
        if args.config == "all"
        else [args.config]
    )

    for config in configs_to_run:
        if config == "no_memory":
            result = run_config_no_memory(qa_pairs, args.use_judge, args.max_qa)
            all_results["no_memory"] = result
            print_results("no_memory (baseline)", result)

        elif config == "graph_base":
            result = run_config_graph(
                conversations, qa_pairs,
                lora_path=None,
                config_name="graph_base",
                use_judge=args.use_judge,
                max_conversations=args.max_conv,
                max_qa=args.max_qa,
            )
            all_results["graph_base"] = result
            print_results("graph_base (Qwen без LoRA)", result)

        elif config == "graph_grpo":
            lora = args.lora_path
            if not os.path.exists(lora):
                print(f"\n  WARNING: LoRA path not found: {lora}")
                print("  Skipping graph_grpo config.")
                continue

            result = run_config_graph(
                conversations, qa_pairs,
                lora_path=lora,
                config_name="graph_grpo",
                use_judge=args.use_judge,
                max_conversations=args.max_conv,
                max_qa=args.max_qa,
            )
            all_results["graph_grpo"] = result
            print_results("graph_grpo (Qwen + LoRA)", result)

    # Comparison
    if len(all_results) > 1:
        print_comparison(all_results)

    # Save results
    if args.output:
        # Strip non-serializable data
        for name, res in all_results.items():
            for r in res.get("results", []):
                r.pop("sources", None)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
