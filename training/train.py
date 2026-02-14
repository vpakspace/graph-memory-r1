"""Training entry point for Graph-Memory-R1.

Usage:
    # Structural reward (format only, no external deps):
    python training/train.py --data data/locomo/locomo10.json --epochs 1

    # Full reward (Neo4j + GPT-4o-mini Answer Agent):
    python training/train.py --data data/locomo/locomo10.json --epochs 2 --reward full

    # Resume from structural checkpoint with full reward:
    python training/train.py --data data/locomo/locomo10.json --epochs 2 --reward full \
        --resume checkpoints/
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset

from config import get_settings
from training.dataset import chunk_conversation, format_for_grpo, load_locomo
from training.grpo_trainer import GRPOMemoryTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Graph-Memory-R1")
    parser.add_argument("--data", required=True, help="Path to LoCoMo dataset JSON")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument(
        "--reward",
        choices=["structural", "full"],
        default="structural",
        help="Reward mode: structural (format only) or full (Neo4j + QA)",
    )
    parser.add_argument("--resume", default=None, help="Resume from checkpoint dir")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}")
    data = load_locomo(args.data)
    print(f"Loaded {data['num_conversations']} conversations, {data['num_qa_pairs']} QA pairs")

    # Create training prompts from conversation chunks
    prompts = []
    for conv in data["conversations"]:
        chunks = chunk_conversation(
            conv["turns"],
            max_tokens=4096,
            conversation_id=conv["id"],
        )
        for chunk in chunks:
            formatted = format_for_grpo(chunk)
            prompts.append(formatted)

    print(f"Created {len(prompts)} training prompts")

    if not prompts:
        print("No training data found. Exiting.")
        return

    # Create HuggingFace Dataset
    train_dataset = Dataset.from_list(prompts)

    # Collect QA pairs for full reward mode
    qa_pairs = []
    if args.reward == "full":
        for qa in data["qa_pairs"]:
            qa_pairs.append({
                "question": qa.question,
                "answer": qa.answer,
                "category": qa.category,
            })
        print(f"Full reward mode: {len(qa_pairs)} QA pairs for evaluation")

    # Setup and run trainer
    settings = get_settings()
    settings.training.epochs = args.epochs
    settings.training.output_dir = args.output

    trainer = GRPOMemoryTrainer(
        output_dir=args.output,
        reward_mode=args.reward,
        qa_pairs=qa_pairs,
        resume_from=args.resume,
    )
    print(f"Setting up GRPO trainer (reward={args.reward})...")
    trainer.setup(dataset=train_dataset)

    print("Starting training...")
    metrics = trainer.train(train_dataset)

    print("Training complete!")
    print(f"  Loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  LoRA saved: {metrics.get('lora_path', 'N/A')}")


if __name__ == "__main__":
    main()
