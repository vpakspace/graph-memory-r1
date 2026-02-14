"""Quick inference demo: load LoRA adapter and generate memory operations.

Usage:
    python scripts/inference_demo.py
    python scripts/inference_demo.py --custom "Alice: I got a new job at Google!"
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a Memory Manager agent. Your job is to manage a graph-based memory "
    "by deciding which operations to perform after reading a conversation chunk.\n\n"
    "Available operations:\n"
    "- graph_memory_add(memory_type, content): Add a new memory node\n"
    "- graph_memory_update(memory_type, node_id, new_content): Update existing node\n"
    "- graph_memory_delete(memory_type, node_id): Delete a node\n"
    "- graph_memory_noop(): No operation needed\n\n"
    "Memory types: core (user profile), semantic (facts), episodic (events)\n\n"
    "Decide which memory operations to perform. Output tool calls in this format:\n"
    '<tool_call>\n{"name": "graph_memory_add", "arguments": {"memory_type": "semantic", "content": "..."}}\n</tool_call>\n'
)

TEST_CONVERSATIONS = [
    # 1. Simple fact
    "Alice: I just moved to San Francisco last week.\nBob: That's great! How do you like it?",
    # 2. Multi-fact
    "Alice: I got promoted to Senior Engineer at Google!\nBob: Congrats! That's amazing.\nAlice: Thanks! I've been there for 3 years now.",
    # 3. Event + preference
    "Alice: We're planning a trip to Japan next month.\nBob: Nice! Do you like sushi?\nAlice: I love sushi, it's my favorite food!",
    # 4. Update scenario
    "Alice: Actually, I changed my mind about the Japan trip. We're going to Korea instead.\nBob: Korea is awesome too!",
]


def load_model(lora_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load base model + LoRA adapter."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, conversation: str, max_new_tokens: int = 512) -> str:
    """Generate memory operations for a conversation chunk."""
    prompt = SYSTEM_PROMPT + f"Conversation chunk:\n{conversation}\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inference demo for Graph-Memory-R1")
    parser.add_argument("--lora", default="checkpoints/final_lora", help="Path to LoRA adapter")
    parser.add_argument("--custom", default=None, help="Custom conversation to test")
    parser.add_argument("--base-only", action="store_true", help="Run without LoRA (baseline)")
    args = parser.parse_args()

    lora_path = args.lora
    base_model = "Qwen/Qwen2.5-3B-Instruct"

    if args.base_only:
        print("=" * 60)
        print("BASELINE MODE (no LoRA)")
        print("=" * 60)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )
        model.eval()
    else:
        print("=" * 60)
        print("LORA MODE (structural training)")
        print("=" * 60)
        model, tokenizer = load_model(lora_path, base_model)

    conversations = TEST_CONVERSATIONS
    if args.custom:
        conversations = [args.custom]

    for i, conv in enumerate(conversations, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {conv[:80]}...")
        print(f"{'─' * 60}")

        output = generate(model, tokenizer, conv)
        print(f"\nGenerated:\n{output}")
        print()


if __name__ == "__main__":
    main()
