"""GRPO Trainer for Memory Manager agent.

Uses trl GRPOTrainer with LoRA fine-tuning on Qwen-2.5-3B.
The training loop:
1. Memory Manager generates graph operations (group of N rollouts)
2. Execute operations on graph (per rollout)
3. Answer Agent answers questions using graph
4. Compute reward (F1 vs gold answer + tool validity + compression)
5. GRPO update (group-relative advantage)
"""

from __future__ import annotations

import os
from typing import Any

from config import get_settings


class GRPOMemoryTrainer:
    """GRPO trainer for the Memory Manager agent."""

    def __init__(self, output_dir: str | None = None):
        self._settings = get_settings()
        self._output_dir = output_dir or self._settings.training.output_dir
        self._model = None
        self._tokenizer = None
        self._trainer = None

    def setup(self) -> None:
        """Load model, tokenizer, and configure GRPO trainer."""
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        settings = self._settings
        model_name = settings.qwen.model_name

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        # LoRA config
        lora_config = LoraConfig(
            r=settings.qwen.lora_rank,
            lora_alpha=settings.qwen.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        # GRPO config
        grpo_config = GRPOConfig(
            output_dir=self._output_dir,
            num_generations=settings.training.grpo_group_size,
            per_device_train_batch_size=settings.training.batch_size,
            gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
            learning_rate=settings.training.learning_rate,
            kl_coef=settings.training.kl_coef,
            max_prompt_length=settings.training.max_prompt_length,
            max_completion_length=settings.training.max_completion_length,
            num_train_epochs=settings.training.epochs,
            logging_steps=10,
            save_steps=100,
            gradient_checkpointing=settings.training.gradient_checkpointing,
        )

        self._trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=[self._reward_fn],
            config=grpo_config,
            peft_config=lora_config,
            processing_class=self._tokenizer,
        )

    def _reward_fn(self, completions: list[str], **kwargs) -> list[float]:
        """Reward function for GRPO.

        Called by trl GRPOTrainer for each group of completions.
        In production, this orchestrates: parse ops → execute → answer → score.
        For initial training, we use a simplified structural reward.
        """
        from memory.operations import parse_tool_calls, validate_operations
        from training.reward import content_placement_score

        rewards = []
        for completion in completions:
            ops = parse_tool_calls(completion)

            # Tool validity
            tool_score = validate_operations(ops) if ops else 0.0

            # Content placement
            placement_score = content_placement_score(ops)

            # Structural reward (valid JSON tool calls)
            has_tool_calls = 1.0 if ops else 0.0

            reward = 0.5 * has_tool_calls + 0.3 * tool_score + 0.2 * placement_score
            rewards.append(reward)

        return rewards

    def train(self, dataset) -> dict[str, Any]:
        """Run GRPO training.

        Args:
            dataset: HuggingFace Dataset with 'prompt' column.

        Returns:
            Training metrics dict.
        """
        if self._trainer is None:
            self.setup()

        result = self._trainer.train()

        # Save LoRA adapters
        save_path = os.path.join(self._output_dir, "final_lora")
        self._trainer.save_model(save_path)

        return {
            "train_loss": result.training_loss if hasattr(result, "training_loss") else None,
            "lora_path": save_path,
        }

    def save(self, path: str | None = None) -> str:
        """Save model and LoRA adapters."""
        save_path = path or os.path.join(self._output_dir, "final_lora")
        if self._trainer:
            self._trainer.save_model(save_path)
        return save_path
