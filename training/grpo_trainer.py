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

import logging
import os
import random
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)


class GRPOMemoryTrainer:
    """GRPO trainer for the Memory Manager agent."""

    def __init__(
        self,
        output_dir: str | None = None,
        reward_mode: str = "structural",
        qa_pairs: list[dict] | None = None,
        resume_from: str | None = None,
    ):
        """
        Args:
            output_dir: Directory for checkpoints.
            reward_mode: "structural" (format only) or "full" (execute + QA).
            qa_pairs: QA pairs from LoCoMo for full reward mode.
            resume_from: Path to LoRA checkpoint to resume from.
        """
        self._settings = get_settings()
        self._output_dir = output_dir or self._settings.training.output_dir
        self._reward_mode = reward_mode
        self._qa_pairs = qa_pairs or []
        self._resume_from = resume_from
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._graph = None
        self._answer_agent = None

    def setup(self, dataset=None) -> None:
        """Load model, tokenizer, and configure GRPO trainer."""
        from peft import LoraConfig, PeftModel
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
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        # Resume from LoRA checkpoint or create new LoRA
        lora_config = None
        if self._resume_from:
            lora_path = self._resume_from
            if os.path.isdir(lora_path) and not os.path.exists(
                os.path.join(lora_path, "adapter_config.json")
            ):
                # Try final_lora subdirectory
                candidate = os.path.join(lora_path, "final_lora")
                if os.path.exists(os.path.join(candidate, "adapter_config.json")):
                    lora_path = candidate
            logger.info("Resuming from LoRA checkpoint: %s", lora_path)
            self._model = PeftModel.from_pretrained(
                self._model, lora_path, is_trainable=True,
            )
        else:
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
            beta=settings.training.kl_coef,
            max_completion_length=settings.training.max_completion_length,
            num_train_epochs=settings.training.epochs,
            logging_steps=10,
            save_steps=100,
            gradient_checkpointing=settings.training.gradient_checkpointing,
        )

        self._trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=[self._reward_fn],
            args=grpo_config,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=self._tokenizer,
        )

    def _init_full_reward(self) -> None:
        """Initialize Neo4j graph + Answer Agent for full reward mode."""
        if self._graph is not None:
            return
        from agents.answer_agent import AnswerAgent
        from memory.graph_memory import GraphMemory

        self._graph = GraphMemory()
        self._graph.init_schema()
        self._answer_agent = AnswerAgent(self._graph)
        logger.info("Full reward mode: Neo4j + Answer Agent initialized")

    def _reward_fn(self, prompts, completions, **kwargs) -> list[float]:
        """Reward function for GRPO.

        Two modes:
        - "structural": checks tool call format only (fast, no external deps)
        - "full": execute ops on Neo4j → Answer Agent → F1 vs gold answer

        Args:
            prompts: List of prompt strings.
            completions: List of completion strings.
        """
        if self._reward_mode == "full":
            return self._full_reward(prompts, completions)
        return self._structural_reward(completions)

    def _structural_reward(self, completions) -> list[float]:
        """Structural reward: only checks tool call format validity."""
        from memory.operations import parse_tool_calls, validate_operations
        from training.reward import content_placement_score

        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else str(completion)
            ops = parse_tool_calls(text)

            tool_score = validate_operations(ops) if ops else 0.0
            placement_score = content_placement_score(ops)
            has_tool_calls = 1.0 if ops else 0.0

            reward = 0.5 * has_tool_calls + 0.3 * tool_score + 0.2 * placement_score
            rewards.append(reward)

        return rewards

    def _full_reward(self, prompts, completions) -> list[float]:
        """Full reward: execute ops → answer questions → F1 vs gold.

        For each completion:
        1. Parse tool calls
        2. Execute on a clean graph
        3. Pick a random QA pair from LoCoMo
        4. Answer Agent answers using graph memory
        5. Compute composite reward (r_qa + r_tool + r_compress + r_valid)
        """
        from memory.operations import execute_operations, parse_tool_calls
        from training.reward import compute_reward

        self._init_full_reward()

        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else str(completion)
            ops = parse_tool_calls(text)

            if not ops:
                rewards.append(0.0)
                continue

            try:
                # Clean graph for this rollout
                self._graph.clear()

                # Execute operations
                execute_operations(self._graph, ops)

                # Pick random QA pair
                if self._qa_pairs:
                    qa = random.choice(self._qa_pairs)
                    question = qa.get("question", qa.get("q", ""))
                    gold_answer = qa.get("answer", qa.get("a", ""))
                else:
                    question, gold_answer = "", ""

                # Answer Agent answers using graph
                if question and self._answer_agent:
                    result = self._answer_agent.answer_with_memory(question)
                    predicted_answer = result.get("answer", "")
                else:
                    predicted_answer = ""

                # Composite reward
                graph_stats = self._graph.get_stats()
                reward = compute_reward(
                    predicted_answer=predicted_answer,
                    gold_answer=gold_answer,
                    operations=ops,
                    graph_stats=graph_stats,
                )
                rewards.append(reward)

            except Exception as e:
                logger.warning("Full reward error: %s", e)
                rewards.append(0.0)

        return rewards

    def train(self, dataset) -> dict[str, Any]:
        """Run GRPO training.

        Args:
            dataset: HuggingFace Dataset with 'prompt' column.

        Returns:
            Training metrics dict.
        """
        if self._trainer is None:
            self.setup(dataset=dataset)

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
