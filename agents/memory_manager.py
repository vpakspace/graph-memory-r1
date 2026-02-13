"""Memory Manager agent: Qwen-2.5-3B with LoRA for graph memory operations.

The Memory Manager reads conversation chunks and decides which graph operations
to perform (ADD/UPDATE/DELETE/NOOP). It is trained via GRPO to optimize
memory management for downstream QA performance.
"""

from __future__ import annotations

import json
import re
from typing import Any

from config import get_settings
from memory.graph_memory import GraphMemory
from memory.operations import MemoryOperation, OperationResult, execute_operations, parse_tool_calls


class MemoryManager:
    """Memory Manager agent that manages graph memory."""

    def __init__(
        self,
        graph: GraphMemory,
        model=None,
        tokenizer=None,
    ):
        self._graph = graph
        self._model = model
        self._tokenizer = tokenizer
        self._settings = get_settings()

    def load_model(self) -> None:
        """Load Qwen model and tokenizer. Call once at startup."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self._settings.qwen.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def load_lora(self, lora_path: str) -> None:
        """Load LoRA adapters on top of base model."""
        if self._model is None:
            self.load_model()

        from peft import PeftModel
        self._model = PeftModel.from_pretrained(self._model, lora_path)

    def process_chunk(self, conversation_chunk: str) -> OperationResult:
        """Process a conversation chunk: generate and execute memory operations."""
        # Get current graph state
        graph_state = self._graph.render_memory(
            max_tokens=self._settings.graph_memory_max_tokens,
        )

        # Generate operations
        prompt = self._build_prompt(conversation_chunk, graph_state)
        response = self._generate(prompt)

        # Parse and execute
        operations = parse_tool_calls(response)
        if not operations:
            operations = [MemoryOperation(action="noop")]

        return execute_operations(self._graph, operations)

    def generate_operations(self, conversation_chunk: str) -> tuple[str, list[MemoryOperation]]:
        """Generate operations without executing them. Used for GRPO training.

        Returns (raw_response, parsed_operations).
        """
        graph_state = self._graph.render_memory(
            max_tokens=self._settings.graph_memory_max_tokens,
        )
        prompt = self._build_prompt(conversation_chunk, graph_state)
        response = self._generate(prompt)
        operations = parse_tool_calls(response)
        return response, operations

    def _build_prompt(self, chunk: str, graph_state: str) -> str:
        """Build the Memory Manager prompt."""
        system = (
            "You are a Memory Manager agent. Your job is to manage a graph-based memory "
            "by deciding which operations to perform after reading a conversation chunk.\n\n"
            "Available operations:\n"
            "- graph_memory_add(memory_type, content): Add a new memory node\n"
            "  memory_type: 'core' (user profile), 'semantic' (facts), 'episodic' (events)\n"
            "- graph_memory_update(memory_type, node_id, new_content): Update existing node\n"
            "- graph_memory_delete(memory_type, node_id): Delete outdated node\n"
            "- graph_memory_noop(): No operation needed\n\n"
            "Rules:\n"
            "1. Store important facts as semantic nodes\n"
            "2. Store events/experiences as episodic nodes\n"
            "3. Update core memory for user preferences/profile changes\n"
            "4. Delete outdated or contradicted information\n"
            "5. Use noop if nothing important needs storing\n"
            "6. Be selective — don't store trivial information\n"
        )

        user = ""
        if graph_state.strip():
            user += f"Current graph memory:\n{graph_state}\n\n"
        user += f"Conversation chunk:\n{chunk}\n\n"
        user += "Output your memory operations:"

        return json.dumps([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    def _generate(self, prompt_json: str) -> str:
        """Generate text using the loaded model."""
        if self._model is None or self._tokenizer is None:
            # Fallback: return noop if model not loaded
            return '<tool_call>\n{"name": "graph_memory_noop", "arguments": {}}\n</tool_call>'

        messages = json.loads(prompt_json)
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._settings.qwen.max_new_tokens,
            temperature=self._settings.qwen.temperature,
            do_sample=True,
        )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response
