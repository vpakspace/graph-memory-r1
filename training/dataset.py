"""LoCoMo dataset loader for GRPO training.

LoCoMo: Long Context Multi-Turn Conversation dataset for memory evaluation.
Parses conversations, extracts QA pairs, formats for GRPO training.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    speaker: str
    text: str
    turn_id: int = 0


@dataclass
class QAPair:
    """A question-answer pair from LoCoMo."""
    question: str
    answer: str
    category: str = ""  # multi-hop, temporal, single-hop, open-domain, adversarial
    conversation_id: str = ""


@dataclass
class ConversationChunk:
    """A chunk of conversation for processing."""
    text: str
    conversation_id: str
    chunk_index: int = 0
    qa_pairs: list[QAPair] = field(default_factory=list)


def load_locomo(data_path: str) -> dict[str, Any]:
    """Load LoCoMo dataset from JSON file.

    Expected format (LoCoMo / MAMGA):
    [
        {
            "conversation_id": "...",
            "conversation": [
                {"speaker": "User1", "text": "..."},
                ...
            ],
            "questions": [
                {"question": "...", "answer": "...", "category": "..."},
                ...
            ]
        },
        ...
    ]
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    all_qa_pairs = []

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("data", data.get("conversations", [data]))
    else:
        raise ValueError(f"Unexpected dataset format: {type(data)}")

    for item in items:
        conv_id = str(item.get("conversation_id", item.get("sample_id", item.get("id", len(conversations)))))

        # Parse conversation turns — handle LoCoMo session-based format
        turns = []
        raw_conv = item.get("conversation", item.get("turns", []))
        if isinstance(raw_conv, dict):
            # LoCoMo format: dict with session_N keys containing turn lists
            turn_idx = 0
            session_num = 1
            while f"session_{session_num}" in raw_conv:
                session_turns = raw_conv[f"session_{session_num}"]
                for turn in session_turns:
                    if isinstance(turn, dict):
                        speaker = turn.get("speaker", f"Speaker{turn_idx % 2}")
                        text = turn.get("text", "")
                        turns.append(ConversationTurn(speaker=speaker, text=text, turn_id=turn_idx))
                        turn_idx += 1
                session_num += 1
        elif isinstance(raw_conv, list):
            for i, turn in enumerate(raw_conv):
                if isinstance(turn, dict):
                    speaker = turn.get("speaker", turn.get("role", f"Speaker{i % 2}"))
                    text = turn.get("text", turn.get("content", ""))
                elif isinstance(turn, str):
                    speaker = f"Speaker{i % 2}"
                    text = turn
                else:
                    continue
                turns.append(ConversationTurn(speaker=speaker, text=text, turn_id=i))

        # Parse QA pairs — handle both 'qa' and 'questions' keys
        qa_pairs = []
        raw_qa = item.get("qa", item.get("questions", item.get("qa_pairs", [])))
        for qa in raw_qa:
            if isinstance(qa, dict):
                category = qa.get("category", qa.get("type", ""))
                qa_pairs.append(QAPair(
                    question=qa.get("question", qa.get("q", "")),
                    answer=qa.get("answer", qa.get("a", "")),
                    category=str(category),
                    conversation_id=conv_id,
                ))

        conversations.append({
            "id": conv_id,
            "turns": turns,
            "qa_pairs": qa_pairs,
        })
        all_qa_pairs.extend(qa_pairs)

    return {
        "conversations": conversations,
        "qa_pairs": all_qa_pairs,
        "num_conversations": len(conversations),
        "num_qa_pairs": len(all_qa_pairs),
    }


def chunk_conversation(
    turns: list[ConversationTurn],
    max_tokens: int = 4096,
    conversation_id: str = "",
) -> list[ConversationChunk]:
    """Split conversation turns into chunks of ~max_tokens each.

    Rough token estimate: 1 token ~ 4 chars.
    """
    max_chars = max_tokens * 4
    chunks = []
    current_text = ""
    chunk_idx = 0

    for turn in turns:
        line = f"{turn.speaker}: {turn.text}\n"
        if len(current_text) + len(line) > max_chars and current_text:
            chunks.append(ConversationChunk(
                text=current_text.strip(),
                conversation_id=conversation_id,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
            current_text = ""
        current_text += line

    if current_text.strip():
        chunks.append(ConversationChunk(
            text=current_text.strip(),
            conversation_id=conversation_id,
            chunk_index=chunk_idx,
        ))

    return chunks


def format_for_grpo(
    chunk: ConversationChunk,
    graph_state: str = "",
) -> dict[str, str]:
    """Format a conversation chunk + graph state for GRPO training prompt.

    Returns dict with 'prompt' key for the Memory Manager.
    """
    prompt = (
        "You are a Memory Manager agent. Your job is to manage a graph-based memory "
        "by deciding which operations to perform after reading a conversation chunk.\n\n"
        "Available operations:\n"
        "- graph_memory_add(memory_type, content): Add a new memory node\n"
        "- graph_memory_update(memory_type, node_id, new_content): Update existing node\n"
        "- graph_memory_delete(memory_type, node_id): Delete a node\n"
        "- graph_memory_noop(): No operation needed\n\n"
        "Memory types: core (user profile), semantic (facts), episodic (events)\n\n"
    )

    if graph_state:
        prompt += f"Current graph memory state:\n{graph_state}\n\n"

    prompt += f"Conversation chunk:\n{chunk.text}\n\n"
    prompt += (
        "Decide which memory operations to perform. Output tool calls in this format:\n"
        '<tool_call>\n{"name": "graph_memory_add", "arguments": {"memory_type": "semantic", "content": "..."}}\n</tool_call>\n'
    )

    return {"prompt": prompt}


def create_splits(
    qa_pairs: list[QAPair],
    train_ratio: float = 0.1,
    val_ratio: float = 0.05,
) -> dict[str, list[QAPair]]:
    """Split QA pairs into train/val/test sets."""
    n = len(qa_pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": qa_pairs[:train_end],
        "val": qa_pairs[train_end:val_end],
        "test": qa_pairs[val_end:],
    }
