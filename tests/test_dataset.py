"""Tests for LoCoMo dataset loader."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from training.dataset import (
    ConversationChunk,
    ConversationTurn,
    QAPair,
    chunk_conversation,
    create_splits,
    format_for_grpo,
    load_locomo,
)


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal LoCoMo-style dataset file."""
    data = [
        {
            "conversation_id": "conv1",
            "conversation": [
                {"speaker": "Alice", "text": "I live in Paris."},
                {"speaker": "Bob", "text": "That's cool! I'm from London."},
                {"speaker": "Alice", "text": "I work as a software engineer."},
            ],
            "questions": [
                {"question": "Where does Alice live?", "answer": "Paris", "category": "single-hop"},
                {"question": "What is Alice's job?", "answer": "software engineer", "category": "single-hop"},
            ],
        },
        {
            "conversation_id": "conv2",
            "conversation": [
                {"speaker": "Charlie", "text": "I had pizza for lunch yesterday."},
            ],
            "questions": [
                {"question": "What did Charlie eat?", "answer": "pizza", "category": "temporal"},
            ],
        },
    ]
    filepath = tmp_path / "test_locomo.json"
    with open(filepath, "w") as f:
        json.dump(data, f)
    return str(filepath)


class TestLoadLocomo:
    def test_load(self, sample_dataset):
        data = load_locomo(sample_dataset)
        assert data["num_conversations"] == 2
        assert data["num_qa_pairs"] == 3

    def test_conversations(self, sample_dataset):
        data = load_locomo(sample_dataset)
        conv = data["conversations"][0]
        assert conv["id"] == "conv1"
        assert len(conv["turns"]) == 3
        assert conv["turns"][0].speaker == "Alice"

    def test_qa_pairs(self, sample_dataset):
        data = load_locomo(sample_dataset)
        qa = data["qa_pairs"][0]
        assert qa.question == "Where does Alice live?"
        assert qa.answer == "Paris"
        assert qa.category == "single-hop"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_locomo("/nonexistent/path.json")


class TestChunkConversation:
    def test_single_chunk(self):
        turns = [
            ConversationTurn(speaker="A", text="Hello", turn_id=0),
            ConversationTurn(speaker="B", text="Hi", turn_id=1),
        ]
        chunks = chunk_conversation(turns, max_tokens=4096)
        assert len(chunks) == 1
        assert "A: Hello" in chunks[0].text

    def test_multiple_chunks(self):
        # Create a long conversation
        turns = [
            ConversationTurn(speaker="A", text="x" * 1000, turn_id=i)
            for i in range(20)
        ]
        chunks = chunk_conversation(turns, max_tokens=1024)
        assert len(chunks) > 1

    def test_empty(self):
        chunks = chunk_conversation([])
        assert len(chunks) == 0

    def test_chunk_index(self):
        turns = [
            ConversationTurn(speaker="A", text="x" * 5000, turn_id=i)
            for i in range(5)
        ]
        chunks = chunk_conversation(turns, max_tokens=1024)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestFormatForGrpo:
    def test_format(self):
        chunk = ConversationChunk(text="User: Hello", conversation_id="1")
        result = format_for_grpo(chunk)
        assert "prompt" in result
        assert "Memory Manager" in result["prompt"]
        assert "User: Hello" in result["prompt"]

    def test_with_graph_state(self):
        chunk = ConversationChunk(text="Test", conversation_id="1")
        result = format_for_grpo(chunk, graph_state="## Core\nProfile data")
        assert "Profile data" in result["prompt"]


class TestCreateSplits:
    def test_splits(self):
        qa = [QAPair(question=f"Q{i}", answer=f"A{i}") for i in range(100)]
        splits = create_splits(qa, train_ratio=0.1, val_ratio=0.05)
        assert len(splits["train"]) == 10
        assert len(splits["val"]) == 5
        assert len(splits["test"]) == 85

    def test_empty(self):
        splits = create_splits([])
        assert len(splits["train"]) == 0
