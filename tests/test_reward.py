"""Tests for reward functions."""

from __future__ import annotations

import pytest

from training.reward import (
    compression_reward,
    content_placement_score,
    exact_match,
    f1_score,
    compute_reward,
)
from memory.operations import MemoryOperation


class TestF1Score:
    def test_exact_match(self):
        assert f1_score("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert f1_score("hello", "world") == 0.0

    def test_partial_overlap(self):
        score = f1_score("hello world foo", "hello world bar")
        assert 0.5 < score < 1.0

    def test_empty_both(self):
        assert f1_score("", "") == 1.0

    def test_empty_prediction(self):
        assert f1_score("", "hello") == 0.0

    def test_case_insensitive(self):
        assert f1_score("Hello World", "hello world") == 1.0

    def test_punctuation_ignored(self):
        assert f1_score("hello, world!", "hello world") == 1.0


class TestExactMatch:
    def test_match(self):
        assert exact_match("hello world", "hello world") == 1.0

    def test_no_match(self):
        assert exact_match("hello", "world") == 0.0

    def test_normalized(self):
        assert exact_match("Hello, World!", "hello world") == 1.0


class TestCompressionReward:
    def test_empty_graph(self):
        assert compression_reward({"total_nodes": 0}) == 1.0

    def test_full_graph(self):
        # 40 nodes * 50 tokens = 2000 tokens ~ max
        reward = compression_reward({"total_nodes": 40}, max_tokens=2048)
        assert reward >= 0.0
        assert reward < 0.5

    def test_over_max(self):
        reward = compression_reward({"total_nodes": 100}, max_tokens=100)
        assert reward == 0.0


class TestContentPlacement:
    def test_correct_semantic(self):
        ops = [MemoryOperation(
            action="add", memory_type="semantic",
            content="Python is a programming language",
        )]
        assert content_placement_score(ops) == 1.0

    def test_correct_episodic(self):
        ops = [MemoryOperation(
            action="add", memory_type="episodic",
            content="I visited Paris yesterday",
        )]
        assert content_placement_score(ops) == 1.0

    def test_correct_core(self):
        ops = [MemoryOperation(
            action="add", memory_type="core",
            content="My name is John and I prefer Python",
        )]
        assert content_placement_score(ops) == 1.0

    def test_empty(self):
        assert content_placement_score([]) == 0.0

    def test_noop_ignored(self):
        ops = [MemoryOperation(action="noop")]
        # noop has no add/update, so total=0 → default 1.0
        assert content_placement_score(ops) == 1.0


class TestComputeReward:
    def test_perfect_answer(self):
        reward = compute_reward(
            predicted_answer="42",
            gold_answer="42",
            operations=[MemoryOperation(
                action="add", memory_type="semantic",
                content="The answer is 42",
            )],
            graph_stats={"total_nodes": 5},
        )
        assert reward > 0.5

    def test_wrong_answer(self):
        reward = compute_reward(
            predicted_answer="completely wrong",
            gold_answer="42",
            operations=[MemoryOperation(action="noop")],
            graph_stats={"total_nodes": 5},
        )
        assert reward < 0.5

    def test_no_operations(self):
        reward = compute_reward(
            predicted_answer="test",
            gold_answer="test",
            operations=[],
            graph_stats={"total_nodes": 0},
        )
        assert isinstance(reward, float)
