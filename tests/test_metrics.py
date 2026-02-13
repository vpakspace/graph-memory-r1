"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from evaluation.metrics import bleu1, compute_all_metrics, exact_match, f1_score


class TestF1:
    def test_identical(self):
        assert f1_score("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self):
        assert f1_score("cat", "dog") == 0.0

    def test_partial(self):
        score = f1_score("the cat sat on the mat", "the cat")
        assert 0.4 < score < 0.8

    def test_empty(self):
        assert f1_score("", "") == 1.0
        assert f1_score("hello", "") == 0.0
        assert f1_score("", "hello") == 0.0


class TestExactMatch:
    def test_match(self):
        assert exact_match("yes", "yes") == 1.0

    def test_normalized_match(self):
        assert exact_match("Yes!", "yes") == 1.0

    def test_no_match(self):
        assert exact_match("yes", "no") == 0.0


class TestBleu1:
    def test_identical(self):
        assert bleu1("the cat", "the cat") == 1.0

    def test_no_overlap(self):
        assert bleu1("cat", "dog") == 0.0

    def test_partial(self):
        score = bleu1("the cat sat", "the cat")
        assert 0.0 < score <= 1.0

    def test_empty_pred(self):
        assert bleu1("", "hello") == 0.0

    def test_empty_ref(self):
        assert bleu1("hello", "") == 0.0


class TestComputeAll:
    def test_returns_all(self):
        result = compute_all_metrics("hello world", "hello world")
        assert "f1" in result
        assert "em" in result
        assert "bleu1" in result
        assert result["f1"] == 1.0
        assert result["em"] == 1.0
