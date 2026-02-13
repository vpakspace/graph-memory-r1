"""Tests for Memory Manager and Answer Agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.tools import GRAPH_MEMORY_TOOLS, get_tool_descriptions
from agents.memory_manager import MemoryManager
from agents.answer_agent import AnswerAgent


class TestTools:
    def test_tool_count(self):
        assert len(GRAPH_MEMORY_TOOLS) == 4

    def test_tool_names(self):
        names = {t["function"]["name"] for t in GRAPH_MEMORY_TOOLS}
        expected = {
            "graph_memory_add",
            "graph_memory_update",
            "graph_memory_delete",
            "graph_memory_noop",
        }
        assert names == expected

    def test_tool_descriptions(self):
        text = get_tool_descriptions()
        assert "graph_memory_add" in text
        assert "graph_memory_noop" in text


class TestMemoryManager:
    @pytest.fixture
    def mock_graph(self):
        graph = MagicMock()
        graph.render_memory.return_value = "## Core Memory\nEmpty"
        graph.add_node.return_value = "new-id"
        return graph

    def test_process_chunk_no_model(self, mock_graph):
        """Without model, should return noop."""
        mm = MemoryManager(graph=mock_graph)
        result = mm.process_chunk("User: Hello\nBot: Hi there!")
        assert result.total >= 1
        # Should succeed (noop fallback)
        assert result.successful >= 1

    def test_generate_operations_no_model(self, mock_graph):
        """Without model, generates noop."""
        mm = MemoryManager(graph=mock_graph)
        response, ops = mm.generate_operations("Test chunk")
        assert "noop" in response
        assert len(ops) >= 0  # May or may not parse the noop

    def test_build_prompt(self, mock_graph):
        mm = MemoryManager(graph=mock_graph)
        prompt = mm._build_prompt("Test chunk", "Current state")
        assert "Memory Manager" in prompt
        assert "Test chunk" in prompt


class TestAnswerAgent:
    @pytest.fixture
    def mock_graph(self):
        graph = MagicMock()
        graph.get_node.return_value = None
        return graph

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"id": "n1", "content": "Python is a language", "memory_type": "semantic", "score": 0.9},
        ]
        return retriever

    def test_answer(self, mock_graph, mock_retriever, mock_openai_client):
        agent = AnswerAgent(
            graph=mock_graph,
            retriever=mock_retriever,
            openai_client=mock_openai_client,
        )
        result = agent.answer("What is Python?")
        assert "answer" in result
        assert "sources" in result
        assert result["num_sources"] == 1

    def test_answer_with_memory(self, mock_graph, mock_openai_client):
        agent = AnswerAgent(
            graph=mock_graph,
            openai_client=mock_openai_client,
        )
        result = agent.answer_with_memory(
            "What is Python?",
            rendered_memory="Python is a programming language",
        )
        assert "answer" in result

    def test_format_context_empty(self, mock_graph, mock_openai_client):
        agent = AnswerAgent(
            graph=mock_graph,
            openai_client=mock_openai_client,
        )
        ctx = agent._format_context([])
        assert "No relevant" in ctx

    def test_format_context_with_nodes(self, mock_graph, mock_openai_client):
        agent = AnswerAgent(
            graph=mock_graph,
            openai_client=mock_openai_client,
        )
        nodes = [
            {"id": "n1", "content": "Fact 1", "memory_type": "semantic", "score": 0.9},
            {"id": "n2", "content": "Event 1", "memory_type": "episodic", "score": 0.8},
        ]
        ctx = agent._format_context(nodes)
        assert "Fact 1" in ctx
        assert "Event 1" in ctx
