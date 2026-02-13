"""Tests for GraphMemory CRUD operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memory.graph_memory import GraphMemory, _label, _validate_type


class TestValidation:
    def test_valid_types(self):
        for t in ("core", "semantic", "episodic"):
            _validate_type(t)  # should not raise

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid memory_type"):
            _validate_type("invalid")

    def test_label_mapping(self):
        assert _label("core") == "CoreMemory"
        assert _label("semantic") == "SemanticNode"
        assert _label("episodic") == "EpisodicNode"


class TestGraphMemory:
    @pytest.fixture
    def mock_session(self):
        session = MagicMock()
        result = MagicMock()
        result.single.return_value = {"c": 1}
        session.run.return_value = result
        return session

    @pytest.fixture
    def mock_driver(self, mock_session):
        driver = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver

    @pytest.fixture
    def graph(self, mock_driver):
        return GraphMemory(driver=mock_driver)

    def test_add_semantic_node(self, graph, mock_session):
        node_id = graph.add_node("semantic", "Test fact")
        assert node_id  # non-empty string
        assert mock_session.run.called

    def test_add_core_memory(self, graph, mock_session):
        node_id = graph.add_node("core", "User likes Python")
        assert node_id == "core"
        # Should call MERGE for core
        call_args = mock_session.run.call_args_list[-1]
        assert "MERGE" in call_args[0][0]

    def test_update_core_memory(self, graph, mock_session):
        result = graph.update_node("core", "core", "Updated profile")
        assert result is True

    def test_update_semantic_node(self, graph, mock_session):
        mock_result = MagicMock()
        mock_result.single.return_value = {"id": "abc123"}
        mock_session.run.return_value = mock_result

        result = graph.update_node("semantic", "abc123", "Updated fact")
        assert result is True

    def test_update_nonexistent_node(self, graph, mock_session):
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        result = graph.update_node("semantic", "nonexistent", "content")
        assert result is False

    def test_delete_core(self, graph, mock_session):
        result = graph.delete_node("core", "core")
        assert result is True

    def test_delete_semantic_node(self, graph, mock_session):
        result = graph.delete_node("semantic", "abc123")
        assert result is True

    def test_get_node(self, graph, mock_session):
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value={"id": "abc", "content": "test"})
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        node = graph.get_node("semantic", "abc")
        assert node is not None

    def test_render_memory(self, graph, mock_session):
        # Mock returns empty results
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        text = graph.render_memory()
        assert isinstance(text, str)

    def test_get_stats(self, graph, mock_session):
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        stats = graph.get_stats()
        assert "total_nodes" in stats
        assert "total_edges" in stats

    def test_clear(self, graph, mock_session):
        graph.clear()
        assert mock_session.run.called

    def test_close(self, graph):
        graph.close()
        graph._driver.close.assert_called_once()

    def test_add_with_embedder(self, mock_driver, mock_session):
        embedder = MagicMock(return_value=[0.1] * 1536)
        graph = GraphMemory(driver=mock_driver, embedder=embedder)

        # Mock vector search to return empty (no similar nodes)
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        node_id = graph.add_node("semantic", "Test with embedding")
        assert node_id
        embedder.assert_called_with("Test with embedding")

    def test_search_keyword(self, graph, mock_session):
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        results = graph.search("test query")
        assert isinstance(results, list)
