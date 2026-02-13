"""Tests for memory operations parsing and execution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memory.operations import (
    MemoryOperation,
    OperationResult,
    execute_operations,
    parse_tool_calls,
    validate_operations,
)


class TestParseToolCalls:
    def test_parse_add(self):
        text = '''<tool_call>
{"name": "graph_memory_add", "arguments": {"memory_type": "semantic", "content": "User likes Python"}}
</tool_call>'''
        ops = parse_tool_calls(text)
        assert len(ops) == 1
        assert ops[0].action == "add"
        assert ops[0].memory_type == "semantic"
        assert ops[0].content == "User likes Python"

    def test_parse_update(self):
        text = '''<tool_call>
{"name": "graph_memory_update", "arguments": {"memory_type": "semantic", "node_id": "abc123", "new_content": "Updated fact"}}
</tool_call>'''
        ops = parse_tool_calls(text)
        assert len(ops) == 1
        assert ops[0].action == "update"
        assert ops[0].node_id == "abc123"

    def test_parse_delete(self):
        text = '''<tool_call>
{"name": "graph_memory_delete", "arguments": {"memory_type": "episodic", "node_id": "xyz789"}}
</tool_call>'''
        ops = parse_tool_calls(text)
        assert len(ops) == 1
        assert ops[0].action == "delete"

    def test_parse_noop(self):
        text = '''<tool_call>
{"name": "graph_memory_noop", "arguments": {}}
</tool_call>'''
        ops = parse_tool_calls(text)
        assert len(ops) == 1
        assert ops[0].action == "noop"

    def test_parse_multiple(self):
        text = '''<tool_call>
{"name": "graph_memory_add", "arguments": {"memory_type": "semantic", "content": "Fact 1"}}
</tool_call>
<tool_call>
{"name": "graph_memory_add", "arguments": {"memory_type": "episodic", "content": "Event 1"}}
</tool_call>'''
        ops = parse_tool_calls(text)
        assert len(ops) == 2

    def test_parse_invalid_json(self):
        text = '''<tool_call>
not valid json
</tool_call>'''
        ops = parse_tool_calls(text)
        assert len(ops) == 0

    def test_parse_unknown_function(self):
        text = '''<tool_call>
{"name": "unknown_function", "arguments": {}}
</tool_call>'''
        ops = parse_tool_calls(text)
        assert len(ops) == 0

    def test_parse_plain_json(self):
        text = '{"name": "graph_memory_add", "arguments": {"memory_type": "semantic", "content": "Plain JSON"}}'
        ops = parse_tool_calls(text)
        assert len(ops) == 1
        assert ops[0].content == "Plain JSON"


class TestExecuteOperations:
    @pytest.fixture
    def mock_graph(self):
        graph = MagicMock(spec=["add_node", "update_node", "delete_node"])
        graph.add_node.return_value = "new-id"
        graph.update_node.return_value = True
        graph.delete_node.return_value = True
        return graph

    def test_execute_add(self, mock_graph):
        ops = [MemoryOperation(action="add", memory_type="semantic", content="Test")]
        result = execute_operations(mock_graph, ops)
        assert result.total == 1
        assert result.successful == 1
        mock_graph.add_node.assert_called_once_with("semantic", "Test")

    def test_execute_update(self, mock_graph):
        ops = [MemoryOperation(
            action="update", memory_type="semantic",
            node_id="abc", content="Updated",
        )]
        result = execute_operations(mock_graph, ops)
        assert result.successful == 1
        mock_graph.update_node.assert_called_once_with("semantic", "abc", "Updated")

    def test_execute_delete(self, mock_graph):
        ops = [MemoryOperation(
            action="delete", memory_type="semantic", node_id="abc",
        )]
        result = execute_operations(mock_graph, ops)
        assert result.successful == 1
        mock_graph.delete_node.assert_called_once_with("semantic", "abc")

    def test_execute_noop(self, mock_graph):
        ops = [MemoryOperation(action="noop")]
        result = execute_operations(mock_graph, ops)
        assert result.successful == 1

    def test_execute_error(self, mock_graph):
        mock_graph.add_node.side_effect = Exception("DB error")
        ops = [MemoryOperation(action="add", memory_type="semantic", content="Test")]
        result = execute_operations(mock_graph, ops)
        assert result.failed == 1
        assert "Error" in result.operations[0].result

    def test_execute_unknown_action(self, mock_graph):
        ops = [MemoryOperation(action="unknown")]
        result = execute_operations(mock_graph, ops)
        assert result.failed == 1


class TestValidateOperations:
    def test_valid_add(self):
        ops = [MemoryOperation(action="add", memory_type="semantic", content="Test")]
        assert validate_operations(ops) == 1.0

    def test_invalid_add_no_content(self):
        ops = [MemoryOperation(action="add", memory_type="semantic", content="")]
        assert validate_operations(ops) == 0.0

    def test_valid_noop(self):
        ops = [MemoryOperation(action="noop")]
        assert validate_operations(ops) == 1.0

    def test_mixed(self):
        ops = [
            MemoryOperation(action="add", memory_type="semantic", content="Test"),
            MemoryOperation(action="add", memory_type="", content="Missing type"),
        ]
        assert validate_operations(ops) == 0.5

    def test_empty(self):
        assert validate_operations([]) == 0.0
