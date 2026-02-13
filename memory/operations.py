"""Graph memory operations: ADD, UPDATE, DELETE, NOOP.

Parses tool calls from Memory Manager LLM output and executes them on GraphMemory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from memory.graph_memory import GraphMemory


@dataclass
class MemoryOperation:
    """A single memory operation."""
    action: str  # add, update, delete, noop
    memory_type: str = ""
    node_id: str = ""
    content: str = ""
    result: str = ""
    success: bool = False


@dataclass
class OperationResult:
    """Result of executing a batch of operations."""
    operations: list[MemoryOperation] = field(default_factory=list)
    total: int = 0
    successful: int = 0
    failed: int = 0


def parse_tool_calls(text: str) -> list[MemoryOperation]:
    """Parse tool calls from LLM output.

    Supports format:
    <tool_call>
    {"name": "graph_memory_add", "arguments": {"memory_type": "semantic", "content": "..."}}
    </tool_call>
    """
    ops: list[MemoryOperation] = []

    # Extract tool_call blocks
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            call = json.loads(match.strip())
            name = call.get("name", "")
            args = call.get("arguments", {})
            op = _call_to_operation(name, args)
            if op:
                ops.append(op)
        except json.JSONDecodeError:
            continue

    # Also support plain JSON tool calls (no XML wrapper)
    if not ops:
        # Try parsing entire text as JSON
        try:
            call = json.loads(text.strip())
            if isinstance(call, dict) and call.get("name", "").startswith("graph_memory_"):
                op = _call_to_operation(call["name"], call.get("arguments", {}))
                if op:
                    ops.append(op)
        except json.JSONDecodeError:
            pass

    return ops


def _call_to_operation(name: str, args: dict[str, Any]) -> MemoryOperation | None:
    """Convert a tool call name + args to a MemoryOperation."""
    if name == "graph_memory_add":
        return MemoryOperation(
            action="add",
            memory_type=args.get("memory_type", "semantic"),
            content=args.get("content", ""),
        )
    elif name == "graph_memory_update":
        return MemoryOperation(
            action="update",
            memory_type=args.get("memory_type", "semantic"),
            node_id=args.get("node_id", ""),
            content=args.get("new_content", args.get("content", "")),
        )
    elif name == "graph_memory_delete":
        return MemoryOperation(
            action="delete",
            memory_type=args.get("memory_type", "semantic"),
            node_id=args.get("node_id", ""),
        )
    elif name == "graph_memory_noop":
        return MemoryOperation(action="noop")
    return None


def execute_operations(
    graph: GraphMemory,
    operations: list[MemoryOperation],
) -> OperationResult:
    """Execute a list of memory operations on the graph."""
    result = OperationResult(total=len(operations))

    for op in operations:
        try:
            if op.action == "add":
                node_id = graph.add_node(op.memory_type, op.content)
                op.node_id = node_id
                op.result = f"Added node {node_id}"
                op.success = True

            elif op.action == "update":
                updated = graph.update_node(op.memory_type, op.node_id, op.content)
                op.result = f"Updated node {op.node_id}" if updated else "Node not found"
                op.success = updated

            elif op.action == "delete":
                deleted = graph.delete_node(op.memory_type, op.node_id)
                op.result = f"Deleted node {op.node_id}" if deleted else "Node not found"
                op.success = deleted

            elif op.action == "noop":
                op.result = "No operation"
                op.success = True

            else:
                op.result = f"Unknown action: {op.action}"
                op.success = False

        except Exception as e:
            op.result = f"Error: {e}"
            op.success = False

        result.operations.append(op)

    result.successful = sum(1 for o in result.operations if o.success)
    result.failed = result.total - result.successful
    return result


def validate_operations(operations: list[MemoryOperation]) -> float:
    """Score how well-formed the operations are (0.0-1.0)."""
    if not operations:
        return 0.0

    valid = 0
    for op in operations:
        if op.action == "noop":
            valid += 1
        elif op.action == "add" and op.memory_type and op.content:
            valid += 1
        elif op.action == "update" and op.memory_type and op.node_id and op.content:
            valid += 1
        elif op.action == "delete" and op.memory_type and op.node_id:
            valid += 1

    return valid / len(operations)
