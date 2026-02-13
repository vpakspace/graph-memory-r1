"""Tool schemas for Memory Manager function calling.

Defines the tools available to the Memory Manager agent in OpenAI function calling format.
"""

from __future__ import annotations

GRAPH_MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "graph_memory_add",
            "description": (
                "Add a new memory node to the graph. Use 'core' for user profile/preferences, "
                "'semantic' for facts and knowledge, 'episodic' for events and experiences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": ["core", "semantic", "episodic"],
                        "description": "Type of memory to add",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content of the memory to store",
                    },
                },
                "required": ["memory_type", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_memory_update",
            "description": "Update an existing memory node with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": ["core", "semantic", "episodic"],
                    },
                    "node_id": {
                        "type": "string",
                        "description": "ID of the node to update",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content for the node",
                    },
                },
                "required": ["memory_type", "node_id", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_memory_delete",
            "description": "Delete a memory node that is outdated or incorrect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": ["core", "semantic", "episodic"],
                    },
                    "node_id": {
                        "type": "string",
                        "description": "ID of the node to delete",
                    },
                },
                "required": ["memory_type", "node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_memory_noop",
            "description": "No operation needed. Use when the conversation chunk has no important information to store.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def get_tool_descriptions() -> str:
    """Format tool schemas as text for prompt injection."""
    lines = []
    for tool in GRAPH_MEMORY_TOOLS:
        func = tool["function"]
        name = func["name"]
        desc = func["description"]
        params = func["parameters"].get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type', 'str')}" for k, v in params.items()
        )
        lines.append(f"- {name}({param_str}): {desc}")
    return "\n".join(lines)
