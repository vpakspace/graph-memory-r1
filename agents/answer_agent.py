"""Answer Agent: GPT-4o-mini for question answering using graph memory.

The Answer Agent retrieves relevant nodes from graph memory and generates
answers. It is NOT trained (frozen) — only the Memory Manager learns.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import get_settings
from memory.graph_memory import GraphMemory
from memory.retriever import GraphRetriever


class AnswerAgent:
    """Answer questions using graph memory context."""

    def __init__(
        self,
        graph: GraphMemory,
        retriever: GraphRetriever | None = None,
        openai_client: OpenAI | None = None,
    ):
        self._graph = graph
        self._retriever = retriever or GraphRetriever(graph)
        settings = get_settings()
        self._client = openai_client or OpenAI(api_key=settings.openai.api_key)
        self._model = settings.openai.llm_model

    def answer(self, question: str, top_k: int = 10) -> dict[str, Any]:
        """Answer a question using graph memory.

        Returns dict with: answer, sources, confidence.
        """
        # Retrieve relevant nodes
        nodes = self._retriever.retrieve(question, top_k=top_k)

        # Format graph context
        context = self._format_context(nodes)

        # Generate answer
        answer = self._generate(question, context)

        return {
            "answer": answer,
            "sources": nodes,
            "num_sources": len(nodes),
            "context_length": len(context),
        }

    def answer_with_memory(
        self, question: str, rendered_memory: str = ""
    ) -> dict[str, Any]:
        """Answer using pre-rendered memory (used during GRPO evaluation)."""
        if not rendered_memory:
            rendered_memory = self._graph.render_memory()

        answer = self._generate(question, rendered_memory)

        return {
            "answer": answer,
            "context_length": len(rendered_memory),
        }

    def _format_context(self, nodes: list[dict[str, Any]]) -> str:
        """Format retrieved nodes as context for the LLM."""
        if not nodes:
            return "No relevant information found in memory."

        parts = []

        # Core memory
        core = self._graph.get_node("core", "core")
        if core and core.get("content"):
            parts.append(f"## Core Memory\n{core['content']}")

        # Group by memory type
        semantic = [n for n in nodes if n.get("memory_type") == "semantic"]
        episodic = [n for n in nodes if n.get("memory_type") == "episodic"]

        if semantic:
            lines = []
            for i, n in enumerate(semantic, 1):
                score = n.get("rrf_score", n.get("score", 0))
                lines.append(f"{i}. [{n.get('id', '?')}] {n.get('content', '')} (score: {score:.3f})")
            parts.append("## Relevant Facts (Semantic)\n" + "\n".join(lines))

        if episodic:
            lines = []
            for i, n in enumerate(episodic, 1):
                score = n.get("rrf_score", n.get("score", 0))
                lines.append(f"{i}. [{n.get('id', '?')}] {n.get('content', '')} (score: {score:.3f})")
            parts.append("## Relevant Events (Episodic)\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "No relevant information found."

    def _generate(self, question: str, context: str) -> str:
        """Generate answer using OpenAI."""
        system = (
            "You are a helpful assistant answering questions based on information from a memory graph. "
            "Use ONLY the provided memory context to answer. If the information is not in the context, "
            "say so clearly. Be concise and accurate."
        )

        user = f"Memory Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error generating answer: {e}"
