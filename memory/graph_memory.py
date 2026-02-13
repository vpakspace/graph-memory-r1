"""Graph-based memory backed by Neo4j.

Three memory types:
- Core: singleton profile (preferences, personality)
- Semantic: facts and knowledge nodes
- Episodic: events and experiences nodes
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase

from config import get_settings

MEMORY_TYPES = ("core", "semantic", "episodic")


class GraphMemory:
    """Neo4j-backed graph memory with three memory types."""

    def __init__(self, driver=None, embedder=None):
        settings = get_settings()
        if driver is None:
            self._driver = GraphDatabase.driver(
                settings.neo4j.uri,
                auth=(settings.neo4j.user, settings.neo4j.password),
            )
        else:
            self._driver = driver
        self._embedder = embedder  # callable: str -> list[float]
        self._settings = settings

    # ── Schema ────────────────────────────────────────────────

    def init_schema(self) -> None:
        """Create indexes and constraints."""
        with self._driver.session() as s:
            # Unique IDs
            s.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (n:SemanticNode) REQUIRE n.id IS UNIQUE"
            )
            s.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (n:EpisodicNode) REQUIRE n.id IS UNIQUE"
            )
            s.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (n:CoreMemory) REQUIRE n.id IS UNIQUE"
            )
            # Vector index for similarity search
            try:
                s.run(
                    "CREATE VECTOR INDEX semantic_embeddings IF NOT EXISTS "
                    "FOR (n:SemanticNode) ON (n.embedding) "
                    "OPTIONS {indexConfig: {"
                    "  `vector.dimensions`: 1536,"
                    "  `vector.similarity_function`: 'cosine'"
                    "}}"
                )
                s.run(
                    "CREATE VECTOR INDEX episodic_embeddings IF NOT EXISTS "
                    "FOR (n:EpisodicNode) ON (n.embedding) "
                    "OPTIONS {indexConfig: {"
                    "  `vector.dimensions`: 1536,"
                    "  `vector.similarity_function`: 'cosine'"
                    "}}"
                )
            except Exception:
                pass  # Vector indexes may already exist

    # ── CRUD ──────────────────────────────────────────────────

    def add_node(
        self,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a node to the graph. Returns node ID."""
        _validate_type(memory_type)
        node_id = str(uuid.uuid4())[:8]
        embedding = self._embed(content) if self._embedder else None
        now = datetime.now(timezone.utc).isoformat()

        if memory_type == "core":
            self._upsert_core(content, now)
            return "core"

        label = _label(memory_type)
        props: dict[str, Any] = {
            "id": node_id,
            "content": content,
            "created_at": now,
        }
        if embedding:
            props["embedding"] = embedding
        if metadata:
            props["metadata"] = str(metadata)

        with self._driver.session() as s:
            s.run(f"CREATE (n:{label} $props)", props=props)

        # Auto-link to similar nodes
        if embedding:
            self._auto_link(memory_type, node_id, embedding)

        return node_id

    def update_node(
        self,
        memory_type: str,
        node_id: str,
        new_content: str,
    ) -> bool:
        """Update node content, re-embed, re-link."""
        _validate_type(memory_type)

        if memory_type == "core":
            now = datetime.now(timezone.utc).isoformat()
            self._upsert_core(new_content, now)
            return True

        label = _label(memory_type)
        embedding = self._embed(new_content) if self._embedder else None
        now = datetime.now(timezone.utc).isoformat()

        params: dict[str, Any] = {
            "id": node_id,
            "content": new_content,
            "updated_at": now,
        }
        set_clause = "n.content = $content, n.updated_at = $updated_at"
        if embedding:
            params["embedding"] = embedding
            set_clause += ", n.embedding = $embedding"

        with self._driver.session() as s:
            result = s.run(
                f"MATCH (n:{label} {{id: $id}}) SET {set_clause} RETURN n.id",
                **params,
            )
            updated = result.single() is not None

        if updated and embedding:
            # Remove old RELATED_TO edges and re-link
            with self._driver.session() as s:
                s.run(
                    f"MATCH (n:{label} {{id: $id}})-[r:RELATED_TO]-() DELETE r",
                    id=node_id,
                )
            self._auto_link(memory_type, node_id, embedding)

        return updated

    def delete_node(self, memory_type: str, node_id: str) -> bool:
        """Delete a node and its edges."""
        _validate_type(memory_type)

        if memory_type == "core":
            with self._driver.session() as s:
                s.run("MATCH (n:CoreMemory) DETACH DELETE n")
            return True

        label = _label(memory_type)
        with self._driver.session() as s:
            result = s.run(
                f"MATCH (n:{label} {{id: $id}}) DETACH DELETE n RETURN count(*) AS c",
                id=node_id,
            )
            record = result.single()
            return record is not None and record["c"] > 0

    def get_node(self, memory_type: str, node_id: str) -> dict[str, Any] | None:
        """Get a single node by ID."""
        _validate_type(memory_type)

        if memory_type == "core":
            with self._driver.session() as s:
                result = s.run("MATCH (n:CoreMemory) RETURN n LIMIT 1")
                record = result.single()
                if record:
                    return dict(record["n"])
            return None

        label = _label(memory_type)
        with self._driver.session() as s:
            result = s.run(
                f"MATCH (n:{label} {{id: $id}}) RETURN n",
                id=node_id,
            )
            record = result.single()
            if record:
                return dict(record["n"])
        return None

    # ── Search ────────────────────────────────────────────────

    def search(
        self,
        query: str,
        memory_type: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid search: BM25 keyword + vector similarity."""
        results = []

        types = [memory_type] if memory_type else ["semantic", "episodic"]

        for mt in types:
            # Keyword search (contains)
            label = _label(mt)
            with self._driver.session() as s:
                records = s.run(
                    f"MATCH (n:{label}) "
                    f"WHERE toLower(n.content) CONTAINS toLower($query) "
                    f"RETURN n.id AS id, n.content AS content, "
                    f"n.created_at AS created_at, '{mt}' AS memory_type, "
                    f"1.0 AS score",
                    query=query,
                )
                for r in records:
                    results.append(dict(r))

            # Vector search (if embedder available)
            if self._embedder:
                query_embedding = self._embed(query)
                vector_results = self._vector_search(label, query_embedding, top_k)
                for vr in vector_results:
                    vr["memory_type"] = mt
                    results.append(vr)

        # Deduplicate by id, keep highest score
        seen: dict[str, dict] = {}
        for r in results:
            rid = r.get("id", "core")
            if rid not in seen or r.get("score", 0) > seen[rid].get("score", 0):
                seen[rid] = r

        # Sort by score descending
        ranked = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        return ranked[:top_k]

    def _vector_search(
        self, label: str, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """Vector similarity search using Neo4j vector index."""
        index_name = (
            "semantic_embeddings"
            if label == "SemanticNode"
            else "episodic_embeddings"
        )
        results = []
        with self._driver.session() as s:
            try:
                records = s.run(
                    f"CALL db.index.vector.queryNodes('{index_name}', $k, $embedding) "
                    f"YIELD node, score "
                    f"RETURN node.id AS id, node.content AS content, "
                    f"node.created_at AS created_at, score",
                    k=top_k,
                    embedding=query_embedding,
                )
                for r in records:
                    results.append(dict(r))
            except Exception:
                pass  # Index may not exist yet
        return results

    # ── Render ────────────────────────────────────────────────

    def render_memory(self, max_tokens: int = 2048) -> str:
        """Format current graph state as text for LLM context."""
        parts = []

        # Core memory
        with self._driver.session() as s:
            result = s.run("MATCH (n:CoreMemory) RETURN n.content AS content LIMIT 1")
            record = result.single()
            if record and record["content"]:
                parts.append(f"## Core Memory\n{record['content']}")

        # Semantic nodes
        with self._driver.session() as s:
            records = s.run(
                "MATCH (n:SemanticNode) "
                "RETURN n.id AS id, n.content AS content "
                "ORDER BY n.created_at DESC LIMIT 50"
            )
            semantic = [dict(r) for r in records]
        if semantic:
            lines = [f"- [{n['id']}] {n['content']}" for n in semantic]
            parts.append("## Semantic Memory (Facts)\n" + "\n".join(lines))

        # Episodic nodes
        with self._driver.session() as s:
            records = s.run(
                "MATCH (n:EpisodicNode) "
                "RETURN n.id AS id, n.content AS content "
                "ORDER BY n.created_at DESC LIMIT 50"
            )
            episodic = [dict(r) for r in records]
        if episodic:
            lines = [f"- [{n['id']}] {n['content']}" for n in episodic]
            parts.append("## Episodic Memory (Events)\n" + "\n".join(lines))

        # Relationships
        with self._driver.session() as s:
            records = s.run(
                "MATCH (a)-[r:RELATED_TO]->(b) "
                "RETURN a.id AS from_id, b.id AS to_id, "
                "r.weight AS weight LIMIT 100"
            )
            edges = [dict(r) for r in records]
        if edges:
            lines = [
                f"- {e['from_id']} -> {e['to_id']} (weight: {e.get('weight', 0):.2f})"
                for e in edges
            ]
            parts.append("## Relationships\n" + "\n".join(lines))

        text = "\n\n".join(parts)

        # Truncate to max_tokens (rough: 1 token ~ 4 chars)
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"

        return text

    # ── Stats ─────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return node/edge counts."""
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count"
            )
            node_counts = {r["label"]: r["count"] for r in result}

            result = s.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")
            edge_counts = {r["type"]: r["count"] for r in result}

        return {
            "nodes": node_counts,
            "edges": edge_counts,
            "total_nodes": sum(node_counts.values()),
            "total_edges": sum(edge_counts.values()),
        }

    # ── Cleanup ───────────────────────────────────────────────

    def clear(self) -> None:
        """Delete all graph memory data."""
        with self._driver.session() as s:
            s.run(
                "MATCH (n) WHERE n:CoreMemory OR n:SemanticNode OR n:EpisodicNode "
                "DETACH DELETE n"
            )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()

    # ── Private ───────────────────────────────────────────────

    def _upsert_core(self, content: str, updated_at: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (n:CoreMemory {id: 'core'}) "
                "SET n.content = $content, n.updated_at = $updated_at",
                content=content,
                updated_at=updated_at,
            )

    def _embed(self, text: str) -> list[float]:
        if self._embedder:
            return self._embedder(text)
        return []

    def _auto_link(
        self, memory_type: str, node_id: str, embedding: list[float]
    ) -> None:
        """Create RELATED_TO edges to similar existing nodes."""
        label = _label(memory_type)
        similar = self._vector_search(label, embedding, top_k=5)
        threshold = self._settings.similarity_threshold

        with self._driver.session() as s:
            for node in similar:
                other_id = node.get("id")
                score = node.get("score", 0)
                if other_id and other_id != node_id and score >= threshold:
                    s.run(
                        f"MATCH (a:{label} {{id: $a_id}}), (b:{label} {{id: $b_id}}) "
                        f"MERGE (a)-[:RELATED_TO {{weight: $weight}}]->(b)",
                        a_id=node_id,
                        b_id=other_id,
                        weight=score,
                    )


def _validate_type(memory_type: str) -> None:
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"Invalid memory_type '{memory_type}', must be one of {MEMORY_TYPES}")


def _label(memory_type: str) -> str:
    return {
        "core": "CoreMemory",
        "semantic": "SemanticNode",
        "episodic": "EpisodicNode",
    }[memory_type]
