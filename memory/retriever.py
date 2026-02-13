"""Graph retrieval: hybrid BM25 + vector + graph traversal.

Combines three retrieval methods via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from memory.graph_memory import GraphMemory


class GraphRetriever:
    """Retrieve relevant memory nodes using hybrid search."""

    def __init__(self, graph: GraphMemory, rrf_k: int = 60):
        self._graph = graph
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid retrieval: BM25 + vector + graph traversal, fused via RRF."""
        types = [memory_type] if memory_type else ["semantic", "episodic"]

        all_nodes = self._get_all_nodes(types)
        if not all_nodes:
            return []

        # BM25 ranking
        bm25_ranked = self._bm25_search(query, all_nodes)

        # Vector ranking (via GraphMemory.search)
        vector_ranked = self._graph.search(query, memory_type=memory_type, top_k=top_k * 2)

        # Graph traversal (expand from top vector hits)
        traversal_ranked = self._graph_traversal(vector_ranked[:3], types)

        # RRF fusion
        fused = self._rrf_fuse(
            [bm25_ranked, vector_ranked, traversal_ranked],
            k=self._rrf_k,
        )

        return fused[:top_k]

    def _get_all_nodes(self, types: list[str]) -> list[dict[str, Any]]:
        """Fetch all nodes of given types for BM25 indexing."""
        nodes = []
        for mt in types:
            label = "SemanticNode" if mt == "semantic" else "EpisodicNode"
            with self._graph._driver.session() as s:
                records = s.run(
                    f"MATCH (n:{label}) "
                    f"RETURN n.id AS id, n.content AS content, '{mt}' AS memory_type"
                )
                for r in records:
                    nodes.append(dict(r))
        return nodes

    def _bm25_search(
        self, query: str, nodes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """BM25 keyword search."""
        if not nodes:
            return []

        corpus = [n.get("content", "") for n in nodes]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        scored_nodes = []
        for i, node in enumerate(nodes):
            scored_nodes.append({**node, "score": float(scores[i])})

        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        return scored_nodes

    def _graph_traversal(
        self,
        anchor_nodes: list[dict[str, Any]],
        types: list[str],
    ) -> list[dict[str, Any]]:
        """Expand from anchor nodes via 1-2 hop graph traversal."""
        if not anchor_nodes:
            return []

        neighbors = []
        seen_ids = set()

        for anchor in anchor_nodes:
            anchor_id = anchor.get("id")
            if not anchor_id:
                continue

            for mt in types:
                label = "SemanticNode" if mt == "semantic" else "EpisodicNode"
                with self._graph._driver.session() as s:
                    # 1-2 hop neighbors
                    records = s.run(
                        f"MATCH (a:{label} {{id: $id}})-[r*1..2]-(b:{label}) "
                        f"WHERE b.id <> $id "
                        f"RETURN DISTINCT b.id AS id, b.content AS content, "
                        f"'{mt}' AS memory_type, 0.5 AS score",
                        id=anchor_id,
                    )
                    for r in records:
                        node = dict(r)
                        nid = node.get("id")
                        if nid and nid not in seen_ids:
                            seen_ids.add(nid)
                            neighbors.append(node)

        return neighbors

    def _rrf_fuse(
        self,
        rankings: list[list[dict[str, Any]]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of multiple rankings."""
        scores: dict[str, float] = {}
        node_map: dict[str, dict[str, Any]] = {}

        for ranking in rankings:
            for rank, node in enumerate(ranking):
                nid = node.get("id", "")
                if not nid:
                    continue
                scores[nid] = scores.get(nid, 0) + 1.0 / (k + rank + 1)
                if nid not in node_map:
                    node_map[nid] = node

        # Sort by RRF score
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for nid in sorted_ids:
            node = node_map[nid].copy()
            node["rrf_score"] = scores[nid]
            result.append(node)

        return result
