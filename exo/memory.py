"""语义记忆检索。"""
from __future__ import annotations

from exo.config import AppConfig
from exo.db import Database
from exo.embeddings import Embedder, cosine_similarity
from exo.models import Fragment, Node


class MemoryService:
    def __init__(self, config: AppConfig, db: Database, embedder: Embedder):
        self.config = config
        self.db = db
        self.embedder = embedder

    async def search_fragments(self, query: str, top_k: int = 5) -> list[tuple[Fragment, float]]:
        vectors = await self.embedder.embed([query])
        if not vectors:
            return []
        query_vec = vectors[0]

        rows = self.db.list_fragments(limit=10000)
        embeddings = self.db.all_embeddings("fragments")
        scored: list[tuple[Fragment, float]] = []
        for row in rows:
            emb = embeddings.get(row["id"])
            if emb is None:
                continue
            score = cosine_similarity(query_vec, emb)
            scored.append(
                (
                    Fragment(
                        id=row["id"],
                        content=row["content"],
                        source=row["source"],
                        tags=row["tags"].split(",") if row["tags"] else [],
                        created_at=row["created_at"],
                        woven_at=row["woven_at"],
                    ),
                    score,
                )
            )
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def search_nodes(self, query: str, top_k: int = 5) -> list[tuple[Node, float]]:
        vectors = await self.embedder.embed([query])
        if not vectors:
            return []
        query_vec = vectors[0]

        rows = self.db.list_nodes(limit=10000)
        embeddings = self.db.all_embeddings("nodes")
        scored: list[tuple[Node, float]] = []
        for row in rows:
            emb = embeddings.get(row["id"])
            if emb is None:
                continue
            score = cosine_similarity(query_vec, emb)
            scored.append(
                (
                    Node(
                        id=row["id"],
                        label=row["label"],
                        node_type=row["node_type"],
                        description=row["description"],
                    ),
                    score,
                )
            )
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def hidden_connections(
        self, top_k: int = 5
    ) -> list[tuple[Node, Node, float]]:
        """发现跨碎片但尚未直接连接的相似节点对。"""
        rows = self.db.list_nodes(limit=10000)
        embeddings = self.db.all_embeddings("nodes")
        nodes = [
            Node(
                id=row["id"],
                label=row["label"],
                node_type=row["node_type"],
                description=row["description"],
            )
            for row in rows
            if row["id"] in embeddings
        ]

        edges = self.db.list_edges()
        connected = {(e["source_id"], e["target_id"]) for e in edges}
        connected.update({(e["target_id"], e["source_id"]) for e in edges})

        pairs: list[tuple[Node, Node, float]] = []
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                if a.id == b.id or (a.id, b.id) in connected:
                    continue
                score = cosine_similarity(embeddings[a.id], embeddings[b.id])
                if score >= self.config.graph.similarity_threshold:
                    pairs.append((a, b, score))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]

    async def forgotten_gems(self, top_k: int = 5) -> list[Node]:
        """返回长期未出现但可能重要的节点（目前按 created_at 最旧排序）。"""
        rows = self.db.list_nodes(limit=10000)
        rows.sort(key=lambda r: r["updated_at"] or r["created_at"])
        return [
            Node(
                id=row["id"],
                label=row["label"],
                node_type=row["node_type"],
                description=row["description"],
            )
            for row in rows[:top_k]
        ]
