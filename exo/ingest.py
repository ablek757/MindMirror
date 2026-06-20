"""碎片摄入模块。"""
from __future__ import annotations

from exo.config import AppConfig
from exo.db import Database
from exo.embeddings import Embedder
from exo.models import Fragment


class IngestService:
    def __init__(self, config: AppConfig, db: Database, embedder: Embedder):
        self.config = config
        self.db = db
        self.embedder = embedder

    async def add(
        self, content: str, source: str = "manual", tags: list[str] | None = None
    ) -> Fragment:
        tags = tags or []
        row_id = self.db.insert_fragment(
            content=content.strip(),
            source=source,
            tags=",".join(tags),
        )
        # 为碎片生成 embedding
        vectors = await self.embedder.embed([content])
        if vectors:
            self.db.save_embedding("fragments", row_id, vectors[0])

        fragment = Fragment(
            id=row_id,
            content=content.strip(),
            source=source,
            tags=tags,
        )

        if self.config.graph.auto_weave:
            from exo.weaver import Weaver
            weaver = Weaver(self.config, self.db, self.embedder)
            await weaver.weave_fragment(fragment)

        return fragment

    def list_fragments(self, limit: int = 100) -> list[Fragment]:
        rows = self.db.list_fragments(limit)
        return [
            Fragment(
                id=row["id"],
                content=row["content"],
                source=row["source"],
                tags=row["tags"].split(",") if row["tags"] else [],
                created_at=row["created_at"],
                woven_at=row["woven_at"],
            )
            for row in rows
        ]
