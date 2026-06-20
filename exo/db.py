"""SQLite 数据库连接与基础操作。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from exo.config import AppConfig


class Database:
    def __init__(self, config: AppConfig):
        self.config = config
        self.path = Path(config.db.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        if not schema_path.exists():
            return
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
        with self._connect() as conn:
            conn.executescript(sql)

    def execute(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> sqlite3.Cursor:
        with self._connect() as conn:
            return conn.execute(sql, params)

    def fetchone(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()

    def fetchall(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def insert_fragment(self, content: str, source: str, tags: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO fragments (content, source, tags) VALUES (?, ?, ?)",
                (content, source, tags),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_unwoven_fragments(self) -> list[sqlite3.Row]:
        return self.fetchall(
            "SELECT * FROM fragments WHERE woven_at IS NULL ORDER BY created_at"
        )

    def mark_fragment_woven(self, fragment_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE fragments SET woven_at = CURRENT_TIMESTAMP WHERE id = ?",
                (fragment_id,),
            )
            conn.commit()

    def list_fragments(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.fetchall(
            "SELECT * FROM fragments ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    def save_embedding(self, table: str, row_id: int, vector: list[float]) -> None:
        blob = json.dumps(vector, ensure_ascii=False).encode("utf-8")
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {table} SET embedding_blob = ? WHERE id = ?", (blob, row_id)
            )
            conn.commit()

    def get_embedding(self, table: str, row_id: int) -> list[float] | None:
        row = self.fetchone(
            f"SELECT embedding_blob FROM {table} WHERE id = ?", (row_id,)
        )
        if row and row["embedding_blob"]:
            return json.loads(row["embedding_blob"].decode("utf-8"))
        return None

    def all_embeddings(self, table: str) -> dict[int, list[float]]:
        rows = self.fetchall(f"SELECT id, embedding_blob FROM {table} WHERE embedding_blob IS NOT NULL")
        return {
            row["id"]: json.loads(row["embedding_blob"].decode("utf-8"))
            for row in rows
        }

    def add_or_get_node(
        self, label: str, node_type: str, description: str = ""
    ) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM nodes WHERE label = ? AND node_type = ?",
                (label, node_type),
            ).fetchone()
            if row:
                node_id = row["id"]
                if description:
                    conn.execute(
                        "UPDATE nodes SET description = ? WHERE id = ?",
                        (description, node_id),
                    )
                conn.commit()
                return node_id  # type: ignore[return-value]

            cur = conn.execute(
                "INSERT INTO nodes (label, node_type, description) VALUES (?, ?, ?)",
                (label, node_type, description),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_node_by_id(self, node_id: int) -> sqlite3.Row | None:
        return self.fetchone("SELECT * FROM nodes WHERE id = ?", (node_id,))

    def list_nodes(self, limit: int = 1000) -> list[sqlite3.Row]:
        return self.fetchall("SELECT * FROM nodes ORDER BY updated_at DESC LIMIT ?", (limit,))

    def add_edge(
        self, source_id: int, target_id: int, relation: str = "relates_to", weight: float = 1.0
    ) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, weight FROM edges WHERE source_id = ? AND target_id = ? AND relation = ?",
                (source_id, target_id, relation),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE edges SET weight = weight + ? WHERE id = ?",
                    (weight, row["id"]),
                )
                conn.commit()
                return row["id"]  # type: ignore[return-value]

            cur = conn.execute(
                "INSERT INTO edges (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)",
                (source_id, target_id, relation, weight),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def list_edges(self) -> list[sqlite3.Row]:
        return self.fetchall("SELECT * FROM edges")

    def link_fragment_node(self, fragment_id: int, node_id: int, confidence: float = 1.0) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO fragment_nodes (fragment_id, node_id, confidence) VALUES (?, ?, ?)",
                (fragment_id, node_id, confidence),
            )
            conn.commit()

    def get_fragment_nodes(self, fragment_id: int) -> list[sqlite3.Row]:
        return self.fetchall(
            """
            SELECT n.* FROM nodes n
            JOIN fragment_nodes fn ON n.id = fn.node_id
            WHERE fn.fragment_id = ?
            """,
            (fragment_id,),
        )

    def get_node_fragments(self, node_id: int) -> list[sqlite3.Row]:
        return self.fetchall(
            """
            SELECT f.* FROM fragments f
            JOIN fragment_nodes fn ON f.id = fn.fragment_id
            WHERE fn.node_id = ?
            ORDER BY f.created_at DESC
            """,
            (node_id,),
        )

    def get_stats(self) -> dict[str, int]:
        with self._connect() as conn:
            fragments = conn.execute(
                "SELECT COUNT(*) FROM fragments"
            ).fetchone()[0]
            nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            unwoven = conn.execute(
                "SELECT COUNT(*) FROM fragments WHERE woven_at IS NULL"
            ).fetchone()[0]
            return {
                "fragments": fragments,
                "nodes": nodes,
                "edges": edges,
                "unwoven_fragments": unwoven,
            }

    def add_conversation_message(self, session_id: str, role: str, content: str, mode: str = "default") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (session_id, role, content, mode) VALUES (?, ?, ?, ?)",
                (session_id, role, content, mode),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_conversation_history(self, session_id: str, limit: int = 20) -> list[sqlite3.Row]:
        return self.fetchall(
            "SELECT * FROM conversations WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )

    def add_insight(self, session_id: str | None, content: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO insights (session_id, content) VALUES (?, ?)",
                (session_id, content),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]
