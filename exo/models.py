"""Pydantic 数据模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Fragment(BaseModel):
    id: int | None = None
    content: str
    source: str = "manual"
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    woven_at: datetime | None = None

    def tags_str(self) -> str:
        return ",".join(self.tags)


class Node(BaseModel):
    id: int | None = None
    label: str
    node_type: str = "concept"
    description: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Edge(BaseModel):
    id: int | None = None
    source_id: int
    target_id: int
    relation: str = "relates_to"
    weight: float = 1.0


class ExtractedConcept(BaseModel):
    label: str
    node_type: str = "concept"
    description: str = ""


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation: str


class WeaveResult(BaseModel):
    fragment_id: int
    nodes: list[Node]
    edges: list[Edge]


class ChatMessage(BaseModel):
    role: str
    content: str
    mode: str = "default"


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    mode: str = "default"


class GraphExport(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
