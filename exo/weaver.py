"""认知编织：从碎片中提取实体关系并织入图谱。"""
from __future__ import annotations

import json
import re
from typing import Any

from exo.config import AppConfig
from exo.db import Database
from exo.embeddings import Embedder, cosine_similarity
from exo.llm import LLMClient, build_messages, get_default_prompt
from exo.models import Edge, ExtractedConcept, ExtractedRelation, Fragment, Node, WeaveResult


class Weaver:
    def __init__(
        self,
        config: AppConfig,
        db: Database,
        embedder: Embedder,
        llm: LLMClient | None = None,
    ):
        self.config = config
        self.db = db
        self.embedder = embedder
        self.llm = llm or LLMClient(config)

    async def weave_all(self) -> list[WeaveResult]:
        rows = self.db.get_unwoven_fragments()
        results: list[WeaveResult] = []
        for row in rows:
            fragment = Fragment(
                id=row["id"],
                content=row["content"],
                source=row["source"],
                tags=row["tags"].split(",") if row["tags"] else [],
                created_at=row["created_at"],
            )
            result = await self.weave_fragment(fragment)
            results.append(result)
        return results

    async def weave_fragment(self, fragment: Fragment) -> WeaveResult:
        extraction = await self._extract(fragment)
        nodes: list[Node] = []
        label_to_id: dict[str, int] = {}

        # 限制节点数量
        extraction.nodes = extraction.nodes[: self.config.graph.max_nodes_per_fragment]

        # 获取现有节点用于相似度合并
        existing_nodes = self.db.list_nodes(limit=10000)
        existing_embeddings = self.db.all_embeddings("nodes")

        for concept in extraction.nodes:
            # 1. 尝试合并相似节点
            merged_id = await self._find_similar_node(
                concept, existing_nodes, existing_embeddings
            )
            if merged_id is not None:
                label_to_id[concept.label] = merged_id
                node_row = self.db.get_node_by_id(merged_id)
                if node_row:
                    nodes.append(Node(**dict(node_row)))
                self.db.link_fragment_node(fragment.id or 0, merged_id, confidence=0.9)
                continue

            # 2. 创建新节点
            node_id = self.db.add_or_get_node(
                concept.label, concept.node_type, concept.description
            )
            label_to_id[concept.label] = node_id
            nodes.append(
                Node(
                    id=node_id,
                    label=concept.label,
                    node_type=concept.node_type,
                    description=concept.description,
                )
            )
            self.db.link_fragment_node(fragment.id or 0, node_id, confidence=1.0)

            # 3. 生成并保存节点 embedding
            text = f"{concept.label} {concept.node_type} {concept.description}".strip()
            vectors = await self.embedder.embed([text])
            if vectors:
                self.db.save_embedding("nodes", node_id, vectors[0])
                existing_embeddings[node_id] = vectors[0]

        edges: list[Edge] = []
        for rel in extraction.edges:
            source_id = label_to_id.get(rel.source)
            target_id = label_to_id.get(rel.target)
            if source_id and target_id and source_id != target_id:
                edge_id = self.db.add_edge(source_id, target_id, rel.relation)
                edges.append(
                    Edge(
                        id=edge_id,
                        source_id=source_id,
                        target_id=target_id,
                        relation=rel.relation,
                    )
                )

        self.db.mark_fragment_woven(fragment.id or 0)
        return WeaveResult(fragment_id=fragment.id or 0, nodes=nodes, edges=edges)

    async def _extract(self, fragment: Fragment) -> "ExtractionResult":
        prompt = get_default_prompt()
        messages = build_messages(prompt, fragment.content)
        response = await self.llm.complete(messages)
        if not isinstance(response, str):
            response = ""
        return self._parse_extraction(response)

    def _parse_extraction(self, text: str) -> "ExtractionResult":
        """解析 LLM 返回的 JSON，失败时返回保守的兜底结果。"""
        try:
            # 尝试提取 JSON 代码块
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
            if match:
                text = match.group(1)
            data = json.loads(text)
            nodes = [
                ExtractedConcept(
                    label=str(n.get("label", "")),
                    node_type=n.get("node_type", "concept"),
                    description=n.get("description", ""),
                )
                for n in data.get("nodes", [])
                if n.get("label")
            ]
            edges = [
                ExtractedRelation(
                    source=str(e.get("source", "")),
                    target=str(e.get("target", "")),
                    relation=e.get("relation", "relates_to"),
                )
                for e in data.get("edges", [])
                if e.get("source") and e.get("target")
            ]
            return ExtractionResult(nodes=nodes, edges=edges)
        except (json.JSONDecodeError, ValueError, TypeError):
            # 兜底：从文本中提取几个关键词作为节点
            words = re.findall(r"[\u4e00-\u9fa5]{2,8}|[a-zA-Z]{4,}", text)
            unique = list(dict.fromkeys(words))[:5]
            return ExtractionResult(
                nodes=[ExtractedConcept(label=w, node_type="concept") for w in unique],
                edges=[],
            )

    async def _find_similar_node(
        self,
        concept: ExtractedConcept,
        existing_nodes: list[Any],
        existing_embeddings: dict[int, list[float]],
    ) -> int | None:
        if not existing_nodes:
            return None

        text = f"{concept.label} {concept.node_type} {concept.description}".strip()
        vectors = await self.embedder.embed([text])
        if not vectors:
            return None
        new_vec = vectors[0]

        best_id: int | None = None
        best_score = 0.0
        for node in existing_nodes:
            node_id = node["id"]
            emb = existing_embeddings.get(node_id)
            if emb is None:
                continue
            score = cosine_similarity(new_vec, emb)
            if score > best_score:
                best_score = score
                best_id = node_id

        if best_score >= self.config.graph.similarity_threshold:
            return best_id
        return None


class ExtractionResult:
    def __init__(self, nodes: list[ExtractedConcept], edges: list[ExtractedRelation]):
        self.nodes = nodes
        self.edges = edges
