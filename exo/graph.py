"""图存储与导出服务。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import networkx as nx

from exo.db import Database


class GraphService:
    def __init__(self, db: Database):
        self.db = db

    def stats(self) -> dict[str, int]:
        return self.db.get_stats()

    def build_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        nodes = self.db.list_nodes(limit=10000)
        for row in nodes:
            G.add_node(
                row["id"],
                label=row["label"],
                node_type=row["node_type"],
                description=row["description"],
            )
        edges = self.db.list_edges()
        for row in edges:
            G.add_edge(
                row["source_id"],
                row["target_id"],
                relation=row["relation"],
                weight=row["weight"],
            )
        return G

    def export_graph(self) -> dict[str, Any]:
        nodes = self.db.list_nodes(limit=10000)
        edges = self.db.list_edges()
        return {
            "nodes": [
                {
                    "id": row["id"],
                    "label": row["label"],
                    "type": row["node_type"],
                    "description": row["description"],
                }
                for row in nodes
            ],
            "edges": [
                {
                    "id": row["id"],
                    "source": row["source_id"],
                    "target": row["target_id"],
                    "relation": row["relation"],
                    "weight": row["weight"],
                }
                for row in edges
            ],
        }

    def export_markdown(self) -> str:
        nodes = self.db.list_nodes(limit=10000)
        fragments = self.db.list_fragments(limit=1000)
        lines = ["# 个人认知外脑图谱", ""]
        lines.append(f"> 导出时间：{datetime.now().isoformat()}")
        lines.append("")
        lines.append(f"## 节点（{len(nodes)}）")
        lines.append("")
        for row in nodes:
            lines.append(f"### {row['label']} `#{row['id']}`")
            lines.append(f"- 类型：{row['node_type']}")
            if row["description"]:
                lines.append(f"- 描述：{row['description']}")
            lines.append("")

        edges = self.db.list_edges()
        lines.append(f"## 关系（{len(edges)}）")
        lines.append("")
        for row in edges:
            src = self.db.get_node_by_id(row["source_id"])
            tgt = self.db.get_node_by_id(row["target_id"])
            src_label = src["label"] if src else row["source_id"]
            tgt_label = tgt["label"] if tgt else row["target_id"]
            lines.append(f"- **{src_label}** → *{row['relation']}* → **{tgt_label}**")
        lines.append("")

        lines.append(f"## 碎片（{len(fragments)}）")
        lines.append("")
        for row in fragments:
            tags = row["tags"]
            tag_str = f" `[{tags}]`" if tags else ""
            lines.append(f"- {row['content']}{tag_str}")
        return "\n".join(lines)

    def export_gexf(self) -> str:
        G = self.build_networkx()
        return "\n".join(nx.generate_gexf(G))
