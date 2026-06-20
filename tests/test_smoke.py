"""无 LLM 冒烟测试。"""
from __future__ import annotations

import asyncio
import gc
import os
import sys
from pathlib import Path

# 确保能导入项目根目录下的 exo 包
sys.path.insert(0, str(Path(__file__).parent.parent))

# 强制使用 mock 模式
os.environ["EXO_MOCK_LLM"] = "1"

from exo.chat import ChatEngine
from exo.config import load_config
from exo.db import Database
from exo.embeddings import Embedder
from exo.graph import GraphService
from exo.ingest import IngestService
from exo.llm import LLMClient
from exo.memory import MemoryService
from exo.weaver import Weaver


def test_full_flow() -> None:
    db_path = Path(__file__).parent.parent / "data" / "test_exo.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    # 创建临时配置
    config = load_config()
    config.db.path = str(db_path)
    config.llm.provider = "mock"
    config.embedding.provider = "mock"

    db = Database(config)
    embedder = Embedder(config)
    llm = LLMClient(config)

    try:
        asyncio.run(_run_flow(config, db, embedder, llm))
    finally:
        asyncio.run(embedder.close())
        asyncio.run(llm.close())
        del db
        gc.collect()
        if db_path.exists():
            db_path.unlink()


async def _run_flow(config, db, embedder, llm) -> None:
    ingest_service = IngestService(config, db, embedder)

    # 1. 摄入碎片
    f1 = await ingest_service.add(
        "知识的复利来自连接，而不是堆积。", tags=["阅读", "认知"]
    )
    f2 = await ingest_service.add(
        "想做一个把日常碎片自动编织成知识图谱的工具。", tags=["项目", "AI"]
    )
    assert f1.id is not None
    assert f2.id is not None

    # 2. 编织
    weaver = Weaver(config, db, embedder, llm)
    results = await weaver.weave_all()
    assert len(results) == 2
    total_nodes = sum(len(r.nodes) for r in results)
    assert total_nodes > 0

    # 3. 图统计
    graph_service = GraphService(db)
    stats = graph_service.stats()
    assert stats["fragments"] == 2
    assert stats["nodes"] > 0
    assert stats["unwoven_fragments"] == 0

    # 4. 导出
    md = graph_service.export_markdown()
    assert "个人认知外脑图谱" in md
    json_data = graph_service.export_graph()
    assert "nodes" in json_data and "edges" in json_data

    # 5. 语义记忆
    memory = MemoryService(config, db, embedder)
    hits = await memory.search_fragments("知识图谱", top_k=3)
    assert len(hits) > 0

    # 6. 对话
    chat_engine = ChatEngine(config, db, llm, embedder)
    response = await chat_engine.respond_text("test-session", "请介绍这个项目")
    assert "mock-mode" in response

    print("[OK] 所有冒烟测试通过")


if __name__ == "__main__":
    test_full_flow()
