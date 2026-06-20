"""FastAPI 服务入口。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from exo.chat import ChatEngine
from exo.config import load_config
from exo.db import Database
from exo.embeddings import Embedder
from exo.graph import GraphService
from exo.ingest import IngestService
from exo.llm import LLMClient
from exo.memory import MemoryService
from exo.models import ChatRequest, Fragment
from exo.weaver import Weaver


config = load_config()
db = Database(config)
embedder = Embedder(config)
llm = LLMClient(config)
ingest_service = IngestService(config, db, embedder)
graph_service = GraphService(db)
memory_service = MemoryService(config, db, embedder)
chat_engine = ChatEngine(config, db, llm, embedder)
weaver = Weaver(config, db, embedder, llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await llm.close()
    await embedder.close()


app = FastAPI(title="个人认知外脑", version="0.1.0", lifespan=lifespan)

# 静态前端
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(web_dir / "index.html"))


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    return graph_service.stats()


@app.post("/api/fragments")
async def create_fragment(request: Request) -> dict[str, Any]:
    data = await request.json()
    content = data.get("content", "").strip()
    source = data.get("source", "web")
    tags = [t.strip() for t in data.get("tags", "").split(",") if t.strip()]
    if not content:
        return {"error": "content is required"}
    fragment = await ingest_service.add(content, source=source, tags=tags)
    return {"id": fragment.id, "content": fragment.content, "tags": fragment.tags}


@app.get("/api/fragments")
async def list_fragments(limit: int = 100) -> list[dict[str, Any]]:
    rows = ingest_service.list_fragments(limit)
    return [
        {
            "id": f.id,
            "content": f.content,
            "source": f.source,
            "tags": f.tags,
            "created_at": f.created_at,
            "woven_at": f.woven_at,
        }
        for f in rows
    ]


@app.post("/api/weave")
async def weave() -> dict[str, Any]:
    results = await weaver.weave_all()
    total_nodes = sum(len(r.nodes) for r in results)
    total_edges = sum(len(r.edges) for r in results)
    return {
        "woven": len(results),
        "nodes_added": total_nodes,
        "edges_added": total_edges,
    }


@app.get("/api/graph")
async def graph_data() -> dict[str, Any]:
    return graph_service.export_graph()


@app.get("/api/nodes/{node_id}")
async def node_detail(node_id: int) -> dict[str, Any]:
    node = db.get_node_by_id(node_id)
    if not node:
        return {"error": "node not found"}
    fragments = db.get_node_fragments(node_id)
    return {
        "id": node["id"],
        "label": node["label"],
        "type": node["node_type"],
        "description": node["description"],
        "fragments": [
            {"id": f["id"], "content": f["content"], "created_at": f["created_at"]}
            for f in fragments
        ],
    }


@app.get("/api/memory/search")
async def memory_search(q: str, top_k: int = 5) -> dict[str, Any]:
    fragment_hits = await memory_service.search_fragments(q, top_k)
    node_hits = await memory_service.search_nodes(q, top_k)
    return {
        "fragments": [
            {"id": f.id, "content": f.content, "score": round(score, 3)}
            for f, score in fragment_hits
        ],
        "nodes": [
            {"id": n.id, "label": n.label, "type": n.node_type, "score": round(score, 3)}
            for n, score in node_hits
        ],
    }


@app.get("/api/memory/connections")
async def hidden_connections(top_k: int = 5) -> dict[str, Any]:
    pairs = await memory_service.hidden_connections(top_k)
    return {
        "connections": [
            {
                "a": {"id": a.id, "label": a.label, "type": a.node_type},
                "b": {"id": b.id, "label": b.label, "type": b.node_type},
                "score": round(score, 3),
            }
            for a, b, score in pairs
        ]
    }


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    session_id = req.session_id or str(uuid4())[:8]

    async def event_stream() -> AsyncIterator[str]:
        async for token in chat_engine.respond(session_id, req.message, req.mode):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/export/markdown")
async def export_markdown() -> dict[str, str]:
    return {"content": graph_service.export_markdown()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
