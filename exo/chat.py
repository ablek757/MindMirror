"""对话引擎：基于图谱与语义记忆的认知伙伴。"""
from __future__ import annotations

from typing import AsyncIterator

from exo.config import AppConfig
from exo.db import Database
from exo.embeddings import Embedder
from exo.llm import LLMClient, build_messages
from exo.memory import MemoryService
from exo.models import ChatMessage


class ChatEngine:
    def __init__(
        self,
        config: AppConfig,
        db: Database,
        llm: LLMClient,
        embedder: Embedder,
    ):
        self.config = config
        self.db = db
        self.llm = llm
        self.embedder = embedder
        self.memory = MemoryService(config, db, embedder)

    async def respond(
        self, session_id: str, message: str, mode: str = "default"
    ) -> AsyncIterator[str]:
        # 1. 检索相关记忆
        fragment_hits = await self.memory.search_fragments(
            message, top_k=self.config.chat.max_context_fragments
        )
        node_hits = await self.memory.search_nodes(
            message, top_k=self.config.chat.max_context_nodes
        )

        context_parts: list[str] = []
        if fragment_hits:
            context_parts.append("相关记忆碎片：")
            for frag, score in fragment_hits:
                context_parts.append(f"- [{score:.2f}] {frag.content}")
        if node_hits:
            context_parts.append("相关图谱节点：")
            for node, score in node_hits:
                desc = f"（{node.description}）" if node.description else ""
                context_parts.append(f"- [{score:.2f}] {node.label} {desc}")

        if not context_parts:
            context_parts.append("（当前知识库中暂未找到直接相关的记忆或节点）")

        context = "\n".join(context_parts)

        # 2. 构建系统提示
        system_role = self.config.chat.system_role
        mode_desc = self.config.chat.modes.get(mode, "")
        system_prompt = f"{system_role}\n\n当前模式：{mode}\n{mode_desc}".strip()

        # 3. 加入最近对话历史
        history_rows = self.db.get_conversation_history(session_id, limit=10)
        history_rows.reverse()
        history: list[dict[str, str]] = []
        for row in history_rows:
            history.append({"role": row["role"], "content": row["content"]})

        user_content = f"{context}\n\n用户当前输入：{message}".strip()
        messages = build_messages(system_prompt, user_content, history)

        # 4. 流式生成并保存
        self.db.add_conversation_message(session_id, "user", message, mode)

        full_response = ""
        stream = await self.llm.complete(messages, stream=True)
        if isinstance(stream, str):
            # 非流式兼容
            full_response = stream
            yield stream
        else:
            async for token in stream:
                full_response += token
                yield token

        self.db.add_conversation_message(session_id, "assistant", full_response, mode)

    async def respond_text(self, session_id: str, message: str, mode: str = "default") -> str:
        parts = []
        async for token in self.respond(session_id, message, mode):
            parts.append(token)
        return "".join(parts)
