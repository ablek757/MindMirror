"""LLM 调用封装。"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import httpx

from exo.config import AppConfig


class LLMClient:
    def __init__(self, config: AppConfig):
        self.config = config.llm
        self.embedding_config = config.embedding
        self.provider = self.config.provider.lower()
        self.mock = self.provider == "mock" or not self.config.api_key
        self.base_url = self.config.base_url or "https://api.openai.com/v1"
        self.api_key = self.config.api_key
        self.client = httpx.AsyncClient(timeout=self.config.timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """非流式返回完整字符串；流式返回 token 迭代器。"""
        if self.mock:
            if stream:
                return self._mock_stream(messages)
            return self._mock_complete(messages)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens

        if stream:
            payload["stream"] = True
            return self._stream(payload)

        resp = await self.client.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def _stream(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST", f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def _mock_complete(self, messages: list[dict[str, str]]) -> str:
        """无 LLM 时的降级输出，便于测试与演示。"""
        system_msg = ""
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
            if m.get("role") == "system":
                system_msg = m.get("content", "")

        # 如果系统提示要求 JSON，返回一个基于关键词的模拟抽取
        if "json" in system_msg.lower():
            return self._mock_json_extraction(user_msg)

        return (
            "[mock-mode] 这是一个模拟回复。当前没有配置 LLM API Key，"
            f"所以无法基于你的输入「{user_msg[:40]}...」做真实推理。"
            "请配置 OPENAI_API_KEY 以启用完整能力。"
        )

    def _mock_json_extraction(self, text: str) -> str:
        import re

        # mock 模式下做简单关键词抽取：连续中文字符 + 英文单词
        seen = set()
        candidates: list[str] = []
        # 抽取 4-8 个连续中文字符作为短语
        for phrase in re.findall(r"[\u4e00-\u9fa5]{4,8}", text):
            # 取前 4 个字作为概念，避免过碎
            w = phrase[:4]
            if w not in seen:
                seen.add(w)
                candidates.append(w)
        # 英文单词
        for w in re.findall(r"[a-zA-Z]{4,}", text):
            if w not in seen:
                seen.add(w)
                candidates.append(w)

        candidates = candidates[:8]
        nodes = [
            {"label": w, "node_type": "concept", "description": ""}
            for w in candidates
        ]
        edges = []
        for i in range(len(candidates) - 1):
            edges.append(
                {
                    "source": candidates[i],
                    "target": candidates[i + 1],
                    "relation": "关联",
                }
            )
        return json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)

    async def _mock_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        text = self._mock_complete(messages)
        for word in text:
            yield word

    async def close(self) -> None:
        await self.client.aclose()


def get_default_prompt() -> str:
    return (
        "你是一个帮助用户整理思维碎片的助手。请从用户提供的文本中提取关键实体、概念、问题或项目，"
        "并识别它们之间的关系。输出为 JSON 格式："
        '{"nodes":[{"label":"名称","node_type":"concept|person|project|question|idea","description":"简短描述"}],'
        '"edges":[{"source":"源节点label","target":"目标节点label","relation":"关系名称"}]}'
    )


def build_messages(
    system_prompt: str, user_content: str, history: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages
