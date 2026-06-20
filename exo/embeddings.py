"""Embedding 封装。"""
from __future__ import annotations

import json
import math
import random
from typing import Any

import httpx
import numpy as np

from exo.config import AppConfig


class Embedder:
    def __init__(self, config: AppConfig):
        self.config = config.embedding
        self.provider = self.config.provider.lower()
        self.mock = self.provider == "mock" or not self.config.api_key
        self.base_url = self.config.base_url or "https://api.openai.com/v1"
        self.api_key = self.config.api_key
        self.dimensions = self.config.dimensions
        self._local_model: Any = None
        self.client = httpx.AsyncClient(timeout=60)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "openai" and self.api_key:
            return await self._openai_embed(texts)
        if self.provider == "sentence_transformers":
            return self._local_embed(texts)
        return self._mock_embed(texts)

    async def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.config.model,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = await self.client.post(
            f"{self.base_url}/embeddings", headers=headers, json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        results = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in results]

    def _local_embed(self, texts: list[str]) -> list[list[float]]:
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "未安装 sentence-transformers，请执行 pip install sentence-transformers"
                ) from exc
            self._local_model = SentenceTransformer(self.config.model)
        vectors = self._local_model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def _mock_embed(self, texts: list[str]) -> list[list[float]]:
        """基于文本哈希生成确定性伪 embedding，便于无 Key 测试。"""
        vectors = []
        for text in texts:
            seed = hash(text) % (2**32)
            rng = random.Random(seed)
            vec = [rng.uniform(-1, 1) for _ in range(self.dimensions)]
            norm = math.sqrt(sum(x * x for x in vec))
            vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors

    async def close(self) -> None:
        await self.client.aclose()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)


def serialize_vector(vector: list[float]) -> bytes:
    return json.dumps(vector, ensure_ascii=False).encode("utf-8")


def deserialize_vector(blob: bytes) -> list[float]:
    return json.loads(blob.decode("utf-8"))
