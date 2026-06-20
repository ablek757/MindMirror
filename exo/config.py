"""配置加载模块。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 加载项目根目录 .env 文件（如果存在）
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 60


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = ""
    dimensions: int = 1536


class DBConfig(BaseModel):
    path: str = "data/exo.db"


class GraphConfig(BaseModel):
    auto_weave: bool = False
    similarity_threshold: float = 0.82
    min_edge_weight: float = 1.0
    max_nodes_per_fragment: int = 8


class ChatConfig(BaseModel):
    system_role: str = ""
    modes: dict[str, str] = Field(default_factory=dict)
    max_context_fragments: int = 6
    max_context_nodes: int = 6


class AppConfig(BaseModel):
    app: dict[str, Any] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """从 YAML 文件加载配置，并用环境变量覆盖关键字段。"""
    path = Path(path)
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    config = AppConfig(**raw)

    # 环境变量覆盖
    config.llm.api_key = os.getenv("OPENAI_API_KEY", config.llm.api_key)
    config.llm.base_url = os.getenv("OPENAI_BASE_URL", config.llm.base_url)
    config.llm.model = os.getenv("OPENAI_MODEL", config.llm.model)
    if os.getenv("EXO_MOCK_LLM"):
        config.llm.provider = "mock"
        config.embedding.provider = "mock"

    config.embedding.api_key = os.getenv("OPENAI_API_KEY", config.embedding.api_key)
    config.embedding.base_url = os.getenv("OPENAI_BASE_URL", config.embedding.base_url)

    # 确保数据目录存在
    db_path = Path(config.db.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return config
