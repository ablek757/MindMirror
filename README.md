# 个人认知外脑（Personal Cognitive Exo-Brain）

一个帮你把日常碎片信息自动编织成会生长的个人思维图谱，并在你需要时以对话方式陪你思考、质疑和创作的 AI 第二大脑。

> 与 `deep-research-agent`（外向型研究助理）互补：本项目是**内向型个人知识花园**。

## 核心能力

- **碎片沉淀**：一句话想法、阅读摘录、剪贴内容随手录入，自动带上时间戳与标签。
- **认知编织**：AI 自动提取实体、概念、问题与关系，把碎片织进知识图谱。
- **图谱生长**：节点与关系不断累积、合并、演化，形成可视化的个人思维网络。
- **语义记忆**：基于 embedding 的相似度检索，帮你发现“很久以前好像想过类似的事”。
- **对话思考**：以认知伙伴身份陪你澄清、质疑、发散、收敛，支持 `/think`、`/challenge`、`/create` 三种模式。
- **本地优先**：所有数据保存在本地 SQLite，不上传云端。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 LLM（可选但推荐）

复制示例环境变量并填入你的 Key：

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
```

或临时设置：

```bash
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

> 没有 Key 也能运行：设置 `EXO_MOCK_LLM=1` 即可使用 mock 模式与本地 embedding，适合体验与测试。

### 3. CLI 使用

```bash
# 摄入一个碎片
python main.py ingest "今天看到一段话：知识的复利来自连接，而不是堆积。" --tags 阅读,认知

# 再摄入几个
python main.py ingest "想做一个把日常碎片自动编织成知识图谱的工具" --tags 项目,AI
python main.py ingest "问题在于，碎片信息越多，越难找到它们之间的隐藏关联。" --tags 问题,认知

# 执行认知编织（把碎片变成图谱）
python main.py weave

# 查看图谱统计
python main.py graph stats

# 导出图谱为 Markdown
python main.py export --format markdown --output output/mind.md
```

### 4. 启动 Web 界面

```bash
python api.py
```

浏览器打开 http://localhost:8000 即可：
- 浏览可交互的知识图谱
- 查看碎片时间线
- 与认知伙伴对话

### 5. 冒烟测试（无需 LLM）

```bash
python tests/test_smoke.py
```

## 命令参考

| 命令 | 说明 |
|------|------|
| `python main.py ingest <text>` | 摄入碎片 |
| `python main.py weave` | 对未编织碎片执行认知编织 |
| `python main.py graph stats` | 查看图谱统计 |
| `python main.py export --format markdown/json/gexf` | 导出图谱 |
| `python main.py chat` | 启动交互式对话（命令行） |
| `python api.py` | 启动 FastAPI 服务与 Web 前端 |

## 项目结构

```
cognitive-exo-brain/
├── main.py              # CLI 入口
├── api.py               # FastAPI 服务
├── config.yaml          # 默认配置
├── requirements.txt     # Python 依赖
├── web/index.html       # 可视化前端
├── exo/                 # 核心包
│   ├── config.py        # 配置加载
│   ├── models.py        # 数据模型
│   ├── db.py            # SQLite 连接
│   ├── llm.py           # LLM 调用
│   ├── embeddings.py    # Embedding 封装
│   ├── ingest.py        # 碎片摄入
│   ├── weaver.py        # 认知编织
│   ├── graph.py         # 图存储
│   ├── memory.py        # 语义记忆
│   └── chat.py          # 对话引擎
└── tests/
    └── test_smoke.py    # 无 LLM 冒烟测试
```

## 配置说明

`config.yaml` 关键项：

| 配置项 | 说明 |
|--------|------|
| `llm.model` | 模型名 |
| `llm.base_url` | OpenAI 兼容接口地址 |
| `llm.temperature` | 生成温度 |
| `embedding.provider` | `openai` / `sentence_transformers` / `mock` |
| `embedding.model` | embedding 模型名 |
| `db.path` | SQLite 文件路径 |
| `graph.auto_weave` | 摄入后是否自动编织 |

## 许可证

MIT
