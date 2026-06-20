"""CLI 入口。"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import click

from exo.chat import ChatEngine
from exo.config import AppConfig, load_config
from exo.db import Database
from exo.embeddings import Embedder
from exo.graph import GraphService
from exo.ingest import IngestService
from exo.llm import LLMClient
from exo.weaver import Weaver


@click.group()
@click.option("--config", "config_path", default="config.yaml", help="配置文件路径")
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """个人认知外脑 CLI。"""
    ctx.ensure_object(dict)
    config = load_config(config_path)
    ctx.obj["config"] = config
    ctx.obj["db"] = Database(config)
    ctx.obj["embedder"] = Embedder(config)
    ctx.obj["llm"] = LLMClient(config)


@cli.command()
@click.argument("content")
@click.option("--source", default="manual", help="碎片来源")
@click.option("--tags", default="", help="逗号分隔标签")
@click.pass_context
def ingest(ctx: click.Context, content: str, source: str, tags: str) -> None:
    """摄入一个思维碎片。"""
    config: AppConfig = ctx.obj["config"]
    db: Database = ctx.obj["db"]
    embedder: Embedder = ctx.obj["embedder"]
    service = IngestService(config, db, embedder)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    async def _run() -> None:
        fragment = await service.add(content, source=source, tags=tag_list)
        click.echo(f"✓ 已摄入碎片 #{fragment.id}")
        if config.graph.auto_weave:
            click.echo("  已自动编织到知识图谱。")

    asyncio.run(_run())


@cli.command()
@click.option("--limit", default=20, help="最多显示条数")
@click.pass_context
def fragments(ctx: click.Context, limit: int) -> None:
    """列出最近摄入的碎片。"""
    db: Database = ctx.obj["db"]
    service = IngestService(ctx.obj["config"], db, ctx.obj["embedder"])
    rows = service.list_fragments(limit)
    for f in rows:
        status = "✓" if f.woven_at else "○"
        tags = f" [{', '.join(f.tags)}]" if f.tags else ""
        click.echo(f"{status} #{f.id}{tags} {f.content[:80]}{'...' if len(f.content) > 80 else ''}")


@cli.command()
@click.pass_context
def weave(ctx: click.Context) -> None:
    """对未编织的碎片执行认知编织。"""
    config: AppConfig = ctx.obj["config"]
    db: Database = ctx.obj["db"]
    embedder: Embedder = ctx.obj["embedder"]
    llm: LLMClient = ctx.obj["llm"]
    weaver = Weaver(config, db, embedder, llm)

    async def _run() -> None:
        results = await weaver.weave_all()
        total_nodes = sum(len(r.nodes) for r in results)
        total_edges = sum(len(r.edges) for r in results)
        click.echo(f"✓ 编织完成：处理 {len(results)} 个碎片，生成 {total_nodes} 个节点，{total_edges} 条关系。")

    asyncio.run(_run())


@cli.group()
@click.pass_context
def graph(ctx: click.Context) -> None:
    """知识图谱操作。"""
    ctx.obj["graph"] = GraphService(ctx.obj["db"])


@graph.command(name="stats")
@click.pass_context
def graph_stats(ctx: click.Context) -> None:
    """查看图谱统计。"""
    service: GraphService = ctx.obj["graph"]
    stats = service.stats()
    click.echo("知识图谱统计：")
    for key, value in stats.items():
        click.echo(f"  {key}: {value}")


@cli.command()
@click.option("--format", "fmt", default="markdown", type=click.Choice(["markdown", "json", "gexf"]))
@click.option("--output", default=None, help="输出文件路径")
@click.pass_context
def export(ctx: click.Context, fmt: str, output: str | None) -> None:
    """导出知识图谱。"""
    service = GraphService(ctx.obj["db"])
    if fmt == "markdown":
        text = service.export_markdown()
    elif fmt == "json":
        import json
        data = service.export_graph()
        text = json.dumps(data, ensure_ascii=False, indent=2)
    elif fmt == "gexf":
        text = service.export_gexf()
    else:
        raise click.BadParameter(f"不支持的格式：{fmt}")

    if output:
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"✓ 已导出到 {output}")
    else:
        click.echo(text)


@cli.command()
@click.option("--session", default=None, help="会话 ID")
@click.pass_context
def chat(ctx: click.Context, session: str | None) -> None:
    """启动交互式对话。"""
    config: AppConfig = ctx.obj["config"]
    db: Database = ctx.obj["db"]
    llm: LLMClient = ctx.obj["llm"]
    embedder: Embedder = ctx.obj["embedder"]
    engine = ChatEngine(config, db, llm, embedder)
    session_id = session or str(uuid.uuid4())[:8]
    click.echo("进入认知对话模式。输入 /quit 退出，/think /challenge /create 切换模式。")
    click.echo(f"会话 ID: {session_id}")

    async def _run() -> None:
        current_mode = "default"
        while True:
            try:
                user_input = click.prompt("你", type=str)
            except click.exceptions.Abort:
                break
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input == "/quit":
                break
            if user_input in ("/think", "/challenge", "/create"):
                current_mode = user_input.lstrip("/")
                click.echo(f"  已切换到 {current_mode} 模式")
                continue

            click.echo("认知伙伴：", nl=False)
            async for token in engine.respond(session_id, user_input, mode=current_mode):
                click.echo(token, nl=False)
            click.echo()

    try:
        asyncio.run(_run())
    finally:
        asyncio.run(llm.close())
        asyncio.run(embedder.close())


if __name__ == "__main__":
    cli()
