"""Asset Publisher: 3軸企画展開・5段階執筆・記事管理・エクスポート・サニタイズ・空白テーマ・シリーズ編纂。"""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from core.asset_publisher.exporter import SeriesCompiler, export_note_markdown, export_obsidian_markdown, export_series_markdown
from core.asset_publisher.matrix_transpiler import research_log_from_experiment, research_log_from_structure
from core.asset_publisher.sanitizer import ComplianceSanitizer
from core.asset_publisher.schemas import ContentAxis, ContentMatrix, ContentPlan, PublishedArticle, ResearchLog
from core.asset_publisher.writer_pipeline import WriterConfig
from web_studio.backend.api_models import ArticleRequest, MatrixRequest, SeriesRequest, TextRequest
from web_studio.backend.deps import dump, get_services, not_found, require_structure
from web_studio.backend.jobs import JobContext, run_or_submit
from web_studio.backend.services import FoundryServices

router = APIRouter(prefix="/api/publisher", tags=["publisher"])


def _research_log(services: FoundryServices, body: MatrixRequest) -> ResearchLog:
    provided = [value for value in (body.structure_id, body.research_log, body.experiment) if value is not None]
    if len(provided) != 1:
        raise HTTPException(status_code=400, detail="structure_id / research_log / experiment のいずれか1つを指定してください")
    if body.structure_id is not None:
        return research_log_from_structure(require_structure(services, body.structure_id))
    if body.research_log is not None:
        return body.research_log
    return research_log_from_experiment(body.experiment or {})


def _require_matrix(services: FoundryServices, matrix_id: str) -> ContentMatrix:
    matrix = services.article_repo.get_matrix(matrix_id)
    if matrix is None:
        raise not_found("企画マトリクス", matrix_id)
    return matrix


def _require_article(services: FoundryServices, article_id: str) -> PublishedArticle:
    article = services.article_repo.get(article_id)
    if article is None:
        raise not_found("記事", article_id)
    return article


@router.post("/matrix")
async def create_matrix(body: MatrixRequest, services: FoundryServices = Depends(get_services)) -> Any:
    log = _research_log(services, body)

    async def runner(ctx: JobContext) -> dict[str, Any]:
        await ctx.update(0.1, "matrix", "研究ログを3軸（創作理論・AI開発・ビジネス）へ展開しています")
        matrix = await services.matrix_transpiler().expand(log)
        if body.save:
            services.article_repo.save_matrix(matrix)
            services.research_logs.save(matrix.id, log)
        await ctx.update(0.95, "save", "企画マトリクスを保存しました")
        return matrix.model_dump(mode="json")

    return await run_or_submit(
        services.jobs, kind="publisher.matrix", run_async=body.run_async, runner=runner, params={"source": log.title}
    )


@router.get("/matrices")
def list_matrices(
    limit: int = Query(50, ge=1, le=500),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    return {"matrices": services.article_repo.list_matrices(limit=limit)}


@router.get("/matrices/{matrix_id}")
def get_matrix(matrix_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    return _require_matrix(services, matrix_id).model_dump(mode="json")


def _resolve_article_inputs(services: FoundryServices, body: ArticleRequest) -> tuple[ContentPlan, ResearchLog | None]:
    log: ResearchLog | None = body.research_log
    if body.plan is not None:
        plan = body.plan
    elif body.matrix_id is not None:
        matrix = _require_matrix(services, body.matrix_id)
        axis = body.axis or matrix.plans[0].axis
        candidates = [candidate for candidate in matrix.plans if candidate.axis == ContentAxis(axis)]
        if not candidates:
            raise HTTPException(status_code=400, detail=f"マトリクス {body.matrix_id} に軸 {ContentAxis(axis).value} の企画がありません")
        plan = candidates[0]
        if log is None:
            log = services.research_logs.get(body.matrix_id)
    else:
        raise HTTPException(status_code=400, detail="plan または matrix_id（＋axis）を指定してください")
    if log is None and body.structure_id is not None:
        log = research_log_from_structure(require_structure(services, body.structure_id))
    return plan, log


@router.post("/articles")
async def create_article(body: ArticleRequest, services: FoundryServices = Depends(get_services)) -> Any:
    plan, log = _resolve_article_inputs(services, body)
    config = WriterConfig.model_validate(body.config)

    async def runner(ctx: JobContext) -> dict[str, Any]:
        article = await services.writer(config).run(plan, log, on_progress=ctx.pipeline_progress)
        services.article_repo.save(article)
        await ctx.update(0.97, "save", "記事を保存しました")
        return article.model_dump(mode="json")

    return await run_or_submit(
        services.jobs,
        kind="publisher.article",
        run_async=body.run_async,
        runner=runner,
        params={"axis": plan.axis.value, "title": plan.title},
    )


@router.get("/articles")
def list_articles(
    axis: ContentAxis | None = None,
    source_id: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    repo = services.article_repo
    axis_value = axis.value if axis is not None else None
    items = repo.list(axis=axis_value, source_id=source_id, limit=limit, offset=offset)
    return {"items": dump(items), "total": repo.count(axis_value)}


@router.get("/articles/{article_id}")
def get_article(article_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    return _require_article(services, article_id).model_dump(mode="json")


@router.delete("/articles/{article_id}")
def delete_article(article_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    if not services.article_repo.delete(article_id):
        raise not_found("記事", article_id)
    return {"deleted": True, "id": article_id}


@router.get("/articles/{article_id}/export")
def export_article(
    article_id: str,
    format: Literal["note", "obsidian"] = "obsidian",  # noqa: A002 - API のクエリ名
    services: FoundryServices = Depends(get_services),
) -> Response:
    article = _require_article(services, article_id)
    if format == "note":
        text = export_note_markdown(article)
    else:
        tags = set(article.tags)
        related = [
            other.title
            for other in services.article_repo.all()
            if other.id != article.id and tags.intersection(other.tags)
        ][:8]
        text = export_obsidian_markdown(article, related)
    filename = f"{article.id}_{format}.md"
    return Response(
        content=text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


@router.post("/sanitize")
def sanitize(body: TextRequest) -> dict[str, Any]:
    text, report = ComplianceSanitizer().sanitize(body.text)
    return {"text": text, "report": report.model_dump(mode="json")}


@router.post("/verify")
def verify(body: TextRequest) -> dict[str, Any]:
    return ComplianceSanitizer().verify(body.text).model_dump(mode="json")


@router.get("/gaps")
def content_gaps(
    top_k: int = Query(5, ge=1, le=30),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    graph = services.knowledge_graph()
    gaps = graph.detect_gaps(top_k=top_k)
    return {"gaps": dump(gaps), "stats": graph.stats(), "mermaid": graph.to_mermaid()}


@router.post("/series")
def compile_series(body: SeriesRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    articles = [_require_article(services, article_id) for article_id in body.article_ids]
    volume, markdown = SeriesCompiler().compile(articles, body.title, body.description)
    if body.format == "note":
        markdown = export_series_markdown(volume, articles, "note")
    return {"volume": volume.model_dump(mode="json"), "markdown": markdown, "format": body.format}
