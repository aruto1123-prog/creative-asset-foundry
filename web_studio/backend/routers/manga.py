"""Manga Studio: ネーム生成（文法検証ループ込み）・検証・自動修正・SVG・レイヤー・ComfyUI コンパイル／投入。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from core.common import new_id
from core.manga_pipeline.comfy_client import ComfyProgressEvent
from core.manga_pipeline.comfy_compiler import CompiledPanelWorkflow, CompilerConfig, ComfyWorkflowCompiler
from core.manga_pipeline.grammar_validator import MangaGrammarValidator
from core.manga_pipeline.layout_renderer import RenderOptions, SVGLayoutRenderer
from core.manga_pipeline.schemas import GrammarReport, Storyboard
from core.manga_pipeline.storyboard_generator import StoryboardConfig, StoryboardFixer
from core.structure_engine.schemas import UniversalStructure
from core.style_alchemy.model_optimizer import RecipeExporter
from core.style_alchemy.schemas import CheckpointRef, ModelArchitecture, StyleRecipe
from web_studio.backend.api_models import ComfyCompileRequest, ComfyQueueRequest, StoryboardCreateRequest
from web_studio.backend.deps import dump, get_services, not_found, require_structure
from web_studio.backend.jobs import JobContext, run_or_submit
from web_studio.backend.services import FoundryServices

router = APIRouter(prefix="/api/manga", tags=["manga"])

DEFAULT_CHECKPOINTS: dict[ModelArchitecture, str] = {
    ModelArchitecture.SDXL: "sd_xl_base_1.0.safetensors",
    ModelArchitecture.SD15: "v1-5-pruned-emaonly.safetensors",
    ModelArchitecture.FLUX: "flux1-dev.safetensors",
    ModelArchitecture.ANIMA: "anima.safetensors",
}
DEFAULT_STYLE_TOKENS = ["monochrome", "manga", "lineart", "screentone", "high contrast ink"]
DEFAULT_NEGATIVE_TOKENS = ["color", "lowres", "blurry", "watermark", "signature", "text"]


def default_recipe(architecture: ModelArchitecture) -> StyleRecipe:
    """レシピ未指定時の既定（モノクロ漫画原稿向け）レシピ。"""
    return StyleRecipe(
        name=f"既定 {architecture.value.upper()} 漫画レシピ",
        architecture=architecture,
        base_checkpoint=CheckpointRef(name=DEFAULT_CHECKPOINTS[architecture], architecture=architecture),
        style_tokens=list(DEFAULT_STYLE_TOKENS),
        negative_tokens=list(DEFAULT_NEGATIVE_TOKENS),
    )


def resolve_recipe(body: ComfyCompileRequest) -> StyleRecipe:
    """リクエストからレシピを決定する（明示レシピ → 最適化パラメータ → 既定）。"""
    architecture = body.architecture
    if body.recipe is not None:
        return body.recipe
    if body.recipe_params is not None:
        model_a = body.model_a or CheckpointRef(name=DEFAULT_CHECKPOINTS[architecture], architecture=architecture)
        model_b = body.model_b or CheckpointRef(name=f"caf_style_{architecture.value}.safetensors", architecture=architecture)
        return RecipeExporter().params_to_recipe(
            body.recipe_params,
            architecture=architecture,
            model_a=model_a,
            model_b=model_b,
            loras=body.loras,
            name="最適化パラメータ由来レシピ",
        )
    recipe = default_recipe(architecture)
    updates: dict[str, Any] = {}
    if body.model_a is not None:
        updates["base_checkpoint"] = body.model_a.model_copy(update={"architecture": architecture})
    if body.loras:
        updates["loras"] = list(body.loras)
    return StyleRecipe.model_validate({**recipe.model_dump(mode="json"), **dump(updates)}) if updates else recipe


def _load(services: FoundryServices, storyboard_id: str) -> tuple[Storyboard, StoryboardConfig]:
    loaded = services.storyboards.get(storyboard_id)
    if loaded is None:
        raise not_found("ネーム", storyboard_id)
    return loaded


def _structure_of(services: FoundryServices, storyboard: Storyboard) -> UniversalStructure | None:
    return services.structure_repo.get(storyboard.structure_id)


def _compile(services: FoundryServices, storyboard: Storyboard, body: ComfyCompileRequest, limit: int | None = None) -> tuple[StyleRecipe, list[CompiledPanelWorkflow]]:
    recipe = resolve_recipe(body)
    config = CompilerConfig(base_resolution=body.base_resolution) if body.base_resolution else CompilerConfig()
    config = config.model_copy(update={"filename_prefix": services.settings.comfy.output_prefix})
    compiler = ComfyWorkflowCompiler(recipe, config)
    if body.panel_ids:
        panels = []
        for panel_id in body.panel_ids:
            panel = storyboard.panel(panel_id)
            if panel is None:
                raise not_found("コマ", panel_id)
            panels.append(panel)
    else:
        panels = storyboard.panels_in_reading_order()
    if limit is not None:
        panels = panels[:limit]
    return recipe, [compiler.compile_panel(storyboard, panel) for panel in panels]


def _fix_loop(storyboard: Storyboard, config: StoryboardConfig, structure: UniversalStructure | None) -> Storyboard:
    """検証 → 規則別修復を反復し、未合格なら安全レイアウトで再構成する。"""
    validator = MangaGrammarValidator(config)
    fixer = StoryboardFixer(config)
    current = storyboard
    report = validator.validate(current, structure)
    best, best_report = current, report
    iteration = 0
    while iteration < config.max_fix_iterations and not report.passed:
        iteration += 1
        candidate = fixer.apply(current, report, structure)
        candidate_report = validator.validate(candidate, structure)
        candidate_report.iteration = iteration
        candidate.revision_log.append(f"自動修正 {iteration}: スコア {candidate_report.score:.1f}")
        current, report = candidate, candidate_report
        if (candidate_report.passed, candidate_report.score) > (best_report.passed, best_report.score):
            best, best_report = candidate, candidate_report
    current, report = best, best_report
    if not report.passed:
        current = fixer.safe_rebuild(current, structure)
        report = validator.validate(current, structure)
        if not report.passed:
            current = fixer.safe_rebuild(current, structure, aggressive=True)
            report = validator.validate(current, structure)
        current.revision_log.append(f"安全レイアウト再構成: スコア {report.score:.1f}")
    current.grammar = report
    return Storyboard.model_validate(current.model_dump())


def _summary(storyboard: Storyboard) -> dict[str, Any]:
    grammar: GrammarReport | None = storyboard.grammar
    return {
        "id": storyboard.id,
        "title": storyboard.title,
        "page_count": storyboard.page_count,
        "panel_count": len(storyboard.panels_in_reading_order()),
        "score": None if grammar is None else grammar.score,
        "passed": None if grammar is None else grammar.passed,
    }


@router.post("/storyboards")
async def create_storyboard(body: StoryboardCreateRequest, services: FoundryServices = Depends(get_services)) -> Any:
    structure = require_structure(services, body.structure_id)
    config = StoryboardConfig.model_validate(body.config)

    async def runner(ctx: JobContext) -> dict[str, Any]:
        await ctx.update(0.1, "storyboard", "構造からページ割り・コマ割り・カメラ・台詞を展開しています")
        storyboard = await services.storyboard_generator(config).generate_validated(structure, body.theme)
        await ctx.update(0.85, "grammar", "漫画文法バリデータの合格を確認しました")
        if body.title:
            storyboard.title = body.title
        if services.storyboards.exists(storyboard.id):
            storyboard = storyboard.model_copy(update={"id": new_id("sb")})
        services.storyboards.save(storyboard, config)
        await ctx.update(0.95, "save", "ネームを保存しました")
        return storyboard.model_dump(mode="json")

    return await run_or_submit(
        services.jobs,
        kind="manga.storyboard",
        run_async=body.run_async,
        runner=runner,
        params={"structure_id": body.structure_id, "config": config.model_dump(mode="json")},
    )


@router.get("/storyboards")
def list_storyboards(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    return {"items": dump(services.storyboards.list(limit=limit, offset=offset)), "total": services.storyboards.count()}


@router.get("/storyboards/{storyboard_id}")
def get_storyboard(storyboard_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    storyboard, _config = _load(services, storyboard_id)
    return storyboard.model_dump(mode="json")


@router.get("/storyboards/{storyboard_id}/config")
def get_storyboard_config(storyboard_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    _storyboard, config = _load(services, storyboard_id)
    return config.model_dump(mode="json")


@router.delete("/storyboards/{storyboard_id}")
def delete_storyboard(storyboard_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    if not services.storyboards.delete(storyboard_id):
        raise not_found("ネーム", storyboard_id)
    return {"deleted": True, "id": storyboard_id}


@router.post("/storyboards/{storyboard_id}/validate")
def validate_storyboard(storyboard_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    storyboard, config = _load(services, storyboard_id)
    report = MangaGrammarValidator(config).validate(storyboard, _structure_of(services, storyboard))
    updated = storyboard.model_copy(update={"grammar": report})
    services.storyboards.save(updated, config)
    return report.model_dump(mode="json")


@router.post("/storyboards/{storyboard_id}/autofix")
def autofix_storyboard(storyboard_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    storyboard, config = _load(services, storyboard_id)
    fixed = _fix_loop(storyboard, config, _structure_of(services, storyboard))
    services.storyboards.save(fixed, config)
    return fixed.model_dump(mode="json")


def _renderer(gaze: bool, labels: bool, guides: bool, sketch: bool) -> SVGLayoutRenderer:
    return SVGLayoutRenderer(
        RenderOptions(show_gaze_vectors=gaze, show_panel_labels=labels, show_bleed_guides=guides, draft_sketch=sketch)
    )


def _require_page(storyboard: Storyboard, page_index: int) -> None:
    if not 0 <= page_index < storyboard.page_count:
        raise HTTPException(status_code=404, detail=f"ページ {page_index} は存在しません（0〜{storyboard.page_count - 1}）")


@router.get("/storyboards/{storyboard_id}/pages/{page_index}/svg")
def page_svg(
    storyboard_id: str,
    page_index: int,
    gaze: bool = False,
    labels: bool = True,
    guides: bool = True,
    sketch: bool = True,
    services: FoundryServices = Depends(get_services),
) -> Response:
    storyboard, _config = _load(services, storyboard_id)
    _require_page(storyboard, page_index)
    svg = _renderer(gaze, labels, guides, sketch).render_page(storyboard, page_index)
    return Response(content=svg, media_type="image/svg+xml")


@router.get("/storyboards/{storyboard_id}/spreads/{right_index}/{left_index}/svg")
def spread_svg(
    storyboard_id: str,
    right_index: int,
    left_index: int,
    gaze: bool = False,
    labels: bool = True,
    guides: bool = True,
    sketch: bool = True,
    services: FoundryServices = Depends(get_services),
) -> Response:
    storyboard, _config = _load(services, storyboard_id)
    _require_page(storyboard, right_index)
    _require_page(storyboard, left_index)
    svg = _renderer(gaze, labels, guides, sketch).render_spread(storyboard, right_index, left_index)
    return Response(content=svg, media_type="image/svg+xml")


@router.get("/storyboards/{storyboard_id}/layers")
def layers(storyboard_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    storyboard, _config = _load(services, storyboard_id)
    return SVGLayoutRenderer().layers(storyboard)


@router.post("/storyboards/{storyboard_id}/comfy")
def compile_comfy(
    storyboard_id: str,
    body: ComfyCompileRequest | None = None,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    storyboard, _config = _load(services, storyboard_id)
    recipe, workflows = _compile(services, storyboard, body or ComfyCompileRequest())
    return {"storyboard": _summary(storyboard), "recipe": dump(recipe), "workflows": dump(workflows), "count": len(workflows)}


@router.post("/storyboards/{storyboard_id}/comfy/queue")
async def queue_comfy(
    storyboard_id: str,
    body: ComfyQueueRequest | None = None,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    request = body or ComfyQueueRequest()
    storyboard, _config = _load(services, storyboard_id)
    recipe, workflows = _compile(services, storyboard, request, limit=request.max_panels)
    total = len(workflows)

    async def runner(ctx: JobContext) -> dict[str, Any]:
        base_url = await services.comfy_base_url()
        client = services.comfy_client(base_url)
        results: list[dict[str, Any]] = []
        for index, workflow in enumerate(workflows):
            stage = f"panel {index + 1}/{total}"

            async def on_progress(event: ComfyProgressEvent, index: int = index, stage: str = stage) -> None:
                fraction = event.progress if event.type == "progress" else 0.0
                await ctx.update((index + min(1.0, max(0.0, fraction))) / max(1, total), stage, f"{workflow.panel_id}: {event.type}")

            await ctx.update(index / max(1, total), stage, f"{workflow.panel_id} をキューへ投入しています")
            result = await client.run_workflow(workflow.prompt, on_progress=on_progress)
            results.append({"panel_id": workflow.panel_id, "page_index": workflow.page_index, **result.model_dump(mode="json")})
        succeeded = sum(1 for item in results if item["status"] == "success")
        return {
            "base_url": base_url,
            "mock": services.settings.comfy.use_mock,
            "recipe": recipe.name,
            "results": results,
            "succeeded": succeeded,
            "failed": total - succeeded,
        }

    record = services.jobs.submit(
        "manga.comfy_queue",
        runner,
        params={"storyboard_id": storyboard_id, "panels": total, "mock": services.settings.comfy.use_mock},
    )
    return {"job_id": record.id, "status": record.status, "kind": record.kind, "panels": total}
