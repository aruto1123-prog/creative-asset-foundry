"""Style Alchemy Arena: 候補・A/B 判定・ランキング・選好ベクトル・ベイズ最適化・レシピ出力・データセット・トークン・グリッド。"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from core.common import stable_seed
from core.style_alchemy.comfy_export import validate_prompt
from core.style_alchemy.dataset_curator import DatasetCurator
from core.style_alchemy.grid_laboratory import TokenEntry
from core.style_alchemy.model_optimizer import BayesianStyleOptimizer, ParameterSpace, RecipeExporter
from core.style_alchemy.preference_engine import (
    BradleyTerryModel,
    Comparison,
    PreferenceAnalyzer,
    StyleCandidate,
    rank_candidates,
    suggest_next_pair,
)
from core.style_alchemy.preview import render_candidate_preview_svg
from core.style_alchemy.schemas import ModelArchitecture, StyleRecipe
from web_studio.backend.api_models import (
    BootstrapRequest,
    CandidateCreateRequest,
    ComparisonRequest,
    DatasetAnalyzeRequest,
    GridComfyRequest,
    GridCreateRequest,
    GridResultRequest,
    RecipeExportRequest,
    SuggestRequest,
)
from web_studio.backend.deps import conflict, dump, get_services, not_found, split_names
from web_studio.backend.services import FoundryServices

router = APIRouter(prefix="/api/style", tags=["style"])

MIN_COMPARISONS_FOR_VECTOR = 3


def _space(architecture: ModelArchitecture, lora_names: list[str], grouping: Literal["grouped", "per_block"]) -> ParameterSpace:
    return ParameterSpace.for_style_lab(architecture, lora_names=split_names(lora_names), grouping=grouping)


def _require_candidate(services: FoundryServices, candidate_id: str) -> StyleCandidate:
    candidate = services.style_repo.get_candidate(candidate_id)
    if candidate is None:
        raise not_found("候補", candidate_id)
    return candidate


# -- 候補 -------------------------------------------------------------------------


@router.get("/candidates")
def list_candidates(
    source: str | None = None,
    limit: int = Query(1000, ge=1, le=5000),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    repo = services.style_repo
    return {"candidates": dump(repo.list_candidates(source=source, limit=limit)), "total": repo.count_candidates()}


@router.post("/candidates")
def create_candidate(body: CandidateCreateRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    repo = services.style_repo
    label = body.label.strip() or f"候補 {repo.count_candidates() + 1:02d}"
    candidate = StyleCandidate(label=label, params=body.params, recipe=body.recipe, image_ref=body.image_ref, source=body.source)
    repo.save_candidate(candidate)
    return candidate.model_dump(mode="json")


@router.post("/candidates/bootstrap")
def bootstrap_candidates(body: BootstrapRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    repo = services.style_repo
    space = _space(body.architecture, body.lora_names, body.grouping)
    offset = repo.count_candidates()
    created: list[StyleCandidate] = []
    for index, params in enumerate(space.sample_lhs(body.n, seed=body.seed)):
        candidate = StyleCandidate(label=f"LHS候補 {offset + index + 1:02d}", params=params, source="bootstrap")
        repo.save_candidate(candidate)
        created.append(candidate)
    return {"candidates": dump(created), "space": space.model_dump(mode="json")}


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    return _require_candidate(services, candidate_id).model_dump(mode="json")


@router.delete("/candidates/{candidate_id}")
def delete_candidate(candidate_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    if not services.style_repo.delete_candidate(candidate_id):
        raise not_found("候補", candidate_id)
    return {"deleted": True, "id": candidate_id}


@router.get("/candidates/{candidate_id}/preview.svg")
def candidate_preview(
    candidate_id: str,
    width: int = Query(512, ge=64, le=2048),
    height: int = Query(512, ge=64, le=2048),
    label: str | None = Query(None, max_length=40),
    services: FoundryServices = Depends(get_services),
) -> Response:
    candidate = _require_candidate(services, candidate_id)
    svg = render_candidate_preview_svg(
        candidate.params,
        seed=stable_seed("caf.preview", candidate.id),
        width=width,
        height=height,
        label=label,
    )
    return Response(content=svg, media_type="image/svg+xml")


# -- アリーナ・判定・ランキング ------------------------------------------------------


@router.get("/arena/next")
def arena_next(
    seed: int | None = None,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    repo = services.style_repo
    candidates = repo.list_candidates()
    comparisons = repo.list_comparisons()
    pair = suggest_next_pair(candidates, comparisons, seed=len(comparisons) if seed is None else seed)
    if pair is None:
        raise conflict("A/B 判定には候補が2件以上必要です（/api/style/candidates/bootstrap で初期候補を生成できます）")
    by_id = {candidate.id: candidate for candidate in candidates}
    left, right = by_id[pair[0]], by_id[pair[1]]
    return {"left": dump(left), "right": dump(right), "comparisons": len(comparisons)}


@router.get("/comparisons")
def list_comparisons(
    limit: int | None = Query(None, ge=1, le=10000),
    session_id: str | None = None,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    repo = services.style_repo
    return {"comparisons": dump(repo.list_comparisons(session_id=session_id, limit=limit)), "total": repo.count_comparisons()}


@router.post("/comparisons")
def create_comparison(body: ComparisonRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    repo = services.style_repo
    if body.left_id == body.right_id:
        raise HTTPException(status_code=400, detail="同一候補同士は比較できません")
    _require_candidate(services, body.left_id)
    _require_candidate(services, body.right_id)
    comparison = Comparison(
        left_id=body.left_id,
        right_id=body.right_id,
        outcome=body.outcome,
        response_time_ms=body.response_time_ms,
        session_id=body.session_id,
    )
    repo.save_comparison(comparison)
    ratings = rank_candidates(repo.list_candidates(), repo.list_comparisons())
    return {"comparison": dump(comparison), "rankings": dump(ratings[:10])}


@router.get("/rankings")
def rankings(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    repo = services.style_repo
    candidates = repo.list_candidates()
    comparisons = repo.list_comparisons()
    if not candidates:
        return {"ratings": [], "n_candidates": 0, "n_comparisons": len(comparisons), "bradley_terry": None}
    ratings = rank_candidates(candidates, comparisons)
    bt = BradleyTerryModel().fit([c.id for c in candidates], comparisons)
    return {
        "ratings": dump(ratings),
        "n_candidates": len(candidates),
        "n_comparisons": len(comparisons),
        "bradley_terry": {"log_likelihood": bt.log_likelihood, "iterations": bt.iterations, "converged": bt.converged},
    }


@router.get("/preference-vector")
def preference_vector(
    method: Literal["pairwise_logistic", "ridge_bt"] = "pairwise_logistic",
    n_bootstrap: int = Query(100, ge=0, le=1000),
    seed: int = 0,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    repo = services.style_repo
    candidates = repo.list_candidates()
    comparisons = repo.list_comparisons()
    if len(candidates) < 2 or len(comparisons) < MIN_COMPARISONS_FOR_VECTOR:
        raise conflict(f"美的選好ベクトルの推定には候補2件以上・比較{MIN_COMPARISONS_FOR_VECTOR}件以上が必要です")
    vector = PreferenceAnalyzer(n_bootstrap=n_bootstrap, seed=seed, method=method).analyze(candidates, comparisons)
    return vector.model_dump(mode="json")


# -- 最適化・レシピ --------------------------------------------------------------------


@router.get("/optimizer/space")
def optimizer_space(
    architecture: ModelArchitecture = ModelArchitecture.SDXL,
    lora_names: list[str] | None = Query(None),
    grouping: Literal["grouped", "per_block"] = "grouped",
) -> dict[str, Any]:
    space = _space(architecture, lora_names or [], grouping)
    return {"space": space.model_dump(mode="json"), "groups": space.groups(), "dims": len(space.dims)}


@router.post("/optimizer/suggest")
def optimizer_suggest(body: SuggestRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    repo = services.style_repo
    space = _space(body.architecture, body.lora_names, body.grouping)
    optimizer = BayesianStyleOptimizer(space, seed=body.seed, acquisition=body.acquisition)
    candidates = repo.list_candidates()
    comparisons = repo.list_comparisons()
    by_id = {candidate.id: candidate for candidate in candidates}
    if candidates and comparisons:
        for rating in rank_candidates(candidates, comparisons):
            source = by_id[rating.candidate_id].params
            params = {dim.name: float(source.get(dim.name, dim.default)) for dim in space.dims}
            optimizer.add_observation(params, rating.bt_elo_scale, rating.candidate_id)
    suggestions = optimizer.suggest(n=body.n, strategy=body.strategy)
    sensitivity: dict[str, float] = {}
    if len(optimizer.observations) >= 2:
        try:
            sensitivity = optimizer.sensitivity()
        except (ValueError, ArithmeticError):
            sensitivity = {}
    registered: list[StyleCandidate] = []
    if body.register_candidates:
        for suggestion in suggestions:
            candidate = StyleCandidate(label=f"最適化提案 #{suggestion.rank}", params=suggestion.params, source="optimizer")
            repo.save_candidate(candidate)
            registered.append(candidate)
    best = optimizer.best()
    return {
        "suggestions": dump(suggestions),
        "best": dump(best) if best is not None else None,
        "observations": len(optimizer.observations),
        "sensitivity": sensitivity,
        "registered": dump(registered),
        "space": space.model_dump(mode="json"),
    }


@router.post("/recipes/export")
def export_recipe(body: RecipeExportRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    exporter = RecipeExporter()
    recipe = exporter.params_to_recipe(
        body.params,
        architecture=body.architecture,
        model_a=body.model_a,
        model_b=body.model_b,
        loras=body.loras,
        name=body.name,
        model_c=body.model_c,
    )
    workflow = exporter.to_comfy_workflow(recipe, body.prompt, body.negative, body.seed, body.width, body.height)
    services.style_repo.save_recipe(recipe)
    return {
        "recipe": recipe.model_dump(mode="json"),
        "merge_config": exporter.to_merge_config(recipe),
        "merge_script": exporter.to_merge_script(recipe),
        "comfy_workflow": workflow,
        "validation": validate_prompt(workflow),
    }


@router.get("/recipes")
def list_recipes(
    limit: int = Query(200, ge=1, le=2000),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    return {"recipes": dump(services.style_repo.list_recipes(limit=limit))}


# -- データセット ------------------------------------------------------------------------


@router.post("/dataset/analyze")
async def analyze_dataset(body: DatasetAnalyzeRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    records = list(body.records or []) or DatasetCurator.parse_caption_lines(body.captions_text or "")
    if not records:
        raise HTTPException(status_code=400, detail="captions_text または records にキャプションを1件以上指定してください")
    curator = services.dataset_curator(body.trigger_token, body.use_llm)
    cleansed = await curator.cleanse_with_llm(records) if body.use_llm else curator.cleanse(records)
    report = curator.analyze(cleansed)
    report_id = services.style_repo.save_dataset_report(report, name=body.name) if body.save else None
    return {"cleansed": dump(cleansed), "report": report.model_dump(mode="json"), "report_id": report_id}


@router.get("/dataset/reports")
def list_dataset_reports(
    limit: int = Query(100, ge=1, le=1000),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    return {"reports": services.style_repo.list_dataset_reports(limit=limit)}


# -- トークン辞書 ------------------------------------------------------------------------


@router.get("/tokens")
def list_tokens(category: str | None = None, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    dictionary = services.style_repo.load_dictionary()
    return {"tokens": dump(dictionary.list(category)), "categories": dictionary.categories()}


@router.post("/tokens")
def add_token(entry: TokenEntry, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    services.style_repo.load_dictionary()
    services.style_repo.save_token(entry)
    return entry.model_dump(mode="json")


@router.delete("/tokens")
def delete_token(
    token: str = Query(..., min_length=1),
    category: str | None = None,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    services.style_repo.load_dictionary()
    if not services.style_repo.delete_token(token, category):
        raise not_found("トークン", token)
    return {"deleted": True, "token": token, "category": category}


# -- グリッド実験 ------------------------------------------------------------------------


@router.post("/grid")
def create_grid(body: GridCreateRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    lab = services.grid_lab()
    experiment = lab.create_experiment(
        body.name,
        body.base_prompt,
        body.x_axis,
        body.y_axis,
        body.z_axis,
        negative_prompt=body.negative_prompt,
        base_params=body.base_params,
        seed=body.seed,
    )
    return {"experiment": dump(experiment), "cells": dump(lab.cells(experiment.id))}


@router.get("/grid")
def list_grids(
    limit: int = Query(200, ge=1, le=2000),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    return {"experiments": dump(services.style_repo.list_experiments(limit=limit))}


@router.get("/grid/{experiment_id}")
def get_grid(experiment_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    lab = services.grid_lab()
    experiment = lab.get_experiment(experiment_id)
    if experiment is None:
        raise not_found("グリッド実験", experiment_id)
    return {
        "experiment": dump(experiment),
        "cells": dump(lab.cells(experiment_id)),
        "analysis": dump(lab.analyze(experiment_id)),
    }


@router.post("/grid/{experiment_id}/results")
def record_grid_result(
    experiment_id: str,
    body: GridResultRequest,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    lab = services.grid_lab()
    if lab.get_experiment(experiment_id) is None:
        raise not_found("グリッド実験", experiment_id)
    if not body.cell_id.startswith(f"{experiment_id}_"):
        raise HTTPException(status_code=400, detail=f"セル {body.cell_id} は実験 {experiment_id} に属していません")
    cell = lab.record_result(body.cell_id, body.score, image_ref=body.image_ref, notes=body.notes)
    return {"cell": dump(cell), "analysis": dump(lab.analyze(experiment_id))}


@router.post("/grid/{experiment_id}/comfy")
def grid_comfy_batch(
    experiment_id: str,
    body: GridComfyRequest,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    lab = services.grid_lab()
    if lab.get_experiment(experiment_id) is None:
        raise not_found("グリッド実験", experiment_id)
    model_a = body.model_a.model_copy(update={"architecture": body.architecture})
    if body.model_b is not None:
        recipe = RecipeExporter().params_to_recipe(
            {},
            architecture=body.architecture,
            model_a=model_a,
            model_b=body.model_b,
            loras=body.loras,
            name=f"グリッド {experiment_id}",
        )
    else:
        recipe = StyleRecipe(
            name=f"グリッド {experiment_id}",
            architecture=body.architecture,
            base_checkpoint=model_a,
            loras=body.loras,
        )
    batch = lab.to_comfy_batch(experiment_id, recipe, width=body.width, height=body.height)
    return {"recipe": dump(recipe), "batch": batch, "count": len(batch)}
