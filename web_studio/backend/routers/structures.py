"""Structure Lab: 構造の一覧・保存・波形編集・抽出・合成・変異・転用・類似検索・可視化・シード。"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response

from core.structure_engine.composer import (
    MUTATION_DESCRIPTIONS,
    MUTATION_LABELS_JA,
    MutationType,
    merge_structures_with_report,
    mutate_structure,
)
from core.structure_engine.consistency import audit_structure
from core.structure_engine.schemas import ALL_CHANNELS, DEFAULT_CHANNELS, EmotionalInformationWave, UniversalStructure
from core.structure_engine.visualizer import beats_to_mermaid, graph_to_mermaid, wave_to_svg
from web_studio.backend.api_models import (
    ExtractRequest,
    MergeRequest,
    MutateRequest,
    SearchRequest,
    SeedLoadRequest,
    TranspileRequest,
)
from web_studio.backend.deps import dump, get_services, not_found, require_structure
from web_studio.backend.jobs import JobContext, run_or_submit
from web_studio.backend.services import FoundryServices

router = APIRouter(prefix="/api", tags=["structures"])


@router.get("/structures")
def list_structures(
    category: str | None = None,
    source_kind: str | None = None,
    q: str | None = None,
    tag: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    order: Literal["id", "recent", "title"] = "id",
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    repo = services.structure_repo
    items = repo.list(category=category, source_kind=source_kind, tag=tag, query=q, limit=limit, offset=offset, order=order)
    total = repo.count(category, source_kind=source_kind, tag=tag, query=q)
    return {"items": dump(items), "total": total, "limit": limit, "offset": offset}


@router.post("/structures")
def save_structure(structure: UniversalStructure, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    services.structure_repo.save(structure)
    return structure.model_dump(mode="json")


@router.get("/structures/mutations")
def list_mutations() -> list[dict[str, str]]:
    return [
        {"type": kind.value, "label": MUTATION_LABELS_JA[kind.value], "description": MUTATION_DESCRIPTIONS[kind.value]}
        for kind in MutationType
    ]


@router.post("/structures/extract")
async def extract_structure(body: ExtractRequest, services: FoundryServices = Depends(get_services)) -> Any:
    async def runner(ctx: JobContext) -> dict[str, Any]:
        await ctx.update(0.1, "deflavor", "固有名詞・ジャンル装飾を抽象化しています")
        result = await services.extractor().extract(
            body.text, source_kind=body.source_kind, title_hint=body.title_hint, category=body.category
        )
        await ctx.update(0.85, "harness", "4段階推論ハーネスで5階層構造を確定しました")
        if body.save:
            services.structure_repo.save(result.structure)
        await ctx.update(0.95, "save", "構造を保存しました" if body.save else "保存せずに返します")
        return result.model_dump(mode="json")

    return await run_or_submit(
        services.jobs,
        kind="structure.extract",
        run_async=body.run_async,
        runner=runner,
        params={"source_kind": body.source_kind, "chars": len(body.text)},
    )


@router.post("/structures/merge")
def merge(body: MergeRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    struct_a = require_structure(services, body.struct_a_id)
    struct_b = require_structure(services, body.struct_b_id)
    merged, report = merge_structures_with_report(
        struct_a, struct_b, body.weight_topology, body.weight_wave, title=body.title
    )
    if body.save:
        services.structure_repo.save(merged)
    return {"structure": dump(merged), "report": dump(report), "audit": dump(audit_structure(merged))}


@router.post("/structures/mutate")
def mutate(body: MutateRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    original = require_structure(services, body.structure_id)
    mutated = mutate_structure(original, body.mutation_type, intensity=body.intensity, seed=body.seed)
    if body.save:
        services.structure_repo.save(mutated)
    return {"structure": dump(mutated), "audit": dump(audit_structure(mutated))}


@router.post("/structures/search")
def search(body: SearchRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    query: UniversalStructure | EmotionalInformationWave
    exclude: tuple[str, ...] = ()
    if body.structure_id:
        query = require_structure(services, body.structure_id)
        exclude = (body.structure_id,)
    elif body.wave is not None:
        query = body.wave
    else:
        raise HTTPException(status_code=400, detail="structure_id または wave のどちらかを指定してください")
    hits = services.structure_repo.search_similar(
        query, method=body.method, top_k=body.top_k, category=body.category, exclude_ids=exclude
    )
    return {"hits": dump(hits), "method": body.method}


@router.get("/structures/{structure_id}")
def get_structure(structure_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    return require_structure(services, structure_id).model_dump(mode="json")


@router.delete("/structures/{structure_id}")
def delete_structure(structure_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    if not services.structure_repo.delete(structure_id):
        raise not_found("構造", structure_id)
    return {"deleted": True, "id": structure_id}


@router.put("/structures/{structure_id}/wave")
def update_wave(
    structure_id: str,
    wave: EmotionalInformationWave,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    structure = require_structure(services, structure_id)
    payload = structure.model_dump(mode="json")
    wave_payload = wave.model_dump(mode="json")
    wave_payload.pop("information_asymmetry", None)
    payload["wave"] = wave_payload
    updated = UniversalStructure.model_validate(payload)
    services.structure_repo.save(updated)
    return updated.model_dump(mode="json")


@router.get("/structures/{structure_id}/audit")
def audit(structure_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    structure = require_structure(services, structure_id)
    issues = audit_structure(structure)
    return {"issues": dump(issues), "errors": sum(1 for issue in issues if issue.severity == "error")}


@router.post("/structures/{structure_id}/transpile")
async def transpile(
    structure_id: str,
    body: TranspileRequest,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    structure = require_structure(services, structure_id)
    design = await services.transpiler().transpile(structure, body.domain, body.theme_variables)
    return design.model_dump(mode="json")


@router.get("/structures/{structure_id}/mermaid")
def mermaid(structure_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, str]:
    structure = require_structure(services, structure_id)
    return {"graph": graph_to_mermaid(structure), "beats": beats_to_mermaid(structure)}


@router.get("/structures/{structure_id}/wave.svg")
def wave_svg(
    structure_id: str,
    compare_id: str | None = None,
    width: int = Query(800, ge=240, le=2400),
    height: int = Query(320, ge=160, le=1200),
    channels: str | None = None,
    title: bool = True,
    services: FoundryServices = Depends(get_services),
) -> Response:
    structure = require_structure(services, structure_id)
    selected = tuple(DEFAULT_CHANNELS)
    if channels:
        requested = tuple(part.strip() for part in channels.split(",") if part.strip())
        unknown = [name for name in requested if name not in ALL_CHANNELS]
        if unknown or not requested:
            raise HTTPException(
                status_code=400,
                detail=f"未知のチャネルです: {', '.join(unknown) or '(空)'}（候補: {', '.join(ALL_CHANNELS)}）",
            )
        selected = requested
    compare = None
    if compare_id:
        other = require_structure(services, compare_id)
        compare = [(other.metadata.title, other.wave)]
    svg = wave_to_svg(
        structure.wave,
        width=width,
        height=height,
        channels=selected,
        beats=structure.beats,
        title=structure.metadata.title if title else None,
        compare=compare,
    )
    return Response(content=svg, media_type="image/svg+xml")


@router.post("/seeds/load")
def load_seeds(
    body: SeedLoadRequest | None = Body(None),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    request = body or SeedLoadRequest()
    return services.load_seeds(overwrite=request.overwrite)


@router.get("/seeds/summary")
def seeds_summary(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    return services.seed_summary()
