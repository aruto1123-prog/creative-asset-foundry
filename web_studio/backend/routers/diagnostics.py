"""Manga Studio: ネーム構造ドクター（診断・履歴・オーバーレイ・処方適用・画像検出・合成画像）。

診断エンジン（``core.manga_pipeline.diagnostics``）・オーバーレイ（``diagnostics_overlay``）・
画像逆解析（``page_vision`` / ``png_codec``）はリクエスト時に遅延 import する（起動時に NumPy 解析系を
読み込まず、導入されていない環境では 503 で明示する）。
"""

from __future__ import annotations

import asyncio
import base64
import importlib
from types import ModuleType
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from core.common import new_id, truncate
from core.manga_pipeline.diagnostics_schemas import DiagnosticReport, DiagnosticSourceType
from core.manga_pipeline.schemas import Storyboard
from core.manga_pipeline.storyboard_generator import StoryboardConfig
from core.structure_engine.schemas import UniversalStructure
from web_studio.backend.api_models import DiagnosticApplyRequest, DiagnosticDetectRequest, DiagnosticRunRequest
from web_studio.backend.deps import conflict, dump, get_services, not_found, require_structure
from web_studio.backend.jobs import JobContext, run_or_submit
from web_studio.backend.services import FoundryServices

__all__ = ["router", "SAMPLE_VARIANTS", "THUMBNAIL_MAX_SIDE"]

router = APIRouter(prefix="/api/manga", tags=["manga-diagnostics"])

SAMPLE_VARIANTS = ("standard", "slanted", "dense", "splash")
THUMBNAIL_MAX_SIDE = 800
_APPLIED_SUFFIX = "（処方適用後）"


def _module(name: str) -> ModuleType:
    """ネーム構造ドクター関連モジュールを遅延 import する（未導入なら 503）。"""
    qualified = f"core.manga_pipeline.{name}"
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name == qualified:
            raise HTTPException(status_code=503, detail=f"ネーム構造ドクターのモジュールが利用できません: {qualified}") from exc
        raise


def _engine(services: FoundryServices, *, simulate: bool = True, max_prescriptions: int = 12, use_llm: bool = False) -> Any:
    diagnostics = _module("diagnostics")
    config = diagnostics.DiagnosticsConfig(
        simulate_prescriptions=simulate,
        max_prescriptions=max_prescriptions,
        use_llm_commentary=use_llm,
    )
    return diagnostics.StoryboardDiagnosticsEngine(config, llm_client=services.llm_client)


def _target(services: FoundryServices, target_structure_id: str | None, fallback_structure_id: str | None = None) -> UniversalStructure | None:
    """目標構造（明示指定は必須存在、フォールバックは存在すれば使用）。"""
    if target_structure_id:
        return require_structure(services, target_structure_id)
    if fallback_structure_id:
        return services.structure_repo.get(fallback_structure_id)
    return None


def _decode_images(values: list[str]) -> list[bytes]:
    png_codec = _module("png_codec")
    blobs: list[bytes] = []
    for index, value in enumerate(values):
        try:
            blobs.append(png_codec.decode_data_url(value))
        except ValueError as exc:
            raise ValueError(f"画像 {index + 1} を解釈できません: {exc}") from exc
    return blobs


def _thumbnail_data_url(data: bytes, max_side: int = THUMBNAIL_MAX_SIDE) -> str:
    """オーバーレイ背景用に長辺 max_side 以下へ面積平均で縮小したグレースケール PNG の data URL。"""
    png_codec = _module("png_codec")
    gray = np.asarray(png_codec.decode_png(data), dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("画像はグレースケールへ変換できる形式である必要があります")
    height, width = gray.shape
    factor = max(1, int(np.ceil(max(height, width) / float(max_side))))
    if factor > 1:
        cropped_h = (height // factor) * factor
        cropped_w = (width // factor) * factor
        if cropped_h >= factor and cropped_w >= factor:
            gray = gray[:cropped_h, :cropped_w].reshape(cropped_h // factor, factor, cropped_w // factor, factor).mean(axis=(1, 3))
    encoded = png_codec.encode_png_gray(np.clip(gray, 0.0, 1.0).astype(np.float32))
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


async def _call_engine(engine: Any, method: str, use_llm: bool, *args: Any, **kwargs: Any) -> DiagnosticReport:
    """LLM 編集者コメントを使う場合は *_async、それ以外はスレッドで同期版を実行する。"""
    if use_llm:
        return await getattr(engine, f"{method}_async")(*args, **kwargs)
    return await asyncio.to_thread(getattr(engine, method), *args, **kwargs)


async def _diagnose(services: FoundryServices, body: DiagnosticRunRequest, ctx: JobContext) -> DiagnosticReport:
    engine = _engine(services, simulate=body.simulate, max_prescriptions=body.max_prescriptions, use_llm=body.use_llm)
    source = body.source_name()
    target: UniversalStructure | None
    if source in ("storyboard_id", "storyboard"):
        if source == "storyboard_id":
            loaded = services.storyboards.get(str(body.storyboard_id))
            if loaded is None:
                raise not_found("ネーム", str(body.storyboard_id))
            storyboard: Storyboard = loaded[0]
            source_id: str | None = body.storyboard_id
        else:
            assert body.storyboard is not None
            storyboard = body.storyboard
            source_id = storyboard.id
        target = _target(services, body.target_structure_id, storyboard.structure_id)
        await ctx.update(0.15, "analyze", "コマ割り・セリフ配置・情報密度を逆解析しています")
        report = await _call_engine(
            engine,
            "diagnose",
            body.use_llm,
            storyboard,
            target=target,
            source_type=DiagnosticSourceType.STORYBOARD,
            source_id=source_id,
            title=body.title,
        )
    elif source == "annotation":
        assert body.annotation is not None
        annotation = body.annotation
        if body.title:
            annotation = annotation.model_copy(update={"title": body.title})
        target = _target(services, body.target_structure_id)
        await ctx.update(0.15, "analyze", "原稿アノテーションをネームへ正規化して逆解析しています")
        report = await _call_engine(
            engine,
            "diagnose_annotation",
            body.use_llm,
            annotation,
            target=target,
            source_type=DiagnosticSourceType.ANNOTATION,
            source_id=None,
        )
    else:
        images = list(body.images or [])
        await ctx.update(0.05, "decode", f"{len(images)} ページの画像を復号しています")
        blobs = await asyncio.to_thread(_decode_images, images)
        await ctx.update(0.15, "detect", "大津法二値化と連結成分でコマ・フキダシを検出しています")
        page_vision = _module("page_vision")
        analyzer = page_vision.PageVisionAnalyzer()
        detection = await asyncio.to_thread(
            lambda: analyzer.analyze_images(blobs, first_page_side=body.first_page_side, title=body.title)
        )
        target = _target(services, body.target_structure_id)
        await ctx.update(0.4, "analyze", "検出結果を逆解析して5軸で採点しています")
        report = await _call_engine(
            engine,
            "diagnose_annotation",
            body.use_llm,
            detection.annotation,
            target=target,
            source_type=DiagnosticSourceType.IMAGE,
            source_id=None,
        )
        await ctx.update(0.85, "thumbnail", "オーバーレイ用の縮小画像を作成しています")
        thumbnails = await asyncio.to_thread(lambda: [_thumbnail_data_url(blob) for blob in blobs])
        report = report.model_copy(update={"page_images": thumbnails})
    updates: dict[str, Any] = {}
    if body.title:
        updates["title"] = body.title
    if report.target_structure_id is None and target is not None:
        updates["target_structure_id"] = target.id
    if updates:
        report = report.model_copy(update=updates)
    await ctx.update(0.95, "save", "診断レポートを保存しています")
    services.diagnostics.save(report)
    return report


def _require_report(services: FoundryServices, report_id: str) -> DiagnosticReport:
    report = services.diagnostics.get(report_id)
    if report is None:
        raise not_found("診断レポート", report_id)
    return report


# ---------------------------------------------------------------------------
# エンドポイント（固定パスを {report_id} より先に登録する）
# ---------------------------------------------------------------------------


@router.post("/diagnostics")
async def run_diagnostics(body: DiagnosticRunRequest, services: FoundryServices = Depends(get_services)) -> Any:
    if body.storyboard_id and not services.storyboards.exists(body.storyboard_id):
        raise not_found("ネーム", body.storyboard_id)
    if body.target_structure_id:
        require_structure(services, body.target_structure_id)

    async def runner(ctx: JobContext) -> dict[str, Any]:
        report = await _diagnose(services, body, ctx)
        return report.model_dump(mode="json")

    return await run_or_submit(
        services.jobs,
        kind="manga.diagnostics",
        run_async=body.run_async,
        runner=runner,
        params={
            "source": body.source_name(),
            "storyboard_id": body.storyboard_id,
            "target_structure_id": body.target_structure_id,
            "images": len(body.images or []),
            "use_llm": body.use_llm,
            "simulate": body.simulate,
        },
    )


@router.get("/diagnostics")
def list_diagnostics(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    source_id: str | None = None,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    return {
        "items": dump(services.diagnostics.list(limit=limit, offset=offset, source_id=source_id)),
        "total": services.diagnostics.count(),
    }


@router.get("/diagnostics/sample-image.png")
def sample_image(
    seed: int = Query(0, ge=0, le=1_000_000),
    variant: Literal["standard", "slanted", "dense", "splash"] = "standard",
) -> Response:
    page_vision = _module("page_vision")
    png_codec = _module("png_codec")
    image, _annotation = page_vision.synthesize_page_image(seed=seed, variant=variant)
    return Response(content=png_codec.encode_png_gray(image), media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/diagnostics/detect")
async def detect_panels(body: DiagnosticDetectRequest, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    del services
    blobs = await asyncio.to_thread(_decode_images, list(body.images))
    page_vision = _module("page_vision")
    analyzer = page_vision.PageVisionAnalyzer()
    result = await asyncio.to_thread(lambda: analyzer.analyze_images(blobs, first_page_side=body.first_page_side, title=body.title))
    return dump(result)


@router.get("/diagnostics/{report_id}")
def get_diagnostic(report_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    return _require_report(services, report_id).model_dump(mode="json")


@router.delete("/diagnostics/{report_id}")
def delete_diagnostic(report_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    if not services.diagnostics.delete(report_id):
        raise not_found("診断レポート", report_id)
    return {"deleted": True, "id": report_id}


@router.get("/diagnostics/{report_id}/pages/{page_index}/svg")
def diagnostic_overlay_svg(
    report_id: str,
    page_index: int,
    heat: bool = True,
    gaze: bool = True,
    pins: bool = True,
    order: bool = True,
    image: bool = True,
    services: FoundryServices = Depends(get_services),
) -> Response:
    report = _require_report(services, report_id)
    if report.storyboard is None:
        raise conflict("この診断レポートにはオーバーレイを描画できるネームが含まれていません")
    page_count = report.storyboard.page_count
    if not 0 <= page_index < page_count:
        raise HTTPException(status_code=404, detail=f"ページ {page_index} は存在しません（0〜{page_count - 1}）")
    overlay = _module("diagnostics_overlay")
    options = overlay.DiagnosticOverlayOptions(
        show_heat=heat,
        show_gaze=gaze,
        show_pins=pins,
        show_balloon_order=order,
        show_image=image,
    )
    svg = overlay.render_diagnostic_overlay(report, page_index, options)
    return Response(content=svg, media_type="image/svg+xml")


@router.post("/diagnostics/{report_id}/apply")
async def apply_prescriptions(
    report_id: str,
    body: DiagnosticApplyRequest,
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    report = _require_report(services, report_id)
    known = {prescription.id for prescription in report.prescriptions}
    unknown = [prescription_id for prescription_id in body.prescription_ids if prescription_id not in known]
    if unknown:
        raise ValueError(f"このレポートに存在しない処方です: {', '.join(unknown)}")
    storyboard = report.storyboard
    config = StoryboardConfig()
    if report.source_type == DiagnosticSourceType.STORYBOARD and report.source_id:
        loaded = services.storyboards.get(report.source_id)
        if loaded is not None:
            config = loaded[1]
            if storyboard is None:
                storyboard = loaded[0]
    if storyboard is None:
        raise conflict("この診断レポートには処方を適用できるネームが含まれていません")
    target = services.structure_repo.get(report.target_structure_id) if report.target_structure_id else None
    engine = _engine(services)
    new_storyboard, new_report = await asyncio.to_thread(
        lambda: engine.apply_prescriptions(storyboard, report, list(body.prescription_ids), target=target)
    )
    base_title = report.title if report.title.endswith(_APPLIED_SUFFIX) else f"{report.title}{_APPLIED_SUFFIX}"
    updates: dict[str, Any] = {"title": truncate(base_title, 200)}
    if new_report.id == report.id or services.diagnostics.exists(new_report.id):
        updates["id"] = new_id("diag")
    saved_storyboard_id: str | None = None
    result_storyboard = new_storyboard
    if body.save_storyboard:
        storyboard_title = new_storyboard.title if new_storyboard.title.endswith("（ドクター処方適用）") else f"{new_storyboard.title}（ドクター処方適用）"
        result_storyboard = new_storyboard.model_copy(update={"id": new_id("sb"), "title": truncate(storyboard_title, 200)})
        services.storyboards.save(result_storyboard, config)
        saved_storyboard_id = result_storyboard.id
        updates.update(
            {
                "source_type": DiagnosticSourceType.STORYBOARD,
                "source_id": saved_storyboard_id,
                "storyboard": result_storyboard,
            }
        )
    new_report = new_report.model_copy(update=updates)
    services.diagnostics.save(new_report)
    return {
        "storyboard": result_storyboard.model_dump(mode="json"),
        "report": new_report.model_dump(mode="json"),
        "score_before": report.overall_score,
        "score_after": new_report.overall_score,
        "saved_storyboard_id": saved_storyboard_id,
        "applied": list(body.prescription_ids),
    }


@router.get("/storyboards/{storyboard_id}/diagnostics")
async def diagnose_saved_storyboard(
    storyboard_id: str,
    simulate: bool = True,
    max_prescriptions: int = Query(12, ge=1, le=50),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    loaded = services.storyboards.get(storyboard_id)
    if loaded is None:
        raise not_found("ネーム", storyboard_id)
    storyboard = loaded[0]
    target = services.structure_repo.get(storyboard.structure_id)
    engine = _engine(services, simulate=simulate, max_prescriptions=max_prescriptions)
    report = await asyncio.to_thread(
        lambda: engine.diagnose(
            storyboard,
            target=target,
            source_type=DiagnosticSourceType.STORYBOARD,
            source_id=storyboard_id,
            title=None,
        )
    )
    services.diagnostics.save(report)
    return report.model_dump(mode="json")
