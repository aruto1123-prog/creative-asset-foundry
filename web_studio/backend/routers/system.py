"""System & Local AI Config: ヘルス・設定・接続テスト・DB 統計／バックアップ・テレメトリ。"""

from __future__ import annotations

import asyncio
import datetime as _dt
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from core import __version__
from core.manga_pipeline.comfy_client import ComfyClient
from web_studio.backend.deps import get_services, not_found
from web_studio.backend.services import FoundryServices

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    settings = services.settings
    status = services.llm_status()
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": status["provider"],
        "llm_configured_provider": status["configured_provider"],
        "llm_fallback_active": status["fallback_active"],
        "llm_model": status["model"],
        "comfy_base_url": settings.comfy.base_url,
        "comfy_mock": settings.comfy.use_mock,
        "db_path": str(settings.db_path),
        "structures": services.structure_repo.count(),
        "seeds_loaded": services.structure_repo.count(source_kind="seed"),
        "running_jobs": services.jobs.running_count(),
        "ws_clients": services.hub.client_count,
        "started_at": services.started_at,
    }


@router.get("/system/config")
async def get_config(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    return services.settings.public_dict()


@router.put("/system/config")
async def update_config(
    payload: dict[str, Any] = Body(...),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    settings = await services.reload_settings(payload)
    return settings.public_dict()


@router.post("/system/llm/test")
async def test_llm(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        info = await services.llm_client.health_check()
    except Exception as exc:  # noqa: BLE001 - 接続失敗は結果として返す
        info = {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "models": []}
    status = services.llm_status()
    return {
        "ok": bool(info.get("ok")),
        "provider": status["provider"],
        "configured_provider": status["configured_provider"],
        "fallback_active": status["fallback_active"],
        "model": status["model"],
        "detail": str(info.get("detail", "")),
        "models": list(info.get("models", []) or []),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


@router.post("/system/comfy/test")
async def test_comfy(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    started = time.perf_counter()
    use_mock = services.settings.comfy.use_mock
    base_url = services.settings.comfy.base_url
    try:
        base_url = await services.comfy_base_url()
        client = ComfyClient(base_url, timeout_s=5.0)
        stats = await asyncio.wait_for(client.system_stats(), timeout=8.0)
    except Exception as exc:  # noqa: BLE001 - 接続失敗は結果として返す
        return {
            "ok": False,
            "mock": use_mock,
            "base_url": base_url,
            "detail": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    return {
        "ok": True,
        "mock": use_mock,
        "base_url": base_url,
        "detail": "内蔵モック ComfyUI に接続しました" if use_mock else "ComfyUI に接続しました",
        "system_stats": stats,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


@router.get("/system/db/stats")
def db_stats(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    db = services.db
    return {
        "path": str(db.path),
        "size_bytes": db.size_bytes(),
        "tables": db.table_stats(),
        "schemas": db.registered_schemas(),
    }


def _backup_entry(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at": _dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc).isoformat(timespec="seconds"),
    }


@router.post("/system/db/backup")
def create_backup(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    backup_dir = services.settings.backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = backup_dir / f"foundry_{stamp}.sqlite3"
    suffix = 1
    while target.exists():
        target = backup_dir / f"foundry_{stamp}_{suffix}.sqlite3"
        suffix += 1
    path = services.db.backup(target)
    return {**_backup_entry(path), "created_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")}


@router.get("/system/db/backups")
def list_backups(services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    backup_dir = services.settings.backup_dir
    files = sorted(backup_dir.glob("foundry_*.sqlite3"), key=lambda p: p.stat().st_mtime, reverse=True) if backup_dir.exists() else []
    return {"backup_dir": str(backup_dir), "backups": [_backup_entry(path) for path in files]}


@router.get("/system/telemetry")
def telemetry(
    limit: int = Query(50, ge=1, le=500),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    store = services.llm_client.telemetry
    if store is None:
        return {"calls": [], "stats": {"totals": {}, "by_task_stage": [], "runs": {}}, "runs": []}
    return {"calls": store.recent_calls(limit), "stats": store.stats(), "runs": store.recent_runs(min(limit, 100))}


@router.get("/system/telemetry/runs/{run_id}")
def telemetry_run(run_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    store = services.llm_client.telemetry
    run = store.get_run(run_id) if store is not None else None
    if run is None:
        raise not_found("ハーネス実行", run_id)
    return run
