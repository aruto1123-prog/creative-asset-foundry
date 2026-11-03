"""ルーター共通の依存関数とヘルパー。"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel

from core.structure_engine.schemas import UniversalStructure
from web_studio.backend.services import FoundryServices

__all__ = ["get_services", "require_structure", "not_found", "conflict", "dump", "split_names"]


def get_services(request: Request) -> FoundryServices:
    return request.app.state.services


def not_found(kind: str, identifier: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{kind}が見つかりません: {identifier}")


def conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=409, detail=detail)


def require_structure(services: FoundryServices, structure_id: str) -> UniversalStructure:
    structure = services.structure_repo.get(structure_id)
    if structure is None:
        raise not_found("構造", structure_id)
    return structure


def dump(value: Any) -> Any:
    """Pydantic モデル（およびその list / dict）を JSON 互換へ変換する。"""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [dump(item) for item in value]
    if isinstance(value, tuple):
        return [dump(item) for item in value]
    if isinstance(value, dict):
        return {key: dump(item) for key, item in value.items()}
    return value


def split_names(values: list[str] | None) -> list[str]:
    """クエリの繰り返し指定・カンマ区切り指定の両方を受け付けて名前リストへ正規化する。"""
    names: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part not in names:
                names.append(part)
    return names
