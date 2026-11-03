"""初期プリセット「普遍的構造データベース」のローダーと SQLite 初期シード投入。

シード JSON 形式::

    {"schema_version": "1.0", "category": "<StructureCategory>", "description": "...",
     "patterns": [<UniversalStructure>, ...]}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from core.structure_engine.schemas import UniversalStructure

if TYPE_CHECKING:
    from core.structure_engine.repository import StructureRepository

__all__ = [
    "SEED_FILES",
    "SEED_SCHEMA_VERSION",
    "SeedReport",
    "seed_dir",
    "load_seed_file",
    "load_all_seeds",
    "seed_database",
]

SEED_SCHEMA_VERSION = "1.0"
SEED_FILES: dict[str, str] = {
    "narrative_archetype": "narrative_archetypes.json",
    "manga_panel_rhythm": "manga_panel_rhythm_patterns.json",
    "persuasion_marketing": "persuasion_and_marketing_structures.json",
}


class SeedReport(BaseModel):
    """シード投入結果。"""

    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    total: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)


def seed_dir() -> Path:
    """シード JSON の格納ディレクトリ。"""
    return Path(__file__).resolve().parent


def _resolve(filename: str | Path, directory: Path | None) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    return (directory or seed_dir()) / candidate


def load_seed_file(filename: str | Path, directory: Path | None = None) -> list[UniversalStructure]:
    """シード JSON を読み込み、全パターンを検証済み ``UniversalStructure`` で返す。"""
    path = _resolve(filename, directory)
    if not path.exists():
        raise FileNotFoundError(f"シードファイルが見つかりません: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("patterns"), list):
        raise ValueError(f"シードファイルの形式が不正です（patterns 配列が必要）: {path}")
    version = str(payload.get("schema_version", ""))
    if version.split(".")[0] != SEED_SCHEMA_VERSION.split(".")[0]:
        raise ValueError(f"未対応のシードスキーマバージョンです: {version}（期待 {SEED_SCHEMA_VERSION}）")
    category = payload.get("category")
    structures: list[UniversalStructure] = []
    for index, item in enumerate(payload["patterns"]):
        structure = UniversalStructure.model_validate(item)
        if category and structure.metadata.category.value != category:
            raise ValueError(
                f"{path.name} の {index} 件目（{structure.id}）のカテゴリ {structure.metadata.category.value} が"
                f"ファイルのカテゴリ {category} と一致しません"
            )
        structures.append(structure)
    ids = [s.id for s in structures]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path.name} に重複IDがあります: {', '.join(duplicates)}")
    return structures


def load_all_seeds(directory: Path | None = None) -> list[UniversalStructure]:
    """全カテゴリのシード（物語原型 → 漫画リズム → 説得構造の順）を読み込む。"""
    structures: list[UniversalStructure] = []
    for filename in SEED_FILES.values():
        structures.extend(load_seed_file(filename, directory))
    return structures


def seed_database(
    repo: "StructureRepository",
    *,
    overwrite: bool = False,
    directory: Path | None = None,
    structures: Iterable[UniversalStructure] | None = None,
) -> SeedReport:
    """シードを SQLite へ投入する。既存IDは overwrite=False ならスキップ、True なら更新。"""
    items = list(structures) if structures is not None else load_all_seeds(directory)
    report = SeedReport(total=len(items))
    for structure in items:
        category = structure.metadata.category.value
        report.by_category[category] = report.by_category.get(category, 0) + 1
        if repo.exists(structure.id):
            if overwrite:
                repo.save(structure)
                report.updated += 1
            else:
                report.skipped += 1
        else:
            repo.save(structure)
            report.inserted += 1
    return report
