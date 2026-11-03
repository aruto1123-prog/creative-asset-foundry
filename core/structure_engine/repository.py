"""普遍的構造の SQLite 永続化と波形類似度検索。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.common import utc_now_iso
from core.db import Database
from core.structure_engine.schemas import (
    DEFAULT_CHANNELS,
    EmotionalInformationWave,
    UniversalStructure,
)
from core.structure_engine.similarity import SimilarityMethod, wave_similarity

__all__ = ["StructureSummary", "SimilarityHit", "StructureRepository", "STRUCTURE_DDL"]

STRUCTURE_DDL = """
CREATE TABLE IF NOT EXISTS structures (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    category TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    data_json TEXT NOT NULL,
    peak_t REAL NOT NULL DEFAULT 0,
    suspense REAL NOT NULL DEFAULT 0,
    surprise REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_structures_category ON structures(category);
CREATE INDEX IF NOT EXISTS idx_structures_source_kind ON structures(source_kind);
CREATE INDEX IF NOT EXISTS idx_structures_updated ON structures(updated_at);
"""

_ORDER_SQL = {
    "id": "id ASC",
    "recent": "updated_at DESC, id ASC",
    "title": "title ASC, id ASC",
}


class StructureSummary(BaseModel):
    """一覧表示用の構造要約。"""

    id: str
    title: str
    summary: str
    category: str
    source_kind: str
    tags: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    peak_t: float
    suspense: float
    surprise: float


class SimilarityHit(BaseModel):
    """類似度検索の1件。"""

    id: str
    title: str
    category: str
    score: float = Field(..., ge=0.0, le=1.0)
    dtw_distance: float | None = None
    cosine: float | None = None


class StructureRepository:
    """``structures`` テーブルへの CRUD と類似検索。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        db.ensure_schema("structure_engine.structures", STRUCTURE_DDL)

    # -- 書き込み ---------------------------------------------------------

    def save(self, structure: UniversalStructure) -> str:
        """構造を保存する（同一IDは上書き。created_at は初回保存時刻を維持）。"""
        now = utc_now_iso()
        existing = self.db.query_one("SELECT created_at FROM structures WHERE id = ?", (structure.id,))
        created_at = existing["created_at"] if existing else (structure.metadata.created_at or now)
        stats = structure.wave.summary()
        self.db.execute(
            """
            INSERT INTO structures (id, title, summary, category, source_kind, tags_json, data_json,
                                    peak_t, suspense, surprise, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                category = excluded.category,
                source_kind = excluded.source_kind,
                tags_json = excluded.tags_json,
                data_json = excluded.data_json,
                peak_t = excluded.peak_t,
                suspense = excluded.suspense,
                surprise = excluded.surprise,
                updated_at = excluded.updated_at
            """,
            (
                structure.id,
                structure.metadata.title,
                structure.metadata.summary,
                structure.metadata.category.value,
                structure.metadata.source_kind.value,
                json.dumps(structure.metadata.tags, ensure_ascii=False),
                structure.model_dump_json(),
                float(stats["peak_t"]),
                float(stats["suspense"]),
                float(stats["surprise"]),
                created_at,
                now,
            ),
        )
        return structure.id

    def save_many(self, structures: Iterable[UniversalStructure]) -> int:
        count = 0
        for structure in structures:
            self.save(structure)
            count += 1
        return count

    def delete(self, structure_id: str) -> bool:
        result = self.db.execute("DELETE FROM structures WHERE id = ?", (structure_id,))
        return result.rowcount > 0

    # -- 読み出し ---------------------------------------------------------

    def exists(self, structure_id: str) -> bool:
        return self.db.query_one("SELECT 1 AS ok FROM structures WHERE id = ?", (structure_id,)) is not None

    def get(self, structure_id: str) -> UniversalStructure | None:
        row = self.db.query_one("SELECT data_json FROM structures WHERE id = ?", (structure_id,))
        if row is None:
            return None
        return UniversalStructure.model_validate_json(row["data_json"])

    @staticmethod
    def _filters(
        category: str | None,
        source_kind: str | None,
        tag: str | None,
        query: str | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if source_kind:
            clauses.append("source_kind = ?")
            params.append(source_kind)
        if tag:
            clauses.append("tags_json LIKE ?")
            params.append(f"%{json.dumps(tag, ensure_ascii=False)}%")
        if query:
            clauses.append("(title LIKE ? OR summary LIKE ? OR tags_json LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like, like])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def list(
        self,
        *,
        category: str | None = None,
        source_kind: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 200,
        offset: int = 0,
        order: Literal["id", "recent", "title"] = "id",
    ) -> list[StructureSummary]:
        """フィルタ付き一覧（query は title / summary / tags の部分一致）。"""
        where, params = self._filters(category, source_kind, tag, query)
        order_sql = _ORDER_SQL.get(order, _ORDER_SQL["id"])
        rows = self.db.query(
            f"SELECT id, title, summary, category, source_kind, tags_json, created_at, updated_at, "
            f"peak_t, suspense, surprise FROM structures{where} ORDER BY {order_sql} LIMIT ? OFFSET ?",
            (*params, max(0, int(limit)), max(0, int(offset))),
        )
        return [
            StructureSummary(
                id=row["id"],
                title=row["title"],
                summary=row["summary"],
                category=row["category"],
                source_kind=row["source_kind"],
                tags=json.loads(row["tags_json"] or "[]"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                peak_t=float(row["peak_t"]),
                suspense=float(row["suspense"]),
                surprise=float(row["surprise"]),
            )
            for row in rows
        ]

    def count(
        self,
        category: str | None = None,
        *,
        source_kind: str | None = None,
        tag: str | None = None,
        query: str | None = None,
    ) -> int:
        where, params = self._filters(category, source_kind, tag, query)
        return int(self.db.scalar(f"SELECT COUNT(*) FROM structures{where}", params) or 0)

    def all(self, category: str | None = None) -> list[UniversalStructure]:
        where, params = self._filters(category, None, None, None)
        rows = self.db.query(f"SELECT data_json FROM structures{where} ORDER BY id ASC", params)
        return [UniversalStructure.model_validate_json(row["data_json"]) for row in rows]

    def category_counts(self) -> dict[str, int]:
        rows = self.db.query("SELECT category, COUNT(*) AS n FROM structures GROUP BY category ORDER BY category")
        return {row["category"]: int(row["n"]) for row in rows}

    # -- 類似検索 ---------------------------------------------------------

    def search_similar(
        self,
        query: UniversalStructure | EmotionalInformationWave,
        *,
        method: SimilarityMethod = "hybrid",
        top_k: int = 5,
        category: str | None = None,
        exclude_ids: Sequence[str] = (),
        channels: Sequence[str] = DEFAULT_CHANNELS,
    ) -> list[SimilarityHit]:
        """波形の DTW / コサイン / ハイブリッド類似度で上位 top_k 件を返す（score 降順）。"""
        query_wave = query.wave if isinstance(query, UniversalStructure) else query
        excluded = set(exclude_ids)
        where, params = self._filters(category, None, None, None)
        rows = self.db.query(f"SELECT id, title, category, data_json FROM structures{where}", params)
        hits: list[SimilarityHit] = []
        for row in rows:
            if row["id"] in excluded:
                continue
            payload = json.loads(row["data_json"])
            candidate_wave = EmotionalInformationWave.model_validate(payload["wave"])
            result = wave_similarity(query_wave, candidate_wave, method=method, channels=channels)
            hits.append(
                SimilarityHit(
                    id=row["id"],
                    title=row["title"],
                    category=row["category"],
                    score=round(result["score"], 6),
                    dtw_distance=round(result["dtw_distance"], 6),
                    cosine=round(result["cosine"], 6),
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.id))
        return hits[: max(0, int(top_k))]

    def export_all(self) -> list[dict[str, Any]]:
        return [structure.model_dump(mode="json") for structure in self.all()]
