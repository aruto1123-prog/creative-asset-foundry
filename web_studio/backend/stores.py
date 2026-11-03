"""バックエンド独自の永続化（ネーム・企画マトリクスに紐づく研究ログ）。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from core.asset_publisher.schemas import ResearchLog
from core.common import utc_now_iso
from core.db import Database
from core.manga_pipeline.schemas import Storyboard
from core.manga_pipeline.storyboard_generator import StoryboardConfig

__all__ = ["StoryboardSummary", "StoryboardStore", "ResearchLogStore"]

STORE_DDL = """
CREATE TABLE IF NOT EXISTS backend_storyboards (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    structure_id TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    panel_count INTEGER NOT NULL,
    score REAL,
    passed INTEGER,
    config_json TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_backend_storyboards_created ON backend_storyboards(created_at);
CREATE INDEX IF NOT EXISTS idx_backend_storyboards_structure ON backend_storyboards(structure_id);
CREATE TABLE IF NOT EXISTS backend_research_logs (
    key TEXT PRIMARY KEY,
    log_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class StoryboardSummary(BaseModel):
    """ネーム一覧の1行。"""

    id: str
    title: str
    structure_id: str
    page_count: int
    panel_count: int
    score: float | None = None
    passed: bool | None = None
    created_at: str
    updated_at: str


class StoryboardStore:
    """Storyboard（ネーム）と生成設定の保存。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        db.ensure_schema("web_studio.stores", STORE_DDL)

    def exists(self, storyboard_id: str) -> bool:
        return self.db.query_one("SELECT 1 AS hit FROM backend_storyboards WHERE id = ?", (storyboard_id,)) is not None

    def save(self, storyboard: Storyboard, config: StoryboardConfig) -> str:
        now = utc_now_iso()
        existing = self.db.query_one("SELECT created_at FROM backend_storyboards WHERE id = ?", (storyboard.id,))
        created_at = existing["created_at"] if existing else (storyboard.created_at or now)
        grammar = storyboard.grammar
        self.db.execute(
            "INSERT INTO backend_storyboards(id, title, structure_id, page_count, panel_count, score, passed, config_json, data_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, structure_id=excluded.structure_id, page_count=excluded.page_count, "
            "panel_count=excluded.panel_count, score=excluded.score, passed=excluded.passed, config_json=excluded.config_json, "
            "data_json=excluded.data_json, updated_at=excluded.updated_at",
            (
                storyboard.id,
                storyboard.title,
                storyboard.structure_id,
                storyboard.page_count,
                len(storyboard.panels_in_reading_order()),
                None if grammar is None else float(grammar.score),
                None if grammar is None else int(bool(grammar.passed)),
                config.model_dump_json(),
                storyboard.model_dump_json(),
                created_at,
                now,
            ),
        )
        return storyboard.id

    def get(self, storyboard_id: str) -> tuple[Storyboard, StoryboardConfig] | None:
        row = self.db.query_one("SELECT config_json, data_json FROM backend_storyboards WHERE id = ?", (storyboard_id,))
        if row is None:
            return None
        return Storyboard.model_validate_json(row["data_json"]), StoryboardConfig.model_validate_json(row["config_json"])

    def list(self, limit: int = 100, offset: int = 0) -> list[StoryboardSummary]:
        rows = self.db.query(
            "SELECT id, title, structure_id, page_count, panel_count, score, passed, created_at, updated_at "
            "FROM backend_storyboards ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (max(1, int(limit)), max(0, int(offset))),
        )
        return [
            StoryboardSummary(
                id=row["id"],
                title=row["title"],
                structure_id=row["structure_id"],
                page_count=int(row["page_count"]),
                panel_count=int(row["panel_count"]),
                score=None if row["score"] is None else float(row["score"]),
                passed=None if row["passed"] is None else bool(row["passed"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def delete(self, storyboard_id: str) -> bool:
        return self.db.execute("DELETE FROM backend_storyboards WHERE id = ?", (storyboard_id,)).rowcount > 0

    def count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM backend_storyboards") or 0)


class ResearchLogStore:
    """企画マトリクス ID 等に紐づく研究ログの保存（記事生成時の根拠データとして再利用）。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        db.ensure_schema("web_studio.stores", STORE_DDL)

    def save(self, key: str, log: ResearchLog) -> None:
        self.db.execute(
            "INSERT INTO backend_research_logs(key, log_json, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET log_json=excluded.log_json",
            (key, log.model_dump_json(), utc_now_iso()),
        )

    def get(self, key: str) -> ResearchLog | None:
        row = self.db.query_one("SELECT log_json FROM backend_research_logs WHERE key = ?", (key,))
        return ResearchLog.model_validate_json(row["log_json"]) if row else None

    def keys(self) -> list[str]:
        return [row["key"] for row in self.db.query("SELECT key FROM backend_research_logs ORDER BY created_at DESC")]

    def raw(self, key: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT log_json FROM backend_research_logs WHERE key = ?", (key,))
        return json.loads(row["log_json"]) if row else None
