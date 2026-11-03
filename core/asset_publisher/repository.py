"""Asset Publisher の SQLite 永続化（記事・企画マトリクス）。"""

from __future__ import annotations

import json
from typing import Any

from core.asset_publisher.schemas import ArticleSummary, ContentMatrix, PublishedArticle
from core.common import utc_now_iso
from core.db import Database

__all__ = ["ArticleRepository"]

_DDL = """
CREATE TABLE IF NOT EXISTS publisher_articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    axis TEXT NOT NULL,
    source_id TEXT,
    final_score REAL NOT NULL DEFAULT 0,
    char_count INTEGER NOT NULL DEFAULT 0,
    tags_json TEXT NOT NULL DEFAULT '[]',
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_publisher_articles_axis ON publisher_articles(axis);
CREATE INDEX IF NOT EXISTS idx_publisher_articles_source ON publisher_articles(source_id);
CREATE TABLE IF NOT EXISTS publisher_matrices (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_title TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_publisher_matrices_source ON publisher_matrices(source_id);
"""


class ArticleRepository:
    """公開記事と企画マトリクスの CRUD。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.db.ensure_schema("asset_publisher", _DDL)

    # ---------------------------------------------------------------- articles
    def save(self, article: PublishedArticle) -> str:
        now = utc_now_iso()
        char_count = article.draft.total_chars
        self.db.execute(
            """
            INSERT INTO publisher_articles(id, title, axis, source_id, final_score, char_count, tags_json, data_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, axis=excluded.axis, source_id=excluded.source_id,
                final_score=excluded.final_score, char_count=excluded.char_count, tags_json=excluded.tags_json,
                data_json=excluded.data_json, updated_at=excluded.updated_at
            """,
            (
                article.id,
                article.title,
                article.axis.value,
                article.source_id,
                float(article.final_score),
                int(char_count),
                json.dumps(article.tags, ensure_ascii=False),
                article.model_dump_json(),
                article.created_at or now,
                now,
            ),
        )
        return article.id

    def get(self, article_id: str) -> PublishedArticle | None:
        row = self.db.query_one("SELECT data_json FROM publisher_articles WHERE id = ?", (article_id,))
        if row is None:
            return None
        return PublishedArticle.model_validate_json(row["data_json"])

    def list(self, *, axis: str | None = None, source_id: str | None = None, limit: int = 100, offset: int = 0) -> list[ArticleSummary]:
        clauses: list[str] = []
        params: list[Any] = []
        if axis:
            clauses.append("axis = ?")
            params.append(axis)
        if source_id:
            clauses.append("source_id = ?")
            params.append(source_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(
            f"""
            SELECT id, title, axis, source_id, final_score, char_count, tags_json, created_at
            FROM publisher_articles {where}
            ORDER BY created_at DESC, id ASC LIMIT ? OFFSET ?
            """,
            (*params, int(limit), int(offset)),
        )
        return [
            ArticleSummary(
                id=row["id"],
                title=row["title"],
                axis=row["axis"],
                source_id=row["source_id"] or "",
                final_score=row["final_score"],
                tags=json.loads(row["tags_json"] or "[]"),
                char_count=row["char_count"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def all(self) -> list[PublishedArticle]:
        rows = self.db.query("SELECT data_json FROM publisher_articles ORDER BY created_at ASC, id ASC")
        return [PublishedArticle.model_validate_json(row["data_json"]) for row in rows]

    def delete(self, article_id: str) -> bool:
        return self.db.execute("DELETE FROM publisher_articles WHERE id = ?", (article_id,)).rowcount > 0

    def count(self, axis: str | None = None) -> int:
        if axis:
            return int(self.db.scalar("SELECT COUNT(*) FROM publisher_articles WHERE axis = ?", (axis,)) or 0)
        return int(self.db.scalar("SELECT COUNT(*) FROM publisher_articles") or 0)

    # ---------------------------------------------------------------- matrices
    def save_matrix(self, matrix: ContentMatrix) -> str:
        now = utc_now_iso()
        self.db.execute(
            """
            INSERT INTO publisher_matrices(id, source_id, source_title, data_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id=excluded.source_id, source_title=excluded.source_title,
                data_json=excluded.data_json, updated_at=excluded.updated_at
            """,
            (matrix.id, matrix.source_id, matrix.source_title, matrix.model_dump_json(), matrix.created_at or now, now),
        )
        return matrix.id

    def get_matrix(self, matrix_id: str) -> ContentMatrix | None:
        row = self.db.query_one("SELECT data_json FROM publisher_matrices WHERE id = ?", (matrix_id,))
        if row is None:
            return None
        return ContentMatrix.model_validate_json(row["data_json"])

    def list_matrices(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, source_id, source_title, created_at FROM publisher_matrices ORDER BY created_at DESC, id ASC LIMIT ?",
            (int(limit),),
        )
        return [dict(row) for row in rows]

    def delete_matrix(self, matrix_id: str) -> bool:
        return self.db.execute("DELETE FROM publisher_matrices WHERE id = ?", (matrix_id,)).rowcount > 0
