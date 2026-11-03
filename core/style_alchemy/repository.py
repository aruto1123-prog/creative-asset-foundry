"""Style Alchemy の SQLite 永続化（候補・比較・観測・レシピ・グリッド実験・トークン辞書・データセット分析）。"""

from __future__ import annotations

import json
from typing import Any

from core.common import new_id, utc_now_iso
from core.db import Database
from core.style_alchemy.dataset_curator import DatasetGapReport
from core.style_alchemy.grid_laboratory import GridCell, GridExperiment, TokenDictionary, TokenEntry
from core.style_alchemy.model_optimizer import Observation
from core.style_alchemy.preference_engine import Comparison, StyleCandidate
from core.style_alchemy.schemas import StyleRecipe

__all__ = ["StyleLabRepository", "STYLE_LAB_DDL"]

STYLE_LAB_DDL = """
CREATE TABLE IF NOT EXISTS style_candidates (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    params_json TEXT NOT NULL,
    recipe_json TEXT,
    image_ref TEXT,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS style_comparisons (
    id TEXT PRIMARY KEY,
    left_id TEXT NOT NULL,
    right_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    response_time_ms REAL,
    session_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_style_comparisons_created ON style_comparisons(created_at);
CREATE TABLE IF NOT EXISTS style_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT,
    params_json TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS style_recipes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    architecture TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grid_experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grid_cells (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    x_index INTEGER NOT NULL,
    y_index INTEGER NOT NULL,
    z_index INTEGER NOT NULL,
    score REAL,
    image_ref TEXT,
    data_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_grid_cells_experiment ON grid_cells(experiment_id);
CREATE TABLE IF NOT EXISTS style_tokens (
    token TEXT NOT NULL,
    category TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    description TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (token, category)
);
CREATE TABLE IF NOT EXISTS dataset_reports (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class StyleLabRepository:
    """Style Alchemy の全永続データを扱うリポジトリ。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        db.ensure_schema("style_alchemy", STYLE_LAB_DDL)

    # -- 候補 ----------------------------------------------------------------

    def save_candidate(self, candidate: StyleCandidate) -> str:
        self.db.execute(
            """INSERT INTO style_candidates(id, label, params_json, recipe_json, image_ref, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET label=excluded.label, params_json=excluded.params_json,
                   recipe_json=excluded.recipe_json, image_ref=excluded.image_ref, source=excluded.source""",
            (
                candidate.id,
                candidate.label,
                _dumps(candidate.params),
                candidate.recipe.model_dump_json() if candidate.recipe is not None else None,
                candidate.image_ref,
                candidate.source,
                candidate.created_at,
            ),
        )
        return candidate.id

    @staticmethod
    def _candidate_from_row(row: dict[str, Any]) -> StyleCandidate:
        return StyleCandidate(
            id=row["id"],
            label=row["label"],
            params=json.loads(row["params_json"]),
            recipe=StyleRecipe.model_validate_json(row["recipe_json"]) if row.get("recipe_json") else None,
            image_ref=row.get("image_ref"),
            source=row["source"],
            created_at=row["created_at"],
        )

    def get_candidate(self, candidate_id: str) -> StyleCandidate | None:
        row = self.db.query_one("SELECT * FROM style_candidates WHERE id = ?", (candidate_id,))
        return self._candidate_from_row(row) if row else None

    def list_candidates(self, source: str | None = None, limit: int = 1000) -> list[StyleCandidate]:
        if source is None:
            rows = self.db.query("SELECT * FROM style_candidates ORDER BY created_at, rowid LIMIT ?", (limit,))
        else:
            rows = self.db.query(
                "SELECT * FROM style_candidates WHERE source = ? ORDER BY created_at, rowid LIMIT ?", (source, limit)
            )
        return [self._candidate_from_row(row) for row in rows]

    def delete_candidate(self, candidate_id: str) -> bool:
        """候補と、その候補を含む比較を削除する。"""
        result = self.db.execute("DELETE FROM style_candidates WHERE id = ?", (candidate_id,))
        self.db.execute("DELETE FROM style_comparisons WHERE left_id = ? OR right_id = ?", (candidate_id, candidate_id))
        self.db.execute("DELETE FROM style_observations WHERE candidate_id = ?", (candidate_id,))
        return result.rowcount > 0

    def count_candidates(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM style_candidates") or 0)

    # -- 比較 ----------------------------------------------------------------

    def save_comparison(self, comparison: Comparison) -> str:
        self.db.execute(
            """INSERT OR REPLACE INTO style_comparisons(id, left_id, right_id, outcome, response_time_ms, session_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                comparison.id,
                comparison.left_id,
                comparison.right_id,
                comparison.outcome,
                comparison.response_time_ms,
                comparison.session_id,
                comparison.created_at,
            ),
        )
        return comparison.id

    def list_comparisons(self, session_id: str | None = None, limit: int | None = None) -> list[Comparison]:
        sql = "SELECT * FROM style_comparisons"
        params: list[Any] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at, rowid"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [Comparison.model_validate(row) for row in self.db.query(sql, params)]

    def delete_comparison(self, comparison_id: str) -> bool:
        return self.db.execute("DELETE FROM style_comparisons WHERE id = ?", (comparison_id,)).rowcount > 0

    def count_comparisons(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM style_comparisons") or 0)

    # -- 観測 ----------------------------------------------------------------

    def add_observation(self, observation: Observation) -> int:
        result = self.db.execute(
            "INSERT INTO style_observations(candidate_id, params_json, score, created_at) VALUES (?, ?, ?, ?)",
            (observation.candidate_id, _dumps(observation.params), observation.score, observation.created_at),
        )
        return int(result.lastrowid or 0)

    def list_observations(self) -> list[Observation]:
        rows = self.db.query("SELECT * FROM style_observations ORDER BY id")
        return [
            Observation(params=json.loads(row["params_json"]), score=row["score"], candidate_id=row["candidate_id"], created_at=row["created_at"])
            for row in rows
        ]

    def clear_observations(self) -> int:
        return self.db.execute("DELETE FROM style_observations").rowcount

    # -- レシピ --------------------------------------------------------------

    def save_recipe(self, recipe: StyleRecipe) -> str:
        self.db.execute(
            """INSERT INTO style_recipes(id, name, architecture, data_json, created_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, architecture=excluded.architecture, data_json=excluded.data_json""",
            (recipe.id, recipe.name, recipe.architecture.value, recipe.model_dump_json(), recipe.created_at),
        )
        return recipe.id

    def get_recipe(self, recipe_id: str) -> StyleRecipe | None:
        row = self.db.query_one("SELECT data_json FROM style_recipes WHERE id = ?", (recipe_id,))
        return StyleRecipe.model_validate_json(row["data_json"]) if row else None

    def list_recipes(self, limit: int = 200) -> list[StyleRecipe]:
        rows = self.db.query("SELECT data_json FROM style_recipes ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))
        return [StyleRecipe.model_validate_json(row["data_json"]) for row in rows]

    def delete_recipe(self, recipe_id: str) -> bool:
        return self.db.execute("DELETE FROM style_recipes WHERE id = ?", (recipe_id,)).rowcount > 0

    # -- グリッド実験 ----------------------------------------------------------

    def save_experiment(self, experiment: GridExperiment) -> str:
        self.db.execute(
            """INSERT INTO grid_experiments(id, name, status, data_json, created_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, status=excluded.status, data_json=excluded.data_json""",
            (experiment.id, experiment.name, experiment.status, experiment.model_dump_json(), experiment.created_at),
        )
        return experiment.id

    def get_experiment(self, experiment_id: str) -> GridExperiment | None:
        row = self.db.query_one("SELECT data_json FROM grid_experiments WHERE id = ?", (experiment_id,))
        return GridExperiment.model_validate_json(row["data_json"]) if row else None

    def list_experiments(self, limit: int = 200) -> list[GridExperiment]:
        rows = self.db.query("SELECT data_json FROM grid_experiments ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))
        return [GridExperiment.model_validate_json(row["data_json"]) for row in rows]

    def delete_experiment(self, experiment_id: str) -> bool:
        result = self.db.execute("DELETE FROM grid_experiments WHERE id = ?", (experiment_id,))
        self.db.execute("DELETE FROM grid_cells WHERE experiment_id = ?", (experiment_id,))
        return result.rowcount > 0

    def save_cells(self, cells: list[GridCell]) -> int:
        now = utc_now_iso()
        return self.db.executemany(
            """INSERT INTO grid_cells(id, experiment_id, x_index, y_index, z_index, score, image_ref, data_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET score=excluded.score, image_ref=excluded.image_ref,
                   data_json=excluded.data_json, updated_at=excluded.updated_at""",
            [
                (c.id, c.experiment_id, c.x_index, c.y_index, c.z_index, c.score, c.image_ref, c.model_dump_json(), now)
                for c in cells
            ],
        )

    def update_cell(self, cell: GridCell) -> None:
        self.save_cells([cell])

    def get_cell(self, cell_id: str) -> GridCell | None:
        row = self.db.query_one("SELECT data_json FROM grid_cells WHERE id = ?", (cell_id,))
        return GridCell.model_validate_json(row["data_json"]) if row else None

    def list_cells(self, experiment_id: str) -> list[GridCell]:
        rows = self.db.query(
            "SELECT data_json FROM grid_cells WHERE experiment_id = ? ORDER BY z_index, y_index, x_index", (experiment_id,)
        )
        return [GridCell.model_validate_json(row["data_json"]) for row in rows]

    # -- トークン辞書 ----------------------------------------------------------

    def save_token(self, entry: TokenEntry) -> None:
        self.db.execute(
            """INSERT INTO style_tokens(token, category, weight, description, tags_json) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(token, category) DO UPDATE SET weight=excluded.weight, description=excluded.description,
                   tags_json=excluded.tags_json""",
            (entry.token, entry.category, entry.weight, entry.description, _dumps(entry.tags)),
        )

    def list_tokens(self, category: str | None = None) -> list[TokenEntry]:
        if category is None:
            rows = self.db.query("SELECT * FROM style_tokens ORDER BY category, token")
        else:
            rows = self.db.query("SELECT * FROM style_tokens WHERE category = ? ORDER BY token", (category,))
        return [
            TokenEntry(token=r["token"], category=r["category"], weight=r["weight"], description=r["description"], tags=json.loads(r["tags_json"]))
            for r in rows
        ]

    def delete_token(self, token: str, category: str | None = None) -> bool:
        if category is None:
            result = self.db.execute("DELETE FROM style_tokens WHERE token = ?", (token,))
        else:
            result = self.db.execute("DELETE FROM style_tokens WHERE token = ? AND category = ?", (token, category))
        return result.rowcount > 0

    def save_dictionary(self, dictionary: TokenDictionary) -> int:
        entries = dictionary.list()
        for entry in entries:
            self.save_token(entry)
        return len(entries)

    def load_dictionary(self) -> TokenDictionary:
        """DB のトークン辞書（未登録なら既定語彙で初期化して保存）。"""
        entries = self.list_tokens()
        if not entries:
            dictionary = TokenDictionary()
            self.save_dictionary(dictionary)
            return dictionary
        return TokenDictionary(entries=entries, include_defaults=False)

    # -- データセット分析 ------------------------------------------------------

    def save_dataset_report(self, report: DatasetGapReport, name: str = "dataset analysis", report_id: str | None = None) -> str:
        identifier = report_id or new_id("dataset")
        self.db.execute(
            """INSERT INTO dataset_reports(id, name, data_json, created_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, data_json=excluded.data_json""",
            (identifier, name, report.model_dump_json(), report.created_at),
        )
        return identifier

    def get_dataset_report(self, report_id: str) -> DatasetGapReport | None:
        row = self.db.query_one("SELECT data_json FROM dataset_reports WHERE id = ?", (report_id,))
        return DatasetGapReport.model_validate_json(row["data_json"]) if row else None

    def list_dataset_reports(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, name, created_at, data_json FROM dataset_reports ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        )
        summaries: list[dict[str, Any]] = []
        for row in rows:
            data = json.loads(row["data_json"])
            summaries.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "total_images": data.get("total_images", 0),
                    "balance_score": data.get("balance_score", 0.0),
                    "total_additional_images": data.get("total_additional_images", 0),
                }
            )
        return summaries

    def delete_dataset_report(self, report_id: str) -> bool:
        return self.db.execute("DELETE FROM dataset_reports WHERE id = ?", (report_id,)).rowcount > 0
