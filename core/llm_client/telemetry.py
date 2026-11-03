"""LLM 呼び出しとハーネス実行のテレメトリ（SQLite 永続化）。

- ``llm_calls``: 全呼び出し（成功・失敗・修復試行）のトークン数・レイテンシ・思考ログ
- ``harness_runs``: 4段階ハーネス1回分の集計
- ``harness_stage_logs``: 各段（decompose / synthesize / critique / refine）のログ
"""

from __future__ import annotations

import json
from typing import Any

from core.common import truncate, utc_now_iso
from core.db import Database
from core.llm_client.types import StageLog, TokenUsage

__all__ = ["TelemetryStore", "TELEMETRY_DDL"]

TELEMETRY_DDL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    task TEXT NOT NULL,
    stage TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    attempt INTEGER NOT NULL DEFAULT 1,
    success INTEGER NOT NULL,
    error TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    request_excerpt TEXT,
    response_excerpt TEXT,
    reasoning_log TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_task_stage ON llm_calls(task, stage);
CREATE TABLE IF NOT EXISTS harness_runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    objective TEXT,
    status TEXT NOT NULL,
    stages TEXT NOT NULL DEFAULT '[]',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_latency_ms REAL NOT NULL DEFAULT 0,
    critique_score REAL,
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS harness_stage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL,
    latency_ms REAL NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    thought_log TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stage_logs_run ON harness_stage_logs(run_id);
"""

EXCERPT_LIMIT = 4000
REASONING_LIMIT = 20000


class TelemetryStore:
    """推論トークン数・レイテンシ・思考ログの SQLite 記録器。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        db.ensure_schema("llm_telemetry", TELEMETRY_DDL)

    # -- 記録 -------------------------------------------------------------

    def record_call(
        self,
        *,
        run_id: str | None,
        task: str,
        stage: str,
        provider: str,
        model: str,
        attempt: int = 1,
        success: bool,
        error: str | None = None,
        usage: TokenUsage | None = None,
        latency_ms: float = 0.0,
        request_excerpt: str = "",
        response_excerpt: str = "",
        reasoning_log: str = "",
    ) -> int:
        """1回の LLM 呼び出しを記録し、行 ID を返す。"""
        usage = usage or TokenUsage()
        result = self.db.execute(
            """
            INSERT INTO llm_calls (
                run_id, task, stage, provider, model, attempt, success, error,
                prompt_tokens, completion_tokens, reasoning_tokens, total_tokens, estimated,
                latency_ms, request_excerpt, response_excerpt, reasoning_log, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                task,
                stage,
                provider,
                model,
                int(attempt),
                1 if success else 0,
                truncate(error, EXCERPT_LIMIT) if error else None,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.reasoning_tokens,
                usage.total_tokens,
                1 if usage.estimated else 0,
                float(latency_ms),
                truncate(request_excerpt or "", EXCERPT_LIMIT),
                truncate(response_excerpt or "", EXCERPT_LIMIT),
                truncate(reasoning_log or "", REASONING_LIMIT),
                utc_now_iso(),
            ),
        )
        return int(result.lastrowid or 0)

    def start_run(self, run_id: str, task: str, objective: str, stages: list[str]) -> None:
        """ハーネス実行の開始を記録する。"""
        self.db.execute(
            """
            INSERT OR REPLACE INTO harness_runs (run_id, task, objective, status, stages, started_at)
            VALUES (?, ?, ?, 'running', ?, ?)
            """,
            (run_id, task, truncate(objective or "", EXCERPT_LIMIT), json.dumps(list(stages)), utc_now_iso()),
        )

    def record_stage(self, run_id: str, log: StageLog) -> None:
        """ハーネス1段のログを記録する。"""
        self.db.execute(
            """
            INSERT INTO harness_stage_logs (
                run_id, stage, ordinal, attempts, success, latency_ms,
                prompt_tokens, completion_tokens, reasoning_tokens, summary, thought_log, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                log.stage,
                log.ordinal,
                log.attempts,
                1 if log.success else 0,
                float(log.latency_ms),
                log.usage.prompt_tokens,
                log.usage.completion_tokens,
                log.usage.reasoning_tokens,
                truncate(log.summary or "", EXCERPT_LIMIT),
                truncate(log.thought_excerpt or "", REASONING_LIMIT),
                utc_now_iso(),
            ),
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        usage: TokenUsage,
        latency_ms: float,
        critique_score: float | None = None,
        error: str | None = None,
    ) -> None:
        """ハーネス実行の完了（成功・失敗）を記録する。"""
        self.db.execute(
            """
            UPDATE harness_runs
               SET status = ?, prompt_tokens = ?, completion_tokens = ?, reasoning_tokens = ?,
                   total_tokens = ?, total_latency_ms = ?, critique_score = ?, error = ?, finished_at = ?
             WHERE run_id = ?
            """,
            (
                status,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.reasoning_tokens,
                usage.total_tokens,
                float(latency_ms),
                critique_score,
                truncate(error, EXCERPT_LIMIT) if error else None,
                utc_now_iso(),
                run_id,
            ),
        )

    # -- 参照 -------------------------------------------------------------

    @staticmethod
    def _decode_call(row: dict[str, Any]) -> dict[str, Any]:
        row["success"] = bool(row.get("success"))
        row["estimated"] = bool(row.get("estimated"))
        return row

    @staticmethod
    def _decode_run(row: dict[str, Any]) -> dict[str, Any]:
        try:
            row["stages"] = json.loads(row.get("stages") or "[]")
        except json.JSONDecodeError:
            row["stages"] = []
        return row

    def recent_calls(self, limit: int = 50) -> list[dict[str, Any]]:
        """新しい順の呼び出し履歴。"""
        rows = self.db.query("SELECT * FROM llm_calls ORDER BY id DESC LIMIT ?", (max(1, int(limit)),))
        return [self._decode_call(row) for row in rows]

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """新しい順のハーネス実行履歴。"""
        rows = self.db.query(
            "SELECT * FROM harness_runs ORDER BY started_at DESC, rowid DESC LIMIT ?", (max(1, int(limit)),)
        )
        return [self._decode_run(row) for row in rows]

    def calls_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """指定実行に属する呼び出し（時系列順）。"""
        rows = self.db.query("SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id ASC", (run_id,))
        return [self._decode_call(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """実行の集計・段ログ・呼び出しを返す。"""
        row = self.db.query_one("SELECT * FROM harness_runs WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        run = self._decode_run(row)
        stage_rows = self.db.query(
            "SELECT * FROM harness_stage_logs WHERE run_id = ? ORDER BY ordinal ASC, id ASC", (run_id,)
        )
        for stage_row in stage_rows:
            stage_row["success"] = bool(stage_row.get("success"))
        run["stage_logs"] = stage_rows
        run["calls"] = self.calls_for_run(run_id)
        return run

    def stats(self) -> dict[str, Any]:
        """総計・task/stage 別集計・実行状態別件数。"""
        totals_row = self.db.query_one(
            """
            SELECT COUNT(*) AS calls,
                   COALESCE(SUM(success), 0) AS successes,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                   COALESCE(SUM(total_tokens), 0) AS total_tokens,
                   COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(MAX(latency_ms), 0) AS max_latency_ms
              FROM llm_calls
            """
        ) or {}
        calls = int(totals_row.get("calls") or 0)
        successes = int(totals_row.get("successes") or 0)
        totals = {
            "calls": calls,
            "successes": successes,
            "failures": calls - successes,
            "success_rate": (successes / calls) if calls else 0.0,
            "prompt_tokens": int(totals_row.get("prompt_tokens") or 0),
            "completion_tokens": int(totals_row.get("completion_tokens") or 0),
            "reasoning_tokens": int(totals_row.get("reasoning_tokens") or 0),
            "total_tokens": int(totals_row.get("total_tokens") or 0),
            "avg_latency_ms": float(totals_row.get("avg_latency_ms") or 0.0),
            "max_latency_ms": float(totals_row.get("max_latency_ms") or 0.0),
        }
        grouped = self.db.query(
            """
            SELECT task, stage,
                   COUNT(*) AS calls,
                   COALESCE(SUM(success), 0) AS successes,
                   COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                   COALESCE(SUM(total_tokens), 0) AS total_tokens
              FROM llm_calls
             GROUP BY task, stage
             ORDER BY task, stage
            """
        )
        by_task_stage = []
        for row in grouped:
            group_calls = int(row["calls"])
            group_successes = int(row["successes"])
            by_task_stage.append(
                {
                    "task": row["task"],
                    "stage": row["stage"],
                    "calls": group_calls,
                    "successes": group_successes,
                    "success_rate": (group_successes / group_calls) if group_calls else 0.0,
                    "avg_latency_ms": float(row["avg_latency_ms"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "completion_tokens": int(row["completion_tokens"]),
                    "reasoning_tokens": int(row["reasoning_tokens"]),
                    "total_tokens": int(row["total_tokens"]),
                }
            )
        run_rows = self.db.query("SELECT status, COUNT(*) AS n FROM harness_runs GROUP BY status")
        runs = {row["status"]: int(row["n"]) for row in run_rows}
        return {"totals": totals, "by_task_stage": by_task_stage, "runs": runs}

    def clear(self) -> None:
        """全テレメトリを削除する（保守用）。"""
        self.db.execute("DELETE FROM llm_calls")
        self.db.execute("DELETE FROM harness_stage_logs")
        self.db.execute("DELETE FROM harness_runs")
