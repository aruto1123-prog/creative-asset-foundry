"""非同期ジョブ管理（抽出・ネーム生成・執筆・ComfyUI 投入など重い処理）。

ジョブは asyncio タスクとして実行し、状態（queued / running / succeeded / failed）・進捗・段階・メッセージ・
結果・エラーを SQLite（``backend_jobs``）へ永続化しつつ、WebSocket ハブへ進捗イベントを配信する。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from core.common import new_id, utc_now_iso
from core.db import Database
from web_studio.backend.ws_hub import WebSocketHub

__all__ = [
    "JobStatus",
    "JobRecord",
    "JobContext",
    "JobManager",
    "JobRunner",
    "detached_context",
    "run_or_submit",
    "TERMINAL_STATUSES",
]

logger = logging.getLogger("caf.backend.jobs")

JobStatus = Literal["queued", "running", "succeeded", "failed"]
TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed"})

JOBS_DDL = """
CREATE TABLE IF NOT EXISTS backend_jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    stage TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL DEFAULT '',
    params_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_backend_jobs_created ON backend_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_backend_jobs_status ON backend_jobs(status);
"""


class JobRecord(BaseModel):
    """ジョブの状態。"""

    id: str
    kind: str
    status: JobStatus = "queued"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    stage: str = "queued"
    message: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    def event(self) -> dict[str, Any]:
        """WebSocket 配信用のイベント表現（結果本体は含めない）。"""
        return {
            "type": "job",
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class JobContext:
    """ジョブ実行関数へ渡す進捗報告ハンドル（同期実行時は何もしない）。"""

    def __init__(self, manager: "JobManager | None", job_id: str | None) -> None:
        self._manager = manager
        self._job_id = job_id

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def is_detached(self) -> bool:
        return self._manager is None or self._job_id is None

    async def update(self, progress: float | None = None, stage: str | None = None, message: str | None = None) -> None:
        """進捗を更新して配信する。"""
        if self._manager is None or self._job_id is None:
            return
        await self._manager.update(self._job_id, progress=progress, stage=stage, message=message)

    async def pipeline_progress(self, stage: str, progress: float, message: str) -> None:
        """``on_progress(stage, progress, message)`` 形式のコールバック（0.05〜0.95 に写像）。"""
        await self.update(progress=0.05 + 0.9 * max(0.0, min(1.0, float(progress))), stage=stage, message=message)


JobRunner = Callable[[JobContext], Awaitable[Any]]


def detached_context() -> JobContext:
    """同期実行用の空コンテキスト。"""
    return JobContext(None, None)


class JobManager:
    """ジョブの投入・実行・状態管理。"""

    def __init__(self, db: Database, hub: WebSocketHub, *, max_memory: int = 200, persist_interval_s: float = 0.5) -> None:
        self.db = db
        self.hub = hub
        self.max_memory = max(10, int(max_memory))
        self.persist_interval_s = float(persist_interval_s)
        self._jobs: dict[str, JobRecord] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._last_persist: dict[str, float] = {}
        db.ensure_schema("web_studio.backend_jobs", JOBS_DDL)

    # -- 永続化 ---------------------------------------------------------------

    def _persist(self, record: JobRecord) -> None:
        result_json = None if record.result is None else json.dumps(record.result, ensure_ascii=False)
        self.db.execute(
            "INSERT INTO backend_jobs(id, kind, status, progress, stage, message, params_json, result_json, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, progress=excluded.progress, stage=excluded.stage, "
            "message=excluded.message, result_json=excluded.result_json, error=excluded.error, updated_at=excluded.updated_at",
            (
                record.id,
                record.kind,
                record.status,
                float(record.progress),
                record.stage,
                record.message,
                json.dumps(record.params, ensure_ascii=False),
                result_json,
                record.error,
                record.created_at,
                record.updated_at,
            ),
        )
        self._last_persist[record.id] = time.monotonic()

    @staticmethod
    def _from_row(row: dict[str, Any], include_result: bool = True) -> JobRecord:
        result = json.loads(row["result_json"]) if include_result and row.get("result_json") else None
        return JobRecord(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            progress=float(row["progress"]),
            stage=row["stage"],
            message=row["message"],
            params=json.loads(row["params_json"] or "{}"),
            result=result,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _trim_memory(self) -> None:
        if len(self._jobs) <= self.max_memory:
            return
        finished = [job for job in self._jobs.values() if job.status in TERMINAL_STATUSES]
        finished.sort(key=lambda job: job.updated_at)
        for job in finished[: len(self._jobs) - self.max_memory]:
            self._jobs.pop(job.id, None)
            self._last_persist.pop(job.id, None)

    # -- 実行 ---------------------------------------------------------------

    def submit(self, kind: str, runner: JobRunner, *, params: dict[str, Any] | None = None) -> JobRecord:
        """ジョブを登録し、実行中のイベントループでタスクとして開始する。"""
        loop = asyncio.get_running_loop()
        record = JobRecord(id=new_id("job"), kind=kind, params=jsonable_encoder(params or {}), message="投入しました")
        self._jobs[record.id] = record
        self._persist(record)
        task = loop.create_task(self._run(record.id, runner), name=f"caf-job-{record.id}")
        self._tasks[record.id] = task
        task.add_done_callback(lambda _t, job_id=record.id: self._tasks.pop(job_id, None))
        return record

    async def _run(self, job_id: str, runner: JobRunner) -> None:
        await self.update(job_id, status="running", progress=0.01, stage="running", message="実行を開始しました")
        try:
            result = await runner(JobContext(self, job_id))
        except asyncio.CancelledError:
            await self.update(job_id, status="failed", stage="cancelled", message="キャンセルされました", error="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - ジョブ失敗は記録して配信する
            logger.exception("ジョブ %s が失敗しました", job_id)
            await self.update(
                job_id,
                status="failed",
                stage="failed",
                message="失敗しました",
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        await self.update(
            job_id,
            status="succeeded",
            progress=1.0,
            stage="completed",
            message="完了しました",
            result=jsonable_encoder(result),
        )

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        stage: str | None = None,
        message: str | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> JobRecord:
        """ジョブ状態を更新し、必要に応じて永続化してイベントを配信する。"""
        record = self._jobs.get(job_id) or self.get(job_id)
        if record is None:
            raise KeyError(f"ジョブが見つかりません: {job_id}")
        changes: dict[str, Any] = {"updated_at": utc_now_iso()}
        if status is not None:
            changes["status"] = status
        if progress is not None:
            changes["progress"] = max(record.progress, min(1.0, max(0.0, float(progress))))
        if stage is not None:
            changes["stage"] = stage
        if message is not None:
            changes["message"] = message
        if result is not None:
            changes["result"] = result
        if error is not None:
            changes["error"] = error
        updated = record.model_copy(update=changes)
        self._jobs[job_id] = updated
        elapsed = time.monotonic() - self._last_persist.get(job_id, 0.0)
        if status is not None or result is not None or error is not None or elapsed >= self.persist_interval_s:
            self._persist(updated)
        await self.hub.broadcast(updated.event())
        if updated.status in TERMINAL_STATUSES:
            self._trim_memory()
        return updated

    # -- 参照 ---------------------------------------------------------------

    def get(self, job_id: str) -> JobRecord | None:
        if job_id in self._jobs:
            return self._jobs[job_id]
        row = self.db.query_one("SELECT * FROM backend_jobs WHERE id = ?", (job_id,))
        return self._from_row(row) if row else None

    def list(self, limit: int = 50) -> list[JobRecord]:
        rows = self.db.query(
            "SELECT * FROM backend_jobs ORDER BY created_at DESC, id DESC LIMIT ?", (max(1, int(limit)),)
        )
        records: list[JobRecord] = []
        for row in rows:
            live = self._jobs.get(row["id"])
            source = live if live is not None else self._from_row(row, include_result=False)
            records.append(source.model_copy(update={"result": None}))
        return records

    def running_count(self) -> int:
        return sum(1 for job in self._jobs.values() if job.status in ("queued", "running"))

    async def wait(self, job_id: str, timeout_s: float = 60.0) -> JobRecord:
        """ジョブ完了を待つ（主にテスト・内部利用）。"""
        task = self._tasks.get(job_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        record = self.get(job_id)
        if record is None:
            raise KeyError(f"ジョブが見つかりません: {job_id}")
        return record

    async def shutdown(self) -> None:
        """実行中ジョブをキャンセルして終了を待つ。"""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()


async def run_or_submit(
    manager: JobManager,
    *,
    kind: str,
    run_async: bool,
    runner: JobRunner,
    params: dict[str, Any] | None = None,
) -> Any:
    """``async: true`` ならジョブ化して ID を返し、それ以外は同期実行して結果を返す。"""
    if run_async:
        record = manager.submit(kind, runner, params=params)
        return {"job_id": record.id, "status": record.status, "kind": kind}
    return await runner(detached_context())
