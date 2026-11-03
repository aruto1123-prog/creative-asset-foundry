"""SQLite 永続化基盤。

- 操作ごとに接続を開閉するスレッドセーフな設計（FastAPI のスレッドプールから安全に利用可能）。
- WAL モードで読み書きの並行性を確保し、書き込みはプロセス内 RLock で直列化する。
- 各モジュールは ``ensure_schema(name, ddl)`` で自分のテーブル群を冪等に作成する。
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.common import utc_now_iso

__all__ = ["Database", "ExecResult"]

_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS _schema_registry (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ExecResult:
    """``execute`` の結果。"""

    lastrowid: int | None
    rowcount: int


class Database:
    """SQLite データベースへの薄く堅牢なアクセス層。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._schemas: set[str] = set()
        with self._write_lock, self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_REGISTRY_DDL)

    def __repr__(self) -> str:
        return f"Database(path={str(self.path)!r})"

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """接続を開き、正常終了でコミット・例外でロールバックしてクローズする。"""
        conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> ExecResult:
        """単一の更新系SQLを実行する。"""
        with self._write_lock, self.connect() as conn:
            cur = conn.execute(sql, params)
            return ExecResult(lastrowid=cur.lastrowid, rowcount=cur.rowcount)

    def executemany(self, sql: str, seq: Iterable[Sequence[Any] | Mapping[str, Any]]) -> int:
        """同一SQLをパラメータ列で一括実行し、影響行数を返す。"""
        with self._write_lock, self.connect() as conn:
            cur = conn.executemany(sql, list(seq))
            return cur.rowcount

    def query(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> list[dict[str, Any]]:
        """SELECT を実行し、各行を dict で返す。"""
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def query_one(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> dict[str, Any] | None:
        """先頭1行のみを返す（無ければ None）。"""
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def scalar(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Any:
        """先頭行の先頭列を返す（無ければ None）。"""
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return row[0] if row is not None else None

    def ensure_schema(self, name: str, ddl: str) -> None:
        """DDL スクリプトを冪等に適用し、登録簿へ記録する。"""
        if name in self._schemas:
            return
        with self._write_lock:
            if name in self._schemas:
                return
            with self.connect() as conn:
                conn.executescript(ddl)
                conn.execute(
                    "INSERT OR REPLACE INTO _schema_registry(name, applied_at) VALUES (?, ?)",
                    (name, utc_now_iso()),
                )
            self._schemas.add(name)

    def registered_schemas(self) -> list[str]:
        """適用済みスキーマ名の一覧。"""
        return [row["name"] for row in self.query("SELECT name FROM _schema_registry ORDER BY name")]

    def tables(self) -> list[str]:
        """ユーザーテーブル名の一覧（内部テーブルを除く）。"""
        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row["name"] for row in rows]

    def table_stats(self) -> dict[str, int]:
        """テーブルごとの行数。"""
        stats: dict[str, int] = {}
        for table in self.tables():
            safe = table.replace('"', '""')
            stats[table] = int(self.scalar(f'SELECT COUNT(*) FROM "{safe}"') or 0)
        return stats

    def size_bytes(self) -> int:
        """DB本体＋WAL＋SHMの合計サイズ。"""
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                total += candidate.stat().st_size
        return total

    def backup(self, dest: str | Path) -> Path:
        """SQLite オンラインバックアップAPIで整合性のあるスナップショットを作成する。"""
        dest_path = Path(dest).expanduser().resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            src = sqlite3.connect(str(self.path), timeout=30.0)
            try:
                dst = sqlite3.connect(str(dest_path))
                try:
                    src.backup(dst)
                finally:
                    dst.close()
            finally:
                src.close()
        return dest_path
