#!/usr/bin/env python3
"""ローカル LLM（DeepSeek-V4.1-Flash）の接続・構造化出力・4段階ハーネスを実機で検証する。

使い方（リポジトリ直下で仮想環境を有効化した状態）:
  python local_tools/check_llm.py --provider openai_compatible --base-url http://localhost:8000/v1 --model DeepSeek-V4.1-Flash
  python local_tools/check_llm.py --provider ollama --base-url http://localhost:11434/v1 --model <Ollama のモデルタグ>

--mode auto（既定）は json_schema → json_object → prompt_only の順に試し、
「LLM 経由で抽出に成功し、自己修復リトライと失敗呼び出しが最も少ないモード」を推奨する。
--save を付けると推奨設定を data/config.json に保存する（起動中のスタジオは再起動で反映）。
--provider mock は配線確認用（実機を使わない）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import FoundrySettings, LLMSettings  # noqa: E402
from core.db import Database  # noqa: E402
from core.llm_client import create_llm_client  # noqa: E402
from core.structure_engine.extractor import StructureExtractor  # noqa: E402
from core.structure_engine.samples import sample_text_ja  # noqa: E402

MODES = ("json_schema", "json_object", "prompt_only")


def build_settings(args: argparse.Namespace, mode: str) -> LLMSettings:
    settings = LLMSettings(
        provider=args.provider,
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        structured_output=mode,
        api_key=args.api_key,
    )
    if args.provider == "ollama":
        settings.ollama_base_url = args.base_url
        settings.ollama_model = args.model
    else:
        settings.base_url = args.base_url
        settings.model = args.model
    return settings


async def evaluate(args: argparse.Namespace, mode: str, db: Database) -> dict[str, Any]:
    client = create_llm_client(build_settings(args, mode), db=db)
    record: dict[str, Any] = {"mode": mode, "ok": False}
    try:
        health = await client.health_check()
        record["models"] = list(health.get("models") or [])
        if not health.get("ok"):
            record["error"] = f"ヘルスチェック失敗: {health.get('detail')}"
            return record
        started = time.perf_counter()
        result = await StructureExtractor(client).extract(sample_text_ja(), source_kind="synopsis")
        record["seconds"] = round(time.perf_counter() - started, 1)
        record["used_llm"] = bool(result.used_llm)
        record["critique_score"] = result.critique_score
        record["title"] = result.structure.metadata.title
        run = client.telemetry.get_run(result.harness_run_id) if (client.telemetry and result.harness_run_id) else None
        calls = list((run or {}).get("calls", []))
        record["calls"] = len(calls)
        record["failed_calls"] = sum(1 for call in calls if not call.get("success"))
        record["repairs"] = sum(1 for call in calls if int(call.get("attempt") or 1) > 1)
        record["prompt_tokens"] = sum(int(call.get("prompt_tokens") or 0) for call in calls)
        record["completion_tokens"] = sum(int(call.get("completion_tokens") or 0) for call in calls)
        record["reasoning_tokens"] = sum(int(call.get("reasoning_tokens") or 0) for call in calls)
        record["stages"] = [str(stage.get("stage")) for stage in (run or {}).get("stage_logs", [])]
        record["ok"] = record["used_llm"]
        if not record["ok"]:
            record["error"] = "ハーネスが失敗しヒューリスティック抽出へフォールバックしました（data/ のテレメトリを確認）"
    except Exception as exc:  # noqa: BLE001 - 検証結果として報告する
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        await client.aclose()
    return record


def rank_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if record.get("ok") else 1,
        int(record.get("repairs", 99)),
        int(record.get("failed_calls", 99)),
        float(record.get("seconds", 1e9)),
    )


def save_settings(args: argparse.Namespace, mode: str) -> Path:
    settings = FoundrySettings.load()
    settings.llm.provider = args.provider
    if args.provider == "ollama":
        settings.llm.ollama_base_url = args.base_url
        settings.llm.ollama_model = args.model
    else:
        settings.llm.base_url = args.base_url
        settings.llm.model = args.model
    settings.llm.structured_output = mode  # type: ignore[assignment]
    settings.llm.timeout_s = args.timeout
    settings.llm.max_tokens = args.max_tokens
    settings.llm.api_key = args.api_key
    return settings.save()


def print_record(record: dict[str, Any]) -> None:
    status = "OK " if record.get("ok") else "NG "
    print(f"[{status}] mode={record['mode']}")
    for key in ("models", "seconds", "used_llm", "critique_score", "calls", "repairs", "failed_calls",
                "prompt_tokens", "completion_tokens", "reasoning_tokens", "stages", "title", "error"):
        if key in record:
            print(f"      {key}: {record[key]}")


async def main_async(args: argparse.Namespace) -> int:
    if os.environ.get("CAF_LLM_PROVIDER") and os.environ["CAF_LLM_PROVIDER"] != args.provider:
        print(f"注意: 環境変数 CAF_LLM_PROVIDER={os.environ['CAF_LLM_PROVIDER']} が設定値より優先されます。unset してから再実行してください。")
        return 2
    modes = MODES if args.mode == "auto" else (args.mode,)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="caf_llm_check_") as tmp:
        db = Database(Path(tmp) / "telemetry.sqlite3")
        for mode in modes:
            print(f"--- structured_output={mode} を検証中（実機では数分かかる場合があります）")
            record = await evaluate(args, mode, db)
            print_record(record)
            records.append(record)
    best = sorted(records, key=rank_key)[0]
    if not best.get("ok"):
        print("\n結論: どのモードでも LLM 経由の構造化出力に成功しませんでした。サーバ起動・モデル名・タイムアウトを確認してください。")
        return 1
    print(f"\n推奨: provider={args.provider} structured_output={best['mode']}（修復 {best['repairs']} 回 / 失敗呼び出し {best['failed_calls']} 件 / {best['seconds']} 秒）")
    if args.save:
        path = save_settings(args, best["mode"])
        print(f"設定を保存しました: {path}（起動中のスタジオは再起動で反映）")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ローカル LLM の実機検証")
    parser.add_argument("--provider", choices=["openai_compatible", "ollama", "mock"], default="openai_compatible")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--mode", choices=["auto", *MODES], default="auto")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--save", action="store_true", help="推奨設定を data/config.json に保存する")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
