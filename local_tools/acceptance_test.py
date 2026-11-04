#!/usr/bin/env python3
"""CAF ローカル実用受け入れテスト（起動中のスタジオに対して全ループを HTTP API で一気通貫実行）。

  構造抽出（実機 LLM）→ シードと合成 → ネーム生成と漫画文法検証 → SVG 取得
  → ComfyUI ワークフロー生成（--checkpoint 指定時は実機 ComfyUI へ投入）
  → 3軸企画 → 5段階執筆 → 一人称「私」・URL 除去の検証 → note 出力 → DB バックアップ

使い方（スタジオ起動済み）:
  python local_tools/acceptance_test.py                         # 実機 LLM 必須（モックなら失敗）
  python local_tools/acceptance_test.py --allow-mock            # 配線確認（モック LLM でも可）
  python local_tools/acceptance_test.py --checkpoint <ckpt名>   # ComfyUI 実生成まで（先頭2コマ）
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

SAMPLE_TEXT = (
    "港町の古書店で働く青年は、亡き祖父が遺した未完の地図を手がかりに、町から消えた灯台の謎を追い始める。"
    "幼なじみの記者は協力を申し出るが、実は町の再開発を進める会社から青年の調査を見張るよう命じられていた。"
    "青年は地図の空白が祖父の失敗ではなく、意図的な隠蔽だったと気づき、記者の裏切りを知って孤立する。"
    "それでも青年は自らの店を手放して最後の手がかりを買い戻し、嵐の夜に灯台の跡地で真実を公にする。"
)


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        print(f"[{'OK' if ok else 'NG'}] {name} {detail}")
        return ok

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.rows)


def wait_job(client: httpx.Client, job_id: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = client.get(f"/api/jobs/{job_id}").json()
        if last.get("status") in {"succeeded", "failed"}:
            return last
        print(f"      ジョブ {last.get('status')} {round(float(last.get('progress') or 0.0) * 100)}% {last.get('stage') or ''}", end="\r", flush=True)
        time.sleep(2.0)
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CAF ローカル実用受け入れテスト")
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--allow-mock", action="store_true", help="モック LLM / モック ComfyUI でも合格とする")
    parser.add_argument("--checkpoint", default=None, help="ComfyUI のチェックポイント名（指定時は実機へ投入）")
    parser.add_argument("--architecture", default="sdxl", choices=["sdxl", "sd15", "flux", "anima"])
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)

    check = Check()
    client = httpx.Client(base_url=args.base, timeout=args.timeout, trust_env=False)

    health = client.get("/api/health").json()
    real_llm = health.get("llm_provider") != "mock" and not health.get("llm_fallback_active")
    check.record("スタジオ稼働", health.get("status") == "ok", f"seeds={health.get('seeds_loaded')} structures={health.get('structures')}")
    check.record("実機 LLM 接続", real_llm or args.allow_mock, f"provider={health.get('llm_provider')} fallback={health.get('llm_fallback_active')}")
    llm = client.post("/api/system/llm/test").json()
    check.record("LLM 接続テスト", bool(llm.get("ok")), f"model={llm.get('model')} {llm.get('latency_ms')}ms")
    comfy = client.post("/api/system/comfy/test").json()
    comfy_required = bool(args.checkpoint) and not args.allow_mock
    check.record("ComfyUI 接続テスト", bool(comfy.get("ok")) or not comfy_required, f"mock={comfy.get('mock')} {comfy.get('detail')}")

    extraction = client.post("/api/structures/extract", json={"text": SAMPLE_TEXT, "source_kind": "synopsis"}).json()
    structure = extraction["structure"]
    check.record("構造抽出（4段階ハーネス）", bool(extraction.get("used_llm")) or args.allow_mock,
                 f"used_llm={extraction.get('used_llm')} critique={extraction.get('critique_score')} title={structure['metadata']['title']}")
    joined = " ".join([structure["metadata"]["title"], structure["metadata"]["summary"]])
    check.record("脱フレーバー（固有の装飾が残らない）", "古書店" not in joined and "灯台" not in joined, "")

    seeds = client.get("/api/structures", params={"category": "narrative_archetype", "limit": 3}).json()["items"]
    merged_response = client.post(
        "/api/structures/merge",
        json={"struct_a_id": structure["id"], "struct_b_id": seeds[0]["id"], "weight_topology": 0.65, "weight_wave": 0.7},
    ).json()
    merged = merged_response["structure"]
    check.record("構造合成", bool(merged.get("id")), f"{merged['metadata']['title']}")

    storyboard = client.post(
        "/api/manga/storyboards",
        json={"structure_id": merged["id"], "theme": {"THEME": "消えた灯台"}, "config": {"target_pages": args.pages}},
    ).json()
    grammar = storyboard.get("grammar") or {}
    check.record("ネーム生成と漫画文法検証", bool(grammar.get("passed")), f"pages={len(storyboard['pages'])} score={grammar.get('score')}")
    svg = client.get(f"/api/manga/storyboards/{storyboard['id']}/pages/0/svg", params={"gaze": "true"}).text
    try:
        ET.fromstring(svg)
        svg_ok = True
    except ET.ParseError:
        svg_ok = False
    check.record("SVG ネーム出力", svg_ok, f"{len(svg)} bytes")

    recipe = {
        "name": "acceptance",
        "architecture": args.architecture,
        "base_checkpoint": {"name": args.checkpoint or "sd_xl_base_1.0.safetensors", "architecture": args.architecture},
        "style_tokens": ["monochrome", "manga", "lineart"],
    }
    compiled = client.post(f"/api/manga/storyboards/{storyboard['id']}/comfy", json={"architecture": args.architecture, "recipe": recipe}).json()
    check.record("ComfyUI ワークフロー生成", int(compiled.get("count") or 0) > 0, f"{compiled.get('count')} コマ分")
    if args.checkpoint or args.allow_mock:
        queued = client.post(
            f"/api/manga/storyboards/{storyboard['id']}/comfy/queue",
            json={"architecture": args.architecture, "recipe": recipe, "max_panels": 2},
        ).json()
        job = wait_job(client, queued["job_id"], args.timeout)
        result = job.get("result") or {}
        print()
        check.record("ComfyUI 投入と生成", job.get("status") == "succeeded" and int(result.get("failed", 1)) == 0,
                     f"mock={result.get('mock')} succeeded={result.get('succeeded')} error={job.get('error')}")

    matrix = client.post("/api/publisher/matrix", json={"structure_id": merged["id"]}).json()
    check.record("3軸企画展開", len(matrix.get("plans", [])) == 3, "")
    article = client.post("/api/publisher/articles", json={"matrix_id": matrix["id"], "axis": "creator_theory"}).json()
    check.record("5段階執筆", float(article.get("final_score") or 0.0) >= 80.0, f"score={article.get('final_score')} title={article.get('title')}")
    verify = client.post("/api/publisher/verify", json={"text": article["markdown"]}).json()
    check.record("一人称「私」統一・URL 完全除去", bool(verify.get("first_person_unified")) and bool(verify.get("url_free")), "")
    note = client.get(f"/api/publisher/articles/{article['id']}/export", params={"format": "note"}).text
    check.record("note 用 Markdown 出力", len(note) > 500, f"{len(note)} 文字")

    backup = client.post("/api/system/db/backup").json()
    check.record("DB バックアップ", bool(backup.get("path")), str(backup.get("path")))
    telemetry = client.get("/api/system/telemetry").json()
    totals = (telemetry.get("stats") or {}).get("totals") or {}
    check.record("テレメトリ記録", int(totals.get("calls") or 0) > 0,
                 f"calls={totals.get('calls')} tokens={totals.get('total_tokens')} avg_ms={round(float(totals.get('avg_latency_ms') or 0))}")

    print("\n受け入れテスト: " + ("合格" if check.passed else "不合格"))
    return 0 if check.passed else 1


if __name__ == "__main__":
    sys.exit(main())
