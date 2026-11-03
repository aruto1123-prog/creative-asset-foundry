"""Structure Engine 用モック LLM ハンドラ。

``MockLLMAdapter`` が初回呼び出し時に遅延 import し、以下のタスクに対して
スキーマ完全準拠かつ入力内容から決定論的に導出されたリアルな応答を返す。

- ``structure.extract``（synthesize / refine）: 抽象化済みテキストからのヒューリスティック5階層抽出。
  refine 段では批評の必須修正件数を反映した改善（要約への反映注記・タグ付与）を行う。
- ``structure.transpile``（synthesize / refine）: 決定論的設計（baseline）から文章部分を構成し、
  refine 段では各セクションの感情目標を本文へ織り込んで具体化する。
"""

from __future__ import annotations

from typing import Any

from core.common import truncate
from core.llm_client import MockContext, register_mock_handler
from core.structure_engine.extractor import heuristic_extract

__all__ = ["mock_structure_extract", "mock_structure_transpile"]


def _payload_text(ctx: MockContext) -> str:
    text = str(ctx.context.get("abstracted_text") or "")
    if text.strip():
        return text
    payload = ctx.context.get("payload")
    if isinstance(payload, dict):
        candidate = str(payload.get("abstracted_text") or "")
        if candidate.strip():
            return candidate
    objective = str(ctx.context.get("objective") or "")
    return objective if objective.strip() else "主体が目標を追い、限界の後に前提が反転し、代償を払って突破する。"


@register_mock_handler("structure.extract", stages=("synthesize", "refine"))
def mock_structure_extract(ctx: MockContext) -> dict[str, Any]:
    layers = heuristic_extract(
        _payload_text(ctx),
        source_kind=str(ctx.context.get("source_kind") or "synopsis"),
        title_hint=ctx.context.get("title_hint"),
        genre_signals=ctx.context.get("genre_signals") or None,
    )
    data = layers.model_dump(mode="json")
    data["wave"].pop("information_asymmetry", None)
    if ctx.stage == "refine":
        must_fix = [str(item) for item in (ctx.context.get("must_fix") or []) if str(item).strip()]
        if must_fix:
            data["summary"] = truncate(
                f"{data['summary']}（批評で指摘された {len(must_fix)} 件の修正点を反映し、因果と抽象度を再点検済み）", 4000
            )
            if "批評反映済み" not in data["tags"]:
                data["tags"] = [*data["tags"], "批評反映済み"]
    return data


@register_mock_handler("structure.transpile", stages=("synthesize", "refine"))
def mock_structure_transpile(ctx: MockContext) -> dict[str, Any]:
    baseline = ctx.context.get("baseline") or {}
    sections_in = baseline.get("sections") or []
    sections: list[dict[str, Any]] = []
    for section in sections_in:
        content = str(section.get("content") or section.get("heading") or "構成要素")
        emotional_goal = str(section.get("emotional_goal") or "受け手の関心を維持する")
        if ctx.stage == "refine":
            content = f"{content} 受け手に残す感情: {emotional_goal}。"
        sections.append(
            {
                "index": int(section.get("index", len(sections))),
                "heading": truncate(str(section.get("heading") or f"セクション{len(sections) + 1}"), 200),
                "content": truncate(content, 4000),
                "emotional_goal": truncate(emotional_goal, 400),
                "information_reveal": truncate(str(section.get("information_reveal") or "情報量を維持する。"), 600),
            }
        )
    if not sections:
        sections.append(
            {
                "index": 0,
                "heading": "全体構成",
                "content": "純粋骨格の力学と波形に沿って構成する。",
                "emotional_goal": "受け手の関心を維持する",
                "information_reveal": "情報量を維持する。",
            }
        )
    return {
        "title": truncate(str(baseline.get("title") or "トランスパイル設計"), 200),
        "logline": truncate(str(baseline.get("logline") or "純粋骨格を対象領域の設計へ変換する。"), 600),
        "sections": sections,
    }
