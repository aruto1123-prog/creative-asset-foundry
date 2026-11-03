"""Manga Pipeline のモックLLMハンドラ。

``manga.dialogue`` の Synthesize / Refine 段に対し、ハーネスが context へ渡したコマ情報から
テンプレート台詞エンジン（別系統の言い回し）で PanelDialogueBatch を決定論的に合成する。
"""

from __future__ import annotations

from typing import Any

from core.llm_client import MockContext, register_mock_handler
from core.manga_pipeline.storyboard_generator import build_dialogue_batch

__all__ = ["mock_manga_dialogue"]


@register_mock_handler("manga.dialogue", stages=("synthesize", "refine"))
def mock_manga_dialogue(ctx: MockContext) -> dict[str, Any]:
    panels = ctx.context.get("panels") or []
    max_chars = int(ctx.context.get("max_chars", 40))
    seed = int(ctx.context.get("seed", 0))
    variant = 1 if ctx.stage == "synthesize" else 2
    batch = build_dialogue_batch(list(panels), seed=seed, max_chars=max_chars, variant_offset=variant)
    return batch.model_dump(mode="json")
