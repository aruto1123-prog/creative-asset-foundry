"""Style Alchemy のモックLLMハンドラ（決定論的ベースラインからリアルな応答を合成）。

``core.llm_client.mock.MOCK_HANDLER_MODULES`` に登録されており、モックアダプタの初回呼び出し時に読み込まれる。
"""

from __future__ import annotations

from typing import Any

from core.llm_client import MockContext, register_mock_handler
from core.style_alchemy.dataset_curator import CaptionRecord, DatasetCurator

__all__ = ["mock_caption_cleanse"]


@register_mock_handler("style.caption_cleanse", stages=("synthesize", "refine"))
def mock_caption_cleanse(ctx: MockContext) -> dict[str, Any]:
    """ルールベース精製器の出力を、LLM が返すべき CleansedCaptionBatch として返す。"""
    records_payload = ctx.context.get("records")
    if records_payload is None:
        payload = ctx.context.get("payload") or {}
        records_payload = payload.get("records", []) if isinstance(payload, dict) else []
    records = [CaptionRecord.model_validate(item) for item in records_payload or []]
    trigger = str(ctx.context.get("trigger_token") or "caf_style")
    curator = DatasetCurator(trigger_token=trigger)
    return {"captions": [item.model_dump(mode="json") for item in curator.cleanse(records)]}
