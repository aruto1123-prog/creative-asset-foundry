"""LLM 基盤の例外階層。

``LLMError`` を基底に、接続障害・応答異常・JSON 抽出失敗・構造化出力失敗・ハーネス失敗を区別する。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LLMError",
    "LLMConnectionError",
    "LLMResponseError",
    "JSONExtractionError",
    "StructuredOutputError",
    "HarnessError",
]


class LLMError(Exception):
    """CAF の LLM 基盤が送出する全例外の基底クラス。"""


class LLMConnectionError(LLMError):
    """エンドポイントへ到達できない、または一時障害（429/5xx/通信断）が再試行上限まで続いた。"""


class LLMResponseError(LLMError):
    """サーバが恒久的な異常応答（429 以外の 4xx・不正な JSON・choices 欠落など）を返した。"""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class JSONExtractionError(LLMError):
    """応答テキストから JSON を抽出・修復できなかった。"""

    def __init__(self, message: str, *, text: str = "") -> None:
        super().__init__(message)
        self.text = text


class StructuredOutputError(LLMError):
    """自己修復リトライを尽くしても出力がスキーマ検証を通過しなかった。"""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        validation_errors: str = "",
        attempts: int = 0,
        schema_name: str | None = None,
        usage: Any = None,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.validation_errors = validation_errors
        self.attempts = attempts
        self.schema_name = schema_name
        self.usage = usage
        self.latency_ms = latency_ms


class HarnessError(LLMError):
    """多段階ハーネスの必須段（Synthesize）が失敗した。"""

    def __init__(self, message: str, *, stage: str | None = None, run_id: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.run_id = run_id
