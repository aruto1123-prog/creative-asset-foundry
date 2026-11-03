"""LLM 基盤の型定義。

OpenAI 互換 API / Ollama / モックの全アダプタが共有するリクエスト・レスポンス・使用量の型と、
4段階ハーネスの段階ログ（``StageLog``）・ルーブリック（``RubricCriterion``）を定義する。
``StageLog`` / ``RubricCriterion`` は循環 import を避けるためここに置き、``harness`` から再エクスポートする。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

__all__ = [
    "T",
    "ChatMessage",
    "system_msg",
    "user_msg",
    "assistant_msg",
    "HarnessStage",
    "LLMRequest",
    "TokenUsage",
    "LLMResponse",
    "StructuredResult",
    "StageLog",
    "RubricCriterion",
    "DEFAULT_RUBRIC",
]

T = TypeVar("T", bound=BaseModel)


class ChatMessage(BaseModel):
    """チャット形式の1メッセージ。"""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


def system_msg(text: str) -> ChatMessage:
    """system ロールのメッセージを生成する。"""
    return ChatMessage(role="system", content=text)


def user_msg(text: str) -> ChatMessage:
    """user ロールのメッセージを生成する。"""
    return ChatMessage(role="user", content=text)


def assistant_msg(text: str) -> ChatMessage:
    """assistant ロールのメッセージを生成する。"""
    return ChatMessage(role="assistant", content=text)


class HarnessStage(str, Enum):
    """推論段階。``SINGLE`` はハーネス外の単発呼び出し、``REPAIR`` は明示的な修復要求。"""

    DECOMPOSE = "decompose"
    SYNTHESIZE = "synthesize"
    CRITIQUE = "critique"
    REFINE = "refine"
    REPAIR = "repair"
    SINGLE = "single"


class LLMRequest(BaseModel):
    """アダプタへ渡す正規化済みリクエスト。

    ``context`` はモックハンドラ用の構造化コンテキストであり、実 LLM には送信されない。
    """

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1)
    response_schema: dict[str, Any] | None = None
    schema_name: str | None = None
    task: str = "generic"
    stage: HarnessStage = HarnessStage.SINGLE
    context: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    attempt: int = Field(1, ge=1)
    json_mode: bool = False


class TokenUsage(BaseModel):
    """トークン使用量。

    OpenAI 互換 API の慣行に従い ``completion_tokens`` は推論（思考）トークンを含み、
    ``reasoning_tokens`` はその内訳である。したがって ``total_tokens = prompt + completion``。
    """

    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    reasoning_tokens: int = Field(0, ge=0)
    estimated: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            estimated=self.estimated or other.estimated,
        )


class LLMResponse(BaseModel):
    """アダプタからの応答。``reasoning`` は reasoning_content / ``<think>`` 由来の思考ログ。"""

    content: str
    reasoning: str = ""
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(0.0, ge=0.0)
    finish_reason: str = "stop"
    raw: dict[str, Any] | None = None


class StructuredResult(BaseModel, Generic[T]):
    """構造化生成の結果。``total_*`` は自己修復リトライを含む全試行の合計。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: T
    response: LLMResponse
    attempts: int = Field(1, ge=1)
    raw_text: str = ""
    syntax_repaired: bool = False
    total_usage: TokenUsage = Field(default_factory=TokenUsage)
    total_latency_ms: float = Field(0.0, ge=0.0)


class StageLog(BaseModel):
    """ハーネス各段の実行ログ。"""

    stage: str
    ordinal: int = Field(ge=1)
    latency_ms: float = Field(0.0, ge=0.0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    attempts: int = Field(0, ge=0)
    success: bool = True
    summary: str = ""
    thought_excerpt: str = ""


class RubricCriterion(BaseModel):
    """Critique 段の評価観点。"""

    key: str = Field(min_length=1)
    description: str
    weight: float = Field(1.0, gt=0.0)


DEFAULT_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        key="logical_consistency",
        description="因果の順序・数値の整合・参照の実在性に矛盾がないか（後の出来事が原因になっていない、範囲外の値がない）",
        weight=1.2,
    ),
    RubricCriterion(
        key="structural_depth",
        description="表層の言い換えに留まらず、力学・因果・認知メカニズムの層まで掘り下げられているか",
        weight=1.2,
    ),
    RubricCriterion(
        key="abstraction_quality",
        description="固有名詞やジャンル固有の装飾を剥がし、他分野へ転用可能な抽象度に到達しているか",
        weight=1.0,
    ),
    RubricCriterion(
        key="schema_fidelity",
        description="スキーマの全必須項目・列挙値・数値範囲・配列長・参照整合性を満たしているか",
        weight=1.0,
    ),
    RubricCriterion(
        key="actionability",
        description="下流工程（合成・転用・生成・執筆）がそのまま利用できる具体性と一貫性があるか",
        weight=0.8,
    ),
]
