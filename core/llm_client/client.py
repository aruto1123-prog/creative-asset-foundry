"""高水準 LLM クライアント。

- ``chat``: 単発の対話呼び出し（``<think>`` 除去・テレメトリ記録）
- ``generate_structured``: Pydantic モデルに厳密準拠した構造化生成。
  応答から JSON を抽出・修復し、検証に失敗した場合は検証エラーをフィードバックして自己修復リトライする。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from core.common import truncate
from core.llm_client.adapters import AutoFallbackAdapter, BaseLLMAdapter
from core.llm_client.errors import JSONExtractionError, LLMError, StructuredOutputError
from core.llm_client.json_repair import format_validation_errors, parse_json_lenient_with_info, strip_reasoning_tags
from core.llm_client.prompts import build_repair_prompt, build_schema_instruction
from core.llm_client.telemetry import TelemetryStore
from core.llm_client.types import (
    ChatMessage,
    HarnessStage,
    LLMRequest,
    LLMResponse,
    StructuredResult,
    T,
    TokenUsage,
    assistant_msg,
    system_msg,
    user_msg,
)

__all__ = ["LLMClient"]

logger = logging.getLogger(__name__)

_FAILED_OUTPUT_ECHO_LIMIT = 6000


def _with_schema_instruction(messages: list[ChatMessage], schema_name: str, schema: dict[str, Any]) -> list[ChatMessage]:
    """スキーマ指示を最後の system メッセージ末尾へ追記（無ければ先頭に system を追加）する。"""
    instruction = build_schema_instruction(schema_name, schema)
    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        if result[index].role == "system":
            merged = f"{result[index].content.rstrip()}\n\n{instruction}"
            result[index] = system_msg(merged)
            return result
    return [system_msg(instruction), *result]


def _normalize_response(response: LLMResponse) -> LLMResponse:
    """本文に残った ``<think>`` を思考ログへ移した応答を返す。"""
    lowered = response.content.lower()
    if "think" not in lowered or "<" not in lowered:
        return response
    content, thoughts = strip_reasoning_tags(response.content)
    if not thoughts and content == response.content:
        return response
    reasoning = response.reasoning
    if thoughts and thoughts.strip() not in (reasoning or ""):
        reasoning = f"{reasoning}\n{thoughts}".strip() if reasoning else thoughts
    return response.model_copy(update={"content": content, "reasoning": reasoning})


def _request_excerpt(request: LLMRequest) -> str:
    tail = request.messages[-2:] if len(request.messages) >= 2 else request.messages
    return "\n".join(f"[{message.role}] {message.content}" for message in tail)


class LLMClient:
    """アダプタを包み、構造化生成・自己修復・テレメトリを提供するクライアント。"""

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        telemetry: TelemetryStore | None = None,
        default_model: str | None = None,
        max_repair_attempts: int = 3,
    ) -> None:
        self.adapter = adapter
        self.telemetry = telemetry
        self.default_model = default_model
        self.max_repair_attempts = max(0, int(max_repair_attempts))

    @property
    def provider(self) -> str:
        return self.adapter.provider

    @property
    def effective_provider(self) -> str:
        """自動選択アダプタの場合は実際に選択されたプロバイダ名。"""
        if isinstance(self.adapter, AutoFallbackAdapter):
            return self.adapter.selected_provider
        return self.adapter.provider

    @property
    def fallback_active(self) -> bool:
        return bool(getattr(self.adapter, "fallback_active", False))

    @property
    def model(self) -> str:
        return self.default_model or self.adapter.model

    # -- 内部 -------------------------------------------------------------

    def _record(
        self,
        request: LLMRequest,
        response: LLMResponse | None,
        *,
        success: bool,
        error: str | None,
        latency_ms: float,
    ) -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.record_call(
                run_id=request.run_id,
                task=request.task,
                stage=request.stage.value,
                provider=response.provider if response is not None else self.effective_provider,
                model=response.model if response is not None else (request.model or self.model),
                attempt=request.attempt,
                success=success,
                error=error,
                usage=response.usage if response is not None else TokenUsage(),
                latency_ms=response.latency_ms if response is not None else latency_ms,
                request_excerpt=_request_excerpt(request),
                response_excerpt=response.content if response is not None else "",
                reasoning_log=response.reasoning if response is not None else "",
            )
        except sqlite3.Error as exc:
            logger.warning("テレメトリ記録に失敗しました: %s", exc)

    async def _invoke(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self.adapter.complete(request)
        except LLMError as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record(request, None, success=False, error=f"{type(exc).__name__}: {exc}", latency_ms=latency_ms)
            raise
        return _normalize_response(response)

    # -- 公開 API ---------------------------------------------------------

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        task: str = "generic",
        stage: HarnessStage = HarnessStage.SINGLE,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """単発のチャット補完。"""
        request = LLMRequest(
            messages=list(messages),
            model=self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            task=task,
            stage=stage,
            context=dict(context or {}),
            run_id=run_id,
            json_mode=json_mode,
        )
        response = await self._invoke(request)
        self._record(request, response, success=True, error=None, latency_ms=response.latency_ms)
        return response

    async def generate_structured(
        self,
        output_model: type[T],
        messages: list[ChatMessage],
        *,
        task: str,
        stage: HarnessStage = HarnessStage.SINGLE,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
        max_repair_attempts: int | None = None,
    ) -> StructuredResult[T]:
        """``output_model`` に厳密準拠した構造化出力を生成する（自己修復リトライ付き）。"""
        schema = output_model.model_json_schema()
        schema_name = output_model.__name__
        repairs = self.max_repair_attempts if max_repair_attempts is None else max(0, int(max_repair_attempts))
        total_attempts = repairs + 1
        conversation = _with_schema_instruction(list(messages), schema_name, schema)
        base_context = dict(context or {})
        total_usage = TokenUsage()
        total_latency = 0.0
        last_errors = ""
        last_text = ""
        for attempt in range(1, total_attempts + 1):
            attempt_context = dict(base_context)
            if attempt > 1:
                attempt_context["repair"] = True
                attempt_context["previous_errors"] = last_errors
                attempt_context["original_stage"] = stage.value
            request = LLMRequest(
                messages=conversation,
                model=self.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_schema=schema,
                schema_name=schema_name,
                task=task,
                stage=stage,
                context=attempt_context,
                run_id=run_id,
                attempt=attempt,
                json_mode=True,
            )
            response = await self._invoke(request)
            total_usage = total_usage + response.usage
            total_latency += response.latency_ms
            text = response.content
            try:
                data, repaired = parse_json_lenient_with_info(text)
                value = output_model.model_validate(data)
            except JSONExtractionError as exc:
                errors_text = format_validation_errors(exc)
            except ValidationError as exc:
                errors_text = format_validation_errors(exc)
            except (ValueError, TypeError) as exc:
                errors_text = f"- ({type(exc).__name__}): {exc}"
            else:
                self._record(request, response, success=True, error=None, latency_ms=response.latency_ms)
                result_type = StructuredResult[output_model]  # type: ignore[valid-type]
                return result_type(
                    value=value,
                    response=response,
                    attempts=attempt,
                    raw_text=text,
                    syntax_repaired=repaired,
                    total_usage=total_usage,
                    total_latency_ms=total_latency,
                )
            self._record(request, response, success=False, error=errors_text, latency_ms=response.latency_ms)
            last_errors = errors_text
            last_text = text
            logger.info("構造化出力の検証に失敗（%s, attempt %s/%s）", schema_name, attempt, total_attempts)
            echo = truncate(text, _FAILED_OUTPUT_ECHO_LIMIT) or "（空の応答）"
            conversation = [*conversation, assistant_msg(echo), user_msg(build_repair_prompt(errors_text, schema_name))]
        raise StructuredOutputError(
            f"{schema_name} の構造化出力が {total_attempts} 回の試行で検証を通過しませんでした",
            raw_text=last_text,
            validation_errors=last_errors,
            attempts=total_attempts,
            schema_name=schema_name,
            usage=total_usage,
            latency_ms=total_latency,
        )

    async def structured(
        self,
        output_model: type[T],
        *,
        system: str,
        user: str,
        task: str,
        stage: HarnessStage = HarnessStage.SINGLE,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        context: dict[str, Any] | None = None,
    ) -> StructuredResult[T]:
        """system / user の2メッセージから構造化生成する簡易版。"""
        return await self.generate_structured(
            output_model,
            [system_msg(system), user_msg(user)],
            task=task,
            stage=stage,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context,
        )

    async def health_check(self) -> dict[str, Any]:
        """アダプタの疎通確認。"""
        return await self.adapter.health_check()

    async def aclose(self) -> None:
        """アダプタ資源を解放する。"""
        await self.adapter.aclose()


def is_structured_model(value: Any) -> bool:
    """値が Pydantic モデルのクラスであるか。"""
    return isinstance(value, type) and issubclass(value, BaseModel)
