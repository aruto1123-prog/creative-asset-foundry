"""4段階思考増幅ハーネス（Multi-Stage Reasoning Harness）。

Step 1 ``Decompose``（本質要素・制約の分解）→ Step 2 ``Synthesize``（スキーマ厳密準拠ドラフト）→
Step 3 ``Critique``（ルーブリック自己批評）→ Step 4 ``Refine``（批評反映の最終版）を自動で回す。

- Decompose / Critique は補助段であり、失敗しても警告を残して続行する。
- Synthesize の失敗は ``HarnessError``。Refine の失敗はドラフトを採用して続行する。
- ``post_validate`` がドメイン固有の問題を返した場合、``max_refine_rounds`` まで追加 Refine を行う。
"""

from __future__ import annotations

import inspect
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable
from typing import Any, Generic, Literal

from pydantic import BaseModel, Field

from core.common import truncate
from core.config import HarnessSettings
from core.llm_client.client import LLMClient
from core.llm_client.errors import HarnessError, LLMError, StructuredOutputError
from core.llm_client.prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    DEEPSEEK_BASE_PERSONA,
    REFINE_DIRECTIVE,
    SYNTHESIZE_DIRECTIVE,
    build_critique_user_prompt,
    build_decompose_user_prompt,
    build_refine_user_prompt,
    build_synthesize_user_prompt,
)
from core.llm_client.types import (
    DEFAULT_RUBRIC,
    HarnessStage,
    RubricCriterion,
    StageLog,
    StructuredResult,
    T,
    TokenUsage,
    system_msg,
    user_msg,
)

__all__ = [
    "ElementItem",
    "ConstraintItem",
    "DecompositionResult",
    "CriterionScore",
    "CritiqueReport",
    "RubricCriterion",
    "DEFAULT_RUBRIC",
    "HarnessConfig",
    "StageLog",
    "HarnessResult",
    "MultiStageReasoningHarness",
]

logger = logging.getLogger(__name__)

_PAYLOAD_TEXT_LIMIT = 60_000
_THOUGHT_EXCERPT_LIMIT = 2_000


class ElementItem(BaseModel):
    """課題の中核要素。"""

    name: str = Field(min_length=1)
    description: str
    importance: float = Field(0.5, ge=0.0, le=1.0)


class ConstraintItem(BaseModel):
    """課題の制約。"""

    name: str = Field(min_length=1)
    description: str
    kind: Literal["hard", "soft"] = "hard"


class DecompositionResult(BaseModel):
    """Step 1 Decompose の出力。"""

    essence: str = Field(min_length=1)
    core_elements: list[ElementItem] = Field(min_length=1)
    constraints: list[ConstraintItem] = Field(min_length=1)
    hidden_assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)
    abstraction_targets: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class CriterionScore(BaseModel):
    """ルーブリック1観点の採点。"""

    key: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=10.0)
    rationale: str = ""


class CritiqueReport(BaseModel):
    """Step 3 Critique の出力。"""

    scores: list[CriterionScore] = Field(min_length=1)
    logical_contradictions: list[str] = Field(default_factory=list)
    shallow_points: list[str] = Field(default_factory=list)
    abstraction_gaps: list[str] = Field(default_factory=list)
    must_fix: list[str] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=100.0)
    verdict: Literal["accept", "revise"]
    summary: str = ""


class HarnessConfig(HarnessSettings):
    """ハーネス実行設定（``HarnessSettings`` と同一フィールド）。"""

    @classmethod
    def from_settings(cls, settings: HarnessSettings | None) -> "HarnessConfig":
        if settings is None:
            return cls()
        return cls.model_validate(settings.model_dump())

    def temperature_for(self, stage: str | HarnessStage, default: float = 0.3) -> float:
        key = stage.value if isinstance(stage, HarnessStage) else str(stage)
        value = self.stage_temperatures.get(key, default)
        return float(min(max(value, 0.0), 2.0))

    def planned_stages(self) -> list[str]:
        stages: list[str] = []
        if self.enable_decompose:
            stages.append(HarnessStage.DECOMPOSE.value)
        stages.append(HarnessStage.SYNTHESIZE.value)
        if self.enable_critique:
            stages.append(HarnessStage.CRITIQUE.value)
        if self.enable_refine:
            stages.append(HarnessStage.REFINE.value)
        return stages


class HarnessResult(BaseModel, Generic[T]):
    """ハーネス1回分の結果。"""

    run_id: str
    task: str
    output: T
    draft: T | None = None
    decomposition: DecompositionResult | None = None
    critique: CritiqueReport | None = None
    stage_logs: list[StageLog] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    total_latency_ms: float = 0.0
    refined: bool = False
    refine_rounds: int = 0
    warnings: list[str] = Field(default_factory=list)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _payload_to_text(payload: Any) -> str:
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, default=str)
    return truncate(text, _PAYLOAD_TEXT_LIMIT)


def _dump_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=2)


class MultiStageReasoningHarness:
    """Decompose → Synthesize → Critique → Refine を自動実行するハーネス。"""

    def __init__(self, client: LLMClient, config: HarnessConfig | None = None) -> None:
        self.client = client
        self.config = config or HarnessConfig()

    # -- ログ -------------------------------------------------------------

    @staticmethod
    def _log_from_result(stage: str, ordinal: int, result: StructuredResult[Any], summary: str) -> StageLog:
        return StageLog(
            stage=stage,
            ordinal=ordinal,
            latency_ms=result.total_latency_ms,
            usage=result.total_usage,
            attempts=result.attempts,
            success=True,
            summary=summary,
            thought_excerpt=truncate(result.response.reasoning or "", _THOUGHT_EXCERPT_LIMIT),
        )

    @staticmethod
    def _log_from_error(stage: str, ordinal: int, error: LLMError, started: float, summary: str) -> StageLog:
        usage = getattr(error, "usage", None)
        latency = getattr(error, "latency_ms", 0.0) or (time.perf_counter() - started) * 1000.0
        attempts = getattr(error, "attempts", 1) or 1
        return StageLog(
            stage=stage,
            ordinal=ordinal,
            latency_ms=float(latency),
            usage=usage if isinstance(usage, TokenUsage) else TokenUsage(),
            attempts=int(attempts),
            success=False,
            summary=summary,
            thought_excerpt="",
        )

    async def _emit(
        self,
        run_id: str,
        log: StageLog,
        logs: list[StageLog],
        warnings: list[str],
        on_stage: Callable[[StageLog], Any] | None,
    ) -> None:
        logs.append(log)
        telemetry = self.client.telemetry
        if telemetry is not None:
            try:
                telemetry.record_stage(run_id, log)
            except sqlite3.Error as exc:
                logger.warning("段ログの記録に失敗しました: %s", exc)
        if on_stage is not None:
            try:
                outcome = on_stage(log)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception as exc:  # 進捗通知の失敗で推論本体を止めない
                warnings.append(f"on_stage コールバックで例外: {exc}")
                logger.warning("on_stage コールバックで例外: %s", exc)

    # -- 本体 -------------------------------------------------------------

    async def run(
        self,
        *,
        task: str,
        objective: str,
        payload: Any,
        output_model: type[T],
        system_prompt: str,
        rubric: list[RubricCriterion] | None = None,
        context: dict[str, Any] | None = None,
        config: HarnessConfig | None = None,
        post_validate: Callable[[T], list[str]] | None = None,
        on_stage: Callable[[StageLog], Any] | None = None,
    ) -> HarnessResult[T]:
        """4段階ハーネスを実行し、最終構造化データと全段のログを返す。"""
        cfg = config or self.config
        criteria = list(rubric or DEFAULT_RUBRIC)
        rubric_dicts = [criterion.model_dump() for criterion in criteria]
        run_id = uuid.uuid4().hex
        payload_text = _payload_to_text(payload)
        base_context: dict[str, Any] = {**dict(context or {}), "objective": objective, "payload": _to_jsonable(payload)}
        target_schema = output_model.model_json_schema()
        schema_name = output_model.__name__
        persona_system = f"{DEEPSEEK_BASE_PERSONA}\n\n{system_prompt}".strip()
        logs: list[StageLog] = []
        warnings: list[str] = []
        total_usage = TokenUsage()
        run_started = time.perf_counter()
        ordinal = 0
        telemetry = self.client.telemetry
        if telemetry is not None:
            try:
                telemetry.start_run(run_id, task, objective, cfg.planned_stages())
            except sqlite3.Error as exc:
                logger.warning("実行開始の記録に失敗しました: %s", exc)

        def finish(status: str, critique_score: float | None, error: str | None = None) -> None:
            if telemetry is None:
                return
            try:
                telemetry.finish_run(
                    run_id,
                    status=status,
                    usage=total_usage,
                    latency_ms=(time.perf_counter() - run_started) * 1000.0,
                    critique_score=critique_score,
                    error=error,
                )
            except sqlite3.Error as exc:
                logger.warning("実行完了の記録に失敗しました: %s", exc)

        # Step 1: Decompose ------------------------------------------------
        decomposition: DecompositionResult | None = None
        if cfg.enable_decompose:
            ordinal += 1
            started = time.perf_counter()
            messages = [
                system_msg(f"{DEEPSEEK_BASE_PERSONA}\n\n{DECOMPOSE_SYSTEM_PROMPT}"),
                user_msg(build_decompose_user_prompt(objective, payload_text, schema_name, target_schema)),
            ]
            stage_context = {
                **base_context,
                "target_schema_name": schema_name,
                "target_required_fields": list(target_schema.get("required") or []),
            }
            try:
                result = await self.client.generate_structured(
                    DecompositionResult,
                    messages,
                    task=task,
                    stage=HarnessStage.DECOMPOSE,
                    temperature=cfg.temperature_for(HarnessStage.DECOMPOSE),
                    run_id=run_id,
                    context=stage_context,
                    max_repair_attempts=cfg.max_repair_attempts,
                )
            except LLMError as exc:
                warnings.append(f"decompose 段が失敗したため分解なしで続行: {exc}")
                log = self._log_from_error("decompose", ordinal, exc, started, f"失敗（分解なしで続行）: {exc}")
            else:
                decomposition = result.value
                log = self._log_from_result(
                    "decompose",
                    ordinal,
                    result,
                    f"中核要素 {len(decomposition.core_elements)} / 制約 {len(decomposition.constraints)}: "
                    f"{truncate(decomposition.essence, 160)}",
                )
            total_usage = total_usage + log.usage
            await self._emit(run_id, log, logs, warnings, on_stage)

        # Step 2: Synthesize -----------------------------------------------
        ordinal += 1
        started = time.perf_counter()
        decomposition_json = _dump_json(decomposition) if decomposition is not None else None
        messages = [
            system_msg(f"{persona_system}\n\n{SYNTHESIZE_DIRECTIVE}"),
            user_msg(build_synthesize_user_prompt(objective, payload_text, decomposition_json)),
        ]
        stage_context = {
            **base_context,
            "decomposition": decomposition.model_dump(mode="json") if decomposition is not None else None,
        }
        try:
            synth = await self.client.generate_structured(
                output_model,
                messages,
                task=task,
                stage=HarnessStage.SYNTHESIZE,
                temperature=cfg.temperature_for(HarnessStage.SYNTHESIZE),
                run_id=run_id,
                context=stage_context,
                max_repair_attempts=cfg.max_repair_attempts,
            )
        except LLMError as exc:
            log = self._log_from_error("synthesize", ordinal, exc, started, f"失敗: {exc}")
            total_usage = total_usage + log.usage
            await self._emit(run_id, log, logs, warnings, on_stage)
            finish("failed", None, str(exc))
            raise HarnessError(f"{task}: synthesize 段で失敗しました: {exc}", stage="synthesize", run_id=run_id) from exc
        draft = synth.value
        log = self._log_from_result(
            "synthesize",
            ordinal,
            synth,
            f"{schema_name} ドラフト生成（試行 {synth.attempts} 回{'・構文修復あり' if synth.syntax_repaired else ''}）",
        )
        total_usage = total_usage + log.usage
        await self._emit(run_id, log, logs, warnings, on_stage)
        draft_dump = draft.model_dump(mode="json")

        # Step 3: Critique -------------------------------------------------
        critique: CritiqueReport | None = None
        if cfg.enable_critique:
            ordinal += 1
            started = time.perf_counter()
            messages = [
                system_msg(f"{DEEPSEEK_BASE_PERSONA}\n\n{CRITIQUE_SYSTEM_PROMPT}"),
                user_msg(
                    build_critique_user_prompt(
                        objective, payload_text, _dump_json(draft), rubric_dicts, cfg.critique_accept_score
                    )
                ),
            ]
            stage_context = {
                **base_context,
                "draft": draft_dump,
                "rubric": rubric_dicts,
                "accept_score": cfg.critique_accept_score,
            }
            try:
                crit = await self.client.generate_structured(
                    CritiqueReport,
                    messages,
                    task=task,
                    stage=HarnessStage.CRITIQUE,
                    temperature=cfg.temperature_for(HarnessStage.CRITIQUE),
                    run_id=run_id,
                    context=stage_context,
                    max_repair_attempts=cfg.max_repair_attempts,
                )
            except LLMError as exc:
                warnings.append(f"critique 段が失敗したため批評なしで続行: {exc}")
                log = self._log_from_error("critique", ordinal, exc, started, f"失敗（批評なしで続行）: {exc}")
            else:
                critique = crit.value
                log = self._log_from_result(
                    "critique",
                    ordinal,
                    crit,
                    f"総合 {critique.overall_score:.1f} 点 / 判定 {critique.verdict} / 必須修正 {len(critique.must_fix)} 件",
                )
            total_usage = total_usage + log.usage
            await self._emit(run_id, log, logs, warnings, on_stage)

        # Step 4: Refine ---------------------------------------------------
        final = draft
        refined = False
        refine_rounds = 0
        accepted = (
            critique is not None
            and critique.verdict == "accept"
            and critique.overall_score >= cfg.critique_accept_score
        )
        should_refine = cfg.enable_refine and not (cfg.skip_refine_when_accepted and accepted)
        critique_dump = critique.model_dump(mode="json") if critique is not None else None

        async def refine_once(current: T, must_fix: list[str], label: str) -> T | None:
            nonlocal ordinal, total_usage
            ordinal += 1
            refine_started = time.perf_counter()
            refine_messages = [
                system_msg(f"{persona_system}\n\n{REFINE_DIRECTIVE}"),
                user_msg(
                    build_refine_user_prompt(
                        objective,
                        payload_text,
                        _dump_json(current),
                        json.dumps(critique_dump, ensure_ascii=False, indent=2) if critique_dump else None,
                        must_fix,
                    )
                ),
            ]
            refine_context = {
                **base_context,
                "draft": current.model_dump(mode="json"),
                "critique": critique_dump,
                "must_fix": list(must_fix),
            }
            try:
                refined_result = await self.client.generate_structured(
                    output_model,
                    refine_messages,
                    task=task,
                    stage=HarnessStage.REFINE,
                    temperature=cfg.temperature_for(HarnessStage.REFINE),
                    run_id=run_id,
                    context=refine_context,
                    max_repair_attempts=cfg.max_repair_attempts,
                )
            except LLMError as exc:
                warnings.append(f"refine 段（{label}）が失敗したため直前の版を採用: {exc}")
                refine_log = self._log_from_error("refine", ordinal, exc, refine_started, f"失敗（{label}）: {exc}")
                total_usage = total_usage + refine_log.usage
                await self._emit(run_id, refine_log, logs, warnings, on_stage)
                return None
            refine_log = self._log_from_result(
                "refine",
                ordinal,
                refined_result,
                f"{label}: 修正事項 {len(must_fix)} 件を反映（試行 {refined_result.attempts} 回）",
            )
            total_usage = total_usage + refine_log.usage
            await self._emit(run_id, refine_log, logs, warnings, on_stage)
            return refined_result.value

        if should_refine:
            base_fixes = list(critique.must_fix) if critique is not None else []
            candidate = await refine_once(final, base_fixes, "批評反映")
            if candidate is not None:
                final = candidate
                refined = True
                refine_rounds += 1

        if post_validate is not None:
            issues = [str(issue) for issue in (post_validate(final) or [])]
            extra_rounds = 0
            while issues and cfg.enable_refine and extra_rounds < cfg.max_refine_rounds:
                extra_rounds += 1
                candidate = await refine_once(final, issues, f"検証問題の追加修正 {extra_rounds}")
                if candidate is None:
                    break
                final = candidate
                refined = True
                refine_rounds += 1
                issues = [str(issue) for issue in (post_validate(final) or [])]
            if issues:
                warnings.append(f"post_validate の問題が未解消: {issues[:5]}")

        finish("succeeded", critique.overall_score if critique is not None else None)
        result_type = HarnessResult[output_model]  # type: ignore[valid-type]
        return result_type(
            run_id=run_id,
            task=task,
            output=final,
            draft=draft,
            decomposition=decomposition,
            critique=critique,
            stage_logs=logs,
            usage=total_usage,
            total_latency_ms=(time.perf_counter() - run_started) * 1000.0,
            refined=refined,
            refine_rounds=refine_rounds,
            warnings=warnings,
        )


def structured_error_summary(error: StructuredOutputError) -> dict[str, Any]:
    """``StructuredOutputError`` を API 応答等で扱いやすい dict に変換する。"""
    return {
        "message": str(error),
        "schema_name": error.schema_name,
        "attempts": error.attempts,
        "validation_errors": error.validation_errors,
        "raw_text": truncate(error.raw_text or "", 2000),
    }
