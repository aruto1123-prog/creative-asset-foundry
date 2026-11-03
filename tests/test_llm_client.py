"""Phase 1: ``core.llm_client`` の単体テスト（決定論的・外部ネットワーク非依存）。"""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

from core.config import HarnessSettings, LLMSettings
from core.llm_client import (
    AutoFallbackAdapter,
    CritiqueReport,
    DecompositionResult,
    HarnessConfig,
    HarnessError,
    HarnessStage,
    JSONExtractionError,
    JsonSchemaSynthesizer,
    LLMClient,
    LLMConnectionError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    MockLLMAdapter,
    MultiStageReasoningHarness,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    StageLog,
    StructuredOutputError,
    TelemetryStore,
    TokenUsage,
    create_adapter,
    create_harness,
    create_llm_client,
    ensure_mock_handler_modules_loaded,
    estimate_messages_tokens,
    estimate_tokens,
    extract_json_block,
    format_validation_errors,
    get_mock_handler,
    parse_json_lenient,
    parse_json_lenient_with_info,
    register_mock_handler,
    registered_mock_handlers,
    repair_json_text,
    reset_mock_handlers_loaded,
    resolve_provider,
    strip_reasoning_tags,
    system_msg,
    unregister_mock_handler,
    user_msg,
)
from core.llm_client import mock as mock_module

# ---------------------------------------------------------------------------
# テスト用モデル
# ---------------------------------------------------------------------------


class Echo(BaseModel):
    value: str


class Rich(BaseModel):
    value: str
    flag: bool
    items: list[int]


class Inner(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    kind: Literal["agent", "desire", "lack"]
    weight: float = Field(0.5, ge=0.0, le=1.0)
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,15}$")


class Mode(str, Enum):
    FAST = "fast"
    DEEP = "deep"


class Outer(BaseModel):
    title: str = Field(min_length=3, max_length=60)
    description: str
    score: float = Field(ge=0, le=100)
    count: int = Field(ge=1, le=5)
    ratio: float = Field(gt=0.0, lt=1.0)
    multiple: int = Field(multiple_of=5, ge=10, le=50)
    tags: list[str] = Field(min_length=2, max_length=4)
    items: list[Inner] = Field(min_length=1)
    mode: Mode
    optional_inner: Inner | None = None
    mapping: dict[str, float] = Field(default_factory=dict)
    pair: tuple[int, str]
    created_at: datetime
    uid: uuid.UUID
    flag: bool
    version_code: str = Field(pattern=r"^v\d+\.\d+$")


class Tree(BaseModel):
    name: str
    children: list["Tree"] = Field(default_factory=list)


Tree.model_rebuild()


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_handler_modules(monkeypatch: pytest.MonkeyPatch):
    """並行開発中の他モジュールのハンドラを読み込まず、Phase 1 単体で検証する。"""
    monkeypatch.setattr(mock_module, "MOCK_HANDLER_MODULES", ())
    reset_mock_handlers_loaded()
    yield
    reset_mock_handlers_loaded()


@pytest.fixture
def handlers():
    """テスト専用ハンドラを登録し、終了時に解除する。"""
    registered: list[tuple[str, Any]] = []

    def register(task: str, stages: Any = "*"):
        def decorator(func):
            register_mock_handler(task, stages)(func)
            registered.append((task, stages))
            return func

        return decorator

    yield register
    for task, stages in registered:
        unregister_mock_handler(task, stages)


def _request(
    task: str = "test.basic",
    stage: HarnessStage = HarnessStage.SINGLE,
    schema_model: type[BaseModel] | None = None,
    content: str = "入力テキスト",
    context: dict[str, Any] | None = None,
) -> LLMRequest:
    return LLMRequest(
        messages=[system_msg("system"), user_msg(content)],
        task=task,
        stage=stage,
        response_schema=schema_model.model_json_schema() if schema_model else None,
        schema_name=schema_model.__name__ if schema_model else None,
        context=dict(context or {}),
    )


def _completion_body(content: str, reasoning: str | None = None, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    body: dict[str, Any] = {
        "id": "cmpl-test",
        "object": "chat.completion",
        "model": "DeepSeek-V4.1-Flash",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


# ---------------------------------------------------------------------------
# json_repair
# ---------------------------------------------------------------------------


def test_strip_reasoning_tags_closed_block() -> None:
    content, thoughts = strip_reasoning_tags('<think>分析中</think>\n{"a": 1}')
    assert content == '{"a": 1}'
    assert thoughts == "分析中"


def test_strip_reasoning_tags_unclosed_block() -> None:
    content, thoughts = strip_reasoning_tags("<think>途中で切れた思考 {未完")
    assert content == ""
    assert "途中で切れた思考" in thoughts


def test_strip_reasoning_tags_leading_close_tag_only() -> None:
    content, thoughts = strip_reasoning_tags('前置きの思考</think>{"ok": true}')
    assert content == '{"ok": true}'
    assert thoughts == "前置きの思考"


def test_extract_json_block_prefers_fenced_block() -> None:
    text = '説明 {"noise": 0}\n```json\n{"a": [1, 2], "b": "x"}\n```\n以上'
    assert json.loads(extract_json_block(text) or "") == {"a": [1, 2], "b": "x"}


def test_extract_json_block_ignores_brackets_and_fences_inside_strings() -> None:
    text = 'prefix {"a": "x}y", "code": "```mermaid\\nA-->B\\n```", "b": {"c": [1]}} suffix'
    assert json.loads(extract_json_block(text) or "") == {
        "a": "x}y",
        "code": "```mermaid\nA-->B\n```",
        "b": {"c": [1]},
    }


def test_extract_json_block_returns_none_without_brackets() -> None:
    assert extract_json_block("括弧を含まない文章です。") is None


REPAIR_CASES: list[tuple[str, Any]] = [
    ('{"a": 1, "b": [1, 2,],}', {"a": 1, "b": [1, 2]}),
    ('{\n // 行コメント\n "a": 1, /* ブロック */ "b": 2\n}', {"a": 1, "b": 2}),
    ("{'a': 'it's ok', 'b': 'x'}", {"a": "it's ok", "b": "x"}),
    ('{a: 1, b_c: "x"}', {"a": 1, "b_c": "x"}),
    ('{"a": True, "b": None, "c": False}', {"a": True, "b": None, "c": False}),
    ('{"a": undefined, "b": NaN, "c": -Infinity}', {"a": None, "b": None, "c": None}),
    ('{"status": in progress, "k": 構造}', {"status": "in progress", "k": "構造"}),
    ('{"a": "line1\nline2\tend"}', {"a": "line1\nline2\tend"}),
    ('{"a": "he said "hi" to me", "b": 1}', {"a": 'he said "hi" to me', "b": 1}),
    ('{\n"a": 1\n"b": 2\n}', {"a": 1, "b": 2}),
    ('[\n{"a": 1}\n{"b": 2}\n]', [{"a": 1}, {"b": 2}]),
    ('{"a": {"b": [1, 2', {"a": {"b": [1, 2]}}),
    ('{"a": "unfinished', {"a": "unfinished"}),
    ('{"a": 1, "b":', {"a": 1, "b": None}),
    ('{"a": 1, "b"', {"a": 1, "b": None}),
    ("{“a”: “値”}", {"a": "値"}),
    ('{"a": "“引用”", "b": 1,}', {"a": "“引用”", "b": 1}),
    ('{"a": [1, 2}', {"a": [1, 2]}),
    ('{"a": +1, "b": .5, "c": 007, "d": 1.}', {"a": 1, "b": 0.5, "c": 7, "d": 1}),
    ('{"date": 2026-01-01}', {"date": "2026-01-01"}),
    ('```json\n{"x": [1, 2,\n```', {"x": [1, 2]}),
]


@pytest.mark.parametrize(("broken", "expected"), REPAIR_CASES)
def test_parse_json_lenient_repairs_broken_json(broken: str, expected: Any) -> None:
    data, repaired = parse_json_lenient_with_info(broken)
    assert data == expected
    assert repaired is True


def test_repair_json_text_output_is_strict_json() -> None:
    repaired = repair_json_text('{name: "構造", list: [1, 2,], nested: {ok: True,}')
    assert json.loads(repaired) == {"name": "構造", "list": [1, 2], "nested": {"ok": True}}


def test_parse_json_lenient_reports_no_repair_for_clean_and_fenced_json() -> None:
    assert parse_json_lenient_with_info('{"a": 1}') == ({"a": 1}, False)
    assert parse_json_lenient_with_info('<think>x</think>```json\n{"a": 1}\n```') == ({"a": 1}, False)
    assert parse_json_lenient_with_info('{"a": 1}}]') == ({"a": 1}, False)


def test_parse_json_lenient_recovers_answer_inside_unclosed_think() -> None:
    assert parse_json_lenient('<think>考察の後に答える {"a": 2}') == {"a": 2}


def test_parse_json_lenient_raises_on_prose_and_empty() -> None:
    with pytest.raises(JSONExtractionError):
        parse_json_lenient("申し訳ありませんが出力できません。")
    with pytest.raises(JSONExtractionError):
        parse_json_lenient("   ")


def test_format_validation_errors_lists_paths_and_inputs() -> None:
    with pytest.raises(ValidationError) as info:
        Inner.model_validate({"id": "Bad Id", "kind": "other", "code": "X"})
    text = format_validation_errors(info.value)
    assert "- id:" in text
    assert "- kind:" in text
    assert "- code:" in text
    assert "input=" in text
    assert format_validation_errors(JSONExtractionError("壊れています")).startswith("- (json):")


# ---------------------------------------------------------------------------
# トークン推定・型
# ---------------------------------------------------------------------------


def test_estimate_tokens_mixed_scripts() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("構造的反転") == 5
    assert estimate_messages_tokens([user_msg("abcd")]) == 2 + 4 + 1


def test_token_usage_addition_and_total() -> None:
    total = TokenUsage(prompt_tokens=3, completion_tokens=4, reasoning_tokens=1) + TokenUsage(
        prompt_tokens=1, completion_tokens=1, estimated=True
    )
    assert total == TokenUsage(prompt_tokens=4, completion_tokens=5, reasoning_tokens=1, estimated=True)
    assert total.total_tokens == 9
    assert total.model_dump()["total_tokens"] == 9


# ---------------------------------------------------------------------------
# JsonSchemaSynthesizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [Outer, Tree, Inner, DecompositionResult, CritiqueReport, StageLog, LLMResponse])
def test_synthesizer_outputs_validate_against_models(model: type[BaseModel]) -> None:
    schema = model.model_json_schema()
    for seed in range(15):
        model.model_validate(JsonSchemaSynthesizer(schema, seed=seed).generate())


def test_synthesizer_is_deterministic_and_seed_sensitive() -> None:
    schema = Outer.model_json_schema()
    first = JsonSchemaSynthesizer(schema, seed=7).generate()
    second = JsonSchemaSynthesizer(schema, seed=7).generate()
    third = JsonSchemaSynthesizer(schema, seed=8).generate()
    assert first == second
    assert first != third


def test_synthesizer_produces_realistic_values_without_urls() -> None:
    for seed in range(10):
        data = JsonSchemaSynthesizer(Outer.model_json_schema(), seed=seed).generate()
        dumped = json.dumps(data, ensure_ascii=False)
        assert "http" not in dumped
        assert "www." not in dumped
        assert any("぀" <= ch <= "鿿" for ch in data["description"])
        assert re.fullmatch(r"v\d+\.\d+", data["version_code"])
        assert data["multiple"] % 5 == 0
        assert all(re.fullmatch(r"[a-z][a-z0-9_]{0,31}", item["id"]) for item in data["items"])


def test_synthesizer_terminates_on_recursive_models() -> None:
    def depth(node: dict[str, Any]) -> int:
        children = node.get("children") or []
        return 1 + max((depth(child) for child in children), default=0)

    data = JsonSchemaSynthesizer(Tree.model_json_schema(), seed=3).generate()
    assert depth(data) <= JsonSchemaSynthesizer.MAX_DEPTH + 1


# ---------------------------------------------------------------------------
# MockLLMAdapter / ハンドラ登録
# ---------------------------------------------------------------------------


async def test_mock_adapter_is_deterministic() -> None:
    request = _request(schema_model=Outer)
    first = await MockLLMAdapter(seed=11).complete(request)
    second = await MockLLMAdapter(seed=11).complete(request)
    other = await MockLLMAdapter(seed=12).complete(request)
    assert first.content == second.content
    assert parse_json_lenient(first.content) == parse_json_lenient(second.content)
    assert parse_json_lenient(first.content) != parse_json_lenient(other.content)
    assert first.provider == "mock"
    assert first.usage.estimated is True
    assert first.usage.reasoning_tokens > 0
    assert first.content.startswith("<think>")
    assert first.reasoning.startswith("[mock-deepseek]")
    Outer.model_validate(parse_json_lenient(first.content))


async def test_mock_handler_resolution_order(handlers) -> None:
    @handlers("test.order", "synthesize")
    def _stage_specific(ctx):
        return {"value": "stage-specific"}

    @handlers("test.order", "*")
    def _wildcard(ctx):
        return {"value": "wildcard"}

    adapter = MockLLMAdapter(emit_think_tags=False)
    synth = await adapter.complete(_request("test.order", HarnessStage.SYNTHESIZE, Echo))
    refine = await adapter.complete(_request("test.order", HarnessStage.REFINE, Echo))
    decompose = await adapter.complete(_request("test.order", HarnessStage.DECOMPOSE, DecompositionResult))
    assert parse_json_lenient(synth.content) == {"value": "stage-specific"}
    assert parse_json_lenient(refine.content) == {"value": "wildcard"}
    DecompositionResult.model_validate(parse_json_lenient(decompose.content))
    assert get_mock_handler("test.order", "critique") is not _wildcard
    assert get_mock_handler("test.order", "single") is _wildcard
    assert ("*", "critique") in registered_mock_handlers()
    assert ("*", "decompose") in registered_mock_handlers()


async def test_mock_plain_chat_returns_text() -> None:
    adapter = MockLLMAdapter(emit_think_tags=False)
    response = await adapter.complete(_request("test.chat", content="漫画のめくり演出と緊張曲線について"))
    assert "モック応答" in response.content
    assert "```" not in response.content


async def test_default_decompose_reflects_payload_keywords() -> None:
    adapter = MockLLMAdapter()
    request = _request(
        "test.decompose",
        HarnessStage.DECOMPOSE,
        DecompositionResult,
        context={
            "objective": "対立構造を抽出する",
            "payload": "裏切りと秘密。裏切りが反転を生む。",
            "target_schema_name": "Outer",
            "target_required_fields": ["title", "score"],
        },
    )
    data = DecompositionResult.model_validate(parse_json_lenient((await adapter.complete(request)).content))
    assert data.core_elements[0].name == "裏切"
    assert any("title" in constraint.description for constraint in data.constraints)
    assert [e.importance for e in data.core_elements] == sorted((e.importance for e in data.core_elements), reverse=True)


async def test_default_critique_analyzes_the_draft() -> None:
    adapter = MockLLMAdapter()
    rubric = [
        {"key": "logical_consistency", "description": "矛盾", "weight": 1.0},
        {"key": "schema_fidelity", "description": "準拠", "weight": 2.0},
    ]
    good = {"title": "構造的反転の設計", "items": [{"id": "a_1", "description": "主体の欲求が対立を駆動し、反転の原因となる。"}]}
    bad = {"title": "", "items": [{"id": "a_1", "description": ""}, {"id": "a_1", "description": ""}], "note": "John"}

    def critique_request(draft: Any) -> LLMRequest:
        return _request(
            "test.critique",
            HarnessStage.CRITIQUE,
            CritiqueReport,
            context={"draft": draft, "rubric": rubric, "accept_score": 80.0},
        )

    good_report = CritiqueReport.model_validate(parse_json_lenient((await adapter.complete(critique_request(good))).content))
    bad_report = CritiqueReport.model_validate(parse_json_lenient((await adapter.complete(critique_request(bad))).content))
    assert [score.key for score in good_report.scores] == ["logical_consistency", "schema_fidelity"]
    assert good_report.overall_score > bad_report.overall_score
    assert good_report.verdict == "accept"
    assert bad_report.verdict == "revise"
    assert bad_report.must_fix
    assert bad_report.logical_contradictions
    assert any("John" in gap for gap in bad_report.abstraction_gaps)


def test_mock_failure_plan_validation_and_queue() -> None:
    with pytest.raises(ValueError):
        MockLLMAdapter(failure_plan={"x:y": ["explode"]})
    adapter = MockLLMAdapter()
    adapter.plan_failures("test.q:*", "garbage", "ok")
    assert adapter.pending_failures() == {"test.q:*": ["garbage", "ok"]}


def test_handler_modules_lazy_loading(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_file = tmp_path / "caf_tmp_handlers_ok.py"
    module_file.write_text(
        "from core.llm_client import register_mock_handler\n\n"
        "@register_mock_handler('test.lazy', 'synthesize')\n"
        "def _handler(ctx):\n"
        "    return {'value': 'lazy'}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        mock_module, "MOCK_HANDLER_MODULES", ("caf_tmp_handlers_ok", "core.llm_client.caf_missing_handlers")
    )
    reset_mock_handlers_loaded()
    try:
        loaded = ensure_mock_handler_modules_loaded()
        assert loaded == ["caf_tmp_handlers_ok"]
        assert get_mock_handler("test.lazy", "synthesize") is not None
        assert ensure_mock_handler_modules_loaded() == loaded
    finally:
        unregister_mock_handler("test.lazy", "synthesize")
        sys.modules.pop("caf_tmp_handlers_ok", None)


def test_handler_module_internal_error_is_raised(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "caf_tmp_handlers_bad.py").write_text("import caf_dependency_that_does_not_exist_xyz\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(mock_module, "MOCK_HANDLER_MODULES", ("caf_tmp_handlers_bad",))
    reset_mock_handlers_loaded()
    try:
        with pytest.raises(ModuleNotFoundError):
            ensure_mock_handler_modules_loaded()
    finally:
        sys.modules.pop("caf_tmp_handlers_bad", None)


# ---------------------------------------------------------------------------
# LLMClient: 構造化生成・自己修復
# ---------------------------------------------------------------------------


async def test_generate_structured_success_with_handler(handlers, tmp_db) -> None:
    @handlers("test.ok", "single")
    def _handler(ctx):
        return {"value": "完成した構造"}

    adapter = MockLLMAdapter()
    client = create_llm_client(db=tmp_db, adapter=adapter)
    result = await client.generate_structured(
        Echo, [system_msg("あなたは構造研究者である"), user_msg("出力せよ")], task="test.ok"
    )
    assert result.value == Echo(value="完成した構造")
    assert result.attempts == 1
    assert result.syntax_repaired is False
    assert "<think>" not in result.raw_text
    assert result.response.reasoning
    first_system = adapter.calls[0].messages[0]
    assert first_system.role == "system"
    assert first_system.content.startswith("あなたは構造研究者である")
    assert "JSON Schema「Echo」" in first_system.content
    assert client.telemetry is not None
    rows = client.telemetry.recent_calls()
    assert len(rows) == 1
    assert rows[0]["success"] is True
    assert rows[0]["task"] == "test.ok"
    assert rows[0]["reasoning_log"]


async def test_schema_instruction_inserted_when_no_system_message() -> None:
    adapter = MockLLMAdapter()
    client = LLMClient(adapter)
    result = await client.generate_structured(Echo, [user_msg("出力せよ")], task="test.nosys")
    assert isinstance(result.value, Echo)
    assert adapter.calls[0].messages[0].role == "system"
    assert "JSON Schema「Echo」" in adapter.calls[0].messages[0].content
    assert adapter.calls[0].json_mode is True


@pytest.mark.parametrize("behavior", ["garbage", "schema_violation", "empty"])
async def test_self_repair_recovers_on_second_attempt(behavior: str, handlers, tmp_db) -> None:
    @handlers("test.repair", "single")
    def _handler(ctx):
        return {"value": "修復後" if ctx.is_repair else "初回"}

    adapter = MockLLMAdapter(failure_plan={"test.repair:single": [behavior]})
    client = create_llm_client(db=tmp_db, adapter=adapter)
    result = await client.generate_structured(Echo, [user_msg("出力せよ")], task="test.repair")
    assert result.attempts == 2
    assert result.value.value == "修復後"
    second = adapter.calls[1]
    assert second.attempt == 2
    assert second.context["repair"] is True
    assert second.context["previous_errors"]
    assert second.messages[-1].role == "user"
    assert "検証エラー" in second.messages[-1].content
    assert second.messages[-2].role == "assistant"
    assert client.telemetry is not None
    rows = client.telemetry.recent_calls()
    assert [row["success"] for row in reversed(rows)] == [False, True]
    assert rows[-1]["error"]
    assert result.total_usage.total_tokens > result.response.usage.total_tokens


async def test_repairable_json_needs_no_retry(handlers) -> None:
    @handlers("test.repairable", "single")
    def _handler(ctx):
        return {"value": "構文修復", "flag": True, "items": [1, 2]}

    adapter = MockLLMAdapter(failure_plan={"test.repairable:single": ["repairable_json"]})
    result = await LLMClient(adapter).generate_structured(Rich, [user_msg("出力せよ")], task="test.repairable")
    assert result.attempts == 1
    assert result.syntax_repaired is True
    assert result.value == Rich(value="構文修復", flag=True, items=[1, 2])


async def test_structured_output_error_after_exhausting_attempts() -> None:
    adapter = MockLLMAdapter(failure_plan={"test.fail:*": ["garbage"] * 10})
    client = LLMClient(adapter, max_repair_attempts=2)
    with pytest.raises(StructuredOutputError) as info:
        await client.generate_structured(Echo, [user_msg("x")], task="test.fail")
    assert info.value.attempts == 3
    assert len(adapter.calls) == 3
    assert info.value.schema_name == "Echo"
    assert info.value.validation_errors.startswith("- (json)")
    assert info.value.usage.total_tokens > 0


async def test_connection_error_propagates_and_is_recorded(tmp_db) -> None:
    adapter = MockLLMAdapter(failure_plan={"test.conn:single": ["connection_error"]})
    client = create_llm_client(db=tmp_db, adapter=adapter)
    with pytest.raises(LLMConnectionError):
        await client.generate_structured(Echo, [user_msg("x")], task="test.conn")
    assert client.telemetry is not None
    rows = client.telemetry.recent_calls()
    assert rows[0]["success"] is False
    assert "LLMConnectionError" in rows[0]["error"]
    recovered = await client.generate_structured(Echo, [user_msg("x")], task="test.conn")
    assert recovered.attempts == 1


async def test_chat_strips_think_tags_and_records(tmp_db) -> None:
    client = create_llm_client(db=tmp_db, adapter=MockLLMAdapter())
    response = await client.chat([user_msg("緊張と緩和の設計について")], task="test.chat")
    assert "<think>" not in response.content
    assert response.reasoning.startswith("[mock-deepseek]")
    assert client.telemetry is not None
    assert client.telemetry.stats()["totals"]["calls"] == 1


async def test_structured_convenience_method() -> None:
    result = await LLMClient(MockLLMAdapter()).structured(Echo, system="構造研究者", user="出力", task="test.conv")
    assert isinstance(result.value, Echo)


# ---------------------------------------------------------------------------
# テレメトリ
# ---------------------------------------------------------------------------


def test_telemetry_store_truncates_and_aggregates(tmp_db) -> None:
    store = TelemetryStore(tmp_db)
    store.record_call(
        run_id=None,
        task="test.t",
        stage="single",
        provider="mock",
        model="m",
        success=True,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, reasoning_tokens=2),
        latency_ms=12.5,
        request_excerpt="r" * 9000,
        response_excerpt="a" * 9000,
        reasoning_log="思" * 30000,
    )
    store.record_call(
        run_id=None, task="test.t", stage="single", provider="mock", model="m", success=False, error="boom", latency_ms=7.5
    )
    rows = store.recent_calls()
    assert len(rows) == 2
    assert rows[0]["success"] is False
    assert len(rows[1]["response_excerpt"]) == 4000
    assert len(rows[1]["reasoning_log"]) == 20000
    stats = store.stats()
    assert stats["totals"]["calls"] == 2
    assert stats["totals"]["failures"] == 1
    assert stats["totals"]["total_tokens"] == 15
    assert stats["totals"]["success_rate"] == 0.5
    assert abs(stats["totals"]["avg_latency_ms"] - 10.0) < 1e-9
    assert stats["by_task_stage"][0]["calls"] == 2


# ---------------------------------------------------------------------------
# 4段階ハーネス
# ---------------------------------------------------------------------------


async def test_harness_runs_four_stages_in_order(handlers, tmp_db) -> None:
    seen: dict[str, dict[str, Any]] = {}

    @handlers("test.harness", ("synthesize", "refine"))
    def _handler(ctx):
        seen[ctx.stage] = dict(ctx.context)
        return {"value": "最終版" if ctx.stage == "refine" else "ドラフト"}

    adapter = MockLLMAdapter()
    client = create_llm_client(db=tmp_db, adapter=adapter)
    config = HarnessConfig(stage_temperatures={"decompose": 0.41, "synthesize": 0.31, "critique": 0.11, "refine": 0.21})
    events: list[str] = []

    async def on_stage(log: StageLog) -> None:
        events.append(log.stage)

    payload = {"text": "師の裏切りが明かされ、主人公は代償を払って突破する。"}
    result = await MultiStageReasoningHarness(client, config).run(
        task="test.harness",
        objective="対立と反転の構造を抽出する",
        payload=payload,
        output_model=Echo,
        system_prompt="構造抽出エンジン",
        context={"custom": 42},
        on_stage=on_stage,
    )
    assert [call.stage.value for call in adapter.calls] == ["decompose", "synthesize", "critique", "refine"]
    assert [call.temperature for call in adapter.calls] == [0.41, 0.31, 0.11, 0.21]
    assert events == ["decompose", "synthesize", "critique", "refine"]
    assert [log.ordinal for log in result.stage_logs] == [1, 2, 3, 4]
    assert all(log.success for log in result.stage_logs)
    assert result.output.value == "最終版"
    assert result.draft is not None and result.draft.value == "ドラフト"
    assert result.refined is True and result.refine_rounds == 1
    assert result.decomposition is not None and result.critique is not None
    assert seen["synthesize"]["decomposition"]["essence"]
    assert seen["synthesize"]["custom"] == 42
    assert seen["synthesize"]["objective"] == "対立と反転の構造を抽出する"
    assert seen["synthesize"]["payload"] == payload
    critique_context = adapter.calls[2].context
    assert critique_context["draft"] == {"value": "ドラフト"}
    assert critique_context["accept_score"] == 85.0
    assert len(critique_context["rubric"]) == 5
    assert seen["refine"]["draft"] == {"value": "ドラフト"}
    assert seen["refine"]["critique"]["overall_score"] == result.critique.overall_score
    assert "must_fix" in seen["refine"]
    stage_total = sum((log.usage for log in result.stage_logs), TokenUsage())
    assert result.usage.total_tokens == stage_total.total_tokens > 0
    assert client.telemetry is not None
    run = client.telemetry.get_run(result.run_id)
    assert run is not None
    assert run["status"] == "succeeded"
    assert [stage["stage"] for stage in run["stage_logs"]] == ["decompose", "synthesize", "critique", "refine"]
    assert len(run["calls"]) == 4
    assert run["total_tokens"] == result.usage.total_tokens
    assert run["critique_score"] == pytest.approx(result.critique.overall_score)
    stats = client.telemetry.stats()
    assert ("test.harness", "critique") in {(row["task"], row["stage"]) for row in stats["by_task_stage"]}
    assert stats["runs"] == {"succeeded": 1}


async def test_harness_tolerates_decompose_and_critique_failures(handlers) -> None:
    @handlers("test.tolerant", ("synthesize", "refine"))
    def _handler(ctx):
        return {"value": ctx.stage}

    adapter = MockLLMAdapter(
        failure_plan={"test.tolerant:decompose": ["garbage"] * 3, "test.tolerant:critique": ["connection_error"]}
    )
    harness = MultiStageReasoningHarness(LLMClient(adapter), HarnessConfig(max_repair_attempts=1))
    result = await harness.run(task="test.tolerant", objective="o", payload="p", output_model=Echo, system_prompt="s")
    assert result.decomposition is None
    assert result.critique is None
    assert result.output.value == "refine"
    assert [(log.stage, log.success) for log in result.stage_logs] == [
        ("decompose", False),
        ("synthesize", True),
        ("critique", False),
        ("refine", True),
    ]
    assert result.stage_logs[0].attempts == 2
    assert len(result.warnings) == 2


async def test_harness_synthesize_failure_raises(tmp_db) -> None:
    adapter = MockLLMAdapter(failure_plan={"test.synthfail:synthesize": ["garbage"] * 5})
    client = create_llm_client(db=tmp_db, adapter=adapter)
    harness = MultiStageReasoningHarness(client, HarnessConfig(max_repair_attempts=1))
    with pytest.raises(HarnessError) as info:
        await harness.run(task="test.synthfail", objective="o", payload="p", output_model=Echo, system_prompt="s")
    assert info.value.stage == "synthesize"
    assert client.telemetry is not None
    run = client.telemetry.get_run(info.value.run_id or "")
    assert run is not None and run["status"] == "failed"


async def test_harness_skips_refine_when_accepted() -> None:
    adapter = MockLLMAdapter()
    config = HarnessConfig(skip_refine_when_accepted=True, critique_accept_score=0.0)
    result = await MultiStageReasoningHarness(LLMClient(adapter), config).run(
        task="test.skip", objective="o", payload="p", output_model=Echo, system_prompt="s"
    )
    assert [call.stage.value for call in adapter.calls] == ["decompose", "synthesize", "critique"]
    assert result.refined is False
    assert result.output == result.draft


async def test_harness_minimal_stage_configuration() -> None:
    adapter = MockLLMAdapter()
    config = HarnessConfig(enable_decompose=False, enable_critique=False, enable_refine=False)
    result = await MultiStageReasoningHarness(LLMClient(adapter), config).run(
        task="test.minimal", objective="o", payload="p", output_model=Echo, system_prompt="s"
    )
    assert [call.stage.value for call in adapter.calls] == ["synthesize"]
    assert len(result.stage_logs) == 1
    assert result.refined is False


async def test_harness_post_validate_triggers_extra_refine(handlers) -> None:
    counter = {"refine": 0}
    captured: list[list[str]] = []

    @handlers("test.post", "synthesize")
    def _synth(ctx):
        return {"value": "draft"}

    @handlers("test.post", "refine")
    def _refine(ctx):
        counter["refine"] += 1
        captured.append(list(ctx.context.get("must_fix") or []))
        return {"value": f"refined-{counter['refine']}"}

    def post_validate(output: Echo) -> list[str]:
        return [] if output.value == "refined-2" else [f"値 {output.value} は未完成"]

    result = await MultiStageReasoningHarness(LLMClient(MockLLMAdapter()), HarnessConfig(max_refine_rounds=2)).run(
        task="test.post", objective="o", payload="p", output_model=Echo, system_prompt="s", post_validate=post_validate
    )
    assert result.output.value == "refined-2"
    assert result.refine_rounds == 2
    assert captured[1] == ["値 refined-1 は未完成"]
    assert not any("post_validate" in warning for warning in result.warnings)


async def test_harness_with_generic_synthesis_produces_valid_complex_output() -> None:
    result = await MultiStageReasoningHarness(LLMClient(MockLLMAdapter())).run(
        task="test.generic_outer", objective="構造化", payload={"k": "v"}, output_model=Outer, system_prompt="s"
    )
    assert isinstance(result.output, Outer)
    Outer.model_validate(result.output.model_dump())


# ---------------------------------------------------------------------------
# OpenAI 互換 / Ollama アダプタ（httpx.MockTransport）
# ---------------------------------------------------------------------------


def test_openai_payload_response_format_modes() -> None:
    request = LLMRequest(
        messages=[user_msg("x")], response_schema=Echo.model_json_schema(), schema_name="Echo Schema!", json_mode=True
    )
    schema_mode = OpenAICompatibleAdapter(structured_output="json_schema").build_payload(request)
    assert schema_mode["response_format"]["type"] == "json_schema"
    assert schema_mode["response_format"]["json_schema"]["name"] == "Echo_Schema"
    assert schema_mode["response_format"]["json_schema"]["schema"]["title"] == "Echo"
    object_mode = OpenAICompatibleAdapter(structured_output="json_object")._build_payload(request)
    assert object_mode["response_format"] == {"type": "json_object"}
    prompt_mode = OpenAICompatibleAdapter(structured_output="prompt_only").build_payload(request)
    assert "response_format" not in prompt_mode
    json_only = OpenAICompatibleAdapter(structured_output="json_schema").build_payload(
        LLMRequest(messages=[user_msg("x")], json_mode=True)
    )
    assert json_only["response_format"] == {"type": "json_object"}
    plain = OpenAICompatibleAdapter().build_payload(LLMRequest(messages=[user_msg("x")]))
    assert "response_format" not in plain
    assert plain["model"] == "DeepSeek-V4.1-Flash"
    assert plain["stream"] is False
    assert OpenAICompatibleAdapter(extra_body={"top_p": 0.9}).build_payload(request)["top_p"] == 0.9
    with pytest.raises(ValueError):
        OpenAICompatibleAdapter(structured_output="xml")


async def test_openai_complete_parses_reasoning_think_and_usage() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_completion_body(
                '<think>内部思考</think>{"value": "結果"}',
                reasoning="サーバ思考",
                usage={
                    "prompt_tokens": 120,
                    "completion_tokens": 45,
                    "total_tokens": 165,
                    "completion_tokens_details": {"reasoning_tokens": 30},
                },
            ),
        )

    adapter = OpenAICompatibleAdapter(api_key="secret", transport=httpx.MockTransport(handler), retry_backoff_s=0)
    request = LLMRequest(
        messages=[system_msg("s"), user_msg("u")],
        response_schema=Echo.model_json_schema(),
        schema_name="Echo",
        temperature=0.25,
        max_tokens=512,
    )
    response = await adapter.complete(request)
    await adapter.aclose()
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["messages"][1] == {"role": "user", "content": "u"}
    assert captured["body"]["temperature"] == 0.25
    assert captured["body"]["max_tokens"] == 512
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert response.content == '{"value": "結果"}'
    assert "サーバ思考" in response.reasoning
    assert "内部思考" in response.reasoning
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.reasoning_tokens) == (120, 45, 30)
    assert response.usage.total_tokens == 165
    assert response.usage.estimated is False
    assert response.provider == "openai_compatible"
    assert response.model == "DeepSeek-V4.1-Flash"


async def test_openai_missing_usage_is_estimated() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_completion_body("構造化された応答です")))
    response = await OpenAICompatibleAdapter(transport=transport).complete(LLMRequest(messages=[user_msg("質問")]))
    assert response.usage.estimated is True
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


async def test_openai_retries_on_503_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=_completion_body("ok"))

    adapter = OpenAICompatibleAdapter(transport=httpx.MockTransport(handler), max_transport_retries=2, retry_backoff_s=0)
    response = await adapter.complete(LLMRequest(messages=[user_msg("x")]))
    assert response.content == "ok"
    assert calls["n"] == 2


async def test_openai_429_exhausts_retries() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text="rate limited")

    adapter = OpenAICompatibleAdapter(transport=httpx.MockTransport(handler), max_transport_retries=2, retry_backoff_s=0)
    with pytest.raises(LLMConnectionError):
        await adapter.complete(LLMRequest(messages=[user_msg("x")]))
    assert calls["n"] == 3


async def test_openai_client_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    adapter = OpenAICompatibleAdapter(transport=httpx.MockTransport(handler), max_transport_retries=3, retry_backoff_s=0)
    with pytest.raises(LLMResponseError) as info:
        await adapter.complete(LLMRequest(messages=[user_msg("x")]))
    assert info.value.status_code == 400
    assert calls["n"] == 1


async def test_openai_empty_choices_is_response_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMResponseError):
        await OpenAICompatibleAdapter(transport=transport).complete(LLMRequest(messages=[user_msg("x")]))


async def test_openai_connection_failure_raises_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    adapter = OpenAICompatibleAdapter(transport=httpx.MockTransport(handler), max_transport_retries=1, retry_backoff_s=0)
    with pytest.raises(LLMConnectionError):
        await adapter.complete(LLMRequest(messages=[user_msg("x")]))
    health = await adapter.health_check()
    assert health["ok"] is False
    assert health["provider"] == "openai_compatible"


async def test_openai_list_models_and_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "DeepSeek-V4.1-Flash"}, {"id": "other"}]})

    adapter = OpenAICompatibleAdapter(transport=httpx.MockTransport(handler))
    assert await adapter.list_models() == ["DeepSeek-V4.1-Flash", "other"]
    health = await adapter.health_check()
    assert health["ok"] is True
    assert health["model_available"] is True


async def test_ollama_health_uses_native_api_tags() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "deepseek-v4.1-flash:latest"}]})
        return httpx.Response(404)

    adapter = OllamaAdapter(transport=httpx.MockTransport(handler))
    assert adapter.root_url == "http://localhost:11434"
    health = await adapter.health_check()
    assert health["ok"] is True
    assert health["provider"] == "ollama"
    assert health["model_available"] is True
    assert seen == ["/api/tags"]


async def test_ollama_health_falls_back_to_v1_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(404)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "deepseek-v4.1-flash"}]})
        return httpx.Response(500)

    adapter = OllamaAdapter(transport=httpx.MockTransport(handler), retry_backoff_s=0)
    health = await adapter.health_check()
    assert health["ok"] is True
    assert health["models"] == ["deepseek-v4.1-flash"]


async def test_ollama_completion_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:11434/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "deepseek-v4.1-flash"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=_completion_body('{"value": "ollama"}', usage={"prompt_tokens": 5, "completion_tokens": 3}))

    adapter = OllamaAdapter(transport=httpx.MockTransport(handler))
    response = await adapter.complete(
        LLMRequest(messages=[user_msg("x")], response_schema=Echo.model_json_schema(), schema_name="Echo")
    )
    assert response.provider == "ollama"
    assert json.loads(response.content) == {"value": "ollama"}
    assert response.usage.total_tokens == 8


# ---------------------------------------------------------------------------
# ファクトリ / AutoFallback
# ---------------------------------------------------------------------------


def test_create_adapter_by_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAF_LLM_PROVIDER", raising=False)
    mock_adapter = create_adapter(LLMSettings(provider="mock", mock_seed=5))
    assert isinstance(mock_adapter, MockLLMAdapter)
    assert mock_adapter.seed == 5
    openai_adapter = create_adapter(
        LLMSettings(
            provider="openai_compatible",
            base_url="http://127.0.0.1:9000/v1",
            model="m1",
            structured_output="json_schema",
        )
    )
    assert type(openai_adapter) is OpenAICompatibleAdapter
    assert openai_adapter.base_url == "http://127.0.0.1:9000/v1"
    assert openai_adapter.model == "m1"
    assert openai_adapter.structured_output == "json_schema"
    ollama_adapter = create_adapter(LLMSettings(provider="ollama", ollama_model="deepseek-test"))
    assert isinstance(ollama_adapter, OllamaAdapter)
    assert ollama_adapter.model == "deepseek-test"
    auto_adapter = create_adapter(LLMSettings(provider="auto"))
    assert isinstance(auto_adapter, AutoFallbackAdapter)
    assert isinstance(auto_adapter.fallback, MockLLMAdapter)
    assert [type(adapter) for adapter in auto_adapter.primaries] == [OpenAICompatibleAdapter, OllamaAdapter]


def test_env_provider_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAF_LLM_PROVIDER", "mock")
    assert isinstance(create_adapter(LLMSettings(provider="openai_compatible")), MockLLMAdapter)
    monkeypatch.setenv("CAF_LLM_PROVIDER", "ollama")
    assert isinstance(create_adapter(LLMSettings(provider="mock")), OllamaAdapter)
    monkeypatch.setenv("CAF_LLM_PROVIDER", "invalid-value")
    assert isinstance(create_adapter(LLMSettings(provider="mock")), MockLLMAdapter)
    assert resolve_provider(LLMSettings(provider="mock")) == "mock"


def test_create_llm_client_wires_telemetry_and_repairs(tmp_db, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAF_LLM_PROVIDER", "mock")
    settings = LLMSettings(harness=HarnessSettings(max_repair_attempts=5, enable_critique=False))
    client = create_llm_client(settings, db=tmp_db)
    assert isinstance(client.adapter, MockLLMAdapter)
    assert isinstance(client.telemetry, TelemetryStore)
    assert client.max_repair_attempts == 5
    assert create_llm_client(settings).telemetry is None
    harness = create_harness(settings, db=tmp_db)
    assert harness.config.max_repair_attempts == 5
    assert harness.config.enable_critique is False
    assert HarnessConfig.from_settings(settings.harness).planned_stages() == ["decompose", "synthesize", "refine"]


def test_mock_llm_client_fixture(mock_llm_client) -> None:
    assert mock_llm_client.provider == "mock"
    assert mock_llm_client.telemetry is not None


async def test_auto_fallback_switches_to_mock_when_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    primary = OpenAICompatibleAdapter(transport=httpx.MockTransport(refuse), max_transport_retries=0)
    secondary = OllamaAdapter(transport=httpx.MockTransport(refuse), max_transport_retries=0)
    auto = AutoFallbackAdapter([primary, secondary], MockLLMAdapter(), probe_timeout_s=1.0)
    client = LLMClient(auto)
    result = await client.generate_structured(Echo, [user_msg("x")], task="test.auto")
    assert isinstance(result.value, Echo)
    assert result.response.provider == "mock"
    assert auto.fallback_active is True
    assert auto.selected_provider == "mock"
    assert client.fallback_active is True
    assert client.effective_provider == "mock"
    assert len(auto.probe_results) == 2
    assert not any(probe["ok"] for probe in auto.probe_results)
    health = await auto.health_check()
    assert health["ok"] is True
    assert health["fallback_active"] is True
    assert health["selected_provider"] == "mock"


async def test_auto_uses_healthy_primary() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "DeepSeek-V4.1-Flash"}]})
        return httpx.Response(200, json=_completion_body('{"value": "実機応答"}', usage={"prompt_tokens": 3, "completion_tokens": 2}))

    auto = AutoFallbackAdapter([OpenAICompatibleAdapter(transport=httpx.MockTransport(handler))], MockLLMAdapter())
    result = await LLMClient(auto).generate_structured(Echo, [user_msg("x")], task="test.auto_primary")
    assert result.value.value == "実機応答"
    assert result.response.provider == "openai_compatible"
    assert auto.fallback_active is False
    assert auto.selected_provider == "openai_compatible"


async def test_auto_falls_back_when_primary_breaks_later() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        raise httpx.ConnectError("dropped", request=request)

    primary = OpenAICompatibleAdapter(transport=httpx.MockTransport(handler), max_transport_retries=0)
    auto = AutoFallbackAdapter([primary], MockLLMAdapter())
    response = await auto.complete(LLMRequest(messages=[user_msg("x")]))
    assert response.provider == "mock"
    assert auto.fallback_active is True
