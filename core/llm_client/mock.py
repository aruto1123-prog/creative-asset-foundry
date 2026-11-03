"""高精度モック LLM アダプタ（DI 差し替え用）。

クラウド VM / CI など実機 LLM が存在しない環境で、全 Pydantic スキーマに準拠した
決定論的かつリアルな構造化応答を返す。

- タスク別ハンドラ登録（``register_mock_handler``）: 各モジュールの ``mock_handlers.py`` が、
  決定論的ベースラインからリアルな応答を合成するハンドラを登録する。
- 汎用 ``JsonSchemaSynthesizer``: ハンドラ未登録のスキーマに対し、制約を満たすインスタンスを合成する。
- 既定ハンドラ: ハーネス内部段（decompose / critique）を入力・ドラフトの実解析から構成する。
- 障害注入（``failure_plan``）: 壊れた JSON・スキーマ違反・接続障害などを再現し、自己修復経路を検証する。
"""

from __future__ import annotations

import asyncio
import copy
import datetime as _dt
import hashlib
import importlib
import inspect
import json
import logging
import math
import random
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from core.common import clamp, stable_seed, truncate
from core.llm_client.adapters import BaseLLMAdapter
from core.llm_client.errors import LLMConnectionError
from core.llm_client.tokens import estimate_messages_tokens, estimate_tokens
from core.llm_client.types import DEFAULT_RUBRIC, HarnessStage, LLMRequest, LLMResponse, TokenUsage

try:  # Python 3.11+
    from re import _parser as _sre_parse  # type: ignore[attr-defined]
except ImportError:  # Python 3.10
    import sre_parse as _sre_parse  # type: ignore[no-redef]

__all__ = [
    "MockContext",
    "MockHandler",
    "MockLLMAdapter",
    "JsonSchemaSynthesizer",
    "register_mock_handler",
    "unregister_mock_handler",
    "get_mock_handler",
    "registered_mock_handlers",
    "MOCK_HANDLER_MODULES",
    "ensure_mock_handler_modules_loaded",
    "reset_mock_handlers_loaded",
    "extract_keywords",
    "FAILURE_BEHAVIORS",
]

logger = logging.getLogger(__name__)

FAILURE_BEHAVIORS = frozenset(
    {"ok", "garbage", "schema_violation", "repairable_json", "connection_error", "empty"}
)
_HARNESS_INTERNAL_STAGES = frozenset({HarnessStage.DECOMPOSE.value, HarnessStage.CRITIQUE.value})

MOCK_HANDLER_MODULES: tuple[str, ...] = (
    "core.structure_engine.mock_handlers",
    "core.style_alchemy.mock_handlers",
    "core.manga_pipeline.mock_handlers",
    "core.asset_publisher.mock_handlers",
)

# ---------------------------------------------------------------------------
# MockContext / ハンドラ登録簿
# ---------------------------------------------------------------------------


@dataclass
class MockContext:
    """モックハンドラへ渡される呼び出し文脈。"""

    request: LLMRequest
    task: str
    stage: str
    schema: dict[str, Any] | None
    schema_name: str | None
    context: dict[str, Any]
    seed: int
    rng: random.Random
    attempt: int = 1

    def synthesize(self, schema: dict[str, Any] | None = None) -> Any:
        """JSON Schema（省略時はリクエストのスキーマ）から決定論的インスタンスを生成する。"""
        target = schema if schema is not None else self.schema
        if target is None:
            return {}
        return JsonSchemaSynthesizer(target, seed=self.seed).generate()

    @property
    def draft(self) -> Any:
        return self.context.get("draft")

    @property
    def critique(self) -> Any:
        return self.context.get("critique")

    @property
    def is_repair(self) -> bool:
        return bool(self.context.get("repair")) or self.stage == HarnessStage.REPAIR.value

    @property
    def system_prompt(self) -> str:
        return "\n".join(m.content for m in self.request.messages if m.role == "system")

    @property
    def last_user_message(self) -> str:
        for message in reversed(self.request.messages):
            if message.role == "user":
                return message.content
        return ""


MockHandler = Callable[[MockContext], Any]

_HANDLERS: dict[tuple[str, str], MockHandler] = {}
_REGISTRY_LOCK = threading.RLock()
_MODULES_LOADED = False
_LOADED_MODULE_NAMES: list[str] = []
_LOAD_LOCK = threading.Lock()


def _stage_value(stage: str | HarnessStage) -> str:
    return stage.value if isinstance(stage, HarnessStage) else str(stage)


def register_mock_handler(task: str, stages: str | HarnessStage | Iterable[str | HarnessStage] = "*") -> Callable[[MockHandler], MockHandler]:
    """``(task, stage)`` に対するモックハンドラを登録するデコレータ。``stages="*"`` は全段。"""
    if isinstance(stages, (str, HarnessStage)):
        stage_list = [_stage_value(stages)]
    else:
        stage_list = [_stage_value(s) for s in stages]

    def decorator(func: MockHandler) -> MockHandler:
        with _REGISTRY_LOCK:
            for stage in stage_list:
                _HANDLERS[(task, stage)] = func
        return func

    return decorator


def unregister_mock_handler(task: str, stages: str | HarnessStage | Iterable[str | HarnessStage] = "*") -> int:
    """登録を解除し、解除件数を返す。"""
    if isinstance(stages, (str, HarnessStage)):
        stage_list = [_stage_value(stages)]
    else:
        stage_list = [_stage_value(s) for s in stages]
    removed = 0
    with _REGISTRY_LOCK:
        for stage in stage_list:
            if _HANDLERS.pop((task, stage), None) is not None:
                removed += 1
    return removed


def get_mock_handler(task: str, stage: str | HarnessStage) -> MockHandler | None:
    """ハンドラを解決する。

    通常段: ``(task, stage)`` → ``(task, "*")`` → ``("*", stage)``。
    ハーネス内部段（decompose / critique）は出力スキーマがハーネス固有のため、
    ``(task, stage)`` → ``("*", stage)`` → ``(task, "*")`` の順で既定ハンドラを優先する。
    """
    stage_name = _stage_value(stage)
    if stage_name in _HARNESS_INTERNAL_STAGES:
        order = ((task, stage_name), ("*", stage_name), (task, "*"))
    else:
        order = ((task, stage_name), (task, "*"), ("*", stage_name))
    with _REGISTRY_LOCK:
        for key in order:
            handler = _HANDLERS.get(key)
            if handler is not None:
                return handler
    return None


def registered_mock_handlers() -> list[tuple[str, str]]:
    """登録済み ``(task, stage)`` の一覧。"""
    with _REGISTRY_LOCK:
        return sorted(_HANDLERS.keys())


def _module_parents(name: str) -> set[str]:
    parts = name.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts) + 1)}


def ensure_mock_handler_modules_loaded() -> list[str]:
    """``MOCK_HANDLER_MODULES`` を一度だけ遅延 import し、読み込めたモジュール名を返す。

    当該モジュール自身（またはその親パッケージ）が未実装で ``ModuleNotFoundError`` になる場合のみ無視し、
    それ以外の例外（ハンドラモジュール内部の不具合）はそのまま送出する。
    """
    global _MODULES_LOADED, _LOADED_MODULE_NAMES
    if _MODULES_LOADED:
        return list(_LOADED_MODULE_NAMES)
    with _LOAD_LOCK:
        if _MODULES_LOADED:
            return list(_LOADED_MODULE_NAMES)
        loaded: list[str] = []
        for module_name in MOCK_HANDLER_MODULES:
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if (exc.name or "") in _module_parents(module_name):
                    logger.debug("モックハンドラモジュール未実装のため無視: %s", module_name)
                    continue
                raise
            loaded.append(module_name)
        _LOADED_MODULE_NAMES = loaded
        _MODULES_LOADED = True
    return list(loaded)


def reset_mock_handlers_loaded() -> None:
    """遅延 import 済みフラグを解除する（次回 ``complete()`` で再読込）。"""
    global _MODULES_LOADED, _LOADED_MODULE_NAMES
    with _LOAD_LOCK:
        _MODULES_LOADED = False
        _LOADED_MODULE_NAMES = []


# ---------------------------------------------------------------------------
# 語彙・キーワード抽出
# ---------------------------------------------------------------------------

_NOUNS = [
    "構造", "対立", "欲求", "欠落", "象徴", "緊張", "解放", "反転", "代償", "余韻",
    "共感", "認知", "情報", "視点", "伏線", "動機", "葛藤", "選択", "信頼", "境界",
    "均衡", "変容", "期待", "予兆", "沈黙", "秘密", "約束", "喪失", "再生", "決断",
]
_ADJECTIVES = [
    "普遍的な", "非対称な", "潜在的な", "決定的な", "構造的な", "静かな",
    "劇的な", "不可逆な", "逆説的な", "本質的な", "段階的な", "鮮烈な",
]
_SENTENCE_TEMPLATES = [
    "{a}{n1}が{n2}を駆動し、受け手の{n3}を段階的に高める。",
    "{n1}と{n2}の非対称性が、全体の{n3}を規定している。",
    "表層の{n1}を剥がすと、{a}{n2}の力学が現れる。",
    "{n1}の提示から{n2}の反転までを一貫した因果で結ぶ。",
    "{n1}を最小限の情報で示し、{n2}への{n3}を持続させる。",
    "{a}{n1}は{n2}の代償として現れ、{n3}の意味を書き換える。",
    "{n1}が満たされない状態を保つことで、{n2}への関心が{n3}へと変わる。",
]
_TITLE_TEMPLATES = [
    "{n1}と{n2}の{n3}",
    "{a}{n1}の設計",
    "{n1}から{n2}への転換",
    "{n1}を軸にした{n2}の再構成",
    "{a}{n1}と{n2}",
]
_SLUG_WORDS = [
    "core", "tension", "desire", "lack", "symbol", "twist", "release", "anchor",
    "pivot", "signal", "frame", "vector", "arc", "beat", "gap", "mirror",
]
_SCREAMING_WORDS = [
    "PROTAGONIST", "GOAL", "CONFLICT", "IRONY", "REVERSAL", "COST", "SYMBOL", "SETTING",
    "AUDIENCE", "OFFER", "EVIDENCE", "EMOTION", "LACK", "DESIRE",
]
_PROMPT_TOKENS = [
    "monochrome", "dynamic angle", "clean lineart", "dramatic lighting",
    "screentone", "high contrast", "wide shot", "close-up",
]
_SENTENCE_HINTS = (
    "description", "summary", "rationale", "explanation", "detail", "reason", "intent", "purpose",
    "essence", "function", "effect", "trigger", "note", "feedback", "message", "evidence", "claim",
    "insight", "thesis", "logline", "content", "body", "text", "lead", "closing", "context", "comment",
    "hypothesis", "analysis", "guide", "instruction", "answer", "objective", "premise", "pain", "angle",
    "hook", "assumption", "criteria", "criterion", "risk", "constraint", "point", "contradiction", "fix",
    "action", "step", "goal", "invariant", "pacing", "reveal", "anchor", "driver", "cta", "plan",
)
_TITLE_HINTS = ("title", "heading", "headline", "caption")
_LABEL_HINTS = ("name", "label", "role", "persona", "speaker", "placeholder")
_WORD_HINTS = ("tag", "keyword", "concept", "term", "emotion", "word", "genre", "domain", "lineage")
_UNIT_NUMBER_HINTS = (
    "weight", "ratio", "strength", "importance", "intensity", "confidence", "probability", "share",
    "rate", "tension", "release", "load", "level", "density", "salience", "knowledge", "coverage",
)
_SCORE_HINTS = ("score", "total", "overall", "percent", "points")
_COUNT_HINTS = ("count", "index", "page", "number", "num", "size", "steps", "ordinal", "attempt", "rank")

_KW_RE = re.compile(r"[一-鿿㐀-䶿々〆ヵヶ]{2,}|[゠-ヿー]{3,}|[A-Za-z][A-Za-z\-]{3,}")
_KW_STOPWORDS = frozenset(
    {
        "the", "and", "with", "from", "that", "this", "json", "schema", "type", "string", "number",
        "object", "array", "integer", "boolean", "null", "true", "false", "properties", "required",
        "items", "title", "description", "default", "minimum", "maximum", "minitems", "maxitems",
        "anyof", "enum", "const", "pattern", "format", "additionalproperties", "minlength", "maxlength",
    }
)


def extract_keywords(text: str, limit: int = 8) -> list[tuple[str, int]]:
    """漢字連続・カタカナ連続・英単語を頻度順（同率は初出順）に抽出する。"""
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for match in _KW_RE.finditer(text or ""):
        word = match.group(0)
        key = word.lower() if word.isascii() else word
        if key in _KW_STOPWORDS:
            continue
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, match.start())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    return ranked[: max(0, limit)]


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t]


def _singular(name: str) -> str:
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss") and len(name) > 1:
        return name[:-1]
    return name


def _has_hint(name: str, hints: tuple[str, ...]) -> bool:
    lowered = (name or "").lower()
    tokens = _name_tokens(lowered)
    for hint in hints:
        if hint in tokens or _singular(hint) in tokens:
            return True
        if any(token.startswith(hint) for token in tokens):
            return True
    return False


# ---------------------------------------------------------------------------
# 正規表現からの文字列生成
# ---------------------------------------------------------------------------

_MAXREPEAT_THRESHOLD = 10_000


def _op_name(op: Any) -> str:
    return str(getattr(op, "name", op)).upper()


class _RegexStringGenerator:
    """``re`` の構文木を辿って、パターンに一致する文字列を決定論的に生成する。"""

    def __init__(self, rng: random.Random, max_repeat_extra: int = 3) -> None:
        self.rng = rng
        self.extra = max_repeat_extra

    def generate(self, pattern: str) -> str | None:
        try:
            parsed = _sre_parse.parse(pattern)
        except Exception:  # 解釈不能なパターンは生成対象外
            return None
        try:
            return self._emit(list(parsed))
        except (RecursionError, ValueError, TypeError, IndexError, OverflowError):
            return None

    def _emit(self, items: list[Any]) -> str:
        parts: list[str] = []
        for op, av in items:
            name = _op_name(op)
            if name == "LITERAL":
                parts.append(chr(av))
            elif name == "NOT_LITERAL":
                parts.append("a" if chr(av) != "a" else "b")
            elif name == "ANY":
                parts.append(self.rng.choice("abcxyz"))
            elif name == "IN":
                parts.append(self._emit_in(list(av)))
            elif name == "BRANCH":
                branches = list(av[1])
                parts.append(self._emit(list(self.rng.choice(branches))))
            elif name == "SUBPATTERN":
                parts.append(self._emit(list(av[-1])))
            elif name == "ATOMIC_GROUP":
                parts.append(self._emit(list(av)))
            elif name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
                low, high, sub = av
                if high >= _MAXREPEAT_THRESHOLD:
                    high = low + self.extra
                upper = max(low, min(high, low + self.extra))
                count = self.rng.randint(low, upper)
                parts.append("".join(self._emit(list(sub)) for _ in range(count)))
            elif name == "CATEGORY":
                parts.append(self._emit_category(av))
        return "".join(parts)

    def _category_chars(self, category: Any) -> str:
        name = _op_name(category)
        if name == "CATEGORY_DIGIT":
            return "0123456789"
        if name == "CATEGORY_WORD":
            return "abcdefghijklmnopqrstuvwxyz0123456789_"
        if name == "CATEGORY_SPACE":
            return " "
        if name == "CATEGORY_NOT_DIGIT":
            return "abcdefghij"
        if name == "CATEGORY_NOT_WORD":
            return "-. "
        return "abcdefghij"

    def _emit_category(self, category: Any) -> str:
        return self.rng.choice(self._category_chars(category))

    def _emit_in(self, items: list[Any]) -> str:
        negate = False
        options: list[str] = []
        excluded: set[str] = set()
        for op, av in items:
            name = _op_name(op)
            if name == "NEGATE":
                negate = True
            elif name == "LITERAL":
                options.append(chr(av))
                excluded.add(chr(av))
            elif name == "RANGE":
                low, high = av
                span = max(0, high - low)
                options.append(chr(low + self.rng.randint(0, min(span, 25))))
                if span < 2048:
                    excluded.update(chr(code) for code in range(low, high + 1))
            elif name == "CATEGORY":
                chars = self._category_chars(av)
                options.append(self.rng.choice(chars))
                excluded.update(chars)
        if negate:
            for candidate in "axqZ0_-.":
                if candidate not in excluded:
                    return candidate
            return "あ"
        return self.rng.choice(options) if options else "a"


# ---------------------------------------------------------------------------
# JSON Schema 合成器
# ---------------------------------------------------------------------------


def _is_null_schema(schema: Any) -> bool:
    return isinstance(schema, dict) and (schema.get("type") == "null" or schema.get("const", 0) is None and "const" in schema)


def _merge_schemas(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key == "properties" and isinstance(value, dict):
            merged["properties"] = {**merged.get("properties", {}), **value}
        elif key == "required" and isinstance(value, list):
            merged["required"] = sorted(set(merged.get("required", [])) | set(value))
        else:
            merged[key] = value
    return merged


class JsonSchemaSynthesizer:
    """JSON Schema（pydantic v2 の ``model_json_schema()`` 出力を想定）から制約を満たす値を生成する。"""

    MAX_DEPTH = 7

    def __init__(self, schema: dict[str, Any], seed: int = 0) -> None:
        self.root: dict[str, Any] = schema if isinstance(schema, dict) else {}
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self._regex = _RegexStringGenerator(self.rng)
        self._serial = 0

    def generate(self) -> Any:
        """ルートスキーマのインスタンスを生成する。"""
        return self._gen(self.root, "root", 0)

    # -- 参照解決 ---------------------------------------------------------

    def _lookup_ref(self, ref: str) -> dict[str, Any]:
        if not ref.startswith("#"):
            return {}
        node: Any = self.root
        path = ref[1:].lstrip("/")
        if not path:
            return node if isinstance(node, dict) else {}
        for part in path.split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return {}
        return node if isinstance(node, dict) else {}

    def _resolve(self, schema: Any) -> Any:
        hops = 0
        while isinstance(schema, dict) and "$ref" in schema and hops < 32:
            target = self._lookup_ref(str(schema["$ref"]))
            siblings = {k: v for k, v in schema.items() if k != "$ref"}
            schema = {**target, **siblings}
            hops += 1
        return schema

    # -- 生成本体 ---------------------------------------------------------

    def _next_serial(self) -> int:
        self._serial += 1
        return self._serial

    def _gen(self, schema: Any, name: str, depth: int) -> Any:
        schema = self._resolve(schema)
        if schema is True or not isinstance(schema, dict):
            return self._sentence()
        if "const" in schema:
            return copy.deepcopy(schema["const"])
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            non_null = [v for v in enum_values if v is not None]
            return copy.deepcopy(self.rng.choice(non_null or enum_values))
        if "allOf" in schema:
            merged: dict[str, Any] = {}
            for sub in schema["allOf"]:
                merged = _merge_schemas(merged, self._resolve(sub))
            outer = {k: v for k, v in schema.items() if k != "allOf"}
            return self._gen(_merge_schemas(merged, outer), name, depth)
        for key in ("anyOf", "oneOf"):
            if key in schema and isinstance(schema[key], list) and schema[key]:
                options = [self._resolve(option) for option in schema[key]]
                non_null = [option for option in options if not _is_null_schema(option)]
                if (depth >= self.MAX_DEPTH or not non_null) and len(non_null) < len(options):
                    return None
                chosen = non_null[0] if non_null else options[0]
                outer = {k: v for k, v in schema.items() if k not in (key, "default", "title", "description")}
                return self._gen(_merge_schemas(chosen, outer) if outer else chosen, name, depth)
        typ = schema.get("type")
        if isinstance(typ, list):
            non_null_types = [t for t in typ if t != "null"]
            typ = non_null_types[0] if non_null_types else "null"
        if typ is None:
            if "properties" in schema or "additionalProperties" in schema:
                typ = "object"
            elif "items" in schema or "prefixItems" in schema:
                typ = "array"
            elif "default" in schema:
                return copy.deepcopy(schema["default"])
            else:
                typ = "string"
        if typ == "object":
            return self._gen_object(schema, name, depth)
        if typ == "array":
            return self._gen_array(schema, name, depth)
        if typ == "integer":
            return self._gen_integer(schema, name)
        if typ == "number":
            return self._gen_number(schema, name)
        if typ == "boolean":
            return self._gen_boolean(name)
        if typ == "null":
            return None
        return self._gen_string(schema, name)

    def _include_optional(self, depth: int) -> bool:
        if depth >= self.MAX_DEPTH - 1:
            return False
        threshold = 0.85 if depth < 2 else 0.6 if depth < 4 else 0.3
        return self.rng.random() < threshold

    def _dict_keys(self, name: str, count: int, key_schema: dict[str, Any] | None) -> list[str]:
        keys: list[str] = []
        if key_schema:
            attempts = 0
            while len(keys) < count and attempts < count * 8:
                attempts += 1
                candidate = str(self._gen_string(key_schema, "key"))
                if candidate not in keys:
                    keys.append(candidate)
            if keys:
                return keys
        lowered = (name or "").lower()
        if "example" in lowered:
            pool = ["manga", "marketing", "article"]
        elif "metric" in lowered:
            pool = ["score", "coverage", "accuracy", "latency_ms"]
        elif "weight" in lowered:
            pool = ["IN00", "MID", "OUT00", "BASE"]
        elif "theme" in lowered or "binding" in lowered or "variable" in lowered:
            pool = ["THEME", "SETTING", "PROTAGONIST_GOAL"]
        else:
            pool = ["alpha", "beta", "gamma", "delta", "epsilon"]
        return pool[: max(0, count)] if count <= len(pool) else pool + [f"key_{i}" for i in range(count - len(pool))]

    def _gen_object(self, schema: dict[str, Any], name: str, depth: int) -> dict[str, Any]:
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        result: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            if prop_name in required or self._include_optional(depth):
                result[prop_name] = self._gen(prop_schema, prop_name, depth + 1)
        additional = schema.get("additionalProperties")
        if not properties and additional is not False:
            min_props = int(schema.get("minProperties", 0))
            max_props = schema.get("maxProperties")
            if depth >= self.MAX_DEPTH:
                count = min_props
            else:
                count = max(min_props, 2 if depth < 4 else 1)
            if max_props is not None:
                count = min(count, int(max_props))
            value_schema = additional if isinstance(additional, dict) else {"type": "string"}
            key_schema = schema.get("propertyNames") if isinstance(schema.get("propertyNames"), dict) else None
            for key in self._dict_keys(name, count, key_schema):
                result[key] = self._gen(value_schema, key, depth + 1)
        return result

    def _gen_array(self, schema: dict[str, Any], name: str, depth: int) -> list[Any]:
        prefix = schema.get("prefixItems")
        if isinstance(prefix, list) and prefix:
            return [self._gen(item_schema, name, depth + 1) for item_schema in prefix]
        items_schema = schema.get("items", {})
        min_items = int(schema.get("minItems", 0))
        max_items = schema.get("maxItems")
        if depth >= self.MAX_DEPTH:
            count = min_items
        else:
            base = 3 if depth < 3 else 2 if depth < 5 else 1
            count = max(min_items, base + self.rng.randint(0, 1))
        if max_items is not None:
            count = min(count, int(max_items))
        count = max(count, min_items)
        unique = bool(schema.get("uniqueItems"))
        singular = _singular(name)
        out: list[Any] = []
        seen: set[str] = set()
        attempts = 0
        while len(out) < count and attempts < max(10, count * 12):
            attempts += 1
            item = self._gen(items_schema, singular, depth + 1)
            if unique:
                marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
                if marker in seen:
                    continue
                seen.add(marker)
            out.append(item)
        return out

    # -- スカラー ---------------------------------------------------------

    def _bounds(self, schema: dict[str, Any], name: str, integer: bool) -> tuple[float, float, bool, bool]:
        low = schema.get("minimum")
        high = schema.get("maximum")
        ex_low = schema.get("exclusiveMinimum")
        ex_high = schema.get("exclusiveMaximum")
        low_exclusive = False
        high_exclusive = False
        if isinstance(ex_low, bool):
            low_exclusive = ex_low
        elif ex_low is not None:
            low, low_exclusive = ex_low, True
        if isinstance(ex_high, bool):
            high_exclusive = ex_high
        elif ex_high is not None:
            high, high_exclusive = ex_high, True
        if low is None and high is None:
            if _has_hint(name, _SCORE_HINTS):
                low, high = 0.0, 100.0
            elif _has_hint(name, ("latency", "ms", "duration")):
                low, high = 10.0, 2000.0
            elif integer and _has_hint(name, _COUNT_HINTS):
                low, high = 1.0, 12.0
            elif integer:
                low, high = 0.0, 10.0
            else:
                low, high = 0.0, 1.0
        elif low is None:
            high_value = float(high)
            low = 0.0 if high_value > 0 else high_value - 10.0
        elif high is None:
            low_value = float(low)
            span = 100.0 if _has_hint(name, _SCORE_HINTS) else 10.0 if integer else 1.0
            if _has_hint(name, _UNIT_NUMBER_HINTS) and low_value < 1.0:
                high = max(low_value + 0.5, 1.0)
            else:
                high = low_value + span
        return float(low), float(high), low_exclusive, high_exclusive

    def _gen_number(self, schema: dict[str, Any], name: str) -> float:
        low, high, low_ex, high_ex = self._bounds(schema, name, integer=False)
        if high < low:
            low, high = high, low
        span = high - low
        eps = max(span * 1e-6, 1e-9)
        lo_eff = low + (eps if low_ex else 0.0)
        hi_eff = high - (eps if high_ex else 0.0)
        if hi_eff < lo_eff:
            hi_eff = lo_eff
        value = lo_eff + (hi_eff - lo_eff) * self.rng.uniform(0.25, 0.85)
        multiple = schema.get("multipleOf")
        if multiple:
            step = float(multiple)
            candidate = math.ceil(lo_eff / step) * step
            steps_available = max(0, int(math.floor((hi_eff - candidate) / step)))
            value = candidate + step * self.rng.randint(0, steps_available)
            return float(round(value, 10))
        value = round(value, 4)
        if value <= low and low_ex:
            value = lo_eff
        if value >= high and high_ex:
            value = hi_eff
        return float(min(max(value, lo_eff), hi_eff))

    def _gen_integer(self, schema: dict[str, Any], name: str) -> int:
        low, high, low_ex, high_ex = self._bounds(schema, name, integer=True)
        lo_int = int(math.floor(low)) + 1 if low_ex and float(low).is_integer() else int(math.ceil(low))
        hi_int = int(math.ceil(high)) - 1 if high_ex and float(high).is_integer() else int(math.floor(high))
        if hi_int < lo_int:
            hi_int = lo_int
        multiple = schema.get("multipleOf")
        if multiple:
            step = int(multiple)
            first = int(math.ceil(lo_int / step)) * step
            choices = list(range(first, hi_int + 1, step)) or [first]
            return int(self.rng.choice(choices))
        span = hi_int - lo_int
        if span <= 0:
            return lo_int
        inner_low = lo_int + span // 4
        inner_high = max(inner_low, hi_int - span // 4)
        return int(self.rng.randint(inner_low, inner_high))

    def _gen_boolean(self, name: str) -> bool:
        lowered = (name or "").lower()
        if lowered.startswith("is_") or lowered in ("passed", "ok", "success", "accepted", "enabled", "required"):
            return True
        return self.rng.random() < 0.55

    def _pick(self, pool: list[str]) -> str:
        return self.rng.choice(pool)

    def _sentence(self) -> str:
        nouns = self.rng.sample(_NOUNS, 3)
        template = self._pick(_SENTENCE_TEMPLATES)
        return template.format(a=self._pick(_ADJECTIVES), n1=nouns[0], n2=nouns[1], n3=nouns[2])

    def _title(self) -> str:
        nouns = self.rng.sample(_NOUNS, 3)
        template = self._pick(_TITLE_TEMPLATES)
        return template.format(a=self._pick(_ADJECTIVES), n1=nouns[0], n2=nouns[1], n3=nouns[2])

    def _slug(self, name: str) -> str:
        base = _singular((name or "item").lower())
        base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
        if not base or base in ("id", "root", "key", "ref", "slug", "code"):
            base = self._pick(_SLUG_WORDS)
        if not base[0].isalpha():
            base = f"x_{base}"
        return f"{base}_{self._next_serial()}"

    def _datetime(self) -> str:
        base = _dt.datetime(2026, 1, 1, 9, 0, 0, tzinfo=_dt.timezone.utc)
        moment = base + _dt.timedelta(minutes=self.rng.randint(0, 525_600))
        return moment.isoformat()

    def _string_for_name(self, name: str) -> str:
        lowered = (name or "").lower()
        tokens = _name_tokens(lowered)
        if lowered in ("id", "uid") or lowered.endswith("_id") or lowered.endswith("_ids") or lowered in ("slug", "ref"):
            return self._slug(name)
        if any(t in ("url", "uri", "link", "href", "website") for t in tokens):
            return ""
        if any(t in ("created", "updated", "timestamp", "datetime") for t in tokens) or lowered.endswith("_at"):
            return self._datetime()
        if "version" in tokens:
            return f"1.{self.rng.randint(0, 9)}.{self.rng.randint(0, 9)}"
        if any(t in ("language", "lang", "locale") for t in tokens):
            return "ja"
        if "color" in tokens or "colour" in tokens:
            return f"#{self.rng.randint(0, 0xFFFFFF):06x}"
        if "prompt" in tokens:
            return ", ".join(self.rng.sample(_PROMPT_TOKENS, 3))
        if any(t in ("mermaid", "diagram") for t in tokens):
            return "flowchart LR\n  A[提示] --> B[反転]\n  B --> C[余韻]"
        if _has_hint(lowered, _TITLE_HINTS):
            return self._title()
        if _has_hint(lowered, _LABEL_HINTS):
            return f"{self._pick(_ADJECTIVES)}{self._pick(_NOUNS)}"
        if _has_hint(lowered, _WORD_HINTS):
            return self._pick(_NOUNS)
        if _has_hint(lowered, _SENTENCE_HINTS):
            if self.rng.random() < 0.4:
                return self._sentence() + self._sentence()
            return self._sentence()
        return self._sentence()

    def _string_for_pattern(self, pattern: str, name: str, min_len: int, max_len: int | None) -> str:
        candidates: list[str] = []
        serial = self._next_serial()
        base = re.sub(r"[^a-z0-9]+", "_", _singular((name or "item").lower())).strip("_") or "item"
        if not base[0].isalpha() or base in ("id", "root", "key", "name", "ref", "slug", "code"):
            base = self._pick(_SLUG_WORDS)
        candidates.append(f"{base}_{serial}")
        candidates.append(f"{self._pick(_SLUG_WORDS)}_{base}_{serial}")
        candidates.append(f"{self._pick(_SCREAMING_WORDS)}_{self._pick(_SCREAMING_WORDS)}")
        candidates.append(self._pick(_SCREAMING_WORDS))
        candidates.append(f"{base}-{serial}")
        candidates.append(f"{base}{serial}")
        candidates.append(str(self.rng.randint(10, 99999)))
        candidates.append(f"{self.rng.randint(0, 0xFFFFFF):06x}")
        candidates.append(self._datetime()[:10])
        candidates.append(self._pick(_NOUNS))
        for _ in range(6):
            generated = self._regex.generate(pattern)
            if generated is not None:
                candidates.append(generated)
        for candidate in candidates:
            if not self._length_ok(candidate, min_len, max_len):
                continue
            try:
                if re.search(pattern, candidate):
                    return candidate
            except re.error:
                break
        for candidate in candidates:
            try:
                if re.search(pattern, candidate):
                    return self._fit_length(candidate, min_len, max_len)
            except re.error:
                break
        return self._fit_length(candidates[0], min_len, max_len)

    @staticmethod
    def _length_ok(value: str, min_len: int, max_len: int | None) -> bool:
        return len(value) >= min_len and (max_len is None or len(value) <= max_len)

    def _fit_length(self, value: str, min_len: int, max_len: int | None) -> str:
        result = value
        guard = 0
        while len(result) < min_len and guard < 50:
            result += self._sentence()
            guard += 1
        if len(result) < min_len:
            result = result + "x" * (min_len - len(result))
        if max_len is not None and len(result) > max_len:
            result = result[:max_len]
        return result

    def _gen_string(self, schema: dict[str, Any], name: str) -> str:
        fmt = schema.get("format")
        min_len = int(schema.get("minLength", 0))
        max_len_raw = schema.get("maxLength")
        max_len = int(max_len_raw) if max_len_raw is not None else None
        if fmt == "date-time":
            return self._datetime()
        if fmt == "date":
            return self._datetime()[:10]
        if fmt == "time":
            return f"{self.rng.randint(0, 23):02d}:{self.rng.randint(0, 59):02d}:{self.rng.randint(0, 59):02d}"
        if fmt == "uuid":
            return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
        if fmt == "email":
            return "researcher@localhost"
        if fmt in ("uri", "url", "iri", "uri-reference", "iri-reference"):
            return f"urn:caf:{self._slug(name)}"
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and pattern:
            return self._string_for_pattern(pattern, name, min_len, max_len)
        return self._fit_length(self._string_for_name(name), min_len, max_len)


# ---------------------------------------------------------------------------
# 既定ハンドラ: decompose / critique
# ---------------------------------------------------------------------------


def _payload_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _default_decompose(ctx: MockContext) -> dict[str, Any]:
    """入力の頻出概念と出力スキーマの必須項目から分解結果を構成する。"""
    objective = str(ctx.context.get("objective") or "")
    payload_text = truncate(_payload_text(ctx.context.get("payload")), 20_000)
    if not objective and not payload_text:
        payload_text = ctx.last_user_message
    keywords = extract_keywords(f"{objective}\n{payload_text}", limit=8)
    target = str(ctx.context.get("target_schema_name") or "出力スキーマ")
    required = [str(r) for r in (ctx.context.get("target_required_fields") or [])]
    if keywords:
        elements = [
            {
                "name": word,
                "description": f"「{word}」は入力中に{count}回現れる中核要素であり、{target}の骨格を決める手掛かりになる。",
                "importance": round(clamp(0.95 - 0.08 * index, 0.2, 1.0), 3),
            }
            for index, (word, count) in enumerate(keywords)
        ]
    else:
        elements = [
            {
                "name": "主題",
                "description": "入力から明示的な概念語が得られないため、要求全体を一つの主題として扱う。",
                "importance": 0.8,
            }
        ]
    required_text = "、".join(required[:10]) if required else "全必須項目"
    constraints = [
        {"name": "スキーマ厳密準拠", "description": f"{target}の必須項目（{required_text}）を欠落なく満たす。", "kind": "hard"},
        {"name": "値域と参照の整合", "description": "数値範囲・列挙値・ID参照を出力内で矛盾なく保つ。", "kind": "hard"},
        {"name": "脱フレーバー", "description": "固有名詞やジャンル固有の装飾は抽象的な役割・機能へ写像する。", "kind": "soft"},
        {"name": "因果の明示", "description": "各要素がなぜ必要か、何を引き起こすかを因果で接続する。", "kind": "soft"},
    ]
    top = [word for word, _ in keywords[:3]]
    if len(top) >= 2:
        essence = f"課題の本質は、{top[0]}と{top[1]}の関係を、{target}として再利用可能な構造へ定式化することにある。"
    elif top:
        essence = f"課題の本質は、{top[0]}を中心とする力学を{target}として定式化することにある。"
    else:
        essence = f"課題の本質は、要求を{target}として矛盾なく構造化することにある。"
    return {
        "essence": essence,
        "core_elements": elements,
        "constraints": constraints,
        "hidden_assumptions": [
            "入力の記述順序は、情報が受け手へ開示される順序を概ね反映している。",
            "頻出する概念ほど構造上の中心に位置する可能性が高い（ただし象徴は低頻度でも重要になりうる）。",
        ],
        "success_criteria": [
            f"{target}の全必須項目が意味的に充足されている。",
            "因果・数値・参照に矛盾がない。",
            "他分野へ転用しても力学が保存される抽象度に達している。",
        ],
        "abstraction_targets": [f"「{word}」を役割・機能の水準へ写像する" for word in top] or ["固有の文脈を役割と機能へ写像する"],
        "risks": [
            "頻出語が主体ではなく場所や象徴である可能性。",
            "出来事の時系列と情報開示の順序が異なる構成（倒叙・回想）を見落とす可能性。",
        ],
    }


_DESCRIPTIVE_KEY_HINTS = (
    "description", "summary", "rationale", "explanation", "function", "effect", "trigger", "essence",
    "content", "body", "text", "thesis", "logline", "claim", "evidence", "intent", "purpose", "guide",
)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_PROPER_NOUN_ALLOW = frozenset(
    {"Mermaid", "Elo", "Bradley", "Terry", "Gaussian", "Markdown", "Obsidian", "Python", "Comfy", "Flux", "Lora"}
)


@dataclass
class _DraftStats:
    strings: int = 0
    empty_strings: int = 0
    short_strings: int = 0
    lists: int = 0
    empty_lists: int = 0
    numbers: int = 0
    invalid_numbers: int = 0
    duplicate_lists: int = 0
    total_chars: int = 0
    max_depth: int = 0
    fields: int = 0
    empty_paths: list[str] = field(default_factory=list)
    short_paths: list[str] = field(default_factory=list)
    duplicate_paths: list[str] = field(default_factory=list)
    invalid_paths: list[str] = field(default_factory=list)
    empty_list_paths: list[str] = field(default_factory=list)
    proper_nouns: list[str] = field(default_factory=list)


def _walk_draft(value: Any, path: str, depth: int, stats: _DraftStats) -> None:
    stats.max_depth = max(stats.max_depth, depth)
    if isinstance(value, dict):
        for key, child in value.items():
            stats.fields += 1
            _walk_draft(child, f"{path}.{key}" if path else str(key), depth + 1, stats)
        return
    if isinstance(value, list):
        stats.lists += 1
        if not value:
            stats.empty_lists += 1
            stats.empty_list_paths.append(path or "(root)")
        if value and all(isinstance(item, (str, dict)) for item in value):
            markers = [json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) for item in value]
            if len(set(markers)) < len(markers):
                stats.duplicate_lists += 1
                stats.duplicate_paths.append(path or "(root)")
        for index, item in enumerate(value):
            _walk_draft(item, f"{path}[{index}]", depth + 1, stats)
        return
    if isinstance(value, str):
        stats.strings += 1
        stats.total_chars += len(value)
        key = path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        stripped = value.strip()
        if not stripped:
            stats.empty_strings += 1
            stats.empty_paths.append(path or "(root)")
        elif len(stripped) < 8 and any(hint in key for hint in _DESCRIPTIVE_KEY_HINTS):
            stats.short_strings += 1
            stats.short_paths.append(path or "(root)")
        for match in _PROPER_NOUN_RE.finditer(value):
            word = match.group(0)
            if word not in _PROPER_NOUN_ALLOW and word not in stats.proper_nouns and len(stats.proper_nouns) < 8:
                stats.proper_nouns.append(word)
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        stats.numbers += 1
        if isinstance(value, float) and not math.isfinite(value):
            stats.invalid_numbers += 1
            stats.invalid_paths.append(path or "(root)")


def _criterion_rationale(key: str, stats: _DraftStats, score: float) -> str:
    if key == "logical_consistency":
        return f"重複リスト {stats.duplicate_lists} 件、非有限数値 {stats.invalid_numbers} 件を検出。採点 {score:.1f}。"
    if key == "structural_depth":
        return f"総文字数 {stats.total_chars}、最大深さ {stats.max_depth}、短文 {stats.short_strings} 件から深さを評価。採点 {score:.1f}。"
    if key == "abstraction_quality":
        nouns = "、".join(stats.proper_nouns[:4]) or "なし"
        return f"固有名詞らしき語: {nouns}。短文 {stats.short_strings} 件。採点 {score:.1f}。"
    if key == "schema_fidelity":
        return f"文字列 {stats.strings} 件中の空欄 {stats.empty_strings} 件、空リスト {stats.empty_lists} 件から充足度を評価。採点 {score:.1f}。"
    if key == "actionability":
        return f"項目数 {stats.fields}、リスト {stats.lists} 件、空欄 {stats.empty_strings} 件から下流での利用可能性を評価。採点 {score:.1f}。"
    return f"全体統計（文字列 {stats.strings}・空欄 {stats.empty_strings}・重複 {stats.duplicate_lists}）から評価。採点 {score:.1f}。"


def _default_critique(ctx: MockContext) -> dict[str, Any]:
    """ドラフトを実際に走査し、ルーブリック各観点を 0〜10 で決定論的に採点する。"""
    draft = ctx.context.get("draft")
    rubric_raw = ctx.context.get("rubric") or [criterion.model_dump() for criterion in DEFAULT_RUBRIC]
    rubric: list[tuple[str, float]] = []
    for item in rubric_raw:
        if isinstance(item, dict) and item.get("key"):
            rubric.append((str(item["key"]), float(item.get("weight", 1.0) or 1.0)))
        elif isinstance(item, BaseModel):
            rubric.append((str(getattr(item, "key")), float(getattr(item, "weight", 1.0))))
    if not rubric:
        rubric = [(criterion.key, criterion.weight) for criterion in DEFAULT_RUBRIC]
    accept_score = float(ctx.context.get("accept_score", 85.0))
    stats = _DraftStats()
    if draft is None:
        scores = [{"key": key, "score": 2.0, "rationale": "評価対象のドラフトが存在しない。"} for key, _ in rubric]
        overall = 20.0
        return {
            "scores": scores,
            "logical_contradictions": [],
            "shallow_points": ["ドラフトが空のため内容を評価できない。"],
            "abstraction_gaps": [],
            "must_fix": ["スキーマに準拠した完全なドラフトを生成する。"],
            "overall_score": overall,
            "verdict": "accept" if overall >= accept_score else "revise",
            "summary": "ドラフト欠落のため総合 20.0 点。",
        }
    _walk_draft(draft, "", 0, stats)
    n_strings = max(1, stats.strings)
    empty_ratio = stats.empty_strings / n_strings
    short_ratio = stats.short_strings / n_strings
    duplicate_penalty = min(stats.duplicate_lists, 3)
    base_scores = {
        "logical_consistency": 9.0 - 1.2 * duplicate_penalty - 2.5 * min(stats.invalid_numbers, 2),
        "structural_depth": 8.6
        - (1.5 if stats.total_chars < 200 else 0.0)
        - (1.0 if stats.max_depth < 2 else 0.0)
        - 2.5 * short_ratio,
        "abstraction_quality": 8.8 - 0.6 * min(len(stats.proper_nouns), 4) - 2.0 * short_ratio,
        "schema_fidelity": (9.2 - 5.0 * empty_ratio - 0.5 * min(stats.empty_lists, 3)) if isinstance(draft, dict) else 2.0,
        "actionability": 8.4 - 3.0 * empty_ratio - 0.4 * min(stats.empty_lists, 4),
    }
    fallback_score = sum(base_scores.values()) / len(base_scores)
    scores: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in rubric:
        raw = base_scores.get(key, fallback_score) + ctx.rng.uniform(-0.15, 0.15)
        score = round(clamp(raw, 0.0, 10.0), 2)
        scores.append({"key": key, "score": score, "rationale": _criterion_rationale(key, stats, score)})
        weighted_sum += score * weight
        weight_total += weight
    overall = round(clamp(weighted_sum / max(weight_total, 1e-9) * 10.0, 0.0, 100.0), 2)
    contradictions = [f"{path} に重複した要素がある。" for path in stats.duplicate_paths[:5]]
    contradictions += [f"{path} に非有限の数値がある。" for path in stats.invalid_paths[:3]]
    shallow = [f"{path} の記述が短く、機能・因果まで掘り下げられていない。" for path in stats.short_paths[:5]]
    gaps = [f"固有名詞らしき語「{word}」が残存しており、役割へ抽象化すべき。" for word in stats.proper_nouns[:4]]
    must_fix: list[str] = []
    must_fix += [f"{path} が空欄。文脈から整合する具体的内容で埋める。" for path in stats.empty_paths[:5]]
    must_fix += [f"{path} の重複要素を統合し、各要素を固有の機能で区別する。" for path in stats.duplicate_paths[:3]]
    must_fix += [f"{path} を因果と機能が分かる一文以上に拡充する。" for path in stats.short_paths[:3]]
    must_fix += [f"「{word}」を抽象的な役割名へ置換する。" for word in stats.proper_nouns[:3]]
    verdict = "accept" if overall >= accept_score and not contradictions else "revise"
    if verdict == "revise" and not must_fix:
        must_fix.append("各要素の因果関係と数値の根拠を一段深く記述し、抽象と具体の対応を明示する。")
    summary = (
        f"総合 {overall:.1f} 点（基準 {accept_score:.1f}）。文字列 {stats.strings} 件・空欄 {stats.empty_strings} 件・"
        f"重複 {stats.duplicate_lists} 件・短文 {stats.short_strings} 件を検出し、判定は {verdict}。"
    )
    return {
        "scores": scores,
        "logical_contradictions": contradictions,
        "shallow_points": shallow,
        "abstraction_gaps": gaps,
        "must_fix": must_fix,
        "overall_score": overall,
        "verdict": verdict,
        "summary": summary,
    }


register_mock_handler("*", HarnessStage.DECOMPOSE)(_default_decompose)
register_mock_handler("*", HarnessStage.CRITIQUE)(_default_critique)


# ---------------------------------------------------------------------------
# モックアダプタ
# ---------------------------------------------------------------------------


class MockLLMAdapter(BaseLLMAdapter):
    """決定論的な高精度モック LLM。"""

    provider = "mock"

    def __init__(
        self,
        seed: int = 20260101,
        model: str = "mock-deepseek-v4.1-flash",
        failure_plan: dict[str, list[str]] | None = None,
        emit_think_tags: bool = True,
        simulated_latency_ms: float = 0.0,
        max_recorded_calls: int = 2000,
    ) -> None:
        self.seed = int(seed)
        self.model = model
        self.emit_think_tags = emit_think_tags
        self.simulated_latency_ms = max(0.0, float(simulated_latency_ms))
        self.max_recorded_calls = max(1, int(max_recorded_calls))
        self.calls: list[LLMRequest] = []
        self._failure_plan: dict[str, deque[str]] = {}
        self._plan_lock = threading.Lock()
        for key, behaviors in (failure_plan or {}).items():
            self.plan_failures(key, *behaviors)

    def plan_failures(self, key: str, *behaviors: str) -> None:
        """``"task:stage"`` / ``"task:*"`` / ``"*:*"`` キーに挙動キューを追加する。"""
        unknown = [b for b in behaviors if b not in FAILURE_BEHAVIORS]
        if unknown:
            raise ValueError(f"未知の障害挙動: {unknown}（許可: {sorted(FAILURE_BEHAVIORS)}）")
        with self._plan_lock:
            self._failure_plan.setdefault(key, deque()).extend(behaviors)

    def pending_failures(self) -> dict[str, list[str]]:
        """未消費の障害キュー。"""
        with self._plan_lock:
            return {key: list(queue) for key, queue in self._failure_plan.items() if queue}

    def _next_behavior(self, task: str, stage: str) -> str:
        with self._plan_lock:
            for key in (f"{task}:{stage}", f"{task}:*", "*:*"):
                queue = self._failure_plan.get(key)
                if queue:
                    return queue.popleft()
        return "ok"

    def _seed_for(self, request: LLMRequest) -> int:
        digest = hashlib.sha256()
        for message in request.messages:
            digest.update(message.role.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(message.content.encode("utf-8"))
            digest.update(b"\x1e")
        return stable_seed(
            self.seed, request.task, request.stage.value, request.schema_name or "", request.attempt, digest.hexdigest()
        )

    def _record(self, request: LLMRequest) -> None:
        self.calls.append(request)
        overflow = len(self.calls) - self.max_recorded_calls
        if overflow > 0:
            del self.calls[:overflow]

    async def _produce_payload(self, ctx: MockContext) -> tuple[Any, str]:
        lookup_stage = ctx.stage
        if lookup_stage == HarnessStage.REPAIR.value:
            lookup_stage = str(ctx.context.get("original_stage") or HarnessStage.SYNTHESIZE.value)
        handler = get_mock_handler(ctx.task, lookup_stage)
        result: Any = None
        strategy = "JsonSchemaSynthesizer"
        if handler is not None:
            strategy = getattr(handler, "__qualname__", getattr(handler, "__name__", "handler"))
            result = handler(ctx)
            if inspect.isawaitable(result):
                result = await result
        if result is None:
            if ctx.schema is not None:
                result = ctx.synthesize()
                strategy = "JsonSchemaSynthesizer"
            else:
                result = self._plain_text_reply(ctx)
                strategy = "plain_text"
        if isinstance(result, BaseModel):
            result = result.model_dump(mode="json")
        return result, strategy

    def _plain_text_reply(self, ctx: MockContext) -> str:
        message = ctx.last_user_message
        keywords = [word for word, _ in extract_keywords(message, limit=4)]
        focus = "、".join(keywords) if keywords else "入力全体"
        nouns = ctx.rng.sample(_NOUNS, 2)
        return (
            f"【モック応答】要点は「{focus}」です。{nouns[0]}と{nouns[1]}の関係を軸に整理すると、"
            "前提・転換・帰結の三段で構造化できます。"
        )

    @staticmethod
    def _garbage_text(ctx: MockContext) -> str:
        return (
            "申し訳ありません。指定された形式での出力を準備していましたが、"
            f"{ctx.schema_name or '要求'} の構成を確定できませんでした。条件を整理してから再度お試しください。"
        )

    @staticmethod
    def _violate_schema(payload: Any, ctx: MockContext) -> Any:
        schema = ctx.schema or {}
        if isinstance(payload, dict):
            broken = dict(payload)
            required = [r for r in (schema.get("required") or []) if r in broken]
            if required:
                broken.pop(required[0])
                return broken
            properties = schema.get("properties") or {}
            for name, prop in properties.items():
                prop_type = prop.get("type") if isinstance(prop, dict) else None
                if prop_type in ("number", "integer", "array", "boolean"):
                    broken[name] = "invalid-type-value"
                    return broken
            return [broken]
        return {"unexpected_payload": True}

    @staticmethod
    def _corrupt_repairably(body: str) -> str:
        corrupted = re.sub(r"([^\s{\[,])(\n\s*[}\]])", r"\1,\2", body)
        corrupted = corrupted.replace(": true", ": True").replace(": false", ": False").replace(": null", ": None")
        corrupted = "// mock: 修復可能な崩れを含む JSON\n" + corrupted
        stripped = corrupted.rstrip()
        if stripped.endswith("}"):
            corrupted = stripped[:-1]
        return corrupted

    @staticmethod
    def _reasoning_log(ctx: MockContext, behavior: str, strategy: str) -> str:
        stage_notes = {
            HarnessStage.DECOMPOSE.value: "入力から本質要素と制約を抽出し、重要度順に整列する。",
            HarnessStage.SYNTHESIZE.value: "分解結果を踏まえ、スキーマの必須項目を満たすドラフトを構成する。",
            HarnessStage.CRITIQUE.value: "ドラフトを走査し、空欄・重複・短文・数値範囲をルーブリックで採点する。",
            HarnessStage.REFINE.value: "批評の must_fix を反映し、整合性を保ったまま最終版を出力する。",
            HarnessStage.REPAIR.value: "検証エラーを解消した完全な JSON を再出力する。",
            HarnessStage.SINGLE.value: "単発要求として応答を構成する。",
        }
        lines = [
            f"[mock-deepseek] task={ctx.task} stage={ctx.stage} attempt={ctx.attempt} seed={ctx.seed}",
            f"- 受信メッセージ {len(ctx.request.messages)} 件 / 推定入力トークン {estimate_messages_tokens(ctx.request.messages)}",
            f"- 出力スキーマ: {ctx.schema_name or '自由記述'}",
            f"- 生成戦略: {strategy}",
            f"- 方針: {stage_notes.get(ctx.stage, stage_notes[HarnessStage.SINGLE.value])}",
        ]
        if ctx.is_repair:
            lines.append("- 前回出力の検証エラーを修復対象として再生成する。")
        if behavior != "ok":
            lines.append(f"- 障害注入モード: {behavior}")
        return "\n".join(lines)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """決定論的な応答を返す（障害注入計画があればそれに従う）。"""
        ensure_mock_handler_modules_loaded()
        self._record(request)
        started = time.perf_counter()
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        stage = request.stage.value
        behavior = self._next_behavior(request.task, stage)
        if behavior == "connection_error":
            raise LLMConnectionError(f"[mock] 擬似接続障害 task={request.task} stage={stage}")
        seed = self._seed_for(request)
        ctx = MockContext(
            request=request,
            task=request.task,
            stage=stage,
            schema=request.response_schema,
            schema_name=request.schema_name,
            context=dict(request.context),
            seed=seed,
            rng=random.Random(seed),
            attempt=request.attempt,
        )
        strategy = "none"
        if behavior == "empty":
            body = ""
        elif behavior == "garbage":
            body = self._garbage_text(ctx)
            strategy = "garbage"
        else:
            payload, strategy = await self._produce_payload(ctx)
            if behavior == "schema_violation":
                payload = self._violate_schema(payload, ctx)
            if isinstance(payload, str) and request.response_schema is None and not request.json_mode:
                body = payload
            else:
                text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2, default=str)
                if behavior == "repairable_json":
                    text = self._corrupt_repairably(text)
                body = f"```json\n{text}\n```"
        reasoning = self._reasoning_log(ctx, behavior, strategy)
        content = f"<think>\n{reasoning}\n</think>\n{body}" if self.emit_think_tags and body else body
        reasoning_tokens = estimate_tokens(reasoning)
        usage = TokenUsage(
            prompt_tokens=estimate_messages_tokens(request.messages),
            completion_tokens=estimate_tokens(body) + reasoning_tokens,
            reasoning_tokens=reasoning_tokens,
            estimated=True,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            model=request.model or self.model,
            provider=self.provider,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason="stop",
            raw={"mock": True, "behavior": behavior, "seed": seed, "strategy": strategy},
        )

    async def health_check(self) -> dict[str, Any]:
        """モックは常に利用可能。"""
        return {
            "ok": True,
            "provider": self.provider,
            "model": self.model,
            "detail": "決定論的モック LLM（外部接続なし）",
            "models": [self.model],
            "handlers": [f"{task}:{stage}" for task, stage in registered_mock_handlers()],
        }
