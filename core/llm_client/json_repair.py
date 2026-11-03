"""LLM 応答からの JSON 抽出と自己修復。

ローカル LLM の出力に頻出する崩れ方を文字単位スキャナで決定論的に修復する。

1. ``<think>…</think>``（未閉鎖・先頭の閉じタグのみも含む）の除去と思考ログの分離
2. ```json フェンス／最初のバランスした ``{…}`` ``[…]`` の抽出（文字列中の括弧・バッククォートを無視）
3. 字句正規化（パス1）: スマートクォート、``//``・``/* */``・``#`` コメント、Python/JS リテラル、
   単引用文字列、未引用キー、裸の単語値、文字列内の生改行・制御文字、文字列内の未エスケープ ``"``、数値表記
4. 構造修復（パス2）: 末尾カンマ除去、欠落カンマ・コロン挿入、途中切断の補完（null 補完・括弧クローズ）、
   不一致・過剰な閉じ括弧の整理
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import ValidationError

from core.common import truncate
from core.llm_client.errors import JSONExtractionError

__all__ = [
    "strip_reasoning_tags",
    "extract_json_block",
    "repair_json_text",
    "parse_json_lenient",
    "parse_json_lenient_with_info",
    "format_validation_errors",
    "loads_strict",
]

_THINK_OPEN_RE = re.compile(r"<\s*think(?:ing)?\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think(?:ing)?\s*>", re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+\-]*[ \t]*\r?\n?")
_TRAILING_FENCE_RE = re.compile(r"\s*```\s*\Z")
_JSON_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_INTEGER_RE = re.compile(r"-?\d+\Z")
_HEX4_RE = re.compile(r"[0-9a-fA-F]{4}")

_SMART_DOUBLE = frozenset({"“", "”", "„", "‟", "″", "＂"})
_SMART_SINGLE = frozenset({"‘", "’", "‚", "‛"})
_NUMBER_CHARS = frozenset("0123456789+-.eE")
_PHRASE_STOP = frozenset(',}]{["\n\r')
_LITERAL_MAP: dict[str, str] = {
    "true": "true",
    "True": "true",
    "TRUE": "true",
    "false": "false",
    "False": "false",
    "FALSE": "false",
    "null": "null",
    "Null": "null",
    "NULL": "null",
    "None": "null",
    "none": "null",
    "nil": "null",
    "undefined": "null",
    "NaN": "null",
    "nan": "null",
    "Infinity": "null",
    "infinity": "null",
    "inf": "null",
    "Inf": "null",
}
_INFINITY_WORDS = frozenset({"Infinity", "infinity", "inf", "Inf"})


# ---------------------------------------------------------------------------
# 思考タグ
# ---------------------------------------------------------------------------


def strip_reasoning_tags(text: str) -> tuple[str, str]:
    """``<think>`` ブロックを除去し ``(本文, 思考ログ)`` を返す。

    - 閉じた ``<think>…</think>`` は思考ログへ移す。
    - 未閉鎖の ``<think>`` 以降はすべて思考ログとみなす。
    - 開きタグ無しで先頭側に ``</think>`` だけがある場合（サーバが開きタグを省略する実装）は、
      その手前を思考ログとみなす。
    """
    if not text:
        return "", ""
    reasoning: list[str] = []
    body = text
    first_open = _THINK_OPEN_RE.search(body)
    first_close = _THINK_CLOSE_RE.search(body)
    if first_close is not None and (first_open is None or first_close.start() < first_open.start()):
        reasoning.append(body[: first_close.start()])
        body = body[first_close.end() :]
    out: list[str] = []
    pos = 0
    while True:
        m_open = _THINK_OPEN_RE.search(body, pos)
        if m_open is None:
            out.append(body[pos:])
            break
        out.append(body[pos : m_open.start()])
        m_close = _THINK_CLOSE_RE.search(body, m_open.end())
        if m_close is None:
            reasoning.append(body[m_open.end() :])
            break
        reasoning.append(body[m_open.end() : m_close.start()])
        pos = m_close.end()
    content = _THINK_CLOSE_RE.sub("", "".join(out))
    thoughts = "\n".join(part.strip() for part in reasoning if part.strip())
    return content.strip(), thoughts


# ---------------------------------------------------------------------------
# ブロック抽出
# ---------------------------------------------------------------------------


def _scan_balanced(text: str, start: int) -> tuple[int, bool]:
    """``start`` の開き括弧に対応する終端位置を返す（文字列リテラル内の括弧は無視）。

    戻り値は ``(終端の次の位置, バランスしたか)``。単引用文字列は改行で強制終了する
    （アポストロフィ混入時の暴走を防ぐ）。
    """
    depth = 0
    in_str: str | None = None
    escape = False
    n = len(text)
    i = start
    while i < n:
        ch = text[i]
        if in_str is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            elif ch == "\n" and in_str == "'":
                in_str = None
        elif ch == '"' or ch == "'":
            in_str = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth <= 0:
                return i + 1, True
        i += 1
    return n, False


def _skip_ws(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos] in " \t\r\n":
        pos += 1
    return pos


def _loads_ok(candidate: str) -> bool:
    try:
        loads_strict(candidate)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def _choose_candidate(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    parseable = [c for c in candidates if _loads_ok(c)]
    pool = parseable or candidates
    best = pool[0]
    for candidate in pool[1:]:
        if len(candidate) > len(best):
            best = candidate
    return best


def extract_json_block(text: str) -> str | None:
    """テキストから JSON 本体と思われる部分文字列を取り出す（見つからなければ None）。

    コードフェンス直後の括弧ブロックを最優先し、無ければ本文中のトップレベル括弧ブロックを走査する。
    複数候補がある場合は「そのまま parse できる候補の中で最長のもの」、無ければ最長の候補を返す。
    閉じフェンスではなく括弧の対応で終端を決めるため、文字列中に ``` を含む JSON でも切断されない。
    """
    if not text:
        return None
    fenced: list[str] = []
    for match in _FENCE_OPEN_RE.finditer(text):
        start = _skip_ws(text, match.end())
        if start < len(text) and text[start] in "{[":
            end, balanced = _scan_balanced(text, start)
            if balanced:
                fenced.append(text[start:end])
            else:
                fenced.append(_TRAILING_FENCE_RE.sub("", text[start:]))
    chosen = _choose_candidate(fenced)
    if chosen is not None:
        return chosen
    loose: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] in "{[":
            end, balanced = _scan_balanced(text, i)
            if balanced:
                loose.append(text[i:end])
                i = end
                continue
            loose.append(_TRAILING_FENCE_RE.sub("", text[i:]))
            break
        i += 1
    return _choose_candidate(loose)


# ---------------------------------------------------------------------------
# パス1: 字句正規化
# ---------------------------------------------------------------------------


def _normalize_number(token: str) -> str:
    """JSON 非準拠の数値表記（+1, .5, 1., 007, 1e）を正規化する。数値化できなければ JSON 文字列にする。"""
    if _JSON_NUMBER_RE.match(token):
        return token
    t = token[1:] if token.startswith("+") else token
    negative = t.startswith("-")
    body = t[1:] if negative else t
    body = re.sub(r"[eE][+-]?\Z", "", body)
    if body.startswith("."):
        body = "0" + body
    if body.endswith("."):
        body = body[:-1]
    leading = re.match(r"0+(?=\d)", body)
    if leading:
        body = body[leading.end() :]
    candidate = ("-" if negative else "") + body
    if _JSON_NUMBER_RE.match(candidate):
        return candidate
    try:
        value = float(candidate)
    except ValueError:
        return json.dumps(token, ensure_ascii=False)
    if not math.isfinite(value):
        return "null"
    if _INTEGER_RE.match(candidate):
        return str(int(candidate))
    return json.dumps(value)


def _is_string_terminator(s: str, k: int) -> bool:
    """引用符の直後位置 ``k`` から見て、その引用符が文字列の終端とみなせるか。"""
    n = len(s)
    saw_newline = False
    while k < n and s[k] in " \t\r\n":
        if s[k] in "\r\n":
            saw_newline = True
        k += 1
    if k >= n:
        return True
    ch = s[k]
    if ch in ",:}]":
        return True
    if saw_newline:
        return True
    if ch == "/" and k + 1 < n and s[k + 1] in "/*":
        return True
    return ch == "#"


def _read_string(s: str, i: int, out: list[str]) -> int:
    """位置 ``i`` の引用符から文字列を読み、正規 JSON 文字列として ``out`` に追加し、次の位置を返す。"""
    n = len(s)
    opener = s[i]
    if opener == '"':
        closers = frozenset({'"'})
    elif opener == "'":
        closers = frozenset({"'"})
    elif opener in _SMART_DOUBLE:
        closers = frozenset({'"', "”", "“"})
    else:
        closers = frozenset({"'", "’", "‘"})
    buf: list[str] = ['"']
    j = i + 1
    while j < n:
        c = s[j]
        if c == "\\":
            if j + 1 >= n:
                buf.append("\\\\")
                j += 1
                continue
            esc = s[j + 1]
            if esc == "'":
                buf.append("'")
                j += 2
                continue
            if esc in '"\\/bfnrt':
                buf.append("\\" + esc)
                j += 2
                continue
            if esc == "u" and _HEX4_RE.match(s, j + 2):
                buf.append(s[j : j + 6])
                j += 6
                continue
            if esc == "\n":
                buf.append("\\n")
                j += 2
                continue
            buf.append("\\\\")
            j += 1
            continue
        if c in closers:
            if _is_string_terminator(s, j + 1):
                buf.append('"')
                out.append("".join(buf))
                return j + 1
            buf.append('\\"' if c == '"' else c)
            j += 1
            continue
        if c == '"':
            buf.append('\\"')
        elif c == "\n":
            buf.append("\\n")
        elif c == "\r":
            buf.append("\\r")
        elif c == "\t":
            buf.append("\\t")
        elif ord(c) < 0x20:
            buf.append(f"\\u{ord(c):04x}")
        else:
            buf.append(c)
        j += 1
    buf.append('"')
    out.append("".join(buf))
    return n


def _read_phrase(s: str, i: int) -> tuple[str, int]:
    """裸の値を構造区切り文字の直前まで読み取る。"""
    n = len(s)
    j = i
    while j < n and s[j] not in _PHRASE_STOP:
        j += 1
    phrase = s[i:j].rstrip()
    return phrase, i + len(phrase)


def _normalize_lexical(s: str) -> str:
    out: list[str] = []
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        nxt = s[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = s.find("\n", i)
            i = n if end < 0 else end
            continue
        if ch == "/" and nxt == "*":
            end = s.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch == "#":
            end = s.find("\n", i)
            i = n if end < 0 else end
            continue
        if ch in "\"'" or ch in _SMART_DOUBLE or ch in _SMART_SINGLE:
            i = _read_string(s, i, out)
            continue
        if ch.isdigit() or (ch in "+-." and (nxt.isdigit() or (nxt == "." and ch in "+-"))):
            j = i + 1
            while j < n and s[j] in _NUMBER_CHARS:
                j += 1
            if j < n and (s[j].isalpha() or s[j] == "_"):
                phrase, i = _read_phrase(s, i)
                out.append(json.dumps(phrase, ensure_ascii=False))
                continue
            out.append(_normalize_number(s[i:j]))
            i = j
            continue
        if ch.isalpha() or ch in "_$":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] in "_$-."):
                j += 1
            word = s[i:j]
            k = _skip_ws(s, j)
            if k < n and s[k] == ":":
                out.append(json.dumps(word, ensure_ascii=False))
                i = j
                continue
            literal = _LITERAL_MAP.get(word)
            if literal is not None:
                if word in _INFINITY_WORDS and out and out[-1] == "-":
                    out.pop()
                out.append(literal)
                i = j
                continue
            phrase, i = _read_phrase(s, i)
            out.append(json.dumps(phrase, ensure_ascii=False))
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# パス2: 構造修復
# ---------------------------------------------------------------------------


def _tokenize(s: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch in "{}[],:":
            tokens.append(("punct", ch))
            i += 1
            continue
        if ch == '"':
            j = i + 1
            escape = False
            while j < n:
                c = s[j]
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    break
                j += 1
            if j >= n:
                tokens.append(("string", s[i:] + '"'))
                i = n
            else:
                tokens.append(("string", s[i : j + 1]))
                i = j + 1
            continue
        if ch == "-" or ch.isdigit():
            j = i + 1
            while j < n and s[j] in _NUMBER_CHARS:
                j += 1
            normalized = _normalize_number(s[i:j])
            kind = "string" if normalized.startswith('"') else "number"
            tokens.append((kind, normalized))
            i = j
            continue
        if ch.isalpha():
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            word = s[i:j]
            if word in ("true", "false", "null"):
                tokens.append(("literal", word))
            else:
                tokens.append(("string", json.dumps(word, ensure_ascii=False)))
            i = j
            continue
        i += 1
    return tokens


class _StructuralEmitter:
    """トークン列を状態機械で辿り、構文的に正しい JSON を再構成する。"""

    def __init__(self) -> None:
        self.out: list[str] = []
        self.stack: list[list[str]] = []
        self.root: list[str] = ["root", "value"]
        self.synthetic_keys = 0

    def frame(self) -> list[str]:
        return self.stack[-1] if self.stack else self.root

    def after_value(self) -> None:
        frame = self.frame()
        frame[1] = "done" if frame is self.root else "comma_or_end"

    def close_top(self) -> None:
        container, state = self.stack.pop()
        if container == "{":
            if state == "colon":
                self.out.append(":null")
            elif state == "value":
                self.out.append("null")
            elif state == "key_or_end" and self.out and self.out[-1] == ",":
                self.out.pop()
            self.out.append("}")
        else:
            if state == "value_or_end" and self.out and self.out[-1] == ",":
                self.out.pop()
            self.out.append("]")
        self.after_value()

    def synthetic_key(self) -> None:
        self.synthetic_keys += 1
        self.out.append(json.dumps(f"_item{self.synthetic_keys}"))
        self.out.append(":")

    def begin_container(self, opener: str) -> bool:
        frame = self.frame()
        container, state = frame
        if frame is self.root:
            if state != "value":
                return False
        elif container == "{":
            if state == "colon":
                self.out.append(":")
                frame[1] = "value"
            elif state != "value":
                if state == "comma_or_end":
                    self.out.append(",")
                self.synthetic_key()
                frame[1] = "value"
        elif state == "comma_or_end":
            self.out.append(",")
            frame[1] = "value_or_end"
        self.out.append(opener)
        self.stack.append([opener, "key_or_end" if opener == "{" else "value_or_end"])
        return True

    def emit_scalar(self, kind: str, text: str) -> None:
        frame = self.frame()
        container, state = frame
        if frame is not self.root and container == "{":
            if state in ("key_or_end", "comma_or_end"):
                if state == "comma_or_end":
                    self.out.append(",")
                key_text = text if kind == "string" else json.dumps(text, ensure_ascii=False)
                self.out.append(key_text)
                frame[1] = "colon"
                return
            if state == "colon":
                self.out.append(":")
                frame[1] = "value"
        elif frame is not self.root and container == "[":
            if state == "comma_or_end":
                self.out.append(",")
                frame[1] = "value_or_end"
        self.out.append(text)
        self.after_value()

    def feed(self, tokens: list[tuple[str, str]]) -> str:
        for kind, text in tokens:
            frame = self.frame()
            if frame is self.root and frame[1] == "done":
                break
            if kind == "punct":
                self._feed_punct(text)
            else:
                self.emit_scalar(kind, text)
        while self.stack:
            self.close_top()
        return "".join(self.out)

    def _feed_punct(self, text: str) -> None:
        frame = self.frame()
        container, state = frame
        if text == ",":
            if state == "comma_or_end" and frame is not self.root:
                self.out.append(",")
                frame[1] = "key_or_end" if container == "{" else "value_or_end"
            return
        if text == ":":
            if state == "colon":
                self.out.append(":")
                frame[1] = "value"
            return
        if text in "}]":
            want = "{" if text == "}" else "["
            if not any(entry[0] == want for entry in self.stack):
                return
            while self.stack and self.stack[-1][0] != want:
                self.close_top()
            self.close_top()
            return
        self.begin_container(text)


def repair_json_text(text: str) -> str:
    """崩れた JSON テキストを、``json.loads`` 可能な JSON テキストへ決定論的に修復する。"""
    if not text:
        return ""
    normalized = _normalize_lexical(text)
    tokens = _tokenize(normalized)
    return _StructuralEmitter().feed(tokens)


# ---------------------------------------------------------------------------
# 高水準 API
# ---------------------------------------------------------------------------


def loads_strict(text: str) -> Any:
    """標準 JSON として解析する（NaN/Infinity は null 扱い）。"""
    return json.loads(text, parse_constant=lambda _constant: None)


def parse_json_lenient_with_info(text: str) -> tuple[Any, bool]:
    """「直接解析 → ブロック抽出 → 修復」の順で JSON を得て ``(データ, 修復を要したか)`` を返す。

    思考タグを除去した本文に JSON が見当たらない場合のみ、未閉鎖 ``<think>`` 内へ答えが入り込んだ
    ケースの救済として元テキスト全体も探索する。全て失敗した場合は ``JSONExtractionError``。
    """
    if text is None or not str(text).strip():
        raise JSONExtractionError("応答が空のため JSON を抽出できません", text=text or "")
    raw = str(text)
    content, _thoughts = strip_reasoning_tags(raw)
    sources = [content]
    if raw != content and ("{" not in content and "[" not in content):
        sources.append(raw)
    last_error: Exception | None = None
    for source in sources:
        stripped = source.strip()
        if stripped:
            try:
                return loads_strict(stripped), False
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        block = extract_json_block(source)
        if block is None:
            continue
        try:
            return loads_strict(block), False
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
        repaired = repair_json_text(block)
        if repaired:
            try:
                return loads_strict(repaired), True
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
    detail = f": {last_error}" if last_error is not None else "（JSON らしき括弧ブロックがありません）"
    raise JSONExtractionError(f"JSON を抽出・修復できませんでした{detail}", text=raw)


def parse_json_lenient(text: str) -> Any:
    """``parse_json_lenient_with_info`` のデータ部分のみを返す。"""
    data, _repaired = parse_json_lenient_with_info(text)
    return data


def format_validation_errors(error: BaseException, limit: int = 20) -> str:
    """検証エラーを「- loc: msg (input=…)」形式の箇条書きに整形する（修復プロンプト用）。"""
    if isinstance(error, ValidationError):
        items = error.errors(include_url=False)
        lines: list[str] = []
        for item in items[: max(1, limit)]:
            loc = ".".join(str(part) for part in item.get("loc", ())) or "(root)"
            message = item.get("msg", "")
            value = item.get("input")
            try:
                rendered = json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                rendered = repr(value)
            lines.append(f"- {loc}: {message} (input={truncate(rendered, 160)})")
        if len(items) > limit:
            lines.append(f"- 他 {len(items) - limit} 件の検証エラー")
        return "\n".join(lines)
    if isinstance(error, JSONExtractionError):
        return f"- (json): {error}"
    return f"- ({type(error).__name__}): {error}"
