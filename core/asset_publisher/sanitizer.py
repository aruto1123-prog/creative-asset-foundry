"""Compliance & Style Sanitizer。

記事テキストに対して次の2つを厳格に保証する。

1. **一人称の「私」統一**: 僕・俺・わたし・筆者などの一人称表現を「私」（複数形は「私たち」）へ置換する。
   下僕・公僕などの合成語、「自分自身で」のような再帰用法、ひらがな語の部分一致（お礼＝おれい 等）は
   境界条件と文脈判定で誤爆させない。通常のコードブロック・インラインコードは対象外、
   Mermaid ブロック内のラベルは対象とする。
2. **URLの100%除去**: http/https 等のスキームURL、``www.``、スキーム無しドメイン（短縮URL・アフィリエイト系含む）、
   全角URL、難読化URL（hxxp・``[.]`` 等）、Markdown インライン/画像/参照リンク、自動リンク、HTML アンカー・
   埋め込み要素、Mermaid の ``click``/``href`` 行、mailto/javascript/data URI を検出・除去する。

全角URLは「全角ASCII→半角」の1対1変換（文字数不変）を施した影文字列上で検出し、
同一位置の元文字列を除去する。除去後は空括弧・空リスト項目・ラベルだけの行などを後始末し、
処理は冪等（2回目の適用で変化なし）である。
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from core.asset_publisher.schemas import (
    ArticleDraft,
    DraftSection,
    RemovedURL,
    Replacement,
    SanitizationReport,
    split_markdown_segments,
)

__all__ = [
    "FIRST_PERSON_CANONICAL",
    "FIRST_PERSON_PLURAL",
    "unify_first_person",
    "detect_first_person_variants",
    "remove_urls",
    "find_urls",
    "sanitize_text",
    "ComplianceSanitizer",
]

FIRST_PERSON_CANONICAL = "私"
FIRST_PERSON_PLURAL = "私たち"

# ---------------------------------------------------------------------------
# テキスト領域（prose / inline_code / code / mermaid）
# ---------------------------------------------------------------------------

_INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")

_PROSE = frozenset({"prose"})
_MERMAID = frozenset({"mermaid"})
_FIRST_PERSON_KINDS = frozenset({"prose", "mermaid"})
_ALL_KINDS = frozenset({"prose", "inline_code", "code", "mermaid"})


@dataclass(frozen=True)
class _Region:
    start: int
    end: int
    kind: str


def _regions(text: str) -> list[_Region]:
    regions: list[_Region] = []
    position = 0
    for kind, chunk in split_markdown_segments(text):
        start, end = position, position + len(chunk)
        if kind == "prose":
            cursor = start
            for match in _INLINE_CODE_RE.finditer(chunk):
                code_start, code_end = start + match.start(), start + match.end()
                if code_start > cursor:
                    regions.append(_Region(cursor, code_start, "prose"))
                regions.append(_Region(code_start, code_end, "inline_code"))
                cursor = code_end
            if cursor < end:
                regions.append(_Region(cursor, end, "prose"))
        else:
            regions.append(_Region(start, end, kind))
        position = end
    return regions


class _RegionIndex:
    """指定種別の領域に完全に含まれるスパンかを高速判定する。"""

    def __init__(self, regions: list[_Region], kinds: frozenset[str]) -> None:
        allowed = [(r.start, r.end) for r in regions if r.kind in kinds]
        self._starts = [a for a, _ in allowed]
        self._ends = [b for _, b in allowed]

    def contains(self, start: int, end: int) -> bool:
        index = bisect.bisect_right(self._starts, start) - 1
        if index < 0:
            return False
        return start >= self._starts[index] and end <= self._ends[index]


# ---------------------------------------------------------------------------
# 一人称統一
# ---------------------------------------------------------------------------

_HB = r"(?<![ぁ-ゖゝゞー])"
_KB = r"(?<![ァ-ヺヽヾー])"
_FOLLOW = (
    r"(?=(?:たち|達|ら|ども|共|自身|は|が|を|に|へ|と|で|や|の|も|から|まで|より|だけ|しか|こそ|さえ|なら|って"
    r"|[、。，．,.!?！？\s」』）)\]】〉》・…]|$))"
)
_NOT_RASHII = r"(?!し[いくさかけ])"
_SERVANT_COMPOUND = r"(?<![下公従家忠老学臣奴義])"

_FIRST_PERSON_RULES: list[tuple[str, str]] = [
    (_SERVANT_COMPOUND + r"僕(?:たち|達|等|ら" + _NOT_RASHII + r")", FIRST_PERSON_PLURAL),
    (r"俺(?:たち|達|等|ら" + _NOT_RASHII + r")", FIRST_PERSON_PLURAL),
    (_HB + r"ぼく(?:たち|ら" + _NOT_RASHII + r")" + _FOLLOW, FIRST_PERSON_PLURAL),
    (_KB + r"ボク(?:たち|達|ラ|ら" + _NOT_RASHII + r")" + _FOLLOW, FIRST_PERSON_PLURAL),
    (_HB + r"おれ(?:たち|ら" + _NOT_RASHII + r")" + _FOLLOW, FIRST_PERSON_PLURAL),
    (_KB + r"オレ(?:たち|達|ラ|ら" + _NOT_RASHII + r")" + _FOLLOW, FIRST_PERSON_PLURAL),
    (r"我々|我等|ワレワレ", FIRST_PERSON_PLURAL),
    (_HB + r"われわれ", FIRST_PERSON_PLURAL),
    (r"私(?:達|共|ども)", FIRST_PERSON_PLURAL),
    (_HB + r"わたくし(?:たち|ども|共|達)" + _FOLLOW, FIRST_PERSON_PLURAL),
    (_HB + r"わたし(?:たち|達|ども|ら" + _NOT_RASHII + r")" + _FOLLOW, FIRST_PERSON_PLURAL),
    (_KB + r"ワタシ(?:たち|達)" + _FOLLOW, FIRST_PERSON_PLURAL),
    (_HB + r"あたし(?:たち|達|ら" + _NOT_RASHII + r")" + _FOLLOW, FIRST_PERSON_PLURAL),
    (r"(?<![のた])筆者ら", FIRST_PERSON_PLURAL),
    (r"俺様", FIRST_PERSON_CANONICAL),
    (_SERVANT_COMPOUND + r"僕(?!婢)", FIRST_PERSON_CANONICAL),
    (r"俺", FIRST_PERSON_CANONICAL),
    (_HB + r"ぼく" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_KB + r"ボク" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_HB + r"おれ" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_KB + r"オレ" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_HB + r"わたくし" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_HB + r"わたし" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_KB + r"ワタシ" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_HB + r"あたくし" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_HB + r"あたし" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_KB + r"アタシ" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_HB + r"わし" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_KB + r"ワシ" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_HB + r"おいら" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (_KB + r"オイラ" + _FOLLOW, FIRST_PERSON_CANONICAL),
    (r"拙者|吾輩|我輩", FIRST_PERSON_CANONICAL),
    (r"小生(?!意気)", FIRST_PERSON_CANONICAL),
    (r"(?<![のた])筆者", FIRST_PERSON_CANONICAL),
    (r"当方", FIRST_PERSON_CANONICAL),
]

_FIRST_PERSON_RE = re.compile("|".join(f"(?P<r{i}>{pattern})" for i, (pattern, _) in enumerate(_FIRST_PERSON_RULES)))
_FIRST_PERSON_REPLACEMENT = {f"r{i}": replacement for i, (_, replacement) in enumerate(_FIRST_PERSON_RULES)}

# 「自分」は主語用法（文頭・読点直後で は/が/も/の/に が続く）かつ他者主語が先行しない場合のみ一人称とみなす
_JIBUN_RE = re.compile(r"(?<![ご御各])自分(?P<plural>たち|達|ら" + _NOT_RASHII + r")?(?!自身)(?=(?:は|が|も|の|に))")
_SENTENCE_BOUNDARY = set("。！？!?\n")
_CLAUSE_STARTERS = ("、", "，", ",", "「", "『", "（", "(", "・")
_LINE_PREFIX_RE = re.compile(r"^[ \t　]*(?:[-*+]|\d+[.)]|>)?[ \t　]*$")
_OTHER_SUBJECT_RE = re.compile(
    r"(?:読者|あなた|各自|誰|彼女|彼ら|彼|相手|皆さん|皆|みんな|人々|ユーザー|顧客|受講者|学習者|主人公|キャラクター"
    r"|作者|本人|子ども|子供|生徒|社員|部下|上司|クライアント)(?:さん|様|たち|ら|自身)?(?:は|が|も|こそ)"
)


def _jibun_is_first_person(text: str, start: int) -> bool:
    cursor = start - 1
    while cursor >= 0 and text[cursor] not in _SENTENCE_BOUNDARY:
        cursor -= 1
    prefix = text[cursor + 1 : start]
    stripped = prefix.rstrip(" \t　")
    at_sentence_start = _LINE_PREFIX_RE.match(stripped) is not None
    after_clause = stripped.endswith(_CLAUSE_STARTERS)
    if not (at_sentence_start or after_clause):
        return False
    return _OTHER_SUBJECT_RE.search(prefix) is None


def _first_person_matches(text: str) -> list[tuple[int, int, str, str]]:
    """(開始, 終了, 元表記, 置換後) の一覧（コード領域を除く）。"""
    index = _RegionIndex(_regions(text), _FIRST_PERSON_KINDS)
    found: list[tuple[int, int, str, str]] = []
    for match in _FIRST_PERSON_RE.finditer(text):
        if not match.group(0) or not index.contains(match.start(), match.end()):
            continue
        replacement = _FIRST_PERSON_REPLACEMENT[match.lastgroup or ""]
        found.append((match.start(), match.end(), match.group(0), replacement))
    for match in _JIBUN_RE.finditer(text):
        if not index.contains(match.start(), match.end()):
            continue
        if not _jibun_is_first_person(text, match.start()):
            continue
        replacement = FIRST_PERSON_PLURAL if match.group("plural") else FIRST_PERSON_CANONICAL
        found.append((match.start(), match.end(), match.group(0), replacement))
    found.sort(key=lambda item: item[0])
    return found


def detect_first_person_variants(text: str) -> list[str]:
    """「私」以外の一人称表現（残存）を出現順に返す。"""
    return [original for _, _, original, _ in _first_person_matches(text or "")]


def unify_first_person(text: str) -> tuple[str, list[Replacement]]:
    """一人称表現を「私」/「私たち」へ統一する。index は入力テキスト上の位置。"""
    source = text or ""
    matches = _first_person_matches(source)
    if not matches:
        return source, []
    pieces: list[str] = []
    replacements: list[Replacement] = []
    cursor = 0
    for start, end, original, replacement in matches:
        if start < cursor:
            continue
        pieces.append(source[cursor:start])
        pieces.append(replacement)
        context = source[max(0, start - 12) : min(len(source), end + 12)].replace("\n", " ")
        replacements.append(Replacement(original=original, replacement=replacement, index=start, context=context))
        cursor = end
    pieces.append(source[cursor:])
    return "".join(pieces), replacements


# ---------------------------------------------------------------------------
# URL 検出・除去
# ---------------------------------------------------------------------------

_WIDTH_TABLE = {code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)}
_WIDTH_TABLE[0x3000] = 0x20


def _fold(text: str) -> str:
    """全角ASCII（U+FF01〜FF5E）と全角空白を半角へ1対1で写像する（文字数不変）。"""
    return text.translate(_WIDTH_TABLE)


_URLCHARS = r"[A-Za-z0-9\-._~:/?#\[\]@!$&()*+,;=%]"
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_PATH = r"(?:[/?#]" + _URLCHARS + r"*)?"
_PORT = r"(?::\d{2,5})?"

_SCHEME_URL_RE = re.compile(
    r"(?<![A-Za-z0-9+.\-])(?:https?|ftps?|sftp|file|wss?|git|ssh|irc|rtsp|smb|telnet|gopher|nntp|news)://"
    + _URLCHARS
    + r"*",
    re.IGNORECASE,
)
_PSEUDO_TAIL = r"[!#-;=?-~]"
_PSEUDO_SCHEME_RE = re.compile(
    r"(?<![A-Za-z0-9+.\-])(?:(?:mailto|javascript|vbscript):"
    + _PSEUDO_TAIL
    + r"+|data:[a-z]+/[a-z0-9.+\-]+[;,]"
    + _PSEUDO_TAIL
    + r"*)",
    re.IGNORECASE,
)
_PROTOCOL_RELATIVE_RE = re.compile(
    r"(?<![A-Za-z0-9:/])//(?:" + _LABEL + r"\.)+[a-z]{2,24}(?![A-Za-z0-9\-])" + _PORT + _PATH,
    re.IGNORECASE,
)
_DEFANGED_SCHEME_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:hxxps?|h\*\*ps?|ttps?)(?::|\[:\]|\(:\))//" + _URLCHARS + r"*",
    re.IGNORECASE,
)
_DEFANGED_COLON_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:https?|ftp)(?:\[:\]|\(:\)|\[://\])(?://)?" + _URLCHARS + r"*",
    re.IGNORECASE,
)
_DEFANG_DOT = r"(?:\[\.\]|\(\.\)|\[dot\]|\(dot\)|\{\.\})"
_DEFANG_DOT_RE = re.compile(_DEFANG_DOT, re.IGNORECASE)
_DEFANGED_DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9.\-\[(])(?:" + _LABEL + r"(?:" + _DEFANG_DOT + r"|\.))+[a-z]{2,24}(?![A-Za-z0-9\-])" + _PATH,
    re.IGNORECASE,
)
_DOT_WORD_RE = re.compile(
    r"(?<![A-Za-z0-9])[a-z0-9][a-z0-9-]*(?:[ \t]+dot[ \t]+[a-z0-9][a-z0-9-]*)+(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_WWW_RE = re.compile(
    r"(?<![A-Za-z0-9.\-/@])www\d{0,3}\.(?:" + _LABEL + r"\.)*" + _LABEL + _PORT + _PATH,
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9.\-/@_\\])(?P<host>(?:" + _LABEL + r"\.)+(?P<tld>[a-z]{2,24}))(?![A-Za-z0-9\-_])(?P<rest>" + _PORT + _PATH + r")",
    re.IGNORECASE,
)
_AUTOLINK_RE = re.compile(
    r"<(?:[a-z][a-z0-9+.\-]{1,31}:[^\s<>]*|www\.[^\s<>]+|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})>",
    re.IGNORECASE,
)
_HTML_PAIRED_RE = re.compile(
    r"<\s*(script|iframe|video|audio|object|picture|noscript|embed)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_ANCHOR_RE = re.compile(r"<\s*a\b[^>]*>(?P<inner>.*?)<\s*/\s*a\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(
    r"<\s*/?\s*(?P<tag>a|img|iframe|script|link|source|video|audio|embed|object|track|picture|meta|base|form|frame"
    r"|frameset|area|param|noscript)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_STRIP_RE = re.compile(r"<[^>]+>")
_REFERENCE_DEF_RE = re.compile(
    r"^[ ]{0,3}\[(?P<label>[^\]\n^][^\]\n]*)\]:[ \t]*(?P<dest>\S+)(?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^)\n]*\)))?[ \t]*(?:\n|$)",
    re.MULTILINE,
)
_REFERENCE_FULL_RE = re.compile(r"\[(?P<text>[^\]\n]+)\]\[(?P<ref>[^\]\n]*)\]")
_REFERENCE_SHORTCUT_RE = re.compile(r"(?<![\]!\[])\[(?P<text>[^\]\n\[]+)\](?![\[(:])")
_MERMAID_LINK_LINE_RE = re.compile(r"^[ \t]*(?:click[ \t]+\S[^\n]*|[^\n]*\bhref\b[^\n]*)(?:\n|$)", re.MULTILINE | re.IGNORECASE)

SHORTENER_HOSTS: frozenset[str] = frozenset(
    {
        "bit.ly", "bitly.com", "amzn.to", "amzn.asia", "amzn.eu", "a.co", "a.r10.to", "r10.to", "t.co", "goo.gl",
        "tinyurl.com", "ow.ly", "buff.ly", "lnkd.in", "rebrand.ly", "is.gd", "v.gd", "cutt.ly", "x.gd", "tiny.cc",
        "shorturl.at", "rb.gy", "t.ly", "bl.ink", "s.id", "qr.ae", "youtu.be", "fb.me", "wp.me", "git.io", "db.tt",
        "trib.al", "dlvr.it", "ift.tt", "po.st", "adf.ly", "lin.ee", "onl.bz", "onl.tw", "urx.blue", "urx2.nu",
        "u0u1.net", "px.a8.net", "af.moshimo.com", "c.af.moshimo.com", "h.accesstrade.net", "hb.afl.rakuten.co.jp",
        "afl.rakuten.co.jp", "ck.jp.ap.valuecommerce.com", "click.linksynergy.com", "rcm-fe.amazon-adsystem.com",
    }
)
STRONG_TLDS: frozenset[str] = frozenset(
    {
        "com", "net", "org", "jp", "info", "biz", "xyz", "site", "online", "shop", "store", "tech", "blog", "page",
        "link", "club", "top", "work", "news", "live", "life", "world", "today", "space", "website", "cloud",
        "digital", "studio", "design", "media", "art", "fun", "games", "global", "group", "inc", "ltd", "company",
        "agency", "email", "jobs", "mobi", "name", "pro", "travel", "academy", "center", "solutions", "systems",
        "network", "social", "tokyo", "osaka", "nagoya", "kyoto", "yokohama", "gov", "edu", "mil", "int", "asia",
        "museum", "wiki", "download", "review", "guru", "ninja", "rocks", "land", "zone", "plus", "icu", "vip",
        "moe", "cafe", "tv", "app", "dev", "io", "ai", "me", "co",
    }
)
_KNOWN_SECOND_LEVEL: frozenset[str] = frozenset({"co", "ne", "or", "ac", "go", "ed", "gr", "lg", "com", "net", "org", "gov", "edu"})
_LIBRARY_LIKE_TLDS: frozenset[str] = frozenset({"app", "dev", "io", "ai", "me", "co"})
FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        "py", "md", "js", "ts", "tsx", "jsx", "rs", "sh", "rb", "pl", "cs", "java", "kt", "swift", "go", "cpp", "hpp",
        "json", "yaml", "yml", "toml", "ini", "cfg", "txt", "csv", "tsv", "html", "htm", "css", "scss", "less", "png",
        "jpg", "jpeg", "gif", "svg", "webp", "bmp", "pdf", "zip", "gz", "tar", "bz2", "xz", "safetensors", "ckpt", "pt",
        "pth", "bin", "onnx", "gguf", "exe", "dll", "so", "dylib", "log", "lock", "env", "sql", "sqlite", "sqlite3",
        "db", "ipynb", "vue", "mp4", "mp3", "wav", "mov", "avi", "php", "lua", "tex", "bib", "xml", "rst", "adoc",
        "bat", "ps1", "cmd", "conf", "wasm", "map", "jsonl", "parquet", "npz", "npy", "pkl", "whl", "egg", "cfm",
    }
)

_TRAILING_PUNCT = ".,;:!?*_~'\""


def _trim_url(candidate: str) -> str:
    """末尾の句読点と不釣り合いな閉じ括弧を取り除く。"""
    value = candidate
    while value:
        last = value[-1]
        if last in _TRAILING_PUNCT:
            value = value[:-1]
            continue
        if last == ")" and value.count(")") > value.count("("):
            value = value[:-1]
            continue
        if last == "]" and value.count("]") > value.count("["):
            value = value[:-1]
            continue
        if last == "}" and value.count("}") > value.count("{"):
            value = value[:-1]
            continue
        break
    return value


def _domain_is_url(host: str, tld: str, rest: str) -> bool:
    """スキーム無しドメイン候補を URL とみなすか判定する（ファイル名・技術語の誤爆防止）。"""
    host_l = host.lower()
    tld_l = tld.lower()
    if host_l in SHORTENER_HOSTS:
        return True
    if tld_l in FILE_EXTENSIONS:
        return False
    labels = host_l.split(".")
    has_path = bool(rest)
    if tld_l in _LIBRARY_LIKE_TLDS:
        return has_path or (len(labels) >= 3 and labels[-2] in _KNOWN_SECOND_LEVEL)
    if tld_l in STRONG_TLDS:
        return True
    if len(tld_l) == 2:
        return has_path or (len(labels) >= 3 and labels[-2] in _KNOWN_SECOND_LEVEL)
    return False


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    kind: str
    kept: str


Detector = Callable[[str, str], list[_Span]]


def _simple_detector(pattern: re.Pattern[str], kind: str, *, trim: bool = False, use_shadow: bool = True) -> Detector:
    """正規表現1本による検出器。URL本体は全角を畳んだ影文字列、Markdown/HTML 構文は元テキストで照合する。"""

    def detect(shadow: str, original: str) -> list[_Span]:
        target = shadow if use_shadow else original
        spans: list[_Span] = []
        for match in pattern.finditer(target):
            start, end = match.start(), match.end()
            if trim:
                trimmed = _trim_url(target[start:end])
                end = start + len(trimmed)
            if end > start:
                spans.append(_Span(start, end, kind, ""))
        return spans

    return detect


def _detect_html_anchor(shadow: str, original: str) -> list[_Span]:
    spans: list[_Span] = []
    for match in _HTML_ANCHOR_RE.finditer(original):
        inner = original[match.start("inner") : match.end("inner")]
        spans.append(_Span(match.start(), match.end(), "html_anchor", _HTML_STRIP_RE.sub("", inner)))
    return spans


def _detect_html_tags(shadow: str, original: str) -> list[_Span]:
    spans: list[_Span] = []
    for match in _HTML_TAG_RE.finditer(original):
        kind = "html_anchor" if match.group("tag").lower() == "a" else "html_media"
        spans.append(_Span(match.start(), match.end(), kind, ""))
    return spans


def _scan_inline_links(shadow: str, image: bool) -> list[tuple[int, int, int, int]]:
    """Markdown インラインリンク/画像を (全体開始, 全体終了, ラベル開始, ラベル終了) で返す（括弧の入れ子対応）。"""
    results: list[tuple[int, int, int, int]] = []
    length = len(shadow)
    cursor = 0
    while cursor < length:
        open_bracket = shadow.find("[", cursor)
        if open_bracket < 0:
            break
        is_image = open_bracket > 0 and shadow[open_bracket - 1] == "!"
        if is_image != image:
            cursor = open_bracket + 1
            continue
        depth = 0
        position = open_bracket
        close_bracket = -1
        while position < length:
            char = shadow[position]
            if char == "\\":
                position += 2
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    close_bracket = position
                    break
            elif char == "\n" and position + 1 < length and shadow[position + 1] == "\n":
                break
            position += 1
        if close_bracket < 0 or close_bracket + 1 >= length or shadow[close_bracket + 1] != "(":
            cursor = open_bracket + 1
            continue
        depth = 0
        position = close_bracket + 1
        close_paren = -1
        while position < length:
            char = shadow[position]
            if char == "\\":
                position += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close_paren = position
                    break
            elif char == "\n":
                break
            position += 1
        if close_paren < 0:
            cursor = open_bracket + 1
            continue
        start = open_bracket - 1 if image else open_bracket
        results.append((start, close_paren + 1, open_bracket + 1, close_bracket))
        cursor = close_paren + 1
    return results


def _detect_md_images(shadow: str, original: str) -> list[_Span]:
    return [_Span(start, end, "markdown_image", "") for start, end, _, _ in _scan_inline_links(original, image=True)]


def _detect_md_links(shadow: str, original: str) -> list[_Span]:
    return [
        _Span(start, end, "markdown_link", original[label_start:label_end])
        for start, end, label_start, label_end in _scan_inline_links(original, image=False)
    ]


def _detect_references(shadow: str, original: str) -> list[_Span]:
    spans: list[_Span] = []
    labels: set[str] = set()
    for match in _REFERENCE_DEF_RE.finditer(original):
        labels.add(match.group("label").strip().lower())
        spans.append(_Span(match.start(), match.end(), "reference_definition", ""))
    if not labels:
        return spans
    for match in _REFERENCE_FULL_RE.finditer(original):
        text = match.group("text")
        ref = (match.group("ref") or text).strip().lower()
        if ref in labels:
            spans.append(_Span(match.start(), match.end(), "reference_link", original[match.start("text") : match.end("text")]))
    for match in _REFERENCE_SHORTCUT_RE.finditer(original):
        if match.group("text").strip().lower() in labels:
            spans.append(
                _Span(match.start(), match.end(), "reference_link", original[match.start("text") : match.end("text")])
            )
    return spans


def _detect_defanged(shadow: str, original: str) -> list[_Span]:
    spans: list[_Span] = []
    for pattern in (_DEFANGED_SCHEME_RE, _DEFANGED_COLON_RE):
        for match in pattern.finditer(shadow):
            trimmed = _trim_url(match.group(0))
            if trimmed:
                spans.append(_Span(match.start(), match.start() + len(trimmed), "defanged_url", ""))
    for match in _DEFANGED_DOMAIN_RE.finditer(shadow):
        token = match.group(0)
        if not _DEFANG_DOT_RE.search(token):
            continue
        trimmed = _trim_url(token)
        if trimmed:
            spans.append(_Span(match.start(), match.start() + len(trimmed), "defanged_url", ""))
    for match in _DOT_WORD_RE.finditer(shadow):
        last_label = re.split(r"[ \t]+dot[ \t]+", match.group(0), flags=re.IGNORECASE)[-1].lower()
        if last_label in STRONG_TLDS or len(last_label) == 2:
            spans.append(_Span(match.start(), match.end(), "defanged_url", ""))
    return spans


def _detect_domains(shadow: str, original: str) -> list[_Span]:
    spans: list[_Span] = []
    for match in _DOMAIN_RE.finditer(shadow):
        rest = _trim_url(match.group("rest"))
        if not _domain_is_url(match.group("host"), match.group("tld"), rest):
            continue
        end = match.start("rest") + len(rest)
        spans.append(_Span(match.start(), end, "domain_like", ""))
    return spans


def _detect_mermaid_links(shadow: str, original: str) -> list[_Span]:
    return [_Span(m.start(), m.end(), "mermaid_click", "") for m in _MERMAID_LINK_LINE_RE.finditer(original)]


_PASSES: list[tuple[str, Detector, frozenset[str]]] = [
    ("html_paired", _simple_detector(_HTML_PAIRED_RE, "html_media", use_shadow=False), _PROSE),
    ("html_anchor", _detect_html_anchor, _PROSE),
    ("html_tag", _detect_html_tags, _PROSE),
    ("markdown_image", _detect_md_images, _PROSE),
    ("markdown_link", _detect_md_links, _PROSE),
    ("reference", _detect_references, _PROSE),
    ("autolink", _simple_detector(_AUTOLINK_RE, "autolink", use_shadow=False), _PROSE),
    ("mermaid_click", _detect_mermaid_links, _MERMAID),
    ("scheme", _simple_detector(_SCHEME_URL_RE, "bare_url", trim=True), _ALL_KINDS),
    ("pseudo_scheme", _simple_detector(_PSEUDO_SCHEME_RE, "bare_url", trim=True), _ALL_KINDS),
    ("protocol_relative", _simple_detector(_PROTOCOL_RELATIVE_RE, "bare_url", trim=True), _ALL_KINDS),
    ("defanged", _detect_defanged, _ALL_KINDS),
    ("www", _simple_detector(_WWW_RE, "bare_url", trim=True), _ALL_KINDS),
    ("domain", _detect_domains, _ALL_KINDS),
]


def _collect(
    text: str,
    detector: Detector,
    kinds: frozenset[str],
    *,
    regions: list[_Region] | None = None,
    shadow: str | None = None,
) -> list[_Span]:
    index = _RegionIndex(regions if regions is not None else _regions(text), kinds)
    folded = shadow if shadow is not None else _fold(text)
    return [span for span in detector(folded, text) if index.contains(span.start, span.end)]


def _apply_spans(text: str, spans: Iterable[_Span]) -> tuple[str, list[RemovedURL]]:
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    pieces: list[str] = []
    removed: list[RemovedURL] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor:
            continue
        pieces.append(text[cursor : span.start])
        pieces.append(span.kept)
        removed.append(RemovedURL(kind=span.kind, original=text[span.start : span.end], kept_text=span.kept))  # type: ignore[arg-type]
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces), removed


def find_urls(text: str) -> list[str]:
    """あらゆる形式のURL・リンク構文を検出して出現順（重複除去）に返す。"""
    source = text or ""
    regions = _regions(source)
    shadow = _fold(source)
    found: list[tuple[int, str]] = []
    for _, detector, kinds in _PASSES:
        for span in _collect(source, detector, kinds, regions=regions, shadow=shadow):
            found.append((span.start, source[span.start : span.end]))
    found.sort(key=lambda item: item[0])
    unique: list[str] = []
    for _, value in found:
        if value not in unique:
            unique.append(value)
    return unique


# ---------------------------------------------------------------------------
# 後始末
# ---------------------------------------------------------------------------

_EMPTY_PARENS_RE = re.compile(r"\([ \t　]*\)|（[ \t　]*）")
_LABEL_ONLY_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*+][ \t]+|\d+[.)][ \t]+)?(?:参照|出典|リンク|ソース|引用元|URL|Source|Reference|Link)s?[ \t]*[:：]?[ \t]*(?:\n|$)",
    re.MULTILINE | re.IGNORECASE,
)
_EMPTY_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]*(?:\n|$)", re.MULTILINE)
_INNER_SPACES_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+(?=\n|$)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+(?=[、。，．])")
# URL 除去で生じる読点の重複（「、、」）と読点・句点の衝突（「、。」「。、」）を畳み込む
_DUPLICATE_COMMA_RE = re.compile(r"、(?:[ \t　]*、)+")
_COMMA_BEFORE_PERIOD_RE = re.compile(r"、[ \t　]*。")
_COMMA_AFTER_PERIOD_RE = re.compile(r"。[ \t　]*、")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_LEADING_SPACES_RE = re.compile(r"^[ ]{1,3}(?=[^\s\-*+\d>|])", re.MULTILINE)


def _cleanup_prose(chunk: str) -> str:
    previous = None
    current = chunk
    rounds = 0
    while previous != current and rounds < 6:
        previous = current
        rounds += 1
        pieces: list[str] = []
        cursor = 0
        for match in _INLINE_CODE_RE.finditer(current):
            if not match.group(2).strip():
                pieces.append(current[cursor : match.start()])
                cursor = match.end()
        pieces.append(current[cursor:])
        current = "".join(pieces)
        current = _EMPTY_PARENS_RE.sub("", current)
        current = _LABEL_ONLY_LINE_RE.sub("", current)
        current = _EMPTY_LIST_ITEM_RE.sub("", current)
        current = _INNER_SPACES_RE.sub(" ", current)
        current = _SPACE_BEFORE_PUNCT_RE.sub("", current)
        current = _DUPLICATE_COMMA_RE.sub("、", current)
        current = _COMMA_BEFORE_PERIOD_RE.sub("。", current)
        current = _COMMA_AFTER_PERIOD_RE.sub("。", current)
        current = _TRAILING_SPACES_RE.sub("", current)
        current = _LEADING_SPACES_RE.sub("", current)
        current = _MANY_NEWLINES_RE.sub("\n\n", current)
    return current


def _cleanup_text(text: str) -> str:
    return "".join(
        _cleanup_prose(chunk) if kind == "prose" else chunk for kind, chunk in split_markdown_segments(text or "")
    )


def remove_urls(text: str) -> tuple[str, list[RemovedURL]]:
    """URL・リンク構文を除去する（リンクテキストは保持）。戻り値は (除去後テキスト, 除去記録)。"""
    current = text or ""
    removed: list[RemovedURL] = []
    for _round in range(4):
        changed = False
        for _, detector, kinds in _PASSES:
            spans = _collect(current, detector, kinds)
            if not spans:
                continue
            current, records = _apply_spans(current, spans)
            removed.extend(records)
            changed = changed or bool(records)
        current = _cleanup_text(current)
        if not changed or not find_urls(current):
            break
    return current, removed


# ---------------------------------------------------------------------------
# 統合サニタイズ
# ---------------------------------------------------------------------------


def sanitize_text(text: str) -> tuple[str, SanitizationReport]:
    """URL完全除去 → 一人称統一 → 後始末 → 残存検査を行う。"""
    without_urls, removed = remove_urls(text or "")
    unified, replacements = unify_first_person(without_urls)
    cleaned = _cleanup_text(unified)
    residual_urls = find_urls(cleaned)
    if residual_urls:
        cleaned, extra = remove_urls(cleaned)
        removed.extend(extra)
        residual_urls = find_urls(cleaned)
    residual_first_person = detect_first_person_variants(cleaned)
    report = SanitizationReport(
        first_person_replacements=replacements,
        urls_removed=removed,
        first_person_unified=not residual_first_person,
        url_free=not residual_urls,
        residual_first_person=residual_first_person,
        residual_urls=residual_urls,
    )
    return cleaned, report


class ComplianceSanitizer:
    """記事コンプライアンス（一人称「私」統一・URL完全除去）の実行器。"""

    def sanitize(self, text: str) -> tuple[str, SanitizationReport]:
        return sanitize_text(text)

    def verify(self, text: str) -> SanitizationReport:
        """テキストを変更せず、残存する一人称変種とURLを検査する。"""
        residual_first_person = detect_first_person_variants(text or "")
        residual_urls = find_urls(text or "")
        return SanitizationReport(
            first_person_unified=not residual_first_person,
            url_free=not residual_urls,
            residual_first_person=residual_first_person,
            residual_urls=residual_urls,
        )

    def sanitize_draft(self, draft: ArticleDraft) -> tuple[ArticleDraft, SanitizationReport]:
        """タイトル・リード・各章の見出しと本文・結語へ適用し、統合報告を返す。"""
        replacements: list[Replacement] = []
        removed: list[RemovedURL] = []

        def _clean(value: str, fallback: str) -> str:
            cleaned, report = sanitize_text(value)
            replacements.extend(report.first_person_replacements)
            removed.extend(report.urls_removed)
            cleaned = cleaned.strip()
            return cleaned if cleaned else fallback

        title = _clean(draft.title, "無題")
        lead = _clean(draft.lead, title)
        sections: list[DraftSection] = []
        for section in draft.sections:
            heading = _clean(section.heading, section.role.value)
            body = _clean(section.body_markdown, heading)
            sections.append(DraftSection(role=section.role, heading=heading, body_markdown=body))
        closing = _clean(draft.closing, title)
        cleaned_draft = ArticleDraft(title=title, lead=lead, sections=sections, closing=closing)
        verification = self.verify(cleaned_draft.to_markdown())
        report = SanitizationReport(
            first_person_replacements=replacements,
            urls_removed=removed,
            first_person_unified=verification.first_person_unified,
            url_free=verification.url_free,
            residual_first_person=verification.residual_first_person,
            residual_urls=verification.residual_urls,
        )
        return cleaned_draft, report
