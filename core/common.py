"""全モジュール共通の小さなユーティリティ群。

プロセス間・実行間で不変な安定ハッシュ（Python組込み ``hash`` はプロセスごとに
ランダム化されるため決定論性が要求される箇所では使わない）や、ID・時刻生成を提供する。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import re
import unicodedata
import uuid

__all__ = [
    "utc_now_iso",
    "new_id",
    "stable_hash",
    "stable_seed",
    "clamp",
    "slugify",
    "truncate",
    "safe_float",
]

_SEED_MODULUS = 2**31 - 1


def utc_now_iso() -> str:
    """UTCの現在時刻を秒精度のISO 8601文字列で返す。"""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """``{prefix}_{12桁hex}`` 形式の一意IDを生成する。"""
    clean = re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_") or "id"
    return f"{clean}_{uuid.uuid4().hex[:12]}"


def stable_hash(*parts: object) -> int:
    """引数列からプロセス非依存の64bit整数ハッシュを計算する。"""
    joined = "\x1f".join(repr(p) if not isinstance(p, str) else p for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def stable_seed(*parts: object) -> int:
    """``random.Random`` / ``numpy.random.default_rng`` 用の決定論的シード（0 <= seed < 2**31-1）。"""
    return stable_hash(*parts) % _SEED_MODULUS


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """値を閉区間 [lo, hi] に収める。NaN は lo として扱う。"""
    if lo > hi:
        lo, hi = hi, lo
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return lo
    return lo if x < lo else hi if x > hi else float(x)


def slugify(text: str, max_len: int = 48) -> str:
    """ASCIIスラッグを生成する。ASCII成分が無い場合（和文のみ等）は安定ハッシュ由来の値を返す。"""
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    if not slug:
        slug = f"h{stable_hash(text or '') % 16**10:010x}"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    if not slug[0].isalpha():
        slug = f"s_{slug}"[:max_len]
    return slug


def truncate(text: str, limit: int) -> str:
    """``limit`` 文字を超える場合は末尾を省略記号で切り詰める。"""
    if limit <= 0:
        return ""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "…"


def safe_float(value: object, default: float = 0.0) -> float:
    """数値変換に失敗した場合や非有限値の場合に既定値を返す float 変換。"""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result
