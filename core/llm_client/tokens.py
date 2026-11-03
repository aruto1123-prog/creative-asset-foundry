"""トークン数の推定。

サーバが ``usage`` を返さない場合（一部のローカル推論サーバ・モック）に用いる。
DeepSeek 系トークナイザの経験則として、ASCII は約4文字/トークン、CJK（漢字・かな・ハングル・全角記号）は
約1文字/トークン、その他の非ASCII文字は約2文字/トークンで見積もる。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

__all__ = ["estimate_tokens", "estimate_messages_tokens", "is_cjk_char"]

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),  # CJK 記号・句読点
    (0x3040, 0x309F),  # ひらがな
    (0x30A0, 0x30FF),  # カタカナ
    (0x31F0, 0x31FF),  # カタカナ拡張
    (0x3400, 0x4DBF),  # CJK 統合漢字拡張A
    (0x4E00, 0x9FFF),  # CJK 統合漢字
    (0xAC00, 0xD7AF),  # ハングル音節
    (0xF900, 0xFAFF),  # CJK 互換漢字
    (0xFF00, 0xFFEF),  # 半角・全角形
    (0x20000, 0x2FA1F),  # CJK 拡張B〜互換補助
)

_MESSAGE_OVERHEAD_TOKENS = 4
_REPLY_PRIMING_TOKENS = 2


def is_cjk_char(ch: str) -> bool:
    """1文字が CJK 系の範囲に属するか。"""
    code = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= code <= hi:
            return True
    return False


def estimate_tokens(text: str | None) -> int:
    """テキストのトークン数を推定する（空文字は0、非空は最低1）。"""
    if not text:
        return 0
    ascii_chars = 0
    cjk_chars = 0
    other_chars = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_chars += 1
        elif is_cjk_char(ch):
            cjk_chars += 1
        else:
            other_chars += 1
    estimate = cjk_chars * 1.0 + ascii_chars / 4.0 + other_chars / 2.0
    return max(1, int(math.ceil(estimate)))


def estimate_messages_tokens(messages: Iterable[object]) -> int:
    """チャットメッセージ列（``content`` 属性または dict の ``content`` キー）の入力トークン数を推定する。"""
    total = _REPLY_PRIMING_TOKENS
    for message in messages:
        if isinstance(message, dict):
            content = str(message.get("content", ""))
        else:
            content = str(getattr(message, "content", ""))
        total += _MESSAGE_OVERHEAD_TOKENS + estimate_tokens(content)
    return total
