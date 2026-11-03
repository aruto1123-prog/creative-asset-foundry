"""縦書きセリフの文節改行と禁則処理、フキダシ寸法計算。

形態素解析器に依存しない決定論的ヒューリスティック:

1. 不可分単位（アトム）: カタカナ語（長音・中黒含む）、英数字連続、短い「」『』内語句、その他1文字。
2. 一次文節境界: 句読点群の直後、ひらがな（付属語）→ 漢字/カタカナ/英数/開き括弧 への遷移位置。
3. 二次文節境界（一次文節が列長を超える場合のみ使用）: 助詞・活用語尾（は/が/を/に/へ/と/で/も/の/や/
   から/まで/より/ね/よ/て/た/ます/です/ない 等）の直後。
4. 列への詰め込み: 文節を貪欲に列（max_chars_per_column）へ詰め、列長を超える文節のみ分割する。
5. 禁則: 行頭禁則文字（、。」』）！？ー…・小書き仮名 等）は前列末尾へ追い込み（最大2字のぶら下げ）、
   行末禁則文字（「『（ 等）は次列へ追い出す。

戻り値の列は読み順（右の列から左の列へ）。
"""

from __future__ import annotations

import re

__all__ = [
    "LINE_START_PROHIBITED",
    "LINE_END_PROHIBITED",
    "VERTICAL_FORMS",
    "ROTATE_IN_VERTICAL",
    "SMALL_KANA",
    "char_class",
    "visible_length",
    "tokenize_atoms",
    "split_bunsetsu",
    "break_japanese_text",
    "balloon_size",
    "fit_font_size",
    "split_text_for_balloons",
    "place_balloons",
]

LINE_START_PROHIBITED = set("、。，．,.」』）)】〕〉》”’！？!?ー～〜…‥・：；:;ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々ゝゞヽヾ")
LINE_END_PROHIBITED = set("「『（(【〔〈《“‘")
SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")

# 縦書き用の字形置換（Unicode 縦書き用形）と回転対象
VERTICAL_FORMS: dict[str, str] = {
    "、": "︑",
    "。": "︒",
    "，": "︐",
    "「": "﹁",
    "」": "﹂",
    "『": "﹃",
    "』": "﹄",
    "（": "︵",
    "）": "︶",
    "(": "︵",
    ")": "︶",
    "【": "︻",
    "】": "︼",
    "〔": "︹",
    "〕": "︺",
    "…": "︙",
    "‥": "︰",
    "！": "︕",
    "？": "︖",
    "：": "︓",
    "；": "︔",
}
ROTATE_IN_VERTICAL = set("ー～〜-－—―→←=＝")

_PRIMARY_PUNCT = set("、。，．,.！？!?…‥」』）)】〕〉》”’")
_OPEN_BRACKETS = set("「『（(【〔〈《“‘")
_SECONDARY_MARKERS = (
    "から",
    "まで",
    "より",
    "けど",
    "ので",
    "のに",
    "って",
    "ます",
    "です",
    "ない",
    "でも",
    "には",
    "では",
    "とは",
    "は",
    "が",
    "を",
    "に",
    "へ",
    "と",
    "で",
    "も",
    "の",
    "や",
    "ね",
    "よ",
    "て",
    "た",
)
_HONORIFIC_PREFIXES = {"お", "ご", "御"}
_ALNUM_RE = re.compile(r"[A-Za-z0-9Ａ-Ｚａ-ｚ０-９%％.,:'_\-]+")
_KATAKANA_RE = re.compile(r"[ァ-ヺーヽヾ・]+")
_QUOTE_RE = re.compile(r"[「『][^「『」』]{0,8}[」』]")


def char_class(ch: str) -> str:
    """文字種: H(ひらがな) K(カタカナ) J(漢字) A(英数) P(閉じ・句読点) O(開き括弧) S(空白) X(その他)。"""
    if not ch:
        return "X"
    if ch.isspace():
        return "S"
    code = ord(ch)
    if ch in _PRIMARY_PUNCT:
        return "P"
    if ch in _OPEN_BRACKETS:
        return "O"
    if 0x3041 <= code <= 0x309F:
        return "H"
    if 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or ch == "ー":
        return "K"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or ch in "々〆ヶ〇":
        return "J"
    if ch.isalnum():
        return "A"
    return "X"


def visible_length(text: str) -> int:
    """空白を除いた文字数。"""
    return len("".join(text.split()))


def tokenize_atoms(text: str, max_chars_per_column: int = 7) -> list[str]:
    """不可分単位（アトム）列へ分解する。"""
    atoms: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        quote = _QUOTE_RE.match(text, i)
        if quote and len(quote.group(0)) <= max_chars_per_column:
            atoms.append(quote.group(0))
            i = quote.end()
            continue
        katakana = _KATAKANA_RE.match(text, i)
        if katakana and char_class(ch) == "K":
            atoms.append(katakana.group(0))
            i = katakana.end()
            continue
        alnum = _ALNUM_RE.match(text, i)
        if alnum and char_class(ch) == "A":
            atoms.append(alnum.group(0))
            i = alnum.end()
            continue
        atoms.append(ch)
        i += 1
    return atoms


def _atom_class(atom: str) -> str:
    """アトムの種別（短い鉤括弧句は Q）。"""
    if atom[0] in "「『" and len(atom) > 1 and atom[-1] in "」』":
        return "Q"
    return char_class(atom[0])


def split_bunsetsu(text: str, max_chars_per_column: int = 7) -> list[str]:
    """一次文節境界で分割した文節列。"""
    atoms = tokenize_atoms(text, max_chars_per_column)
    if not atoms:
        return []
    chunks: list[str] = []
    current = ""
    previous_class = ""
    for atom in atoms:
        cls = _atom_class(atom)
        boundary = False
        if current and current not in _HONORIFIC_PREFIXES:
            if previous_class == "P" and cls != "P":
                boundary = True
            elif cls in ("O", "Q") and previous_class != "O":
                boundary = True
            elif previous_class in ("H", "Q") and cls in ("J", "K", "A"):
                boundary = True
        if boundary:
            chunks.append(current)
            current = ""
        current += atom
        if cls == "Q":
            previous_class = "Q"
        elif atom[-1] in _PRIMARY_PUNCT:
            previous_class = "P"
        else:
            previous_class = cls
    if current:
        chunks.append(current)
    return chunks


def _secondary_split(chunk: str, limit: int) -> list[str]:
    """列長を超える文節を、助詞・活用語尾の直後で分割する（無ければ強制分割）。"""
    if visible_length(chunk) <= limit:
        return [chunk]
    best = -1
    for size in range(min(limit, len(chunk) - 1), 0, -1):
        head = chunk[:size]
        if any(head.endswith(marker) for marker in _SECONDARY_MARKERS) and char_class(chunk[size]) != "K":
            best = size
            break
    if best <= 0:
        cut = limit
        while cut > 1 and (
            (char_class(chunk[cut - 1]) == "K" and char_class(chunk[cut]) == "K")
            or (char_class(chunk[cut - 1]) == "A" and char_class(chunk[cut]) == "A")
        ):
            cut -= 1
        if cut <= 1:
            cut = limit
        best = cut
    return [chunk[:best]] + _secondary_split(chunk[best:], limit)


def _leading_prohibited(column: str) -> int:
    count = 0
    while count < len(column) and column[count] in LINE_START_PROHIBITED:
        count += 1
    return count


def _split_long_column(column: str, limit: int) -> tuple[str, str]:
    """列長上限を超えた列を、後半が行頭禁則文字で始まらない位置で2分割する。"""
    cut = min(limit, len(column) - 1)
    while cut > 1 and (column[cut] in LINE_START_PROHIBITED or column[cut - 1] in LINE_END_PROHIBITED):
        cut -= 1
    return column[:cut], column[cut:]


def _apply_kinsoku(columns: list[str], limit: int) -> list[str]:
    """行頭禁則（追い込み／追い出し）と行末禁則（追い出し）を適用する。"""
    cols = [c for c in columns if c]
    hang = limit + 2
    i = 0
    guard = 0
    while i < len(cols) and guard < 512:
        guard += 1
        if i > 0:
            run = _leading_prohibited(cols[i])
            if run:
                previous = cols[i - 1]
                if len(previous) + run <= hang:
                    cols[i - 1] = previous + cols[i][:run]
                    cols[i] = cols[i][run:]
                else:
                    k = 1
                    while k < len(previous) and previous[-k] in LINE_START_PROHIBITED:
                        k += 1
                    cols[i - 1] = previous[:-k]
                    cols[i] = previous[-k:] + cols[i]
                    if not cols[i - 1]:
                        del cols[i - 1]
                        i = max(0, i - 1)
                        continue
                if not cols[i]:
                    del cols[i]
                    continue
        if i + 1 < len(cols):
            moved = ""
            while cols[i] and cols[i][-1] in LINE_END_PROHIBITED:
                moved = cols[i][-1] + moved
                cols[i] = cols[i][:-1]
            if moved:
                cols[i + 1] = moved + cols[i + 1]
                if not cols[i]:
                    del cols[i]
                    continue
        if len(cols[i]) > hang:
            head, tail = _split_long_column(cols[i], limit)
            cols[i] = head
            cols.insert(i + 1, tail)
        i += 1
    return [c for c in cols if c]


def break_japanese_text(text: str, max_chars_per_column: int = 7) -> list[str]:
    """縦書き用に文節単位で改行した列（読み順＝右→左）を返す。明示的な改行は強制改列。"""
    limit = max(2, int(max_chars_per_column))
    columns: list[str] = []
    for paragraph in str(text).replace("\r", "").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current = ""
        for chunk in split_bunsetsu(paragraph, limit):
            for piece in _secondary_split(chunk, limit):
                if not current:
                    current = piece
                elif visible_length(current) + visible_length(piece) <= limit:
                    current += piece
                else:
                    columns.append(current)
                    current = piece
        if current:
            columns.append(current)
    return _apply_kinsoku(columns, limit)


def fit_font_size(columns: list[str], width: float, height: float, *, base: float = 22.0, inset: float = 0.72) -> float:
    """フキダシ寸法に収まる縦書き文字サイズ（楕円内接係数 inset）。"""
    if not columns:
        return base
    n_cols = len(columns)
    max_len = max(len(c) for c in columns)
    by_width = (width * inset) / (n_cols * 1.12)
    by_height = (height * inset) / (max_len * 1.02)
    return max(8.0, min(base, by_width, by_height))


def balloon_size(columns: list[str], font_size: float = 22.0, kind: str = "normal") -> tuple[float, float]:
    """列構成からフキダシの幅・高さを算出する。"""
    n_cols = max(1, len(columns))
    max_len = max((len(c) for c in columns), default=1)
    text_w = n_cols * font_size * 1.12
    text_h = max_len * font_size * 1.02
    factor = {"normal": 1.5, "shout": 1.7, "monologue": 1.56, "flash": 1.75}.get(kind, 1.5)
    width = text_w * factor + 12.0
    height = text_h * factor + 12.0
    return (round(width, 2), round(height, 2))


def split_text_for_balloons(text: str, max_chars: int) -> list[str]:
    """1フキダシの上限文字数で台詞を自然な位置（句点・読点・文節）で分割する。"""
    clean = "".join(str(text).split())
    if visible_length(clean) <= max_chars:
        return [clean] if clean else []
    sentences = re.split(r"(?<=[。！？!?…])", clean)
    pieces: list[str] = []
    buffer = ""
    for sentence in (s for s in sentences if s):
        if visible_length(buffer + sentence) <= max_chars:
            buffer += sentence
            continue
        if buffer:
            pieces.append(buffer)
            buffer = ""
        if visible_length(sentence) <= max_chars:
            buffer = sentence
            continue
        chunk = ""
        for part in split_bunsetsu(sentence, max_chars):
            if visible_length(chunk + part) <= max_chars:
                chunk += part
            else:
                if chunk:
                    pieces.append(chunk)
                while visible_length(part) > max_chars:
                    pieces.append(part[:max_chars])
                    part = part[max_chars:]
                chunk = part
        buffer = chunk
    if buffer:
        pieces.append(buffer)
    return [p for p in pieces if p]


def _face_overlap(box: tuple[float, float, float, float], faces: list[tuple[float, float, float]]) -> float:
    x0, y0, x1, y1 = box
    total = 0.0
    for fx, fy, radius in faces:
        nearest_x = min(max(fx, x0), x1)
        nearest_y = min(max(fy, y0), y1)
        distance = ((fx - nearest_x) ** 2 + (fy - nearest_y) ** 2) ** 0.5
        if distance < radius:
            total += radius - distance
    return total


def place_balloons(
    box: tuple[float, float, float, float],
    sizes: list[tuple[float, float]],
    faces: list[tuple[float, float, float]] | None = None,
    *,
    pad: float = 10.0,
    spacing: float = 8.0,
) -> list[tuple[float, float, float, float]]:
    """コマのコア矩形内に、右上から読み順（右→左、段送りは下へ）でフキダシを配置する。

    戻り値は各フキダシの (中心x, 中心y, 幅, 高さ)。コマに収まらない寸法は縮小し、
    顔位置（faces: 中心と半径）との重なりが最小になる位置を横方向の候補から選ぶ。
    """
    x0, y0, x1, y1 = box
    faces = faces or []
    max_w = max(20.0, (x1 - x0) - 2 * pad)
    max_h = max(20.0, (y1 - y0) - 2 * pad)
    placements: list[tuple[float, float, float, float]] = []
    cursor_right = x1 - pad
    row_top = y0 + pad
    row_bottom = row_top
    for raw_w, raw_h in sizes:
        scale = min(1.0, max_w / raw_w, (max_h * 0.92) / raw_h)
        w = raw_w * scale
        h = raw_h * scale
        if placements and cursor_right - w < x0 + pad:
            cursor_right = x1 - pad
            row_top = row_bottom + spacing
        limit_left = x0 + pad + w / 2.0
        best: tuple[float, float] | None = None
        best_cost = float("inf")
        step = max(4.0, (x1 - x0) * 0.08)
        for k in range(5):
            cx = cursor_right - w / 2.0 - k * step
            if cx < limit_left and k > 0:
                break
            cy = row_top + h / 2.0
            cx = min(max(cx, limit_left), x1 - pad - w / 2.0) if w < max_w else (x0 + x1) / 2.0
            cy = min(max(cy, y0 + pad + h / 2.0), y1 - pad - h / 2.0) if h < max_h else (y0 + y1) / 2.0
            cost = _face_overlap((cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0), faces) + k * 0.5
            if cost < best_cost:
                best_cost = cost
                best = (cx, cy)
        assert best is not None
        cx, cy = best
        placements.append((round(cx, 2), round(cy, 2), round(w, 2), round(h, 2)))
        cursor_right = cx - w / 2.0 - spacing
        row_bottom = max(row_bottom, cy + h / 2.0)
    return placements
