"""PNG デコーダ／エンコーダ（NumPy と標準ライブラリのみ）。

ネーム構造ドクターの画像逆解析（``page_vision``）は外部の画像ライブラリに依存しない。ブラウザは任意形式の
原稿画像を canvas で PNG 化して送信するため、PNG 仕様（ISO/IEC 15948）の非インタレース画像を完全に復号できれば
十分である。

- チャンク: IHDR / PLTE / tRNS / IDAT / IEND を解釈し、全チャンクの CRC-32 を検証する。未知の必須チャンクは拒否、
  補助チャンク（gAMA・pHYs・tEXt 等）は読み飛ばす。
- フィルタ: 型 0〜4（None / Sub / Up / Average / Paeth）。0〜2 のみの画像は行単位のベクトル演算、Average・Paeth を
  含む画像は「反対角線（x+y=一定）上の画素は互いに独立」という依存構造を利用した波面法でベクトル化して復元する
  （2000×1400 RGBA でも 1 秒前後）。
- 画素形式: color type 0/2/3/4/6、bit depth 1/2/4/8/16。アルファ（および tRNS）は白背景で合成し、
  輝度 Y = 0.299R + 0.587G + 0.114B の float32 グレースケール（白=1.0）を返す。
- Adam7 インタレースは非対応（明確な ValueError）。
"""

from __future__ import annotations

import base64
import binascii
import re
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PNG_SIGNATURE",
    "MAX_PIXELS",
    "PNGFormatError",
    "PNGInfo",
    "png_info",
    "decode_png",
    "decode_png_rgba",
    "encode_png_gray",
    "encode_png_rgb",
    "encode_data_url",
    "decode_data_url",
]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PIXELS = 60_000_000

_CHANNELS: dict[int, int] = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_ALLOWED_DEPTHS: dict[int, tuple[int, ...]] = {
    0: (1, 2, 4, 8, 16),
    2: (8, 16),
    3: (1, 2, 4, 8),
    4: (8, 16),
    6: (8, 16),
}
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


class PNGFormatError(ValueError):
    """不正または非対応の PNG データ。"""


@dataclass(frozen=True)
class PNGInfo:
    """IHDR の内容。"""

    width: int
    height: int
    bit_depth: int
    color_type: int
    interlace: int

    @property
    def channels(self) -> int:
        return _CHANNELS[self.color_type]

    @property
    def bits_per_pixel(self) -> int:
        return self.channels * self.bit_depth

    @property
    def filter_unit(self) -> int:
        """フィルタ演算の基準バイト幅（1 画素のバイト数、1 未満は 1）。"""
        return max(1, self.bits_per_pixel // 8)

    @property
    def stride(self) -> int:
        """フィルタ型バイトを除いた 1 行のバイト数。"""
        return (self.width * self.bits_per_pixel + 7) // 8


@dataclass
class _Parsed:
    info: PNGInfo
    palette: np.ndarray | None
    trns: bytes | None
    compressed: bytes


def _read_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise PNGFormatError("PNG データは bytes で渡してください")
    data = bytes(data)
    if not data.startswith(PNG_SIGNATURE):
        raise PNGFormatError("PNG シグネチャが不正です（PNG 画像ではありません）")
    position = len(PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    while position < len(data):
        if position + 8 > len(data):
            raise PNGFormatError("チャンクヘッダが途中で切れています")
        (length,) = struct.unpack(">I", data[position : position + 4])
        tag = data[position + 4 : position + 8]
        if length > 0x7FFFFFFF:
            raise PNGFormatError("チャンク長が仕様の上限を超えています")
        if not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in tag):
            raise PNGFormatError(f"不正なチャンク型です: {tag!r}")
        end = position + 8 + length
        if end + 4 > len(data):
            raise PNGFormatError(f"チャンク {tag.decode('ascii')} が途中で切れています")
        payload = data[position + 8 : end]
        (stored_crc,) = struct.unpack(">I", data[end : end + 4])
        if zlib.crc32(tag + payload) & 0xFFFFFFFF != stored_crc:
            raise PNGFormatError(f"チャンク {tag.decode('ascii')} の CRC が一致しません（データが破損しています）")
        chunks.append((tag, payload))
        position = end + 4
        if tag == b"IEND":
            break
    return chunks


def _parse_header(payload: bytes) -> PNGInfo:
    if len(payload) != 13:
        raise PNGFormatError("IHDR の長さが不正です")
    width, height, depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", payload)
    if width == 0 or height == 0:
        raise PNGFormatError("画像の幅・高さは 1 以上が必要です")
    if width * height > MAX_PIXELS:
        raise PNGFormatError(f"画素数が上限（{MAX_PIXELS:,} 画素）を超えています")
    if color_type not in _CHANNELS:
        raise PNGFormatError(f"未知の color type です: {color_type}")
    if depth not in _ALLOWED_DEPTHS[color_type]:
        raise PNGFormatError(f"color type {color_type} では bit depth {depth} は使用できません")
    if compression != 0 or filter_method != 0:
        raise PNGFormatError("未知の圧縮方式またはフィルタ方式です")
    if interlace == 1:
        raise PNGFormatError("Adam7 インタレース PNG は非対応です（非インタレースで保存し直してください）")
    if interlace != 0:
        raise PNGFormatError(f"未知のインタレース方式です: {interlace}")
    return PNGInfo(width=width, height=height, bit_depth=depth, color_type=color_type, interlace=interlace)


def _parse(data: bytes) -> _Parsed:
    chunks = _read_chunks(data)
    if not chunks or chunks[0][0] != b"IHDR":
        raise PNGFormatError("最初のチャンクが IHDR ではありません")
    info = _parse_header(chunks[0][1])
    palette: np.ndarray | None = None
    trns: bytes | None = None
    compressed: list[bytes] = []
    seen_end = False
    for tag, payload in chunks[1:]:
        if tag == b"IHDR":
            raise PNGFormatError("IHDR チャンクが重複しています")
        if tag == b"PLTE":
            if len(payload) % 3 != 0 or not 3 <= len(payload) <= 768:
                raise PNGFormatError("PLTE チャンクの長さが不正です")
            palette = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.float64)
        elif tag == b"tRNS":
            trns = payload
        elif tag == b"IDAT":
            compressed.append(payload)
        elif tag == b"IEND":
            seen_end = True
        elif 65 <= tag[0] <= 90:
            raise PNGFormatError(f"未知の必須チャンクです: {tag.decode('ascii')}")
    if not compressed:
        raise PNGFormatError("IDAT チャンクがありません")
    if not seen_end:
        raise PNGFormatError("IEND チャンクがありません（データが途中で切れています）")
    if info.color_type == 3 and palette is None:
        raise PNGFormatError("パレット画像に PLTE チャンクがありません")
    return _Parsed(info=info, palette=palette, trns=trns, compressed=b"".join(compressed))


def png_info(data: bytes) -> PNGInfo:
    """画素を復号せずに IHDR を読む（寸法・形式の事前検査用）。"""
    chunks = _read_chunks(data)
    if not chunks or chunks[0][0] != b"IHDR":
        raise PNGFormatError("最初のチャンクが IHDR ではありません")
    return _parse_header(chunks[0][1])


# ---------------------------------------------------------------------------
# フィルタの復元
# ---------------------------------------------------------------------------


def _unfilter_rows(data: np.ndarray, filter_types: np.ndarray, unit: int) -> np.ndarray:
    """フィルタ型 0〜2 のみの画像を行単位のベクトル演算で復元する。"""
    height, stride = data.shape
    out = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        line = data[y]
        kind = int(filter_types[y])
        if kind == 0:
            current = line.copy()
        elif kind == 1:
            current = np.cumsum(line.reshape(-1, unit), axis=0, dtype=np.uint8).reshape(-1)
        else:
            current = line + previous
        out[y] = current
        previous = current
    return out


def _unfilter_wavefront(data: np.ndarray, filter_types: np.ndarray, unit: int) -> np.ndarray:
    """Average・Paeth を含む画像の復元。

    画素 (y, x) の再構成値は左 (y, x-1)・上 (y-1, x)・左上 (y-1, x-1) にのみ依存するため、反対角線 d = x + y 上の
    画素は互いに独立で、d-1・d-2 上の値だけから同時に計算できる。行ごとに異なるフィルタ型も同時に処理する。
    """
    height, stride = data.shape
    units = stride // unit
    filtered = data.reshape(height, units, unit).astype(np.int32)
    recon = np.zeros((height, units, unit), dtype=np.int32)
    kinds = filter_types.astype(np.int32)
    for diagonal in range(height + units - 1):
        y_lo = max(0, diagonal - units + 1)
        y_hi = min(height - 1, diagonal)
        ys = np.arange(y_lo, y_hi + 1)
        xs = diagonal - ys
        has_left = (xs > 0)[:, None]
        has_up = (ys > 0)[:, None]
        left_x = np.maximum(xs - 1, 0)
        up_y = np.maximum(ys - 1, 0)
        a = np.where(has_left, recon[ys, left_x], 0)
        b = np.where(has_up, recon[up_y, xs], 0)
        c = np.where(has_left & has_up, recon[up_y, left_x], 0)
        kind = kinds[ys][:, None]
        estimate = a + b - c
        pa = np.abs(estimate - a)
        pb = np.abs(estimate - b)
        pc = np.abs(estimate - c)
        paeth = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
        predictor = np.where(
            kind == 1,
            a,
            np.where(kind == 2, b, np.where(kind == 3, (a + b) >> 1, np.where(kind == 4, paeth, 0))),
        )
        recon[ys, xs] = (filtered[ys, xs] + predictor) & 255
    return recon.reshape(height, stride).astype(np.uint8)


def _unfilter(raw: bytes, info: PNGInfo) -> np.ndarray:
    stride = info.stride
    expected = info.height * (stride + 1)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    if buffer.size < expected:
        raise PNGFormatError("画像データが不足しています（IDAT が途中で切れています）")
    rows = buffer[:expected].reshape(info.height, stride + 1)
    filter_types = rows[:, 0]
    if int(filter_types.max()) > 4:
        raise PNGFormatError("不正なフィルタ型が含まれています")
    data = rows[:, 1:]
    unit = info.filter_unit
    if np.any(filter_types >= 3):
        return _unfilter_wavefront(data, filter_types, unit)
    return _unfilter_rows(data, filter_types, unit)


def _unpack_samples(pixels: np.ndarray, info: PNGInfo) -> np.ndarray:
    """フィルタ復元済みのバイト列を (H, W, C) のサンプル値（整数）へ展開する。"""
    height, width, channels, depth = info.height, info.width, info.channels, info.bit_depth
    if depth == 8:
        return pixels[:, : width * channels].reshape(height, width, channels).astype(np.int64)
    if depth == 16:
        pairs = pixels[:, : width * channels * 2].reshape(height, width * channels, 2).astype(np.int64)
        return ((pairs[..., 0] << 8) | pairs[..., 1]).reshape(height, width, channels)
    per_byte = 8 // depth
    mask = (1 << depth) - 1
    shifts = (np.arange(per_byte - 1, -1, -1, dtype=np.int64) * depth)[None, None, :]
    values = (pixels.astype(np.int64)[:, :, None] >> shifts) & mask
    return values.reshape(height, -1)[:, :width].reshape(height, width, 1)


def _rgba(parsed: _Parsed) -> tuple[np.ndarray, np.ndarray]:
    """(RGB [0,1] float64 (H,W,3), alpha [0,1] float64 (H,W))。"""
    info = parsed.info
    try:
        raw = zlib.decompress(parsed.compressed)
    except zlib.error as exc:
        raise PNGFormatError(f"IDAT の zlib 伸長に失敗しました: {exc}") from exc
    samples = _unpack_samples(_unfilter(raw, info), info)
    max_value = float((1 << info.bit_depth) - 1)
    trns = parsed.trns
    color_type = info.color_type
    alpha = np.ones(samples.shape[:2], dtype=np.float64)
    if color_type == 0:
        gray = samples[..., 0]
        if trns is not None and len(trns) >= 2:
            (transparent,) = struct.unpack(">H", trns[:2])
            alpha = np.where(gray == transparent, 0.0, 1.0)
        level = gray.astype(np.float64) / max_value
        rgb = np.repeat(level[..., None], 3, axis=2)
    elif color_type == 2:
        if trns is not None and len(trns) >= 6:
            key = np.array(struct.unpack(">HHH", trns[:6]), dtype=np.int64)
            alpha = np.where(np.all(samples == key[None, None, :], axis=2), 0.0, 1.0)
        rgb = samples.astype(np.float64) / max_value
    elif color_type == 3:
        palette = parsed.palette
        assert palette is not None
        index = samples[..., 0]
        if int(index.max()) >= palette.shape[0]:
            raise PNGFormatError("パレットの範囲外の色番号が含まれています")
        rgb = palette[index] / 255.0
        table = np.full(palette.shape[0], 255.0)
        if trns is not None:
            entries = np.frombuffer(trns[: palette.shape[0]], dtype=np.uint8).astype(np.float64)
            table[: entries.size] = entries
        alpha = table[index] / 255.0
    elif color_type == 4:
        level = samples[..., 0].astype(np.float64) / max_value
        alpha = samples[..., 1].astype(np.float64) / max_value
        rgb = np.repeat(level[..., None], 3, axis=2)
    else:
        rgb = samples[..., :3].astype(np.float64) / max_value
        alpha = samples[..., 3].astype(np.float64) / max_value
    return rgb, alpha


def decode_png(data: bytes) -> np.ndarray:
    """PNG を復号し、白背景で合成した float32 グレースケール（白=1.0、形状 (H, W)）を返す。"""
    rgb, alpha = _rgba(_parse(data))
    luminance = rgb @ _LUMA
    composited = luminance * alpha + (1.0 - alpha)
    return np.clip(composited, 0.0, 1.0).astype(np.float32)


def decode_png_rgba(data: bytes) -> np.ndarray:
    """PNG を復号し、float32 の RGBA（[0,1]、形状 (H, W, 4)）を返す。"""
    rgb, alpha = _rgba(_parse(data))
    return np.concatenate([rgb, alpha[..., None]], axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# エンコード
# ---------------------------------------------------------------------------


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def _to_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == bool:
        return np.where(array, 255, 0).astype(np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("画像に NaN または無限大が含まれています")
        return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(array, 0, 255).astype(np.uint8)
    raise ValueError(f"画素型 {array.dtype} は PNG へ変換できません")


def _encode(rows: np.ndarray, width: int, height: int, color_type: int, unit: int, level: int) -> bytes:
    """Sub（先頭行）と Up（2 行目以降）フィルタで行を符号化し、PNG バイト列を組み立てる。"""
    filtered = np.empty((height, rows.shape[1] + 1), dtype=np.uint8)
    filtered[:, 0] = 2
    filtered[0, 0] = 1
    first = rows[0]
    shifted = np.concatenate([np.zeros(unit, dtype=np.uint8), first[:-unit]]) if first.size > unit else np.zeros_like(first)
    filtered[0, 1:] = first - shifted
    if height > 1:
        filtered[1:, 1:] = rows[1:] - rows[:-1]
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(filtered.tobytes(), level))
        + _chunk(b"IEND", b"")
    )


def encode_png_gray(image: Any, *, compression_level: int = 6) -> bytes:
    """2 次元（または (H, W, 1)）配列を 8bit グレースケール PNG に符号化する。

    浮動小数は [0, 1]（白=1）、整数は [0, 255]、bool は True=白 として解釈する。
    """
    array = _to_uint8(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("グレースケール画像は空でない 2 次元配列で渡してください")
    height, width = array.shape
    return _encode(np.ascontiguousarray(array), width, height, 0, 1, compression_level)


def encode_png_rgb(image: Any, *, compression_level: int = 6) -> bytes:
    """(H, W, 3) または (H, W, 4)（アルファは白背景で合成）の配列を 8bit RGB PNG に符号化する。"""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("RGB 画像は (H, W, 3) または (H, W, 4) の配列で渡してください")
    if array.shape[2] == 4:
        values = array.astype(np.float64)
        if not np.issubdtype(array.dtype, np.floating):
            values = values / 255.0
        alpha = np.clip(values[..., 3:4], 0.0, 1.0)
        array = values[..., :3] * alpha + (1.0 - alpha)
    rgb = _to_uint8(array)
    height, width = rgb.shape[:2]
    return _encode(np.ascontiguousarray(rgb.reshape(height, width * 3)), width, height, 2, 3, compression_level)


def encode_data_url(png: bytes) -> str:
    """PNG バイト列を ``data:image/png;base64,...`` 形式へ変換する。"""
    return "data:image/png;base64," + base64.b64encode(bytes(png)).decode("ascii")


_WHITESPACE_RE = re.compile(r"\s+")


def decode_data_url(data: str | bytes) -> bytes:
    """``data:<mime>;base64,<payload>`` 形式または素の base64 文字列をバイト列へ復号する。"""
    if isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("data URL / base64 は ASCII 文字列で渡してください") from exc
    elif isinstance(data, str):
        text = data
    else:
        raise ValueError("data URL / base64 は文字列で渡してください")
    text = text.strip()
    if text[:5].lower() == "data:":
        header, separator, payload = text.partition(",")
        if not separator:
            raise ValueError("data URL にカンマ区切りがありません")
        if ";base64" not in header.lower():
            raise ValueError("base64 エンコードされた data URL のみ対応しています")
    else:
        payload = text
    payload = _WHITESPACE_RE.sub("", payload).replace("-", "+").replace("_", "/")
    if not payload:
        raise ValueError("データが空です")
    remainder = len(payload.rstrip("=")) % 4
    if remainder == 1:
        raise ValueError("base64 の長さが不正です")
    payload = payload.rstrip("=") + "=" * ((4 - remainder) % 4)
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("base64 として解釈できないデータです") from exc
