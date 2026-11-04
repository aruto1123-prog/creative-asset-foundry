"""ページ画像からのコマ・フキダシ逆解析（Page Vision）。

既存の漫画原稿画像（PNG。ブラウザで任意形式を PNG 化して送信）から、ネーム構造ドクターが診断できる
``PageAnnotation`` を自動生成する。外部の画像ライブラリ・機械学習には依存せず、NumPy と古典的画像処理のみで動く。

処理の流れ（``PageVisionAnalyzer.detect``）:

1. グレースケール化（白=1.0）し、長辺 ``detail_long_side`` へ面積平均で縮小した「詳細画像」を作る。
2. 大津法でしきい値 t を求め、墨（ink = 画素 < t）を二値化する。
3. 墨マスクを長辺 ``analysis_long_side`` の「解析画像」へ縮小する。縮小はブロック内の墨画素率が
   ``pool_ink_fraction`` 以上なら墨とする「墨保存プーリング」で、細いコマ枠線を消さずにガター（余白）を残す。
4. 解析画像の白画素を 4 近傍で連結成分に分け、画像外周の帯（``border_band_ratio``）に触れる成分を背景
   （ページ余白＋ガター）とする。外周に細い黒縁がある走査画像でも背景を取り違えない。
5. 背景以外を 8 近傍で連結成分に分け、面積比が ``min_panel_area_ratio`` 以上のものをコマとする（斜めゴマ対応）。
   フキダシの尻尾などがガターを横切って 2 コマが細く連結した成分は、侵食で分離できれば分割する。
   ガターが見つからない（非背景が 1 成分で面積比 > 0.9）ときは投影プロファイルの再帰 XY カットへ切り替える。
6. 各コマについて、外接矩形・凸包多角形（Andrew の単調鎖）・傾き・充填率と凸性による信頼度を求め、
   詳細画像上でコマ枠線を侵食で除いた内部の墨比率（ベタ・トーン・描線を含む黒面積率）を測る。
7. コマ内部で枠に接しない白の連結成分のうち、穴埋め後の面積比・充填率（楕円≈0.785）・縦横比が条件を満たす
   ものをフキダシとし、内部の墨（字形）を縦書き列に分解して「列の高さ ÷（列幅×字送り比）」で文字数を推定する。
8. 全座標をページ座標（幅 ``page_width``、高さは画像の縦横比）へ変換し、コマを日本式の読み順（右上→左下）に並べる。

``synthesize_page_image`` は正解アノテーション付きの合成原稿画像を生成する（テストと UI デモ用）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.manga_pipeline.diagnostics_schemas import (
    AnnotatedBalloon,
    AnnotatedPage,
    AnnotatedPanel,
    DetectedBalloon,
    DetectedPanel,
    ImageDetectionResult,
    PageAnnotation,
    PageDetection,
)
from core.manga_pipeline.geometry import xy_cut_order
from core.manga_pipeline.png_codec import decode_data_url, decode_png
from core.manga_pipeline.schemas import BalloonKind, PageSide, Point

__all__ = [
    "SYNTH_VARIANTS",
    "VisionConfig",
    "PageVisionAnalyzer",
    "to_grayscale",
    "area_downscale",
    "otsu_threshold",
    "label_components",
    "convex_hull",
    "simplify_polygon",
    "polygon_area",
    "synthesize_page_image",
]

SYNTH_VARIANTS: tuple[str, ...] = ("standard", "slanted", "dense", "splash")


class VisionConfig(BaseModel):
    """画像逆解析の設定。"""

    model_config = ConfigDict(extra="ignore")

    analysis_long_side: int = Field(520, ge=96, le=4096, description="コマ検出に用いる解析画像の長辺 px")
    detail_long_side: int = Field(1200, ge=200, le=6000, description="ベタ比率・フキダシ解析に用いる詳細画像の長辺 px")
    min_panel_area_ratio: float = Field(0.01, gt=0.0, lt=1.0, description="コマとみなす最小面積比（ページ面積比）")
    min_balloon_area_ratio: float = Field(0.004, gt=0.0, lt=1.0, description="フキダシとみなす最小面積比（コマ面積比）")
    max_balloon_area_ratio: float = Field(0.6, gt=0.0, le=1.0, description="フキダシとみなす最大面積比（コマ面積比）")
    page_width: float = Field(1000.0, gt=0.0, description="出力ページ座標系の幅")
    ink_threshold: float | None = Field(None, gt=0.0, lt=1.0, description="墨判定しきい値（None で大津法）")
    pool_ink_fraction: float = Field(0.15, gt=0.0, le=1.0, description="解析画像へ縮小する際に墨とするブロック内墨画素率")
    border_band_ratio: float = Field(0.012, gt=0.0, lt=0.2, description="背景の種とする外周帯の幅（長辺比）")
    gutter_min_px: int = Field(3, ge=1, le=64, description="XY カットで採用するガターの最小幅（解析画像 px）")
    profile_ink_tolerance: float = Field(0.004, ge=0.0, lt=0.2, description="投影プロファイルで白行・白列とみなす墨率")
    frame_erosion_px: int = Field(3, ge=0, le=32, description="コマ内部の解析でコマ枠線を除く侵食量（詳細画像 px）")
    balloon_fill_min: float = Field(0.55, gt=0.0, lt=1.0)
    balloon_fill_max: float = Field(0.92, gt=0.0, le=1.0)
    balloon_aspect_min: float = Field(0.25, gt=0.0)
    balloon_aspect_max: float = Field(4.0, gt=0.0)
    char_cell_aspect: float = Field(1.1, gt=0.3, lt=3.0, description="縦書き1文字の字送り（列幅に対する高さの比）")
    split_bridges: bool = Field(True, description="細い連結部で繋がった複数コマを侵食で分離する")
    max_panels: int = Field(24, ge=1, le=24)

    @model_validator(mode="after")
    def _check_ranges(self) -> "VisionConfig":
        if self.balloon_fill_min >= self.balloon_fill_max:
            raise ValueError("balloon_fill_min は balloon_fill_max より小さい必要があります")
        if self.balloon_aspect_min >= self.balloon_aspect_max:
            raise ValueError("balloon_aspect_min は balloon_aspect_max より小さい必要があります")
        if self.min_balloon_area_ratio >= self.max_balloon_area_ratio:
            raise ValueError("min_balloon_area_ratio は max_balloon_area_ratio より小さい必要があります")
        if self.analysis_long_side > self.detail_long_side:
            raise ValueError("analysis_long_side は detail_long_side 以下である必要があります")
        return self


# ---------------------------------------------------------------------------
# 基本処理: グレースケール化・縮小・しきい値
# ---------------------------------------------------------------------------


def to_grayscale(image: Any) -> np.ndarray:
    """任意の画像配列を float32 グレースケール（白=1.0）へ変換する。

    - (H, W): 浮動小数は [0,1]（最大値が 1 を超える場合は [0,255] とみなす）、整数は型の最大値で正規化、bool は True=白。
    - (H, W, 3) / (H, W, 4): 輝度 0.299R+0.587G+0.114B。アルファは白背景で合成。
    """
    array = np.asarray(image)
    if array.dtype == bool:
        array = array.astype(np.float32)
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            array = array[..., 0]
        elif channels in (3, 4):
            values = _normalize_values(array)
            luminance = values[..., 0] * 0.299 + values[..., 1] * 0.587 + values[..., 2] * 0.114
            if channels == 4:
                alpha = values[..., 3]
                luminance = luminance * alpha + (1.0 - alpha)
            return np.clip(luminance, 0.0, 1.0).astype(np.float32)
        else:
            raise ValueError("カラー画像はチャネル数 3（RGB）または 4（RGBA）で渡してください")
    if array.ndim != 2:
        raise ValueError("画像は (H, W) のグレースケール、または (H, W, 3/4) の RGB(A) 配列で渡してください")
    return np.clip(_normalize_values(array), 0.0, 1.0).astype(np.float32)


def _normalize_values(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(info.max) if info.max <= 65535 else 255.0
        return array.astype(np.float64) / scale
    values = array.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("画像に NaN または無限大が含まれています")
    if values.size and float(values.max()) > 1.0 + 1e-6:
        values = values / 255.0
    return values


def _target_shape(height: int, width: int, long_side: int) -> tuple[int, int]:
    longest = max(height, width)
    if longest <= long_side:
        return height, width
    scale = longest / float(long_side)
    return max(1, int(round(height / scale))), max(1, int(round(width / scale)))


def _block_edges(size: int, blocks: int) -> np.ndarray:
    return np.round(np.linspace(0, size, blocks + 1)).astype(np.int64)


def _block_mean(values: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """積分画像による面積平均縮小（出力の各画素は入力の 1px 以上のブロック平均）。"""
    height, width = values.shape
    if out_height == height and out_width == width:
        return values.astype(np.float64)
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(values, axis=0, dtype=np.float64), axis=1)
    ys = _block_edges(height, out_height)
    xs = _block_edges(width, out_width)
    total = (
        integral[np.ix_(ys[1:], xs[1:])]
        - integral[np.ix_(ys[:-1], xs[1:])]
        - integral[np.ix_(ys[1:], xs[:-1])]
        + integral[np.ix_(ys[:-1], xs[:-1])]
    )
    area = np.outer(np.diff(ys), np.diff(xs)).astype(np.float64)
    return total / area


def area_downscale(gray: np.ndarray, long_side: int) -> np.ndarray:
    """長辺が long_side を超える場合のみ面積平均で縮小する。"""
    array = np.asarray(gray, dtype=np.float64)
    out_height, out_width = _target_shape(array.shape[0], array.shape[1], long_side)
    return _block_mean(array, out_height, out_width).astype(np.float32)


def otsu_threshold(gray: np.ndarray) -> float:
    """大津法（256 ビンのヒストグラムでクラス間分散を最大化）による二値化しきい値（[0,1]）。

    画素 < しきい値 を墨とみなす。値が一様な画像では 0.5 を返す。
    """
    values = np.asarray(gray, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("空の画像です")
    if float(values.max()) > 1.0 + 1e-6:
        values = values / 255.0
    values = np.clip(values, 0.0, 1.0)
    if float(values.max() - values.min()) < 1e-6:
        return 0.5
    histogram, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    probability = histogram.astype(np.float64) / values.size
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * centers)
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(denominator > 1e-12, (mu_total * omega - mu) ** 2 / denominator, -1.0)
    best = int(np.argmax(between))
    threshold = float(edges[best + 1])
    return float(min(max(threshold, 1.0 / 512.0), 1.0 - 1.0 / 512.0))


# ---------------------------------------------------------------------------
# 連結成分（行ランの Union-Find）
# ---------------------------------------------------------------------------


@dataclass
class _Runs:
    labels: np.ndarray
    count: int
    rows: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    run_labels: np.ndarray

    def stats(self) -> dict[str, np.ndarray]:
        """ラベルごとの面積と外接矩形（終端は排他的）。添字 0 は未使用。"""
        size = self.count + 1
        lengths = (self.ends - self.starts).astype(np.float64)
        area = np.bincount(self.run_labels, weights=lengths, minlength=size)
        x0 = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
        x1 = np.zeros(size, dtype=np.int64)
        y0 = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
        y1 = np.zeros(size, dtype=np.int64)
        np.minimum.at(x0, self.run_labels, self.starts)
        np.maximum.at(x1, self.run_labels, self.ends)
        np.minimum.at(y0, self.run_labels, self.rows)
        np.maximum.at(y1, self.run_labels, self.rows + 1)
        return {"area": area, "x0": x0, "x1": x1, "y0": y0, "y1": y1}


def _row_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    diff = np.diff(padded, axis=1)
    start_rows, start_cols = np.nonzero(diff == 1)
    _end_rows, end_cols = np.nonzero(diff == -1)
    return start_rows.astype(np.int64), start_cols.astype(np.int64), end_cols.astype(np.int64)


def _label_runs(mask: np.ndarray, connectivity: int = 4) -> _Runs:
    if connectivity not in (4, 8):
        raise ValueError("connectivity は 4 または 8 を指定してください")
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("マスクは 2 次元配列で渡してください")
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    rows, starts, ends = _row_runs(binary) if binary.size else (np.zeros(0, np.int64),) * 3
    total_runs = int(rows.size)
    if total_runs == 0:
        return _Runs(labels, 0, rows, starts, ends, np.zeros(0, dtype=np.int64))
    parent = list(range(total_runs))
    row_ptr = np.searchsorted(rows, np.arange(height + 1)).tolist()
    start_list = starts.tolist()
    end_list = ends.tolist()
    slack = 1 if connectivity == 8 else 0

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for y in range(1, height):
        i, i_end = row_ptr[y], row_ptr[y + 1]
        j, j_end = row_ptr[y - 1], row_ptr[y]
        while i < i_end and j < j_end:
            a_start, a_end = start_list[i], end_list[i]
            b_start, b_end = start_list[j], end_list[j]
            if a_start < b_end + slack and b_start < a_end + slack:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    if root_i < root_j:
                        parent[root_j] = root_i
                    else:
                        parent[root_i] = root_j
            if a_end < b_end:
                i += 1
            elif a_end > b_end:
                j += 1
            else:
                i += 1
                j += 1
    mapping: dict[int, int] = {}
    run_labels = np.empty(total_runs, dtype=np.int64)
    for index in range(total_runs):
        root = find(index)
        label = mapping.get(root)
        if label is None:
            label = len(mapping) + 1
            mapping[root] = label
        run_labels[index] = label
    lengths = ends - starts
    offsets = rows * width + starts - np.concatenate(([0], np.cumsum(lengths)[:-1]))
    flat_index = np.repeat(offsets, lengths) + np.arange(int(lengths.sum()))
    labels.ravel()[flat_index] = np.repeat(run_labels, lengths).astype(np.int32)
    return _Runs(labels, len(mapping), rows, starts, ends, run_labels)


def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """二値マスクの連結成分ラベリング（行ごとのランをノードとする Union-Find の 2 パス）。

    戻り値は (ラベル配列 int32（背景 0、成分は行優先の初出順に 1..n）, 成分数)。
    """
    runs = _label_runs(mask, connectivity)
    return runs.labels, runs.count


# ---------------------------------------------------------------------------
# 形態演算・幾何
# ---------------------------------------------------------------------------


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """4 近傍の侵食（配列外は偽とみなす）。"""
    current = np.asarray(mask, dtype=bool)
    for _ in range(max(0, iterations)):
        if not current.any():
            break
        inner = current.copy()
        inner[1:, :] &= current[:-1, :]
        inner[:-1, :] &= current[1:, :]
        inner[:, 1:] &= current[:, :-1]
        inner[:, :-1] &= current[:, 1:]
        inner[0, :] = False
        inner[-1, :] = False
        inner[:, 0] = False
        inner[:, -1] = False
        current = inner
    return current


def convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew の単調鎖法による凸包（共線点は除去）。"""
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 2:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """多角形の面積（靴紐公式の絶対値）。"""
    if len(points) < 3:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2.0)


def simplify_polygon(
    points: Sequence[tuple[float, float]], *, max_vertices: int = 12, tolerance_ratio: float = 0.004
) -> list[tuple[float, float]]:
    """凸多角形の頂点を、除去しても面積がほぼ変わらないもの（階段状の斜辺など）から順に間引く。"""
    vertices = [(float(x), float(y)) for x, y in points]
    if len(vertices) <= 4:
        return vertices
    total = polygon_area(vertices)
    tolerance = max(total * tolerance_ratio, 1e-9)

    def triangle(index: int) -> float:
        a = vertices[index - 1]
        b = vertices[index]
        c = vertices[(index + 1) % len(vertices)]
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0

    while len(vertices) > 4:
        areas = [triangle(i) for i in range(len(vertices))]
        smallest = int(np.argmin(areas))
        if areas[smallest] > tolerance and len(vertices) <= max_vertices:
            break
        vertices.pop(smallest)
    return vertices


def _max_slant(vertices: Sequence[tuple[float, float]]) -> float:
    """主要な辺（周長の 12% 以上）の、最寄りの軸からの最大傾き（度）。"""
    if len(vertices) < 3:
        return 0.0
    edges = []
    perimeter = 0.0
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        perimeter += length
        edges.append((start, end, length))
    slant = 0.0
    for start, end, length in edges:
        if perimeter <= 0 or length < 0.12 * perimeter:
            continue
        angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
        deviation = abs(((angle + 45.0) % 90.0) - 45.0)
        slant = max(slant, deviation)
    return slant


@dataclass
class _Region:
    """解析画像上のコマ領域（y0..y1, x0..x1 は排他的終端、mask は外接矩形内の充填マスク）。"""

    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray
    source: str = "component"

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def corner_points(self) -> list[tuple[float, float]]:
        rows_on = np.nonzero(self.mask.any(axis=1))[0]
        if rows_on.size == 0:
            return []
        sub = self.mask[rows_on]
        left = np.argmax(sub, axis=1)
        right = self.mask.shape[1] - np.argmax(sub[:, ::-1], axis=1)
        points: list[tuple[float, float]] = []
        for row, lo, hi in zip(rows_on.tolist(), left.tolist(), right.tolist()):
            y_top = float(self.y0 + row)
            points.extend(
                [
                    (float(self.x0 + lo), y_top),
                    (float(self.x0 + hi), y_top),
                    (float(self.x0 + lo), y_top + 1.0),
                    (float(self.x0 + hi), y_top + 1.0),
                ]
            )
        return points


def _region_from_mask(mask: np.ndarray, source: str) -> _Region | None:
    rows_on = np.nonzero(mask.any(axis=1))[0]
    cols_on = np.nonzero(mask.any(axis=0))[0]
    if rows_on.size == 0 or cols_on.size == 0:
        return None
    y0, y1 = int(rows_on[0]), int(rows_on[-1]) + 1
    x0, x1 = int(cols_on[0]), int(cols_on[-1]) + 1
    return _Region(y0, y1, x0, x1, mask[y0:y1, x0:x1].copy(), source)


def _white_segments(flags: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """真（白）の連続が min_gap 以上の箇所で区切った、非白の区間列。"""
    size = int(flags.size)
    gaps: list[tuple[int, int]] = []
    index = 0
    while index < size:
        if flags[index]:
            start = index
            while index < size and flags[index]:
                index += 1
            if index - start >= min_gap:
                gaps.append((start, index))
        else:
            index += 1
    segments: list[tuple[int, int]] = []
    cursor = 0
    for start, end in gaps:
        if start > cursor:
            segments.append((cursor, start))
        cursor = end
    if cursor < size:
        segments.append((cursor, size))
    return segments


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------


class PageVisionAnalyzer:
    """ページ画像からコマ・フキダシ・ベタ比率を逆解析する。"""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()

    # -- 公開 API -----------------------------------------------------------

    def detect(self, image: np.ndarray, *, page_index: int = 0) -> PageDetection:
        """1 ページ分の画像（グレースケール／RGB(A) 配列）からコマとフキダシを検出する。"""
        cfg = self.config
        gray = to_grayscale(image)
        height, width = gray.shape
        if height < 32 or width < 32:
            raise ValueError("画像が小さすぎます（縦横とも 32px 以上が必要です）")
        detail = area_downscale(gray, cfg.detail_long_side)
        detail_h, detail_w = detail.shape
        threshold = cfg.ink_threshold if cfg.ink_threshold is not None else otsu_threshold(detail)
        ink_detail = detail < threshold
        small_h, small_w = _target_shape(detail_h, detail_w, cfg.analysis_long_side)
        if (small_h, small_w) == (detail_h, detail_w):
            ink_small = ink_detail.copy()
        else:
            ink_small = _block_mean(ink_detail.astype(np.float64), small_h, small_w) >= cfg.pool_ink_fraction
        warnings: list[str] = []
        if float(detail.max() - detail.min()) < 0.15:
            warnings.append("画像のコントラストが低いため、検出精度が低下している可能性があります")
        regions, notes = self._find_panel_regions(ink_small)
        warnings.extend(notes)
        page_width = cfg.page_width
        page_height = page_width * height / width
        factors = _Factors(
            small=(small_h, small_w),
            detail=(detail_h, detail_w),
            page=(page_width, page_height),
        )
        panels = [self._panel_from_region(region, ink_detail, factors) for region in regions]
        if len(panels) > cfg.max_panels:
            panels = sorted(panels, key=lambda p: p.area_ratio, reverse=True)[: cfg.max_panels]
            warnings.append(f"コマ数が上限 {cfg.max_panels} を超えたため、面積の大きい {cfg.max_panels} コマに絞りました")
        if panels:
            tolerance = max(4.0, page_width * 0.006)
            order = xy_cut_order([p.bbox for p in panels], tol=tolerance)
            panels = [panels[i] for i in order]
        return PageDetection(
            page_index=page_index,
            width=page_width,
            height=page_height,
            image_width=width,
            image_height=height,
            threshold=float(threshold),
            panels=panels,
            warnings=warnings,
        )

    def detect_png(self, data: bytes | str, *, page_index: int = 0) -> PageDetection:
        """PNG バイト列（または data URL / base64 文字列）から検出する。"""
        payload = decode_data_url(data) if isinstance(data, str) else bytes(data)
        return self.detect(decode_png(payload), page_index=page_index)

    def to_annotation(
        self,
        detections: Sequence[PageDetection],
        *,
        title: str = "画像から逆解析した原稿",
        first_page_side: PageSide | str = PageSide.RIGHT,
    ) -> PageAnnotation:
        """検出結果をネーム構造ドクター用の PageAnnotation へ変換する（ページの左右は先頭から交互）。"""
        if not detections:
            raise ValueError("検出結果が空です")
        if len(detections) > 64:
            raise ValueError("1 回に解析できるのは 64 ページまでです")
        first = PageSide(first_page_side)
        other = PageSide.LEFT if first == PageSide.RIGHT else PageSide.RIGHT
        pages: list[AnnotatedPage] = []
        total_panels = 0
        total_balloons = 0
        fallback_pages: list[int] = []
        for position, detection in enumerate(sorted(detections, key=lambda d: d.page_index)):
            side = first if position % 2 == 0 else other
            panels: list[AnnotatedPanel] = []
            for panel in detection.panels[: self.config.max_panels]:
                balloons = [
                    AnnotatedBalloon(bbox=balloon.bbox, char_count=balloon.estimated_chars, kind=BalloonKind.NORMAL)
                    for balloon in panel.balloons
                    if balloon.bbox[2] > balloon.bbox[0] and balloon.bbox[3] > balloon.bbox[1]
                ]
                total_balloons += len(balloons)
                panels.append(
                    AnnotatedPanel(
                        bbox=panel.bbox,
                        polygon=list(panel.polygon) if len(panel.polygon) >= 3 else None,
                        ink_ratio=panel.ink_ratio,
                        balloons=balloons,
                        note=f"自動検出（信頼度 {panel.confidence:.2f}・傾き {panel.slant_deg:.1f}°）",
                    )
                )
            if not panels:
                margin_x = detection.width * 0.05
                margin_y = detection.height * 0.05
                panels.append(
                    AnnotatedPanel(
                        bbox=(margin_x, margin_y, detection.width - margin_x, detection.height - margin_y),
                        note="コマを検出できなかったため、ページ全体を 1 コマとして扱います",
                    )
                )
                fallback_pages.append(position + 1)
            total_panels += len(panels)
            pages.append(AnnotatedPage(width=detection.width, height=detection.height, side=side, panels=panels))
        notes = f"画像 {len(pages)} ページから {total_panels} コマ・{total_balloons} フキダシを自動検出"
        if fallback_pages:
            notes += "（コマ未検出のページ: " + ", ".join(f"p{n}" for n in fallback_pages) + "）"
        return PageAnnotation(title=title, first_page_side=first, pages=pages, notes=notes)

    def analyze_images(
        self,
        images: Sequence[bytes | str],
        *,
        first_page_side: PageSide | str = PageSide.RIGHT,
        title: str | None = None,
    ) -> ImageDetectionResult:
        """複数ページの PNG（または data URL）を解析し、検出結果と PageAnnotation をまとめて返す。"""
        if not images:
            raise ValueError("画像が 1 枚もありません")
        if len(images) > 64:
            raise ValueError("1 回に解析できるのは 64 ページまでです")
        detections = [self.detect_png(data, page_index=index) for index, data in enumerate(images)]
        annotation = self.to_annotation(
            detections, title=title or "画像から逆解析した原稿", first_page_side=first_page_side
        )
        return ImageDetectionResult(pages=detections, annotation=annotation)

    # -- コマ領域の検出 ------------------------------------------------------

    def _find_panel_regions(self, ink: np.ndarray) -> tuple[list[_Region], list[str]]:
        cfg = self.config
        height, width = ink.shape
        notes: list[str] = []
        white = ~ink
        white_runs = _label_runs(white, 4)
        band = max(2, int(round(cfg.border_band_ratio * max(height, width))))
        edge = np.zeros((height, width), dtype=bool)
        edge[:band, :] = True
        edge[-band:, :] = True
        edge[:, :band] = True
        edge[:, -band:] = True
        seeds = np.unique(white_runs.labels[edge & white])
        seeds = seeds[seeds > 0]
        background = np.isin(white_runs.labels, seeds) if seeds.size else np.zeros((height, width), dtype=bool)
        foreground = ~background
        runs = _label_runs(foreground, 8)
        stats = runs.stats()
        page_area = float(height * width)
        min_area = cfg.min_panel_area_ratio * page_area
        regions: list[_Region] = []
        for label in range(1, runs.count + 1):
            if stats["area"][label] < min_area:
                continue
            y0, y1 = int(stats["y0"][label]), int(stats["y1"][label])
            x0, x1 = int(stats["x0"][label]), int(stats["x1"][label])
            mask = runs.labels[y0:y1, x0:x1] == label
            regions.append(_Region(y0, y1, x0, x1, mask, "component"))
        if not regions or (len(regions) == 1 and regions[0].area > 0.9 * page_area):
            box = (0, height, 0, width) if not regions else (regions[0].y0, regions[0].y1, regions[0].x0, regions[0].x1)
            boxes = self._xy_cut(ink, box, 0)
            boxes = [b for b in boxes if (b[1] - b[0]) * (b[3] - b[2]) >= min_area]
            if len(boxes) > 1 or (not regions and boxes):
                notes.append("ガター（コマ間の余白）を検出できなかったため、投影プロファイルによる再帰 XY カットで分割しました")
                regions = [
                    _Region(y0, y1, x0, x1, np.ones((y1 - y0, x1 - x0), dtype=bool), "xycut")
                    for y0, y1, x0, x1 in boxes
                ]
            elif regions:
                notes.append("ページ全面が 1 つの領域として検出されました（断ち切りの 1 枚絵の可能性があります）")
        if cfg.split_bridges:
            refined: list[_Region] = []
            for region in regions:
                parts = [region]
                if region.source == "component" and region.area > 2.5 * min_area and self._convexity(region) < 0.82:
                    parts = self._split_bridged(region, min_area)
                    if len(parts) > 1:
                        notes.append(f"細い連結部で繋がった領域を {len(parts)} コマに分離しました")
                refined.extend(parts)
            regions = refined
        if not regions:
            notes.append("コマを検出できませんでした")
        return regions, notes

    def _xy_cut(self, ink: np.ndarray, box: tuple[int, int, int, int], depth: int) -> list[tuple[int, int, int, int]]:
        cfg = self.config
        y0, y1, x0, x1 = box
        if y1 - y0 < 2 or x1 - x0 < 2:
            return []
        sub = ink[y0:y1, x0:x1]
        rows_on = np.nonzero(sub.mean(axis=1) > cfg.profile_ink_tolerance)[0]
        cols_on = np.nonzero(sub.mean(axis=0) > cfg.profile_ink_tolerance)[0]
        if rows_on.size == 0 or cols_on.size == 0:
            return []
        y0, y1 = y0 + int(rows_on[0]), y0 + int(rows_on[-1]) + 1
        x0, x1 = x0 + int(cols_on[0]), x0 + int(cols_on[-1]) + 1
        if depth >= 16:
            return [(y0, y1, x0, x1)]
        sub = ink[y0:y1, x0:x1]
        for axis in (1, 0):
            profile = sub.mean(axis=axis) <= cfg.profile_ink_tolerance
            segments = _white_segments(profile, cfg.gutter_min_px)
            if len(segments) > 1:
                result: list[tuple[int, int, int, int]] = []
                for start, end in segments:
                    child = (y0 + start, y0 + end, x0, x1) if axis == 1 else (y0, y1, x0 + start, x0 + end)
                    result.extend(self._xy_cut(ink, child, depth + 1))
                return result
        return [(y0, y1, x0, x1)]

    def _convexity(self, region: _Region) -> float:
        hull_area = polygon_area(convex_hull(region.corner_points()))
        if hull_area <= 0:
            return 1.0
        return min(1.0, region.area / hull_area)

    def _split_bridged(self, region: _Region, min_area: float) -> list[_Region]:
        mask = region.mask
        for radius in (2, 3, 4):
            eroded = _erode(mask, radius)
            runs = _label_runs(eroded, 8)
            if runs.count < 2:
                continue
            stats = runs.stats()
            floor_area = max(0.6 * min_area, 0.12 * float(mask.sum()))
            big = [label for label in range(1, runs.count + 1) if stats["area"][label] >= floor_area]
            if len(big) < 2:
                continue
            grown = np.zeros(mask.shape, dtype=np.int32)
            for index, label in enumerate(big, start=1):
                grown[runs.labels == label] = index
            for _ in range(radius + 2):
                changed = False
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    shifted = np.zeros_like(grown)
                    src_y = slice(max(0, -dy), grown.shape[0] - max(0, dy))
                    dst_y = slice(max(0, dy), grown.shape[0] - max(0, -dy))
                    src_x = slice(max(0, -dx), grown.shape[1] - max(0, dx))
                    dst_x = slice(max(0, dx), grown.shape[1] - max(0, -dx))
                    shifted[dst_y, dst_x] = grown[src_y, src_x]
                    take = (grown == 0) & mask & (shifted > 0)
                    if take.any():
                        grown[take] = shifted[take]
                        changed = True
                if not changed:
                    break
            parts: list[_Region] = []
            for index in range(1, len(big) + 1):
                part = _region_from_mask(grown == index, "split")
                if part is not None:
                    part.y0 += region.y0
                    part.y1 += region.y0
                    part.x0 += region.x0
                    part.x1 += region.x0
                    parts.append(part)
            if len(parts) >= 2:
                return parts
        return [region]

    # -- コマ内部の解析 ------------------------------------------------------

    def _panel_from_region(self, region: _Region, ink_detail: np.ndarray, factors: "_Factors") -> DetectedPanel:
        cfg = self.config
        small_h, small_w = factors.small
        detail_h, detail_w = factors.detail
        page_w, page_h = factors.page
        scale_y = detail_h / small_h
        scale_x = detail_w / small_w
        dy0 = int(math.floor(region.y0 * scale_y))
        dy1 = min(detail_h, int(math.ceil(region.y1 * scale_y)))
        dx0 = int(math.floor(region.x0 * scale_x))
        dx1 = min(detail_w, int(math.ceil(region.x1 * scale_x)))
        rows = np.clip(((np.arange(dy0, dy1) + 0.5) / scale_y).astype(np.int64), region.y0, region.y1 - 1) - region.y0
        cols = np.clip(((np.arange(dx0, dx1) + 0.5) / scale_x).astype(np.int64), region.x0, region.x1 - 1) - region.x0
        mask_detail = region.mask[np.ix_(rows, cols)]
        erosion = cfg.frame_erosion_px + int(math.ceil(max(scale_y, scale_x)))
        interior = _erode(mask_detail, erosion)
        panel_pixels = float(mask_detail.sum())
        if interior.sum() < 0.25 * panel_pixels:
            interior = _erode(mask_detail, max(1, erosion // 2))
        if not interior.any():
            interior = mask_detail
        ink_crop = ink_detail[dy0:dy1, dx0:dx1]
        ink_ratio = float(ink_crop[interior].mean()) if interior.any() else 0.0
        detail_to_page = (page_w / detail_w, page_h / detail_h)
        balloons = self._detect_balloons(ink_crop, interior, panel_pixels, (dy0, dx0), detail_to_page)

        to_page_x = page_w / small_w
        to_page_y = page_h / small_h
        hull = convex_hull(region.corner_points())
        simplified = simplify_polygon(hull)
        polygon_page = [(x * to_page_x, y * to_page_y) for x, y in simplified]
        hull_area = polygon_area(hull)
        convexity = min(1.0, region.area / hull_area) if hull_area > 0 else 1.0
        fill = region.area / float((region.y1 - region.y0) * (region.x1 - region.x0))
        if region.source == "xycut":
            confidence = 0.55
        else:
            confidence = convexity**2 * (0.7 + 0.3 * min(1.0, fill / 0.9))
        bbox = (region.x0 * to_page_x, region.y0 * to_page_y, region.x1 * to_page_x, region.y1 * to_page_y)
        return DetectedPanel(
            bbox=tuple(round(v, 3) for v in bbox),  # type: ignore[arg-type]
            polygon=[Point(x=round(x, 3), y=round(y, 3)) for x, y in polygon_page],
            area_ratio=min(1.0, region.area / float(small_h * small_w)),
            ink_ratio=min(1.0, max(0.0, ink_ratio)),
            slant_deg=round(_max_slant(polygon_page), 2),
            confidence=round(min(1.0, max(0.0, confidence)), 3),
            balloons=balloons,
        )

    def _detect_balloons(
        self,
        ink_crop: np.ndarray,
        interior: np.ndarray,
        panel_pixels: float,
        offset: tuple[int, int],
        detail_to_page: tuple[float, float],
    ) -> list[DetectedBalloon]:
        cfg = self.config
        if panel_pixels <= 0:
            return []
        white = (~ink_crop) & interior
        runs = _label_runs(white, 4)
        if runs.count == 0:
            return []
        boundary = interior & ~_erode(interior, 1)
        touching = set(np.unique(runs.labels[boundary & white]).tolist())
        stats = runs.stats()
        min_pixels = cfg.min_balloon_area_ratio * panel_pixels
        max_pixels = cfg.max_balloon_area_ratio * panel_pixels
        candidates = [
            label
            for label in range(1, runs.count + 1)
            if label not in touching and stats["area"][label] >= 0.5 * min_pixels
        ]
        candidates.sort(key=lambda label: -stats["area"][label])
        offset_y, offset_x = offset
        scale_x, scale_y = detail_to_page
        balloons: list[DetectedBalloon] = []
        for label in candidates[:40]:
            y0, y1 = int(stats["y0"][label]), int(stats["y1"][label])
            x0, x1 = int(stats["x0"][label]), int(stats["x1"][label])
            component = runs.labels[y0:y1, x0:x1] == label
            filled = _fill_holes(component)
            filled_pixels = float(filled.sum())
            if not min_pixels <= filled_pixels <= max_pixels:
                continue
            box_h, box_w = y1 - y0, x1 - x0
            fill_ratio = filled_pixels / float(box_h * box_w)
            aspect = box_h / float(box_w)
            if not cfg.balloon_fill_min <= fill_ratio <= cfg.balloon_fill_max:
                continue
            if not cfg.balloon_aspect_min <= aspect <= cfg.balloon_aspect_max:
                continue
            glyphs = filled & ~component
            chars = self._estimate_chars(glyphs)
            shape_score = 1.0 - min(1.0, abs(fill_ratio - 0.785) / 0.25)
            confidence = 0.35 + 0.45 * shape_score + (0.2 if chars > 0 else 0.0)
            bbox = (
                (offset_x + x0) * scale_x,
                (offset_y + y0) * scale_y,
                (offset_x + x1) * scale_x,
                (offset_y + y1) * scale_y,
            )
            balloons.append(
                DetectedBalloon(
                    bbox=tuple(round(v, 3) for v in bbox),  # type: ignore[arg-type]
                    area_ratio=min(1.0, filled_pixels / panel_pixels),
                    fill_ratio=min(1.0, fill_ratio),
                    estimated_chars=chars,
                    confidence=round(min(1.0, max(0.0, confidence)), 3),
                )
            )
        balloons.sort(key=lambda b: (-b.bbox[2], b.bbox[1]))
        return balloons

    def _estimate_chars(self, glyphs: np.ndarray) -> int:
        """縦書きの字形マスクを列に分解し、列ごとに「高さ ÷（列幅×字送り比）」で文字数を推定する。"""
        if not glyphs.any():
            return 0
        columns = glyphs.any(axis=0)
        raw_runs: list[tuple[int, int]] = []
        index = 0
        size = int(columns.size)
        while index < size:
            if columns[index]:
                start = index
                while index < size and columns[index]:
                    index += 1
                raw_runs.append((start, index))
            else:
                index += 1
        if not raw_runs:
            return 0
        widest = max(end - start for start, end in raw_runs)
        merge_gap = max(1, int(round(0.25 * widest)))
        merged: list[tuple[int, int]] = []
        for start, end in raw_runs:
            if merged and start - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        total = 0
        for start, end in merged:
            column = glyphs[:, start:end]
            rows_on = np.nonzero(column.any(axis=1))[0]
            if rows_on.size == 0:
                continue
            height = int(rows_on[-1] - rows_on[0] + 1)
            width = max(1, end - start)
            total += max(1, int(round(height / (width * self.config.char_cell_aspect))))
        return total


@dataclass(frozen=True)
class _Factors:
    small: tuple[int, int]
    detail: tuple[int, int]
    page: tuple[float, float]


def _fill_holes(component: np.ndarray) -> np.ndarray:
    """成分マスクの穴（外接矩形の縁に連結しない補集合、8 近傍）を埋める。"""
    complement = ~component
    if not complement.any():
        return component.copy()
    runs = _label_runs(complement, 8)
    border_labels = set(np.unique(np.concatenate([
        runs.labels[0, :],
        runs.labels[-1, :],
        runs.labels[:, 0],
        runs.labels[:, -1],
    ])).tolist())
    holes = (runs.labels > 0) & ~np.isin(runs.labels, list(border_labels))
    return component | holes


# ---------------------------------------------------------------------------
# 合成原稿画像（テスト・UI デモ用）
# ---------------------------------------------------------------------------


def _convex_mask(points: Sequence[tuple[float, float]], height: int, width: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """凸多角形の画素マスク（外接矩形内）と外接矩形 (y0, y1, x0, x1) を返す。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, int(math.floor(min(xs))))
    x1 = min(width, int(math.ceil(max(xs))))
    y0 = max(0, int(math.floor(min(ys))))
    y1 = min(height, int(math.ceil(max(ys))))
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    cx = grid_x + 0.5
    cy = grid_y + 0.5
    orientation = 0.0
    count = len(points)
    for index in range(count):
        ax, ay = points[index]
        bx, by = points[(index + 1) % count]
        orientation += ax * by - bx * ay
    sign = 1.0 if orientation >= 0 else -1.0
    inside = np.ones(cx.shape, dtype=bool)
    for index in range(count):
        ax, ay = points[index]
        bx, by = points[(index + 1) % count]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        inside &= sign * cross >= 0.0
    return inside, (y0, y1, x0, x1)


def _intersect(horizontal: tuple[float, float, float, float], vertical: tuple[float, float, float, float]) -> tuple[float, float]:
    """水平寄りの直線（(xa, ya)-(xb, yb)）と垂直寄りの直線の交点。"""
    hx0, hy0, hx1, hy1 = horizontal
    vx0, vy0, vx1, vy1 = vertical
    slope_h = (hy1 - hy0) / (hx1 - hx0) if hx1 != hx0 else 0.0
    slope_v = (vx1 - vx0) / (vy1 - vy0) if vy1 != vy0 else 0.0
    denominator = 1.0 - slope_v * slope_h
    x = (vx0 + slope_v * (hy0 - vy0 - slope_h * hx0)) / denominator
    y = hy0 + slope_h * (x - hx0)
    return (x, y)


def synthesize_page_image(
    *, seed: int = 0, variant: str = "standard", width: int = 700, height: int = 990
) -> tuple[np.ndarray, PageAnnotation]:
    """正解アノテーション付きの合成漫画原稿（グレースケール、白=1.0）を生成する。

    variant: "standard"（矩形の段組み）/ "slanted"（斜めガター）/ "dense"（多コマ・多フキダシ）/ "splash"（1 枚絵）。
    正解の座標はページ座標（幅 1000、高さは縦横比）で、コマは読み順（上段から、段内は右から）に並ぶ。
    正解の ink_ratio はコマ枠線を除いた内部の暗画素（< 0.5）率、フキダシの char_count は描いた字形の数。
    """
    if variant not in SYNTH_VARIANTS:
        raise ValueError(f"variant は {', '.join(SYNTH_VARIANTS)} のいずれかを指定してください")
    if width < 240 or height < 240:
        raise ValueError("合成画像の縦横は 240px 以上が必要です")
    rng = np.random.default_rng(seed)
    image = np.ones((height, width), dtype=np.float32)
    margin_x = round(width * 0.075)
    frame = (float(margin_x), float(round(height * 0.065)), float(width - margin_x), float(height - round(height * 0.075)))
    gap_y = max(10.0, round(height * 0.024))
    gap_x = max(9.0, round(width * 0.018))
    line = max(3, int(round(width * 0.0045)))
    glyph = max(4, int(round(width * 0.0085)))
    glyph_gap_v = max(1, int(round(glyph * 0.17)))
    glyph_gap_h = max(3, int(round(glyph * 0.7)))

    if variant == "splash":
        tier_counts = [1]
    elif variant == "dense":
        tier_counts = [int(rng.integers(2, 4)) for _ in range(int(rng.integers(4, 6)))]
    elif variant == "slanted":
        tier_counts = [2, int(rng.integers(1, 3)), 2]
    else:
        tier_counts = [int(rng.choice([1, 2, 3], p=[0.25, 0.5, 0.25])) for _ in range(int(rng.integers(3, 5)))]
    slanted = variant == "slanted"
    fx0, fy0, fx1, fy1 = frame
    weights = rng.uniform(0.75, 1.35, len(tier_counts))
    usable = (fy1 - fy0) - gap_y * (len(tier_counts) - 1)
    cuts = fy0 + np.concatenate([[0.0], np.cumsum(weights / weights.sum() * usable)])
    boundaries: list[tuple[float, float]] = []
    for k in range(1, len(tier_counts)):
        center = cuts[k] + gap_y * (k - 0.5)
        offset = float(rng.uniform(0.02, 0.045) * height * rng.choice([-1.0, 1.0])) if slanted else 0.0
        boundaries.append((center + offset, center - offset))

    def tier_edge(index: int, top: bool) -> tuple[float, float, float, float]:
        if top and index == 0:
            return (fx0, fy0, fx1, fy0)
        if not top and index == len(tier_counts) - 1:
            return (fx0, fy1, fx1, fy1)
        left_y, right_y = boundaries[index - 1] if top else boundaries[index]
        shift = gap_y / 2.0 if top else -gap_y / 2.0
        return (fx0, left_y + shift, fx1, right_y + shift)

    panel_points: list[list[tuple[float, float]]] = []
    for tier_index, count in enumerate(tier_counts):
        top = tier_edge(tier_index, True)
        bottom = tier_edge(tier_index, False)
        tier_top = min(top[1], top[3])
        tier_bottom = max(bottom[1], bottom[3])
        column_weights = rng.uniform(0.7, 1.3, count)
        usable_w = (fx1 - fx0) - gap_x * (count - 1)
        xs = fx0 + np.concatenate([[0.0], np.cumsum(column_weights / column_weights.sum() * usable_w)])
        verticals: list[tuple[float, float, float, float]] = []
        for k in range(1, count):
            center = xs[k] + gap_x * (k - 0.5)
            tilt = float(rng.uniform(0.03, 0.06) * (tier_bottom - tier_top) * rng.choice([-1.0, 1.0])) if slanted else 0.0
            verticals.append((center + tilt, tier_top, center - tilt, tier_bottom))
        row_points: list[list[tuple[float, float]]] = []
        for column in range(count):
            if column == 0:
                left = (fx0, tier_top, fx0, tier_bottom)
            else:
                vx0, vy0, vx1, vy1 = verticals[column - 1]
                left = (vx0 + gap_x / 2.0, vy0, vx1 + gap_x / 2.0, vy1)
            if column == count - 1:
                right = (fx1, tier_top, fx1, tier_bottom)
            else:
                vx0, vy0, vx1, vy1 = verticals[column]
                right = (vx0 - gap_x / 2.0, vy0, vx1 - gap_x / 2.0, vy1)
            corners = [_intersect(top, left), _intersect(top, right), _intersect(bottom, right), _intersect(bottom, left)]
            row_points.append(corners)
        panel_points.extend(reversed(row_points))

    scale = 1000.0 / width
    page_height = height * scale
    annotated_panels: list[AnnotatedPanel] = []
    balloon_ranges = {"standard": (0, 3), "slanted": (0, 3), "dense": (1, 3), "splash": (2, 4)}
    for corners in panel_points:
        mask, (y0, y1, x0, x1) = _convex_mask(corners, height, width)
        region = image[y0:y1, x0:x1]
        region[mask] = 1.0
        interior = _erode(mask, line)
        region[mask & ~interior] = 0.0
        content = _erode(mask, line + 4)
        box_h, box_w = y1 - y0, x1 - x0
        beta_box: tuple[int, int, int, int] | None = None
        if rng.random() < 0.45:
            bh = int(box_h * rng.uniform(0.25, 0.42))
            bw = int(box_w * rng.uniform(0.25, 0.45))
            corner = int(rng.integers(0, 4))
            by0 = 0 if corner in (0, 1) else box_h - bh
            bx0 = 0 if corner in (0, 2) else box_w - bw
            beta_box = (by0, by0 + bh, bx0, bx0 + bw)
            patch = np.zeros(mask.shape, dtype=bool)
            patch[by0 : by0 + bh, bx0 : bx0 + bw] = True
            region[patch & interior] = 0.0
        if rng.random() < 0.5:
            dark = bool(rng.random() < 0.5)
            spacing = 4
            dot = 3 if dark else 2
            th = int(box_h * rng.uniform(0.2, 0.4))
            tw = int(box_w * rng.uniform(0.3, 0.6))
            ty0 = int(rng.integers(0, max(1, box_h - th)))
            tx0 = int(rng.integers(0, max(1, box_w - tw)))
            grid_y, grid_x = np.mgrid[0:box_h, 0:box_w]
            dots = ((grid_y % spacing) < dot) & ((grid_x % spacing) < dot)
            area = np.zeros(mask.shape, dtype=bool)
            area[ty0 : ty0 + th, tx0 : tx0 + tw] = True
            region[dots & area & content] = 0.0
        if rng.random() < 0.55:
            radius = max(6, int(min(box_w, box_h) * rng.uniform(0.07, 0.12)))
            cx = int(rng.uniform(0.25, 0.75) * box_w)
            body_top = int(box_h * rng.uniform(0.45, 0.6))
            head_cy = body_top - radius
            silhouette_box = (head_cy - radius, box_h, cx - int(radius * 1.6), cx + int(radius * 1.6))
            separated = True
            if beta_box is not None:
                sy0, sy1, sx0, sx1 = silhouette_box
                by0, by1, bx0, bx1 = beta_box
                separated = sx1 + 8 < bx0 or bx1 + 8 < sx0 or sy1 + 8 < by0 or by1 + 8 < sy0
            if separated:
                grid_y, grid_x = np.mgrid[0:box_h, 0:box_w]
                head = (grid_x - cx) ** 2 + (grid_y - head_cy) ** 2 <= radius**2
                body = (np.abs(grid_x - cx) <= int(radius * 1.5)) & (grid_y >= body_top)
                region[(head | body) & interior] = 0.0
        low, high = balloon_ranges[variant]
        wanted = int(rng.integers(low, high))
        placement = _erode(mask, line + 12)
        placed: list[tuple[int, int, int, int]] = []
        balloons: list[AnnotatedBalloon] = []
        for _ in range(wanted):
            for _attempt in range(40):
                semi_x = max(22.0, box_w * rng.uniform(0.09, 0.15))
                semi_y = min(semi_x * rng.uniform(1.3, 1.9), box_h * 0.4)
                semi_x = min(semi_x, box_w * 0.3)
                if semi_y < 20.0:
                    continue
                cx = rng.uniform(semi_x + 4, box_w - semi_x - 4)
                cy = rng.uniform(semi_y + 4, box_h - semi_y - 4)
                bb = (int(cy - semi_y - 4), int(cy + semi_y + 5), int(cx - semi_x - 4), int(cx + semi_x + 5))
                if bb[0] < 0 or bb[2] < 0 or bb[1] > box_h or bb[3] > box_w:
                    continue
                if not placement[bb[0] : bb[1], bb[2] : bb[3]].all():
                    continue
                if any(not (bb[1] + 8 < p[0] or p[1] + 8 < bb[0] or bb[3] + 8 < p[2] or p[3] + 8 < bb[2]) for p in placed):
                    continue
                placed.append(bb)
                grid_y, grid_x = np.mgrid[bb[0] : bb[1], bb[2] : bb[3]]
                inner = ((grid_x + 0.5 - cx) / semi_x) ** 2 + ((grid_y + 0.5 - cy) / semi_y) ** 2 <= 1.0
                outer = ((grid_x + 0.5 - cx) / (semi_x + 2.0)) ** 2 + ((grid_y + 0.5 - cy) / (semi_y + 2.0)) ** 2 <= 1.0
                window = region[bb[0] : bb[1], bb[2] : bb[3]]
                window[outer] = 0.0
                window[inner] = 1.0
                half_w = semi_x * 0.62
                half_h = semi_y * 0.62
                columns = max(1, min(4, int((2 * half_w + glyph_gap_h) // (glyph + glyph_gap_h))))
                per_column = max(1, min(9, int((2 * half_h + glyph_gap_v) // (glyph + glyph_gap_v))))
                block_w = columns * glyph + (columns - 1) * glyph_gap_h
                start_x = int(round(cx - block_w / 2.0))
                start_y = int(round(cy - half_h))
                char_total = 0
                for column in range(columns):
                    count = per_column if column == 0 else int(rng.integers(max(1, per_column - 3), per_column + 1))
                    gx = start_x + column * (glyph + glyph_gap_h)
                    for k in range(count):
                        gy = start_y + k * (glyph + glyph_gap_v)
                        region[gy : gy + glyph, gx : gx + glyph] = 0.0
                    char_total += count
                balloons.append(
                    AnnotatedBalloon(
                        bbox=(
                            (x0 + cx - semi_x - 2.0) * scale,
                            (y0 + cy - semi_y - 2.0) * scale,
                            (x0 + cx + semi_x + 2.0) * scale,
                            (y0 + cy + semi_y + 2.0) * scale,
                        ),
                        char_count=char_total,
                        kind=BalloonKind.NORMAL,
                    )
                )
                break
        interior_gt = _erode(mask, line + 3)
        ink_ratio = float((region[interior_gt] < 0.5).mean()) if interior_gt.any() else 0.0
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        annotated_panels.append(
            AnnotatedPanel(
                bbox=(min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale),
                polygon=[Point(x=x * scale, y=y * scale) for x, y in corners],
                ink_ratio=min(1.0, max(0.0, ink_ratio)),
                balloons=sorted(balloons, key=lambda b: (-b.bbox[2], b.bbox[1])),
                note="合成原稿の正解コマ",
            )
        )
    page = AnnotatedPage(
        width=1000.0,
        height=page_height,
        side=PageSide.RIGHT,
        panels=annotated_panels,
        reading_order=list(range(len(annotated_panels))),
    )
    annotation = PageAnnotation(
        title=f"合成原稿（{variant}・seed={seed}）",
        pages=[page],
        notes="page_vision.synthesize_page_image による正解付き合成原稿",
    )
    return image, annotation
