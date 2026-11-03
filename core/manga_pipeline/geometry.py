"""漫画原稿の幾何演算と視線誘導（Z字読み順）計算。

- 多角形演算: 面積（Shoelace）・重心・外接矩形・点包含（Ray casting）・辺列挙・凸性判定・
  Sutherland–Hodgman クリップ・線分交差。
- 読み順: 日本式（右上→左下）を **再帰 XY カット** で求める。まず全幅を貫く水平ガター（上→下）、
  無ければ全高を貫く垂直ガター（右→左）で分割し、各部分を再帰的に処理する。斜めゴマは
  「中央高さでの水平断面」「中央幅での垂直断面」から得るコア矩形で判定するため、台形の外接矩形が
  隣接コマと重なっても正しく分割できる。
- 視線誘導: 前コマの出口（左下寄り）から次コマの入口（右上寄り）へのベクトルを求め、
  同段での右向き移動・段跨ぎでの上向き移動を逆行、遷移線分同士の真の交差を交差として検出する。

本モジュールはスキーマに依存しない純粋関数群（コマは ``.id`` と ``.polygon`` を持つ任意オブジェクト）で、
``compute_gaze_transitions`` のみ戻り値生成のためにスキーマを遅延 import する。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.manga_pipeline.schemas import GazeTransition, Page

__all__ = [
    "Pt",
    "EPS",
    "polygon_signed_area",
    "polygon_area",
    "polygon_centroid",
    "polygon_bbox",
    "polygon_edges",
    "point_in_polygon",
    "point_segment_distance",
    "is_convex",
    "clip_polygon",
    "segments_intersect",
    "segment_intersection",
    "rect_points",
    "horizontal_extent_at",
    "vertical_extent_at",
    "core_box",
    "xy_cut_order",
    "compute_reading_order",
    "panel_exit_point",
    "panel_entry_point",
    "same_tier",
    "compute_gaze_transitions",
    "normalize_vector",
    "plan_tiers",
    "tier_fractions",
    "layout_tiers",
]

Pt = tuple[float, float]
EPS = 1e-9


def _as_points(points: Iterable[Any]) -> list[Pt]:
    """Point モデル・タプル・リストを (x, y) タプル列へ正規化する。"""
    result: list[Pt] = []
    for p in points:
        if hasattr(p, "x") and hasattr(p, "y"):
            result.append((float(p.x), float(p.y)))
        else:
            result.append((float(p[0]), float(p[1])))
    return result


def polygon_signed_area(points: Sequence[Any]) -> float:
    """符号付き面積（画面座標系 y下向きでは時計回りが正）。"""
    pts = _as_points(points)
    n = len(pts)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def polygon_area(points: Sequence[Any]) -> float:
    return abs(polygon_signed_area(points))


def polygon_centroid(points: Sequence[Any]) -> Pt:
    """多角形の重心（面積が退化している場合は頂点平均）。"""
    pts = _as_points(points)
    if not pts:
        return (0.0, 0.0)
    area = polygon_signed_area(pts)
    if abs(area) < EPS:
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def polygon_bbox(points: Sequence[Any]) -> tuple[float, float, float, float]:
    pts = _as_points(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def polygon_edges(points: Sequence[Any]) -> list[tuple[Pt, Pt]]:
    pts = _as_points(points)
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def point_segment_distance(p: Pt, a: Pt, b: Pt) -> float:
    """点と線分の距離。"""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < EPS:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_in_polygon(point: Any, points: Sequence[Any], *, boundary_tolerance: float = 1e-6) -> bool:
    """点包含判定（境界上は内側とみなす）。"""
    px, py = _as_points([point])[0]
    pts = _as_points(points)
    for a, b in polygon_edges(pts):
        if point_segment_distance((px, py), a, b) <= boundary_tolerance:
            return True
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > py) != (yj > py):
            x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_cross:
                inside = not inside
        j = i
    return inside


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex(points: Sequence[Any]) -> bool:
    """凸多角形か（共線頂点は無視）。"""
    pts = _as_points(points)
    n = len(pts)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        cross = _cross(pts[i], pts[(i + 1) % n], pts[(i + 2) % n])
        if abs(cross) < 1e-9:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return sign != 0


def clip_polygon(subject: Sequence[Any], clip: Sequence[Any]) -> list[Pt]:
    """Sutherland–Hodgman 法で subject を凸多角形 clip で切り抜く（向き非依存）。"""
    output = _as_points(subject)
    clip_pts = _as_points(clip)
    if len(clip_pts) < 3 or len(output) < 3:
        return []
    orientation = 1.0 if polygon_signed_area(clip_pts) >= 0 else -1.0

    def inside(p: Pt, a: Pt, b: Pt) -> bool:
        return orientation * _cross(a, b, p) >= -1e-9

    def intersection(p1: Pt, p2: Pt, a: Pt, b: Pt) -> Pt:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < EPS:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    for i in range(len(clip_pts)):
        a = clip_pts[i]
        b = clip_pts[(i + 1) % len(clip_pts)]
        input_list = output
        output = []
        if not input_list:
            break
        previous = input_list[-1]
        for current in input_list:
            if inside(current, a, b):
                if not inside(previous, a, b):
                    output.append(intersection(previous, current, a, b))
                output.append(current)
            elif inside(previous, a, b):
                output.append(intersection(previous, current, a, b))
            previous = current
    return output


def _orientation(a: Pt, b: Pt, c: Pt) -> int:
    value = _cross(a, b, c)
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a: Pt, b: Pt, p: Pt) -> bool:
    return (
        min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
    )


def segments_intersect(p1: Any, p2: Any, p3: Any, p4: Any, *, proper: bool = True) -> bool:
    """線分 p1p2 と p3p4 の交差判定。proper=True では端点接触・共線重なりを交差とみなさない。"""
    a, b, c, d = _as_points([p1, p2, p3, p4])
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if proper:
        return False
    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    if o4 == 0 and _on_segment(c, d, b):
        return True
    return False


def segment_intersection(p1: Any, p2: Any, p3: Any, p4: Any) -> Pt | None:
    """2線分の交点（交差しなければ None）。"""
    if not segments_intersect(p1, p2, p3, p4, proper=False):
        return None
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _as_points([p1, p2, p3, p4])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < EPS:
        return (x3, y3)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def rect_points(x0: float, y0: float, x1: float, y1: float) -> list[Pt]:
    """矩形頂点（左上→右上→右下→左下、画面座標で時計回り）。"""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def horizontal_extent_at(points: Sequence[Any], y: float) -> tuple[float, float] | None:
    """高さ y の水平線と多角形の交差区間 [xmin, xmax]。"""
    xs: list[float] = []
    for (x1, y1), (x2, y2) in polygon_edges(points):
        if abs(y2 - y1) < EPS:
            if abs(y - y1) < 1e-6:
                xs.extend([x1, x2])
            continue
        if min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    if not xs:
        return None
    return (min(xs), max(xs))


def vertical_extent_at(points: Sequence[Any], x: float) -> tuple[float, float] | None:
    """位置 x の垂直線と多角形の交差区間 [ymin, ymax]。"""
    ys: list[float] = []
    for (x1, y1), (x2, y2) in polygon_edges(points):
        if abs(x2 - x1) < EPS:
            if abs(x - x1) < 1e-6:
                ys.extend([y1, y2])
            continue
        if min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9:
            t = (x - x1) / (x2 - x1)
            ys.append(y1 + t * (y2 - y1))
    if not ys:
        return None
    return (min(ys), max(ys))


def core_box(points: Sequence[Any]) -> tuple[float, float, float, float]:
    """斜めゴマでも隣接コマと重ならない「コア矩形」（中央断面の範囲）。"""
    x0, y0, x1, y1 = polygon_bbox(points)
    mid_y = (y0 + y1) / 2.0
    mid_x = (x0 + x1) / 2.0
    horizontal = horizontal_extent_at(points, mid_y) or (x0, x1)
    vertical = vertical_extent_at(points, mid_x) or (y0, y1)
    return (horizontal[0], vertical[0], horizontal[1], vertical[1])


def _partition(indices: list[int], intervals: dict[int, tuple[float, float]], tol: float) -> list[list[int]]:
    """区間の重なりでクラスタ分割（ガターで分離されたグループ列を昇順で返す）。"""
    shrunk: list[tuple[float, float, int]] = []
    for index in indices:
        a, b = intervals[index]
        lo, hi = a + tol, b - tol
        if lo > hi:
            mid = (a + b) / 2.0
            lo = hi = mid
        shrunk.append((lo, hi, index))
    shrunk.sort()
    groups: list[list[int]] = []
    current: list[int] = []
    current_end = -math.inf
    for lo, hi, index in shrunk:
        if current and lo >= current_end:
            groups.append(current)
            current = []
            current_end = -math.inf
        current.append(index)
        current_end = max(current_end, hi)
    if current:
        groups.append(current)
    return groups


def xy_cut_order(boxes: Sequence[tuple[float, float, float, float]], tol: float = 6.0) -> list[int]:
    """日本式読み順（右上→左下）を再帰 XY カットで求め、インデックス列を返す。"""
    vertical_intervals = {i: (b[1], b[3]) for i, b in enumerate(boxes)}
    horizontal_intervals = {i: (b[0], b[2]) for i, b in enumerate(boxes)}

    def recurse(indices: list[int]) -> list[int]:
        if len(indices) <= 1:
            return list(indices)
        bands = _partition(indices, vertical_intervals, tol)
        if len(bands) > 1:
            ordered: list[int] = []
            for band in bands:
                ordered.extend(recurse(band))
            return ordered
        columns = _partition(indices, horizontal_intervals, tol)
        if len(columns) > 1:
            ordered = []
            for column in reversed(columns):
                ordered.extend(recurse(column))
            return ordered
        return sorted(indices, key=lambda i: (round(boxes[i][1] / 40.0), -boxes[i][2], boxes[i][1]))

    return recurse(list(range(len(boxes))))


def _polygon_points_of(panel: Any) -> list[Pt]:
    polygon = panel.polygon
    if hasattr(polygon, "points"):
        return _as_points(polygon.points)
    return _as_points(polygon)


def compute_reading_order(panels: Sequence[Any], tol: float = 6.0) -> list[str]:
    """コマ列の自然な読み順（パネルID列）。"""
    boxes = [core_box(_polygon_points_of(panel)) for panel in panels]
    order = xy_cut_order(boxes, tol=tol)
    return [panels[i].id for i in order]


def panel_exit_point(points: Sequence[Any]) -> Pt:
    """コマの視線出口（左下寄り）。"""
    x0, y0, x1, y1 = core_box(points)
    return (x0 + 0.2 * (x1 - x0), y0 + 0.8 * (y1 - y0))


def panel_entry_point(points: Sequence[Any]) -> Pt:
    """コマの視線入口（右上寄り）。"""
    x0, y0, x1, y1 = core_box(points)
    return (x1 - 0.2 * (x1 - x0), y0 + 0.2 * (y1 - y0))


def same_tier(points_a: Sequence[Any], points_b: Sequence[Any], ratio: float = 0.5) -> bool:
    """2コマが同じ段にあるか（コア矩形の縦方向重なりが低い方の高さの ratio 以上）。"""
    _, ay0, _, ay1 = core_box(points_a)
    _, by0, _, by1 = core_box(points_b)
    overlap = min(ay1, by1) - max(ay0, by0)
    min_height = max(1e-6, min(ay1 - ay0, by1 - by0))
    return overlap / min_height >= ratio


def normalize_vector(dx: float, dy: float, default: Pt = (-1.0, 0.0)) -> Pt:
    length = math.hypot(dx, dy)
    if length < EPS:
        return default
    return (dx / length, dy / length)


def compute_gaze_transitions(page: "Page", tol_x: float = 20.0, tol_y: float = 20.0) -> list["GazeTransition"]:
    """ページ内のコマ順（index_in_page 順）に沿った視線遷移と逆行・交差の判定。"""
    from core.manga_pipeline.schemas import GazeTransition, Point

    panels = sorted(page.panels, key=lambda p: p.index_in_page)
    transitions: list[GazeTransition] = []
    segments: list[tuple[Pt, Pt]] = []
    for current, following in zip(panels, panels[1:]):
        a_pts = _polygon_points_of(current)
        b_pts = _polygon_points_of(following)
        start = panel_exit_point(a_pts)
        end = panel_entry_point(b_pts)
        dx, dy = end[0] - start[0], end[1] - start[1]
        tier = same_tier(a_pts, b_pts)
        if tier:
            backtrack = dx > tol_x
        else:
            backtrack = dy < -tol_y
        transitions.append(
            GazeTransition(
                from_panel=current.id,
                to_panel=following.id,
                start=Point(x=start[0], y=start[1]),
                end=Point(x=end[0], y=end[1]),
                vector=Point(x=dx, y=dy),
                is_backtrack=backtrack,
                same_tier=tier,
            )
        )
        segments.append((start, end))
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments_intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1], proper=True):
                transitions[i].crosses.append(j)
                transitions[j].crosses.append(i)
    return transitions


# ---------------------------------------------------------------------------
# 段（tier）ベースのコマ割り実現
# ---------------------------------------------------------------------------


def plan_tiers(large_flags: Sequence[bool], *, max_tiers: int = 4, capacity: int = 3, pattern: int = 0) -> list[list[int]]:
    """読み順のスロット列を段へ割り付ける。大ゴマ（large）は単独の段を占有する。

    通常スロットの連続区間（run）ごとに段数を配分し、各段の枚数を均等化する。
    pattern は余りの配置（前寄せ/後寄せ）を決める決定論的パラメータ。
    """
    n = len(large_flags)
    if n == 0:
        return []
    runs: list[list[int]] = []
    sequence: list[tuple[str, list[int]]] = []
    current: list[int] = []
    for index, large in enumerate(large_flags):
        if large:
            if current:
                sequence.append(("run", current))
                runs.append(current)
                current = []
            sequence.append(("large", [index]))
        else:
            current.append(index)
    if current:
        sequence.append(("run", current))
        runs.append(current)
    n_large = sum(1 for kind, _ in sequence if kind == "large")
    available = max(len(runs), max_tiers - n_large)
    cap = max(1, capacity)
    allocation: dict[int, int] = {}
    for run_index, run in enumerate(runs):
        length = len(run)
        desired = int(math.floor(length / 2.0 + 0.5)) if length > 2 else 1
        allocation[run_index] = max(1, max(int(math.ceil(length / cap)), desired))
    while sum(allocation.values()) > available:
        candidates = [k for k, v in allocation.items() if v > 1]
        if not candidates:
            break
        widest = max(candidates, key=lambda k: (allocation[k], -k))
        allocation[widest] -= 1
    tiers: list[list[int]] = []
    run_counter = 0
    for kind, members in sequence:
        if kind == "large":
            tiers.append(list(members))
            continue
        k = allocation[run_counter]
        run_counter += 1
        base, extra = divmod(len(members), k)
        sizes = [base] * k
        order = list(range(k)) if pattern % 2 == 0 else list(range(k - 1, -1, -1))
        for j in order[:extra]:
            sizes[j] += 1
        cursor = 0
        for size in sizes:
            if size > 0:
                tiers.append(members[cursor : cursor + size])
                cursor += size
    return tiers


def tier_fractions(
    weights: Sequence[float],
    large_flags: Sequence[bool],
    *,
    large_min: float | None = None,
    normal_min: float = 0.14,
    fixed: dict[int, float] | None = None,
) -> list[float]:
    """段の高さ配分（合計1）。大ゴマ段は large_min 以上、通常段は normal_min 以上を保証する。"""
    count = len(weights)
    if count == 0:
        return []
    fixed = dict(fixed or {})
    n_large = sum(1 for f in large_flags if f)
    if large_min is None:
        large_min = 0.46 if n_large <= 1 else 0.42
    minimums = [large_min if large_flags[i] else normal_min for i in range(count)]
    for index, value in fixed.items():
        minimums[index] = value
    total_min = sum(minimums)
    if total_min > 1.0:
        scale = 1.0 / total_min
        minimums = [m * scale for m in minimums]
        total_min = 1.0
    raw = [max(1e-6, float(w)) for w in weights]
    fractions = list(minimums)
    remaining = 1.0 - total_min
    free = [i for i in range(count) if i not in fixed]
    if remaining > 1e-9 and free:
        weight_sum = sum(raw[i] for i in free)
        for i in free:
            fractions[i] += remaining * raw[i] / weight_sum
    elif remaining > 1e-9:
        fractions = [f + remaining / count for f in fractions]
    total = sum(fractions)
    return [f / total for f in fractions]


def layout_tiers(
    frame: tuple[float, float, float, float],
    tiers_weights: Sequence[Sequence[float]],
    tier_fracs: Sequence[float],
    *,
    gutter_x: float,
    gutter_y: float,
    slants: Sequence[float] | None = None,
    min_width_share: float = 0.22,
) -> list[list[list[Pt]]]:
    """段ごとのコマ多角形（各コマは右→左の読み順、頂点は左上→右上→右下→左下）。

    slants[t] は段 t 内の縦ガターの傾き（上端で +s、下端で −s だけ x をずらす台形化）。
    """
    x0, y0, x1, y1 = frame
    count = len(tiers_weights)
    slant_values = list(slants) if slants is not None else [0.0] * count
    available_h = (y1 - y0) - gutter_y * max(0, count - 1)
    result: list[list[list[Pt]]] = []
    top = y0
    for tier_index, weights in enumerate(tiers_weights):
        height = available_h * float(tier_fracs[tier_index])
        bottom = y1 if tier_index == count - 1 else top + height
        k = len(weights)
        available_w = (x1 - x0) - gutter_x * max(0, k - 1)
        raw = [max(1e-6, float(w)) for w in weights]
        total = sum(raw)
        min_share = min(min_width_share, 0.9 / k) if k > 1 else 1.0
        shares = [max(min_share, r / total) for r in raw]
        share_sum = sum(shares)
        widths = [available_w * s / share_sum for s in shares]
        slant = float(slant_values[tier_index]) if tier_index < len(slant_values) else 0.0
        if k > 1:
            slant = max(-0.25 * min(widths), min(0.25 * min(widths), slant))
        else:
            slant = 0.0
        rights: list[float] = []
        lefts: list[float] = []
        cursor = x1
        for j in range(k):
            rights.append(cursor)
            cursor -= widths[j]
            lefts.append(cursor if j < k - 1 else x0)
            cursor -= gutter_x
        polygons: list[list[Pt]] = []
        for j in range(k):
            right_top = rights[j] + (slant if j > 0 else 0.0)
            right_bottom = rights[j] - (slant if j > 0 else 0.0)
            left_top = lefts[j] + (slant if j < k - 1 else 0.0)
            left_bottom = lefts[j] - (slant if j < k - 1 else 0.0)
            polygons.append([(left_top, top), (right_top, top), (right_bottom, bottom), (left_bottom, bottom)])
        result.append(polygons)
        top = bottom + gutter_y
    return result
