"""ネーム構造ドクター（Storyboard Diagnostics Engine）。

既存の漫画ページ／ネームを逆解析し、「どこを直せば読者の視線誘導と感情の緩急がより劇的になるか」を
定量診断する。

処理の流れ:

1. **正規化** — 入力（``Storyboard`` / ``PageAnnotation``）を内部ネームへ揃える。アノテーションは
   ページ寸法から一様スケールし、読み順は明示値か再帰 XY カットで決める。
2. **逆解析** — コマ面積比・傾き・断ち切り・台詞量・フキダシ被覆率・ベタ比率と、視覚的手がかりからの
   推定劇的強度 I を算出する::

       I = clip(0.30·size + 0.20·shot + 0.12·angle + 0.12·ink + 0.08·sfx + 0.08·shout + 0.05·slant + 0.05·special)
       size = min(1, area_ratio / 0.5)^0.7

   ページ単位では視線遷移（逆行・交差・Z字適合率・経路長）、読み順の曖昧さ、情報密度、めくりの強度を求める。
3. **5軸採点** — 視線誘導・感情の緩急・情報密度・コマ割りリズム・めくり効果（各0〜100）と重み付き総合点。
4. **所見** — 閾値を外れた箇所を severity・観測値・目標値・ピン座標つきで列挙する。
5. **処方箋** — 所見ごとに12種の修正演算子の候補を作り、ネームの複製へ実際に適用して再採点する
   反実仮想シミュレーションで期待改善量を測り、改善量の大きい順に提示する。
6. **編集者コメント** — 決定論ベースラインを作り、LLM があれば多段階ハーネスで増幅する。
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from core.common import clamp, stable_hash, stable_seed
from core.manga_pipeline import geometry as geo
from core.manga_pipeline.diagnostics_schemas import (
    AXIS_LABELS_JA,
    DEFAULT_AXIS_WEIGHTS,
    PRESCRIPTION_LABELS_JA,
    SEVERITY_ORDER,
    AxisScore,
    DiagnosisCommentary,
    DiagnosticAxis,
    DiagnosticFinding,
    DiagnosticReport,
    DiagnosticSourceType,
    IntensityCurve,
    PageAnnotation,
    PageMetrics,
    PanelMetrics,
    Prescription,
    PrescriptionOperation,
    grade_for,
)
from core.manga_pipeline.diagnostics_schemas import AnnotatedBalloon, AnnotatedPage, AnnotatedPanel
from core.manga_pipeline.geometry import compute_gaze_transitions, compute_reading_order
from core.manga_pipeline.grammar_validator import MangaGrammarValidator, page_information_density
from core.manga_pipeline.schemas import (
    Balloon,
    BalloonKind,
    CameraAngle,
    CameraWork,
    CharacterAction,
    CharacterProfile,
    CompositionRule,
    GazeTransition,
    Page,
    PageGeometry,
    PageSide,
    PageTurnRole,
    Panel,
    PanelShape,
    Point,
    Polygon,
    ShotSize,
    SoundEffect,
    Storyboard,
)
from core.manga_pipeline.storyboard_generator import (
    StoryboardConfig,
    finalize_reading_order,
    layout_balloons,
    layout_page,
    page_roles,
    page_sides,
    refresh_page_metrics,
    template_lines,
)
from core.manga_pipeline.text_layout import break_japanese_text, split_text_for_balloons
from core.structure_engine.schemas import EmotionalInformationWave, UniversalStructure
from core.structure_engine.similarity import dtw_distance

logger = logging.getLogger(__name__)

__all__ = [
    "DiagnosticsConfig",
    "StoryboardDiagnosticsEngine",
    "FINDING_CODES",
    "annotation_to_storyboard",
    "storyboard_to_annotation",
    "estimate_panel_intensity",
    "position_label",
]

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

SHOT_WEIGHT: dict[str, float] = {"ECU": 1.0, "CU": 0.8, "MS": 0.45, "FS": 0.35, "ELS": 0.55}
ANGLE_WEIGHT: dict[str, float] = {"dutch": 0.9, "low": 0.75, "high": 0.6, "eye_level": 0.3}
SHOT_LENS: dict[ShotSize, int] = {ShotSize.ECU: 105, ShotSize.CU: 85, ShotSize.MS: 50, ShotSize.FS: 35, ShotSize.ELS: 24}
SHOT_CLOSER: dict[ShotSize, ShotSize] = {
    ShotSize.ELS: ShotSize.FS,
    ShotSize.FS: ShotSize.MS,
    ShotSize.MS: ShotSize.CU,
    ShotSize.CU: ShotSize.ECU,
    ShotSize.ECU: ShotSize.ECU,
}
SHOT_CHARACTER_SCALE: dict[ShotSize, float] = {
    ShotSize.ECU: 1.0,
    ShotSize.CU: 0.85,
    ShotSize.MS: 0.62,
    ShotSize.FS: 0.8,
    ShotSize.ELS: 0.28,
}
SHOT_INK_FACTOR: dict[str, float] = {"ECU": 1.15, "CU": 1.1, "MS": 1.0, "FS": 0.9, "ELS": 0.8}
SHOT_LABELS: dict[str, str] = {"ECU": "超アップ", "CU": "アップ", "MS": "ミディアム", "FS": "全身", "ELS": "超ロング"}
ANGLE_LABELS: dict[str, str] = {"eye_level": "アイレベル", "high": "俯瞰", "low": "アオリ", "dutch": "ダッチアングル"}

SEVERITY_WEIGHT: dict[str, float] = {"critical": 4.0, "major": 3.0, "minor": 2.0, "info": 1.0}
HEURISTIC_GAIN: dict[str, float] = {"critical": 4.0, "major": 2.5, "minor": 1.0, "info": 0.5}
MAX_PANELS_PER_PAGE = 8

FINDING_CODES: dict[str, tuple[DiagnosticAxis, str]] = {
    "GF_BACKTRACK": (DiagnosticAxis.GAZE_FLOW, "視線の逆行"),
    "GF_CROSSING": (DiagnosticAxis.GAZE_FLOW, "視線経路の交差"),
    "GF_AMBIGUOUS_ORDER": (DiagnosticAxis.GAZE_FLOW, "読み順の曖昧さ"),
    "GF_BALLOON_ORDER": (DiagnosticAxis.GAZE_FLOW, "フキダシ読み順の不一致"),
    "GF_LONG_JUMP": (DiagnosticAxis.GAZE_FLOW, "視線の長距離跳躍"),
    "ED_FLAT_SEQUENCE": (DiagnosticAxis.EMOTIONAL_DYNAMICS, "強度の平坦な連続"),
    "ED_WEAK_PEAK": (DiagnosticAxis.EMOTIONAL_DYNAMICS, "山場の視覚的な弱さ"),
    "ED_NO_PAUSE": (DiagnosticAxis.EMOTIONAL_DYNAMICS, "山場直前の溜め不足"),
    "ED_MISALIGNED_CURVE": (DiagnosticAxis.EMOTIONAL_DYNAMICS, "目標曲線とのずれ"),
    "ED_LOW_RANGE": (DiagnosticAxis.EMOTIONAL_DYNAMICS, "強度レンジの不足"),
    "ID_OVERLOADED_BALLOON": (DiagnosticAxis.INFORMATION_DENSITY, "フキダシの文字数過多"),
    "ID_OVERLOADED_PAGE": (DiagnosticAxis.INFORMATION_DENSITY, "ページの情報過密"),
    "ID_CROWDED_PANEL": (DiagnosticAxis.INFORMATION_DENSITY, "フキダシによる絵の圧迫"),
    "ID_FLAT_DENSITY": (DiagnosticAxis.INFORMATION_DENSITY, "情報密度の緩急不足"),
    "ID_SPARSE_RUN": (DiagnosticAxis.INFORMATION_DENSITY, "疎なページの連続"),
    "PR_UNIFORM_GRID": (DiagnosticAxis.PANEL_RHYTHM, "均一なコマ割り"),
    "PR_MONOTONOUS_SHOT": (DiagnosticAxis.PANEL_RHYTHM, "同一ショットの連続"),
    "PR_MONOTONOUS_ANGLE": (DiagnosticAxis.PANEL_RHYTHM, "同一アングルの連続"),
    "PR_PANEL_COUNT": (DiagnosticAxis.PANEL_RHYTHM, "ページのコマ数過多"),
    "PR_NO_SPECIAL": (DiagnosticAxis.PANEL_RHYTHM, "特殊ゴマの不足・乱用"),
    "PT_WEAK_REVEAL": (DiagnosticAxis.PAGE_TURN, "めくり直後の弱さ"),
    "PT_WEAK_HOOK": (DiagnosticAxis.PAGE_TURN, "めくり前の引きの弱さ"),
    "PT_PEAK_MID_PAGE": (DiagnosticAxis.PAGE_TURN, "山場のページ中段への埋没"),
}

_OPERATION_AXIS: dict[PrescriptionOperation, DiagnosticAxis] = {
    PrescriptionOperation.ENLARGE_PANEL: DiagnosticAxis.EMOTIONAL_DYNAMICS,
    PrescriptionOperation.CONVERT_TO_SPLASH: DiagnosticAxis.EMOTIONAL_DYNAMICS,
    PrescriptionOperation.MOVE_PEAK_TO_REVEAL: DiagnosticAxis.PAGE_TURN,
    PrescriptionOperation.STRENGTHEN_HOOK: DiagnosticAxis.PAGE_TURN,
    PrescriptionOperation.INSERT_PAUSE: DiagnosticAxis.EMOTIONAL_DYNAMICS,
    PrescriptionOperation.VARY_SHOT: DiagnosticAxis.PANEL_RHYTHM,
    PrescriptionOperation.SPLIT_BALLOON: DiagnosticAxis.INFORMATION_DENSITY,
    PrescriptionOperation.MOVE_BALLOON: DiagnosticAxis.GAZE_FLOW,
    PrescriptionOperation.REORDER_PANELS: DiagnosticAxis.GAZE_FLOW,
    PrescriptionOperation.REBALANCE_DENSITY: DiagnosticAxis.INFORMATION_DENSITY,
    PrescriptionOperation.ADD_CONTRAST: DiagnosticAxis.PANEL_RHYTHM,
    PrescriptionOperation.ADD_SLANT: DiagnosticAxis.PANEL_RHYTHM,
}


class DiagnosticsConfig(BaseModel):
    """ネーム構造ドクターの設定。"""

    model_config = ConfigDict(extra="ignore")

    max_chars_per_balloon: int = Field(40, ge=8, le=200)
    comfortable_chars_per_page: tuple[int, int] = (30, 220)
    overloaded_chars_per_page: int = Field(320, ge=40, le=5000)
    crowded_coverage: float = Field(0.45, ge=0.05, le=1.0)
    axis_weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_AXIS_WEIGHTS))
    simulate_prescriptions: bool = True
    max_prescriptions: int = Field(12, ge=0, le=60)
    max_candidates: int = Field(36, ge=1, le=200)
    use_llm_commentary: bool = False
    gaze_tolerance: float = Field(20.0, ge=0.0, le=200.0)


# ---------------------------------------------------------------------------
# 補助データ構造
# ---------------------------------------------------------------------------


@dataclass
class _Aux:
    """アノテーション由来の補助値（ベタ比率・作者意図の緊張度）。"""

    ink: dict[str, float] = field(default_factory=dict)
    hint: dict[str, float] = field(default_factory=dict)


@dataclass
class _Context:
    source_type: DiagnosticSourceType
    target_wave: EmotionalInformationWave | None
    structure: UniversalStructure | None
    aux: _Aux
    use_intent: bool


@dataclass
class _Analysis:
    board: Storyboard
    pages: list[Page]
    panel_metrics: list[PanelMetrics]
    page_metrics: list[PageMetrics]
    seq: list[PanelMetrics]
    transitions: dict[int, list[GazeTransition]]
    balloon_violations: list[tuple[str, str, str]]
    balloon_pairs: int
    long_jumps: list[tuple[int, GazeTransition]]
    ambiguous_pairs: list[tuple[int, str, str]]
    flat_runs: list[tuple[int, int]]
    peaks_without_pause: list[int]
    intended_peak: int | None
    has_targets: bool
    correlation: float | None
    intensity_range: float
    axes: list[AxisScore]
    overall: float
    curve: IntensityCurve
    density_windows: list[tuple[int, float]]
    sparse_runs: list[tuple[int, int]]
    uniform_pages: list[int]
    shot_runs: list[list[str]]
    angle_runs: list[list[str]]
    special_ratio: float
    top_turn_ratio: float
    buried_peak: bool


@dataclass
class _Candidate:
    operation: PrescriptionOperation
    finding_ids: list[str]
    page_index: int | None
    panel_ids: list[str]
    balloon_ids: list[str]
    params: dict[str, Any]
    priority: float
    severity: str

    def key(self) -> tuple[Any, ...]:
        return (
            self.operation.value,
            self.page_index,
            tuple(self.panel_ids),
            tuple(sorted((k, str(v)) for k, v in self.params.items())),
        )

    def family(self) -> tuple[Any, ...]:
        """パラメータ違い（pattern 等）を同一視したキー。"""
        params = {k: v for k, v in self.params.items() if k != "pattern"}
        return (
            self.operation.value,
            self.page_index,
            tuple(self.panel_ids),
            tuple(sorted((k, str(v)) for k, v in params.items())),
        )


# ---------------------------------------------------------------------------
# 幾何・指標の補助関数
# ---------------------------------------------------------------------------


def _slant_deg(points: list[tuple[float, float]]) -> float:
    """多角形辺の軸からの最大傾き（度）。短すぎる辺は無視する。"""
    edges = geo.polygon_edges(points)
    perimeter = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges)
    worst = 0.0
    for a, b in edges:
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if perimeter <= 0 or length < 0.05 * perimeter:
            continue
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        worst = max(worst, min(angle, 90.0 - angle))
    return round(worst, 3)


def _union_bbox(panels: list[Panel]) -> tuple[float, float, float, float]:
    boxes = [panel.polygon.bbox() for panel in panels]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _vertical_overlap_ratio(a: Panel, b: Panel) -> float:
    _, ay0, _, ay1 = a.polygon.core_box()
    _, by0, _, by1 = b.polygon.core_box()
    overlap = min(ay1, by1) - max(ay0, by0)
    min_height = max(1e-6, min(ay1 - ay0, by1 - by0))
    return max(0.0, overlap) / min_height


def _is_special(panel: Panel) -> bool:
    return panel.emphasis != "normal" or panel.breakout or panel.shape == PanelShape.SLANTED


def _has_shout(panel: Panel) -> bool:
    return any(b.kind in (BalloonKind.SHOUT, BalloonKind.FLASH) for b in panel.balloons)


def _balloon_coverage(panel: Panel) -> float:
    area = panel.polygon.area()
    if area <= 1e-6:
        return 0.0
    covered = sum(math.pi / 4.0 * b.width * b.height for b in panel.balloons)
    return float(clamp(covered / area, 0.0, 1.0))


def _estimate_ink(panel: Panel, page: Page) -> float:
    base = page.black_ratio_target
    value = base * SHOT_INK_FACTOR.get(panel.camera.shot.value, 1.0)
    if panel.emphasis != "normal":
        value += 0.08
    if panel.camera.angle == CameraAngle.DUTCH:
        value += 0.05
    return float(clamp(value, 0.0, 1.0))


def estimate_panel_intensity(
    *,
    area_ratio: float,
    shot: str,
    angle: str,
    ink_ratio: float,
    sfx_count: int,
    has_shout: bool,
    slant_deg: float,
    special: bool,
    hint: float | None = None,
) -> float:
    """視覚的手がかりからの推定劇的強度 I ∈ [0, 1]（hint があれば 50:50 で混合）。"""
    size = min(1.0, max(0.0, area_ratio) / 0.5) ** 0.7
    value = (
        0.30 * size
        + 0.20 * SHOT_WEIGHT.get(shot, 0.45)
        + 0.12 * ANGLE_WEIGHT.get(angle, 0.3)
        + 0.12 * min(1.0, max(0.0, ink_ratio) / 0.6)
        + 0.08 * min(1.0, sfx_count / 2.0)
        + 0.08 * (1.0 if has_shout else 0.0)
        + 0.05 * min(1.0, abs(slant_deg) / 12.0)
        + 0.05 * (1.0 if special else 0.0)
    )
    value = clamp(value, 0.0, 1.0)
    if hint is not None:
        value = 0.5 * value + 0.5 * clamp(hint, 0.0, 1.0)
    return round(float(value), 4)


def position_label(panel: Panel, page: Page, geometry: PageGeometry) -> str:
    """ページ内のコマ位置の日本語表現（「右上」「中段左」「上段（全幅）」等）。"""
    fx0, fy0, fx1, fy1 = geometry.frame_rect(page.side)
    x0, y0, x1, y1 = panel.polygon.bbox()
    fw = max(1.0, fx1 - fx0)
    fh = max(1.0, fy1 - fy0)
    u = ((x0 + x1) / 2.0 - fx0) / fw
    v = ((y0 + y1) / 2.0 - fy0) / fh
    if (x1 - x0) >= 0.8 * fw and (y1 - y0) >= 0.8 * fh:
        return "全面"
    vert = "上" if v < 1.0 / 3.0 else "下" if v > 2.0 / 3.0 else "中"
    if (x1 - x0) >= 0.8 * fw:
        return {"上": "上段", "中": "中段", "下": "下段"}[vert] + "（全幅）"
    horiz = "右" if u > 2.0 / 3.0 else "左" if u < 1.0 / 3.0 else "中央"
    table = {
        ("右", "上"): "右上",
        ("左", "上"): "左上",
        ("右", "下"): "右下",
        ("左", "下"): "左下",
        ("中央", "上"): "上段中央",
        ("中央", "下"): "下段中央",
        ("右", "中"): "中段右",
        ("左", "中"): "中段左",
        ("中央", "中"): "中央",
    }
    return table[(horiz, vert)]


def _locate(board: Storyboard, panel_id: str) -> tuple[Panel | None, Page | None]:
    for page in board.pages:
        for panel in page.panels:
            if panel.id == panel_id:
                return panel, page
    return None, None


def _page_by_index(board: Storyboard, index: int | None) -> Page | None:
    if index is None:
        return None
    for page in board.pages:
        if page.index == index:
            return page
    return None


def _loc(board: Storyboard, panel_id: str) -> str:
    panel, page = _locate(board, panel_id)
    if panel is None or page is None:
        return f"コマ {panel_id}"
    return f"p{page.number} {position_label(panel, page, board.geometry)}のコマ"


def _in_spread(board: Storyboard, page: Page) -> bool:
    return board.spread_of_page(page.index) is not None or any(p.spread_role != "none" for p in page.panels)


def _set_order(page: Page, panels: list[Panel]) -> None:
    page.panels = list(panels)
    for i, panel in enumerate(page.panels):
        panel.index_in_page = i
        panel.page_index = page.index


def _remap_point(point: Point, old: tuple[float, float, float, float], new: tuple[float, float, float, float]) -> Point:
    ox0, oy0, ox1, oy1 = old
    nx0, ny0, nx1, ny1 = new
    u = (point.x - ox0) / max(1e-6, ox1 - ox0)
    v = (point.y - oy0) / max(1e-6, oy1 - oy0)
    return Point(x=round(nx0 + clamp(u, -0.2, 1.2) * (nx1 - nx0), 2), y=round(ny0 + clamp(v, -0.2, 1.2) * (ny1 - ny0), 2))


def _theme_word(board: Storyboard) -> str:
    for key in ("THEME", "theme", "テーマ"):
        if board.theme.get(key, "").strip():
            return board.theme[key].strip()[:10]
    for value in board.theme.values():
        if value.strip():
            return value.strip()[:10]
    return "あの場所"


# ---------------------------------------------------------------------------
# アノテーション ⇔ ネーム
# ---------------------------------------------------------------------------


def _estimate_shot(character_count: int, area_ratio: float) -> ShotSize:
    if character_count <= 0:
        return ShotSize.ELS if area_ratio >= 0.35 else ShotSize.FS
    if character_count == 1:
        if area_ratio < 0.12:
            return ShotSize.ECU
        if area_ratio < 0.3:
            return ShotSize.CU
        return ShotSize.MS
    if character_count == 2:
        return ShotSize.MS if area_ratio < 0.2 else ShotSize.FS
    return ShotSize.ELS if area_ratio >= 0.3 else ShotSize.FS


def _estimate_geometry(width: float, height: float, unions: list[tuple[tuple[float, float, float, float], PageSide]]) -> PageGeometry:
    tops: list[float] = []
    bottoms: list[float] = []
    inners: list[float] = []
    outers: list[float] = []
    for (x0, y0, x1, y1), side in unions:
        tops.append(y0)
        bottoms.append(height - y1)
        if side == PageSide.RIGHT:
            inners.append(x0)
            outers.append(width - x1)
        else:
            inners.append(width - x1)
            outers.append(x0)

    def _median(values: list[float], limit: float) -> float:
        if not values:
            return 0.0
        return round(float(clamp(float(np.median(values)), 0.0, limit)), 2)

    return PageGeometry(
        width=width,
        height=height,
        bleed=30.0,
        margin_top=_median(tops, 0.3 * height),
        margin_bottom=_median(bottoms, 0.3 * height),
        margin_outer=_median(outers, 0.3 * width),
        margin_inner=_median(inners, 0.3 * width),
    )


def _annotation_to_board(annotation: PageAnnotation) -> tuple[Storyboard, _Aux]:
    """アノテーションを内部ネームへ正規化し、補助値（ベタ比率・意図強度）を返す。"""
    first = annotation.pages[0]
    width = 1000.0
    height = round(width * first.height / first.width, 3)
    defaults = page_sides(len(annotation.pages), annotation.first_page_side)
    sides = [page.side if page.side is not None else defaults[i] for i, page in enumerate(annotation.pages)]
    roles = page_roles(sides)
    scaled: list[tuple[float, list[list[tuple[float, float]]]]] = []
    unions: list[tuple[tuple[float, float, float, float], PageSide]] = []
    for index, apage in enumerate(annotation.pages):
        scale = min(width / apage.width, height / apage.height)
        point_lists = [[(round(x * scale, 3), round(y * scale, 3)) for x, y in apanel.points()] for apanel in apage.panels]
        scaled.append((scale, point_lists))
        xs = [x for pts in point_lists for x, _ in pts]
        ys = [y for pts in point_lists for _, y in pts]
        unions.append(((min(xs), min(ys), max(xs), max(ys)), sides[index]))
    geometry = _estimate_geometry(width, height, unions)
    max_characters = max((apanel.character_count for apage in annotation.pages for apanel in apage.panels), default=0)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    characters = [
        CharacterProfile(
            id=f"c{k}",
            role="character",
            name_placeholder=f"人物{letters[k % 26]}",
            visual_description="manga character, monochrome ink",
        )
        for k in range(max_characters)
    ]
    aux = _Aux()
    pages: list[Page] = []
    for index, apage in enumerate(annotation.pages):
        scale, point_lists = scaled[index]
        union = unions[index][0]
        union_area = max(1.0, _bbox_area(union))
        built: list[Panel] = []
        for j, apanel in enumerate(apage.panels):
            panel_id = f"p{index:02d}_k{j:02d}"
            points = point_lists[j]
            area_ratio = clamp(geo.polygon_area(points) / union_area, 0.0, 1.0)
            slant = _slant_deg(points)
            if len(points) > 4:
                shape = PanelShape.IRREGULAR
            elif slant > 1.5:
                shape = PanelShape.SLANTED
            else:
                shape = PanelShape.RECT
            shot = apanel.shot or _estimate_shot(apanel.character_count, area_ratio)
            angle = apanel.angle or CameraAngle.EYE_LEVEL
            camera = CameraWork(
                shot=shot,
                angle=angle,
                lens_mm=SHOT_LENS[shot],
                composition=CompositionRule.RULE_OF_THIRDS,
                dutch_degrees=15.0 if angle == CameraAngle.DUTCH else 0.0,
            )
            count = apanel.character_count
            xs = [0.5] if count <= 1 else list(np.linspace(0.75, 0.25, count))
            actions = [
                CharacterAction(
                    character_id=f"c{k}",
                    position=Point(x=round(float(xs[k]), 3), y=0.56),
                    scale=SHOT_CHARACTER_SCALE[shot],
                )
                for k in range(count)
            ]
            balloon_boxes: list[tuple[float, float, float, float]] = []
            balloons: list[Balloon] = []
            for k, aballoon in enumerate(apanel.balloons):
                bx0, by0, bx1, by1 = (v * scale for v in aballoon.bbox)
                text = "".join(aballoon.text.split())
                if not text:
                    text = "■" * max(1, int(aballoon.char_count or 1))
                balloons.append(
                    Balloon(
                        id=f"{panel_id}_b{k}",
                        kind=aballoon.kind,
                        speaker_id=aballoon.speaker,
                        text=text[:400],
                        columns=break_japanese_text(text[:400], 7),
                        center=Point(x=round((bx0 + bx1) / 2.0, 2), y=round((by0 + by1) / 2.0, 2)),
                        width=round(max(1.0, bx1 - bx0), 2),
                        height=round(max(1.0, by1 - by0), 2),
                    )
                )
                balloon_boxes.append((bx0, by0, bx1, by1))
            if apanel.balloons:
                hints = [b.order_hint for b in apanel.balloons]
                if all(h is not None for h in hints):
                    order = sorted(range(len(balloons)), key=lambda i: (hints[i], i))
                else:
                    order = geo.xy_cut_order(balloon_boxes, tol=4.0)
                for rank, i in enumerate(order):
                    balloons[i].reading_order = rank
            cx, cy = geo.polygon_centroid(points)
            sfx = [
                SoundEffect(
                    id=f"{panel_id}_s{k}",
                    text="描き文字",
                    position=Point(x=round(cx - 30.0 * k, 2), y=round(cy + 20.0 * k, 2)),
                    style="impact",
                )
                for k in range(apanel.sfx_count)
            ]
            is_silent = apanel.is_silent if apanel.is_silent is not None else (not apanel.balloons and apanel.sfx_count == 0)
            if len(apage.panels) == 1 and area_ratio >= 0.85:
                emphasis = "splash"
            elif area_ratio >= 0.4:
                emphasis = "large"
            else:
                emphasis = "normal"
            panel = Panel(
                id=panel_id,
                page_index=index,
                index_in_page=j,
                polygon=Polygon.from_tuples(points),
                shape=shape,
                bleed_sides=list(apanel.bleed_sides),
                breakout=apanel.breakout,
                camera=camera,
                characters=actions,
                balloons=balloons,
                sfx=sfx,
                description=apanel.note or f"既存原稿 p{index + 1} コマ{j + 1}",
                t=0.0,
                tension=0.5,
                emphasis=emphasis,  # type: ignore[arg-type]
                is_silent=bool(is_silent),
            )
            built.append(panel)
            if apanel.ink_ratio is not None:
                aux.ink[panel_id] = float(apanel.ink_ratio)
            if apanel.intensity_hint is not None:
                aux.hint[panel_id] = float(apanel.intensity_hint)
        if apage.reading_order is not None:
            ordered = [built[i] for i in apage.reading_order]
        else:
            ids = compute_reading_order(built)
            by_id = {p.id: p for p in built}
            ordered = [by_id[i] for i in ids]
        page = Page(
            index=index,
            number=index + 1,
            side=sides[index],
            page_turn_role=roles[index],
        )
        _set_order(page, ordered)
        pages.append(page)
    digest = stable_hash(annotation.model_dump_json()) % 16**10
    board = Storyboard(
        id=f"sb_ann_{digest:010x}",
        title=annotation.title,
        structure_id="annotation",
        geometry=geometry,
        first_page_side=annotation.first_page_side,
        pages=pages,
        characters=characters,
        seed=0,
    )
    finalize_reading_order(board)
    ordered_all = board.panels_in_reading_order()
    total = max(1, len(ordered_all))
    for k, panel in enumerate(ordered_all):
        panel.t = round((k + 0.5) / total, 6)
    for page in board.pages:
        refresh_page_metrics(page)
        union = _union_bbox(page.panels)
        union_area = max(1.0, _bbox_area(union))
        for panel in page.panels:
            hint = aux.hint.get(panel.id)
            if hint is not None:
                panel.tension = round(hint, 4)
            else:
                ink = aux.ink.get(panel.id, _estimate_ink(panel, page))
                panel.tension = estimate_panel_intensity(
                    area_ratio=clamp(panel.polygon.area() / union_area, 0.0, 1.0),
                    shot=panel.camera.shot.value,
                    angle=panel.camera.angle.value,
                    ink_ratio=ink,
                    sfx_count=len(panel.sfx),
                    has_shout=_has_shout(panel),
                    slant_deg=_slant_deg(panel.polygon.tuples()),
                    special=_is_special(panel),
                )
        refresh_page_metrics(page)
        page.t_start = min(p.t for p in page.panels)
        page.t_end = max(p.t for p in page.panels)
    return Storyboard.model_validate(board.model_dump()), aux


def annotation_to_storyboard(annotation: PageAnnotation) -> Storyboard:
    """既存原稿のアノテーションを内部ネーム（Storyboard）へ正規化する。"""
    board, _ = _annotation_to_board(annotation)
    return board


def storyboard_to_annotation(storyboard: Storyboard) -> PageAnnotation:
    """ネームをアノテーション形式へ書き出す（既存原稿の手直し・外部ツール連携用）。"""
    geometry = storyboard.geometry
    pages: list[AnnotatedPage] = []
    for page in sorted(storyboard.pages, key=lambda p: p.index):
        panels: list[AnnotatedPanel] = []
        for panel in page.ordered_panels():
            points = panel.polygon.tuples()
            box = panel.polygon.bbox()
            is_rect = len(points) == 4 and _slant_deg(points) < 0.01
            balloons = [
                AnnotatedBalloon(
                    text=balloon.text,
                    bbox=(
                        round(balloon.center.x - balloon.width / 2.0, 3),
                        round(balloon.center.y - balloon.height / 2.0, 3),
                        round(balloon.center.x + balloon.width / 2.0, 3),
                        round(balloon.center.y + balloon.height / 2.0, 3),
                    ),
                    kind=balloon.kind,
                    speaker=balloon.speaker_id,
                    order_hint=balloon.reading_order,
                )
                for balloon in sorted(panel.balloons, key=lambda b: b.reading_order)
            ]
            panels.append(
                AnnotatedPanel(
                    bbox=box if is_rect else None,
                    polygon=None if is_rect else [Point(x=x, y=y) for x, y in points],
                    balloons=balloons,
                    shot=panel.camera.shot,
                    angle=panel.camera.angle,
                    character_count=min(20, len(panel.characters)),
                    sfx_count=min(20, len(panel.sfx)),
                    is_silent=panel.is_silent,
                    bleed_sides=list(panel.bleed_sides),
                    breakout=panel.breakout,
                    intensity_hint=round(float(panel.tension), 4),
                    note=panel.description[:1000],
                )
            )
        pages.append(AnnotatedPage(width=geometry.width, height=geometry.height, side=page.side, panels=panels))
    return PageAnnotation(title=storyboard.title, first_page_side=storyboard.first_page_side, pages=pages)


# ---------------------------------------------------------------------------
# エンジン本体
# ---------------------------------------------------------------------------


def _target_wave(target: Any) -> EmotionalInformationWave | None:
    if isinstance(target, UniversalStructure):
        return target.wave
    if isinstance(target, EmotionalInformationWave):
        return target
    return None


def _aux_from_report(report: DiagnosticReport) -> _Aux:
    aux = _Aux()
    if report.source_type == DiagnosticSourceType.STORYBOARD:
        return aux
    for metric in report.panels:
        aux.ink[metric.panel_id] = metric.ink_ratio
        if report.target_structure_id is None and metric.target_tension is not None:
            aux.hint[metric.panel_id] = metric.target_tension
    return aux


class StoryboardDiagnosticsEngine:
    """ネーム構造ドクター。"""

    def __init__(self, config: DiagnosticsConfig | None = None, llm_client: Any | None = None) -> None:
        self.config = config or DiagnosticsConfig()
        self.llm_client = llm_client
        self._story_config = StoryboardConfig(max_chars_per_balloon=self.config.max_chars_per_balloon)

    # -- 公開 API --------------------------------------------------------------

    def diagnose(
        self,
        storyboard: Storyboard,
        *,
        target: UniversalStructure | EmotionalInformationWave | None = None,
        source_type: DiagnosticSourceType = DiagnosticSourceType.STORYBOARD,
        source_id: str | None = None,
        title: str | None = None,
    ) -> DiagnosticReport:
        """ネームを診断してレポートを返す（決定論的）。"""
        board = storyboard.model_copy(deep=True)
        source = DiagnosticSourceType(source_type)
        ctx = self._context(target, source, _Aux(), use_intent=source == DiagnosticSourceType.STORYBOARD)
        resolved_source_id = source_id if source_id is not None else (storyboard.id if source == DiagnosticSourceType.STORYBOARD else None)
        return self._build_report(board, ctx, source_id=resolved_source_id, title=title or board.title)

    def diagnose_annotation(
        self,
        annotation: PageAnnotation,
        *,
        target: UniversalStructure | EmotionalInformationWave | None = None,
        source_type: DiagnosticSourceType = DiagnosticSourceType.ANNOTATION,
        source_id: str | None = None,
    ) -> DiagnosticReport:
        """既存原稿アノテーションを診断する。"""
        board, aux = _annotation_to_board(annotation)
        ctx = self._context(target, DiagnosticSourceType(source_type), aux, use_intent=False)
        return self._build_report(board, ctx, source_id=source_id, title=annotation.title)

    async def diagnose_async(self, storyboard: Storyboard, *, use_llm: bool | None = None, **kwargs: Any) -> DiagnosticReport:
        """診断＋（LLM があれば）多段階ハーネスによる編集者コメントの増幅。"""
        report = self.diagnose(storyboard, **kwargs)
        return await self._amplify(report, use_llm)

    async def diagnose_annotation_async(self, annotation: PageAnnotation, *, use_llm: bool | None = None, **kwargs: Any) -> DiagnosticReport:
        report = self.diagnose_annotation(annotation, **kwargs)
        return await self._amplify(report, use_llm)

    def score(
        self,
        storyboard: Storyboard,
        *,
        target: UniversalStructure | EmotionalInformationWave | None = None,
    ) -> tuple[float, list[AxisScore]]:
        """高速採点（所見・処方を作らない）。戻り値は (総合点, 5軸スコア)。"""
        ctx = self._context(target, DiagnosticSourceType.STORYBOARD, _Aux(), use_intent=True)
        analysis = self._analyze(storyboard, ctx)
        return round(analysis.overall, 2), analysis.axes

    def apply_operation(
        self,
        storyboard: Storyboard,
        operation: PrescriptionOperation | str,
        params: dict[str, Any],
        *,
        target: UniversalStructure | EmotionalInformationWave | None = None,
    ) -> Storyboard | None:
        """修正演算子を複製へ適用した新しいネームを返す（適用不能なら None）。"""
        board = storyboard.model_copy(deep=True)
        ctx = self._context(target, DiagnosticSourceType.STORYBOARD, _Aux(), use_intent=True)
        if not self._apply_operation(board, PrescriptionOperation(operation), dict(params), ctx):
            return None
        return Storyboard.model_validate(board.model_dump())

    def apply_prescriptions(
        self,
        storyboard: Storyboard,
        report: DiagnosticReport,
        prescription_ids: list[str],
        *,
        target: UniversalStructure | EmotionalInformationWave | None = None,
    ) -> tuple[Storyboard, DiagnosticReport]:
        """指定処方を順に適用（総合点を下げるものと適用不能なものはスキップ）し、再診断する。"""
        board = storyboard.model_copy(deep=True)
        source = report.source_type
        ctx = self._context(target, source, _aux_from_report(report), use_intent=source == DiagnosticSourceType.STORYBOARD)
        by_id = {rx.id: rx for rx in report.prescriptions}
        current = self._analyze(board, ctx).overall
        for prescription_id in prescription_ids:
            rx = by_id.get(prescription_id)
            if rx is None:
                board.revision_log.append(f"ネーム構造ドクター: 処方 {prescription_id} はレポートに存在しないためスキップ")
                continue
            trial = board.model_copy(deep=True)
            try:
                applied = self._apply_operation(trial, rx.operation, dict(rx.params), ctx)
            except (ValueError, KeyError, IndexError) as exc:
                logger.warning("処方 %s の適用に失敗しました: %s", rx.id, exc)
                applied = False
            if not applied:
                board.revision_log.append(f"ネーム構造ドクター: {rx.title} は適用条件を満たさないためスキップ")
                continue
            trial_score = self._analyze(trial, ctx).overall
            if trial_score + 1e-9 < current:
                board.revision_log.append(f"ネーム構造ドクター: {rx.title} は総合点を下げるためスキップ（{current:.2f}→{trial_score:.2f}）")
                continue
            board = trial
            board.revision_log.append(f"ネーム構造ドクター適用: {rx.title}（{current:.2f}→{trial_score:.2f}）")
            current = trial_score
        finalize_reading_order(board)
        for page in board.pages:
            refresh_page_metrics(page)
        board = Storyboard.model_validate(board.model_dump())
        structure = target if isinstance(target, UniversalStructure) else None
        board.grammar = MangaGrammarValidator(self._story_config).validate(board, structure)
        new_report = self._build_report(board, ctx, source_id=report.source_id, title=report.title)
        new_report.page_images = list(report.page_images)
        if report.target_structure_id and not new_report.target_structure_id:
            new_report.target_structure_id = report.target_structure_id
        return board, new_report

    # -- 内部: 文脈 ---------------------------------------------------------------

    def _context(self, target: Any, source: DiagnosticSourceType, aux: _Aux, *, use_intent: bool) -> _Context:
        return _Context(
            source_type=source,
            target_wave=_target_wave(target),
            structure=target if isinstance(target, UniversalStructure) else None,
            aux=aux,
            use_intent=use_intent,
        )

    # -- 内部: 逆解析 ---------------------------------------------------------------

    def _analyze(self, board: Storyboard, ctx: _Context) -> _Analysis:
        cfg = self.config
        pages = sorted(board.pages, key=lambda p: p.index)
        if not any(page.panels for page in pages):
            raise ValueError("診断対象のコマがありません")
        geometry = board.geometry
        diagonal = math.hypot(geometry.width, geometry.height)
        panel_metrics: list[PanelMetrics] = []
        metric_by_id: dict[str, PanelMetrics] = {}
        for page in pages:
            panels = page.ordered_panels()
            if not panels:
                continue
            union_area = max(1.0, _bbox_area(_union_bbox(panels)))
            for panel in panels:
                points = panel.polygon.tuples()
                bbox = panel.polygon.bbox()
                area_ratio = float(clamp(panel.polygon.area() / union_area, 0.0, 1.0))
                slant = _slant_deg(points)
                ink = ctx.aux.ink.get(panel.id)
                if ink is None:
                    ink = _estimate_ink(panel, page)
                special = panel.emphasis in ("splash", "spread") or panel.breakout or bool(panel.bleed_sides)
                intensity = estimate_panel_intensity(
                    area_ratio=area_ratio,
                    shot=panel.camera.shot.value,
                    angle=panel.camera.angle.value,
                    ink_ratio=ink,
                    sfx_count=len(panel.sfx),
                    has_shout=_has_shout(panel),
                    slant_deg=slant,
                    special=special,
                    hint=ctx.aux.hint.get(panel.id),
                )
                if ctx.target_wave is not None:
                    target_tension: float | None = round(ctx.target_wave.sample("tension", panel.t), 4)
                elif ctx.use_intent:
                    target_tension = round(float(panel.tension), 4)
                else:
                    hint = ctx.aux.hint.get(panel.id)
                    target_tension = round(hint, 4) if hint is not None else None
                metric = PanelMetrics(
                    panel_id=panel.id,
                    page_index=page.index,
                    index_in_page=panel.index_in_page,
                    reading_order=panel.reading_order,
                    area_ratio=round(area_ratio, 4),
                    aspect=round(max(1e-3, (bbox[2] - bbox[0]) / max(1e-6, bbox[3] - bbox[1])), 4),
                    slant_deg=slant,
                    is_bleed=bool(panel.bleed_sides),
                    is_breakout=panel.breakout,
                    is_silent=panel.is_silent or (not panel.balloons and not panel.sfx),
                    shot=panel.camera.shot.value,
                    angle=panel.camera.angle.value,
                    char_count=panel.text_char_count(),
                    balloon_count=len(panel.balloons),
                    balloon_coverage=round(_balloon_coverage(panel), 4),
                    ink_ratio=round(float(clamp(ink, 0.0, 1.0)), 4),
                    intensity=intensity,
                    target_tension=target_tension,
                    t=round(float(clamp(panel.t, 0.0, 1.0)), 6),
                )
                panel_metrics.append(metric)
                metric_by_id[panel.id] = metric
        panel_lookup = {panel.id: panel for page in pages for panel in page.panels}
        seq = [m for m in panel_metrics if panel_lookup[m.panel_id].spread_role != "continuation"]
        intensities = np.asarray([m.intensity for m in seq], dtype=float)
        n = len(seq)
        has_targets = n > 0 and all(m.target_tension is not None for m in seq)
        targets = np.asarray([m.target_tension if m.target_tension is not None else 0.0 for m in seq], dtype=float)

        # --- 視線遷移・ページ指標 ---
        transitions: dict[int, list[GazeTransition]] = {}
        balloon_violations: list[tuple[str, str, str]] = []
        balloon_pairs = 0
        long_jumps: list[tuple[int, GazeTransition]] = []
        ambiguous_pairs: list[tuple[int, str, str]] = []
        page_metrics: list[PageMetrics] = []
        mean_all = float(np.mean(intensities)) if n else 0.0
        comfortable_low, comfortable_high = cfg.comfortable_chars_per_page
        for page in pages:
            panels = page.ordered_panels()
            page_transitions = compute_gaze_transitions(page, tol_x=cfg.gaze_tolerance, tol_y=cfg.gaze_tolerance) if len(panels) > 1 else []
            transitions[page.index] = page_transitions
            backtracks = sum(1 for t in page_transitions if t.is_backtrack)
            crossing_pairs = sum(len(t.crosses) for t in page_transitions) // 2
            path = sum(math.hypot(t.vector.x, t.vector.y) for t in page_transitions)
            for transition in page_transitions:
                if math.hypot(transition.vector.x, transition.vector.y) > 0.9 * diagonal:
                    long_jumps.append((page.index, transition))
            ambiguous = 0
            for a, b in zip(panels, panels[1:]):
                ratio = _vertical_overlap_ratio(a, b)
                if 0.25 <= ratio <= 0.75:
                    ambiguous += 1
                    ambiguous_pairs.append((page.index, a.id, b.id))
            for panel in panels:
                ordered = sorted(panel.balloons, key=lambda bl: bl.reading_order)
                for a, b in zip(ordered, ordered[1:]):
                    balloon_pairs += 1
                    to_the_right = b.center.x > a.center.x + 8.0
                    not_lower = b.center.y < a.center.y + 0.5 * min(a.height, b.height)
                    if to_the_right and not_lower:
                        balloon_violations.append((panel.id, a.id, b.id))
            chars = sum(p.text_char_count() for p in panels)
            balloons = sum(len(p.balloons) for p in panels)
            if chars > cfg.overloaded_chars_per_page:
                level = "overloaded"
            elif chars > comfortable_high:
                level = "dense"
            elif chars < comfortable_low:
                level = "sparse"
            else:
                level = "balanced"
            if (balloons > 12 or len(panels) > MAX_PANELS_PER_PAGE) and level in ("sparse", "balanced"):
                level = "dense"
            areas = np.asarray([p.polygon.area() for p in panels], dtype=float)
            area_cv = float(np.std(areas) / np.mean(areas)) if len(areas) > 1 and float(np.mean(areas)) > 0 else 0.0
            page_intensities = [metric_by_id[p.id].intensity for p in panels]
            role = page.page_turn_role
            reveal_strength = max(page_intensities[:2]) if role == PageTurnRole.REVEAL and page_intensities else None
            hook_strength = page_intensities[-1] if role == PageTurnRole.HOOK and page_intensities else None
            count_t = len(page_transitions)
            backtrack_rate = backtracks / count_t if count_t else 0.0
            crossing_rate = min(1.0, crossing_pairs / count_t) if count_t else 0.0
            ambiguity = ambiguous / max(1, len(panels) - 1) if len(panels) > 1 else 0.0
            density_penalty = {"overloaded": 25.0, "dense": 6.0, "sparse": 0.0, "balanced": 0.0}[level]
            uniform = len(panels) >= 4 and area_cv < 0.2
            internal_contrast = (max(page_intensities) - float(np.mean(page_intensities))) if page_intensities else 0.0
            page_score = (
                100.0
                - 30.0 * backtrack_rate
                - 15.0 * crossing_rate
                - 10.0 * ambiguity
                - density_penalty
                - (12.0 if uniform else 0.0)
                - (8.0 if len(panels) >= 3 and internal_contrast < 0.08 else 0.0)
            )
            page_metrics.append(
                PageMetrics(
                    page_index=page.index,
                    side=page.side.value,
                    page_turn_role=role.value,
                    panel_count=len(panels),
                    char_count=chars,
                    balloon_count=balloons,
                    information_density=page_information_density(page),
                    density_level=level,  # type: ignore[arg-type]
                    area_cv=round(area_cv, 4),
                    max_area_ratio=round(max((metric_by_id[p.id].area_ratio for p in panels), default=0.0), 4),
                    mean_intensity=round(float(np.mean(page_intensities)) if page_intensities else 0.0, 4),
                    peak_intensity=round(max(page_intensities, default=0.0), 4),
                    gaze_path_length=round(path / diagonal, 4),
                    backtracks=backtracks,
                    crossings=crossing_pairs,
                    z_flow_conformity=round(1.0 - backtrack_rate, 4),
                    order_ambiguity=round(ambiguity, 4),
                    reveal_strength=round(reveal_strength, 4) if reveal_strength is not None else None,
                    hook_strength=round(hook_strength, 4) if hook_strength is not None else None,
                    score=round(float(clamp(page_score, 0.0, 100.0)), 2),
                )
            )

        # --- 軸1: 視線誘導 ---
        total_transitions = sum(len(v) for v in transitions.values())
        total_backtracks = sum(pm.backtracks for pm in page_metrics)
        total_crossings = sum(pm.crossings for pm in page_metrics)
        consecutive_pairs = sum(max(0, pm.panel_count - 1) for pm in page_metrics)
        backtrack_rate = total_backtracks / total_transitions if total_transitions else 0.0
        crossing_rate = min(1.0, total_crossings / total_transitions) if total_transitions else 0.0
        conformity = 1.0 - backtrack_rate
        ambiguity = len(ambiguous_pairs) / consecutive_pairs if consecutive_pairs else 0.0
        balloon_violation_rate = len(balloon_violations) / balloon_pairs if balloon_pairs else 0.0
        long_penalty = min(15.0, 3.0 * len(long_jumps))
        gaze_score = clamp(
            100.0
            - 45.0 * backtrack_rate
            - 25.0 * crossing_rate
            - 15.0 * (1.0 - conformity)
            - 10.0 * ambiguity
            - 5.0 * balloon_violation_rate
            - long_penalty,
            0.0,
            100.0,
        )
        gaze_axis = AxisScore(
            axis=DiagnosticAxis.GAZE_FLOW,
            label=AXIS_LABELS_JA[DiagnosticAxis.GAZE_FLOW.value],
            score=round(gaze_score, 2),
            weight=float(cfg.axis_weights.get(DiagnosticAxis.GAZE_FLOW.value, 0.0)),
            metrics={
                "transitions": float(total_transitions),
                "backtrack_rate": round(backtrack_rate, 4),
                "crossing_rate": round(crossing_rate, 4),
                "z_flow_conformity": round(conformity, 4),
                "order_ambiguity": round(ambiguity, 4),
                "balloon_order_violation_rate": round(balloon_violation_rate, 4),
                "long_jumps": float(len(long_jumps)),
            },
            summary=(
                f"遷移 {total_transitions} 本中、逆行 {total_backtracks}・交差 {total_crossings}・"
                f"曖昧な段 {len(ambiguous_pairs)}・フキダシ順不一致 {len(balloon_violations)}。"
                + ("右上→左下の Z 字が明快です。" if gaze_score >= 85 else "読み順の迷いが生じる箇所があります。")
            ),
        )

        # --- 軸2: 感情の緩急 ---
        if n >= 2:
            intensity_range = float(np.percentile(intensities, 90) - np.percentile(intensities, 10))
            contrast = float(np.mean(np.abs(np.diff(intensities))))
        else:
            intensity_range = 0.0
            contrast = 0.0
        range_score = min(1.0, intensity_range / 0.4)
        contrast_score = min(1.0, contrast / 0.12)
        flat_runs: list[tuple[int, int]] = []
        run_start = 0
        for i in range(1, n + 1):
            continues = i < n and abs(float(intensities[i] - intensities[i - 1])) < 0.05
            if not continues:
                if i - run_start >= 4:
                    flat_runs.append((run_start, i - 1))
                run_start = i
        flat_penalty = min(30.0, sum(6.0 * ((end - start + 1) - 3) for start, end in flat_runs))
        if n:
            peak_index = int(np.argmax(targets)) if has_targets else int(np.argmax(intensities))
            rank_pct = float(np.sum(intensities > intensities[peak_index])) / n
            areas_seq = np.asarray([m.area_ratio for m in seq], dtype=float)
            area_pct = float(np.sum(areas_seq > areas_seq[peak_index])) / n
            rank_score = clamp(1.0 - max(0.0, rank_pct - 0.1) / 0.5, 0.0, 1.0)
            area_score = clamp(1.0 - max(0.0, area_pct - 0.1) / 0.5, 0.0, 1.0)
            peak_score = 0.6 * rank_score + 0.4 * area_score
        else:
            peak_index = None
            peak_score = 0.0
        peaks = [
            k
            for k in range(1, n)
            if intensities[k] >= 0.7 and intensities[k] >= intensities[k - 1] and (k == n - 1 or intensities[k] >= intensities[k + 1])
        ]
        peaks_without_pause: list[int] = []
        for k in peaks:
            has_pause = any(
                j >= 0 and (intensities[k] - intensities[j] >= 0.25 or seq[j].is_silent) for j in (k - 1, k - 2)
            )
            if not has_pause:
                peaks_without_pause.append(k)
        pause_ratio = (len(peaks) - len(peaks_without_pause)) / len(peaks) if peaks else 0.6
        correlation: float | None = None
        if has_targets and n >= 3 and float(np.std(intensities)) > 1e-9 and float(np.std(targets)) > 1e-9:
            correlation = float(np.corrcoef(intensities, targets)[0, 1])
        if correlation is not None:
            corr_score = clamp((correlation + 0.2) / 0.9, 0.0, 1.0)
            emotion_raw = 100.0 * (0.2 * range_score + 0.2 * contrast_score + 0.2 * peak_score + 0.15 * pause_ratio + 0.25 * corr_score)
        else:
            emotion_raw = 100.0 * (0.27 * range_score + 0.27 * contrast_score + 0.26 * peak_score + 0.2 * pause_ratio)
        emotion_score = clamp(emotion_raw - flat_penalty, 0.0, 100.0)
        emotion_axis = AxisScore(
            axis=DiagnosticAxis.EMOTIONAL_DYNAMICS,
            label=AXIS_LABELS_JA[DiagnosticAxis.EMOTIONAL_DYNAMICS.value],
            score=round(emotion_score, 2),
            weight=float(cfg.axis_weights.get(DiagnosticAxis.EMOTIONAL_DYNAMICS.value, 0.0)),
            metrics={
                "intensity_range": round(intensity_range, 4),
                "mean_adjacent_change": round(contrast, 4),
                "flat_runs": float(len(flat_runs)),
                "peak_emphasis": round(peak_score, 4),
                "pause_ratio": round(pause_ratio, 4),
                "correlation": round(correlation, 4) if correlation is not None else -2.0,
            },
            summary=(
                f"推定強度のレンジ {intensity_range:.2f}・隣接変化 {contrast:.2f}・平坦区間 {len(flat_runs)}・"
                f"溜めのある山場 {len(peaks) - len(peaks_without_pause)}/{len(peaks)}"
                + (f"・目標曲線との相関 {correlation:.2f}" if correlation is not None else "")
                + "。"
            ),
        )

        # --- 軸3: 情報密度 ---
        over_balloons = [
            (panel.id, balloon.id)
            for page in pages
            for panel in page.ordered_panels()
            for balloon in panel.balloons
            if balloon.char_count > cfg.max_chars_per_balloon
        ]
        crowded = [m for m in panel_metrics if m.balloon_coverage > cfg.crowded_coverage]
        overloaded_pages = [pm for pm in page_metrics if pm.density_level == "overloaded"]
        dense_pages = [pm for pm in page_metrics if pm.density_level == "dense"]
        density_windows: list[tuple[int, float]] = []
        flat_density_penalty = 0.0
        densities = [pm.information_density for pm in page_metrics]
        for i in range(len(page_metrics) - 2):
            window = np.asarray(densities[i : i + 3], dtype=float)
            mean = float(np.mean(window))
            cv = float(np.std(window) / mean) if mean > 1e-9 else 0.0
            if cv < 0.15:
                density_windows.append((page_metrics[i + 1].page_index, cv))
                flat_density_penalty += 5.0 * (0.15 - cv) / 0.15
        flat_density_penalty = min(20.0, flat_density_penalty)
        sparse_runs: list[tuple[int, int]] = []
        run_start_page: int | None = None
        for i, pm in enumerate(page_metrics + [None]):  # type: ignore[list-item]
            sparse = pm is not None and pm.density_level == "sparse"
            if sparse and run_start_page is None:
                run_start_page = i
            elif not sparse and run_start_page is not None:
                if i - run_start_page >= 3:
                    sparse_runs.append((page_metrics[run_start_page].page_index, page_metrics[i - 1].page_index))
                run_start_page = None
        density_score = clamp(
            100.0
            - min(30.0, 6.0 * len(over_balloons))
            - min(36.0, 12.0 * len(overloaded_pages))
            - min(12.0, 3.0 * len(dense_pages))
            - min(20.0, 4.0 * len(crowded))
            - flat_density_penalty
            - min(16.0, 8.0 * len(sparse_runs)),
            0.0,
            100.0,
        )
        density_axis = AxisScore(
            axis=DiagnosticAxis.INFORMATION_DENSITY,
            label=AXIS_LABELS_JA[DiagnosticAxis.INFORMATION_DENSITY.value],
            score=round(density_score, 2),
            weight=float(cfg.axis_weights.get(DiagnosticAxis.INFORMATION_DENSITY.value, 0.0)),
            metrics={
                "overloaded_balloons": float(len(over_balloons)),
                "overloaded_pages": float(len(overloaded_pages)),
                "dense_pages": float(len(dense_pages)),
                "crowded_panels": float(len(crowded)),
                "flat_density_windows": float(len(density_windows)),
                "sparse_runs": float(len(sparse_runs)),
                "mean_chars_per_page": round(float(np.mean([pm.char_count for pm in page_metrics])) if page_metrics else 0.0, 2),
            },
            summary=(
                f"文字数過多のフキダシ {len(over_balloons)}・過密ページ {len(overloaded_pages)}・"
                f"絵を圧迫するコマ {len(crowded)}・密度が平板な区間 {len(density_windows)}。"
            ),
        )

        # --- 軸4: コマ割りリズム ---
        uniform_pages = [pm.page_index for pm in page_metrics if pm.panel_count >= 4 and pm.area_cv < 0.2]
        rhythm_panels = [panel_lookup[m.panel_id] for m in seq]
        shot_runs = self._runs(rhythm_panels, "shot")
        angle_runs = self._runs(rhythm_panels, "angle")
        crowded_pages = [pm for pm in page_metrics if pm.panel_count > MAX_PANELS_PER_PAGE]
        single_ratio = sum(1 for pm in page_metrics if pm.panel_count == 1) / len(page_metrics) if page_metrics else 0.0
        special_pages = sum(1 for page in pages if any(_is_special(p) for p in page.panels))
        special_ratio = special_pages / len(pages) if pages else 0.0
        special_penalty = 0.0
        if special_ratio < 0.2:
            special_penalty = (0.2 - special_ratio) / 0.2 * 15.0
        elif special_ratio > 0.8:
            special_penalty = (special_ratio - 0.8) / 0.2 * 10.0
        rhythm_score = clamp(
            100.0
            - min(30.0, 10.0 * len(uniform_pages))
            - min(20.0, 5.0 * len(shot_runs))
            - min(15.0, 3.0 * len(angle_runs))
            - min(18.0, 6.0 * len(crowded_pages))
            - (10.0 if single_ratio > 0.3 else 0.0)
            - special_penalty,
            0.0,
            100.0,
        )
        rhythm_axis = AxisScore(
            axis=DiagnosticAxis.PANEL_RHYTHM,
            label=AXIS_LABELS_JA[DiagnosticAxis.PANEL_RHYTHM.value],
            score=round(rhythm_score, 2),
            weight=float(cfg.axis_weights.get(DiagnosticAxis.PANEL_RHYTHM.value, 0.0)),
            metrics={
                "uniform_pages": float(len(uniform_pages)),
                "shot_runs": float(len(shot_runs)),
                "angle_runs": float(len(angle_runs)),
                "crowded_pages": float(len(crowded_pages)),
                "single_panel_ratio": round(single_ratio, 4),
                "special_page_ratio": round(special_ratio, 4),
                "mean_area_cv": round(float(np.mean([pm.area_cv for pm in page_metrics])) if page_metrics else 0.0, 4),
            },
            summary=(
                f"均一グリッドのページ {len(uniform_pages)}・同一ショット連続 {len(shot_runs)}・同一アングル連続 {len(angle_runs)}・"
                f"特殊ゴマを含むページ比率 {special_ratio:.0%}。"
            ),
        )

        # --- 軸5: めくり効果 ---
        reveal_values = [pm.reveal_strength for pm in page_metrics if pm.reveal_strength is not None]
        hook_values = [pm.hook_strength for pm in page_metrics if pm.hook_strength is not None]
        reveal_part = clamp(0.5 + (float(np.mean(reveal_values)) - mean_all) / 0.3, 0.0, 1.0) if reveal_values else 0.75
        hook_part = clamp(0.5 + (float(np.mean(hook_values)) - mean_all) / 0.3, 0.0, 1.0) if hook_values else 0.75
        page_by_index = {page.index: page for page in pages}

        def at_turn_position(metric: PanelMetrics) -> bool:
            panel = panel_lookup[metric.panel_id]
            page = page_by_index[metric.page_index]
            if page.page_turn_role == PageTurnRole.REVEAL and metric.index_in_page <= 1:
                return True
            return panel.spread_role != "none" or panel.emphasis in ("splash", "spread")

        top_turn_ratio = 0.0
        if n:
            k = max(1, int(math.ceil(0.1 * n)))
            top = sorted(range(n), key=lambda i: (-intensities[i], i))[:k]
            top_turn_ratio = sum(1 for i in top if at_turn_position(seq[i])) / k
        top_part = clamp(top_turn_ratio / 0.5, 0.0, 1.0)
        buried_peak = False
        if peak_index is not None:
            peak_metric = seq[peak_index]
            peak_panel = panel_lookup[peak_metric.panel_id]
            last_index = len(page_by_index[peak_metric.page_index].panels) - 1
            buried_peak = (
                peak_metric.area_ratio < 0.2
                and not at_turn_position(peak_metric)
                and 0 < peak_metric.index_in_page < last_index
                and peak_panel.emphasis == "normal"
            )
        if len(pages) < 2:
            turn_score = 75.0
            turn_summary = "単ページのためめくり効果は参考値です。"
        else:
            turn_score = clamp(100.0 * (0.35 * reveal_part + 0.25 * hook_part + 0.40 * top_part) - (25.0 if buried_peak else 0.0), 0.0, 100.0)
            turn_summary = (
                f"めくり直後の強度 {np.mean(reveal_values) if reveal_values else 0.0:.2f}・めくり前の引き {np.mean(hook_values) if hook_values else 0.0:.2f}"
                f"（全体平均 {mean_all:.2f}）・強度上位コマのめくり配置率 {top_turn_ratio:.0%}"
                + ("・山場がページ中段に埋没" if buried_peak else "")
                + "。"
            )
        turn_axis = AxisScore(
            axis=DiagnosticAxis.PAGE_TURN,
            label=AXIS_LABELS_JA[DiagnosticAxis.PAGE_TURN.value],
            score=round(turn_score, 2),
            weight=float(cfg.axis_weights.get(DiagnosticAxis.PAGE_TURN.value, 0.0)),
            metrics={
                "reveal_strength": round(float(np.mean(reveal_values)), 4) if reveal_values else -1.0,
                "hook_strength": round(float(np.mean(hook_values)), 4) if hook_values else -1.0,
                "mean_intensity": round(mean_all, 4),
                "top_turn_ratio": round(top_turn_ratio, 4),
                "buried_peak": 1.0 if buried_peak else 0.0,
            },
            summary=turn_summary,
        )

        axes = [gaze_axis, emotion_axis, density_axis, rhythm_axis, turn_axis]
        weight_total = sum(axis.weight for axis in axes)
        overall = sum(axis.score * axis.weight for axis in axes) / weight_total if weight_total > 0 else float(np.mean([a.score for a in axes]))

        dtw: float | None = None
        if has_targets and n >= 2:
            grid = np.linspace(0.0, 1.0, 21)
            positions = np.linspace(0.0, 1.0, n)
            dtw = round(float(dtw_distance(np.interp(grid, positions, intensities), np.interp(grid, positions, targets))), 4)
        curve = IntensityCurve(
            t=[m.t for m in seq],
            inferred=[m.intensity for m in seq],
            target=[float(m.target_tension) for m in seq] if has_targets else None,  # type: ignore[arg-type]
            correlation=round(correlation, 4) if correlation is not None else None,
            dtw_distance=dtw,
            panel_ids=[m.panel_id for m in seq],
        )
        return _Analysis(
            board=board,
            pages=pages,
            panel_metrics=panel_metrics,
            page_metrics=page_metrics,
            seq=seq,
            transitions=transitions,
            balloon_violations=balloon_violations,
            balloon_pairs=balloon_pairs,
            long_jumps=long_jumps,
            ambiguous_pairs=ambiguous_pairs,
            flat_runs=flat_runs,
            peaks_without_pause=peaks_without_pause,
            intended_peak=peak_index,
            has_targets=has_targets,
            correlation=correlation,
            intensity_range=intensity_range,
            axes=axes,
            overall=float(clamp(overall, 0.0, 100.0)),
            curve=curve,
            density_windows=density_windows,
            sparse_runs=sparse_runs,
            uniform_pages=uniform_pages,
            shot_runs=shot_runs,
            angle_runs=angle_runs,
            special_ratio=special_ratio,
            top_turn_ratio=top_turn_ratio,
            buried_peak=buried_peak,
        )

    @staticmethod
    def _runs(panels: list[Panel], attribute: str) -> list[list[str]]:
        runs: list[list[str]] = []
        current: list[Panel] = []
        for panel in panels + [None]:  # type: ignore[list-item]
            value = getattr(panel.camera, attribute).value if panel is not None else None
            if current and value == getattr(current[-1].camera, attribute).value:
                current.append(panel)
                continue
            if len(current) >= 3:
                runs.append([p.id for p in current])
            current = [panel] if panel is not None else []
        return runs

    # -- 内部: 所見 ---------------------------------------------------------------

    def _findings(self, an: _Analysis) -> list[DiagnosticFinding]:
        cfg = self.config
        board = an.board
        geometry = board.geometry
        panel_lookup = {panel.id: panel for page in an.pages for panel in page.panels}
        page_lookup = {page.index: page for page in an.pages}
        metric_by_id = {m.panel_id: m for m in an.panel_metrics}
        raw: list[dict[str, Any]] = []

        def centroid(panel_id: str) -> Point:
            return panel_lookup[panel_id].polygon.centroid()

        def add(
            code: str,
            severity: str,
            *,
            page_index: int | None,
            panel_ids: list[str],
            title: str,
            detail: str,
            metric: str,
            observed: float,
            target: float,
            anchor: Point | None,
            balloon_ids: list[str] | None = None,
        ) -> None:
            raw.append(
                {
                    "code": code,
                    "severity": severity,
                    "page_index": page_index,
                    "panel_ids": panel_ids,
                    "balloon_ids": balloon_ids or [],
                    "title": title,
                    "detail": detail,
                    "metric": metric,
                    "observed": round(float(observed), 4),
                    "target": round(float(target), 4),
                    "anchor": anchor,
                }
            )

        # 視線誘導
        for page_index, page_transitions in an.transitions.items():
            page = page_lookup[page_index]
            for transition in page_transitions:
                mid = Point(x=(transition.start.x + transition.end.x) / 2.0, y=(transition.start.y + transition.end.y) / 2.0)
                if transition.is_backtrack:
                    observed = transition.vector.x if transition.same_tier else -transition.vector.y
                    add(
                        "GF_BACKTRACK",
                        "major",
                        page_index=page_index,
                        panel_ids=[transition.from_panel, transition.to_panel],
                        title=f"{_loc(board, transition.from_panel)}→{position_label(panel_lookup[transition.to_panel], page, geometry)}のコマで視線が逆行",
                        detail=(
                            "右上→左下の Z 字に反して視線が"
                            + ("右へ戻ります" if transition.same_tier else "上の段へ戻ります")
                            + "。読者が読み順に迷い、没入が途切れます。"
                        ),
                        metric="backtrack_offset_px",
                        observed=observed,
                        target=0.0,
                        anchor=mid,
                    )
            seen: set[tuple[int, int]] = set()
            for i, transition in enumerate(page_transitions):
                for j in transition.crosses:
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    other = page_transitions[j]
                    point = geo.segment_intersection(
                        transition.start.as_tuple(), transition.end.as_tuple(), other.start.as_tuple(), other.end.as_tuple()
                    )
                    anchor = Point(x=point[0], y=point[1]) if point is not None else Point(x=transition.end.x, y=transition.end.y)
                    add(
                        "GF_CROSSING",
                        "minor",
                        page_index=page_index,
                        panel_ids=[transition.from_panel, transition.to_panel, other.from_panel, other.to_panel],
                        title=f"p{page.number} で視線経路が交差",
                        detail="2本の視線遷移が交差しており、どちらのコマを先に読むかが曖昧になります。",
                        metric="crossing_pairs",
                        observed=1.0,
                        target=0.0,
                        anchor=anchor,
                    )
        for page_index, first_id, second_id in an.ambiguous_pairs:
            page = page_lookup[page_index]
            ratio = _vertical_overlap_ratio(panel_lookup[first_id], panel_lookup[second_id])
            a = centroid(first_id)
            b = centroid(second_id)
            add(
                "GF_AMBIGUOUS_ORDER",
                "minor",
                page_index=page_index,
                panel_ids=[first_id, second_id],
                title=f"{_loc(board, first_id)}と{position_label(panel_lookup[second_id], page, geometry)}のコマの段が曖昧",
                detail=f"2コマの縦方向の重なりが {ratio:.0%} で、同じ段か別の段かを読者が判断しにくい配置です。",
                metric="vertical_overlap_ratio",
                observed=ratio,
                target=0.0 if ratio < 0.5 else 1.0,
                anchor=Point(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0),
            )
        for panel_id, first_balloon, second_balloon in an.balloon_violations:
            panel = panel_lookup[panel_id]
            balloon = next(b for b in panel.balloons if b.id == second_balloon)
            add(
                "GF_BALLOON_ORDER",
                "minor",
                page_index=panel.page_index,
                panel_ids=[panel_id],
                balloon_ids=[first_balloon, second_balloon],
                title=f"{_loc(board, panel_id)}のフキダシ読み順が逆",
                detail="後に読むべきフキダシが前のフキダシより右上にあり、台詞の順序が入れ替わって読まれます。",
                metric="balloon_order_violation",
                observed=1.0,
                target=0.0,
                anchor=Point(x=balloon.center.x, y=balloon.center.y),
            )
        diagonal = math.hypot(geometry.width, geometry.height)
        for page_index, transition in an.long_jumps:
            length = math.hypot(transition.vector.x, transition.vector.y)
            add(
                "GF_LONG_JUMP",
                "minor",
                page_index=page_index,
                panel_ids=[transition.from_panel, transition.to_panel],
                title=f"{_loc(board, transition.from_panel)}から次のコマへの視線移動が長すぎる",
                detail=f"視線の移動距離がページ対角線の {length / diagonal:.0%} に達し、読みのテンポが途切れます。",
                metric="jump_length_ratio",
                observed=length / diagonal,
                target=0.9,
                anchor=Point(x=(transition.start.x + transition.end.x) / 2.0, y=(transition.start.y + transition.end.y) / 2.0),
            )

        # 感情の緩急
        for start, end in an.flat_runs:
            ids = [an.seq[i].panel_id for i in range(start, end + 1)]
            middle = ids[len(ids) // 2]
            length = end - start + 1
            add(
                "ED_FLAT_SEQUENCE",
                "critical" if length >= 7 else "major",
                page_index=panel_lookup[middle].page_index,
                panel_ids=ids,
                title=f"{_loc(board, ids[0])}から {length} コマ、推定強度がほぼ一定",
                detail="コマの大きさ・ショット・ベタ量がほぼ同じで、感情の起伏が画面から伝わりません。",
                metric="flat_run_length",
                observed=float(length),
                target=3.0,
                anchor=centroid(middle),
            )
        if an.intended_peak is not None and an.seq:
            peak = an.seq[an.intended_peak]
            intensities = np.asarray([m.intensity for m in an.seq], dtype=float)
            threshold = float(np.percentile(intensities, 90))
            rank_pct = float(np.sum(intensities > peak.intensity)) / len(an.seq)
            if rank_pct > 0.25 or peak.area_ratio < 0.25:
                add(
                    "ED_WEAK_PEAK",
                    "critical" if rank_pct > 0.5 else "major",
                    page_index=peak.page_index,
                    panel_ids=[peak.panel_id],
                    title=f"山場の{_loc(board, peak.panel_id)}が視覚的に弱い",
                    detail=(
                        f"意図された最大緊張のコマの推定強度が {peak.intensity:.2f}（上位 {rank_pct:.0%}）、"
                        f"面積比 {peak.area_ratio:.2f} にとどまり、山場として目に飛び込みません。"
                    ),
                    metric="peak_intensity",
                    observed=peak.intensity,
                    target=max(0.75, threshold),
                    anchor=centroid(peak.panel_id),
                )
        for k in an.peaks_without_pause:
            metric = an.seq[k]
            add(
                "ED_NO_PAUSE",
                "minor",
                page_index=metric.page_index,
                panel_ids=[metric.panel_id],
                title=f"{_loc(board, metric.panel_id)}（山場）の直前に溜めがない",
                detail="山場の直前のコマも強度が高く、静から動への落差（溜め）が作られていません。",
                metric="pre_peak_drop",
                observed=float(metric.intensity - min(an.seq[j].intensity for j in (k - 1, k - 2) if j >= 0)),
                target=0.25,
                anchor=centroid(metric.panel_id),
            )
        if an.correlation is not None and an.correlation < 0.3:
            add(
                "ED_MISALIGNED_CURVE",
                "major",
                page_index=None,
                panel_ids=[],
                title="推定強度の曲線が目標の感情曲線と噛み合っていない",
                detail=f"コマの見た目の強さと物語上の緊張の相関が {an.correlation:.2f} と低く、盛り上がるべき箇所で画面が静かです。",
                metric="intensity_target_correlation",
                observed=an.correlation,
                target=0.5,
                anchor=None,
            )
        if len(an.seq) >= 4 and an.intensity_range < 0.3:
            add(
                "ED_LOW_RANGE",
                "major",
                page_index=None,
                panel_ids=[],
                title="作品全体で強度の幅が狭い",
                detail=f"推定強度の p90−p10 が {an.intensity_range:.2f} しかなく、大ゴマと小ゴマ・寄りと引きの差が乏しい構成です。",
                metric="intensity_range",
                observed=an.intensity_range,
                target=0.4,
                anchor=None,
            )

        # 情報密度
        for page in an.pages:
            for panel in page.ordered_panels():
                for balloon in panel.balloons:
                    if balloon.char_count > cfg.max_chars_per_balloon:
                        add(
                            "ID_OVERLOADED_BALLOON",
                            "major",
                            page_index=page.index,
                            panel_ids=[panel.id],
                            balloon_ids=[balloon.id],
                            title=f"{_loc(board, panel.id)}のフキダシが {balloon.char_count} 字",
                            detail=f"1フキダシ {cfg.max_chars_per_balloon} 字を超え、縦書きで読み切るのに負担がかかります。",
                            metric="chars_per_balloon",
                            observed=float(balloon.char_count),
                            target=float(cfg.max_chars_per_balloon),
                            anchor=Point(x=balloon.center.x, y=balloon.center.y),
                        )
        for pm in an.page_metrics:
            page = page_lookup[pm.page_index]
            if pm.density_level == "overloaded":
                add(
                    "ID_OVERLOADED_PAGE",
                    "critical" if pm.char_count > 1.5 * cfg.overloaded_chars_per_page else "major",
                    page_index=pm.page_index,
                    panel_ids=[p.id for p in page.ordered_panels()],
                    title=f"p{page.number} の台詞が {pm.char_count} 字で過密",
                    detail="1ページの文字量が多すぎ、絵を読む時間が奪われています。",
                    metric="chars_per_page",
                    observed=float(pm.char_count),
                    target=float(cfg.comfortable_chars_per_page[1]),
                    anchor=Point(x=geometry.width * 0.5, y=geometry.margin_top * 0.5),
                )
        for metric in an.panel_metrics:
            if metric.balloon_coverage > cfg.crowded_coverage:
                add(
                    "ID_CROWDED_PANEL",
                    "minor",
                    page_index=metric.page_index,
                    panel_ids=[metric.panel_id],
                    title=f"{_loc(board, metric.panel_id)}がフキダシで埋まっている",
                    detail=f"フキダシがコマ面積の {metric.balloon_coverage:.0%} を覆い、絵が見えません。",
                    metric="balloon_coverage",
                    observed=metric.balloon_coverage,
                    target=cfg.crowded_coverage,
                    anchor=centroid(metric.panel_id),
                )
        for page_index, cv in an.density_windows:
            page = page_lookup[page_index]
            add(
                "ID_FLAT_DENSITY",
                "minor",
                page_index=page_index,
                panel_ids=[],
                title=f"p{page.number} 前後3ページの情報密度が平板",
                detail=f"情報密度の変動係数が {cv:.2f} で、会話の密なページと絵で見せる疎なページの緩急がありません。",
                metric="density_cv",
                observed=cv,
                target=0.15,
                anchor=Point(x=geometry.width * 0.5, y=geometry.height - geometry.margin_bottom * 0.5),
            )
        for start_page, end_page in an.sparse_runs:
            page = page_lookup[start_page]
            add(
                "ID_SPARSE_RUN",
                "info",
                page_index=start_page,
                panel_ids=[],
                title=f"p{page.number}〜p{page_lookup[end_page].number} で台詞の少ないページが連続",
                detail="絵で見せるページが続きすぎると、状況の理解が追いつかなくなります。",
                metric="sparse_run_length",
                observed=float(end_page - start_page + 1),
                target=2.0,
                anchor=Point(x=geometry.width * 0.5, y=geometry.margin_top * 0.5),
            )

        # コマ割りリズム
        for page_index in an.uniform_pages:
            page = page_lookup[page_index]
            pm = next(p for p in an.page_metrics if p.page_index == page_index)
            add(
                "PR_UNIFORM_GRID",
                "major",
                page_index=page_index,
                panel_ids=[p.id for p in page.ordered_panels()],
                title=f"p{page.number} のコマが均一なグリッド",
                detail=f"{pm.panel_count} コマの面積の変動係数が {pm.area_cv:.2f} しかなく、どこを見せたいページかが伝わりません。",
                metric="area_cv",
                observed=pm.area_cv,
                target=0.35,
                anchor=Point(x=geometry.width * 0.5, y=geometry.height * 0.5),
            )
        for code, runs, label in (("PR_MONOTONOUS_SHOT", an.shot_runs, "ショット"), ("PR_MONOTONOUS_ANGLE", an.angle_runs, "アングル")):
            for run in runs:
                middle = run[len(run) // 2]
                panel = panel_lookup[middle]
                value = panel.camera.shot.value if code == "PR_MONOTONOUS_SHOT" else panel.camera.angle.value
                value_label = SHOT_LABELS.get(value, value) if code == "PR_MONOTONOUS_SHOT" else ANGLE_LABELS.get(value, value)
                add(
                    code,
                    "minor",
                    page_index=panel.page_index,
                    panel_ids=list(run),
                    title=f"{_loc(board, run[0])}から{label}「{value_label}」が {len(run)} コマ連続",
                    detail=f"同じ{label}が続き、画面が単調になっています。",
                    metric=f"{'shot' if code == 'PR_MONOTONOUS_SHOT' else 'angle'}_run_length",
                    observed=float(len(run)),
                    target=2.0,
                    anchor=centroid(middle),
                )
        for pm in an.page_metrics:
            if pm.panel_count > MAX_PANELS_PER_PAGE:
                page = page_lookup[pm.page_index]
                add(
                    "PR_PANEL_COUNT",
                    "minor",
                    page_index=pm.page_index,
                    panel_ids=[p.id for p in page.ordered_panels()],
                    title=f"p{page.number} のコマ数が {pm.panel_count}",
                    detail=f"1ページ {MAX_PANELS_PER_PAGE} コマを超え、1コマあたりの絵が小さくなりすぎています。",
                    metric="panels_per_page",
                    observed=float(pm.panel_count),
                    target=float(MAX_PANELS_PER_PAGE),
                    anchor=Point(x=geometry.width * 0.5, y=geometry.margin_top * 0.5),
                )
        if len(an.pages) >= 3 and (an.special_ratio < 0.2 or an.special_ratio > 0.8):
            overuse = an.special_ratio > 0.8
            add(
                "PR_NO_SPECIAL",
                "minor" if overuse else "info",
                page_index=None,
                panel_ids=[],
                title="大ゴマ・斜めゴマ・ぶち抜きの使用が" + ("多すぎる" if overuse else "少なすぎる"),
                detail=(
                    f"特殊ゴマを含むページの比率が {an.special_ratio:.0%} です。"
                    + ("強調が乱用され、本当の山場が埋もれます。" if overuse else "画面の見せ場が乏しく、ページをめくる動機が弱まります。")
                ),
                metric="special_page_ratio",
                observed=an.special_ratio,
                target=0.8 if overuse else 0.2,
                anchor=None,
            )

        # めくり効果
        if len(an.pages) >= 2:
            mean_all = float(np.mean([m.intensity for m in an.seq])) if an.seq else 0.0
            for pm in an.page_metrics:
                page = page_lookup[pm.page_index]
                ordered = page.ordered_panels()
                if pm.reveal_strength is not None and pm.reveal_strength < mean_all:
                    first = ordered[0]
                    add(
                        "PT_WEAK_REVEAL",
                        "major" if pm.reveal_strength < mean_all - 0.1 else "minor",
                        page_index=pm.page_index,
                        panel_ids=[p.id for p in ordered[:2]],
                        title=f"p{page.number}（めくり直後）の先頭コマが弱い",
                        detail=f"ページをめくった瞬間の強度が {pm.reveal_strength:.2f} で全体平均 {mean_all:.2f} を下回り、めくりの驚きがありません。",
                        metric="reveal_strength",
                        observed=pm.reveal_strength,
                        target=mean_all + 0.15,
                        anchor=centroid(first.id),
                    )
                if pm.hook_strength is not None and pm.hook_strength < pm.mean_intensity:
                    last = ordered[-1]
                    add(
                        "PT_WEAK_HOOK",
                        "minor",
                        page_index=pm.page_index,
                        panel_ids=[last.id],
                        title=f"p{page.number}（めくり前）の最終コマに引きがない",
                        detail=f"ページ末尾のコマの強度 {pm.hook_strength:.2f} がページ平均 {pm.mean_intensity:.2f} を下回り、次をめくらせる問いが残りません。",
                        metric="hook_strength",
                        observed=pm.hook_strength,
                        target=pm.mean_intensity + 0.1,
                        anchor=centroid(last.id),
                    )
            if an.buried_peak and an.intended_peak is not None:
                peak = an.seq[an.intended_peak]
                add(
                    "PT_PEAK_MID_PAGE",
                    "critical",
                    page_index=peak.page_index,
                    panel_ids=[peak.panel_id],
                    title=f"山場が{_loc(board, peak.panel_id)}（ページ中段の小ゴマ）に埋没",
                    detail="最大の山場がめくり直後でも見開きでもない中段の小さなコマにあり、読者が素通りします。",
                    metric="peak_area_ratio",
                    observed=peak.area_ratio,
                    target=0.4,
                    anchor=centroid(peak.panel_id),
                )

        raw.sort(key=lambda item: (SEVERITY_ORDER[item["severity"]], item["page_index"] if item["page_index"] is not None else -1, item["code"]))
        findings: list[DiagnosticFinding] = []
        for index, item in enumerate(raw, start=1):
            axis, _ = FINDING_CODES[item["code"]]
            findings.append(DiagnosticFinding(id=f"f{index:02d}", axis=axis, **item))
        return findings

    # -- 内部: 処方候補 -------------------------------------------------------------

    def _candidates(self, an: _Analysis, findings: list[DiagnosticFinding]) -> list[_Candidate]:
        board = an.board
        panel_lookup = {panel.id: panel for page in an.pages for panel in page.panels}
        page_lookup = {page.index: page for page in an.pages}
        metric_by_id = {m.panel_id: m for m in an.panel_metrics}
        candidates: list[_Candidate] = []

        def add(op: PrescriptionOperation, finding: DiagnosticFinding, *, page_index: int | None, panel_ids: list[str] | None = None, balloon_ids: list[str] | None = None, **params: Any) -> None:
            candidates.append(
                _Candidate(
                    operation=op,
                    finding_ids=[finding.id],
                    page_index=page_index,
                    panel_ids=list(panel_ids or []),
                    balloon_ids=list(balloon_ids or []),
                    params=params,
                    priority=SEVERITY_WEIGHT[finding.severity],
                    severity=finding.severity,
                )
            )

        def strongest_on_page(page: Page) -> Panel | None:
            panels = [p for p in page.ordered_panels() if p.spread_role == "none"]
            if not panels:
                return None

            def importance(panel: Panel) -> float:
                metric = metric_by_id.get(panel.id)
                if metric is not None and metric.target_tension is not None:
                    return metric.target_tension
                return panel.tension

            return max(panels, key=lambda p: (importance(p), -p.index_in_page))

        for finding in findings:
            code = finding.code
            if code in ("GF_BACKTRACK", "GF_CROSSING", "GF_AMBIGUOUS_ORDER", "GF_LONG_JUMP") and finding.page_index is not None:
                for pattern in (0, 1):
                    add(PrescriptionOperation.REORDER_PANELS, finding, page_index=finding.page_index, page=finding.page_index, pattern=pattern)
            elif code == "GF_BALLOON_ORDER" and finding.panel_ids:
                add(PrescriptionOperation.MOVE_BALLOON, finding, page_index=finding.page_index, panel_ids=finding.panel_ids[:1], panel_id=finding.panel_ids[0])
            elif code == "ED_FLAT_SEQUENCE" and finding.panel_ids:
                middle = finding.panel_ids[len(finding.panel_ids) // 2]
                page = page_lookup.get(panel_lookup[middle].page_index)
                if page is not None:
                    add(PrescriptionOperation.ADD_CONTRAST, finding, page_index=page.index, page=page.index)
                add(PrescriptionOperation.VARY_SHOT, finding, page_index=panel_lookup[middle].page_index, panel_ids=[middle], panel_id=middle, attribute="shot")
            elif code in ("ED_WEAK_PEAK", "PT_PEAK_MID_PAGE") and finding.panel_ids:
                peak_id = finding.panel_ids[0]
                peak_panel = panel_lookup[peak_id]
                page_index = peak_panel.page_index
                add(PrescriptionOperation.ENLARGE_PANEL, finding, page_index=page_index, panel_ids=[peak_id], panel_id=peak_id)
                add(PrescriptionOperation.CONVERT_TO_SPLASH, finding, page_index=page_index, panel_ids=[peak_id], panel_id=peak_id)
                add(PrescriptionOperation.MOVE_PEAK_TO_REVEAL, finding, page_index=page_index, panel_ids=[peak_id], panel_id=peak_id)
            elif code == "ED_NO_PAUSE" and finding.panel_ids:
                peak_id = finding.panel_ids[0]
                add(PrescriptionOperation.INSERT_PAUSE, finding, page_index=panel_lookup[peak_id].page_index, panel_ids=[peak_id], panel_id=peak_id)
            elif code in ("ED_MISALIGNED_CURVE", "ED_LOW_RANGE"):
                gaps = [
                    (m.target_tension - m.intensity if m.target_tension is not None else 0.0, m.panel_id)
                    for m in an.seq
                ]
                gaps.sort(key=lambda item: (-item[0], item[1]))
                for _, panel_id in gaps[:2]:
                    add(PrescriptionOperation.ENLARGE_PANEL, finding, page_index=panel_lookup[panel_id].page_index, panel_ids=[panel_id], panel_id=panel_id)
                if code == "ED_LOW_RANGE":
                    flattest = min(an.page_metrics, key=lambda pm: (pm.area_cv, pm.page_index))
                    add(PrescriptionOperation.ADD_CONTRAST, finding, page_index=flattest.page_index, page=flattest.page_index)
                    hottest = max(an.page_metrics, key=lambda pm: (pm.mean_intensity, -pm.page_index))
                    add(PrescriptionOperation.ADD_SLANT, finding, page_index=hottest.page_index, page=hottest.page_index, pattern=0)
            elif code == "ID_OVERLOADED_BALLOON" and finding.balloon_ids:
                add(
                    PrescriptionOperation.SPLIT_BALLOON,
                    finding,
                    page_index=finding.page_index,
                    panel_ids=finding.panel_ids[:1],
                    balloon_ids=finding.balloon_ids[:1],
                    panel_id=finding.panel_ids[0],
                    balloon_id=finding.balloon_ids[0],
                )
            elif code in ("ID_OVERLOADED_PAGE", "PR_PANEL_COUNT") and finding.page_index is not None:
                add(PrescriptionOperation.REBALANCE_DENSITY, finding, page_index=finding.page_index, page=finding.page_index, mode="overload")
            elif code == "ID_CROWDED_PANEL" and finding.panel_ids:
                panel = panel_lookup[finding.panel_ids[0]]
                longest = max(panel.balloons, key=lambda b: (b.char_count, b.id)) if panel.balloons else None
                if longest is not None and longest.char_count > max(8, self.config.max_chars_per_balloon // 2):
                    add(
                        PrescriptionOperation.SPLIT_BALLOON,
                        finding,
                        page_index=finding.page_index,
                        panel_ids=[panel.id],
                        balloon_ids=[longest.id],
                        panel_id=panel.id,
                        balloon_id=longest.id,
                        limit=max(8, self.config.max_chars_per_balloon // 2),
                    )
                add(PrescriptionOperation.MOVE_BALLOON, finding, page_index=finding.page_index, panel_ids=[panel.id], panel_id=panel.id)
            elif code == "ID_FLAT_DENSITY" and finding.page_index is not None:
                add(PrescriptionOperation.REBALANCE_DENSITY, finding, page_index=finding.page_index, page=finding.page_index, mode="contrast")
            elif code == "PR_UNIFORM_GRID" and finding.page_index is not None:
                add(PrescriptionOperation.ADD_CONTRAST, finding, page_index=finding.page_index, page=finding.page_index)
                add(PrescriptionOperation.ADD_SLANT, finding, page_index=finding.page_index, page=finding.page_index, pattern=0)
            elif code in ("PR_MONOTONOUS_SHOT", "PR_MONOTONOUS_ANGLE") and finding.panel_ids:
                middle = finding.panel_ids[len(finding.panel_ids) // 2]
                attribute = "shot" if code == "PR_MONOTONOUS_SHOT" else "angle"
                add(PrescriptionOperation.VARY_SHOT, finding, page_index=panel_lookup[middle].page_index, panel_ids=[middle], panel_id=middle, attribute=attribute)
            elif code == "PR_NO_SPECIAL":
                if an.special_ratio < 0.2:
                    hottest = max(an.page_metrics, key=lambda pm: (pm.mean_intensity, -pm.page_index))
                    add(PrescriptionOperation.ADD_SLANT, finding, page_index=hottest.page_index, page=hottest.page_index, pattern=0)
                    page = page_lookup[hottest.page_index]
                    strongest = strongest_on_page(page)
                    if strongest is not None:
                        add(PrescriptionOperation.ENLARGE_PANEL, finding, page_index=page.index, panel_ids=[strongest.id], panel_id=strongest.id)
            elif code == "PT_WEAK_REVEAL" and finding.page_index is not None:
                page = page_lookup[finding.page_index]
                first = page.ordered_panels()[0]
                add(PrescriptionOperation.ENLARGE_PANEL, finding, page_index=page.index, panel_ids=[first.id], panel_id=first.id)
                add(PrescriptionOperation.VARY_SHOT, finding, page_index=page.index, panel_ids=[first.id], panel_id=first.id, attribute="shot", direction="closer")
                previous = page_lookup.get(page.index - 1)
                if previous is not None:
                    movable = [p for p in previous.ordered_panels()[:-1] if p.spread_role == "none"]
                    if movable:
                        best = max(movable, key=lambda p: (metric_by_id[p.id].target_tension if metric_by_id[p.id].target_tension is not None else p.tension, -p.index_in_page))
                        add(PrescriptionOperation.MOVE_PEAK_TO_REVEAL, finding, page_index=previous.index, panel_ids=[best.id], panel_id=best.id)
            elif code == "PT_WEAK_HOOK" and finding.panel_ids:
                add(PrescriptionOperation.STRENGTHEN_HOOK, finding, page_index=finding.page_index, panel_ids=finding.panel_ids[:1], panel_id=finding.panel_ids[0])

        merged: dict[tuple[Any, ...], _Candidate] = {}
        for candidate in candidates:
            key = candidate.key()
            if key in merged:
                existing = merged[key]
                for finding_id in candidate.finding_ids:
                    if finding_id not in existing.finding_ids:
                        existing.finding_ids.append(finding_id)
                existing.priority += 0.25 * candidate.priority
            else:
                merged[key] = candidate
        ordered = sorted(merged.values(), key=lambda c: (-c.priority, c.key()))
        return ordered[: self.config.max_candidates]

    # -- 内部: 処方の生成（シミュレーション） ------------------------------------------

    def _prescriptions(self, board: Storyboard, an: _Analysis, findings: list[DiagnosticFinding], ctx: _Context) -> list[Prescription]:
        cfg = self.config
        if cfg.max_prescriptions <= 0:
            return []
        candidates = self._candidates(an, findings)
        base_axes = {axis.axis.value: axis.score for axis in an.axes}
        results: list[tuple[float, _Candidate, dict[str, float], Storyboard | None, bool]] = []
        if cfg.simulate_prescriptions:
            best_by_family: dict[tuple[Any, ...], tuple[float, _Candidate, dict[str, float], Storyboard]] = {}
            for candidate in candidates:
                trial = board.model_copy(deep=True)
                try:
                    applied = self._apply_operation(trial, candidate.operation, dict(candidate.params), ctx)
                    if not applied:
                        continue
                    trial_an = self._analyze(trial, ctx)
                except (ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
                    logger.debug("処方候補 %s のシミュレーションに失敗: %s", candidate.operation.value, exc)
                    continue
                gain = trial_an.overall - an.overall
                if gain <= 1e-6:
                    continue
                axis_gains = {axis.axis.value: round(axis.score - base_axes[axis.axis.value], 2) for axis in trial_an.axes}
                family = candidate.family()
                current = best_by_family.get(family)
                if current is None or gain > current[0] + 1e-9:
                    if current is not None:
                        for finding_id in current[1].finding_ids:
                            if finding_id not in candidate.finding_ids:
                                candidate.finding_ids.append(finding_id)
                    best_by_family[family] = (gain, candidate, axis_gains, trial)
                else:
                    for finding_id in candidate.finding_ids:
                        if finding_id not in current[1].finding_ids:
                            current[1].finding_ids.append(finding_id)
            results = [(gain, cand, axis_gains, trial, True) for gain, cand, axis_gains, trial in best_by_family.values()]
            results.sort(key=lambda item: (-item[0], item[1].key()))
        else:
            for candidate in candidates:
                results.append((HEURISTIC_GAIN[candidate.severity], candidate, {}, None, False))
            results.sort(key=lambda item: (-item[0], item[1].key()))
        prescriptions: list[Prescription] = []
        for rank, (gain, candidate, axis_gains, trial, simulated) in enumerate(results[: cfg.max_prescriptions], start=1):
            label = PRESCRIPTION_LABELS_JA[candidate.operation.value]
            location = self._location_text(board, candidate)
            prescriptions.append(
                Prescription(
                    id=f"rx{rank:02d}",
                    operation=candidate.operation,
                    axis=_OPERATION_AXIS[candidate.operation],
                    finding_ids=list(candidate.finding_ids),
                    page_index=candidate.page_index,
                    panel_ids=list(candidate.panel_ids),
                    balloon_ids=list(candidate.balloon_ids),
                    title=f"{location}: {label}",
                    instruction=self._instruction(board, trial, candidate),
                    expected_gain=round(float(gain), 2),
                    axis_gains=axis_gains,
                    simulated=simulated,
                    applicable=True,
                    params=dict(candidate.params),
                    rank=rank,
                )
            )
        return prescriptions

    @staticmethod
    def _location_text(board: Storyboard, candidate: _Candidate) -> str:
        if candidate.panel_ids:
            return _loc(board, candidate.panel_ids[0])
        page = _page_by_index(board, candidate.page_index)
        if page is not None:
            return f"p{page.number}"
        return "作品全体"

    def _instruction(self, board: Storyboard, trial: Storyboard | None, candidate: _Candidate) -> str:
        op = candidate.operation
        params = candidate.params
        panel_id = params.get("panel_id")
        loc = _loc(board, panel_id) if panel_id else self._location_text(board, candidate)

        def area_ratio(target: Storyboard | None, pid: str | None) -> float | None:
            if target is None or pid is None:
                return None
            panel, page = _locate(target, pid)
            if panel is None or page is None:
                return None
            union = max(1.0, _bbox_area(_union_bbox(page.panels)))
            return panel.polygon.area() / union

        if op == PrescriptionOperation.ENLARGE_PANEL:
            before = area_ratio(board, panel_id)
            after = area_ratio(trial, panel_id)
            change = f"（面積比 {before:.2f}→{after:.2f}）" if before is not None and after is not None else ""
            return f"{loc}を段幅いっぱいの大ゴマに拡大する{change}。同じ段の他のコマは縮めるか前後の段へ送る。"
        if op == PrescriptionOperation.CONVERT_TO_SPLASH:
            panel, page = _locate(board, panel_id) if panel_id else (None, None)
            others = len(page.panels) - 1 if page is not None else 0
            return f"{loc}を1ページ大ゴマ（四辺断ち切り）にし、同ページの他 {others} コマを前後のページへ送る。"
        if op == PrescriptionOperation.MOVE_PEAK_TO_REVEAL:
            panel, page = _locate(board, panel_id) if panel_id else (None, None)
            if page is not None and page.page_turn_role == PageTurnRole.REVEAL:
                before = [p for p in page.ordered_panels() if p.index_in_page < (panel.index_in_page if panel else 0)]
                return f"{loc}より前の {len(before)} コマを前ページ末尾へ送り、山場をめくり直後のページ先頭に置く。"
            nxt = _page_by_index(board, page.index + 1) if page is not None else None
            target_page = f"p{nxt.number}" if nxt is not None else "次ページ"
            return f"{loc}とそれ以降のコマを{target_page}（めくり直後の右ページ）の先頭へ送り、山場をめくりで見せる。"
        if op == PrescriptionOperation.STRENGTHEN_HOOK:
            panel, _ = _locate(board, panel_id) if panel_id else (None, None)
            if panel is not None:
                closer = SHOT_CLOSER[panel.camera.shot]
                return (
                    f"{loc}（めくり直前）のショットを{SHOT_LABELS[panel.camera.shot.value]}→{SHOT_LABELS[closer.value]}に寄せ、"
                    "アオリ気味に撮って、答えを言わずに問いを残すモノローグで終える。"
                )
            return f"{loc}をめくり前の引きとして強化する。"
        if op == PrescriptionOperation.INSERT_PAUSE:
            return f"{loc}の直前に台詞のない「間」のコマ（引きの画・アイレベル）を1枚挟み、静から動への落差を作る。"
        if op == PrescriptionOperation.VARY_SHOT:
            panel, _ = _locate(board, panel_id) if panel_id else (None, None)
            attribute = params.get("attribute", "shot")
            after_panel, _ = _locate(trial, panel_id) if (trial is not None and panel_id) else (None, None)
            if panel is not None and after_panel is not None:
                if attribute == "shot":
                    return f"{loc}のショットを{SHOT_LABELS[panel.camera.shot.value]}→{SHOT_LABELS[after_panel.camera.shot.value]}に変え、単調さを断つ。"
                return f"{loc}のアングルを{ANGLE_LABELS[panel.camera.angle.value]}→{ANGLE_LABELS[after_panel.camera.angle.value]}に変え、単調さを断つ。"
            return f"{loc}の{'ショット' if attribute == 'shot' else 'アングル'}を前後と変える。"
        if op == PrescriptionOperation.SPLIT_BALLOON:
            panel, _ = _locate(board, panel_id) if panel_id else (None, None)
            balloon = next((b for b in panel.balloons if b.id == params.get("balloon_id")), None) if panel is not None else None
            limit = int(params.get("limit", self.config.max_chars_per_balloon))
            if balloon is not None:
                pieces = split_text_for_balloons(balloon.text, limit)
                return f"{loc}のフキダシ（{balloon.char_count}字）を句読点・文節で {len(pieces)} 個に分割し、各 {limit} 字以内に収める。"
            return f"{loc}の長いフキダシを分割する。"
        if op == PrescriptionOperation.MOVE_BALLOON:
            return f"{loc}のフキダシを右上→左下の読み順に並べ直し、顔に掛からない位置へ移す。"
        if op == PrescriptionOperation.REORDER_PANELS:
            page = _page_by_index(board, params.get("page"))
            before = sum(1 for t in compute_gaze_transitions(page) if t.is_backtrack) if page is not None and len(page.panels) > 1 else 0
            after_page = _page_by_index(trial, params.get("page")) if trial is not None else None
            after = sum(1 for t in compute_gaze_transitions(after_page) if t.is_backtrack) if after_page is not None and len(after_page.panels) > 1 else 0
            number = page.number if page is not None else "?"
            return f"p{number} のコマを段単位で組み直し、右上→左下の Z 字で読めるようにする（逆行 {before}→{after}）。"
        if op == PrescriptionOperation.REBALANCE_DENSITY:
            page = _page_by_index(board, params.get("page"))
            after_page = _page_by_index(trial, params.get("page")) if trial is not None else None
            number = page.number if page is not None else "?"
            if page is not None and after_page is not None:
                return (
                    f"p{number} の端のコマを隣のページへ送り、情報密度を {page_information_density(page):.1f}→"
                    f"{page_information_density(after_page):.1f} に調整して前後との緩急をつける。"
                )
            return f"p{number} の情報密度を隣接ページへ分散する。"
        if op == PrescriptionOperation.ADD_CONTRAST:
            page = _page_by_index(board, params.get("page"))
            after_page = _page_by_index(trial, params.get("page")) if trial is not None else None

            def cv(target_page: Page | None) -> float:
                if target_page is None or len(target_page.panels) < 2:
                    return 0.0
                areas = np.asarray([p.polygon.area() for p in target_page.panels], dtype=float)
                return float(np.std(areas) / np.mean(areas)) if float(np.mean(areas)) > 0 else 0.0

            number = page.number if page is not None else "?"
            return f"p{number} で最も重要なコマを大ゴマにし、コマ面積に大小の差をつける（変動係数 {cv(page):.2f}→{cv(after_page):.2f}）。"
        if op == PrescriptionOperation.ADD_SLANT:
            page = _page_by_index(board, params.get("page"))
            after_page = _page_by_index(trial, params.get("page")) if trial is not None else None
            slant = max((_slant_deg(p.polygon.tuples()) for p in after_page.panels), default=0.0) if after_page is not None else 0.0
            number = page.number if page is not None else "?"
            return f"p{number} の縦ガターを斜めに切り、動勢を加える（最大傾き {slant:.0f}°）。"
        return f"{loc}を修正する。"

    # -- 内部: 修正演算子 ---------------------------------------------------------------

    def _relayout(self, board: Storyboard, page: Page, *, allow_slant: bool | None = None, pattern: int = 0) -> None:
        """ページを再配置し、フキダシ・描き文字をコマの新しい位置へ写像する。"""
        old_boxes = {panel.id: panel.polygon.core_box() for panel in page.panels}
        if allow_slant is None:
            allow_slant = any(panel.shape == PanelShape.SLANTED for panel in page.panels)
        layout_page(page, board, allow_slant=allow_slant, pattern=pattern)
        for panel in page.panels:
            old = old_boxes.get(panel.id)
            new = panel.polygon.core_box()
            if old is None:
                layout_balloons(panel, self._story_config)
                continue
            sx = (new[2] - new[0]) / max(1e-6, old[2] - old[0])
            sy = (new[3] - new[1]) / max(1e-6, old[3] - old[1])
            for balloon in panel.balloons:
                balloon.center = _remap_point(balloon.center, old, new)
                balloon.width = round(max(20.0, min(balloon.width * sx, (new[2] - new[0]) * 0.95)), 2)
                balloon.height = round(max(20.0, min(balloon.height * sy, (new[3] - new[1]) * 0.95)), 2)
                if balloon.tail_target is not None:
                    balloon.tail_target = _remap_point(balloon.tail_target, old, new)
            for effect in panel.sfx:
                effect.position = _remap_point(effect.position, old, new)
        refresh_page_metrics(page)

    def _finish(self, board: Storyboard) -> None:
        finalize_reading_order(board)
        for page in board.pages:
            refresh_page_metrics(page)

    def _apply_operation(self, board: Storyboard, operation: PrescriptionOperation, params: dict[str, Any], ctx: _Context) -> bool:
        handlers = {
            PrescriptionOperation.ENLARGE_PANEL: self._op_enlarge_panel,
            PrescriptionOperation.CONVERT_TO_SPLASH: self._op_convert_to_splash,
            PrescriptionOperation.MOVE_PEAK_TO_REVEAL: self._op_move_peak_to_reveal,
            PrescriptionOperation.STRENGTHEN_HOOK: self._op_strengthen_hook,
            PrescriptionOperation.INSERT_PAUSE: self._op_insert_pause,
            PrescriptionOperation.VARY_SHOT: self._op_vary_shot,
            PrescriptionOperation.SPLIT_BALLOON: self._op_split_balloon,
            PrescriptionOperation.MOVE_BALLOON: self._op_move_balloon,
            PrescriptionOperation.REORDER_PANELS: self._op_reorder_panels,
            PrescriptionOperation.REBALANCE_DENSITY: self._op_rebalance_density,
            PrescriptionOperation.ADD_CONTRAST: self._op_add_contrast,
            PrescriptionOperation.ADD_SLANT: self._op_add_slant,
        }
        applied = handlers[operation](board, params, ctx)
        if applied:
            self._finish(board)
        return applied

    def _op_enlarge_panel(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        panel, page = _locate(board, str(params.get("panel_id", "")))
        if panel is None or page is None or panel.spread_role != "none" or panel.emphasis != "normal":
            return False
        if _in_spread(board, page):
            return False
        large = [p for p in page.panels if p.emphasis == "large" and p.id != panel.id]
        while len(large) >= 2:
            weakest = min(large, key=lambda p: (p.tension, p.id))
            weakest.emphasis = "normal"
            large = [p for p in large if p.id != weakest.id]
        panel.emphasis = "large"
        self._relayout(board, page)
        return True

    def _move_to_previous_end(self, board: Storyboard, source: Page, panels: list[Panel]) -> Page | None:
        previous = _page_by_index(board, source.index - 1)
        if previous is None or _in_spread(board, previous) or len(previous.panels) + len(panels) > MAX_PANELS_PER_PAGE:
            return None
        ids = {p.id for p in panels}
        _set_order(previous, previous.ordered_panels() + panels)
        _set_order(source, [p for p in source.ordered_panels() if p.id not in ids])
        return previous

    def _move_to_next_start(self, board: Storyboard, source: Page, panels: list[Panel]) -> Page | None:
        nxt = _page_by_index(board, source.index + 1)
        if nxt is None or _in_spread(board, nxt) or len(nxt.panels) + len(panels) > MAX_PANELS_PER_PAGE:
            return None
        ids = {p.id for p in panels}
        _set_order(nxt, panels + nxt.ordered_panels())
        _set_order(source, [p for p in source.ordered_panels() if p.id not in ids])
        return nxt

    def _op_convert_to_splash(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        panel, page = _locate(board, str(params.get("panel_id", "")))
        if panel is None or page is None or panel.spread_role != "none" or _in_spread(board, page):
            return False
        ordered = page.ordered_panels()
        position = next(i for i, p in enumerate(ordered) if p.id == panel.id)
        before = ordered[:position]
        after = ordered[position + 1 :]
        previous = _page_by_index(board, page.index - 1)
        nxt = _page_by_index(board, page.index + 1)
        if before and (previous is None or _in_spread(board, previous) or len(previous.panels) + len(before) > MAX_PANELS_PER_PAGE):
            return False
        if after and (nxt is None or _in_spread(board, nxt) or len(nxt.panels) + len(after) > MAX_PANELS_PER_PAGE):
            return False
        if not before and not after and panel.emphasis == "splash":
            return False
        touched: list[Page] = [page]
        if before:
            moved = self._move_to_previous_end(board, page, before)
            if moved is None:
                return False
            touched.append(moved)
        if after:
            moved = self._move_to_next_start(board, page, after)
            if moved is None:
                return False
            touched.append(moved)
        panel.emphasis = "splash"
        for target in touched:
            self._relayout(board, target)
        return True

    def _op_move_peak_to_reveal(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        panel, page = _locate(board, str(params.get("panel_id", "")))
        if panel is None or page is None or panel.spread_role != "none" or _in_spread(board, page):
            return False
        ordered = page.ordered_panels()
        position = next(i for i, p in enumerate(ordered) if p.id == panel.id)
        if page.page_turn_role == PageTurnRole.REVEAL:
            if position == 0:
                return False
            moved = self._move_to_previous_end(board, page, ordered[:position])
            if moved is None:
                return False
            self._relayout(board, moved)
            self._relayout(board, page)
            return True
        nxt = _page_by_index(board, page.index + 1)
        if nxt is None or nxt.page_turn_role != PageTurnRole.REVEAL or position == 0:
            return False
        moved = self._move_to_next_start(board, page, ordered[position:])
        if moved is None:
            return False
        self._relayout(board, page)
        self._relayout(board, moved)
        return True

    def _new_balloon(self, panel: Panel, kind: BalloonKind, text: str, speaker_id: str | None) -> Balloon:
        x0, y0, x1, y1 = panel.polygon.core_box()
        index = len(panel.balloons)
        balloon_id = f"{panel.id}_b{index}"
        existing = {b.id for b in panel.balloons}
        while balloon_id in existing:
            index += 1
            balloon_id = f"{panel.id}_b{index}"
        return Balloon(
            id=balloon_id,
            kind=kind,
            speaker_id=speaker_id,
            text=text,
            center=Point(x=round((x0 + x1) / 2.0, 2), y=round((y0 + y1) / 2.0, 2)),
            width=80.0,
            height=80.0,
            reading_order=index,
        )

    def _op_strengthen_hook(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        panel, page = _locate(board, str(params.get("panel_id", "")))
        if panel is None or page is None or panel.spread_role == "continuation":
            return False
        changed = False
        closer = SHOT_CLOSER[panel.camera.shot]
        update: dict[str, Any] = {}
        if closer != panel.camera.shot:
            update["shot"] = closer
            update["lens_mm"] = SHOT_LENS[closer]
        if panel.camera.angle == CameraAngle.EYE_LEVEL:
            update["angle"] = CameraAngle.LOW
        if update:
            panel.camera = panel.camera.model_copy(update=update)
            scale = SHOT_CHARACTER_SCALE[panel.camera.shot]
            for character in panel.characters:
                character.scale = scale
            changed = True
        if not panel.balloons:
            rng = random.Random(stable_seed("caf.doctor.hook", panel.id))
            lines = template_lines(None, max(0.6, panel.tension), rng=rng, theme_word=_theme_word(board), hook=True)
            text = lines[0][1] if lines else "まさか……"
            speaker = panel.characters[0].character_id if panel.characters else None
            panel.balloons.append(self._new_balloon(panel, BalloonKind.MONOLOGUE, text, speaker))
            panel.is_silent = False
            layout_balloons(panel, self._story_config)
            changed = True
        if "page_turn_hook" not in panel.prompt_hints:
            panel.prompt_hints.append("page_turn_hook")
        return changed

    def _op_insert_pause(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        peak, page = _locate(board, str(params.get("panel_id", "")))
        if peak is None or page is None or peak.spread_role != "none" or _in_spread(board, page):
            return False
        if len(page.panels) >= MAX_PANELS_PER_PAGE:
            return False
        ordered = page.ordered_panels()
        position = next(i for i, p in enumerate(ordered) if p.id == peak.id)
        previous = ordered[position - 1] if position > 0 else None
        pause_id = f"{peak.id}_pause"
        while board.panel(pause_id) is not None:
            pause_id += "x"
        shot = ShotSize.ELS if peak.camera.shot != ShotSize.ELS else ShotSize.FS
        angle = CameraAngle.EYE_LEVEL if peak.camera.angle != CameraAngle.EYE_LEVEL else CameraAngle.HIGH
        base_tension = previous.tension if previous is not None else peak.tension
        pause = Panel(
            id=pause_id,
            page_index=page.index,
            index_in_page=position,
            polygon=peak.polygon.model_copy(deep=True),
            shape=PanelShape.RECT,
            camera=CameraWork(
                shot=shot,
                angle=angle,
                lens_mm=SHOT_LENS[shot],
                composition=CompositionRule.NEGATIVE_SPACE,
            ),
            characters=[c.model_copy(update={"scale": SHOT_CHARACTER_SCALE[shot]}) for c in peak.characters[:1]],
            balloons=[],
            sfx=[],
            description="間（無音）: 山場の直前に呼吸を置く",
            t=round(max(0.0, peak.t - 0.001), 6),
            tension=round(float(clamp(min(base_tension, peak.tension) - 0.3, 0.0, 1.0)), 4),
            emphasis="normal",
            is_silent=True,
        )
        _set_order(page, ordered[:position] + [pause] + ordered[position:])
        self._relayout(board, page)
        return True

    def _op_vary_shot(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        panel, page = _locate(board, str(params.get("panel_id", "")))
        if panel is None or page is None or panel.spread_role == "continuation":
            return False
        attribute = str(params.get("attribute", "shot"))
        sequence = [p for p in board.panels_in_reading_order() if p.spread_role != "continuation"]
        position = next(i for i, p in enumerate(sequence) if p.id == panel.id)
        neighbors = [sequence[j] for j in (position - 1, position + 1) if 0 <= j < len(sequence)]
        if attribute == "shot":
            current = panel.camera.shot
            closer_first = params.get("direction") == "closer" or panel.tension >= 0.6
            pool = [ShotSize.CU, ShotSize.ECU, ShotSize.MS, ShotSize.FS, ShotSize.ELS] if closer_first else [ShotSize.FS, ShotSize.ELS, ShotSize.MS, ShotSize.CU, ShotSize.ECU]
            if params.get("direction") == "closer":
                order = [ShotSize.ELS, ShotSize.FS, ShotSize.MS, ShotSize.CU, ShotSize.ECU]
                pool = [s for s in pool if order.index(s) > order.index(current)] or pool
            used = {n.camera.shot for n in neighbors} | {current}
            choice = next((s for s in pool if s not in used), None)
            if choice is None:
                return False
            panel.camera = panel.camera.model_copy(update={"shot": choice, "lens_mm": SHOT_LENS[choice]})
            for character in panel.characters:
                character.scale = SHOT_CHARACTER_SCALE[choice]
            return True
        current_angle = panel.camera.angle
        pool_angles = (
            [CameraAngle.LOW, CameraAngle.DUTCH, CameraAngle.HIGH, CameraAngle.EYE_LEVEL]
            if panel.tension >= 0.6
            else [CameraAngle.HIGH, CameraAngle.LOW, CameraAngle.EYE_LEVEL, CameraAngle.DUTCH]
        )
        used_angles = {n.camera.angle for n in neighbors} | {current_angle}
        new_angle = next((a for a in pool_angles if a not in used_angles), None)
        if new_angle is None:
            return False
        dutch = round(10.0 + 20.0 * panel.tension, 1) if new_angle == CameraAngle.DUTCH else 0.0
        panel.camera = panel.camera.model_copy(update={"angle": new_angle, "dutch_degrees": dutch})
        return True

    def _op_split_balloon(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        panel, page = _locate(board, str(params.get("panel_id", "")))
        if panel is None or page is None:
            return False
        balloon_id = params.get("balloon_id")
        index = next((i for i, b in enumerate(panel.balloons) if b.id == balloon_id), None)
        if index is None:
            return False
        balloon = panel.balloons[index]
        limit = int(params.get("limit", self.config.max_chars_per_balloon))
        pieces = split_text_for_balloons(balloon.text, limit)
        if len(pieces) < 2:
            return False
        replacement = [balloon.model_copy(update={"text": piece, "columns": []}) for piece in pieces]
        rebuilt = panel.balloons[:index] + replacement + panel.balloons[index + 1 :]
        cap = self._story_config.max_balloons_per_panel
        overflow: list[Balloon] = []
        if len(rebuilt) > cap:
            ordered_panels = page.ordered_panels()
            position = next(i for i, p in enumerate(ordered_panels) if p.id == panel.id)
            receiver = ordered_panels[position + 1] if position + 1 < len(ordered_panels) else None
            if receiver is not None and receiver.spread_role == "none" and len(receiver.balloons) + (len(rebuilt) - cap) <= cap:
                overflow = rebuilt[cap:]
                rebuilt = rebuilt[:cap]
                for item in overflow:
                    receiver.balloons.append(item.model_copy(update={"id": f"{receiver.id}_m{len(receiver.balloons)}"}))
                for i, item in enumerate(receiver.balloons):
                    item.id = f"{receiver.id}_b{i}"
                receiver.is_silent = False
                layout_balloons(receiver, self._story_config)
        panel.balloons = rebuilt
        for i, item in enumerate(panel.balloons):
            item.id = f"{panel.id}_b{i}"
        layout_balloons(panel, self._story_config)
        return True

    def _op_move_balloon(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        panel, _ = _locate(board, str(params.get("panel_id", "")))
        if panel is None or not panel.balloons:
            return False
        before = [(b.id, b.center.x, b.center.y, b.width, b.height) for b in panel.balloons]
        layout_balloons(panel, self._story_config)
        after = [(b.id, b.center.x, b.center.y, b.width, b.height) for b in panel.balloons]
        return before != after

    def _op_reorder_panels(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        page = _page_by_index(board, params.get("page"))
        if page is None or len(page.panels) < 2 or _in_spread(board, page):
            return False
        self._relayout(board, page, allow_slant=False, pattern=int(params.get("pattern", 0)))
        return True

    def _op_rebalance_density(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        page = _page_by_index(board, params.get("page"))
        if page is None or len(page.panels) < 2 or _in_spread(board, page):
            return False
        mode = str(params.get("mode", "overload"))
        ordered = page.ordered_panels()
        options: list[tuple[float, str, Page]] = []
        previous = _page_by_index(board, page.index - 1)
        nxt = _page_by_index(board, page.index + 1)
        if previous is not None and not _in_spread(board, previous) and len(previous.panels) < MAX_PANELS_PER_PAGE:
            options.append((page_information_density(previous), "previous", previous))
        if nxt is not None and not _in_spread(board, nxt) and len(nxt.panels) < MAX_PANELS_PER_PAGE:
            options.append((page_information_density(nxt), "next", nxt))
        if not options:
            return False
        if mode == "contrast":
            _, direction, _ = max(options, key=lambda item: (item[0], item[1]))
        else:
            _, direction, _ = min(options, key=lambda item: (item[0], item[1]))
        if direction == "previous":
            moved = self._move_to_previous_end(board, page, [ordered[0]])
        else:
            moved = self._move_to_next_start(board, page, [ordered[-1]])
        if moved is None:
            return False
        self._relayout(board, page)
        self._relayout(board, moved)
        return True

    def _op_add_contrast(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        page = _page_by_index(board, params.get("page"))
        if page is None or len(page.panels) < 2 or _in_spread(board, page):
            return False
        candidates = [p for p in page.ordered_panels() if p.emphasis == "normal" and p.spread_role == "none"]
        if not candidates or sum(1 for p in page.panels if p.emphasis == "large") >= 2:
            return False

        def importance(panel: Panel) -> float:
            if ctx.target_wave is not None:
                return ctx.target_wave.sample("tension", panel.t)
            hint = ctx.aux.hint.get(panel.id)
            return hint if hint is not None else panel.tension

        chosen = max(candidates, key=lambda p: (importance(p), -p.index_in_page))
        chosen.emphasis = "large"
        self._relayout(board, page)
        return True

    def _op_add_slant(self, board: Storyboard, params: dict[str, Any], ctx: _Context) -> bool:
        page = _page_by_index(board, params.get("page"))
        if page is None or len(page.panels) < 2 or _in_spread(board, page):
            return False
        if any(panel.shape == PanelShape.SLANTED for panel in page.panels):
            return False
        page.tension_mean = max(page.tension_mean, 0.72)
        self._relayout(board, page, allow_slant=True, pattern=int(params.get("pattern", 0)))
        return any(panel.shape == PanelShape.SLANTED for panel in page.panels)

    # -- 内部: レポートと講評 -------------------------------------------------------------

    def _build_report(self, board: Storyboard, ctx: _Context, *, source_id: str | None, title: str) -> DiagnosticReport:
        analysis = self._analyze(board, ctx)
        findings = self._findings(analysis)
        prescriptions = self._prescriptions(board, analysis, findings, ctx)
        overall = round(analysis.overall, 2)
        report = DiagnosticReport(
            title=title[:200] if title else "ネーム診断",
            source_type=ctx.source_type,
            source_id=source_id,
            target_structure_id=ctx.structure.id if ctx.structure is not None else None,
            overall_score=overall,
            grade=grade_for(overall),
            axes=analysis.axes,
            pages=analysis.page_metrics,
            panels=analysis.panel_metrics,
            findings=findings,
            prescriptions=prescriptions,
            intensity_curve=analysis.curve,
            storyboard=board.model_copy(deep=True),
        )
        report.commentary = render_commentary(self.baseline_commentary(report))
        return report

    def baseline_commentary(self, report: DiagnosticReport) -> DiagnosisCommentary:
        """決定論ベースラインの編集者コメント。"""
        axes = sorted(report.axes, key=lambda a: (-a.score, a.axis.value))
        best = axes[0]
        worst = axes[-1]
        strengths = [f"{axis.label}（{axis.score:.0f}点）: {axis.summary}" for axis in axes if axis.score >= 75.0][:3]
        if not strengths:
            strengths = [f"{best.label}（{best.score:.0f}点）が相対的に安定しています。"]
        priorities = [
            f"{rx.title}（期待 +{rx.expected_gain:.1f}点）: {rx.instruction}" for rx in report.top_prescriptions(5)
        ]
        if not priorities:
            serious = sorted(report.findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.id))[:3]
            priorities = [f"{f.title}: {f.detail}" for f in serious]
        if not priorities:
            priorities = ["現状の構成を維持しつつ、山場の直前にもう一拍の「間」を置けるかを検討してください。"]
        critical = [f for f in report.findings if f.severity == "critical"]
        top = report.top_prescriptions(1)
        note_parts = [
            f"総合 {report.overall_score:.1f} 点（{report.grade}）。{best.label}が最も強く、{worst.label}（{worst.score:.0f}点）に最大の伸びしろがあります。"
        ]
        if critical:
            note_parts.append(f"まず「{critical[0].title}」を解消しないと、他の工夫が読者に届きません。")
        if top:
            note_parts.append(
                f"一手だけ直すなら「{top[0].title}」です。シミュレーション上、これだけで総合点が {top[0].expected_gain:.1f} 点上がります。"
            )
        if report.intensity_curve.correlation is not None:
            corr = report.intensity_curve.correlation
            note_parts.append(
                "画面の強弱は物語の緊張とよく連動しています。"
                if corr >= 0.5
                else f"画面の強弱と物語の緊張の相関が {corr:.2f} と弱く、盛り上げたい箇所ほどコマを大きく・寄りで見せる余地があります。"
            )
        note_parts.append("直したら必ずもう一度通して読み、めくりの瞬間に驚きが残っているかを確かめてください。")
        headline = f"総合 {report.overall_score:.1f} 点（{report.grade}）— {best.label}が強み、{worst.label}に伸びしろ"
        return DiagnosisCommentary(
            headline=headline[:200],
            strengths=strengths[:8],
            priorities=priorities[:8],
            editor_note="".join(note_parts)[:2000],
        )

    async def _amplify(self, report: DiagnosticReport, use_llm: bool | None) -> DiagnosticReport:
        wants = self.config.use_llm_commentary if use_llm is None else use_llm
        if not wants or self.llm_client is None:
            return report
        from core.llm_client import HarnessError, MultiStageReasoningHarness, StructuredOutputError
        from core.manga_pipeline.prompts import DIAGNOSIS_COMMENTARY_SYSTEM_PROMPT, DIAGNOSIS_RUBRIC

        baseline = self.baseline_commentary(report)
        payload = {
            "title": report.title,
            "overall_score": report.overall_score,
            "grade": report.grade,
            "axes": [{"axis": a.axis.value, "label": a.label, "score": a.score, "summary": a.summary} for a in report.axes],
            "findings": [
                {"id": f.id, "severity": f.severity, "code": f.code, "title": f.title, "detail": f.detail}
                for f in sorted(report.findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.id))[:12]
            ],
            "prescriptions": [
                {"id": rx.id, "title": rx.title, "instruction": rx.instruction, "expected_gain": rx.expected_gain}
                for rx in report.top_prescriptions(8)
            ],
            "intensity_correlation": report.intensity_curve.correlation,
        }
        harness = MultiStageReasoningHarness(self.llm_client)
        try:
            result = await harness.run(
                task="manga.diagnosis_commentary",
                objective="ネーム診断の定量結果から、優先順位の明確な編集者の講評を書く",
                payload=payload,
                output_model=DiagnosisCommentary,
                system_prompt=DIAGNOSIS_COMMENTARY_SYSTEM_PROMPT,
                rubric=DIAGNOSIS_RUBRIC,
                context={"baseline": baseline.model_dump(mode="json")},
            )
        except (HarnessError, StructuredOutputError) as exc:
            logger.warning("編集者コメントの LLM 増幅に失敗したためベースラインを維持します: %s", exc)
            return report
        report.commentary = render_commentary(result.output)
        report.used_llm = True
        return report


def render_commentary(commentary: DiagnosisCommentary) -> str:
    """編集者コメントを表示用のテキストへ整形する。"""
    lines = [commentary.headline, "", "■ 強み"]
    lines.extend(f"- {item}" for item in commentary.strengths)
    lines.extend(["", "■ 優先課題"])
    lines.extend(f"{i}. {item}" for i, item in enumerate(commentary.priorities, start=1))
    lines.extend(["", "■ 編集者より", commentary.editor_note])
    return "\n".join(lines)


__all__.append("render_commentary")
