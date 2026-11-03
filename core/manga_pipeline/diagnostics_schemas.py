"""ネーム構造ドクター（Storyboard Diagnostics Engine）の共有スキーマ。

既存の漫画ページ／ネームを次の3経路のいずれかで受け取り、コマ割り・セリフ配置・情報密度を逆解析する。

1. ``Storyboard``  : 本システムで生成・保存したネーム（またはそれを編集したもの）
2. ``PageAnnotation`` : 既存原稿のコマ枠・フキダシ・台詞を手入力／半自動で記述した JSON
3. ページ画像      : PNG（ブラウザで任意形式を PNG 化して送信）から ``page_vision`` が自動検出

解析結果は5軸（視線誘導・感情の緩急・情報密度・コマ割りリズム・めくり効果）で採点され、
所見（Finding）ごとに「どこを・どう直せば・総合スコアが何点上がるか」を反実仮想シミュレーション
（処方を実際に適用して再採点）で定量化した処方箋（Prescription）を、期待改善量の大きい順に返す。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import new_id, utc_now_iso
from core.manga_pipeline.schemas import BalloonKind, CameraAngle, PageSide, Point, ShotSize, Storyboard

__all__ = [
    "BBox",
    "DiagnosticAxis",
    "AXIS_LABELS_JA",
    "DEFAULT_AXIS_WEIGHTS",
    "DiagnosticSourceType",
    "Severity",
    "SEVERITY_ORDER",
    "SEVERITY_LABELS_JA",
    "PrescriptionOperation",
    "PRESCRIPTION_LABELS_JA",
    "AnnotatedBalloon",
    "AnnotatedPanel",
    "AnnotatedPage",
    "PageAnnotation",
    "DetectedBalloon",
    "DetectedPanel",
    "PageDetection",
    "ImageDetectionResult",
    "PanelMetrics",
    "PageMetrics",
    "AxisScore",
    "DiagnosticFinding",
    "Prescription",
    "IntensityCurve",
    "DiagnosisCommentary",
    "DiagnosticReport",
    "grade_for",
]

BBox = tuple[float, float, float, float]
"""(x0, y0, x1, y1) のページ座標矩形（x0<x1, y0<y1）。"""


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class DiagnosticAxis(str, Enum):
    """診断の5軸。"""

    GAZE_FLOW = "gaze_flow"
    EMOTIONAL_DYNAMICS = "emotional_dynamics"
    INFORMATION_DENSITY = "information_density"
    PANEL_RHYTHM = "panel_rhythm"
    PAGE_TURN = "page_turn"


AXIS_LABELS_JA: dict[str, str] = {
    DiagnosticAxis.GAZE_FLOW.value: "視線誘導",
    DiagnosticAxis.EMOTIONAL_DYNAMICS.value: "感情の緩急",
    DiagnosticAxis.INFORMATION_DENSITY.value: "情報密度",
    DiagnosticAxis.PANEL_RHYTHM.value: "コマ割りリズム",
    DiagnosticAxis.PAGE_TURN.value: "めくり効果",
}

DEFAULT_AXIS_WEIGHTS: dict[str, float] = {
    DiagnosticAxis.GAZE_FLOW.value: 0.25,
    DiagnosticAxis.EMOTIONAL_DYNAMICS.value: 0.25,
    DiagnosticAxis.INFORMATION_DENSITY.value: 0.15,
    DiagnosticAxis.PANEL_RHYTHM.value: 0.20,
    DiagnosticAxis.PAGE_TURN.value: 0.15,
}


class DiagnosticSourceType(str, Enum):
    STORYBOARD = "storyboard"
    ANNOTATION = "annotation"
    IMAGE = "image"


Severity = Literal["critical", "major", "minor", "info"]

SEVERITY_ORDER: dict[str, int] = {"critical": 0, "major": 1, "minor": 2, "info": 3}

SEVERITY_LABELS_JA: dict[str, str] = {
    "critical": "致命的",
    "major": "重要",
    "minor": "軽微",
    "info": "参考",
}


class PrescriptionOperation(str, Enum):
    """処方の演算子（Storyboard に自動適用可能な修正操作）。"""

    ENLARGE_PANEL = "enlarge_panel"
    CONVERT_TO_SPLASH = "convert_to_splash"
    MOVE_PEAK_TO_REVEAL = "move_peak_to_reveal"
    STRENGTHEN_HOOK = "strengthen_hook"
    INSERT_PAUSE = "insert_pause"
    VARY_SHOT = "vary_shot"
    SPLIT_BALLOON = "split_balloon"
    MOVE_BALLOON = "move_balloon"
    REORDER_PANELS = "reorder_panels"
    REBALANCE_DENSITY = "rebalance_density"
    ADD_CONTRAST = "add_contrast"
    ADD_SLANT = "add_slant"


PRESCRIPTION_LABELS_JA: dict[str, str] = {
    PrescriptionOperation.ENLARGE_PANEL.value: "山場のコマを拡大",
    PrescriptionOperation.CONVERT_TO_SPLASH.value: "1ページ大ゴマ化",
    PrescriptionOperation.MOVE_PEAK_TO_REVEAL.value: "山場をめくり直後へ移動",
    PrescriptionOperation.STRENGTHEN_HOOK.value: "めくり前の引きを強化",
    PrescriptionOperation.INSERT_PAUSE.value: "山場直前に「間」を挿入",
    PrescriptionOperation.VARY_SHOT.value: "ショット・アングルの単調さを打破",
    PrescriptionOperation.SPLIT_BALLOON.value: "長台詞のフキダシを分割",
    PrescriptionOperation.MOVE_BALLOON.value: "フキダシの読み順を是正",
    PrescriptionOperation.REORDER_PANELS.value: "コマの段組みを組み直す",
    PrescriptionOperation.REBALANCE_DENSITY.value: "情報密度を隣接ページへ分散",
    PrescriptionOperation.ADD_CONTRAST.value: "コマ面積に大小の緩急をつける",
    PrescriptionOperation.ADD_SLANT.value: "斜めガターで動勢を加える",
}


# ---------------------------------------------------------------------------
# 既存原稿アノテーション（入力）
# ---------------------------------------------------------------------------


def _check_bbox(bbox: BBox, label: str) -> None:
    x0, y0, x1, y1 = bbox
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"{label} の bbox は x0<x1 かつ y0<y1 である必要があります: {bbox}")


class AnnotatedBalloon(_Base):
    """既存原稿のフキダシ。台詞が不明な場合は char_count（推定文字数）で密度を表す。"""

    text: str = Field("", max_length=400)
    char_count: int | None = Field(None, ge=0, le=2000)
    bbox: BBox
    kind: BalloonKind = BalloonKind.NORMAL
    speaker: str | None = None
    order_hint: int | None = Field(None, ge=0, description="作者が意図したコマ内の読み順（任意）")

    @model_validator(mode="after")
    def _validate_bbox(self) -> "AnnotatedBalloon":
        _check_bbox(self.bbox, "フキダシ")
        return self

    def effective_chars(self) -> int:
        """空白を除いた台詞文字数（台詞が無ければ char_count）。"""
        stripped = "".join(self.text.split())
        if stripped:
            return len(stripped)
        return int(self.char_count or 0)


class AnnotatedPanel(_Base):
    """既存原稿のコマ。bbox または polygon（3点以上）のどちらかが必須。"""

    bbox: BBox | None = None
    polygon: list[Point] | None = None
    balloons: list[AnnotatedBalloon] = Field(default_factory=list)
    shot: ShotSize | None = None
    angle: CameraAngle | None = None
    character_count: int = Field(0, ge=0, le=20)
    ink_ratio: float | None = Field(None, ge=0.0, le=1.0, description="ベタ・トーンを含む黒面積率")
    sfx_count: int = Field(0, ge=0, le=20)
    is_silent: bool | None = None
    bleed_sides: list[Literal["top", "bottom", "left", "right"]] = Field(default_factory=list)
    breakout: bool = False
    intensity_hint: float | None = Field(None, ge=0.0, le=1.0, description="作者が意図した緊張度（任意）")
    note: str = Field("", max_length=1000)

    @model_validator(mode="after")
    def _validate_shape(self) -> "AnnotatedPanel":
        if self.polygon is not None and len(self.polygon) < 3:
            raise ValueError("コマの polygon は3点以上が必要です")
        if self.bbox is None and self.polygon is None:
            raise ValueError("コマには bbox または polygon が必要です")
        if self.bbox is not None:
            _check_bbox(self.bbox, "コマ")
        return self

    def points(self) -> list[tuple[float, float]]:
        """コマの輪郭点（polygon 優先、無ければ bbox の4隅を時計回り）。"""
        if self.polygon is not None:
            return [(p.x, p.y) for p in self.polygon]
        assert self.bbox is not None
        x0, y0, x1, y1 = self.bbox
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def bounds(self) -> BBox:
        pts = self.points()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))


class AnnotatedPage(_Base):
    """既存原稿の1ページ。reading_order はコマ index の明示的な読み順（無ければ XY カットで推定）。"""

    width: float = Field(1000.0, gt=0)
    height: float = Field(1414.0, gt=0)
    side: PageSide | None = None
    panels: list[AnnotatedPanel] = Field(..., min_length=1, max_length=24)
    reading_order: list[int] | None = None

    @model_validator(mode="after")
    def _validate_order(self) -> "AnnotatedPage":
        if self.reading_order is not None and sorted(self.reading_order) != list(range(len(self.panels))):
            raise ValueError("reading_order はコマ index（0..n-1）の順列である必要があります")
        return self


class PageAnnotation(_Base):
    """既存原稿（複数ページ）のアノテーション。"""

    title: str = Field("既存原稿", min_length=1, max_length=200)
    first_page_side: PageSide = PageSide.RIGHT
    pages: list[AnnotatedPage] = Field(..., min_length=1, max_length=64)
    notes: str = Field("", max_length=2000)


# ---------------------------------------------------------------------------
# 画像からの自動検出
# ---------------------------------------------------------------------------


class DetectedBalloon(_Base):
    bbox: BBox
    area_ratio: float = Field(..., ge=0.0, le=1.0, description="コマ面積に対するフキダシ面積比")
    fill_ratio: float = Field(..., ge=0.0, le=1.0, description="外接矩形に対する白領域の充填率（楕円≈0.785）")
    estimated_chars: int = Field(0, ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class DetectedPanel(_Base):
    bbox: BBox
    polygon: list[Point]
    area_ratio: float = Field(..., ge=0.0, le=1.0, description="ページ面積に対するコマ面積比")
    ink_ratio: float = Field(..., ge=0.0, le=1.0)
    slant_deg: float = 0.0
    confidence: float = Field(..., ge=0.0, le=1.0)
    balloons: list[DetectedBalloon] = Field(default_factory=list)


class PageDetection(_Base):
    page_index: int = Field(..., ge=0)
    width: float = Field(..., gt=0, description="ページ座標系の幅（既定1000）")
    height: float = Field(..., gt=0)
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    threshold: float = Field(..., ge=0.0, le=1.0, description="二値化しきい値（大津法）")
    panels: list[DetectedPanel] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ImageDetectionResult(_Base):
    pages: list[PageDetection]
    annotation: PageAnnotation


# ---------------------------------------------------------------------------
# 解析結果
# ---------------------------------------------------------------------------


class PanelMetrics(_Base):
    panel_id: str
    page_index: int
    index_in_page: int
    reading_order: int
    area_ratio: float = Field(..., ge=0.0, le=1.0, description="ページ内コマ領域（全コマの外接矩形）に対する面積比")
    aspect: float = Field(..., gt=0.0)
    slant_deg: float = 0.0
    is_bleed: bool = False
    is_breakout: bool = False
    is_silent: bool = False
    shot: str | None = None
    angle: str | None = None
    char_count: int = Field(0, ge=0)
    balloon_count: int = Field(0, ge=0)
    balloon_coverage: float = Field(0.0, ge=0.0, le=1.0)
    ink_ratio: float = Field(0.0, ge=0.0, le=1.0)
    intensity: float = Field(..., ge=0.0, le=1.0, description="視覚的手がかりから推定した劇的強度")
    target_tension: float | None = Field(None, ge=0.0, le=1.0)
    t: float = Field(..., ge=0.0, le=1.0)


class PageMetrics(_Base):
    page_index: int
    side: str
    page_turn_role: str
    panel_count: int = Field(..., ge=0)
    char_count: int = Field(0, ge=0)
    balloon_count: int = Field(0, ge=0)
    information_density: float = Field(0.0, ge=0.0)
    density_level: Literal["sparse", "balanced", "dense", "overloaded"] = "balanced"
    area_cv: float = Field(0.0, ge=0.0, description="コマ面積の変動係数（大小の緩急）")
    max_area_ratio: float = Field(0.0, ge=0.0, le=1.0)
    mean_intensity: float = Field(0.0, ge=0.0, le=1.0)
    peak_intensity: float = Field(0.0, ge=0.0, le=1.0)
    gaze_path_length: float = Field(0.0, ge=0.0, description="視線経路長（ページ対角線長で正規化）")
    backtracks: int = Field(0, ge=0)
    crossings: int = Field(0, ge=0)
    z_flow_conformity: float = Field(1.0, ge=0.0, le=1.0)
    order_ambiguity: float = Field(0.0, ge=0.0, le=1.0)
    reveal_strength: float | None = Field(None, ge=0.0, le=1.0)
    hook_strength: float | None = Field(None, ge=0.0, le=1.0)
    score: float = Field(..., ge=0.0, le=100.0)


class AxisScore(_Base):
    axis: DiagnosticAxis
    label: str
    score: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., ge=0.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    summary: str = ""


class DiagnosticFinding(_Base):
    id: str
    axis: DiagnosticAxis
    severity: Severity
    code: str
    page_index: int | None = None
    panel_ids: list[str] = Field(default_factory=list)
    balloon_ids: list[str] = Field(default_factory=list)
    title: str
    detail: str
    metric: str
    observed: float
    target: float
    anchor: Point | None = Field(None, description="オーバーレイのピン位置（ページ座標）")


class Prescription(_Base):
    id: str
    operation: PrescriptionOperation
    axis: DiagnosticAxis
    finding_ids: list[str] = Field(default_factory=list)
    page_index: int | None = None
    panel_ids: list[str] = Field(default_factory=list)
    balloon_ids: list[str] = Field(default_factory=list)
    title: str
    instruction: str
    expected_gain: float = Field(0.0, description="総合スコアの期待改善量（反実仮想シミュレーション）")
    axis_gains: dict[str, float] = Field(default_factory=dict)
    simulated: bool = False
    applicable: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    rank: int = Field(0, ge=0)


class IntensityCurve(_Base):
    t: list[float]
    inferred: list[float]
    target: list[float] | None = None
    correlation: float | None = None
    dtw_distance: float | None = None
    panel_ids: list[str] = Field(default_factory=list)


class DiagnosisCommentary(_Base):
    """LLM 増幅（task ``manga.diagnosis_commentary``）の出力スキーマ。"""

    headline: str = Field(..., min_length=1, max_length=200)
    strengths: list[str] = Field(default_factory=list, max_length=8)
    priorities: list[str] = Field(..., min_length=1, max_length=8)
    editor_note: str = Field(..., min_length=1, max_length=2000)


def grade_for(score: float) -> Literal["S", "A", "B", "C", "D"]:
    """総合スコアの評語（S≥90, A≥80, B≥70, C≥60, それ未満 D）。"""
    if score >= 90.0:
        return "S"
    if score >= 80.0:
        return "A"
    if score >= 70.0:
        return "B"
    if score >= 60.0:
        return "C"
    return "D"


class DiagnosticReport(_Base):
    """ネーム構造ドクターの診断レポート。"""

    id: str = Field(default_factory=lambda: new_id("diag"))
    title: str = Field(..., min_length=1, max_length=200)
    source_type: DiagnosticSourceType
    source_id: str | None = None
    target_structure_id: str | None = None
    overall_score: float = Field(..., ge=0.0, le=100.0)
    grade: Literal["S", "A", "B", "C", "D"]
    axes: list[AxisScore]
    pages: list[PageMetrics]
    panels: list[PanelMetrics]
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    prescriptions: list[Prescription] = Field(default_factory=list)
    intensity_curve: IntensityCurve
    commentary: str = ""
    used_llm: bool = False
    storyboard: Storyboard | None = Field(None, description="解析に用いた正規化ネーム（オーバーレイ描画・処方適用に使用）")
    page_images: list[str] = Field(default_factory=list, description="画像入力時の縮小ページ画像（data URL）")
    created_at: str = Field(default_factory=utc_now_iso)

    def axis(self, axis: DiagnosticAxis | str) -> AxisScore:
        value = axis.value if isinstance(axis, DiagnosticAxis) else str(axis)
        for item in self.axes:
            if item.axis.value == value:
                return item
        raise KeyError(value)

    def top_prescriptions(self, n: int = 5) -> list[Prescription]:
        return sorted(self.prescriptions, key=lambda p: (-p.expected_gain, p.rank))[: max(0, n)]

    def findings_for_page(self, page_index: int) -> list[DiagnosticFinding]:
        return [f for f in self.findings if f.page_index == page_index]
