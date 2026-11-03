"""漫画演出・コマ割り・視線誘導の構造化スキーマ。

座標系: 仕上がりページ（トリム）を幅1000×高さ1414（B判系 √2 比）とし、左上原点・y 下向き。
断ち切り（bleed）は各辺30の外側領域。右綴じ（日本式）の見開きでは右ページが先に読まれ、
右ページのノド（綴じ側）は左辺、左ページのノドは右辺となる。

- ページ役割: ``REVEAL`` = めくり直後の右ページ（左ページの次）、``HOOK`` = めくり前の左ページ終端。
- コマは多角形（矩形・斜め割り台形・変形）で表し、``bleed_sides`` の辺は断ち切りまで延長する。
- フキダシは縦書きの列（右→左）を保持し、オノマトペは配置ベクトル・回転を持つ。
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from core.common import new_id, utc_now_iso
from core.manga_pipeline import geometry as geo

__all__ = [
    "PageSide",
    "PageTurnRole",
    "PageGeometry",
    "Point",
    "Polygon",
    "ShotSize",
    "SHOT_ORDER",
    "SHOT_LABELS_JA",
    "CameraAngle",
    "ANGLE_LABELS_JA",
    "CompositionRule",
    "CameraWork",
    "CharacterProfile",
    "CharacterAction",
    "BalloonKind",
    "Balloon",
    "SoundEffect",
    "PanelShape",
    "Panel",
    "Page",
    "Spread",
    "GazeTransition",
    "GrammarViolation",
    "GrammarReport",
    "Storyboard",
    "DialogueLine",
    "PanelDialogue",
    "PanelDialogueBatch",
    "BleedSide",
    "Emphasis",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


BleedSide = Literal["top", "bottom", "left", "right"]
Emphasis = Literal["normal", "large", "splash", "spread"]


class PageSide(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class PageTurnRole(str, Enum):
    NONE = "none"
    HOOK = "hook"
    REVEAL = "reveal"


class PageGeometry(_Base):
    """ページ寸法と基本枠マージン（単位: 仕上がり座標）。"""

    width: float = Field(1000.0, gt=0)
    height: float = Field(1414.0, gt=0)
    bleed: float = Field(30.0, ge=0)
    margin_top: float = Field(90.0, ge=0)
    margin_bottom: float = Field(110.0, ge=0)
    margin_outer: float = Field(70.0, ge=0)
    margin_inner: float = Field(90.0, ge=0)
    gutter_x: float = Field(16.0, ge=0, description="同一段内の左右のコマ間隔")
    gutter_y: float = Field(30.0, ge=0, description="段と段の間隔")

    def frame_rect(self, side: PageSide | str) -> tuple[float, float, float, float]:
        """基本枠 (x0, y0, x1, y1)。右ページはノド（内側）が左辺、左ページは右辺。"""
        page_side = PageSide(side)
        if page_side == PageSide.RIGHT:
            x0, x1 = self.margin_inner, self.width - self.margin_outer
        else:
            x0, x1 = self.margin_outer, self.width - self.margin_inner
        return (x0, self.margin_top, x1, self.height - self.margin_bottom)

    def frame_area(self, side: PageSide | str) -> float:
        x0, y0, x1, y1 = self.frame_rect(side)
        return (x1 - x0) * (y1 - y0)

    def bleed_rect(self) -> tuple[float, float, float, float]:
        return (-self.bleed, -self.bleed, self.width + self.bleed, self.height + self.bleed)

    def trim_rect(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, self.width, self.height)


class Point(_Base):
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class Polygon(_Base):
    """コマ枠多角形（ページ座標）。"""

    points: list[Point] = Field(..., min_length=3)

    @classmethod
    def from_tuples(cls, points: list[tuple[float, float]]) -> "Polygon":
        return cls(points=[Point(x=float(x), y=float(y)) for x, y in points])

    @classmethod
    def rect(cls, x0: float, y0: float, x1: float, y1: float) -> "Polygon":
        return cls.from_tuples(geo.rect_points(x0, y0, x1, y1))

    def tuples(self) -> list[tuple[float, float]]:
        return [(p.x, p.y) for p in self.points]

    def area(self) -> float:
        return geo.polygon_area(self.tuples())

    def centroid(self) -> Point:
        x, y = geo.polygon_centroid(self.tuples())
        return Point(x=x, y=y)

    def bbox(self) -> tuple[float, float, float, float]:
        return geo.polygon_bbox(self.tuples())

    def contains(self, point: Point | tuple[float, float]) -> bool:
        return geo.point_in_polygon(point, self.tuples())

    def edges(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return geo.polygon_edges(self.tuples())

    def is_convex(self) -> bool:
        return geo.is_convex(self.tuples())

    def core_box(self) -> tuple[float, float, float, float]:
        return geo.core_box(self.tuples())

    def translate(self, dx: float, dy: float) -> "Polygon":
        return Polygon.from_tuples([(x + dx, y + dy) for x, y in self.tuples()])


class ShotSize(str, Enum):
    ECU = "ECU"
    CU = "CU"
    MS = "MS"
    FS = "FS"
    ELS = "ELS"


SHOT_ORDER: tuple[ShotSize, ...] = (ShotSize.ECU, ShotSize.CU, ShotSize.MS, ShotSize.FS, ShotSize.ELS)

SHOT_LABELS_JA: dict[str, str] = {
    ShotSize.ECU.value: "超クローズアップ",
    ShotSize.CU.value: "クローズアップ",
    ShotSize.MS.value: "ミディアムショット",
    ShotSize.FS.value: "フルショット",
    ShotSize.ELS.value: "超ロングショット",
}


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye_level"
    HIGH = "high"
    LOW = "low"
    DUTCH = "dutch"


ANGLE_LABELS_JA: dict[str, str] = {
    CameraAngle.EYE_LEVEL.value: "アイレベル",
    CameraAngle.HIGH.value: "俯瞰",
    CameraAngle.LOW.value: "アオリ",
    CameraAngle.DUTCH.value: "ダッチアングル",
}


class CompositionRule(str, Enum):
    RULE_OF_THIRDS = "rule_of_thirds"
    CENTERED = "centered"
    DIAGONAL = "diagonal"
    GOLDEN_SPIRAL = "golden_spiral"
    SYMMETRY = "symmetry"
    FRAME_IN_FRAME = "frame_in_frame"
    LEADING_LINES = "leading_lines"
    NEGATIVE_SPACE = "negative_space"


class CameraWork(_Base):
    """カメラワーク（ショットサイズ・アングル・レンズ焦点距離・構図）。"""

    shot: ShotSize
    angle: CameraAngle = CameraAngle.EYE_LEVEL
    lens_mm: int = Field(50, ge=18, le=135)
    composition: CompositionRule = CompositionRule.RULE_OF_THIRDS
    dutch_degrees: float = Field(0.0, ge=-45.0, le=45.0)
    notes: str = Field("", max_length=500)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def horizontal_fov_deg(self) -> float:
        """35mm判（横36mm）換算の水平画角。"""
        return round(math.degrees(2.0 * math.atan(36.0 / (2.0 * float(self.lens_mm)))), 3)


class CharacterProfile(_Base):
    """登場人物（構造グラフの主体ノードに対応する抽象人物）。"""

    id: str
    role: str
    name_placeholder: str
    visual_description: str
    seed: int | None = None


class CharacterAction(_Base):
    """コマ内での人物の演技（位置はコマ内正規化座標 [0,1]）。"""

    character_id: str
    position: Point
    scale: float = Field(0.5, ge=0.05, le=1.0)
    pose: str = "standing"
    expression: str = "neutral"
    gaze: Point = Field(default_factory=lambda: Point(x=-1.0, y=0.0))
    action: str = ""

    @field_validator("position")
    @classmethod
    def _clamp_position(cls, value: Point) -> Point:
        return Point(x=min(1.0, max(0.0, value.x)), y=min(1.0, max(0.0, value.y)))

    @field_validator("gaze")
    @classmethod
    def _normalize_gaze(cls, value: Point) -> Point:
        x, y = geo.normalize_vector(value.x, value.y)
        return Point(x=round(x, 6), y=round(y, 6))


class BalloonKind(str, Enum):
    NORMAL = "normal"
    SHOUT = "shout"
    MONOLOGUE = "monologue"
    FLASH = "flash"


class Balloon(_Base):
    """フキダシ（座標はページ座標、columns は縦書き列を読み順＝右→左で保持）。"""

    id: str
    kind: BalloonKind = BalloonKind.NORMAL
    speaker_id: str | None = None
    text: str = Field(..., min_length=1, max_length=400)
    columns: list[str] = Field(default_factory=list)
    center: Point
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    tail_target: Point | None = None
    reading_order: int = Field(0, ge=0)

    @property
    def char_count(self) -> int:
        return len("".join(self.text.split()))


class SoundEffect(_Base):
    """オノマトペ（描き文字）。"""

    id: str
    text: str = Field(..., min_length=1, max_length=40)
    position: Point
    direction: Point = Field(default_factory=lambda: Point(x=1.0, y=0.0))
    scale: float = Field(1.0, ge=0.2, le=4.0)
    rotation_deg: float = Field(0.0, ge=-180.0, le=180.0)
    style: Literal["impact", "motion", "ambient", "emotion"] = "impact"

    @field_validator("direction")
    @classmethod
    def _normalize_direction(cls, value: Point) -> Point:
        x, y = geo.normalize_vector(value.x, value.y, default=(1.0, 0.0))
        return Point(x=round(x, 6), y=round(y, 6))


class PanelShape(str, Enum):
    RECT = "rect"
    SLANTED = "slanted"
    IRREGULAR = "irregular"
    BORDERLESS = "borderless"


class Panel(_Base):
    """コマ。"""

    id: str
    page_index: int = Field(..., ge=0)
    index_in_page: int = Field(..., ge=0)
    reading_order: int = Field(0, ge=0)
    polygon: Polygon
    shape: PanelShape = PanelShape.RECT
    bleed_sides: list[BleedSide] = Field(default_factory=list)
    breakout: bool = False
    camera: CameraWork
    characters: list[CharacterAction] = Field(default_factory=list)
    balloons: list[Balloon] = Field(default_factory=list)
    sfx: list[SoundEffect] = Field(default_factory=list)
    description: str = Field("", max_length=2000)
    beat_id: str | None = None
    secondary_beat_ids: list[str] = Field(default_factory=list)
    t: float = Field(0.0, ge=0.0, le=1.0)
    tension: float = Field(0.5, ge=0.0, le=1.0)
    emphasis: Emphasis = "normal"
    is_silent: bool = False
    spread_role: Literal["none", "main", "continuation"] = "none"
    prompt_hints: list[str] = Field(default_factory=list)

    @field_validator("bleed_sides")
    @classmethod
    def _unique_sides(cls, value: list[str]) -> list[str]:
        ordered: list[str] = []
        for side in value:
            if side not in ordered:
                ordered.append(side)
        return ordered

    def text_char_count(self) -> int:
        return sum(balloon.char_count for balloon in self.balloons)

    def beat_ids(self) -> list[str]:
        ids = [self.beat_id] if self.beat_id else []
        return ids + [b for b in self.secondary_beat_ids if b not in ids]


class Page(_Base):
    """ページ。"""

    index: int = Field(..., ge=0)
    number: int = Field(..., ge=1)
    side: PageSide
    panels: list[Panel] = Field(default_factory=list)
    spread_id: str | None = None
    page_turn_role: PageTurnRole = PageTurnRole.NONE
    information_density: float = Field(0.0, ge=0.0)
    black_ratio_target: float = Field(0.3, ge=0.0, le=1.0)
    tension_mean: float = Field(0.5, ge=0.0, le=1.0)
    t_start: float = Field(0.0, ge=0.0, le=1.0)
    t_end: float = Field(1.0, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)

    def ordered_panels(self) -> list[Panel]:
        return sorted(self.panels, key=lambda p: p.index_in_page)


class Spread(_Base):
    """見開き（右ページ＋左ページ）。"""

    id: str
    right_page: int = Field(..., ge=0)
    left_page: int = Field(..., ge=0)
    kind: Literal["facing", "double_page_panel"] = "facing"

    @model_validator(mode="after")
    def _adjacent(self) -> "Spread":
        if self.left_page != self.right_page + 1:
            raise ValueError("見開きの左ページは右ページの直後である必要があります")
        return self


class GazeTransition(_Base):
    """コマ間の視線遷移ベクトル。"""

    from_panel: str
    to_panel: str
    start: Point
    end: Point
    vector: Point
    is_backtrack: bool = False
    same_tier: bool = False
    crosses: list[int] = Field(default_factory=list)


class GrammarViolation(_Base):
    """漫画文法違反。"""

    rule_id: str
    severity: Literal["error", "warning", "info"]
    page_index: int | None = None
    panel_ids: list[str] = Field(default_factory=list)
    message: str
    fix_instruction: str
    penalty: float = Field(0.0, ge=0.0)


class GrammarReport(_Base):
    """漫画文法監査レポート。"""

    score: float = Field(..., ge=0.0, le=100.0)
    passed: bool
    violations: list[GrammarViolation] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    iteration: int = Field(0, ge=0)

    def errors(self) -> list[GrammarViolation]:
        return [v for v in self.violations if v.severity == "error"]

    def warnings(self) -> list[GrammarViolation]:
        return [v for v in self.violations if v.severity == "warning"]

    def by_rule(self) -> dict[str, list[GrammarViolation]]:
        grouped: dict[str, list[GrammarViolation]] = {}
        for violation in self.violations:
            grouped.setdefault(violation.rule_id, []).append(violation)
        return grouped


class Storyboard(_Base):
    """ネーム（ストーリーボード）全体。"""

    id: str = Field(default_factory=lambda: new_id("sb"))
    title: str = Field(..., min_length=1, max_length=200)
    structure_id: str
    theme: dict[str, str] = Field(default_factory=dict)
    geometry: PageGeometry = Field(default_factory=PageGeometry)
    first_page_side: PageSide = PageSide.RIGHT
    pages: list[Page] = Field(default_factory=list)
    spreads: list[Spread] = Field(default_factory=list)
    characters: list[CharacterProfile] = Field(default_factory=list)
    seed: int = 0
    grammar: GrammarReport | None = None
    revision_log: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def panels_in_reading_order(self) -> list[Panel]:
        ordered: list[Panel] = []
        for page in sorted(self.pages, key=lambda p: p.index):
            ordered.extend(page.ordered_panels())
        return ordered

    def panel(self, panel_id: str) -> Panel | None:
        for page in self.pages:
            for panel in page.panels:
                if panel.id == panel_id:
                    return panel
        return None

    def page_of_panel(self, panel_id: str) -> Page | None:
        for page in self.pages:
            if any(panel.id == panel_id for panel in page.panels):
                return page
        return None

    def character(self, character_id: str) -> CharacterProfile | None:
        for character in self.characters:
            if character.id == character_id:
                return character
        return None

    def spread_of_page(self, page_index: int) -> Spread | None:
        for spread in self.spreads:
            if page_index in (spread.right_page, spread.left_page):
                return spread
        return None

    def summary(self) -> dict[str, Any]:
        panels = self.panels_in_reading_order()
        return {
            "id": self.id,
            "title": self.title,
            "structure_id": self.structure_id,
            "pages": self.page_count,
            "panels": len(panels),
            "balloons": sum(len(p.balloons) for p in panels),
            "spreads": len(self.spreads),
            "grammar_score": self.grammar.score if self.grammar else None,
            "grammar_passed": self.grammar.passed if self.grammar else None,
            "created_at": self.created_at,
        }


class DialogueLine(_Base):
    """LLM 台詞増幅の1行。"""

    speaker_id: str | None = None
    kind: BalloonKind = BalloonKind.NORMAL
    text: str = Field(..., min_length=1, max_length=120)


class PanelDialogue(_Base):
    panel_id: str
    lines: list[DialogueLine] = Field(default_factory=list, max_length=4)
    silent: bool = False


class PanelDialogueBatch(_Base):
    """LLM 台詞増幅の出力（コマごとの台詞列）。"""

    storyboard_title: str = ""
    panels: list[PanelDialogue] = Field(..., min_length=1)
