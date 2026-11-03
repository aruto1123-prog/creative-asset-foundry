"""構造駆動ネーム生成器と漫画文法自己修正ループ。

``UniversalStructure`` の感情曲線（緊張度・認知負荷）・情報開示曲線・ビート因果チェーンから、

1. ページ配分: 重み w(t) = 0.6 + 0.8·tension + 0.3·cognitive_load の累積を target_pages 等分する
   （緊張の高い区間ほど多くのページを使い、時間を引き延ばす）。
2. めくり設計: 転換点（pivotal）上位ビートを「めくり直後の右ページ（REVEAL）」の先頭コマへ寄せるよう
   ページ境界を区分的に再配分し、直前の左ページ（HOOK）終端に「引き」のコマを置く。
   大域緊張最大ビートは見開き（REVEAL 右ページ＋次の左ページ）または splash で強調する。
3. コマ割り: 高緊張ページほど少コマ・大ゴマ、情報/会話の多いページほど多コマ。段（tier）ベースで、
   高緊張ページは縦ガターを傾けた斜め割り（台形）、強調ゴマは断ち切り・ぶち抜き。
4. カメラ: ビート種別と緊張度から ショットサイズ/アングル/レンズ/構図 を決め、3連続の単調さを生成時点で回避。
5. 演技と台詞: 主体ノード→人物、ビート→表情・ポーズ・台詞テンプレート、縦書き文節改行、フキダシ配置、オノマトペ。

を決定論的に展開する。``generate_validated`` は Manga Grammar Validator による監査と
``StoryboardFixer`` による規則別修復を反復し、最終的に規則準拠の安全レイアウトで必ず合格させる。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from core.common import clamp, stable_seed
from core.manga_pipeline import geometry as geo
from core.manga_pipeline.geometry import compute_gaze_transitions, compute_reading_order
from core.manga_pipeline.grammar_validator import (
    MangaGrammarValidator,
    emphasis_beat_ids,
    identify_peak_beat,
    page_information_density,
)
from core.manga_pipeline.schemas import (
    Balloon,
    BalloonKind,
    CameraAngle,
    CameraWork,
    CharacterAction,
    CharacterProfile,
    CompositionRule,
    DialogueLine,
    GrammarReport,
    Page,
    PageSide,
    PageTurnRole,
    Panel,
    PanelDialogue,
    PanelDialogueBatch,
    PanelShape,
    Point,
    Polygon,
    ShotSize,
    SoundEffect,
    Spread,
    Storyboard,
)
from core.manga_pipeline.text_layout import (
    balloon_size,
    break_japanese_text,
    place_balloons,
    split_text_for_balloons,
    visible_length,
)
from core.structure_engine.schemas import EdgeKind, NodeKind, StructuralBeat, UniversalStructure

logger = logging.getLogger(__name__)

__all__ = [
    "StoryboardConfig",
    "StoryboardGenerator",
    "StoryboardFixer",
    "MangaGrammarValidator",
    "compute_reading_order",
    "compute_gaze_transitions",
    "break_japanese_text",
    "page_sides",
    "page_roles",
    "allocate_pages",
    "layout_page",
    "plan_camera",
    "enforce_camera_variety",
    "layout_balloons",
    "place_sfx",
    "refresh_page_metrics",
    "finalize_reading_order",
    "template_lines",
    "build_dialogue_batch",
    "apply_dialogue_batch",
]


class StoryboardConfig(BaseModel):
    """ネーム生成設定。"""

    model_config = ConfigDict(extra="ignore")

    target_pages: int = Field(16, ge=4, le=64)
    min_panels_per_page: int = Field(2, ge=1, le=8)
    max_panels_per_page: int = Field(7, ge=2, le=12)
    first_page_side: PageSide = PageSide.RIGHT
    max_chars_per_balloon: int = Field(40, ge=8, le=120)
    max_chars_per_column: int = Field(7, ge=3, le=16)
    max_balloons_per_panel: int = Field(3, ge=1, le=6)
    grammar_threshold: float = Field(80.0, ge=0.0, le=100.0)
    max_fix_iterations: int = Field(6, ge=0, le=20)
    seed: int = 0
    use_llm_dialogue: bool = False
    font_size: float = Field(22.0, ge=10.0, le=48.0)


# ---------------------------------------------------------------------------
# 演出辞書（ビート種別 → カメラ・演技・台詞）
# ---------------------------------------------------------------------------

S = ShotSize
A = CameraAngle

SHOT_PREFERENCES: dict[str, list[ShotSize]] = {
    "hook": [S.ECU, S.CU, S.ELS],
    "premise": [S.ELS, S.FS, S.MS],
    "setup": [S.FS, S.ELS, S.MS],
    "escalation": [S.MS, S.FS, S.CU],
    "local_optimization_limit": [S.MS, S.CU, S.FS],
    "false_victory": [S.FS, S.MS, S.CU],
    "crisis": [S.ELS, S.CU, S.FS],
    "revelation": [S.ECU, S.CU, S.MS],
    "structural_twist": [S.CU, S.ECU, S.MS],
    "costly_breakthrough": [S.FS, S.ELS, S.CU],
    "climax": [S.ELS, S.FS, S.ECU],
    "payoff": [S.MS, S.CU, S.FS],
    "afterglow": [S.ELS, S.FS, S.MS],
    "transition": [S.ELS, S.FS, S.MS],
}

ANGLE_PREFERENCES: dict[str, list[CameraAngle]] = {
    "hook": [A.DUTCH, A.EYE_LEVEL, A.HIGH],
    "premise": [A.EYE_LEVEL, A.HIGH, A.LOW],
    "setup": [A.EYE_LEVEL, A.HIGH, A.LOW],
    "escalation": [A.EYE_LEVEL, A.LOW, A.HIGH],
    "local_optimization_limit": [A.HIGH, A.EYE_LEVEL, A.DUTCH],
    "false_victory": [A.LOW, A.EYE_LEVEL, A.HIGH],
    "crisis": [A.HIGH, A.EYE_LEVEL, A.DUTCH],
    "revelation": [A.DUTCH, A.EYE_LEVEL, A.LOW],
    "structural_twist": [A.DUTCH, A.LOW, A.EYE_LEVEL],
    "costly_breakthrough": [A.LOW, A.DUTCH, A.EYE_LEVEL],
    "climax": [A.LOW, A.DUTCH, A.HIGH],
    "payoff": [A.EYE_LEVEL, A.HIGH, A.LOW],
    "afterglow": [A.EYE_LEVEL, A.HIGH, A.LOW],
    "transition": [A.EYE_LEVEL, A.HIGH, A.LOW],
}

EXPRESSIONS: dict[str, str] = {
    "hook": "uneasy",
    "premise": "resolute",
    "setup": "calm",
    "escalation": "focused",
    "local_optimization_limit": "frustrated",
    "false_victory": "triumphant smile",
    "crisis": "despair",
    "revelation": "shocked",
    "structural_twist": "wide-eyed shock",
    "costly_breakthrough": "determined, gritted teeth",
    "climax": "fierce shout",
    "payoff": "relieved",
    "afterglow": "serene smile",
    "transition": "pensive",
}

POSES: dict[str, str] = {
    "hook": "looking back over shoulder",
    "premise": "standing tall",
    "setup": "preparing",
    "escalation": "advancing forward",
    "local_optimization_limit": "clenching fist",
    "false_victory": "raising arm",
    "crisis": "kneeling, head down",
    "revelation": "stepping back",
    "structural_twist": "frozen in place",
    "costly_breakthrough": "lunging forward",
    "climax": "dynamic action pose",
    "payoff": "holding an object close",
    "afterglow": "walking away",
    "transition": "walking",
}

SCENE_HINTS: dict[str, list[str]] = {
    "hook": ["mysterious atmosphere", "foreshadowing"],
    "premise": ["establishing shot", "everyday scenery"],
    "setup": ["preparation", "quiet tension"],
    "escalation": ["rising tension", "speed lines"],
    "local_optimization_limit": ["frustration", "heavy atmosphere"],
    "false_victory": ["triumphant mood", "bright light"],
    "crisis": ["isolation", "heavy shadows"],
    "revelation": ["shock", "dramatic backlight"],
    "structural_twist": ["dramatic lighting", "high contrast"],
    "costly_breakthrough": ["intense action", "speed lines", "impact"],
    "climax": ["epic scale", "impact frame", "speed lines"],
    "payoff": ["warm light", "quiet emotion"],
    "afterglow": ["calm", "wide open sky"],
    "transition": ["scenery", "passage of time"],
}

ACTION_BEATS = {"escalation", "false_victory", "crisis", "revelation", "structural_twist", "costly_breakthrough", "climax"}

DIALOGUE_TEMPLATES: dict[str, list[list[tuple[str, str]]]] = {
    "hook": [
        [("monologue", "……また、あの日の夢だ。")],
        [("monologue", "{THEME}——全ては、そこから始まった。")],
        [("normal", "何かが、おかしい。"), ("monologue", "胸の奥がざわつく。")],
    ],
    "premise": [
        [("normal", "言われた通りにやれば、届くはずだ。")],
        [("normal", "{THEME}を取り戻す。"), ("normal", "そのためなら、何でもする。")],
        [("normal", "これが正しいやり方なんだ。")],
    ],
    "setup": [
        [("normal", "準備はできてる。")],
        [("normal", "まずは、ここからだ。")],
        [("monologue", "まだ誰も気づいていない。")],
    ],
    "escalation": [
        [("normal", "いける……このまま押し切る！")],
        [("normal", "まだだ、まだ足りない。")],
        [("normal", "一歩ずつ、確実に。"), ("normal", "焦るな。")],
    ],
    "local_optimization_limit": [
        [("monologue", "どうして……同じやり方なのに。")],
        [("normal", "これ以上、進めない……？")],
        [("monologue", "何かが根本的に違う。")],
    ],
    "false_victory": [
        [("shout", "やった……！"), ("normal", "これで全部、元通りだ。")],
        [("normal", "勝った……のか？")],
        [("normal", "終わったんだ、全部。")],
    ],
    "crisis": [
        [("monologue", "もう、何も残っていない。")],
        [("monologue", "何を信じてきたんだ……。")],
        [("normal", "ここまで、なのか……。")],
    ],
    "revelation": [
        [("flash", "まさか……！")],
        [("flash", "全部、仕組まれていた……！")],
        [("normal", "嘘だ……そんなはずは。")],
    ],
    "structural_twist": [
        [("flash", "お前だったのか……！")],
        [("flash", "前提が、ひっくり返る。")],
        [("normal", "正しかったのは、逆だった。")],
    ],
    "costly_breakthrough": [
        [("shout", "これが、払うべき代償だ！")],
        [("shout", "全部くれてやる……だから！")],
        [("shout", "ここで、終わらせる！！")],
    ],
    "climax": [
        [("shout", "うおおおおっ！！")],
        [("shout", "届けえええっ！！")],
        [("shout", "これで、最後だ！")],
    ],
    "payoff": [
        [("normal", "ずっと、これを探してたんだ。")],
        [("normal", "ようやく、わかった。")],
        [("normal", "あの言葉の意味が。")],
    ],
    "afterglow": [
        [("monologue", "……行こう。新しい場所へ。")],
        [("monologue", "{THEME}は、もう怖くない。")],
        [("normal", "ここから、始めよう。")],
    ],
    "transition": [
        [("normal", "……行こう。")],
        [("normal", "時間がない。")],
        [("monologue", "そして、夜が明けた。")],
    ],
}

CONNECTIVE_TEMPLATES: dict[str, list[list[tuple[str, str]]]] = {
    "low": [[("normal", "……そうか。")], [("normal", "少し、休もう。")], [("monologue", "静かだ。")], [("normal", "それで、どうする？")]],
    "mid": [[("normal", "気をつけろ。")], [("normal", "まだ終わってない。")], [("normal", "見ろ、あれを。")], [("monologue", "違和感が消えない。")]],
    "high": [[("shout", "来るぞ！")], [("normal", "くっ……！")], [("shout", "逃げろ！")], [("normal", "まだ……だ！")]],
}

HOOK_TEMPLATES: list[list[tuple[str, str]]] = [
    [("normal", "……その時だった。")],
    [("monologue", "扉の向こうに、答えがある。")],
    [("normal", "まさか、あれは……。")],
]

# 直前ビート（局面）に応じた繋ぎゴマの反応台詞
PHASE_REACTIONS: dict[str, list[list[tuple[str, str]]]] = {
    "hook": [[("monologue", "思い出したくない。")], [("normal", "……誰だ？")], [("monologue", "あの日から、全部が変わった。")], [("normal", "まだ、夢の中みたいだ。")]],
    "premise": [[("normal", "本当に、それでいいのか？")], [("normal", "迷ってる暇はない。")], [("normal", "任せておけ。")], [("monologue", "信じるしかない。")]],
    "setup": [[("normal", "段取りは頭に入ってる。")], [("normal", "合図を待て。")], [("monologue", "静かすぎる。")], [("normal", "準備は、いいか。")]],
    "escalation": [[("normal", "この調子だ。")], [("normal", "手応えがある。")], [("normal", "もう一押しだ！")], [("monologue", "届く、きっと届く。")]],
    "local_optimization_limit": [[("normal", "なぜだ……。")], [("monologue", "壁が、見えない。")], [("normal", "同じことの繰り返しだ。")], [("monologue", "何かを見落としている。")]],
    "false_victory": [[("normal", "やっと……終わった。")], [("normal", "ははっ、拍子抜けだな。")], [("monologue", "本当に、これで？")], [("normal", "乾杯しよう。")]],
    "crisis": [[("monologue", "……。")], [("normal", "立てない……。")], [("monologue", "何も、聞こえない。")], [("normal", "もう、いい……。")]],
    "revelation": [[("normal", "どういう……ことだ。")], [("monologue", "全部、最初から。")], [("normal", "答えろ！")], [("monologue", "足元が、崩れていく。")]],
    "structural_twist": [[("normal", "信じていたのに。")], [("monologue", "世界が、反転する。")], [("normal", "なぜ、黙っていた。")], [("monologue", "なら、今までの全ては。")]],
    "costly_breakthrough": [[("shout", "まだだ！！")], [("normal", "これで、いい。")], [("monologue", "失うものは、もうない。")], [("normal", "見ていてくれ。")]],
    "climax": [[("shout", "いけえっ！！")], [("normal", "ぐっ……！")], [("monologue", "今しかない。")], [("normal", "終わりだ！")]],
    "payoff": [[("normal", "そういうことか。")], [("normal", "ありがとう。")], [("monologue", "ようやく、繋がった。")], [("normal", "これが、答えだ。")]],
    "afterglow": [[("monologue", "風が、変わった。")], [("normal", "また、いつか。")], [("monologue", "新しい朝だ。")], [("normal", "帰ろう。")]],
    "transition": [[("normal", "行くぞ。")], [("monologue", "時間が流れていく。")], [("normal", "次は、どこへ？")], [("monologue", "季節が変わる。")]],
}
PHASE_SILENCE: dict[str, float] = {
    "crisis": 0.5,
    "climax": 0.45,
    "structural_twist": 0.4,
    "revelation": 0.4,
    "afterglow": 0.4,
    "premise": 0.15,
    "setup": 0.2,
}

SFX_LIBRARY: dict[str, list[tuple[str, str]]] = {
    "impact": [("ドンッ", "impact"), ("バキッ", "impact"), ("ドゴォ", "impact")],
    "motion": [("ザッ", "motion"), ("ダッ", "motion"), ("ヒュッ", "motion")],
    "ambient": [("ゴゴゴ", "ambient"), ("ザワ……", "ambient")],
    "emotion": [("ドクン", "emotion"), ("シン……", "emotion")],
}

_TRAIT_HAIR = ["short black hair", "long silver hair", "messy brown hair", "tied-back dark hair", "cropped white hair", "wavy auburn hair"]
_TRAIT_CLOTHES = ["worn travel coat", "school uniform", "tailored dark suit", "hooded jacket", "plain shirt and vest", "long cloak"]
_TRAIT_FEATURES = ["sharp eyes", "gentle eyes", "faint scar on cheek", "thin glasses", "tired eyes", "piercing gaze"]
_ROLE_VISUALS = {
    "protagonist": "young protagonist, determined expression",
    "antagonist": "imposing antagonist, cold expression",
    "mentor": "older mentor figure, composed expression",
    "ally": "loyal companion, earnest expression",
    "other": "enigmatic figure, unreadable expression",
}
_ROLE_NAMES = {"protagonist": "主人公", "antagonist": "対立者", "mentor": "導き手", "ally": "協力者", "other": "第三者"}


# ---------------------------------------------------------------------------
# ページ構成
# ---------------------------------------------------------------------------


def page_sides(count: int, first: PageSide | str = PageSide.RIGHT) -> list[PageSide]:
    """ページの左右（交互）。"""
    start = PageSide(first)
    other = PageSide.LEFT if start == PageSide.RIGHT else PageSide.RIGHT
    return [start if i % 2 == 0 else other for i in range(count)]


def page_roles(sides: list[PageSide]) -> list[PageTurnRole]:
    """めくり役割: 左ページの次の右ページ=REVEAL、右ページの前の左ページ=HOOK。"""
    roles: list[PageTurnRole] = []
    for i, side in enumerate(sides):
        if side == PageSide.RIGHT and i > 0 and sides[i - 1] == PageSide.LEFT:
            roles.append(PageTurnRole.REVEAL)
        elif side == PageSide.LEFT and i + 1 < len(sides) and sides[i + 1] == PageSide.RIGHT:
            roles.append(PageTurnRole.HOOK)
        else:
            roles.append(PageTurnRole.NONE)
    return roles


def _beat_score(beat: StructuralBeat, peak_id: str | None) -> float:
    return (
        abs(beat.tension_delta)
        + abs(beat.information_shift)
        + (0.3 if beat.is_turning_point else 0.0)
        + (0.6 if beat.id == peak_id else 0.0)
    )


def allocate_pages(
    structure: UniversalStructure,
    count: int,
    roles: list[PageTurnRole],
    *,
    emphasis_ids: list[str],
    peak_id: str | None,
    preferred_reveals: set[int] | None = None,
    max_anchors: int | None = None,
) -> tuple[list[float], dict[int, str]]:
    """ページ境界 t（count+1 点）と、REVEAL ページ先頭へ固定したビートの対応を返す。"""
    grid = np.linspace(0.0, 1.0, 401)
    wave = structure.wave
    tension = np.interp(grid, wave.t, wave.tension)
    load = np.interp(grid, wave.t, wave.cognitive_load)
    weight = 0.6 + 0.8 * tension + 0.3 * load
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(grid))])
    cumulative /= cumulative[-1]

    def forward(t: float) -> float:
        return float(np.interp(t, grid, cumulative))

    def inverse(u: float) -> float:
        return float(np.interp(u, cumulative, grid))

    base = [inverse(k / count) for k in range(count + 1)]
    base[0], base[-1] = 0.0, 1.0
    reveal_pages = [i for i, role in enumerate(roles) if role == PageTurnRole.REVEAL]
    preferred = preferred_reveals or set()
    beats = {beat.id: beat for beat in structure.beats.beats}
    candidates = [beats[i] for i in emphasis_ids if i in beats]
    limit = max_anchors if max_anchors is not None else max(1, round(count / 6) + 1)
    ranked = sorted(candidates, key=lambda b: (-_beat_score(b, peak_id), b.t))[:limit]
    ranked.sort(key=lambda b: b.t)
    min_span = 0.2 / count
    anchors: dict[int, float] = {0: 0.0, count: 1.0}
    assigned: dict[int, str] = {}
    last_page, last_t = 0, 0.0
    for idx, beat in enumerate(ranked):
        remaining = len(ranked) - idx - 1
        options = [r for r in reveal_pages if r > last_page]
        if remaining and len(options) > remaining:
            options = options[: len(options) - remaining]
        start_t = max(0.0, beat.t - 0.004)
        feasible = [
            r
            for r in options
            if start_t - last_t >= (r - last_page) * min_span and (1.0 - start_t) >= (count - r) * min_span
        ]
        if not feasible:
            continue
        chosen = min(feasible, key=lambda r: abs(base[r] - beat.t) - (0.04 if r in preferred else 0.0))
        anchors[chosen] = start_t
        assigned[chosen] = beat.id
        last_page, last_t = chosen, start_t
    keys = sorted(anchors)
    bounds = [0.0] * (count + 1)
    for ka, kb in zip(keys, keys[1:]):
        ua, ub = forward(anchors[ka]), forward(anchors[kb])
        for k in range(ka, kb + 1):
            bounds[k] = inverse(ua + (k - ka) / (kb - ka) * (ub - ua))
        bounds[ka], bounds[kb] = anchors[ka], anchors[kb]
    for k in range(1, count + 1):
        if bounds[k] <= bounds[k - 1]:
            bounds[k] = min(1.0, bounds[k - 1] + 1e-4)
    bounds[-1] = 1.0
    return bounds, assigned


@dataclass
class _PagePlan:
    index: int
    side: PageSide
    role: PageTurnRole
    t_start: float
    t_end: float
    beats: list[StructuralBeat] = field(default_factory=list)
    anchor_beat_id: str | None = None
    mode: str = "normal"
    panel_count: int = 4
    tension_mean: float = 0.5
    load_mean: float = 0.5
    rhythm_count: int | None = None
    rhythm_emphasis: bool = False


@dataclass
class _Slot:
    beat: StructuralBeat | None
    secondary: list[StructuralBeat]
    t: float
    tension: float
    large: bool = False
    hook: bool = False
    pause: bool = False
    spread_role: str = "none"


# ---------------------------------------------------------------------------
# レイアウト実現（生成器と修復器で共用）
# ---------------------------------------------------------------------------


def _outer_side(side: PageSide) -> str:
    return "right" if side == PageSide.RIGHT else "left"


def _inner_side(side: PageSide) -> str:
    return "left" if side == PageSide.RIGHT else "right"


def _importance(panel: Panel) -> float:
    return 0.6 + panel.tension + (0.4 if panel.beat_id else 0.0) + (0.3 if panel.emphasis != "normal" else 0.0)


def layout_page(page: Page, storyboard: Storyboard, *, allow_slant: bool = True, pattern: int = 0) -> None:
    """ページ内コマの多角形・形状・断ち切り・ぶち抜きを、現在のコマ順と強調指定から再計算する。"""
    geometry = storyboard.geometry
    frame = geometry.frame_rect(page.side)
    x0, y0, x1, y1 = frame
    panels = page.ordered_panels()
    if not panels:
        return
    outer = _outer_side(page.side)
    inner = _inner_side(page.side)
    if len(panels) == 1 and (panels[0].emphasis in ("splash", "spread") or panels[0].spread_role == "main"):
        panel = panels[0]
        panel.polygon = Polygon.rect(x0, y0, x1, y1)
        panel.shape = PanelShape.RECT
        panel.bleed_sides = ["top", "bottom", "left", "right"]
        panel.breakout = panel.tension >= 0.8
        return
    has_continuation = panels[0].spread_role == "continuation"
    flags = [p.emphasis in ("large", "splash", "spread") for p in panels]
    body_panels = panels[1:] if has_continuation else panels
    body_flags = flags[1:] if has_continuation else flags
    large_positions = [i for i, f in enumerate(body_flags) if f]
    max_large = 1 if has_continuation else 2
    if len(large_positions) > max_large:
        keep = set(sorted(large_positions, key=lambda i: -body_panels[i].tension)[:max_large])
        for i in large_positions:
            if i not in keep:
                body_flags[i] = False
                body_panels[i].emphasis = "normal"
    flags = ([True] if has_continuation else []) + list(body_flags)
    n_large = sum(1 for f in body_flags if f)
    max_tiers = 3 if (n_large >= 2 or has_continuation) else 4
    body_tiers = geo.plan_tiers(body_flags, max_tiers=max_tiers - (1 if has_continuation else 0), capacity=3, pattern=pattern)
    offset = 1 if has_continuation else 0
    tiers: list[list[int]] = ([[0]] if has_continuation else []) + [[i + offset for i in tier] for tier in body_tiers]
    tier_large = [any(flags[i] for i in tier) if not (has_continuation and t == 0) else True for t, tier in enumerate(tiers)]
    count = len(tiers)
    frame_h = y1 - y0
    available = frame_h - geometry.gutter_y * max(0, count - 1)
    large_min = min(0.9, 0.415 * frame_h / max(1.0, available)) if count > 1 else 1.0
    weights = [max(_importance(panels[i]) for i in tier) for tier in tiers]
    fixed = {0: 0.45} if has_continuation and count > 1 else None
    fracs = geo.tier_fractions(weights, tier_large, large_min=large_min, normal_min=0.13, fixed=fixed)
    slants: list[float] = []
    for t, tier in enumerate(tiers):
        tier_tension = max(panels[i].tension for i in tier)
        if allow_slant and len(tier) >= 2 and not tier_large[t] and page.tension_mean >= 0.62:
            magnitude = clamp((page.tension_mean - 0.55) / 0.4, 0.0, 1.0) * 60.0
            direction = 1.0 if (t + pattern) % 2 == 0 else -1.0
            slants.append(round(direction * magnitude * (0.7 + 0.3 * tier_tension), 2))
        else:
            slants.append(0.0)
    tiers_weights = [[_importance(panels[i]) for i in tier] for tier in tiers]
    polygons = geo.layout_tiers(frame, tiers_weights, fracs, gutter_x=geometry.gutter_x, gutter_y=geometry.gutter_y, slants=slants)
    for t, tier in enumerate(tiers):
        for j, panel_index in enumerate(tier):
            panel = panels[panel_index]
            panel.polygon = Polygon.from_tuples([(round(x, 2), round(y, 2)) for x, y in polygons[t][j]])
            panel.shape = PanelShape.SLANTED if abs(slants[t]) > 1e-6 else PanelShape.RECT
            sides: list[str] = []
            is_large = tier_large[t]
            touches_outer = (j == 0 and outer == "right") or (j == len(tier) - 1 and outer == "left")
            if panel.spread_role == "continuation":
                sides = ["top", inner, outer]
            elif is_large:
                if t == 0:
                    sides.append("top")
                if t == count - 1:
                    sides.append("bottom")
                sides.append(outer)
            elif panel.tension >= 0.8 and touches_outer:
                sides.append(outer)
            panel.bleed_sides = sides  # type: ignore[assignment]
            panel.breakout = bool(is_large and panel.tension >= 0.85 and panel.spread_role != "continuation")
            if is_large and panel.emphasis == "normal" and panel.spread_role == "none":
                panel.emphasis = "large"


# ---------------------------------------------------------------------------
# カメラ・演技・台詞・フキダシ
# ---------------------------------------------------------------------------


def _beat_key(beat: StructuralBeat | None) -> str | None:
    return beat.beat_type.value if beat is not None else None


def _shot_preferences(panel: Panel, beat: StructuralBeat | None) -> list[ShotSize]:
    key = _beat_key(beat)
    if key in SHOT_PREFERENCES:
        prefs = list(SHOT_PREFERENCES[key])
    elif "page_turn_hook" in panel.prompt_hints:
        prefs = [S.CU, S.ECU, S.MS]
    elif panel.is_silent:
        prefs = [S.ELS, S.FS, S.CU] if panel.tension < 0.55 else [S.CU, S.ECU, S.FS]
    elif panel.tension >= 0.7:
        prefs = [S.CU, S.FS, S.ECU]
    else:
        prefs = [S.MS, S.CU, S.FS]
    for shot in (S.MS, S.CU, S.FS, S.ELS, S.ECU):
        if shot not in prefs:
            prefs.append(shot)
    return prefs


def _angle_preferences(panel: Panel, beat: StructuralBeat | None) -> list[CameraAngle]:
    key = _beat_key(beat)
    if key in ANGLE_PREFERENCES:
        prefs = list(ANGLE_PREFERENCES[key])
        if key == "hook" and panel.tension < 0.5:
            prefs = [A.EYE_LEVEL, A.HIGH, A.DUTCH]
    elif panel.tension >= 0.7:
        prefs = [A.LOW, A.DUTCH, A.EYE_LEVEL]
    elif panel.tension <= 0.35:
        prefs = [A.EYE_LEVEL, A.HIGH, A.LOW]
    else:
        prefs = [A.EYE_LEVEL, A.LOW, A.HIGH]
    for angle in (A.EYE_LEVEL, A.HIGH, A.LOW, A.DUTCH):
        if angle not in prefs:
            prefs.append(angle)
    return prefs


def _lens_for(shot: ShotSize, tension: float, key: str | None, rng: random.Random) -> int:
    base = {S.ECU: 105, S.CU: 85, S.MS: 50, S.FS: 35, S.ELS: 24}[shot]
    if shot in (S.FS, S.ELS, S.MS) and tension >= 0.75 and (key in ACTION_BEATS or key is None):
        base = 18 if shot == S.ELS else 24
    if shot in (S.CU, S.ECU) and tension >= 0.7:
        base = 120 if shot == S.CU else 135
    jitter = rng.choice([-5, 0, 0, 5])
    return int(clamp(base + jitter, 18, 135))


def _composition_for(shot: ShotSize, angle: CameraAngle, panel: Panel, beat: StructuralBeat | None) -> CompositionRule:
    if angle == A.DUTCH:
        return CompositionRule.DIAGONAL
    if shot == S.ELS:
        return CompositionRule.NEGATIVE_SPACE if panel.tension < 0.5 else CompositionRule.LEADING_LINES
    if shot == S.FS:
        if len(panel.characters) >= 2 and panel.tension >= 0.7:
            return CompositionRule.SYMMETRY
        return CompositionRule.DIAGONAL if panel.tension >= 0.6 else CompositionRule.RULE_OF_THIRDS
    if shot == S.MS:
        if beat is not None and beat.information_shift < -0.15:
            return CompositionRule.FRAME_IN_FRAME
        return CompositionRule.RULE_OF_THIRDS
    if shot == S.CU:
        return CompositionRule.GOLDEN_SPIRAL if panel.tension >= 0.6 else CompositionRule.CENTERED
    return CompositionRule.CENTERED


def _pick(prefs: list[Any], history: list[Any]) -> Any:
    for value in prefs:
        if not (len(history) >= 2 and history[-1] == value and history[-2] == value):
            return value
    return prefs[0]


def plan_camera(storyboard: Storyboard, structure: UniversalStructure | None, seed: int = 0) -> None:
    """全コマのカメラワークを読み順で決定する（同一ショット/アングルの3連続を生成時点で回避）。"""
    rng = random.Random(stable_seed("caf.camera", storyboard.structure_id, seed))
    beats = {b.id: b for b in structure.beats.beats} if structure is not None else {}
    shots: list[ShotSize] = []
    angles: list[CameraAngle] = []
    mains: dict[str, CameraWork] = {}
    for panel in storyboard.panels_in_reading_order():
        if panel.spread_role == "continuation":
            continue
        beat = beats.get(panel.beat_id) if panel.beat_id else None
        shot_prefs = _shot_preferences(panel, beat)
        angle_prefs = _angle_preferences(panel, beat)
        if beat is None and rng.random() < 0.35:
            shot_prefs[0], shot_prefs[1] = shot_prefs[1], shot_prefs[0]
        if beat is None and rng.random() < 0.3:
            angle_prefs[0], angle_prefs[1] = angle_prefs[1], angle_prefs[0]
        shot = _pick(shot_prefs, shots)
        angle = _pick(angle_prefs, angles)
        key = _beat_key(beat)
        dutch = 0.0
        if angle == A.DUTCH:
            dutch = round((10.0 + 20.0 * panel.tension) * (1 if rng.random() < 0.5 else -1), 1)
        panel.camera = CameraWork(
            shot=shot,
            angle=angle,
            lens_mm=_lens_for(shot, panel.tension, key, rng),
            composition=_composition_for(shot, angle, panel, beat),
            dutch_degrees=dutch,
            notes=_camera_note(shot, angle, key),
        )
        shots.append(shot)
        angles.append(angle)
        if panel.spread_role == "main":
            mains[panel.beat_id or panel.id] = panel.camera
    for panel in storyboard.panels_in_reading_order():
        if panel.spread_role == "continuation":
            source = mains.get(panel.beat_id or "")
            if source is not None:
                panel.camera = source.model_copy()


def _camera_note(shot: ShotSize, angle: CameraAngle, key: str | None) -> str:
    reasons = {
        A.HIGH: "俯瞰で無力さ・孤立を示す",
        A.LOW: "アオリで威圧・決意を示す",
        A.DUTCH: "ダッチで不安定さ・動揺を示す",
        A.EYE_LEVEL: "アイレベルで観察的に見せる",
    }
    return f"{shot.value}/{reasons[angle]}" + (f"（{key}）" if key else "")


def enforce_camera_variety(storyboard: Storyboard) -> int:
    """読み順で同一ショット/アングルが3連続しないよう最小変更で修正し、変更数を返す。"""
    panels = [p for p in storyboard.panels_in_reading_order() if p.spread_role != "continuation"]
    changes = 0
    order = [S.ECU, S.CU, S.MS, S.FS, S.ELS]
    for attribute in ("shot", "angle"):
        for i in range(2, len(panels)):
            a = getattr(panels[i - 2].camera, attribute)
            b = getattr(panels[i - 1].camera, attribute)
            c = getattr(panels[i].camera, attribute)
            if not (a == b == c):
                continue
            nxt = getattr(panels[i + 1].camera, attribute) if i + 1 < len(panels) else None
            if attribute == "shot":
                idx = order.index(c)
                candidates = [order[j] for j in (idx + 1, idx - 1, idx + 2, idx - 2) if 0 <= j < len(order)]
                if panels[i].tension >= 0.6:
                    candidates.sort(key=lambda s: order.index(s))
                new_value = next(v for v in candidates if v != c and v != nxt) if any(v != c and v != nxt for v in candidates) else candidates[0]
                panels[i].camera = panels[i].camera.model_copy(
                    update={"shot": new_value, "lens_mm": {S.ECU: 105, S.CU: 85, S.MS: 50, S.FS: 35, S.ELS: 24}[new_value]}
                )
            else:
                pool = [A.EYE_LEVEL, A.HIGH, A.LOW, A.DUTCH]
                if c != A.EYE_LEVEL:
                    pool = [A.EYE_LEVEL] + [v for v in pool if v != A.EYE_LEVEL]
                elif panels[i].tension >= 0.6:
                    pool = [A.LOW, A.DUTCH, A.HIGH, A.EYE_LEVEL]
                else:
                    pool = [A.HIGH, A.LOW, A.DUTCH, A.EYE_LEVEL]
                new_angle = next(v for v in pool if v != c and v != nxt)
                update: dict[str, Any] = {"angle": new_angle}
                update["dutch_degrees"] = round(10.0 + 20.0 * panels[i].tension, 1) if new_angle == A.DUTCH else 0.0
                panels[i].camera = panels[i].camera.model_copy(update=update)
            changes += 1
    mains = {p.beat_id: p.camera for p in panels if p.spread_role == "main"}
    for panel in storyboard.panels_in_reading_order():
        if panel.spread_role == "continuation" and panel.beat_id in mains:
            panel.camera = mains[panel.beat_id].model_copy()
    return changes


def _face_points(panel: Panel) -> dict[str, tuple[float, float, float]]:
    x0, y0, x1, y1 = panel.polygon.core_box()
    width, height = x1 - x0, y1 - y0
    faces: dict[str, tuple[float, float, float]] = {}
    for character in panel.characters:
        fx = x0 + character.position.x * width
        fy = y0 + max(0.05, character.position.y - 0.3 * character.scale) * height
        radius = max(16.0, 0.17 * character.scale * min(width, height))
        faces[character.character_id] = (fx, fy, radius)
    return faces


def layout_balloons(panel: Panel, config: StoryboardConfig) -> None:
    """コマ内のフキダシ（列・寸法・中心・尾・読み順）を再計算する。"""
    if not panel.balloons:
        return
    faces = _face_points(panel)
    sizes: list[tuple[float, float]] = []
    for balloon in panel.balloons:
        balloon.columns = break_japanese_text(balloon.text, config.max_chars_per_column)
        sizes.append(balloon_size(balloon.columns, config.font_size, balloon.kind.value))
    placements = place_balloons(panel.polygon.core_box(), sizes, list(faces.values()))
    for order, (balloon, (cx, cy, w, h)) in enumerate(zip(panel.balloons, placements)):
        balloon.center = Point(x=cx, y=cy)
        balloon.width = w
        balloon.height = h
        balloon.reading_order = order
        target = faces.get(balloon.speaker_id or "")
        if balloon.kind in (BalloonKind.NORMAL, BalloonKind.SHOUT) and target is not None:
            fx, fy, radius = target
            inside = abs(fx - cx) <= w / 2.0 and abs(fy - cy) <= h / 2.0
            balloon.tail_target = None if inside else Point(x=round(fx, 2), y=round(fy + radius * 0.5, 2))
        else:
            balloon.tail_target = None


def place_sfx(panel: Panel, beat_key: str | None) -> None:
    """高緊張のアクションコマにオノマトペを配置する。"""
    panel.sfx = []
    if panel.spread_role == "continuation":
        return
    if panel.tension < 0.72 or (beat_key is not None and beat_key not in ACTION_BEATS):
        return
    seed = stable_seed("caf.sfx", panel.id)
    rng = random.Random(seed)
    if beat_key is None and (panel.tension < 0.8 or rng.random() < 0.5):
        return
    if beat_key in ("costly_breakthrough", "climax", "escalation"):
        family = "impact" if panel.tension >= 0.8 else "motion"
    elif beat_key in ("revelation", "structural_twist", "crisis"):
        family = "emotion" if beat_key != "crisis" else "ambient"
    else:
        family = "ambient"
    text, style = rng.choice(SFX_LIBRARY[family])
    x0, y0, x1, y1 = panel.polygon.core_box()
    px = x0 + (x1 - x0) * rng.uniform(0.18, 0.36)
    py = y0 + (y1 - y0) * rng.uniform(0.62, 0.82)
    direction = (-0.7, 0.7) if family == "impact" else (-1.0, 0.15) if family == "motion" else (0.0, 1.0)
    panel.sfx.append(
        SoundEffect(
            id=f"{panel.id}_s0",
            text=text,
            position=Point(x=round(px, 2), y=round(py, 2)),
            direction=Point(x=direction[0], y=direction[1]),
            scale=round(0.8 + panel.tension, 3),
            rotation_deg=round(rng.uniform(-16.0, 16.0), 2),
            style=style,  # type: ignore[arg-type]
        )
    )


def _stage_panel(panel: Panel, beat: StructuralBeat | None) -> None:
    """ショットに応じた人物の位置・スケール・視線・表情・ポーズ。"""
    if not panel.characters:
        return
    shot = panel.camera.shot
    if shot == S.ECU and len(panel.characters) > 1:
        panel.characters = panel.characters[:1]
    count = len(panel.characters)
    scale = {S.ECU: 1.0, S.CU: 0.85, S.MS: 0.62, S.FS: 0.8, S.ELS: 0.28}[shot]
    if count >= 2 and shot not in (S.ECU,):
        scale *= 0.9
    y = {S.ECU: 0.5, S.CU: 0.52, S.MS: 0.56, S.FS: 0.55, S.ELS: 0.68}[shot]
    if count == 1:
        if panel.camera.composition in (CompositionRule.CENTERED, CompositionRule.SYMMETRY):
            xs = [0.5]
        else:
            xs = [0.62 if panel.index_in_page % 2 == 0 else 0.4]
    elif count == 2:
        xs = [0.7, 0.3]
    else:
        xs = [0.78, 0.5, 0.22][:count]
    key = _beat_key(beat)
    if key is None:
        expression = "tense" if panel.tension >= 0.7 else "thoughtful" if panel.tension >= 0.4 else "relaxed"
        pose = "bracing" if panel.tension >= 0.7 else "standing"
    else:
        expression = EXPRESSIONS.get(key, "neutral")
        pose = POSES.get(key, "standing")
    staged: list[CharacterAction] = []
    for index, character in enumerate(panel.characters):
        if count == 1:
            gaze = (0.0, 1.0) if shot == S.ECU else (-0.85, 0.25)
        else:
            gaze = (-1.0, 0.05) if index == 0 else (1.0, 0.05)
        staged.append(
            CharacterAction(
                character_id=character.character_id,
                position=Point(x=xs[min(index, len(xs) - 1)], y=y),
                scale=round(clamp(scale, 0.05, 1.0), 3),
                pose=pose if index == 0 else ("facing the other" if count > 1 else pose),
                expression=expression if index == 0 else ("stern" if panel.tension >= 0.6 else "attentive"),
                gaze=Point(x=gaze[0], y=gaze[1]),
                action=character.action,
            )
        )
    panel.characters = staged


def template_lines(
    key: str | None,
    tension: float,
    *,
    rng: random.Random,
    theme_word: str,
    hook: bool = False,
    variant_offset: int = 0,
    phase: str | None = None,
    usage: dict[str, int] | None = None,
) -> list[tuple[str, str]]:
    """ビート種別・局面・緊張度からテンプレート台詞（種別, 本文）を選ぶ。

    usage を渡すと作品内で使用回数の少ない言い回しを優先し、同じ台詞の反復を避ける。
    """
    if hook:
        pool = HOOK_TEMPLATES
    elif key in DIALOGUE_TEMPLATES:
        pool = DIALOGUE_TEMPLATES[key]
    elif phase in PHASE_REACTIONS:
        pool = PHASE_REACTIONS[phase]
    else:
        band = "high" if tension >= 0.7 else "mid" if tension >= 0.4 else "low"
        pool = CONNECTIVE_TEMPLATES[band]
    start = (rng.randrange(len(pool)) + variant_offset) % len(pool)
    order = [(start + k) % len(pool) for k in range(len(pool))]
    if usage is not None:
        rank = {index: position for position, index in enumerate(order)}
        order = sorted(order, key=lambda i: (sum(usage.get(text, 0) for _, text in pool[i]), rank[i]))
    index = order[0]
    if usage is not None:
        for _, text in pool[index]:
            usage[text] = usage.get(text, 0) + 1
    lines: list[tuple[str, str]] = []
    for kind, text in pool[index]:
        rendered = text.replace("{THEME}", theme_word)
        if kind == "shout" and tension < 0.55:
            kind = "normal"
        lines.append((kind, rendered))
    return lines


def _phase_key(structure: UniversalStructure, t: float) -> str | None:
    """時刻 t の局面（直前のビート種別。無ければ最初のビート）。"""
    beats = sorted(structure.beats.beats, key=lambda b: b.t)
    if not beats:
        return None
    previous = [b for b in beats if b.t <= t + 1e-9]
    return (previous[-1] if previous else beats[0]).beat_type.value


def _theme_word(theme: dict[str, str]) -> str:
    for key in ("THEME", "theme", "テーマ"):
        if theme.get(key):
            return theme[key].strip()[:10]
    for value in theme.values():
        if value.strip():
            return value.strip()[:10]
    return "あの場所"


def _make_balloons(panel: Panel, lines: list[tuple[str, str]], config: StoryboardConfig) -> list[Balloon]:
    speakers = [c.character_id for c in panel.characters]
    balloons: list[Balloon] = []
    for line_index, (kind, text) in enumerate(lines):
        speaker = speakers[line_index % len(speakers)] if speakers else None
        for piece in split_text_for_balloons(text, config.max_chars_per_balloon):
            if len(balloons) >= config.max_balloons_per_panel:
                break
            balloons.append(
                Balloon(
                    id=f"{panel.id}_b{len(balloons)}",
                    kind=BalloonKind(kind),
                    speaker_id=speaker,
                    text=piece,
                    columns=break_japanese_text(piece, config.max_chars_per_column),
                    center=panel.polygon.centroid(),
                    width=60.0,
                    height=80.0,
                    reading_order=len(balloons),
                )
            )
    return balloons


def refresh_page_metrics(page: Page) -> None:
    """ページの情報密度・緊張平均・ベタ比率目標を更新する。"""
    panels = page.panels
    page.information_density = page_information_density(page)
    page.tension_mean = round(float(np.mean([p.tension for p in panels])) if panels else 0.0, 4)
    emphasized = any(p.emphasis != "normal" for p in panels)
    page.black_ratio_target = round(clamp(0.12 + 0.55 * page.tension_mean + (0.08 if emphasized else 0.0), 0.08, 0.85), 4)


def finalize_reading_order(storyboard: Storyboard) -> None:
    """ページ内・全体の読み順インデックスを確定する。"""
    counter = 0
    storyboard.pages.sort(key=lambda p: p.index)
    for page in storyboard.pages:
        ordered = page.ordered_panels()
        for i, panel in enumerate(ordered):
            panel.index_in_page = i
            panel.page_index = page.index
            panel.reading_order = counter
            counter += 1
        page.panels = ordered


# ---------------------------------------------------------------------------
# 台詞の LLM 増幅（バッチ構築・適用）
# ---------------------------------------------------------------------------


def build_dialogue_batch(panels: list[dict[str, Any]], *, seed: int = 0, max_chars: int = 40, variant_offset: int = 1) -> PanelDialogueBatch:
    """コマ情報から台詞バッチを決定論的に構築する（モックLLMの応答・オフライン増幅に使用）。"""
    entries: list[PanelDialogue] = []
    for info in panels:
        panel_id = str(info.get("panel_id"))
        rng = random.Random(stable_seed("caf.dialogue", panel_id, seed))
        if info.get("silent"):
            entries.append(PanelDialogue(panel_id=panel_id, lines=[], silent=True))
            continue
        lines = template_lines(
            info.get("beat_type"),
            float(info.get("tension", 0.5)),
            rng=rng,
            theme_word=str(info.get("theme_word", "あの場所")),
            hook=bool(info.get("hook")),
            variant_offset=variant_offset,
        )
        speakers = [c.get("id") for c in info.get("characters", []) if c.get("id")]
        dialogue_lines = []
        for index, (kind, text) in enumerate(lines[:3]):
            clipped = text if visible_length(text) <= max_chars else text[:max_chars]
            dialogue_lines.append(
                DialogueLine(speaker_id=speakers[index % len(speakers)] if speakers else None, kind=BalloonKind(kind), text=clipped)
            )
        entries.append(PanelDialogue(panel_id=panel_id, lines=dialogue_lines, silent=not dialogue_lines))
    if not entries:
        entries.append(PanelDialogue(panel_id="none", lines=[], silent=True))
    return PanelDialogueBatch(storyboard_title="", panels=entries)


def apply_dialogue_batch(storyboard: Storyboard, batch: PanelDialogueBatch, config: StoryboardConfig) -> int:
    """台詞バッチをネームへ適用し、変更したコマ数を返す。"""
    changed = 0
    for entry in batch.panels:
        panel = storyboard.panel(entry.panel_id)
        if panel is None or panel.spread_role == "continuation":
            continue
        valid_speakers = {c.character_id for c in panel.characters}
        if entry.silent or not entry.lines:
            if panel.emphasis == "normal" and not panel.beat_id:
                panel.balloons = []
                panel.is_silent = True
                changed += 1
            continue
        lines = [(line.kind.value, line.text) for line in entry.lines]
        panel.balloons = _make_balloons(panel, lines, config)
        for balloon, line in zip(panel.balloons, entry.lines):
            if line.speaker_id in valid_speakers:
                balloon.speaker_id = line.speaker_id
        panel.is_silent = not panel.balloons
        layout_balloons(panel, config)
        changed += 1
    return changed


# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------


class StoryboardGenerator:
    """構造駆動ネーム生成器。"""

    def __init__(self, config: StoryboardConfig | None = None, llm_client: Any | None = None) -> None:
        self.config = config or StoryboardConfig()
        self.llm_client = llm_client
        self.validator = MangaGrammarValidator(self.config)

    # -- 人物 ------------------------------------------------------------------

    def _build_characters(
        self, structure: UniversalStructure, theme: dict[str, str], characters: list[CharacterProfile] | None
    ) -> list[CharacterProfile]:
        if characters:
            return [c.model_copy() for c in characters]
        graph = structure.tension_graph
        agents = sorted(graph.nodes_of_kind(NodeKind.AGENT), key=lambda n: (-n.weight, n.id))[:4]
        if not agents:
            agents = sorted(graph.nodes, key=lambda n: (-n.weight, n.id))[:1]
        protagonist = agents[0].id
        used_names: dict[str, int] = {}
        profiles: list[CharacterProfile] = []
        for node in agents:
            role_key = self._classify_role(node, protagonist, structure)
            base_name = _ROLE_NAMES[role_key]
            used_names[base_name] = used_names.get(base_name, 0) + 1
            name = base_name if used_names[base_name] == 1 else f"{base_name}{chr(ord('A') + used_names[base_name] - 1)}"
            rng = random.Random(stable_seed("caf.character", structure.id, node.id, self.config.seed))
            visual = ", ".join(
                [
                    _ROLE_VISUALS[role_key],
                    rng.choice(_TRAIT_HAIR),
                    rng.choice(_TRAIT_CLOTHES),
                    rng.choice(_TRAIT_FEATURES),
                ]
            )
            profiles.append(
                CharacterProfile(
                    id=node.id,
                    role=role_key,
                    name_placeholder=name,
                    visual_description=visual,
                    seed=stable_seed("caf.character.seed", structure.id, node.id),
                )
            )
        return profiles

    @staticmethod
    def _classify_role(node: Any, protagonist_id: str, structure: UniversalStructure) -> str:
        if node.id == protagonist_id:
            return "protagonist"
        role_text = f"{node.role} {node.label}".lower()
        hostile = False
        supportive = False
        for edge in structure.tension_graph.edges:
            if {edge.source, edge.target} != {node.id, protagonist_id}:
                continue
            if edge.kind in (EdgeKind.OPPOSITION, EdgeKind.BETRAYAL):
                hostile = True
            elif edge.kind == EdgeKind.POWER_GRADIENT and edge.source == node.id and edge.gradient > 0.3:
                hostile = True
            elif edge.kind in (EdgeKind.DEPENDENCY, EdgeKind.PURSUIT, EdgeKind.ASYMMETRY):
                supportive = True
        if any(word in role_text for word in ("mentor", "guardian", "guide", "師", "導")):
            return "mentor"
        if hostile:
            return "antagonist"
        if supportive:
            return "ally"
        return "other"

    # -- ページ計画 --------------------------------------------------------------

    def _rhythm_map(self, structure: UniversalStructure, count: int, roles: list[PageTurnRole]) -> dict[str, Any] | None:
        profile = structure.rhythm_profile
        if profile is None:
            return None
        source_pages = profile.page_count
        reveal_pages = [i for i, role in enumerate(roles) if role == PageTurnRole.REVEAL]

        def to_ours(index: int) -> int:
            if source_pages <= 1:
                return 0
            return int(round(index * (count - 1) / (source_pages - 1)))

        def nearest_reveal(index: int) -> int | None:
            if not reveal_pages:
                return None
            return min(reveal_pages, key=lambda r: (abs(r - index), r))

        counts = []
        for k in range(count):
            source = int(round(k * (source_pages - 1) / max(1, count - 1))) if count > 1 else 0
            counts.append(profile.panels_per_page[min(source, source_pages - 1)])
        reveals = {r for r in (nearest_reveal(to_ours(i)) for i in profile.page_turn_reveals) if r is not None}
        spread = nearest_reveal(to_ours(profile.spread_pages[0])) if profile.spread_pages else None
        return {
            "counts": counts,
            "emphasis": {to_ours(i) for i in profile.emphasis_pages},
            "reveals": reveals | ({spread} if spread is not None else set()),
            "spread": spread,
        }

    def _plans(self, structure: UniversalStructure) -> tuple[list[_PagePlan], str | None, list[str]]:
        cfg = self.config
        count = cfg.target_pages
        sides = page_sides(count, cfg.first_page_side)
        roles = page_roles(sides)
        peak = identify_peak_beat(structure)
        peak_id = peak.id if peak is not None else None
        emphasis_ids = emphasis_beat_ids(structure)
        rhythm = self._rhythm_map(structure, count, roles)
        bounds, anchors = allocate_pages(
            structure,
            count,
            roles,
            emphasis_ids=emphasis_ids,
            peak_id=peak_id,
            preferred_reveals=rhythm["reveals"] if rhythm else None,
        )
        anchor_page_of_beat = {beat_id: page for page, beat_id in anchors.items()}
        spread_page: int | None = None
        if peak_id in anchor_page_of_beat and count >= 8:
            p = anchor_page_of_beat[peak_id]
            if p + 1 < count and sides[p] == PageSide.RIGHT and sides[p + 1] == PageSide.LEFT:
                spread_page = p
                upper = bounds[p + 2] if p + 2 <= count else 1.0
                peak_t = peak.t if peak is not None else bounds[p]
                candidate = peak_t + 0.001
                bounds[p + 1] = candidate if candidate < upper - 1e-4 else (bounds[p] + upper) / 2.0
        wave = structure.wave
        plans: list[_PagePlan] = []
        beats_sorted = sorted(structure.beats.beats, key=lambda b: b.t)
        for k in range(count):
            t0, t1 = bounds[k], bounds[k + 1]
            inside = [b for b in beats_sorted if (t0 <= b.t < t1) or (k == count - 1 and b.t >= t0)]
            samples = np.linspace(t0, t1, 9)
            plan = _PagePlan(
                index=k,
                side=sides[k],
                role=roles[k],
                t_start=round(t0, 6),
                t_end=round(t1, 6),
                beats=inside,
                anchor_beat_id=anchors.get(k),
                tension_mean=float(np.mean(np.interp(samples, wave.t, wave.tension))),
                load_mean=float(np.mean(np.interp(samples, wave.t, wave.cognitive_load))),
                rhythm_count=rhythm["counts"][k] if rhythm else None,
                rhythm_emphasis=bool(rhythm and k in rhythm["emphasis"]),
            )
            if spread_page is not None and k == spread_page:
                plan.mode = "spread_main"
            elif spread_page is not None and k == spread_page + 1:
                plan.mode = "spread_cont"
            plans.append(plan)
        if spread_page is None and peak_id is not None:
            for plan in plans:
                if any(b.id == peak_id for b in plan.beats) and len(plan.beats) == 1 and plan.mode == "normal":
                    plan.mode = "splash"
        for plan in plans:
            plan.panel_count = self._panel_count(plan)
        self._modulate_counts(plans, emphasis_ids)
        return plans, peak_id, emphasis_ids

    def _panel_count(self, plan: _PagePlan) -> int:
        cfg = self.config
        if plan.mode in ("spread_main", "splash"):
            return 1
        lo, hi = cfg.min_panels_per_page, cfg.max_panels_per_page
        base = hi - (hi - lo) * (plan.tension_mean ** 1.15)
        if plan.load_mean > 0.62:
            base += 0.8
        if plan.anchor_beat_id:
            base -= 0.7
        if plan.rhythm_emphasis:
            base -= 1.0
        if plan.rhythm_count is not None:
            base = 0.6 * plan.rhythm_count + 0.4 * base
        count = int(round(base))
        if plan.mode == "spread_cont":
            return int(clamp(count, 1, 4))
        return int(clamp(count, lo, hi))

    def _modulate_counts(self, plans: list[_PagePlan], emphasis_ids: list[str]) -> None:
        """連続3ページのコマ数が平板にならないよう中間ページを増減する（緩急の付与）。"""
        lo, hi = self.config.min_panels_per_page, self.config.max_panels_per_page
        for _ in range(3):
            changed = False
            for i in range(len(plans) - 2):
                window = [plans[j].panel_count for j in range(i, i + 3)]
                if max(window) - min(window) >= 2:
                    continue
                middle = plans[i + 1]
                if middle.mode != "normal":
                    continue
                required = max(1, len(middle.beats))
                neighbors = (window[0] + window[2]) / 2.0
                options = []
                if middle.panel_count >= neighbors:
                    options = [middle.panel_count + 2, middle.panel_count - 2, middle.panel_count + 1]
                else:
                    options = [middle.panel_count - 2, middle.panel_count + 2, middle.panel_count - 1]
                for option in options:
                    if max(lo, min(required, hi)) <= option <= hi:
                        middle.panel_count = option
                        changed = True
                        break
            if not changed:
                break

    # -- スロット ---------------------------------------------------------------

    def _build_slots(self, plan: _PagePlan, structure: UniversalStructure, emphasis_ids: set[str], peak_id: str | None, next_plan: _PagePlan | None) -> list[_Slot]:
        cfg = self.config
        wave = structure.wave
        beats = list(plan.beats)
        if plan.mode == "spread_main":
            peak_beat = next((b for b in beats if b.id == peak_id), beats[0] if beats else None)
            secondary = [b for b in beats if peak_beat is not None and b.id != peak_beat.id]
            t = peak_beat.t if peak_beat is not None else plan.t_start
            return [_Slot(beat=peak_beat, secondary=secondary, t=t, tension=wave.sample("tension", t), large=True, spread_role="main")]
        slots_prefix: list[_Slot] = []
        if plan.mode == "spread_cont":
            peak_beat = next((b for b in structure.beats.beats if b.id == peak_id), None)
            slots_prefix.append(
                _Slot(beat=peak_beat, secondary=[], t=plan.t_start, tension=wave.sample("tension", plan.t_start), large=True, spread_role="continuation")
            )
        if plan.mode == "splash":
            beat = beats[0]
            return [_Slot(beat=beat, secondary=beats[1:], t=beat.t, tension=wave.sample("tension", beat.t), large=True)]
        max_large = 1 if plan.mode == "spread_cont" else 2
        emph = [b for b in beats if b.id in emphasis_ids]
        keep_large = {b.id for b in sorted(emph, key=lambda b: (-_beat_score(b, peak_id), b.t))[:max_large]}
        primaries: list[StructuralBeat] = []
        secondary: dict[str, list[StructuralBeat]] = {}
        for beat in beats:
            if beat.id in emphasis_ids and beat.id not in keep_large:
                host = min((b for b in beats if b.id in keep_large), key=lambda b: abs(b.t - beat.t))
                secondary.setdefault(host.id, []).append(beat)
            else:
                primaries.append(beat)
        n_large = sum(1 for b in primaries if b.id in keep_large)
        hi = cfg.max_panels_per_page - len(slots_prefix)
        if n_large >= 2:
            normal_cap = 3
        elif n_large == 1:
            normal_cap = max(1, min(hi - 1, 6))
        else:
            normal_cap = max(1, hi)
        if plan.mode == "spread_cont":
            normal_cap = min(normal_cap, 3)
        normal = [b for b in primaries if b.id not in keep_large]
        while len(normal) > normal_cap:
            victim = min(normal, key=lambda b: (_beat_score(b, peak_id), -b.t))
            index = primaries.index(victim)
            neighbors = [primaries[j] for j in (index - 1, index + 1) if 0 <= j < len(primaries)]
            host = neighbors[0] if neighbors else primaries[0]
            secondary.setdefault(host.id, []).append(victim)
            primaries.remove(victim)
            normal.remove(victim)
        if n_large >= 2:
            primaries = self._contiguous_normals(primaries, keep_large, secondary)
        target = plan.panel_count - len(slots_prefix)
        upper = n_large + normal_cap
        n_slots = int(clamp(target, max(1, len(primaries)), max(1, len(primaries), upper)))
        positions = self._slot_positions(plan, primaries, n_slots)
        slots: list[_Slot] = []
        by_position = {pos: beat for beat, pos in zip(primaries, positions)}
        for i in range(n_slots):
            beat = by_position.get(i)
            if beat is not None:
                t = beat.t
            else:
                t = plan.t_start + (i + 0.5) / n_slots * (plan.t_end - plan.t_start)
            slots.append(
                _Slot(
                    beat=beat,
                    secondary=secondary.get(beat.id, []) if beat is not None else [],
                    t=float(clamp(t, 0.0, 1.0)),
                    tension=float(wave.sample("tension", t)),
                    large=beat is not None and beat.id in keep_large,
                )
            )
        for i in range(1, len(slots)):
            if slots[i].t < slots[i - 1].t:
                slots[i].t = slots[i - 1].t
        if n_large >= 2:
            slots = self._drop_split_connectives(slots)
        for i, slot in enumerate(slots):
            if slot.beat is None and i + 1 < len(slots) and slots[i + 1].large:
                slot.pause = True
        if plan.role == PageTurnRole.HOOK and next_plan is not None and next_plan.anchor_beat_id and slots:
            last = slots[-1]
            if not last.large:
                last.hook = True
                last.tension = max(last.tension, float(wave.sample("tension", next_plan.t_start)) - 0.05)
        return slots_prefix + slots

    @staticmethod
    def _contiguous_normals(primaries: list[StructuralBeat], keep_large: set[str], secondary: dict[str, list[StructuralBeat]]) -> list[StructuralBeat]:
        """大ゴマ2つのページで通常コマを1つの連続区間に収める（他区間は隣の大ゴマへ統合）。"""
        runs: list[list[StructuralBeat]] = []
        current: list[StructuralBeat] = []
        for beat in primaries:
            if beat.id in keep_large:
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(beat)
        if current:
            runs.append(current)
        if len(runs) <= 1:
            return primaries
        keep_run = max(runs, key=len)
        result: list[StructuralBeat] = []
        for beat in primaries:
            if beat.id in keep_large or beat in keep_run:
                result.append(beat)
                continue
            larges = [b for b in primaries if b.id in keep_large]
            host = min(larges, key=lambda b: abs(b.t - beat.t))
            secondary.setdefault(host.id, []).append(beat)
        return result

    @staticmethod
    def _drop_split_connectives(slots: list[_Slot]) -> list[_Slot]:
        """大ゴマ2つのページで、通常区間を分断する接続コマを除去する。"""
        runs: list[list[int]] = []
        current: list[int] = []
        for i, slot in enumerate(slots):
            if slot.large:
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(i)
        if current:
            runs.append(current)
        if len(runs) <= 1:
            return slots
        keep = max(runs, key=lambda r: (sum(1 for i in r if slots[i].beat is not None), len(r)))
        keep_set = set(keep)
        return [slot for i, slot in enumerate(slots) if slot.large or i in keep_set or slot.beat is not None]

    @staticmethod
    def _slot_positions(plan: _PagePlan, beats: list[StructuralBeat], n_slots: int) -> list[int]:
        if not beats:
            return []
        span = max(1e-6, plan.t_end - plan.t_start)
        positions: list[int] = []
        last = -1
        for index, beat in enumerate(beats):
            remaining = len(beats) - index - 1
            if beat.id == plan.anchor_beat_id and index == 0:
                desired = 0
            else:
                desired = int(clamp(int((beat.t - plan.t_start) / span * n_slots), 0, n_slots - 1))
            position = max(last + 1, desired)
            position = min(position, n_slots - 1 - remaining)
            positions.append(position)
            last = position
        return positions

    # -- ページ実現 --------------------------------------------------------------

    def _realize_page(self, plan: _PagePlan, slots: list[_Slot], storyboard_id: str, profiles: list[CharacterProfile], structure: UniversalStructure, previous_cast: list[str]) -> tuple[Page, list[str]]:
        cfg = self.config
        panels: list[Panel] = []
        beat_lookup = {b.id: b for b in structure.beats.beats}
        protagonist = profiles[0].id if profiles else None
        cast_ids = {p.id for p in profiles}
        cast_memory = list(previous_cast)
        for index, slot in enumerate(slots):
            beat = slot.beat
            involved: list[str] = []
            for candidate in ([beat] if beat else []) + slot.secondary:
                for node_id in candidate.involved_nodes:
                    if node_id in cast_ids and node_id not in involved:
                        involved.append(node_id)
            if not involved:
                involved = cast_memory[:2] if cast_memory else ([protagonist] if protagonist else [])
            involved = involved[:3]
            cast_memory = involved or cast_memory
            emphasis = "normal"
            if slot.spread_role in ("main", "continuation"):
                emphasis = "spread"
            elif plan.mode == "splash":
                emphasis = "splash"
            elif slot.large:
                emphasis = "large"
            hints: list[str] = []
            key = beat.beat_type.value if beat else None
            if key:
                hints.extend(SCENE_HINTS.get(key, []))
            if slot.hook:
                hints.append("page_turn_hook")
            if slot.spread_role == "continuation":
                hints.append("spread_continuation")
            description = self._describe(beat, slot, beat_lookup)
            panel_id = f"pg{plan.index:02d}_pn{index:02d}"
            panels.append(
                Panel(
                    id=panel_id,
                    page_index=plan.index,
                    index_in_page=index,
                    polygon=Polygon.rect(0.0, 0.0, 1.0, 1.0),
                    camera=CameraWork(shot=S.MS),
                    characters=[
                        CharacterAction(character_id=cid, position=Point(x=0.5, y=0.55), action=(beat.label if beat else ""))
                        for cid in involved
                    ],
                    description=description,
                    beat_id=beat.id if beat else None,
                    secondary_beat_ids=[b.id for b in slot.secondary],
                    t=float(clamp(slot.t, 0.0, 1.0)),
                    tension=float(clamp(slot.tension, 0.0, 1.0)),
                    emphasis=emphasis,  # type: ignore[arg-type]
                    is_silent=slot.pause or slot.spread_role == "continuation",
                    spread_role=slot.spread_role,  # type: ignore[arg-type]
                    prompt_hints=hints,
                )
            )
        page = Page(
            index=plan.index,
            number=plan.index + 1,
            side=plan.side,
            panels=panels,
            page_turn_role=plan.role,
            t_start=float(clamp(plan.t_start, 0.0, 1.0)),
            t_end=float(clamp(plan.t_end, 0.0, 1.0)),
            tension_mean=float(clamp(plan.tension_mean, 0.0, 1.0)),
        )
        notes = []
        if plan.anchor_beat_id:
            notes.append(f"めくり直後の先頭コマで転換点ビート {plan.anchor_beat_id} を開示")
        if plan.role == PageTurnRole.HOOK and slots and slots[-1].hook:
            notes.append("ページ終端に次ページへの引きを配置")
        if plan.mode == "spread_main":
            notes.append("見開き大ゴマ（右ページ側）")
        if plan.mode == "spread_cont":
            notes.append("見開き大ゴマ（左ページ側の続き）")
        if plan.mode == "splash":
            notes.append("1ページ大ゴマ（splash）で緊張ピークを強調")
        page.notes = notes
        return page, cast_memory

    @staticmethod
    def _describe(beat: StructuralBeat | None, slot: _Slot, beat_lookup: dict[str, StructuralBeat]) -> str:
        if slot.spread_role == "continuation" and beat is not None:
            return f"見開きの続き: {beat.label}"
        if beat is not None:
            extra = f"（併せて: {'、'.join(b.label for b in slot.secondary)}）" if slot.secondary else ""
            return f"{beat.label} — {beat.function}{extra}"
        if slot.hook:
            return "引き: 次ページの開示を予感させる寄りのコマ"
        if slot.pause:
            return "溜め: 大ゴマ直前の無音の間"
        return "繋ぎ: 直前の出来事への反応と状況の推移"

    def _assign_dialogue(self, storyboard: Storyboard, structure: UniversalStructure, theme_word: str) -> None:
        beats = {b.id: b for b in structure.beats.beats}
        usage: dict[str, int] = {}
        for page in storyboard.pages:
            for panel in page.ordered_panels():
                if panel.spread_role == "continuation" or panel.is_silent:
                    panel.balloons = []
                    continue
                rng = random.Random(stable_seed("caf.dialogue", panel.id, self.config.seed))
                beat = beats.get(panel.beat_id) if panel.beat_id else None
                hook = "page_turn_hook" in panel.prompt_hints
                phase = _phase_key(structure, panel.t) if beat is None else None
                if beat is None and not hook and rng.random() < PHASE_SILENCE.get(phase or "", 0.25):
                    panel.is_silent = True
                    panel.balloons = []
                    continue
                lines = template_lines(
                    beat.beat_type.value if beat else None,
                    panel.tension,
                    rng=rng,
                    theme_word=theme_word,
                    hook=hook,
                    phase=phase,
                    usage=usage,
                )
                if panel.emphasis in ("large", "splash", "spread") and len(lines) > 1:
                    lines = lines[:1]
                panel.balloons = _make_balloons(panel, lines, self.config)
                panel.is_silent = not panel.balloons

    def _finish_page(self, page: Page, storyboard: Storyboard, structure: UniversalStructure) -> None:
        beats = {b.id: b for b in structure.beats.beats}
        for panel in page.panels:
            beat = beats.get(panel.beat_id) if panel.beat_id else None
            _stage_panel(panel, beat)
            layout_balloons(panel, self.config)
            place_sfx(panel, beat.beat_type.value if beat else None)
        refresh_page_metrics(page)

    def generate(
        self,
        structure: UniversalStructure,
        theme: dict[str, str] | None = None,
        characters: list[CharacterProfile] | None = None,
    ) -> Storyboard:
        """構造から決定論的にネームを生成する（文法監査は行わない）。"""
        cfg = self.config
        theme_map = {str(k): str(v) for k, v in (theme or {}).items()}
        plans, peak_id, emphasis_ids = self._plans(structure)
        profiles = self._build_characters(structure, theme_map, characters)
        storyboard_id = f"sb_{stable_seed('caf.storyboard', structure.id, cfg.seed, cfg.target_pages) % 16**10:010x}"
        pages: list[Page] = []
        cast: list[str] = []
        emphasis_set = set(emphasis_ids)
        for k, plan in enumerate(plans):
            slots = self._build_slots(plan, structure, emphasis_set, peak_id, plans[k + 1] if k + 1 < len(plans) else None)
            page, cast = self._realize_page(plan, slots, storyboard_id, profiles, structure, cast)
            pages.append(page)
        spreads: list[Spread] = []
        for page in pages:
            if page.side == PageSide.RIGHT and page.index + 1 < len(pages) and pages[page.index + 1].side == PageSide.LEFT:
                left = pages[page.index + 1]
                is_double = page.panels and page.panels[0].spread_role == "main"
                spread = Spread(
                    id=f"spread_{page.index:02d}_{left.index:02d}",
                    right_page=page.index,
                    left_page=left.index,
                    kind="double_page_panel" if is_double else "facing",
                )
                spreads.append(spread)
                page.spread_id = spread.id
                left.spread_id = spread.id
        title = structure.metadata.title
        if theme_map:
            title = f"{title}（{_theme_word(theme_map)}）"
        storyboard = Storyboard(
            id=storyboard_id,
            title=title[:200],
            structure_id=structure.id,
            theme=theme_map,
            first_page_side=cfg.first_page_side,
            pages=pages,
            spreads=spreads,
            characters=profiles,
            seed=cfg.seed,
        )
        for page in storyboard.pages:
            for panel in page.panels:
                panel.tension = float(clamp(panel.tension, 0.0, 1.0))
            refresh_page_metrics(page)
            layout_page(page, storyboard, pattern=stable_seed("caf.pattern", storyboard_id, page.index) % 2)
        self._assign_dialogue(storyboard, structure, _theme_word(theme_map))
        plan_camera(storyboard, structure, cfg.seed)
        for page in storyboard.pages:
            self._finish_page(page, storyboard, structure)
        finalize_reading_order(storyboard)
        storyboard.revision_log.append(
            f"初期生成: {storyboard.page_count}ページ / {len(storyboard.panels_in_reading_order())}コマ"
            f" / 見開き大ゴマ {sum(1 for s in spreads if s.kind == 'double_page_panel')} 箇所"
        )
        return Storyboard.model_validate(storyboard.model_dump())

    # -- 監査ループ --------------------------------------------------------------

    def _validate_loop(self, storyboard: Storyboard, structure: UniversalStructure) -> Storyboard:
        fixer = StoryboardFixer(self.config)
        current = storyboard
        report = self.validator.validate(current, structure)
        best, best_report = current, report
        iteration = 0
        while iteration < self.config.max_fix_iterations and not (report.passed and report.score >= 95.0):
            iteration += 1
            candidate = fixer.apply(current, report, structure)
            candidate_report = self.validator.validate(candidate, structure)
            candidate_report.iteration = iteration
            candidate.revision_log.append(
                f"修正 {iteration}: スコア {candidate_report.score:.1f} / 違反 {len(candidate_report.violations)} 件"
                f"（error {len(candidate_report.errors())}）"
            )
            current, report = candidate, candidate_report
            improved = (candidate_report.passed, candidate_report.score) > (best_report.passed, best_report.score)
            if improved:
                best, best_report = candidate, candidate_report
            elif best_report.passed:
                break
        current, report = best, best_report
        if not report.passed:
            current = fixer.safe_rebuild(current, structure)
            report = self.validator.validate(current, structure)
            report.iteration = iteration + 1
            current.revision_log.append(f"安全レイアウト再構成: スコア {report.score:.1f}")
            if not report.passed:
                current = fixer.safe_rebuild(current, structure, aggressive=True)
                report = self.validator.validate(current, structure)
                report.iteration = iteration + 2
                current.revision_log.append(f"安全レイアウト再構成（強）: スコア {report.score:.1f}")
        current.grammar = report
        return Storyboard.model_validate(current.model_dump())

    def generate_validated_sync(
        self,
        structure: UniversalStructure,
        theme: dict[str, str] | None = None,
        characters: list[CharacterProfile] | None = None,
    ) -> Storyboard:
        """生成 → 漫画文法監査 → 自動修正ループ（同期版、LLM 不使用）。"""
        return self._validate_loop(self.generate(structure, theme, characters), structure)

    async def generate_validated(
        self,
        structure: UniversalStructure,
        theme: dict[str, str] | None = None,
        characters: list[CharacterProfile] | None = None,
    ) -> Storyboard:
        """生成 →（任意）LLM 台詞増幅 → 漫画文法監査 → 自動修正ループ。"""
        storyboard = self.generate(structure, theme, characters)
        if self.config.use_llm_dialogue and self.llm_client is not None:
            storyboard = await self._enrich_dialogue(storyboard, structure)
        return self._validate_loop(storyboard, structure)

    def _dialogue_payload(self, storyboard: Storyboard, structure: UniversalStructure) -> list[dict[str, Any]]:
        beats = {b.id: b for b in structure.beats.beats}
        theme_word = _theme_word(storyboard.theme)
        payload: list[dict[str, Any]] = []
        for panel in storyboard.panels_in_reading_order():
            if panel.spread_role == "continuation":
                continue
            beat = beats.get(panel.beat_id) if panel.beat_id else None
            payload.append(
                {
                    "panel_id": panel.id,
                    "beat_type": beat.beat_type.value if beat else None,
                    "beat_label": beat.label if beat else "",
                    "beat_function": beat.function if beat else panel.description,
                    "tension": round(panel.tension, 4),
                    "emphasis": panel.emphasis,
                    "hook": "page_turn_hook" in panel.prompt_hints,
                    "silent": panel.is_silent,
                    "theme_word": theme_word,
                    "characters": [
                        {
                            "id": c.character_id,
                            "name": (storyboard.character(c.character_id).name_placeholder if storyboard.character(c.character_id) else c.character_id),
                            "role": (storyboard.character(c.character_id).role if storyboard.character(c.character_id) else ""),
                        }
                        for c in panel.characters
                    ],
                    "current_lines": [{"speaker_id": b.speaker_id, "kind": b.kind.value, "text": b.text} for b in panel.balloons],
                    "max_chars": self.config.max_chars_per_balloon,
                }
            )
        return payload

    async def _enrich_dialogue(self, storyboard: Storyboard, structure: UniversalStructure) -> Storyboard:
        from core.llm_client import HarnessError, MultiStageReasoningHarness, StructuredOutputError
        from core.manga_pipeline.prompts import DIALOGUE_RUBRIC, MANGA_DIALOGUE_SYSTEM_PROMPT

        payload = self._dialogue_payload(storyboard, structure)
        known_ids = {entry["panel_id"] for entry in payload}
        max_chars = self.config.max_chars_per_balloon

        def post_validate(batch: PanelDialogueBatch) -> list[str]:
            issues: list[str] = []
            for entry in batch.panels:
                if entry.panel_id not in known_ids:
                    issues.append(f"未知のコマID {entry.panel_id} が含まれています")
                for line in entry.lines:
                    if visible_length(line.text) > max_chars:
                        issues.append(f"コマ {entry.panel_id} の台詞が {max_chars} 字を超えています: {line.text[:20]}")
            return issues

        harness = MultiStageReasoningHarness(self.llm_client)
        try:
            result = await harness.run(
                task="manga.dialogue",
                objective="ネームの各コマの台詞を、演出意図と文字数制約を守って増幅・推敲する",
                payload={"title": storyboard.title, "panels": payload},
                output_model=PanelDialogueBatch,
                system_prompt=MANGA_DIALOGUE_SYSTEM_PROMPT,
                rubric=DIALOGUE_RUBRIC,
                context={"panels": payload, "max_chars": max_chars, "seed": self.config.seed},
                post_validate=post_validate,
            )
        except (HarnessError, StructuredOutputError) as exc:
            logger.warning("LLM 台詞増幅に失敗したためテンプレート台詞を維持します: %s", exc)
            storyboard.revision_log.append(f"LLM台詞増幅失敗（テンプレート維持）: {exc}")
            return storyboard
        enriched = storyboard.model_copy(deep=True)
        changed = apply_dialogue_batch(enriched, result.output, self.config)
        for page in enriched.pages:
            refresh_page_metrics(page)
        enriched.revision_log.append(f"LLM台詞増幅: run_id={result.run_id} / {changed} コマ更新")
        return enriched


# ---------------------------------------------------------------------------
# 修復器
# ---------------------------------------------------------------------------


class StoryboardFixer:
    """漫画文法違反の規則別・決定論的修復。"""

    def __init__(self, config: StoryboardConfig | None = None) -> None:
        self.config = config or StoryboardConfig()

    def apply(self, storyboard: Storyboard, report: GrammarReport, structure: UniversalStructure | None = None) -> Storyboard:
        """レポートの違反に応じて修復したコピーを返す。"""
        board = storyboard.model_copy(deep=True)
        by_rule = report.by_rule()
        log: list[str] = []
        if "PEAK_EMPHASIS" in by_rule:
            log.extend(self._fix_peak(board, by_rule["PEAK_EMPHASIS"], structure))
        if "BALLOON_OVERLOAD" in by_rule:
            log.extend(self._fix_overload(board, by_rule["BALLOON_OVERLOAD"]))
        relayout_pages = {v.page_index for rule in ("GAZE_BACKTRACK", "GAZE_CROSSING") for v in by_rule.get(rule, []) if v.page_index is not None}
        for page in board.pages:
            if page.index in relayout_pages:
                layout_page(page, board, allow_slant=False)
                self._refresh_page(page, board, structure)
                log.append(f"ページ {page.number}: 視線誘導のため矩形段組へ再構成")
        if "DENSITY_CONTRAST" in by_rule:
            log.extend(self._fix_density(board, by_rule["DENSITY_CONTRAST"], structure))
        if "PAGE_TURN_HOOK" in by_rule:
            log.extend(self._fix_hooks(board, by_rule["PAGE_TURN_HOOK"]))
        if "BALLOON_ORDER" in by_rule:
            for violation in by_rule["BALLOON_ORDER"]:
                for panel_id in violation.panel_ids:
                    panel = board.panel(panel_id)
                    if panel is not None:
                        layout_balloons(panel, self.config)
                        log.append(f"コマ {panel_id}: フキダシを読み順に再配置")
        changes = enforce_camera_variety(board)
        if changes:
            log.append(f"カメラの単調さを {changes} 箇所修正")
        finalize_reading_order(board)
        for page in board.pages:
            refresh_page_metrics(page)
        board.revision_log.extend(log)
        return Storyboard.model_validate(board.model_dump())

    # -- 個別修復 ---------------------------------------------------------------

    def _refresh_page(self, page: Page, board: Storyboard, structure: UniversalStructure | None) -> None:
        beats = {b.id: b for b in structure.beats.beats} if structure is not None else {}
        for panel in page.panels:
            beat = beats.get(panel.beat_id) if panel.beat_id else None
            _stage_panel(panel, beat)
            layout_balloons(panel, self.config)
            place_sfx(panel, beat.beat_type.value if beat else None)
        refresh_page_metrics(page)

    def _consolidate_large(self, page: Page, board: Storyboard, keep_ids: list[str]) -> list[str]:
        """1ページの大ゴマ候補が多すぎる場合、余剰コマのビートを残す大ゴマへ統合する。"""
        log: list[str] = []
        has_continuation = any(p.spread_role == "continuation" for p in page.panels)
        limit = 1 if has_continuation else 2
        large = [p for p in page.ordered_panels() if p.id in keep_ids or p.emphasis in ("large", "splash")]
        if len(large) <= limit:
            return log
        ranked = sorted(large, key=lambda p: (-(p.id in keep_ids), -p.tension))
        keepers = ranked[:limit]
        for panel in ranked[limit:]:
            host = min(keepers, key=lambda k: abs(k.index_in_page - panel.index_in_page))
            host.secondary_beat_ids.extend(b for b in panel.beat_ids() if b not in host.beat_ids())
            room = self.config.max_balloons_per_panel - len(host.balloons)
            host.balloons.extend(panel.balloons[: max(0, room)])
            page.panels = [p for p in page.panels if p.id != panel.id]
            log.append(f"ページ {page.number}: コマ {panel.id} を大ゴマ {host.id} へ統合")
        for i, panel in enumerate(page.ordered_panels()):
            panel.index_in_page = i
        return log

    def _fix_peak(self, board: Storyboard, violations: list[Any], structure: UniversalStructure | None) -> list[str]:
        log: list[str] = []
        targets: dict[int, list[str]] = {}
        for violation in violations:
            for panel_id in violation.panel_ids:
                page = board.page_of_panel(panel_id)
                if page is not None:
                    targets.setdefault(page.index, []).append(panel_id)
        for page_index, panel_ids in targets.items():
            page = next(p for p in board.pages if p.index == page_index)
            log.extend(self._consolidate_large(page, board, panel_ids))
            for panel_id in panel_ids:
                panel = board.panel(panel_id)
                if panel is not None and panel.emphasis == "normal":
                    panel.emphasis = "large"
            layout_page(page, board, allow_slant=False)
            self._refresh_page(page, board, structure)
            log.append(f"ページ {page.number}: 転換点コマ {', '.join(panel_ids)} を大ゴマ化")
        return log

    def _fix_overload(self, board: Storyboard, violations: list[Any]) -> list[str]:
        log: list[str] = []
        cfg = self.config
        for violation in violations:
            for panel_id in violation.panel_ids:
                panel = board.panel(panel_id)
                page = board.page_of_panel(panel_id)
                if panel is None or page is None:
                    continue
                rebuilt: list[Balloon] = []
                for balloon in panel.balloons:
                    for piece in split_text_for_balloons(balloon.text, cfg.max_chars_per_balloon):
                        rebuilt.append(balloon.model_copy(update={"text": piece, "id": f"{panel.id}_b{len(rebuilt)}"}))
                limit_total = int(2.2 * cfg.max_chars_per_balloon)
                keep: list[Balloon] = []
                overflow: list[Balloon] = []
                total = 0
                for balloon in rebuilt:
                    if len(keep) < cfg.max_balloons_per_panel and total + balloon.char_count <= limit_total:
                        keep.append(balloon)
                        total += balloon.char_count
                    else:
                        overflow.append(balloon)
                panel.balloons = keep
                ordered = page.ordered_panels()
                position = next(i for i, p in enumerate(ordered) if p.id == panel.id)
                neighbors = [ordered[j] for j in (position + 1, position - 1) if 0 <= j < len(ordered)]
                for balloon in overflow:
                    receiver = next(
                        (
                            n
                            for n in neighbors
                            if n.spread_role != "continuation"
                            and len(n.balloons) < cfg.max_balloons_per_panel
                            and n.text_char_count() + balloon.char_count <= limit_total
                        ),
                        None,
                    )
                    if receiver is not None:
                        receiver.balloons.append(balloon.model_copy(update={"id": f"{receiver.id}_b{len(receiver.balloons)}"}))
                        receiver.is_silent = False
                        layout_balloons(receiver, cfg)
                        log.append(f"コマ {panel.id}: 台詞を {receiver.id} へ送り")
                    else:
                        log.append(f"コマ {panel.id}: 収まらない台詞「{balloon.text[:12]}」を削除")
                for i, balloon in enumerate(panel.balloons):
                    balloon.id = f"{panel.id}_b{i}"
                layout_balloons(panel, cfg)
                log.append(f"コマ {panel.id}: フキダシを分割・整理")
        return log

    def _fix_density(self, board: Storyboard, violations: list[Any], structure: UniversalStructure | None) -> list[str]:
        log: list[str] = []
        cfg = self.config
        threshold = MangaGrammarValidator(cfg).density_cv_threshold
        pages = sorted(board.pages, key=lambda p: p.index)

        def cv(window: list[Page]) -> float:
            values = np.asarray([page_information_density(p) for p in window], dtype=float)
            mean = float(np.mean(values))
            return float(np.std(values) / mean) if mean > 1e-9 else 0.0

        for violation in violations:
            if violation.page_index is None:
                continue
            position = next((i for i, p in enumerate(pages) if p.index == violation.page_index), None)
            if position is None:
                continue
            page = pages[position]
            if violation.severity == "error":
                removed = self._remove_connective(page, board, structure, count=2)
                for panel in page.panels:
                    panel.balloons = panel.balloons[:2]
                    layout_balloons(panel, cfg)
                refresh_page_metrics(page)
                log.append(f"ページ {page.number}: 過密のため接続コマ {removed} 枚削除・台詞整理")
                continue
            window = pages[max(0, position - 1) : position + 2]
            if len(window) < 3:
                continue
            neighbors = [p for p in window if p.index != page.index]
            neighbor_mean = float(np.mean([page_information_density(p) for p in neighbors]))
            denser = page_information_density(page) >= neighbor_mean
            actions = 0
            for _ in range(4):
                if cv(window) >= threshold + 0.02:
                    break
                changed = False
                if denser:
                    changed = bool(self._add_connective(page, board, structure, count=1))
                    if not changed:
                        target = max(neighbors, key=page_information_density)
                        changed = bool(self._remove_connective(target, board, structure, count=1)) or self._silence_connectives(target)
                else:
                    changed = bool(self._remove_connective(page, board, structure, count=1)) or self._silence_connectives(page)
                    if not changed:
                        target = min(neighbors, key=page_information_density)
                        changed = bool(self._add_connective(target, board, structure, count=1))
                for item in window:
                    refresh_page_metrics(item)
                if not changed:
                    break
                actions += 1
            if actions:
                direction = "密に" if denser else "疎に"
                log.append(f"ページ {page.number}: 緩急をつけるため前後との密度差を拡大（{direction}、{actions} 手）")
        return log

    def _remove_connective(self, page: Page, board: Storyboard, structure: UniversalStructure | None, count: int) -> int:
        removed = 0
        for _ in range(count):
            candidates = [p for p in page.ordered_panels() if not p.beat_id and p.spread_role == "none" and "page_turn_hook" not in p.prompt_hints]
            if not candidates or len(page.panels) <= 1:
                break
            victim = min(candidates, key=lambda p: (p.tension, -p.index_in_page))
            page.panels = [p for p in page.panels if p.id != victim.id]
            removed += 1
        if removed:
            for i, panel in enumerate(page.ordered_panels()):
                panel.index_in_page = i
            layout_page(page, board, allow_slant=False)
            self._refresh_page(page, board, structure)
        return removed

    def _add_connective(self, page: Page, board: Storyboard, structure: UniversalStructure | None, count: int) -> int:
        cfg = self.config
        added = 0
        large = sum(1 for p in page.panels if p.emphasis != "normal")
        capacity = cfg.max_panels_per_page - len(page.panels)
        if large >= 2:
            capacity = min(capacity, 3 - sum(1 for p in page.panels if p.emphasis == "normal"))
        theme_word = _theme_word(board.theme)
        for _ in range(min(count, max(0, capacity))):
            ordered = page.ordered_panels()
            anchor_index = len(ordered)
            if ordered and ordered[-1].spread_role == "none" and "page_turn_hook" in ordered[-1].prompt_hints:
                anchor_index = len(ordered) - 1
            reference = ordered[anchor_index - 1] if anchor_index > 0 else ordered[0]
            panel_id = f"pg{page.index:02d}_ad{len(page.panels):02d}"
            while board.panel(panel_id) is not None:
                panel_id = panel_id + "x"
            rng = random.Random(stable_seed("caf.connective", panel_id))
            new_panel = Panel(
                id=panel_id,
                page_index=page.index,
                index_in_page=anchor_index,
                polygon=reference.polygon.model_copy(),
                camera=reference.camera.model_copy(),
                characters=[c.model_copy() for c in reference.characters[:2]],
                description="繋ぎ: 会話による状況整理",
                t=reference.t,
                tension=float(clamp(reference.tension * 0.9, 0.0, 1.0)),
            )
            for panel in ordered[anchor_index:]:
                panel.index_in_page += 1
            page.panels.append(new_panel)
            new_panel.balloons = _make_balloons(new_panel, template_lines(None, new_panel.tension, rng=rng, theme_word=theme_word), cfg)
            added += 1
        if added:
            layout_page(page, board, allow_slant=False)
            enforce_camera_variety(board)
            self._refresh_page(page, board, structure)
        return added

    def _silence_connectives(self, page: Page) -> bool:
        changed = False
        for panel in page.panels:
            if not panel.beat_id and panel.spread_role == "none" and panel.balloons and "page_turn_hook" not in panel.prompt_hints:
                panel.balloons = []
                panel.is_silent = True
                changed = True
        return changed

    def _fix_hooks(self, board: Storyboard, violations: list[Any]) -> list[str]:
        log: list[str] = []
        theme_word = _theme_word(board.theme)
        for violation in violations:
            for panel_id in violation.panel_ids:
                panel = board.panel(panel_id)
                page = board.page_of_panel(panel_id)
                if panel is None or page is None:
                    continue
                if "page_turn_hook" not in panel.prompt_hints:
                    panel.prompt_hints.append("page_turn_hook")
                panel.tension = float(clamp(max(panel.tension, page.tension_mean + 0.02), 0.0, 1.0))
                if not panel.balloons and panel.spread_role == "none":
                    rng = random.Random(stable_seed("caf.hook", panel.id))
                    panel.balloons = _make_balloons(panel, template_lines(None, panel.tension, rng=rng, theme_word=theme_word, hook=True), self.config)
                    panel.is_silent = False
                    layout_balloons(panel, self.config)
                log.append(f"コマ {panel_id}: めくり前の引きを強化")
        return log

    # -- 安全再構成 --------------------------------------------------------------

    def safe_rebuild(self, storyboard: Storyboard, structure: UniversalStructure | None = None, *, aggressive: bool = False) -> Storyboard:
        """全ページを規則準拠の安全レイアウトで再構成する（矩形段組・強調保証・台詞整理・緩急調整）。"""
        board = storyboard.model_copy(deep=True)
        cfg = self.config
        validator = MangaGrammarValidator(cfg)
        targets = set(emphasis_beat_ids(structure)) if structure is not None else set()
        balloon_cap = 2 if aggressive else cfg.max_balloons_per_panel
        for page in board.pages:
            emphasis_panels = [p.id for p in page.panels if set(p.beat_ids()) & targets]
            self._consolidate_large(page, board, emphasis_panels)
            for panel in page.panels:
                if set(panel.beat_ids()) & targets and panel.emphasis == "normal":
                    panel.emphasis = "large"
            max_normal = 3 if sum(1 for p in page.panels if p.emphasis != "normal") >= 2 else 6
            normals = [p for p in page.ordered_panels() if p.emphasis == "normal"]
            while len(normals) > max_normal:
                victims = [p for p in normals if not p.beat_id and "page_turn_hook" not in p.prompt_hints]
                if not victims:
                    host = normals[-2]
                    tail = normals[-1]
                    host.secondary_beat_ids.extend(tail.beat_ids())
                    page.panels = [p for p in page.panels if p.id != tail.id]
                else:
                    victim = victims[0]
                    page.panels = [p for p in page.panels if p.id != victim.id]
                normals = [p for p in page.ordered_panels() if p.emphasis == "normal"]
            for i, panel in enumerate(page.ordered_panels()):
                panel.index_in_page = i
            limit_total = int(2.0 * cfg.max_chars_per_balloon)
            for panel in page.panels:
                trimmed: list[Balloon] = []
                total = 0
                for balloon in panel.balloons:
                    for piece in split_text_for_balloons(balloon.text, cfg.max_chars_per_balloon):
                        if len(trimmed) < balloon_cap and total + visible_length(piece) <= limit_total:
                            trimmed.append(balloon.model_copy(update={"text": piece, "id": f"{panel.id}_b{len(trimmed)}"}))
                            total += visible_length(piece)
                panel.balloons = trimmed
            layout_page(page, board, allow_slant=False)
            self._refresh_page(page, board, structure)
        enforce_camera_variety(board)
        finalize_reading_order(board)
        for _ in range(6 if aggressive else 3):
            report = validator.validate(board, structure)
            density = [v for v in report.violations if v.rule_id == "DENSITY_CONTRAST"]
            hooks = [v for v in report.violations if v.rule_id == "PAGE_TURN_HOOK"]
            if not density and not hooks:
                break
            if density:
                self._fix_density(board, density, structure)
            if hooks:
                self._fix_hooks(board, hooks)
            enforce_camera_variety(board)
            finalize_reading_order(board)
        for page in board.pages:
            refresh_page_metrics(page)
        board.revision_log.append("安全レイアウト: 矩形段組・強調保証・台詞整理・緩急調整を適用")
        return Storyboard.model_validate(board.model_dump())

