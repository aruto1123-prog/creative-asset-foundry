"""Phase 4: 構造駆動型・漫画ネーム＆ComfyUI生成パイプラインのテスト。"""

from __future__ import annotations

import asyncio
import json
import math
import xml.etree.ElementTree as ET

import httpx
import pytest

from core.manga_pipeline import geometry as geo
from core.manga_pipeline.comfy_client import ComfyClient, ComfyConnectionError, ComfyError
from core.manga_pipeline.comfy_compiler import (
    CompilerConfig,
    ComfyWorkflowCompiler,
    ControlGuide,
    SeedManager,
)
from core.manga_pipeline.comfy_mock_server import MockComfyServer, make_png
from core.manga_pipeline.grammar_validator import (
    MangaGrammarValidator,
    emphasis_beat_ids,
    identify_peak_beat,
    page_information_density,
    panel_area_ratio,
)
from core.manga_pipeline.layout_renderer import RenderOptions, SVGLayoutRenderer
from core.manga_pipeline.schemas import (
    Balloon,
    BalloonKind,
    CameraAngle,
    CameraWork,
    Page,
    PageSide,
    PageTurnRole,
    Panel,
    Point,
    Polygon,
    ShotSize,
    Storyboard,
)
from core.manga_pipeline.storyboard_generator import (
    StoryboardConfig,
    StoryboardFixer,
    StoryboardGenerator,
    allocate_pages,
    page_roles,
    page_sides,
)
from core.manga_pipeline.text_layout import (
    LINE_END_PROHIBITED,
    LINE_START_PROHIBITED,
    break_japanese_text,
    place_balloons,
    split_text_for_balloons,
)
from core.structure_engine.samples import sample_structure, sample_structures
from core.structure_engine.schemas import PanelRhythmProfile
from core.style_alchemy.comfy_export import validate_prompt
from core.style_alchemy.schemas import (
    BlockMergeRecipe,
    CheckpointRef,
    LoRASpec,
    ModelArchitecture,
    StyleRecipe,
)

PAGE_COUNTS = (8, 12, 16, 24)


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------


def _storyboard(kind: str = "reversal", pages: int = 12, seed: int = 3, **config: object) -> Storyboard:
    generator = StoryboardGenerator(StoryboardConfig(target_pages=pages, seed=seed, **config))
    return generator.generate_validated_sync(sample_structure(kind), theme={"THEME": "失われた居場所"})


def _recipe(arch: ModelArchitecture = ModelArchitecture.SDXL) -> StyleRecipe:
    a = CheckpointRef(name=f"{arch.value}_a.safetensors", architecture=arch)
    b = CheckpointRef(name=f"{arch.value}_b.safetensors", architecture=arch)
    return StyleRecipe(
        name="test recipe",
        architecture=arch,
        base_checkpoint=a,
        merge=BlockMergeRecipe(architecture=arch, model_a=a, model_b=b, default_weight=0.35),
        loras=[LoRASpec(name="ink_style.safetensors", strength_model=0.8, trigger_words=["caf_ink"])],
        style_tokens=["monochrome", "manga"],
    )


def _camera(shot: ShotSize = ShotSize.MS, angle: CameraAngle = CameraAngle.EYE_LEVEL) -> CameraWork:
    return CameraWork(shot=shot, angle=angle, lens_mm=50)


def _manual_page(index: int, rects: list[tuple[float, float, float, float]], *, side: PageSide = PageSide.RIGHT) -> Page:
    panels = [
        Panel(
            id=f"m{index}_{i}",
            page_index=index,
            index_in_page=i,
            polygon=Polygon.rect(*rect),
            camera=_camera(),
            tension=0.5,
        )
        for i, rect in enumerate(rects)
    ]
    return Page(index=index, number=index + 1, side=side, panels=panels)


def _manual_storyboard(pages: list[Page]) -> Storyboard:
    return Storyboard(title="手動ネーム", structure_id="manual", pages=pages)


# ---------------------------------------------------------------------------
# 幾何
# ---------------------------------------------------------------------------


def test_polygon_basics() -> None:
    square = geo.rect_points(0, 0, 10, 10)
    assert geo.polygon_area(square) == pytest.approx(100.0)
    assert geo.polygon_centroid(square) == pytest.approx((5.0, 5.0))
    assert geo.polygon_bbox(square) == (0, 0, 10, 10)
    triangle = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0)]
    assert geo.polygon_area(triangle) == pytest.approx(18.0)
    assert geo.polygon_centroid(triangle) == pytest.approx((2.0, 2.0))
    assert geo.point_in_polygon((5, 5), square)
    assert geo.point_in_polygon((10, 5), square)
    assert not geo.point_in_polygon((11, 5), square)
    assert geo.is_convex(square)
    assert not geo.is_convex([(0, 0), (10, 0), (5, 3), (10, 10), (0, 10)])
    assert len(geo.polygon_edges(square)) == 4
    clipped = geo.clip_polygon(square, geo.rect_points(5, 5, 15, 15))
    assert geo.polygon_area(clipped) == pytest.approx(25.0)
    model = Polygon.rect(0, 0, 4, 2)
    assert model.area() == pytest.approx(8.0)
    assert model.contains(Point(x=1, y=1))
    assert model.centroid().x == pytest.approx(2.0)


def test_segment_intersection() -> None:
    assert geo.segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert geo.segment_intersection((0, 0), (10, 10), (0, 10), (10, 0)) == pytest.approx((5.0, 5.0))
    assert not geo.segments_intersect((0, 0), (5, 5), (5, 5), (10, 0))
    assert geo.segments_intersect((0, 0), (5, 5), (5, 5), (10, 0), proper=False)
    assert not geo.segments_intersect((0, 0), (10, 0), (0, 1), (10, 1))


def test_reading_order_two_tiers() -> None:
    page = _manual_page(0, [(100, 100, 480, 600), (520, 100, 900, 600), (100, 640, 480, 1200), (520, 640, 900, 1200)])
    order = geo.compute_reading_order(page.panels)
    assert order == ["m0_1", "m0_0", "m0_3", "m0_2"]


def test_reading_order_tall_right_panel() -> None:
    page = _manual_page(0, [(100, 100, 480, 390), (100, 410, 480, 700), (500, 100, 900, 700)])
    assert geo.compute_reading_order(page.panels) == ["m0_2", "m0_0", "m0_1"]


def test_reading_order_slanted_panels() -> None:
    polys = geo.layout_tiers((90, 90, 930, 1304), [[1, 1, 1], [1, 1]], [0.5, 0.5], gutter_x=16, gutter_y=30, slants=[45, -40])
    boxes = [geo.core_box(p) for tier in polys for p in tier]
    raw_boxes = [geo.polygon_bbox(p) for tier in polys for p in tier]
    assert raw_boxes[0][0] < raw_boxes[1][2]
    assert geo.xy_cut_order(boxes) == list(range(5))


@pytest.mark.parametrize(
    "flags",
    [[False] * 5, [False] * 7, [True, False, False, False], [False, False, True, False, False], [True, False, False, True], [False] * 9],
)
def test_plan_tiers_and_layout_order(flags: list[bool]) -> None:
    tiers = geo.plan_tiers(flags, max_tiers=4, capacity=3)
    assert [i for tier in tiers for i in tier] == list(range(len(flags)))
    for tier in tiers:
        if any(flags[i] for i in tier):
            assert len(tier) == 1
    fracs = geo.tier_fractions([1.0] * len(tiers), [any(flags[i] for i in t) for t in tiers])
    assert sum(fracs) == pytest.approx(1.0)
    polys = geo.layout_tiers((90, 90, 930, 1304), [[1.0] * len(t) for t in tiers], fracs, gutter_x=16, gutter_y=30, slants=[30.0] * len(tiers))
    boxes = [geo.core_box(p) for tier in polys for p in tier]
    assert geo.xy_cut_order(boxes) == list(range(len(flags)))


def test_gaze_transitions_natural_backtrack_and_crossing() -> None:
    natural = _manual_page(0, [(520, 100, 900, 600), (100, 100, 480, 600), (520, 640, 900, 1200), (100, 640, 480, 1200)])
    transitions = geo.compute_gaze_transitions(natural)
    assert len(transitions) == 3
    assert not any(t.is_backtrack for t in transitions)
    assert not any(t.crosses for t in transitions)
    backward = _manual_page(1, [(100, 100, 480, 600), (520, 100, 900, 600)])
    assert geo.compute_gaze_transitions(backward)[0].is_backtrack
    crossing = _manual_page(2, [(520, 100, 900, 600), (100, 640, 480, 1200), (100, 100, 480, 600), (520, 640, 900, 1200)])
    result = geo.compute_gaze_transitions(crossing)
    assert any(t.crosses for t in result)
    assert any(t.is_backtrack for t in result)


# ---------------------------------------------------------------------------
# 縦書き・禁則
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "お前だったのか……！",
        "全部、最初から仕組まれていた……！",
        "形見の「鍵」を使って、結界を打ち破る。",
        "そんな……嘘だろ？ありえない！",
        "グレゴールの結界を、今ここで打ち破る！",
        "うおおおおおおおおっ！！",
    ],
)
def test_break_japanese_text_kinsoku(text: str) -> None:
    columns = break_japanese_text(text, 7)
    assert "".join(columns) == "".join(text.split())
    assert all(len(c) <= 9 for c in columns)
    assert all(c[0] not in LINE_START_PROHIBITED for c in columns[1:])
    assert all(c[-1] not in LINE_END_PROHIBITED for c in columns[:-1])


def test_katakana_word_is_not_split() -> None:
    columns = break_japanese_text("グレゴールの結界", 7)
    assert any("グレゴール" in c for c in columns)


def test_split_text_for_balloons_and_placement() -> None:
    text = "全部、最初から仕組まれていた。信じていたものは嘘だった。それでも前へ進む。"
    pieces = split_text_for_balloons(text, 18)
    assert "".join(pieces) == text
    assert all(len(p) <= 18 for p in pieces)
    box = (100.0, 100.0, 600.0, 500.0)
    placements = place_balloons(box, [(120, 200), (110, 180), (100, 160)], [(450.0, 300.0, 40.0)])
    for cx, cy, w, h in placements:
        assert box[0] <= cx - w / 2 and cx + w / 2 <= box[2]
        assert box[1] <= cy - h / 2 and cy + h / 2 <= box[3]
    for (ax, ay, _, ah), (bx, by, _, bh) in zip(placements, placements[1:]):
        assert bx < ax or by > ay


# ---------------------------------------------------------------------------
# ページ構成・生成
# ---------------------------------------------------------------------------


def test_page_sides_and_roles() -> None:
    sides = page_sides(6, PageSide.RIGHT)
    assert [s.value for s in sides] == ["right", "left", "right", "left", "right", "left"]
    roles = page_roles(sides)
    assert roles[0] == PageTurnRole.NONE
    assert roles[1] == PageTurnRole.HOOK and roles[2] == PageTurnRole.REVEAL
    assert roles[-1] == PageTurnRole.NONE
    left_first = page_roles(page_sides(4, PageSide.LEFT))
    assert left_first == [PageTurnRole.HOOK, PageTurnRole.REVEAL, PageTurnRole.HOOK, PageTurnRole.REVEAL]


def test_allocate_pages_anchors_pivotal_beats_on_reveal_pages() -> None:
    structure = sample_structure("reversal")
    sides = page_sides(12, PageSide.RIGHT)
    roles = page_roles(sides)
    peak = identify_peak_beat(structure)
    bounds, anchors = allocate_pages(structure, 12, roles, emphasis_ids=emphasis_beat_ids(structure), peak_id=peak.id)
    assert bounds[0] == 0.0 and bounds[-1] == 1.0
    assert all(b2 > b1 for b1, b2 in zip(bounds, bounds[1:]))
    assert anchors
    beats = {b.id: b for b in structure.beats.beats}
    for page_index, beat_id in anchors.items():
        assert roles[page_index] == PageTurnRole.REVEAL
        assert bounds[page_index] <= beats[beat_id].t < bounds[page_index + 1]


@pytest.mark.parametrize("kind", ["reversal", "inverted", "persuasion"])
@pytest.mark.parametrize("pages", PAGE_COUNTS)
def test_generate_validated_always_passes(kind: str, pages: int) -> None:
    config = StoryboardConfig(target_pages=pages, seed=1)
    structure = sample_structure(kind)
    storyboard = StoryboardGenerator(config).generate_validated_sync(structure)
    report = storyboard.grammar
    assert report is not None and report.passed
    assert report.score >= config.grammar_threshold
    assert not report.errors()
    assert storyboard.page_count == pages
    assert [p.side for p in storyboard.pages] == page_sides(pages, PageSide.RIGHT)
    panels = storyboard.panels_in_reading_order()
    assert [p.reading_order for p in panels] == list(range(len(panels)))
    covered = {beat_id for panel in panels for beat_id in panel.beat_ids()}
    assert {b.id for b in structure.beats.beats} <= covered
    validator = MangaGrammarValidator(config)
    for beat_id in emphasis_beat_ids(structure):
        holders = [(panel, storyboard.page_of_panel(panel.id)) for panel in panels if beat_id in panel.beat_ids()]
        assert any(validator._panel_emphasized(panel, storyboard, page) for panel, page in holders)
    bx0, by0, bx1, by1 = storyboard.geometry.bleed_rect()
    for panel in panels:
        assert 18 <= panel.camera.lens_mm <= 135
        x0, y0, x1, y1 = panel.polygon.bbox()
        assert bx0 <= x0 and x1 <= bx1 and by0 <= y0 and y1 <= by1
        for balloon in panel.balloons:
            assert balloon.char_count <= config.max_chars_per_balloon
            assert balloon.columns


def test_generation_is_deterministic() -> None:
    a = _storyboard(seed=9).model_dump(exclude={"created_at"})
    b = _storyboard(seed=9).model_dump(exclude={"created_at"})
    assert a == b
    c = _storyboard(seed=10).model_dump(exclude={"created_at"})
    assert a != c


def test_peak_spread_and_page_turn_design() -> None:
    structure = sample_structure("reversal")
    storyboard = _storyboard("reversal", pages=12)
    doubles = [s for s in storyboard.spreads if s.kind == "double_page_panel"]
    assert len(doubles) == 1
    spread = doubles[0]
    right = next(p for p in storyboard.pages if p.index == spread.right_page)
    assert right.side == PageSide.RIGHT and right.page_turn_role == PageTurnRole.REVEAL
    peak = identify_peak_beat(structure)
    assert any(peak.id in panel.beat_ids() and panel.emphasis == "spread" for panel in right.panels)
    left = next(p for p in storyboard.pages if p.index == spread.left_page)
    assert left.ordered_panels()[0].spread_role == "continuation"
    reveal_pages = [p for p in storyboard.pages if p.page_turn_role == PageTurnRole.REVEAL]
    assert reveal_pages
    hooks = [p for p in storyboard.pages if any("page_turn_hook" in panel.prompt_hints for panel in p.panels)]
    assert hooks and all(p.page_turn_role == PageTurnRole.HOOK for p in hooks)
    assert any(panel.shape.value == "slanted" for panel in storyboard.panels_in_reading_order())
    assert any(panel.bleed_sides for panel in storyboard.panels_in_reading_order())


def test_left_first_and_rhythm_profile() -> None:
    structure = sample_structure("inverted")
    left_first = StoryboardGenerator(StoryboardConfig(target_pages=10, first_page_side=PageSide.LEFT)).generate_validated_sync(structure)
    assert left_first.pages[0].side == PageSide.LEFT
    assert left_first.grammar.passed
    profiled = structure.model_copy(
        update={
            "rhythm_profile": PanelRhythmProfile(
                page_count=6,
                panels_per_page=[5, 3, 6, 2, 4, 1],
                emphasis_pages=[3, 5],
                page_turn_reveals=[2, 4],
                spread_pages=[2],
                silent_panels_ratio=0.2,
                pacing_notes=["導入", "溜め", "反転"],
            )
        }
    )
    generator = StoryboardGenerator(StoryboardConfig(target_pages=12))
    rhythm = generator._rhythm_map(profiled, 12, page_roles(page_sides(12, PageSide.RIGHT)))
    assert rhythm is not None and len(rhythm["counts"]) == 12
    assert all(page_roles(page_sides(12, PageSide.RIGHT))[r] == PageTurnRole.REVEAL for r in rhythm["reveals"])
    storyboard = generator.generate_validated_sync(profiled)
    assert storyboard.grammar.passed


def test_characters_and_staging() -> None:
    storyboard = _storyboard()
    assert storyboard.characters[0].role == "protagonist"
    assert storyboard.characters[0].name_placeholder == "主人公"
    assert all(c.visual_description for c in storyboard.characters)
    ids = {c.id for c in storyboard.characters}
    for panel in storyboard.panels_in_reading_order():
        for action in panel.characters:
            assert action.character_id in ids
            assert 0.0 <= action.position.x <= 1.0
            assert math.isclose(math.hypot(action.gaze.x, action.gaze.y), 1.0, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# 漫画文法バリデータ
# ---------------------------------------------------------------------------


def _rules(report) -> set[str]:
    return {v.rule_id for v in report.violations}


def test_validator_detects_monotonous_camera() -> None:
    storyboard = _storyboard()
    panels = [p for p in storyboard.panels_in_reading_order() if p.spread_role != "continuation"]
    for panel in panels[:4]:
        panel.camera = CameraWork(shot=ShotSize.MS, angle=CameraAngle.HIGH, lens_mm=50)
    report = MangaGrammarValidator().validate(storyboard, sample_structure("reversal"))
    assert {"MONOTONOUS_SHOT", "MONOTONOUS_ANGLE"} <= _rules(report)
    shot_violation = next(v for v in report.violations if v.rule_id == "MONOTONOUS_SHOT")
    assert len(shot_violation.panel_ids) >= 3 and shot_violation.fix_instruction


def test_validator_detects_peak_without_emphasis() -> None:
    structure = sample_structure("reversal")
    storyboard = _storyboard()
    target = emphasis_beat_ids(structure)[0]
    panel = next(p for p in storyboard.panels_in_reading_order() if target in p.beat_ids())
    page = storyboard.page_of_panel(panel.id)
    page.page_turn_role = PageTurnRole.NONE
    panel.emphasis = "normal"
    x0, y0, _, _ = panel.polygon.bbox()
    panel.polygon = Polygon.rect(x0, y0, x0 + 120, y0 + 120)
    report = MangaGrammarValidator().validate(storyboard, structure)
    assert "PEAK_EMPHASIS" in _rules(report)
    assert not report.passed
    fixed = StoryboardFixer().apply(storyboard, report, structure)
    again = MangaGrammarValidator().validate(fixed, structure)
    assert "PEAK_EMPHASIS" not in _rules(again)


def test_validator_detects_balloon_overload_and_fixer_splits() -> None:
    structure = sample_structure("reversal")
    storyboard = _storyboard()
    panel = next(p for p in storyboard.panels_in_reading_order() if p.balloons)
    long_text = "これは非常に長い台詞であり一つのフキダシに収めるには明らかに文字数が多すぎるため分割が必要になるはずだ。" * 2
    panel.balloons[0] = panel.balloons[0].model_copy(update={"text": long_text})
    report = MangaGrammarValidator().validate(storyboard, structure)
    overload = [v for v in report.violations if v.rule_id == "BALLOON_OVERLOAD"]
    assert {v.severity for v in overload} == {"warning", "error"}
    fixed = StoryboardFixer().apply(storyboard, report, structure)
    again = MangaGrammarValidator().validate(fixed, structure)
    assert not [v for v in again.violations if v.rule_id == "BALLOON_OVERLOAD"]


def test_validator_detects_gaze_backtrack_and_crossing() -> None:
    backtrack = _manual_storyboard([_manual_page(0, [(90, 90, 480, 650), (500, 90, 930, 650), (90, 680, 930, 1304)])])
    report = MangaGrammarValidator().validate(backtrack)
    assert "GAZE_BACKTRACK" in _rules(report)
    crossing = _manual_storyboard([_manual_page(0, [(520, 100, 900, 600), (100, 640, 480, 1200), (100, 100, 480, 600), (520, 640, 900, 1200)])])
    report = MangaGrammarValidator().validate(crossing)
    assert {"GAZE_CROSSING", "GAZE_BACKTRACK"} <= _rules(report)


def test_validator_detects_density_contrast_and_overflow() -> None:
    pages = [_manual_page(i, [(90, 90, 930, 650), (90, 680, 930, 1304)], side=PageSide.RIGHT if i % 2 == 0 else PageSide.LEFT) for i in range(5)]
    flat = _manual_storyboard(pages)
    report = MangaGrammarValidator().validate(flat)
    density = [v for v in report.violations if v.rule_id == "DENSITY_CONTRAST"]
    assert len(density) == 1 and density[0].severity == "warning"
    crowded_panel = pages[2].panels[0]
    crowded_panel.balloons = [
        Balloon(id=f"x{i}", text="あ" * 40, center=Point(x=500, y=300), width=100, height=200, reading_order=i) for i in range(20)
    ]
    report = MangaGrammarValidator().validate(flat)
    assert any(v.rule_id == "DENSITY_CONTRAST" and v.severity == "error" for v in report.violations)
    assert page_information_density(pages[2]) > 26


def test_validator_detects_hook_and_balloon_order() -> None:
    structure = sample_structure("reversal")
    storyboard = _storyboard()
    hook_page = next(p for p in storyboard.pages if any("page_turn_hook" in panel.prompt_hints for panel in p.panels))
    last = hook_page.ordered_panels()[-1]
    last.prompt_hints = [h for h in last.prompt_hints if h != "page_turn_hook"]
    last.tension = 0.0
    panel = next(p for p in storyboard.panels_in_reading_order() if p.balloons)
    first = panel.balloons[0]
    panel.balloons.append(
        Balloon(
            id=f"{panel.id}_late",
            text="右上の後続",
            center=Point(x=first.center.x + 80.0, y=first.center.y - 10.0),
            width=first.width,
            height=first.height,
            reading_order=first.reading_order + 1,
        )
    )
    report = MangaGrammarValidator().validate(storyboard, structure)
    rules = _rules(report)
    assert "PAGE_TURN_HOOK" in rules
    assert "BALLOON_ORDER" in rules
    infos = [v for v in report.violations if v.severity == "info"]
    assert all(v.penalty == 1.0 for v in infos)
    assert report.score == pytest.approx(max(0.0, 100.0 - sum(v.penalty for v in report.violations)))


def test_safe_rebuild_always_passes() -> None:
    structure = sample_structure("inverted")
    storyboard = _storyboard("inverted", pages=16)
    for panel in storyboard.panels_in_reading_order():
        panel.camera = CameraWork(shot=ShotSize.CU, angle=CameraAngle.LOW, lens_mm=85)
        panel.emphasis = "normal"
    rebuilt = StoryboardFixer().safe_rebuild(storyboard, structure, aggressive=True)
    report = MangaGrammarValidator().validate(rebuilt, structure)
    assert report.passed and not report.errors()
    assert any("安全レイアウト" in line for line in rebuilt.revision_log)


async def test_llm_dialogue_enrichment(mock_llm_client) -> None:
    structure = sample_structure("persuasion")
    generator = StoryboardGenerator(StoryboardConfig(target_pages=8, seed=5, use_llm_dialogue=True), llm_client=mock_llm_client)
    storyboard = await generator.generate_validated(structure, theme={"THEME": "努力の方向"})
    assert storyboard.grammar.passed
    assert any(line.startswith("LLM台詞増幅") for line in storyboard.revision_log)
    calls = mock_llm_client.telemetry.recent_calls(50)
    assert any(call["task"] == "manga.dialogue" for call in calls)
    stages = {call["stage"] for call in calls if call["task"] == "manga.dialogue"}
    assert {"decompose", "synthesize", "critique", "refine"} <= stages


async def test_llm_dialogue_failure_falls_back_to_templates(tmp_db) -> None:
    from core.llm_client import LLMClient, MockLLMAdapter, TelemetryStore

    adapter = MockLLMAdapter(failure_plan={"manga.dialogue:synthesize": ["garbage"] * 12})
    client = LLMClient(adapter, telemetry=TelemetryStore(tmp_db), max_repair_attempts=1)
    generator = StoryboardGenerator(StoryboardConfig(target_pages=8, seed=5, use_llm_dialogue=True), llm_client=client)
    storyboard = await generator.generate_validated(sample_structure("reversal"))
    assert storyboard.grammar.passed
    assert any("LLM台詞増幅失敗" in line for line in storyboard.revision_log)


def test_mock_handler_registered() -> None:
    import core.manga_pipeline.mock_handlers  # noqa: F401
    from core.llm_client import get_mock_handler

    assert get_mock_handler("manga.dialogue", "synthesize") is not None


# ---------------------------------------------------------------------------
# ComfyUI コンパイラ
# ---------------------------------------------------------------------------


def test_compile_storyboard_valid_workflows() -> None:
    storyboard = _storyboard("inverted", pages=8)
    compiler = ComfyWorkflowCompiler(_recipe())
    workflows = compiler.compile_storyboard(storyboard)
    panels = storyboard.panels_in_reading_order()
    assert len(workflows) == len(panels)
    for workflow, panel in zip(workflows, panels):
        assert validate_prompt(workflow.prompt) == []
        assert workflow.width % 64 == 0 and workflow.height % 64 == 0
        x0, y0, x1, y1 = panel.polygon.bbox()
        aspect = (x1 - x0) * (2.0 if panel.spread_role != "none" else 1.0) / (y1 - y0)
        assert workflow.width / workflow.height == pytest.approx(aspect, rel=0.2)
        assert storyboard.id in workflow.filename_prefix
        classes = {node["class_type"] for node in workflow.prompt.values()}
        assert {"KSampler", "SaveImage", "VAEDecode", "ModelMergeSDXL", "LoraLoader"} <= classes
        assert "close-up" in workflow.positive_text or "shot" in workflow.positive_text
    crops = {w.crop for w in workflows}
    assert "right_half" in crops and "left_half" in crops


def test_regional_conditioning_for_multiple_characters() -> None:
    storyboard = _storyboard("reversal", pages=8)
    compiler = ComfyWorkflowCompiler(_recipe())
    panel = next(p for p in storyboard.panels_in_reading_order() if len(p.characters) >= 2)
    compiled = compiler.compile_panel(storyboard, panel)
    set_masks = [n for n in compiled.prompt.values() if n["class_type"] == "ConditioningSetMask"]
    assert len(set_masks) == len(panel.characters) + 1
    assert len(compiled.regions) == len(panel.characters)
    for region in compiled.regions:
        assert region.x % 8 == 0 and region.y % 8 == 0 and region.width % 8 == 0 and region.height % 8 == 0
        assert region.x + region.width <= compiled.width and region.y + region.height <= compiled.height
    ordered = sorted(compiled.regions, key=lambda r: r.x)
    for a, b in zip(ordered, ordered[1:]):
        assert a.x + a.width <= b.x + 8
    combines = [n for n in compiled.prompt.values() if n["class_type"] == "ConditioningCombine"]
    assert len(combines) == len(panel.characters)


def test_controlnet_chain_and_seed_management() -> None:
    storyboard = _storyboard("reversal", pages=8)
    seeds = SeedManager(base_seed=7)
    compiler = ComfyWorkflowCompiler(_recipe(), CompilerConfig(seed_base=7), seed_manager=seeds)
    panel = storyboard.panels_in_reading_order()[0]
    guides = [ControlGuide(kind="openpose", image_name="pose.png"), ControlGuide(kind="depth", image_name="depth.png", strength=0.5)]
    compiled = compiler.compile_panel(storyboard, panel, guides)
    applies = {node_id: n for node_id, n in compiled.prompt.items() if n["class_type"] == "ControlNetApplyAdvanced"}
    assert len(applies) == 2
    first_id, second_id = sorted(applies, key=int)
    assert applies[second_id]["inputs"]["positive"] == [first_id, 0]
    assert applies[second_id]["inputs"]["negative"] == [first_id, 1]
    sampler = next(n for n in compiled.prompt.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["positive"] == [second_id, 0]
    assert sampler["inputs"]["negative"] == [second_id, 1]
    assert validate_prompt(compiled.prompt) == []
    again = ComfyWorkflowCompiler(_recipe(), CompilerConfig(seed_base=7)).compile_panel(storyboard, panel)
    assert again.seed == compiled.seed
    seeds.lock(panel.id, 123456)
    assert compiler.compile_panel(storyboard, panel).seed == 123456
    assert seeds.is_locked(panel.id)
    seeds.unlock(panel.id)
    assert compiler.compile_panel(storyboard, panel).seed == compiled.seed
    assert any(entry["locked"] for entry in seeds.history)


def test_flux_recipe_compiles() -> None:
    storyboard = _storyboard("persuasion", pages=8)
    compiled = ComfyWorkflowCompiler(_recipe(ModelArchitecture.FLUX)).compile_panel(storyboard, storyboard.panels_in_reading_order()[1])
    classes = {n["class_type"] for n in compiled.prompt.values()}
    assert {"FluxGuidance", "EmptySD3LatentImage", "UNETLoader", "DualCLIPLoader", "ModelMergeFlux1"} <= classes
    sampler = next(n for n in compiled.prompt.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["cfg"] == 1.0
    assert validate_prompt(compiled.prompt) == []


# ---------------------------------------------------------------------------
# ComfyUI クライアント
# ---------------------------------------------------------------------------


def test_make_png_signature() -> None:
    data = make_png(16, 8, seed=3)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in data and b"IEND" in data


async def test_client_with_mock_server_end_to_end() -> None:
    storyboard = _storyboard("inverted", pages=8)
    workflow = ComfyWorkflowCompiler(_recipe()).compile_storyboard(storyboard)[0]
    events = []
    async with MockComfyServer() as server:
        client = ComfyClient(server.base_url, poll_interval_s=0.2)
        result = await client.run_workflow(workflow.prompt, on_progress=events.append, timeout_s=30)
        assert result.status == "success"
        assert result.outputs
        progress = [e for e in events if e.type == "progress"]
        assert progress and progress[-1].value == progress[-1].max
        assert progress[-1].progress == pytest.approx(1.0)
        ref = next(iter(result.outputs.values()))[0]
        image = await client.view_image(ref.filename, ref.subfolder, ref.type)
        assert image.startswith(b"\x89PNG")
        history = await client.get_history(result.prompt_id)
        assert history["status"]["status_str"] == "success"
        again = await client.wait_for_completion(result.prompt_id, timeout_s=5)
        assert again.status == "success"
        info = await client.get_object_info("KSampler")
        assert "KSampler" in info
        stats = await client.system_stats()
        assert stats["devices"]
        await client.interrupt()
        broken = dict(workflow.prompt)
        broken["999"] = {"class_type": "VAEDecode", "inputs": {"samples": ["888", 0], "vae": ["1", 2]}}
        with pytest.raises(ComfyError) as excinfo:
            await client.queue_prompt(broken)
        assert excinfo.value.status_code == 400


class _FakeWebSocket:
    def __init__(self, messages: list[str | bytes]) -> None:
        self.messages = list(messages)

    async def recv(self) -> str | bytes:
        if self.messages:
            return self.messages.pop(0)
        await asyncio.sleep(30)
        return ""

    async def __aenter__(self) -> "_FakeWebSocket":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _transport_handler(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "pid-1", "number": 1, "node_errors": {}})
        if request.url.path == "/history/pid-1":
            return httpx.Response(
                200,
                json={"pid-1": {"outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "caf", "type": "output"}]}}, "status": {"status_str": "success", "completed": True}}},
            )
        return httpx.Response(404, json={})

    return handler


async def test_client_websocket_flow_with_fakes() -> None:
    requests: list[httpx.Request] = []
    messages = [
        json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 1}}}}),
        b"\x00\x01binary-preview",
        json.dumps({"type": "execution_start", "data": {"prompt_id": "pid-1"}}),
        json.dumps({"type": "progress", "data": {"value": 1, "max": 4, "prompt_id": "pid-1", "node": "3"}}),
        json.dumps({"type": "progress", "data": {"value": 4, "max": 4, "prompt_id": "pid-1", "node": "3"}}),
        json.dumps({"type": "executing", "data": {"node": "3", "prompt_id": "other"}}),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "pid-1"}}),
    ]
    received = []

    async def on_progress(event) -> None:
        received.append(event)

    client = ComfyClient(
        "http://127.0.0.1:8188",
        client_id="test-client",
        transport=httpx.MockTransport(_transport_handler(requests)),
        ws_connect=lambda url: _FakeWebSocket(messages),
        poll_interval_s=0.05,
    )
    assert client.ws_url == "ws://127.0.0.1:8188/ws?clientId=test-client"
    assert ComfyClient("https://example.test/base").ws_url.startswith("wss://example.test/base/ws?clientId=")
    result = await client.run_workflow({"1": {"class_type": "SaveImage", "inputs": {}}}, on_progress=on_progress, timeout_s=5)
    assert result.status == "success"
    assert result.outputs["9"][0].filename == "a.png"
    assert [e.type for e in received] == ["status", "execution_start", "progress", "progress", "executing"]
    assert received[2].progress == pytest.approx(0.25)
    body = json.loads(requests[0].content)
    assert body["client_id"] == "test-client" and "prompt" in body


async def test_client_falls_back_to_history_polling_and_errors() -> None:
    requests: list[httpx.Request] = []

    def failing_connect(url: str):
        raise OSError("websocket unavailable")

    client = ComfyClient(
        "http://127.0.0.1:8188",
        transport=httpx.MockTransport(_transport_handler(requests)),
        ws_connect=failing_connect,
        poll_interval_s=0.05,
    )
    result = await client.run_workflow({"1": {"class_type": "SaveImage", "inputs": {}}}, timeout_s=5)
    assert result.status == "success" and result.outputs

    def server_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(ComfyError):
        await ComfyClient("http://127.0.0.1:8188", transport=httpx.MockTransport(server_error)).queue_prompt({"1": {}})

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ComfyConnectionError):
        await ComfyClient("http://127.0.0.1:8188", transport=httpx.MockTransport(unreachable)).system_stats()

    event = ComfyClient.parse_event({"type": "execution_success", "data": {"prompt_id": "x"}})
    assert event.progress == 1.0 and event.prompt_id == "x"


# ---------------------------------------------------------------------------
# SVG レンダラー
# ---------------------------------------------------------------------------


def test_render_pages_and_spread() -> None:
    storyboard = _storyboard("reversal", pages=12)
    renderer = SVGLayoutRenderer(RenderOptions(show_gaze_vectors=True))
    svgs = renderer.render_all(storyboard)
    assert len(svgs) == storyboard.page_count
    for svg in svgs:
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")
        assert root.attrib["viewBox"].startswith("-30 -30")
    spread = next(s for s in storyboard.spreads if s.kind == "double_page_panel")
    spread_svg = renderer.render_spread(storyboard, spread.right_page, spread.left_page)
    root = ET.fromstring(spread_svg)
    nested = [el for el in root if el.tag.endswith("svg")]
    assert len(nested) == 2
    right = next(el for el in nested if el.attrib["data-page-index"] == str(spread.right_page))
    assert float(right.attrib["x"]) > 0.0


def test_render_bleed_edges_and_gaze_toggle() -> None:
    storyboard = _storyboard("reversal", pages=12)
    renderer = SVGLayoutRenderer(RenderOptions(show_gaze_vectors=True))
    found_bleed = found_plain = False
    for page in storyboard.pages:
        svg = renderer.render_page(storyboard, page.index)
        root = ET.fromstring(svg)
        frames = {el.attrib["data-panel-id"]: int(el.attrib["data-edges"]) for el in root.iter() if el.attrib.get("class") == "panel-frame"}
        for panel in page.panels:
            if len(panel.polygon.points) != 4:
                continue
            if panel.bleed_sides:
                assert frames[panel.id] == 4 - len(panel.bleed_sides)
                found_bleed = True
            else:
                assert frames[panel.id] == 4
                found_plain = True
    assert found_bleed and found_plain
    on = sum(svg.count('class="gaze-arrow"') for svg in renderer.render_all(storyboard))
    off = sum(svg.count('class="gaze-arrow"') for svg in SVGLayoutRenderer(RenderOptions(show_gaze_vectors=False)).render_all(storyboard))
    assert on > 0 and off == 0


def test_render_vertical_text_count_images_and_escaping() -> None:
    storyboard = _storyboard("inverted", pages=8)
    panel = next(p for p in storyboard.panels_in_reading_order() if p.balloons)
    panel.balloons[0] = panel.balloons[0].model_copy(update={"text": "<危険&記号>", "columns": ["<危険&", "記号>"]})
    renderer = SVGLayoutRenderer(RenderOptions(image_hrefs={panel.id: "images/panel_0001.png"}))
    svgs = renderer.render_all(storyboard)
    for svg in svgs:
        ET.fromstring(svg)
    expected = sum(len("".join(b.columns)) for p in storyboard.panels_in_reading_order() for b in p.balloons)
    assert sum(svg.count('class="vchar"') for svg in svgs) == expected
    page_svg = svgs[panel.page_index]
    assert 'href="images/panel_0001.png"' in page_svg
    assert "&lt;" in page_svg and "&amp;" in page_svg


def test_layers_json_structure() -> None:
    storyboard = _storyboard("persuasion", pages=8)
    layers = SVGLayoutRenderer().layers(storyboard)
    assert layers["storyboard_id"] == storyboard.id
    assert len(layers["pages"]) == storyboard.page_count
    for page_layers, page in zip(layers["pages"], sorted(storyboard.pages, key=lambda p: p.index)):
        kinds = [layer["type"] for layer in page_layers["layers"]]
        assert kinds.count("guide") == 3
        assert kinds.count("frame") == len(page.panels)
        assert kinds.count("image") == len(page.panels)
        balloons = sum(len(p.balloons) for p in page.panels)
        assert kinds.count("balloon") == balloons and kinds.count("text") == balloons
        assert kinds.count("sfx") == sum(len(p.sfx) for p in page.panels)
        for layer in page_layers["layers"]:
            if layer["type"] == "text":
                assert layer["writing_mode"] == "vertical-rl" and layer["columns"]
    json.dumps(layers, ensure_ascii=False)


def test_area_ratio_and_density_helpers() -> None:
    storyboard = _storyboard("reversal", pages=8)
    page = storyboard.pages[0]
    total = sum(panel_area_ratio(panel, storyboard, page) for panel in page.panels)
    assert 0.5 < total <= 1.0 + 1e-6
    assert page_information_density(page) == pytest.approx(page.information_density, rel=1e-6)


def test_balloon_kinds_present() -> None:
    kinds = set()
    for structure in sample_structures():
        storyboard = StoryboardGenerator(StoryboardConfig(target_pages=12, seed=2)).generate_validated_sync(structure)
        kinds |= {b.kind for p in storyboard.panels_in_reading_order() for b in p.balloons}
    assert {BalloonKind.NORMAL, BalloonKind.SHOUT, BalloonKind.MONOLOGUE, BalloonKind.FLASH} <= kinds


def test_lazy_package_attributes() -> None:
    import core.manga_pipeline as manga

    assert manga.storyboard_generator.StoryboardGenerator is StoryboardGenerator
    assert manga.Storyboard is Storyboard
    with pytest.raises(AttributeError):
        getattr(manga, "does_not_exist")
