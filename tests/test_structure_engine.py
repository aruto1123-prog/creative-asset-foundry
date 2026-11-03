"""Phase 2: 普遍的構造抽出＆異分野転用エンジンのテスト。"""

from __future__ import annotations

import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from core.llm_client import HarnessConfig, LLMClient, MockLLMAdapter, TelemetryStore
from core.structure_engine.composer import (
    MUTATION_DESCRIPTIONS,
    MutationType,
    align_nodes,
    hungarian_assignment,
    interpolate_waves,
    merge_structures,
    merge_structures_with_report,
    mutate_structure,
)
from core.structure_engine.consistency import audit_structure, has_errors, issue_codes
from core.structure_engine.extractor import (
    StructureExtractor,
    deflavor_text,
    find_flavor_leaks,
    heuristic_extract,
    scrub_flavor,
    split_sentences,
)
from core.structure_engine.repository import StructureRepository
from core.structure_engine.samples import sample_structure, sample_structures, sample_text_en, sample_text_ja
from core.structure_engine.schemas import (
    BeatType,
    DomainAgnosticBlueprint,
    DynamicTensionGraph,
    EmotionalInformationWave,
    StructureCategory,
    StructureLayers,
    UniversalStructure,
)
from core.structure_engine.seeds.loader import SEED_FILES, load_all_seeds, load_seed_file, seed_database
from core.structure_engine.similarity import (
    cosine_similarity,
    dtw_distance,
    dtw_path,
    graph_similarity,
    structure_similarity,
    wave_similarity,
)
from core.structure_engine.transpiler import (
    StructureTranspiler,
    TargetDomain,
    transpile_deterministic,
)
from core.structure_engine.visualizer import (
    beats_to_mermaid,
    graph_to_mermaid,
    structure_report_markdown,
    wave_to_svg,
)

JA_PROPER_NOUNS = ("レオン", "ハート", "グレゴール", "アルカディア", "ミスト", "セリス")
EN_PROPER_NOUNS = ("Mara", "Quinn", "Victor", "Hale", "Helix", "Avalon", "Theo")


def _payload(structure: UniversalStructure) -> dict:
    data = structure.model_dump(mode="json")
    data["wave"].pop("information_asymmetry", None)
    return data


def _flat_wave(value: float = 0.5) -> dict:
    grid = [round(i / 20, 6) for i in range(21)]
    return {
        "t": grid,
        "tension": [value] * 21,
        "release": [0.2] * 21,
        "audience_knowledge": [0.5] * 21,
        "character_knowledge": [0.5] * 21,
        "cognitive_load": [0.4] * 21,
    }


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


def test_wave_shape_and_range_validation() -> None:
    good = _flat_wave()
    EmotionalInformationWave.model_validate(good)
    short = {k: v[:20] for k, v in good.items()}
    short["t"] = [round(i / 19, 6) for i in range(20)]
    with pytest.raises(ValidationError):
        EmotionalInformationWave.model_validate(short)
    with pytest.raises(ValidationError):
        EmotionalInformationWave.model_validate({**good, "tension": [1.2] * 21})
    with pytest.raises(ValidationError):
        EmotionalInformationWave.model_validate({**good, "t": [round(i / 21, 6) for i in range(21)]})
    with pytest.raises(ValidationError):
        EmotionalInformationWave.model_validate({**good, "release": [0.2] * 22})


def test_wave_computed_asymmetry_roundtrip_and_metrics() -> None:
    inverted = sample_structure("inverted")
    reversal = sample_structure("reversal")
    dumped = inverted.wave.model_dump(mode="json")
    assert "information_asymmetry" in dumped
    restored = EmotionalInformationWave.model_validate(dumped)
    assert restored.information_asymmetry == inverted.wave.information_asymmetry
    assert inverted.wave.suspense_coefficient() > reversal.wave.suspense_coefficient()
    resampled = reversal.wave.resample(41)
    assert resampled.n_points == 41
    assert abs(resampled.peak_t() - reversal.wave.peak_t()) <= 0.06
    summary = reversal.wave.summary()
    assert {"peak_t", "suspense", "surprise", "catharsis"} <= set(summary)


def test_cross_layer_reference_validation() -> None:
    data = _payload(sample_structure("reversal"))
    bad_beat = json.loads(json.dumps(data))
    bad_beat["beats"]["beats"][0]["involved_nodes"] = ["ghost_node"]
    with pytest.raises(ValidationError):
        UniversalStructure.model_validate(bad_beat)
    bad_mechanism = json.loads(json.dumps(data))
    bad_mechanism["mechanics"]["mechanisms"][0]["linked_beats"] = ["no_such_beat"]
    with pytest.raises(ValidationError):
        UniversalStructure.model_validate(bad_mechanism)
    bad_slot = json.loads(json.dumps(data))
    bad_slot["tension_graph"]["nodes"][0]["slot"] = "UNDEFINED_SLOT"
    with pytest.raises(ValidationError):
        UniversalStructure.model_validate(bad_slot)
    bad_cause = json.loads(json.dumps(data))
    bad_cause["beats"]["beats"][1]["causes"] = ["afterglow"]
    with pytest.raises(ValidationError):
        UniversalStructure.model_validate(bad_cause)


def test_blueprint_validation_and_render() -> None:
    with pytest.raises(ValidationError):
        DomainAgnosticBlueprint.model_validate(
            {"template": "{{UNDEFINED_SLOT}} が目標を追う骨格テンプレートです。", "slots": [{"name": "HERO", "description": "主体"}], "invariants": ["x"]}
        )
    with pytest.raises(ValidationError):
        DomainAgnosticBlueprint.model_validate(
            {"template": "スロットを一切含まない骨格テンプレートの文です。", "slots": [{"name": "HERO", "description": "主体"}], "invariants": ["x"]}
        )
    blueprint = sample_structure("reversal").blueprint
    with pytest.raises(ValueError):
        blueprint.render({"PROTAGONIST": "少年"})
    rendered = blueprint.render({"PROTAGONIST": "少年"}, strict=False)
    assert "少年" in rendered and "〔" in rendered
    assert blueprint.placeholders()[0] == "PROTAGONIST"


def test_graph_validation_rules() -> None:
    nodes = [{"id": "a", "kind": "agent", "role": "r", "label": "l"}, {"id": "b", "kind": "desire", "role": "r", "label": "l"}]
    with pytest.raises(ValidationError):
        DynamicTensionGraph.model_validate({"nodes": nodes, "edges": [{"source": "a", "target": "a", "kind": "pursuit"}]})
    with pytest.raises(ValidationError):
        DynamicTensionGraph.model_validate(
            {"nodes": nodes, "edges": [{"source": "a", "target": "b", "kind": "pursuit"}, {"source": "a", "target": "b", "kind": "pursuit"}]}
        )
    graph = sample_structure("reversal").tension_graph
    assert abs(sum(graph.power_balance().values())) < 1e-9


# ---------------------------------------------------------------------------
# deflavor / heuristic extract
# ---------------------------------------------------------------------------


def test_deflavor_japanese_removes_proper_nouns_and_genre_terms() -> None:
    result = deflavor_text(sample_text_ja())
    surfaces = {token.surface for token in result.tokens}
    assert {"レオン・ハート", "グレゴール", "セリス", "アルカディア", "ミスト村"} <= surfaces
    for noun in JA_PROPER_NOUNS:
        assert noun not in result.abstracted_text
    for term in ("魔法", "禁術", "結界", "魔力", "古文書"):
        assert term not in result.abstracted_text
    assert "[人物A]" in result.abstracted_text
    assert max(result.genre_signals, key=result.genre_signals.get) == "fantasy"
    leon = [t for t in result.tokens if t.surface in ("レオン・ハート", "レオン")]
    assert len({t.placeholder for t in leon}) == 1
    assert all(t.category == "place" for t in result.tokens if t.surface in ("アルカディア", "ミスト村"))


def test_deflavor_english_and_aliases() -> None:
    result = deflavor_text(sample_text_en())
    for noun in EN_PROPER_NOUNS:
        assert noun not in result.abstracted_text
    categories = {t.surface: t.category for t in result.tokens}
    assert categories["Helix Corporation"] == "organization"
    assert categories["New Avalon"] == "place"
    hale = {t.placeholder for t in result.tokens if t.surface in ("Victor Hale", "Hale")}
    assert len(hale) == 1
    assert "In the city of" in result.abstracted_text


def test_deflavor_honorifics_brackets_surnames_and_common_words() -> None:
    text = "佐藤さんは『ネメシス』に入った。田中ハルトはライバルのシステムを壊した。東都大学の山田太郎は笑った。真剣に剣を振るう。"
    result = deflavor_text(text)
    surfaces = {t.surface: t.category for t in result.tokens}
    assert surfaces["佐藤"] == "person"
    assert surfaces["ネメシス"] == "title"
    assert surfaces["田中ハルト"] == "person"
    assert surfaces["東都大学"] == "place"
    assert surfaces["山田太郎"] == "person"
    assert "ライバル" in result.abstracted_text and "システム" in result.abstracted_text
    assert "真剣に武器を振るう" in result.abstracted_text
    assert deflavor_text("").tokens == []


def test_split_sentences_mixed() -> None:
    assert split_sentences("一文目。二文目！ Third sentence. Fourth one?") == ["一文目。", "二文目！", "Third sentence.", "Fourth one?"]


@pytest.mark.parametrize("text", [sample_text_ja(), sample_text_en(), "主人公は旅に出た。", "x"])
def test_heuristic_extract_produces_valid_clean_structure(text: str) -> None:
    layers = heuristic_extract(text)
    assert isinstance(layers, StructureLayers)
    structure = UniversalStructure.from_layers(layers, id="st_heuristic_probe")
    assert not has_errors(audit_structure(structure))
    assert "peak_without_pivot" not in issue_codes(audit_structure(structure))
    assert len(structure.beats.beats) >= 6
    assert len(structure.beats.turning_points()) >= 2
    assert 3 <= len(structure.mechanics.mechanisms) <= 5
    breakthrough = structure.beats.by_type(BeatType.COSTLY_BREAKTHROUGH)[0]
    assert abs(breakthrough.t - structure.wave.peak_t()) < 1e-6
    tokens = deflavor_text(text).tokens
    assert find_flavor_leaks(structure, tokens) == []
    joined = "\n".join(structure.text_fields())
    for noun in (*JA_PROPER_NOUNS, *EN_PROPER_NOUNS):
        assert noun not in joined


def test_heuristic_extract_is_deterministic_and_rejects_empty() -> None:
    first = heuristic_extract(sample_text_ja()).model_dump()
    second = heuristic_extract(sample_text_ja()).model_dump()
    assert first == second
    with pytest.raises(ValueError):
        heuristic_extract("   ")


def test_heuristic_detects_betrayal_and_roles() -> None:
    layers = heuristic_extract(sample_text_ja())
    kinds = {edge.kind.value for edge in layers.tension_graph.edges}
    assert {"betrayal", "dependency", "opposition", "pursuit"} <= kinds
    roles = {node.id: node.role for node in layers.tension_graph.nodes}
    assert roles["antagonist"] == "false_mentor"
    assert "ally" in roles
    assert layers.beats.get("twist").beat_type == BeatType.STRUCTURAL_TWIST


# ---------------------------------------------------------------------------
# extractor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extractor_with_mock_llm(mock_llm_client) -> None:
    extractor = StructureExtractor(mock_llm_client)
    result = await extractor.extract(sample_text_ja(), category=StructureCategory.NARRATIVE_ARCHETYPE)
    assert result.used_llm
    assert result.harness_run_id
    assert result.critique_score is not None
    structure = result.structure
    assert structure.metadata.source_kind.value == "extracted"
    assert structure.metadata.category == StructureCategory.NARRATIVE_ARCHETYPE
    assert find_flavor_leaks(structure, result.deflavor.tokens) == []
    joined = "\n".join(structure.text_fields())
    for noun in JA_PROPER_NOUNS:
        assert noun not in joined
    run = mock_llm_client.telemetry.get_run(result.harness_run_id)
    assert [log["stage"] for log in run["stage_logs"]] == ["decompose", "synthesize", "critique", "refine"]
    assert run["calls"]


@pytest.mark.asyncio
async def test_extractor_without_llm() -> None:
    result = await StructureExtractor().extract(sample_text_en(), source_kind="lp")
    assert not result.used_llm
    assert result.harness_run_id is None
    assert result.structure.metadata.origin_domain == "marketing"
    with pytest.raises(ValueError):
        await StructureExtractor().extract("  ")


@pytest.mark.asyncio
async def test_extractor_falls_back_when_harness_fails(tmp_db) -> None:
    adapter = MockLLMAdapter(failure_plan={"structure.extract:synthesize": ["garbage"] * 6})
    client = LLMClient(adapter, telemetry=TelemetryStore(tmp_db), max_repair_attempts=1)
    extractor = StructureExtractor(client, HarnessConfig(max_repair_attempts=1))
    result = await extractor.extract(sample_text_ja())
    assert not result.used_llm
    assert result.harness_run_id is None
    assert find_flavor_leaks(result.structure, result.deflavor.tokens) == []


def test_find_and_scrub_flavor_leaks() -> None:
    tokens = deflavor_text(sample_text_ja()).tokens
    data = _payload(sample_structure("reversal"))
    data["tension_graph"]["nodes"][0]["label"] = "レオンという主体"
    data["beats"]["beats"][0]["function"] = "[人物A]が魔法を使う"
    data["metadata"]["lineage"] = ["レオン型の系譜"]
    leaky = UniversalStructure.model_validate(data)
    leaks = find_flavor_leaks(leaky, tokens)
    assert {"レオン", "[人物A]", "魔法"} <= set(leaks)
    cleaned = scrub_flavor(leaky, tokens)
    assert find_flavor_leaks(cleaned, tokens) == []
    assert cleaned.metadata.lineage == ["レオン型の系譜"]
    assert cleaned.tension_graph.nodes[0].label == "ある人物という主体"
    assert cleaned.beats.beats[0].function == "ある人物が特殊能力を使う"


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------


def _peak_wave(peak: float) -> EmotionalInformationWave:
    return EmotionalInformationWave.from_keyframes(
        {"tension": [(0.0, 0.1), (peak - 0.08, 0.15), (peak, 0.95), (peak + 0.08, 0.15), (1.0, 0.1)]}
    )


def test_wave_similarity_self_and_bounds() -> None:
    for structure in sample_structures():
        result = wave_similarity(structure.wave, structure.wave)
        assert result["score"] == pytest.approx(1.0)
        assert result["dtw_distance"] == pytest.approx(0.0)
    a, b, c = sample_structures()
    for method in ("dtw", "cosine", "hybrid"):
        score = wave_similarity(a.wave, b.wave, method=method)["score"]
        assert 0.0 <= score < 1.0
    with pytest.raises(ValueError):
        wave_similarity(a.wave, b.wave, method="euclid")  # type: ignore[arg-type]


def test_dtw_is_more_robust_to_time_shift_than_cosine() -> None:
    result = wave_similarity(_peak_wave(0.4), _peak_wave(0.55), channels=("tension",))
    cosine_score = (result["cosine"] + 1.0) / 2.0
    assert result["dtw_similarity"] > 0.8
    assert result["dtw_similarity"] > cosine_score + 0.2


def test_dtw_window_and_path_properties() -> None:
    x = np.sin(np.linspace(0, 6, 30))
    y = np.sin(np.linspace(0.6, 6.6, 30))
    diagonal = dtw_distance(x, y, window=0)
    default = dtw_distance(x, y)
    full = dtw_distance(x, y, window=30)
    assert diagonal >= default >= full - 1e-12
    assert diagonal == pytest.approx(float(np.mean(np.abs(x - y))))
    distance, path = dtw_path(np.arange(10.0), np.arange(15.0) * (9 / 14))
    assert path[0] == (0, 0) and path[-1] == (9, 14)
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        assert 0 <= i1 - i0 <= 1 and 0 <= j1 - j0 <= 1 and (i1, j1) != (i0, j0)
    assert distance >= 0.0


def test_cosine_graph_and_structure_similarity() -> None:
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    a, b, _ = sample_structures()
    assert graph_similarity(a.tension_graph, a.tension_graph) == pytest.approx(1.0)
    assert graph_similarity(a.tension_graph, b.tension_graph) < 1.0
    assert structure_similarity(a, a) == pytest.approx(1.0)
    assert structure_similarity(a, b) < structure_similarity(a, a)


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------


def test_audit_samples_have_no_errors() -> None:
    for structure in sample_structures():
        issues = audit_structure(structure)
        assert not has_errors(issues)
    for kind in ("reversal", "inverted", "persuasion"):
        assert issue_codes(audit_structure(sample_structure(kind))) == set()


def _variant(kind: str, edit) -> UniversalStructure:
    data = _payload(sample_structure(kind))
    edit(data)
    return UniversalStructure.model_validate(data)


def _no_agent(data: dict) -> None:
    for node in data["tension_graph"]["nodes"]:
        if node["kind"] == "agent":
            node["kind"] = "symbol"


def _remove_dependency(data: dict) -> None:
    data["tension_graph"]["edges"] = [e for e in data["tension_graph"]["edges"] if e["kind"] != "dependency"]


def _orphan(data: dict) -> None:
    data["beats"]["beats"][2]["causes"] = []


def _no_pivot(data: dict) -> None:
    for beat in data["beats"]["beats"]:
        beat["is_turning_point"] = False
        if beat["beat_type"] in ("false_victory", "revelation", "climax", "crisis", "structural_twist", "costly_breakthrough"):
            beat["beat_type"] = "escalation"


def _flat(data: dict) -> None:
    data["wave"] = _flat_wave()
    for beat in data["beats"]["beats"]:
        beat["information_shift"] = 0.0


def _wrong_twist_sign(data: dict) -> None:
    for beat in data["beats"]["beats"]:
        if beat["beat_type"] == "structural_twist":
            beat["information_shift"] = -0.5


def _no_resolution(data: dict) -> None:
    for beat in data["beats"]["beats"]:
        if beat["beat_type"] in ("climax", "afterglow", "payoff", "costly_breakthrough"):
            beat["beat_type"] = "transition"


def _unused_slot(data: dict) -> None:
    data["blueprint"]["slots"].append({"name": "EXTRA_SLOT", "description": "任意の補助スロット", "required": False})


@pytest.mark.parametrize(
    ("code", "kind", "edit"),
    [
        ("no_agent", "persuasion", _no_agent),
        ("betrayal_without_dependency", "persuasion", _remove_dependency),
        ("orphan_beat", "inverted", _orphan),
        ("peak_without_pivot", "inverted", _no_pivot),
        ("turning_point_flat", "persuasion", _flat),
        ("flat_wave", "persuasion", _flat),
        ("twist_info_sign", "reversal", _wrong_twist_sign),
        ("missing_resolution", "inverted", _no_resolution),
        ("slot_unused", "inverted", _unused_slot),
    ],
)
def test_audit_detects_issue(code: str, kind: str, edit) -> None:
    issues = audit_structure(_variant(kind, edit))
    assert code in issue_codes(issues)
    if code == "no_agent":
        assert has_errors(issues)


# ---------------------------------------------------------------------------
# composer
# ---------------------------------------------------------------------------


def test_hungarian_matches_bruteforce() -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        matrix = rng.random((n, m))
        cost = sum(matrix[i, j] for i, j in hungarian_assignment(matrix))
        if n <= m:
            best = min(sum(matrix[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        else:
            best = min(sum(matrix[p[j], j] for j in range(m)) for p in itertools.permutations(range(n), m))
        assert cost == pytest.approx(best)


def test_align_nodes_identity() -> None:
    structure = sample_structure("reversal")
    mapping, costs = align_nodes(structure.tension_graph, structure.tension_graph)
    assert mapping == {node.id: node.id for node in structure.tension_graph.nodes}
    assert all(cost == pytest.approx(0.0) for cost in costs.values())


def test_interpolate_waves_extremes() -> None:
    a = sample_structure("reversal").wave
    b = sample_structure("inverted").wave
    for align in (False, True):
        pure_a = interpolate_waves(a, b, 0.0, align=align)
        pure_b = interpolate_waves(a, b, 1.0, align=align)
        np.testing.assert_allclose(pure_a.channel("tension"), a.channel("tension"), atol=1e-3)
        np.testing.assert_allclose(pure_b.channel("tension"), b.channel("tension"), atol=1e-3)
    half = interpolate_waves(a, b, 0.5, align=False)
    expected = 0.5 * a.channel("tension") + 0.5 * b.channel("tension")
    np.testing.assert_allclose(half.channel("tension"), expected, atol=1e-3)
    aligned = interpolate_waves(a, b, 0.7)
    assert aligned.n_points == 21


def test_merge_structures_basic_report() -> None:
    a, b = sample_structure("reversal"), sample_structure("inverted")
    merged, report = merge_structures_with_report(a, b, 0.7, 0.7)
    assert merged.metadata.source_kind.value == "merged"
    assert merged.metadata.parent_ids == [a.id, b.id]
    assert merged.metadata.title == f"{a.metadata.title} × {b.metadata.title}"
    assert set(report.node_mapping) == {node.id for node in b.tension_graph.nodes}
    assert report.weight_topology == pytest.approx(0.7) and report.weight_wave == pytest.approx(0.7)
    assert report.dtw_alignment_cost >= 0.0
    codes = issue_codes(audit_structure(merged))
    assert "no_agent" not in codes and "peak_without_pivot" not in codes
    assert "合成波形" in merged.blueprint.invariants[-1]
    assert set(a.blueprint.placeholders()) <= set(merged.blueprint.placeholders())
    assert merge_structures(a, b, 0.7, 0.7, new_id="mg_fixed_id").id == "mg_fixed_id"


def test_merge_topology_weight_extremes() -> None:
    a, b = sample_structure("reversal"), sample_structure("inverted")
    pure_a, _ = merge_structures_with_report(a, b, 1.0, 0.5)
    assert not any(node.id.startswith("b_") for node in pure_a.tension_graph.nodes)
    a_keys = {(e.source, e.target, e.kind.value) for e in a.tension_graph.edges}
    merged_keys = {(e.source, e.target, e.kind.value) for e in pure_a.tension_graph.edges}
    assert a_keys <= merged_keys
    pure_b, report = merge_structures_with_report(a, b, 0.0, 0.5)
    keys = {(e.source, e.target, e.kind.value) for e in pure_b.tension_graph.edges}
    for edge in b.tension_graph.edges:
        source, target = report.node_mapping[edge.source], report.node_mapping[edge.target]
        assert source and target
        assert (source, target, edge.kind.value) in keys


def test_merge_wave_weight_extremes() -> None:
    a, b = sample_structure("reversal"), sample_structure("persuasion")
    wave_b = merge_structures(a, b, 0.7, 1.0).wave
    wave_a = merge_structures(a, b, 0.7, 0.0).wave
    np.testing.assert_allclose(wave_b.channel("tension"), b.wave.channel("tension"), atol=1e-3)
    np.testing.assert_allclose(wave_a.channel("tension"), a.wave.channel("tension"), atol=1e-3)


def test_merge_repairs_connectivity_and_betrayal() -> None:
    def _disconnect(data: dict) -> None:
        data["tension_graph"]["edges"] = [
            edge
            for edge in data["tension_graph"]["edges"]
            if not (edge["source"] == "seeker" and edge["target"] == "wound")
        ]

    reversal = _variant("reversal", _disconnect)
    assert "disconnected_graph" in issue_codes(audit_structure(reversal))
    inverted = sample_structure("inverted")
    merged, report = merge_structures_with_report(reversal, inverted, 1.0, 0.0)
    assert "disconnected_graph" not in issue_codes(audit_structure(merged))
    assert any("非連結" in fix for fix in report.fixes_applied)
    for structure in sample_structures():
        for other in sample_structures():
            combined = merge_structures(structure, other, 0.6, 0.6)
            assert "betrayal_without_dependency" not in issue_codes(audit_structure(combined))
            assert not has_errors(audit_structure(combined))


@pytest.mark.parametrize("mutation", list(MutationType))
@pytest.mark.parametrize("kind", ["reversal", "inverted", "persuasion"])
def test_mutations_produce_valid_structures(mutation: MutationType, kind: str) -> None:
    original = sample_structure(kind)
    snapshot = original.model_dump()
    mutated = mutate_structure(original, mutation, seed=5)
    assert mutated.metadata.source_kind.value == "mutated"
    assert mutated.metadata.parent_ids == [original.id]
    assert mutated.id != original.id
    assert not has_errors(audit_structure(mutated))
    assert mutation.value in MUTATION_DESCRIPTIONS
    assert original.model_dump() == snapshot


def _strip_volatile(structure: UniversalStructure) -> dict:
    data = structure.model_dump(mode="json")
    data.pop("id")
    data["metadata"].pop("created_at")
    return data


def test_mutation_numeric_effects() -> None:
    reversal = sample_structure("reversal")

    disclosed = mutate_structure(reversal, MutationType.INVERT_DISCLOSURE)
    assert disclosed.wave.suspense_coefficient() > reversal.wave.suspense_coefficient() + 0.05
    assert disclosed.wave.surprise_coefficient() < reversal.wave.surprise_coefficient()
    assert disclosed.beats.beats[0].id == "irony_setup"

    inverted_power = mutate_structure(reversal, MutationType.INVERT_POWER_GRADIENT)
    before = reversal.tension_graph.power_balance()
    after = inverted_power.tension_graph.power_balance()
    for node_id, value in before.items():
        if abs(value) > 1e-6:
            assert after[node_id] == pytest.approx(-value)

    false_climax = mutate_structure(reversal, MutationType.INSERT_FALSE_CLIMAX, seed=1)
    fv = false_climax.beats.by_type(BeatType.FALSE_VICTORY)[0]
    collapse = false_climax.beats.get("false_victory_collapse")
    wave = false_climax.wave
    assert wave.sample("tension", fv.t) > wave.sample("tension", fv.t - 0.05)
    assert wave.sample("tension", fv.t) > wave.sample("tension", fv.t + 0.05)
    assert wave.sample("tension", fv.t) - wave.sample("tension", collapse.t) > 0.2

    reversed_struct = mutate_structure(reversal, MutationType.REVERSE_CHRONOLOGY)
    original_rev = reversal.wave.channel("tension")[::-1]
    corr = float(np.corrcoef(reversed_struct.wave.channel("tension")[1:-1], original_rev[1:-1])[0, 1])
    assert corr > 0.7
    beats = reversed_struct.beats.beats
    assert beats[0].beat_type == BeatType.HOOK and beats[-1].beat_type == BeatType.PAYOFF
    assert all(beat.causes == [beats[i].id] for i, beat in enumerate(beats[1:]))

    betrayed = mutate_structure(sample_structure("inverted"), MutationType.ADD_BETRAYAL)
    betrayals = betrayed.tension_graph.edges_of_kind("betrayal")
    assert betrayals and betrayals[0].activation_t == pytest.approx(0.68)
    pair = {betrayals[0].source, betrayals[0].target}
    assert any({e.source, e.target} == pair for e in betrayed.tension_graph.edges_of_kind("dependency"))
    assert betrayed.beats.get("betrayal_reveal").t == pytest.approx(0.68)

    swapped = mutate_structure(reversal, MutationType.PERSPECTIVE_SWAP)
    roles = {node.id: node.role for node in swapped.tension_graph.nodes}
    original_roles = {node.id: node.role for node in reversal.tension_graph.nodes}
    assert roles["seeker"] == original_roles["guardian"] and roles["guardian"] == original_roles["seeker"]
    assert swapped.mechanics.empathy_anchor.startswith("視点交換")

    compressed = mutate_structure(reversal, MutationType.COMPRESS_MIDDLE)
    assert compressed.beats.get("twist").t < 0.6
    assert compressed.wave.peak_t() < reversal.wave.peak_t()


def test_mutation_intensity_zero_and_determinism() -> None:
    reversal = sample_structure("reversal")
    for mutation in (MutationType.INVERT_DISCLOSURE, MutationType.INSERT_FALSE_CLIMAX, MutationType.ADD_BETRAYAL):
        unchanged = mutate_structure(reversal, mutation, intensity=0.0)
        np.testing.assert_allclose(unchanged.wave.as_matrix(), reversal.wave.as_matrix(), atol=1e-4)
    first = mutate_structure(reversal, MutationType.INSERT_FALSE_CLIMAX, seed=9)
    second = mutate_structure(reversal, MutationType.INSERT_FALSE_CLIMAX, seed=9)
    assert _strip_volatile(first) == _strip_volatile(second)
    with pytest.raises(ValueError):
        mutate_structure(reversal, "not_a_mutation")


# ---------------------------------------------------------------------------
# transpiler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", list(TargetDomain))
@pytest.mark.parametrize("kind", ["reversal", "inverted", "persuasion"])
def test_transpile_deterministic_preserves_structure(domain: TargetDomain, kind: str) -> None:
    structure = sample_structure(kind)
    design = transpile_deterministic(structure, domain, {"THEME": "失われた居場所"})
    assert [section.beat_id for section in design.sections] == [beat.id for beat in structure.beats.beats]
    assert design.fidelity.beat_coverage == pytest.approx(1.0)
    assert design.fidelity.wave_correlation > 0.9
    assert design.fidelity.topology_preserved
    assert design.fidelity.score > 0.9
    assert "失われた居場所" in design.rendered_blueprint
    for section in design.sections:
        beat = structure.beats.get(section.beat_id)
        assert section.tension_target == pytest.approx(structure.wave.sample("tension", beat.t), abs=1e-4)
        assert section.t_start <= beat.t <= section.t_end
    expected_keys = {
        TargetDomain.MANGA_PLOT: {"characters", "page_plan", "page_turn_reveals", "spread_candidates"},
        TargetDomain.MARKETING_LP: {"funnel_stages", "sections", "cta", "objections"},
        TargetDomain.ARTICLE: {"thesis", "outline", "figures", "evidence_plan"},
    }[domain]
    assert expected_keys <= set(design.domain_payload)


def test_transpile_manga_page_plan_allocation() -> None:
    structure = sample_structure("reversal")
    for pages in ("16", "24"):
        design = transpile_deterministic(structure, "manga_plot", {"PAGES": pages})
        plan = design.domain_payload["page_plan"]
        assert sum(item["pages"] for item in plan) == int(pages)
        assert design.domain_payload["pages_total"] == int(pages)
    with pytest.raises(ValueError):
        transpile_deterministic(structure, "screenplay")


@pytest.mark.asyncio
async def test_transpile_llm_path_keeps_numbers(mock_llm_client) -> None:
    structure = sample_structure("persuasion")
    baseline = transpile_deterministic(structure, TargetDomain.ARTICLE, {"THEME": "独学"})
    design = await StructureTranspiler(mock_llm_client).transpile(structure, TargetDomain.ARTICLE, {"THEME": "独学"})
    assert design.domain_payload["harness_run_id"]
    assert [s.tension_target for s in design.sections] == [s.tension_target for s in baseline.sections]
    assert all("LLM増幅" in s.notes for s in design.sections)
    assert design.fidelity.score == pytest.approx(baseline.fidelity.score)


# ---------------------------------------------------------------------------
# repository
# ---------------------------------------------------------------------------


def test_repository_crud_and_filters(tmp_db) -> None:
    repo = StructureRepository(tmp_db)
    saved = sample_structures()
    for structure in saved:
        repo.save(structure)
    assert repo.count() == 3
    assert repo.get("sample_inverted") == saved[1]
    assert repo.get("missing_structure") is None
    assert [s.id for s in repo.list(category="persuasion_marketing")] == ["sample_persuasion"]
    assert [s.id for s in repo.list(tag="サスペンス")] == ["sample_inverted"]
    assert "sample_inverted" in [s.id for s in repo.list(query="倒叙")]
    assert repo.count(category="custom") == 2
    assert repo.category_counts() == {"custom": 2, "persuasion_marketing": 1}
    first_created = repo.list(query="庇護者")[0].created_at
    repo.save(sample_structure("reversal"))
    assert repo.list(query="庇護者")[0].created_at == first_created
    summary = repo.list(order="title", limit=1)[0]
    assert summary.peak_t > 0
    assert repo.delete("sample_reversal") is True
    assert repo.delete("sample_reversal") is False
    assert repo.count() == 2


def test_repository_similarity_search(tmp_db) -> None:
    repo = StructureRepository(tmp_db)
    for structure in sample_structures():
        repo.save(structure)
    for kind in ("reversal", "inverted"):
        repo.save(mutate_structure(sample_structure(kind), MutationType.COMPRESS_MIDDLE, new_id=f"mu_{kind}_compressed"))
    query = sample_structure("reversal")
    hits = repo.search_similar(query, top_k=10)
    assert hits[0].id == "sample_reversal" and hits[0].score == pytest.approx(1.0)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    excluded = repo.search_similar(query, method="dtw", top_k=3, exclude_ids=("sample_reversal",))
    assert "sample_reversal" not in {h.id for h in excluded}
    assert len(excluded) == 3
    only_custom = repo.search_similar(query.wave, category="persuasion_marketing")
    assert [h.id for h in only_custom] == ["sample_persuasion"]


# ---------------------------------------------------------------------------
# visualizer
# ---------------------------------------------------------------------------


def test_graph_to_mermaid_shapes_and_arrows() -> None:
    structure = sample_structure("reversal")
    code = graph_to_mermaid(structure)
    assert code.startswith("flowchart LR")
    for shape in ('["', '(["', '{{"', '[("'):
        assert shape in code
    assert "-.->" in code and "==>" in code
    data = _payload(structure)
    data["tension_graph"]["nodes"][0]["label"] = 'いわゆる"英雄"'
    escaped = graph_to_mermaid(UniversalStructure.model_validate(data).tension_graph)
    assert "#quot;英雄#quot;" in escaped


def test_beats_to_mermaid_edges_and_classes() -> None:
    structure = sample_structure("inverted")
    code = beats_to_mermaid(structure)
    assert code.startswith("flowchart TD")
    assert sum(1 for line in code.splitlines() if " --> " in line) == len(structure.beats.causal_edges())
    assert ":::turning" in code


def test_wave_to_svg_is_valid_xml() -> None:
    structure = sample_structure("reversal")
    other = sample_structure("inverted")
    svg = wave_to_svg(structure.wave, beats=structure.beats, title="<A&B> 波形", compare=[("比較", other.wave)])
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    channels = [p for p in root.iter(f"{ns}polyline") if p.get("class") == "channel"]
    compares = [p for p in root.iter(f"{ns}polyline") if p.get("class") == "compare"]
    markers = [g for g in root.iter(f"{ns}g") if g.get("class") == "beat-marker"]
    assert len(channels) == 4
    assert len(compares) == 4
    assert len(markers) == len(structure.beats.beats)
    assert "&lt;A&amp;B&gt;" in svg


def test_structure_report_markdown_sections() -> None:
    report = structure_report_markdown(sample_structure("persuasion"))
    assert report.startswith("# 常識転覆と新しい枠組みの提示")
    assert report.count("```mermaid") == 2
    assert "Layer 5" in report and "不変条件" in report


# ---------------------------------------------------------------------------
# seeds loader
# ---------------------------------------------------------------------------


def _write_seed_files(directory: Path) -> dict[str, list[str]]:
    plan = {
        "narrative_archetype": ("na", ["reversal", "inverted"]),
        "manga_panel_rhythm": ("mr", ["inverted"]),
        "persuasion_marketing": ("pm", ["persuasion", "reversal"]),
    }
    written: dict[str, list[str]] = {}
    for category, (prefix, kinds) in plan.items():
        patterns = []
        for index, kind in enumerate(kinds, start=1):
            data = _payload(sample_structure(kind))
            data["id"] = f"{prefix}_{index:02d}_{kind}"
            data["metadata"]["category"] = category
            data["metadata"]["source_kind"] = "seed"
            patterns.append(data)
        written[category] = [p["id"] for p in patterns]
        (directory / SEED_FILES[category]).write_text(
            json.dumps({"schema_version": "1.0", "category": category, "description": "テスト用", "patterns": patterns}, ensure_ascii=False),
            encoding="utf-8",
        )
    return written


def test_seed_loader_roundtrip(tmp_path: Path, tmp_db) -> None:
    written = _write_seed_files(tmp_path)
    narrative = load_seed_file(SEED_FILES["narrative_archetype"], tmp_path)
    assert [s.id for s in narrative] == written["narrative_archetype"]
    everything = load_all_seeds(tmp_path)
    assert len(everything) == 5
    repo = StructureRepository(tmp_db)
    first = seed_database(repo, directory=tmp_path)
    assert (first.inserted, first.skipped, first.updated, first.total) == (5, 0, 0, 5)
    assert first.by_category == {"narrative_archetype": 2, "manga_panel_rhythm": 1, "persuasion_marketing": 2}
    second = seed_database(repo, directory=tmp_path)
    assert (second.inserted, second.skipped) == (0, 5)
    third = seed_database(repo, directory=tmp_path, overwrite=True)
    assert third.updated == 5
    assert repo.count() == 5
    explicit = seed_database(repo, structures=[sample_structure("persuasion")])
    assert explicit.inserted == 1


def test_seed_loader_rejects_invalid_files(tmp_path: Path) -> None:
    _write_seed_files(tmp_path)
    path = tmp_path / SEED_FILES["manga_panel_rhythm"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["category"] = "persuasion_marketing"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_seed_file(path)
    payload["category"] = "manga_panel_rhythm"
    payload["schema_version"] = "9.0"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_seed_file(path)
    with pytest.raises(FileNotFoundError):
        load_seed_file("no_such_file.json", tmp_path)
