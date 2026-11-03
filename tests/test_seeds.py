"""Phase 6 / Phase 9: 初期プリセット「普遍的構造データベース」100パターンの検証。"""

from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pytest

from core.structure_engine.schemas import StructureCategory, UniversalStructure
from core.structure_engine.seeds.loader import (
    SEED_FILES,
    load_all_seeds,
    load_seed_file,
    seed_database,
    seed_dir,
)

_PREFIX = {
    StructureCategory.NARRATIVE_ARCHETYPE.value: "na_",
    StructureCategory.MANGA_PANEL_RHYTHM.value: "mr_",
    StructureCategory.PERSUASION_MARKETING.value: "pm_",
    StructureCategory.ADVANCED_NARRATIVE.value: "an_",
    StructureCategory.MANGA_DIRECTION.value: "md_",
    StructureCategory.EMOTIONAL_WAVE.value: "ew_",
}
_EXPECTED_COUNTS = {
    StructureCategory.NARRATIVE_ARCHETYPE.value: 20,
    StructureCategory.MANGA_PANEL_RHYTHM.value: 20,
    StructureCategory.PERSUASION_MARKETING.value: 20,
    StructureCategory.ADVANCED_NARRATIVE.value: 14,
    StructureCategory.MANGA_DIRECTION.value: 13,
    StructureCategory.EMOTIONAL_WAVE.value: 13,
}
_TOTAL = sum(_EXPECTED_COUNTS.values())
_RHYTHM_CATEGORIES = {StructureCategory.MANGA_PANEL_RHYTHM, StructureCategory.MANGA_DIRECTION}
_EXTENSION_CATEGORIES = {
    StructureCategory.ADVANCED_NARRATIVE.value,
    StructureCategory.MANGA_DIRECTION.value,
    StructureCategory.EMOTIONAL_WAVE.value,
}
_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)


@pytest.fixture(scope="module")
def all_seeds() -> list[UniversalStructure]:
    return load_all_seeds()


def _tension(structure: UniversalStructure) -> np.ndarray:
    return structure.wave.resample_arrays(21)["tension"]


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def test_total_is_one_hundred() -> None:
    assert _TOTAL == 100


def test_seed_files_exist_with_expected_counts() -> None:
    assert set(SEED_FILES) == set(_PREFIX) == set(_EXPECTED_COUNTS)
    for category, filename in SEED_FILES.items():
        path = seed_dir() / filename
        assert path.exists(), f"シードファイルがありません: {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "1.0"
        assert payload["category"] == category
        assert payload["description"]
        assert len(payload["patterns"]) == _EXPECTED_COUNTS[category]
        structures = load_seed_file(filename)
        assert len(structures) == _EXPECTED_COUNTS[category]
        assert all(s.metadata.category.value == category for s in structures)


def test_hundred_unique_valid_structures(all_seeds: list[UniversalStructure]) -> None:
    assert len(all_seeds) == _TOTAL
    ids = [s.id for s in all_seeds]
    assert len(set(ids)) == _TOTAL
    for structure in all_seeds:
        assert structure.id.startswith(_PREFIX[structure.metadata.category.value])
        assert structure.metadata.source_kind.value == "seed"
        UniversalStructure.model_validate(structure.model_dump(mode="json"))


def test_ids_are_sequential_within_category(all_seeds: list[UniversalStructure]) -> None:
    by_category: dict[str, list[str]] = {}
    for structure in all_seeds:
        by_category.setdefault(structure.metadata.category.value, []).append(structure.id)
    for category, ids in by_category.items():
        numbers = sorted(int(identifier.split("_")[1]) for identifier in ids)
        assert numbers == list(range(1, _EXPECTED_COUNTS[category] + 1)), category


@pytest.mark.parametrize("index", range(_TOTAL))
def test_seed_quality_criteria(all_seeds: list[UniversalStructure], index: int) -> None:
    s = all_seeds[index]
    assert 4 <= len(s.tension_graph.nodes) <= 8, s.id
    assert 5 <= len(s.tension_graph.edges) <= 12, s.id
    assert 6 <= len(s.beats.beats) <= 10, s.id
    assert len(s.beats.pivotal_beats()) >= 2, s.id
    assert 3 <= len(s.mechanics.mechanisms) <= 5, s.id
    assert 6 <= len(s.blueprint.slots) <= 10, s.id
    assert 3 <= len(s.blueprint.invariants) <= 6, s.id
    assert s.wave.n_points >= 21, s.id
    assert s.transfer_guide.manga.strip(), s.id
    assert s.transfer_guide.marketing.strip(), s.id
    assert s.transfer_guide.article.strip(), s.id
    if s.metadata.category in _RHYTHM_CATEGORIES:
        assert s.rhythm_profile is not None, s.id
    for text in s.text_fields():
        assert not _URL_RE.search(text), f"{s.id}: URL を含みます: {text[:60]}"


def test_all_seeds_pass_consistency_audit(all_seeds: list[UniversalStructure]) -> None:
    from core.structure_engine.consistency import audit_structure

    problems = [
        f"{structure.id}: {issue.code}（{issue.location}）{issue.message}"
        for structure in all_seeds
        for issue in audit_structure(structure)
    ]
    assert problems == [], "整合性監査の指摘があります:\n" + "\n".join(problems)


def test_waves_are_distinct_within_category(all_seeds: list[UniversalStructure]) -> None:
    by_category: dict[str, list[UniversalStructure]] = {}
    for s in all_seeds:
        by_category.setdefault(s.metadata.category.value, []).append(s)
    for category, items in by_category.items():
        limit = 0.97 if category in _EXTENSION_CATEGORIES else 0.985
        for a, b in itertools.combinations(items, 2):
            corr = _corr(_tension(a), _tension(b))
            if corr is None:
                continue
            assert corr < limit, f"{category}: {a.id} と {b.id} の緊張波形が酷似しています（r={corr:.3f}）"


def test_extension_waves_are_distinct_from_original_sixty(all_seeds: list[UniversalStructure]) -> None:
    extension = [s for s in all_seeds if s.metadata.category.value in _EXTENSION_CATEGORIES]
    original = [s for s in all_seeds if s.metadata.category.value not in _EXTENSION_CATEGORIES]
    assert len(extension) == 40 and len(original) == 60
    for a in extension:
        for b in original:
            corr = _corr(_tension(a), _tension(b))
            if corr is None:
                continue
            assert corr < 0.985, f"{a.id} が既存シード {b.id} と酷似しています（r={corr:.3f}）"


def test_emotional_wave_presets_cover_distinct_dynamics(all_seeds: list[UniversalStructure]) -> None:
    presets = [s for s in all_seeds if s.metadata.category == StructureCategory.EMOTIONAL_WAVE]
    assert len(presets) == 13
    peak_positions = {round(s.wave.peak_t("tension"), 2) for s in presets}
    assert len(peak_positions) >= 6, "感情波形プリセットのピーク位置が偏っています"
    summaries = [s.wave.summary() for s in presets]
    assert max(item["suspense"] for item in summaries) > min(item["suspense"] for item in summaries) + 0.05
    assert max(item["catharsis"] for item in summaries) > min(item["catharsis"] for item in summaries) + 0.1


def test_seed_database_is_idempotent(tmp_db) -> None:
    from core.structure_engine.repository import StructureRepository

    repo = StructureRepository(tmp_db)
    first = seed_database(repo)
    assert first.total == _TOTAL
    assert first.inserted == _TOTAL
    assert sum(first.by_category.values()) == _TOTAL
    second = seed_database(repo)
    assert second.inserted == 0
    assert second.skipped == _TOTAL
    third = seed_database(repo, overwrite=True)
    assert third.updated == _TOTAL
    assert repo.count() == _TOTAL
    for category, expected in _EXPECTED_COUNTS.items():
        assert repo.count(category=category) == expected


def test_seed_database_upgrades_existing_sixty(tmp_db) -> None:
    """既存の60件DBに対し、拡張40件だけが追加投入される（アップグレード経路）。"""
    from core.structure_engine.repository import StructureRepository

    repo = StructureRepository(tmp_db)
    original = [s for s in load_all_seeds() if s.metadata.category.value not in _EXTENSION_CATEGORIES]
    for structure in original:
        repo.save(structure)
    assert repo.count() == 60
    report = seed_database(repo)
    assert report.inserted == 40
    assert report.skipped == 60
    assert repo.count() == _TOTAL


def test_seed_similarity_search(tmp_db) -> None:
    from core.structure_engine.repository import StructureRepository

    repo = StructureRepository(tmp_db)
    seed_database(repo)
    query = repo.get("na_01_call_and_return") or repo.all()[0]
    hits = repo.search_similar(query, method="hybrid", top_k=5, exclude_ids=(query.id,))
    assert len(hits) == 5
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert query.id not in {h.id for h in hits}
    wave_hits = repo.search_similar(query, method="dtw", top_k=5, category="emotional_wave")
    assert len(wave_hits) == 5
    assert all(h.category == "emotional_wave" for h in wave_hits)
