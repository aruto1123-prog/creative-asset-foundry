"""Phase 6: 初期プリセット「普遍的構造データベース」60パターンの検証。"""

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
}
_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)


@pytest.fixture(scope="module")
def all_seeds() -> list[UniversalStructure]:
    return load_all_seeds()


def test_seed_files_exist_with_twenty_patterns() -> None:
    assert set(SEED_FILES) == set(_PREFIX)
    for category, filename in SEED_FILES.items():
        path = seed_dir() / filename
        assert path.exists(), f"シードファイルがありません: {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "1.0"
        assert payload["category"] == category
        assert payload["description"]
        assert len(payload["patterns"]) == 20
        structures = load_seed_file(filename)
        assert len(structures) == 20
        assert all(s.metadata.category.value == category for s in structures)


def test_sixty_unique_valid_structures(all_seeds: list[UniversalStructure]) -> None:
    assert len(all_seeds) == 60
    ids = [s.id for s in all_seeds]
    assert len(set(ids)) == 60
    for structure in all_seeds:
        assert structure.id.startswith(_PREFIX[structure.metadata.category.value])
        assert structure.metadata.source_kind.value == "seed"
        UniversalStructure.model_validate(structure.model_dump(mode="json"))


@pytest.mark.parametrize("index", range(60))
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
    if s.metadata.category == StructureCategory.MANGA_PANEL_RHYTHM:
        assert s.rhythm_profile is not None, s.id
    for text in s.text_fields():
        assert not _URL_RE.search(text), f"{s.id}: URL を含みます: {text[:60]}"


def test_waves_are_distinct_within_category(all_seeds: list[UniversalStructure]) -> None:
    by_category: dict[str, list[UniversalStructure]] = {}
    for s in all_seeds:
        by_category.setdefault(s.metadata.category.value, []).append(s)
    for category, items in by_category.items():
        for a, b in itertools.combinations(items, 2):
            ta = a.wave.resample_arrays(21)["tension"]
            tb = b.wave.resample_arrays(21)["tension"]
            if np.std(ta) < 1e-9 or np.std(tb) < 1e-9:
                continue
            corr = float(np.corrcoef(ta, tb)[0, 1])
            assert corr < 0.985, f"{category}: {a.id} と {b.id} の緊張波形が酷似しています（r={corr:.3f}）"


def test_seed_database_is_idempotent(tmp_db) -> None:
    from core.structure_engine.repository import StructureRepository

    repo = StructureRepository(tmp_db)
    first = seed_database(repo)
    assert first.total == 60
    assert first.inserted == 60
    assert sum(first.by_category.values()) == 60
    second = seed_database(repo)
    assert second.inserted == 0
    assert second.skipped == 60
    third = seed_database(repo, overwrite=True)
    assert third.updated == 60
    assert repo.count() == 60
    for category in _PREFIX:
        assert repo.count(category=category) == 20


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
