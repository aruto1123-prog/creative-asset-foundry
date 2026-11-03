"""Phase 7: Web Studio バックエンド（FastAPI）全エンドポイント・WebSocket・ジョブの結合テスト。"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core.config import FoundrySettings
from core.structure_engine.samples import sample_structure, sample_text_ja
from core.style_alchemy.comfy_export import validate_prompt
from web_studio.backend.app import create_app

TERMINAL = {"succeeded", "failed"}


# ---------------------------------------------------------------------------
# フィクスチャ・ヘルパー
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def studio(tmp_path_factory: pytest.TempPathFactory):
    data_dir = tmp_path_factory.mktemp("studio_data")
    settings = FoundrySettings.load(data_dir)
    settings.llm.provider = "mock"
    settings.comfy.use_mock = True
    settings.ensure_dirs()
    dist = tmp_path_factory.mktemp("studio_dist")
    (dist / "index.html").write_text("<!doctype html><html><body><div id='root'>CAF Studio</div></body></html>", encoding="utf-8")
    (dist / "assets").mkdir()
    (dist / "assets" / "app.js").write_text("console.log('caf');", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    app = create_app(settings, frontend_dist=dist)
    with TestClient(app) as client:
        yield client


def _services(client: TestClient):
    return client.app.state.services


def _svg_root(text: str) -> ET.Element:
    root = ET.fromstring(text)
    assert root.tag.endswith("svg")
    return root


def _wait_job(client: TestClient, job_id: str, timeout_s: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        record = client.get(f"/api/jobs/{job_id}").json()
        if record["status"] in TERMINAL:
            return record
        time.sleep(0.05)
    raise AssertionError(f"ジョブ {job_id} が {timeout_s} 秒以内に完了しませんでした")


def _drain_job_events(ws, job_id: str, max_messages: int = 400) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for _ in range(max_messages):
        message = ws.receive_json()
        if message.get("type") == "job" and message.get("job_id") == job_id:
            events.append(message)
            if message["status"] in TERMINAL:
                return events
    raise AssertionError(f"ジョブ {job_id} の終了イベントを受信できませんでした")


def _first_id(client: TestClient, category: str) -> str:
    items = client.get("/api/structures", params={"category": category, "limit": 1}).json()["items"]
    assert items, f"{category} の構造がありません"
    return items[0]["id"]


def _any_structure_id(client: TestClient) -> str:
    items = client.get("/api/structures", params={"limit": 1}).json()["items"]
    assert items
    return items[0]["id"]


# ---------------------------------------------------------------------------
# System / Health / Seeds
# ---------------------------------------------------------------------------


def test_health_reports_status_and_seed_counts(studio: TestClient) -> None:
    health = studio.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["llm_provider"] == "mock"
    assert health["comfy_mock"] is True
    assert health["db_path"].endswith("foundry.sqlite3")
    summary = studio.get("/api/seeds/summary").json()
    assert summary["total"] == health["seeds_loaded"]
    assert summary["total"] == sum(summary["by_category"].values())
    assert summary["total"] > 0
    assert set(summary["files"]) == {"narrative_archetype", "manga_panel_rhythm", "persuasion_marketing"}
    assert health["structures"] >= summary["total"]


def test_seed_load_is_idempotent(studio: TestClient) -> None:
    total = studio.get("/api/seeds/summary").json()["total"]
    report = studio.post("/api/seeds/load", json={}).json()
    assert report["inserted"] == 0
    assert report["skipped"] == report["total"] == total
    no_body = studio.post("/api/seeds/load")
    assert no_body.status_code == 200
    overwrite = studio.post("/api/seeds/load", json={"overwrite": True}).json()
    assert overwrite["updated"] == total
    assert overwrite["inserted"] == 0
    assert "missing_files" in overwrite and "invalid_files" in overwrite


def test_create_app_without_arguments_reads_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "env_data"
    monkeypatch.setenv("CAF_DATA_DIR", str(data_dir))
    app = create_app(frontend_dist=tmp_path / "no_dist")
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert health["db_path"] == str((data_dir / "foundry.sqlite3").resolve())
        assert health["seeds_loaded"] > 0
        assert client.get("/").status_code == 404
    assert app.state.frontend_mounted is False


# ---------------------------------------------------------------------------
# Structure Lab
# ---------------------------------------------------------------------------


def test_structure_listing_filters_and_lookup(studio: TestClient) -> None:
    page = studio.get("/api/structures", params={"limit": 5}).json()
    assert len(page["items"]) == 5
    assert page["total"] >= 5
    persuasion = studio.get("/api/structures", params={"category": "persuasion_marketing", "limit": 100}).json()
    assert persuasion["items"] and all(item["category"] == "persuasion_marketing" for item in persuasion["items"])
    seeds = studio.get("/api/structures", params={"source_kind": "seed", "limit": 1}).json()
    assert seeds["total"] == studio.get("/api/seeds/summary").json()["total"]
    first = persuasion["items"][0]
    detail = studio.get(f"/api/structures/{first['id']}").json()
    assert detail["id"] == first["id"]
    assert len(detail["wave"]["t"]) >= 21
    word = first["title"][:2]
    searched = studio.get("/api/structures", params={"q": word}).json()
    assert any(item["id"] == first["id"] for item in searched["items"])
    assert studio.get("/api/structures/does_not_exist").status_code == 404


def test_structure_crud_wave_edit_and_audit(studio: TestClient) -> None:
    structure = sample_structure("reversal")
    created = studio.post("/api/structures", json=structure.model_dump(mode="json"))
    assert created.status_code == 200
    assert studio.get(f"/api/structures/{structure.id}").json()["metadata"]["title"] == structure.metadata.title

    wave = structure.wave.model_dump(mode="json")
    wave.pop("information_asymmetry")
    wave["tension"] = [round(min(1.0, value * 0.9), 4) for value in wave["tension"]]
    updated = studio.put(f"/api/structures/{structure.id}/wave", json=wave)
    assert updated.status_code == 200
    assert updated.json()["wave"]["tension"] == wave["tension"]
    assert studio.get(f"/api/structures/{structure.id}").json()["wave"]["tension"] == wave["tension"]

    broken = dict(wave)
    broken["t"] = [i / 9 for i in range(10)]
    for key in ("tension", "release", "audience_knowledge", "character_knowledge", "cognitive_load"):
        broken[key] = [0.5] * 10
    assert studio.put(f"/api/structures/{structure.id}/wave", json=broken).status_code == 422
    out_of_range = dict(wave)
    out_of_range["tension"] = [1.5] * len(wave["t"])
    assert studio.put(f"/api/structures/{structure.id}/wave", json=out_of_range).status_code == 422

    audit = studio.get(f"/api/structures/{structure.id}/audit").json()
    assert audit["errors"] == 0
    assert isinstance(audit["issues"], list)

    assert studio.delete(f"/api/structures/{structure.id}").json()["deleted"] is True
    assert studio.get(f"/api/structures/{structure.id}").status_code == 404
    assert studio.delete(f"/api/structures/{structure.id}").status_code == 404
    invalid = structure.model_dump(mode="json")
    invalid["id"] = "X"
    assert studio.post("/api/structures", json=invalid).status_code == 422


def test_extract_sync_removes_flavor_and_saves(studio: TestClient) -> None:
    response = studio.post("/api/structures/extract", json={"text": sample_text_ja(), "title_hint": "抽出テスト"})
    assert response.status_code == 200
    result = response.json()
    assert result["used_llm"] is True
    assert result["harness_run_id"]
    structure = result["structure"]
    assert structure["metadata"]["source_kind"] == "extracted"
    serialized = json.dumps({k: v for k, v in structure.items() if k != "metadata"}, ensure_ascii=False)
    for proper_noun in ("レオン", "グレゴール", "アルカディア", "セリス"):
        assert proper_noun not in serialized
    assert studio.get(f"/api/structures/{structure['id']}").status_code == 200
    unsaved = studio.post("/api/structures/extract", json={"text": "主人公は宿敵に挑み、代償を払って勝つ。", "save": False}).json()
    assert studio.get(f"/api/structures/{unsaved['structure']['id']}").status_code == 404
    assert studio.post("/api/structures/extract", json={"text": ""}).status_code == 422


def test_merge_mutate_transpile_search_and_visuals(studio: TestClient) -> None:
    a_id = _first_id(studio, "narrative_archetype")
    b_id = _first_id(studio, "manga_panel_rhythm")
    merged = studio.post(
        "/api/structures/merge", json={"struct_a_id": a_id, "struct_b_id": b_id, "weight_topology": 0.6, "weight_wave": 0.8}
    )
    assert merged.status_code == 200
    payload = merged.json()
    merged_structure = payload["structure"]
    assert merged_structure["metadata"]["source_kind"] == "merged"
    assert merged_structure["metadata"]["parent_ids"] == [a_id, b_id]
    assert payload["report"]["weight_wave"] == pytest.approx(0.8)
    assert not [issue for issue in payload["audit"] if issue["severity"] == "error"]
    merged_id = merged_structure["id"]
    assert studio.get(f"/api/structures/{merged_id}").status_code == 200
    assert studio.post("/api/structures/merge", json={"struct_a_id": a_id, "struct_b_id": "missing_x"}).status_code == 404

    mutations = studio.get("/api/structures/mutations").json()
    assert len(mutations) == 7
    assert {m["type"] for m in mutations} >= {"invert_disclosure", "insert_false_climax", "invert_power_gradient"}
    mutated = studio.post(
        "/api/structures/mutate", json={"structure_id": merged_id, "mutation_type": "insert_false_climax", "seed": 3}
    ).json()
    assert mutated["structure"]["metadata"]["source_kind"] == "mutated"
    assert mutated["structure"]["metadata"]["parent_ids"] == [merged_id]
    bad_mutation = studio.post("/api/structures/mutate", json={"structure_id": merged_id, "mutation_type": "explode"})
    assert bad_mutation.status_code == 422

    for domain in ("manga_plot", "marketing_lp", "article"):
        design = studio.post(
            f"/api/structures/{merged_id}/transpile", json={"domain": domain, "theme_variables": {"THEME": "失われた居場所"}}
        )
        assert design.status_code == 200, design.text
        body = design.json()
        assert body["domain"] == domain
        assert body["sections"]
        assert 0.0 <= body["fidelity"]["score"] <= 1.0
    assert studio.post(f"/api/structures/{merged_id}/transpile", json={"domain": "poetry"}).status_code == 422

    hits = studio.post("/api/structures/search", json={"structure_id": a_id, "top_k": 4, "method": "hybrid"}).json()["hits"]
    assert len(hits) == 4
    assert a_id not in {hit["id"] for hit in hits}
    assert [hit["score"] for hit in hits] == sorted((hit["score"] for hit in hits), reverse=True)
    wave = studio.get(f"/api/structures/{a_id}").json()["wave"]
    wave.pop("information_asymmetry")
    by_wave = studio.post("/api/structures/search", json={"wave": wave, "method": "dtw", "top_k": 3, "category": "narrative_archetype"})
    assert by_wave.status_code == 200
    assert by_wave.json()["hits"][0]["id"] == a_id
    assert studio.post("/api/structures/search", json={}).status_code == 400

    mermaid = studio.get(f"/api/structures/{merged_id}/mermaid").json()
    assert mermaid["graph"].startswith("flowchart")
    assert "flowchart" in mermaid["beats"]
    svg = studio.get(f"/api/structures/{merged_id}/wave.svg", params={"compare_id": a_id})
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    _svg_root(svg.text)
    custom = studio.get(f"/api/structures/{a_id}/wave.svg", params={"channels": "tension,release", "width": 600, "title": False})
    _svg_root(custom.text)
    assert studio.get(f"/api/structures/{a_id}/wave.svg", params={"channels": "nonsense"}).status_code == 400


# ---------------------------------------------------------------------------
# Style Alchemy Arena
# ---------------------------------------------------------------------------


def _utility(params: dict[str, float]) -> float:
    return 2.0 * params["style.black_fill_ratio"] - 1.5 * params["style.line_taper"]


def test_style_candidates_arena_preferences_and_optimizer(studio: TestClient) -> None:
    assert studio.get("/api/style/arena/next").status_code == 409
    assert studio.get("/api/style/preference-vector").status_code == 409
    assert studio.get("/api/style/rankings").json()["ratings"] == []

    boot = studio.post(
        "/api/style/candidates/bootstrap", json={"n": 4, "architecture": "sdxl", "lora_names": ["ink_style"], "seed": 2}
    ).json()
    assert len(boot["candidates"]) == 4
    first = boot["candidates"][0]
    assert "style.black_fill_ratio" in first["params"]
    assert any(name.startswith("lora.ink_style.") for name in first["params"])
    preview = studio.get(f"/api/style/candidates/{first['id']}/preview.svg", params={"width": 256, "height": 256})
    assert preview.headers["content-type"].startswith("image/svg+xml")
    _svg_root(preview.text)
    assert studio.get(f"/api/style/candidates/{first['id']}").json()["id"] == first["id"]
    for candidate in boot["candidates"]:
        assert studio.delete(f"/api/style/candidates/{candidate['id']}").json()["deleted"] is True
    assert studio.delete(f"/api/style/candidates/{first['id']}").status_code == 404
    assert studio.get("/api/style/candidates/unknown/preview.svg").status_code == 404

    grid_values = [0.1, 0.4, 0.7, 1.0]
    created: dict[str, dict[str, float]] = {}
    for black in grid_values:
        for taper in grid_values[:3]:
            params = {"style.black_fill_ratio": black, "style.line_taper": taper}
            candidate = studio.post("/api/style/candidates", json={"label": f"黒{black}・線{taper}", "params": params}).json()
            created[candidate["id"]] = params
    listing = studio.get("/api/style/candidates").json()
    assert listing["total"] == len(created) == 12

    for _ in range(60):
        pair = studio.get("/api/style/arena/next").json()
        left, right = pair["left"], pair["right"]
        assert left["id"] != right["id"]
        diff = _utility(created[left["id"]]) - _utility(created[right["id"]])
        outcome = "left" if diff > 0 else "right" if diff < 0 else "tie"
        result = studio.post(
            "/api/style/comparisons",
            json={"left_id": left["id"], "right_id": right["id"], "outcome": outcome, "response_time_ms": 420.0},
        )
        assert result.status_code == 200
        assert result.json()["rankings"]
    assert studio.get("/api/style/comparisons").json()["total"] == 60
    assert studio.post("/api/style/comparisons", json={"left_id": left["id"], "right_id": left["id"], "outcome": "left"}).status_code == 400
    assert studio.post("/api/style/comparisons", json={"left_id": left["id"], "right_id": "ghost", "outcome": "left"}).status_code == 404
    assert studio.post("/api/style/comparisons", json={"left_id": left["id"], "right_id": right["id"], "outcome": "draw"}).status_code == 422

    rankings = studio.get("/api/style/rankings").json()
    assert rankings["n_candidates"] == 12
    assert [entry["rank"] for entry in rankings["ratings"]] == list(range(1, 13))
    assert rankings["bradley_terry"]["converged"] is True
    utilities = sorted((_utility(p) for p in created.values()), reverse=True)
    top_utility = _utility(created[rankings["ratings"][0]["candidate_id"]])
    assert top_utility >= utilities[2]

    vector = studio.get("/api/style/preference-vector", params={"n_bootstrap": 30}).json()
    features = {feature["name"]: feature for feature in vector["features"]}
    assert features["style.black_fill_ratio"]["direction"] == "prefer_higher"
    assert features["style.line_taper"]["direction"] == "prefer_lower"
    ridge = studio.get("/api/style/preference-vector", params={"method": "ridge_bt", "n_bootstrap": 10}).json()
    assert ridge["method"] == "ridge_bt"

    space = studio.get("/api/style/optimizer/space", params={"architecture": "sdxl", "lora_names": "ink_style"}).json()
    names = [dim["name"] for dim in space["space"]["dims"]]
    assert "merge.IN" in names and "lora.ink_style.strength" in names
    assert space["dims"] == len(names)

    suggestion = studio.post(
        "/api/style/optimizer/suggest", json={"n": 2, "lora_names": ["ink_style"], "register": True, "strategy": "gp"}
    ).json()
    assert len(suggestion["suggestions"]) == 2
    assert suggestion["observations"] == 12
    assert len(suggestion["registered"]) == 2
    assert suggestion["best"] is not None
    bounds = {dim["name"]: (dim["low"], dim["high"]) for dim in suggestion["space"]["dims"]}
    for item in suggestion["suggestions"]:
        for name, value in item["params"].items():
            assert bounds[name][0] - 1e-9 <= value <= bounds[name][1] + 1e-9

    export = studio.post(
        "/api/style/recipes/export",
        json={
            "params": suggestion["suggestions"][0]["params"],
            "architecture": "sdxl",
            "loras": [{"name": "ink_style.safetensors", "trigger_words": ["caf_ink"]}],
            "name": "APIテストレシピ",
            "width": 832,
            "height": 1216,
        },
    ).json()
    assert export["validation"] == []
    assert validate_prompt(export["comfy_workflow"]) == []
    assert export["recipe"]["merge"] is not None
    compile(export["merge_script"], "<merge>", "exec")
    assert export["merge_config"]
    assert studio.get("/api/style/recipes").json()["recipes"]


def test_dataset_tokens_and_grid_laboratory(studio: TestClient) -> None:
    lines = []
    for index in range(24):
        if index % 2 == 0:
            tags = "1girl, crying, from above, simple background, monochrome, masterpiece"
        else:
            tags = "1girl, smile, from side, detailed background, sunlight, best quality"
        lines.append(f"img{index:03d}: {tags}")
    analysis = studio.post("/api/style/dataset/analyze", json={"captions_text": "\n".join(lines), "trigger_token": "caf_ink"}).json()
    assert analysis["report"]["total_images"] == 24
    assert analysis["report_id"]
    assert analysis["cleansed"][0]["caption"].startswith("caf_ink")
    assert any("monochrome" in item["removed_style_tags"] for item in analysis["cleansed"])
    pairs = {frozenset((e["tag_a"], e["tag_b"])) for e in analysis["report"]["entanglements"]}
    assert frozenset(("crying", "from above")) in pairs
    with_llm = studio.post("/api/style/dataset/analyze", json={"captions_text": "\n".join(lines[:8]), "use_llm": True, "save": False})
    assert with_llm.status_code == 200
    assert with_llm.json()["report_id"] is None
    assert studio.post("/api/style/dataset/analyze", json={"captions_text": "   "}).status_code == 400
    assert studio.get("/api/style/dataset/reports").json()["reports"]

    tokens = studio.get("/api/style/tokens").json()
    assert len(tokens["categories"]) >= 6
    added = studio.post(
        "/api/style/tokens", json={"token": "caf test ink", "category": "line_art", "weight": 1.2, "description": "テスト用"}
    )
    assert added.status_code == 200
    line_art = studio.get("/api/style/tokens", params={"category": "line_art"}).json()["tokens"]
    assert any(token["token"] == "caf test ink" for token in line_art)
    assert studio.delete("/api/style/tokens", params={"token": "caf test ink", "category": "line_art"}).json()["deleted"] is True
    assert studio.delete("/api/style/tokens", params={"token": "caf test ink", "category": "line_art"}).status_code == 404

    grid = studio.post(
        "/api/style/grid",
        json={
            "name": "黒ベタ×線画",
            "base_prompt": "1girl, manga panel",
            "x_axis": {"name": "black", "kind": "param", "target": "style.black_fill_ratio", "values": [0.2, 0.5, 0.8]},
            "y_axis": {"name": "line", "kind": "token", "target": "line_art", "values": ["clean lineart", "rough sketch lines"]},
        },
    ).json()
    experiment_id = grid["experiment"]["id"]
    cells = grid["cells"]
    assert len(cells) == 6
    for cell in cells:
        response = studio.post(
            f"/api/style/grid/{experiment_id}/results",
            json={"cell_id": cell["id"], "score": 10.0 * cell["x_index"] + cell["y_index"]},
        )
        assert response.status_code == 200
    detail = studio.get(f"/api/style/grid/{experiment_id}").json()
    assert detail["experiment"]["status"] == "completed"
    best = max(cells, key=lambda c: 10.0 * c["x_index"] + c["y_index"])
    assert detail["analysis"]["best_cell_id"] == best["id"]
    assert detail["analysis"]["n_scored"] == 6
    assert any(e["id"] == experiment_id for e in studio.get("/api/style/grid").json()["experiments"])
    assert studio.post(f"/api/style/grid/{experiment_id}/results", json={"cell_id": "other_x0", "score": 1}).status_code == 400
    assert studio.post(f"/api/style/grid/{experiment_id}/results", json={"cell_id": f"{experiment_id}_x99_y0_z0", "score": 1}).status_code == 404
    assert studio.get("/api/style/grid/grid_missing").status_code == 404
    batch = studio.post(f"/api/style/grid/{experiment_id}/comfy", json={"architecture": "sdxl"}).json()
    assert batch["count"] == 6
    assert all(validate_prompt(item["workflow"]) == [] for item in batch["batch"])
    merged_batch = studio.post(
        f"/api/style/grid/{experiment_id}/comfy", json={"architecture": "sdxl", "model_b": {"name": "b.safetensors"}}
    ).json()
    assert merged_batch["recipe"]["merge"] is not None


# ---------------------------------------------------------------------------
# Manga Studio
# ---------------------------------------------------------------------------


def test_manga_storyboard_lifecycle_and_comfy(studio: TestClient) -> None:
    structure_id = _first_id(studio, "narrative_archetype")
    response = studio.post(
        "/api/manga/storyboards",
        json={
            "structure_id": structure_id,
            "theme": {"THEME": "失われた居場所"},
            "config": {"target_pages": 8, "seed": 3},
            "title": "APIテストネーム",
        },
    )
    assert response.status_code == 200, response.text
    storyboard = response.json()
    storyboard_id = storyboard["id"]
    assert storyboard["title"] == "APIテストネーム"
    assert storyboard["grammar"]["passed"] is True
    assert storyboard["grammar"]["score"] >= 80.0
    assert len(storyboard["pages"]) >= 4

    listing = studio.get("/api/manga/storyboards").json()
    assert any(item["id"] == storyboard_id and item["passed"] for item in listing["items"])
    assert studio.get(f"/api/manga/storyboards/{storyboard_id}").json()["id"] == storyboard_id
    assert studio.get(f"/api/manga/storyboards/{storyboard_id}/config").json()["target_pages"] == 8

    report = studio.post(f"/api/manga/storyboards/{storyboard_id}/validate").json()
    assert 0.0 <= report["score"] <= 100.0
    assert "violations" in report
    fixed = studio.post(f"/api/manga/storyboards/{storyboard_id}/autofix").json()
    assert fixed["grammar"]["passed"] is True

    page_svg = studio.get(f"/api/manga/storyboards/{storyboard_id}/pages/0/svg", params={"gaze": True})
    assert page_svg.headers["content-type"].startswith("image/svg+xml")
    _svg_root(page_svg.text)
    plain_svg = studio.get(f"/api/manga/storyboards/{storyboard_id}/pages/0/svg", params={"gaze": False, "labels": False})
    _svg_root(plain_svg.text)
    assert studio.get(f"/api/manga/storyboards/{storyboard_id}/pages/99/svg").status_code == 404
    spread = studio.get(f"/api/manga/storyboards/{storyboard_id}/spreads/0/1/svg")
    _svg_root(spread.text)
    layers = studio.get(f"/api/manga/storyboards/{storyboard_id}/layers").json()
    assert len(layers["pages"]) == len(fixed["pages"])

    compiled = studio.post(f"/api/manga/storyboards/{storyboard_id}/comfy", json={}).json()
    panel_total = sum(len(page["panels"]) for page in fixed["pages"])
    assert compiled["count"] == panel_total
    assert all(validate_prompt(workflow["prompt"]) == [] for workflow in compiled["workflows"])
    assert all(workflow["width"] % 8 == 0 and workflow["height"] % 8 == 0 for workflow in compiled["workflows"])
    first_panel = compiled["workflows"][0]["panel_id"]
    single = studio.post(f"/api/manga/storyboards/{storyboard_id}/comfy", json={"panel_ids": [first_panel]}).json()
    assert single["count"] == 1
    tuned = studio.post(
        f"/api/manga/storyboards/{storyboard_id}/comfy",
        json={"panel_ids": [first_panel], "recipe_params": {"merge.IN": 0.3, "merge.OUT": 0.7, "style.black_fill_ratio": 0.8}},
    ).json()
    assert tuned["recipe"]["merge"] is not None
    explicit = studio.post(
        f"/api/manga/storyboards/{storyboard_id}/comfy",
        json={"panel_ids": [first_panel], "recipe": tuned["recipe"]},
    ).json()
    assert explicit["recipe"]["name"] == tuned["recipe"]["name"]
    assert studio.post(f"/api/manga/storyboards/{storyboard_id}/comfy", json={"panel_ids": ["ghost"]}).status_code == 404

    queued = studio.post(f"/api/manga/storyboards/{storyboard_id}/comfy/queue", json={"panel_ids": [first_panel]}).json()
    assert queued["panels"] == 1
    record = _wait_job(studio, queued["job_id"])
    assert record["status"] == "succeeded", record
    assert record["result"]["succeeded"] == 1
    assert record["result"]["results"][0]["status"] == "success"
    assert record["result"]["results"][0]["outputs"]

    assert studio.post("/api/manga/storyboards", json={"structure_id": "missing_structure"}).status_code == 404
    invalid_config = studio.post("/api/manga/storyboards", json={"structure_id": structure_id, "config": {"target_pages": 2}})
    assert invalid_config.status_code == 400
    assert studio.delete(f"/api/manga/storyboards/{storyboard_id}").json()["deleted"] is True
    assert studio.get(f"/api/manga/storyboards/{storyboard_id}").status_code == 404
    assert studio.delete(f"/api/manga/storyboards/{storyboard_id}").status_code == 404


# ---------------------------------------------------------------------------
# Asset Publisher
# ---------------------------------------------------------------------------


def test_publisher_matrix_articles_exports_and_series(studio: TestClient) -> None:
    structure_id = _first_id(studio, "persuasion_marketing")
    matrix = studio.post("/api/publisher/matrix", json={"structure_id": structure_id}).json()
    assert {plan["axis"] for plan in matrix["plans"]} == {"creator_theory", "ai_engineering", "business_marketing"}
    matrix_id = matrix["id"]
    assert studio.get(f"/api/publisher/matrices/{matrix_id}").json()["id"] == matrix_id
    assert any(item["id"] == matrix_id for item in studio.get("/api/publisher/matrices").json()["matrices"])
    assert studio.post("/api/publisher/matrix", json={}).status_code == 400
    assert studio.post("/api/publisher/matrix", json={"structure_id": structure_id, "experiment": {"x": 1}}).status_code == 400
    custom_log = {
        "title": "線の入り抜き強度と選好の関係",
        "summary": "一対比較の結果、線の入り抜きを弱めた候補が一貫して好まれた。",
        "findings": [{"claim": "入り抜きを弱めると勝率が上がる", "evidence": "60回の比較で勝率72%", "metrics": {"win_rate": 0.72}}],
        "tags": ["選好学習"],
    }
    log_matrix = studio.post("/api/publisher/matrix", json={"research_log": custom_log})
    assert log_matrix.status_code == 200
    assert studio.get("/api/publisher/matrices/missing").status_code == 404

    article = studio.post("/api/publisher/articles", json={"matrix_id": matrix_id, "axis": "ai_engineering"})
    assert article.status_code == 200, article.text
    article_a = article.json()
    assert article_a["axis"] == "ai_engineering"
    assert article_a["final_score"] >= 80.0
    assert article_a["sanitization"]["url_free"] is True
    assert article_a["sanitization"]["first_person_unified"] is True
    plan_c = next(plan for plan in matrix["plans"] if plan["axis"] == "creator_theory")
    article_b = studio.post("/api/publisher/articles", json={"plan": plan_c, "structure_id": structure_id}).json()
    assert article_b["axis"] == "creator_theory"
    assert studio.post("/api/publisher/articles", json={}).status_code == 400
    assert studio.post("/api/publisher/articles", json={"matrix_id": "missing"}).status_code == 404

    listing = studio.get("/api/publisher/articles").json()
    assert listing["total"] >= 2
    filtered = studio.get("/api/publisher/articles", params={"axis": "ai_engineering"}).json()
    assert filtered["items"] and all(item["axis"] == "ai_engineering" for item in filtered["items"])
    assert studio.get(f"/api/publisher/articles/{article_a['id']}").json()["title"] == article_a["title"]

    note = studio.get(f"/api/publisher/articles/{article_a['id']}/export", params={"format": "note"})
    assert note.headers["content-type"].startswith("text/markdown")
    assert "attachment" in note.headers["content-disposition"]
    assert "```mermaid" not in note.text
    obsidian = studio.get(f"/api/publisher/articles/{article_a['id']}/export", params={"format": "obsidian"})
    assert obsidian.text.startswith("---")
    for text in (note.text, obsidian.text):
        assert "http://" not in text and "https://" not in text
    assert studio.get(f"/api/publisher/articles/{article_a['id']}/export", params={"format": "pdf"}).status_code == 422

    dirty = "僕はこの手法を https://example.com/aff?id=1 と [詳細](http://bit.ly/xyz) で紹介した。俺たちは検証した。"
    cleaned = studio.post("/api/publisher/sanitize", json={"text": dirty}).json()
    assert "私" in cleaned["text"]
    assert "http" not in cleaned["text"] and "bit.ly" not in cleaned["text"]
    assert cleaned["report"]["url_free"] is True and cleaned["report"]["first_person_unified"] is True
    raw_report = studio.post("/api/publisher/verify", json={"text": dirty}).json()
    assert raw_report["url_free"] is False and raw_report["first_person_unified"] is False
    assert studio.post("/api/publisher/verify", json={"text": cleaned["text"]}).json()["url_free"] is True

    gaps = studio.get("/api/publisher/gaps", params={"top_k": 3}).json()
    assert len(gaps["gaps"]) <= 3
    assert gaps["stats"]
    assert gaps["mermaid"]

    series = studio.post(
        "/api/publisher/series", json={"article_ids": [article_a["id"], article_b["id"]], "title": "構造と絵柄の教本"}
    ).json()
    ordered = [aid for chapter in series["volume"]["chapters"] for aid in chapter["article_ids"]]
    assert sorted(ordered) == sorted([article_a["id"], article_b["id"]])
    assert series["markdown"]
    note_series = studio.post(
        "/api/publisher/series", json={"article_ids": [article_a["id"]], "title": "note版", "format": "note"}
    ).json()
    assert note_series["format"] == "note"
    assert studio.post("/api/publisher/series", json={"article_ids": ["missing"], "title": "x"}).status_code == 404

    assert studio.delete(f"/api/publisher/articles/{article_b['id']}").json()["deleted"] is True
    assert studio.get(f"/api/publisher/articles/{article_b['id']}").status_code == 404
    assert studio.delete(f"/api/publisher/articles/{article_b['id']}").status_code == 404


# ---------------------------------------------------------------------------
# Jobs / WebSocket
# ---------------------------------------------------------------------------


def test_websocket_ping_pong_and_async_jobs(studio: TestClient) -> None:
    with studio.websocket_connect("/ws/progress") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
        ws.send_text("hello?")
        assert ws.receive_json()["type"] == "error"

        extract = studio.post("/api/structures/extract", json={"text": sample_text_ja(), "async": True}).json()
        assert set(extract) >= {"job_id", "status", "kind"}
        record = _wait_job(studio, extract["job_id"])
        assert record["status"] == "succeeded"
        events = _drain_job_events(ws, extract["job_id"])
        assert events[-1]["status"] == "succeeded"
        assert any(event["status"] == "running" for event in events)
        assert [event["progress"] for event in events] == sorted(event["progress"] for event in events)
        structure_id = record["result"]["structure"]["id"]

        storyboard_job = studio.post(
            "/api/manga/storyboards", json={"structure_id": structure_id, "config": {"target_pages": 8}, "async": True}
        ).json()
        storyboard_record = _wait_job(studio, storyboard_job["job_id"])
        assert storyboard_record["status"] == "succeeded"
        assert storyboard_record["result"]["grammar"]["passed"] is True
        assert _drain_job_events(ws, storyboard_job["job_id"])[-1]["status"] == "succeeded"

        matrix_job = studio.post("/api/publisher/matrix", json={"structure_id": structure_id, "async": True}).json()
        matrix_record = _wait_job(studio, matrix_job["job_id"])
        assert matrix_record["status"] == "succeeded"
        _drain_job_events(ws, matrix_job["job_id"])
        article_job = studio.post(
            "/api/publisher/articles",
            json={"matrix_id": matrix_record["result"]["id"], "axis": "business_marketing", "async": True},
        ).json()
        article_record = _wait_job(studio, article_job["job_id"])
        assert article_record["status"] == "succeeded"
        article_events = _drain_job_events(ws, article_job["job_id"])
        stages = {event["stage"] for event in article_events}
        assert len(stages) >= 4
        assert article_record["result"]["final_score"] >= 80.0

    jobs = studio.get("/api/jobs", params={"limit": 20}).json()["jobs"]
    ids = {job["id"] for job in jobs}
    assert {extract["job_id"], storyboard_job["job_id"], article_job["job_id"]} <= ids
    assert all(job["result"] is None for job in jobs)
    assert studio.get("/api/jobs/job_missing").status_code == 404


def test_failed_job_is_reported(studio: TestClient) -> None:
    services = _services(studio)
    structure_id = _any_structure_id(studio)
    storyboard = studio.post(
        "/api/manga/storyboards", json={"structure_id": structure_id, "config": {"target_pages": 8}}
    ).json()
    original_mock = services.settings.comfy.use_mock
    original_url = services.settings.comfy.base_url
    try:
        services.settings.comfy.use_mock = False
        services.settings.comfy.base_url = "http://127.0.0.1:9"
        queued = studio.post(f"/api/manga/storyboards/{storyboard['id']}/comfy/queue", json={"max_panels": 1}).json()
        record = _wait_job(studio, queued["job_id"])
    finally:
        services.settings.comfy.use_mock = original_mock
        services.settings.comfy.base_url = original_url
    assert record["status"] == "failed"
    assert record["error"]


# ---------------------------------------------------------------------------
# System config / DB / telemetry / SPA
# ---------------------------------------------------------------------------


def test_connection_tests_db_stats_backup_and_telemetry(studio: TestClient) -> None:
    llm = studio.post("/api/system/llm/test").json()
    assert llm["ok"] is True
    assert llm["provider"] == "mock"
    assert llm["latency_ms"] >= 0.0
    comfy = studio.post("/api/system/comfy/test").json()
    assert comfy["ok"] is True
    assert comfy["mock"] is True
    assert "system_stats" in comfy

    stats = studio.get("/api/system/db/stats").json()
    assert stats["size_bytes"] > 0
    assert stats["tables"]["structures"] >= 1
    assert "backend_jobs" in stats["tables"]
    backup = studio.post("/api/system/db/backup").json()
    assert Path(backup["path"]).exists()
    assert backup["size_bytes"] > 0
    second = studio.post("/api/system/db/backup").json()
    assert second["name"] != backup["name"]
    names = {item["name"] for item in studio.get("/api/system/db/backups").json()["backups"]}
    assert {backup["name"], second["name"]} <= names

    telemetry = studio.get("/api/system/telemetry", params={"limit": 10}).json()
    assert telemetry["calls"]
    assert telemetry["stats"]["totals"]["calls"] > 0
    assert telemetry["runs"]
    run_id = telemetry["runs"][0]["run_id"]
    run = studio.get(f"/api/system/telemetry/runs/{run_id}")
    assert run.status_code == 200
    assert studio.get("/api/system/telemetry/runs/run_missing").status_code == 404


def test_config_update_persists_and_rebuilds_llm(studio: TestClient) -> None:
    services = _services(studio)
    before_client = services.llm_client
    config = studio.get("/api/system/config").json()
    assert "***" in config["llm"]["api_key"]
    response = studio.put(
        "/api/system/config",
        json={"llm": {"model": "DeepSeek-V4.1-Flash-Studio", "harness": {"critique_accept_score": 80}}, "comfy": {"timeout_s": 123.0}},
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["llm"]["model"] == "DeepSeek-V4.1-Flash-Studio"
    assert updated["llm"]["harness"]["critique_accept_score"] == 80
    assert updated["llm"]["harness"]["enable_decompose"] is True
    assert updated["comfy"]["timeout_s"] == 123.0
    assert services.llm_client is not before_client
    persisted = json.loads(services.settings.config_path.read_text(encoding="utf-8"))
    assert persisted["llm"]["model"] == "DeepSeek-V4.1-Flash-Studio"
    assert persisted["comfy"]["timeout_s"] == 123.0

    original_key = services.settings.llm.api_key
    studio.put("/api/system/config", json={"llm": {"api_key": updated["llm"]["api_key"]}})
    assert services.settings.llm.api_key == original_key
    studio.put("/api/system/config", json={"llm": {"api_key": "sk-local-secret"}})
    assert services.settings.llm.api_key == "sk-local-secret"
    assert "secret" not in studio.get("/api/system/config").json()["llm"]["api_key"]

    assert studio.put("/api/system/config", json={"llm": {"temperature": 9.0}}).status_code == 400
    assert studio.put("/api/system/config", json={"database": {}}).status_code == 400
    assert studio.put("/api/system/config", json={"llm": "invalid"}).status_code == 400
    assert studio.put("/api/system/config", json=[1, 2]).status_code == 422
    assert studio.post("/api/system/llm/test").json()["ok"] is True


def test_spa_fallback_and_static_assets(studio: TestClient) -> None:
    index = studio.get("/")
    assert index.status_code == 200
    assert "CAF Studio" in index.text
    deep = studio.get("/manga/storyboards/123")
    assert deep.status_code == 200
    assert "CAF Studio" in deep.text
    asset = studio.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text
    favicon = studio.get("/favicon.svg")
    assert favicon.status_code == 200
    assert studio.get("/api/not-a-route").status_code == 404
    assert studio.get("/api/not-a-route").json()["detail"]
    assert studio.get("/ws/nothing").status_code == 404
