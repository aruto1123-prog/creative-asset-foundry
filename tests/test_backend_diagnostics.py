"""Phase 10: ネーム構造ドクター API（診断・履歴・オーバーレイ・処方適用・画像検出・合成画像）の結合テスト。"""

from __future__ import annotations

import base64
import copy
import time
import xml.etree.ElementTree as ET
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core.config import FoundrySettings
from web_studio.backend.app import create_app

TERMINAL = {"succeeded", "failed"}
AXES = {"gaze_flow", "emotional_dynamics", "information_density", "panel_rhythm", "page_turn"}


# ---------------------------------------------------------------------------
# フィクスチャ・ヘルパー
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def studio(tmp_path_factory: pytest.TempPathFactory):
    data_dir = tmp_path_factory.mktemp("doctor_data")
    settings = FoundrySettings.load(data_dir)
    settings.llm.provider = "mock"
    settings.comfy.use_mock = True
    settings.ensure_dirs()
    dist = tmp_path_factory.mktemp("doctor_dist")
    (dist / "index.html").write_text("<!doctype html><html><body><div id='root'></div></body></html>", encoding="utf-8")
    app = create_app(settings, frontend_dist=dist)
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def storyboard(studio: TestClient) -> dict[str, Any]:
    items = studio.get("/api/structures", params={"category": "narrative_archetype", "limit": 1}).json()["items"]
    assert items, "物語原型シードがありません"
    response = studio.post(
        "/api/manga/storyboards",
        json={"structure_id": items[0]["id"], "theme": {"THEME": "失われた約束"}, "config": {"target_pages": 6, "seed": 3}},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _svg_root(text: str) -> ET.Element:
    root = ET.fromstring(text)
    assert root.tag.endswith("svg")
    return root


def _wait_job(client: TestClient, job_id: str, timeout_s: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        record = client.get(f"/api/jobs/{job_id}").json()
        if record["status"] in TERMINAL:
            return record
        time.sleep(0.05)
    raise AssertionError(f"ジョブ {job_id} が {timeout_s} 秒以内に完了しませんでした")


def _assert_report_shape(report: dict[str, Any]) -> None:
    assert report["id"].startswith("diag")
    assert 0.0 <= report["overall_score"] <= 100.0
    assert report["grade"] in {"S", "A", "B", "C", "D"}
    assert {axis["axis"] for axis in report["axes"]} == AXES
    for axis in report["axes"]:
        assert 0.0 <= axis["score"] <= 100.0
    assert report["pages"]
    assert report["panels"]
    assert report["storyboard"] is not None
    curve = report["intensity_curve"]
    assert len(curve["t"]) == len(curve["inferred"])
    gains = [item["expected_gain"] for item in report["prescriptions"]]
    assert gains == sorted(gains, reverse=True)


def _degraded(storyboard: dict[str, Any]) -> dict[str, Any]:
    """単調なショット・アングル、強調なし、過多な台詞へ劣化させたネーム。"""
    degraded = copy.deepcopy(storyboard)
    degraded["id"] = f"{storyboard['id']}_degraded"
    long_text = "それでも私はここで立ち止まるわけにはいかないし、あの日の約束をどうしても果たさなければならないのだと何度も自分に言い聞かせていた。"
    first_balloon_done = False
    for page in degraded["pages"]:
        for panel in page["panels"]:
            panel["camera"]["shot"] = "MS"
            panel["camera"]["angle"] = "eye_level"
            panel["emphasis"] = "normal"
            if panel["balloons"] and not first_balloon_done:
                panel["balloons"][0]["text"] = long_text
                first_balloon_done = True
    degraded["grammar"] = None
    return degraded


def _annotation(western_order: bool = False) -> dict[str, Any]:
    """見開き2ページの既存原稿アノテーション。western_order=True は左→右の誤った読み順を明示する。"""
    page_1 = {
        "width": 1000,
        "height": 1414,
        "panels": [
            {"bbox": [520, 90, 930, 470], "shot": "ELS", "character_count": 1, "balloons": [{"text": "ここが始まりの場所だ。", "bbox": [820, 110, 910, 300]}]},
            {"bbox": [70, 90, 500, 470], "shot": "MS", "character_count": 2, "balloons": [{"text": "本当に戻ってきたのか？", "bbox": [380, 110, 470, 300]}]},
            {"bbox": [70, 500, 930, 900], "shot": "CU", "angle": "low", "character_count": 1, "ink_ratio": 0.4, "sfx_count": 1, "balloons": [{"text": "", "char_count": 9, "bbox": [760, 520, 900, 760], "kind": "shout"}]},
            {"bbox": [520, 930, 930, 1304], "shot": "MS", "character_count": 2, "balloons": [{"text": "なら、案内しよう。", "bbox": [800, 950, 900, 1120]}]},
            {"bbox": [70, 930, 500, 1304], "shot": "FS", "character_count": 2, "balloons": []},
        ],
    }
    page_2 = {
        "width": 1000,
        "height": 1414,
        "panels": [
            {"bbox": [70, 90, 930, 360], "shot": "ELS", "ink_ratio": 0.45, "balloons": []},
            {"bbox": [520, 390, 930, 760], "shot": "CU", "character_count": 1, "balloons": [{"text": "……誰かが待っていた。", "bbox": [820, 410, 910, 640], "kind": "monologue"}]},
            {"bbox": [70, 390, 500, 760], "shot": "MS", "angle": "dutch", "character_count": 1, "balloons": [{"text": "その扉を開けるな！", "bbox": [380, 410, 480, 620], "kind": "shout"}]},
            {"bbox": [70, 790, 930, 1304], "shot": "ECU", "angle": "low", "character_count": 1, "ink_ratio": 0.55, "sfx_count": 2, "intensity_hint": 0.9, "balloons": []},
        ],
    }
    if western_order:
        page_1["reading_order"] = [1, 0, 2, 4, 3]
        page_2["reading_order"] = [0, 2, 1, 3]
    return {"title": "既存原稿テスト" + ("（西洋式読み順）" if western_order else ""), "first_page_side": "right", "pages": [page_1, page_2]}


def _sample_data_url(client: TestClient, seed: int, variant: str) -> str:
    response = client.get("/api/manga/diagnostics/sample-image.png", params={"seed": seed, "variant": variant})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/png")
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    return "data:image/png;base64," + base64.b64encode(response.content).decode("ascii")


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------


def test_diagnose_saved_storyboard_overlay_and_history(studio: TestClient, storyboard: dict[str, Any]) -> None:
    response = studio.post("/api/manga/diagnostics", json={"storyboard_id": storyboard["id"], "max_prescriptions": 8})
    assert response.status_code == 200, response.text
    report = response.json()
    _assert_report_shape(report)
    assert report["source_type"] == "storyboard"
    assert report["source_id"] == storyboard["id"]
    assert report["target_structure_id"] == storyboard["structure_id"]
    assert len(report["panels"]) == sum(len(page["panels"]) for page in storyboard["pages"])
    assert len(report["prescriptions"]) <= 8

    svg = studio.get(f"/api/manga/diagnostics/{report['id']}/pages/0/svg", params={"heat": True, "gaze": True, "pins": True, "order": True})
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    _svg_root(svg.text)
    plain = studio.get(f"/api/manga/diagnostics/{report['id']}/pages/0/svg", params={"heat": False, "gaze": False, "pins": False, "order": False})
    assert plain.status_code == 200
    _svg_root(plain.text)
    assert len(plain.text) < len(svg.text)
    assert studio.get(f"/api/manga/diagnostics/{report['id']}/pages/999/svg").status_code == 404

    fetched = studio.get(f"/api/manga/diagnostics/{report['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["overall_score"] == report["overall_score"]

    listing = studio.get("/api/manga/diagnostics").json()
    assert report["id"] in {item["id"] for item in listing["items"]}
    assert listing["total"] >= 1
    summary = next(item for item in listing["items"] if item["id"] == report["id"])
    assert summary["page_count"] == len(storyboard["pages"])
    assert summary["findings"] == len(report["findings"])
    filtered = studio.get("/api/manga/diagnostics", params={"source_id": storyboard["id"]}).json()["items"]
    assert filtered and all(item["source_id"] == storyboard["id"] for item in filtered)


def test_degraded_storyboard_prescriptions_apply_and_save(studio: TestClient, storyboard: dict[str, Any]) -> None:
    degraded = _degraded(storyboard)
    response = studio.post("/api/manga/diagnostics", json={"storyboard": degraded, "simulate": True, "max_prescriptions": 10})
    assert response.status_code == 200, response.text
    report = response.json()
    _assert_report_shape(report)
    assert report["findings"], "劣化ネームから所見が検出されていません"
    applicable = [item for item in report["prescriptions"] if item["applicable"]]
    assert applicable, "自動適用可能な処方がありません"
    assert all(item["expected_gain"] > 0 for item in report["prescriptions"] if item["simulated"])

    applied = studio.post(
        f"/api/manga/diagnostics/{report['id']}/apply",
        json={"prescription_ids": [applicable[0]["id"]], "save_storyboard": True},
    )
    assert applied.status_code == 200, applied.text
    result = applied.json()
    assert result["score_after"] >= result["score_before"] - 1e-6
    assert result["score_before"] == pytest.approx(report["overall_score"])
    saved_id = result["saved_storyboard_id"]
    assert saved_id and saved_id != degraded["id"]
    assert studio.get(f"/api/manga/storyboards/{saved_id}").status_code == 200
    new_report = result["report"]
    assert new_report["id"] != report["id"]
    assert new_report["source_type"] == "storyboard"
    assert new_report["source_id"] == saved_id
    assert studio.get(f"/api/manga/diagnostics/{new_report['id']}").status_code == 200

    unsaved = studio.post(
        f"/api/manga/diagnostics/{report['id']}/apply",
        json={"prescription_ids": [applicable[0]["id"]], "save_storyboard": False},
    )
    assert unsaved.status_code == 200
    assert unsaved.json()["saved_storyboard_id"] is None

    unknown = studio.post(f"/api/manga/diagnostics/{report['id']}/apply", json={"prescription_ids": ["rx_missing"]})
    assert unknown.status_code == 400
    empty = studio.post(f"/api/manga/diagnostics/{report['id']}/apply", json={"prescription_ids": []})
    assert empty.status_code == 422


def test_annotation_diagnosis_detects_western_reading_order(studio: TestClient) -> None:
    natural = studio.post("/api/manga/diagnostics", json={"annotation": _annotation(False), "simulate": False})
    assert natural.status_code == 200, natural.text
    natural_report = natural.json()
    _assert_report_shape(natural_report)
    assert natural_report["source_type"] == "annotation"
    assert len(natural_report["pages"]) == 2
    assert sum(1 for panel in natural_report["panels"] if panel["page_index"] == 0) == 5

    western = studio.post("/api/manga/diagnostics", json={"annotation": _annotation(True), "simulate": False, "title": "西洋式読み順"})
    assert western.status_code == 200, western.text
    western_report = western.json()
    assert western_report["title"] == "西洋式読み順"

    def gaze(report: dict[str, Any]) -> float:
        return next(axis["score"] for axis in report["axes"] if axis["axis"] == "gaze_flow")

    assert gaze(western_report) < gaze(natural_report)
    assert sum(page["backtracks"] for page in western_report["pages"]) > sum(page["backtracks"] for page in natural_report["pages"])
    svg = studio.get(f"/api/manga/diagnostics/{western_report['id']}/pages/1/svg")
    assert svg.status_code == 200
    _svg_root(svg.text)


def test_sample_image_detect_and_image_diagnosis(studio: TestClient) -> None:
    first = _sample_data_url(studio, 3, "standard")
    second = _sample_data_url(studio, 4, "slanted")
    detected = studio.post("/api/manga/diagnostics/detect", json={"images": [first, second], "first_page_side": "right", "title": "合成原稿"})
    assert detected.status_code == 200, detected.text
    detection = detected.json()
    assert len(detection["pages"]) == 2
    assert all(len(page["panels"]) >= 2 for page in detection["pages"])
    assert detection["annotation"]["title"] == "合成原稿"
    assert len(detection["annotation"]["pages"]) == 2

    response = studio.post("/api/manga/diagnostics", json={"images": [first, second], "first_page_side": "right", "simulate": False})
    assert response.status_code == 200, response.text
    report = response.json()
    _assert_report_shape(report)
    assert report["source_type"] == "image"
    assert len(report["page_images"]) == 2
    assert all(image.startswith("data:image/png;base64,") for image in report["page_images"])
    with_image = studio.get(f"/api/manga/diagnostics/{report['id']}/pages/0/svg", params={"image": True})
    assert with_image.status_code == 200
    _svg_root(with_image.text)
    assert "data:image/png;base64," in with_image.text
    without_image = studio.get(f"/api/manga/diagnostics/{report['id']}/pages/0/svg", params={"image": False})
    assert "data:image/png;base64," not in without_image.text

    assert studio.get("/api/manga/diagnostics/sample-image.png", params={"variant": "bogus"}).status_code == 422
    broken = studio.post("/api/manga/diagnostics", json={"images": ["data:image/png;base64,bm90LWEtcG5n"]})
    assert broken.status_code == 400
    broken_detect = studio.post("/api/manga/diagnostics/detect", json={"images": ["%%%"]})
    assert broken_detect.status_code == 400


def test_validation_and_not_found(studio: TestClient, storyboard: dict[str, Any]) -> None:
    assert studio.post("/api/manga/diagnostics", json={}).status_code == 422
    both = studio.post("/api/manga/diagnostics", json={"storyboard_id": storyboard["id"], "annotation": _annotation()})
    assert both.status_code == 422
    assert studio.post("/api/manga/diagnostics", json={"storyboard_id": "sb_missing_000"}).status_code == 404
    missing_target = studio.post("/api/manga/diagnostics", json={"storyboard_id": storyboard["id"], "target_structure_id": "st_missing_000"})
    assert missing_target.status_code == 404
    invalid_annotation = studio.post("/api/manga/diagnostics", json={"annotation": {"title": "x", "pages": [{"panels": [{"balloons": []}]}]}})
    assert invalid_annotation.status_code == 422
    assert studio.get("/api/manga/diagnostics/diag_missing_000").status_code == 404
    assert studio.delete("/api/manga/diagnostics/diag_missing_000").status_code == 404
    assert studio.get("/api/manga/diagnostics/diag_missing_000/pages/0/svg").status_code == 404
    assert studio.post("/api/manga/diagnostics/diag_missing_000/apply", json={"prescription_ids": ["rx"]}).status_code == 404
    assert studio.get("/api/manga/storyboards/sb_missing_000/diagnostics").status_code == 404
    assert studio.post("/api/manga/diagnostics/detect", json={"images": []}).status_code == 422


def test_storyboard_shortcut_target_and_delete(studio: TestClient, storyboard: dict[str, Any]) -> None:
    target_id = studio.get("/api/structures", params={"category": "persuasion_marketing", "limit": 1}).json()["items"][0]["id"]
    explicit = studio.post("/api/manga/diagnostics", json={"storyboard_id": storyboard["id"], "target_structure_id": target_id, "simulate": False})
    assert explicit.status_code == 200, explicit.text
    assert explicit.json()["target_structure_id"] == target_id
    assert explicit.json()["intensity_curve"]["target"] is not None

    shortcut = studio.get(f"/api/manga/storyboards/{storyboard['id']}/diagnostics", params={"simulate": False})
    assert shortcut.status_code == 200, shortcut.text
    report = shortcut.json()
    _assert_report_shape(report)
    assert report["source_id"] == storyboard["id"]
    assert report["target_structure_id"] == storyboard["structure_id"]

    deleted = studio.delete(f"/api/manga/diagnostics/{report['id']}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "id": report["id"]}
    assert studio.get(f"/api/manga/diagnostics/{report['id']}").status_code == 404


def test_llm_commentary(studio: TestClient, storyboard: dict[str, Any]) -> None:
    response = studio.post("/api/manga/diagnostics", json={"storyboard_id": storyboard["id"], "use_llm": True, "simulate": False})
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["used_llm"] is True
    assert report["commentary"].strip()


def test_async_diagnosis_job_with_websocket(studio: TestClient, storyboard: dict[str, Any]) -> None:
    with studio.websocket_connect("/ws/progress") as ws:
        assert ws.receive_json()["type"] == "hello"
        submitted = studio.post("/api/manga/diagnostics", json={"storyboard_id": storyboard["id"], "async": True, "simulate": False})
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job_id"]
        events: list[dict[str, Any]] = []
        for _ in range(400):
            message = ws.receive_json()
            if message.get("type") == "job" and message.get("job_id") == job_id:
                events.append(message)
                if message["status"] in TERMINAL:
                    break
    assert events, "ジョブ進捗イベントを受信できませんでした"
    assert events[-1]["status"] == "succeeded"
    assert any(event["stage"] == "analyze" for event in events)
    record = _wait_job(studio, job_id)
    assert record["status"] == "succeeded"
    assert record["result"]["source_id"] == storyboard["id"]
    assert studio.get(f"/api/manga/diagnostics/{record['result']['id']}").status_code == 200
