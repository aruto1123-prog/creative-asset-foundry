"""高度な漫画演出（manga_direction）13パターンの著作スクリプト。

連続見開き・縦長ゴマ・枠線崩壊・定点反復・ぶち抜き・モノローグ深潜・沈黙と一言・斜めゴマ・
小ゴマのタイムラプス・視線撹乱・対比見開き・回想の枠線表現・最終コマの大余白という、
コマ枠そのものを演出装置として用いる13の骨格を、手書き設計のコンパクトな定義（タプル列）から
``UniversalStructure`` として組み立てる。

波形は5チャネルすべてを 6〜12 個のキーフレーム（正規化時間 0.05 刻みの格子点上）で設計し、
``EmotionalInformationWave.from_keyframes``（PCHIP単調補間）で 21 点（20分割）に展開する。
転換点ビートは必ずキーフレーム上の緊張度の極値に置く。

ページ左右の慣行: page 0 を右ページとし、偶数インデックス=右ページ・奇数=左ページ
（見開きは「右ページ開始・左ページ終端」）。めくりは奇数（左）→偶数（右）の境界で起きるため、
めくり直後のリビール（page_turn_reveals）と見開きの右ページ側（spread_pages）は偶数インデックスに置く。

再生成: ``python -m core.structure_engine.seeds.authoring.build_manga_direction``
（``--check`` を付けると JSON を書かずに検証のみ行う）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from core.structure_engine.consistency import audit_structure
from core.structure_engine.schemas import (
    WAVE_CHANNELS,
    BeatType,
    EdgeKind,
    EmotionalInformationWave,
    UniversalStructure,
)

SEED_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = SEED_DIR / "advanced_manga_direction_patterns.json"
EXISTING_SEED_FILES = (
    "narrative_archetypes.json",
    "manga_panel_rhythm_patterns.json",
    "persuasion_and_marketing_structures.json",
)
SEED_CREATED_AT = "2026-11-03T00:00:00+00:00"
CATEGORY = "manga_direction"
PATTERN_COUNT = 13
FILE_DESCRIPTION = (
    "高度な漫画演出骨格13選。コマ枠・見開き・余白・読み順そのものを演出装置として扱い、"
    "連続見開きの二連撃、縦長ゴマの落下感、心理と同期した枠線崩壊、定点構図の反復と差分、"
    "ぶち抜き登場の威圧、モノローグの深潜、沈黙から一言の決め、斜めゴマの加速、"
    "小ゴマ連打のタイムラプス、視線誘導の意図的な撹乱、対比見開き、回想の枠線表現、"
    "最終コマの大余白を、ページ単位のコマ数・強調ページ・めくり位置（rhythm_profile）と"
    "21点の感情・情報波形で記述する。"
)
ID_PATTERN = re.compile(r"^md_(\d{2})_[a-z0-9_]+$")
CATEGORY_CORRELATION_LIMIT = 0.97
EXISTING_CORRELATION_LIMIT = 0.985
SIGN_CHECK_THRESHOLD = 0.25
SIGN_WINDOW = 0.1
FINAL_BEAT_TYPES = frozenset({BeatType.AFTERGLOW, BeatType.PAYOFF, BeatType.CLIMAX})

PATTERNS: list[dict[str, Any]] = []


def _nodes(rows: list[tuple]) -> list[dict[str, Any]]:
    """(id, kind, role, label, description, weight, slot) → ノード辞書。"""
    return [
        {"id": r[0], "kind": r[1], "role": r[2], "label": r[3], "description": r[4], "weight": r[5], "slot": r[6]}
        for r in rows
    ]


def _edges(rows: list[tuple]) -> list[dict[str, Any]]:
    """(source, target, kind, intensity, gradient, label, activation_t) → エッジ辞書。"""
    return [
        {
            "source": r[0],
            "target": r[1],
            "kind": r[2],
            "intensity": r[3],
            "gradient": r[4],
            "label": r[5],
            "activation_t": r[6],
        }
        for r in rows
    ]


def _beats(rows: list[tuple]) -> list[dict[str, Any]]:
    """(id, type, t, span, label, function, causes, tension_delta, information_shift, nodes, turning) → ビート辞書。"""
    return [
        {
            "id": r[0],
            "beat_type": r[1],
            "t": r[2],
            "span": r[3],
            "label": r[4],
            "function": r[5],
            "causes": list(r[6]),
            "tension_delta": r[7],
            "information_shift": r[8],
            "involved_nodes": list(r[9]),
            "is_turning_point": r[10],
        }
        for r in rows
    ]


def _mechanisms(rows: list[tuple]) -> list[dict[str, Any]]:
    """(id, category, name, description, trigger, effect, theory, strength, beats, nodes) → メカニズム辞書。"""
    return [
        {
            "id": r[0],
            "category": r[1],
            "name": r[2],
            "description": r[3],
            "trigger": r[4],
            "effect": r[5],
            "theory": r[6],
            "strength": r[7],
            "linked_beats": list(r[8]),
            "linked_nodes": list(r[9]),
        }
        for r in rows
    ]


def _slots(rows: list[tuple]) -> list[dict[str, Any]]:
    """(name, description, category, constraints, (manga, marketing, article)) → スロット辞書。"""
    return [
        {
            "name": r[0],
            "description": r[1],
            "category": r[2],
            "constraints": list(r[3]),
            "examples": {"manga": r[4][0], "marketing": r[4][1], "article": r[4][2]},
            "required": True,
        }
        for r in rows
    ]


def build_structure(spec: dict[str, Any]) -> UniversalStructure:
    """1パターン定義から検証済みの UniversalStructure を組み立てる。"""
    wave = EmotionalInformationWave.from_keyframes(spec["waves"], n=21)
    payload: dict[str, Any] = {
        "id": spec["id"],
        "metadata": {
            "title": spec["title"],
            "summary": spec["summary"],
            "category": CATEGORY,
            "source_kind": "seed",
            "tags": list(spec["tags"]),
            "origin_domain": "manga",
            "lineage": list(spec["lineage"]),
            "parent_ids": [],
            "language": "ja",
            "version": 1,
            "created_at": SEED_CREATED_AT,
            "notes": spec.get("notes", ""),
        },
        "tension_graph": {"nodes": _nodes(spec["nodes"]), "edges": _edges(spec["edges"])},
        "wave": wave.model_dump(mode="json"),
        "beats": {"beats": _beats(spec["beats"])},
        "mechanics": {
            "mechanisms": _mechanisms(spec["mechanisms"]),
            "empathy_anchor": spec["anchor"],
            "immersion_drivers": list(spec["drivers"]),
            "emotion_sequence": list(spec["emotions"]),
        },
        "blueprint": {
            "template": spec["template"],
            "slots": _slots(spec["slots"]),
            "invariants": list(spec["invariants"]),
        },
        "transfer_guide": dict(spec["guide"]),
        "rhythm_profile": dict(spec["rhythm"]),
    }
    return UniversalStructure.model_validate(payload)


def build_all() -> list[UniversalStructure]:
    return [build_structure(spec) for spec in PATTERNS]


def _is_local_extremum(values: np.ndarray, index: int) -> bool:
    neighbours = [values[j] for j in (index - 1, index + 1) if 0 <= j < values.size]
    value = values[index]
    return all(value >= v - 1e-9 for v in neighbours) or all(value <= v + 1e-9 for v in neighbours)


def _is_local_maximum(values: np.ndarray, index: int) -> bool:
    neighbours = [values[j] for j in (index - 1, index + 1) if 0 <= j < values.size]
    return all(values[index] >= v - 1e-9 for v in neighbours)


def load_existing_tensions(seed_dir: Path = SEED_DIR) -> dict[str, np.ndarray]:
    """既存シード（60件）の緊張度波形を ID→配列 で読み込む（波形相関の検査用）。"""
    tensions: dict[str, np.ndarray] = {}
    for filename in EXISTING_SEED_FILES:
        path = seed_dir / filename
        if not path.exists():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        for pattern in document.get("patterns", []):
            tensions[pattern["id"]] = np.asarray(pattern["wave"]["tension"], dtype=float)
    return tensions


def _tension_change(structure: UniversalStructure, t: float) -> float:
    """ビートへ流れ込む緊張度変化（t<0.1 では前方差分）。"""
    wave = structure.wave
    if t < SIGN_WINDOW:
        return wave.sample("tension", min(1.0, t + SIGN_WINDOW)) - wave.sample("tension", t)
    return wave.sample("tension", t) - wave.sample("tension", t - SIGN_WINDOW)


def _asymmetry_change(structure: UniversalStructure, t: float) -> float:
    wave = structure.wave
    return wave.sample("information_asymmetry", min(1.0, t + SIGN_WINDOW)) - wave.sample(
        "information_asymmetry", max(0.0, t - SIGN_WINDOW)
    )


def _validate_rhythm(sid: str, structure: UniversalStructure, problems: list[str]) -> None:
    profile = structure.rhythm_profile
    if profile is None:
        problems.append(f"{sid}: rhythm_profile がありません")
        return
    if not 3 <= profile.page_count <= 16:
        problems.append(f"{sid}: page_count {profile.page_count} が3〜16の範囲外")
    if any(i % 2 for i in profile.page_turn_reveals):
        problems.append(f"{sid}: page_turn_reveals に奇数（左ページ）インデックスがあります")
    if not profile.page_turn_reveals:
        problems.append(f"{sid}: page_turn_reveals が空です")
    if any(i % 2 for i in profile.spread_pages):
        problems.append(f"{sid}: spread_pages に奇数インデックスがあります")
    if len(profile.pacing_notes) < profile.page_count:
        problems.append(f"{sid}: pacing_notes がページ数より少ない")
    tension = structure.wave.channel("tension")
    peak_t = structure.wave.peak_t("tension")
    denominator = max(1, profile.page_count - 1)
    tolerance = max(0.06, 0.5 / denominator + 0.02)
    accent_pages = set(profile.emphasis_pages) | set(profile.page_turn_reveals)
    accent_pages |= set(profile.spread_pages) | {i + 1 for i in profile.spread_pages}
    if not accent_pages:
        problems.append(f"{sid}: 強調・めくり・見開きページが1つもありません")
    elif min(abs(i / denominator - peak_t) for i in accent_pages) > tolerance:
        problems.append(f"{sid}: 緊張度ピーク t={peak_t} と強調/めくり/見開きページの位置が整合しません")
    grid = np.asarray(structure.wave.t)
    mean_tension = float(tension.mean())
    for page in profile.page_turn_reveals:
        position = page / denominator
        indices = [i for i in range(tension.size) if abs(grid[i] - position) <= tolerance]
        peaks = [i for i in indices if _is_local_maximum(tension, i) and tension[i] > mean_tension]
        if not peaks:
            problems.append(f"{sid}: めくりページ {page} の近傍に平均を上回る緊張度のピークがありません")


def validate_patterns(specs: list[dict[str, Any]], structures: list[UniversalStructure]) -> list[str]:
    """SPEC §10/§14 の品質基準・リズム整合・符号整合・波形多様性を検査し、問題の一覧を返す。"""
    problems: list[str] = []
    if len(structures) != PATTERN_COUNT:
        problems.append(f"パターン数が{PATTERN_COUNT}ではありません: {len(structures)}")
    ids = [s.id for s in structures]
    if len(set(ids)) != len(ids):
        problems.append("IDが重複しています")
    numbers: list[int] = []
    for s in structures:
        match = ID_PATTERN.match(s.id)
        if not match:
            problems.append(f"{s.id}: ID形式が md_XX_<slug> ではありません")
        else:
            numbers.append(int(match.group(1)))
    if sorted(numbers) != list(range(1, PATTERN_COUNT + 1)):
        problems.append(f"ID番号が01〜{PATTERN_COUNT}の連番ではありません: {sorted(numbers)}")
    used_edge_kinds: set[str] = set()
    for spec, s in zip(specs, structures):
        sid = s.id
        graph = s.tension_graph
        if not 4 <= len(graph.nodes) <= 8:
            problems.append(f"{sid}: ノード数 {len(graph.nodes)} が4〜8の範囲外")
        if any(node.slot is None for node in graph.nodes):
            problems.append(f"{sid}: slot 未割当のノードがあります")
        if not 5 <= len(graph.edges) <= 12:
            problems.append(f"{sid}: エッジ数 {len(graph.edges)} が5〜12の範囲外")
        kinds = {e.kind.value for e in graph.edges}
        used_edge_kinds |= kinds
        if len(kinds) < 4:
            problems.append(f"{sid}: エッジ種別が {len(kinds)} 種しかありません（4種以上）")
        if not any(e.activation_t is not None for e in graph.edges):
            problems.append(f"{sid}: activation_t を持つエッジがありません")
        if not any(abs(e.gradient) > 0 for e in graph.edges):
            problems.append(f"{sid}: gradient を持つエッジがありません")
        beats = s.beats.beats
        if not 6 <= len(beats) <= 10:
            problems.append(f"{sid}: ビート数 {len(beats)} が6〜10の範囲外")
        if beats and beats[-1].beat_type not in FINAL_BEAT_TYPES:
            problems.append(f"{sid}: 最後のビート {beats[-1].id} が afterglow/payoff/climax ではありません")
        turning = s.beats.turning_points()
        if len(turning) < 2:
            problems.append(f"{sid}: 転換点が2未満")
        tension = s.wave.channel("tension")
        for beat in turning:
            index = int(round(beat.t * (tension.size - 1)))
            if abs(index / (tension.size - 1) - beat.t) > 1e-9:
                problems.append(f"{sid}: 転換点 {beat.id} の t={beat.t} が波形格子点上にありません")
            elif not _is_local_extremum(tension, index):
                problems.append(f"{sid}: 転換点 {beat.id} (t={beat.t}) が緊張度の極値と一致しません")
        peak_t = s.wave.peak_t("tension")
        if not any(abs(b.t - peak_t) <= 0.1 + 1e-9 for b in turning):
            problems.append(f"{sid}: 緊張度ピーク t={peak_t} の ±0.1 に転換点ビートがありません")
        for beat in beats:
            if abs(beat.tension_delta) >= SIGN_CHECK_THRESHOLD:
                observed = _tension_change(s, beat.t)
                if observed * beat.tension_delta <= 0:
                    problems.append(
                        f"{sid}: ビート {beat.id} の tension_delta={beat.tension_delta:+.2f} が波形の変化 {observed:+.3f} と逆向きです"
                    )
            if abs(beat.information_shift) >= SIGN_CHECK_THRESHOLD:
                observed = _asymmetry_change(s, beat.t)
                if observed * beat.information_shift <= 0:
                    problems.append(
                        f"{sid}: ビート {beat.id} の information_shift={beat.information_shift:+.2f} が非対称性の変化 {observed:+.3f} と逆向きです"
                    )
        if not 3 <= len(s.mechanics.mechanisms) <= 5:
            problems.append(f"{sid}: メカニズム数 {len(s.mechanics.mechanisms)} が3〜5の範囲外")
        if any(not m.theory for m in s.mechanics.mechanisms):
            problems.append(f"{sid}: theory が空のメカニズムがあります")
        if not 6 <= len(s.blueprint.slots) <= 10:
            problems.append(f"{sid}: スロット数 {len(s.blueprint.slots)} が6〜10の範囲外")
        for slot in s.blueprint.slots:
            if any(not slot.examples.get(key) for key in ("manga", "marketing", "article")):
                problems.append(f"{sid}: スロット {slot.name} の examples が3領域揃っていません")
        if not 3 <= len(s.blueprint.invariants) <= 6:
            problems.append(f"{sid}: 不変条件数 {len(s.blueprint.invariants)} が3〜6の範囲外")
        guide = s.transfer_guide
        if not (guide.manga and guide.marketing and guide.article):
            problems.append(f"{sid}: transfer_guide が3領域すべて記述されていません")
        if not 3 <= len(s.metadata.tags) <= 6:
            problems.append(f"{sid}: タグ数 {len(s.metadata.tags)} が3〜6の範囲外")
        if not s.metadata.lineage:
            problems.append(f"{sid}: lineage が空です")
        for text in s.text_fields():
            if "http" in text.lower() or "www." in text.lower():
                problems.append(f"{sid}: URL らしき文字列を含みます: {text[:40]}")
        for channel in WAVE_CHANNELS:
            points = spec["waves"].get(channel, [])
            if not 6 <= len(points) <= 12:
                problems.append(f"{sid}: {channel} のキーフレーム数 {len(points)} が6〜12の範囲外")
            for t_value, _ in points:
                if abs(round(t_value * 20) / 20 - t_value) > 1e-9:
                    problems.append(f"{sid}: {channel} のキーフレーム t={t_value} が0.05刻みの格子点上にありません")
        for issue in audit_structure(s):
            problems.append(f"{sid}: 整合性監査 {issue.code}（{issue.location}）: {issue.message}")
        _validate_rhythm(sid, s, problems)
    missing_kinds = {k.value for k in EdgeKind} - used_edge_kinds
    if missing_kinds:
        problems.append(f"全体で未使用のエッジ種別: {sorted(missing_kinds)}")
    tensions = [s.wave.channel("tension") for s in structures]
    for i in range(len(tensions)):
        for j in range(i + 1, len(tensions)):
            r = float(np.corrcoef(tensions[i], tensions[j])[0, 1])
            if r >= CATEGORY_CORRELATION_LIMIT:
                problems.append(f"カテゴリ内の緊張度波形の相関が高すぎます: {ids[i]} × {ids[j]} r={r:.3f}")
    existing = load_existing_tensions()
    for sid, tension in zip(ids, tensions):
        for other_id, other in existing.items():
            if other.size != tension.size:
                continue
            r = float(np.corrcoef(tension, other)[0, 1])
            if r >= EXISTING_CORRELATION_LIMIT:
                problems.append(f"既存シードとの緊張度波形の相関が高すぎます: {sid} × {other_id} r={r:.3f}")
    return problems


def correlation_summary(structures: list[UniversalStructure]) -> dict[str, float]:
    """カテゴリ内・既存シードとの最大相関（報告用）。"""
    tensions = [s.wave.channel("tension") for s in structures]
    within = 0.0
    for i in range(len(tensions)):
        for j in range(i + 1, len(tensions)):
            within = max(within, float(np.corrcoef(tensions[i], tensions[j])[0, 1]))
    existing = load_existing_tensions()
    against = 0.0
    for tension in tensions:
        for other in existing.values():
            if other.size == tension.size:
                against = max(against, float(np.corrcoef(tension, other)[0, 1]))
    return {"within_category_max": round(within, 4), "against_existing_max": round(against, 4)}


def write_json(structures: list[UniversalStructure], path: Path = OUTPUT_PATH) -> Path:
    document = {
        "schema_version": "1.0",
        "category": CATEGORY,
        "description": FILE_DESCRIPTION,
        "patterns": [s.model_dump(mode="json") for s in structures],
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    structures = build_all()
    problems = validate_patterns(PATTERNS, structures)
    if problems:
        for problem in problems:
            print(f"[NG] {problem}", file=sys.stderr)
        return 1
    summary = correlation_summary(structures)
    if "--check" in args:
        print(f"[OK] {len(structures)} パターンの検証に合格しました（相関 {summary}）")
        return 0
    path = write_json(structures)
    print(f"[OK] {len(structures)} パターンを書き出しました: {path}（相関 {summary}）")
    return 0


# ---------------------------------------------------------------------------
# 01: 連続見開きの二連撃
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_01_double_spread_combo",
        "title": "連続見開きの二連撃",
        "summary": "小ゴマの加速で臨界まで溜めた力をめくり直後の第1見開きで解放して決着を信じさせ、強敵の持ち直しを小ゴマの1ページで挟んでから、二度目のめくり直後に同じ構図をより巨大なスケールで反復する第2見開きで上書きする、二段構えの圧倒の骨格。",
        "tags": ["見開き", "連続見開き", "二段構え", "スケール", "決着"],
        "lineage": ["見開き大ゴマの連続使用による圧倒の演出", "偽の決着からの上書き（ダブルクライマックス）", "同構図反復によるスケール比較の技法"],
        "waves": {
            "tension": [(0.0, 0.3), (0.15, 0.55), (0.25, 0.86), (0.4, 0.58), (0.5, 0.7), (0.65, 0.98), (0.8, 0.6), (0.9, 0.42), (1.0, 0.32)],
            "release": [(0.0, 0.05), (0.2, 0.08), (0.25, 0.28), (0.4, 0.1), (0.6, 0.12), (0.65, 0.38), (0.85, 0.65), (1.0, 0.82)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.3), (0.25, 0.55), (0.5, 0.6), (0.65, 0.85), (0.8, 0.92), (1.0, 0.96)],
            "character_knowledge": [(0.0, 0.15), (0.2, 0.35), (0.25, 0.5), (0.5, 0.62), (0.65, 0.8), (0.8, 0.9), (1.0, 0.96)],
            "cognitive_load": [(0.0, 0.35), (0.15, 0.55), (0.25, 0.3), (0.4, 0.5), (0.5, 0.55), (0.65, 0.25), (0.85, 0.4), (1.0, 0.3)],
        },
        "nodes": [
            ("striker", "agent", "protagonist", "二撃目を隠し持つ主体", "第1撃の後にさらに大きな力を解放する主体。余力の存在を最後まで悟らせない", 0.9, "STRIKER"),
            ("rival", "agent", "antagonist", "第1撃を受けても立ち上がる強敵", "偽の決着を覆して持ち直し、第2撃のスケールを引き出す相手", 0.85, "RESILIENT_RIVAL"),
            ("decisive_win", "desire", "goal", "揺るがない決着", "主体が求める、反論の余地のない勝利", 0.8, "DECISIVE_RESULT"),
            ("hidden_reserve", "lack", "withheld_power", "読者に伏せられた余力", "主体が温存している力。第1見開きの時点では読者にも相手にも知らされていない", 0.75, "HIDDEN_RESERVE"),
            ("first_spread", "symbol", "false_finale", "第1見開きの決め絵", "決着に見える一撃の全景。後に前座だったと判明する", 0.7, "FIRST_SPREAD"),
            ("second_spread", "symbol", "true_finale", "第2見開きの決め絵", "第1撃を構図ごと上書きする真の決め絵", 0.95, "SECOND_SPREAD"),
        ],
        "edges": [
            ("striker", "rival", "opposition", 0.9, 0.0, "決着をかけた正面衝突", 0.0),
            ("striker", "decisive_win", "pursuit", 0.9, 0.0, "反論の余地のない勝利を追う", 0.0),
            ("rival", "striker", "power_gradient", 0.7, 0.5, "第1撃を受けてなお上回る耐久", 0.4),
            ("striker", "hidden_reserve", "dependency", 0.75, -0.2, "勝敗を温存した余力に託す", 0.1),
            ("hidden_reserve", "rival", "asymmetry", 0.8, 0.7, "余力は相手にも読者にも見えていない", 0.2),
            ("first_spread", "decisive_win", "embodiment", 0.6, 0.0, "偽の決着を体現する第1撃", 0.25),
            ("second_spread", "first_spread", "opposition", 0.7, 0.6, "第2撃が第1撃の構図を上書きする", 0.65),
            ("second_spread", "hidden_reserve", "embodiment", 0.85, 0.0, "温存された余力の全開放", 0.65),
        ],
        "beats": [
            ("b_buildup", "hook", 0.05, 0.08, "小ゴマ連打の溜め", "細く速い小ゴマで力を臨界へ積み上げ、読者の呼吸を浅くする", [], 0.2, -0.05, ["striker", "rival"], False),
            ("b_accelerate", "escalation", 0.15, 0.06, "左ページ末尾の踏み込み", "左ページ最終コマで主体が踏み込む瞬間だけを切り取り、めくりへ誘う", ["b_buildup"], 0.2, 0.05, ["striker", "decisive_win"], False),
            ("b_first_spread", "false_victory", 0.25, 0.06, "第1見開き：決着に見える一撃", "めくり直後の見開きで渾身の一撃を全景で描き、決着したと読者に信じさせる", ["b_accelerate"], 0.3, 0.1, ["striker", "rival", "first_spread"], True),
            ("b_rival_rises", "crisis", 0.4, 0.05, "強敵の持ち直し", "煙の中から立ち上がる相手を小さなコマで示し、決着の確信を一度崩す", ["b_first_spread"], -0.25, -0.05, ["rival", "striker"], True),
            ("b_reserve_hint", "setup", 0.5, 0.05, "余力の予兆", "主体の手元のECUで温存していた力の気配だけを見せ、左ページ末尾で引く", ["b_rival_rises"], 0.12, 0.05, ["striker", "hidden_reserve"], False),
            ("b_second_spread", "climax", 0.65, 0.06, "第2見開き：構図ごと上書きする真の一撃", "二度目のめくり直後、第1見開きと同じ構図をより巨大なスケールで反復し、真の決着を刻む", ["b_reserve_hint", "b_first_spread"], 0.3, 0.1, ["striker", "rival", "second_spread", "hidden_reserve"], True),
            ("b_aftermath", "payoff", 0.85, 0.08, "静寂の余波", "崩れ落ちる相手と立ち尽くす主体を引きの構図で描き、二撃の差を余波で測らせる", ["b_second_spread"], -0.3, 0.05, ["striker", "rival", "decisive_win"], False),
            ("b_settle", "afterglow", 1.0, 0.04, "呼吸の回復", "会話の小コマで日常の呼吸へ戻し、二連撃の記憶を余韻として残す", ["b_aftermath"], -0.15, 0.0, ["striker"], False),
        ],
        "mechanisms": [
            ("m_false_finale", "prediction_error", "偽の決着による予測の上書き", "第1見開きで決着を確信させ、強敵の持ち直しで予測を崩してから第2見開きで上書きする", "決着に見える第1見開きの直後の持ち直し", "第2撃のスケールが予測誤差として強く記憶される", "予測符号化理論", 0.85, ["b_first_spread", "b_rival_rises", "b_second_spread"], ["rival"]),
            ("m_contrast_scale", "contrast_effect", "二つの見開きの対比", "同じ構図を二度描くことで、第2撃の規模が第1撃との比較で相対的に増幅される", "同構図・同視線方向の反復", "第2撃が数値ではなく体感として巨大に感じられる", "対比効果（コントラスト効果）", 0.8, ["b_second_spread"], ["first_spread", "second_spread"]),
            ("m_turn_reward", "anticipation_reward", "めくりの予期と報酬", "左ページ末尾の踏み込みと余力の予兆が、めくりのたびに報酬を予期させる", "左ページ最終コマの引き", "ページをめくる動作そのものが期待の高まりになる", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.7, ["b_accelerate", "b_reserve_hint"], ["hidden_reserve"]),
            ("m_two_stage_catharsis", "catharsis", "二段の解放", "一度目の解放を寸止めにして二度目で全開放することで、解放の総量が最大化する", "偽の決着と真の決着の二段構え", "読後に長いカタルシスが残る", "カタルシス理論", 0.7, ["b_second_spread", "b_aftermath"], ["striker"]),
        ],
        "anchor": "限界の先にもう一段の力を隠し持ち、決着の瞬間まで諦めない主体の執念",
        "drivers": ["決着に見える第1見開きの衝撃", "強敵の持ち直しによる確信の崩壊", "第1撃を構図ごと上書きする第2見開き"],
        "emotions": ["緊迫", "高揚", "確信", "動揺", "期待", "圧倒", "余韻"],
        "template": "{{STRIKER}}は{{DECISIVE_RESULT}}を求め、小ゴマの連打で力を溜めてから、めくり直後の{{FIRST_SPREAD}}で渾身の一撃を放つ。しかし{{RESILIENT_RIVAL}}は持ち直し、決着の確信は崩れる。{{STRIKER}}は{{HIDDEN_RESERVE}}を解き放ち、二度目のめくり直後の{{SECOND_SPREAD}}で第1撃の構図ごと上書きして真の決着を刻む。",
        "slots": [
            ("STRIKER", "二撃目を隠し持つ主体", "agent", ["第1撃の時点で余力の存在を読者に悟らせない"], ("満身創痍で立つ主人公", "二段階の提案を準備したブランド", "二つ目の論拠を温存した筆者")),
            ("RESILIENT_RIVAL", "第1撃を受けても持ち直す相手", "agent", ["持ち直しは小さなコマで示す"], ("煙の中から立ち上がる宿敵", "最初の提案では動かない慎重な顧客", "第一の反証では崩れない有力な通説")),
            ("DECISIVE_RESULT", "反論の余地のない決着", "goal", ["第2撃でしか到達しない"], ("宿敵の完全な敗北", "迷いのない導入決定", "通説の明確な更新")),
            ("HIDDEN_RESERVE", "読者にも伏せられた余力", "lack", ["第1見開きまで一切示さない", "予兆はECU一コマに限る"], ("封印していた奥義", "上位プランの追加価値", "決定的な追試データ")),
            ("FIRST_SPREAD", "決着に見える第1見開き", "symbol", ["めくり直後の右ページから始める"], ("大地を割る一撃の全景", "業界標準を超える基本性能の提示", "第一の反証データの図解")),
            ("SECOND_SPREAD", "第1撃を構図ごと上書きする第2見開き", "symbol", ["第1見開きと同じ構図をより大きなスケールで反復する"], ("空まで裂く二撃目の全景", "基本性能をさらに上回る上位価値の提示", "通説そのものを覆す決定的データの図解")),
        ],
        "invariants": [
            "見開きは必ずめくり直後の右ページから始め、2回とも右→左に跨がせる",
            "第1見開きと第2見開きの間に、持ち直しを示す小ゴマ中心のページを1枚以上挟む",
            "第2見開きは第1見開きと同じ構図・同じ視線方向で、スケールだけを拡大する",
            "見開き内の視線誘導は右上の主体から左下の着弾点へ一直線に流す",
        ],
        "guide": {
            "manga": "10ページ構成。p1（右）: 5コマの小ゴマで構えと間合い（MS・アイレベル中心）。p2（左）: 6コマの斜めゴマで加速し、最終コマは踏み込む足元のECU・アオリで引く。p3〜p4: めくり直後の第1見開き（ELS・広角24mm、右上の主体から左下の着弾点へ）。p5（右）: 煙の中から立ち上がる相手を4コマの小ゴマで示す（CU・俯瞰で主体の動揺）。p6（左）: 5コマで余力の予兆を手元ECUで1コマ挟み、最終コマで溜める。p7〜p8: 二度目のめくり直後の第2見開き（第1見開きと同構図を18mmの超広角で拡大）。p9（右）: 3コマの引きの余波（ELS・俯瞰）。p10（左）: 4コマの会話で日常の呼吸へ戻す。",
            "marketing": "LPで最初の価値提案（第1見開き）を全幅のビジュアルで示して一度納得させ、続くセクションで顧客がまだ残す懸念（強敵の持ち直し）を正直に提示する。その直後、同じレイアウト・同じ見出し構造で、懸念ごと上書きする上位価値（第2見開き）を全幅で提示し、比較によって価値の大きさを体感させてからCTAへ進む。",
            "article": "記事中盤で第一の論拠を図解付きで提示していったん結論が出たように見せ、続く節で有力な反論を誠実に検討する。その後、第一の図解と同じ形式で、反論ごと覆す決定的な追試データを示し、二段の論証によって結論の強度を最大化する。",
        },
        "rhythm": {
            "page_count": 10,
            "panels_per_page": [5, 6, 1, 1, 4, 5, 1, 1, 3, 4],
            "emphasis_pages": [2, 6],
            "page_turn_reveals": [2, 6],
            "spread_pages": [2, 6],
            "silent_panels_ratio": 0.15,
            "pacing_notes": [
                "p1（右）: 5コマの小ゴマで構えと間合いを刻み、呼吸を浅くする",
                "p2（左）: 6コマの斜めゴマで加速し、最終コマの踏み込みでめくりへ誘う",
                "p3（右・めくり直後）: 第1見開きの右半分。主体の振りかぶりから一撃が走り出す",
                "p4（左）: 第1見開きの左半分。着弾点の全景で決着を信じさせる",
                "p5（右）: 4コマの小ゴマで煙の中の持ち直しを示し、確信を崩す",
                "p6（左）: 5コマで余力の予兆を手元ECUで1コマ挟み、最終コマで溜める",
                "p7（右・めくり直後）: 第2見開きの右半分。第1見開きと同構図でスケールを拡大する",
                "p8（左）: 第2見開きの左半分。第1撃を上書きする着弾の全景",
                "p9（右）: 3コマの引きの余波で二撃の差を測らせる",
                "p10（左）: 4コマの会話で日常の呼吸へ戻す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 02: 縦長ゴマの落下感（垂直移動による時間の重力）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_02_vertical_fall_gravity",
        "title": "縦長ゴマの落下感（垂直移動による時間の重力）",
        "summary": "足場の崩落を左ページ末尾の引きに置き、めくり直後の縦長ゴマ3本で落下と深淵の全貌を同時に開示し、無重力の一瞬を細い空白で挟んでから再び加速させ、二度目のめくり直後の1ページ大ゴマで着地の衝撃を刻む、視線の垂直移動を時間の重力に変える骨格。",
        "tags": ["縦長ゴマ", "落下", "垂直移動", "時間の重力", "めくり"],
        "lineage": ["縦長ゴマによる落下・下降の表現慣行", "視線の垂直移動と時間経過の対応（読書行為の身体性）", "無重力の一瞬（ハングタイム）の演出"],
        "waves": {
            "tension": [(0.0, 0.25), (0.15, 0.38), (0.3, 0.55), (0.4, 0.8), (0.5, 0.68), (0.65, 0.86), (0.8, 0.97), (0.9, 0.55), (1.0, 0.35)],
            "release": [(0.0, 0.08), (0.3, 0.06), (0.4, 0.1), (0.5, 0.3), (0.65, 0.08), (0.8, 0.15), (0.9, 0.6), (1.0, 0.78)],
            "audience_knowledge": [(0.0, 0.1), (0.3, 0.3), (0.4, 0.6), (0.6, 0.65), (0.8, 0.8), (0.9, 0.86), (1.0, 0.9)],
            "character_knowledge": [(0.0, 0.2), (0.3, 0.4), (0.4, 0.55), (0.6, 0.65), (0.8, 0.85), (0.9, 0.88), (1.0, 0.9)],
            "cognitive_load": [(0.0, 0.4), (0.3, 0.5), (0.4, 0.3), (0.5, 0.2), (0.65, 0.35), (0.8, 0.25), (1.0, 0.45)],
        },
        "nodes": [
            ("faller", "agent", "protagonist", "足場を失って落ちる主体", "崩落に巻き込まれ、抗えない重力に身を預けざるを得なくなる主体", 0.9, "FALLER"),
            ("reaching_hand", "agent", "rescuer", "深淵の縁から手を伸ばす者", "落下の終盤に間に合おうとする協力者。主体はその存在にまだ気づいていない", 0.65, "REACHING_HAND"),
            ("survival", "desire", "goal", "着地して生き延びること", "落下の間ずっと主体を駆動する切実な欲求", 0.85, "SURVIVAL_GOAL"),
            ("lost_footing", "lack", "lost_ground", "失われた足場", "主体が体重を預けていた拠り所。崩落によって主体を裏切る", 0.75, "LOST_FOOTING"),
            ("abyss", "symbol", "depth", "底の見えない深淵", "縦長ゴマの長さそのものとして描かれる距離と時間", 0.8, "ABYSS"),
            ("vertical_frame", "symbol", "tall_panel", "縦に引き伸ばされたコマ枠", "視線を上から下へ長く滑らせ、落下時間を読書時間に変換する装置", 0.7, "VERTICAL_PANEL"),
        ],
        "edges": [
            ("faller", "survival", "pursuit", 0.9, 0.0, "生き延びるための必死の追求", 0.0),
            ("abyss", "faller", "power_gradient", 0.85, 0.8, "深淵の引力が主体を圧倒する", 0.4),
            ("faller", "lost_footing", "dependency", 0.7, -0.3, "足場に体重を預けていた", 0.0),
            ("lost_footing", "faller", "betrayal", 0.75, 0.0, "足場が崩れて主体を裏切る", 0.3),
            ("reaching_hand", "abyss", "opposition", 0.6, 0.0, "深淵の引力に抗って手を伸ばす", 0.65),
            ("vertical_frame", "abyss", "embodiment", 0.7, 0.0, "縦長ゴマが落下距離を体現する", 0.4),
            ("faller", "reaching_hand", "asymmetry", 0.5, -0.4, "主体は差し伸べられた手にまだ気づいていない", 0.5),
        ],
        "beats": [
            ("b_ledge", "hook", 0.05, 0.08, "縁での対峙", "高所の縁で主体と状況を横長のコマで提示し、下方の存在を意識させる", [], 0.15, 0.0, ["faller", "abyss"], False),
            ("b_crumble", "escalation", 0.3, 0.08, "足場の崩落", "細かい亀裂の小コマを重ね、左ページ最終コマで足元が抜ける瞬間を切って引く", ["b_ledge"], 0.2, 0.05, ["faller", "lost_footing"], False),
            ("b_drop", "revelation", 0.4, 0.06, "落下の始まりと深淵の全貌", "めくり直後の縦長ゴマ3本で落下が始まり、底の見えない深さが初めて読者に開示される", ["b_crumble"], 0.25, 0.3, ["faller", "abyss", "vertical_frame"], True),
            ("b_weightless", "transition", 0.5, 0.05, "無重力の一瞬", "細い空白コマで時間を止め、宙に浮いた主体の表情だけを置く", ["b_drop"], -0.12, 0.0, ["faller"], True),
            ("b_accelerate", "escalation", 0.65, 0.08, "加速する落下", "縦長ゴマを2本に減らして速度線を増し、視線の落下速度を上げる", ["b_weightless"], 0.18, 0.05, ["faller", "vertical_frame", "reaching_hand"], False),
            ("b_impact", "climax", 0.8, 0.05, "着地の一枚絵", "二度目のめくり直後の1ページ大ゴマで着地の衝撃を全景で刻む", ["b_accelerate"], 0.25, 0.05, ["faller", "abyss", "reaching_hand"], True),
            ("b_aftermath", "afterglow", 1.0, 0.06, "砂煙の静寂", "横長の静かなコマで砂煙と差し伸べられた手を描き、落下の時間を余韻に変える", ["b_impact"], -0.3, 0.0, ["faller", "reaching_hand", "survival"], False),
        ],
        "mechanisms": [
            ("m_reading_gravity", "narrative_transportation", "読書行為の重力化", "縦長ゴマを上から下へ読む視線運動が、落下する身体感覚と同期する", "縦に引き伸ばされたコマ枠の連続", "読者が自分の視線で落下を体験する", "物語移入理論（身体化された読書）", 0.85, ["b_drop", "b_accelerate"], ["vertical_frame"]),
            ("m_suspended_time", "anticipation_reward", "宙吊りの予期", "無重力の一瞬で時間を止め、着地の瞬間への予期を最大化する", "細い空白コマと宙に浮いた表情", "着地の一枚絵が強い報酬として届く", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.75, ["b_weightless", "b_impact"], ["faller"]),
            ("m_loss_of_ground", "loss_aversion", "拠り所を失う恐怖", "当たり前に預けていた足場を失う体験が、読者自身の安全基盤への不安を呼び起こす", "足元が抜ける瞬間の引き", "主体の生存に強く感情移入する", "損失回避（プロスペクト理論）", 0.7, ["b_crumble", "b_drop"], ["lost_footing", "faller"]),
        ],
        "anchor": "信じていた足場を突然失い、それでも生き延びようともがく身体の切実さ",
        "drivers": ["左ページ末尾で足元が抜ける引き", "縦長ゴマが体感させる落下距離", "無重力の一瞬から着地までの予期"],
        "emotions": ["緊張", "恐怖", "絶望", "静止", "加速", "衝撃", "安堵"],
        "template": "{{FALLER}}は{{LOST_FOOTING}}を失い、{{SURVIVAL_GOAL}}だけを頼りに{{ABYSS}}へ落ちていく。めくり直後の{{VERTICAL_PANEL}}が落下距離を読書時間に変え、無重力の一瞬を挟んで再び加速する。二度目のめくり直後の大ゴマで着地の衝撃が刻まれ、砂煙の向こうで{{REACHING_HAND}}の存在が明かされる。",
        "slots": [
            ("FALLER", "足場を失って落ちる主体", "agent", ["落下中の表情は無重力の一瞬にだけ大きく描く"], ("崖から落ちる主人公", "急激な市場変化に飲まれる担当者", "前提が崩れた研究者")),
            ("LOST_FOOTING", "主体が預けていた拠り所", "lack", ["崩落は左ページ最終コマに置く"], ("崩れる岩棚", "突然終了した主要取引", "否定された前提理論")),
            ("SURVIVAL_GOAL", "生き延びること", "goal", ["落下の間ずっと保持される"], ("生還", "事業の存続", "研究の継続")),
            ("ABYSS", "底の見えない深淵", "symbol", ["めくり直後に初めて全貌を見せる"], ("霧の底の谷", "見通せない市場の下落", "解けない問題の深さ")),
            ("VERTICAL_PANEL", "縦に引き伸ばされたコマ枠", "symbol", ["ページ高の8割以上の縦長ゴマを使う"], ("ページを貫く縦長ゴマ3本", "縦長スクロールで下降を見せるグラフ", "段階的に深掘りする縦の図解")),
            ("REACHING_HAND", "深淵の縁から手を伸ばす者", "agent", ["着地まで正体を明かさない"], ("崖上から伸びる仲間の手", "危機に駆けつける支援パートナー", "思わぬ方向から届く反証データ")),
        ],
        "invariants": [
            "落下は縦長ゴマ（ページ高の8割以上）で描き、視線を上から下へ長く滑らせる",
            "足場の崩落は左ページ最終コマに置き、落下の開始はめくり直後の右ページに置く",
            "無重力の一瞬は細い空白コマで挟み、セリフを置かない",
            "着地は二度目のめくり直後の1ページ大ゴマで描く",
        ],
        "guide": {
            "manga": "6ページ構成。p1（右）: 5コマで高所の縁と状況を提示（ELSで高さ、MSで主体）。p2（左）: 4コマで亀裂を小コマに重ね、最終コマで足元が抜ける瞬間を俯瞰のECUで引く。p3（右・めくり直後）: ページを縦に貫く縦長ゴマ3本で落下と深淵の全貌を開示（俯瞰・広角18mm）。p4（左）: 縦長ゴマ2本で無重力の一瞬（表情CU・ダッチ）と加速（速度線）を描く。p5（右・めくり直後）: 1ページ大ゴマで着地の衝撃（アオリ・広角）。p6（左）: 4コマの横長ゴマで砂煙と差し伸べられた手を静かに描く。",
            "marketing": "LPで安定していた前提（足場）が崩れる瞬間を区切りの直前に置き、スクロール直後に縦長の図解で下落の深さを一気に見せる。いったん余白で読者の思考を止め、続けて縦方向に加速する事例を並べたうえで、底を打った地点で解決策（差し伸べられた手）を提示する。",
            "article": "前提が崩れる瞬間を章末に置き、次章冒頭で問題の深さを縦長の図表で一挙に示す。短い一文の段落で思考を止める間を作り、加速する事例を重ねてから、底に到達した地点で新しい足場となる解決策を提示する。",
        },
        "rhythm": {
            "page_count": 6,
            "panels_per_page": [5, 4, 3, 2, 1, 4],
            "emphasis_pages": [2, 3, 4],
            "page_turn_reveals": [2, 4],
            "spread_pages": [],
            "silent_panels_ratio": 0.3,
            "pacing_notes": [
                "p1（右）: 5コマで高所の縁と下方の存在を提示する",
                "p2（左）: 4コマで亀裂を重ね、最終コマで足元が抜ける瞬間を切って引く",
                "p3（右・めくり直後）: ページを貫く縦長ゴマ3本で落下と深淵の全貌を同時に開示する",
                "p4（左）: 縦長ゴマ2本で無重力の一瞬と再加速を描き、ページ末尾で地面を予感させる",
                "p5（右・めくり直後）: 1ページ大ゴマで着地の衝撃を刻む",
                "p6（左）: 4コマの横長ゴマで砂煙と差し伸べられた手を静かに描く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 03: コマ枠の崩壊（心理の破綻と枠線消失）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_03_frame_collapse",
        "title": "コマ枠の崩壊（心理の破綻と枠線消失）",
        "summary": "整った枠線の日常から枠線を少しずつ細らせ、めくり直後の亀裂、断片化、枠線が完全に消える1ページの崩壊、漂う断片を経て、三度目のめくり直後に太い枠線で心を組み直す、枠線そのものを心理の状態計として扱う骨格。",
        "tags": ["コマ枠", "心理描写", "崩壊", "再構築", "枠線"],
        "lineage": ["枠線の太さ・有無による心理表現の慣行", "断片化したコマによる解離の描写", "枠線の回復による再統合の象徴化"],
        "waves": {
            "tension": [(0.0, 0.3), (0.15, 0.42), (0.3, 0.68), (0.4, 0.55), (0.6, 0.97), (0.7, 0.62), (0.85, 0.76), (0.95, 0.45), (1.0, 0.38)],
            "release": [(0.0, 0.1), (0.3, 0.06), (0.45, 0.12), (0.6, 0.04), (0.7, 0.35), (0.85, 0.3), (0.95, 0.7), (1.0, 0.78)],
            "audience_knowledge": [(0.0, 0.15), (0.3, 0.35), (0.45, 0.4), (0.6, 0.7), (0.85, 0.85), (1.0, 0.92)],
            "character_knowledge": [(0.0, 0.2), (0.3, 0.3), (0.45, 0.32), (0.6, 0.5), (0.85, 0.82), (1.0, 0.92)],
            "cognitive_load": [(0.0, 0.35), (0.3, 0.55), (0.45, 0.65), (0.6, 0.85), (0.7, 0.6), (0.85, 0.45), (1.0, 0.3)],
        },
        "nodes": [
            ("self", "agent", "protagonist", "心の均衡を失いかける主体", "外圧に耐えきれず、知覚の枠組みそのものが崩れていく主体", 0.9, "FRACTURING_SELF"),
            ("pressure", "agent", "external_pressure", "主体を追い詰める外圧", "喪失・恐怖・重圧など、心の枠を内側から押し広げる力", 0.8, "EXTERNAL_PRESSURE"),
            ("stability", "desire", "goal", "心の均衡の回復", "主体が取り戻そうとする、世界を区切って理解できる状態", 0.8, "INNER_STABILITY"),
            ("broken_frame", "lack", "lost_structure", "失われた心の枠組み", "出来事を順序立てて受け止めるための内的構造の欠落", 0.75, "BROKEN_FRAME"),
            ("panel_border", "symbol", "panel_frame", "コマの枠線", "主体の心理状態を映す計器。太さと有無で均衡の度合いを示す", 0.85, "PANEL_BORDER"),
            ("anchor_memory", "symbol", "anchor", "立ち直りのよりどころとなる記憶", "崩壊の底で主体を繋ぎ止める一つの具体的な記憶", 0.65, "ANCHOR_MEMORY"),
        ],
        "edges": [
            ("pressure", "self", "power_gradient", 0.8, 0.7, "外圧が主体の許容量を上回る", 0.2),
            ("self", "stability", "pursuit", 0.85, 0.0, "均衡の回復を求める", 0.0),
            ("self", "panel_border", "dependency", 0.7, 0.0, "枠線の秩序に心を預ける", 0.0),
            ("panel_border", "broken_frame", "embodiment", 0.8, 0.0, "消えゆく枠線が心の破綻を体現する", 0.3),
            ("pressure", "stability", "opposition", 0.75, 0.0, "外圧が均衡を打ち砕く", 0.3),
            ("anchor_memory", "self", "asymmetry", 0.6, 0.5, "記憶だけが主体の知らない支えを示す", 0.85),
            ("self", "anchor_memory", "pursuit", 0.55, 0.0, "断片の中から記憶を掴み直す", 0.8),
        ],
        "beats": [
            ("b_order", "hook", 0.05, 0.06, "整った枠線の日常", "均一な太さの枠線で区切られた6コマで、主体の均衡状態を基準線として示す", [], 0.08, 0.0, ["self", "panel_border"], False),
            ("b_thinning", "escalation", 0.15, 0.08, "細くなる枠線", "外圧の兆しとともに枠線を細らせ、左ページ末尾のコマを破線にして引く", ["b_order"], 0.12, 0.05, ["pressure", "panel_border"], False),
            ("b_crack", "structural_twist", 0.3, 0.06, "最初の亀裂", "めくり直後のコマで枠線が割れ、主体の認識の秩序が初めて壊れたことを示す", ["b_thinning"], 0.25, 0.1, ["self", "panel_border", "broken_frame"], True),
            ("b_fragment", "crisis", 0.4, 0.06, "断片化する視界", "コマが大小ばらばらの断片に分かれ、読み順が揺らぐ", ["b_crack"], -0.13, 0.05, ["self", "broken_frame"], True),
            ("b_collapse", "climax", 0.6, 0.06, "枠線消失の一枚絵", "二度目のめくり直後、枠線の一切ない1ページに主体と外圧を溶け合わせて描く", ["b_fragment"], 0.4, 0.2, ["self", "pressure", "broken_frame"], True),
            ("b_float", "transition", 0.7, 0.05, "漂う断片", "枠のない余白に小さな断片コマが漂い、その中に記憶の一片が紛れ込む", ["b_collapse"], -0.3, 0.05, ["anchor_memory", "self"], False),
            ("b_reassemble", "costly_breakthrough", 0.85, 0.06, "太い枠線での再構築", "三度目のめくり直後、記憶を掴んだ主体が太い枠線のコマで世界を区切り直す", ["b_float"], 0.14, 0.05, ["self", "anchor_memory", "stability"], True),
            ("b_new_frame", "afterglow", 1.0, 0.05, "新しい枠の日常", "以前より太い枠線の4コマで日常を描き直し、変化した均衡を示す", ["b_reassemble"], -0.3, 0.0, ["self", "panel_border", "stability"], False),
        ],
        "mechanisms": [
            ("m_frame_as_gauge", "narrative_transportation", "枠線という心理計", "枠線の太さと有無が主体の心理状態と同期し、読者は絵柄より先に枠の変化で異変を感じ取る", "段階的に細り、割れ、消える枠線", "言語化されない心理の破綻を身体的に体験する", "物語移入理論", 0.85, ["b_thinning", "b_crack", "b_collapse"], ["panel_border"]),
            ("m_schema_break", "prediction_error", "読みの枠組みの破壊", "コマ枠という読書の前提そのものが壊れることで、強い予測誤差と不安が生じる", "枠線消失の1ページ", "読者自身が足場を失う感覚を共有する", "スキーマ理論と予測誤差", 0.8, ["b_collapse"], ["broken_frame"]),
            ("m_reintegration", "catharsis", "再統合のカタルシス", "太い枠線で世界を区切り直す瞬間に、崩壊した秩序が回復する解放感が生まれる", "記憶を掴んだ直後の太い枠線", "崩壊の深さに比例した安堵", "カタルシス理論", 0.75, ["b_reassemble", "b_new_frame"], ["anchor_memory", "self"]),
            ("m_empathy_fracture", "empathy_transfer", "破綻への共感", "主体の心の枠組みが崩れる過程を読者が読書体験として追体験する", "断片化する視界と揺らぐ読み順", "主体の苦痛を自分事として受け取る", "共感の知覚‐行為モデル", 0.65, ["b_fragment"], ["self"]),
        ],
        "anchor": "外からの重圧で心の枠組みが壊れていくとき、それでも一つの記憶にすがって自分を組み直そうとする人間の強さ",
        "drivers": ["細くなり割れていく枠線の不穏", "枠線が消える1ページの衝撃", "太い枠線で世界を区切り直す瞬間"],
        "emotions": ["平穏", "不安", "動揺", "混乱", "虚無", "微光", "回復"],
        "template": "{{FRACTURING_SELF}}の日常は均一な{{PANEL_BORDER}}で区切られているが、{{EXTERNAL_PRESSURE}}によって枠線は細り、めくり直後に割れ、二度目のめくり直後には完全に消えて{{BROKEN_FRAME}}が露わになる。漂う断片の中で{{ANCHOR_MEMORY}}を掴んだ主体は、三度目のめくり直後に太い枠線で世界を区切り直し、{{INNER_STABILITY}}を新しい形で取り戻す。",
        "slots": [
            ("FRACTURING_SELF", "心の均衡を失いかける主体", "agent", ["崩壊中も主体の表情だけは描き続ける"], ("喪失に直面した主人公", "相次ぐトラブルに疲弊した担当者", "前提を失った研究者")),
            ("PANEL_BORDER", "心理状態を映すコマの枠線", "symbol", ["太さと有無以外の装飾を変えない"], ("細る枠線と消える枠線", "崩れていくダッシュボードの罫線", "章ごとに変わる図の枠")),
            ("EXTERNAL_PRESSURE", "主体を追い詰める外圧", "agent", ["枠線の変化と同期させる"], ("大切な人の喪失", "同時多発する障害対応", "反証データの連続")),
            ("BROKEN_FRAME", "失われた心の枠組み", "lack", ["枠線消失の1ページで露わにする"], ("出来事を受け止められない空白", "優先順位を失った業務", "説明体系の崩壊")),
            ("ANCHOR_MEMORY", "立ち直りのよりどころとなる記憶", "symbol", ["崩壊の前に一度だけ伏線として描く"], ("幼い日の約束", "最初の顧客の感謝の言葉", "研究を始めた原点の問い")),
            ("INNER_STABILITY", "取り戻される心の均衡", "goal", ["元と同じではなく変化した形で回復させる"], ("喪失を抱えて生きる覚悟", "新しい業務設計", "更新された理論体系")),
        ],
        "invariants": [
            "枠線の太さは均衡度と単調に対応させ、崩壊へ向かうほど細く・破線に・無しへと変える",
            "亀裂・消失・再構築はいずれもめくり直後の右ページ冒頭に置く",
            "枠線消失のページでも主体の顔は判読できる大きさで残し、視線の着地点を確保する",
            "再構築後の枠線は崩壊前より太くし、変化した均衡を可視化する",
        ],
        "guide": {
            "manga": "8ページ構成。p1（右）: 均一な太さの枠線で6コマの日常（MS・アイレベル）。p2（左）: 5コマで枠線を細らせ、最終コマを破線にして引く。p3（右・めくり直後）: 4コマ、冒頭コマの枠線が割れる（CU・ダッチアングル）。p4（左）: 4コマの大小ばらばらの断片ゴマで読み順を揺らす。p5（右・めくり直後）: 枠線なしの1ページで主体と外圧を溶け合わせる（ECU・俯瞰）。p6（左）: 5つの小さな断片コマが余白に漂い、その一つに記憶を置く。p7（右・めくり直後）: 太い枠線の3コマで記憶を掴み世界を区切り直す（アオリ）。p8（左）: 太い枠線の4コマで変化した日常を描く。",
            "marketing": "LPのセクション罫線や余白を顧客の不安の高まりと同期させ、問題の深刻化に合わせて区切りを崩していく。最も混乱した状態を区切りのない全面ビジュアルで見せた後、太く明確な区切りのフレームワーク（解決策）で情報を整理し直し、秩序の回復をCTAへの導線にする。",
            "article": "記事の構成そのものを崩壊と再構築の比喩として用い、問題の章では段落を短く断片化し、最も深刻な局面で見出しのない一節を置く。その後、明確な見出しと図解で議論を区切り直す章を設け、読者に秩序の回復を体験させる。",
        },
        "rhythm": {
            "page_count": 8,
            "panels_per_page": [6, 5, 4, 4, 1, 5, 3, 4],
            "emphasis_pages": [2, 4, 6],
            "page_turn_reveals": [2, 4, 6],
            "spread_pages": [],
            "silent_panels_ratio": 0.35,
            "pacing_notes": [
                "p1（右）: 均一な太さの枠線で6コマの日常を描き、基準線を作る",
                "p2（左）: 5コマで枠線を少しずつ細らせ、最終コマを破線にして引く",
                "p3（右・めくり直後）: 4コマ、冒頭コマの枠線が割れて秩序の崩れを開示する",
                "p4（左）: 大小ばらばらの断片4コマで読み順を揺らし、崩壊を予感させる",
                "p5（右・めくり直後）: 枠線のない1ページで主体と外圧を溶け合わせる",
                "p6（左）: 余白に漂う5つの断片コマの一つに記憶を紛れ込ませる",
                "p7（右・めくり直後）: 太い枠線の3コマで記憶を掴み世界を区切り直す",
                "p8（左）: 以前より太い枠線の4コマで変化した日常を描く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 04: 定点構図の反復と差分
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_04_fixed_point_difference",
        "title": "定点構図の反復と差分",
        "summary": "同じ画角・同じ構図のコマを反復して読者の目を慣らし、めくり直後に最初の差分を気づかせ、取り繕う日常の多コマを挟んでから、二度目のめくり直後に同じ構図の中の不在を1ページ大ゴマで突きつける、変化を差分だけで語る定点観測の骨格。",
        "tags": ["定点観測", "反復", "差分", "不在", "静かな衝撃"],
        "lineage": ["固定カメラによる定点反復の演出", "差異の知覚と習慣化（反復による馴化と脱馴化）", "不在によって存在を語る喪失表現"],
        "waves": {
            "tension": [(0.0, 0.22), (0.15, 0.25), (0.3, 0.3), (0.4, 0.62), (0.45, 0.46), (0.6, 0.42), (0.7, 0.5), (0.8, 0.94), (0.9, 0.48), (1.0, 0.2)],
            "release": [(0.0, 0.4), (0.3, 0.35), (0.4, 0.1), (0.6, 0.2), (0.8, 0.05), (0.9, 0.35), (1.0, 0.72)],
            "audience_knowledge": [(0.0, 0.2), (0.3, 0.25), (0.4, 0.5), (0.7, 0.55), (0.8, 0.9), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.25), (0.3, 0.3), (0.4, 0.35), (0.7, 0.45), (0.8, 0.6), (1.0, 0.85)],
            "cognitive_load": [(0.0, 0.3), (0.2, 0.55), (0.4, 0.65), (0.6, 0.5), (0.8, 0.3), (1.0, 0.2)],
        },
        "nodes": [
            ("observer", "agent", "viewpoint", "定点の前で日々を過ごす主体", "同じ場所・同じ時刻を繰り返し生き、変化を見落とし続ける主体", 0.85, "OBSERVER"),
            ("absent_one", "agent", "departing", "定点から静かに消えていく者", "反復の中で少しずつ痕跡を薄め、最後に不在として現れる存在", 0.8, "ABSENT_ONE"),
            ("continuity", "desire", "goal", "変わらない日々の継続", "主体が無意識に信じている、明日も同じ構図が続くという前提", 0.75, "CONTINUITY"),
            ("difference", "lack", "unnoticed_change", "見落とされた差分", "反復の中に紛れた小さな変化。読者だけが先に気づく", 0.7, "SUBTLE_DIFFERENCE"),
            ("fixed_frame", "symbol", "fixed_composition", "反復される同一構図", "同じ画角・同じ配置で繰り返されるコマ。差分を測る物差しになる", 0.85, "FIXED_FRAME"),
            ("empty_object", "symbol", "absence_marker", "空いた椅子", "最後の大ゴマで不在そのものを体現する物", 0.8, "EMPTY_OBJECT"),
        ],
        "edges": [
            ("observer", "continuity", "pursuit", 0.8, 0.0, "変わらない日々を求める", 0.0),
            ("observer", "absent_one", "dependency", 0.75, -0.2, "いて当然という依存", 0.0),
            ("absent_one", "observer", "asymmetry", 0.7, 0.6, "去る者だけが別れを知っている", 0.4),
            ("fixed_frame", "continuity", "embodiment", 0.7, 0.0, "同一構図が継続の前提を体現する", 0.0),
            ("difference", "fixed_frame", "opposition", 0.65, 0.0, "差分が反復を侵食する", 0.4),
            ("empty_object", "absent_one", "embodiment", 0.8, 0.0, "空いた椅子が不在を体現する", 0.8),
            ("difference", "observer", "power_gradient", 0.6, 0.5, "差分が観察者の確信を崩す", 0.7),
        ],
        "beats": [
            ("b_establish", "hook", 0.05, 0.08, "定点の確立", "同じ画角の4コマで場所と人物の配置を基準線として刻む", [], 0.05, 0.0, ["observer", "fixed_frame"], False),
            ("b_repeat", "premise", 0.25, 0.08, "反復が生む安心", "次のページも同じ構図で繰り返し、読者の目を慣らしながら最後のコマにだけ微かな違和を置く", ["b_establish"], 0.05, 0.05, ["observer", "absent_one", "fixed_frame"], False),
            ("b_first_diff", "revelation", 0.4, 0.05, "最初の差分", "めくり直後、同じ構図の中で一つだけ欠けた物をCUで示し、読者に変化を気づかせる", ["b_repeat"], 0.3, 0.25, ["difference", "fixed_frame", "absent_one"], True),
            ("b_normalcy", "crisis", 0.6, 0.08, "取り繕う日常", "主体は変化に気づかないまま多コマの会話を続け、読者だけが不安を抱える", ["b_first_diff"], -0.15, 0.05, ["observer", "continuity"], False),
            ("b_creep", "escalation", 0.7, 0.06, "差分の増殖", "同じ構図の中の差分が二つ三つと増え、左ページ末尾で主体が初めて振り返る", ["b_normalcy"], 0.1, 0.05, ["difference", "observer"], False),
            ("b_absence", "structural_twist", 0.8, 0.05, "同じ構図の中の不在", "二度目のめくり直後、最初と同じ構図の1ページ大ゴマに空いた椅子だけが残る", ["b_creep"], 0.4, 0.3, ["empty_object", "absent_one", "observer"], True),
            ("b_still", "afterglow", 1.0, 0.05, "無人の定点", "最後に人物のいない同じ構図を小さく置き、時間だけが流れ続けることを示す", ["b_absence"], -0.25, 0.0, ["fixed_frame", "observer"], False),
        ],
        "mechanisms": [
            ("m_habituation", "contrast_effect", "馴化と脱馴化", "同一構図の反復で知覚を慣らし、わずかな差分を強いコントラストとして浮かび上がらせる", "同じ画角のコマの連続", "小さな変化が大きな意味を帯びて見える", "馴化・脱馴化の知覚理論", 0.85, ["b_repeat", "b_first_diff"], ["fixed_frame", "difference"]),
            ("m_reader_first", "dramatic_irony", "読者だけが気づく差分", "主体が見落とす変化を読者だけが先に見つけ、知っているのに伝えられない緊張が続く", "主体の無自覚な会話と背景の差分", "静かなサスペンスと胸騒ぎ", "劇的アイロニー", 0.8, ["b_normalcy", "b_creep"], ["observer"]),
            ("m_absence_presence", "loss_aversion", "不在による存在の確認", "空いた椅子が、それまでの反復の中にいた存在の重さを逆算させる", "同じ構図の中の不在", "失ったものの大きさが遅れて押し寄せる", "損失回避（プロスペクト理論）", 0.75, ["b_absence", "b_still"], ["empty_object", "absent_one"]),
        ],
        "anchor": "当たり前に続くと信じていた日々の中から、大切な誰かが静かに消えていく喪失の予感",
        "drivers": ["同じ構図の中の間違い探し", "読者だけが知っている変化", "空いた椅子の1ページ"],
        "emotions": ["安らぎ", "既視感", "違和感", "胸騒ぎ", "焦燥", "喪失", "静寂"],
        "template": "{{OBSERVER}}の日々は{{FIXED_FRAME}}で反復され、{{CONTINUITY}}が当然のように続いている。めくり直後に{{SUBTLE_DIFFERENCE}}が読者にだけ示され、取り繕う日常の背後で差分は増殖する。二度目のめくり直後、最初と同じ構図の中に{{EMPTY_OBJECT}}だけが残り、{{ABSENT_ONE}}の不在が突きつけられる。",
        "slots": [
            ("OBSERVER", "定点の前で日々を過ごす主体", "agent", ["変化に気づくのは読者より遅らせる"], ("縁側で朝食をとる少年", "毎週同じ数値を確認する担当者", "毎年同じ観測を続ける研究者")),
            ("FIXED_FRAME", "反復される同一構図", "symbol", ["画角・配置・コマの大きさを変えない"], ("同じ縁側を同じ角度で描くコマ", "毎週同じ形式のレポート画面", "定点観測の写真群")),
            ("CONTINUITY", "変わらない日々の継続", "goal", ["言葉で語らず反復で示す"], ("家族そろった朝", "安定した継続率", "変わらない観測値")),
            ("SUBTLE_DIFFERENCE", "見落とされた差分", "lack", ["最初は一コマ・一点に限る"], ("一つ減った湯呑み", "わずかに下がった利用頻度", "観測値の小さなずれ")),
            ("EMPTY_OBJECT", "不在を体現する物", "symbol", ["最初の構図と完全に同じ画角で描く"], ("空いた椅子", "空欄になった顧客リスト", "途切れた観測記録")),
            ("ABSENT_ONE", "定点から消えていく存在", "agent", ["最後まで退場の場面を描かない"], ("祖母", "静かに離れていった常連顧客", "失われた生態系の一種")),
        ],
        "invariants": [
            "反復コマは画角・人物配置・コマの大きさを厳密に揃え、差分以外の変化を入れない",
            "最初の差分と最後の不在はいずれもめくり直後の右ページ冒頭に置く",
            "読者が主体より先に差分に気づく順序を崩さない",
            "最後の大ゴマは最初のコマと同じ構図で描き、視線を同じ位置から入らせる",
        ],
        "guide": {
            "manga": "6ページ構成。p1（右）: 同じ画角の4コマ（FS・アイレベル・50mm固定）で場所と人物を確立。p2（左）: 同構図4コマを反復し、最終コマの背景に微かな違和を置く。p3（右・めくり直後）: 3コマ、冒頭で欠けた物をCUで示し差分を気づかせる。p4（左）: 5コマのMS会話で取り繕う日常、背景の差分が増え、最終コマで主体が振り返る（CU）。p5（右・めくり直後）: 最初と同じ構図の1ページ大ゴマに空いた椅子だけを置く。p6（左）: 3コマ、無人の同構図を小さく置いて時間の経過を示す。",
            "marketing": "LPで顧客の日常業務を同じ形式のダッシュボード画面として繰り返し見せ、スクロールの区切り直後に小さな異常値だけを強調する。気づかないまま続く業務の描写を挟み、最後に同じ画面の中で大切な指標が空欄になった状態を見せて、継続的な観測の重要性と解決策へ導く。",
            "article": "同じ形式の観測記録を章ごとに反復提示し、ある章の冒頭でわずかな差分を指摘する。気づかれないまま進む事例を挟み、最後に同じ形式の記録の中で決定的な欠落を示して、定点観測と差分の読み取りの価値を論じる。",
        },
        "rhythm": {
            "page_count": 6,
            "panels_per_page": [4, 4, 3, 5, 1, 3],
            "emphasis_pages": [2, 4],
            "page_turn_reveals": [2, 4],
            "spread_pages": [],
            "silent_panels_ratio": 0.4,
            "pacing_notes": [
                "p1（右）: 同じ画角の4コマで場所と人物の配置を基準線として刻む",
                "p2（左）: 同構図の4コマを反復し、最終コマの背景に微かな違和を置く",
                "p3（右・めくり直後）: 3コマ、冒頭で欠けた物をCUで示し最初の差分を気づかせる",
                "p4（左）: 5コマの会話で取り繕う日常を描き、背景の差分を増やして最終コマで振り返らせる",
                "p5（右・めくり直後）: 最初と同じ構図の1ページ大ゴマに空いた椅子だけを置く",
                "p6（左）: 3コマ、人物のいない同構図を小さく置いて時間の流れを示す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 05: ぶち抜き登場と枠越えの威圧
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_05_breakout_entrance",
        "title": "ぶち抜き登場と枠越えの威圧",
        "summary": "群衆の恐慌と噂、近づく足音の小ゴマで正体を伏せたまま期待を溜め、静まり返る一瞬を左ページ末尾に置いてから、めくり直後に枠線を物理的に破って登場させ、続くページで複数のコマに跨がる姿で空間そのものを支配させる、枠越えによる威圧の骨格。",
        "tags": ["ぶち抜き", "登場", "威圧", "枠越え", "強敵"],
        "lineage": ["ぶち抜きゴマによる人物の存在感誇示の慣行", "正体を伏せた接近と全身見せの登場演出", "コマ枠を物理境界として扱うメタ表現"],
        "waves": {
            "tension": [(0.0, 0.35), (0.15, 0.5), (0.3, 0.44), (0.45, 0.62), (0.55, 0.56), (0.65, 0.96), (0.75, 0.78), (0.9, 0.9), (1.0, 0.74)],
            "release": [(0.0, 0.1), (0.3, 0.15), (0.55, 0.05), (0.65, 0.1), (0.75, 0.25), (0.9, 0.12), (1.0, 0.2)],
            "audience_knowledge": [(0.0, 0.1), (0.3, 0.2), (0.55, 0.3), (0.65, 0.75), (0.9, 0.82), (1.0, 0.85)],
            "character_knowledge": [(0.0, 0.3), (0.3, 0.4), (0.55, 0.5), (0.65, 0.7), (0.9, 0.75), (1.0, 0.8)],
            "cognitive_load": [(0.0, 0.6), (0.3, 0.5), (0.55, 0.35), (0.65, 0.25), (0.8, 0.45), (1.0, 0.4)],
        },
        "nodes": [
            ("intruder", "agent", "overwhelming_figure", "枠を破って現れる圧倒的存在", "正体を伏せたまま接近し、コマ枠そのものを破壊して登場する存在", 0.95, "OVERWHELMING_FIGURE"),
            ("crowd", "agent", "bystanders", "恐慌に陥る群衆", "噂で先に正体を知り、枠の内側という安全圏に逃げ込もうとする人々", 0.6, "CROWD"),
            ("witness", "agent", "protagonist", "逃げずに見届ける者", "群衆の中で一人だけ立ち止まり、威圧に静かに抗う主体", 0.75, "WITNESS"),
            ("dominance", "desire", "goal", "空間の支配", "侵入者が示そうとする、場とページそのものを掌握する力", 0.8, "DOMINANCE"),
            ("lost_safety", "lack", "broken_refuge", "失われる安全圏", "コマ枠の内側にあると信じられていた安全の喪失", 0.7, "LOST_SAFETY"),
            ("frame_border", "symbol", "panel_frame", "破られるコマ枠", "安全圏の境界であり、侵入者が物理的に破壊する対象", 0.85, "FRAME_BORDER"),
        ],
        "edges": [
            ("intruder", "crowd", "power_gradient", 0.9, 0.9, "圧倒的な力の差", 0.65),
            ("intruder", "dominance", "pursuit", 0.8, 0.0, "空間の支配を誇示する", 0.0),
            ("crowd", "lost_safety", "dependency", 0.7, -0.2, "枠の内側という安全圏に頼る", 0.0),
            ("frame_border", "lost_safety", "embodiment", 0.75, 0.0, "枠線が安全圏を体現する", 0.0),
            ("intruder", "frame_border", "opposition", 0.85, 0.0, "枠線を破壊して侵入する", 0.65),
            ("crowd", "intruder", "asymmetry", 0.6, 0.5, "群衆は噂で正体を先に知っている", 0.3),
            ("witness", "intruder", "opposition", 0.6, 0.0, "見届ける者の静かな抵抗", 0.9),
        ],
        "beats": [
            ("b_panic", "hook", 0.05, 0.08, "群衆の恐慌", "6コマの小ゴマで逃げ惑う群衆を描き、原因を見せずに異変だけを伝える", [], 0.2, 0.0, ["crowd"], False),
            ("b_rumor", "setup", 0.3, 0.08, "噂と影", "群衆の会話で正体の噂を断片的に語らせ、読者より人物の方が多くを知る状態を作る", ["b_panic"], -0.05, -0.05, ["crowd", "intruder"], False),
            ("b_footsteps", "escalation", 0.45, 0.06, "近づく足音", "足音の描き文字と揺れる小物の小コマで接近を示す", ["b_rumor"], 0.18, 0.05, ["intruder", "frame_border"], False),
            ("b_hush", "transition", 0.55, 0.05, "静まり返る一瞬", "左ページ最終コマを無音の小ゴマにし、全員が同じ方向を見る瞬間で止める", ["b_footsteps"], -0.06, 0.0, ["crowd", "witness"], True),
            ("b_breakout", "revelation", 0.65, 0.06, "枠をぶち抜く登場", "めくり直後、侵入者の全身がコマ枠を破って隣のコマへはみ出し、正体が一挙に開示される", ["b_hush"], 0.4, 0.35, ["intruder", "frame_border", "lost_safety"], True),
            ("b_overlap", "escalation", 0.75, 0.06, "複数コマに跨る威圧", "侵入者の体が3つのコマに跨り、群衆のコマを上から覆う", ["b_breakout"], -0.1, 0.05, ["intruder", "crowd", "dominance"], False),
            ("b_declaration", "climax", 0.9, 0.05, "支配の宣告", "ページを縦断する侵入者のアオリの大ゴマで、場の支配を言葉で宣告する", ["b_overlap"], 0.12, 0.05, ["intruder", "witness", "dominance"], True),
            ("b_dread", "payoff", 1.0, 0.05, "支配された空間", "見届ける者の小さなCUで、威圧に抗う意志だけを残して締める", ["b_declaration"], -0.16, 0.0, ["witness", "intruder"], False),
        ],
        "mechanisms": [
            ("m_withheld_entrance", "curiosity_gap", "伏せられた正体", "噂と足音だけで接近を描き、正体への問いを最大限に引き伸ばす", "影・足音・群衆の反応", "めくり直後の登場が強烈な答えになる", "情報ギャップ理論", 0.8, ["b_rumor", "b_footsteps"], ["intruder"]),
            ("m_boundary_violation", "prediction_error", "境界侵犯の衝撃", "読書の約束事であるコマ枠が物理的に破られ、読者の予測そのものが侵される", "枠線を破る全身のぶち抜き", "侵入者の力が読書空間にまで及ぶ実感", "スキーマ違反と予測誤差", 0.9, ["b_breakout", "b_overlap"], ["frame_border"]),
            ("m_authority_awe", "authority", "畏怖による支配", "複数コマを覆う巨大な姿とアオリの構図が、抗いがたい権威の感覚を生む", "コマを跨ぐ巨大な姿とアオリの大ゴマ", "読者が群衆と同じ畏怖を体感する", "権威と服従の心理", 0.7, ["b_declaration"], ["intruder", "crowd"]),
            ("m_lone_resistance", "empathy_transfer", "ただ一人の抵抗", "恐慌の中で一人だけ立ち止まる者に読者の共感が集まる", "群衆の中で動かない見届ける者", "次の展開への期待と応援", "共感の知覚‐行為モデル", 0.65, ["b_hush", "b_dread"], ["witness"]),
        ],
        "anchor": "誰もが逃げ出す圧倒的な存在の前で、それでも目を逸らさずに立ち尽くす者の恐怖と意志",
        "drivers": ["噂と足音だけで伏せられた正体", "枠線が物理的に破られる衝撃", "複数のコマを覆う巨大な姿"],
        "emotions": ["恐慌", "不安", "予感", "静止", "戦慄", "畏怖", "決意"],
        "template": "{{CROWD}}の恐慌と噂によって正体を伏せたまま{{OVERWHELMING_FIGURE}}が近づき、静まり返る一瞬の後、めくり直後に{{FRAME_BORDER}}を破って全身を現す。安全圏だったはずの枠は意味を失い、{{LOST_SAFETY}}が露わになる。侵入者は複数のコマに跨って{{DOMINANCE}}を宣告し、ただ一人{{WITNESS}}だけが目を逸らさずに立ち尽くす。",
        "slots": [
            ("CROWD", "恐慌に陥る群衆", "agent", ["原因を見せずに反応だけを描く"], ("逃げ惑う市場の人々", "動揺する既存顧客層", "混乱する専門家コミュニティ")),
            ("OVERWHELMING_FIGURE", "枠を破って現れる圧倒的存在", "agent", ["登場まで全身を見せない", "登場はめくり直後に置く"], ("伝説の強敵", "市場を塗り替える新興勢力", "通説を根底から覆す新理論")),
            ("FRAME_BORDER", "破られるコマ枠", "symbol", ["破壊は物理的な破片として描く"], ("砕け散る枠線", "業界の常識という境界", "学問分野の境界線")),
            ("LOST_SAFETY", "失われる安全圏", "lack", ["破壊の直後に示す"], ("守られていたはずの街", "安泰だった市場シェア", "揺るがないと信じられた定説")),
            ("DOMINANCE", "空間の支配", "goal", ["言葉の宣告は最後の大ゴマに限る"], ("この場の全てを支配するという宣言", "市場の再定義", "パラダイムの転換")),
            ("WITNESS", "逃げずに見届ける者", "agent", ["群衆の中で唯一動かない人物として配置する"], ("拳を握る主人公", "変化を機会と捉える経営者", "新理論を検証しようとする研究者")),
        ],
        "invariants": [
            "侵入者の全身はめくり直後の右ページまで一切見せず、影・足音・反応だけで接近を描く",
            "ぶち抜きは枠線を物理的に破る描写とし、はみ出しは読み順の次のコマへ向ける",
            "威圧のページでは侵入者が複数コマに跨っても、視線は右上の頭部から左下の群衆へ流す",
            "見届ける者は全編を通じて群衆と逆向きの視線で配置する",
        ],
        "guide": {
            "manga": "4ページ構成。p1（右）: 6コマの小ゴマで逃げ惑う群衆（俯瞰・広角、原因は画面外）。p2（左）: 5コマで噂の会話と足音の描き文字、揺れる小物のECU、最終コマは全員が同じ方向を見る無音のコマ（アイレベル）。p3（右・めくり直後）: 2コマ、侵入者の全身が上段の枠線を破って下段へはみ出すぶち抜き（アオリ・広角18mm）。p4（左）: 3コマ、侵入者の体が3コマに跨って群衆を覆い、最後に見届ける者のCU（望遠100mm）で抵抗の意志を残す。",
            "marketing": "LPで市場の混乱と噂を先に見せ、新しい解決策の正体を伏せたまま期待を溜める。区切り直後に、既存のレイアウト枠を破るデザインで解決策を大きく登場させ、続くセクションで複数の比較表にまたがるその優位を示し、変化を機会と捉える読者をCTAへ導く。",
            "article": "業界の混乱と噂を導入で描き、新理論の名前や中身を伏せて関心を高める。章の切れ目で既存の分類の枠に収まらない新理論として大きく提示し、複数の分野の論点にまたがるその影響力を示してから、検証の姿勢を持つ読者への提言で締める。",
        },
        "rhythm": {
            "page_count": 4,
            "panels_per_page": [6, 5, 2, 3],
            "emphasis_pages": [2, 3],
            "page_turn_reveals": [2],
            "spread_pages": [],
            "silent_panels_ratio": 0.2,
            "pacing_notes": [
                "p1（右）: 6コマの小ゴマで逃げ惑う群衆を描き、原因を画面外に置く",
                "p2（左）: 5コマで噂と足音を重ね、最終コマを全員が同じ方向を見る無音のコマにして引く",
                "p3（右・めくり直後）: 2コマ、侵入者の全身が枠線を破ってはみ出すぶち抜きで登場させる",
                "p4（左）: 3コマ、侵入者が3コマに跨って群衆を覆い、見届ける者のCUで締める",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 06: モノローグの連鎖と内面の深潜
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_06_monologue_deep_dive",
        "title": "モノローグの連鎖と内面の深潜",
        "summary": "外界の小さな出来事から独白が増殖し、めくり直後に独白の箱がコマを覆って記憶の断片へ潜行し、黒地に白文字の最深層の沈黙を経て核心の記憶へ浮上し、最後のめくり直後に1ページで核心の自覚を言語化させる、内面の深度をページの明度とテキスト量で設計する骨格。",
        "tags": ["モノローグ", "内面描写", "記憶", "自覚", "黒地"],
        "lineage": ["独白の箱（ナレーションボックス）の連鎖による内面描写", "意識の流れの手法", "黒地白文字による深層心理の表現慣行"],
        "waves": {
            "tension": [(0.0, 0.35), (0.15, 0.3), (0.3, 0.66), (0.4, 0.52), (0.55, 0.44), (0.7, 0.58), (0.85, 0.72), (1.0, 0.95)],
            "release": [(0.0, 0.2), (0.3, 0.1), (0.55, 0.3), (0.7, 0.15), (0.85, 0.1), (1.0, 0.45)],
            "audience_knowledge": [(0.0, 0.1), (0.3, 0.45), (0.55, 0.62), (0.85, 0.75), (0.95, 0.85), (1.0, 0.92)],
            "character_knowledge": [(0.0, 0.15), (0.3, 0.28), (0.55, 0.38), (0.85, 0.5), (0.95, 0.7), (1.0, 0.9)],
            "cognitive_load": [(0.0, 0.3), (0.3, 0.6), (0.45, 0.8), (0.55, 0.85), (0.7, 0.7), (0.85, 0.55), (1.0, 0.35)],
        },
        "nodes": [
            ("speaker", "agent", "protagonist", "独白する主体", "外界の出来事をきっかけに、自分自身へ向けて語り始める主体", 0.9, "NARRATING_SELF"),
            ("outer_world", "agent", "outer_world", "独白を引き出す外界", "小さな出来事や他者の一言として主体の内面を揺さぶる外側の世界", 0.55, "OUTER_TRIGGER"),
            ("self_understanding", "desire", "goal", "自分自身の理解", "主体が深潜の果てに求める、自分の行動の本当の理由", 0.85, "SELF_UNDERSTANDING"),
            ("buried_memory", "lack", "repressed_memory", "埋もれた核心の記憶", "主体自身にも隠されていた記憶。深潜の最深部で姿を現す", 0.8, "BURIED_MEMORY"),
            ("monologue_box", "symbol", "narration_box", "コマを覆う独白の箱", "テキスト量の増加として内面への沈降を可視化する装置", 0.75, "MONOLOGUE_BOX"),
            ("black_ground", "symbol", "deep_layer", "黒地の深層", "背景を黒く塗りつぶし白文字だけを置くことで最深部の沈黙を示す", 0.7, "BLACK_GROUND"),
        ],
        "edges": [
            ("speaker", "self_understanding", "pursuit", 0.85, 0.0, "自分の本当の理由を追う", 0.0),
            ("outer_world", "speaker", "power_gradient", 0.55, 0.4, "外界の出来事が独白を引き出す", 0.05),
            ("buried_memory", "speaker", "asymmetry", 0.75, 0.7, "記憶は本人にも隠されている", 0.3),
            ("monologue_box", "buried_memory", "embodiment", 0.7, 0.0, "増殖する独白が記憶への沈降を体現する", 0.3),
            ("black_ground", "buried_memory", "embodiment", 0.65, 0.0, "黒地が最深層の沈黙を体現する", 0.55),
            ("speaker", "outer_world", "opposition", 0.5, 0.0, "言語化した自己が外界に向き直る", 0.95),
            ("speaker", "monologue_box", "dependency", 0.6, 0.0, "独白に思考を預ける", 0.15),
        ],
        "beats": [
            ("b_surface", "hook", 0.05, 0.06, "外界の小さな出来事", "会話中心の5コマで日常の一場面を描き、主体の視線だけが一瞬止まるコマを置く", [], 0.05, 0.0, ["speaker", "outer_world"], False),
            ("b_first_mono", "setup", 0.15, 0.06, "最初の独白", "主体の横顔に小さな独白の箱が一つ現れ、左ページ末尾で箱が二つ三つと増えて引く", ["b_surface"], -0.05, 0.1, ["speaker", "monologue_box"], False),
            ("b_dive", "revelation", 0.3, 0.06, "記憶の断片への潜行", "めくり直後、独白の箱がコマを覆い尽くし、背景が記憶の断片に置き換わる", ["b_first_mono"], 0.3, 0.3, ["speaker", "monologue_box", "buried_memory"], True),
            ("b_abyss", "crisis", 0.55, 0.06, "最深層の沈黙", "黒地に白文字だけの2コマで、言葉にならない感情の底を描く", ["b_dive"], -0.15, 0.05, ["black_ground", "buried_memory"], True),
            ("b_memory_core", "escalation", 0.7, 0.06, "核心の記憶", "黒地の中に一コマだけ鮮明な記憶の場面が浮かび上がる", ["b_abyss"], 0.15, 0.05, ["buried_memory", "speaker"], False),
            ("b_ascent", "escalation", 0.85, 0.06, "浮上と言語化", "独白の箱が減り始め、背景が少しずつ白へ戻る中で主体が言葉を選び直す", ["b_memory_core"], 0.12, -0.05, ["speaker", "self_understanding"], False),
            ("b_realization", "climax", 1.0, 0.04, "核心の自覚", "最後のめくり直後の1ページで、主体が自分の本当の理由を一行の言葉にする", ["b_ascent"], 0.22, -0.2, ["speaker", "self_understanding", "outer_world"], True),
        ],
        "mechanisms": [
            ("m_interior_access", "dramatic_irony", "内面への特権的接近", "独白によって読者だけが主体の内面を知り、周囲の人物との認識差が生まれる", "コマを覆う独白の箱", "主体への深い同一化と、周囲との緊張", "劇的アイロニー", 0.75, ["b_dive", "b_memory_core"], ["speaker", "monologue_box"]),
            ("m_depth_by_luminance", "narrative_transportation", "明度による深度表現", "背景の明度とテキスト量を内面の深さに対応させ、読者を物理的に沈ませる", "白地から黒地への移行", "読者が主体の心の底まで同行する", "物語移入理論", 0.8, ["b_abyss"], ["black_ground"]),
            ("m_insight_closure", "zeigarnik", "未完の問いと自覚", "深潜の間ずっと開いていた問いが、最後の一行で閉じられる", "核心の自覚の1ページ", "強い納得と読後の余韻", "未完了効果（ツァイガルニク効果）", 0.7, ["b_first_mono", "b_realization"], ["self_understanding"]),
            ("m_identity_mirror", "identity_projection", "自己投影の鏡", "言語化の過程が読者自身の内省を誘い、主体の自覚を自分事として受け取らせる", "言葉を選び直す浮上の過程", "読者自身の記憶が呼び起こされる", "自己参照効果", 0.65, ["b_ascent"], ["speaker"]),
        ],
        "anchor": "自分でも気づいていなかった本当の理由に、言葉を重ねながら少しずつ近づいていく内省の痛みと解放",
        "drivers": ["増殖する独白の箱", "黒地に白文字の最深層", "最後の1ページの自覚"],
        "emotions": ["違和感", "内省", "郷愁", "沈潜", "痛み", "浮上", "自覚"],
        "template": "{{OUTER_TRIGGER}}をきっかけに{{NARRATING_SELF}}の{{MONOLOGUE_BOX}}が増殖し、めくり直後にコマを覆って記憶の断片へ潜行する。{{BLACK_GROUND}}の最深層で言葉は途切れ、{{BURIED_MEMORY}}が一コマだけ鮮明に浮かび上がる。浮上しながら言葉を選び直した主体は、最後のめくり直後に{{SELF_UNDERSTANDING}}を一行の言葉として自覚する。",
        "slots": [
            ("OUTER_TRIGGER", "独白を引き出す外界の出来事", "agent", ["ごく小さな出来事に留める"], ("友人の何気ない一言", "顧客からの短い問い合わせ", "読者からの素朴な質問")),
            ("NARRATING_SELF", "独白する主体", "agent", ["独白の間も表情を描き続ける"], ("迷いを抱えた主人公", "創業の原点を見失いかけた経営者", "研究の意義を問い直す筆者")),
            ("MONOLOGUE_BOX", "コマを覆う独白の箱", "symbol", ["テキスト量を深度に比例させる"], ("横顔を覆う四角い独白の箱", "顧客の心の声を並べたテキスト群", "自問自答の段落の連鎖")),
            ("BLACK_GROUND", "黒地の深層", "symbol", ["白文字だけを置き絵を描かない"], ("黒く塗りつぶした背景の2コマ", "暗転した背景に浮かぶ一文", "余白に置かれた短い一行")),
            ("BURIED_MEMORY", "埋もれた核心の記憶", "lack", ["一コマだけ鮮明に描く"], ("幼い日の約束の場面", "最初の顧客に救われた日", "研究を始めたきっかけの体験")),
            ("SELF_UNDERSTANDING", "自分自身の理解", "goal", ["一行の言葉で言い切る"], ("自分が戦う本当の理由", "事業の存在意義", "研究を続ける根本の動機")),
        ],
        "invariants": [
            "独白の箱の量と背景の暗さを内面の深度に比例させ、深潜と浮上を単調に設計する",
            "深潜の開始と核心の自覚はめくり直後の右ページに置く",
            "最深層の黒地のコマには絵を置かず、白文字の視線を右上から左下へ一直線に流す",
            "核心の記憶は一コマだけ鮮明に描き、他の記憶の断片と画調で区別する",
        ],
        "guide": {
            "manga": "7ページ構成。p1（右）: 5コマの会話（MS・アイレベル）で日常を描き、主体の視線が止まるCUを1コマ。p2（左）: 4コマで独白の箱を増やし、最終コマは箱に覆われ始めた横顔（CU）。p3（右・めくり直後）: 3コマ、独白の箱がコマを覆い背景が記憶の断片に置き換わる（ダッチ・望遠）。p4（左）: 黒地白文字の2コマで最深層の沈黙。p5（右）: 3コマ、黒地の中に一コマだけ鮮明な核心の記憶（FS・俯瞰）。p6（左）: 4コマで独白の箱を減らし背景を白へ戻す浮上、最終コマで顔を上げる。p7（右・めくり直後）: 1ページで核心の自覚の一行（ECU・アイレベル・85mm）。",
            "marketing": "LPで顧客の小さな違和感から始め、心の声を並べたテキストで課題の内面へ深く潜る。最も深い不安を余白に一文だけ置いて示し、そこから本当に求めていた価値を言語化していく流れで浮上させ、最後に一行の価値提案でCTAへつなぐ。",
            "article": "身近な問いから始め、自問自答の段落を重ねて論点を深めていく。最も根本的な問いを短い一行の段落として独立させ、その答えを核となる体験とともに言語化し、結論の一文で読者自身の内省を促して締める。",
        },
        "rhythm": {
            "page_count": 7,
            "panels_per_page": [5, 4, 3, 2, 3, 4, 1],
            "emphasis_pages": [2, 6],
            "page_turn_reveals": [2, 6],
            "spread_pages": [],
            "silent_panels_ratio": 0.25,
            "pacing_notes": [
                "p1（右）: 5コマの会話で日常を描き、主体の視線が止まるCUを1コマ挟む",
                "p2（左）: 4コマで独白の箱を増やし、最終コマで箱に覆われ始めた横顔を引きにする",
                "p3（右・めくり直後）: 3コマ、独白の箱がコマを覆い背景が記憶の断片に置き換わる",
                "p4（左）: 黒地白文字の2コマで最深層の沈黙を描く",
                "p5（右）: 3コマ、黒地の中に一コマだけ鮮明な核心の記憶を浮かび上がらせる",
                "p6（左）: 4コマで独白の箱を減らし背景を白へ戻し、最終コマで主体が顔を上げる",
                "p7（右・めくり直後）: 1ページで核心の自覚を一行の言葉にする",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 07: 無音の連続コマから一言の決め
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_07_silence_then_one_line",
        "title": "無音の連続コマから一言の決め",
        "summary": "台詞の応酬が飽和した直後に沈黙を落とし、セリフのない小ゴマを重ねて沈黙の濃度を上げ、左ページ末尾で息を止めさせてから、めくり直後の大ゴマに一言だけを置いて全ての緊張を受け止めさせる、台詞の量の落差で一言の重さを最大化する骨格。",
        "tags": ["沈黙", "無音コマ", "決め台詞", "間", "会話劇"],
        "lineage": ["無音コマの連続による間の演出", "台詞の飽和と沈黙の落差（緩急の対比）", "一言の決め台詞を大ゴマに置く慣行"],
        "waves": {
            "tension": [(0.0, 0.42), (0.15, 0.58), (0.25, 0.48), (0.4, 0.56), (0.55, 0.74), (0.65, 0.95), (0.75, 0.6), (0.9, 0.5), (1.0, 0.44)],
            "release": [(0.0, 0.05), (0.15, 0.04), (0.25, 0.15), (0.55, 0.05), (0.65, 0.2), (0.75, 0.7), (1.0, 0.78)],
            "audience_knowledge": [(0.0, 0.2), (0.25, 0.3), (0.55, 0.4), (0.65, 0.8), (0.85, 0.85), (1.0, 0.88)],
            "character_knowledge": [(0.0, 0.3), (0.25, 0.35), (0.55, 0.45), (0.65, 0.78), (0.85, 0.85), (1.0, 0.88)],
            "cognitive_load": [(0.0, 0.75), (0.15, 0.8), (0.25, 0.4), (0.45, 0.2), (0.6, 0.15), (0.75, 0.35), (1.0, 0.3)],
        },
        "nodes": [
            ("speaker", "agent", "protagonist", "一言を探す主体", "言葉を尽くしても届かなかった末に、最後の一言を選ぶ主体", 0.9, "SPEAKER"),
            ("listener", "agent", "counterpart", "沈黙で応じる相手", "言い争いの後に口を閉ざし、その沈黙の意味を読ませる相手", 0.8, "LISTENER"),
            ("connection", "desire", "goal", "通じ合うこと", "二人が本当に求めている、言葉を超えた理解", 0.85, "CONNECTION"),
            ("unsaid", "lack", "unspoken_feeling", "言えずにいる想い", "台詞の応酬の下に埋もれ、沈黙の中で膨らんでいく本音", 0.8, "UNSAID_FEELING"),
            ("silence", "symbol", "silent_panels", "セリフのない連続コマ", "沈黙の時間と濃度を可視化する無音の小ゴマ群", 0.75, "SILENT_PANELS"),
            ("one_line", "symbol", "decisive_line", "一言の決め台詞", "沈黙の全ての緊張を受け止める、大ゴマの中の一行", 0.9, "DECISIVE_LINE"),
        ],
        "edges": [
            ("speaker", "listener", "opposition", 0.7, 0.0, "言い争いの応酬", 0.05),
            ("speaker", "connection", "pursuit", 0.8, 0.0, "言葉を超えた理解を求める", 0.0),
            ("listener", "speaker", "asymmetry", 0.6, 0.4, "相手の沈黙の意味を主体は測りかねる", 0.3),
            ("unsaid", "speaker", "power_gradient", 0.7, 0.6, "言えずにいる想いが主体を押し潰す", 0.25),
            ("silence", "unsaid", "embodiment", 0.75, 0.0, "無音のコマが言えない想いを体現する", 0.25),
            ("one_line", "connection", "embodiment", 0.8, 0.0, "一言が通じ合いを体現する", 0.65),
            ("listener", "connection", "dependency", 0.55, 0.0, "相手も同じ一言を待っていた", 0.7),
        ],
        "beats": [
            ("b_argument", "hook", 0.05, 0.06, "台詞の応酬", "6コマに大小のフキダシを詰め込み、言葉が言葉を呼ぶ口論を描く", [], 0.15, 0.0, ["speaker", "listener"], False),
            ("b_outburst", "escalation", 0.15, 0.05, "言葉の飽和", "最も大きな叫びのフキダシで口論が頂点に達する", ["b_argument"], 0.12, 0.05, ["speaker", "listener"], True),
            ("b_silence_falls", "transition", 0.25, 0.05, "沈黙の始まり", "フキダシが突然消え、二人の間に何もない空間を置く", ["b_outburst"], -0.1, 0.0, ["silence", "listener"], True),
            ("b_thick_silence", "escalation", 0.55, 0.08, "沈黙の濃度", "手元・窓・時計などセリフのない小ゴマを重ね、左ページ末尾で主体が口を開きかけるコマで止める", ["b_silence_falls"], 0.18, 0.05, ["silence", "unsaid", "speaker"], False),
            ("b_one_line", "revelation", 0.65, 0.05, "一言の決め", "めくり直後の大ゴマに一言だけを置き、沈黙の全てをその一行に受け止めさせる", ["b_thick_silence"], 0.21, 0.3, ["one_line", "speaker", "unsaid"], True),
            ("b_reception", "payoff", 0.8, 0.06, "受け止める沈黙", "相手の表情の小さな変化を無音のCUで描き、一言が届いたことだけを示す", ["b_one_line"], -0.3, 0.05, ["listener", "connection"], False),
            ("b_echo", "afterglow", 1.0, 0.05, "言葉の残響", "引きの構図で二人の距離が縮まったことを示し、台詞なしで締める", ["b_reception"], -0.08, 0.0, ["speaker", "listener", "connection"], False),
        ],
        "mechanisms": [
            ("m_density_contrast", "contrast_effect", "台詞量の落差", "台詞の飽和から沈黙への急落が、最後の一言の重さを相対的に最大化する", "フキダシで埋まったコマと無音のコマの対比", "一行の言葉が大きな事件のように響く", "対比効果（コントラスト効果）", 0.85, ["b_argument", "b_one_line"], ["silence", "one_line"]),
            ("m_silence_projection", "identity_projection", "沈黙への投影", "セリフのない小ゴマは読者が自分の言葉で埋める余白になる", "手元・窓・時計の無音の小ゴマ", "読者自身の記憶や感情が沈黙に流れ込む", "自己参照効果", 0.7, ["b_thick_silence"], ["unsaid"]),
            ("m_breath_reward", "anticipation_reward", "息止めと解放", "左ページ末尾で口を開きかけたまま止めることで、めくり直後の一言への予期が最大化する", "口を開きかけたコマでのページ終端", "めくった瞬間の強い報酬感", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.75, ["b_thick_silence", "b_one_line"], ["speaker"]),
        ],
        "anchor": "言葉を尽くしても伝わらなかった後で、たった一言にすべてを託すしかない瞬間の切実さ",
        "drivers": ["口論から沈黙への急落", "セリフのない小ゴマの濃度", "めくり直後の一言"],
        "emotions": ["苛立ち", "激昂", "虚脱", "緊張", "決意", "解放", "安らぎ"],
        "template": "{{SPEAKER}}と{{LISTENER}}の口論は言葉の飽和に達し、突然の沈黙が落ちる。{{SILENT_PANELS}}が重なるほど{{UNSAID_FEELING}}は膨らみ、左ページ末尾で主体は口を開きかけたまま止まる。めくり直後の大ゴマに置かれた{{DECISIVE_LINE}}がすべての緊張を受け止め、言葉を超えた{{CONNECTION}}が静かに訪れる。",
        "slots": [
            ("SPEAKER", "一言を探す主体", "agent", ["沈黙の間は表情だけで心理を描く"], ("想いを伝えられない主人公", "提案を言い直す営業担当", "結論を一文で言い切る筆者")),
            ("LISTENER", "沈黙で応じる相手", "agent", ["沈黙の意味を台詞で説明しない"], ("口を閉ざした幼なじみ", "黙って考え込む顧客", "反論を控える読者")),
            ("SILENT_PANELS", "セリフのない連続コマ", "symbol", ["3コマ以上連続させる", "手元・小物・風景の寄りで構成する"], ("湯気の立つ湯呑みと時計の秒針", "考え込む顧客の手元の写真", "一行だけの段落の連続")),
            ("UNSAID_FEELING", "言えずにいる想い", "lack", ["台詞では一度も言わない"], ("本当は離れたくないという想い", "本当に必要としている価値", "読者が抱える言語化されていない疑問")),
            ("DECISIVE_LINE", "一言の決め台詞", "symbol", ["15文字以内に収める", "めくり直後の大ゴマに置く"], ("「ここにいて」", "「あなたのための仕組みです」", "「答えは、問いの中にある」")),
            ("CONNECTION", "通じ合うこと", "goal", ["結果を説明せず表情と距離で示す"], ("二人の距離が縮まる", "顧客の頷き", "読者の納得")),
        ],
        "invariants": [
            "沈黙は3コマ以上の無音の小ゴマで描き、フキダシを一切置かない",
            "一言はめくり直後の大ゴマに置き、15文字以内に収める",
            "沈黙のコマは視線を右上から左下へゆっくり流すため、ほぼ等しい大きさで並べる",
            "一言の後は台詞を増やさず、表情と構図だけで受容を示す",
        ],
        "guide": {
            "manga": "4ページ構成。p1（右）: 6コマに大小のフキダシを詰めた口論（MSの切り返し、最大の叫びはCU・アオリ）。p2（左）: フキダシのない5コマ（湯呑みECU・窓の外FS・時計ECU・相手の背中MS・口を開きかけた主体のCU）で沈黙を濃くして止める。p3（右・めくり直後）: 2コマ、上段の大ゴマに一言だけ（CU・アイレベル・85mm）、下段に相手の目のECU。p4（左）: 3コマ、無音の表情変化と引きの構図（FS・俯瞰）で距離が縮まったことを示す。",
            "marketing": "LPの前半で機能説明や比較を詰め込みすぎた従来の訴求を戯画的に見せ、急に余白の多いセクションへ切り替えて顧客の本音の沈黙を写真だけで表現する。区切り直後にたった一行の価値提案を大きく置き、その後は説明を足さず導入事例の表情写真でCTAへつなぐ。",
            "article": "前半で論点を細かく積み上げた議論の応酬を描き、突然短い一行の段落を重ねて思考を止める間を作る。章の切れ目の直後に核心の一文だけを独立させ、その後は補足を最小限にして読者自身に考えさせる余白で締める。",
        },
        "rhythm": {
            "page_count": 4,
            "panels_per_page": [6, 5, 2, 3],
            "emphasis_pages": [2],
            "page_turn_reveals": [2],
            "spread_pages": [],
            "silent_panels_ratio": 0.55,
            "pacing_notes": [
                "p1（右）: 6コマに大小のフキダシを詰め込み、口論を最大の叫びで頂点に到達させる",
                "p2（左）: フキダシのない5コマで沈黙を濃くし、口を開きかけた主体のCUで止める",
                "p3（右・めくり直後）: 上段の大ゴマに一言だけを置き、下段で相手の目を受ける",
                "p4（左）: 3コマの無音の表情変化と引きの構図で、言葉が届いたことを示す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 08: 斜めゴマの加速とアクション連鎖
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_08_slanted_acceleration",
        "title": "斜めゴマの加速とアクション連鎖",
        "summary": "水平な構えのページから斜めゴマの初撃へ入り、反撃・連撃・紙一重の回避と、ページごとにガターの傾きとコマの細さを増して体感速度を上げ、三度目のめくり直後の見開きで決定打を刻んでから水平なコマへ戻して呼吸を回復させる、傾きを速度計として使うアクション連鎖の骨格。",
        "tags": ["斜めゴマ", "アクション", "加速", "連撃", "見開き"],
        "lineage": ["斜めガターによる動勢表現の慣行", "コマの細分化による時間の加速（ショット長の短縮）", "決定打の見開きと水平コマへの帰還"],
        "waves": {
            "tension": [(0.0, 0.3), (0.1, 0.42), (0.2, 0.66), (0.3, 0.52), (0.45, 0.8), (0.55, 0.64), (0.7, 0.98), (0.8, 0.7), (0.9, 0.45), (1.0, 0.34)],
            "release": [(0.0, 0.1), (0.2, 0.05), (0.3, 0.2), (0.45, 0.06), (0.55, 0.18), (0.7, 0.05), (0.8, 0.5), (1.0, 0.8)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.3), (0.45, 0.45), (0.55, 0.5), (0.7, 0.75), (0.9, 0.85), (1.0, 0.88)],
            "character_knowledge": [(0.0, 0.25), (0.2, 0.32), (0.45, 0.5), (0.55, 0.55), (0.7, 0.72), (0.9, 0.85), (1.0, 0.88)],
            "cognitive_load": [(0.0, 0.3), (0.2, 0.45), (0.45, 0.6), (0.55, 0.5), (0.7, 0.35), (0.85, 0.3), (1.0, 0.25)],
        },
        "nodes": [
            ("fighter", "agent", "protagonist", "加速する主体", "連撃の中で相手の隙を探り、傾きを増すコマとともに速度を上げていく主体", 0.9, "FIGHTER"),
            ("opponent", "agent", "antagonist", "互角に渡り合う相手", "反撃と回避で主体の連撃を受け止め、決定打の直前まで均衡を保つ相手", 0.85, "OPPONENT"),
            ("victory", "desire", "goal", "決定打による勝利", "連鎖の果てに主体が掴もうとする一瞬の勝機", 0.85, "VICTORY"),
            ("opening", "lack", "hidden_gap", "一瞬の隙", "相手自身も気づいていない防御の綻び。連撃の果てに露わになる", 0.75, "OPENING"),
            ("slant_gutter", "symbol", "slanted_gutter", "傾いたガター", "傾きの角度で体感速度を可視化する装置", 0.8, "SLANTED_GUTTER"),
            ("speed_lines", "symbol", "speed_lines", "集中線と速度線", "決定打の瞬間に視線を一点へ射抜く線の群れ", 0.65, "SPEED_LINES"),
        ],
        "edges": [
            ("fighter", "opponent", "opposition", 0.9, 0.0, "互角の打ち合い", 0.2),
            ("fighter", "victory", "pursuit", 0.85, 0.0, "勝機を掴みに行く", 0.0),
            ("opponent", "fighter", "power_gradient", 0.7, 0.5, "反撃で主導権を握り返す", 0.3),
            ("fighter", "opening", "pursuit", 0.7, 0.0, "連撃の中で隙を探る", 0.45),
            ("slant_gutter", "victory", "embodiment", 0.6, 0.0, "傾きが勝機への加速を体現する", 0.45),
            ("speed_lines", "opening", "embodiment", 0.55, 0.0, "集中線が一瞬の隙を射抜く", 0.7),
            ("opponent", "opening", "asymmetry", 0.6, -0.5, "相手は自らの隙に気づいていない", 0.55),
        ],
        "beats": [
            ("b_stance", "hook", 0.05, 0.06, "構えと間合い", "水平なガターの5コマで構えと距離を測り、速度の基準線を作る", [], 0.12, 0.0, ["fighter", "opponent"], False),
            ("b_clash", "escalation", 0.2, 0.06, "斜めゴマの初撃", "めくり直後、ガターが初めて傾き、初撃の交錯を斜めの4コマで描く", ["b_stance"], 0.25, 0.05, ["fighter", "opponent", "slant_gutter"], True),
            ("b_counter", "crisis", 0.3, 0.05, "反撃と体勢の崩れ", "相手の反撃で主体が体勢を崩し、傾きが逆向きになる", ["b_clash"], -0.14, 0.0, ["opponent", "fighter"], True),
            ("b_chain", "escalation", 0.45, 0.06, "加速する連撃", "二度目のめくり直後、細い斜めゴマ6コマで連撃を刻み、体感速度を最大まで上げる", ["b_counter"], 0.28, 0.05, ["fighter", "slant_gutter", "opening"], True),
            ("b_near_miss", "crisis", 0.55, 0.05, "紙一重の回避", "相手の一撃を紙一重で躱す3コマで息を止め、左ページ末尾で隙を見つけた目のECUで引く", ["b_chain"], -0.16, 0.05, ["fighter", "opponent", "opening"], True),
            ("b_strike", "climax", 0.7, 0.06, "見開きの決定打", "三度目のめくり直後の見開きで、集中線が隙を射抜く決定打を全景で刻む", ["b_near_miss"], 0.34, 0.15, ["fighter", "opponent", "speed_lines", "victory"], True),
            ("b_aftermath", "payoff", 0.85, 0.06, "余波と静止", "水平なガターに戻した4コマで余波を描き、速度の急停止を体感させる", ["b_strike"], -0.3, 0.05, ["fighter", "opponent"], False),
            ("b_close", "afterglow", 1.0, 0.05, "呼吸の回復", "水平な3コマの会話で呼吸を取り戻し、次の局面への余韻を残す", ["b_aftermath"], -0.12, 0.0, ["fighter", "victory"], False),
        ],
        "mechanisms": [
            ("m_slant_velocity", "narrative_transportation", "傾きによる体感速度", "ガターの傾きとコマの細さがページごとに増し、読者の視線そのものが加速する", "傾きを増す斜めガター", "アクションの速度を読書のリズムとして体験する", "物語移入理論（身体化された読書）", 0.85, ["b_clash", "b_chain"], ["slant_gutter"]),
            ("m_rhythm_reward", "anticipation_reward", "連鎖のリズムと予期", "打ち合いと回避の往復が規則的なリズムを作り、決定打への予期が段階的に高まる", "交互に訪れる攻勢と劣勢", "決定打の見開きが強い報酬として届く", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.8, ["b_counter", "b_near_miss", "b_strike"], ["fighter", "opponent"]),
            ("m_sudden_stop", "contrast_effect", "急停止の対比", "最大速度の見開きの直後に水平なコマへ戻すことで、静止の重さが際立つ", "斜めから水平への切替", "決着の重みと余韻が強く残る", "対比効果（コントラスト効果）", 0.7, ["b_aftermath"], ["slant_gutter"]),
        ],
        "anchor": "互角の相手との極限の打ち合いの中で、一瞬の隙を見逃さずに勝機を掴み取る集中",
        "drivers": ["ページごとに傾きを増すガター", "攻勢と劣勢の規則的な往復", "隙を射抜く見開きの決定打"],
        "emotions": ["集中", "高揚", "焦り", "加速", "息止め", "爆発", "静寂"],
        "template": "{{FIGHTER}}は{{OPPONENT}}と水平な構えで向き合い、めくり直後に{{SLANTED_GUTTER}}が傾いて初撃が交錯する。反撃と連撃、紙一重の回避を重ねるたびに傾きは増し、ついに{{OPENING}}が露わになる。三度目のめくり直後の見開きで{{SPEED_LINES}}が隙を射抜き、{{VICTORY}}が刻まれた後、水平なコマで呼吸が戻る。",
        "slots": [
            ("FIGHTER", "加速する主体", "agent", ["構えのページでは水平な構図に置く"], ("剣を構える主人公", "競合と競り合う挑戦者企業", "反論と応酬を重ねる論者")),
            ("OPPONENT", "互角に渡り合う相手", "agent", ["決定打の直前まで均衡を保つ"], ("宿命のライバル", "市場の首位企業", "強固な対立仮説")),
            ("SLANTED_GUTTER", "傾いたガター", "symbol", ["傾きはページごとに単調に増やす"], ("10度から30度へ傾きを増すガター", "斜めに区切られた比較セクション", "段階的に短くなる段落")),
            ("OPENING", "一瞬の隙", "lack", ["左ページ末尾の目のECUで初めて示す"], ("構えの綻び", "首位企業の見落とした顧客層", "対立仮説の説明できない例外")),
            ("SPEED_LINES", "集中線と速度線", "symbol", ["決定打の見開きで一点に収束させる"], ("隙へ収束する集中線", "一点に集約されるデータの矢印", "一つの結論へ収束する論点図")),
            ("VICTORY", "決定打による勝利", "goal", ["決定打の後は水平なコマで静止させる"], ("一閃の勝利", "シェアの逆転", "論争の決着")),
        ],
        "invariants": [
            "ガターの傾きとコマの細さはページごとに単調に増やし、決定打で最大にする",
            "決定打は三度目のめくり直後の見開きに置き、集中線を隙の一点へ収束させる",
            "傾いたコマでも視線は右上から左下へ流れる向きに傾けて逆行させない",
            "決定打の直後は水平なガターへ戻して速度の急停止を示す",
        ],
        "guide": {
            "manga": "10ページ構成。p1（右）: 水平な5コマで構えと間合い（FS・アイレベル）。p2（左）: 4コマ、最終コマで踏み込む足元を引く。p3（右・めくり直後）: 傾き10度の斜めゴマ4コマで初撃の交錯（MS・広角24mm）。p4（左）: 5コマで反撃と体勢の崩れ（ダッチ・俯瞰）。p5（右・めくり直後）: 傾き20度の細い斜めゴマ6コマで連撃（CU連打・アオリ）。p6（左）: 3コマで紙一重の回避と隙を見つけた目のECU。p7〜p8（めくり直後の見開き）: 傾き30度の構図に集中線を収束させた決定打（ELS・18mm）。p9（右）: 水平な4コマで余波。p10（左）: 水平な3コマの会話で呼吸を戻す。",
            "marketing": "LPで競合との比較を段階的に加速させ、セクションごとに斜めの区切りと短い見出しで読むテンポを上げる。最も白熱した比較の直後に、顧客の隠れたニーズ（隙）を一点で示し、全幅のビジュアルで決定的な優位を提示してから、落ち着いた水平レイアウトで申込みへ導く。",
            "article": "論争を応酬の形式で段階的に加速させ、段落を短くして論点の往復を速める。対立仮説の見落としを一点で突く決定的な論証を章の冒頭に大きく置き、その後は落ち着いた長い段落で結論の意味を整理する。",
        },
        "rhythm": {
            "page_count": 10,
            "panels_per_page": [5, 4, 4, 5, 6, 3, 1, 1, 4, 3],
            "emphasis_pages": [2, 4, 6],
            "page_turn_reveals": [2, 4, 6],
            "spread_pages": [6],
            "silent_panels_ratio": 0.2,
            "pacing_notes": [
                "p1（右）: 水平な5コマで構えと間合いを測り、速度の基準線を作る",
                "p2（左）: 4コマで間合いを詰め、最終コマで踏み込む足元を引きにする",
                "p3（右・めくり直後）: 傾き10度の斜めゴマ4コマで初撃の交錯を描く",
                "p4（左）: 5コマで反撃と体勢の崩れを逆向きの傾きで描く",
                "p5（右・めくり直後）: 傾き20度の細い斜めゴマ6コマで連撃を刻む",
                "p6（左）: 3コマで紙一重の回避と隙を見つけた目のECUで引く",
                "p7（右・めくり直後）: 決定打の見開きの右半分。集中線が隙へ収束し始める",
                "p8（左）: 決定打の見開きの左半分。射抜かれた隙と着弾の全景",
                "p9（右）: 水平な4コマで余波を描き速度を急停止させる",
                "p10（左）: 水平な3コマの会話で呼吸を回復させる",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 09: 小ゴマ連打のタイムラプス（時間の圧縮）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_09_timelapse_compression",
        "title": "小ゴマ連打のタイムラプス（時間の圧縮）",
        "summary": "誓いの一言で目標を固定し、季節や日付の記号を添えた12コマの小ゴマで長い時間を一気に圧縮し、めくり直後の1ページで時間を跳んだ結果を一枚絵として突きつけ、周囲の反応と新たな壁を挟んで最後のめくり直後に成長の証明を描く、コマ数と時間量を反比例させる骨格。",
        "tags": ["タイムラプス", "時間圧縮", "小ゴマ", "成長", "モンタージュ"],
        "lineage": ["モンタージュによる時間圧縮の映画技法", "小ゴマの連打による経過時間の表現慣行", "誓いと証明の成長譚構造"],
        "waves": {
            "tension": [(0.0, 0.35), (0.1, 0.48), (0.2, 0.3), (0.35, 0.42), (0.5, 0.82), (0.6, 0.55), (0.75, 0.6), (0.85, 0.72), (0.95, 0.96), (1.0, 0.82)],
            "release": [(0.0, 0.1), (0.1, 0.05), (0.2, 0.35), (0.35, 0.4), (0.5, 0.55), (0.6, 0.3), (0.85, 0.1), (1.0, 0.4)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.2), (0.35, 0.35), (0.5, 0.72), (0.75, 0.78), (0.95, 0.9), (1.0, 0.92)],
            "character_knowledge": [(0.0, 0.15), (0.2, 0.3), (0.35, 0.5), (0.5, 0.68), (0.75, 0.75), (0.95, 0.85), (1.0, 0.92)],
            "cognitive_load": [(0.0, 0.3), (0.2, 0.85), (0.35, 0.9), (0.5, 0.3), (0.75, 0.45), (0.95, 0.35), (1.0, 0.3)],
        },
        "nodes": [
            ("trainee", "agent", "protagonist", "時間を積み重ねる主体", "誓いを胸に、描かれない長い時間を一コマずつ積み上げていく主体", 0.9, "TRAINEE"),
            ("watcher", "agent", "mentor", "見守り試す者", "主体の積み重ねを見守り、最後に試験として立ちはだかる存在", 0.7, "WATCHER"),
            ("mastery", "desire", "goal", "到達すべき境地", "誓いの一言で固定され、時間の圧縮を通じて目指される成長", 0.85, "MASTERY"),
            ("elapsed_time", "lack", "invisible_time", "描かれない時間", "小ゴマの間に省略された膨大な日々。読者は断片からその量を推し量る", 0.75, "ELAPSED_TIME"),
            ("small_panels", "symbol", "panel_grid", "小ゴマの連打", "ページを細かく分割したコマの格子。コマ数が時間量を表す", 0.8, "SMALL_PANEL_GRID"),
            ("time_marker", "symbol", "season_sign", "季節と日付の記号", "桜・雪・カレンダーなど、コマごとに時間の経過を刻む記号", 0.6, "TIME_MARKER"),
        ],
        "edges": [
            ("trainee", "mastery", "pursuit", 0.9, 0.0, "境地への到達を目指す", 0.0),
            ("watcher", "trainee", "power_gradient", 0.6, 0.5, "見守る者が成長を測る", 0.1),
            ("trainee", "watcher", "dependency", 0.65, 0.0, "導きを支えに積み重ねる", 0.1),
            ("elapsed_time", "trainee", "asymmetry", 0.6, 0.4, "圧縮された時間の中身は読者に一部しか見せない", 0.35),
            ("small_panels", "elapsed_time", "embodiment", 0.75, 0.0, "小ゴマの数が時間量を体現する", 0.3),
            ("time_marker", "elapsed_time", "embodiment", 0.6, 0.0, "季節の記号が経過を刻む", 0.35),
            ("watcher", "mastery", "opposition", 0.55, 0.0, "最終試験として立ちはだかる", 0.85),
        ],
        "beats": [
            ("b_vow", "hook", 0.1, 0.06, "誓いの一言", "4コマの最後に主体の誓いを大きめのフキダシで置き、目標を固定する", [], 0.13, 0.0, ["trainee", "mastery"], True),
            ("b_compress", "setup", 0.2, 0.05, "時間圧縮の開始", "左ページを12の小ゴマに分割し、最初のコマに季節の記号を置く", ["b_vow"], -0.18, 0.05, ["small_panels", "time_marker"], True),
            ("b_montage", "escalation", 0.35, 0.08, "小ゴマ12連の積層", "同じ動作を季節だけ変えて反復し、失敗と継続の断片で膨大な時間を圧縮する", ["b_compress"], 0.12, 0.15, ["trainee", "small_panels", "elapsed_time"], False),
            ("b_result", "revelation", 0.5, 0.06, "時間を跳んだ結果の一枚絵", "めくり直後の1ページで、変貌した主体の姿を一枚絵として突きつける", ["b_montage"], 0.4, 0.3, ["trainee", "elapsed_time", "mastery"], True),
            ("b_reactions", "crisis", 0.6, 0.06, "周囲の反応と新たな壁", "周囲の驚きの小コマを重ねた後、まだ届かない領域を見守る者が示す", ["b_result"], -0.27, -0.05, ["watcher", "trainee"], True),
            ("b_challenge", "escalation", 0.85, 0.06, "最終試験の幕開け", "見守る者が試験として立ちはだかり、左ページ末尾で構えた二人を引く", ["b_reactions"], 0.15, 0.05, ["watcher", "trainee", "mastery"], False),
            ("b_final_test", "climax", 0.95, 0.04, "成長の証明", "最後のめくり直後の大ゴマで、圧縮された時間のすべてが一つの動作に結晶する", ["b_challenge"], 0.25, 0.1, ["trainee", "watcher", "mastery"], True),
            ("b_after", "payoff", 1.0, 0.03, "次なる時間へ", "見守る者の小さな頷きで、新しい時間の始まりを示す", ["b_final_test"], -0.14, 0.0, ["watcher", "trainee"], False),
        ],
        "mechanisms": [
            ("m_time_density", "curiosity_gap", "省略された時間への想像", "小ゴマの間に省略された日々を読者が想像で補い、時間の重みを自ら構築する", "季節の記号を添えた小ゴマの連打", "結果の一枚絵に読者自身の想像した努力が重なる", "情報ギャップ理論", 0.75, ["b_montage"], ["elapsed_time", "small_panels"]),
            ("m_leap_contrast", "contrast_effect", "跳躍の対比", "細かな小ゴマの格子から1ページ大ゴマへの急変が、成長の大きさを空間の差として見せる", "12コマから1コマへの切替", "変貌の衝撃が視覚的に増幅される", "対比効果（コントラスト効果）", 0.85, ["b_result"], ["small_panels"]),
            ("m_commitment", "commitment_consistency", "誓いの一貫性", "冒頭の誓いの一言が積み重ねの全てを方向づけ、最終試験で回収される", "冒頭の誓いと最後の証明", "一貫した意志への深い納得", "コミットメントと一貫性の原理", 0.7, ["b_vow", "b_final_test"], ["trainee", "mastery"]),
            ("m_efficacy", "self_efficacy", "積み重ねの効力感", "小さな反復が確かな成長に結びつく過程が、読者自身の努力への効力感を呼び起こす", "反復の小ゴマと結果の一枚絵", "自分も積み重ねれば届くという実感", "自己効力感理論", 0.65, ["b_montage", "b_result"], ["trainee"]),
        ],
        "anchor": "結果の出ない日々を誰にも見られずに積み重ね続ける孤独と、それが報われる瞬間への渇望",
        "drivers": ["季節が移ろう小ゴマの連打", "時間を跳んだ結果の一枚絵", "誓いを回収する最終試験"],
        "emotions": ["決意", "忍耐", "単調", "驚嘆", "不安", "緊迫", "達成"],
        "template": "{{TRAINEE}}は{{MASTERY}}への誓いを立て、{{SMALL_PANEL_GRID}}に{{TIME_MARKER}}を刻みながら{{ELAPSED_TIME}}を一気に圧縮していく。めくり直後の1ページで時間を跳んだ結果が突きつけられ、周囲が驚く中で{{WATCHER}}は新たな壁を示す。最後のめくり直後、圧縮されたすべての時間が一つの動作に結晶し、誓いは証明される。",
        "slots": [
            ("TRAINEE", "時間を積み重ねる主体", "agent", ["小ゴマの中では同じ構図で反復させる"], ("修行に打ち込む主人公", "改善を続けるチーム", "長期観測を続ける研究者")),
            ("MASTERY", "到達すべき境地", "goal", ["冒頭の誓いの言葉で固定する"], ("師を超える技", "業界最高水準の品質", "仮説を実証できる精度")),
            ("SMALL_PANEL_GRID", "小ゴマの連打", "symbol", ["1ページに10コマ以上を等分配置する"], ("12分割の小ゴマ", "月次の改善記録を並べた格子", "観測データの小さな表の連続")),
            ("TIME_MARKER", "季節と日付の記号", "symbol", ["各コマの同じ位置に置く"], ("桜・入道雲・紅葉・雪", "四半期ごとのグラフ", "観測年の表記")),
            ("ELAPSED_TIME", "描かれない時間", "lack", ["失敗の断片を必ず混ぜる"], ("三年の修行", "二年間の改善サイクル", "十年の観測期間")),
            ("WATCHER", "見守り試す者", "agent", ["最終試験まで実力を見せない"], ("無言の師匠", "厳しい監査役の顧客", "査読者")),
        ],
        "invariants": [
            "時間の圧縮は1ページ10コマ以上の等分割の小ゴマで行い、各コマの同じ位置に時間の記号を置く",
            "時間を跳んだ結果はめくり直後の1ページ大ゴマで見せる",
            "小ゴマの格子の視線誘導は右上から左下へ規則的に流し、行ごとに季節を進める",
            "冒頭の誓いの言葉と最後の証明の動作を対応させて回収する",
        ],
        "guide": {
            "manga": "5ページ構成。p1（右）: 4コマで出来事と誓いの一言（最後のコマは誓う主体のCU・アオリ）。p2（左）: 12分割の小ゴマで修行の反復、各コマ右上に季節の記号（FS固定・アイレベル）、最終コマで夜明けの空を引きにする。p3（右・めくり直後）: 1ページ大ゴマで変貌した主体の全身（FS・アオリ・広角24mm）。p4（左）: 5コマで周囲の驚き（CU連打）と見守る者が示す新たな壁、最終コマで構えた二人を引く。p5（右・めくり直後）: 3コマ、上段の大ゴマで最終試験の一撃（ELS）、下段で見守る者の小さな頷き。",
            "marketing": "LPでブランドの約束（誓い）を冒頭に置き、継続的な改善の記録を小さなカードの格子で一気に見せて時間の積み重ねを伝える。区切り直後に改善の結果を全幅のビフォーアフターで提示し、顧客の驚きの声と残る課題を示した上で、最終的な品質保証で申込みへ導く。",
            "article": "研究や取り組みの出発点となる誓いを冒頭に示し、長期にわたる試行錯誤を小さな表や箇条書きの連続で圧縮して提示する。章の冒頭で積み重ねの成果を一つの図で示し、残された課題と最終的な検証の結果で締めくくる。",
        },
        "rhythm": {
            "page_count": 5,
            "panels_per_page": [4, 12, 1, 5, 3],
            "emphasis_pages": [2, 4],
            "page_turn_reveals": [2, 4],
            "spread_pages": [],
            "silent_panels_ratio": 0.45,
            "pacing_notes": [
                "p1（右）: 4コマで出来事を描き、最後のコマに誓いの一言を置く",
                "p2（左）: 12分割の小ゴマで修行の反復を季節の記号とともに圧縮し、最終コマで夜明けを引く",
                "p3（右・めくり直後）: 1ページ大ゴマで時間を跳んだ結果の全身を突きつける",
                "p4（左）: 5コマで周囲の驚きと新たな壁を示し、構えた二人で引く",
                "p5（右・めくり直後）: 3コマ、上段の大ゴマで成長の証明、下段で見守る者の頷きを置く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 10: 視線誘導の意図的な撹乱（混乱の演出）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "md_10_gaze_disruption",
        "title": "視線誘導の意図的な撹乱（混乱の演出）",
        "summary": "整然とした読み順のページに小さな違和を混ぜ、めくり直後に重なり合うコマで読み順そのものを撹乱し、次のページで散乱したコマによって読者の視線を迷子にした後、二度目のめくり直後に一点を指す大ゴマの声で視線と現実を取り戻させる、混乱を読書体験そのものとして再現する骨格。",
        "tags": ["視線誘導", "撹乱", "混乱", "幻覚", "回復"],
        "lineage": ["読み順の意図的な破壊による混乱表現", "主観的知覚の歪みを画面構成で示す技法", "アンカーとなる大ゴマによる視線の回復"],
        "waves": {
            "tension": [(0.0, 0.3), (0.15, 0.4), (0.3, 0.5), (0.45, 0.75), (0.5, 0.7), (0.6, 0.97), (0.7, 0.6), (0.8, 0.72), (0.9, 0.45), (1.0, 0.35)],
            "release": [(0.0, 0.15), (0.3, 0.1), (0.45, 0.05), (0.6, 0.02), (0.7, 0.2), (0.8, 0.45), (1.0, 0.75)],
            "audience_knowledge": [(0.0, 0.2), (0.3, 0.25), (0.45, 0.2), (0.6, 0.15), (0.8, 0.7), (0.9, 0.8), (1.0, 0.85)],
            "character_knowledge": [(0.0, 0.3), (0.3, 0.28), (0.45, 0.22), (0.6, 0.2), (0.8, 0.6), (0.9, 0.78), (1.0, 0.85)],
            "cognitive_load": [(0.0, 0.3), (0.3, 0.5), (0.45, 0.8), (0.6, 0.95), (0.7, 0.7), (0.8, 0.4), (1.0, 0.25)],
        },
        "nodes": [
            ("lost_one", "agent", "protagonist", "知覚の迷子になる主体", "幻覚や攻撃によって現実の手がかりを失い、読む順序ごと混乱していく主体", 0.9, "DISORIENTED_SELF"),
            ("chaos_source", "agent", "antagonist", "混乱を生む力", "主体の知覚を歪める幻覚・術・恐慌など、読み順を破壊する力", 0.8, "CHAOS_SOURCE"),
            ("guiding_voice", "agent", "anchor_figure", "正しい方向を示す声", "混乱の外側から主体に呼びかけ、視線の錨となる存在", 0.7, "GUIDING_VOICE"),
            ("clarity", "desire", "goal", "現実の把握", "主体が取り戻そうとする、順序立てて世界を理解できる状態", 0.85, "CLARITY"),
            ("lost_ground", "lack", "lost_reference", "立脚点の喪失", "何が現実で何が幻なのかを判断するための基準の欠落", 0.75, "LOST_GROUND"),
            ("reading_order", "symbol", "reading_order", "読み順", "右上から左下へ流れるはずの読書の約束事。撹乱の対象となる", 0.8, "READING_ORDER"),
            ("anchor_panel", "symbol", "anchor_panel", "一点を指す大ゴマ", "混乱の後に視線を一か所へ集め、現実へ引き戻す大ゴマ", 0.75, "ANCHOR_PANEL"),
        ],
        "edges": [
            ("chaos_source", "lost_one", "power_gradient", 0.85, 0.8, "混乱が主体の知覚を圧倒する", 0.45),
            ("lost_one", "clarity", "pursuit", 0.85, 0.0, "現実の把握を求める", 0.0),
            ("chaos_source", "reading_order", "opposition", 0.8, 0.0, "読み順そのものを破壊する", 0.45),
            ("lost_one", "reading_order", "dependency", 0.7, 0.0, "読み順に身を委ねている", 0.0),
            ("reading_order", "lost_one", "betrayal", 0.65, 0.0, "信じた読み順が主体を迷わせる", 0.45),
            ("guiding_voice", "lost_one", "asymmetry", 0.6, 0.6, "声だけが正しい方向を知っている", 0.8),
            ("anchor_panel", "clarity", "embodiment", 0.75, 0.0, "大ゴマが現実の把握を体現する", 0.8),
            ("lost_ground", "chaos_source", "embodiment", 0.6, 0.0, "立脚点の喪失が混乱として現れる", 0.45),
        ],
        "beats": [
            ("b_order", "hook", 0.05, 0.06, "整然とした読み順", "教科書的な右上から左下への5コマで、読書の約束事を基準線として示す", [], 0.05, 0.0, ["lost_one", "reading_order"], False),
            ("b_glitch", "setup", 0.3, 0.08, "小さな違和", "6コマのうち2コマの大きさと位置をわずかにずらし、左ページ末尾で同じ顔が二度現れて引く", ["b_order"], 0.1, -0.05, ["chaos_source", "reading_order"], False),
            ("b_disorder", "structural_twist", 0.45, 0.06, "読み順の撹乱", "めくり直後、7コマが重なり合い、どれから読むべきか分からない配置で混乱を体験させる", ["b_glitch"], 0.25, -0.1, ["chaos_source", "reading_order", "lost_one"], True),
            ("b_confusion", "crisis", 0.6, 0.06, "視線の迷子と最大の混乱", "8コマが散乱し、逆さのコマや同じ場面の反復で読者の視線を迷子にする", ["b_disorder"], 0.25, -0.05, ["lost_one", "lost_ground", "chaos_source"], True),
            ("b_anchor", "revelation", 0.8, 0.06, "一点を指す大ゴマの声", "二度目のめくり直後、外側からの声とともに一点を指す大ゴマが視線を集め、現実が戻る", ["b_confusion"], 0.12, 0.15, ["guiding_voice", "anchor_panel", "lost_one"], True),
            ("b_restore", "payoff", 0.9, 0.05, "格子の回復", "整った格子の4コマで、混乱の正体と主体の現在地を順序立てて示す", ["b_anchor"], -0.2, 0.05, ["lost_one", "clarity", "reading_order"], False),
            ("b_calm", "afterglow", 1.0, 0.04, "整った呼吸", "引きの構図で主体と声の主を並べ、静かな呼吸で締める", ["b_restore"], -0.1, 0.0, ["lost_one", "guiding_voice"], False),
        ],
        "mechanisms": [
            ("m_reading_breakdown", "prediction_error", "読書の約束事の崩壊", "右上から左下という読み順の前提が壊れ、読者自身が主体と同じ混乱を体験する", "重なり合い散乱するコマ", "混乱が説明ではなく体験として伝わる", "スキーマ違反と予測誤差", 0.9, ["b_disorder", "b_confusion"], ["reading_order"]),
            ("m_shared_disorientation", "empathy_transfer", "混乱の共有", "主体と読者が同時に迷子になることで、主体の恐怖への共感が最大化する", "逆さのコマと同じ場面の反復", "主体の不安を自分の不安として感じる", "共感の知覚‐行為モデル", 0.75, ["b_confusion"], ["lost_one"]),
            ("m_anchor_relief", "catharsis", "錨による解放", "一点を指す大ゴマが視線を集めた瞬間、混乱の緊張が一挙に解ける", "外側からの声と一点を指す大ゴマ", "安堵と現実感の回復", "カタルシス理論", 0.75, ["b_anchor", "b_restore"], ["anchor_panel", "guiding_voice"]),
        ],
        "anchor": "何が現実か分からなくなる恐怖の中で、自分を呼ぶ一つの声だけを頼りに戻ってこようとする心",
        "drivers": ["読み順がずれていく違和", "重なり散乱するコマの迷路", "一点を指す大ゴマの声"],
        "emotions": ["平静", "違和感", "困惑", "恐慌", "孤立", "安堵", "静けさ"],
        "template": "{{DISORIENTED_SELF}}の世界は整った{{READING_ORDER}}で描かれていたが、{{CHAOS_SOURCE}}によって小さな違和が生まれ、めくり直後にコマが重なり合って読み順は崩れる。{{LOST_GROUND}}のまま視線は散乱するコマの中で迷子になる。二度目のめくり直後、{{GUIDING_VOICE}}とともに{{ANCHOR_PANEL}}が一点を指し、主体は{{CLARITY}}を取り戻す。",
        "slots": [
            ("DISORIENTED_SELF", "知覚の迷子になる主体", "agent", ["混乱中も主体の顔を最低1コマは置く"], ("幻術にかかった主人公", "情報過多で判断できない顧客", "矛盾するデータに戸惑う分析者")),
            ("READING_ORDER", "読み順", "symbol", ["撹乱前に教科書的な順序を一度見せる"], ("右上から左下への5コマ", "整理された比較表", "論理的な章立て")),
            ("CHAOS_SOURCE", "混乱を生む力", "agent", ["正体は回復後まで明かさない"], ("幻術を操る敵", "乱立する選択肢と広告", "相反する研究結果の洪水")),
            ("LOST_GROUND", "立脚点の喪失", "lack", ["散乱のページで最大にする"], ("現実と幻の区別の喪失", "判断基準の喪失", "評価軸の喪失")),
            ("GUIDING_VOICE", "正しい方向を示す声", "agent", ["声は画面外から届かせる"], ("仲間の呼びかけ", "信頼できる専門家の助言", "一つの確かな原則")),
            ("ANCHOR_PANEL", "一点を指す大ゴマ", "symbol", ["めくり直後に置き、視線を一点へ集める"], ("指差す手の大ゴマ", "一つの推奨プランの強調表示", "核心の一文を示す図")),
            ("CLARITY", "現実の把握", "goal", ["回復後は整った格子で示す"], ("幻術の解除", "迷いのない選択", "整理された結論")),
        ],
        "invariants": [
            "撹乱の前に教科書的な読み順のページを必ず置き、基準線を読者に刻む",
            "撹乱は段階的に強め、重なり合うページから散乱するページへと進める",
            "回復の大ゴマはめくり直後の右ページに置き、視線を一点へ集める",
            "撹乱のページでも主体の顔を最低1コマは判読できる大きさで残す",
        ],
        "guide": {
            "manga": "6ページ構成。p1（右）: 教科書的な5コマ（MS・アイレベル）。p2（左）: 6コマのうち2コマの大きさと位置をずらし、最終コマで同じ顔が二度現れる（CU・ダッチ）。p3（右・めくり直後）: 7コマが重なり合う配置（コマ同士が30%ずつ重なる、ダッチアングル多用）。p4（左）: 8コマを散乱させ、1コマは上下逆さ、同じ場面を2回反復（ECU・広角18mmの歪み）。p5（右・めくり直後）: 2コマ、上段の大ゴマで画面外の声と一点を指す手（CU・アイレベル・85mm）。p6（左）: 整った格子の4コマで現在地を示し、最後に引きの構図。",
            "marketing": "LPで情報過多の比較画面を段階的に崩して見せ、顧客が選べなくなる混乱を体験させる。最も混乱したセクションの直後に、一つの推奨プランだけを大きく指し示すセクションを置き、整理された格子の比較表で判断の根拠を示してからCTAへ導く。",
            "article": "相反する主張や研究結果を意図的に並べて読者に混乱を体験させ、最も混乱した節の直後に一つの確かな原則を独立した段落で示す。その原則を軸に情報を整理し直す章で結論へ導く。",
        },
        "rhythm": {
            "page_count": 6,
            "panels_per_page": [5, 6, 7, 8, 2, 4],
            "emphasis_pages": [2, 3, 4],
            "page_turn_reveals": [2, 4],
            "spread_pages": [],
            "silent_panels_ratio": 0.2,
            "pacing_notes": [
                "p1（右）: 教科書的な読み順の5コマで基準線を刻む",
                "p2（左）: 6コマのうち2コマをずらし、最終コマで同じ顔が二度現れる違和で引く",
                "p3（右・めくり直後）: 7コマを重なり合わせ、どれから読むべきか分からなくする",
                "p4（左）: 8コマを散乱させ、逆さのコマと同場面の反復で視線を迷子にする",
                "p5（右・めくり直後）: 2コマ、画面外の声と一点を指す大ゴマで視線を回復させる",
                "p6（左）: 整った格子の4コマで現在地を示し、引きの構図で締める",
            ],
        },
    }
)
