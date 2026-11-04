"""高度な物語構造（advanced_narrative）14パターンの著作スクリプト。

枠物語・信頼できない語り手・二重時間軸・群像の収束・メタフィクション・円環・逆算・多重反転・
視点転換・分岐と収束・喪失受容・伏線の一斉回収・悲喜劇・観測者効果といった「語りの水準や時間の
編成そのものが感情を動かす」高度な物語構造を、固有名詞を一切含まない抽象表現で手書き設計し、
``UniversalStructure`` として検証したうえで
``core/structure_engine/seeds/advanced_narrative_structures.json`` を生成する。

実行: ``python -m core.structure_engine.seeds.authoring.build_advanced_narrative``
検証のみ: ``python -m core.structure_engine.seeds.authoring.build_advanced_narrative --check``

設計規約（``quality_problems`` / ``validate_all`` が機械検査する）:
- 波形は5チャネルすべてを6〜12個のキーフレームで明示し、PCHIP 単調補間で21点へ展開する。
  転換点ビートの t はキーフレーム（緊張度の局所極値）または読者認知量の跳躍位置に置く。
- tension_delta / information_shift（|値|≥0.1）の符号は、ビート直前区間の緊張度・情報非対称性の
  実変化と一致させる。大域ピーク ±0.1 には必ず転換点ビートを置く。
- 全ノードにブループリントのスロットを割り当て、力学グラフは連結に保つ。裏切りエッジには同一ペアの
  依存エッジを先行させる。最後のビートは余韻・回収・頂点のいずれかで締める。
- 物語論上の原型名・系譜は ``metadata.lineage`` にのみ記載し、5階層の本体には書かない。
- 整合性監査（``audit_structure``）の指摘は0件、カテゴリ内の緊張波形相関は 0.97 未満、
  既存60シードとの緊張波形相関は 0.985 未満とする。

パターン定義のタプル形式:
- nodes:      (id, kind, role, label, description, weight, slot)
- edges:      (source, target, kind, intensity, gradient, label, activation_t)
- beats:      (id, beat_type, t, span, label, function, causes, tension_delta, information_shift,
               involved_nodes, is_turning_point)
- mechanisms: (id, category, name, description, trigger, effect, theory, strength, linked_beats, linked_nodes)
- slots:      (name, category, description, constraints, (manga例, marketing例, article例), required)
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
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
OUTPUT_PATH = SEED_DIR / "advanced_narrative_structures.json"
EXISTING_SEED_FILES = (
    "narrative_archetypes.json",
    "manga_panel_rhythm_patterns.json",
    "persuasion_and_marketing_structures.json",
)
SCHEMA_VERSION = "1.0"
CATEGORY = "advanced_narrative"
ID_PREFIX = "an"
DESCRIPTION = (
    "語りの水準・時間の編成・視点の配置そのものが読者の感情と理解を動かす高度な物語構造14選。"
    "枠物語の反転、信頼できない語り手、二重時間軸の合流、群像の収束、メタフィクション、円環、記憶の逆算、"
    "多重反転、敵対者視点による価値反転、分岐と収束、喪失受容、伏線の一斉回収、悲喜劇、観測者効果を、"
    "21点の感情・情報波形、力学グラフ、因果ビート、認知メカニズム、抽象変数ブループリント、"
    "漫画・マーケティング・記事への転用ガイドとともに収録する。"
)
EXPECTED_COUNT = 14
QUALITY_BOUNDS: dict[str, tuple[int, int]] = {
    "nodes": (4, 8),
    "edges": (5, 12),
    "beats": (6, 10),
    "turning_points": (2, 10),
    "mechanisms": (3, 5),
    "slots": (6, 10),
    "invariants": (3, 6),
}
KEYFRAME_BOUNDS = (6, 12)
MAX_TENSION_CORRELATION = 0.97
MAX_EXISTING_CORRELATION = 0.985
MIN_GUIDE_CHARS = 60
SIGN_CHECK_THRESHOLD = 0.1
PEAK_TOLERANCE = 0.1
AUDIENCE_JUMP = 0.1
FINAL_BEAT_TYPES = frozenset({BeatType.AFTERGLOW, BeatType.PAYOFF, BeatType.CLIMAX})
_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)

PATTERNS: list[dict[str, Any]] = []


def _node(spec: tuple) -> dict[str, Any]:
    node_id, kind, role, label, description, weight, slot = spec
    return {
        "id": node_id,
        "kind": kind,
        "role": role,
        "label": label,
        "description": description,
        "weight": weight,
        "slot": slot,
    }


def _edge(spec: tuple) -> dict[str, Any]:
    source, target, kind, intensity, gradient, label, activation_t = spec
    return {
        "source": source,
        "target": target,
        "kind": kind,
        "intensity": intensity,
        "gradient": gradient,
        "label": label,
        "activation_t": activation_t,
    }


def _beat(spec: tuple) -> dict[str, Any]:
    (beat_id, beat_type, t, span, label, function, causes, tension_delta, information_shift, nodes, turning) = spec
    return {
        "id": beat_id,
        "beat_type": beat_type,
        "t": t,
        "span": span,
        "label": label,
        "function": function,
        "causes": list(causes),
        "tension_delta": tension_delta,
        "information_shift": information_shift,
        "involved_nodes": list(nodes),
        "is_turning_point": turning,
    }


def _mechanism(spec: tuple) -> dict[str, Any]:
    (mech_id, category, name, description, trigger, effect, theory, strength, beats, nodes) = spec
    return {
        "id": mech_id,
        "category": category,
        "name": name,
        "description": description,
        "trigger": trigger,
        "effect": effect,
        "theory": theory,
        "strength": strength,
        "linked_beats": list(beats),
        "linked_nodes": list(nodes),
    }


def _slot(spec: tuple) -> dict[str, Any]:
    name, category, description, constraints, examples, required = spec
    manga, marketing, article = examples
    return {
        "name": name,
        "category": category,
        "description": description,
        "constraints": list(constraints),
        "examples": {"manga": manga, "marketing": marketing, "article": article},
        "required": required,
    }


def pattern_id(spec: dict[str, Any]) -> str:
    return f"{ID_PREFIX}_{spec['num']:02d}_{spec['slug']}"


def build_structure(spec: dict[str, Any]) -> UniversalStructure:
    """1パターンの定義から UniversalStructure を組み立てて検証する。"""
    wave = EmotionalInformationWave.from_keyframes(spec["wave"], n=21)
    payload = {
        "id": pattern_id(spec),
        "metadata": {
            "title": spec["title"],
            "summary": spec["summary"],
            "category": CATEGORY,
            "source_kind": "seed",
            "tags": list(spec["tags"]),
            "origin_domain": spec["origin"],
            "lineage": list(spec["lineage"]),
            "parent_ids": [],
            "language": "ja",
            "version": 1,
            "created_at": "2026-11-03T00:00:00+00:00",
            "notes": spec.get("notes", ""),
        },
        "tension_graph": {
            "nodes": [_node(n) for n in spec["nodes"]],
            "edges": [_edge(e) for e in spec["edges"]],
        },
        "wave": wave.model_dump(mode="json"),
        "beats": {"beats": [_beat(b) for b in spec["beats"]]},
        "mechanics": {
            "mechanisms": [_mechanism(m) for m in spec["mechanisms"]],
            "empathy_anchor": spec["anchor"],
            "immersion_drivers": list(spec["drivers"]),
            "emotion_sequence": list(spec["emotions"]),
        },
        "blueprint": {
            "template": spec["template"],
            "slots": [_slot(s) for s in spec["slots"]],
            "invariants": list(spec["invariants"]),
        },
        "transfer_guide": dict(spec["guide"]),
        "rhythm_profile": None,
    }
    return UniversalStructure.model_validate(payload)


def build_all() -> list[UniversalStructure]:
    return [build_structure(spec) for spec in sorted(PATTERNS, key=lambda s: s["num"])]


def _grid_index(t: float) -> int:
    return int(round(t * 20))


def _is_tension_extremum(tension: np.ndarray, index: int) -> bool:
    last = len(tension) - 1
    if index <= 0:
        return bool(tension[0] != tension[1])
    if index >= last:
        return bool(tension[last] != tension[last - 1])
    left, center, right = tension[index - 1], tension[index], tension[index + 1]
    return bool((center >= left and center >= right) or (center <= left and center <= right))


def quality_problems(spec: dict[str, Any], structure: UniversalStructure) -> list[str]:
    """SPEC §10 / §14 の品質基準と、波形・ビート・グラフの整合性を検査して問題点を返す。"""
    sid = structure.id
    problems: list[str] = []
    graph = structure.tension_graph
    beats = structure.beats.beats
    counts = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "beats": len(beats),
        "turning_points": len(structure.beats.turning_points()),
        "mechanisms": len(structure.mechanics.mechanisms),
        "slots": len(structure.blueprint.slots),
        "invariants": len(structure.blueprint.invariants),
    }
    for key, (low, high) in QUALITY_BOUNDS.items():
        if not low <= counts[key] <= high:
            problems.append(f"{sid}: {key}={counts[key]} が {low}〜{high} の範囲外")
    for field in ("manga", "marketing", "article"):
        if len(getattr(structure.transfer_guide, field).strip()) < MIN_GUIDE_CHARS:
            problems.append(f"{sid}: 転用ガイド {field} が短すぎます")
    if set(spec["wave"]) != set(WAVE_CHANNELS):
        problems.append(f"{sid}: 波形キーフレームは5チャネルすべてを明示する必要があります")
    for channel, frames in spec["wave"].items():
        if not KEYFRAME_BOUNDS[0] <= len(frames) <= KEYFRAME_BOUNDS[1]:
            problems.append(f"{sid}: {channel} のキーフレーム数 {len(frames)} が範囲外")
    if not 3 <= len(structure.metadata.tags) <= 6:
        problems.append(f"{sid}: tags は3〜6個")
    if not structure.metadata.lineage:
        problems.append(f"{sid}: lineage が空です")
    unslotted = [node.id for node in graph.nodes if not node.slot]
    if unslotted:
        problems.append(f"{sid}: スロット未割当のノードがあります: {', '.join(unslotted)}")
    if len({edge.kind for edge in graph.edges}) < 4:
        problems.append(f"{sid}: エッジ種別の多様性が不足しています（4種以上）")
    for edge in graph.edges:
        if edge.kind != EdgeKind.BETRAYAL:
            continue
        dependencies = [
            other
            for other in graph.edges
            if other.kind == EdgeKind.DEPENDENCY and {other.source, other.target} == {edge.source, edge.target}
        ]
        if not dependencies:
            problems.append(f"{sid}: 裏切り {edge.source}->{edge.target} に先行する依存エッジがありません")
        elif edge.activation_t is not None and all(
            dep.activation_t is not None and dep.activation_t > edge.activation_t for dep in dependencies
        ):
            problems.append(f"{sid}: 裏切り {edge.source}->{edge.target} が依存より先に顕在化しています")
    for mechanism in structure.mechanics.mechanisms:
        if not mechanism.theory.strip():
            problems.append(f"{sid}: メカニズム {mechanism.id} に theory がありません")
    for slot in structure.blueprint.slots:
        if any(not str(slot.examples.get(domain, "")).strip() for domain in ("manga", "marketing", "article")):
            problems.append(f"{sid}: スロット {slot.name} の examples が3領域揃っていません")
    for text in structure.text_fields() + list(structure.metadata.lineage):
        if _URL_RE.search(text):
            problems.append(f"{sid}: URL らしき文字列を含みます: {text[:40]}")

    wave = structure.wave
    tension = wave.channel("tension")
    for index, beat in enumerate(beats):
        if index > 0 and not beat.causes:
            problems.append(f"{sid}: ビート {beat.id} に因果（causes）がありません")
        if abs(beat.tension_delta) >= SIGN_CHECK_THRESHOLD:
            change = wave.sample("tension", beat.t) - wave.sample("tension", max(0.0, beat.t - 0.1))
            if np.sign(change) != np.sign(beat.tension_delta) or abs(change) < 1e-6:
                problems.append(
                    f"{sid}: ビート {beat.id} の tension_delta={beat.tension_delta:+.2f} が波形変化 {change:+.3f} と不整合"
                )
        if abs(beat.information_shift) >= SIGN_CHECK_THRESHOLD:
            asym_change = wave.sample("information_asymmetry", min(1.0, beat.t + 0.05)) - wave.sample(
                "information_asymmetry", max(0.0, beat.t - 0.1)
            )
            if np.sign(asym_change) != np.sign(beat.information_shift) or abs(asym_change) < 1e-6:
                problems.append(
                    f"{sid}: ビート {beat.id} の information_shift={beat.information_shift:+.2f} が"
                    f"非対称性変化 {asym_change:+.3f} と不整合"
                )
        if beat.is_turning_point:
            jump = wave.sample("audience_knowledge", min(1.0, beat.t + 0.05)) - wave.sample(
                "audience_knowledge", max(0.0, beat.t - 0.05)
            )
            if not (_is_tension_extremum(tension, _grid_index(beat.t)) or jump >= AUDIENCE_JUMP):
                problems.append(f"{sid}: 転換点 {beat.id}（t={beat.t}）が緊張度の極値にも認知量の跳躍にもありません")
    if beats and beats[-1].beat_type not in FINAL_BEAT_TYPES:
        problems.append(f"{sid}: 最後のビートは afterglow / payoff / climax のいずれかである必要があります")
    peak_t = float(wave.t[int(np.argmax(tension))])
    if not any(beat.is_turning_point and abs(beat.t - peak_t) <= PEAK_TOLERANCE + 1e-9 for beat in beats):
        problems.append(f"{sid}: 緊張度の大域ピーク t={peak_t:.2f} の ±{PEAK_TOLERANCE} に転換点がありません")
    return problems


def tension_correlations(structures: list[UniversalStructure]) -> list[tuple[str, str, float]]:
    """カテゴリ内全ペアの緊張波形 Pearson 相関。"""
    result: list[tuple[str, str, float]] = []
    for first, second in combinations(structures, 2):
        r = float(np.corrcoef(first.wave.channel("tension"), second.wave.channel("tension"))[0, 1])
        result.append((first.id, second.id, r))
    return result


def load_existing_tensions() -> dict[str, np.ndarray]:
    """既存60シード（3ファイル）の緊張波形を21点に揃えて読み込む。"""
    tensions: dict[str, np.ndarray] = {}
    grid = np.linspace(0.0, 1.0, 21)
    for filename in EXISTING_SEED_FILES:
        path = SEED_DIR / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for pattern in payload.get("patterns", []):
            wave = pattern["wave"]
            tensions[pattern["id"]] = np.interp(grid, np.asarray(wave["t"], dtype=float), np.asarray(wave["tension"], dtype=float))
    return tensions


def existing_correlations(structures: list[UniversalStructure]) -> list[tuple[str, str, float]]:
    """新規パターンと既存シードの緊張波形相関（全組合せ）。"""
    existing = load_existing_tensions()
    result: list[tuple[str, str, float]] = []
    for structure in structures:
        mine = structure.wave.resample_arrays(21)["tension"]
        for other_id, other in existing.items():
            if np.std(mine) < 1e-9 or np.std(other) < 1e-9:
                continue
            result.append((structure.id, other_id, float(np.corrcoef(mine, other)[0, 1])))
    return result


def validate_all(structures: list[UniversalStructure]) -> list[str]:
    """件数・ID一意性・品質基準・監査・波形多様性を総合検査する。"""
    problems: list[str] = []
    if len(structures) != EXPECTED_COUNT:
        problems.append(f"パターン数 {len(structures)} が {EXPECTED_COUNT} ではありません")
    ids = [s.id for s in structures]
    if len(ids) != len(set(ids)):
        problems.append("パターンIDが重複しています")
    nums = sorted(spec["num"] for spec in PATTERNS)
    if nums != list(range(1, len(nums) + 1)):
        problems.append(f"パターン番号が連番ではありません: {nums}")
    spec_by_id = {pattern_id(spec): spec for spec in PATTERNS}
    for structure in structures:
        problems.extend(quality_problems(spec_by_id[structure.id], structure))
        for issue in audit_structure(structure):
            problems.append(f"{structure.id}: 整合性監査 {issue.code}（{issue.location}）: {issue.message}")
    for first, second, r in tension_correlations(structures):
        if r >= MAX_TENSION_CORRELATION:
            problems.append(f"カテゴリ内の緊張波形が類似しすぎています: {first} × {second} r={r:.4f}")
    for mine, other, r in existing_correlations(structures):
        if r >= MAX_EXISTING_CORRELATION:
            problems.append(f"既存シードと緊張波形が類似しすぎています: {mine} × {other} r={r:.4f}")
    return problems


def write_json(structures: list[UniversalStructure], path: Path = OUTPUT_PATH) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "category": CATEGORY,
        "description": DESCRIPTION,
        "patterns": [s.model_dump(mode="json") for s in structures],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="高度な物語構造シード（14パターン）の検証とJSON生成")
    parser.add_argument("--check", action="store_true", help="検証のみ行いJSONを書き出さない")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="出力先JSONパス")
    args = parser.parse_args(argv)
    structures = build_all()
    problems = validate_all(structures)
    if problems:
        for problem in problems:
            print(f"[NG] {problem}")
        return 1
    inner = tension_correlations(structures)
    outer = existing_correlations(structures)
    max_inner = max(inner, key=lambda item: item[2]) if inner else ("", "", 0.0)
    max_outer = max(outer, key=lambda item: item[2]) if outer else ("", "", 0.0)
    print(
        f"[OK] {len(structures)} パターン検証完了（カテゴリ内最大相関 {max_inner[2]:.4f}: {max_inner[0]} × {max_inner[1]} / "
        f"既存シードとの最大相関 {max_outer[2]:.4f}: {max_outer[0]} × {max_outer[1]}）"
    )
    if not args.check:
        written = write_json(structures, args.output)
        print(f"[OK] 書き出し: {written}")
    return 0


# ---------------------------------------------------------------------------
# パターン定義
# ---------------------------------------------------------------------------

PATTERNS.append(
    {
        "num": 1,
        "slug": "nested_frame_reversal",
        "title": "入れ子の語り（枠物語の真相反転）",
        "origin": "literature",
        "summary": (
            "外枠の語り手が切迫した状況の中で内側の物語を語り聞かせ、内側の物語の宙吊りの結末が、外枠の語りの場そのものの意味を反転させる構造。"
            "聞き手と読者は内側の物語に没入するが、最後に「なぜ今この話が語られたのか」が明かされ、語る行為自体が事件だったと判明する。"
        ),
        "tags": ["枠物語", "入れ子構造", "語りの反転", "二重水準"],
        "lineage": ["枠物語（物語内物語）", "語りの水準論（物語言説の階層）", "夜ごとの語りで命をつなぐ説話形式"],
        "nodes": [
            ("narrator", "agent", "outer_narrator", "切迫した場で物語を語り始める語り手", "語りの結末と目的を最初から知りながら、それを物語の形でしか差し出せない。語りの主導権を握り続けることで聞き手を目的地へ導く。", 0.9, "OUTER_NARRATOR"),
            ("listener", "agent", "outer_listener", "語りに引き込まれる聞き手", "読者の代理として内側の物語に没入する。自分が物語の当事者であることをまだ知らない。", 0.8, "LISTENER"),
            ("inner_hero", "agent", "inner_protagonist", "内側の物語の主人公", "取り返しのつかない選択へ向かう人物。語り手の過去の姿であり、同時に聞き手の運命の写し鏡として機能する。", 0.75, "INNER_PROTAGONIST"),
            ("hidden_purpose", "desire", "narration_motive", "語りに込められた真の目的", "警告・告白・継承のいずれか。語りが終わる瞬間にだけ姿を現し、全編の意味を確定させる。", 0.85, "NARRATION_MOTIVE"),
            ("unspoken_debt", "lack", "concealed_guilt", "語り手が直接口にできない負い目", "語りの順序と省略を支配する見えない力。内側の物語の宙吊りの結末に刻まれている。", 0.7, "CONCEALED_GUILT"),
            ("embedded_object", "symbol", "bridging_token", "内側と外枠の両方に現れる物", "二つの水準を接続する物証。外枠に現れた瞬間、内側の物語が実在の出来事だったと証明する。", 0.6, "BRIDGING_TOKEN"),
        ],
        "edges": [
            ("narrator", "listener", "asymmetry", 0.8, 0.8, "語り手だけが結末と目的を知る", 0.05),
            ("listener", "narrator", "dependency", 0.6, -0.3, "物語の続きを語り手に委ねる", 0.15),
            ("narrator", "hidden_purpose", "pursuit", 0.85, 0.0, "語りを通じた目的の遂行", 0.05),
            ("unspoken_debt", "narrator", "power_gradient", 0.7, 0.6, "負い目が語りの順序と省略を強いる", 0.3),
            ("inner_hero", "unspoken_debt", "embodiment", 0.65, 0.0, "内側の主人公が負い目の原型を体現する", 0.6),
            ("inner_hero", "listener", "opposition", 0.5, 0.0, "内側の選択と聞き手の選択の衝突", 0.7),
            ("embedded_object", "hidden_purpose", "embodiment", 0.6, 0.0, "外枠に現れた物が語りの目的を示す", 0.8),
            ("listener", "embedded_object", "pursuit", 0.5, 0.0, "聞き手が手がかりを手に取る", 0.85),
        ],
        "beats": [
            ("frame_open", "hook", 0.05, 0.05, "語りの場の切迫", "閉ざされた夜や迫る期限の中で語りが始まり、外枠そのものに不穏な問いが立つ。", [], 0.2, 0.05, ["narrator", "listener"], False),
            ("inner_begin", "setup", 0.15, 0.06, "内側の物語への移行", "語りの水準が切り替わり、内側の主人公の穏やかな日常と欠落が示される。", ["frame_open"], -0.12, 0.1, ["inner_hero", "narrator"], False),
            ("inner_rise", "escalation", 0.35, 0.08, "内側の葛藤の上昇", "内側の主人公が取り返しのつかない選択へ近づき、聞き手は物語へ深く没入していく。", ["inner_begin"], 0.2, 0.08, ["inner_hero", "unspoken_debt"], False),
            ("frame_interrupt", "transition", 0.45, 0.05, "外枠への割り込み", "聞き手が話を遮り、外枠の時間的切迫が再確認されて内側の熱が一度冷まされる。", ["inner_rise"], -0.1, 0.05, ["listener", "narrator"], False),
            ("inner_climax", "crisis", 0.6, 0.06, "内側の物語の頂点", "内側の主人公が決定的な選択を誤り、語り手の負い目の原型となる出来事が語られる。", ["frame_interrupt"], 0.15, 0.05, ["inner_hero", "unspoken_debt"], True),
            ("inner_end", "transition", 0.7, 0.05, "語られない結末", "内側の物語が結末を語らないまま途切れ、聞き手と読者に宙吊りの問いが残る。", ["inner_climax"], -0.2, 0.03, ["narrator", "listener"], False),
            ("frame_twist", "structural_twist", 0.8, 0.05, "枠の反転", "外枠に象徴物が現れ、語り手が内側の主人公の成れの果てであり、語りの目的が聞き手自身に向けられていたと判明する。", ["inner_end"], 0.4, 0.3, ["narrator", "listener", "embedded_object", "hidden_purpose"], True),
            ("listener_choice", "costly_breakthrough", 0.9, 0.05, "聞き手の選択", "聞き手が語りを受け取り、語り手の負い目の代償を自ら引き受ける決断を下す。", ["frame_twist"], -0.3, 0.05, ["listener", "unspoken_debt"], False),
            ("dawn", "afterglow", 0.97, 0.03, "語り終えた夜明け", "語りの場が解かれ、象徴物が新しい持ち主の手に残ることで物語が継承される。", ["listener_choice"], -0.2, 0.02, ["listener", "embedded_object"], False),
        ],
        "mechanisms": [
            ("motive_gap", "curiosity_gap", "語りの動機への問い", "なぜ今この話が語られるのかという問いが、内側の物語を二重の注意で読ませる。", "外枠の切迫と語り手の沈黙の提示", "内側の細部に外枠の意味を探し続ける持続的な注意", "情報ギャップ理論", 0.75, ["frame_open", "frame_interrupt"], ["narrator", "hidden_purpose"]),
            ("inner_transport", "narrative_transportation", "内側の物語への移入", "水準の切り替えによって聞き手と読者が内側の世界へ深く運ばれ、外枠を一時的に忘れる。", "語りの様式の転換と内側の主人公の欠落描写", "外枠へ引き戻されたときの落差の増幅", "物語移入理論", 0.7, ["inner_begin", "inner_rise"], ["inner_hero", "listener"]),
            ("level_error", "prediction_error", "水準の反転による予測誤差", "語りの場そのものが事件だったと判明し、読者が保持していた水準の区別が崩れる。", "外枠への象徴物の出現", "全編の再解釈と強い驚き", "予測符号化理論（スキーマ違反）", 0.9, ["frame_twist"], ["narrator", "embedded_object"]),
            ("suspended_end", "zeigarnik", "宙吊りの結末の記憶優位", "語られなかった内側の結末が未完了課題として保持され、反転の瞬間に一挙に回収される。", "内側の物語の途切れ", "反転時の理解の快感と再読欲求", "ツァイガルニク効果（未完了課題の記憶優位）", 0.65, ["inner_end", "frame_twist"], ["inner_hero"]),
        ],
        "anchor": "誰にも打ち明けられない負い目を、物語という形でしか差し出せない語り手の孤独",
        "drivers": ["語りが途切れるたびに再浮上する外枠の切迫", "内側の主人公と聞き手の運命の相似", "二つの水準に現れる同じ物"],
        "emotions": ["不穏", "好奇", "没入", "焦燥", "戦慄", "驚愕", "覚悟", "余韻"],
        "template": (
            "{{OUTER_NARRATOR}}は切迫した夜に{{LISTENER}}へ物語を語り始める。物語の中で{{INNER_PROTAGONIST}}は取り返しのつかない選択へ向かうが、"
            "結末は語られないまま途切れる。{{BRIDGING_TOKEN}}が外枠に現れたとき、語りの真の目的である{{NARRATION_MOTIVE}}と語り手の"
            "{{CONCEALED_GUILT}}が明らかになり、{{LISTENER}}は{{FRAME_COST}}を引き受けて物語を受け継ぐ。"
        ),
        "slots": [
            ("OUTER_NARRATOR", "agent", "語りの場を支配する語り手", ["結末を知りながら語りの順序で隠す"], ("病床で昔話をする老人", "ブランドの原点を語る創業者", "取材で過去を回想する当事者"), True),
            ("LISTENER", "audience", "語りに引き込まれる聞き手（読者の代理）", ["自分が当事者だと終盤まで知らない"], ("看病に来た孫", "購入を迷う見込み顧客", "記事の読者"), True),
            ("INNER_PROTAGONIST", "agent", "内側の物語の主人公", ["語り手の過去または聞き手の未来の写し鏡にする"], ("若き日の旅人", "かつて同じ課題に苦しんだ担当者", "事例の当事者"), True),
            ("NARRATION_MOTIVE", "goal", "語りに込められた真の目的", ["終盤まで明示しない"], ("孫を同じ過ちから守ること", "顧客に最後の一歩を踏み出させること", "読者に判断基準を手渡すこと"), True),
            ("CONCEALED_GUILT", "lack", "語り手が直接は語れない負い目", [], ("見捨てた友への悔恨", "初期製品の失敗", "筆者自身の見落とし"), True),
            ("BRIDGING_TOKEN", "symbol", "内側と外枠の両方に現れる物証", ["序盤の内側の物語で一度だけ描写する"], ("錆びた懐中時計", "初期の試作品", "冒頭と結論で繰り返す同じ図表"), True),
            ("FRAME_COST", "cost", "物語を受け継ぐ代償", [], ("家業を継ぐ覚悟", "慣れた手法の放棄", "自説の撤回"), True),
        ],
        "invariants": [
            "内側の物語の結末は外枠の反転まで語り切らない",
            "外枠の切迫（期限・危険）を内側の物語の要所で再提示する",
            "外枠と内側の両方に同一の象徴物を登場させ、反転の物証とする",
            "反転は t≈0.8 に置き、語りの行為そのものの意味を書き換える",
        ],
        "guide": {
            "manga": (
                "全24ページ想定。1〜2pで外枠（夜の病室など）の切迫を描き、3pから内側の物語へ移る際はコマ枠の様式（断ち切り・白枠）を変えて水準差を視覚化する。"
                "内側の頂点は14〜15p、16pで内側の物語を途切れさせて外枠へ戻る。19p左ページ終端に象徴物を置く引きを作り、めくった20p右ページ冒頭で語り手の正体を大ゴマで明かす。"
                "最終2ページは外枠のみで、夜明けの光と象徴物の受け渡しを無言のコマで描く。"
            ),
            "marketing": (
                "ブランドの原点の物語（内側）を語りながら、見込み顧客の現在の切迫（外枠）を各セクション冒頭で再提示する。"
                "中盤で物語を途切れさせて比較表や証拠を挟み、終盤で『この物語の当事者はあなたと同じ立場だった』と枠を反転させ、移行コストを正直に示して申込へ繋ぐ。"
            ),
            "article": (
                "冒頭で筆者が今この話を書く理由を伏せたまま事例（内側）を語り始め、要所で読者の現状（外枠）に立ち返る。"
                "結論部で事例の当事者が筆者自身だったこと、または読者と同じ立場だったことを明かし、提示した枠組みの意味を反転させる。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.3), (0.05, 0.42), (0.15, 0.3), (0.35, 0.62), (0.45, 0.5), (0.6, 0.75), (0.7, 0.55), (0.8, 0.95), (0.9, 0.6), (1.0, 0.35)],
            "release": [(0.0, 0.05), (0.15, 0.2), (0.35, 0.1), (0.45, 0.25), (0.6, 0.08), (0.7, 0.3), (0.8, 0.05), (0.9, 0.55), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.05), (0.1, 0.12), (0.3, 0.3), (0.5, 0.45), (0.7, 0.55), (0.78, 0.6), (0.85, 0.92), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.5), (0.1, 0.55), (0.3, 0.6), (0.5, 0.62), (0.7, 0.66), (0.78, 0.7), (0.85, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.35), (0.15, 0.45), (0.35, 0.55), (0.45, 0.6), (0.6, 0.7), (0.8, 0.85), (0.9, 0.6), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 2,
        "slug": "unreliable_narrator",
        "title": "信頼できない語り手",
        "origin": "literature",
        "summary": (
            "一人称の語り手が出来事を語るが、語り手自身が真相を知りながら隠している、あるいは自己欺瞞によって歪めている構造。"
            "読者は語り手の視点を信じて推論を重ねるが、細部の綻びが臨界に達した瞬間に語り全体が別の意味へ反転し、冒頭からの全ての言葉が再解釈を迫られる。"
        ),
        "tags": ["信頼できない語り手", "自己欺瞞", "叙述の反転", "再解釈"],
        "lineage": ["信頼できない語り手（語りの信頼性の理論）", "叙述そのものを仕掛けとする推理", "自己欺瞞の心理劇"],
        "nodes": [
            ("narrator", "agent", "unreliable_narrator", "真相を語らない一人称の語り手", "温かく理性的な声で語るが、守りたい自己像のために真相を省略し、出来事の順序を組み替えている。", 0.95, "NARRATOR"),
            ("absent_other", "agent", "absent_figure", "語りの中心にいながら不在の人物", "語り手の回想の中でだけ姿を見せる。その不在こそが事件の核心である。", 0.75, "ABSENT_FIGURE"),
            ("witness", "agent", "doubting_witness", "語りに違和感を示す周縁の人物", "語り手の説明に小さな疑問を挟み、読者に最初の亀裂を与える。", 0.6, "DOUBTING_WITNESS"),
            ("self_image", "desire", "protected_self_image", "語り手が守ろうとする自己像", "善良で被害者である自分。語りの全ての取捨選択はこの像を守るために行われる。", 0.8, "PROTECTED_SELF_IMAGE"),
            ("suppressed_truth", "lack", "suppressed_truth", "抑圧された真相", "語り手が直視できない自分の行為。語りの空白と矛盾の形でだけ姿を現す。", 0.85, "SUPPRESSED_TRUTH"),
            ("telltale", "symbol", "telltale_detail", "語りの綻びを示す細部", "語り手が無造作に描写する小物や時刻。終盤で真相の物証へ反転する。", 0.6, "TELLTALE_DETAIL"),
        ],
        "edges": [
            ("narrator", "self_image", "pursuit", 0.9, 0.0, "自己像の防衛", 0.0),
            ("narrator", "absent_other", "dependency", 0.55, -0.2, "かつての拠り所への依存", 0.05),
            ("narrator", "absent_other", "asymmetry", 0.85, 0.8, "不在の真相を語り手だけが知る", 0.1),
            ("narrator", "witness", "power_gradient", 0.55, 0.4, "語りの主導権", 0.2),
            ("suppressed_truth", "narrator", "power_gradient", 0.75, 0.7, "抑圧された真相が語りを歪める", 0.3),
            ("witness", "narrator", "opposition", 0.6, 0.0, "違和感の指摘", 0.4),
            ("telltale", "suppressed_truth", "embodiment", 0.7, 0.0, "細部が真相を漏らす", 0.72),
            ("narrator", "absent_other", "betrayal", 0.85, 0.0, "語り手こそが不在を生んだ", 0.85),
        ],
        "beats": [
            ("confession_frame", "hook", 0.1, 0.05, "穏やかな一人称の語り出し", "語り手が誠実な証言者として語り始め、読者は無条件に視点を共有する。", [], 0.15, -0.05, ["narrator", "absent_other"], False),
            ("ordinary_account", "premise", 0.2, 0.06, "語り手の目に映る日常", "不在の人物との思い出が美しく語られ、語り手の善良な自己像が確立される。", ["confession_frame"], -0.05, 0.05, ["narrator", "absent_other", "self_image"], False),
            ("first_doubt", "escalation", 0.4, 0.06, "最初の違和感", "周縁の人物が語り手の説明に小さな疑問を挟み、時刻や順序の食い違いが一つ生じる。", ["ordinary_account"], 0.12, 0.05, ["witness", "narrator"], False),
            ("rationalization", "setup", 0.5, 0.05, "語り手の合理化", "語り手が違和感を巧みに丸め込み、読者もいったん語りを信じ直す。", ["first_doubt"], -0.05, 0.03, ["narrator", "self_image"], False),
            ("mounting_cracks", "local_optimization_limit", 0.65, 0.06, "合理化の限界", "綻びが増え、語り手の説明が自己矛盾を起こし始める。語りの空白が目立つようになる。", ["rationalization"], 0.15, 0.05, ["narrator", "witness", "suppressed_truth"], True),
            ("memory_gap", "crisis", 0.72, 0.04, "記憶の空白", "語り手自身が自分の記憶の空白に直面し、語りが一瞬途切れる。", ["mounting_cracks"], -0.05, 0.02, ["narrator", "suppressed_truth"], False),
            ("collapse", "structural_twist", 0.85, 0.05, "語りの崩壊", "無造作に描写されてきた細部が真相の物証へ反転し、語り手こそが不在の原因だったと判明する。", ["memory_gap"], 0.35, 0.35, ["narrator", "telltale", "absent_other", "suppressed_truth"], True),
            ("rereading", "payoff", 0.93, 0.04, "全編の再解釈", "冒頭からの穏やかな語りが、罪から目を背けるための言葉として読み直される。", ["collapse"], -0.4, 0.06, ["narrator", "self_image"], False),
            ("last_words", "afterglow", 0.98, 0.02, "語り手の最後の言葉", "語り手がなおも自己像に縋るか、初めて真相を自分の言葉で語るかの余韻を残す。", ["rereading"], -0.1, 0.02, ["narrator"], False),
        ],
        "mechanisms": [
            ("trust_dissonance", "cognitive_dissonance", "信頼と違和感の不協和", "語り手への信頼と積み重なる矛盾が衝突し、読者自身が解消を求めて読み進める。", "周縁の人物の疑問と語り手の合理化", "語りへの能動的な検証姿勢", "認知的不協和理論", 0.75, ["first_doubt", "rationalization", "mounting_cracks"], ["narrator", "witness"]),
            ("narration_error", "prediction_error", "語りの反転による予測誤差", "読者が共有していた視点そのものが偽りだったと判明し、最大級の予測誤差が生じる。", "細部の物証化", "強い驚きと全編の再読欲求", "予測符号化理論（スキーマ違反）", 0.9, ["collapse"], ["narrator", "telltale"]),
            ("complicit_judgment", "moral_judgment", "共犯的な道徳判断", "語り手に共感していた読者が、その共感を自ら裁き直すことを迫られる。", "語り手の加害の露見", "自分の判断の脆さへの省察", "道徳基盤理論", 0.7, ["collapse", "rereading"], ["narrator", "absent_other"]),
            ("gap_hunger", "curiosity_gap", "空白への問い", "語りの空白そのものが問いとなり、読者は語られないものを探し始める。", "記憶の空白と時刻の食い違い", "細部への注意の集中", "情報ギャップ理論", 0.65, ["first_doubt", "memory_gap"], ["suppressed_truth"]),
        ],
        "anchor": "自分を善良な人間だと信じたいという、誰もが抱える自己像への執着",
        "drivers": ["語り手の温かい声と細部の不穏さの落差", "周縁の人物だけが示す違和感", "語りの空白が増えていく不安"],
        "emotions": ["安心", "共感", "違和", "再信頼", "疑念", "戦慄", "愕然", "沈思"],
        "template": (
            "{{NARRATOR}}は{{ABSENT_FIGURE}}との日々を誠実に語り、{{PROTECTED_SELF_IMAGE}}を読者と共有する。"
            "{{DOUBTING_WITNESS}}の疑問が語りに亀裂を入れるたびに語り手は合理化を重ねるが、無造作に語られた{{TELLTALE_DETAIL}}が"
            "{{SUPPRESSED_TRUTH}}を指し示したとき語りは崩壊し、{{REREADING_PAYOFF}}によって全編が別の意味へ反転する。"
        ),
        "slots": [
            ("NARRATOR", "agent", "真相を語らない一人称の語り手", ["序盤は誠実で好感の持てる声にする"], ("穏やかな一人称の少女", "自社の歩みを語るブランド", "自説を語る筆者"), True),
            ("ABSENT_FIGURE", "agent", "語りの中心にいて不在の人物", [], ("姿を消した親友", "去っていった初期顧客", "引用されない反対意見の持ち主"), True),
            ("DOUBTING_WITNESS", "agent", "違和感を示す周縁の人物", ["疑問は小さく具体的にする"], ("無口な担任教師", "鋭い質問をする比較サイトの声", "反証データを示す第三者"), True),
            ("PROTECTED_SELF_IMAGE", "goal", "語り手が守る自己像", [], ("友を守った自分", "顧客想いの企業像", "正しかった自分の主張"), True),
            ("SUPPRESSED_TRUTH", "lack", "抑圧された真相", ["語りの空白と矛盾でだけ示す"], ("友を突き放した夜", "不都合な解約理由", "前提にしていた誤り"), True),
            ("TELLTALE_DETAIL", "evidence", "真相を漏らす細部", ["序盤に無造作に一度だけ描写する"], ("止まったままの時計", "隠されていた比較項目", "脚注の小さな数値"), True),
            ("REREADING_PAYOFF", "reversal", "全編の再解釈", [], ("冒頭の独白の意味の反転", "誠実な情報開示への転換", "結論の自己修正"), True),
        ],
        "invariants": [
            "語りの虚偽は省略と順序の組み替えで作り、明白な嘘はつかせない",
            "真相の物証となる細部は序盤に無造作に提示しておく",
            "周縁の人物の違和感を中盤までに最低二度挿入する",
            "反転は t≈0.85 に置き、反転後に冒頭の語りを再提示して意味の変化を示す",
        ],
        "guide": {
            "manga": (
                "全20ページ想定。語り手の独白はモノローグのフキダシで統一し、画面には語りと矛盾する小物（止まった時計など）を背景に小さく描き込む。"
                "6p・11pで周縁の人物の疑問を小さなコマで挿入し、15p左ページ終端で語り手の記憶の空白を白抜きのコマで示す。めくった16p右ページで細部が物証に反転する瞬間を大ゴマにし、"
                "最終ページで1pと同じ構図・同じ独白を再掲して意味の変化だけを読ませる。"
            ),
            "marketing": (
                "業界の常識を語る『信頼できる語り手』として旧来の訴求を提示した後、見落とされていた数値（細部）を示して語りの前提を崩す。"
                "比較検討の場面で不都合な情報も自ら開示する姿勢を見せ、誠実さへの転換そのものを価値提案に組み込む。"
            ),
            "article": (
                "筆者自身の過去の主張を誠実な語りとして再現し、読者が納得した後でデータの細部から矛盾を示して自己修正する構成にする。"
                "結論で冒頭の一文を再掲し、同じ言葉が異なる意味を持つことを示して学びを定着させる。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.25), (0.1, 0.4), (0.2, 0.33), (0.4, 0.52), (0.5, 0.46), (0.65, 0.68), (0.72, 0.6), (0.85, 0.97), (0.93, 0.55), (1.0, 0.4)],
            "release": [(0.0, 0.05), (0.2, 0.2), (0.4, 0.1), (0.5, 0.25), (0.65, 0.08), (0.72, 0.2), (0.85, 0.05), (0.93, 0.5), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.2), (0.4, 0.28), (0.65, 0.35), (0.8, 0.4), (0.85, 0.85), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.8), (0.2, 0.8), (0.4, 0.82), (0.65, 0.85), (0.8, 0.86), (0.85, 0.9), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.3), (0.2, 0.4), (0.4, 0.55), (0.65, 0.7), (0.8, 0.78), (0.85, 0.9), (0.93, 0.7), (1.0, 0.45)],
        },
    }
)

PATTERNS.append(
    {
        "num": 3,
        "slug": "dual_timeline_convergence",
        "title": "二重時間軸の合流",
        "origin": "film",
        "summary": (
            "過去と現在の二つの時間軸が交互に語られ、終盤で過去の線が現在の一点に追いつく構造。読者は両方の線を見ているため人物より先に二つの関係に気づき、"
            "劇的アイロニーを抱えたまま合流点を待つ。合流の瞬間、人物は過去の行為が現在の事件の原因だったことを知り、断絶していたつながりが修復される。"
        ),
        "tags": ["二重時間軸", "交互進行", "合流", "因果の回収"],
        "lineage": ["交差編集（クロスカッティング）", "過去と現在の並行叙述", "世代を越える因果の叙述"],
        "nodes": [
            ("present_seeker", "agent", "present_protagonist", "説明のつかない事件の原因を追う現在の主体", "現在の線の主人公。過去の線の存在を知らないまま、過去の決断の帰結と向き合わされる。", 0.9, "PRESENT_PROTAGONIST"),
            ("past_actor", "agent", "past_protagonist", "ある決断へ向かって生きる過去の主体", "過去の線の主人公。その決断が現在の脅威と救いの両方の起点となる。", 0.85, "PAST_PROTAGONIST"),
            ("present_threat", "agent", "present_antagonist", "過去の決断の帰結として生じた現在の脅威", "現在の主体を追い詰めるが、その存在理由は過去の線でしか説明できない。", 0.7, "PRESENT_THREAT"),
            ("causal_answer", "desire", "causal_answer", "現在の事件の原因の理解", "現在の主体が求める答え。合流点でのみ与えられる。", 0.8, "CAUSAL_ANSWER"),
            ("severed_link", "lack", "severed_link", "断ち切られた世代と記憶のつながり", "二つの線が互いを知らないことそのものが欠落であり、現在の主体を孤立させる。", 0.65, "SEVERED_LINK"),
            ("meeting_place", "symbol", "convergence_point", "二つの線が交わる場所・物・瞬間", "過去の主体が向かい、現在の主体が辿り着く一点。合流の物証となる。", 0.7, "CONVERGENCE_POINT"),
        ],
        "edges": [
            ("present_seeker", "causal_answer", "pursuit", 0.85, 0.0, "事件の原因の探求", 0.1),
            ("severed_link", "present_seeker", "power_gradient", 0.6, 0.5, "断絶が主体を孤立させる", 0.15),
            ("past_actor", "present_seeker", "asymmetry", 0.7, 0.7, "過去の主体だけが原因を知る", 0.2),
            ("present_threat", "present_seeker", "opposition", 0.85, 0.3, "現在の脅威の接近", 0.3),
            ("past_actor", "meeting_place", "pursuit", 0.7, 0.0, "過去の主体が向かう場所", 0.4),
            ("present_threat", "past_actor", "dependency", 0.6, -0.2, "脅威は過去の決断に由来する", 0.6),
            ("present_seeker", "meeting_place", "pursuit", 0.65, 0.0, "現在の主体が辿り着く場所", 0.6),
            ("meeting_place", "causal_answer", "embodiment", 0.75, 0.0, "合流点が答えを体現する", 0.78),
        ],
        "beats": [
            ("present_incident", "hook", 0.1, 0.05, "現在の不可解な事件", "現在の線で説明のつかない出来事が起こり、主体は原因の探索を始める。", [], 0.2, 0.05, ["present_seeker", "present_threat"], False),
            ("past_opening", "setup", 0.2, 0.06, "過去の線の穏やかな始まり", "時間が切り替わり、過去の主体の日常と絆が穏やかに描かれる。読者だけが二つの線を見比べ始める。", ["present_incident"], -0.25, 0.1, ["past_actor"], False),
            ("present_pressure", "escalation", 0.3, 0.05, "現在の脅威の接近", "現在の線に戻ると脅威が迫り、主体は孤立を深める。", ["past_opening"], 0.3, 0.05, ["present_seeker", "present_threat", "severed_link"], False),
            ("past_bond", "setup", 0.4, 0.05, "過去の絆と選択の芽", "過去の線で主体が守るべきものを得て、後の決断の伏線が置かれる。", ["present_pressure"], -0.22, 0.08, ["past_actor", "meeting_place"], False),
            ("echo", "local_optimization_limit", 0.5, 0.05, "符合と行き詰まり", "現在の捜索が行き詰まる一方、過去の線との奇妙な符合を読者だけが確信する。", ["past_bond"], 0.3, 0.06, ["present_seeker", "past_actor"], True),
            ("past_decision", "crisis", 0.6, 0.05, "過去の決定的な決断", "過去の主体が全てを賭けた決断を下し、その代償が現在へ続くことが暗示される。", ["echo"], -0.23, 0.04, ["past_actor", "present_threat"], False),
            ("convergence", "revelation", 0.78, 0.05, "二つの線の合流", "過去の線が現在の一点に追いつき、人物が因果の全貌を知る。読者の先行していた理解に人物が追いつく。", ["past_decision"], 0.35, -0.38, ["present_seeker", "past_actor", "meeting_place", "causal_answer"], True),
            ("mending", "costly_breakthrough", 0.9, 0.05, "断絶の修復", "現在の主体が過去の決断の代償を引き受け、脅威を終わらせることでつながりを結び直す。", ["convergence"], -0.45, 0.04, ["present_seeker", "present_threat", "severed_link"], False),
            ("one_timeline", "afterglow", 0.98, 0.02, "一つの時間", "二つの線が一つの時間として流れ始め、合流点に新しい意味が宿る。", ["mending"], -0.18, 0.02, ["present_seeker", "meeting_place"], False),
        ],
        "mechanisms": [
            ("cross_irony", "dramatic_irony", "両線を知る読者の劇的アイロニー", "読者だけが過去と現在の両方を見ているため、人物の行動の一つ一つが二重の意味を帯びる。", "過去の線と現在の線の交互提示", "合流点への強い予期と緊張", "劇的アイロニー", 0.85, ["past_opening", "echo"], ["past_actor", "present_seeker"]),
            ("cause_gap", "curiosity_gap", "原因への問い", "現在の不可解な事件が原因の空白を生み、過去の線を読み解く動機になる。", "冒頭の説明のつかない出来事", "過去の細部への注意", "情報ギャップ理論", 0.7, ["present_incident"], ["present_seeker", "causal_answer"]),
            ("convergence_reward", "anticipation_reward", "合流の予期と報酬", "二つの線が近づくほど合流の瞬間への予期が高まり、成就が強い快感を生む。", "符合の増加と場所の一致", "予期の充足による深い満足", "報酬予測誤差理論", 0.75, ["echo", "convergence"], ["meeting_place"]),
            ("lineage_catharsis", "catharsis", "断絶の修復によるカタルシス", "世代と記憶の断絶が修復される瞬間に蓄積した緊張が解放される。", "代償の引き受けと脅威の終結", "安堵と感動", "カタルシス論", 0.7, ["mending", "one_timeline"], ["severed_link"]),
        ],
        "anchor": "自分の知らない誰かの選択が、今の自分の人生を形作っていたという感覚",
        "drivers": ["過去と現在の交互進行が生む落差", "読者だけが気づく符合", "合流点が近づく予感"],
        "emotions": ["困惑", "郷愁", "焦燥", "温かさ", "確信", "衝撃", "安堵", "静けさ"],
        "template": (
            "{{PRESENT_PROTAGONIST}}は説明のつかない事件の原因である{{CAUSAL_ANSWER}}を追い、{{PRESENT_THREAT}}に追い詰められていく。"
            "交互に語られる過去で{{PAST_PROTAGONIST}}は{{CONVERGENCE_POINT}}へ向かって決断を下す。二つの線が合流したとき{{SEVERED_LINK}}の意味が明かされ、"
            "現在の主体は{{INHERITED_COST}}を引き受けてつながりを結び直す。"
        ),
        "slots": [
            ("PRESENT_PROTAGONIST", "agent", "現在の線の主体", [], ("家業を継いだ青年", "原因不明の売上低下に悩む担当者", "現代の課題に向き合う読者"), True),
            ("PAST_PROTAGONIST", "agent", "過去の線の主体", ["現在の主体と直接の面識を持たせない"], ("若き日の祖母", "創業期の開発者", "過去の研究者"), True),
            ("PRESENT_THREAT", "agent", "過去の決断から生じた現在の脅威", [], ("土地を狙う開発業者", "過去の設計に起因する障害", "前提の歪みが生んだ誤解"), True),
            ("CAUSAL_ANSWER", "goal", "現在の事件の原因の理解", [], ("村が守ってきたものの正体", "障害の根本原因", "通説が生まれた経緯"), True),
            ("SEVERED_LINK", "lack", "断ち切られたつながり", [], ("語られなかった家族の歴史", "引き継がれなかった設計思想", "忘れられた原典"), True),
            ("CONVERGENCE_POINT", "symbol", "二つの線が交わる一点", ["過去と現在の両方で同じ構図で描く"], ("古い灯台", "最初の設計書", "原典の一節"), True),
            ("INHERITED_COST", "cost", "過去を引き継ぐ代償", [], ("家を手放す決断", "既存システムの全面改修", "定説の撤回"), True),
        ],
        "invariants": [
            "過去と現在は交互に語り、過去の線は現在の線より穏やかな緊張度で描く",
            "読者が人物より先に二つの線の関係に気づくよう符合を配置する",
            "合流点は過去と現在の両方で同じ場所・物として描く",
            "合流は t≈0.78 に置き、その後に代償を伴う修復を置く",
        ],
        "guide": {
            "manga": (
                "全28ページ想定。現在の線は通常の枠線、過去の線は角を丸めた枠や淡いトーンで区別し、1〜2ページ単位で交互に切り替える。"
                "切り替えの直前と直後に同じ構図のコマ（手元・空・場所）を置いて符合を読者だけに見せる。21p左ページ終端で現在の主体が合流点の前に立つ引きを作り、"
                "めくった22p右ページ見開きで過去と現在の同じ構図を左右に並べて合流を示す。"
            ),
            "marketing": (
                "創業期の開発ストーリー（過去）と顧客の現在の課題（現在）をセクションごとに交互に配置し、両者に共通する一点（設計思想や最初の製品）を中盤で示唆する。"
                "終盤で両者が同じ課題から生まれたことを合流させ、引き継ぐ価値と移行コストを提示してCTAへ繋ぐ。"
            ),
            "article": (
                "現代の問題と、その問題の起源となった過去の出来事を交互に記述し、読者が先に関係に気づくよう伏線を置く。"
                "合流部で起源を明示し、断絶していた知見を現代の実践へ接続する具体的な手順で締める。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.35), (0.1, 0.55), (0.2, 0.3), (0.3, 0.6), (0.4, 0.38), (0.5, 0.68), (0.6, 0.45), (0.78, 0.95), (0.9, 0.5), (1.0, 0.3)],
            "release": [(0.0, 0.05), (0.2, 0.35), (0.3, 0.08), (0.4, 0.3), (0.5, 0.06), (0.6, 0.28), (0.78, 0.05), (0.9, 0.6), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.68), (0.75, 0.78), (0.85, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.2, 0.15), (0.4, 0.22), (0.6, 0.3), (0.75, 0.36), (0.8, 0.8), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.4), (0.2, 0.55), (0.4, 0.65), (0.6, 0.72), (0.78, 0.85), (0.9, 0.55), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 4,
        "slug": "ensemble_convergence",
        "title": "群像劇の収束",
        "origin": "film",
        "summary": (
            "互いに無関係に見える複数の人物の線が並行して進み、それぞれが小さな危機を抱えたまま一つの出来事へ引き寄せられていく構造。"
            "読者だけがすれ違いを見届け、終盤で一人の小さな行動が別の人物の運命を変える連鎖が可視化されたとき、群像全体が一つの意味へ収束する。"
        ),
        "tags": ["群像劇", "並行プロット", "収束", "連鎖"],
        "lineage": ["群像劇（アンサンブル）", "多数の線が交差するハイパーリンク型の物語", "偶然の連鎖による運命の交差"],
        "nodes": [
            ("lead_a", "agent", "ensemble_member_a", "失うものを抱えた第一の人物", "大切なものを守るために孤独な賭けに出ようとしている。", 0.8, "THREAD_A_LEAD"),
            ("lead_b", "agent", "ensemble_member_b", "秘密を抱えた第二の人物", "自分の秘密が第一の人物の運命の鍵であることを知らない。", 0.75, "THREAD_B_LEAD"),
            ("lead_c", "agent", "ensemble_member_c", "決断を先延ばしにする第三の人物", "小さな勇気が出せずにいるが、その一歩が連鎖の引き金になる。", 0.7, "THREAD_C_LEAD"),
            ("shared_need", "desire", "shared_need", "各人が別の言葉で求める同じもの", "居場所、赦し、つながり。人物ごとに表現は違うが本質は同一である。", 0.8, "SHARED_NEED"),
            ("isolation", "lack", "mutual_isolation", "互いの存在を知らない孤立", "全員が独りで問題を抱えていると思い込んでいる。", 0.7, "MUTUAL_ISOLATION"),
            ("event", "symbol", "convergence_event", "全員を一つの場所へ引き寄せる出来事", "祭り、事故、嵐など。各線の小さな危機を一点に集める。", 0.75, "CONVERGENCE_EVENT"),
        ],
        "edges": [
            ("lead_a", "shared_need", "pursuit", 0.7, 0.0, "居場所への希求", 0.0),
            ("lead_b", "shared_need", "pursuit", 0.6, 0.0, "赦しへの希求", 0.1),
            ("isolation", "lead_a", "power_gradient", 0.55, 0.4, "孤立が判断を狭める", 0.2),
            ("lead_b", "lead_a", "asymmetry", 0.65, 0.6, "知らずに運命の鍵を握る", 0.3),
            ("lead_a", "lead_c", "dependency", 0.5, -0.2, "見えない依存", 0.5),
            ("lead_c", "lead_b", "opposition", 0.55, 0.0, "利害の衝突", 0.6),
            ("lead_c", "event", "pursuit", 0.6, 0.0, "出来事へ向かう一歩", 0.72),
            ("event", "shared_need", "embodiment", 0.75, 0.0, "出来事が共通の希求を可視化する", 0.85),
        ],
        "beats": [
            ("mosaic_open", "hook", 0.1, 0.05, "断片のモザイク", "三つの線の断片が脈絡なく提示され、読者は関係の推測を始める。", [], 0.1, 0.1, ["lead_a", "lead_b", "lead_c"], False),
            ("threads_set", "premise", 0.2, 0.06, "それぞれの欠落", "各人物の欠落と目的が確立し、全員が孤立していることが示される。", ["mosaic_open"], -0.02, 0.06, ["lead_a", "lead_b", "lead_c", "isolation"], False),
            ("small_crises", "escalation", 0.3, 0.06, "小さな危機の同時多発", "各線で小さな危機が起こり、並行する緊張が積み上がる。", ["threads_set"], 0.12, 0.05, ["lead_a", "lead_b"], False),
            ("near_miss", "setup", 0.4, 0.05, "すれ違い", "人物同士が知らずに交差する瞬間を読者だけが目撃する。", ["small_crises"], -0.02, 0.1, ["lead_a", "lead_c"], False),
            ("pressure", "local_optimization_limit", 0.5, 0.05, "独力の限界", "各人物が自分の線だけでは解決できない地点に達する。", ["near_miss"], 0.12, 0.05, ["lead_a", "lead_b", "lead_c"], True),
            ("cascade", "crisis", 0.72, 0.06, "連鎖の始動", "第三の人物の一歩が第二の人物の秘密を露わにし、第一の人物の賭けを危機に陥れる。", ["pressure"], 0.22, 0.05, ["lead_c", "lead_b", "lead_a"], False),
            ("convergence", "revelation", 0.85, 0.05, "一点への収束", "全員が出来事の場に集まり、互いの線がつながっていたことを知る。", ["cascade"], 0.24, -0.3, ["lead_a", "lead_b", "lead_c", "event"], True),
            ("collective_choice", "costly_breakthrough", 0.95, 0.04, "群像の選択", "一人が小さな犠牲を払うことで全員の線が救われ、共通の希求が満たされる。", ["convergence"], -0.45, 0.04, ["lead_a", "lead_b", "lead_c", "shared_need"], False),
            ("mosaic_close", "afterglow", 1.0, 0.02, "完成したモザイク", "冒頭の断片が一枚の絵として並び直され、それぞれの日常へ戻っていく。", ["collective_choice"], -0.1, 0.02, ["event"], False),
        ],
        "mechanisms": [
            ("near_miss_irony", "dramatic_irony", "すれ違いの劇的アイロニー", "人物同士が知らずに交差する瞬間を読者だけが見ることで、関係への予期と焦燥が生まれる。", "すれ違いの場面", "人物に呼びかけたくなる焦燥", "劇的アイロニー", 0.8, ["near_miss"], ["lead_a", "lead_c"]),
            ("multi_projection", "identity_projection", "複数の自己投影先", "異なる欠落を持つ複数の人物のいずれかに読者が自分を重ね、群像全体に関与する。", "各人物の欠落の提示", "群像への多面的な感情移入", "同一化理論", 0.7, ["threads_set"], ["lead_a", "lead_b", "lead_c"]),
            ("convergence_reward", "anticipation_reward", "収束の予期と報酬", "交差が増えるほど収束の瞬間への予期が高まり、成就が強い満足をもたらす。", "独力の限界と出来事の予告", "全ての線がつながる快感", "報酬予測誤差理論", 0.75, ["pressure", "convergence"], ["event"]),
            ("ensemble_peak_end", "peak_end", "群像の選択の記憶", "一人の小さな犠牲という頂点と静かな終幕が、群像全体の体験を肯定的に記憶させる。", "群像の選択と余韻", "長く残る温かな記憶", "ピーク・エンドの法則", 0.6, ["collective_choice", "mosaic_close"], ["shared_need"]),
        ],
        "anchor": "自分だけが独りで問題を抱えていると思い込む孤独",
        "drivers": ["読者だけが見るすれ違い", "並行する小さな危機の同時進行", "一点へ引き寄せられていく予感"],
        "emotions": ["好奇", "共感", "焦燥", "もどかしさ", "緊迫", "驚き", "温かさ", "充足"],
        "template": (
            "{{THREAD_A_LEAD}}、{{THREAD_B_LEAD}}、{{THREAD_C_LEAD}}はそれぞれ{{MUTUAL_ISOLATION}}の中で{{SHARED_NEED}}を求め、互いを知らずにすれ違う。"
            "{{CONVERGENCE_EVENT}}が全員を一つの場所へ引き寄せたとき、一人の小さな一歩が連鎖を起こし、{{SMALL_SACRIFICE}}によって全ての線が一つの意味へ収束する。"
        ),
        "slots": [
            ("THREAD_A_LEAD", "agent", "失うものを抱えた第一の人物", [], ("店を畳もうとする店主", "撤退を検討する既存顧客", "課題を抱える第一の読者層"), True),
            ("THREAD_B_LEAD", "agent", "秘密を抱えた第二の人物", [], ("家出した少女", "社内で孤立する担当者", "異なる立場の第二の読者層"), True),
            ("THREAD_C_LEAD", "agent", "決断を先延ばしにする第三の人物", ["その一歩を連鎖の引き金にする"], ("告白できない青年", "導入を迷う決裁者", "行動をためらう第三の読者層"), True),
            ("SHARED_NEED", "goal", "全員が別の言葉で求める同じもの", [], ("帰る場所", "安心して任せられる仕組み", "共通する本質的課題の解決"), True),
            ("MUTUAL_ISOLATION", "lack", "互いを知らない孤立", [], ("同じ町で互いを知らない日々", "部署ごとに分断された情報", "分野ごとに閉じた議論"), True),
            ("CONVERGENCE_EVENT", "symbol", "全員を引き寄せる出来事", ["序盤から日付や予告として示す"], ("夏祭りの夜", "全社説明会", "共通の事例"), True),
            ("SMALL_SACRIFICE", "cost", "収束を可能にする小さな犠牲", [], ("大切な約束を破る勇気", "一部機能の先行導入", "自分の立場の一部の譲歩"), True),
        ],
        "invariants": [
            "各線は単独でも物語として成立する欠落と目的を持つ",
            "人物同士のすれ違いを読者だけに最低一度見せる",
            "収束の出来事は序盤から予告しておく",
            "収束は t≈0.85 に置き、一人の小さな行動が連鎖の起点となる",
        ],
        "guide": {
            "manga": (
                "全32ページ想定。各人物の線を見開き単位で順に担当させ、ページ端に共通の小物（ポスター、時計）を描き込んで同一時間・同一空間を示す。"
                "14〜15pで二人が背中合わせにすれ違う横長の大ゴマを置き、読者だけに交差を見せる。終盤は小ゴマを三列に並べた同時進行のページで加速させ、"
                "27p右ページの見開きで全員が出来事の場に揃う瞬間を描く。"
            ),
            "marketing": (
                "異なる立場の三つの顧客像（ペルソナ）の課題をそれぞれ短いストーリーで並行提示し、共通する本質的課題を中盤で示唆する。"
                "終盤で一つの導入事例が三者すべての課題を解決した連鎖を示し、自分に最も近い立場から申し込める導線を用意する。"
            ),
            "article": (
                "異なる分野・立場の三つの事例を並行して記述し、読者に共通点を探させる。"
                "結論部で共通の構造を一つの枠組みとして提示し、立場ごとの最初の一歩を具体的に示して締める。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.2), (0.1, 0.3), (0.2, 0.28), (0.3, 0.4), (0.4, 0.38), (0.5, 0.5), (0.6, 0.48), (0.72, 0.7), (0.85, 0.94), (0.95, 0.45), (1.0, 0.35)],
            "release": [(0.0, 0.05), (0.12, 0.2), (0.3, 0.15), (0.45, 0.25), (0.6, 0.2), (0.72, 0.1), (0.85, 0.05), (0.95, 0.7), (1.0, 0.9)],
            "audience_knowledge": [(0.0, 0.15), (0.2, 0.35), (0.4, 0.5), (0.6, 0.62), (0.8, 0.75), (0.86, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.2, 0.15), (0.4, 0.2), (0.6, 0.25), (0.8, 0.3), (0.86, 0.7), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.7), (0.15, 0.8), (0.3, 0.75), (0.5, 0.65), (0.7, 0.6), (0.85, 0.7), (0.95, 0.4), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 5,
        "slug": "metafiction_weaponized",
        "title": "メタフィクション（虚構性の武器化）",
        "origin": "manga",
        "summary": (
            "物語が自らを物語として意識し、作中の人物が「これは物語である」という事実を武器として用いる構造。"
            "読者との共犯関係を築いた後、虚構の規則そのものが人物を縛る最大の敵へ反転し、人物はコマ枠や余白といった虚構の装置を逆用して抗う。"
            "最後に読者自身の読む行為が結末に組み込まれる。"
        ),
        "tags": ["メタフィクション", "第四の壁", "自己言及", "読者参加"],
        "lineage": ["メタフィクション", "第四の壁の破壊", "自己言及的叙述"],
        "nodes": [
            ("aware_character", "agent", "self_aware_protagonist", "自分が物語の中にいると気づいた人物", "虚構の規則を知覚し、読者に語りかけ、やがて規則そのものに抗う。", 0.95, "SELF_AWARE_PROTAGONIST"),
            ("author_figure", "agent", "author_function", "物語の規則を司る作者的存在", "結末を決める権能を持つ。姿を見せず、規則の変更としてのみ現れる。", 0.85, "AUTHOR_FUNCTION"),
            ("implied_reader", "agent", "implied_reader", "物語に呼びかけられる読者という存在", "最初は観客として招かれ、最後には結末の決定に参加させられる。", 0.7, "IMPLIED_READER"),
            ("freedom", "desire", "narrative_freedom", "決められた筋書きからの自由", "人物が求める唯一の目標。虚構の外へ出ることではなく、虚構の中で選ぶ権利。", 0.85, "NARRATIVE_FREEDOM"),
            ("fixed_ending", "lack", "predetermined_fate", "決められた結末という欠落", "人物の選択を無意味にする予定調和。規則の反転とともに露わになる。", 0.75, "PREDETERMINED_FATE"),
            ("fiction_device", "symbol", "fiction_device", "虚構性を可視化する装置", "コマ枠、余白、ページの端、フキダシ。人物が最後に武器として掴む。", 0.7, "FICTION_DEVICE"),
        ],
        "edges": [
            ("author_figure", "aware_character", "power_gradient", 0.9, 0.9, "規則を司る者の絶対的優位", 0.1),
            ("aware_character", "freedom", "pursuit", 0.85, 0.0, "筋書きからの自由の希求", 0.2),
            ("aware_character", "implied_reader", "dependency", 0.6, -0.3, "読者の注視に存在を依存する", 0.35),
            ("fixed_ending", "aware_character", "power_gradient", 0.6, 0.6, "決められた結末が行動を縛る", 0.4),
            ("author_figure", "implied_reader", "asymmetry", 0.6, 0.5, "作者は読者の期待を先読みする", 0.5),
            ("aware_character", "author_figure", "opposition", 0.8, 0.0, "規則への反逆", 0.6),
            ("fiction_device", "fixed_ending", "embodiment", 0.7, 0.0, "装置が予定調和を可視化する", 0.6),
            ("aware_character", "fiction_device", "pursuit", 0.65, 0.0, "装置の逆用", 0.68),
        ],
        "beats": [
            ("wink", "hook", 0.1, 0.05, "最初の目配せ", "人物が一瞬だけ読者に視線を向け、第四の壁に最初の綻びが生まれる。", [], 0.12, 0.1, ["aware_character", "implied_reader"], False),
            ("play", "premise", 0.2, 0.05, "規則の遊び", "人物が虚構の約束事を笑いに変え、読者との共犯関係の土台を作る。", ["wink"], -0.1, 0.05, ["aware_character", "implied_reader"], False),
            ("rule_test", "escalation", 0.35, 0.06, "規則の侵犯と報い", "人物が規則を試しに破ると、物語の側から不穏な報いが返ってくる。", ["play"], 0.2, 0.05, ["aware_character", "author_figure"], False),
            ("complicity", "setup", 0.5, 0.05, "共犯関係の成立", "人物と読者が規則の存在を共有し、二人だけの秘密として物語を進める。", ["rule_test"], -0.05, -0.05, ["aware_character", "implied_reader"], False),
            ("rules_turn", "structural_twist", 0.6, 0.05, "規則の反転", "作者的存在が虚構性そのものを人物への武器に転じ、決められた結末が突きつけられる。読者は人物より先にその意味を理解する。", ["complicity"], 0.3, 0.15, ["author_figure", "aware_character", "fixed_ending"], True),
            ("despair", "crisis", 0.68, 0.05, "予定調和の重圧", "人物の選択がことごとく筋書きに回収され、抗う意味が失われかける。", ["rules_turn"], -0.15, 0.03, ["aware_character", "fixed_ending"], False),
            ("counter_device", "costly_breakthrough", 0.8, 0.05, "装置の逆用", "人物がコマ枠や余白を破り、虚構の装置そのものを使って筋書きの外へ一歩踏み出す。", ["despair"], 0.3, 0.05, ["aware_character", "fiction_device"], False),
            ("reader_inclusion", "climax", 0.9, 0.04, "読者の組み込み", "結末を決めるのは読む行為だと告げられ、読者自身が最後の選択に参加させられる。", ["counter_device"], 0.1, -0.12, ["aware_character", "implied_reader", "author_figure"], True),
            ("last_page", "afterglow", 0.97, 0.03, "閉じない最後の頁", "物語が閉じても人物の気配が余白に残り、読者の手元に選択の余韻が残る。", ["reader_inclusion"], -0.3, 0.02, ["implied_reader", "fiction_device"], False),
        ],
        "mechanisms": [
            ("fiction_dissonance", "cognitive_dissonance", "虚構と知りつつ動かされる不協和", "物語だと明示されているのに心が動くという矛盾が、読者に自分の読む行為を意識させる。", "共犯関係の成立と規則の反転", "物語への関与の自覚", "認知的不協和理論", 0.7, ["complicity", "rules_turn"], ["implied_reader"]),
            ("rule_error", "prediction_error", "規則の反転による予測誤差", "遊びだった虚構の規則が人物を縛る武器に変わり、読者の予測が裏切られる。", "作者的存在の介入", "不穏な驚きと緊張", "予測符号化理論（スキーマ違反）", 0.85, ["rules_turn"], ["author_figure", "fixed_ending"]),
            ("reader_projection", "identity_projection", "読者の物語内化", "読者が呼びかけの対象から結末の当事者へ変わり、強い自己投影が起こる。", "結末への読者の組み込み", "物語を自分事として体験する", "同一化理論", 0.8, ["wink", "reader_inclusion"], ["implied_reader"]),
            ("frame_catharsis", "catharsis", "枠を破る解放", "人物が虚構の装置を逆用して筋書きを破る瞬間に、蓄積した閉塞感が一挙に解放される。", "コマ枠を破る大ゴマ", "高揚と解放感", "カタルシス論", 0.75, ["counter_device", "last_page"], ["fiction_device"]),
        ],
        "anchor": "決められた役割や筋書きから自由になりたいという、誰もが抱く息苦しさ",
        "drivers": ["読者だけに向けられる視線", "規則が変わるたびの不穏さ", "虚構の装置そのものが動き出す驚き"],
        "emotions": ["くすぐったさ", "愉快", "不穏", "共犯感", "恐怖", "絶望", "高揚", "余韻"],
        "template": (
            "{{SELF_AWARE_PROTAGONIST}}は自分が物語の中にいることに気づき、{{IMPLIED_READER}}に語りかけながら{{NARRATIVE_FREEDOM}}を求める。"
            "{{AUTHOR_FUNCTION}}が虚構の規則を武器に変えて{{PREDETERMINED_FATE}}を突きつけたとき、人物は{{FICTION_DEVICE}}を逆用して抗い、"
            "最後に{{READER_ACT}}が結末を決める。"
        ),
        "slots": [
            ("SELF_AWARE_PROTAGONIST", "agent", "物語の中にいると気づいた人物", ["序盤は遊びとして第四の壁を破る"], ("脇役の少女", "自社の広告を自覚する案内役", "自分の論の構成を語る筆者"), True),
            ("AUTHOR_FUNCTION", "agent", "規則を司る作者的存在", ["姿ではなく規則の変化として描く"], ("ページの外の声", "業界の慣習", "論文の書式"), True),
            ("IMPLIED_READER", "audience", "呼びかけられる読者", [], ("読者自身", "ページを読んでいる見込み顧客", "記事の読者"), True),
            ("NARRATIVE_FREEDOM", "goal", "筋書きからの自由", [], ("自分で選ぶ結末", "型通りの提案からの自由", "定型的な結論からの自由"), True),
            ("PREDETERMINED_FATE", "lack", "決められた結末", [], ("最終回で消える役回り", "決まりきった購入導線", "結論ありきの議論"), True),
            ("FICTION_DEVICE", "symbol", "虚構性を可視化する装置", ["装置の存在を序盤から描き込む"], ("コマ枠と余白", "ページのスクロールそのもの", "見出しと脚注"), True),
            ("READER_ACT", "reversal", "結末を決める読む行為", [], ("ページをめくる手", "読者自身のクリック", "読後の判断"), True),
        ],
        "invariants": [
            "第四の壁の破壊は最初は遊びとして小さく行い、段階的に拡大する",
            "規則の反転で虚構性を人物に不利な武器へ変える",
            "虚構の装置（枠・余白・頁）を最終局面の武器として回収する",
            "最後に読者の行為を結末に組み込む",
        ],
        "guide": {
            "manga": (
                "全20ページ想定。2pで人物がコマ枠の外の読者へ一瞬視線を向ける小ゴマを置き、5〜8pで枠線に寄りかかる・フキダシを掴むなどの遊びを描く。"
                "12p右ページで余白が黒く塗りつぶされる演出で規則の反転を示し、16pで人物がコマ枠を破って隣のコマへ踏み込むぶち抜きの大ゴマを置く。"
                "最終ページは大きな余白の中に人物の小さな姿だけを残し、読者がページを閉じる行為を結末として描く。"
            ),
            "marketing": (
                "ランディングページ自身がこれは広告ですと自己言及し、典型的な煽り文句を遊びとして解体して読者との信頼を作る。"
                "中盤で業界の型（決められた導線）が顧客を縛ってきた事実を示し、最後に読者自身が選べる複数の申込方法を提示して選択を委ねる。"
            ),
            "article": (
                "記事の構成そのものを冒頭で読者に開示し、典型的な結論の型を一度なぞってみせてから、その型が議論を縛っていたことを示す。"
                "結論部では複数の解釈を併記し、どれを採るかを読者に委ねる形で締める。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.2), (0.1, 0.32), (0.2, 0.22), (0.35, 0.45), (0.5, 0.4), (0.6, 0.7), (0.68, 0.55), (0.8, 0.84), (0.9, 0.98), (0.95, 0.6), (1.0, 0.45)],
            "release": [(0.0, 0.3), (0.1, 0.45), (0.2, 0.35), (0.35, 0.2), (0.5, 0.3), (0.6, 0.05), (0.68, 0.25), (0.88, 0.05), (0.95, 0.6), (1.0, 0.8)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.4), (0.35, 0.5), (0.55, 0.53), (0.65, 0.8), (0.88, 0.85), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.05), (0.2, 0.1), (0.35, 0.3), (0.55, 0.48), (0.65, 0.55), (0.88, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.5), (0.2, 0.55), (0.35, 0.65), (0.6, 0.8), (0.8, 0.85), (0.88, 0.9), (1.0, 0.5)],
        },
    }
)

PATTERNS.append(
    {
        "num": 6,
        "slug": "ouroboros_loop",
        "title": "円環構造（結末が冒頭へ接続する輪）",
        "origin": "film",
        "summary": (
            "物語の結末の出来事が冒頭の出来事の原因となり、時間が閉じた輪を描く構造。読者は冒頭で先出しされた謎の場面を結果として読み進めるが、"
            "終盤でそれが始まりでもあったと知る。主体は輪を断ち切るのではなく引き受けることを選び、物語全体が自己完結した円環として再構成される。"
        ),
        "tags": ["円環構造", "自己原因", "冒頭回帰", "宿命"],
        "lineage": ["円環的叙述", "自己完結する因果の輪（因果の逆説）", "永劫回帰のモチーフ"],
        "nodes": [
            ("wanderer", "agent", "looped_protagonist", "輪の中を生きる主体", "冒頭の謎の場面の意味を知らないまま、自らの行動でその場面へ近づいていく。", 0.95, "LOOPED_PROTAGONIST"),
            ("stranger", "agent", "mysterious_stranger", "冒頭に現れる謎の人物", "主体に助言と物を与える。その正体は主体の未来の姿である。", 0.85, "MYSTERIOUS_STRANGER"),
            ("escape", "desire", "break_the_cycle", "輪を断ち切りたいという願い", "主体が宿命に抗う原動力。抗うほど冒頭の場面に近づく皮肉を生む。", 0.75, "BREAK_THE_CYCLE"),
            ("fate", "lack", "closed_fate", "閉じた宿命という欠落", "選択の余地がないように見える未来。主体の自由の欠如として感じられる。", 0.7, "CLOSED_FATE"),
            ("opening_scene", "symbol", "opening_scene", "冒頭に先出しされた謎の場面", "結末であると同時に始まりでもある場面。最後に内側から描き直される。", 0.8, "OPENING_SCENE"),
            ("relic", "symbol", "looping_object", "輪を巡る物", "誰が作ったのか分からない物。主体が受け取り、やがて自分で冒頭へ運ぶ。", 0.65, "LOOPING_OBJECT"),
        ],
        "edges": [
            ("stranger", "wanderer", "asymmetry", 0.85, 0.85, "謎の人物は全てを知っている", 0.06),
            ("relic", "opening_scene", "embodiment", 0.7, 0.0, "物が冒頭の場面を指し示す", 0.06),
            ("wanderer", "stranger", "dependency", 0.55, -0.4, "謎の人物の助言に従う", 0.3),
            ("wanderer", "relic", "pursuit", 0.6, 0.0, "物の由来を追う", 0.3),
            ("wanderer", "escape", "pursuit", 0.85, 0.0, "宿命からの脱出の希求", 0.45),
            ("fate", "wanderer", "power_gradient", 0.75, 0.7, "閉じた宿命が選択を縛る", 0.6),
            ("stranger", "wanderer", "betrayal", 0.7, 0.0, "助言こそが宿命への道だった", 0.85),
            ("opening_scene", "fate", "embodiment", 0.6, 0.0, "冒頭の場面が宿命を可視化する", 0.85),
        ],
        "beats": [
            ("cold_open", "hook", 0.05, 0.05, "結末の場面の先出し", "謎の人物と輪を巡る物が登場する緊迫した場面が、文脈なしに冒頭で提示される。", [], 0.1, 0.05, ["stranger", "relic", "opening_scene"], False),
            ("ordinary_time", "premise", 0.15, 0.05, "主体の日常", "冒頭の場面とは無関係に見える主体の穏やかな日常が描かれる。", ["cold_open"], -0.45, 0.02, ["wanderer"], False),
            ("relic_found", "setup", 0.3, 0.05, "物との出会い", "主体が謎の人物から物を受け取り、冒頭の場面との最初の接点が生まれる。", ["ordinary_time"], 0.08, 0.03, ["wanderer", "relic", "stranger"], False),
            ("guidance", "transition", 0.45, 0.05, "謎の人物の助言", "謎の人物の助言に従うことで主体は危機を逃れるが、輪の存在には気づかない。", ["relic_found"], -0.05, 0.03, ["wanderer", "stranger"], False),
            ("resistance", "local_optimization_limit", 0.6, 0.05, "抗うほど近づく", "宿命への抵抗が強まるほど、主体の行動が冒頭の場面の構図に近づいていく。", ["guidance"], 0.15, 0.04, ["wanderer", "fate", "escape"], True),
            ("doubt", "crisis", 0.72, 0.05, "正体への疑念", "主体が謎の人物の正体と助言の意図を疑い始め、孤立を深める。", ["resistance"], -0.07, 0.02, ["wanderer", "stranger"], False),
            ("loop_reveal", "structural_twist", 0.85, 0.05, "円環の露見", "謎の人物は未来の主体であり、物は主体自身が冒頭の場面へ運ぶと判明する。読者は人物より先に輪の全体を理解する。", ["doubt"], 0.35, 0.2, ["wanderer", "stranger", "relic", "opening_scene"], True),
            ("closing_loop", "costly_breakthrough", 0.95, 0.04, "輪を閉じる選択", "主体は宿命を断ち切るのではなく引き受け、未来の自分として冒頭の場面へ向かう。", ["loop_reveal"], -0.15, 0.03, ["wanderer", "fate"], False),
            ("return_to_start", "afterglow", 1.0, 0.02, "内側から見た冒頭", "冒頭と同じ場面が、今度は主体の内側から描き直され、輪が静かに閉じる。", ["closing_loop"], -0.03, 0.01, ["wanderer", "opening_scene"], False),
        ],
        "mechanisms": [
            ("cold_open_gap", "curiosity_gap", "先出しされた場面への問い", "文脈のない冒頭の場面が強い問いを生み、以降の全ての場面がその答え探しになる。", "結末の場面の先出し", "冒頭の場面を常に参照する持続的な注意", "情報ギャップ理論", 0.8, ["cold_open"], ["opening_scene"]),
            ("loop_error", "prediction_error", "円環の露見による予測誤差", "結果だと思っていた場面が原因でもあったと判明し、時間の前後の理解が覆る。", "謎の人物の正体の開示", "強い驚きと全編の再構成", "予測符号化理論（スキーマ違反）", 0.9, ["loop_reveal"], ["stranger", "relic"]),
            ("return_reward", "anticipation_reward", "冒頭回帰の予期と報酬", "主体の行動が冒頭の構図に近づくたびに回帰への予期が高まり、成就が満足をもたらす。", "抵抗と構図の一致", "円環が閉じる充足感", "報酬予測誤差理論", 0.7, ["resistance", "return_to_start"], ["opening_scene"]),
            ("open_loop", "zeigarnik", "閉じきらない輪の余韻", "輪が閉じても次の周回が暗示され、未完了感が再読を促す。", "冒頭と同じ場面での終幕", "再読欲求と長い余韻", "ツァイガルニク効果", 0.6, ["return_to_start"], ["wanderer"]),
        ],
        "anchor": "変えられない宿命の中で、それでも自分で選んだと言いたい人間の尊厳",
        "drivers": ["先出しされた場面の謎", "謎の人物の不可解な助言", "抗うほど近づく構図の皮肉"],
        "emotions": ["戦慄", "平穏", "好奇", "信頼", "焦燥", "疑念", "衝撃", "静かな諦観"],
        "template": (
            "冒頭で{{MYSTERIOUS_STRANGER}}と{{LOOPING_OBJECT}}が現れる{{OPENING_SCENE}}が先出しされる。{{LOOPED_PROTAGONIST}}は{{CLOSED_FATE}}に抗い"
            "{{BREAK_THE_CYCLE}}を求めるが、抗うほど冒頭の場面に近づいていく。謎の人物が未来の自分だと知ったとき、主体は{{LOOP_ACCEPTANCE}}を選び、輪は静かに閉じる。"
        ),
        "slots": [
            ("LOOPED_PROTAGONIST", "agent", "輪の中を生きる主体", [], ("時計職人の少年", "同じ失敗を繰り返す組織", "同じ問いに戻ってくる研究者"), True),
            ("MYSTERIOUS_STRANGER", "agent", "冒頭に現れる謎の人物（未来の主体）", ["顔や名前を終盤まで明かさない"], ("フードを被った老人", "数年後の自社の姿", "未来の読者自身"), True),
            ("OPENING_SCENE", "symbol", "冒頭に先出しされる場面", ["最後に同じ構図で内側から描き直す"], ("燃える時計塔", "数年後の導入完了の光景", "結論の一文"), True),
            ("BREAK_THE_CYCLE", "goal", "輪を断ち切りたい願い", [], ("運命を変えること", "失敗の繰り返しからの脱却", "堂々巡りの議論の終結"), True),
            ("CLOSED_FATE", "lack", "閉じた宿命", [], ("変えられない未来の光景", "繰り返される同じ課題", "同じ結論への回帰"), True),
            ("LOOPING_OBJECT", "symbol", "輪を巡る物", ["由来を最後まで説明しない"], ("止まった懐中時計", "受け継がれる設計書", "冒頭と結論の同じ図"), True),
            ("LOOP_ACCEPTANCE", "reversal", "輪を引き受ける選択", [], ("未来の自分として過去へ向かうこと", "失敗の型を資産として受け継ぐこと", "問いに戻ること自体を方法とすること"), True),
        ],
        "invariants": [
            "冒頭で結末の場面を文脈なしに先出しする",
            "主体の抵抗が冒頭の場面の構図へ近づく形で描く",
            "円環の露見は t≈0.85 に置き、謎の人物と物の由来を同時に明かす",
            "最後は冒頭と同じ場面を主体の内側から描き直して閉じる",
        ],
        "guide": {
            "manga": (
                "全24ページ想定。1〜2pで謎の人物と燃える塔のような場面を断ち切りの大ゴマで先出しし、3pからは穏やかな日常へ急転させる。"
                "中盤以降、主体の行動を冒頭のコマと同じ構図・同じアングルで少しずつ描き直していく。19p左ページ終端で主体が物を見つめる引きを作り、"
                "めくった20pで謎の人物の素顔（未来の主体）を明かす。最終ページは1pと同じ構図を主体の視点から描き、同じ台詞を逆の立場で言わせる。"
            ),
            "marketing": (
                "ページ冒頭で導入完了後の数年後の光景（結論）を先に見せ、その光景に至るまでの過程を時系列で語る。"
                "最後に冒頭の光景がこのページを読んでいるあなたの選択から始まると示し、最初の一歩としてのCTAを置く。"
            ),
            "article": (
                "冒頭で結論の一文を先に示し、そこに至る論証を展開した後、結論部で同じ一文を再掲して意味の変化を示す。"
                "読者が冒頭へ戻って読み直したくなるよう、序盤の細部が結論の伏線になる構成にする。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.7), (0.05, 0.82), (0.15, 0.35), (0.3, 0.45), (0.45, 0.4), (0.6, 0.62), (0.72, 0.55), (0.85, 0.9), (0.95, 0.75), (1.0, 0.72)],
            "release": [(0.0, 0.05), (0.05, 0.02), (0.15, 0.45), (0.3, 0.3), (0.45, 0.35), (0.6, 0.1), (0.72, 0.25), (0.85, 0.05), (0.95, 0.4), (1.0, 0.35)],
            "audience_knowledge": [(0.0, 0.3), (0.05, 0.38), (0.2, 0.4), (0.45, 0.48), (0.8, 0.6), (0.85, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.05, 0.12), (0.2, 0.15), (0.45, 0.3), (0.7, 0.45), (0.85, 0.55), (1.0, 0.9)],
            "cognitive_load": [(0.0, 0.6), (0.05, 0.7), (0.15, 0.45), (0.45, 0.55), (0.7, 0.65), (0.85, 0.85), (1.0, 0.7)],
        },
    }
)

PATTERNS.append(
    {
        "num": 7,
        "slug": "amnesia_backward_reconstruction",
        "title": "記憶の欠落からの逆算",
        "origin": "film",
        "summary": (
            "記憶を失った主体が、現在に残された痕跡から過去の自分を逆算して再構成する構造。手がかりが増えるほど過去の自分が望ましくない人物だったことが示唆され、"
            "主体は取り戻すべき自分と今の自分のどちらを選ぶかを迫られる。真相は他者からではなく、主体自身の手で掘り当てられる。"
        ),
        "tags": ["記憶の欠落", "逆算的探索", "アイデンティティ", "自己対決"],
        "lineage": ["記憶喪失の謎解き劇", "結果から原因へ遡る逆向きの叙述", "自己同一性をめぐる哲学劇"],
        "nodes": [
            ("present_self", "agent", "present_self", "記憶を失った今の主体", "痕跡だけを頼りに自分が誰かを再構成しようとする。善良であろうとする今の意志を持つ。", 0.95, "PRESENT_SELF"),
            ("past_self", "agent", "forgotten_self", "忘れられた過去の自分", "痕跡の向こうに浮かび上がるもう一人の主体。今の主体と価値観が対立する。", 0.8, "FORGOTTEN_SELF"),
            ("witness", "agent", "knowing_witness", "過去を知る人物", "主体の過去を知り、都合の良い過去を吹き込むことで主体を操ろうとする。", 0.7, "KNOWING_WITNESS"),
            ("identity", "desire", "recovered_identity", "取り戻したい自分", "主体の行動原理。それが望ましい自分であるという前提に支えられている。", 0.85, "RECOVERED_IDENTITY"),
            ("memory_void", "lack", "memory_void", "記憶の空白", "主体を他者の言葉に依存させる根源的な欠落。", 0.75, "MEMORY_VOID"),
            ("trace", "symbol", "residual_trace", "身体と周囲に残された痕跡", "傷、手帳、習慣、鍵。過去の自分を指し示す唯一の物証。", 0.7, "RESIDUAL_TRACE"),
        ],
        "edges": [
            ("memory_void", "present_self", "power_gradient", 0.7, 0.7, "空白が主体を他者に依存させる", 0.0),
            ("present_self", "identity", "pursuit", 0.9, 0.0, "自己の再構成", 0.08),
            ("witness", "present_self", "asymmetry", 0.8, 0.8, "過去を知る者の優位", 0.2),
            ("present_self", "trace", "pursuit", 0.55, 0.0, "痕跡の収集", 0.2),
            ("present_self", "witness", "dependency", 0.6, -0.4, "語られる過去への依存", 0.3),
            ("trace", "past_self", "embodiment", 0.75, 0.0, "痕跡が過去の自分を体現する", 0.35),
            ("witness", "present_self", "betrayal", 0.65, 0.0, "都合の良い過去の吹き込み", 0.6),
            ("past_self", "present_self", "opposition", 0.85, 0.0, "過去の自分との対決", 0.75),
        ],
        "beats": [
            ("awakening", "hook", 0.08, 0.05, "記憶のない目覚め", "主体が見知らぬ場所で目覚め、身体に残る痕跡だけが過去への手がかりとなる。", [], 0.12, 0.03, ["present_self", "memory_void", "trace"], False),
            ("first_traces", "premise", 0.2, 0.06, "痕跡と仮の自己像", "痕跡を集めて仮の自己像を組み立て、主体は善良な過去を想像する。", ["awakening"], -0.22, 0.05, ["present_self", "trace", "identity"], False),
            ("witness_story", "setup", 0.35, 0.05, "語られる過去", "過去を知る人物が現れ、主体に心地よい過去を語って聞かせる。", ["first_traces"], 0.1, 0.05, ["witness", "present_self"], False),
            ("contradiction", "escalation", 0.48, 0.05, "痕跡と語りの矛盾", "痕跡が語られた過去と食い違い、主体の中に疑念が芽生える。", ["witness_story"], -0.05, 0.05, ["present_self", "trace", "witness"], False),
            ("dark_lead", "escalation", 0.6, 0.05, "暗い手がかり", "痕跡が過去の自分の暗い行為を指し始め、主体は探索を止められなくなる。", ["contradiction"], 0.22, 0.05, ["present_self", "past_self"], False),
            ("self_revelation", "revelation", 0.75, 0.05, "自己の露見", "過去の自分こそが事件の加害者だったと、主体自身が痕跡から掘り当てる。", ["dark_lead"], 0.23, -0.15, ["present_self", "past_self", "trace"], True),
            ("collapse", "crisis", 0.8, 0.04, "自己像の崩壊", "取り戻すべきだった自分が否定され、主体は拠り所を失う。", ["self_revelation"], -0.35, 0.02, ["present_self", "identity"], False),
            ("chosen_self", "costly_breakthrough", 0.9, 0.05, "今の自分の選択", "主体は過去を償う代償を引き受け、今の自分として生きることを選ぶ。", ["collapse"], 0.15, 0.03, ["present_self", "past_self"], True),
            ("new_name", "afterglow", 1.0, 0.02, "新しい名前", "痕跡を消さずに受け入れたまま、主体は新しい名前で歩き出す。", ["chosen_self"], -0.45, 0.01, ["present_self", "trace"], False),
        ],
        "mechanisms": [
            ("void_gap", "curiosity_gap", "空白への問い", "主体と読者が同じ空白を抱えて出発し、痕跡の一つ一つが答えへの期待を生む。", "記憶のない目覚め", "痕跡への強い注意", "情報ギャップ理論", 0.8, ["awakening", "first_traces"], ["memory_void", "trace"]),
            ("blank_projection", "identity_projection", "白紙の主体への投影", "過去を持たない主体に読者が自分を重ね、自己像の再構成を追体験する。", "仮の自己像の構築", "主体の選択の自分事化", "同一化理論", 0.7, ["first_traces"], ["present_self"]),
            ("self_judgment", "moral_judgment", "自己への道徳的審判", "主体が自分の過去を裁く立場に置かれ、読者も同じ判断を迫られる。", "過去の自分の加害の露見", "赦しと責任をめぐる省察", "道徳基盤理論", 0.8, ["self_revelation", "chosen_self"], ["past_self", "present_self"]),
            ("trace_dissonance", "cognitive_dissonance", "語りと痕跡の不協和", "心地よい語りと不穏な痕跡の食い違いが、真相を求める動機を強める。", "痕跡と語りの矛盾", "語り手への疑念と自力探索の決意", "認知的不協和理論", 0.65, ["contradiction"], ["witness", "trace"]),
        ],
        "anchor": "過去の自分を知るのが怖いという、誰もが心の奥に持つ恐れ",
        "drivers": ["痕跡が一つ増えるごとに変わる自己像", "過去を知る人物への依存と疑念", "望ましくない真相への予感"],
        "emotions": ["不安", "希望", "安心", "疑念", "戦慄", "絶望", "決意", "静かな受容"],
        "template": (
            "{{MEMORY_VOID}}を抱えて目覚めた{{PRESENT_SELF}}は、{{RESIDUAL_TRACE}}を頼りに{{RECOVERED_IDENTITY}}を再構成しようとする。"
            "{{KNOWING_WITNESS}}の語る過去は痕跡と食い違い、主体は自らの手で{{FORGOTTEN_SELF}}の真実を掘り当てる。拠り所を失った主体は"
            "{{ATONEMENT_COST}}を引き受け、今の自分として生きることを選ぶ。"
        ),
        "slots": [
            ("PRESENT_SELF", "agent", "記憶を失った今の主体", [], ("病室で目覚めた青年", "前任者の記録を失った新任担当", "前提を忘れて読み始める読者"), True),
            ("FORGOTTEN_SELF", "agent", "忘れられた過去の自分", ["今の主体と価値観を対立させる"], ("かつての冷酷な傭兵", "過去に強引な施策を行った自社", "かつて誤った主張をした筆者"), True),
            ("KNOWING_WITNESS", "agent", "過去を知り操ろうとする人物", [], ("婚約者を名乗る女", "都合の良い過去実績を語る代理店", "都合の良い解釈を広めた論者"), True),
            ("RECOVERED_IDENTITY", "goal", "取り戻したい自分", [], ("優しかったはずの自分", "顧客に愛されていたはずのブランド", "正しかったはずの理論"), True),
            ("MEMORY_VOID", "lack", "記憶の空白", [], ("事故以前の記憶", "引き継がれなかった経緯", "参照されなくなった原典"), True),
            ("RESIDUAL_TRACE", "evidence", "残された痕跡", ["序盤から複数の痕跡を少しずつ提示する"], ("手の甲の古傷", "解約顧客の記録", "古い実験ノート"), True),
            ("ATONEMENT_COST", "cost", "今の自分を選ぶための代償", [], ("過去の罪の告白", "過去の不手際の公表と補償", "過去の主張の撤回"), True),
        ],
        "invariants": [
            "真相は他者の語りではなく主体自身が痕跡から掘り当てる",
            "過去を知る人物の語りは痕跡と食い違うよう設計する",
            "自己の露見は t≈0.75 に置き、直後に自己像の崩壊を置く",
            "結末では過去を消さずに受け入れたうえで今の自分を選ばせる",
        ],
        "guide": {
            "manga": (
                "全24ページ想定。1pは主体の目覚めを真俯瞰の大ゴマで描き、身体の痕跡を超クローズアップで連続提示する。痕跡を見つけるたびに同じ構図の小ゴマを使い、読者に収集の手触りを与える。"
                "17p左ページ終端で主体が最後の痕跡に手を伸ばす引きを作り、めくった18p右ページで過去の自分の行為を断ち切りの大ゴマで描く。"
                "最終ページは鏡の前で痕跡を隠さずに歩き出す主体を引きの構図で描く。"
            ),
            "marketing": (
                "前任者の記録を失った担当者が、残されたデータの痕跡から自社の過去の施策を再構成するストーリーとして語る。"
                "不都合な過去（顧客を失った理由）を自ら掘り当てて公表し、補償と改善を約束する誠実さを価値提案の中核に置く。"
            ),
            "article": (
                "前提を知らない読者と同じ立場から、残された資料の断片を一つずつ検証して過去の経緯を再構成する。"
                "かつての主張の誤りを自ら認めたうえで、新しい立場からの実践的な提言で締める。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.5), (0.08, 0.62), (0.2, 0.4), (0.35, 0.55), (0.48, 0.5), (0.6, 0.72), (0.75, 0.95), (0.8, 0.6), (0.9, 0.75), (1.0, 0.3)],
            "release": [(0.0, 0.05), (0.2, 0.3), (0.35, 0.15), (0.48, 0.3), (0.6, 0.1), (0.75, 0.02), (0.8, 0.35), (0.9, 0.15), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.05), (0.2, 0.2), (0.35, 0.35), (0.5, 0.5), (0.6, 0.62), (0.75, 0.85), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.02), (0.2, 0.12), (0.35, 0.2), (0.5, 0.3), (0.6, 0.36), (0.75, 0.9), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.75), (0.2, 0.6), (0.35, 0.62), (0.5, 0.68), (0.6, 0.7), (0.75, 0.8), (0.9, 0.6), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 8,
        "slug": "layered_reversals",
        "title": "多重反転（反転の反転の反転）",
        "origin": "film",
        "summary": (
            "反転が一度で終わらず、反転の反転、さらにその反転へと三層に重なる構造。読者が新しい真相に適応するたびに次の層が剥がされ、"
            "最終的には最初の見かけに近い地点へ着地しながら、その意味だけが決定的に変わっている。各層で同じ合図が異なる意味を帯びる。"
        ),
        "tags": ["多重反転", "どんでん返し", "読み合い", "真相の層"],
        "lineage": ["どんでん返しの連鎖", "信用詐欺劇（騙し合いの劇）", "層状の真相構造"],
        "nodes": [
            ("mark", "agent", "apparent_victim", "罠にかけられる標的に見える人物", "第一層では被害者、第二層では反撃者、第三層では最初から盤上の駒だったと意味を変え続ける。", 0.8, "APPARENT_VICTIM"),
            ("schemer", "agent", "apparent_schemer", "罠を仕掛ける策士に見える人物", "標的を出し抜いたと信じているが、自らもより大きな設計の内側にいる。", 0.85, "APPARENT_SCHEMER"),
            ("architect", "agent", "hidden_architect", "全てを設計していた見えない人物", "誰も疑わなかった位置に最初から立ち、三つの層すべてを配置していた。", 0.75, "HIDDEN_ARCHITECT"),
            ("prize", "desire", "contested_prize", "全員が奪い合う獲物", "金、地位、秘密。層が変わるたびに本当の価値が書き換わる。", 0.8, "CONTESTED_PRIZE"),
            ("blind_spot", "lack", "perceptual_blind_spot", "全員に共通する認知の死角", "自分は騙す側だという確信。最後の層はこの死角の上に築かれている。", 0.7, "PERCEPTUAL_BLIND_SPOT"),
            ("signal", "symbol", "recurring_signal", "各層で意味を変える合図", "同じ仕草や小物が、層ごとに約束・裏切り・支配の印として読み替えられる。", 0.65, "RECURRING_SIGNAL"),
        ],
        "edges": [
            ("schemer", "prize", "pursuit", 0.85, 0.0, "獲物への執着", 0.15),
            ("mark", "prize", "pursuit", 0.6, 0.0, "守るべき獲物", 0.15),
            ("schemer", "mark", "asymmetry", 0.7, 0.7, "罠の全貌を策士だけが知る", 0.15),
            ("mark", "schemer", "dependency", 0.6, -0.3, "標的は策士を信じる", 0.3),
            ("blind_spot", "mark", "power_gradient", 0.6, 0.5, "死角が判断を誤らせる", 0.3),
            ("schemer", "mark", "betrayal", 0.75, 0.0, "信頼の裏切り", 0.45),
            ("signal", "blind_spot", "embodiment", 0.65, 0.0, "合図が死角を隠す", 0.45),
            ("architect", "schemer", "power_gradient", 0.85, 0.85, "見えない設計者が策士を操る", 0.65),
            ("architect", "mark", "asymmetry", 0.6, 0.6, "標的の行動まで予期されていた", 0.85),
        ],
        "beats": [
            ("setup_con", "hook", 0.15, 0.05, "罠の構図", "策士が標的を狙う構図が提示され、読者は騙す側の視点に立つ。", [], 0.1, 0.05, ["schemer", "mark", "prize"], False),
            ("lure", "premise", 0.3, 0.06, "誘い込み", "標的が策士を信頼し、罠の中心へ自ら歩み寄っていく。", ["setup_con"], -0.04, 0.03, ["mark", "schemer"], False),
            ("first_reversal", "structural_twist", 0.45, 0.04, "第一の反転", "標的は最初から罠に気づいていたと判明し、騙す側と騙される側が入れ替わる。", ["lure"], 0.25, 0.15, ["mark", "schemer", "signal"], True),
            ("regroup", "transition", 0.52, 0.04, "新しい真相への適応", "読者が新しい構図に適応し、標的の反撃を見守り始める。", ["first_reversal"], -0.17, 0.03, ["mark"], False),
            ("second_reversal", "structural_twist", 0.65, 0.04, "第二の反転", "標的の反撃すら策士の想定内だったと判明し、合図の意味が再び書き換わる。", ["regroup"], 0.3, 0.07, ["schemer", "mark", "signal"], True),
            ("false_safety", "false_victory", 0.72, 0.04, "偽の決着", "策士の勝利が確定したかに見え、読者も今度こそ真相に辿り着いたと安堵する。", ["second_reversal"], -0.25, 0.03, ["schemer", "prize"], False),
            ("final_reversal", "revelation", 0.85, 0.05, "第三の反転", "全てを設計していたのは誰も疑わなかった人物だったと明かされ、三つの層が一つの設計図として連結する。", ["false_safety"], 0.39, 0.15, ["architect", "schemer", "mark", "blind_spot"], True),
            ("landing", "payoff", 0.93, 0.04, "見かけへの着地", "表面上は最初の見かけに近い結末へ着地するが、全ての場面の意味が変わっている。", ["final_reversal"], -0.37, 0.04, ["architect", "prize"], False),
            ("last_signal", "afterglow", 1.0, 0.02, "合図の最後の意味", "最後にもう一度だけ合図が示され、読者だけがその本当の意味を知っている。", ["landing"], -0.3, 0.02, ["signal"], False),
        ],
        "mechanisms": [
            ("stacked_error", "prediction_error", "反転の連鎖による予測誤差", "読者が新しい真相に適応するたびに予測が裏切られ、驚きが層ごとに更新される。", "三度の反転", "読みの更新そのものの快感", "予測符号化理論（スキーマ違反）", 0.9, ["first_reversal", "second_reversal", "final_reversal"], ["schemer", "mark", "architect"]),
            ("next_layer", "anticipation_reward", "次の層への予期", "一度反転を経験した読者は次の反転を予期し、安堵の場面にも疑いを向けるようになる。", "偽の決着と適応の間", "能動的な読み合いへの参加", "報酬予測誤差理論", 0.75, ["regroup", "false_safety"], ["signal"]),
            ("setup_gap", "curiosity_gap", "罠の全貌への問い", "冒頭の罠の構図が全貌を伏せたまま示され、誰が何を知っているのかという問いを生む。", "罠の構図の提示", "人物の知識状態への注意", "情報ギャップ理論", 0.65, ["setup_con"], ["schemer", "prize"]),
            ("final_peak", "peak_end", "最終反転と着地の記憶", "最大の反転と静かな着地が全体の評価を決め、再読時の発見を約束する。", "第三の反転と見かけへの着地", "再読欲求と強い印象", "ピーク・エンドの法則", 0.7, ["final_reversal", "landing"], ["architect"]),
        ],
        "anchor": "自分は騙す側だと信じていたい、知性への自負とその脆さ",
        "drivers": ["誰が何を知っているのかの絶え間ない更新", "同じ合図の意味の変化", "安堵の場面に潜む次の反転の気配"],
        "emotions": ["期待", "共犯感", "驚き", "適応", "驚愕", "安堵", "戦慄", "感嘆"],
        "template": (
            "{{APPARENT_SCHEMER}}は{{CONTESTED_PRIZE}}を狙って{{APPARENT_VICTIM}}に罠を仕掛ける。第一の反転で標的が罠に気づいていたと判明し、"
            "第二の反転でそれすら策士の想定内だったと分かる。最後に{{HIDDEN_ARCHITECT}}が全員の{{PERCEPTUAL_BLIND_SPOT}}を利用していたと明かされ、"
            "{{RECURRING_SIGNAL}}は{{FINAL_MEANING}}へ読み替えられる。"
        ),
        "slots": [
            ("APPARENT_VICTIM", "agent", "標的に見える人物", ["層ごとに立場を反転させる"], ("世間知らずの令嬢", "値引きを迫る取引先", "通説に従う実務家"), True),
            ("APPARENT_SCHEMER", "agent", "策士に見える人物", [], ("凄腕の詐欺師", "強気の交渉担当", "通説への反論者"), True),
            ("HIDDEN_ARCHITECT", "agent", "全てを設計していた人物", ["序盤から画面内に置き、疑われない役割を与える"], ("執事", "最初の相談相手だった顧客", "両論の前提を作った原典"), True),
            ("CONTESTED_PRIZE", "goal", "奪い合われる獲物", [], ("一族の遺産", "大型契約", "議論の主導権"), True),
            ("PERCEPTUAL_BLIND_SPOT", "lack", "全員に共通する死角", [], ("自分は騙す側だという確信", "相手より情報を持っているという過信", "反論している自分は正しいという思い込み"), True),
            ("RECURRING_SIGNAL", "symbol", "層ごとに意味を変える合図", ["最初に提示した意味を三回書き換える"], ("指輪を回す仕草", "契約書の一文", "繰り返し引用される一節"), True),
            ("FINAL_MEANING", "reversal", "合図の最後の意味", [], ("最初から決まっていた勝者の印", "当初から用意されていた本命の提案", "両論を統合する新しい解釈"), True),
        ],
        "invariants": [
            "反転は三層とし、各層で同じ合図の意味を書き換える",
            "反転と反転の間に、読者が新しい真相へ適応する短い凪を置く",
            "真の設計者は序盤から画面内に置き、疑われない役割を与える",
            "最後は見かけ上は最初の構図に近い地点へ着地させる",
        ],
        "guide": {
            "manga": (
                "全32ページ想定。合図となる仕草を1pの超クローズアップで提示し、反転ごとに同じ構図・同じサイズのコマで再提示して意味の変化を読ませる。"
                "第一の反転は13p右ページ冒頭、第二の反転は20p右ページ冒頭、第三の反転は27p右ページ見開きと、いずれもめくり直後に置く。"
                "各反転の直前の左ページ終端は人物の笑みのアップで締め、反転のたびに読者に前の頁を読み返させる。"
            ),
            "marketing": (
                "比較検討のLPで、競合の主張への反論（第一の反転）、反論への再反論の想定（第二の反転）、両者が見落とす本質的な前提（第三の反転）の三層で論点を剥がしていく。"
                "最後に提案自体が最初の素朴な要望にこそ最も忠実だったと示して着地させる。"
            ),
            "article": (
                "通説、通説への反論、反論の限界、そして両者を包む前提の見直しという三層の反転で論じる。"
                "結論では冒頭の素朴な問いに戻り、同じ答えが異なる理由で正しかったことを示す。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.3), (0.15, 0.42), (0.3, 0.38), (0.45, 0.7), (0.52, 0.45), (0.65, 0.82), (0.72, 0.5), (0.85, 0.97), (0.93, 0.55), (1.0, 0.35)],
            "release": [(0.0, 0.05), (0.15, 0.15), (0.3, 0.2), (0.45, 0.1), (0.52, 0.4), (0.65, 0.08), (0.72, 0.35), (0.85, 0.05), (0.93, 0.65), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.1), (0.4, 0.25), (0.45, 0.45), (0.6, 0.48), (0.65, 0.62), (0.8, 0.66), (0.85, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.6), (0.4, 0.62), (0.45, 0.64), (0.6, 0.75), (0.65, 0.78), (0.8, 0.88), (0.85, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.4), (0.3, 0.5), (0.45, 0.65), (0.6, 0.7), (0.65, 0.78), (0.85, 0.9), (1.0, 0.45)],
        },
    }
)

PATTERNS.append(
    {
        "num": 9,
        "slug": "antagonist_perspective_inversion",
        "title": "敵対者視点による価値反転",
        "origin": "drama",
        "summary": (
            "前半は主人公の視点で敵対者を絶対悪として描き、中盤で視点を敵対者側へ切り替えて同じ出来事を語り直す構造。"
            "敵対者の論理と喪失が見えたとき読者の道徳的判断は反転し、主人公の正義が別の誰かの悲劇の上に成り立っていたことが明らかになる。"
            "読者だけがそれを知ったまま、主人公は最後の対決へ向かう。"
        ),
        "tags": ["視点転換", "価値反転", "敵対者の論理", "道徳的再評価"],
        "lineage": ["同一事象の多視点的語り直し", "悪役の動機の再評価", "道徳的反転の悲劇"],
        "nodes": [
            ("hero", "agent", "righteous_protagonist", "正義を疑わない主人公", "共同体を守る側に立ち、敵対者を倒すことに一点の迷いもない。", 0.9, "RIGHTEOUS_PROTAGONIST"),
            ("antagonist", "agent", "demonized_antagonist", "絶対悪として描かれる敵対者", "前半は脅威としてのみ描かれるが、視点転換後はかつて共同体に全てを奪われた者として現れる。", 0.85, "DEMONIZED_ANTAGONIST"),
            ("community", "agent", "protected_community", "主人公が守る共同体", "主人公を支え称えるが、敵対者を生んだ過去の罪を隠している。", 0.7, "PROTECTED_COMMUNITY"),
            ("justice", "desire", "righteous_goal", "主人公の正義", "敵対者の打倒による平和。視点転換後はその土台が問い直される。", 0.8, "RIGHTEOUS_GOAL"),
            ("wound", "lack", "antagonist_wound", "敵対者の喪失", "共同体がかつて敵対者から奪ったもの。敵対者の全ての行動の起点である。", 0.8, "ANTAGONIST_WOUND"),
            ("origin_mark", "symbol", "shared_origin", "両者が同じ出来事から生まれたことを示す印", "主人公と敵対者が同じ場所・同じ出来事に由来することを示す物証。", 0.6, "SHARED_ORIGIN"),
        ],
        "edges": [
            ("hero", "justice", "pursuit", 0.85, 0.0, "正義の遂行", 0.1),
            ("hero", "antagonist", "opposition", 0.9, 0.0, "絶対悪との対決", 0.1),
            ("community", "hero", "dependency", 0.65, -0.3, "主人公に守られる共同体", 0.25),
            ("community", "antagonist", "power_gradient", 0.7, 0.7, "共同体がかつて敵対者を排除した", 0.5),
            ("wound", "antagonist", "embodiment", 0.75, 0.0, "喪失が敵対者を形作った", 0.5),
            ("antagonist", "hero", "asymmetry", 0.65, 0.6, "敵対者だけが過去の真相を知る", 0.5),
            ("origin_mark", "wound", "embodiment", 0.6, 0.0, "共有された起源が喪失を証明する", 0.8),
            ("origin_mark", "justice", "embodiment", 0.5, 0.0, "正義の土台に刻まれた起源", 0.8),
        ],
        "beats": [
            ("clear_evil", "hook", 0.1, 0.05, "絶対悪の襲来", "敵対者が共同体を襲い、主人公の視点から疑いようのない悪として提示される。", [], 0.15, 0.02, ["antagonist", "community"], False),
            ("rally", "premise", 0.25, 0.06, "正義の結束", "共同体が主人公の下に結束し、敵対者を倒すことが正義として確立される。", ["clear_evil"], 0.1, 0.02, ["hero", "community", "justice"], False),
            ("hunt", "false_victory", 0.4, 0.05, "追い詰める高揚", "主人公が敵対者を追い詰め、勝利が目前に迫る高揚が頂点に達する。", ["rally"], 0.15, 0.02, ["hero", "antagonist"], True),
            ("perspective_switch", "structural_twist", 0.5, 0.05, "視点の切り替え", "語りが敵対者の側へ移り、同じ出来事が奪われた者の視点から語り直される。", ["hunt"], -0.3, 0.3, ["antagonist", "community"], True),
            ("hidden_wound", "revelation", 0.6, 0.05, "隠された喪失", "共同体がかつて敵対者から全てを奪った事実と、敵対者の論理が開示される。", ["perspective_switch"], 0.1, 0.08, ["antagonist", "wound", "community"], False),
            ("irony_march", "escalation", 0.7, 0.06, "知らずに進む主人公", "読者だけが真相を知ったまま、主人公は正義を疑わず最後の対決へ向かう。", ["hidden_wound"], 0.07, 0.04, ["hero", "justice"], False),
            ("confrontation", "climax", 0.8, 0.05, "真相との衝突", "決戦の最中に主人公が共有された起源の印を見つけ、自らの正義の土台を知る。", ["irony_march"], 0.34, -0.4, ["hero", "antagonist", "origin_mark"], True),
            ("reckoning", "costly_breakthrough", 0.9, 0.05, "罪の引き受け", "主人公が共同体の過去の罪を引き受け、勝利ではなく償いを選ぶ。", ["confrontation"], -0.36, 0.03, ["hero", "community", "wound"], False),
            ("shared_ground", "afterglow", 1.0, 0.02, "起源の地で", "主人公と敵対者の起源の場所に、二人分の印が並べて置かれる。", ["reckoning"], -0.3, 0.01, ["origin_mark", "hero"], False),
        ],
        "mechanisms": [
            ("moral_flip", "moral_judgment", "道徳判断の反転", "前半で下した道徳判断が視点転換によって覆り、読者自身の判断の基準が問われる。", "絶対悪の提示と視点転換", "判断の保留と自己省察", "道徳基盤理論", 0.85, ["clear_evil", "confrontation"], ["hero", "antagonist"]),
            ("villain_empathy", "empathy_transfer", "敵対者への共感移入", "奪われた者の視点に立つことで、敵対者への共感が生まれる。", "視点の切り替えと喪失の開示", "敵対者の痛みの自分事化", "共感の視点取得理論", 0.8, ["perspective_switch", "hidden_wound"], ["antagonist", "wound"]),
            ("righteous_irony", "dramatic_irony", "正義の劇的アイロニー", "読者だけが真相を知ったまま主人公の正義を見守ることで、焦燥と悲しみが生まれる。", "主人公の知らないまま進む決戦", "主人公に真相を伝えたくなる焦燥", "劇的アイロニー", 0.75, ["irony_march"], ["hero"]),
            ("justice_dissonance", "cognitive_dissonance", "正義と罪の不協和", "応援してきた主人公の正義と共同体の罪が両立しないことが、読者に解消を迫る。", "隠された喪失の開示", "償いという結末の受容", "認知的不協和理論", 0.7, ["hidden_wound", "reckoning"], ["community", "justice"]),
        ],
        "anchor": "自分の正しさが誰かの犠牲の上に成り立っていたかもしれないという、静かな恐れ",
        "drivers": ["視点が切り替わる瞬間の落差", "同じ出来事の二つの語り", "読者だけが知る真相"],
        "emotions": ["怒り", "連帯", "高揚", "動揺", "悲しみ", "焦燥", "衝撃", "鎮魂"],
        "template": (
            "{{RIGHTEOUS_PROTAGONIST}}は{{PROTECTED_COMMUNITY}}を守るため、{{DEMONIZED_ANTAGONIST}}を倒すという{{RIGHTEOUS_GOAL}}を掲げる。"
            "視点が敵対者へ切り替わると、共同体がかつて奪った{{ANTAGONIST_WOUND}}が明かされる。決戦の中で{{SHARED_ORIGIN}}を見つけた主人公は、"
            "勝利ではなく{{COLLECTIVE_GUILT}}を引き受けることを選ぶ。"
        ),
        "slots": [
            ("RIGHTEOUS_PROTAGONIST", "agent", "正義を疑わない主人公", [], ("村を守る若き剣士", "業界標準を推進する担当者", "通説を擁護する論者"), True),
            ("DEMONIZED_ANTAGONIST", "agent", "絶対悪として描かれる敵対者", ["前半は内面を一切描かない"], ("村を襲う異形の者", "業界から排除された競合", "異端とされた学説の提唱者"), True),
            ("PROTECTED_COMMUNITY", "agent", "主人公が守る共同体", [], ("山間の村", "既存の業界団体", "主流派の学界"), True),
            ("RIGHTEOUS_GOAL", "goal", "主人公の掲げる正義", [], ("村の平和", "業界の健全化", "学説の正統性の維持"), True),
            ("ANTAGONIST_WOUND", "lack", "敵対者が奪われたもの", ["視点転換まで伏せる"], ("焼かれた故郷", "不当に奪われた市場", "握り潰された研究成果"), True),
            ("SHARED_ORIGIN", "symbol", "両者の共通の起源を示す印", ["序盤に主人公側の持ち物として一度描く"], ("同じ紋の刻まれた刀", "同じ創業者の署名", "同じ師の書いた序文"), True),
            ("COLLECTIVE_GUILT", "cost", "引き受ける共同体の罪", [], ("村の罪の告白と償い", "過去の排除の公式な謝罪", "過去の不当な評価の撤回"), True),
        ],
        "invariants": [
            "前半は敵対者の内面を描かず、脅威としてのみ提示する",
            "視点転換は t≈0.5 に置き、同じ出来事を別の視点で語り直す",
            "読者だけが真相を知った状態で主人公を最後の対決へ向かわせる",
            "結末では勝利ではなく償いを選ばせる",
        ],
        "guide": {
            "manga": (
                "全32ページ想定。前半16pは主人公の視点で、敵対者は影や仮面で顔を見せずアオリの威圧的な構図で描く。"
                "16p左ページ終端で主人公が刃を振り下ろす直前の引きを作り、めくった17p右ページから敵対者の視点に切り替え、同じ場面を逆アングルで描き直す。"
                "後半は主人公側と敵対者側の場面を交互に置き、最終決戦の見開きで共有された起源の印を中央に配置する。"
            ),
            "marketing": (
                "業界の常識として排除されてきた方法（敵対者）を、まず一般的な批判の視点から紹介し、中盤で排除された側の論理とデータを丁寧に語り直す。"
                "最後に両者の共通の起源（同じ顧客課題）を示し、対立ではなく統合の提案としてCTAを置く。"
            ),
            "article": (
                "前半で主流の立場から異端の説を批判的に紹介し、後半でその説の提唱者の論理と排除の経緯を同じ資料から語り直す。"
                "結論で両者に共通する前提を示し、評価の基準そのものを見直す提言で締める。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.25), (0.1, 0.4), (0.25, 0.55), (0.4, 0.75), (0.5, 0.45), (0.6, 0.55), (0.7, 0.62), (0.8, 0.96), (0.9, 0.6), (1.0, 0.3)],
            "release": [(0.0, 0.05), (0.25, 0.15), (0.4, 0.3), (0.5, 0.1), (0.6, 0.2), (0.7, 0.1), (0.8, 0.05), (0.9, 0.45), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.1), (0.25, 0.25), (0.45, 0.3), (0.5, 0.6), (0.7, 0.75), (0.8, 0.85), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.25, 0.22), (0.45, 0.28), (0.5, 0.3), (0.7, 0.35), (0.8, 0.85), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.35), (0.25, 0.45), (0.4, 0.5), (0.5, 0.75), (0.7, 0.7), (0.8, 0.85), (1.0, 0.4)],
        },
    }
)

PATTERNS.append(
    {
        "num": 10,
        "slug": "branching_to_necessity",
        "title": "分岐と収束（可能性の必然化）",
        "origin": "literature",
        "summary": (
            "主体の前に複数の可能性（選択の分岐）が開かれ、物語はそれぞれの分岐を試行して比較する形で進む構造。"
            "どの分岐でも同じ核心の喪失に行き着くと判明したとき、主体は結果を変えるのではなく意味を変える選択へ至り、可能性の束が一つの必然へ収束する。"
            "試行のたびに希望が削られ、最後の選択だけが静かな解放をもたらす。"
        ),
        "tags": ["分岐", "やり直し", "必然", "選択の意味"],
        "lineage": ["分岐する時間の物語", "やり直しを繰り返す試行構造", "運命と自由意志の劇"],
        "nodes": [
            ("chooser", "agent", "branching_protagonist", "分岐をやり直す主体", "大切な人を救うために何度も分岐点へ戻り、別の選択を試み続ける。", 0.95, "BRANCHING_PROTAGONIST"),
            ("beloved", "agent", "threatened_beloved", "失われ続ける大切な人", "どの分岐でも失われる存在。実は自らの運命を知りながら主体に黙っていた。", 0.8, "THREATENED_BELOVED"),
            ("arbiter", "agent", "branch_arbiter", "分岐の規則を体現する存在", "時間や運命の番人。主体の試行を拒まないが、規則の核心だけは変えさせない。", 0.75, "BRANCH_ARBITER"),
            ("altered", "desire", "altered_outcome", "結果を変えたいという願い", "主体の全ての試行の動機。最後に意味を変える願いへと変容する。", 0.85, "ALTERED_OUTCOME"),
            ("inevitable", "lack", "inevitable_loss", "避けられない喪失", "どの分岐でも同じ形で訪れる核心の喪失。", 0.8, "INEVITABLE_LOSS"),
            ("crossroads", "symbol", "branching_point", "全ての分岐が始まる地点", "主体が何度も戻る場所。最後に花を手向ける場所へ意味を変える。", 0.65, "BRANCHING_POINT"),
        ],
        "edges": [
            ("chooser", "altered", "pursuit", 0.9, 0.0, "結果を変える試行", 0.0),
            ("chooser", "beloved", "dependency", 0.7, -0.2, "存在の拠り所", 0.0),
            ("arbiter", "chooser", "power_gradient", 0.85, 0.85, "規則を司る者の優位", 0.2),
            ("arbiter", "chooser", "asymmetry", 0.6, 0.6, "規則の全貌を知る", 0.2),
            ("chooser", "arbiter", "opposition", 0.75, 0.0, "規則への挑戦", 0.45),
            ("inevitable", "beloved", "embodiment", 0.7, 0.0, "喪失が大切な人に宿る", 0.7),
            ("beloved", "chooser", "asymmetry", 0.5, 0.4, "自らの運命を知っていた", 0.75),
            ("crossroads", "inevitable", "embodiment", 0.6, 0.0, "分岐点が喪失を可視化する", 0.88),
        ],
        "beats": [
            ("omen", "hook", 0.05, 0.05, "喪失の予兆", "分岐点で大切な人を失う瞬間の予兆が描かれ、主体の試行の動機が示される。", [], 0.05, 0.03, ["chooser", "beloved", "crossroads"], False),
            ("first_branch", "premise", 0.2, 0.05, "最初の試行と喪失", "最初の選択が喪失で終わり、主体は分岐点へ戻る力を知る。", ["omen"], 0.17, 0.02, ["chooser", "beloved", "inevitable"], False),
            ("rewind", "transition", 0.25, 0.04, "やり直し", "時間が巻き戻り、主体は別の選択を試みるために分岐点へ立つ。", ["first_branch"], -0.25, 0.02, ["chooser", "crossroads"], False),
            ("second_branch", "escalation", 0.45, 0.06, "第二の試行", "全く別の選択をしても、異なる経路で同じ喪失へ行き着く。", ["rewind"], 0.2, 0.02, ["chooser", "beloved"], False),
            ("pattern_noticed", "local_optimization_limit", 0.5, 0.04, "規則の発見", "どの分岐でも同じ核心が失われる規則に主体が気づき、希望が大きく削られる。", ["second_branch"], -0.3, 0.03, ["chooser", "arbiter", "inevitable"], True),
            ("desperate_branch", "crisis", 0.7, 0.06, "規則への挑戦", "主体が規則そのものを破ろうとする最後の試行に出るが、代償だけが増えていく。", ["pattern_noticed"], 0.22, 0.02, ["chooser", "arbiter"], True),
            ("meaning_turn", "structural_twist", 0.75, 0.04, "問いの転回", "大切な人が自らの運命を知っていたと分かり、結果ではなく意味を変えるという発想に至る。", ["desperate_branch"], -0.34, 0.04, ["chooser", "beloved"], False),
            ("meaningful_choice", "costly_breakthrough", 0.9, 0.05, "意味を選ぶ", "主体は喪失を受け入れつつ、大切な人と過ごす最後の時間を自ら選び取る。", ["meaning_turn"], 0.4, 0.03, ["chooser", "beloved", "altered"], True),
            ("flowers", "afterglow", 0.98, 0.02, "一つの必然", "全ての分岐が収束した地点に、主体が花を手向ける。", ["meaningful_choice"], -0.5, 0.01, ["chooser", "crossroads"], False),
        ],
        "mechanisms": [
            ("repeated_loss", "loss_aversion", "繰り返される喪失への抵抗", "同じ喪失が分岐ごとに繰り返され、読者の損失回避が試行への強い共感を生む。", "試行のたびの喪失", "主体の執念への共感", "プロスペクト理論（損失回避）", 0.8, ["first_branch", "second_branch"], ["beloved", "inevitable"]),
            ("rule_error", "prediction_error", "規則の発見による予測誤差", "やり直せば救えるという前提が崩れ、物語の問いそのものが変わる。", "規則の発見", "絶望と問いの転換", "予測符号化理論（スキーマ違反）", 0.75, ["pattern_noticed"], ["arbiter"]),
            ("acceptance_catharsis", "catharsis", "意味の選択によるカタルシス", "結果を変えられないまま意味を選ぶ瞬間に、蓄積した喪失の痛みが静かに解放される。", "最後の時間の選択", "涙と静かな解放", "カタルシス論", 0.85, ["meaningful_choice", "flowers"], ["chooser", "beloved"]),
            ("beloved_empathy", "empathy_transfer", "失われる人への共感", "運命を知りながら黙っていた大切な人の視点が開かれ、読者の共感が二重になる。", "問いの転回", "二人分の痛みの共有", "共感の視点取得理論", 0.7, ["omen", "meaning_turn"], ["beloved"]),
        ],
        "anchor": "あのとき別の選択をしていればという、誰もが抱える後悔",
        "drivers": ["試行のたびに増える代償", "同じ喪失へ収束する規則の不気味さ", "大切な人の沈黙の意味"],
        "emotions": ["不安", "希望", "絶望", "執念", "虚無", "足掻き", "気づき", "静かな受容"],
        "template": (
            "{{BRANCHING_PROTAGONIST}}は{{BRANCHING_POINT}}へ何度も戻り、{{THREATENED_BELOVED}}を救うために{{ALTERED_OUTCOME}}を試み続ける。"
            "しかしどの分岐でも{{INEVITABLE_LOSS}}は訪れ、{{BRANCH_ARBITER}}の規則は変わらない。大切な人の沈黙の意味を知った主体は、"
            "結果ではなく{{MEANING_CHOICE}}を選ぶ。"
        ),
        "slots": [
            ("BRANCHING_PROTAGONIST", "agent", "分岐をやり直す主体", [], ("時を戻せる少年", "施策をやり直し続ける担当者", "仮説を立て直し続ける研究者"), True),
            ("THREATENED_BELOVED", "agent", "失われ続ける大切な人", ["運命を知っていた伏線を序盤に置く"], ("病を抱えた幼なじみ", "離脱していく優良顧客", "守りたい理論"), True),
            ("BRANCH_ARBITER", "agent", "分岐の規則を体現する存在", [], ("時計塔の番人", "変えられない市場構造", "変えられない物理的制約"), True),
            ("ALTERED_OUTCOME", "goal", "結果を変えたい願い", [], ("幼なじみを救うこと", "離脱率をゼロにすること", "反証を覆すこと"), True),
            ("INEVITABLE_LOSS", "lack", "避けられない喪失", [], ("幼なじみとの別れ", "一定数の顧客離脱", "仮説の部分的な誤り"), True),
            ("BRANCHING_POINT", "symbol", "全ての分岐が始まる地点", ["同じ構図で繰り返し描く"], ("踏切", "同じ分析画面", "同じ実験条件"), True),
            ("MEANING_CHOICE", "reversal", "結果ではなく意味を選ぶ決断", [], ("最後の一日を共に過ごすこと", "離脱する顧客との良い別れ方の設計", "誤りを含めた理論の再定義"), True),
        ],
        "invariants": [
            "試行は最低三回とし、毎回異なる経路で同じ喪失へ収束させる",
            "試行ごとに解放感（希望）を段階的に減らしていく",
            "大切な人が運命を知っていた伏線を序盤に置く",
            "最後の選択は結果ではなく意味を変えるものとし、t≈0.9 に置く",
        ],
        "guide": {
            "manga": (
                "全28ページ想定。分岐点（踏切など）を毎回同じ構図・同じコマ割りで描き、試行のたびに背景の細部だけを変える。"
                "各試行の喪失の瞬間は右ページ下段の大ゴマ、巻き戻しは小ゴマを逆順に並べる演出で示す。"
                "23p左ページ終端で大切な人が微笑む引きを作り、めくった24pからは時間を戻さない最後の一日を余白の多いコマで描き、最終ページで分岐点に花を置く。"
            ),
            "marketing": (
                "顧客が試してきた複数の解決策（分岐）を順に検証し、どれも同じ根本原因で失敗する構造を示す。"
                "最後に結果をゼロにするのではなく、避けられない損失の意味を変える（設計し直す）提案として価値を提示する。"
            ),
            "article": (
                "複数の代替案を一つずつ検討し、いずれも同じ制約に突き当たることを示した後、問いそのものを立て直す。"
                "制約を受け入れたうえで意味のある選択肢を提示し、読者が今日選べる行動で締める。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.2), (0.2, 0.55), (0.25, 0.25), (0.45, 0.68), (0.5, 0.32), (0.7, 0.8), (0.75, 0.4), (0.9, 0.95), (1.0, 0.25)],
            "release": [(0.0, 0.05), (0.2, 0.4), (0.25, 0.05), (0.45, 0.3), (0.5, 0.05), (0.7, 0.2), (0.75, 0.03), (0.9, 0.1), (0.95, 0.6), (1.0, 0.9)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.25), (0.45, 0.4), (0.7, 0.55), (0.75, 0.75), (0.9, 0.85), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.05), (0.2, 0.25), (0.45, 0.42), (0.7, 0.58), (0.75, 0.78), (0.9, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.4), (0.2, 0.5), (0.45, 0.6), (0.7, 0.72), (0.9, 0.85), (0.95, 0.5), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 11,
        "slug": "five_stages_of_grief",
        "title": "喪失受容の五段階",
        "origin": "drama",
        "summary": (
            "決定的な喪失の直後から、主体が否認・怒り・取引・抑鬱・受容の五段階を順に通過する構造。各段階で主体は異なる相手と異なる手段で喪失に抗い、"
            "遺された言葉との出会いを経て、喪失した対象との関係を内面で結び直す。喪失そのものが新しい生き方の起点へ変わる。"
        ),
        "tags": ["喪失", "悲嘆の過程", "受容", "内面の旅"],
        "lineage": ["悲嘆の五段階モデル", "喪の作業（精神分析）", "通過儀礼としての弔い"],
        "nodes": [
            ("mourner", "agent", "bereaved_protagonist", "大切な存在を失った主体", "喪失を受け入れられず、段階ごとに異なる形で抗い続ける。", 0.95, "BEREAVED_PROTAGONIST"),
            ("lost_one", "agent", "lost_beloved", "失われた存在", "不在のまま物語の中心に居続け、遺された言葉によって最後に主体と対話する。", 0.85, "LOST_BELOVED"),
            ("companion", "agent", "steadfast_companion", "寄り添い続ける同伴者", "主体の怒りを受け止め、同じ喪失を先に越えた者として静かに待つ。", 0.65, "STEADFAST_COMPANION"),
            ("undo", "desire", "undo_the_loss", "喪失を取り消したいという願い", "否認と取引の段階で主体を突き動かす不可能な願い。", 0.8, "UNDO_THE_LOSS"),
            ("absence", "lack", "absence", "埋まらない不在", "日常のあらゆる場面に現れ、主体の時間を止める欠落。", 0.85, "ABSENCE"),
            ("keepsake", "symbol", "keepsake", "失われた存在の形見", "不在を可視化し、最後に手放すか受け継ぐかが問われる物。", 0.65, "KEEPSAKE"),
        ],
        "edges": [
            ("mourner", "lost_one", "dependency", 0.9, -0.3, "失われた存在への依存", 0.0),
            ("absence", "mourner", "power_gradient", 0.85, 0.8, "不在が主体の時間を止める", 0.05),
            ("keepsake", "lost_one", "embodiment", 0.75, 0.0, "形見が失われた存在を宿す", 0.05),
            ("mourner", "undo", "pursuit", 0.8, 0.0, "喪失を取り消す試み", 0.15),
            ("mourner", "companion", "opposition", 0.6, 0.0, "怒りの矛先", 0.3),
            ("companion", "mourner", "asymmetry", 0.5, 0.4, "喪失を先に越えた者の知", 0.45),
            ("keepsake", "absence", "embodiment", 0.6, 0.0, "形見が不在を可視化する", 0.6),
            ("lost_one", "mourner", "asymmetry", 0.6, 0.5, "遺された言葉", 0.8),
        ],
        "beats": [
            ("loss", "hook", 0.05, 0.04, "喪失の瞬間", "決定的な喪失が突然訪れ、主体の世界が止まる。", [], 0.55, 0.0, ["mourner", "lost_one"], True),
            ("denial", "premise", 0.15, 0.05, "否認", "主体は何も起きなかったかのように日常を続け、不在から目を逸らす。", ["loss"], -0.55, 0.01, ["mourner", "absence"], False),
            ("anger", "escalation", 0.3, 0.06, "怒り", "主体の怒りが寄り添う同伴者へ向かい、関係が壊れかける。", ["denial"], 0.33, 0.01, ["mourner", "companion"], False),
            ("bargaining", "false_victory", 0.45, 0.06, "取引", "失ったものを取り戻せるという幻想に縋り、束の間の希望を得る。", ["anger"], -0.2, 0.02, ["mourner", "undo"], False),
            ("depression", "crisis", 0.6, 0.06, "抑鬱", "取引の失敗とともに深い虚無が訪れ、主体の時間が完全に止まる。", ["bargaining"], -0.18, 0.02, ["mourner", "absence"], True),
            ("trigger", "escalation", 0.7, 0.05, "記念日の引き金", "記念日や形見が引き金となり、閉じ込めていた感情が再び揺さぶられる。", ["depression"], 0.23, 0.02, ["mourner", "keepsake"], False),
            ("message", "revelation", 0.8, 0.05, "遺された言葉", "失われた存在が主体の未来を案じて遺した言葉が見つかり、不在の存在と初めて対話する。", ["trigger"], 0.47, 0.04, ["mourner", "lost_one", "keepsake"], True),
            ("acceptance", "costly_breakthrough", 0.9, 0.05, "受容", "主体は関係を内面で結び直し、形見を手放すか受け継ぐかを自ら選ぶ。", ["message"], -0.52, 0.02, ["mourner", "keepsake", "companion"], False),
            ("new_days", "afterglow", 1.0, 0.02, "不在と共にある日常", "不在を抱えたまま、主体の時間が再び動き出す。", ["acceptance"], -0.15, 0.01, ["mourner"], False),
        ],
        "mechanisms": [
            ("grief_empathy", "empathy_transfer", "喪失への共感移入", "読者自身の喪失の記憶が主体に重なり、各段階の痛みを自分事として体験する。", "喪失の瞬間と抑鬱", "深い共感と涙", "共感の視点取得理論", 0.85, ["loss", "depression"], ["mourner", "absence"]),
            ("bargain_aversion", "loss_aversion", "取り戻したい執着", "失ったものを取り戻せるという幻想が損失回避を刺激し、取引の段階に切実さを与える。", "取引の希望", "希望が崩れる痛みの増幅", "プロスペクト理論（損失回避）", 0.65, ["bargaining"], ["undo"]),
            ("message_catharsis", "catharsis", "遺された言葉のカタルシス", "不在の存在との対話が成立した瞬間に、蓄積した悲しみが一挙に解放される。", "遺された言葉の発見", "号泣と安堵", "カタルシス論", 0.9, ["message", "acceptance"], ["lost_one", "keepsake"]),
            ("quiet_end", "peak_end", "静かな終幕の記憶", "最大の感情の頂点と静かな日常の再開が、体験全体を温かく記憶させる。", "受容と日常の再開", "長く残る温かな余韻", "ピーク・エンドの法則", 0.6, ["acceptance", "new_days"], ["mourner"]),
        ],
        "anchor": "大切な存在を失ったとき、時間だけが先に進んでしまうことへの戸惑い",
        "drivers": ["段階ごとに変わる主体の抗い方", "同伴者との壊れかけた関係", "形見に宿る不在の気配"],
        "emotions": ["衝撃", "麻痺", "怒り", "希望", "虚無", "動揺", "号泣", "静かな温かさ"],
        "template": (
            "{{BEREAVED_PROTAGONIST}}は{{LOST_BELOVED}}を失い、{{ABSENCE}}の中で否認・怒り・取引・抑鬱の段階を通過しながら{{UNDO_THE_LOSS}}を求める。"
            "{{STEADFAST_COMPANION}}は怒りを受け止めて待ち続ける。{{KEEPSAKE}}に導かれて{{LAST_MESSAGE}}を見つけたとき、主体は不在の存在と初めて対話し、"
            "関係を内面で結び直して日常へ戻る。"
        ),
        "slots": [
            ("BEREAVED_PROTAGONIST", "agent", "喪失を抱えた主体", [], ("母を亡くした少女", "主力事業を失った経営者", "長年の前提を失った研究者"), True),
            ("LOST_BELOVED", "agent", "失われた存在", ["不在のまま存在感を保たせる"], ("亡き母", "撤退した看板商品", "否定された理論"), True),
            ("STEADFAST_COMPANION", "agent", "寄り添い続ける同伴者", [], ("無口な父", "長年の取引先", "かつての共同研究者"), True),
            ("UNDO_THE_LOSS", "goal", "喪失を取り消したい願い", [], ("母が生きている世界", "看板商品の復活", "理論の延命"), True),
            ("ABSENCE", "lack", "埋まらない不在", [], ("食卓の空いた席", "売場の空いた棚", "参照されなくなった章"), True),
            ("KEEPSAKE", "symbol", "形見", ["序盤から日常の背景に置いておく"], ("母のエプロン", "初代商品の試作品", "恩師の手書きのノート"), True),
            ("LAST_MESSAGE", "evidence", "遺された言葉", [], ("母が録音した声", "創業者の手紙", "恩師が遺した最後の論文の結び"), True),
        ],
        "invariants": [
            "喪失は冒頭で突然訪れさせ、五段階を順に通過させる",
            "怒りの矛先は寄り添う同伴者に向け、関係を一度壊しかける",
            "抑鬱の底で時間を止め、記念日や形見を引き金に再び動かす",
            "遺された言葉による対話を t≈0.8 に置き、その後に受容を置く",
        ],
        "guide": {
            "manga": (
                "全24ページ想定。1pで喪失の瞬間を断ち切りの大ゴマで描き、2pは同じ部屋の何気ない日常を細かいコマで描いて否認の不気味さを出す。"
                "怒りの場面はダッチアングルと叫びのフキダシ、抑鬱の場面は台詞のない余白の多いコマを連続させる。"
                "19p左ページ終端で形見に手を伸ばす引きを作り、めくった20p右ページで遺された言葉を見開きの大ゴマで描き、最終ページは空いた席のある食卓を静かな引きで描く。"
            ),
            "marketing": (
                "主力事業や大切な習慣を失った顧客の心理段階（否認・怒り・取引・停滞）を丁寧に言語化し、各段階で起こりがちな失敗を共感的に示す。"
                "最後に失ったものの価値を引き継ぐ新しい形として提案を示し、急がせないCTAで受容の段階に寄り添う。"
            ),
            "article": (
                "前提や習慣を失ったときの受け止め方を五段階で整理し、各段階での典型的な行動と見落としを具体例で示す。"
                "結論では失ったものから引き継ぐべき核を特定する手順を提示する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.3), (0.05, 0.85), (0.15, 0.3), (0.3, 0.75), (0.45, 0.5), (0.6, 0.22), (0.7, 0.45), (0.8, 0.92), (0.9, 0.4), (1.0, 0.25)],
            "release": [(0.0, 0.1), (0.05, 0.0), (0.15, 0.05), (0.3, 0.1), (0.45, 0.4), (0.6, 0.05), (0.8, 0.1), (0.9, 0.75), (1.0, 0.95)],
            "audience_knowledge": [(0.0, 0.2), (0.05, 0.5), (0.3, 0.55), (0.6, 0.62), (0.8, 0.7), (0.9, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.2), (0.05, 0.5), (0.3, 0.5), (0.6, 0.55), (0.8, 0.65), (0.9, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.3), (0.05, 0.7), (0.15, 0.4), (0.3, 0.6), (0.45, 0.55), (0.6, 0.35), (0.8, 0.8), (0.9, 0.5), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 12,
        "slug": "foreshadowing_cascade",
        "title": "伏線の一斉回収",
        "origin": "manga",
        "summary": (
            "序盤から中盤にかけて、何気ない日常の細部（言い回し・置物・習慣・天候）が意味を持たないかのように散りばめられる構造。"
            "終盤の一点で一つの事実が明かされると、それらの細部が連鎖的に意味を帯び、物語全体が一枚の設計図として一斉に立ち上がる。"
            "長い静けさと一瞬の連鎖の対比が、強い発見の快感を生む。"
        ),
        "tags": ["伏線回収", "日常の細部", "連鎖的開示", "設計美"],
        "lineage": ["伏線と回収（提示された小道具は必ず使われるという原則）", "日常描写の積層構造", "細部が組み上がるパズル型の叙述"],
        "nodes": [
            ("dweller", "agent", "everyday_protagonist", "穏やかな日常を生きる主体", "細部の意味に気づかないまま日常を過ごし、最後に全ての意味を受け取る。", 0.85, "EVERYDAY_PROTAGONIST"),
            ("keeper", "agent", "secret_keeper", "細部の意味を知る人物", "日常を守るために事実を伏せ、細部の中にだけ手がかりを残している。", 0.8, "SECRET_KEEPER"),
            ("hidden_fact", "lack", "hidden_fact", "日常の下に隠された事実", "全ての細部を一つに結ぶ起点。明かされた瞬間に日常の意味が変わる。", 0.85, "HIDDEN_FACT"),
            ("continuity", "desire", "everyday_continuity", "日常が続くこと", "主体と秘密を知る人物の双方が守ろうとするもの。", 0.7, "EVERYDAY_CONTINUITY"),
            ("details", "symbol", "scattered_details", "散りばめられた細部", "繰り返される言い回し、置かれたままの物、毎朝の習慣、決まった天気の話。", 0.75, "SCATTERED_DETAILS"),
            ("keystone", "symbol", "keystone_fact", "一斉回収の起点となる物", "最後に現れる一つの物や言葉。全ての細部を連結する要石。", 0.7, "KEYSTONE_REVEAL"),
        ],
        "edges": [
            ("dweller", "continuity", "pursuit", 0.7, 0.0, "日常の継続", 0.0),
            ("keeper", "dweller", "asymmetry", 0.75, 0.75, "細部の意味を知る者の沈黙", 0.0),
            ("dweller", "keeper", "dependency", 0.6, -0.2, "日常を支える存在への依存", 0.15),
            ("keeper", "continuity", "pursuit", 0.6, 0.0, "日常を守るための沈黙", 0.15),
            ("details", "hidden_fact", "embodiment", 0.8, 0.0, "細部が事実を少しずつ漏らす", 0.3),
            ("hidden_fact", "continuity", "opposition", 0.65, 0.0, "隠された事実が日常を脅かす", 0.8),
            ("hidden_fact", "dweller", "power_gradient", 0.6, 0.5, "事実が主体の世界を書き換える", 0.8),
            ("keystone", "details", "embodiment", 0.7, 0.0, "要石が細部を連結する", 0.8),
        ],
        "beats": [
            ("calm_open", "hook", 0.05, 0.05, "穏やかな日常と最初の細部", "何気ない朝の情景の中に、最初の細部（決まった言い回し）がさりげなく置かれる。", [], 0.02, 0.0, ["dweller", "details"], False),
            ("detail_two", "setup", 0.15, 0.05, "二つ目の細部", "秘密を知る人物の習慣が、意味のない癖のように描かれる。", ["calm_open"], 0.04, 0.0, ["keeper", "details"], False),
            ("detail_three", "setup", 0.3, 0.05, "三つ目の細部", "誰も触れない置物と、毎回同じ天気の話題が日常に溶け込む。", ["detail_two"], -0.03, 0.0, ["dweller", "details"], False),
            ("small_ripple", "escalation", 0.45, 0.05, "小さな波紋", "秘密を知る人物の不自然な沈黙が一瞬だけ描かれ、すぐに日常へ戻る。", ["detail_three"], 0.08, 0.02, ["keeper", "dweller"], False),
            ("lull", "transition", 0.6, 0.05, "束の間の凪", "最も穏やかな日常の場面が置かれ、読者は細部のことを忘れかける。", ["small_ripple"], -0.02, 0.0, ["dweller", "continuity"], False),
            ("unease", "local_optimization_limit", 0.72, 0.05, "違和感の臨界", "日常の説明では細部を説明しきれなくなり、小さな違和感が積み重なる。", ["lull"], 0.11, 0.02, ["dweller", "details"], False),
            ("keystone_reveal", "revelation", 0.8, 0.04, "要石の開示", "一つの事実が明かされ、隠されていた起点が姿を現す。", ["unease"], 0.5, 0.15, ["keeper", "keystone", "hidden_fact"], True),
            ("cascade", "payoff", 0.85, 0.04, "一斉回収", "全ての細部が連鎖的に意味を帯び、日常の場面が次々と別の意味へ読み替えられていく。", ["keystone_reveal"], 0.07, 0.04, ["dweller", "details", "keystone"], True),
            ("redefined", "afterglow", 0.95, 0.04, "再定義された日常", "同じ朝の情景が再び描かれ、今度は全ての細部が愛おしい意味を持って見える。", ["cascade"], -0.5, 0.01, ["dweller", "keeper", "continuity"], False),
        ],
        "mechanisms": [
            ("faint_gap", "curiosity_gap", "微かな問いの蓄積", "意味の分からない細部が意識の端に小さな問いとして蓄積され、回収への潜在的な期待を作る。", "繰り返される細部と不自然な沈黙", "無意識の注意の持続", "情報ギャップ理論", 0.6, ["detail_two", "small_ripple"], ["details", "keeper"]),
            ("calm_transport", "narrative_transportation", "日常への深い移入", "穏やかな日常描写が読者を物語世界に深く沈め、回収時の落差を最大化する。", "長い日常描写", "世界への愛着", "物語移入理論", 0.7, ["calm_open", "lull"], ["dweller"]),
            ("pattern_completion", "anticipation_reward", "連鎖的な意味の完成", "散らばった細部が一つの設計図に組み上がる瞬間、パターン完成の強い快感が生じる。", "要石の開示と一斉回収", "鳥肌と再読欲求", "報酬予測誤差理論", 0.9, ["keystone_reveal", "cascade"], ["keystone", "details"]),
            ("calm_contrast", "contrast_effect", "静けさと連鎖の対比", "長い凪と一瞬の連鎖の落差が、発見の衝撃を何倍にも増幅する。", "凪の直後の開示", "衝撃の増幅", "対比効果", 0.65, ["lull", "keystone_reveal"], ["continuity"]),
        ],
        "anchor": "当たり前だと思っていた日常が、誰かの静かな努力に支えられていたと知る瞬間",
        "drivers": ["何気ない細部の反復", "秘密を知る人物の一瞬の沈黙", "長い凪の後の連鎖"],
        "emotions": ["穏やかさ", "微かな違和", "安らぎ", "引っかかり", "驚き", "鳥肌", "感動", "愛おしさ"],
        "template": (
            "{{EVERYDAY_PROTAGONIST}}は{{EVERYDAY_CONTINUITY}}の中で暮らし、{{SECRET_KEEPER}}はその日常を守るために沈黙している。"
            "日常には{{SCATTERED_DETAILS}}がさりげなく散りばめられ、終盤に{{KEYSTONE_REVEAL}}によって{{HIDDEN_FACT}}が明かされた瞬間、"
            "全ての細部が連鎖的に意味を帯び、{{REDEFINED_EVERYDAY}}が立ち上がる。"
        ),
        "slots": [
            ("EVERYDAY_PROTAGONIST", "agent", "日常を生きる主体", [], ("下宿で暮らす学生", "毎日製品を使う顧客", "日々の業務をこなす実務家"), True),
            ("SECRET_KEEPER", "agent", "細部の意味を知る人物", ["沈黙を一度だけ不自然に描く"], ("下宿の大家", "目立たない改善を続けた開発陣", "地道に記録を残した先人"), True),
            ("HIDDEN_FACT", "lack", "日常の下に隠された事実", ["序盤には決して示唆しすぎない"], ("大家が主人公の亡き親の友だった事実", "全ての細部が一つの思想で設計されていた事実", "全ての手順が一つの原理から導かれていた事実"), True),
            ("EVERYDAY_CONTINUITY", "goal", "続いてほしい日常", [], ("いつもの朝食", "いつもの快適な使い心地", "いつもの業務の流れ"), True),
            ("SCATTERED_DETAILS", "symbol", "散りばめられた細部", ["最低三つを異なる種類（言葉・物・習慣）で置く"], ("毎朝の決まった挨拶", "小さな操作の気配り", "手順書の注記"), True),
            ("KEYSTONE_REVEAL", "symbol", "全てを連結する要石", [], ("押し入れの古い写真", "設計思想を書いた一枚のメモ", "原理を記した一行"), True),
            ("REDEFINED_EVERYDAY", "reversal", "意味が変わった日常", [], ("同じ朝の挨拶の新しい意味", "使い慣れた製品への新しい愛着", "手順の背後の原理の理解"), True),
        ],
        "invariants": [
            "細部は最低三つ、言葉・物・習慣など異なる種類で配置する",
            "回収直前に最も穏やかな凪の場面を置く",
            "回収は t≈0.8 の一点から連鎖的に行い、細部を一つずつ読み替える",
            "最後に冒頭と同じ日常の場面を再提示して意味の変化を示す",
        ],
        "guide": {
            "manga": (
                "全24ページ想定。各細部は背景の一部として小さく描き、同じ構図・同じ位置で2〜3回繰り返して読者の目に馴染ませる。"
                "17〜18pに台詞のない穏やかな見開きを置いて凪を作り、19p左ページ終端で要石の物に手が触れる引きを作る。"
                "めくった20pから小ゴマを連打して過去の細部のコマを再掲し、それぞれに新しい意味の一言を添えて一斉回収する。最終ページは1pと同じ朝の情景で締める。"
            ),
            "marketing": (
                "製品の何気ない使い心地の細部（気配り）をストーリーの中にさりげなく複数配置し、それらが一つの設計思想から生まれていたことを終盤で明かす。"
                "回収の直後に設計思想への共感を軸にしたCTAを置き、既存の体験の意味を変えて購入動機にする。"
            ),
            "article": (
                "序盤に複数の具体的な観察事実を、結論を伏せたまま並べておく。"
                "終盤で一つの原理を提示し、序盤の事実を一つずつその原理で読み替えて体系の美しさを示し、読者の実務への応用で締める。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.2), (0.15, 0.25), (0.3, 0.22), (0.45, 0.3), (0.6, 0.28), (0.72, 0.4), (0.8, 0.88), (0.85, 0.95), (0.93, 0.5), (1.0, 0.3)],
            "release": [(0.0, 0.25), (0.15, 0.35), (0.3, 0.3), (0.45, 0.35), (0.6, 0.3), (0.72, 0.15), (0.8, 0.05), (0.93, 0.7), (1.0, 0.9)],
            "audience_knowledge": [(0.0, 0.1), (0.3, 0.15), (0.6, 0.2), (0.75, 0.25), (0.8, 0.7), (0.85, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.3), (0.3, 0.35), (0.6, 0.4), (0.75, 0.42), (0.8, 0.75), (0.85, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.3), (0.15, 0.35), (0.3, 0.4), (0.45, 0.45), (0.6, 0.5), (0.72, 0.55), (0.8, 0.85), (0.85, 0.9), (1.0, 0.4)],
        },
    }
)

PATTERNS.append(
    {
        "num": 13,
        "slug": "tragicomic_reversal",
        "title": "悲喜劇反転（道化の真実）",
        "origin": "drama",
        "summary": (
            "道化役の人物が周囲を笑わせ続ける軽快な喜劇として始まるが、笑いの一つ一つが道化自身の深い痛みを隠す仮面だったことが少しずつ漏れ出す構造。"
            "読者だけが仮面の綻びを見届け、最大の笑いの場面で仮面が割れたとき、喜劇の全場面が悲劇として再読される。最後に道化は素顔のまま笑う。"
        ),
        "tags": ["悲喜劇", "道化", "仮面", "笑いと痛み"],
        "lineage": ["悲喜劇（笑いと悲劇の混交）", "賢い道化の系譜", "笑いの裏の悲劇"],
        "nodes": [
            ("jester", "agent", "masked_jester", "周囲を笑わせ続ける道化", "笑いを提供することでしか居場所を確保できないと信じ、痛みを仮面の下に隠している。", 0.95, "MASKED_JESTER"),
            ("crowd", "agent", "laughing_crowd", "道化に笑いを求める周囲", "悪意なく道化に笑いを求め続け、その要求が最後に仮面を割る。", 0.7, "LAUGHING_CROWD"),
            ("noticer", "agent", "one_who_notices", "仮面の綻びに気づく人物", "周囲の中でただ一人道化の痛みに気づき、笑いを止めようとする。", 0.65, "ONE_WHO_NOTICES"),
            ("belonging", "desire", "belonging", "居場所", "道化が笑いと引き換えに得ようとしているもの。", 0.8, "BELONGING"),
            ("pain", "lack", "hidden_pain", "笑いの下に隠された痛み", "道化が誰にも見せない喪失や孤独。笑いの量に比例して深まる。", 0.85, "HIDDEN_PAIN"),
            ("mask", "symbol", "comic_mask", "笑いの仮面", "道化の決め顔や決め台詞。割れた瞬間に痛みの象徴へ反転する。", 0.7, "COMIC_MASK"),
        ],
        "edges": [
            ("jester", "belonging", "pursuit", 0.8, 0.0, "笑いと引き換えの居場所", 0.0),
            ("crowd", "jester", "dependency", 0.7, 0.3, "周囲は道化の笑いに頼る", 0.1),
            ("jester", "crowd", "dependency", 0.6, -0.3, "道化は笑いに居場所を依存する", 0.1),
            ("mask", "pain", "embodiment", 0.85, 0.0, "仮面が痛みを覆い隠す", 0.2),
            ("pain", "jester", "power_gradient", 0.7, 0.6, "痛みが道化を突き動かす", 0.35),
            ("noticer", "jester", "asymmetry", 0.6, 0.5, "気づく者だけが痛みを見抜く", 0.55),
            ("noticer", "crowd", "opposition", 0.45, 0.0, "笑いを止めようとする者", 0.65),
            ("crowd", "jester", "betrayal", 0.55, 0.0, "最大の笑いの要求が仮面を割る", 0.75),
        ],
        "beats": [
            ("stage_open", "hook", 0.1, 0.05, "道化の登場", "道化が鮮やかな芸で周囲を沸かせ、軽快な喜劇の幕が上がる。", [], 0.1, 0.02, ["jester", "crowd"], False),
            ("routine", "premise", 0.2, 0.05, "笑いの日常", "道化が笑わせることで居場所を確保する日常が描かれる。", ["stage_open"], -0.07, 0.03, ["jester", "belonging"], False),
            ("crack_one", "setup", 0.35, 0.04, "最初の綻び", "ひとりになった瞬間に一瞬だけ笑顔が消え、読者だけがその表情を目撃する。", ["routine"], 0.09, 0.1, ["jester", "mask", "pain"], True),
            ("cover_up", "transition", 0.45, 0.05, "笑いでの取り繕い", "道化はすぐに大げさな芸で取り繕い、周囲は何も気づかない。", ["crack_one"], -0.05, 0.03, ["jester", "crowd"], False),
            ("noticer_steps_in", "escalation", 0.55, 0.05, "気づく者の接近", "仮面の綻びに気づいた人物が道化に近づき、笑いの裏を問いかける。", ["cover_up"], 0.17, 0.03, ["noticer", "jester"], False),
            ("demand", "local_optimization_limit", 0.65, 0.05, "取り繕いの限界", "周囲がさらに大きな笑いを求め、道化の取り繕いが限界に近づく。", ["noticer_steps_in"], -0.04, 0.03, ["crowd", "jester"], False),
            ("mask_breaks", "structural_twist", 0.75, 0.05, "仮面の崩壊", "最大の笑いの場面で道化が泣き崩れ、周囲は初めて痛みの存在を知る。", ["demand"], 0.57, -0.35, ["jester", "crowd", "mask", "pain"], True),
            ("rereading", "payoff", 0.85, 0.05, "喜劇の再読", "これまでの全ての笑いの場面が、痛みを隠すための記録として意味を変える。", ["mask_breaks"], -0.35, 0.02, ["jester", "noticer"], False),
            ("unmasked_laugh", "afterglow", 0.95, 0.04, "素顔の笑い", "道化が仮面なしで、気づいた人物と共に小さく笑う。", ["rereading"], -0.18, 0.01, ["jester", "noticer", "belonging"], False),
        ],
        "mechanisms": [
            ("jester_irony", "dramatic_irony", "読者だけが見る綻び", "周囲が笑う中で読者だけが道化の痛みを見ており、笑いの場面が切なさを帯びていく。", "最初の綻びと取り繕い", "笑いながら胸が痛む二重の感情", "劇的アイロニー", 0.8, ["crack_one", "noticer_steps_in"], ["jester", "mask"]),
            ("laugh_contrast", "contrast_effect", "笑いと涙の対比", "最大の笑いの場面での崩壊が、喜劇の軽さとの対比で悲しみを何倍にも増幅する。", "仮面の崩壊", "落差による衝撃", "対比効果", 0.85, ["mask_breaks"], ["crowd", "jester"]),
            ("jester_empathy", "empathy_transfer", "道化への共感移入", "笑わせる側の孤独が明かされ、読者自身の取り繕いの経験が重なる。", "仮面の崩壊と喜劇の再読", "道化の痛みの自分事化", "共感の視点取得理論", 0.75, ["mask_breaks", "rereading"], ["pain"]),
            ("unmasked_relief", "catharsis", "素顔の笑いによる解放", "仮面のない笑いが生まれた瞬間に、蓄積した切なさが温かい涙として解放される。", "素顔の笑い", "安堵と温かさ", "カタルシス論", 0.7, ["unmasked_laugh"], ["jester", "noticer"]),
        ],
        "anchor": "場の空気を明るくするために、自分の痛みを笑いで隠してしまう優しさ",
        "drivers": ["笑いの場面に紛れ込む一瞬の素顔", "周囲の悪意なき要求", "気づいた人物との距離"],
        "emotions": ["愉快", "楽しさ", "ひっかかり", "切なさ", "不安", "戦慄", "悲しみ", "温かさ"],
        "template": (
            "{{MASKED_JESTER}}は{{LAUGHING_CROWD}}を笑わせ続けることで{{BELONGING}}を得ているが、{{COMIC_MASK}}の下には{{HIDDEN_PAIN}}が隠れている。"
            "{{ONE_WHO_NOTICES}}だけが綻びに気づき、最大の笑いの場面で仮面が割れたとき、喜劇の全場面が悲劇として読み替えられる。最後に道化は{{UNMASKED_LAUGH}}にたどり着く。"
        ),
        "slots": [
            ("MASKED_JESTER", "agent", "笑わせ続ける道化", ["序盤は徹底して明るく描く"], ("クラスのお調子者", "明るく振る舞う窓口担当", "軽妙な語り口の筆者"), True),
            ("LAUGHING_CROWD", "agent", "笑いを求める周囲", ["悪意を持たせない"], ("クラスメイト", "気軽に要望を重ねる顧客", "軽い読み物を求める読者"), True),
            ("ONE_WHO_NOTICES", "agent", "綻びに気づく人物", [], ("転校生", "長年の担当者", "注意深い一部の読者"), True),
            ("BELONGING", "goal", "笑いと引き換えの居場所", [], ("教室の居場所", "顧客に必要とされる実感", "読まれ続けること"), True),
            ("HIDDEN_PAIN", "lack", "笑いの下の痛み", ["読者にだけ少しずつ見せる"], ("家族の不和", "現場の疲弊", "筆者自身の挫折"), True),
            ("COMIC_MASK", "symbol", "笑いの仮面", ["決め顔・決め台詞として反復する"], ("決めポーズと決め台詞", "いつもの明るい定型文", "毎回の軽妙な書き出し"), True),
            ("UNMASKED_LAUGH", "reversal", "素顔のままの笑い", [], ("友と交わす本当の笑顔", "率直な対話から生まれる信頼", "等身大の言葉で語る文章"), True),
        ],
        "invariants": [
            "序盤は解放度の高い軽快な喜劇として描く",
            "仮面の綻びは読者にだけ短く見せ、周囲には気づかせない",
            "崩壊は最大の笑いの場面（t≈0.75）で起こす",
            "崩壊後に過去の笑いの場面を再提示して意味の反転を示す",
        ],
        "guide": {
            "manga": (
                "全20ページ想定。前半は誇張した表情・集中線・大きな描き文字でテンポよくギャグを重ね、道化がひとりになる瞬間だけ無音の小ゴマで素顔を挟む。"
                "14p左ページ終端で周囲が道化を囲んで大笑いする引きを作り、めくった15p右ページで同じ構図のまま道化の顔だけが泣き崩れている大ゴマを置く。"
                "16〜17pで過去のギャグのコマを縮小して再掲し、最終ページは仮面のない小さな笑顔を余白の中に描く。"
            ),
            "marketing": (
                "明るく軽快なトーンで現場の頑張りを紹介した後、その明るさの裏にあった現場の疲弊と率直な課題を開示する。"
                "取り繕いをやめて課題を共有する姿勢そのものを信頼の根拠とし、等身大の約束をCTAにする。"
            ),
            "article": (
                "軽妙な語り口で始め、要所で筆者自身の失敗や痛みを一瞬だけ覗かせる。"
                "中盤以降で率直な告白に転じ、冒頭の軽い言葉が何を隠していたかを示したうえで、等身大の学びを提示して締める。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.15), (0.1, 0.25), (0.2, 0.18), (0.35, 0.3), (0.45, 0.25), (0.55, 0.42), (0.65, 0.38), (0.75, 0.95), (0.85, 0.6), (0.92, 0.45), (1.0, 0.35)],
            "release": [(0.0, 0.6), (0.1, 0.75), (0.2, 0.65), (0.35, 0.7), (0.45, 0.55), (0.55, 0.45), (0.65, 0.5), (0.75, 0.05), (0.85, 0.2), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.1), (0.35, 0.3), (0.55, 0.5), (0.7, 0.55), (0.75, 0.85), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.05), (0.35, 0.08), (0.55, 0.12), (0.7, 0.15), (0.75, 0.8), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.25), (0.2, 0.3), (0.35, 0.4), (0.55, 0.5), (0.75, 0.8), (0.85, 0.6), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 14,
        "slug": "observer_effect_mystery",
        "title": "観測者効果のミステリー",
        "origin": "literature",
        "summary": (
            "謎を調べる行為そのものが謎の対象を変質させていく構造。探索者が近づくほど証言は変わり、痕跡は書き換わり、最後には探索者自身が謎の原因になっていたことが判明する。"
            "読者は変化が探索者の在不在と連動していることに先に気づき、真相は静的な答えではなく観測と対象の相互作用として提示される。"
        ),
        "tags": ["観測者効果", "自己言及的ミステリー", "変質する真相", "探索者の関与"],
        "lineage": ["観測が対象を変えるという逆説", "探索者自身が謎の原因となる型", "自己言及的な推理劇"],
        "nodes": [
            ("investigator", "agent", "entangled_investigator", "謎を調べる探索者", "客観的な観測者だと信じて記録を取り続けるが、その観測自体が対象を変えていく。", 0.95, "ENTANGLED_INVESTIGATOR"),
            ("subject", "agent", "shifting_subject", "証言や姿を変える調査対象", "探索者が近づくたびに異なる顔を見せる。嘘をついているのではなく、観測に応じて変わっている。", 0.8, "SHIFTING_SUBJECT"),
            ("rival", "agent", "rival_observer", "別の解釈を示す第二の観測者", "同じ対象を別の距離から観測し、探索者とは異なる真相を主張する。", 0.6, "RIVAL_OBSERVER"),
            ("truth", "desire", "fixed_truth", "確定した真相", "探索者が求める一つの答え。観測をやめない限り決して確定しない。", 0.85, "FIXED_TRUTH"),
            ("instability", "lack", "unstable_reality", "不安定な現実", "確かな足場の欠如。探索者を記録への依存へ追い込む。", 0.7, "UNSTABLE_REALITY"),
            ("record", "symbol", "changing_record", "書き換わる記録", "探索者の手帳や記録。最後に探索者自身の筆跡で書き換えられていたと判明する。", 0.7, "CHANGING_RECORD"),
        ],
        "edges": [
            ("investigator", "truth", "pursuit", 0.9, 0.0, "確定した真相の希求", 0.0),
            ("subject", "investigator", "asymmetry", 0.7, 0.6, "対象だけが変化の理由を知る", 0.1),
            ("investigator", "record", "dependency", 0.55, -0.3, "記録に依存する推理", 0.2),
            ("record", "instability", "embodiment", 0.75, 0.0, "記録の変化が不安定さを体現する", 0.2),
            ("investigator", "subject", "power_gradient", 0.6, 0.5, "観測する者の介入力", 0.3),
            ("rival", "investigator", "opposition", 0.65, 0.0, "対抗する解釈", 0.4),
            ("instability", "investigator", "power_gradient", 0.55, 0.4, "揺らぐ現実が探索者を追い詰める", 0.6),
            ("record", "investigator", "betrayal", 0.7, 0.0, "記録は自分の手で書き換えられていた", 0.8),
        ],
        "beats": [
            ("first_observation", "hook", 0.1, 0.05, "最初の観測", "探索者が不可解な現象を記録し、謎の探索が始まる。", [], 0.08, 0.0, ["investigator", "subject", "record"], False),
            ("shifting_testimony", "premise", 0.2, 0.05, "変わる証言", "同じ人物の証言が訪ねるたびに少しずつ変わり、探索者は嘘を疑う。", ["first_observation"], 0.07, 0.05, ["subject", "investigator"], False),
            ("hypothesis", "setup", 0.3, 0.05, "仮説の構築", "探索者は記録を整理して一つの仮説を組み立て、謎を制御できたと感じる。", ["shifting_testimony"], -0.03, 0.03, ["investigator", "record"], False),
            ("rewritten_trace", "escalation", 0.4, 0.05, "書き換わる痕跡", "確かに記録したはずの痕跡が別の形に変わっており、仮説が揺らぐ。", ["hypothesis"], 0.13, 0.07, ["record", "instability"], False),
            ("rival_theory", "transition", 0.5, 0.05, "対抗する解釈", "第二の観測者が全く異なる真相を主張し、探索者は孤立する。", ["rewritten_trace"], -0.03, 0.03, ["rival", "investigator"], False),
            ("pattern_of_presence", "local_optimization_limit", 0.6, 0.05, "臨在の規則", "変化は探索者がいるときにだけ起きていることに、読者だけが気づく。探索者の推理は限界に達する。", ["rival_theory"], 0.14, 0.06, ["investigator", "subject"], True),
            ("entanglement", "structural_twist", 0.8, 0.05, "観測者の関与の露見", "記録の書き換えは探索者自身の筆跡によるものだったと判明し、謎の原因が観測そのものだったと分かる。", ["pattern_of_presence"], 0.34, -0.25, ["investigator", "record", "subject"], True),
            ("letting_go", "costly_breakthrough", 0.9, 0.05, "観測の放棄", "探索者は真相を確定させる欲望を手放し、観測をやめることで対象を解放する。", ["entanglement"], -0.34, 0.03, ["investigator", "truth", "subject"], False),
            ("blank_page", "afterglow", 1.0, 0.02, "白紙の最後の頁", "記録の最後の頁は白紙のまま残り、真相は確定しないまま静かに閉じる。", ["letting_go"], -0.14, 0.01, ["record"], False),
        ],
        "mechanisms": [
            ("anomaly_gap", "curiosity_gap", "変化の理由への問い", "変わる証言と書き換わる痕跡が、なぜ変わるのかという強い問いを生む。", "最初の観測と変わる証言", "細部への執拗な注意", "情報ギャップ理論", 0.75, ["first_observation", "shifting_testimony"], ["subject", "record"]),
            ("presence_irony", "dramatic_irony", "読者だけが気づく臨在の規則", "変化が探索者の在不在と連動していることに読者だけが気づき、探索者への焦燥が生まれる。", "臨在の規則の提示", "探索者に伝えたくなる焦燥", "劇的アイロニー", 0.7, ["pattern_of_presence"], ["investigator"]),
            ("observer_error", "prediction_error", "観測者が原因だった予測誤差", "客観的な観測者という前提が崩れ、謎の構造そのものが反転する。", "自分の筆跡の発見", "強い驚きと世界観の揺らぎ", "予測符号化理論（スキーマ違反）", 0.9, ["entanglement"], ["investigator", "record"]),
            ("truth_dissonance", "cognitive_dissonance", "確定と不確定の不協和", "真相を確定させたい欲望と確定しない現実の衝突が、観測の放棄という選択へ導く。", "対抗する解釈と観測の放棄", "答えのない結末の受容", "認知的不協和理論", 0.65, ["rival_theory", "letting_go"], ["truth", "rival"]),
        ],
        "anchor": "答えを確かめようとするほど、相手や状況を変えてしまうことへの戸惑い",
        "drivers": ["訪ねるたびに変わる証言", "書き換わる記録の不気味さ", "自分自身が原因かもしれないという予感"],
        "emotions": ["好奇", "疑念", "自信", "動揺", "孤立", "確信", "戦慄", "静かな解放"],
        "template": (
            "{{ENTANGLED_INVESTIGATOR}}は{{FIXED_TRUTH}}を求めて{{SHIFTING_SUBJECT}}を観測し、{{CHANGING_RECORD}}に全てを記録する。"
            "しかし証言は変わり、記録は書き換わり、{{RIVAL_OBSERVER}}は別の真相を主張する。{{UNSTABLE_REALITY}}の原因が自分の観測だったと知った探索者は、"
            "{{OBSERVATION_COST}}を受け入れて観測をやめる。"
        ),
        "slots": [
            ("ENTANGLED_INVESTIGATOR", "agent", "観測によって対象を変えてしまう探索者", [], ("怪異を調べる記者", "顧客調査を繰り返す担当者", "測定を重ねる研究者"), True),
            ("SHIFTING_SUBJECT", "agent", "観測に応じて変わる対象", ["嘘ではなく変化として描く"], ("訪ねるたびに話が変わる村人", "調査のたびに回答が変わる顧客", "測定条件で振る舞いが変わる現象"), True),
            ("RIVAL_OBSERVER", "agent", "別の解釈を示す観測者", [], ("先に村を調べた学者", "別の手法で分析する競合", "異なる測定系の研究者"), True),
            ("FIXED_TRUTH", "goal", "確定した真相", [], ("怪異の正体", "顧客の本当のニーズ", "現象の真の値"), True),
            ("UNSTABLE_REALITY", "lack", "揺らぐ現実", [], ("訪ねるたびに変わる村の様子", "調査のたびに変わる数値", "再現しない結果"), True),
            ("CHANGING_RECORD", "symbol", "書き換わる記録", ["探索者の筆跡の癖を序盤に一度描く"], ("記者の手帳", "調査の生データ", "実験ノート"), True),
            ("OBSERVATION_COST", "cost", "観測をやめる代償", [], ("記事にしない決断", "過剰な調査の停止と信頼への投資", "測定方法そのものの見直し"), True),
        ],
        "invariants": [
            "対象の変化は嘘ではなく観測への応答として描く",
            "変化が探索者の在不在と連動していることを読者に先に気づかせる",
            "観測者の関与の露見は t≈0.8 に置き、記録という物証で示す",
            "結末では真相を確定させず、観測の放棄による解放で閉じる",
        ],
        "guide": {
            "manga": (
                "全24ページ想定。探索者の手帳のコマを同じ構図で繰り返し挿入し、ページを追うごとに文字が少しずつ変わっている様子を読者だけに見せる。"
                "証言の場面は同じ人物を同じアングルで描きながら表情だけを変え、探索者がいないコマでは変化が起きないことを背景で示す。"
                "18p左ページ終端で手帳の筆跡に気づく引きを作り、めくった19p右ページで探索者自身の手が書き換えている回想を大ゴマで描く。最終ページは白紙の頁の大ゴマで閉じる。"
            ),
            "marketing": (
                "顧客調査を繰り返すほど回答が変わる現象を、調査そのものが顧客の負担や期待を変えていた構造として示す。"
                "過剰な観測をやめて関係そのものに投資する提案を価値として提示し、低負担で始められる導入方法をCTAにする。"
            ),
            "article": (
                "測定や調査の結果が測定方法そのものに影響されていた事例を、観察の記録とともに検証的に記述する。"
                "結論では確定的な答えではなく、観測の影響を織り込んだ判断の手順を読者に提示する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.3), (0.1, 0.38), (0.2, 0.45), (0.3, 0.42), (0.4, 0.55), (0.5, 0.52), (0.6, 0.66), (0.7, 0.62), (0.8, 0.96), (0.9, 0.62), (1.0, 0.48)],
            "release": [(0.0, 0.1), (0.1, 0.2), (0.2, 0.1), (0.3, 0.22), (0.4, 0.08), (0.5, 0.2), (0.6, 0.06), (0.7, 0.15), (0.8, 0.02), (0.9, 0.3), (1.0, 0.45)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.25), (0.4, 0.4), (0.6, 0.55), (0.75, 0.65), (0.8, 0.85), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.1), (0.2, 0.2), (0.4, 0.28), (0.6, 0.35), (0.75, 0.4), (0.8, 0.88), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.5), (0.2, 0.6), (0.4, 0.7), (0.6, 0.8), (0.8, 0.95), (0.9, 0.75), (1.0, 0.6)],
        },
    }
)


if __name__ == "__main__":
    raise SystemExit(main())
