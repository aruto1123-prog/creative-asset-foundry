"""物語原型（narrative_archetype）20パターンの著作スクリプト。

古今東西の神話・古典劇・映画・漫画に通底する「対立構造・情報非対称性・逆転劇」の純粋骨格を、
固有名詞を一切含まない抽象表現で手書き設計し、``UniversalStructure`` として検証したうえで
``core/structure_engine/seeds/narrative_archetypes.json`` を生成する。

実行: ``python -m core.structure_engine.seeds.authoring.build_narrative_archetypes``
検証のみ: ``python -m core.structure_engine.seeds.authoring.build_narrative_archetypes --check``

設計規約:
- 波形は5チャネルすべてを6〜12個のキーフレームで明示し、PCHIP単調補間で21点へ展開する。
- ビートの t は緊張波形のキーフレーム（極値・急変点）に一致させ、tension_delta / information_shift
  の符号を波形の実際の変化と整合させる（``quality_problems`` が機械検査する）。
- 物語論上の原型名・系譜は ``metadata.lineage`` にのみ記載し、5階層の本体には書かない。
- 裏切りエッジには同一ペアの依存エッジを先行させる（裏切りは信頼の存在を前提とする）。

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
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from core.structure_engine.consistency import audit_structure
from core.structure_engine.schemas import (
    WAVE_CHANNELS,
    EdgeKind,
    EmotionalInformationWave,
    UniversalStructure,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "narrative_archetypes.json"
SCHEMA_VERSION = "1.0"
CATEGORY = "narrative_archetype"
DESCRIPTION = (
    "古今東西の神話・古典劇・映画・漫画から抽出した、読者の感情を強烈に揺さぶる対立構造・情報非対称性・"
    "逆転劇の純粋骨格20選。各パターンは21点の感情・情報波形、力学グラフ、因果ビート、認知メカニズム、"
    "抽象変数ブループリント、漫画・マーケティング・記事への転用ガイドを備える。"
)
EXPECTED_COUNT = 20
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
MIN_GUIDE_CHARS = 60
STRONG_DELTA = 0.25

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
    return f"na_{spec['num']:02d}_{spec['slug']}"


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


def quality_problems(spec: dict[str, Any], structure: UniversalStructure) -> list[str]:
    """SPEC §10 の品質基準と波形・ビート整合性を検査し、問題点の一覧を返す。"""
    sid = structure.id
    problems: list[str] = []
    graph = structure.tension_graph
    counts = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "beats": len(structure.beats.beats),
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
    if sum(1 for node in graph.nodes if node.slot) < 2:
        problems.append(f"{sid}: 主要ノードへのスロット割当が不足しています")
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
    wave = structure.wave
    for index, beat in enumerate(structure.beats.beats):
        if index > 0 and not beat.causes:
            problems.append(f"{sid}: ビート {beat.id} に因果（causes）がありません")
        before = max(0.0, beat.t - 0.1)
        after = min(1.0, beat.t + 0.05)
        tension_change = wave.sample("tension", beat.t) - wave.sample("tension", before)
        if beat.tension_delta >= STRONG_DELTA and tension_change <= 0:
            problems.append(f"{sid}: ビート {beat.id} の tension_delta(+) が波形と不整合 ({tension_change:+.3f})")
        if beat.tension_delta <= -STRONG_DELTA and tension_change >= 0:
            problems.append(f"{sid}: ビート {beat.id} の tension_delta(-) が波形と不整合 ({tension_change:+.3f})")
        asym_change = wave.sample("information_asymmetry", after) - wave.sample("information_asymmetry", before)
        if beat.information_shift >= STRONG_DELTA and asym_change <= 0:
            problems.append(f"{sid}: ビート {beat.id} の information_shift(+) が波形と不整合 ({asym_change:+.3f})")
        if beat.information_shift <= -STRONG_DELTA and asym_change >= 0:
            problems.append(f"{sid}: ビート {beat.id} の information_shift(-) が波形と不整合 ({asym_change:+.3f})")
    return problems


def tension_correlations(structures: list[UniversalStructure]) -> list[tuple[str, str, float]]:
    """全ペアの緊張波形 Pearson 相関。"""
    result: list[tuple[str, str, float]] = []
    for first, second in combinations(structures, 2):
        r = float(np.corrcoef(first.wave.channel("tension"), second.wave.channel("tension"))[0, 1])
        result.append((first.id, second.id, r))
    return result


def validate_all(structures: list[UniversalStructure]) -> list[str]:
    """件数・ID一意性・品質基準・波形多様性を総合検査する。"""
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
            problems.append(f"緊張波形が類似しすぎています: {first} × {second} r={r:.4f}")
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
    parser = argparse.ArgumentParser(description="物語原型シード（20パターン）の検証とJSON生成")
    parser.add_argument("--check", action="store_true", help="検証のみ行いJSONを書き出さない")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="出力先JSONパス")
    args = parser.parse_args(argv)
    structures = build_all()
    problems = validate_all(structures)
    if problems:
        for problem in problems:
            print(f"[NG] {problem}")
        return 1
    correlations = tension_correlations(structures)
    max_pair = max(correlations, key=lambda item: item[2]) if correlations else ("", "", 0.0)
    print(f"[OK] {len(structures)} パターン検証完了（最大緊張相関 {max_pair[2]:.4f}: {max_pair[0]} × {max_pair[1]}）")
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
        "slug": "call_and_return",
        "title": "召命と帰還の円環",
        "origin": "myth",
        "summary": (
            "未完成の主体が日常への召命を受け、境界を越えて試練と象徴的な死を経験し、獲得した恩恵を共同体へ持ち帰る円環構造。"
            "出発点と帰着点が同じ場所でありながら、主体の内面と共同体の秩序が不可逆に更新される。"
        ),
        "tags": ["英雄の旅", "変容", "円環構造", "成長"],
        "lineage": ["英雄の旅（単一神話）", "通過儀礼の三段階（分離・移行・統合）", "民話の機能分析"],
        "nodes": [
            ("seeker", "agent", "protagonist", "日常に留まる未完成の主体", "共同体の中で役割を持てず、自覚されない欠落を抱える。召命に応えることで変容の器となる。", 0.95, "PROTAGONIST"),
            ("guide", "agent", "mentor", "境界の向こうを知る導き手", "主体に道具と知恵を与えるが、試練そのものは決して代行しない。", 0.6, "MENTOR_FIGURE"),
            ("guardian", "agent", "threshold_guardian", "既知と未知の境界を守る関門", "越境の資格を問う存在。主体を退けるのではなく代償の支払いを求める。", 0.55, "THRESHOLD_TEST"),
            ("shadow", "agent", "antagonist", "主体の可能性を反転させた影", "主体が選ばなかった道の化身であり、最大の試練で同じ能力を逆方向に振るう。", 0.8, "SHADOW_SELF"),
            ("boon", "desire", "goal", "共同体を癒やす恩恵", "主体個人の利益ではなく、帰還によって初めて意味を持つ価値。", 0.85, "COMMUNAL_BOON"),
            ("wound", "lack", "inner_lack", "主体の自覚されない欠落", "外的目標の裏に隠れた内的課題。最大の試練で直面させられる。", 0.75, "INNER_WOUND"),
            ("threshold", "symbol", "boundary_symbol", "日常と非日常を分かつ境界の象徴", "越境と帰還の二度にわたって現れ、主体の変化の度合いを可視化する。", 0.5, "BOUNDARY_SYMBOL"),
        ],
        "edges": [
            ("seeker", "boon", "pursuit", 0.85, 0.0, "恩恵の探求", 0.1),
            ("guide", "seeker", "asymmetry", 0.6, 0.6, "導き手だけが道を知る", 0.2),
            ("seeker", "guide", "dependency", 0.55, -0.3, "知恵と道具への依存", 0.22),
            ("guardian", "seeker", "power_gradient", 0.65, 0.5, "越境資格の審査", 0.28),
            ("shadow", "seeker", "opposition", 0.9, 0.3, "選ばなかった自己との対決", 0.55),
            ("seeker", "shadow", "asymmetry", 0.5, -0.5, "影は主体の弱点を知り尽くす", 0.5),
            ("shadow", "wound", "embodiment", 0.7, 0.0, "影は欠落の人格化である", 0.55),
            ("threshold", "wound", "embodiment", 0.45, 0.0, "境界が欠落の深さを映す", 0.28),
            ("threshold", "boon", "embodiment", 0.45, 0.0, "帰還の境界で恩恵が意味を持つ", 0.95),
        ],
        "beats": [
            ("ordinary_call", "hook", 0.1, 0.05, "日常への召命", "停滞した日常に異物が侵入し、主体に越境の選択を迫る。", [], 0.2, 0.1, ["seeker", "boon"], False),
            ("refusal", "premise", 0.18, 0.05, "召命の拒否", "主体が恐れから一度拒むことで、日常の引力と欠落の深さが示される。", ["ordinary_call"], -0.1, 0.0, ["seeker", "wound"], False),
            ("crossing", "setup", 0.28, 0.06, "境界の越境", "導き手の後押しと関門の審査を経て、後戻りできない領域へ踏み込む。", ["refusal"], 0.25, 0.05, ["seeker", "guide", "guardian", "threshold"], True),
            ("trials", "escalation", 0.42, 0.1, "試練と仲間", "連続する小試練で主体の能力は拡張されるが、内的欠落は温存されたまま進む。", ["crossing"], -0.05, 0.1, ["seeker", "guide"], False),
            ("ordeal", "crisis", 0.55, 0.06, "最大の試練", "影との対決で主体は象徴的な死を経験し、目を背けてきた欠落と直面させられる。", ["trials"], 0.35, 0.15, ["seeker", "shadow", "wound"], True),
            ("reward", "payoff", 0.65, 0.06, "恩恵の獲得", "死と再生を経た主体が恩恵を掴み、内的欠落が初めて言語化される。", ["ordeal"], -0.3, -0.1, ["seeker", "boon", "wound"], False),
            ("road_back", "climax", 0.82, 0.07, "帰路の追撃と再生", "持ち帰る途上で影が最後の反撃に出て、主体は恩恵を守るために自己を賭す。", ["reward"], 0.4, 0.0, ["seeker", "shadow", "boon"], True),
            ("return", "afterglow", 0.95, 0.05, "円環の帰還", "同じ場所へ戻った主体が共同体を更新し、境界の象徴が変化を証明する。", ["road_back"], -0.4, 0.05, ["seeker", "boon", "threshold"], False),
        ],
        "mechanisms": [
            ("identity_proj", "identity_projection", "未完成性への自己投影", "主体の欠落と停滞が受け手自身の未達感と重なり、旅を自分事として体験させる。", "召命の拒否と欠落の提示", "受け手が主体の成長を自己の可能性として追体験する", "物語移入理論・同一化理論", 0.8, ["ordinary_call", "refusal"], ["seeker", "wound"]),
            ("death_rebirth", "catharsis", "象徴的死と再生のカタルシス", "最大の試練で一度すべてを失わせ、再生によって蓄積した緊張を一挙に解放する。", "最大の試練直後の恩恵獲得", "安堵と高揚の同時発生", "アリストテレスのカタルシス論", 0.85, ["ordeal", "reward"], ["seeker", "shadow"]),
            ("return_anticipation", "anticipation_reward", "帰還への予期と報酬", "円環構造が帰着点を暗示し、受け手は変化した主体が日常に戻る瞬間を予期し続ける。", "境界の象徴の再出現", "予期の充足による深い満足", "予測報酬誤差理論", 0.65, ["crossing", "return"], ["threshold"]),
            ("peak_end_memory", "peak_end", "ピークと終幕の記憶", "帰路の追撃というピークと静かな帰還という終幕が、体験全体の評価を決定する。", "クライマックス直後の静かな帰還", "物語全体を肯定的に記憶させる", "ピーク・エンドの法則", 0.6, ["road_back", "return"], []),
        ],
        "anchor": "召命を一度拒んでしまう主体の臆病さと、その裏に隠れた自覚されない欠落",
        "drivers": ["境界を越えるたびに更新される世界の規則", "導き手との非対称な知識関係", "影が主体の弱点を知り尽くしている不安"],
        "emotions": ["好奇", "躊躇", "高揚", "恐怖", "絶望", "安堵", "決意", "充足"],
        "template": (
            "停滞した日常に{{CALL_TO_ADVENTURE}}が訪れ、{{PROTAGONIST}}は{{INNER_WOUND}}ゆえに一度それを拒む。"
            "{{MENTOR_FIGURE}}の後押しで{{THRESHOLD_TEST}}を越えた先で、{{PROTAGONIST}}は{{SHADOW_SELF}}と対決して象徴的な死を経験し、"
            "{{COMMUNAL_BOON}}を得る。帰路の追撃を退けて元の場所へ戻ったとき、{{BOUNDARY_SYMBOL}}の変化が{{TRANSFORMATION_PROOF}}を証明する。"
        ),
        "slots": [
            ("PROTAGONIST", "agent", "日常に留まる未完成の主体", ["冒頭で欠落を行動として見せる"], ("才能を持て余す見習い", "現状に不満を抱える見込み顧客", "常識に疑問を持ち始めた読者"), True),
            ("CALL_TO_ADVENTURE", "conflict", "日常に侵入する越境の誘い", [], ("謎めいた依頼", "業界構造の変化の兆し", "通説を揺るがす観測事実"), True),
            ("INNER_WOUND", "lack", "外的目標の裏に隠れた内的課題", ["終盤まで主体自身は言語化できない"], ("過去の失敗への恐れ", "変化への自信のなさ", "前提を疑う習慣の欠如"), True),
            ("MENTOR_FIGURE", "agent", "道具と知恵を与える導き手", ["試練を代行しない"], ("引退した師匠", "先行して成功した導入者", "先行研究の知見"), True),
            ("THRESHOLD_TEST", "conflict", "越境の資格を問う関門", [], ("入門の試験", "最初の小さな決断", "反論への最初の検証"), True),
            ("SHADOW_SELF", "irony", "主体が選ばなかった道の化身", ["主体と同じ能力を逆方向に使う"], ("闇に堕ちた同門", "旧来手法に固執する過去の自分", "誤った一般化"), True),
            ("COMMUNAL_BOON", "goal", "帰還によって意味を持つ恩恵", [], ("故郷を救う秘法", "チーム全体の成果", "読者が持ち帰る判断の枠組み"), True),
            ("BOUNDARY_SYMBOL", "symbol", "越境と帰還で二度現れる境界の象徴", [], ("故郷の門", "導入前後の比較画面", "冒頭と結論で繰り返す同一の問い"), True),
            ("TRANSFORMATION_PROOF", "reversal", "主体の不可逆な変化を示す証拠", [], ("かつての門番が頭を下げる", "数値で示す変化", "同じ問いへの新しい答え"), True),
        ],
        "invariants": [
            "出発点と帰着点は同一の場所・状況で描き、差分のみで変化を示す",
            "最大の試練（象徴的な死）は中盤後半 t≈0.5〜0.6 に置き、その直後に恩恵を与える",
            "帰路に最後の追撃を置き、獲得した恩恵を一度危機にさらす",
            "導き手は試練を代行しない（主体の選択だけが変化の原因である）",
        ],
        "guide": {
            "manga": (
                "全32ページ想定。1〜4pで日常と欠落、5pで召命、6〜7pで拒否。8〜9pの見開きで越境を大ゴマ化し、めくり（9→10p）で非日常の全景を見せる。"
                "最大の試練は18〜20pに置き、20p左ページ終端を主体が倒れ伏す引きゴマで締め、めくった21p右ページ冒頭で再生を大ゴマにする。"
                "帰路の追撃は26〜28p、最終2ページは冒頭1pと同じ構図で描き差分だけで変化を語る。"
            ),
            "marketing": (
                "顧客を主体、ブランドを導き手に配置する。召命は課題の顕在化、拒否は導入への躊躇として正面から扱い、最初の小さな決断（無料体験）を関門に設定する。"
                "最大の試練は導入初期の挫折として予告しておき、恩恵は顧客個人ではなく所属組織への波及で語る。"
            ),
            "article": (
                "冒頭で読者の現状と問いを提示し、通説を疑う契機を召命として示す。反論の検証を関門、論証上の最大の困難を試練として正直に描き、"
                "得られた枠組みを結論で冒頭の問いに再適用して変化を証明する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.15), (0.1, 0.35), (0.18, 0.28), (0.28, 0.5), (0.42, 0.45), (0.55, 0.78), (0.65, 0.5), (0.82, 0.92), (0.92, 0.4), (1.0, 0.18)],
            "release": [(0.0, 0.05), (0.18, 0.12), (0.3, 0.1), (0.45, 0.2), (0.62, 0.55), (0.75, 0.15), (0.88, 0.6), (1.0, 0.9)],
            "audience_knowledge": [(0.0, 0.1), (0.12, 0.25), (0.3, 0.35), (0.5, 0.5), (0.6, 0.65), (0.82, 0.85), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.05), (0.12, 0.15), (0.3, 0.3), (0.5, 0.45), (0.6, 0.7), (0.82, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.25), (0.15, 0.35), (0.3, 0.5), (0.45, 0.55), (0.58, 0.7), (0.7, 0.5), (0.85, 0.75), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 2,
        "slug": "recognition_reversal",
        "title": "発見と逆転の悲劇",
        "origin": "drama",
        "summary": (
            "共同体の災厄の原因を究明する主体が、追及の果てに自らこそが原因であったと発見する。真相の発見と運命の逆転が同一の瞬間に重なり、"
            "読者と主体が同時に真実へ到達することで最大の恐れと憐れみが生まれる。"
        ),
        "tags": ["悲劇", "自己発見", "運命の逆転", "真相究明"],
        "lineage": ["アリストテレス的発見と逆転（アナグノリシスとペリペテイア）", "古典悲劇の運命劇", "探偵役が原因であるという逆説"],
        "nodes": [
            ("investigator", "agent", "protagonist", "災厄の原因を追及する責任ある主体", "共同体への責任感から真相究明を誓うが、その誓い自体が自らを追い詰める装置となる。", 0.95, "PROTAGONIST"),
            ("witness", "agent", "reluctant_witness", "真相を知りながら口を閉ざす証人", "主体を守るために沈黙するが、追及に屈して決定的な証言を与えてしまう。", 0.6, "RELUCTANT_WITNESS"),
            ("seer", "agent", "truth_teller", "真実を告げても信じられない告知者", "最初から答えを示すが、主体の自己像がそれを拒絶させる。", 0.55, "IGNORED_PROPHECY"),
            ("truth", "desire", "goal", "災厄の原因の解明", "共同体を救うための外的目標であり、達成の瞬間に主体自身を破壊する。", 0.9, "SOUGHT_TRUTH"),
            ("blindness", "lack", "tragic_flaw", "自己を例外視する認知の盲点", "主体の聡明さそのものが、自分を疑うことを不可能にしている。", 0.85, "TRAGIC_FLAW"),
            ("plague", "symbol", "communal_curse", "共同体を蝕む災厄の象徴", "主体の隠された過去が外部化したもの。真相の解明と共に意味が反転する。", 0.6, "COMMUNAL_CURSE"),
        ],
        "edges": [
            ("investigator", "truth", "pursuit", 0.95, 0.0, "真相究明の誓い", 0.08),
            ("seer", "investigator", "asymmetry", 0.75, 0.8, "告知者だけが答えを知る", 0.22),
            ("investigator", "seer", "opposition", 0.6, 0.4, "告知の拒絶と糾弾", 0.25),
            ("witness", "investigator", "asymmetry", 0.7, 0.7, "証人が決定的事実を秘匿する", 0.45),
            ("investigator", "witness", "power_gradient", 0.65, 0.7, "統治的権力による証言の強制", 0.72),
            ("investigator", "blindness", "dependency", 0.7, 0.0, "自己像の安定を盲点に依存する", None),
            ("plague", "blindness", "embodiment", 0.6, 0.0, "災厄が盲点の帰結を体現する", 0.82),
            ("truth", "investigator", "opposition", 0.85, -0.2, "解明された真相が主体自身を告発する", 0.82),
        ],
        "beats": [
            ("curse_hook", "hook", 0.0, 0.05, "災厄の提示", "共同体を蝕む災厄が示され、責任ある主体に救済が委ねられる。", [], 0.2, 0.05, ["investigator", "plague"], False),
            ("oath", "premise", 0.08, 0.04, "究明の誓い", "主体が原因の徹底追及と厳罰を公に誓い、自らの退路を断つ。", ["curse_hook"], 0.1, 0.0, ["investigator", "truth"], False),
            ("ignored_prophecy", "setup", 0.22, 0.05, "退けられた告知", "告知者が答えを示すが、主体は自己像を守るために告知者を糾弾する。読者だけが疑いを抱く。", ["oath"], 0.05, 0.25, ["seer", "investigator", "blindness"], False),
            ("first_cracks", "escalation", 0.45, 0.08, "綻びの連鎖", "細部の証言が少しずつ主体の過去と符合し始めるが、主体は別の解釈で塞いでしまう。", ["ignored_prophecy"], 0.1, 0.05, ["investigator", "witness", "blindness"], False),
            ("false_relief", "false_victory", 0.6, 0.05, "偽の安堵", "主体を脅かすはずの予言が外れたかに見える報せが届き、一時的に疑いが晴れる。", ["first_cracks"], -0.15, 0.0, ["investigator", "seer"], False),
            ("forced_testimony", "escalation", 0.72, 0.05, "証言の強制", "主体が権力で証人の沈黙を破らせ、自ら決定的な事実を引き出してしまう。", ["false_relief"], 0.3, 0.1, ["investigator", "witness"], True),
            ("anagnorisis", "revelation", 0.82, 0.04, "発見と逆転", "追及の対象と追及者が同一であったと判明し、救済者が災厄の原因へと反転する。", ["forced_testimony", "ignored_prophecy"], 0.35, -0.3, ["investigator", "truth", "blindness", "plague"], True),
            ("self_verdict", "costly_breakthrough", 0.9, 0.05, "自らへの裁き", "主体は誓いのとおり自らを罰し、見ることを拒んできた代償を身に刻む。", ["anagnorisis", "oath"], -0.25, 0.0, ["investigator"], True),
            ("tragic_afterglow", "afterglow", 1.0, 0.05, "恐れと憐れみの余韻", "災厄は去るが主体は共同体を去り、聡明さの限界という普遍的教訓が残る。", ["self_verdict"], -0.4, 0.0, ["investigator", "plague"], False),
        ],
        "mechanisms": [
            ("irony_gap", "dramatic_irony", "読者だけが先に疑う劇的アイロニー", "告知の段階で読者は真相の輪郭を掴むが主体は拒絶し続け、その差分が一挙手一投足に不吉な二重の意味を与える。", "退けられた告知と綻びの連鎖", "主体の言動すべてに戦慄の二重性を感じる", "劇的アイロニー論", 0.85, ["ignored_prophecy", "first_cracks"], ["seer", "investigator"]),
            ("fear_pity", "catharsis", "恐れと憐れみのカタルシス", "善意と聡明さを持つ主体が自らの盲点で破滅することで、受け手は自分にも起こりうる恐れと主体への憐れみを同時に抱く。", "発見と逆転の同時発生", "感情の浄化と長い余韻", "アリストテレスのカタルシス論", 0.9, ["anagnorisis", "self_verdict"], ["investigator"]),
            ("self_dissonance", "cognitive_dissonance", "自己像と事実の不協和", "主体が反証を別解釈で塞ぐ過程が、受け手自身の自己正当化の癖を鏡のように映す。", "綻びを塞ぐ主体の再解釈", "居心地の悪い自己認識", "認知的不協和理論", 0.7, ["first_cracks", "false_relief"], ["blindness"]),
            ("oath_justice", "moral_judgment", "誓いの帰結への道徳的判断", "主体が公に立てた誓いが自分自身に適用される構図が、公正の感覚を強く刺激する。", "究明の誓いと自らへの裁き", "因果応報への畏怖", "公正世界信念", 0.65, ["oath", "self_verdict"], ["investigator", "truth"]),
        ],
        "anchor": "共同体を救おうとする善意と聡明さが、そのまま自分を破滅へ導いてしまう主体の誠実さ",
        "drivers": ["読者だけが真相の輪郭を先に掴む情報差", "証言が一つずつ主体の過去と符合していく推理の快楽", "誓いが自分に跳ね返ると予感させる構図"],
        "emotions": ["不安", "疑念", "戦慄", "偽りの安堵", "恐怖", "憐れみ", "畏怖"],
        "template": (
            "{{COMMUNAL_CURSE}}に苦しむ共同体を救うため、{{PROTAGONIST}}は{{SOUGHT_TRUTH}}の徹底究明と厳罰を誓う。"
            "{{IGNORED_PROPHECY}}は最初から答えを示しているが、{{TRAGIC_FLAW}}ゆえに退けられる。{{FALSE_RELIEF}}で疑いが一度晴れた直後、"
            "{{RELUCTANT_WITNESS}}の証言によって追及者自身が原因であると判明し、主体は{{SELF_VERDICT}}を自らに下す。"
        ),
        "slots": [
            ("PROTAGONIST", "agent", "災厄の原因を追及する責任ある主体", ["聡明さと善意を冒頭で証明する"], ("名推理で知られる若き領主", "改善を主導する責任者", "自説を検証する研究者"), True),
            ("COMMUNAL_CURSE", "symbol", "共同体を蝕む災厄の象徴", [], ("原因不明の連続失踪", "組織に蔓延する慢性的な不振", "業界で繰り返される失敗"), True),
            ("SOUGHT_TRUTH", "goal", "解明すべき原因", [], ("失踪事件の黒幕", "不振の真因", "失敗の構造的原因"), True),
            ("IGNORED_PROPHECY", "irony", "最初から示されている答え", ["序盤で明示し読者の記憶に残す"], ("占い師の不吉な一言", "現場担当者の率直な指摘", "冒頭で棄却した仮説"), True),
            ("TRAGIC_FLAW", "lack", "自己を例外視する認知の盲点", [], ("自分だけは正しいという確信", "成功体験への固執", "自説への確証バイアス"), True),
            ("FALSE_RELIEF", "reversal", "疑いを一度晴らす偽の安堵", [], ("容疑を晴らす偽のアリバイ", "一時的な数値の改善", "一見有利なデータ"), True),
            ("RELUCTANT_WITNESS", "agent", "真相を知りながら沈黙する証人", [], ("主人公を育てた老僕", "長年仕えた部下", "過去の自分の記録"), True),
            ("SELF_VERDICT", "cost", "自らに下す裁きと代償", [], ("自ら職を辞して街を去る", "責任者自身の撤退と刷新", "自説の公開撤回"), True),
        ],
        "invariants": [
            "追及者と追及対象の同一性は発見の瞬間まで主体に隠し、読者には告知の段階で輪郭だけを与える",
            "発見（認知の転換）と逆転（運命の転換）は同一ビートで起こす",
            "主体を破滅させるのは外敵ではなく、主体自身の美徳と盲点である",
            "偽の安堵を発見の直前に置き、落差を最大化する",
        ],
        "guide": {
            "manga": (
                "全24ページ想定。1〜3pで災厄と主体の聡明さ、4pで誓いを決め台詞の大ゴマにする。7〜8pの告知場面に読者だけが違和感を持つ小ゴマを挿入する。"
                "12〜15pで証言の綻び、16pで偽の安堵。めくり（17→18p）の右ページ冒頭で決定的証言を置き、19〜20pの見開きで発見と逆転を一枚絵にする。"
                "21〜23pは無音の静寂ゴマを多用して自らへの裁きを描き、24pは去りゆく主体の背中で余韻を残す。"
            ),
            "marketing": (
                "顧客が原因は外部にあると信じて解決策を探す構図を描き、診断の結果、原因が顧客自身の慣行にあったと気づかせる。"
                "発見は責めではなく救済の入口として提示し、自らへの裁きを習慣の刷新という前向きな代償へ置き換える。"
            ),
            "article": (
                "通説に沿った原因追及を誠実に進めた結果、問題を生んでいたのは追及に使っていた前提そのものだったと示す構成。"
                "偽の安堵として一見有利なデータを置き、直後の反証で枠組みを反転させる。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.35), (0.08, 0.45), (0.22, 0.5), (0.45, 0.62), (0.6, 0.5), (0.72, 0.74), (0.82, 0.98), (0.9, 0.72), (1.0, 0.3)],
            "release": [(0.0, 0.05), (0.3, 0.08), (0.6, 0.25), (0.8, 0.05), (0.88, 0.5), (0.95, 0.8), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.15), (0.2, 0.45), (0.45, 0.55), (0.7, 0.62), (0.8, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.2, 0.15), (0.45, 0.25), (0.7, 0.35), (0.8, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.3), (0.25, 0.45), (0.5, 0.6), (0.7, 0.75), (0.82, 0.9), (0.92, 0.5), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 3,
        "slug": "devils_bargain",
        "title": "禁じられた取引と代償",
        "origin": "literature",
        "summary": (
            "欠落に苦しむ主体が、禁じられた力を持つ存在と取引して望みを即座に手に入れる。前半は上昇と陶酔が続くが、"
            "契約に隠された代償が後半に一挙に取り立てられ、獲得したものすべてが反転する。"
        ),
        "tags": ["契約", "代償", "欲望", "破滅と救済"],
        "lineage": ["悪魔との契約譚", "願いの代償（三つの願い型民話）", "ファウスト的取引"],
        "nodes": [
            ("bargainer", "agent", "protagonist", "限界に絶望した野心的な主体", "正当な努力では埋められない欠落を前に、禁じられた近道を選んでしまう。", 0.95, "PROTAGONIST"),
            ("tempter", "agent", "tempter", "望みを即座に叶える取引相手", "主体の欠落を正確に見抜き、代償を曖昧な条項に隠して契約を持ちかける。", 0.85, "TEMPTER"),
            ("innocent", "agent", "collateral", "契約の余波を受ける無垢な第三者", "主体が得た力の恩恵と被害を同時に受け、代償の重さを可視化する。", 0.6, "COLLATERAL_INNOCENT"),
            ("wish", "desire", "goal", "即座に得られる成功と承認", "契約によって叶えられる表層の欲望。達成されるほど空虚さが増していく。", 0.8, "SURFACE_WISH"),
            ("inadequacy", "lack", "inner_lack", "自己の無力さへの耐えがたい感覚", "取引を選ばせた根源的な欠落。力を得ても決して埋まらない。", 0.85, "CORE_INADEQUACY"),
            ("contract", "symbol", "binding_symbol", "署名された契約の象徴", "取引の拘束力と隠された条項を一つの物に凝縮した象徴。", 0.7, "HIDDEN_CLAUSE"),
            ("soul", "desire", "true_value", "主体が本当に守るべきだった価値", "取引で差し出したものの真の重さ。失いかけて初めて認識される。", 0.75, "TRUE_VALUE"),
        ],
        "edges": [
            ("bargainer", "wish", "pursuit", 0.9, 0.0, "近道による欲望の追求", 0.1),
            ("tempter", "bargainer", "asymmetry", 0.9, 0.9, "取引相手だけが条項の全貌を知る", 0.15),
            ("bargainer", "tempter", "dependency", 0.8, -0.6, "力の源泉への依存", 0.22),
            ("tempter", "bargainer", "power_gradient", 0.8, 0.85, "契約による支配", 0.3),
            ("tempter", "bargainer", "betrayal", 0.85, 0.8, "隠された条項による取り立て", 0.75),
            ("contract", "soul", "embodiment", 0.7, 0.0, "契約書が差し出した価値を体現する", 0.22),
            ("contract", "inadequacy", "embodiment", 0.5, 0.0, "署名の瞬間に欠落が固定される", 0.22),
            ("innocent", "bargainer", "dependency", 0.5, -0.3, "主体の得た力に生活を委ねる", 0.45),
            ("wish", "soul", "opposition", 0.7, 0.0, "表層の欲望と真の価値の対立", 0.6),
        ],
        "beats": [
            ("despair_hook", "hook", 0.0, 0.05, "限界の露呈", "主体の正当な努力が決定的に報われない瞬間が示され、欠落が耐えがたいものになる。", [], 0.2, 0.0, ["bargainer", "inadequacy"], False),
            ("temptation", "premise", 0.12, 0.05, "取引の申し出", "欠落を正確に見抜いた取引相手が、即座に望みを叶える条件を提示する。", ["despair_hook"], 0.1, 0.15, ["tempter", "bargainer", "inadequacy"], False),
            ("signing", "setup", 0.22, 0.04, "署名", "主体が条項を読み切らぬまま署名して力を得る。緊張は一時的に陶酔へ転じる。", ["temptation"], -0.35, 0.1, ["bargainer", "tempter", "contract"], True),
            ("ascent", "escalation", 0.4, 0.1, "陶酔の上昇", "望みが次々と叶い主体は承認と成功を手にするが、読者だけが小さな不吉の兆しに気づく。", ["signing"], -0.1, 0.2, ["bargainer", "wish", "innocent"], False),
            ("first_toll", "local_optimization_limit", 0.55, 0.05, "最初の取り立て", "成功の裏で無垢な第三者が傷つき、埋めるはずだった欠落がむしろ広がっている。", ["ascent"], 0.15, 0.05, ["innocent", "bargainer", "inadequacy"], False),
            ("clause_revealed", "revelation", 0.75, 0.05, "隠された条項の発動", "契約の真の条項が明かされ、主体が差し出していたものの正体が判明する。", ["first_toll", "signing"], 0.35, -0.3, ["tempter", "contract", "soul"], True),
            ("collection", "crisis", 0.86, 0.05, "取り立ての極点", "力も成功も一斉に反転し、主体は最も大切なものを失う寸前まで追い詰められる。", ["clause_revealed"], 0.1, 0.0, ["bargainer", "tempter", "soul", "innocent"], True),
            ("redemptive_choice", "costly_breakthrough", 0.94, 0.04, "代償を引き受ける選択", "主体は得たものすべてを自ら手放し、真の価値を守るための代償を引き受ける。", ["collection"], -0.3, 0.0, ["bargainer", "soul", "contract"], True),
            ("bitter_afterglow", "afterglow", 1.0, 0.05, "苦い余韻", "欠落は埋まらないまま残るが、主体は近道の代わりに自らの限界を引き受けて歩き出す。", ["redemptive_choice"], -0.2, 0.0, ["bargainer", "inadequacy"], False),
        ],
        "mechanisms": [
            ("fear_of_loss", "loss_aversion", "得たものを失う恐怖", "前半で積み上げた成功が大きいほど、取り立てで失うものへの恐怖が増幅される。", "陶酔の上昇の後の取り立て", "喪失への強い回避感情と緊張", "プロスペクト理論", 0.85, ["ascent", "collection"], ["wish", "bargainer"]),
            ("ominous_irony", "dramatic_irony", "読者だけが知る不吉の兆し", "署名の瞬間に条項の曖昧さを読者にだけ示し、成功の描写すべてに不穏な影を落とす。", "署名場面で読み飛ばされた一文", "幸福な場面ほど不安が募る", "劇的アイロニー論", 0.75, ["signing", "ascent"], ["contract"]),
            ("moral_cost", "moral_judgment", "近道への道徳的判断", "努力を飛ばして得た報酬が取り立てられる構図が、受け手の公正感を満たしつつ主体への同情も生む。", "無垢な第三者の被害", "因果への納得と憐れみの同居", "公正世界信念", 0.7, ["first_toll", "redemptive_choice"], ["innocent"]),
            ("temptation_projection", "identity_projection", "誘惑への自己投影", "誰もが一度は願う近道を主体が選ぶため、受け手は主体の選択を他人事として切り離せない。", "欠落を見抜いた取引の申し出", "自分も同じ選択をしたかもしれないという動揺", "同一化理論", 0.65, ["temptation"], ["bargainer", "inadequacy"]),
        ],
        "anchor": "正当な努力が報われない絶望の中で、近道に手を伸ばしてしまう主体の弱さ",
        "drivers": ["条項の全貌を知らない主体と兆しに気づく読者の情報差", "叶い続ける望みが空虚さを増していく逆説", "取り立てがいつ来るかという予感"],
        "emotions": ["絶望", "誘惑", "陶酔", "違和感", "恐怖", "喪失", "覚悟"],
        "template": (
            "{{CORE_INADEQUACY}}に追い詰められた{{PROTAGONIST}}は、{{TEMPTER}}から{{SURFACE_WISH}}を即座に叶える取引を持ちかけられ、"
            "{{HIDDEN_CLAUSE}}を読み切らぬまま署名する。望みが叶うほど{{COLLATERAL_INNOCENT}}が傷つき、やがて条項が発動して{{TRUE_VALUE}}が"
            "取り立てられようとする。主体は{{REDEMPTIVE_PRICE}}を引き受けることでのみ、真の価値を守り抜く。"
        ),
        "slots": [
            ("PROTAGONIST", "agent", "限界に絶望した野心的な主体", ["冒頭で正当な努力が報われない場面を見せる"], ("才能の壁に阻まれた新人作家", "成果を急ぐ事業責任者", "即効性のある答えを求める読者"), True),
            ("CORE_INADEQUACY", "lack", "取引を選ばせた根源的な欠落", [], ("作品が誰にも届かない無力感", "競合に勝てない焦燥", "成果が出ない不安"), True),
            ("TEMPTER", "agent", "望みを即座に叶える取引相手", ["欠落を正確に言い当てる"], ("願いを叶える謎の存在", "短期的成果を約束する手法", "安易な成功法則"), True),
            ("SURFACE_WISH", "goal", "契約で叶う表層の欲望", [], ("一夜にして得る名声", "急激な売上の増加", "すぐに使える小手先の技"), True),
            ("HIDDEN_CLAUSE", "symbol", "代償を隠した契約の象徴", ["署名場面で読者にだけ一瞬示す"], ("小さな文字の一文", "見落とされた利用条件", "前提条件の注記"), True),
            ("COLLATERAL_INNOCENT", "agent", "契約の余波を受ける無垢な第三者", [], ("主人公を慕う幼なじみ", "疲弊する現場の担当者", "誤った手法を真似る後輩"), True),
            ("TRUE_VALUE", "goal", "主体が本当に守るべき価値", [], ("自分の言葉で描く誇り", "顧客との信頼関係", "長期的な理解力"), True),
            ("REDEMPTIVE_PRICE", "cost", "真の価値を守るために引き受ける代償", [], ("得た名声をすべて返上する", "短期施策を止めて信頼回復に投資する", "近道を捨てて基礎から学び直す"), True),
        ],
        "invariants": [
            "取引の成立は序盤 t≈0.2 に置き、その直後の緊張を大きく下げて陶酔を描く",
            "代償の存在は署名場面で読者にだけ示し、主体には条項の発動まで隠す",
            "取り立ての大きさは獲得したものの大きさに比例させる",
            "解決は契約の破棄ではなく、主体が自ら代償を引き受ける選択によってのみ行う",
        ],
        "guide": {
            "manga": (
                "全40ページ想定。1〜4pで報われない努力、5〜7pで取引の申し出。8〜9pの見開きで署名を描き、条項の一文を小ゴマで読者にだけ見せる。"
                "10〜22pは成功の上昇を明るいトーンとコマ数多めのテンポで描き、各エピソード末に不吉な兆しの小ゴマを一つ置く。23〜26pで最初の取り立て、"
                "めくり（29→30p）の右ページで条項の発動を断ち切りの大ゴマにする。34〜37pで取り立ての極点、38〜39pで代償を引き受ける選択、40pは静かな余韻。"
            ),
            "marketing": (
                "短期的な近道（誇大な成功法則）を取引相手として描き、その代償（信頼の喪失・燃え尽き）を顧客が自分事として想像できるよう具体化する。"
                "自社の提案は近道ではなく、代償を正直に提示したうえで真の価値を守る選択として位置づける。"
            ),
            "article": (
                "即効性のある手法がもたらす初期の成功をまず公平に認め、その後に表面化する隠れたコストをデータで示す。"
                "結論では近道を全否定するのではなく、代償を理解したうえで選ぶための判断基準を提示する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.55), (0.12, 0.65), (0.22, 0.3), (0.4, 0.22), (0.55, 0.38), (0.66, 0.55), (0.75, 0.88), (0.86, 0.96), (0.94, 0.6), (1.0, 0.4)],
            "release": [(0.0, 0.05), (0.22, 0.65), (0.4, 0.7), (0.55, 0.4), (0.75, 0.05), (0.9, 0.2), (1.0, 0.75)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.35), (0.35, 0.55), (0.6, 0.65), (0.75, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.2, 0.2), (0.35, 0.22), (0.6, 0.3), (0.75, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.35), (0.2, 0.5), (0.4, 0.3), (0.6, 0.45), (0.75, 0.8), (0.88, 0.75), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 4,
        "slug": "ashes_to_recognition",
        "title": "灰からの上昇と真価の露見",
        "origin": "myth",
        "summary": (
            "不当に低い地位へ押し込められた主体が、隠された真価を一夜だけ発揮する機会を得る。発揮した真価は時間制限によって一度失われ、"
            "残された痕跡を手がかりに真価が公に認証されることで、地位と評価が反転する。"
        ),
        "tags": ["逆転", "真価の承認", "時間制限", "地位の反転"],
        "lineage": ["灰かぶり型民話（虐げられた者の上昇）", "身分の隠蔽と露見", "認知の印による同定"],
        "nodes": [
            ("overlooked", "agent", "protagonist", "真価を隠され不当に扱われる主体", "勤勉さと誠実さを持ちながら、環境によって低い地位に固定されている。", 0.95, "PROTAGONIST"),
            ("oppressor", "agent", "oppressor", "主体の真価を抑圧する身近な権力", "主体の価値を薄々知りながら、自らの地位を守るために隠蔽する。", 0.75, "OPPRESSOR"),
            ("helper", "agent", "magical_helper", "一夜だけ機会を与える援助者", "主体の誠実さへの返礼として、制限付きの変身の機会を授ける。", 0.55, "HELPER"),
            ("recognizer", "agent", "recognizer", "真価を見抜く高位の認証者", "表層ではなく本質を見る視点を持ち、消えた主体を探し出す。", 0.7, "RECOGNIZER"),
            ("recognition", "desire", "goal", "真価の公的な承認", "地位の上昇そのものではなく、本来の自分として見られること。", 0.85, "TRUE_RECOGNITION"),
            ("invisibility", "lack", "inner_lack", "誰にも見られない存在であること", "主体が慣れてしまった透明化の痛み。", 0.7, "INVISIBILITY"),
            ("token", "symbol", "proof_token", "真価を証明する唯一の痕跡", "時間制限の瞬間に残された物。持ち主の同定によって真価を証明する。", 0.75, "PROOF_TOKEN"),
        ],
        "edges": [
            ("oppressor", "overlooked", "power_gradient", 0.85, 0.85, "身近な権力による地位の固定", 0.0),
            ("oppressor", "overlooked", "asymmetry", 0.6, 0.6, "抑圧者だけが真価を知り隠蔽する", 0.1),
            ("overlooked", "recognition", "pursuit", 0.75, 0.0, "本来の自分として見られる願い", 0.15),
            ("helper", "overlooked", "asymmetry", 0.5, 0.4, "援助者だけが機会の規則を知る", 0.3),
            ("overlooked", "helper", "dependency", 0.6, -0.4, "一夜の変身への依存", 0.3),
            ("recognizer", "overlooked", "pursuit", 0.7, 0.0, "痕跡を手がかりにした探索", 0.72),
            ("oppressor", "recognizer", "opposition", 0.65, 0.2, "同定を妨害する偽装", 0.8),
            ("token", "recognition", "embodiment", 0.8, 0.0, "痕跡が真価の証明を体現する", 0.6),
            ("token", "invisibility", "embodiment", 0.4, 0.0, "残された痕跡が透明化の終わりを予告する", 0.62),
        ],
        "beats": [
            ("invisible_hook", "hook", 0.0, 0.05, "不当な扱い", "誠実な主体が身近な権力によって低い地位へ固定され、真価を誰にも見られない日常が示される。", [], 0.2, 0.1, ["overlooked", "oppressor", "invisibility"], False),
            ("denied_chance", "premise", 0.12, 0.05, "機会の剥奪", "真価を示す場が公に開かれるが、抑圧者が主体の参加を禁じる。", ["invisible_hook"], 0.1, 0.0, ["oppressor", "overlooked", "recognition"], False),
            ("gift_of_night", "setup", 0.3, 0.05, "制限付きの機会", "援助者が誠実さへの返礼として一夜の変身を授け、時間制限という規則を告げる。", ["denied_chance"], -0.2, 0.15, ["helper", "overlooked"], True),
            ("radiance", "payoff", 0.45, 0.06, "真価の発揮", "変身した主体が場の中心で真価を示し、認証者だけがその本質に惹かれる。", ["gift_of_night"], 0.15, 0.1, ["overlooked", "recognizer"], False),
            ("midnight_loss", "crisis", 0.6, 0.04, "時間切れの喪失", "制限時刻の到来で主体は逃げ去り、痕跡だけが残される。真価は再び見えなくなる。", ["radiance"], 0.25, 0.05, ["overlooked", "token", "recognizer"], True),
            ("search", "escalation", 0.72, 0.08, "痕跡による探索", "認証者が痕跡を手がかりに持ち主を探し、抑圧者は偽装で同定を妨げる。", ["midnight_loss"], -0.1, 0.1, ["recognizer", "oppressor", "token"], False),
            ("false_claimants", "local_optimization_limit", 0.82, 0.05, "偽の持ち主たち", "抑圧者側の偽装が痕跡に合わず次々と失敗し、主体の存在は隠され続ける。", ["search"], 0.25, 0.0, ["oppressor", "recognizer", "token"], True),
            ("recognition_moment", "revelation", 0.92, 0.04, "真価の認証", "痕跡が主体に合致し、透明だった存在が公に同定されて地位と評価が反転する。", ["false_claimants", "radiance"], -0.4, -0.4, ["overlooked", "recognizer", "token", "recognition"], True),
            ("ascended_afterglow", "afterglow", 1.0, 0.05, "上昇の余韻", "主体は抑圧者を罰するのではなく許し、見られることを得た静かな幸福が示される。", ["recognition_moment"], -0.2, 0.0, ["overlooked", "oppressor"], False),
        ],
        "mechanisms": [
            ("underdog_empathy", "empathy_transfer", "虐げられた者への共感移入", "誠実さが報われない不当さが受け手の公正感を刺激し、主体の側に感情を強く固定する。", "不当な扱いと機会の剥奪", "主体の勝利を自分のことのように願う", "共感移入・公正世界信念", 0.85, ["invisible_hook", "denied_chance"], ["overlooked", "oppressor"]),
            ("ticking_clock", "scarcity", "時間制限による希少性", "一夜限りという制限が、真価を発揮できる時間を希少で切実なものにする。", "援助者が告げる時間制限", "幸福な場面にも切迫感が混じる", "希少性の原理", 0.7, ["gift_of_night", "midnight_loss"], ["helper"]),
            ("identity_suspense", "dramatic_irony", "正体を知る読者のサスペンス", "読者だけが痕跡の持ち主を知っており、探索と偽装の場面すべてに焦燥を生む。", "偽の持ち主たちの失敗", "早く気づいてほしいという焦燥", "劇的アイロニー論", 0.75, ["search", "false_claimants"], ["token", "recognizer"]),
            ("status_contrast", "contrast_effect", "地位の落差による対比効果", "最底辺の扱いと公的な認証の落差が大きいほど、反転の快感が増幅される。", "真価の認証の瞬間", "爽快な逆転のカタルシス", "対比効果", 0.8, ["recognition_moment"], ["overlooked", "recognition"]),
        ],
        "anchor": "不当に扱われても誠実さを手放さない主体の、誰にも見られない日々の痛み",
        "drivers": ["読者だけが主体の真価と痕跡の持ち主を知る情報差", "一夜限りの時間制限", "偽の持ち主が次々と失敗する反復"],
        "emotions": ["憤り", "憐れみ", "期待", "高揚", "喪失", "焦燥", "爽快", "安らぎ"],
        "template": (
            "{{INVISIBILITY}}に慣れきった{{PROTAGONIST}}は、{{OPPRESSOR}}によって真価を示す場から締め出されている。"
            "{{HELPER}}が授けた{{TIME_LIMIT}}付きの機会で主体は真価を発揮して{{RECOGNIZER}}の目に留まるが、制限の到来で{{PROOF_TOKEN}}だけを残して姿を消す。"
            "痕跡を手がかりにした探索と偽装の失敗を経て、主体は{{TRUE_RECOGNITION}}を得る。"
        ),
        "slots": [
            ("PROTAGONIST", "agent", "真価を隠され不当に扱われる主体", ["冒頭で誠実さを行動で示す"], ("雑用を押し付けられる見習い", "実力が評価されない小さな事業者", "埋もれた良書を探す読者"), True),
            ("INVISIBILITY", "lack", "誰にも見られない存在であること", [], ("名前で呼ばれない日常", "市場で認知されない製品", "埋もれた知見"), True),
            ("OPPRESSOR", "agent", "真価を抑圧する身近な権力", [], ("手柄を奪う兄弟子", "既得権を持つ大手", "旧来の権威"), True),
            ("HELPER", "agent", "一夜だけ機会を与える援助者", [], ("気まぐれな魔法使い", "期間限定の展示枠", "偶然の拡散"), True),
            ("TIME_LIMIT", "conflict", "機会を切実にする時間制限", ["制限は事前に明示する"], ("真夜中の鐘", "締切のある募集", "話題の旬が過ぎる期限"), True),
            ("RECOGNIZER", "agent", "真価を見抜く高位の認証者", [], ("名門の当主", "目利きの仕入れ担当", "信頼される評者"), True),
            ("PROOF_TOKEN", "symbol", "真価を証明する唯一の痕跡", ["持ち主以外に合わない性質を持つ"], ("片方だけ残された装身具", "名刺代わりの試作品", "独自の実測データ"), True),
            ("TRUE_RECOGNITION", "goal", "本来の自分として見られること", [], ("名前を呼ばれての招聘", "正当な取引条件", "評価の定着"), True),
        ],
        "invariants": [
            "主体の真価は冒頭から読者にだけ明かし、作中の世界には隠す",
            "真価の発揮は時間制限付きで行い、一度は必ず失わせる",
            "同定は主体固有の痕跡によって行い、偽の持ち主の失敗を挟んで遅延させる",
            "反転後の主体は抑圧者への報復ではなく寛容を選ぶ",
        ],
        "guide": {
            "manga": (
                "全30ページ想定。1〜5pで不当な扱いを小ゴマ多めで重く描き、6pで機会の剥奪。9〜10pで援助者と時間制限、めくり（11→12p）で変身後の主体を断ち切りの大ゴマで見せる。"
                "13〜17pで真価の発揮、18pで制限時刻と逃走、19p左ページ終端に残された痕跡のアップを置いて引く。20〜26pで探索と偽の持ち主の失敗をテンポよく反復し、"
                "めくり（27→28p）の右ページで痕跡の合致を見開き大ゴマにする。29〜30pは寛容と余韻。"
            ),
            "marketing": (
                "実力があるのに認知されない顧客（または製品）を主体として描き、限られた露出機会での真価の発揮と、その痕跡（実績データ・推薦）が"
                "後から正当な評価につながる流れを設計する。時間制限は定員や期間といった誠実な理由で示す。"
            ),
            "article": (
                "過小評価されている概念や手法を主体に据え、通説がそれを見落としてきた理由を抑圧の構造として分析する。"
                "限られた事例で示された真価と、それを裏づける固有の痕跡（データ）を提示し、最後に正当な位置づけへ反転させる。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.45), (0.12, 0.55), (0.3, 0.35), (0.45, 0.5), (0.6, 0.75), (0.72, 0.6), (0.82, 0.85), (0.92, 0.3), (1.0, 0.1)],
            "release": [(0.0, 0.05), (0.3, 0.3), (0.45, 0.55), (0.6, 0.1), (0.75, 0.15), (0.92, 0.75), (1.0, 0.95)],
            "audience_knowledge": [(0.0, 0.2), (0.3, 0.5), (0.45, 0.6), (0.6, 0.75), (0.82, 0.8), (0.92, 1.0), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.3, 0.15), (0.45, 0.3), (0.6, 0.35), (0.82, 0.5), (0.92, 1.0), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.3), (0.2, 0.35), (0.45, 0.5), (0.6, 0.65), (0.82, 0.7), (0.92, 0.4), (1.0, 0.25)],
        },
    }
)

PATTERNS.append(
    {
        "num": 5,
        "slug": "hubris_fall",
        "title": "傲慢による墜落",
        "origin": "myth",
        "summary": (
            "類まれな能力で頂点に立った主体が、警告を退けて限界を越えようとし、自らの成功の論理そのものによって墜落する。"
            "上昇の局面では緊張は低く、警告が無視されるたびに読者の不安だけが蓄積し、終盤に破滅が一挙に訪れる。"
        ),
        "tags": ["傲慢", "墜落", "警告の無視", "限界"],
        "lineage": ["傲慢と報い（ヒュブリスとネメシス）", "翼を得て太陽に近づきすぎる神話", "成功の罠"],
        "nodes": [
            ("ascender", "agent", "protagonist", "成功の頂点に立つ才能ある主体", "能力によって境界を次々と越えてきた経験が、限界の存在を信じさせない。", 0.95, "PROTAGONIST"),
            ("warner", "agent", "warner", "限界を知り警告する者", "主体を想うがゆえに警告するが、成功の只中にいる主体には届かない。", 0.6, "WARNING_VOICE"),
            ("order", "agent", "natural_order", "越えてはならない限界を課す秩序", "人格を持たない法則として、越境した者に均衡回復の報いを与える。", 0.7, "NATURAL_LIMIT"),
            ("summit", "desire", "goal", "さらなる高みへの到達", "既に十分な成功を得ていながら止まれない、際限のない上昇欲求。", 0.85, "ENDLESS_ASCENT"),
            ("humility", "lack", "inner_lack", "自己の有限性への感覚の欠如", "成功体験の積み重ねが削り取った、限界を感じ取る能力。", 0.8, "MISSING_HUMILITY"),
            ("wings", "symbol", "means_symbol", "上昇を可能にした手段の象徴", "主体の能力の結晶であり、限界を越えた瞬間に崩壊する。", 0.75, "ASCENT_MEANS"),
            ("followers", "agent", "dependents", "主体の成功に巻き込まれる追随者", "主体を称賛し追随するが、墜落の際には共に巻き込まれる。", 0.5, None),
        ],
        "edges": [
            ("ascender", "summit", "pursuit", 0.95, 0.0, "際限のない上昇", 0.05),
            ("warner", "ascender", "asymmetry", 0.65, 0.6, "警告者だけが限界の位置を知る", 0.25),
            ("ascender", "warner", "opposition", 0.55, 0.6, "警告の嘲笑と排除", 0.6),
            ("ascender", "followers", "power_gradient", 0.75, 0.8, "成功による支配と称賛", 0.3),
            ("followers", "ascender", "dependency", 0.6, -0.5, "主体の成功への依存", 0.3),
            ("order", "ascender", "power_gradient", 0.9, 0.95, "越境者への均衡回復", 0.84),
            ("order", "ascender", "opposition", 0.85, 0.7, "秩序による報い", 0.9),
            ("wings", "summit", "embodiment", 0.7, 0.0, "手段が上昇欲求を体現する", 0.05),
            ("wings", "humility", "embodiment", 0.5, 0.0, "手段への過信が謙虚さの欠落を映す", 0.72),
        ],
        "beats": [
            ("triumph_hook", "hook", 0.0, 0.05, "頂点の誇示", "主体が卓越した能力で不可能を成し遂げる場面から始まり、称賛と自信が示される。", [], 0.1, 0.0, ["ascender", "wings", "followers"], False),
            ("first_warning", "premise", 0.25, 0.05, "最初の警告", "警告者が限界の存在を告げるが、主体は過去の成功を根拠に一笑に付す。", ["triumph_hook"], 0.05, 0.2, ["warner", "ascender", "humility"], False),
            ("escalating_ambition", "escalation", 0.45, 0.1, "野心の加速", "上昇のたびに主体はより大きな目標を掲げ、追随者を巻き込んで賭け金を拡大する。", ["first_warning"], 0.1, 0.25, ["ascender", "summit", "followers"], False),
            ("silencing", "local_optimization_limit", 0.6, 0.05, "警告者の排除", "主体は警告者を遠ざけ、限界を感じ取る最後の回路を自ら断ち切る。", ["escalating_ambition"], 0.15, 0.1, ["ascender", "warner"], True),
            ("cracks", "escalation", 0.72, 0.06, "綻びの兆候", "手段に小さな破損が生じ始めるが、主体はそれを無視してさらに高みを目指す。", ["silencing"], 0.2, 0.05, ["wings", "ascender"], False),
            ("overreach", "climax", 0.84, 0.04, "限界の越境", "主体がついに越えてはならない一線を越え、手段が崩壊を始める。", ["cracks"], 0.25, 0.1, ["ascender", "wings", "order"], True),
            ("fall", "crisis", 0.9, 0.04, "墜落", "秩序の報いが一挙に訪れ、主体と追随者は築いたものすべてと共に墜落する。", ["overreach"], 0.15, -0.5, ["ascender", "order", "followers"], True),
            ("lesson_afterglow", "afterglow", 1.0, 0.05, "残された教訓", "生き残った者が警告の意味を悟り、限界を知ることの価値が静かに語られる。", ["fall", "first_warning"], -0.3, 0.0, ["warner", "followers"], False),
        ],
        "mechanisms": [
            ("warning_irony", "dramatic_irony", "警告を知る読者の不安", "読者は警告の正しさを確信しているが主体は聞き入れず、上昇の場面ほど不安が増していく。", "最初の警告と警告者の排除", "成功の描写に不吉さを感じ続ける", "劇的アイロニー論", 0.85, ["first_warning", "silencing"], ["warner", "ascender"]),
            ("nemesis_justice", "moral_judgment", "傲慢への報いの納得", "限界を無視した者が報いを受ける構図が公正の感覚を満たし、墜落に必然性を与える。", "限界の越境と墜落", "畏怖を伴う納得", "公正世界信念", 0.75, ["overreach", "fall"], ["order"]),
            ("success_projection", "identity_projection", "成功者の慢心への自己投影", "受け手自身の小さな成功体験と慢心が主体に重なり、教訓が他人事でなくなる。", "頂点の誇示", "自己の慢心への警戒心", "同一化理論", 0.6, ["triumph_hook", "escalating_ambition"], ["ascender"]),
            ("fall_contrast", "contrast_effect", "頂点と墜落の落差", "前半の称賛と終盤の墜落の高低差が大きいほど、衝撃と教訓が深く刻まれる。", "墜落の瞬間", "強い衝撃と長い余韻", "対比効果", 0.7, ["triumph_hook", "fall"], ["ascender", "followers"]),
        ],
        "anchor": "警告を退けるほどに才能を信じてしまう主体の、悪意のない慢心",
        "drivers": ["主体が無視する警告を読者だけが正しいと知る情報差", "上昇のたびに大きくなる賭け金", "手段に生じる小さな綻び"],
        "emotions": ["感嘆", "憧れ", "違和感", "不安", "戦慄", "衝撃", "教訓"],
        "template": (
            "{{ASCENT_MEANS}}によって頂点に立った{{PROTAGONIST}}は、{{MISSING_HUMILITY}}ゆえに{{ENDLESS_ASCENT}}を止められない。"
            "{{WARNING_VOICE}}が{{NATURAL_LIMIT}}の存在を告げるが、主体は成功を根拠に退け、ついには警告者を遠ざける。"
            "手段に綻びが生じても上昇を続けた主体は一線を越え、{{FALL_CONSEQUENCE}}として築いたものすべてを失う。"
        ),
        "slots": [
            ("PROTAGONIST", "agent", "成功の頂点に立つ才能ある主体", ["冒頭で本物の才能を証明する"], ("天才的な若き剣士", "急成長する企業の創業者", "成功法則を信じる読者"), True),
            ("ASCENT_MEANS", "symbol", "上昇を可能にした手段の象徴", [], ("伝説の翼の装具", "急拡大を支えた主力事業", "これまで通用した手法"), True),
            ("MISSING_HUMILITY", "lack", "自己の有限性への感覚の欠如", [], ("負けを知らない慢心", "成功体験への過信", "反証を見ない態度"), True),
            ("ENDLESS_ASCENT", "goal", "際限のない上昇欲求", [], ("最強の称号", "市場の独占", "すべてを説明できる理論"), True),
            ("WARNING_VOICE", "agent", "限界を知り警告する者", ["主体を想う立場から警告させる"], ("育ての親", "古参の参謀", "地道な反証研究"), True),
            ("NATURAL_LIMIT", "conflict", "越えてはならない限界", [], ("身体が耐えられない禁じ手", "規制と市場の飽和", "手法の適用範囲"), True),
            ("FALL_CONSEQUENCE", "cost", "墜落によって失うもの", [], ("剣も仲間も失う", "事業の崩壊と信用失墜", "積み上げた主張の瓦解"), True),
        ],
        "invariants": [
            "主体の能力は本物として描き、墜落の原因を能力不足ではなく限界の無視に置く",
            "警告は少なくとも二度提示し、二度目に警告者を排除させる",
            "上昇局面の緊張は低く保ち、読者の不安は情報非対称性だけで蓄積させる",
            "墜落は越境の直後に一挙に起こし、段階的な衰退にしない",
        ],
        "guide": {
            "manga": (
                "全28ページ想定。1〜4pは主体の偉業を大ゴマで華々しく描き、5〜7pで称賛と追随者。9pで最初の警告を小さなコマで挟む。"
                "12〜17pは上昇の連続でコマを大きく明るく保ち、各上昇の直後に読者だけが気づく綻びの小ゴマを入れる。18〜19pで警告者の排除。"
                "めくり（23→24p）の右ページで限界の越境を断ち切りの大ゴマにし、25〜26pの見開きで墜落を一枚絵で描く。27〜28pは静寂ゴマで教訓の余韻。"
            ),
            "marketing": (
                "急成長の成功体験が次の失敗の原因になる構図を、顧客の現状に重ねて描く。警告者の役割をデータや第三者の視点として提示し、"
                "自社の提案を限界を可視化する仕組みとして位置づける。恐怖訴求に偏らず、謙虚さを競争優位として語る。"
            ),
            "article": (
                "ある手法の成功事例を本物として認めたうえで、適用範囲を越えて使われた結果の失敗を分析する。"
                "見過ごされた警告の兆候を時系列で示し、限界を検知するための指標を結論として提示する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.2), (0.15, 0.15), (0.3, 0.22), (0.45, 0.32), (0.6, 0.45), (0.72, 0.62), (0.84, 0.85), (0.9, 0.97), (0.95, 0.8), (1.0, 0.55)],
            "release": [(0.0, 0.4), (0.15, 0.5), (0.3, 0.45), (0.5, 0.3), (0.72, 0.1), (0.9, 0.05), (0.95, 0.4), (1.0, 0.6)],
            "audience_knowledge": [(0.0, 0.2), (0.25, 0.4), (0.45, 0.6), (0.6, 0.7), (0.84, 0.85), (0.9, 1.0), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.2), (0.25, 0.22), (0.45, 0.25), (0.6, 0.28), (0.84, 0.4), (0.9, 1.0), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.3), (0.3, 0.35), (0.6, 0.5), (0.84, 0.7), (0.9, 0.85), (1.0, 0.4)],
        },
    }
)

PATTERNS.append(
    {
        "num": 6,
        "slug": "stolen_fire",
        "title": "火を盗む者の代償",
        "origin": "myth",
        "summary": (
            "上位の秩序が独占する力を、虐げられた共同体のために盗み出した主体が、恩恵を広める代わりに長く苛烈な罰を受ける。"
            "盗みの瞬間の高揚と罰の持続が二つの山をなし、最後に主体の犠牲が共同体の自立によって報われる。"
        ),
        "tags": ["犠牲", "反逆", "恩恵の分配", "罰と解放"],
        "lineage": ["火の窃盗神話（文化英雄）", "知識の独占と解放", "受難する恩人"],
        "nodes": [
            ("thief", "agent", "protagonist", "秩序に背いて力を盗む主体", "上位の秩序に属しながら、虐げられた者への共感から禁を破る。", 0.95, "PROTAGONIST"),
            ("sovereign", "agent", "monopolist", "力を独占する上位の秩序", "力の独占によって支配を維持し、分配を秩序への反逆とみなす。", 0.85, "MONOPOLIST"),
            ("community", "agent", "beneficiaries", "力を奪われた共同体", "力を持たないことで従属を強いられてきたが、恩恵を得て自立へ向かう。", 0.7, "BENEFICIARIES"),
            ("fire", "symbol", "power_symbol", "独占された力の象徴", "持つ者を照らし、分け与えるほど増える性質を持つ。", 0.85, "MONOPOLIZED_POWER"),
            ("autonomy", "desire", "goal", "共同体の自立", "主体個人の栄誉ではなく、他者が自分の力で生きられる状態。", 0.8, "COMMUNAL_AUTONOMY"),
            ("punishment", "lack", "imposed_lack", "終わりの見えない罰による自由の喪失", "主体が支払い続ける代償であり、共同体の繁栄と対照される。", 0.75, "ENDLESS_PUNISHMENT"),
        ],
        "edges": [
            ("sovereign", "community", "power_gradient", 0.9, 0.9, "力の独占による支配", 0.0),
            ("thief", "sovereign", "dependency", 0.4, -0.5, "秩序の内側に属する立場", 0.0),
            ("thief", "autonomy", "pursuit", 0.85, 0.0, "共同体の自立への献身", 0.15),
            ("thief", "sovereign", "opposition", 0.85, -0.6, "独占への反逆", 0.35),
            ("thief", "sovereign", "betrayal", 0.8, -0.3, "秩序の内側からの離反", 0.35),
            ("fire", "autonomy", "embodiment", 0.75, 0.0, "力が自立の可能性を体現する", 0.35),
            ("sovereign", "thief", "power_gradient", 0.9, 0.9, "罰による見せしめ", 0.62),
            ("community", "thief", "dependency", 0.6, -0.4, "恩恵の源への依存", 0.45),
            ("community", "autonomy", "pursuit", 0.65, 0.0, "力を使いこなす学習", 0.5),
            ("punishment", "thief", "power_gradient", 0.8, 0.8, "終わりなき罰が主体の自由を奪い続ける", 0.65),
        ],
        "beats": [
            ("dark_hook", "hook", 0.0, 0.05, "奪われた共同体", "力を持たない共同体が寒さと暗さの中で従属を強いられる姿が示される。", [], 0.2, 0.0, ["community", "sovereign"], False),
            ("resolve", "premise", 0.15, 0.05, "離反の決意", "秩序の内側にいる主体が共同体の苦しみに共感し、独占を破る決意を固める。", ["dark_hook"], 0.1, 0.15, ["thief", "community", "autonomy"], False),
            ("theft", "structural_twist", 0.35, 0.05, "力の窃盗", "主体が秩序の監視を欺いて独占された力を盗み出し、共同体へ手渡す。", ["resolve"], 0.4, 0.1, ["thief", "fire", "sovereign"], True),
            ("spreading", "payoff", 0.48, 0.06, "恩恵の拡散", "力は分け与えるほど増え、共同体は暖かさと技術を得て初めて自立の兆しを見せる。", ["theft"], -0.35, 0.05, ["community", "fire", "autonomy"], False),
            ("discovery_punishment", "crisis", 0.62, 0.05, "露見と罰", "独占者が窃盗に気づき、主体を見せしめとして終わりの見えない罰に処す。", ["spreading"], 0.4, -0.3, ["sovereign", "thief", "punishment"], True),
            ("endurance", "escalation", 0.75, 0.1, "罰の持続", "主体は罰を受け続けながらも独占者への服従を拒み、共同体は力の使い方を学び続ける。", ["discovery_punishment"], -0.1, 0.05, ["thief", "punishment", "community"], False),
            ("surrender_offer", "climax", 0.86, 0.05, "屈服の誘い", "独占者が罰の解除と引き換えに共同体から力を取り上げる取引を迫り、主体はこれを拒絶する。", ["endurance"], 0.2, 0.05, ["sovereign", "thief", "fire"], True),
            ("liberation", "costly_breakthrough", 0.94, 0.04, "解放", "自立した共同体の存在が独占の意味を失わせ、主体の罰は解かれる。犠牲は共同体の成熟によって報われる。", ["surrender_offer", "spreading"], -0.4, 0.0, ["community", "thief", "autonomy"], True),
            ("legacy", "afterglow", 1.0, 0.04, "灯の継承", "共同体は受け取った力を次の世代へ手渡し、主体の名ではなく灯そのものが残る。", ["liberation"], -0.15, 0.0, ["community", "fire"], False),
        ],
        "mechanisms": [
            ("sacrifice_empathy", "empathy_transfer", "自己犠牲への共感", "他者のために罰を引き受ける主体の姿が、受け手の利他性と義憤を同時に喚起する。", "露見と罰", "主体への敬意と独占者への怒り", "共感-利他性仮説", 0.85, ["discovery_punishment", "endurance"], ["thief", "punishment"]),
            ("reciprocity_debt", "reciprocity", "恩恵への返礼意識", "共同体が受け取った恩恵の大きさが主体への恩返しを期待させ、解放の場面を待望させる。", "恩恵の拡散", "解放への強い期待", "返報性の原理", 0.65, ["spreading", "liberation"], ["community"]),
            ("injustice_judgment", "moral_judgment", "独占の不正への道徳的判断", "力を独占し分配を罰する秩序の不正が明確なため、窃盗という違反が正義として受け止められる。", "奪われた共同体の描写", "規則違反への肯定的評価", "公正世界信念", 0.75, ["dark_hook", "theft"], ["sovereign", "community"]),
            ("double_peak", "peak_end", "二つの山と静かな終幕", "窃盗と罰という二つの緊張の山の後に灯の継承という穏やかな終幕を置き、全体を希望として記憶させる。", "灯の継承", "犠牲の物語を希望として記憶する", "ピーク・エンドの法則", 0.6, ["theft", "surrender_offer", "legacy"], ["fire"]),
        ],
        "anchor": "自分の安全を捨ててでも、力を奪われた者たちの寒さに耐えられなかった主体の共感",
        "drivers": ["監視を欺く窃盗の緊張", "罰を受け続ける主体と繁栄する共同体の対照", "屈服の取引を拒めるかという問い"],
        "emotions": ["憐れみ", "義憤", "緊張", "歓喜", "怒り", "忍耐", "敬意", "希望"],
        "template": (
            "{{MONOPOLIST}}が{{MONOPOLIZED_POWER}}を独占し、{{BENEFICIARIES}}は従属を強いられている。秩序の内側にいる{{PROTAGONIST}}は"
            "共感から離反して力を盗み、共同体へ手渡す。恩恵が広がる一方で窃盗は露見し、主体は{{ENDLESS_PUNISHMENT}}に処される。"
            "{{SURRENDER_OFFER}}を拒み続けた末、{{COMMUNAL_AUTONOMY}}の実現が主体を解放する。"
        ),
        "slots": [
            ("PROTAGONIST", "agent", "秩序に背いて力を盗む主体", ["秩序の内側に属していることを明示する"], ("宮廷に仕える若き技師", "社内の改革者", "専門知を開く研究者"), True),
            ("MONOPOLIST", "agent", "力を独占する上位の秩序", [], ("技術を独占する支配層", "情報を囲い込む大手", "知識を占有する権威"), True),
            ("MONOPOLIZED_POWER", "symbol", "独占された力の象徴", ["分け与えるほど増える性質を持たせる"], ("燃え続ける聖火", "業界の非公開ノウハウ", "閉ざされた専門知"), True),
            ("BENEFICIARIES", "agent", "力を奪われた共同体", [], ("辺境の村人たち", "中小の事業者", "独学の学習者"), True),
            ("ENDLESS_PUNISHMENT", "cost", "終わりの見えない罰", [], ("岩山への幽閉", "業界からの締め出し", "権威からの集中批判"), True),
            ("SURRENDER_OFFER", "reversal", "屈服と引き換えの取引", [], ("力の回収と引き換えの赦免", "公開停止と引き換えの和解", "主張撤回と引き換えの地位"), True),
            ("COMMUNAL_AUTONOMY", "goal", "共同体の自立", [], ("村が自ら火を起こせる日", "顧客が自走できる状態", "読者が自分で検証できる力"), True),
        ],
        "invariants": [
            "主体は独占者の側（秩序の内側）に属しており、窃盗は離反を伴う",
            "窃盗と罰は二つの独立した緊張の山として配置する",
            "主体の解放は独占者の改心ではなく、共同体の自立によってもたらす",
            "主体は屈服の取引を少なくとも一度明確に拒絶する",
        ],
        "guide": {
            "manga": (
                "全36ページ想定。1〜4pで暗く寒い共同体を黒ベタ多めで描き、5〜7pで主体の決意。10〜13pで監視を欺く潜入をコマ数多めのテンポで描き、"
                "めくり（13→14p）の右ページで盗み出した力を断ち切りの大ゴマで輝かせる。15〜19pは恩恵の拡散を白い画面で明るく描き、20pで露見。"
                "めくり（21→22p）で罰を見開きで提示し、23〜29pで罰の持続と共同体の成長を交互にカットバックする。30〜32pで屈服の取引と拒絶、"
                "33〜34pで解放、35〜36pで灯の継承。"
            ),
            "marketing": (
                "業界が独占してきた価値（ノウハウ・道具）を顧客に開放する立場としてブランドを描く。開放に伴う既存勢力からの反発を隠さず語り、"
                "顧客の自立（自走できる状態）をゴールに据えることで、依存型ではない信頼を構築する。"
            ),
            "article": (
                "閉じた専門知が生む不均衡を事例で示し、その知を開くことの価値と代償を論じる。"
                "知を受け取った側が自立していく過程をデータで追い、結論で読者自身が知を次へ渡す行動を提案する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.3), (0.15, 0.4), (0.35, 0.8), (0.48, 0.42), (0.62, 0.85), (0.75, 0.72), (0.86, 0.88), (0.94, 0.35), (1.0, 0.2)],
            "release": [(0.0, 0.05), (0.35, 0.15), (0.48, 0.7), (0.62, 0.05), (0.78, 0.2), (0.94, 0.8), (1.0, 0.9)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.45), (0.35, 0.55), (0.6, 0.7), (0.86, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.2), (0.2, 0.25), (0.35, 0.3), (0.55, 0.35), (0.62, 0.8), (0.86, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.3), (0.2, 0.45), (0.35, 0.6), (0.5, 0.4), (0.65, 0.55), (0.86, 0.6), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 7,
        "slug": "fractured_testimony",
        "title": "多視点による真実の崩壊",
        "origin": "film",
        "summary": (
            "一つの出来事を複数の当事者が語り直すたびに前の証言が覆され、確定していたはずの事実が崩れていく。"
            "最終的に露わになるのは真実そのものではなく、各人が誠実な自己像を守るために語りを歪めていたという人間の構造である。"
        ),
        "tags": ["多視点", "信頼できない語り手", "真実の相対性", "証言"],
        "lineage": ["多視点証言劇（藪の中型構造）", "信頼できない語り手", "羅生門的効果"],
        "nodes": [
            ("arbiter", "agent", "arbiter", "証言を裁く立場の聴き手", "真実を確定させる責務を負うが、語りが重なるほど確信を失っていく。", 0.7, "ARBITER"),
            ("narrator_a", "agent", "first_witness", "自らの名誉を守る第一の語り手", "自分を英雄的に描く証言で出来事を語り始める。", 0.7, "FIRST_TESTIMONY"),
            ("narrator_b", "agent", "second_witness", "自らの潔白を守る第二の語り手", "第一の証言を覆し、自分を被害者として描き直す。", 0.7, "SECOND_TESTIMONY"),
            ("narrator_c", "agent", "outside_witness", "利害がないはずの第三の目撃者", "中立を装うが、自らの小さな罪を隠すために語りを歪めていたことが最後に露見する。", 0.75, "HIDDEN_WITNESS"),
            ("truth", "desire", "goal", "出来事の唯一の真相", "全員が求めるふりをしながら、誰も本当には望んでいない確定。", 0.8, "UNREACHABLE_TRUTH"),
            ("self_image", "lack", "shared_lack", "誠実な自己像を失う恐怖", "全ての語り手に共通する欠落であり、語りが歪む真因。", 0.85, "SELF_IMAGE_FEAR"),
            ("relic", "symbol", "contested_evidence", "解釈が割れる物的証拠", "証言ごとに異なる意味を与えられ、真実の不確定性を象徴する。", 0.6, "CONTESTED_EVIDENCE"),
        ],
        "edges": [
            ("arbiter", "truth", "pursuit", 0.85, 0.0, "真相の確定", 0.05),
            ("arbiter", "narrator_a", "power_gradient", 0.55, 0.5, "裁く者と裁かれる者", 0.15),
            ("narrator_a", "narrator_b", "opposition", 0.8, 0.2, "証言の相互否定", 0.28),
            ("narrator_b", "narrator_a", "opposition", 0.8, 0.2, "名誉と潔白の衝突", 0.45),
            ("narrator_c", "arbiter", "asymmetry", 0.75, 0.7, "第三者だけが全体を見ていた", 0.55),
            ("arbiter", "narrator_c", "dependency", 0.65, -0.5, "中立の証言への依存", 0.55),
            ("narrator_c", "arbiter", "betrayal", 0.7, 0.5, "中立の仮面の崩壊", 0.78),
            ("relic", "truth", "embodiment", 0.6, 0.0, "証拠が真相の不確定性を体現する", 0.2),
            ("self_image", "truth", "opposition", 0.75, 0.0, "自己像の防衛が真相を歪める", 0.88),
        ],
        "beats": [
            ("incident_hook", "hook", 0.0, 0.05, "出来事の空白", "結果だけが示され、何が起きたのかという空白が聴き手と読者に同時に提示される。", [], 0.15, 0.0, ["arbiter", "relic"], False),
            ("first_account", "setup", 0.18, 0.06, "第一の証言", "第一の語り手が自らを英雄的に描く物語を語り、一見筋の通った真相が成立する。", ["incident_hook"], 0.2, 0.05, ["narrator_a", "arbiter", "relic"], False),
            ("second_account", "revelation", 0.28, 0.05, "第二の証言による反転", "第二の語り手が同じ出来事を別の構図で語り始め、第一の真相が自己美化だったと判明する。", ["first_account"], -0.2, 0.0, ["narrator_b", "narrator_a"], True),
            ("second_climax", "escalation", 0.45, 0.08, "被害者としての語りの頂点", "第二の語りの頂点で、語り手が自らを被害者として描く決定的場面が提示される。", ["second_account"], 0.3, 0.0, ["narrator_b", "relic"], False),
            ("third_account", "transition", 0.55, 0.05, "第三の目撃者の登場", "利害がないはずの目撃者が中立の立場から語り直し、二つの証言の両方が部分的に否定される。", ["second_climax"], -0.2, 0.0, ["narrator_c", "arbiter"], False),
            ("third_climax", "escalation", 0.7, 0.07, "最も醜い真相", "第三の語りが最も醜い真相を描き、聴き手は中立の証言を真実として受け入れかける。", ["third_account"], 0.3, 0.0, ["narrator_c", "narrator_a", "narrator_b"], False),
            ("witness_exposed", "structural_twist", 0.78, 0.04, "中立の仮面の崩壊", "中立と思われた目撃者自身も、小さな罪を隠すために語りを歪めていたと露見する。", ["third_climax"], -0.25, 0.1, ["narrator_c", "arbiter", "relic"], True),
            ("structural_truth", "revelation", 0.88, 0.05, "構造としての真相", "真相の確定ではなく、全員が自己像を守るために語っていたという構造そのものが露わになる。", ["witness_exposed", "second_account"], 0.35, 0.3, ["self_image", "truth", "arbiter"], True),
            ("ambiguous_afterglow", "afterglow", 1.0, 0.05, "判決なき余韻", "聴き手は判決を下せないまま一つの小さな誠実な行為を選び、人間への信頼がかろうじて残る。", ["structural_truth"], -0.4, 0.0, ["arbiter"], False),
        ],
        "mechanisms": [
            ("schema_collapse", "prediction_error", "確定した真相の崩壊", "読者が受け入れた真相が次の証言で覆されるたびに予測誤差が生じ、注意が強制的に再起動される。", "第二・第三の証言による反転", "強い驚きと再解釈の快感", "予測符号化理論", 0.8, ["second_account", "third_account", "witness_exposed"], ["narrator_a", "narrator_b"]),
            ("unreliable_dissonance", "cognitive_dissonance", "矛盾する語りの不協和", "両立しない証言を同時に抱えさせることで、受け手自身に真相を組み立てさせる能動的な関与を生む。", "証言の相互否定", "自分で推理せずにいられない没入", "認知的不協和理論", 0.75, ["second_climax", "third_climax"], ["narrator_a", "narrator_b", "narrator_c"]),
            ("open_loop", "zeigarnik", "確定しない真相の未完了感", "真相を最後まで確定させないことで、物語が受け手の記憶の中で反芻され続ける。", "判決を下せない終幕", "鑑賞後も続く長い反芻", "ツァイガルニク効果", 0.7, ["ambiguous_afterglow"], ["truth"]),
            ("mirror_projection", "identity_projection", "自己正当化への自己投影", "全員が自己像を守るために語るという構造が、受け手自身の記憶の都合のよい書き換えを映し出す。", "構造としての真相の露見", "人間の弱さへの苦い共感", "自己奉仕バイアス", 0.65, ["structural_truth"], ["self_image"]),
        ],
        "anchor": "誰もが嘘をついているのではなく、ただ自分を誠実な人間だと信じていたいだけだという哀しさ",
        "drivers": ["証言のたびに覆る真相の推理", "同じ出来事が別の構図で描き直される再解釈の快感", "中立の証言者すら信用できない不安"],
        "emotions": ["困惑", "納得", "驚愕", "疑念", "嫌悪", "失望", "哀しみ", "かすかな希望"],
        "template": (
            "{{CONTESTED_EVIDENCE}}だけが残された出来事を、{{ARBITER}}は確定させようとする。{{FIRST_TESTIMONY}}は自らを英雄として語り、"
            "{{SECOND_TESTIMONY}}はそれを覆して自らを被害者として語る。中立を装う{{HIDDEN_WITNESS}}の証言も歪みを含んでいたと露見したとき、"
            "浮かび上がるのは{{UNREACHABLE_TRUTH}}ではなく、全員を突き動かす{{SELF_IMAGE_FEAR}}である。それでも聴き手は{{SMALL_SINCERE_ACT}}を選ぶ。"
        ),
        "slots": [
            ("ARBITER", "agent", "証言を裁く立場の聴き手", [], ("出来事を調べる若い記録係", "原因分析を担う責任者", "情報を検証する読者"), True),
            ("CONTESTED_EVIDENCE", "symbol", "解釈が割れる物的証拠", ["各証言で異なる意味を与える"], ("持ち主の分からない短刀", "解釈が割れる売上データ", "引用され続ける統計"), True),
            ("FIRST_TESTIMONY", "agent", "名誉を守る第一の語り手", [], ("勝者を名乗る剣士", "成功を誇る営業部門", "定説を唱える権威"), True),
            ("SECOND_TESTIMONY", "agent", "潔白を守る第二の語り手", [], ("被害を訴える旅人", "責任を否定する開発部門", "定説に反論する批判者"), True),
            ("HIDDEN_WITNESS", "agent", "中立を装う第三の目撃者", ["最後に小さな罪を露見させる"], ("物陰で見ていた木こり", "外部の調査機関", "中立を装う解説者"), True),
            ("UNREACHABLE_TRUTH", "goal", "出来事の唯一の真相", [], ("出来事の真犯人", "失敗の唯一の原因", "唯一正しい解釈"), True),
            ("SELF_IMAGE_FEAR", "lack", "誠実な自己像を失う恐怖", [], ("臆病者と呼ばれる恐れ", "評価を失う恐れ", "誤りを認める恐れ"), True),
            ("SMALL_SINCERE_ACT", "emotion", "それでも選ばれる小さな誠実さ", [], ("捨てられた子を引き取る決意", "責任追及より再発防止を選ぶ", "結論を保留する誠実さ"), True),
        ],
        "invariants": [
            "各証言はそれ単体で完結した筋の通る物語として描き、次の証言で覆す",
            "後の証言ほど真相らしく見せ、最後に中立の証言者の歪みを露見させる",
            "真相の確定ではなく、語りを歪める動機の共通性を最終的な発見とする",
            "終幕に小さな誠実な行為を置き、相対主義の虚無で終わらせない",
        ],
        "guide": {
            "manga": (
                "全40ページ想定。1〜2pで結果だけを見せる。各証言を約9ページの独立したパートとし、パートごとにトーンと線の質感を変えて語り手を区別する。"
                "各パートの頂点は左ページ終端に置き、めくりで次の語り手の顔の大ゴマから始めて反転を見せる。同じ構図の場面を各パートで描き直し、差分で矛盾を示す。"
                "33〜36pで第三者の歪みを露見させ、37〜38pの見開きで全員の顔を並べる構図にする。39〜40pは小さな誠実な行為で締める。"
            ),
            "marketing": (
                "顧客が聞かされてきた複数の説明（各社の主張）を順に検証し、それぞれが売り手の都合で歪められていたことを示す。"
                "自社の主張も一つの視点にすぎないと認めたうえで、顧客が自分で検証できる基準を誠実さの証として提供する。"
            ),
            "article": (
                "一つの現象に対する複数の解釈を順に紹介し、それぞれの証拠と動機を検証する。真相の断定ではなく解釈を歪める共通の動機を構造として抽出し、"
                "読者に判断の保留と検証の手順を提案する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.4), (0.18, 0.62), (0.28, 0.42), (0.45, 0.72), (0.55, 0.5), (0.7, 0.82), (0.78, 0.55), (0.88, 0.92), (1.0, 0.4)],
            "release": [(0.0, 0.05), (0.2, 0.35), (0.3, 0.15), (0.47, 0.4), (0.57, 0.15), (0.72, 0.35), (0.8, 0.1), (1.0, 0.5)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.45), (0.3, 0.35), (0.47, 0.55), (0.57, 0.4), (0.72, 0.6), (0.8, 0.45), (0.9, 0.85), (1.0, 0.8)],
            "character_knowledge": [(0.0, 0.2), (0.2, 0.45), (0.3, 0.3), (0.47, 0.5), (0.57, 0.35), (0.72, 0.55), (0.8, 0.4), (0.9, 0.6), (1.0, 0.65)],
            "cognitive_load": [(0.0, 0.4), (0.2, 0.55), (0.45, 0.7), (0.7, 0.85), (0.88, 0.95), (1.0, 0.6)],
        },
    }
)

PATTERNS.append(
    {
        "num": 8,
        "slug": "inverted_pursuit",
        "title": "倒叙型追跡劇",
        "origin": "drama",
        "summary": (
            "冒頭で犯行のすべてが読者に示され、物語の関心は誰がではなく、いかにして暴かれるかに移る。"
            "完全犯罪を確信する加害者と、些細な違和感から包囲を狭める追跡者の心理戦が、読者だけが全貌を知るサスペンスとして持続する。"
        ),
        "tags": ["倒叙", "追跡", "心理戦", "サスペンス"],
        "lineage": ["倒叙ミステリ", "冴えない刑事が名士を追い詰める刑事劇の構造", "完全犯罪の綻び"],
        "nodes": [
            ("perpetrator", "agent", "antagonist_viewpoint", "完全犯罪を確信する知的な加害者", "地位と知性を持ち、計画の完璧さに誇りを抱く。物語の視点人物でもある。", 0.9, "CONFIDENT_PERPETRATOR"),
            ("pursuer", "agent", "detective", "凡庸を装う執拗な追跡者", "冴えない外見と低い地位で相手を油断させ、些細な違和感を決して手放さない。", 0.9, "UNASSUMING_PURSUER"),
            ("victim", "agent", "victim", "計画の犠牲となった者", "加害者の秘密を知っていたために排除された存在。", 0.5, None),
            ("concealment", "desire", "goal", "犯行の完全な隠蔽", "加害者が守ろうとする地位と体面。", 0.8, "PERFECT_COVER"),
            ("flaw", "lack", "fatal_oversight", "計画の中の見落とし", "加害者の知性ゆえの思い込みが生んだ、たった一つの綻び。", 0.85, "FATAL_OVERSIGHT"),
            ("clue", "symbol", "telltale_detail", "違和感を生む些細な痕跡", "誰もが見過ごす細部が、計画の前提を崩す鍵となる。", 0.75, "TELLTALE_DETAIL"),
            ("proof", "desire", "pursuer_goal", "犯行の立証", "追跡者が求める、言い逃れのできない一点。", 0.75, "PROOF"),
        ],
        "edges": [
            ("victim", "perpetrator", "dependency", 0.6, -0.4, "秘密を共有していた関係", 0.0),
            ("perpetrator", "victim", "betrayal", 0.85, 0.6, "信頼を利用した排除", 0.02),
            ("perpetrator", "concealment", "pursuit", 0.9, 0.0, "隠蔽工作の維持", 0.05),
            ("perpetrator", "pursuer", "power_gradient", 0.7, 0.7, "地位による見下し", 0.2),
            ("pursuer", "perpetrator", "opposition", 0.85, -0.3, "執拗な追及", 0.25),
            ("pursuer", "proof", "pursuit", 0.85, 0.0, "立証への執念", 0.25),
            ("clue", "flaw", "embodiment", 0.8, 0.0, "痕跡が見落としを体現する", 0.35),
            ("pursuer", "perpetrator", "asymmetry", 0.6, 0.5, "追跡者だけが違和感の意味に気づく", 0.75),
            ("flaw", "perpetrator", "power_gradient", 0.7, 0.5, "見落としが加害者の完全性を内側から崩す", 0.78),
        ],
        "beats": [
            ("crime_hook", "hook", 0.0, 0.08, "犯行の全開示", "加害者が周到な計画で犯行を遂げる一部始終が、読者にだけ示される。", [], 0.2, 0.3, ["perpetrator", "victim", "concealment"], False),
            ("cover_complete", "premise", 0.08, 0.05, "完全犯罪の完成", "偽装工作が完了し、加害者は誰にも疑われない立場を確立したと確信する。", ["crime_hook"], 0.1, 0.05, ["perpetrator", "concealment", "flaw"], False),
            ("pursuer_arrives", "setup", 0.2, 0.05, "冴えない追跡者の登場", "凡庸に見える追跡者が現れ、加害者は内心で相手を侮る。緊張は一度緩む。", ["cover_complete"], -0.3, 0.05, ["pursuer", "perpetrator"], False),
            ("first_unease", "escalation", 0.35, 0.06, "最初の違和感", "追跡者が誰も気に留めない細部を口にし、加害者の余裕に小さな亀裂が入る。", ["pursuer_arrives"], 0.1, -0.1, ["pursuer", "clue", "perpetrator"], False),
            ("counter_move", "escalation", 0.5, 0.06, "加害者の反撃", "加害者が地位と知性を駆使して追跡者を誤った方向へ誘導し、包囲は一度後退する。", ["first_unease"], 0.15, 0.0, ["perpetrator", "pursuer"], False),
            ("false_safety", "false_victory", 0.62, 0.05, "偽りの安全", "追跡者が引き下がったかに見え、加害者は勝利を確信して警戒を解く。", ["counter_move"], -0.1, 0.0, ["perpetrator", "concealment"], False),
            ("closing_net", "escalation", 0.75, 0.06, "包囲の収束", "追跡者が違和感を一つの仮説に束ね、加害者の行動を先回りして罠を仕掛ける。", ["false_safety", "first_unease"], 0.25, -0.1, ["pursuer", "clue", "flaw"], True),
            ("trap_sprung", "revelation", 0.9, 0.04, "罠の発動", "加害者自身の行動が見落としを証明してしまい、言い逃れのできない一点が突きつけられる。", ["closing_net"], 0.2, -0.3, ["perpetrator", "pursuer", "clue", "proof"], True),
            ("dignified_collapse", "afterglow", 1.0, 0.05, "敗北の受容", "加害者は追跡者の執念に敬意を示して敗北を受け入れ、知性同士の対決が静かに閉じる。", ["trap_sprung"], -0.3, 0.0, ["perpetrator", "pursuer"], False),
        ],
        "mechanisms": [
            ("inverted_suspense", "dramatic_irony", "全貌を知る読者のサスペンス", "読者は犯行の全貌を知っているため、追跡者の一言ごとに加害者の動揺を読み取り、二重の意味を味わう。", "冒頭の犯行の全開示", "知っているからこそ高まる緊張", "サスペンス論（机の下の時限装置の喩え）", 0.9, ["crime_hook", "first_unease"], ["perpetrator", "pursuer"]),
            ("underestimation_contrast", "contrast_effect", "侮られた者の逆転", "凡庸に見える追跡者が知的な加害者を追い詰める落差が、逆転の快感を増幅する。", "冴えない追跡者の登場と罠の発動", "痛快さと敬意", "対比効果", 0.75, ["pursuer_arrives", "trap_sprung"], ["pursuer"]),
            ("detail_curiosity", "curiosity_gap", "痕跡の意味への好奇心", "読者は犯行を知っていても見落としが何かは知らず、どの細部が決め手になるかという問いが持続する。", "最初の違和感", "細部への注意の集中", "情報ギャップ理論", 0.7, ["first_unease", "closing_net"], ["clue", "flaw"]),
            ("anticipated_fall", "anticipation_reward", "破綻の予期と充足", "完全犯罪がどう崩れるかを予期させ続け、罠の発動でその予期を最も鮮やかな形で満たす。", "罠の発動", "予期の充足による快感", "予測報酬誤差理論", 0.65, ["false_safety", "trap_sprung"], ["perpetrator"]),
        ],
        "anchor": "地位も知性もない者だと侮られながら、たった一つの違和感を決して手放さない追跡者の誠実さ",
        "drivers": ["読者だけが犯行の全貌を知る情報差", "どの細部が決め手になるかという問い", "知性と執念の一騎打ち"],
        "emotions": ["衝撃", "緊張", "侮り", "違和感", "焦燥", "期待", "痛快", "敬意"],
        "template": (
            "{{CONFIDENT_PERPETRATOR}}は{{PERFECT_COVER}}を確信して犯行を遂げ、その一部始終は読者にだけ示される。"
            "{{UNASSUMING_PURSUER}}は侮られながらも{{TELLTALE_DETAIL}}から生じた違和感を手放さず、加害者の反撃と偽りの安全を経て包囲を狭める。"
            "最後に加害者自身の行動が{{FATAL_OVERSIGHT}}を証明し、{{PROOF}}が突きつけられる。"
        ),
        "slots": [
            ("CONFIDENT_PERPETRATOR", "agent", "完全犯罪を確信する知的な加害者", ["地位と知性を冒頭で示す"], ("名声ある外科医", "業界を支配する老舗", "誤った定説を守る権威"), True),
            ("PERFECT_COVER", "goal", "完全な隠蔽", [], ("鉄壁のアリバイ", "表向きの好業績", "整合して見える説明"), True),
            ("UNASSUMING_PURSUER", "agent", "凡庸を装う執拗な追跡者", ["外見や地位で侮られる設定にする"], ("よれた外套の老刑事", "小さな挑戦者企業", "素朴な疑問を持つ読者"), True),
            ("TELLTALE_DETAIL", "symbol", "違和感を生む些細な痕跡", ["冒頭の犯行場面に映り込ませておく"], ("止まった時計", "顧客の小さな不満", "説明できない例外データ"), True),
            ("FATAL_OVERSIGHT", "lack", "計画の中の見落とし", [], ("知らなかった生活習慣", "現場を知らない前提", "検証されていない仮定"), True),
            ("PROOF", "goal", "言い逃れのできない一点", [], ("本人しか知り得ない発言", "決定的な比較実験", "再現可能な反証"), True),
        ],
        "invariants": [
            "犯行の全貌は冒頭で読者に開示し、謎は誰がではなくいかにして暴かれるかに置く",
            "決め手となる痕跡は冒頭の犯行場面に映り込ませておく（公正な手がかり）",
            "追跡者は侮られる外見・地位を持ち、知性の対決を落差として演出する",
            "決着は加害者自身の行動が見落としを証明する形で行う",
        ],
        "guide": {
            "manga": (
                "全30ページ想定。1〜6pで犯行の一部始終を描き、決め手となる痕跡を背景の小物として一コマだけ描き込む。7pで完全犯罪の完成を加害者の余裕の表情の大ゴマで締める。"
                "めくり（7→8p）で冴えない追跡者を全身で登場させ緊張を緩める。9〜20pは会話劇中心で、追跡者の一言に加害者の目元のアップを挟むカットバックを反復する。"
                "21〜22pで偽りの安全、23〜26pで包囲。めくり（27→28p）の右ページで罠の発動と痕跡のアップを大ゴマで置き、29〜30pで敗北の受容。"
            ),
            "marketing": (
                "顧客が既に知っている問題（全貌）を冒頭で開示し、なぜ従来の対策ではそれが見逃され続けるのかを追跡劇として描く。"
                "自社は些細な兆候を手放さない追跡者として位置づけ、決め手のデータで問題を立証する体験を提供する。"
            ),
            "article": (
                "結論（ある通説の誤り）を冒頭で宣言し、その誤りがなぜ長く見過ごされてきたかを、一つの例外データを軸に段階的に立証していく構成。"
                "反論は加害者の反撃として誠実に扱い、偽りの安全の局面を設けてから決定的な反証を示す。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.62), (0.08, 0.75), (0.2, 0.35), (0.35, 0.45), (0.5, 0.58), (0.62, 0.52), (0.75, 0.75), (0.9, 0.96), (0.95, 0.5), (1.0, 0.3)],
            "release": [(0.0, 0.05), (0.2, 0.4), (0.35, 0.2), (0.62, 0.3), (0.8, 0.1), (0.95, 0.8), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.7), (0.08, 0.85), (0.3, 0.85), (0.6, 0.88), (0.9, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.0), (0.2, 0.1), (0.35, 0.25), (0.5, 0.4), (0.75, 0.6), (0.9, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.5), (0.1, 0.6), (0.25, 0.35), (0.5, 0.5), (0.75, 0.65), (0.9, 0.8), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 9,
        "slug": "dual_genius_duel",
        "title": "天才同士の心理戦",
        "origin": "manga",
        "summary": (
            "互いの正体と意図を探り合う二人の天才が、読み合いと裏の裏を重ねる心理戦。読者は両者の内心を交互に知らされるため、"
            "どちらの策が上回るかという劇的アイロニーと、三度の反転による予測誤差が連続する。"
        ),
        "tags": ["心理戦", "読み合い", "頭脳戦", "反転"],
        "lineage": ["頭脳戦の対決構造", "鏡像的ライバル関係", "劇的アイロニーの交互配分"],
        "nodes": [
            ("strategist_a", "agent", "protagonist", "秘密の力を持つ孤高の策略家", "独自の正義のために隠れて行動し、正体の露見を最大の脅威とする。", 0.95, "HIDDEN_STRATEGIST"),
            ("strategist_b", "agent", "rival", "正体を暴こうとする対の天才", "主体と同等の知性を持ち、論理だけで影の正体に迫る。", 0.9, "MIRROR_RIVAL"),
            ("pawn", "agent", "pawn", "両者に利用される駒", "どちらの陣営にも情報を運び、読み合いの媒介となる。", 0.5, "PAWN"),
            ("victory", "desire", "goal", "相手の完全な無力化", "勝利そのものより、知性で相手を上回ったという証明を求める。", 0.85, "INTELLECTUAL_VICTORY"),
            ("isolation", "lack", "shared_lack", "対等な理解者の不在", "両者に共通する孤独であり、互いが唯一の理解者であるという皮肉を生む。", 0.75, "SHARED_ISOLATION"),
            ("rulebook", "symbol", "rule_symbol", "力を縛る規則の象徴", "力の行使条件を定めた規則であり、その抜け穴が最終的な決め手となる。", 0.7, "HIDDEN_RULE"),
        ],
        "edges": [
            ("strategist_a", "strategist_b", "opposition", 0.95, 0.1, "正体を巡る読み合い", 0.15),
            ("strategist_b", "strategist_a", "opposition", 0.9, 0.1, "論理による包囲", 0.2),
            ("strategist_a", "strategist_b", "asymmetry", 0.7, 0.6, "主体だけが力の規則を知る", 0.1),
            ("strategist_b", "strategist_a", "asymmetry", 0.6, 0.5, "対の天才だけが外部の捜査網を握る", 0.35),
            ("strategist_a", "pawn", "power_gradient", 0.7, 0.7, "駒の操作", 0.4),
            ("pawn", "strategist_a", "dependency", 0.6, -0.5, "主体への心酔", 0.4),
            ("pawn", "strategist_a", "betrayal", 0.65, 0.3, "駒の離反", 0.8),
            ("strategist_a", "victory", "pursuit", 0.9, 0.0, "知性の優位の証明", 0.1),
            ("rulebook", "victory", "embodiment", 0.6, 0.0, "規則の抜け穴が勝敗を体現する", 0.82),
            ("isolation", "strategist_a", "power_gradient", 0.6, 0.5, "対等な理解者の不在が主体を孤高の策略へ駆り立てる", 0.1),
        ],
        "beats": [
            ("power_hook", "hook", 0.0, 0.05, "秘密の力の提示", "主体が誰にも知られずに力を行使できることが示され、読者だけが主体の正体を知る。", [], 0.15, 0.25, ["strategist_a", "rulebook"], False),
            ("rival_enters", "premise", 0.15, 0.05, "対の天才の宣戦", "対の天才が影の存在を論理で推定し、公に挑戦状を叩きつける。", ["power_hook"], 0.2, 0.05, ["strategist_b", "strategist_a"], False),
            ("first_probe", "escalation", 0.25, 0.05, "最初の探り合い", "両者が正体を隠したまま接近し、互いを疑いながら協力者を装う。", ["rival_enters"], -0.15, 0.0, ["strategist_a", "strategist_b", "pawn"], False),
            ("first_reversal", "structural_twist", 0.35, 0.04, "第一の反転", "対の天才が主体の盲点を突いて包囲を一挙に狭め、主体は初めて追い詰められる。", ["first_probe"], 0.3, -0.2, ["strategist_b", "strategist_a"], True),
            ("counter_gambit", "escalation", 0.45, 0.06, "規則を使った反撃", "主体は力の規則を逆手に取った策で疑いを晴らし、形勢を押し戻す。", ["first_reversal"], -0.2, 0.15, ["strategist_a", "rulebook"], False),
            ("second_reversal", "structural_twist", 0.58, 0.04, "第二の反転", "駒を介した罠が噛み合い、対の天才が主体の正体へ確信に近づく決定的な一手を打つ。", ["counter_gambit"], 0.35, -0.2, ["strategist_b", "pawn", "strategist_a"], True),
            ("sacrifice_play", "costly_breakthrough", 0.68, 0.05, "捨て身の一手", "主体は自らの記憶や立場を賭ける捨て身の策で、対の天才の包囲を無効化する。", ["second_reversal"], -0.25, 0.1, ["strategist_a", "rulebook", "isolation"], False),
            ("final_reversal", "climax", 0.82, 0.04, "第三の反転", "最後の対決で駒の離反が起こり、規則の抜け穴を見抜いていた側が勝利を確定させる。", ["sacrifice_play", "counter_gambit"], 0.35, -0.1, ["strategist_a", "strategist_b", "pawn", "rulebook"], True),
            ("hollow_afterglow", "afterglow", 1.0, 0.05, "勝者の孤独", "勝敗が決した後、勝者は唯一の理解者を失ったことに気づき、知性の勝利の空虚さが残る。", ["final_reversal"], -0.25, 0.0, ["isolation", "strategist_a", "strategist_b"], False),
        ],
        "mechanisms": [
            ("alternating_irony", "dramatic_irony", "交互に配分される劇的アイロニー", "読者は両者の内心を交互に知らされるため、どちらの策も相手に見えていない部分を把握し、二重の緊張を味わう。", "両者の内心独白の交互配置", "どちらにも肩入れできない緊張", "劇的アイロニー論", 0.9, ["first_probe", "counter_gambit"], ["strategist_a", "strategist_b"]),
            ("triple_reversal", "prediction_error", "三度の反転による予測誤差", "形勢が三度覆ることで読者の予測が連続して裏切られ、次の一手への注意が極限まで高まる。", "第一・第二・第三の反転", "驚愕と再予測の連鎖", "予測符号化理論", 0.85, ["first_reversal", "second_reversal", "final_reversal"], ["strategist_b"]),
            ("rule_curiosity", "curiosity_gap", "規則の抜け穴への好奇心", "力の規則が序盤に提示されるため、読者はどの条項が決め手になるかを推理し続ける。", "規則を使った反撃", "推理への能動的参加", "情報ギャップ理論", 0.7, ["counter_gambit", "final_reversal"], ["rulebook"]),
            ("lonely_empathy", "empathy_transfer", "孤独な天才への共感", "互いだけが理解者であるという皮肉が、敵対する二人の双方に感情移入させる。", "勝者の孤独", "勝敗を超えた哀感", "共感移入", 0.6, ["hollow_afterglow"], ["isolation"]),
        ],
        "anchor": "誰にも理解されない知性を持つ二人が、互いだけを理解者として認めてしまう孤独",
        "drivers": ["両者の内心を交互に知る情報配分", "形勢が三度覆る読み合い", "序盤に提示された規則の抜け穴"],
        "emotions": ["好奇", "緊張", "戦慄", "安堵", "焦燥", "驚愕", "興奮", "空虚"],
        "template": (
            "{{HIDDEN_STRATEGIST}}は{{HIDDEN_RULE}}に縛られた秘密の力を持ち、その正体を{{MIRROR_RIVAL}}が論理だけで暴こうとする。"
            "両者は協力者を装って探り合い、{{PAWN}}を介した罠と反撃によって形勢は三度覆る。最後に規則の抜け穴を見抜いた側が"
            "{{INTELLECTUAL_VICTORY}}を得るが、勝者に残るのは{{SHARED_ISOLATION}}である。"
        ),
        "slots": [
            ("HIDDEN_STRATEGIST", "agent", "秘密の力を持つ孤高の策略家", ["正体の露見を最大の脅威とする"], ("裏で裁きを下す優等生", "非公開戦略を持つ新興企業", "通説に挑む独立研究者"), True),
            ("MIRROR_RIVAL", "agent", "正体を暴こうとする対の天才", ["主体と同等の知性を持たせる"], ("天才的な若き捜査官", "業界最大手の分析部門", "定説を守る論客"), True),
            ("HIDDEN_RULE", "symbol", "力を縛る規則の象徴", ["序盤で全条項を読者に提示する"], ("力の使用条件を記した手帳", "業界の規制と契約条項", "理論の前提条件"), True),
            ("PAWN", "agent", "両者に利用される駒", [], ("主人公に心酔する協力者", "両社と取引する仲介業者", "両陣営に引用されるデータ"), True),
            ("INTELLECTUAL_VICTORY", "goal", "知性の優位の証明", [], ("相手に正体を認めさせる", "市場での決定的優位", "論争の決着"), True),
            ("SHARED_ISOLATION", "lack", "対等な理解者の不在", [], ("勝利を語る相手のいない孤独", "競合を失った市場の停滞", "批判者を失った理論の脆さ"), True),
        ],
        "invariants": [
            "両者の知性は対等に保ち、勝敗を知性差ではなく情報と規則の解釈で決める",
            "形勢の反転は少なくとも三度、間隔を詰めながら配置する",
            "決め手となる規則は序盤で全て読者に提示しておく",
            "読者には両者の内心を交互に開示し、一方的な肩入れを防ぐ",
        ],
        "guide": {
            "manga": (
                "全45ページ想定（連載3話分の配分）。1〜3pで力の提示、4〜6pで宣戦。各話の終端（15p・30p）を反転の左ページに置き、次話冒頭のめくりで相手側の内心独白から始める。"
                "会話劇は両者の顔のアップを左右対称の構図で切り返し、内心独白は枠なしの縦長コマで表す。第三の反転は40〜41pの見開きで両者の視線を交差させる一枚絵にし、"
                "44〜45pで勝者の孤独を引きの構図で描く。"
            ),
            "marketing": (
                "競合との比較を単純な優劣ではなく読み合いの構造で描く。顧客が想定する反論を相手の一手として先回りし、三段階の反転（常識→反証→再反証）で"
                "自社の論理を提示する。規則（価格条件・保証条件）は最初にすべて開示して信頼を得る。"
            ),
            "article": (
                "対立する二つの理論を対等な知性として扱い、それぞれの論拠を交互に検証する。三度の反転を経て決め手となる前提条件を特定し、"
                "結論では勝敗よりも両理論が互いを必要としていたという構造を示す。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.35), (0.15, 0.55), (0.25, 0.4), (0.35, 0.7), (0.45, 0.5), (0.58, 0.85), (0.68, 0.6), (0.82, 0.97), (0.9, 0.7), (1.0, 0.45)],
            "release": [(0.0, 0.05), (0.25, 0.3), (0.35, 0.05), (0.45, 0.35), (0.58, 0.05), (0.68, 0.35), (0.82, 0.05), (0.92, 0.6), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.3), (0.15, 0.5), (0.35, 0.55), (0.4, 0.56), (0.45, 0.7), (0.58, 0.65), (0.7, 0.8), (0.82, 0.75), (0.92, 1.0), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.05), (0.15, 0.15), (0.35, 0.45), (0.4, 0.52), (0.45, 0.4), (0.58, 0.7), (0.7, 0.6), (0.82, 0.85), (0.92, 1.0), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.5), (0.2, 0.6), (0.35, 0.75), (0.58, 0.85), (0.82, 0.95), (0.92, 0.7), (1.0, 0.45)],
        },
    }
)

PATTERNS.append(
    {
        "num": 10,
        "slug": "betrayal_and_revenge",
        "title": "忠誠の裏切りと復讐",
        "origin": "literature",
        "summary": (
            "信じた者たちの裏切りによってすべてを奪われた主体が、長い幽閉と変容を経て別人として帰還し、周到な計画で一人ずつ報いを与えていく。"
            "復讐の完遂が近づくほど主体は人間性を失い、最後に復讐と赦しの間で選択を迫られる。"
        ),
        "tags": ["裏切り", "復讐", "変容", "赦し"],
        "lineage": ["復讐譚の構造", "無実の幽閉と帰還", "仮面の帰還者"],
        "nodes": [
            ("avenger", "agent", "protagonist", "裏切りによってすべてを奪われた主体", "誠実さゆえに信じた者たちに陥れられ、幽閉の中で別人へと変容する。", 0.95, "WRONGED_PROTAGONIST"),
            ("betrayer", "agent", "antagonist", "主体の地位と愛する者を奪った裏切り者", "嫉妬と野心から主体を陥れ、その後に成功を手にしている。", 0.85, "BETRAYER"),
            ("mentor_prisoner", "agent", "mentor", "幽閉の中で出会う導き手", "主体に知識と隠された財の在処を授け、復讐の手段を与える。", 0.6, "PRISON_MENTOR"),
            ("beloved", "agent", "lost_love", "奪われた愛する者", "裏切り者の側に取り込まれ、主体の帰還を知らずに生きている。", 0.6, "LOST_BELOVED"),
            ("justice", "desire", "goal", "奪われたものへの報い", "失われた人生の回復ではなく、裏切り者に同じ痛みを与えること。", 0.85, "VENGEANCE_GOAL"),
            ("humanity", "lack", "inner_lack", "変容の過程で失われていく人間性", "復讐の完遂に近づくほど主体から失われていく温かさ。", 0.8, "LOST_HUMANITY"),
            ("mask", "symbol", "new_identity", "別人としての新しい名と仮面", "帰還した主体の正体を隠す装置であり、元の自分との断絶を象徴する。", 0.7, "MASKED_IDENTITY"),
        ],
        "edges": [
            ("avenger", "betrayer", "dependency", 0.7, -0.2, "友情と信頼", 0.0),
            ("betrayer", "avenger", "betrayal", 0.95, 0.8, "偽りの告発による陥れ", 0.15),
            ("betrayer", "beloved", "power_gradient", 0.6, 0.6, "奪った者の支配", 0.2),
            ("mentor_prisoner", "avenger", "asymmetry", 0.7, 0.7, "幽閉の導き手だけが知る秘密", 0.3),
            ("avenger", "betrayer", "asymmetry", 0.85, 0.85, "仮面の帰還者だけが正体を知る", 0.5),
            ("avenger", "justice", "pursuit", 0.95, 0.0, "周到な報いの計画", 0.42),
            ("avenger", "betrayer", "opposition", 0.9, 0.5, "復讐の執行", 0.62),
            ("mask", "humanity", "embodiment", 0.7, 0.0, "仮面が失われた人間性を体現する", 0.5),
            ("beloved", "avenger", "dependency", 0.4, 0.0, "消えない想い", 0.9),
            ("humanity", "avenger", "power_gradient", 0.7, 0.6, "失われゆく人間性が復讐者を内側から侵食する", 0.55),
        ],
        "beats": [
            ("happy_hook", "hook", 0.0, 0.05, "幸福の頂点", "誠実な主体が地位と愛を手にする寸前の幸福が描かれ、失われるものの大きさが示される。", [], 0.0, 0.0, ["avenger", "beloved", "betrayer"], False),
            ("betrayal", "crisis", 0.15, 0.04, "裏切りと転落", "信じた友の偽りの告発で主体はすべてを奪われ、理由も知らされず幽閉される。", ["happy_hook"], 0.7, 0.3, ["betrayer", "avenger"], True),
            ("despair", "escalation", 0.28, 0.08, "幽閉の絶望", "時間の感覚を失うほどの幽閉の中で、主体は生きる意味を失いかける。", ["betrayal"], -0.25, 0.1, ["avenger", "humanity"], False),
            ("mentor_gift", "revelation", 0.4, 0.05, "導き手の秘密", "幽閉の導き手が裏切りの全容と隠された財の在処を明かし、主体は復讐という目的を得る。", ["despair"], -0.2, 0.25, ["mentor_prisoner", "avenger", "justice"], True),
            ("masked_return", "setup", 0.5, 0.05, "仮面の帰還", "脱出した主体は莫大な富と新しい名を得て、別人として裏切り者たちの前に現れる。", ["mentor_gift"], 0.1, 0.25, ["avenger", "mask", "betrayer"], False),
            ("first_retribution", "escalation", 0.62, 0.06, "最初の報い", "主体は裏切り者の一人を自らの欲望によって破滅させ、計画の精密さを示す。", ["masked_return"], 0.15, 0.0, ["avenger", "betrayer", "justice"], False),
            ("collateral_cost", "local_optimization_limit", 0.72, 0.05, "巻き添えの痛み", "報いの連鎖が無関係な者や愛する者の現在を傷つけ始め、主体の中で人間性が軋む。", ["first_retribution"], 0.15, 0.0, ["beloved", "avenger", "humanity"], False),
            ("unmasking", "climax", 0.86, 0.04, "正体の告白", "最後の裏切り者の前で主体は仮面を外し、奪われた名を名乗って報いを突きつける。", ["collateral_cost"], 0.25, -0.5, ["avenger", "betrayer", "mask"], True),
            ("mercy_choice", "costly_breakthrough", 0.93, 0.04, "赦しの選択", "完遂の直前、主体は愛する者の言葉で人間性を取り戻し、最後の一撃を自ら止める。", ["unmasking", "collateral_cost"], -0.4, 0.0, ["avenger", "beloved", "humanity"], True),
            ("hopeful_afterglow", "afterglow", 1.0, 0.04, "待つことと希望", "復讐は終わるが失われた時間は戻らない。主体は過去ではなく未来へ向けて船出する。", ["mercy_choice"], -0.2, 0.0, ["avenger"], False),
        ],
        "mechanisms": [
            ("injustice_rage", "moral_judgment", "不当な裏切りへの義憤", "誠実な主体が理不尽に奪われる序盤の落差が受け手の義憤を最大化し、復讐への強い期待を生む。", "裏切りと転落", "報いを求める強い感情", "公正世界信念", 0.9, ["betrayal", "despair"], ["betrayer", "avenger"]),
            ("masked_irony", "dramatic_irony", "仮面の帰還者の劇的アイロニー", "読者は帰還者の正体を知っているが裏切り者たちは知らず、あらゆる会話が二重の意味を帯びる。", "仮面の帰還", "罠に気づかぬ相手への痛快な緊張", "劇的アイロニー論", 0.85, ["masked_return", "first_retribution"], ["mask", "betrayer"]),
            ("revenge_anticipation", "anticipation_reward", "報いの予期と執行", "周到な計画が一つずつ実を結ぶたびに予期が満たされ、次の報いへの期待が連鎖する。", "最初の報い", "連鎖する報酬感", "予測報酬誤差理論", 0.75, ["first_retribution", "unmasking"], ["justice"]),
            ("humanity_dissonance", "cognitive_dissonance", "復讐と人間性の不協和", "応援してきた復讐が巻き添えを生む段階で、受け手は復讐を望む自分自身と向き合わされる。", "巻き添えの痛み", "復讐への欲求の揺らぎ", "認知的不協和理論", 0.7, ["collateral_cost", "mercy_choice"], ["humanity"]),
        ],
        "anchor": "誠実に生きていただけの主体が、理由も知らされずにすべてを奪われる理不尽",
        "drivers": ["仮面の帰還者の正体を読者だけが知る情報差", "一人ずつ実を結ぶ周到な計画", "復讐の完遂と人間性の喪失の天秤"],
        "emotions": ["幸福", "衝撃", "絶望", "決意", "痛快", "動揺", "慟哭", "希望"],
        "template": (
            "{{WRONGED_PROTAGONIST}}は信じた{{BETRAYER}}の裏切りによって地位と{{LOST_BELOVED}}を奪われ、幽閉される。{{PRISON_MENTOR}}から真相と手段を授かった主体は、"
            "{{MASKED_IDENTITY}}をまとって帰還し、{{VENGEANCE_GOAL}}を一人ずつ執行していく。報いの連鎖が{{LOST_HUMANITY}}を露わにしたとき、"
            "主体は最後の一撃の前で{{MERCY_CHOICE}}を迫られる。"
        ),
        "slots": [
            ("WRONGED_PROTAGONIST", "agent", "裏切りによってすべてを奪われた主体", ["序盤で誠実さと幸福を十分に描く"], ("将来を約束された若き航海士", "共同創業者に追放された経営者", "功績を奪われた研究者"), True),
            ("BETRAYER", "agent", "主体を陥れた裏切り者", [], ("嫉妬に駆られた親友", "手柄を奪った共同創業者", "業績を盗用した権威"), True),
            ("LOST_BELOVED", "agent", "奪われた愛する者", [], ("婚約者", "かつての顧客", "共に研究した仲間"), True),
            ("PRISON_MENTOR", "agent", "幽閉の中で出会う導き手", [], ("隣の独房の老学者", "どん底で出会う出資者", "埋もれた古典文献"), True),
            ("MASKED_IDENTITY", "symbol", "別人としての新しい名と仮面", [], ("謎の資産家の称号", "新ブランドでの再起", "新しい理論名での発表"), True),
            ("VENGEANCE_GOAL", "goal", "奪われたものへの報い", [], ("裏切り者の破滅", "市場での完全な逆転", "盗用の公的な証明"), True),
            ("LOST_HUMANITY", "lack", "変容の過程で失われる人間性", [], ("笑い方を忘れた心", "利益だけを見る経営", "論破だけを求める姿勢"), True),
            ("MERCY_CHOICE", "reversal", "最後の一撃の前の選択", [], ("剣を収めて名を明かすだけに留める", "買収ではなく協業を選ぶ", "糾弾ではなく公正な記録を残す"), True),
        ],
        "invariants": [
            "裏切りは序盤 t≈0.15 に置き、それ以前に失われるものの大きさを十分に描く",
            "帰還後の主体の正体は読者にだけ開示し、裏切り者たちには最後まで隠す",
            "報いは一人ずつ段階的に執行し、各報いを裏切り者自身の欲望や罪に起因させる",
            "完遂の直前に人間性の回復を賭けた選択を置く",
        ],
        "guide": {
            "manga": (
                "全60ページ想定（前後編）。1〜6pで幸福を明るく描き、7〜8pの見開きで裏切りを一枚絵にする。9〜18pの幽閉は黒ベタ中心で、時間経過を同構図の反復で示す。"
                "19〜22pで導き手の秘密、前編の終端（30p左ページ）を仮面の帰還者の全身大ゴマで引く。後編は会話劇の中に読者だけが分かる二重の意味のセリフを仕込み、"
                "52〜53pの見開きで正体の告白、56〜58pで赦しの選択、59〜60pで船出の余韻。"
            ),
            "marketing": (
                "不当な扱いを受けてきた顧客の悔しさを起点に再起の物語を設計する。復讐ではなく正当な逆転として、顧客が自らの力で取り戻す姿を描き、"
                "最後は過去への報いより未来への投資を選ぶ姿勢でブランドの価値観を示す。"
            ),
            "article": (
                "不当に扱われてきた知見や手法が再評価されるまでの経緯を、裏切り・幽閉・帰還の三段で描く。批判への反撃ではなく公正な検証によって"
                "評価を取り戻す過程を示し、結論で対立の先にある協働を提案する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.15), (0.15, 0.9), (0.28, 0.65), (0.4, 0.42), (0.5, 0.5), (0.62, 0.62), (0.72, 0.75), (0.86, 0.95), (0.93, 0.55), (1.0, 0.35)],
            "release": [(0.0, 0.6), (0.12, 0.5), (0.15, 0.0), (0.4, 0.3), (0.62, 0.35), (0.75, 0.45), (0.86, 0.2), (0.95, 0.55), (1.0, 0.6)],
            "audience_knowledge": [(0.0, 0.1), (0.15, 0.4), (0.3, 0.6), (0.5, 0.85), (0.86, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.15, 0.35), (0.3, 0.3), (0.5, 0.2), (0.72, 0.35), (0.86, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.25), (0.15, 0.55), (0.3, 0.45), (0.5, 0.6), (0.72, 0.75), (0.86, 0.85), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 11,
        "slug": "mentor_loss_inheritance",
        "title": "師の喪失と継承",
        "origin": "manga",
        "summary": (
            "未熟な主体が導き手のもとで成長するが、物語の中盤で導き手は主体を守るために失われる。"
            "喪失の悲嘆を経て主体は導き手の教えの真意を理解し、自らが次の導き手として意志を継承する。"
        ),
        "tags": ["師弟", "喪失", "継承", "成長"],
        "lineage": ["師の死と継承の物語構造", "導き手の退場（英雄の旅の変奏）", "意志の継承"],
        "nodes": [
            ("apprentice", "agent", "protagonist", "未熟で反抗的な弟子", "才能はあるが導き手の教えの意味を理解できず、近道を求めて反発する。", 0.95, "APPRENTICE"),
            ("master", "agent", "mentor", "未完の教えを残す導き手", "主体の未熟さを知りながら、言葉ではなく背中で教えを伝えようとする。", 0.85, "MASTER"),
            ("threat", "agent", "antagonist", "導き手の過去から来る脅威", "導き手がかつて取り逃がした因縁であり、弟子の未熟さを突いて襲来する。", 0.75, "PAST_THREAT"),
            ("successor_circle", "agent", "community", "次の世代として主体を見つめる者たち", "かつての主体のように未熟な者たちであり、継承の証人となる。", 0.45, None),
            ("mastery", "desire", "goal", "導き手を超える力", "当初は自己証明のための力として求められ、最後には守るための力へ意味が変わる。", 0.8, "MASTERY"),
            ("guidance_void", "lack", "inner_lack", "導き手を失った空白", "喪失後に主体を襲う、判断の拠り所の欠如。", 0.8, "GUIDANCE_VOID"),
            ("heirloom", "symbol", "legacy_symbol", "受け継がれる遺品", "導き手の教えを凝縮した遺品であり、主体が真意を理解した瞬間に意味が開く。", 0.75, "LEGACY_HEIRLOOM"),
        ],
        "edges": [
            ("apprentice", "master", "dependency", 0.8, -0.6, "教えへの依存と反発", 0.05),
            ("master", "apprentice", "asymmetry", 0.75, 0.7, "導き手だけが教えの真意を知る", 0.1),
            ("apprentice", "master", "opposition", 0.45, -0.3, "近道を求める反発", 0.2),
            ("apprentice", "mastery", "pursuit", 0.85, 0.0, "力の追求", 0.1),
            ("threat", "master", "opposition", 0.9, 0.3, "過去の因縁", 0.35),
            ("threat", "apprentice", "power_gradient", 0.85, 0.85, "未熟さへの圧倒", 0.45),
            ("heirloom", "guidance_void", "embodiment", 0.7, 0.0, "遺品が空白を埋める鍵を体現する", 0.6),
            ("heirloom", "mastery", "embodiment", 0.6, 0.0, "力の意味の変化を体現する", 0.75),
            ("successor_circle", "apprentice", "dependency", 0.5, -0.4, "新たな導き手への信頼", 0.95),
        ],
        "beats": [
            ("rebel_hook", "hook", 0.0, 0.05, "反抗する弟子", "才能を持て余す主体が導き手の地味な教えに反発し、近道の力を求める姿が示される。", [], 0.1, 0.1, ["apprentice", "master", "mastery"], False),
            ("hidden_lesson", "setup", 0.2, 0.06, "伝わらない教え", "導き手は繰り返しの鍛錬に真意を込めるが、主体には退屈な作業にしか見えない。読者だけが意図を察する。", ["rebel_hook"], 0.1, 0.2, ["master", "apprentice", "heirloom"], False),
            ("threat_returns", "escalation", 0.35, 0.06, "因縁の襲来", "導き手の過去の因縁が姿を現し、平穏な修行の日々に影が差す。", ["hidden_lesson"], 0.1, 0.05, ["threat", "master"], False),
            ("rash_act", "local_optimization_limit", 0.45, 0.04, "未熟な暴走", "主体は力を示そうと独断で因縁に挑み、決定的な窮地を招く。", ["threat_returns"], 0.25, 0.0, ["apprentice", "threat"], False),
            ("master_falls", "crisis", 0.5, 0.04, "導き手の喪失", "導き手は主体を庇って因縁を退けるが、その代償として命を落とす。", ["rash_act"], 0.3, 0.0, ["master", "apprentice", "threat"], True),
            ("grief_void", "transition", 0.6, 0.06, "空白の悲嘆", "判断の拠り所を失った主体は自責に沈み、力の追求そのものを手放しかける。", ["master_falls"], -0.45, 0.0, ["apprentice", "guidance_void"], False),
            ("true_meaning", "revelation", 0.72, 0.05, "教えの真意", "遺品を通じて繰り返しの鍛錬の真意が明かされ、主体は導き手が何を守ろうとしていたかを理解する。", ["grief_void", "hidden_lesson"], -0.05, -0.3, ["heirloom", "apprentice", "master"], True),
            ("inherited_battle", "climax", 0.86, 0.05, "継承の決戦", "主体は導き手の技を自分のものとして因縁に再び挑み、守るための力で勝利する。", ["true_meaning"], 0.4, 0.0, ["apprentice", "threat", "mastery", "heirloom"], True),
            ("new_master", "afterglow", 1.0, 0.05, "次の導き手", "主体はかつての自分のような未熟な者たちを前に、同じ地味な教えを今度は自分の言葉で伝え始める。", ["inherited_battle"], -0.3, 0.0, ["apprentice", "successor_circle"], False),
        ],
        "mechanisms": [
            ("teaching_irony", "dramatic_irony", "伝わらない教えの劇的アイロニー", "読者は導き手の真意を察しているが主体は反発し続け、修行の場面に切なさが宿る。", "伝わらない教え", "もどかしさと喪失の予感", "劇的アイロニー論", 0.7, ["hidden_lesson", "rash_act"], ["master", "apprentice"]),
            ("loss_grief", "loss_aversion", "導き手の喪失の衝撃", "物語の支柱である導き手を中盤で失わせることで、受け手に安全基地の喪失を追体験させる。", "導き手の喪失", "強い悲嘆と主体への同情", "愛着理論・損失回避", 0.85, ["master_falls", "grief_void"], ["master", "guidance_void"]),
            ("meaning_catharsis", "catharsis", "真意の理解による浄化", "反発していた教えの真意を理解する瞬間に、悲嘆が感謝へと転化され涙を伴う解放が生じる。", "教えの真意の開示", "悲嘆から感謝への転化", "カタルシス論", 0.8, ["true_meaning"], ["heirloom"]),
            ("growth_efficacy", "self_efficacy", "継承による自己効力感", "未熟だった主体が導き手の技を自分のものとして使いこなす姿が、受け手に成長の実感を与える。", "継承の決戦", "自分も変われるという希望", "自己効力感理論", 0.7, ["inherited_battle", "new_master"], ["apprentice", "mastery"]),
        ],
        "anchor": "導き手の教えに反発しながらも、本当は認められたいと願っている弟子の未熟さ",
        "drivers": ["読者だけが察する導き手の真意", "因縁の襲来による喪失の予感", "遺品に込められた教えの謎"],
        "emotions": ["反発", "退屈", "不安", "後悔", "慟哭", "感謝", "決意", "継承の誇り"],
        "template": (
            "近道を求める{{APPRENTICE}}は、{{MASTER}}の地味な教えに反発し続ける。{{PAST_THREAT}}の襲来に未熟なまま挑んだ主体を庇い、導き手は失われる。"
            "{{GUIDANCE_VOID}}に沈んだ主体は{{LEGACY_HEIRLOOM}}を通じて教えの真意を理解し、{{MASTERY}}の意味を自己証明から守るための力へと変える。"
            "決戦に勝利した主体は、{{INHERITANCE_ACT}}によって次の導き手となる。"
        ),
        "slots": [
            ("APPRENTICE", "agent", "未熟で反抗的な弟子", ["冒頭で才能と未熟さの両方を見せる"], ("型稽古を嫌う天才肌の少年", "成果を急ぐ新任の管理職", "基礎を軽視する学習者"), True),
            ("MASTER", "agent", "未完の教えを残す導き手", ["教えを言葉で説明しすぎない"], ("寡黙な老剣士", "現場叩き上げの上司", "古典的な理論"), True),
            ("PAST_THREAT", "agent", "導き手の過去から来る脅威", [], ("師が取り逃がした宿敵", "過去の判断が招いた市場危機", "かつて退けた反論の再来"), True),
            ("GUIDANCE_VOID", "lack", "導き手を失った空白", [], ("拠り所を失った迷い", "指揮官不在の混乱", "参照すべき基準の喪失"), True),
            ("LEGACY_HEIRLOOM", "symbol", "受け継がれる遺品", ["序盤から何度も画面に映しておく"], ("使い込まれた木刀", "引き継ぎの手書きメモ", "注釈だらけの原典"), True),
            ("MASTERY", "goal", "導き手を超える力", [], ("師の奥義", "チームを率いる力", "応用できる理解"), True),
            ("INHERITANCE_ACT", "reversal", "次の導き手となる継承の行為", [], ("弟子に同じ型を教える", "部下に権限を委ねる", "読者に問いを手渡す"), True),
        ],
        "invariants": [
            "導き手の喪失は物語の中央 t≈0.5 に置き、主体の未熟な行動を直接の原因とする",
            "教えの真意は序盤から読者にだけ察せられるよう仕込み、喪失後に主体へ開示する",
            "決戦は導き手の技を主体自身の形で使うことで勝利させる",
            "結末で主体を次の導き手の立場に置き、継承の円環を閉じる",
        ],
        "guide": {
            "manga": (
                "全40ページ想定。1〜5pで反発する弟子と地味な修行、8〜10pで遺品を何度も画面に映す。15〜18pで因縁の襲来、19pで弟子の暴走。"
                "めくり（19→20p）の右ページで導き手が弟子を庇う瞬間を断ち切りの大ゴマにし、20〜21pの見開きで喪失を描く。22〜27pは無音の静寂ゴマと雨の表現で悲嘆、"
                "28〜29pで遺品から真意が明かされる回想を枠線なしで挿入する。めくり（33→34p）で継承した技を放つ見開き、39〜40pで次の世代に教える主体を冒頭と同じ構図で描く。"
            ),
            "marketing": (
                "顧客を弟子、成功した先人の知恵を導き手として描き、導き手が去った後も顧客が自走できる仕組み（教えの遺品）を提供価値とする。"
                "依存ではなく継承をゴールに設定し、最終的に顧客自身が誰かを導く姿を未来像として示す。"
            ),
            "article": (
                "古典的な理論が軽視されてきた経緯を弟子の反発として描き、その理論が忘れられたことで生じた問題を示す。"
                "理論の真意を現代の文脈で再解釈し、読者自身が次の世代へ伝える方法を結論とする。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.3), (0.15, 0.4), (0.35, 0.5), (0.45, 0.7), (0.5, 0.95), (0.6, 0.45), (0.7, 0.4), (0.86, 0.9), (0.93, 0.5), (1.0, 0.2)],
            "release": [(0.0, 0.3), (0.2, 0.35), (0.45, 0.1), (0.5, 0.0), (0.62, 0.2), (0.72, 0.5), (0.86, 0.1), (0.95, 0.85), (1.0, 0.9)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.4), (0.45, 0.5), (0.6, 0.65), (0.72, 0.85), (0.86, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.2, 0.15), (0.45, 0.25), (0.6, 0.35), (0.72, 0.85), (0.86, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.3), (0.25, 0.4), (0.5, 0.75), (0.62, 0.45), (0.75, 0.55), (0.86, 0.7), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 12,
        "slug": "closed_circle_paranoia",
        "title": "閉鎖空間の疑心暗鬼",
        "origin": "literature",
        "summary": (
            "外部から遮断された空間に集められた者たちが一人ずつ失われていき、互いへの疑心が連帯を崩壊させる。"
            "犠牲のたびに緊張が段階的に上昇し、偽の結論による一時的な安堵の後、隠されていた内部の構造が露わになる。"
        ),
        "tags": ["閉鎖空間", "疑心暗鬼", "連続する犠牲", "推理"],
        "lineage": ["閉鎖空間ミステリ（クローズドサークル）", "童謡の見立てによる連続事件", "孤島・吹雪の山荘型設定"],
        "nodes": [
            ("survivor", "agent", "protagonist", "疑心の中で理性を保とうとする主体", "集団の中で最後まで推論を手放さず、同時に自分も疑われる立場にいる。", 0.9, "REASONING_SURVIVOR"),
            ("group", "agent", "suspects", "閉じ込められた集団", "それぞれが隠し事を抱え、恐怖の中で互いを疑い始める。", 0.7, "TRAPPED_GROUP"),
            ("orchestrator", "agent", "hidden_antagonist", "集団の中に紛れた仕掛け人", "犠牲者を装うことで疑いの外に出て、過去の罪への裁きを演出する。", 0.85, "HIDDEN_ORCHESTRATOR"),
            ("escape", "desire", "goal", "生還と真相の解明", "外部との連絡が絶たれた中での唯一の目的。", 0.8, "SURVIVAL_GOAL"),
            ("trust", "lack", "collective_lack", "集団内の信頼の欠如", "恐怖が増幅させる相互不信であり、仕掛け人の最大の武器となる。", 0.85, "COLLAPSING_TRUST"),
            ("rhyme", "symbol", "countdown_symbol", "犠牲を予告する見立ての象徴", "一つずつ減っていく象徴物が、次の犠牲と残り時間を可視化する。", 0.75, "COUNTDOWN_SYMBOL"),
            ("isolation_field", "symbol", "barrier", "外部との断絶", "荒天や水域のような遮断の象徴であり、救済の不在を保証する。", 0.55, "ISOLATION_BARRIER"),
        ],
        "edges": [
            ("survivor", "escape", "pursuit", 0.85, 0.0, "生還と解明", 0.05),
            ("isolation_field", "escape", "opposition", 0.6, 0.0, "救済の不在", 0.05),
            ("orchestrator", "group", "asymmetry", 0.9, 0.9, "仕掛け人だけが全員の罪を知る", 0.0),
            ("orchestrator", "group", "power_gradient", 0.8, 0.8, "閉鎖空間の支配", 0.15),
            ("rhyme", "trust", "embodiment", 0.7, 0.0, "減っていく象徴が信頼の崩壊を体現する", 0.2),
            ("group", "survivor", "dependency", 0.55, 0.0, "推論役への依存", 0.25),
            ("survivor", "group", "opposition", 0.6, 0.0, "相互の疑心", 0.4),
            ("group", "orchestrator", "dependency", 0.5, -0.3, "犠牲者を装った者への信頼", 0.5),
            ("orchestrator", "group", "betrayal", 0.85, 0.7, "犠牲者の偽装", 0.55),
            ("trust", "group", "power_gradient", 0.75, 0.7, "信頼の欠如が集団を相互不信へ追い込む", 0.3),
        ],
        "beats": [
            ("gathering_hook", "hook", 0.0, 0.05, "招かれた者たち", "互いに面識のない者たちが閉ざされた空間に集められ、過去の罪を告発する声が響く。", [], 0.15, 0.05, ["group", "orchestrator", "isolation_field"], False),
            ("first_loss", "escalation", 0.12, 0.04, "最初の犠牲", "告発の直後に最初の犠牲が出て象徴物が一つ減る。外部との連絡は既に断たれている。", ["gathering_hook"], 0.4, 0.05, ["group", "rhyme", "isolation_field"], True),
            ("suspicion_spreads", "premise", 0.25, 0.06, "疑心の拡大", "犠牲が外部の者の仕業ではありえないと判明し、集団は互いを疑い始める。", ["first_loss"], -0.1, 0.05, ["survivor", "group", "trust"], False),
            ("second_loss", "escalation", 0.38, 0.04, "第二の犠牲", "見立てどおりに次の犠牲が出て、象徴物の減少が次の犠牲を予告していたと確信される。", ["suspicion_spreads"], 0.25, 0.05, ["group", "rhyme"], False),
            ("feigned_victim", "setup", 0.5, 0.04, "偽装された犠牲", "集団の一人が犠牲となったように見え、疑いの対象から外れる。読者も主体もそれを事実と受け取る。", ["second_loss"], -0.1, -0.05, ["orchestrator", "group"], False),
            ("trust_collapse", "crisis", 0.6, 0.05, "連帯の崩壊", "恐怖が限界を超えて集団は互いを拘束し合い、主体自身も疑いの中心に立たされる。", ["feigned_victim"], 0.25, 0.0, ["survivor", "group", "trust"], True),
            ("false_conclusion", "false_victory", 0.7, 0.05, "偽の結論", "最も疑わしい者が仕掛け人と断定され、集団は一時的な安堵に包まれる。", ["trust_collapse"], -0.4, 0.0, ["survivor", "group"], False),
            ("countdown_continues", "escalation", 0.82, 0.04, "止まらない予告", "断定された者が犠牲となり象徴物はなお減り続ける。偽の結論が崩れ、残された者は極限に追い込まれる。", ["false_conclusion"], 0.45, -0.05, ["rhyme", "group", "survivor"], False),
            ("orchestrator_revealed", "revelation", 0.9, 0.04, "仕掛け人の正体", "主体の推論が偽装された犠牲の矛盾を突き、犠牲者を装っていた仕掛け人の正体と動機が明かされる。", ["countdown_continues", "feigned_victim"], 0.1, 0.0, ["survivor", "orchestrator", "trust"], True),
            ("aftermath", "afterglow", 1.0, 0.05, "生還の代償", "生き残った者は救出されるが、疑い合った記憶と暴かれた過去の罪は消えずに残る。", ["orchestrator_revealed"], -0.5, 0.0, ["survivor", "group"], False),
        ],
        "mechanisms": [
            ("countdown_anticipation", "anticipation_reward", "見立てによる予告と予期", "減っていく象徴物が次の犠牲を予告し、受け手は恐怖とともに次を予期せずにいられない。", "象徴物の減少", "恐怖を伴う持続的な注意", "予期理論", 0.8, ["first_loss", "second_loss", "countdown_continues"], ["rhyme"]),
            ("whodunit_gap", "curiosity_gap", "誰がという情報ギャップ", "仕掛け人が集団の中にいると確信させながら特定させないことで、全員の言動が手がかりとして読まれる。", "疑心の拡大", "能動的な推理への没入", "情報ギャップ理論", 0.85, ["suspicion_spreads", "trust_collapse"], ["group", "orchestrator"]),
            ("false_closure", "prediction_error", "偽の結論の崩壊", "断定と安堵の直後に予告が止まらないことで受け手の予測が裏切られ、緊張が最大化する。", "止まらない予告", "強い衝撃と再推理", "予測符号化理論", 0.75, ["false_conclusion", "countdown_continues"], ["survivor"]),
            ("paranoia_projection", "identity_projection", "疑う側にも疑われる側にもなる自己投影", "主体自身が疑いの中心に立たされることで、受け手は極限状況での自分の振る舞いを想像させられる。", "連帯の崩壊", "不信の連鎖への戦慄", "同一化理論", 0.6, ["trust_collapse"], ["trust", "survivor"]),
        ],
        "anchor": "誰も信じられない状況の中で、それでも推論と理性を手放さない主体の孤独な誠実さ",
        "drivers": ["減っていく象徴物による予告", "全員が容疑者である推理", "偽の結論の後も止まらない犠牲"],
        "emotions": ["不穏", "恐怖", "疑心", "戦慄", "孤立", "安堵", "絶望", "驚愕", "虚脱"],
        "template": (
            "{{ISOLATION_BARRIER}}によって外部から断たれた空間に{{TRAPPED_GROUP}}が集められ、過去の罪が告発される。{{COUNTDOWN_SYMBOL}}が一つ減るたびに犠牲が出て、"
            "{{COLLAPSING_TRUST}}が連帯を崩していく。{{HIDDEN_ORCHESTRATOR}}は犠牲者を装って疑いの外に出る。偽の結論の後も予告は止まらず、"
            "{{REASONING_SURVIVOR}}の推論が偽装の矛盾を突いたとき、{{SURVIVAL_GOAL}}と引き換えに{{BURIED_SIN}}が暴かれる。"
        ),
        "slots": [
            ("REASONING_SURVIVOR", "agent", "疑心の中で理性を保つ主体", ["主体自身も疑われる立場に置く"], ("招かれた若い家庭教師", "危機対応を任された担当者", "通説を疑う読者"), True),
            ("TRAPPED_GROUP", "agent", "閉じ込められた集団", ["全員に隠し事を持たせる"], ("孤島に招かれた十人", "撤退できない共同事業者", "同じ前提を共有する専門家たち"), True),
            ("ISOLATION_BARRIER", "symbol", "外部との断絶", [], ("荒天で閉ざされた孤島", "撤退不能な契約", "反証が届かない閉じた業界"), True),
            ("COUNTDOWN_SYMBOL", "symbol", "犠牲を予告する見立ての象徴", ["序盤で総数を明示する"], ("一つずつ消える人形", "減っていく資金残高", "一つずつ崩れる前提"), True),
            ("COLLAPSING_TRUST", "lack", "集団内の信頼の欠如", [], ("互いを縛り合う恐怖", "部門間の責任の押し付け合い", "相互批判の泥仕合"), True),
            ("HIDDEN_ORCHESTRATOR", "agent", "集団に紛れた仕掛け人", ["犠牲者を装わせる"], ("最初に倒れたはずの判事役", "被害者を装う関係者", "中立を装う前提そのもの"), True),
            ("SURVIVAL_GOAL", "goal", "生還と真相の解明", [], ("救助船の到着", "事業の存続", "議論の決着"), True),
            ("BURIED_SIN", "cost", "暴かれる過去の罪", [], ("全員が隠していた過去の事故", "隠蔽されていた不正", "見て見ぬふりをした反例"), True),
        ],
        "invariants": [
            "外部との断絶を冒頭で確定させ、救済の可能性を物語の終盤まで排除する",
            "犠牲は象徴物の減少と同期させ、次の犠牲を予告する",
            "仕掛け人は犠牲者を装って疑いの外に出し、その偽装の矛盾を最終的な手がかりにする",
            "偽の結論による安堵を終盤直前に置き、その直後に予告を継続させる",
        ],
        "guide": {
            "manga": (
                "全48ページ想定。1〜4pで集められた者たちを一人ずつ紹介し、5pの見開きで閉ざされた空間の全景と象徴物の総数を見せる。各犠牲の発見は左ページ終端の引きに置き、"
                "めくった右ページで象徴物が一つ減った大ゴマから始める。会話劇は顔の並列コマで疑いの視線を交差させ、28〜30pで連帯の崩壊を斜め割りの不安定なコマ割りで描く。"
                "34〜35pで偽の結論の安堵を余白多めに、めくり（37→38p）で止まらない予告を断ち切りの大ゴマにする。43〜44pの見開きで仕掛け人の正体、47〜48pで救出と消えない記憶。"
            ),
            "marketing": (
                "撤退できない状況で一つずつ失われていく資源（時間・予算・信頼）を見立てとして可視化し、顧客の危機感を具体化する。"
                "内部の相互不信こそが本当の敵であると示し、自社の提案を信頼を回復させる仕組みとして提示する。"
            ),
            "article": (
                "閉じた前提を共有する分野で、反例が一つずつ見過ごされていく過程を連続する犠牲として描く。偽の結論（安易な原因の断定）の後にも問題が続くことを示し、"
                "最後に前提そのものに潜む構造的な原因を明らかにする。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.2), (0.12, 0.6), (0.25, 0.5), (0.38, 0.72), (0.5, 0.62), (0.6, 0.85), (0.7, 0.45), (0.82, 0.9), (0.9, 0.98), (1.0, 0.3)],
            "release": [(0.0, 0.2), (0.12, 0.0), (0.25, 0.15), (0.38, 0.05), (0.5, 0.1), (0.7, 0.55), (0.82, 0.05), (0.95, 0.7), (1.0, 0.75)],
            "audience_knowledge": [(0.0, 0.1), (0.12, 0.2), (0.38, 0.3), (0.6, 0.4), (0.7, 0.5), (0.82, 0.55), (0.9, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.12, 0.2), (0.38, 0.3), (0.6, 0.35), (0.7, 0.55), (0.82, 0.5), (0.9, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.3), (0.12, 0.45), (0.38, 0.6), (0.6, 0.75), (0.7, 0.6), (0.82, 0.8), (0.9, 0.95), (1.0, 0.4)],
        },
    }
)

PATTERNS.append(
    {
        "num": 13,
        "slug": "double_life",
        "title": "正体秘匿の二重生活",
        "origin": "film",
        "summary": (
            "公には平凡な顔を持ちながら裏で秘密の役割を担う主体が、二つの生活の両立に引き裂かれる。"
            "露見寸前の危機が周期的に訪れ、最後に最も大切な人の前で正体を明かすか否かの選択が訪れる。"
        ),
        "tags": ["二重生活", "正体", "秘密", "選択"],
        "lineage": ["仮面の英雄（二重身元）の構造", "秘密の共有と露見", "変身譚"],
        "nodes": [
            ("dual_self", "agent", "protagonist", "二つの顔を持つ主体", "公の平凡な顔と秘密の役割の顔を使い分け、どちらの自分も本物だと信じている。", 0.95, "DUAL_PROTAGONIST"),
            ("close_one", "agent", "beloved", "公の顔だけを知る大切な人", "主体の不在や嘘に傷つきながらも信じ続ける存在。", 0.75, "CLOSE_ONE"),
            ("hunter", "agent", "antagonist", "秘密の顔を追う敵対者", "秘密の役割を担う主体を倒すために、その正体を暴こうとする。", 0.8, "IDENTITY_HUNTER"),
            ("suspicious_ally", "agent", "suspicious_peer", "違和感に気づき始める身近な第三者", "主体の不自然な行動を観察し、少しずつ正体に近づいていく。", 0.55, None),
            ("normal_life", "desire", "goal", "平凡な幸福の維持", "秘密の役割と引き換えに失われつつある、ありふれた日常。", 0.8, "ORDINARY_HAPPINESS"),
            ("authenticity", "lack", "inner_lack", "本当の自分を誰にも見せられない孤独", "二重生活が生む、誰とも全てを分かち合えない欠落。", 0.85, "UNSHARED_SELF"),
            ("mask", "symbol", "secret_identity", "秘密の顔の象徴", "役割を担う時に身につける徴であり、露見の危機の焦点となる。", 0.75, "SECRET_MARK"),
        ],
        "edges": [
            ("dual_self", "normal_life", "pursuit", 0.8, 0.0, "日常の維持", 0.05),
            ("dual_self", "close_one", "asymmetry", 0.8, 0.8, "主体だけが秘密を知る", 0.05),
            ("close_one", "dual_self", "dependency", 0.65, -0.2, "嘘を承知の信頼", 0.15),
            ("hunter", "dual_self", "opposition", 0.85, 0.3, "秘密の顔への敵対", 0.2),
            ("hunter", "dual_self", "power_gradient", 0.6, 0.4, "正体を盾にした脅迫", 0.62),
            ("suspicious_ally", "dual_self", "asymmetry", 0.45, -0.2, "違和感の蓄積", 0.35),
            ("mask", "authenticity", "embodiment", 0.75, 0.0, "仮面が分かち合えない自己を体現する", 0.1),
            ("mask", "normal_life", "opposition", 0.6, 0.0, "秘密の役割が日常を侵食する", 0.45),
        ],
        "beats": [
            ("double_hook", "hook", 0.0, 0.05, "二つの顔", "平凡な公の顔と秘密の役割の顔が並行して示され、読者だけが両方を知る。", [], 0.1, 0.3, ["dual_self", "mask", "close_one"], False),
            ("near_miss", "escalation", 0.1, 0.04, "最初の露見の危機", "秘密の役割の最中に大切な人と鉢合わせしかけ、主体は咄嗟の嘘で切り抜ける。", ["double_hook"], 0.35, 0.0, ["dual_self", "close_one", "mask"], False),
            ("cost_of_lies", "premise", 0.18, 0.05, "嘘の代償", "繰り返される不在と嘘が大切な人との関係を少しずつ蝕み、主体は二重生活の限界を感じ始める。", ["near_miss"], -0.3, 0.0, ["close_one", "dual_self", "authenticity"], False),
            ("hunter_closes", "escalation", 0.4, 0.05, "追手の接近", "敵対者が秘密の顔の手がかりを掴み、主体の日常の周辺に姿を現す。", ["cost_of_lies"], 0.35, 0.05, ["hunter", "dual_self", "suspicious_ally"], True),
            ("fragile_peace", "transition", 0.48, 0.06, "束の間の平穏", "追手を一度退けて主体は日常を取り戻したかに見えるが、身近な第三者の違和感は膨らんでいる。", ["hunter_closes"], -0.4, 0.05, ["dual_self", "normal_life", "suspicious_ally"], False),
            ("blackmail", "crisis", 0.62, 0.05, "正体を盾にした脅迫", "敵対者が主体の正体を突き止め、大切な人を盾に二つの生活のどちらかを捨てるよう迫る。", ["fragile_peace"], 0.1, 0.1, ["hunter", "close_one", "dual_self"], False),
            ("unmasking", "climax", 0.75, 0.05, "仮面を外す決断", "大切な人を救うため、主体はその目の前で秘密の顔として戦い、自ら正体を晒す。", ["blackmail"], 0.4, -0.3, ["dual_self", "close_one", "mask", "hunter"], True),
            ("acceptance", "revelation", 0.88, 0.05, "受容と分かち合い", "大切な人は嘘の理由を理解し、主体は初めて本当の自分を分かち合える相手を得る。", ["unmasking", "cost_of_lies"], -0.3, -0.2, ["close_one", "dual_self", "authenticity"], True),
            ("integrated_afterglow", "afterglow", 1.0, 0.05, "統合された日常", "二つの顔は消えないが、秘密を共有する者がいる日常が新しい均衡として描かれる。", ["acceptance"], -0.3, 0.0, ["dual_self", "normal_life"], False),
        ],
        "mechanisms": [
            ("secret_irony", "dramatic_irony", "秘密を知る読者の二重視", "読者は主体の両方の顔を知っているため、日常の何気ない会話がすべて露見の危機として読める。", "二つの顔の並行提示", "日常場面への持続的な緊張", "劇的アイロニー論", 0.85, ["double_hook", "near_miss"], ["dual_self", "close_one"]),
            ("lonely_projection", "identity_projection", "分かち合えない自己への投影", "誰にも本当の自分を見せられない孤独が、受け手自身の社会的な仮面の経験と重なる。", "嘘の代償", "主体の孤独への深い共感", "社会的自己呈示論", 0.75, ["cost_of_lies", "acceptance"], ["authenticity"]),
            ("exposure_loss", "loss_aversion", "露見による喪失の恐怖", "正体が露見すれば日常の幸福がすべて失われるという賭け金が、各危機の緊張を支える。", "正体を盾にした脅迫", "失うことへの強い恐れ", "プロスペクト理論", 0.7, ["blackmail", "unmasking"], ["normal_life", "hunter"]),
            ("acceptance_catharsis", "catharsis", "受容による解放", "嘘を重ねてきた緊張が大切な人の受容によって一挙に解かれ、深い安堵が生まれる。", "受容と分かち合い", "長い緊張からの解放", "カタルシス論", 0.8, ["acceptance"], ["close_one"]),
        ],
        "anchor": "大切な人を守るために嘘をつき続け、その嘘によって大切な人を傷つけてしまう主体の矛盾",
        "drivers": ["読者だけが両方の顔を知る情報差", "周期的に訪れる露見の危機", "嘘の代償が積み上がる関係の変化"],
        "emotions": ["高揚", "ひやり", "罪悪感", "緊張", "安堵", "恐怖", "決意", "解放", "温かさ"],
        "template": (
            "{{DUAL_PROTAGONIST}}は平凡な公の顔の裏で{{SECRET_MARK}}を身につけ、秘密の役割を担っている。{{CLOSE_ONE}}への嘘は{{ORDINARY_HAPPINESS}}を蝕み、"
            "{{UNSHARED_SELF}}が深まっていく。{{IDENTITY_HUNTER}}が正体を突き止めて大切な人を盾に取ったとき、主体はその目の前で仮面を外し、"
            "{{ACCEPTANCE_MOMENT}}によって二つの自分を統合する。"
        ),
        "slots": [
            ("DUAL_PROTAGONIST", "agent", "二つの顔を持つ主体", ["冒頭で両方の顔を並行して見せる"], ("昼は配達員の覆面の守り手", "本業と挑戦を隠して両立する起業家", "二つの専門を持つ研究者"), True),
            ("SECRET_MARK", "symbol", "秘密の顔の象徴", [], ("素顔を隠す仮面", "非公開の別名義", "異分野の手法"), True),
            ("CLOSE_ONE", "agent", "公の顔だけを知る大切な人", [], ("幼なじみ", "長年の取引先", "主流派の同僚"), True),
            ("ORDINARY_HAPPINESS", "goal", "平凡な幸福の維持", [], ("約束した日の食卓", "安定した本業", "既存分野での評価"), True),
            ("UNSHARED_SELF", "lack", "本当の自分を見せられない孤独", [], ("誰にも語れない功績", "語れない経歴", "理解されない越境"), True),
            ("IDENTITY_HUNTER", "agent", "正体を暴こうとする敵対者", [], ("覆面の守り手を追う宿敵", "挑戦を暴こうとする競合", "越境を批判する権威"), True),
            ("ACCEPTANCE_MOMENT", "reversal", "受容による統合の瞬間", [], ("仮面ごと受け入れられる", "二つの事業の統合発表", "二分野の接続の承認"), True),
        ],
        "invariants": [
            "読者には冒頭から両方の顔を開示し、作中の大切な人には最後まで隠す",
            "露見の危機は周期的に訪れ、回を追うごとに大切な人の近くで起こす",
            "正体の開示は強制ではなく、大切な人を守るための主体自身の選択として行う",
            "結末で二重性を消すのではなく、秘密を共有する関係として統合する",
        ],
        "guide": {
            "manga": (
                "全32ページ想定。1〜3pで公の顔と秘密の顔を左右のページで対比させ、4〜5pで最初の露見の危機を左ページ終端のひやりとした表情で引く。"
                "8〜10pは嘘の代償を静かな会話劇で描き、12〜14pで追手の接近を背景の人物として何度か映り込ませる。16〜17pで束の間の平穏、19〜21pで脅迫。"
                "めくり（23→24p）の右ページで大切な人の前で仮面のまま戦う姿を大ゴマにし、24〜25pの見開きで仮面を外す瞬間を描く。28〜30pで受容、31〜32pで統合された日常。"
            ),
            "marketing": (
                "顧客が抱える公私の二重性（本業と挑戦、表向きの役割と本音）を主体として描き、それを隠し続ける消耗を具体化する。"
                "製品を二つの顔を統合する手段として位置づけ、秘密を共有できるコミュニティの存在を安心感の源として提示する。"
            ),
            "article": (
                "ある分野で主流の手法と、隠れて使われている異分野の手法の二重性を描き、隠すことで生じるコストを示す。"
                "二つを統合した事例の受容のされ方を分析し、越境の知見を公にするための手順を提案する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.25), (0.1, 0.65), (0.18, 0.3), (0.32, 0.35), (0.4, 0.75), (0.48, 0.3), (0.6, 0.4), (0.75, 0.95), (0.88, 0.6), (1.0, 0.2)],
            "release": [(0.0, 0.2), (0.12, 0.05), (0.2, 0.45), (0.4, 0.05), (0.5, 0.5), (0.62, 0.3), (0.75, 0.05), (0.9, 0.6), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.6), (0.2, 0.7), (0.4, 0.75), (0.6, 0.8), (0.75, 0.9), (0.9, 1.0), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.05), (0.2, 0.1), (0.4, 0.2), (0.6, 0.3), (0.75, 0.6), (0.9, 1.0), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.4), (0.1, 0.6), (0.2, 0.45), (0.4, 0.65), (0.6, 0.6), (0.75, 0.85), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 14,
        "slug": "time_loop_escape",
        "title": "時間反復からの脱出",
        "origin": "film",
        "summary": (
            "同じ時間を繰り返す閉じた循環に囚われた主体が、反復のたびに知識を蓄積し、失敗の原因を一つずつ潰していく。"
            "表面上は同じ出来事が繰り返されるが主体だけが変化し、最後にループの真の原因が自分自身の内面にあったと気づいて循環を断ち切る。"
        ),
        "tags": ["時間反復", "試行錯誤", "知識の蓄積", "自己変革"],
        "lineage": ["時間反復（ループ）物語", "試行錯誤による学習の構造", "繰り返される一日の寓話"],
        "nodes": [
            ("looper", "agent", "protagonist", "同じ時間を繰り返す主体", "反復の記憶を唯一保持し、周囲が気づかない変化の担い手となる。", 0.95, "LOOPING_PROTAGONIST"),
            ("catastrophe", "agent", "loop_trigger", "循環の終点で起こる破局", "反復ごとに主体を巻き戻す出来事であり、回避の試みに応じて形を変える。", 0.8, "RECURRING_CATASTROPHE"),
            ("anchor_person", "agent", "anchor", "ループの中で主体が救いたい存在", "毎回同じ言動を繰り返すが、主体の接し方によって反応が変化する。", 0.7, "ANCHOR_PERSON"),
            ("escape", "desire", "goal", "循環からの脱出", "当初は時間の先へ進むこと自体が目的であり、やがて意味が変わる。", 0.85, "ESCAPE_GOAL"),
            ("inner_cause", "lack", "true_cause", "ループを生んでいる内面の欠落", "主体が向き合ってこなかった後悔や拒絶であり、真の脱出条件となる。", 0.85, "INNER_CAUSE"),
            ("reset_mark", "symbol", "loop_marker", "反復の開始を告げる徴", "毎回同じ形で現れる合図であり、最後の反復の後にだけ異なる姿を見せる。", 0.65, "RESET_MARKER"),
        ],
        "edges": [
            ("catastrophe", "looper", "power_gradient", 0.85, 0.85, "巻き戻しの支配", 0.05),
            ("looper", "escape", "pursuit", 0.9, 0.0, "脱出の試行", 0.12),
            ("looper", "catastrophe", "opposition", 0.85, -0.5, "破局の回避", 0.15),
            ("looper", "anchor_person", "asymmetry", 0.85, 0.9, "主体だけが反復の記憶を持つ", 0.12),
            ("looper", "anchor_person", "dependency", 0.6, -0.1, "繰り返す一日の支え", 0.35),
            ("anchor_person", "looper", "asymmetry", 0.5, 0.3, "相手だけが主体の本心を見抜く", 0.62),
            ("reset_mark", "inner_cause", "embodiment", 0.6, 0.0, "反復の徴が向き合わない後悔を体現する", 0.5),
            ("inner_cause", "escape", "opposition", 0.8, 0.0, "内面の欠落が脱出を阻む", 0.62),
        ],
        "beats": [
            ("first_day", "hook", 0.0, 0.05, "ありふれた一日", "主体が退屈に感じている一日が描かれ、その終わりに破局が起こる。", [], 0.2, 0.05, ["looper", "catastrophe", "anchor_person"], False),
            ("first_reset", "premise", 0.15, 0.03, "最初の巻き戻し", "破局の直後に主体は同じ朝へ戻り、自分だけが記憶を持っていると気づく。", ["first_day"], -0.45, 0.2, ["looper", "reset_mark"], True),
            ("brute_force", "escalation", 0.3, 0.06, "力ずくの回避", "主体は破局を力ずくで避けようと試行を重ねるが、破局は形を変えて必ず訪れる。", ["first_reset"], 0.4, 0.1, ["looper", "catastrophe"], False),
            ("despair_loops", "local_optimization_limit", 0.5, 0.06, "外的解決の限界", "知識を極めても脱出できず、主体は反復を投げやりに消費し始める。", ["brute_force"], 0.2, 0.05, ["looper", "escape", "reset_mark"], False),
            ("anchor_insight", "revelation", 0.62, 0.05, "相手が見抜く本心", "繰り返しの中で救いたい相手と深く関わった主体は、相手の言葉から自分が何から逃げているかを示される。", ["despair_loops"], 0.0, -0.1, ["anchor_person", "looper", "inner_cause"], True),
            ("inner_confrontation", "costly_breakthrough", 0.75, 0.05, "内面との対峙", "主体はループの真の原因である後悔に向き合い、他者のために一日を使い切る選択をする。", ["anchor_insight"], 0.2, -0.1, ["looper", "inner_cause", "anchor_person"], False),
            ("final_loop", "climax", 0.86, 0.04, "最後の一日", "変化した主体が最後の破局に挑む。今度は回避ではなく受容と献身によって破局の意味を変える。", ["inner_confrontation", "brute_force"], 0.3, 0.0, ["looper", "catastrophe", "escape"], True),
            ("tomorrow", "afterglow", 1.0, 0.04, "訪れる明日", "初めて迎える翌朝、反復の徴は異なる姿を見せ、主体は同じ世界を新しい目で見る。", ["final_loop"], -0.5, 0.2, ["looper", "reset_mark", "anchor_person"], False),
        ],
        "mechanisms": [
            ("repetition_variation", "prediction_error", "反復の中の差分による予測誤差", "同じ出来事の反復で受け手に予測を形成させ、わずかな差分を際立たせて驚きと笑いを生む。", "力ずくの回避の反復", "差分への鋭い注意と快感", "予測符号化理論", 0.8, ["brute_force", "despair_loops"], ["reset_mark"]),
            ("mastery_efficacy", "self_efficacy", "知識の蓄積による熟達感", "反復のたびに主体が上達していく過程が、受け手に試行錯誤の報酬を追体験させる。", "知識の蓄積", "自分も学べば変われるという感覚", "自己効力感理論", 0.7, ["brute_force", "inner_confrontation"], ["looper"]),
            ("loop_irony", "dramatic_irony", "やり直しを知る者の劇的アイロニー", "読者と主体だけが一日が繰り返されると知っており、周囲の何気ない言動が切なさと可笑しさを帯びる。", "最初の巻き戻し", "周囲の無知への愛おしさ", "劇的アイロニー論", 0.65, ["first_reset", "anchor_insight"], ["anchor_person"]),
            ("escape_catharsis", "catharsis", "循環の終わりの解放", "閉じた循環が内面の変化によって断ち切られる瞬間に、蓄積した徒労感が一挙に解放される。", "訪れる明日", "長い閉塞からの解放", "カタルシス論", 0.8, ["final_loop", "tomorrow"], ["escape", "inner_cause"]),
        ],
        "anchor": "同じ一日を何度生き直しても、自分自身の後悔からだけは逃げ続けてしまう主体の弱さ",
        "drivers": ["反復ごとに増える主体の知識と周囲の無知の差", "同じ出来事の中のわずかな差分", "破局が形を変えて訪れる謎"],
        "emotions": ["退屈", "混乱", "挑戦心", "苛立ち", "絶望", "気づき", "献身", "解放"],
        "template": (
            "{{LOOPING_PROTAGONIST}}は{{RECURRING_CATASTROPHE}}で終わる一日を繰り返し、{{RESET_MARKER}}とともに同じ朝へ戻される。"
            "反復の知識で破局を力ずくで避けようとする試みはすべて失敗し、主体は外的な解決の限界に達する。{{ANCHOR_PERSON}}の言葉で{{INNER_CAUSE}}に気づいた主体は"
            "{{ESCAPE_GOAL}}の意味を変え、{{TRANSFORMING_ACT}}によって循環を断ち切る。"
        ),
        "slots": [
            ("LOOPING_PROTAGONIST", "agent", "同じ時間を繰り返す主体", ["冒頭で一日への不満を見せる"], ("毎朝同じ駅で目覚める高校生", "同じ失敗を繰り返す事業担当者", "同じ誤りを繰り返す学習者"), True),
            ("RECURRING_CATASTROPHE", "conflict", "循環の終点で起こる破局", ["回避の試みに応じて形を変える"], ("夕暮れの事故", "毎期の目標未達", "同じ箇所でのつまずき"), True),
            ("RESET_MARKER", "symbol", "反復の開始を告げる徴", ["最後の反復の後にだけ姿を変える"], ("鳴り続ける目覚まし時計", "毎月の定例会議", "繰り返される同じ問題"), True),
            ("ANCHOR_PERSON", "agent", "ループの中で救いたい存在", [], ("隣の席の同級生", "離反しかけている顧客", "質問をくれる読者"), True),
            ("INNER_CAUSE", "lack", "ループを生む内面の欠落", ["序盤から小さく仄めかす"], ("伝えられなかった謝罪", "失敗を認めない組織文化", "前提を疑わない思考習慣"), True),
            ("ESCAPE_GOAL", "goal", "循環からの脱出", [], ("明日を迎えること", "成長軌道への復帰", "次の段階への進歩"), True),
            ("TRANSFORMING_ACT", "reversal", "内面の変化による脱出の行為", [], ("後悔していた言葉を伝える", "失敗を公開して学ぶ仕組みを作る", "前提を書き換える"), True),
        ],
        "invariants": [
            "反復は三回以上描き、回を追うごとに主体の知識と周囲の無知の差を広げる",
            "外的な手段による回避はすべて失敗させ、脱出条件を主体の内面に置く",
            "反復の徴は毎回同じ形で描き、最後の一日の後だけ変化させる",
            "最後の一日は回避ではなく受容と献身によって破局の意味を変える",
        ],
        "guide": {
            "manga": (
                "全40ページ想定。1〜6pで退屈な一日と破局を通常のテンポで描き、7pで巻き戻し。各反復の開始は同じ構図の大ゴマ（反復の徴）から始め、ページをめくるたびに既視感を生む。"
                "2回目以降の反復は同じ場面を小ゴマで高速に処理し、差分だけを大きく描く。破局はそれぞれ左ページ終端に置き、めくりで反復の徴に戻す。"
                "26〜29pで相手が本心を見抜く静かな会話、34〜35pの見開きで最後の破局への献身、39〜40pで初めての翌朝を変化した徴とともに描く。"
            ),
            "marketing": (
                "顧客が繰り返している同じ失敗（毎期の未達・同じ課題の再発）を時間反復として描き、外的な施策をいくら重ねても抜け出せない理由を示す。"
                "真の原因を顧客の習慣や前提に置き、それを変える小さな行動を脱出の鍵として提案する。"
            ),
            "article": (
                "同じ失敗が繰り返される事例を時系列で並べ、対症療法がすべて失敗した理由を分析する。"
                "反復に共通する前提を特定し、その前提を書き換えたときに初めて循環が止まることを示す構成。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.5), (0.12, 0.82), (0.15, 0.3), (0.3, 0.72), (0.33, 0.3), (0.5, 0.78), (0.53, 0.35), (0.7, 0.6), (0.86, 0.97), (1.0, 0.2)],
            "release": [(0.0, 0.05), (0.12, 0.0), (0.15, 0.2), (0.3, 0.05), (0.33, 0.25), (0.5, 0.05), (0.6, 0.4), (0.86, 0.1), (1.0, 0.95)],
            "audience_knowledge": [(0.0, 0.1), (0.15, 0.3), (0.33, 0.45), (0.53, 0.6), (0.7, 0.75), (0.86, 0.85), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.05), (0.15, 0.05), (0.33, 0.08), (0.53, 0.1), (0.7, 0.35), (0.86, 0.6), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.4), (0.15, 0.55), (0.33, 0.6), (0.53, 0.7), (0.7, 0.55), (0.86, 0.85), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 15,
        "slug": "monster_mirror",
        "title": "怪物と人間の境界",
        "origin": "literature",
        "summary": (
            "恐怖の対象として現れた異形の存在が、関わりを通じて人間以上の人間性を示し、恐れて排除しようとする共同体の側が怪物性を露わにする。"
            "恐怖から理解へ、理解から迫害へと感情の対象が反転していく。"
        ),
        "tags": ["異形", "共感の反転", "排除", "人間性"],
        "lineage": ["怪物を通じた人間性の問い", "異類との交流譚", "迫害される他者の寓話"],
        "nodes": [
            ("creature", "agent", "outsider", "恐れられる異形の存在", "外見と出自ゆえに怪物とみなされるが、内面には繊細な感受性と孤独を抱える。", 0.9, "FEARED_OUTSIDER"),
            ("witness", "agent", "protagonist", "異形と関わりを持つ主体", "最初は恐れるが、関わる中で異形の人間性を知り、共同体との板挟みになる。", 0.85, "BRIDGE_PROTAGONIST"),
            ("mob", "agent", "community", "異形を恐れる共同体", "未知への恐怖を正義の名で排除へと転化させ、自らの怪物性に気づかない。", 0.8, "FEARFUL_MOB"),
            ("instigator", "agent", "instigator", "恐怖を煽る扇動者", "共同体の恐怖を利用して自らの地位を固めようとする。", 0.6, "FEAR_MONGER"),
            ("belonging", "desire", "goal", "異形が求める居場所", "怪物ではなく一人の存在として受け入れられること。", 0.8, "DESIRED_BELONGING"),
            ("otherness", "lack", "inner_lack", "共同体に欠けた他者への想像力", "恐怖を生む根源であり、共同体の真の欠落。", 0.85, "MISSING_EMPATHY"),
            ("scar", "symbol", "mark_of_difference", "異形であることの徴", "恐怖の焦点となる外見上の徴であり、最後には別の意味を帯びる。", 0.65, "MARK_OF_DIFFERENCE"),
        ],
        "edges": [
            ("mob", "creature", "opposition", 0.85, 0.6, "未知への恐怖と排除", 0.05),
            ("creature", "belonging", "pursuit", 0.85, 0.0, "受け入れられたい願い", 0.3),
            ("creature", "witness", "dependency", 0.7, -0.3, "唯一の理解者への信頼", 0.35),
            ("witness", "creature", "asymmetry", 0.6, 0.4, "主体だけが異形の内面を知る", 0.3),
            ("instigator", "mob", "power_gradient", 0.7, 0.7, "恐怖の扇動", 0.5),
            ("mob", "creature", "power_gradient", 0.85, 0.85, "数と正義の名による圧力", 0.55),
            ("witness", "creature", "betrayal", 0.6, 0.3, "恐れに負けた一度の背信", 0.62),
            ("witness", "mob", "opposition", 0.6, -0.5, "共同体への異議", 0.85),
            ("scar", "otherness", "embodiment", 0.7, 0.0, "徴が共同体の想像力の欠如を映す", 0.55),
            ("otherness", "mob", "power_gradient", 0.7, 0.6, "他者への想像力の欠如が共同体を排除へ駆り立てる", 0.1),
        ],
        "beats": [
            ("fear_hook", "hook", 0.05, 0.05, "異形の出現", "共同体の前に異形の存在が姿を現し、悲鳴とともに恐怖の対象として刻まれる。", [], 0.25, 0.0, ["creature", "mob", "scar"], False),
            ("first_contact", "premise", 0.15, 0.05, "主体との遭遇", "逃げ遅れた主体が異形と至近で対面し、相手が自分を傷つけなかったことに戸惑う。", ["fear_hook"], -0.2, 0.15, ["witness", "creature"], False),
            ("understanding", "setup", 0.3, 0.08, "内面の発見", "主体は密かに交流を重ね、異形が言葉や美しいものを愛する繊細な存在であると知る。", ["first_contact"], -0.15, 0.3, ["witness", "creature", "belonging"], True),
            ("bond", "payoff", 0.4, 0.06, "絆の成立", "異形は主体を唯一の理解者として信頼し、居場所を得られるかもしれないという希望が生まれる。", ["understanding"], -0.1, 0.05, ["creature", "witness", "belonging"], False),
            ("exposure", "escalation", 0.55, 0.05, "存在の露見", "扇動者が異形の居場所を嗅ぎつけ、共同体の恐怖を煽って排除の機運を高める。", ["bond"], 0.25, 0.0, ["instigator", "mob", "creature"], False),
            ("betrayal_under_fear", "crisis", 0.62, 0.04, "恐れに負けた背信", "圧力に屈した主体が一度だけ異形を見捨て、異形は理解者に裏切られたと知る。", ["exposure", "bond"], 0.15, 0.0, ["witness", "creature", "mob"], True),
            ("hunt", "escalation", 0.75, 0.06, "正義の名の迫害", "共同体は正義の名のもとに異形を追い詰め、その過程で自らの残酷さを露わにしていく。", ["betrayal_under_fear"], 0.15, 0.05, ["mob", "creature", "instigator", "otherness"], False),
            ("final_stand", "climax", 0.88, 0.05, "境界での選択", "追い詰められた異形が報復の力を持ちながら共同体の子どもを救い、主体は共同体の前で異形を庇う。", ["hunt"], 0.2, -0.3, ["creature", "witness", "mob", "scar"], True),
            ("mirror_afterglow", "afterglow", 1.0, 0.05, "どちらが怪物か", "異形は去り、残された共同体は自らの恐怖が生んだ怪物性と向き合わされる。徴は恐怖ではなく悔恨の記憶となる。", ["final_stand"], -0.5, 0.0, ["mob", "otherness", "scar"], False),
        ],
        "mechanisms": [
            ("empathy_reversal", "empathy_transfer", "恐怖から共感への移入の反転", "恐怖の対象として提示した存在の内面を段階的に開示し、受け手の感情移入の対象を反転させる。", "内面の発見", "異形への強い共感と保護欲", "共感移入・態度変容理論", 0.9, ["understanding", "bond"], ["creature"]),
            ("mob_irony", "dramatic_irony", "知る者と知らない者の断絶", "読者は異形の繊細さを知っているが共同体は知らず、迫害の場面すべてが痛ましい誤解として映る。", "存在の露見と迫害", "義憤ともどかしさ", "劇的アイロニー論", 0.8, ["exposure", "hunt"], ["mob", "creature"]),
            ("monstrosity_dissonance", "cognitive_dissonance", "怪物と人間の基準の揺らぎ", "排除する側の残酷さが異形の優しさと対照されることで、受け手の善悪の基準そのものが揺さぶられる。", "正義の名の迫害", "自らの偏見への気づき", "認知的不協和理論", 0.75, ["hunt", "mirror_afterglow"], ["otherness", "mob"]),
            ("guilt_judgment", "moral_judgment", "背信と迫害への道徳的判断", "主体の一度の背信と共同体の迫害が、受け手自身の同調圧力への弱さを問いただす。", "恐れに負けた背信", "罪悪感を伴う省察", "道徳的直観理論", 0.65, ["betrayal_under_fear", "final_stand"], ["witness"]),
        ],
        "anchor": "怪物と呼ばれながら、誰よりも言葉と美しいものを愛している異形の孤独",
        "drivers": ["恐怖から理解へと反転する感情の対象", "読者だけが知る異形の内面", "主体が共同体と異形のどちらにつくかという問い"],
        "emotions": ["恐怖", "戸惑い", "好奇", "温かさ", "不安", "罪悪感", "義憤", "痛み", "悔恨"],
        "template": (
            "{{MARK_OF_DIFFERENCE}}を持つ{{FEARED_OUTSIDER}}は、{{FEARFUL_MOB}}にとって恐怖の対象として現れる。{{BRIDGE_PROTAGONIST}}は交流を通じてその内面を知り、"
            "異形は{{DESIRED_BELONGING}}を夢見る。しかし{{FEAR_MONGER}}が恐怖を煽り、主体は一度圧力に屈して異形を見捨てる。"
            "迫害の果てに異形が{{MERCIFUL_ACT}}を選んだとき、共同体は自らの{{MISSING_EMPATHY}}と向き合わされる。"
        ),
        "slots": [
            ("FEARED_OUTSIDER", "agent", "恐れられる異形の存在", ["外見と内面の落差を明確にする"], ("人里離れて暮らす巨躯の存在", "業界の常識から外れた新手法", "誤解され続けてきた概念"), True),
            ("MARK_OF_DIFFERENCE", "symbol", "異形であることの徴", [], ("顔の大きな傷跡", "見慣れない製品の形", "難解に見える専門用語"), True),
            ("FEARFUL_MOB", "agent", "異形を恐れる共同体", [], ("閉鎖的な村の人々", "変化を恐れる業界", "通説を信じる多数派"), True),
            ("BRIDGE_PROTAGONIST", "agent", "異形と関わる主体", ["一度は圧力に屈する弱さを持たせる"], ("迷い込んだ少女", "早期に導入した担当者", "偏見を検証する読者"), True),
            ("DESIRED_BELONGING", "goal", "異形が求める居場所", [], ("村の祭りへの参加", "市場での正当な評価", "主流の議論への受容"), True),
            ("FEAR_MONGER", "agent", "恐怖を煽る扇動者", [], ("地位を狙う村の有力者", "不安を煽る競合の宣伝", "誇張された批判"), True),
            ("MERCIFUL_ACT", "reversal", "報復ではなく救済を選ぶ行為", [], ("追手の子どもを救う", "批判者の課題を解決する", "反論者の疑問に誠実に答える"), True),
            ("MISSING_EMPATHY", "lack", "共同体に欠けた他者への想像力", [], ("見た目で判断する心", "前例主義", "検証なき拒絶"), True),
        ],
        "invariants": [
            "異形は最初に恐怖の対象として提示し、内面は主体を通じて段階的に開示する",
            "共同体の迫害は悪意ではなく恐怖と正義感から生じるように描く",
            "主体に一度は圧力に屈する弱さを持たせ、単純な善悪の構図を避ける",
            "異形は報復の力を持ちながら救済を選び、怪物性の所在を反転させる",
        ],
        "guide": {
            "manga": (
                "全36ページ想定。1〜2pで異形の出現を逆光と黒ベタのシルエットの大ゴマで恐ろしく描く。5〜6pで主体との遭遇を至近距離の目のアップで。"
                "9〜15pは交流を柔らかい線とトーンで描き、異形の表情を少しずつ豊かにしていく。18〜19pで扇動を群衆の俯瞰で、20pで主体の背信を左ページ終端に置く。"
                "めくり（21→22p）で裏切られた異形の顔を大ゴマにする。24〜29pは迫害を斜めのコマ割りで加速し、めくり（31→32p）で異形が子どもを救う姿を見開きにする。"
                "35〜36pで去った異形と残された共同体を冒頭と同じ構図の逆光で描く。"
            ),
            "marketing": (
                "市場で誤解されている新しい製品や手法を異形として描き、恐怖や偏見の構造を丁寧に分析する。実際に関わった顧客の体験を通じて内面（価値）を開示し、"
                "批判者の課題をも解決する誠実な姿勢で信頼の反転を設計する。"
            ),
            "article": (
                "誤解されてきた概念や少数派の手法を取り上げ、なぜ多数派がそれを恐れ排除してきたかを分析する。"
                "実際に使った人の経験から価値を明らかにし、結論で排除の構造そのものを問い直す。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.45), (0.05, 0.7), (0.15, 0.5), (0.25, 0.38), (0.4, 0.28), (0.55, 0.55), (0.62, 0.68), (0.75, 0.8), (0.88, 0.96), (1.0, 0.45)],
            "release": [(0.0, 0.05), (0.15, 0.15), (0.3, 0.5), (0.42, 0.6), (0.55, 0.2), (0.75, 0.05), (0.92, 0.4), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.1), (0.15, 0.3), (0.3, 0.6), (0.45, 0.75), (0.7, 0.85), (0.9, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.15, 0.12), (0.3, 0.15), (0.45, 0.15), (0.7, 0.2), (0.9, 0.6), (1.0, 0.8)],
            "cognitive_load": [(0.0, 0.45), (0.1, 0.55), (0.3, 0.4), (0.5, 0.45), (0.75, 0.65), (0.88, 0.8), (1.0, 0.4)],
        },
    }
)

PATTERNS.append(
    {
        "num": 16,
        "slug": "underdog_ensemble",
        "title": "落ちこぼれ集団の逆襲",
        "origin": "film",
        "summary": (
            "それぞれ欠点を抱えて居場所を失った者たちが一つの目的のもとに集められ、反発と挫折を経てチームとして結束する。"
            "初戦の敗北で一度崩壊した集団が、各人の欠点を補い合う形で再編され、圧倒的な強者に挑む。"
        ),
        "tags": ["チーム", "落ちこぼれ", "結束", "逆襲"],
        "lineage": ["仲間集め（アンサンブル）の構造", "弱者連合の逆転劇", "村を守る七人の侍型の集団英雄譚"],
        "nodes": [
            ("leader", "agent", "protagonist", "落ちこぼれを束ねる不完全な主体", "自らも挫折を抱えながら、集団に可能性を見いだして束ねようとする。", 0.9, "RELUCTANT_LEADER"),
            ("misfits", "agent", "ensemble", "欠点を抱えた者たちの集団", "それぞれが一芸と致命的な欠点を持ち、互いに反発し合う。", 0.85, "MISFIT_ENSEMBLE"),
            ("giant", "agent", "antagonist", "圧倒的な強者", "資源と実績を持ち、集団を取るに足らない存在として見下す。", 0.85, "OVERWHELMING_GIANT"),
            ("victory", "desire", "goal", "強者からの勝利", "勝利そのものよりも、居場所を失った者たちが存在を証明すること。", 0.85, "UNDERDOG_VICTORY"),
            ("cohesion", "lack", "collective_lack", "互いへの信頼の欠如", "各人の欠点と過去の傷が生む不信であり、集団の最大の弱点。", 0.8, "MISSING_COHESION"),
            ("banner", "symbol", "team_symbol", "集団の結束を象徴する旗印", "最初は寄せ集めの徴として嘲笑され、最後には誇りの象徴となる。", 0.65, "TEAM_BANNER"),
            ("stakes", "desire", "protected", "敗北すれば失われる守るべきもの", "集団が戦う理由となる、彼ら自身の居場所や弱き者たち。", 0.7, "PROTECTED_STAKES"),
        ],
        "edges": [
            ("giant", "misfits", "power_gradient", 0.95, 0.95, "資源と実績による圧倒", 0.05),
            ("leader", "victory", "pursuit", 0.85, 0.0, "存在の証明", 0.1),
            ("leader", "misfits", "opposition", 0.45, 0.1, "反発と衝突", 0.2),
            ("misfits", "leader", "dependency", 0.6, -0.3, "束ね役への渋々の信頼", 0.25),
            ("giant", "leader", "asymmetry", 0.6, 0.6, "強者だけが持つ情報と戦術", 0.45),
            ("banner", "cohesion", "embodiment", 0.7, 0.0, "旗印が結束の有無を体現する", 0.6),
            ("misfits", "stakes", "pursuit", 0.7, 0.0, "守るべきものへの献身", 0.62),
            ("leader", "giant", "asymmetry", 0.55, 0.4, "欠点の組み合わせという誰も知らない武器", 0.75),
            ("misfits", "giant", "opposition", 0.85, -0.8, "弱者の挑戦", 0.85),
            ("cohesion", "misfits", "power_gradient", 0.7, 0.6, "結束の欠如が集団の力を内側から削ぐ", 0.2),
        ],
        "beats": [
            ("misfit_hook", "hook", 0.0, 0.05, "居場所を失った者たち", "それぞれの場所で欠点ゆえに居場所を失った者たちが、短い場面の連続で紹介される。", [], 0.1, 0.05, ["misfits", "leader"], False),
            ("gathering", "premise", 0.1, 0.05, "寄せ集めの結成", "守るべきもののために主体が集団を束ねるが、強者からは嘲笑される。", ["misfit_hook"], -0.05, 0.05, ["leader", "misfits", "giant", "stakes"], False),
            ("friction", "setup", 0.22, 0.06, "反発と衝突", "各人の欠点が衝突し、集団は練習の段階から崩壊しかける。", ["gathering"], 0.15, 0.05, ["misfits", "leader", "cohesion"], False),
            ("small_win", "payoff", 0.35, 0.05, "小さな成功", "偶然の連携で小さな成功を収め、集団に初めて笑いが生まれる。", ["friction"], 0.05, 0.05, ["misfits", "banner"], False),
            ("first_defeat", "crisis", 0.48, 0.05, "初戦の惨敗", "強者との初戦で集団は完膚なきまでに敗れ、互いを責め合って散り散りになる。", ["small_win"], 0.3, 0.05, ["giant", "misfits", "cohesion"], True),
            ("dark_regroup", "transition", 0.62, 0.06, "どん底からの再編", "主体が一人ずつ訪ね歩き、各人の欠点が別の誰かの欠点を補えることに気づく。", ["first_defeat"], -0.35, 0.1, ["leader", "misfits", "cohesion"], True),
            ("new_strategy", "setup", 0.75, 0.05, "欠点を組み合わせた戦術", "欠点の組み合わせによる誰も予想しない戦術が組み上がり、旗印が誇りの象徴として掲げ直される。", ["dark_regroup"], 0.2, 0.2, ["leader", "misfits", "banner"], False),
            ("final_battle", "climax", 0.88, 0.05, "強者への逆襲", "決戦で集団は一時劣勢に陥るが、各人が自らの欠点を武器に変えて連鎖的に逆転する。", ["new_strategy", "first_defeat"], 0.35, -0.15, ["misfits", "giant", "leader", "stakes"], True),
            ("victory_afterglow", "afterglow", 1.0, 0.05, "居場所の獲得", "勝利の後、集団はそれぞれが居場所を得たことを知り、旗印のもとで新しい関係が始まる。", ["final_battle"], -0.6, 0.0, ["misfits", "banner", "victory"], False),
        ],
        "mechanisms": [
            ("underdog_effect", "empathy_transfer", "弱者への応援効果", "圧倒的な力の差と各人の欠点が、受け手に強い応援の感情と自己投影の余地を与える。", "寄せ集めの結成と嘲笑", "集団への強い肩入れ", "アンダードッグ効果", 0.85, ["gathering", "first_defeat"], ["misfits", "giant"]),
            ("ensemble_projection", "identity_projection", "多様な欠点への自己投影", "集団の各人が異なる欠点を持つため、受け手は誰か一人に必ず自分を重ねられる。", "居場所を失った者たちの紹介", "特定の人物への深い愛着", "同一化理論", 0.75, ["misfit_hook", "dark_regroup"], ["misfits"]),
            ("flaw_reversal", "prediction_error", "欠点が武器になる予測誤差", "弱点として描かれてきた特性が決戦で武器に反転し、受け手の予測を気持ちよく裏切る。", "強者への逆襲", "爽快な驚き", "予測符号化理論", 0.8, ["new_strategy", "final_battle"], ["leader"]),
            ("victory_peak_end", "peak_end", "逆転の頂点と温かい終幕", "決戦の逆転という頂点と居場所の獲得という温かい終幕が、体験全体を強い肯定感で記憶させる。", "居場所の獲得", "高揚と満足の余韻", "ピーク・エンドの法則", 0.7, ["final_battle", "victory_afterglow"], ["banner"]),
        ],
        "anchor": "欠点ゆえに居場所を失いながら、それでも誰かに必要とされたいと願う者たちの不器用さ",
        "drivers": ["圧倒的な力の差", "各人の欠点がどう噛み合うかという期待", "初戦の惨敗からの再起"],
        "emotions": ["共感", "苛立ち", "笑い", "期待", "絶望", "決意", "高揚", "歓喜", "温かさ"],
        "template": (
            "欠点ゆえに居場所を失った{{MISFIT_ENSEMBLE}}を、{{RELUCTANT_LEADER}}が{{PROTECTED_STAKES}}のために束ねる。{{OVERWHELMING_GIANT}}は寄せ集めを嘲笑し、"
            "{{MISSING_COHESION}}ゆえに集団は初戦で惨敗して散り散りになる。どん底で各人の欠点が互いを補えると気づいた集団は{{FLAW_STRATEGY}}を組み上げ、"
            "{{TEAM_BANNER}}を誇りとして掲げ直して{{UNDERDOG_VICTORY}}を掴む。"
        ),
        "slots": [
            ("MISFIT_ENSEMBLE", "agent", "欠点を抱えた者たちの集団", ["各人に一芸と致命的な欠点を一つずつ与える"], ("廃部寸前の部活の面々", "寄せ集めの社内プロジェクト", "異分野から集まった研究会"), True),
            ("RELUCTANT_LEADER", "agent", "落ちこぼれを束ねる不完全な主体", [], ("挫折した元選手の顧問", "左遷された中間管理職", "独学の在野研究者"), True),
            ("PROTECTED_STAKES", "goal", "敗北すれば失われる守るべきもの", [], ("取り壊される部室", "撤退が決まりかけた事業", "消えかけた研究分野"), True),
            ("OVERWHELMING_GIANT", "agent", "圧倒的な強者", [], ("全国大会の常勝校", "市場を独占する大手", "主流学派の権威"), True),
            ("MISSING_COHESION", "lack", "互いへの信頼の欠如", [], ("責任の押し付け合い", "部門間の不信", "分野間の相互無理解"), True),
            ("FLAW_STRATEGY", "reversal", "欠点を組み合わせた戦術", ["序盤に描いた欠点を全員分回収する"], ("短所を囮にする連携技", "弱みを差別化に変える戦略", "異分野の盲点を補い合う手法"), True),
            ("TEAM_BANNER", "symbol", "結束を象徴する旗印", [], ("手縫いのチームの旗", "手作りのロゴ", "研究会の名前"), True),
            ("UNDERDOG_VICTORY", "goal", "強者からの勝利", [], ("常勝校からの一勝", "大手からの顧客獲得", "定説への有効な反証"), True),
        ],
        "invariants": [
            "集団の各人に一芸と致命的な欠点を一つずつ与え、序盤で全員分を提示する",
            "初戦は必ず惨敗させ、集団を一度散り散りにする",
            "逆転の鍵は新たな力の獲得ではなく、既存の欠点の組み合わせ方の発見に置く",
            "勝利の意味を存在の証明と居場所の獲得へと拡張して終える",
        ],
        "guide": {
            "manga": (
                "全45ページ想定。1〜6pで一人ずつの欠点を1ページずつの小エピソードで紹介し、7〜8pの見開きで寄せ集めの集合絵を描く。10〜14pで反発と衝突、16〜17pで小さな成功の笑い。"
                "めくり（21→22p）の右ページで初戦の惨敗を断ち切りの大ゴマにし、23〜24pで散り散りの各人をコマの分断で表現する。27〜32pで一人ずつ訪ね歩く再編、33pで旗印を掲げ直す。"
                "38〜41pは決戦を斜め割りで加速し、めくり（41→42p）で欠点が武器になる連鎖逆転を見開きで描く。44〜45pで居場所の獲得。"
            ),
            "marketing": (
                "顧客を欠点を抱えた集団として肯定的に描き、大手のような資源がなくても欠点の組み合わせ方次第で勝てることを示す。"
                "自社の製品を集団を束ねる旗印として位置づけ、初期の失敗を隠さず再起の物語として語る。"
            ),
            "article": (
                "弱小の組織や手法が強者に勝った事例を分析し、勝因が新たな資源ではなく既存の弱点の組み合わせにあったことを示す。"
                "読者が自分の組織の欠点を棚卸しし、組み合わせを設計するためのフレームワークを結論とする。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.35), (0.1, 0.3), (0.22, 0.45), (0.35, 0.5), (0.48, 0.78), (0.55, 0.6), (0.62, 0.4), (0.75, 0.6), (0.88, 0.96), (1.0, 0.15)],
            "release": [(0.0, 0.2), (0.2, 0.35), (0.35, 0.45), (0.48, 0.05), (0.62, 0.25), (0.72, 0.55), (0.88, 0.1), (0.95, 0.85), (1.0, 0.95)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.35), (0.48, 0.5), (0.62, 0.65), (0.75, 0.8), (0.88, 0.85), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.2), (0.2, 0.3), (0.48, 0.55), (0.62, 0.5), (0.75, 0.45), (0.88, 0.6), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.4), (0.15, 0.55), (0.35, 0.5), (0.48, 0.65), (0.62, 0.45), (0.75, 0.6), (0.88, 0.8), (1.0, 0.3)],
        },
    }
)

PATTERNS.append(
    {
        "num": 17,
        "slug": "forbidden_bond",
        "title": "禁じられた絆",
        "origin": "drama",
        "summary": (
            "対立する二つの陣営に属する二人が惹かれ合い、秘密の絆を育む。陣営間の対立が激化するほど絆は試され、すれ違いと誤解が重なって破局寸前に至るが、"
            "二人の選択が陣営の対立そのものの意味を問い直す。"
        ),
        "tags": ["禁じられた関係", "対立陣営", "すれ違い", "和解"],
        "lineage": ["敵対する家の恋人たち（悲恋の構造）", "境界を越える絆", "誤解による悲劇と和解"],
        "nodes": [
            ("lover_a", "agent", "protagonist", "一方の陣営に属する主体", "陣営への忠誠と相手への想いの間で引き裂かれる。", 0.9, "LOVER_A"),
            ("lover_b", "agent", "deuteragonist", "対立陣営に属する相手", "同じく自陣営の期待を背負いながら、境界を越えて主体と心を通わせる。", 0.9, "LOVER_B"),
            ("faction_a", "agent", "faction", "主体の属する陣営", "過去の因縁から相手陣営を敵視し、主体に忠誠を求める。", 0.7, "FACTION_A"),
            ("faction_b", "agent", "rival_faction", "相手の属する陣営", "同じく因縁を抱え、相手に主体の陣営との断絶を迫る。", 0.7, "FACTION_B"),
            ("union", "desire", "goal", "陣営を越えた結びつき", "二人が望む、所属に縛られない関係。", 0.85, "FORBIDDEN_UNION"),
            ("feud", "lack", "inherited_lack", "受け継がれた憎しみの理由の空白", "誰も対立の起源を覚えていないのに続いていく憎しみ。", 0.75, "INHERITED_FEUD"),
            ("token", "symbol", "secret_token", "二人だけの秘密の徴", "絆を確かめ合う小さな物であり、すれ違いの場面で誤解の種にもなる。", 0.65, "SECRET_TOKEN"),
        ],
        "edges": [
            ("faction_a", "faction_b", "opposition", 0.9, 0.0, "受け継がれた対立", 0.0),
            ("lover_a", "union", "pursuit", 0.85, 0.0, "結びつきへの願い", 0.15),
            ("lover_b", "union", "pursuit", 0.85, 0.0, "結びつきへの願い", 0.15),
            ("lover_a", "lover_b", "dependency", 0.8, 0.0, "互いへの想い", 0.2),
            ("token", "union", "embodiment", 0.7, 0.0, "秘密の徴が絆を体現する", 0.2),
            ("faction_a", "lover_a", "power_gradient", 0.75, 0.75, "忠誠の強要", 0.3),
            ("faction_b", "lover_b", "power_gradient", 0.75, 0.75, "断絶の強要", 0.3),
            ("lover_a", "lover_b", "asymmetry", 0.5, 0.3, "伝わらなかった真意", 0.72),
            ("token", "feud", "embodiment", 0.4, 0.0, "徴の誤読が憎しみの連鎖を映す", 0.85),
        ],
        "beats": [
            ("encounter_hook", "hook", 0.0, 0.05, "境界での出会い", "対立する陣営の二人が、互いの所属を知らないまま出会い惹かれ合う。", [], 0.1, 0.05, ["lover_a", "lover_b"], False),
            ("identity_known", "premise", 0.12, 0.04, "所属の判明", "相手が敵対陣営の者だと知り、二人は想いが禁じられたものであると悟る。", ["encounter_hook"], 0.1, 0.05, ["lover_a", "lover_b", "faction_a", "faction_b"], False),
            ("secret_bond", "setup", 0.28, 0.08, "秘密の絆", "二人は秘密の徴を交わして密かに会い続け、陣営の外にある自分を初めて知る。", ["identity_known"], -0.1, 0.05, ["lover_a", "lover_b", "token", "union"], False),
            ("hopeful_plan", "payoff", 0.4, 0.05, "結びつきの誓い", "二人は陣営を越えて結ばれる計画を立て、和解の可能性に希望を抱く。", ["secret_bond"], 0.05, 0.0, ["lover_a", "lover_b", "union"], False),
            ("feud_erupts", "escalation", 0.55, 0.05, "対立の激化", "陣営間の衝突が流血に至り、主体は身近な者を守るために相手陣営の者を傷つけてしまう。", ["hopeful_plan"], 0.3, 0.05, ["faction_a", "faction_b", "lover_a", "feud"], True),
            ("forced_separation", "crisis", 0.72, 0.05, "引き離し", "二人はそれぞれの陣営に引き離され、相手に真意を伝える手段を失う。", ["feud_erupts"], 0.25, 0.15, ["faction_a", "faction_b", "lover_a", "lover_b"], False),
            ("misreading", "climax", 0.85, 0.05, "致命的なすれ違い", "伝言の行き違いと秘密の徴の誤読により、二人はそれぞれ相手が絆を捨てたと信じ込む。", ["forced_separation", "secret_bond"], 0.3, 0.1, ["lover_a", "lover_b", "token"], True),
            ("truth_reunion", "revelation", 0.93, 0.04, "真意の到達", "破局の寸前に徴の本当の意味が届き、二人は互いの想いが変わっていなかったと知る。", ["misreading"], -0.35, -0.3, ["lover_a", "lover_b", "token", "union"], True),
            ("reconciliation_afterglow", "afterglow", 1.0, 0.04, "対立の意味の問い直し", "二人の選択が陣営に対立の起源の空白を突きつけ、憎しみの連鎖に初めて亀裂が入る。", ["truth_reunion"], -0.3, 0.0, ["faction_a", "faction_b", "feud"], False),
        ],
        "mechanisms": [
            ("forbidden_scarcity", "scarcity", "禁止による価値の増幅", "許されない関係であるほど、二人の短い逢瀬が希少で切実なものとして受け止められる。", "秘密の絆", "逢瀬への強い没入", "心理的リアクタンス理論", 0.8, ["secret_bond", "hopeful_plan"], ["union", "token"]),
            ("misreading_irony", "dramatic_irony", "すれ違いを知る読者の焦燥", "読者は伝言の行き違いの真相を知っているが二人は知らず、すれ違いの場面が耐えがたい焦燥を生む。", "致命的なすれ違い", "届いてほしいという強い願い", "劇的アイロニー論", 0.85, ["forced_separation", "misreading"], ["lover_a", "lover_b"]),
            ("feud_judgment", "moral_judgment", "受け継がれた憎しみへの問い", "誰も起源を覚えていない対立が二人を引き裂く理不尽さが、受け手に対立そのものの正当性を問わせる。", "対立の激化", "憎しみの連鎖への批判的視点", "道徳的直観理論", 0.7, ["feud_erupts", "reconciliation_afterglow"], ["feud", "faction_a", "faction_b"]),
            ("reunion_catharsis", "catharsis", "真意の到達による解放", "破局寸前で想いが届く瞬間に、すれ違いで蓄積した焦燥が一挙に解放される。", "真意の到達", "涙を伴う安堵", "カタルシス論", 0.8, ["truth_reunion"], ["token"]),
        ],
        "anchor": "所属という自分では選べないものによって、自分で選んだ想いを否定される二人の痛み",
        "drivers": ["許されない関係の秘密の逢瀬", "陣営間の対立の激化", "読者だけが知るすれ違いの真相"],
        "emotions": ["ときめき", "戸惑い", "幸福", "希望", "衝撃", "引き裂かれる痛み", "焦燥", "安堵", "祈り"],
        "template": (
            "{{FACTION_A}}と{{FACTION_B}}は{{INHERITED_FEUD}}によって憎しみ合っている。{{LOVER_A}}と{{LOVER_B}}は所属を越えて惹かれ合い、"
            "{{SECRET_TOKEN}}を交わして{{FORBIDDEN_UNION}}を誓う。対立の激化が二人を引き離し、伝言の行き違いによる{{FATAL_MISREADING}}が破局を招きかけるが、"
            "徴の真意が届いたとき、二人の選択は対立の起源の空白を陣営に突きつける。"
        ),
        "slots": [
            ("FACTION_A", "agent", "主体の属する陣営", [], ("代々続く武門の一族", "老舗企業", "ある学派"), True),
            ("FACTION_B", "agent", "相手の属する陣営", [], ("対立する商家の一族", "新興の競合企業", "対立する学派"), True),
            ("INHERITED_FEUD", "lack", "受け継がれた憎しみの理由の空白", ["起源を誰も説明できない設定にする"], ("百年前の諍い", "過去の訴訟の遺恨", "かつての論争の感情的対立"), True),
            ("LOVER_A", "agent", "一方の陣営に属する主体", [], ("一族の跡取り", "老舗の若手社員", "学派の若手研究者"), True),
            ("LOVER_B", "agent", "対立陣営に属する相手", [], ("商家の娘", "競合の企画担当", "対立学派の研究者"), True),
            ("SECRET_TOKEN", "symbol", "二人だけの秘密の徴", ["すれ違いの場面で誤読の種にする"], ("片方ずつ持つ根付", "共同開発の試作品", "共著の草稿"), True),
            ("FORBIDDEN_UNION", "goal", "陣営を越えた結びつき", [], ("祝言", "業務提携", "共同研究"), True),
            ("FATAL_MISREADING", "irony", "致命的なすれ違い", [], ("届かなかった手紙", "誤解された提携の打診", "誤読された論文の一節"), True),
        ],
        "invariants": [
            "二人は互いの所属を知る前に惹かれ合い、想いの純粋さを所属から切り離して提示する",
            "陣営の対立は二人の選択と無関係に激化させ、主体自身にも加害の一端を担わせる",
            "すれ違いの真相は読者にだけ開示し、二人には破局の寸前まで隠す",
            "結末で対立そのものの起源と意味を問い直させる",
        ],
        "guide": {
            "manga": (
                "全36ページ想定。1〜4pで所属を知らない出会いを柔らかなトーンで描き、5pで所属の判明を左ページ終端に置く。8〜14pの逢瀬は余白の多いコマで静かに描き、秘密の徴を毎回小ゴマで映す。"
                "めくり（17→18p）で陣営の衝突を断ち切りの大ゴマにし、主体の加害の瞬間を挟む。22〜27pで引き離しと伝言の行き違いを、左右のページで二人を並行描写するカットバックで見せ、"
                "めくり（29→30p）で誤読の瞬間を見開きにする。31〜33pで徴の真意の到達、34〜36pで陣営の人々の表情の変化を描く。"
            ),
            "marketing": (
                "対立してきた二つの立場（部門・業界・顧客と売り手）を陣営として描き、両者の間に生まれる協働の価値を禁じられた絆として提示する。"
                "誤解の構造を丁寧に解きほぐし、自社を両者をつなぐ徴として位置づける。"
            ),
            "article": (
                "対立する二つの学派や立場の歴史を整理し、対立の起源が既に失われていることを示す。両者を結ぶ試みがなぜ誤解されてきたかを分析し、"
                "結論で対立を越えた統合の可能性を提案する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.3), (0.12, 0.4), (0.25, 0.3), (0.4, 0.35), (0.55, 0.62), (0.65, 0.55), (0.72, 0.8), (0.85, 0.97), (0.93, 0.6), (1.0, 0.3)],
            "release": [(0.0, 0.1), (0.15, 0.4), (0.28, 0.6), (0.42, 0.65), (0.55, 0.2), (0.72, 0.05), (0.87, 0.05), (0.95, 0.7), (1.0, 0.75)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.45), (0.4, 0.6), (0.6, 0.7), (0.72, 0.85), (0.85, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.2), (0.2, 0.45), (0.4, 0.6), (0.6, 0.6), (0.72, 0.45), (0.85, 0.5), (0.93, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.35), (0.2, 0.4), (0.4, 0.35), (0.55, 0.55), (0.72, 0.65), (0.85, 0.8), (1.0, 0.35)],
        },
    }
)

PATTERNS.append(
    {
        "num": 18,
        "slug": "paradise_flaw",
        "title": "楽園の綻びと隠された代償",
        "origin": "literature",
        "summary": (
            "完璧に見える共同体の中で満ち足りて暮らす主体が、小さな違和感をきっかけに、その幸福が見えない誰かの犠牲によって成り立っていると知る。"
            "真実を知った主体は、楽園に留まるか、去るか、変えるかの選択を迫られる。"
        ),
        "tags": ["理想郷", "隠された代償", "真実の発見", "倫理的選択"],
        "lineage": ["理想郷の裏側（ユートピアの反転）", "共同体を支える隠された犠牲の寓話", "目覚めの物語"],
        "nodes": [
            ("dweller", "agent", "protagonist", "楽園で満ち足りて暮らす主体", "共同体の規則を疑うことなく幸福を享受しているが、感受性の鋭さゆえに違和感を拾う。", 0.95, "CONTENT_DWELLER"),
            ("guardians", "agent", "system_keepers", "楽園の秩序を守る管理者たち", "犠牲の存在を知りながら、全体の幸福のための必要悪として隠し続ける。", 0.8, "ORDER_KEEPERS"),
            ("hidden_one", "agent", "sacrificed", "見えない場所で犠牲を担う存在", "共同体の幸福の代償を一身に負わされ、存在そのものを隠されている。", 0.85, "HIDDEN_SACRIFICE"),
            ("doubter", "agent", "catalyst", "かつて真実を知った離脱者", "主体に違和感の意味を示唆し、選択の可能性を示す。", 0.55, None),
            ("happiness", "desire", "goal", "完全な幸福の維持", "主体がこれまで疑わなかった価値であり、真実の後に意味が揺らぐ。", 0.75, "PERFECT_HAPPINESS"),
            ("complicity", "lack", "moral_lack", "知らずに加担していた倫理的空白", "幸福の享受そのものが加担であったという、主体の内なる欠落。", 0.85, "UNKNOWING_COMPLICITY"),
            ("anomaly", "symbol", "first_crack", "楽園に生じた小さな違和感の徴", "完璧な日常にただ一つ合わない細部であり、真実への入口となる。", 0.7, "FIRST_ANOMALY"),
        ],
        "edges": [
            ("guardians", "dweller", "asymmetry", 0.85, 0.9, "管理者だけが代償を知る", 0.0),
            ("guardians", "hidden_one", "power_gradient", 0.95, 0.95, "犠牲の強制と隠蔽", 0.0),
            ("dweller", "happiness", "pursuit", 0.7, 0.0, "幸福の享受", 0.05),
            ("dweller", "guardians", "dependency", 0.7, -0.6, "秩序への信頼", 0.05),
            ("anomaly", "complicity", "embodiment", 0.7, 0.0, "違和感の徴が見えない加担を映す", 0.35),
            ("doubter", "dweller", "asymmetry", 0.5, 0.4, "離脱者が示唆する真実", 0.45),
            ("guardians", "dweller", "betrayal", 0.75, 0.6, "幸福の欺瞞", 0.55),
            ("happiness", "hidden_one", "opposition", 0.85, 0.8, "多数の幸福と一人の犠牲の対立", 0.55),
            ("dweller", "hidden_one", "pursuit", 0.6, 0.0, "犠牲者への接近", 0.75),
            ("complicity", "dweller", "power_gradient", 0.65, 0.5, "知らずに続けてきた加担が主体の幸福を蝕む", 0.6),
        ],
        "beats": [
            ("idyll_hook", "hook", 0.0, 0.06, "完璧な日常", "争いも欠乏もない共同体の満ち足りた日常が、隅々まで美しく描かれる。", [], 0.0, 0.0, ["dweller", "happiness", "guardians"], False),
            ("ritual", "premise", 0.15, 0.06, "疑われない規則", "共同体には誰も理由を問わない小さな規則があり、主体もそれを当然のこととして守っている。", ["idyll_hook"], 0.0, 0.05, ["dweller", "guardians"], False),
            ("anomaly_found", "setup", 0.35, 0.05, "小さな違和感", "主体は完璧な日常の中でただ一つ合わない細部に気づき、それを忘れられなくなる。", ["ritual"], 0.15, 0.1, ["dweller", "anomaly"], True),
            ("whisper", "escalation", 0.45, 0.06, "離脱者の示唆", "かつて共同体を去った者が、違和感の先に何があるかを謎めいた言葉で示唆する。", ["anomaly_found"], 0.1, 0.1, ["doubter", "dweller", "anomaly"], False),
            ("discovery", "revelation", 0.55, 0.04, "隠された代償", "主体は規則の向こう側で、共同体の幸福を一身に負わされた存在を発見する。", ["whisper", "ritual"], 0.4, 0.05, ["dweller", "hidden_one", "complicity"], True),
            ("justification", "local_optimization_limit", 0.65, 0.05, "必要悪の論理", "管理者たちは多数の幸福のための犠牲は合理的だと説き、主体の動揺を鎮めようとする。", ["discovery"], -0.2, 0.0, ["guardians", "dweller", "happiness"], False),
            ("confrontation", "crisis", 0.75, 0.05, "加担の自覚", "主体は自らの幸福がその犠牲の上に成り立ってきたと悟り、共同体と自分自身の両方に向き合う。", ["justification"], 0.3, 0.05, ["dweller", "complicity", "hidden_one"], True),
            ("choice", "costly_breakthrough", 0.88, 0.05, "留まるか去るか", "主体は幸福を手放してでも、犠牲の存在を見なかったことにしない選択をする。", ["confrontation"], -0.05, 0.0, ["dweller", "hidden_one", "happiness"], True),
            ("walking_away", "afterglow", 1.0, 0.05, "楽園を去る背中", "主体は楽園を後にし、共同体には問いだけが残る。答えの代わりに、次に違和感を拾う者の存在が暗示される。", ["choice"], -0.35, 0.0, ["dweller", "guardians", "anomaly"], False),
        ],
        "mechanisms": [
            ("idyll_dissonance", "cognitive_dissonance", "幸福と犠牲の不協和", "美しく描かれた幸福が犠牲の上に成り立つと知ることで、受け手は自分が享受している日常の前提を揺さぶられる。", "隠された代償の発見", "自らの日常への問い直し", "認知的不協和理論", 0.9, ["discovery", "confrontation"], ["happiness", "hidden_one"]),
            ("anomaly_curiosity", "curiosity_gap", "違和感の先への好奇心", "完璧な日常の中のただ一つの違和感が、受け手に説明されない空白への強い好奇心を生む。", "小さな違和感", "真実を知りたいという持続的な欲求", "情報ギャップ理論", 0.75, ["anomaly_found", "whisper"], ["anomaly"]),
            ("utilitarian_judgment", "moral_judgment", "多数の幸福と一人の犠牲の道徳判断", "必要悪の論理が提示されることで、受け手自身が功利と尊厳のどちらを選ぶかを問われる。", "必要悪の論理", "答えの出ない倫理的葛藤", "道徳的ジレンマ研究", 0.8, ["justification", "choice"], ["guardians"]),
            ("open_question", "zeigarnik", "答えのない終幕の未完了感", "主体が去るだけで共同体は変わらないまま終わることで、問いが受け手の中で反芻され続ける。", "楽園を去る背中", "鑑賞後も続く倫理的反芻", "ツァイガルニク効果", 0.7, ["walking_away"], ["dweller"]),
        ],
        "anchor": "何も疑わずに幸福だった自分が、知らないうちに誰かの犠牲に加担していたと知る主体の痛み",
        "drivers": ["完璧な日常に生じた一つの違和感", "規則の向こう側への好奇心", "幸福と犠牲のどちらを選ぶかという問い"],
        "emotions": ["安らぎ", "満足", "違和感", "好奇", "戦慄", "動揺", "罪悪感", "決意", "静かな余韻"],
        "template": (
            "{{CONTENT_DWELLER}}は{{PERFECT_HAPPINESS}}に満ちた共同体で、理由を問わない規則を守って暮らしている。{{FIRST_ANOMALY}}に気づいた主体は、"
            "{{ORDER_KEEPERS}}が隠してきた{{HIDDEN_SACRIFICE}}の存在を知る。必要悪の論理に揺さぶられながら{{UNKNOWING_COMPLICITY}}を自覚した主体は、"
            "幸福を手放して{{ETHICAL_CHOICE}}を選ぶ。"
        ),
        "slots": [
            ("CONTENT_DWELLER", "agent", "楽園で満ち足りて暮らす主体", ["冒頭で幸福を十分に味わわせる"], ("祝祭の街に暮らす少年", "好業績企業の社員", "成功法則の信奉者"), True),
            ("PERFECT_HAPPINESS", "goal", "完全な幸福", [], ("飢えも争いもない日々", "右肩上がりの成長", "すべてがうまくいく理論"), True),
            ("FIRST_ANOMALY", "symbol", "小さな違和感の徴", ["序盤の日常描写に一度だけ映り込ませる"], ("閉ざされた地下への扉", "説明のつかない離職率", "理論に合わない一つの事例"), True),
            ("ORDER_KEEPERS", "agent", "秩序を守る管理者たち", [], ("街の長老たち", "経営陣", "理論の提唱者たち"), True),
            ("HIDDEN_SACRIFICE", "agent", "見えない場所で犠牲を担う存在", [], ("地下に閉じ込められた子ども", "過酷な下請けの現場", "切り捨てられた少数事例"), True),
            ("UNKNOWING_COMPLICITY", "lack", "知らずに加担していた倫理的空白", [], ("祝祭を楽しんでいた自分", "成果だけを見ていた自分", "反例を無視していた自分"), True),
            ("ETHICAL_CHOICE", "reversal", "幸福を手放す倫理的選択", [], ("街を去る", "取引構造の是正を求める", "理論の限界を公表する"), True),
        ],
        "invariants": [
            "物語の前半三分の一は緊張を極めて低く保ち、楽園の幸福を十分に描き込む",
            "違和感の徴は日常描写の中に一度だけ自然に紛れ込ませる",
            "管理者を単純な悪として描かず、必要悪の論理に説得力を持たせる",
            "結末で共同体を変えきらず、問いを受け手に委ねる",
        ],
        "guide": {
            "manga": (
                "全30ページ想定。1〜8pは楽園の日常を明るいトーンと大きなコマで丁寧に描き、6pの背景に違和感の徴を一度だけ紛れ込ませる。10〜11pで徴への気づき、13〜14pで離脱者の示唆。"
                "めくり（15→16p）の右ページで隠された代償を黒ベタ中心の断ち切り大ゴマで描き、冒頭の明るさとの落差を最大化する。17〜21pは管理者の論理を静かな会話劇で、"
                "22〜25pで加担の自覚を主体の顔のアップの連続で描く。28〜30pは楽園を去る背中を引きの構図で描き、最後のコマに次に違和感を拾う者を小さく置く。"
            ),
            "marketing": (
                "顧客が当然視している業界の慣行の裏に隠れたコスト（環境負荷・労働・長期的損失）を楽園の綻びとして提示する。"
                "顧客を責めるのではなく、知ったうえで選び直せる選択肢として自社の提案を位置づけ、透明性そのものを価値とする。"
            ),
            "article": (
                "広く受け入れられている成功事例や理論の裏にある見えないコストを、小さな異常値から掘り起こす構成。"
                "必要悪の論理にも公平に紙幅を割き、結論では読者自身が判断するための問いを提示する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.12), (0.15, 0.1), (0.28, 0.15), (0.35, 0.3), (0.45, 0.38), (0.55, 0.78), (0.65, 0.6), (0.75, 0.9), (0.88, 0.86), (1.0, 0.5)],
            "release": [(0.0, 0.5), (0.2, 0.55), (0.35, 0.4), (0.55, 0.05), (0.75, 0.05), (0.9, 0.2), (1.0, 0.35)],
            "audience_knowledge": [(0.0, 0.05), (0.2, 0.1), (0.35, 0.25), (0.45, 0.35), (0.55, 0.75), (0.75, 0.9), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.05), (0.2, 0.05), (0.35, 0.15), (0.45, 0.25), (0.55, 0.7), (0.75, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.2), (0.25, 0.2), (0.35, 0.4), (0.55, 0.75), (0.75, 0.85), (0.88, 0.8), (1.0, 0.5)],
        },
    }
)

PATTERNS.append(
    {
        "num": 19,
        "slug": "savior_corruption",
        "title": "救済者の堕落",
        "origin": "film",
        "summary": (
            "圧政から共同体を解放した英雄が、秩序を守るために力を集め続けるうちに、かつて倒した圧政者と同じ姿へ変貌していく。"
            "かつての同志が英雄に刃を向ける最後の対決で、救済と支配の境界が露わになる。"
        ),
        "tags": ["英雄の堕落", "権力", "理想の変質", "かつての同志"],
        "lineage": ["英雄が竜になる物語（救済者の暴君化）", "革命の自己反転", "権力の腐敗の寓話"],
        "nodes": [
            ("hero", "agent", "protagonist", "共同体を解放した理想に燃える主体", "正義のために戦い勝利したが、守るための力を手放せなくなっていく。", 0.95, "IDEALISTIC_HERO"),
            ("old_tyrant", "agent", "former_antagonist", "かつて倒された圧政者", "物語の序盤で倒されるが、その論理は主体の中に受け継がれていく。", 0.6, "FALLEN_TYRANT"),
            ("comrade", "agent", "former_ally", "主体と共に戦ったかつての同志", "理想を共有していたがゆえに、変質した主体に最初に気づき、最後に刃を向ける。", 0.85, "FORMER_COMRADE"),
            ("people", "agent", "community", "解放されたはずの共同体", "最初は主体を称えるが、やがて新たな抑圧の下で沈黙する。", 0.6, "LIBERATED_PEOPLE"),
            ("order", "desire", "goal", "二度と圧政を生まない完全な秩序", "崇高な目的であるがゆえに、手段の正当化を際限なく許してしまう。", 0.85, "PERFECT_ORDER"),
            ("restraint", "lack", "inner_lack", "力の自制という歯止めの欠如", "勝利の経験が主体から奪った、自らを疑う能力。", 0.8, "MISSING_RESTRAINT"),
            ("crown", "symbol", "power_symbol", "受け継がれる権力の象徴", "圧政者から奪い取った権力の徴であり、身につけた者を同じ姿へ変えていく。", 0.75, "INHERITED_CROWN"),
        ],
        "edges": [
            ("old_tyrant", "people", "power_gradient", 0.9, 0.9, "圧政", 0.0),
            ("hero", "old_tyrant", "opposition", 0.95, -0.4, "解放の戦い", 0.1),
            ("hero", "comrade", "dependency", 0.8, 0.0, "理想を共有する同志の絆", 0.1),
            ("hero", "order", "pursuit", 0.9, 0.0, "完全な秩序の追求", 0.35),
            ("crown", "restraint", "embodiment", 0.75, 0.0, "権力の徴が歯止めの欠如を体現する", 0.4),
            ("comrade", "hero", "asymmetry", 0.5, 0.3, "同志だけが主体の変質を見抜く", 0.5),
            ("hero", "people", "power_gradient", 0.85, 0.85, "守るための支配", 0.55),
            ("hero", "comrade", "betrayal", 0.75, 0.6, "理想の裏切り", 0.7),
            ("comrade", "hero", "opposition", 0.9, -0.5, "かつての英雄への反逆", 0.85),
            ("restraint", "hero", "power_gradient", 0.75, 0.6, "歯止めの欠如が主体を支配へと駆り立てる", 0.5),
        ],
        "beats": [
            ("oppression_hook", "hook", 0.0, 0.05, "圧政下の共同体", "圧政に苦しむ共同体の中から、理想に燃える主体が立ち上がる。", [], 0.2, 0.0, ["hero", "old_tyrant", "people"], False),
            ("liberation_battle", "escalation", 0.15, 0.05, "解放の戦い", "主体と同志は命を賭けて圧政者に挑み、多くの犠牲を払いながら追い詰める。", ["oppression_hook"], 0.4, 0.05, ["hero", "comrade", "old_tyrant"], False),
            ("triumph", "false_victory", 0.25, 0.04, "勝利と戴冠", "圧政者は倒れ、主体は称賛の中で権力の徴を受け継ぐ。共同体は解放を祝う。", ["liberation_battle"], -0.5, 0.0, ["hero", "crown", "people", "old_tyrant"], True),
            ("first_compromise", "setup", 0.38, 0.06, "最初の妥協", "混乱を収めるため主体は一時的な強権を行使し、同志は小さな違和感を覚える。", ["triumph"], 0.05, 0.2, ["hero", "comrade", "order"], False),
            ("justified_means", "escalation", 0.5, 0.08, "目的による手段の正当化", "完全な秩序のためという理由で主体は反対者を排除し始める。読者だけが圧政者の論理との一致に気づく。", ["first_compromise"], 0.15, 0.25, ["hero", "people", "restraint"], False),
            ("mirror_moment", "revelation", 0.6, 0.04, "鏡像の瞬間", "主体はかつて圧政者が口にしたのと同じ言葉を自ら口にし、同志はその変質を確信する。", ["justified_means", "triumph"], 0.1, 0.1, ["hero", "comrade", "crown", "old_tyrant"], True),
            ("break_with_comrade", "crisis", 0.7, 0.05, "同志との決裂", "主体は諫言する同志を投獄し、理想を共有した最後の絆を自ら断ち切る。", ["mirror_moment"], 0.2, 0.05, ["hero", "comrade"], True),
            ("final_confrontation", "climax", 0.88, 0.05, "かつての英雄との対決", "脱出した同志が共同体と共に主体の前に立ち、主体はかつての圧政者の位置に立つ自分を突きつけられる。", ["break_with_comrade"], 0.25, -0.3, ["comrade", "hero", "people", "crown"], True),
            ("cycle_afterglow", "afterglow", 1.0, 0.05, "循環の問い", "主体は退き、権力の徴は次の者の手に渡る。共同体は同じ循環を繰り返さないための問いを抱える。", ["final_confrontation"], -0.5, 0.0, ["crown", "people", "comrade"], False),
        ],
        "mechanisms": [
            ("corruption_irony", "dramatic_irony", "変質を先に知る読者の不安", "読者は主体の言動がかつての圧政者と重なっていくのを主体自身より先に察知し、英雄への愛着と警戒の間で揺れる。", "目的による手段の正当化", "愛着の揺らぎと不安", "劇的アイロニー論", 0.85, ["justified_means", "mirror_moment"], ["hero", "old_tyrant"]),
            ("ideal_dissonance", "cognitive_dissonance", "英雄への共感と批判の不協和", "序盤で強く共感させた英雄を後半で批判させることで、受け手自身が正義の暴走に加担しうることを自覚させる。", "鏡像の瞬間", "自らの正義観への問い直し", "認知的不協和理論", 0.8, ["triumph", "mirror_moment"], ["hero"]),
            ("comrade_empathy", "empathy_transfer", "同志への共感の移行", "物語の後半で受け手の共感の重心が主体から同志へ移り、対決の場面に二重の痛みを与える。", "同志との決裂", "引き裂かれるような共感", "共感移入", 0.7, ["break_with_comrade", "final_confrontation"], ["comrade"]),
            ("cycle_judgment", "moral_judgment", "権力の循環への道徳的判断", "解放者が支配者になる循環を示すことで、受け手に権力そのものへの警戒心を残す。", "循環の問い", "権力構造への批判的視点", "道徳的直観理論", 0.7, ["cycle_afterglow"], ["crown", "people"]),
        ],
        "anchor": "誰よりも強く正義を信じていたからこそ、正義のための手段を疑えなくなっていく主体の悲劇",
        "drivers": ["英雄と圧政者の言動が重なっていく過程", "同志だけが気づく変質", "かつての同志との対決の予感"],
        "emotions": ["怒り", "熱狂", "歓喜", "違和感", "不安", "戦慄", "痛み", "やるせなさ"],
        "template": (
            "{{FALLEN_TYRANT}}の圧政から{{LIBERATED_PEOPLE}}を解放した{{IDEALISTIC_HERO}}は、{{INHERITED_CROWN}}を受け継いで{{PERFECT_ORDER}}を目指す。"
            "{{MISSING_RESTRAINT}}ゆえに目的が手段を正当化し始め、主体はかつての圧政者と同じ言葉を口にする。諫言する{{FORMER_COMRADE}}を退けた主体は、"
            "最後にかつての同志と共同体の前で{{MIRROR_CONFRONTATION}}に直面する。"
        ),
        "slots": [
            ("IDEALISTIC_HERO", "agent", "理想に燃える解放者", ["序盤で理想と勇気を本物として描く"], ("圧政に抗う若き反乱者", "業界の悪弊を壊した改革派経営者", "旧説を覆した研究者"), True),
            ("FALLEN_TYRANT", "agent", "かつて倒された圧政者", [], ("暴君と呼ばれた支配者", "独占を敷いた旧大手", "教条化した旧理論"), True),
            ("LIBERATED_PEOPLE", "agent", "解放されたはずの共同体", [], ("城下の民", "業界の中小事業者", "学界の若手"), True),
            ("INHERITED_CROWN", "symbol", "受け継がれる権力の象徴", ["圧政者から主体へ物理的に渡す"], ("奪った王冠", "買収した旧大手の本社", "定説の座"), True),
            ("PERFECT_ORDER", "goal", "完全な秩序", [], ("争いのない国", "健全な市場", "誤りのない理論体系"), True),
            ("MISSING_RESTRAINT", "lack", "力の自制の欠如", [], ("反対意見を聞かない心", "異論を排除する組織", "批判を受け付けない態度"), True),
            ("FORMER_COMRADE", "agent", "かつての同志", ["理想を最も深く共有した人物にする"], ("共に戦った幼なじみ", "創業時の右腕", "共同研究者"), True),
            ("MIRROR_CONFRONTATION", "reversal", "鏡像としての対決", [], ("同志の剣の前で冠を見つめる", "かつての自社と同じ手法で挑まれる", "自分の旧論文で反証される"), True),
        ],
        "invariants": [
            "主体の理想と勇気は序盤で本物として描き、堕落を偽善ではなく変質として示す",
            "解放の勝利を構造上の偽の勝利として配置し、そこから下降を始める",
            "主体が圧政者と同じ言葉を口にする鏡像の瞬間を必ず置く",
            "最後の対決者はかつての同志とし、理想の継承者と権力の継承者を対置する",
        ],
        "guide": {
            "manga": (
                "全40ページ想定。1〜4pで圧政下の共同体、5〜9pで解放の戦いを迫力ある大ゴマで描き、めくり（9→10p）で勝利と戴冠を見開きの祝祭として描く。"
                "12〜20pは統治の場面を少しずつ冷たいトーンへ移行させ、主体の目元の描写を段階的に圧政者に似せていく。22〜23pで圧政者の過去の台詞と同じフキダシを並置する鏡像の瞬間を作る。"
                "26〜27pで同志の投獄、めくり（33→34p）の右ページで同志と共同体が主体の前に立つ対決を見開きにする。39〜40pで権力の徴が次の手に渡る、冒頭と対になる構図。"
            ),
            "marketing": (
                "業界の悪弊を壊した新興勢力が、成長とともに同じ悪弊を繰り返す危険を率直に語り、自社がそれを防ぐ仕組み（透明性・第三者の点検）を持つことを信頼の根拠として示す。"
                "顧客をかつての同志として位置づけ、声を上げられる関係を約束する。"
            ),
            "article": (
                "旧来の権威を批判して台頭した理論や組織が、やがて同じ硬直に陥る事例を分析する。変質の兆候を時系列で示し、"
                "影響力を持った側が自らを疑い続けるための制度的な歯止めを提案する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.4), (0.15, 0.8), (0.25, 0.2), (0.38, 0.25), (0.5, 0.38), (0.6, 0.5), (0.7, 0.7), (0.8, 0.75), (0.88, 0.96), (1.0, 0.45)],
            "release": [(0.0, 0.05), (0.15, 0.1), (0.25, 0.9), (0.38, 0.6), (0.5, 0.35), (0.7, 0.1), (0.88, 0.05), (0.95, 0.4), (1.0, 0.5)],
            "audience_knowledge": [(0.0, 0.1), (0.25, 0.2), (0.4, 0.4), (0.55, 0.6), (0.7, 0.8), (0.88, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.25, 0.15), (0.4, 0.18), (0.55, 0.25), (0.7, 0.4), (0.88, 0.85), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.35), (0.15, 0.55), (0.25, 0.3), (0.5, 0.45), (0.7, 0.6), (0.88, 0.8), (1.0, 0.45)],
        },
    }
)

PATTERNS.append(
    {
        "num": 20,
        "slug": "homecoming_atonement",
        "title": "帰郷と贖罪",
        "origin": "drama",
        "summary": (
            "過去の過ちによって故郷を捨てた主体が、ある事情で帰郷し、拒絶と不信の中で償いの機会を探る。"
            "過去と同じ状況が再び訪れたとき、主体は今度こそ逃げずに代償を引き受け、赦しと再生を得る。"
        ),
        "tags": ["帰郷", "贖罪", "赦し", "再生"],
        "lineage": ["放蕩者の帰還の寓話", "過去の反復と贖罪の構造", "故郷への帰還劇"],
        "nodes": [
            ("returnee", "agent", "protagonist", "過去の過ちを背負って帰郷した主体", "かつての過ちから逃げて故郷を離れ、罪悪感を抱えたまま生きてきた。", 0.95, "GUILTY_RETURNEE"),
            ("wronged", "agent", "wronged_party", "主体の過ちで傷ついた者", "主体を許せずにいるが、心の奥では帰還を待っていた。", 0.8, "WRONGED_ONE"),
            ("hometown", "agent", "community", "主体を拒む故郷の共同体", "主体の過去を覚えており、帰郷を不信と冷淡さで迎える。", 0.6, "HOMETOWN_COMMUNITY"),
            ("ally", "agent", "bridge", "主体を受け入れる数少ない理解者", "過去を知りながら主体の変化を信じ、共同体との橋渡しとなる。", 0.5, None),
            ("forgiveness", "desire", "goal", "赦しと居場所の回復", "主体が本当に求めているが、口にすることもできない願い。", 0.85, "SOUGHT_FORGIVENESS"),
            ("guilt", "lack", "inner_lack", "逃げたことへの消えない罪悪感", "過ちそのものより、責任から逃げたことが主体を苛む欠落。", 0.85, "UNRESOLVED_GUILT"),
            ("old_place", "symbol", "memory_site", "過ちの記憶が刻まれた場所", "過去の出来事が起きた場所であり、反復の舞台となる。", 0.7, "MEMORY_SITE"),
        ],
        "edges": [
            ("returnee", "wronged", "dependency", 0.6, -0.3, "かつての絆", 0.0),
            ("returnee", "wronged", "betrayal", 0.8, 0.3, "過去の過ちと逃避", 0.02),
            ("hometown", "returnee", "opposition", 0.7, 0.6, "帰郷者への不信", 0.2),
            ("hometown", "returnee", "power_gradient", 0.6, 0.6, "共同体の記憶による裁き", 0.25),
            ("wronged", "returnee", "opposition", 0.75, 0.4, "許せない想い", 0.3),
            ("old_place", "guilt", "embodiment", 0.75, 0.0, "記憶の場所が罪悪感を体現する", 0.3),
            ("returnee", "forgiveness", "pursuit", 0.8, 0.0, "償いの機会の探求", 0.35),
            ("ally", "hometown", "asymmetry", 0.45, 0.3, "理解者だけが主体の変化を知る", 0.45),
            ("guilt", "returnee", "power_gradient", 0.75, 0.6, "消えない罪悪感が主体を縛り続ける", 0.05),
        ],
        "beats": [
            ("past_hook", "hook", 0.0, 0.05, "過去の過ち", "主体が過ちを犯し、責任から逃げて故郷を去った過去の一場面が断片的に示される。", [], 0.2, 0.0, ["returnee", "wronged", "old_place"], False),
            ("return", "premise", 0.12, 0.05, "帰郷", "避けられない事情で主体は故郷に戻り、変わらない風景と変わってしまった人々に迎えられる。", ["past_hook"], -0.1, 0.1, ["returnee", "hometown", "old_place"], False),
            ("rejection", "escalation", 0.28, 0.05, "拒絶", "傷ついた者は主体を拒絶し、共同体の冷淡さが主体の罪悪感を抉る。", ["return"], 0.25, 0.05, ["wronged", "hometown", "returnee", "guilt"], True),
            ("quiet_amends", "setup", 0.45, 0.08, "静かな償い", "主体は言葉ではなく行動で小さな償いを重ね、理解者を通じて共同体の一部が変化に気づき始める。", ["rejection"], -0.2, 0.15, ["returnee", "ally", "hometown"], False),
            ("partial_truth", "revelation", 0.55, 0.05, "語られなかった過去", "過去の出来事には主体だけが背負った語られない事情があったことが、読者にだけ明かされる。", ["quiet_amends", "past_hook"], 0.0, 0.2, ["returnee", "guilt", "old_place"], False),
            ("history_repeats", "crisis", 0.75, 0.05, "過去の反復", "記憶の場所でかつてと同じ危機が再び起こり、主体は再び逃げるか留まるかの選択を迫られる。", ["partial_truth"], 0.45, 0.05, ["returnee", "old_place", "wronged"], True),
            ("atonement_act", "costly_breakthrough", 0.85, 0.05, "逃げない選択", "主体は今度こそ逃げずに危機に立ち向かい、傷ついた者を救うために大きな代償を引き受ける。", ["history_repeats"], -0.1, 0.1, ["returnee", "wronged", "guilt"], True),
            ("forgiveness_moment", "payoff", 0.93, 0.04, "赦しの言葉", "傷ついた者が主体の真実と変化を知り、長年言えなかった赦しの言葉を口にする。", ["atonement_act", "partial_truth"], -0.45, -0.4, ["wronged", "returnee", "forgiveness"], True),
            ("home_afterglow", "afterglow", 1.0, 0.04, "帰るべき場所", "主体は故郷に留まることを選び、記憶の場所は罪の象徴から新しい始まりの場所へと意味を変える。", ["forgiveness_moment"], -0.2, 0.0, ["returnee", "hometown", "old_place"], False),
        ],
        "mechanisms": [
            ("guilt_projection", "identity_projection", "逃げた過去への自己投影", "誰もが持つ逃げてしまった記憶が主体の罪悪感と重なり、受け手は自分の過去と向き合わされる。", "過去の過ちと帰郷", "主体への深い共感と自己省察", "同一化理論", 0.8, ["past_hook", "return"], ["returnee", "guilt"]),
            ("hidden_truth_irony", "dramatic_irony", "語られない事情を知る読者", "読者は主体が背負った語られない事情を知っているが共同体は知らず、拒絶の場面が痛ましく映る。", "語られなかった過去", "誤解が解けてほしいという願い", "劇的アイロニー論", 0.75, ["partial_truth", "atonement_act"], ["hometown", "wronged"]),
            ("second_chance", "prediction_error", "反復される危機での異なる選択", "過去と同じ危機の再来で受け手は再び逃げる主体を予期するが、異なる選択がその予測を裏切り強い感動を生む。", "過去の反復と逃げない選択", "予期の裏切りによる感動", "予測符号化理論", 0.8, ["history_repeats", "atonement_act"], ["old_place"]),
            ("forgiveness_catharsis", "catharsis", "赦しによる長い緊張の解放", "長年の罪悪感と拒絶が赦しの言葉によって一挙に解かれ、涙を伴う解放が生じる。", "赦しの言葉", "深い安堵と温かさ", "カタルシス論", 0.85, ["forgiveness_moment", "home_afterglow"], ["forgiveness"]),
        ],
        "anchor": "過ちそのものより、責任から逃げてしまった自分をずっと許せずにいる主体の罪悪感",
        "drivers": ["拒絶の中で償いの機会を探る緊張", "読者にだけ明かされる過去の事情", "同じ危機が再び訪れたときの選択"],
        "emotions": ["悔恨", "郷愁", "痛み", "孤独", "かすかな希望", "恐怖", "覚悟", "赦し", "温かさ"],
        "template": (
            "かつて{{UNRESOLVED_GUILT}}を残して故郷を去った{{GUILTY_RETURNEE}}は、避けられない事情で帰郷し、{{WRONGED_ONE}}と{{HOMETOWN_COMMUNITY}}の拒絶に直面する。"
            "静かな償いを重ねる中で、読者にだけ{{UNTOLD_CIRCUMSTANCE}}が明かされる。{{MEMORY_SITE}}で過去と同じ危機が再び訪れたとき、"
            "主体は今度こそ逃げずに代償を引き受け、{{SOUGHT_FORGIVENESS}}を得る。"
        ),
        "slots": [
            ("GUILTY_RETURNEE", "agent", "過去の過ちを背負って帰郷した主体", ["冒頭で過去を断片的にだけ見せる"], ("都会から戻った元漁師の青年", "撤退した市場に再参入する企業", "かつて誤りを広めた解説者"), True),
            ("UNRESOLVED_GUILT", "lack", "逃げたことへの消えない罪悪感", [], ("事故の夜に逃げた記憶", "顧客を見捨てた撤退", "訂正しなかった誤情報"), True),
            ("WRONGED_ONE", "agent", "主体の過ちで傷ついた者", [], ("取り残された幼なじみ", "見捨てられた旧顧客", "誤情報で損をした読者"), True),
            ("HOMETOWN_COMMUNITY", "agent", "主体を拒む故郷の共同体", [], ("港町の人々", "かつての取引先の業界", "かつての読者層"), True),
            ("UNTOLD_CIRCUMSTANCE", "irony", "語られなかった過去の事情", ["読者にだけ先に明かす"], ("誰かを庇って罪を被っていた事実", "撤退の裏にあった不可抗力", "誤りの背景にあった情報不足"), True),
            ("MEMORY_SITE", "symbol", "過ちの記憶が刻まれた場所", [], ("事故のあった防波堤", "撤退した旧店舗", "過去の誤った記事"), True),
            ("SOUGHT_FORGIVENESS", "goal", "赦しと居場所の回復", [], ("幼なじみの赦しの言葉", "旧顧客との再契約", "読者からの信頼回復"), True),
        ],
        "invariants": [
            "過去の過ちは冒頭で断片的にのみ示し、全容は物語の後半まで明かさない",
            "償いは言葉ではなく行動で示し、帰郷直後の拒絶を十分に描く",
            "過去と同じ危機を同じ場所で反復させ、主体に異なる選択をさせる",
            "赦しは主体の代償の引き受けと真実の開示の両方を経て初めて与える",
        ],
        "guide": {
            "manga": (
                "全36ページ想定。1〜3pで過去の過ちを粗い線と断片的なコマで描き、4〜5pの見開きで変わらない故郷の風景に帰ってくる主体を小さく描く。"
                "7〜10pで拒絶を左ページ終端の冷たい表情で引き、12〜18pで静かな償いを無言のコマ多めで重ねる。20〜21pで語られなかった事情を回想として読者にだけ見せる。"
                "めくり（25→26p）の右ページで過去と同じ危機を冒頭と同じ構図の大ゴマで反復し、27〜29pで逃げない選択を描く。31〜33pで赦しの言葉、34〜36pで意味の変わった記憶の場所。"
            ),
            "marketing": (
                "一度失敗や撤退で顧客の信頼を失ったブランドが帰還する物語として設計し、過去の過ちを隠さず認めることから始める。"
                "言葉ではなく行動（具体的な改善と補償）で償い、同じ危機が起きたときに今度は逃げない仕組みを約束として示す。"
            ),
            "article": (
                "過去に誤った主張や判断をした経験を正直に開示し、その背景と反省を分析する。同じ状況が再来したときにどう判断を変えたかを具体的に示し、"
                "読者が自分の過去の判断と向き合うための手順を提案する。"
            ),
        },
        "wave": {
            "tension": [(0.0, 0.55), (0.12, 0.45), (0.28, 0.72), (0.4, 0.5), (0.52, 0.38), (0.62, 0.48), (0.75, 0.92), (0.85, 0.8), (0.93, 0.35), (1.0, 0.12)],
            "release": [(0.0, 0.05), (0.12, 0.15), (0.28, 0.05), (0.45, 0.35), (0.55, 0.5), (0.75, 0.05), (0.9, 0.7), (1.0, 0.95)],
            "audience_knowledge": [(0.0, 0.15), (0.15, 0.3), (0.35, 0.45), (0.55, 0.65), (0.72, 0.75), (0.9, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.15), (0.15, 0.15), (0.35, 0.2), (0.55, 0.3), (0.72, 0.35), (0.9, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.4), (0.15, 0.45), (0.3, 0.6), (0.5, 0.45), (0.75, 0.8), (0.9, 0.55), (1.0, 0.3)],
        },
    }
)


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
