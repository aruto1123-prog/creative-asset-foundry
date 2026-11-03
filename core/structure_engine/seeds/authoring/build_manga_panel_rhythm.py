"""漫画特有の演出リズム骨格（manga_panel_rhythm）20パターンの著作スクリプト。

各パターンは手書き設計したコンパクトな定義（タプル列）から ``UniversalStructure`` を組み立てる。
波形は5チャネルすべてを 6〜12 個のキーフレームで設計し、
``EmotionalInformationWave.from_keyframes``（PCHIP単調補間）で 21 点（20分割）に展開する。

ページ左右の慣行: page 0 を右ページとし、偶数インデックス=右ページ・奇数=左ページとする
（見開きは「右ページ開始・左ページ終端」）。めくりは「奇数（左）→偶数（右）」の境界で起きるため、
めくり直後のリビール（page_turn_reveals）と見開き大ゴマの右ページ側（spread_pages）は偶数インデックスに置く。

再生成: ``python -m core.structure_engine.seeds.authoring.build_manga_panel_rhythm``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from core.structure_engine.schemas import WAVE_CHANNELS, EdgeKind, EmotionalInformationWave, UniversalStructure

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "manga_panel_rhythm_patterns.json"
SEED_CREATED_AT = "2026-11-03T00:00:00+00:00"
CATEGORY = "manga_panel_rhythm"
FILE_DESCRIPTION = (
    "漫画特有の演出リズム骨格20選。第1話冒頭の掴み、見開きめくりのリビール、静寂から爆発への緩急、"
    "バトル・心理戦の駆け引き、会話劇の切り返し、回想、ギャグ、感情決壊、追跡、新キャラ登場、ホラー、"
    "時間引き延ばし、群像、決め台詞、日常の間、謎解き、クリフハンガー、告白、覚醒を、"
    "ページ単位のコマ数・強調ページ・めくり位置（rhythm_profile）と21点の感情・情報波形で記述する。"
)
ID_PATTERN = re.compile(r"^mr_(\d{2})_[a-z0-9_]+$")
CORRELATION_LIMIT = 0.97

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


def validate_patterns(specs: list[dict[str, Any]], structures: list[UniversalStructure]) -> list[str]:
    """SPEC §10 の品質基準・リズム整合・波形多様性を検査し、問題の一覧を返す。"""
    problems: list[str] = []
    if len(structures) != 20:
        problems.append(f"パターン数が20ではありません: {len(structures)}")
    ids = [s.id for s in structures]
    if len(set(ids)) != len(ids):
        problems.append("IDが重複しています")
    numbers = []
    for s in structures:
        match = ID_PATTERN.match(s.id)
        if not match:
            problems.append(f"{s.id}: ID形式が mr_XX_<slug> ではありません")
        else:
            numbers.append(int(match.group(1)))
    if sorted(numbers) != list(range(1, 21)):
        problems.append(f"ID番号が01〜20の連番ではありません: {sorted(numbers)}")
    used_edge_kinds: set[str] = set()
    for spec, s in zip(specs, structures):
        sid = s.id
        g = s.tension_graph
        if not 4 <= len(g.nodes) <= 8:
            problems.append(f"{sid}: ノード数 {len(g.nodes)} が4〜8の範囲外")
        if not 5 <= len(g.edges) <= 12:
            problems.append(f"{sid}: エッジ数 {len(g.edges)} が5〜12の範囲外")
        kinds = {e.kind.value for e in g.edges}
        used_edge_kinds |= kinds
        if len(kinds) < 4:
            problems.append(f"{sid}: エッジ種別が {len(kinds)} 種しかありません（4種以上）")
        if not any(e.activation_t is not None for e in g.edges):
            problems.append(f"{sid}: activation_t を持つエッジがありません")
        if not any(abs(e.gradient) > 0 for e in g.edges):
            problems.append(f"{sid}: gradient を持つエッジがありません")
        beats = s.beats.beats
        if not 6 <= len(beats) <= 10:
            problems.append(f"{sid}: ビート数 {len(beats)} が6〜10の範囲外")
        turning = s.beats.turning_points()
        if len(turning) < 2:
            problems.append(f"{sid}: 転換点が2未満")
        if not any(b.causes for b in beats):
            problems.append(f"{sid}: 因果 causes が空です")
        tension = s.wave.channel("tension")
        for beat in turning:
            index = int(round(beat.t * (tension.size - 1)))
            if abs(index / (tension.size - 1) - beat.t) > 1e-9:
                problems.append(f"{sid}: 転換点 {beat.id} の t={beat.t} が波形格子点上にありません")
            elif not _is_local_extremum(tension, index):
                problems.append(f"{sid}: 転換点 {beat.id} (t={beat.t}) が緊張度の極値と一致しません")
        peak_t = s.wave.peak_t("tension")
        if not any(abs(b.t - peak_t) <= 0.05 + 1e-9 for b in turning):
            problems.append(f"{sid}: 緊張度ピーク t={peak_t} の近傍に転換点ビートがありません")
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
        for channel in WAVE_CHANNELS:
            count = len(spec["waves"].get(channel, []))
            if not 6 <= count <= 12:
                problems.append(f"{sid}: {channel} のキーフレーム数 {count} が6〜12の範囲外")
        profile = s.rhythm_profile
        if profile is None:
            problems.append(f"{sid}: rhythm_profile がありません")
            continue
        if not 3 <= profile.page_count <= 24:
            problems.append(f"{sid}: page_count {profile.page_count} が3〜24の範囲外")
        if any(i % 2 for i in profile.page_turn_reveals):
            problems.append(f"{sid}: page_turn_reveals に奇数（左ページ）インデックスがあります")
        if any(i % 2 for i in profile.spread_pages):
            problems.append(f"{sid}: spread_pages に奇数インデックスがあります")
        if len(profile.pacing_notes) < profile.page_count:
            problems.append(f"{sid}: pacing_notes がページ数より少ない")
        denominator = max(1, profile.page_count - 1)
        tolerance = max(0.06, 0.5 / denominator + 0.02)
        accent_pages = set(profile.emphasis_pages) | set(profile.page_turn_reveals)
        accent_pages |= set(profile.spread_pages) | {i + 1 for i in profile.spread_pages}
        if not accent_pages:
            problems.append(f"{sid}: 強調・めくり・見開きページが1つもありません")
        elif min(abs(i / denominator - peak_t) for i in accent_pages) > tolerance:
            problems.append(f"{sid}: 緊張度ピーク t={peak_t} と強調/めくりページの位置が整合しません")
        grid = np.asarray(s.wave.t)
        mean_tension = float(tension.mean())
        for page in profile.page_turn_reveals:
            position = page / denominator
            indices = [i for i in range(tension.size) if abs(grid[i] - position) <= tolerance]
            peaks = [i for i in indices if _is_local_maximum(tension, i) and tension[i] > mean_tension]
            if not peaks:
                problems.append(f"{sid}: めくりページ {page} の近傍に平均を上回る緊張度のピークがありません")
    missing_kinds = {k.value for k in EdgeKind} - used_edge_kinds
    if missing_kinds:
        problems.append(f"全体で未使用のエッジ種別: {sorted(missing_kinds)}")
    tensions = [s.wave.channel("tension") for s in structures]
    for i in range(len(tensions)):
        for j in range(i + 1, len(tensions)):
            r = float(np.corrcoef(tensions[i], tensions[j])[0, 1])
            if r >= CORRELATION_LIMIT:
                problems.append(f"緊張度波形の相関が高すぎます: {ids[i]} × {ids[j]} r={r:.3f}")
    return problems


def write_json(structures: list[UniversalStructure], path: Path = OUTPUT_PATH) -> Path:
    document = {
        "schema_version": "1.0",
        "category": CATEGORY,
        "description": FILE_DESCRIPTION,
        "patterns": [s.model_dump(mode="json") for s in structures],
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    structures = build_all()
    problems = validate_patterns(PATTERNS, structures)
    if problems:
        for problem in problems:
            print(f"[NG] {problem}", file=sys.stderr)
        return 1
    path = write_json(structures)
    print(f"[OK] {len(structures)} パターンを書き出しました: {path}")
    return 0


# ---------------------------------------------------------------------------
# 01: 第1話冒頭3ページの掴み（衝撃→日常→引き）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_01_opening_three_page_grab",
        "title": "第1話冒頭3ページの掴み（衝撃→日常→引き）",
        "summary": "文脈抜きの衝撃一枚絵で問いを先に植え、日常の多コマで基準線を引き、谷底に伏線を一コマだけ置いて、めくり直後の大ゴマで衝撃と日常を一本の因果に結ぶ導入の骨格。",
        "tags": ["第1話", "導入", "掴み", "めくり", "伏線"],
        "lineage": ["プロローグ先行型の導入", "イン・メディアス・レス（事件の途中から始める技法）", "週刊連載第1話の3ページ導入慣行"],
        "waves": {
            "tension": [(0.0, 0.88), (0.1, 0.62), (0.25, 0.35), (0.4, 0.25), (0.6, 0.5), (0.65, 0.58), (0.75, 0.86), (0.85, 0.8), (0.95, 0.9), (1.0, 0.96)],
            "release": [(0.0, 0.05), (0.15, 0.2), (0.3, 0.45), (0.45, 0.55), (0.6, 0.3), (0.8, 0.1), (1.0, 0.05)],
            "audience_knowledge": [(0.0, 0.05), (0.2, 0.25), (0.45, 0.42), (0.7, 0.5), (0.75, 0.82), (0.9, 0.86), (1.0, 0.9)],
            "character_knowledge": [(0.0, 0.15), (0.2, 0.3), (0.45, 0.38), (0.7, 0.4), (0.75, 0.7), (0.9, 0.78), (1.0, 0.82)],
            "cognitive_load": [(0.0, 0.3), (0.2, 0.55), (0.45, 0.6), (0.6, 0.5), (0.75, 0.45), (0.85, 0.35), (1.0, 0.4)],
        },
        "nodes": [
            ("viewpoint", "agent", "viewpoint_character", "日常に属する視点人物", "読者が最初に同一化する主体。冒頭の衝撃とはまだ無関係に見える位置に置かれる", 0.9, "VIEWPOINT_CHARACTER"),
            ("anomaly", "agent", "intruding_force", "日常の外側から侵入する異物", "冒頭の一枚絵で断片だけが示され、正体を伏せたまま日常へ接近する力", 0.85, "SHOCK_IMAGE"),
            ("reader_question", "desire", "reader_curiosity", "冒頭の衝撃は何だったのかという読者の問い", "一枚絵が植え付ける問い。日常描写のあいだ解消されず牽引力として働く", 0.8, "OPENING_QUESTION"),
            ("ordinary_rule", "symbol", "ordinary_order", "繰り返される日常の規則", "多コマで描かれる生活の手触り。後に破られるための基準線", 0.6, "ORDINARY_RULE"),
            ("hidden_link", "lack", "missing_link", "衝撃と日常を結ぶ伏せられた因果", "読者にも視点人物にも欠けている情報。めくり直後に一挙に補完される", 0.75, "LINKING_REVEAL"),
            ("foreshadow", "symbol", "planted_detail", "日常に紛れた一つの違和", "谷底のコマに一つだけ置かれる伏線。再読時に意味が反転する", 0.55, "HIDDEN_FORESHADOW"),
        ],
        "edges": [
            ("anomaly", "viewpoint", "opposition", 0.8, 0.0, "異物が日常の主体を脅かす", 0.75),
            ("anomaly", "ordinary_rule", "power_gradient", 0.7, 0.6, "侵入する力が日常の規則を上回る", 0.75),
            ("viewpoint", "ordinary_rule", "dependency", 0.7, 0.0, "視点人物は日常の規則に寄りかかっている", 0.2),
            ("reader_question", "anomaly", "pursuit", 0.85, 0.0, "読者の問いが異物の正体を追う", 0.0),
            ("hidden_link", "reader_question", "asymmetry", 0.6, -0.5, "伏せられた因果が読者の認知を上回る", 0.05),
            ("foreshadow", "hidden_link", "embodiment", 0.5, 0.0, "違和の一コマが因果を体現する", 0.4),
            ("viewpoint", "hidden_link", "asymmetry", 0.5, -0.3, "視点人物は因果を知らない", 0.6),
        ],
        "beats": [
            ("b_shock", "hook", 0.0, 0.08, "断ち切りの衝撃一枚絵", "結末または異常の断片を文脈抜きで提示し、問いを先に植える", [], 0.6, -0.3, ["anomaly", "reader_question"], False),
            ("b_daily", "premise", 0.2, 0.12, "日常の規則の提示", "視点人物の生活と小さな欠落を多コマの会話で手早く確立する", ["b_shock"], -0.4, 0.1, ["viewpoint", "ordinary_rule"], False),
            ("b_seed_unease", "setup", 0.4, 0.06, "静けさの底に置く違和", "緊張が最も下がった地点で伏線を一コマだけ紛れ込ませる", ["b_daily"], -0.1, 0.05, ["foreshadow", "ordinary_rule"], True),
            ("b_intrusion", "escalation", 0.6, 0.06, "違和の接近", "伏線が視点人物の手の届く範囲へ寄り、左ページ最終コマが引きになる", ["b_seed_unease", "b_shock"], 0.3, 0.1, ["viewpoint", "anomaly"], False),
            ("b_link_reveal", "revelation", 0.75, 0.06, "めくり直後の接続の開示", "右ページ冒頭の大ゴマで冒頭の衝撃と日常が一本の因果で結ばれる", ["b_intrusion"], 0.4, 0.5, ["hidden_link", "anomaly", "viewpoint"], True),
            ("b_refusal", "crisis", 0.85, 0.05, "視点人物の拒絶", "事態を受け入れられない一瞬の反応ゴマで呼吸を作る", ["b_link_reveal"], -0.1, 0.0, ["viewpoint"], False),
            ("b_pull", "hook", 1.0, 0.04, "後戻りできない一線の引き", "ページ末尾で視点人物が境界を越えざるを得ない状況を示し次ページへ牽引する", ["b_refusal", "b_link_reveal"], 0.3, 0.1, ["viewpoint", "anomaly"], True),
        ],
        "mechanisms": [
            ("m_question_first", "curiosity_gap", "問いの先行提示", "答えのない断片を最初に見せ、解消への欲求を日常描写のあいだ持続させる", "文脈を欠いた衝撃の一枚絵", "読者が答えを探して日常のコマを能動的に読む", "情報ギャップ理論", 0.85, ["b_shock", "b_link_reveal"], ["reader_question"]),
            ("m_baseline", "contrast_effect", "日常の基準線", "平穏な規則を先に見せることで、後の侵入の落差を最大化する", "多コマの穏やかな日常", "侵入の瞬間の衝撃が相対的に増幅される", "対比効果（コントラスト効果）", 0.7, ["b_daily", "b_link_reveal"], ["ordinary_rule"]),
            ("m_turn_anticipation", "anticipation_reward", "めくりの予期と報酬", "左ページ末尾の引きが予期を生み、めくり直後の開示が報酬として働く", "左ページ最終コマの引き", "ページをめくる行為そのものが快の予期になる", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.75, ["b_intrusion", "b_link_reveal"], ["hidden_link"]),
            ("m_anchor", "empathy_transfer", "日常への同一化", "読者がまず日常側の主体へ同一化し、侵入を自分事として受け取る", "視点人物の小さな欠落と生活感", "異物の侵入が読者自身の日常への脅威として感じられる", "物語移入理論", 0.65, ["b_daily", "b_refusal"], ["viewpoint"]),
        ],
        "anchor": "冒頭の衝撃とは無関係に見える視点人物の、ありふれた日常と小さな欠落",
        "drivers": ["冒頭の一枚絵が残す未回答の問い", "日常に一コマだけ紛れた違和", "左ページ末尾の引きとめくりの予期"],
        "emotions": ["驚愕", "好奇", "安堵", "不穏", "戦慄", "期待"],
        "template": "第1ページ（右）は{{SHOCK_IMAGE}}を断ち切りの一枚絵で文脈抜きに提示し、読者に{{OPENING_QUESTION}}という問いを植え付ける。第2ページ（左）は{{VIEWPOINT_CHARACTER}}が生きる{{ORDINARY_RULE}}を多コマの会話で手早く確立し、緊張の谷底に{{HIDDEN_FORESHADOW}}を一コマだけ紛れ込ませ、最終コマに{{TURN_PULL}}を置く。めくった第3ページ（右）冒頭の大ゴマで{{LINKING_REVEAL}}を開示し、冒頭の衝撃と日常を一本の因果で結んで次ページへ牽引する。",
        "slots": [
            ("SHOCK_IMAGE", "文脈抜きで提示される衝撃の断片（結末・異常・破局の一瞬）", "symbol", ["説明文を添えない", "正体の核心は伏せる"], ("崩れ落ちる建物の前に立つ後ろ姿", "失敗した顧客の一場面を切り取った冒頭写真", "常識が崩れた瞬間を描く冒頭の一文")),
            ("OPENING_QUESTION", "一枚絵が読者に残す未回答の問い", "goal", ["日常パートでは答えない"], ("この破局は誰に何が起きた結果か", "なぜこの人は失敗したのか", "なぜ通説はこの事例で破綻したのか")),
            ("VIEWPOINT_CHARACTER", "読者が最初に同一化する日常側の主体", "agent", ["冒頭の衝撃とは無関係に見えること"], ("寝坊して駆け出す学生", "日々の業務に追われる担当者", "通説を信じて実践してきた読者像")),
            ("ORDINARY_RULE", "繰り返される生活・業務・思考の規則", "setting", ["多コマで手早く描く", "後で破られる基準線にする"], ("毎朝同じ道を通る習慣", "いつも通りの集客手順", "業界で当然とされる手法")),
            ("HIDDEN_FORESHADOW", "日常に紛れた一つの違和（伏線）", "symbol", ["一コマに限定する", "再読で意味が反転する"], ("背景の掲示板に映る見慣れない紋様", "数値の小さな異常値", "事例中の説明できない例外")),
            ("TURN_PULL", "左ページ最終コマに置く引き", "conflict", ["左ページの最後に置く", "答えは書かない"], ("足元に落ちてくる影", "鳴り止まない通知", "しかし、この例外には続きがあった")),
            ("LINKING_REVEAL", "めくり直後に開示される、衝撃と日常を結ぶ因果", "reversal", ["右ページ冒頭の大ゴマ", "冒頭の一枚絵と同じ構図を反復する"], ("冒頭の後ろ姿が視点人物本人だと判明する", "冒頭の失敗事例が自分の明日の姿だと気づく", "例外こそが新しい法則の入口だと示す")),
        ],
        "invariants": [
            "第1ページは文脈抜きの一枚絵（1コマ・断ち切り）で始める",
            "緊張度の谷は日常パートの中盤に置き、そこに伏線を一コマだけ置く",
            "引きは左ページ最終コマ、開示はめくり直後の右ページ冒頭の大ゴマに置く",
            "視線誘導は右上から左下へのZ字を崩さず、開示の大ゴマへ最短で流し込む",
            "冒頭の問いへの答えは開示の大ゴマまで一切与えない",
        ],
        "guide": {
            "manga": "3ページ構成。p1（右）: 1コマの断ち切り大ゴマ、ELSまたはFSで衝撃の断片、アオリで威圧感、説明セリフなし。p2（左）: 6コマ、MS中心の会話カットバックで日常を確立し、4コマ目付近に背景へ溶け込ませたCUの違和を1コマ、最終6コマ目は視点人物の背後に迫る影をアイレベル・望遠85mmで引きにする。めくって p3（右）: 上段に横長の大ゴマ（冒頭と同構図をFSで反復）で接続を開示し、続く小コマ2つで拒絶の表情CUと一線を越える足元ECUを置いてページ末尾で次へ引く。",
            "marketing": "LP冒頭のファーストビューに文脈抜きの衝撃的な結果（失敗や成功の断片）を置き、直後に読者の日常業務を具体的に描写して同一化させる。日常描写の中に一つだけ小さな異常値を紛れ込ませ、スクロールの区切り（めくりに相当）直後に、冒頭の結果がその異常値の延長にあると接続して問題提起へ進む。",
            "article": "記事冒頭で結論や破綻事例の一場面だけを提示し、読者が普段従っている常識を淡々と記述する。常識の記述の中に説明できない例外を一つ置き、見出しの切れ目の直後で冒頭事例とその例外を結び付け、本論の問いを確定させる。",
        },
        "rhythm": {
            "page_count": 3,
            "panels_per_page": [1, 6, 3],
            "emphasis_pages": [0, 2],
            "page_turn_reveals": [2],
            "spread_pages": [],
            "silent_panels_ratio": 0.2,
            "pacing_notes": [
                "p1（右）: 断ち切り1コマの衝撃一枚絵。セリフなしで問いだけを植える",
                "p2（左）: 6コマの日常。会話の切り返しで規則を確立し、谷底の4コマ目に違和を1コマ、最終コマで引く",
                "p3（右・めくり直後）: 冒頭と同構図の大ゴマで接続を開示し、小コマ2つで拒絶と一線を描いて次へ牽引する",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 02: 見開きめくり大ゴマ・リビール
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_02_spread_turn_reveal",
        "title": "見開きめくり大ゴマ・リビール",
        "summary": "人物の反応だけを先行させて対象を伏せ、細いコマの断片で読者の推測を飽和させた後、左ページ末尾の無音コマで息を止め、めくり直後の見開き大ゴマで全貌を一挙に開示する骨格。",
        "tags": ["見開き", "めくり", "リビール", "大ゴマ", "スケール"],
        "lineage": ["反応先行のショット構成（リアクション・ファースト）", "見開き大ゴマによるスケール提示の慣行", "サプライズ・リビール"],
        "waves": {
            "tension": [(0.0, 0.3), (0.15, 0.38), (0.3, 0.46), (0.45, 0.58), (0.55, 0.7), (0.6, 0.6), (0.65, 0.98), (0.75, 0.88), (0.85, 0.7), (1.0, 0.45)],
            "release": [(0.0, 0.1), (0.3, 0.12), (0.55, 0.06), (0.6, 0.04), (0.65, 0.3), (0.8, 0.6), (1.0, 0.75)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.22), (0.4, 0.3), (0.55, 0.36), (0.6, 0.38), (0.65, 0.88), (0.85, 0.93), (1.0, 0.97)],
            "character_knowledge": [(0.0, 0.15), (0.2, 0.3), (0.4, 0.45), (0.55, 0.55), (0.6, 0.6), (0.65, 0.9), (0.85, 0.93), (1.0, 0.97)],
            "cognitive_load": [(0.0, 0.4), (0.3, 0.52), (0.55, 0.66), (0.6, 0.5), (0.65, 0.22), (0.8, 0.3), (1.0, 0.35)],
        },
        "nodes": [
            ("witness", "agent", "reacting_viewpoint", "対象を見上げる視点人物", "読者より先に対象を目撃し、その反応だけが提示される主体", 0.8, "WITNESS"),
            ("revealed_object", "symbol", "spectacle", "見開きで全貌を現す対象", "部分・影・音だけで小出しにされ、めくり直後に全景で開示される決め絵", 0.95, "SPECTACLE"),
            ("reader_guess", "desire", "reader_prediction", "全貌を推測しようとする読者の欲求", "断片から予測を積み上げ、予測の飽和点でめくりへ向かう", 0.8, "READER_GUESS"),
            ("withheld_view", "lack", "withheld_view", "読者に見せていない視界", "人物は見ているが読者には見えていない領域。めくりで一挙に補完される", 0.85, "WITHHELD_VIEW"),
            ("frame_limit", "symbol", "panel_frame", "小さく区切られたコマ枠", "対象を収めきれない細いコマの連続。窮屈さが解放の落差を生む", 0.5, "FRAGMENT_SEQUENCE"),
        ],
        "edges": [
            ("witness", "withheld_view", "asymmetry", 0.85, 0.7, "人物だけが全貌を見ている", 0.1),
            ("reader_guess", "revealed_object", "pursuit", 0.9, 0.0, "読者の推測が全貌を追う", 0.0),
            ("frame_limit", "reader_guess", "power_gradient", 0.6, 0.5, "コマ枠が読者の視界を制限する", 0.3),
            ("revealed_object", "witness", "power_gradient", 0.75, 0.8, "対象のスケールが人物を圧倒する", 0.65),
            ("withheld_view", "revealed_object", "embodiment", 0.7, 0.0, "伏せられた視界が決め絵を体現する", 0.65),
            ("reader_guess", "witness", "dependency", 0.6, 0.0, "読者は人物の反応に推測を依存する", 0.1),
            ("frame_limit", "revealed_object", "opposition", 0.55, -0.4, "小ゴマの枠と見開きの全景が対立する", 0.65),
        ],
        "beats": [
            ("b_reaction_first", "setup", 0.1, 0.08, "反応の先行提示", "対象を見せずに人物の驚愕表情だけを先に見せ、見えない対象の存在を確定させる", [], 0.15, -0.2, ["witness", "withheld_view"], False),
            ("b_fragments", "escalation", 0.3, 0.12, "断片の積み上げ", "影・足元・音・周囲の反応を細いコマで小出しにし、推測の材料を与える", ["b_reaction_first"], 0.15, 0.05, ["frame_limit", "reader_guess"], False),
            ("b_saturation", "local_optimization_limit", 0.55, 0.05, "推測の飽和", "断片から組み立てた読者の予測が限界に達し、これ以上の小出しが効かなくなる", ["b_fragments"], 0.15, 0.0, ["reader_guess"], False),
            ("b_breath", "transition", 0.6, 0.04, "左ページ末尾の息止め", "左ページ最終コマを無音の小ゴマにして呼吸を止め、めくりの手を誘う", ["b_saturation"], -0.1, 0.0, ["frame_limit", "witness"], True),
            ("b_spread_reveal", "revelation", 0.65, 0.1, "めくり直後の見開き全景", "右ページから左ページへ跨る見開き大ゴマで、伏せていた全貌を一挙に開示する", ["b_breath"], 0.4, 0.5, ["revealed_object", "withheld_view", "witness"], True),
            ("b_detail_payoff", "payoff", 0.75, 0.06, "全景内の伏線回収", "見開き内の視線誘導で、断片コマの影や音が全景のどこに対応するかを拾わせる", ["b_spread_reveal", "b_fragments"], -0.1, 0.05, ["revealed_object", "reader_guess"], False),
            ("b_afterglow", "afterglow", 1.0, 0.08, "余韻の小コマ", "次の右ページを小さな会話コマで始め、圧倒の余韻から物語の時間へ戻す", ["b_detail_payoff"], -0.25, 0.04, ["witness"], False),
        ],
        "mechanisms": [
            ("m_withheld_reaction", "curiosity_gap", "反応先行による情報ギャップ", "人物の反応だけを見せて対象を隠すと、読者は見えない原因を補完しようとする", "驚愕表情のCUと対象の不在", "読者が能動的に対象を想像し、開示への期待が膨らむ", "情報ギャップ理論", 0.85, ["b_reaction_first", "b_fragments"], ["withheld_view", "reader_guess"]),
            ("m_scale_shock", "prediction_error", "スケールの予測誤差", "細いコマで形成された予測を、見開きの全景が規模で上回る", "小ゴマ連続から見開きへの面積の跳躍", "予測を超える規模感が強い驚きと記憶を生む", "予測符号化理論", 0.9, ["b_spread_reveal"], ["revealed_object", "frame_limit"]),
            ("m_turn_reward", "anticipation_reward", "めくりの報酬", "息止めの小ゴマが予期を極大化し、めくり行為が報酬に直結する", "左ページ末尾の無音コマ", "ページをめくる身体動作そのものが快になる", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.8, ["b_breath", "b_spread_reveal"], ["frame_limit"]),
            ("m_peak_memory", "peak_end", "ピークの記憶化", "見開きの一枚絵がエピソード全体の記憶の代表値になる", "見開き大ゴマの強度", "話全体の印象が見開きの体験で上書きされる", "ピーク・エンドの法則", 0.7, ["b_spread_reveal", "b_afterglow"], ["revealed_object"]),
        ],
        "anchor": "対象を見上げて言葉を失う人物の表情（読者はまだ同じものを見ていない）",
        "drivers": ["人物だけが見ている視界への好奇心", "断片から全貌を組み立てる推理の快", "めくりの手を誘う息止めの小ゴマ"],
        "emotions": ["予感", "好奇", "焦燥", "息止め", "圧倒", "畏怖", "余韻"],
        "template": "{{WITNESS}}の反応だけを先に見せ、{{WITHHELD_VIEW}}を読者から隠す。{{FRAGMENT_SEQUENCE}}で断片を小出しにして{{READER_GUESS}}を飽和させ、左ページ最終コマに{{BREATH_PANEL}}を置いて呼吸を止める。めくった右ページから左ページへ跨る見開き大ゴマで{{SPECTACLE}}を全景開示し、見開き内の視線誘導で{{DETAIL_PAYOFF}}を回収する。",
        "slots": [
            ("WITNESS", "対象を先に目撃し反応だけが描かれる主体", "agent", ["反応は表情CUで描く", "対象を同じコマに入れない"], ("見上げて息を呑む主人公", "結果を見て絶句する導入事例の担当者", "データを見て驚く研究者")),
            ("WITHHELD_VIEW", "読者にだけ伏せられている視界", "lack", ["めくりまで全貌を見せない"], ("巨大な存在の全身", "導入後の劇的な変化の全体像", "全データを並べた分布図")),
            ("FRAGMENT_SEQUENCE", "推測材料となる断片の連続", "evidence", ["細いコマで3〜6個", "影・音・部分・周囲の反応に限定"], ("影・足音・揺れる水面・逃げる鳥", "顧客の声の断片を3つ", "部分的な数値を順に提示")),
            ("READER_GUESS", "断片から読者が組み立てる予測", "goal", ["開示の規模が予測を上回るよう設計する"], ("大きな何かがいるという予感", "かなり改善したらしいという期待", "何か傾向があるという推測")),
            ("BREATH_PANEL", "左ページ最終コマの無音の息止め", "emotion", ["セリフなし", "小さく置く"], ("見上げる横顔の無音コマ", "結果発表直前の一行", "次の図を前にした一文の溜め")),
            ("SPECTACLE", "見開き全景で開示される決め絵", "symbol", ["見開き大ゴマ", "人物を小さく入れてスケールを示す"], ("都市を覆う巨影の全景", "導入前後の全体比較図", "全体分布を示す大きな図解")),
            ("DETAIL_PAYOFF", "全景の中で回収される断片の対応", "reversal", ["断片の位置を視線誘導で辿らせる"], ("影の主が全景の片隅の尾だったと分かる", "顧客の声が全体のどこに位置するか示す", "冒頭の部分値が分布の外れ値だと示す")),
        ],
        "invariants": [
            "全貌は見開き大ゴマまで一切見せない",
            "見開き大ゴマは必ずめくり直後の右ページから始める（右ページ開始・左ページ終端）",
            "めくり直前の左ページ最終コマは無音の小ゴマにする",
            "断片コマは細く小さく、見開きとの面積比の落差を最大化する",
            "見開き内の視線誘導は右上の人物から左下の対象細部へ流す",
        ],
        "guide": {
            "manga": "7ページ構成。p1〜p4は5〜7コマの細いコマで構成し、p1に驚愕表情のCU（アオリ）、p2〜p3に影・足元・音のECUと周囲の反応MSを縦長の細コマで並べる。p4（左）は7コマに詰めて推測を飽和させ、最終コマを無音の横顔CU・望遠85mmで小さく置く。めくって p5〜p6 を見開き大ゴマ1枚とし、ELS・超広角18mm・アオリで全景を開示、右上に人物を小さく置いて左下の対象細部へ視線を流す。p7（右）は4コマの会話で余韻を引き取る。",
            "marketing": "LPでは成果を語る顧客の反応（驚きや感謝の声）を先に断片的に見せ、成果の全体像は伏せる。スクロールで断片を重ね、区切りの直前に短い一文で溜めた上で、大きなビジュアル（全体比較図・ビフォーアフターの全景）で一挙に開示する。",
            "article": "記事では人々の驚いた反応や部分的な数値を先に並べ、全体像の図解は後半まで出さない。見出しの切れ目直前に一文の溜めを置き、次の章冒頭で全体像を示す大きな図を掲げて、それまでの断片が図のどこに当たるかを順に回収する。",
        },
        "rhythm": {
            "page_count": 7,
            "panels_per_page": [5, 6, 5, 7, 1, 1, 4],
            "emphasis_pages": [4, 5],
            "page_turn_reveals": [4],
            "spread_pages": [4],
            "silent_panels_ratio": 0.2,
            "pacing_notes": [
                "p1（右）: 5コマ。対象を映さず人物の驚愕表情CUを先行させる",
                "p2（左）: 6コマ。影・足元のECUと周囲の反応を細い縦コマで並べる",
                "p3（右）: 5コマ。音と揺れの断片で推測材料を追加する",
                "p4（左）: 7コマに詰めて推測を飽和させ、最終コマは無音の小ゴマで息を止める",
                "p5（右・めくり直後）: 見開き大ゴマの右半分。人物を右上に小さく置く",
                "p6（左）: 見開き大ゴマの左半分。全景の細部へ視線を流し伏線を回収する",
                "p7（右）: 4コマの会話で余韻から物語の時間へ戻す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 03: 静寂から爆発への緩急
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_03_silence_to_explosion",
        "title": "静寂から爆発への緩急",
        "summary": "音のある日常から唐突に擬音とフキダシを消し、無音コマの連打で読者だけが知る時限を進め、嵐の前の一瞬の弛緩を挟んで、めくり直後の全面断ち切りで爆発と巨大な描き文字により失われた音を一挙に返す骨格。",
        "tags": ["緩急", "無音", "サスペンス", "爆発", "描き文字"],
        "lineage": ["時限爆弾型サスペンス（読者だけが知る危機）", "無音演出から轟音への音響的対比", "嵐の前の静けさ"],
        "waves": {
            "tension": [(0.0, 0.35), (0.1, 0.28), (0.25, 0.34), (0.4, 0.45), (0.55, 0.56), (0.7, 0.66), (0.75, 0.6), (0.8, 1.0), (0.9, 0.62), (1.0, 0.35)],
            "release": [(0.0, 0.2), (0.1, 0.3), (0.4, 0.15), (0.7, 0.05), (0.75, 0.08), (0.8, 0.2), (0.9, 0.6), (1.0, 0.8)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.3), (0.4, 0.42), (0.6, 0.55), (0.75, 0.6), (0.8, 0.85), (1.0, 0.9)],
            "character_knowledge": [(0.0, 0.2), (0.2, 0.22), (0.4, 0.25), (0.6, 0.28), (0.75, 0.3), (0.8, 0.85), (1.0, 0.9)],
            "cognitive_load": [(0.0, 0.45), (0.2, 0.35), (0.4, 0.25), (0.6, 0.2), (0.75, 0.15), (0.8, 0.6), (0.9, 0.5), (1.0, 0.3)],
        },
        "nodes": [
            ("unaware_group", "agent", "unaware_party", "危機に気づかない当事者たち", "静けさの中で日常の所作を続け、迫る破局を知らない主体", 0.75, "UNAWARE_PARTY"),
            ("fuse", "symbol", "burning_fuse", "静かに進行する時限", "無音コマの連打の中で少しずつ短くなる導火線の象徴", 0.85, "FUSE_DETAIL"),
            ("eruptive_force", "agent", "eruptive_force", "臨界で解放される力", "静寂の背後で蓄積され、一コマの爆発として顕在化する力", 0.9, "EXPLOSION_IMAGE"),
            ("reader_dread", "desire", "reader_dread", "破局を予感しつつ見届けたい読者の欲求", "読者だけが時限を見ており、気づけと願いながら頁を進める", 0.8, "READER_DREAD"),
            ("lost_sound", "lack", "absent_sound", "失われた音（擬音とセリフの欠如）", "フキダシも描き文字も消えた空白が、爆発の音量を相対的に増幅する", 0.7, "SILENCE_FIELD"),
        ],
        "edges": [
            ("reader_dread", "unaware_group", "asymmetry", 0.85, 0.8, "読者だけが危機を知っている", 0.2),
            ("fuse", "eruptive_force", "embodiment", 0.8, 0.0, "時限が蓄積する力を体現する", 0.3),
            ("eruptive_force", "unaware_group", "power_gradient", 0.9, 0.9, "解放される力が当事者を圧倒する", 0.8),
            ("eruptive_force", "unaware_group", "opposition", 0.85, 0.0, "力が日常そのものを破壊しにくる", 0.8),
            ("reader_dread", "lost_sound", "dependency", 0.6, 0.0, "読者の予感は無音の空白に支えられる", 0.1),
            ("reader_dread", "fuse", "pursuit", 0.75, 0.0, "読者の視線が時限の進行を追う", 0.3),
            ("lost_sound", "eruptive_force", "opposition", 0.5, -0.6, "無音の場が爆発の轟音と対置される", 0.8),
        ],
        "beats": [
            ("b_ordinary_noise", "premise", 0.0, 0.06, "音のある日常", "会話と擬音のある普通のコマで基準となる音量を提示する", [], 0.0, 0.0, ["unaware_group"], False),
            ("b_sound_drop", "setup", 0.1, 0.05, "音の消失", "フキダシと擬音を唐突に消し、無音コマの連打へ切り替える", ["b_ordinary_noise"], -0.1, 0.05, ["lost_sound"], True),
            ("b_silent_chain", "escalation", 0.4, 0.15, "無音コマの連打", "同サイズの小ゴマで所作・風景・時計を淡々と刻み、時間の進行だけを感じさせる", ["b_sound_drop"], 0.15, 0.1, ["lost_sound", "unaware_group"], False),
            ("b_fuse_detail", "escalation", 0.55, 0.08, "時限の細部", "読者だけに時限の進行をECUで見せ、当事者の無関心と対置する", ["b_silent_chain"], 0.15, 0.15, ["fuse", "reader_dread"], False),
            ("b_last_breath", "transition", 0.75, 0.04, "嵐の前の弛緩", "左ページ末尾で当事者が一瞬だけ緩む小ゴマを置き、緊張をわずかに抜く", ["b_fuse_detail"], -0.1, 0.0, ["unaware_group"], True),
            ("b_explosion", "climax", 0.8, 0.06, "めくり直後の爆発", "右ページ全面の断ち切りで爆発を描き、巨大な描き文字で失われた音を一挙に返す", ["b_last_breath", "b_fuse_detail"], 0.4, 0.55, ["eruptive_force", "lost_sound", "unaware_group"], True),
            ("b_aftermath", "afterglow", 1.0, 0.1, "残響と瓦礫", "粉塵の中の小さな人影を俯瞰で置き、音が引いていく余韻を描く", ["b_explosion"], -0.3, 0.05, ["unaware_group"], False),
        ],
        "mechanisms": [
            ("m_time_bomb", "dramatic_irony", "読者だけが知る時限", "当事者が知らない危機を読者だけに見せ、気づけという焦燥を持続させる", "時限の細部と当事者の無関心の対置", "読者がページを止められなくなるサスペンス", "サスペンス理論（劇的アイロニー）", 0.85, ["b_fuse_detail", "b_silent_chain"], ["reader_dread", "fuse"]),
            ("m_contrast_volume", "contrast_effect", "無音と轟音の対比", "音の欠如を続けた後に最大音量の描き文字を置くと、知覚上の音量が増幅される", "無音コマの連打から描き文字の全面化", "爆発の衝撃が実際の描写以上に大きく感じられる", "対比効果（知覚の順応と対比）", 0.9, ["b_sound_drop", "b_explosion"], ["lost_sound"]),
            ("m_guard_down", "prediction_error", "弛緩直後の破局", "緊張がわずかに抜けた瞬間に破局を置くと、予測の防御が外れた状態で衝撃を受ける", "嵐の前の弛緩コマ", "身構えが解けた瞬間の驚愕", "予測符号化理論", 0.75, ["b_last_breath", "b_explosion"], ["unaware_group"]),
            ("m_catharsis_noise", "catharsis", "音の回帰による解放", "溜め込まれた緊張が音の回帰とともに一挙に放出される", "爆発の描き文字と粉塵の余韻", "張り詰めた身体感覚の解放", "カタルシス（緊張解放）", 0.6, ["b_explosion", "b_aftermath"], ["eruptive_force"]),
        ],
        "anchor": "何も知らずに日常の所作を続ける当事者たちの、無防備な横顔",
        "drivers": ["読者だけが見ている時限", "フキダシの消えた無音のページ", "めくり直前の一瞬の弛緩"],
        "emotions": ["日常", "違和", "焦燥", "息止め", "衝撃", "放心"],
        "template": "まず音のある日常で基準となる音量を示し、{{SOUND_DROP}}によって{{SILENCE_FIELD}}へ切り替える。無音コマの連打で{{UNAWARE_PARTY}}の所作を淡々と刻みながら、読者だけに{{FUSE_DETAIL}}を見せて{{READER_DREAD}}を募らせる。左ページ末尾に{{LAST_RELAXATION}}を置いて緊張をわずかに抜き、めくり直後の右ページ全面で{{EXPLOSION_IMAGE}}と巨大な描き文字により失われた音を一挙に返す。",
        "slots": [
            ("SOUND_DROP", "音が消える切り替えの合図", "setting", ["一コマで切り替える"], ("鳴いていた蝉が一斉に止む", "背景音が消える演出の一行", "それまでの饒舌な語りを一文で断ち切る")),
            ("SILENCE_FIELD", "フキダシも擬音もない無音の場", "setting", ["2ページ以上続ける", "同サイズの小ゴマで刻む"], ("無音の教室の小ゴマ連打", "余白を多くとった静かな構成", "短文だけを淡々と並べる段落")),
            ("UNAWARE_PARTY", "危機に気づかない当事者", "agent", ["危機の方向を見ない"], ("談笑する仲間たち", "問題に気づいていない見込み客", "前提を疑わない従来手法の実践者")),
            ("FUSE_DETAIL", "読者だけが見る時限の細部", "symbol", ["ECUで繰り返し提示する", "進行度を段階的に示す"], ("短くなる導火線", "静かに減り続ける残り枠", "見過ごされている指標の悪化")),
            ("READER_DREAD", "読者が抱く破局の予感", "emotion", ["当事者には共有させない"], ("早く逃げろという焦り", "このままでは損をするという予感", "このままでは破綻するという直感")),
            ("LAST_RELAXATION", "破局直前に置く一瞬の弛緩", "emotion", ["左ページ最終コマ", "笑顔や安堵を小さく"], ("ほっと笑う横顔", "これで安心と思わせる一文", "一見うまくいっているかに見える結果")),
            ("EXPLOSION_IMAGE", "めくり直後に解放される破局の一枚絵", "reversal", ["右ページ全面断ち切り", "最大の描き文字を添える"], ("校舎を貫く爆炎と轟音の描き文字", "放置した場合の損失を示す大きな図", "前提が崩壊する決定的データ")),
        ],
        "invariants": [
            "無音区間ではフキダシと描き文字を一切使わない",
            "破局の直前には必ず一コマの弛緩を置く",
            "爆発はめくり直後の右ページ全面（断ち切り1コマ）に置く",
            "無音区間のコマは同サイズの小ゴマで等間隔に刻み、視線を一定速度で流す",
        ],
        "guide": {
            "manga": "6ページ構成。p1（右）は会話と擬音のある5コマ。p2（左）で擬音を消し、8コマの等サイズ小ゴマで所作・時計・窓の外をMSとECUで交互に刻む。p3（右）6コマで読者だけに時限のECU（俯瞰）を3回挟む。p4（左）4コマ、最終コマに当事者の笑顔CUを小さく置く。めくって p5（右）は断ち切り1コマ、ELS・広角18mm・ダッチアングル15度で爆発、ページ幅いっぱいの描き文字。p6（左）5コマで粉塵の俯瞰ELSから人影MSへ戻す。",
            "marketing": "ページ前半を静かな日常描写（顧客が問題に気づかず業務を続ける様子）で構成し、読者にだけ数値の悪化を示す。直前に今のところ問題はないと安心させる一文を置いた上で、区切りの直後に放置した場合の損失を大きなビジュアルで一挙に示し、解決策の提示へつなぐ。",
            "article": "論説では従来手法がうまく回っている様子を淡々と記述し、その間に見過ごされている指標だけを短く何度か示す。一見問題がないと述べた直後の章冒頭で、決定的なデータによって前提が崩壊する瞬間を提示し、問題の深刻さを体感させる。",
        },
        "rhythm": {
            "page_count": 6,
            "panels_per_page": [5, 8, 6, 4, 1, 5],
            "emphasis_pages": [4],
            "page_turn_reveals": [4],
            "spread_pages": [],
            "silent_panels_ratio": 0.45,
            "pacing_notes": [
                "p1（右）: 5コマ。会話と擬音のある日常で基準の音量を示す",
                "p2（左）: 8コマの等サイズ小ゴマ。擬音を消して無音の所作を刻む",
                "p3（右）: 6コマ。読者だけに時限のECUを挟み焦燥を募らせる",
                "p4（左）: 4コマ。最終コマで当事者が一瞬緩む小ゴマを置く",
                "p5（右・めくり直後）: 断ち切り1コマの爆発と全面の描き文字",
                "p6（左）: 5コマ。粉塵の俯瞰から人影へ戻り、音が引く余韻を描く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 04: バトル：溜め→一閃→残心
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_04_battle_charge_strike_zanshin",
        "title": "バトル：溜め→一閃→残心",
        "summary": "間合いの対峙から部位のECUで溜めを刻み、読み合いの膠着を経てめくり直後の1コマで一閃を描き、結果を描かない静止の一拍・遅れて走る亀裂・構えを解かない残心で締める一騎打ちの骨格。",
        "tags": ["バトル", "一騎打ち", "溜め", "一閃", "残心"],
        "lineage": ["居合・剣戟の間合い演出", "動と静の対比による決着描写", "遅延された結果開示（ディレイド・リアクション）"],
        "waves": {
            "tension": [(0.0, 0.5), (0.15, 0.6), (0.3, 0.72), (0.45, 0.82), (0.5, 0.97), (0.6, 0.78), (0.65, 0.86), (0.75, 0.5), (0.9, 0.3), (1.0, 0.38)],
            "release": [(0.0, 0.1), (0.3, 0.05), (0.5, 0.25), (0.6, 0.15), (0.7, 0.5), (0.85, 0.75), (1.0, 0.6)],
            "audience_knowledge": [(0.0, 0.3), (0.2, 0.38), (0.4, 0.45), (0.5, 0.55), (0.6, 0.56), (0.7, 0.85), (1.0, 0.92)],
            "character_knowledge": [(0.0, 0.3), (0.2, 0.4), (0.4, 0.5), (0.5, 0.7), (0.6, 0.8), (0.7, 0.88), (1.0, 0.92)],
            "cognitive_load": [(0.0, 0.5), (0.2, 0.55), (0.4, 0.4), (0.5, 0.3), (0.6, 0.25), (0.75, 0.4), (1.0, 0.3)],
        },
        "nodes": [
            ("fighter", "agent", "protagonist_fighter", "一撃に全てを賭ける主体", "間合いの中で溜めを作り、一閃で勝負を決めようとする主体", 0.9, "FIGHTER"),
            ("opponent", "agent", "opponent", "拮抗する相手", "主体と同等以上の力を持ち、読み合いで膠着をつくる対手", 0.85, "OPPONENT"),
            ("decisive_strike", "symbol", "decisive_strike", "一閃の決め絵", "溜めの全量が一コマに凝縮される決定打の象徴", 0.95, "STRIKE_IMAGE"),
            ("victory_desire", "desire", "victory", "勝敗の決着を見届けたい欲求", "読者と主体が共有する、一撃の成否への渇望", 0.8, "OUTCOME_QUESTION"),
            ("unknown_outcome", "lack", "suspended_outcome", "結果の空白", "一閃の直後に結果を描かない一拍。主体だけが手応えを知っている", 0.75, "DELAYED_RESULT"),
            ("stance", "symbol", "residual_stance", "解かれない構え（残心）", "決着後も構えを解かない姿勢。勝利の重さと次への緊張を同時に示す", 0.6, "ZANSHIN_IMAGE"),
        ],
        "edges": [
            ("fighter", "opponent", "opposition", 0.95, 0.0, "互いに一撃を狙う拮抗", 0.0),
            ("opponent", "fighter", "power_gradient", 0.7, 0.3, "序盤は相手がわずかに優位", 0.1),
            ("fighter", "decisive_strike", "pursuit", 0.9, 0.0, "主体は一閃の機を追う", 0.2),
            ("decisive_strike", "opponent", "power_gradient", 0.85, 0.9, "一閃が勢力を逆転させる", 0.5),
            ("fighter", "unknown_outcome", "asymmetry", 0.7, 0.6, "主体だけが手応えを知っている", 0.5),
            ("victory_desire", "unknown_outcome", "dependency", 0.6, 0.0, "決着の欲求は結果の空白に宙吊りにされる", 0.55),
            ("stance", "fighter", "embodiment", 0.6, 0.0, "残心が勝者の覚悟を体現する", 0.9),
        ],
        "beats": [
            ("b_standoff", "premise", 0.0, 0.08, "間合いの対峙", "両者の距離と構えを俯瞰とFSで示し、一撃で決まるという勝負の規則を提示する", [], 0.0, 0.0, ["fighter", "opponent"], False),
            ("b_charge", "setup", 0.2, 0.12, "溜めの部位カット", "握り・呼吸・足指・視線のECUを細かく刻み、力の蓄積を身体の部分で描く", ["b_standoff"], 0.2, 0.05, ["fighter", "decisive_strike"], False),
            ("b_deadlock", "local_optimization_limit", 0.45, 0.06, "読み合いの膠着", "互いの読みが釣り合い、待つだけでは勝てない地点に達する", ["b_charge"], 0.1, 0.0, ["fighter", "opponent"], False),
            ("b_strike", "climax", 0.5, 0.05, "めくり直後の一閃", "右ページ1コマの断ち切りで一閃を描き、動線を斜めに切って速度を見せる", ["b_deadlock"], 0.15, 0.1, ["decisive_strike", "fighter"], True),
            ("b_delay", "crisis", 0.6, 0.05, "結果の空白", "一閃の直後に両者が静止する一拍を置き、成否を描かない", ["b_strike"], -0.2, -0.15, ["unknown_outcome", "victory_desire"], True),
            ("b_delayed_reaction", "revelation", 0.65, 0.05, "遅れて来る結果", "相手の装備や身体に一拍遅れて亀裂が走り、決着が開示される", ["b_delay"], 0.08, 0.3, ["opponent", "unknown_outcome"], True),
            ("b_collapse", "payoff", 0.75, 0.08, "崩れ落ち", "相手が崩れ、溜めの部位カットが一閃の伏線だったと回収される", ["b_delayed_reaction", "b_charge"], -0.35, 0.05, ["opponent", "fighter"], False),
            ("b_zanshin", "afterglow", 1.0, 0.1, "残心", "勝者が構えを解かない姿を引きで置き、勝利の重さと次の気配を残す", ["b_collapse"], 0.08, 0.02, ["stance", "fighter"], False),
        ],
        "mechanisms": [
            ("m_embodied_charge", "anticipation_reward", "身体部位による溜めの予期", "部位のECUを積み重ねると読者の身体が力の蓄積を模倣し、解放を予期する", "握り・呼吸・足指のECU連続", "一閃の瞬間に身体的な快感を伴う解放が起きる", "身体化された認知（ミラーニューロン仮説）", 0.8, ["b_charge", "b_strike"], ["fighter"]),
            ("m_suspended_result", "zeigarnik", "結果の宙吊り", "決着の直後に結果を描かない一拍が、未完了の緊張を最大化する", "一閃直後の静止コマ", "読者が結果を確かめずにはいられない", "未完了効果（ツァイガルニク効果）", 0.85, ["b_delay", "b_delayed_reaction"], ["unknown_outcome", "victory_desire"]),
            ("m_info_lag", "dramatic_irony", "手応えの先行", "主体だけが手応えを知る短い非対称が、勝者の余裕と凄みを演出する", "主体の静かな表情と相手の無傷に見える姿", "結果開示の瞬間に主体の格が一段上がる", "劇的アイロニー", 0.65, ["b_delay"], ["fighter", "unknown_outcome"]),
            ("m_peak_end_zanshin", "peak_end", "残心による終端の記憶化", "決着後の静かな残心がピークの記憶に重さと品位を与える", "構えを解かない引きの一枚", "戦いの記憶が品位ある印象で締めくくられる", "ピーク・エンドの法則", 0.6, ["b_zanshin"], ["stance"]),
        ],
        "anchor": "一撃に全てを賭けて呼吸を整える主体の、握りしめた手元",
        "drivers": ["部位カットで蓄積される力の予感", "一閃直後の結果の空白", "一拍遅れて走る亀裂"],
        "emotions": ["緊迫", "凝縮", "解放", "宙吊り", "確信", "畏敬"],
        "template": "{{FIGHTER}}と{{OPPONENT}}の間合いを俯瞰で示し、{{CHARGE_DETAILS}}で溜めを身体の部分に刻む。読み合いが膠着した左ページ末尾からめくり、右ページ1コマで{{STRIKE_IMAGE}}を描く。直後に{{DELAYED_RESULT}}として静止の一拍を置いて{{OUTCOME_QUESTION}}を宙吊りにし、一拍遅れて結果を開示した後、{{ZANSHIN_IMAGE}}で締めくくる。",
        "slots": [
            ("FIGHTER", "一撃に賭ける主体", "agent", ["溜めの間は動かさない"], ("居合の構えの剣士", "勝負所を待つ営業担当", "一つの仮説に賭ける研究者")),
            ("OPPONENT", "拮抗する相手", "agent", ["主体と同等以上の格"], ("同門の宿敵", "強力な競合サービス", "長年支持されてきた定説")),
            ("CHARGE_DETAILS", "溜めを示す身体部位・細部の連続", "evidence", ["ECUで3〜6コマ", "動きは最小限"], ("握り・呼吸・足指・汗", "準備の積み重ねを示す短い事実", "検証の手順を一つずつ示す")),
            ("STRIKE_IMAGE", "一閃の決め絵", "symbol", ["めくり直後の右ページ1コマ", "斜めの動線"], ("斜めに走る白い剣閃", "決定的な一手の提示", "定説を覆す一つの実験結果")),
            ("DELAYED_RESULT", "結果を描かない静止の一拍", "irony", ["一閃の直後に置く", "セリフは最小限"], ("背中合わせで静止する二人", "発表直前の一拍の沈黙", "結果を述べる前の一段落の溜め")),
            ("OUTCOME_QUESTION", "宙吊りにされる勝敗の問い", "goal", ["開示まで答えない"], ("どちらが斬られたのか", "本当に成果は出たのか", "仮説は正しかったのか")),
            ("ZANSHIN_IMAGE", "決着後も構えを解かない終端の姿", "emotion", ["引きの構図", "勝利を誇らない"], ("納刀せず遠くを見る剣士", "次の課題を見据える担当者の横顔", "検証の限界を静かに述べる結語")),
        ],
        "invariants": [
            "一閃はめくり直後の右ページに1コマで描く",
            "一閃の直後に結果を描かない静止の一拍を必ず置く",
            "溜めの区間は部位のECUを小ゴマで刻み、全身ショットを避ける",
            "一閃のコマは斜めの動線で右上から左下へ視線を切り裂く",
            "残心は勝敗開示の後、引きの構図で置く",
        ],
        "guide": {
            "manga": "5ページ構成。p1（右）4コマ: 間合いを俯瞰ELSとFSで示す。p2（左）7コマ: 握り・呼吸・足指・視線のECUを細い小ゴマで刻み、最終コマは相手の目のECU（アイレベル・望遠135mm）で膠着を示す。めくって p3（右）1コマ断ち切り: 一閃を広角24mm・ダッチアングル20度、右上から左下への斜めの剣閃で描く。p4（左）3コマ: 背中合わせの静止MS→相手の亀裂CU→崩れ落ちFS。p5（右）3コマ: 残心を俯瞰ELSで引いて締める。",
            "marketing": "競合比較型LPで、自社と競合の拮抗を冷静に示した上で、準備の積み重ね（技術・実績の細部）を短い事実で刻む。決定的な一手（独自の機能や保証）を大きく一つだけ提示し、直後に結果を急いで語らず一拍置いてから成果データを開示し、最後は誇らず次の改善への姿勢で締める。",
            "article": "論説では対立する二つの立場を公平に提示し、検証手順を一つずつ積み上げる。決定的な実験結果を一つだけ大きく示した後、すぐに結論を述べず一段落の溜めを置き、その上で結論を開示する。結語では結論の限界と次の問いを静かに述べて品位を保つ。",
        },
        "rhythm": {
            "page_count": 5,
            "panels_per_page": [4, 7, 1, 3, 3],
            "emphasis_pages": [2],
            "page_turn_reveals": [2],
            "spread_pages": [],
            "silent_panels_ratio": 0.3,
            "pacing_notes": [
                "p1（右）: 4コマ。間合いと構えを俯瞰とFSで示し勝負の規則を提示する",
                "p2（左）: 7コマ。部位のECUを細かく刻み、最終コマで膠着の視線を置く",
                "p3（右・めくり直後）: 断ち切り1コマの一閃。斜めの剣閃で視線を切り裂く",
                "p4（左）: 3コマ。静止の一拍→遅れて走る亀裂→崩れ落ちで結果を開示する",
                "p5（右）: 3コマ。残心を引きの構図で置き、次の気配を残す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 05: 心理戦：読み合いの三段反転
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_05_mind_game_triple_reversal",
        "title": "心理戦：読み合いの三段反転",
        "summary": "冒頭で規則を明示し、策士の意図不明の仕掛けから、相手が見破る第一の反転、劣勢が布石だった第二の反転、規則の死角を突く最終手を見開きで示す第三の反転へと、めくり位置と同期させて予測を三度上書きする心理戦の骨格。",
        "tags": ["心理戦", "頭脳戦", "反転", "見開き", "規則"],
        "lineage": ["フェアプレイ原則に基づく知略劇", "多段どんでん返し", "ゲーム理論的駆け引きの劇化"],
        "waves": {
            "tension": [(0.0, 0.3), (0.15, 0.45), (0.2, 0.62), (0.3, 0.5), (0.45, 0.78), (0.55, 0.6), (0.65, 0.98), (0.75, 0.7), (0.9, 0.45), (1.0, 0.55)],
            "release": [(0.0, 0.1), (0.2, 0.3), (0.3, 0.4), (0.45, 0.2), (0.55, 0.35), (0.65, 0.15), (0.8, 0.6), (1.0, 0.5)],
            "audience_knowledge": [(0.0, 0.2), (0.15, 0.3), (0.2, 0.5), (0.4, 0.52), (0.45, 0.7), (0.6, 0.72), (0.65, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.4), (0.2, 0.6), (0.45, 0.8), (0.65, 0.95), (0.8, 0.97), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.6), (0.2, 0.7), (0.3, 0.75), (0.45, 0.85), (0.55, 0.8), (0.65, 0.9), (0.8, 0.5), (1.0, 0.45)],
        },
        "nodes": [
            ("strategist", "agent", "strategist", "罠を重ねる策士", "読者にも手の内を明かさず、相手の読みを利用して局面を反転させる主体", 0.9, "STRATEGIST"),
            ("adversary", "agent", "adversary", "読みで対抗する相手", "策士と同等の知性を持ち、一度は勝利を確信する対手", 0.85, "ADVERSARY"),
            ("game_rule", "symbol", "game_rule", "勝負の規則", "冒頭で明示され、全ての反転が従う公正な制約", 0.7, "GAME_RULE"),
            ("hidden_plan", "lack", "concealed_plan", "読者に伏せられた真の策", "三段の反転のたびに一層ずつ開示される策士の計画", 0.85, "HIDDEN_PLAN"),
            ("reader_deduction", "desire", "reader_deduction", "先を読もうとする読者の推理", "規則に基づいて結末を予測しようとし、その予測ごと裏切られる欲求", 0.8, "READER_DEDUCTION"),
            ("winning_move", "symbol", "winning_move", "最終手の決め絵", "三段目の反転を象徴する、規則の死角を突いた一手", 0.9, "FINAL_MOVE"),
        ],
        "edges": [
            ("strategist", "adversary", "opposition", 0.95, 0.0, "知略による正面対立", 0.0),
            ("adversary", "strategist", "power_gradient", 0.7, 0.5, "第一の反転で相手が優位に立つ", 0.2),
            ("strategist", "hidden_plan", "asymmetry", 0.9, 0.8, "策士だけが真の計画を知る", 0.05),
            ("reader_deduction", "hidden_plan", "pursuit", 0.85, 0.0, "読者の推理が伏せられた策を追う", 0.1),
            ("reader_deduction", "game_rule", "dependency", 0.7, 0.0, "読者の推理は規則に依存する", 0.0),
            ("winning_move", "adversary", "power_gradient", 0.9, 0.9, "最終手が勢力を決定的に反転させる", 0.65),
            ("strategist", "adversary", "betrayal", 0.6, 0.0, "協力を装った申し出が罠だったと判明する", 0.45),
            ("hidden_plan", "winning_move", "embodiment", 0.8, 0.0, "伏せられた策が最終手に結晶する", 0.65),
        ],
        "beats": [
            ("b_rules", "premise", 0.0, 0.06, "規則の明示", "勝負の規則を図解的なコマで明確に示し、以後の反転が全て規則内で起きる公正さを担保する", [], 0.0, 0.1, ["game_rule"], False),
            ("b_first_trap", "setup", 0.1, 0.06, "最初の仕掛け", "策士の不可解な一手を見せ、意図は伏せる", ["b_rules"], 0.15, -0.1, ["strategist", "hidden_plan"], False),
            ("b_rev1", "structural_twist", 0.2, 0.05, "第一の反転", "めくり直後に相手が仕掛けを見破り、策士が追い詰められたように見せる", ["b_first_trap"], 0.2, 0.2, ["adversary", "strategist"], True),
            ("b_counter_setup", "setup", 0.3, 0.06, "劣勢の中の布石", "追い詰められた策士が意味ありげな視線を一コマだけ見せる", ["b_rev1"], -0.12, -0.05, ["strategist", "hidden_plan"], False),
            ("b_rev2", "revelation", 0.45, 0.05, "第二の反転", "めくり直後に劣勢そのものが布石だったと開示し、相手の確信を崩す", ["b_counter_setup", "b_rules"], 0.28, 0.2, ["strategist", "adversary", "hidden_plan"], True),
            ("b_doubt", "crisis", 0.55, 0.05, "読者の推理の行き詰まり", "相手が再反撃の兆しを見せ、読者の予測が分岐して決められなくなる", ["b_rev2"], -0.18, 0.0, ["reader_deduction", "adversary"], False),
            ("b_rev3", "climax", 0.65, 0.08, "第三の反転（見開き）", "めくり直後の見開きで、規則の死角を突いた最終手と最初の仕掛けの真意を同時に開示する", ["b_doubt", "b_first_trap"], 0.38, 0.25, ["winning_move", "strategist", "game_rule"], True),
            ("b_logic_unpack", "payoff", 0.75, 0.06, "論理の解説", "最終手に至る論理を規則に照らして短く解説し、後出しでない公正さを確認させる", ["b_rev3"], -0.28, 0.05, ["game_rule", "reader_deduction"], False),
            ("b_cost", "afterglow", 0.9, 0.06, "勝者の代償", "勝利の裏で策士が失ったものを一コマで示し、余韻に陰影を与える", ["b_logic_unpack"], -0.25, 0.02, ["strategist"], False),
            ("b_next_game", "hook", 1.0, 0.04, "次の勝負の予告", "敗者または第三者が次の勝負を示唆して引く", ["b_cost"], 0.1, 0.0, ["adversary"], False),
        ],
        "mechanisms": [
            ("m_fair_play", "commitment_consistency", "規則への公正なコミット", "冒頭で規則を明示すると、読者は規則内での解決を期待し、反転を公正なものとして受け入れる", "図解的な規則提示", "反転が後出しでなく必然と感じられ、納得が驚きを強化する", "コミットメントと一貫性の原理", 0.7, ["b_rules", "b_logic_unpack"], ["game_rule"]),
            ("m_layered_surprise", "prediction_error", "多層の予測誤差", "反転のたびに読者の予測モデルを一段上書きし、次の予測を難しくする", "三段の反転とめくり位置の同期", "反転ごとに驚きが増幅し、最後に最大化する", "予測符号化理論", 0.9, ["b_rev1", "b_rev2", "b_rev3"], ["reader_deduction"]),
            ("m_concealed_mind", "curiosity_gap", "策士の内心の秘匿", "策士の意図を読者にも伏せることで、全ての行動に解釈の余白を残す", "意味ありげな視線の一コマ", "読者が策士の内心を推理し続ける", "情報ギャップ理論", 0.8, ["b_first_trap", "b_counter_setup"], ["hidden_plan", "strategist"]),
            ("m_participation", "self_efficacy", "推理参加の効力感", "規則が明示されているため、読者も推理に参加できるという感覚を持つ", "解ける難度の論理", "読者が自分も読めたかもしれないという参加感を得る", "自己効力感理論", 0.6, ["b_doubt", "b_logic_unpack"], ["reader_deduction"]),
            ("m_bittersweet", "peak_end", "勝者の代償による終端", "勝利の後に代償を置くことで、ピークの興奮に陰影が加わり記憶に残る", "勝者が失ったものの一コマ", "単純な爽快感でなく重みのある記憶になる", "ピーク・エンドの法則", 0.5, ["b_cost"], ["strategist"]),
        ],
        "anchor": "劣勢の中でも一瞬だけ見せる策士の静かな視線",
        "drivers": ["明示された規則と、その死角への期待", "反転ごとに上書きされる予測", "策士の内心に対する終わらない推理"],
        "emotions": ["緊張", "確信", "動揺", "疑念", "驚嘆", "納得", "感傷"],
        "template": "{{GAME_RULE}}を冒頭で明示し、{{STRATEGIST}}が意図不明の仕掛けを打つ。めくり直後に{{ADVERSARY}}が仕掛けを見破る第一の反転、劣勢そのものが布石だったと示す第二の反転を置き、{{READER_DEDUCTION}}を行き詰まらせる。めくり直後の見開きで{{FINAL_MOVE}}と{{HIDDEN_PLAN}}を同時に開示する第三の反転を置き、規則に照らした解説の後、{{VICTOR_COST}}で余韻を残す。",
        "slots": [
            ("GAME_RULE", "勝負の公正な規則", "setting", ["冒頭で図解的に明示する", "後から追加しない"], ("札を三回交換できる賭け", "比較検討の評価基準", "議論の前提となる定義")),
            ("STRATEGIST", "手の内を伏せる策士", "agent", ["内心を独白させない"], ("無表情の天才勝負師", "一見不利な条件を出す売り手", "反直観的な主張をする論者")),
            ("ADVERSARY", "一度は勝利を確信する相手", "agent", ["策士と同等の知性"], ("全てを読んだと笑う対戦相手", "競合の定番提案", "有力な反論")),
            ("READER_DEDUCTION", "規則に基づく読者の予測", "goal", ["二度裏切ってから最後に満たす"], ("次はこう来るはずという読み", "結局は価格で選ぶだろうという予断", "結論はこうなるだろうという予想")),
            ("FINAL_MOVE", "規則の死角を突いた最終手", "reversal", ["見開きで開示", "規則内で成立"], ("最初に捨てた札が切り札だった", "無料提供が実は本命の導線だった", "最初の反例が理論の決定的証拠だった")),
            ("HIDDEN_PLAN", "伏せられていた真の計画", "lack", ["最初の仕掛けと接続する"], ("一手目から仕組まれた誘導", "段階的な体験設計", "論証全体の隠れた骨格")),
            ("VICTOR_COST", "勝者が払った代償", "cost", ["一コマで示す", "勝利を相対化する"], ("友を失った策士の背中", "短期利益を手放した判断", "単純な結論を捨てた代償")),
        ],
        "invariants": [
            "全ての反転は冒頭で明示した規則の内側で成立させる",
            "三段の反転はそれぞれめくり直後の右ページに置き、最後の反転は見開きにする",
            "反転の強度は段ごとに増し、緊張度のピークは第三の反転に置く",
            "策士の内心は最後の開示まで独白させない",
            "反転の間には必ず緊張の谷（劣勢・疑念）を挟む",
        ],
        "guide": {
            "manga": "10ページ構成。p1（右）6コマ: 規則を図解的なMSと手元ECUで明示。p2（左）7コマ: 仕掛けと相手の観察を切り返しMS/CUで。めくって p3（右）3コマ: 第一の反転、相手の笑みCUをアオリで大きく。p4（左）6コマ: 劣勢、策士の視線ECUを一コマ。めくって p5（右）3コマ: 第二の反転、策士の静かなCUを俯瞰から。p6（左）6コマ: 相手の再反撃の兆しで推理を分岐させる。めくって p7〜p8 見開き: 最終手を手元ECUと全景ELSの合成で、ダッチアングル10度。p9（右）4コマで論理の解説、p10（左）3コマで代償と次の勝負の予告。",
            "marketing": "比較検討型のセールス文脈で、評価基準を冒頭で明示し、一見不利な提案（無料・制限付き）を置く。読者が見抜いたと思う地点で第一の反転、その制限自体が価値の証明だった第二の反転、最後に全体の体験設計を開示する第三の反転を置き、全て最初の評価基準で説明できることを確認させる。",
            "article": "論説では議論の前提となる定義を最初に明示し、反直観的な主張を置く。有力な反論で主張が崩れたように見せた後、その反論自体が主張を補強する事例だったと示し、最後に冒頭の反例が理論全体の決定的証拠だったと開示する。結論では単純な結論を捨てた代償にも触れる。",
        },
        "rhythm": {
            "page_count": 10,
            "panels_per_page": [6, 7, 3, 6, 3, 6, 1, 1, 4, 3],
            "emphasis_pages": [2, 4, 6, 7],
            "page_turn_reveals": [2, 4, 6],
            "spread_pages": [6],
            "silent_panels_ratio": 0.1,
            "pacing_notes": [
                "p1（右）: 6コマ。規則を図解的なコマで明示する",
                "p2（左）: 7コマ。策士の不可解な仕掛けと相手の観察を切り返す",
                "p3（右・めくり直後）: 3コマ。第一の反転を相手の笑みの大ゴマで示す",
                "p4（左）: 6コマ。劣勢の描写の中に策士の視線を一コマだけ置く",
                "p5（右・めくり直後）: 3コマ。第二の反転で劣勢が布石だったと開示する",
                "p6（左）: 6コマ。相手の再反撃の兆しで読者の推理を分岐させる",
                "p7（右・めくり直後）: 見開き大ゴマの右半分。最終手の手元を大きく置く",
                "p8（左）: 見開き大ゴマの左半分。最初の仕掛けとの接続を全景で示す",
                "p9（右）: 4コマ。規則に照らして論理を短く解説する",
                "p10（左）: 3コマ。勝者の代償と次の勝負の予告で引く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 06: 会話劇のカットバック（切り返し）リズム
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_06_dialogue_cutback",
        "title": "会話劇のカットバック（切り返し）リズム",
        "summary": "二人の位置関係を確立した上で肩越しの切り返しを刻み、はぐらかしで緊張を逃がしつつ間隔とショットを詰め、セリフのない沈黙の切り返しを経て、めくり直後にイマジナリーラインを越える構図で核心の一言を置く会話劇の骨格。",
        "tags": ["会話劇", "切り返し", "サブテクスト", "沈黙", "イマジナリーライン"],
        "lineage": ["ショット・リバースショットの古典的編集文法", "サブテクスト重視の対話劇", "180度ルールと軸越えの演出"],
        "waves": {
            "tension": [(0.0, 0.25), (0.1, 0.35), (0.2, 0.3), (0.3, 0.45), (0.4, 0.38), (0.5, 0.58), (0.6, 0.5), (0.7, 0.72), (0.8, 0.62), (0.85, 0.95), (1.0, 0.4)],
            "release": [(0.0, 0.2), (0.2, 0.3), (0.4, 0.32), (0.6, 0.28), (0.8, 0.15), (0.85, 0.1), (0.95, 0.6), (1.0, 0.75)],
            "audience_knowledge": [(0.0, 0.15), (0.2, 0.25), (0.4, 0.35), (0.6, 0.45), (0.8, 0.55), (0.85, 0.85), (1.0, 0.9)],
            "character_knowledge": [(0.0, 0.4), (0.2, 0.45), (0.4, 0.5), (0.6, 0.55), (0.8, 0.6), (0.85, 0.85), (1.0, 0.9)],
            "cognitive_load": [(0.0, 0.5), (0.3, 0.6), (0.5, 0.65), (0.7, 0.7), (0.8, 0.6), (0.85, 0.35), (1.0, 0.3)],
        },
        "nodes": [
            ("speaker_a", "agent", "concealing_speaker", "本心を隠して話す主体", "表層の言葉で核心を避け続け、沈黙の後に一言だけ本心を漏らす主体", 0.85, "SPEAKER_A"),
            ("speaker_b", "agent", "probing_speaker", "核心へ踏み込もうとする相手", "問いと切り返しで相手の防御を一枚ずつ剥がしていく主体", 0.8, "SPEAKER_B"),
            ("subtext", "lack", "subtext", "言葉にされない本心", "セリフの裏に伏せられ、表情と間でのみ示唆される情報", 0.9, "SUBTEXT"),
            ("understanding", "desire", "mutual_understanding", "互いに分かり合いたいという欲求", "両者と読者が共有する、本心に触れたいという渇望", 0.75, "UNDERSTANDING_GOAL"),
            ("imaginary_line", "symbol", "imaginary_line", "イマジナリーライン", "二人の位置関係を固定する視線の軸。核心の一言で初めて越えられる", 0.6, "AXIS_BREAK"),
        ],
        "edges": [
            ("speaker_b", "speaker_a", "opposition", 0.6, 0.0, "問いと回避の応酬", 0.1),
            ("speaker_a", "subtext", "asymmetry", 0.85, 0.8, "話者だけが本心を知る", 0.0),
            ("speaker_b", "subtext", "pursuit", 0.8, 0.0, "相手は本心へ踏み込もうとする", 0.2),
            ("speaker_a", "speaker_b", "dependency", 0.7, 0.0, "本心を隠しながらも相手を必要としている", 0.0),
            ("speaker_b", "speaker_a", "power_gradient", 0.55, 0.4, "問う側が会話の主導権を握っていく", 0.5),
            ("imaginary_line", "understanding", "embodiment", 0.65, 0.0, "軸を越える瞬間が理解を体現する", 0.85),
            ("subtext", "understanding", "opposition", 0.5, -0.3, "隠された本心が理解を阻む", 0.3),
        ],
        "beats": [
            ("b_two_shot", "premise", 0.0, 0.06, "位置関係の確立", "二人を同一コマに収め、イマジナリーラインと距離を確定させる", [], 0.0, 0.0, ["speaker_a", "speaker_b", "imaginary_line"], False),
            ("b_volley1", "escalation", 0.3, 0.1, "第一の応酬", "肩越しショットの切り返しで問いと回避を交互に刻む", ["b_two_shot"], 0.15, 0.05, ["speaker_a", "speaker_b"], False),
            ("b_deflect", "setup", 0.4, 0.05, "はぐらかし", "冗談や話題転換で緊張をいったん逃がし、本心の在処だけを示唆する", ["b_volley1"], -0.07, 0.05, ["speaker_a", "subtext"], False),
            ("b_volley2", "escalation", 0.5, 0.08, "第二の応酬", "切り返しの間隔を詰め、ショットサイズをMSからCUへ寄せる", ["b_deflect"], 0.2, 0.05, ["speaker_b", "subtext"], False),
            ("b_probe", "local_optimization_limit", 0.7, 0.06, "表層の言葉の限界", "言葉での攻防が尽き、これ以上の切り返しでは核心に届かない地点に達する", ["b_volley2"], 0.14, 0.05, ["speaker_b", "speaker_a"], False),
            ("b_pause", "transition", 0.8, 0.04, "沈黙の切り返し", "セリフのない表情CUの切り返しを左ページ末尾に置き、間で本心を示唆する", ["b_probe"], -0.1, 0.0, ["subtext", "imaginary_line"], True),
            ("b_key_line", "revelation", 0.85, 0.05, "核心の一言", "めくり直後にイマジナリーラインを越える構図で、本心の一言を置く", ["b_pause"], 0.33, 0.3, ["speaker_a", "subtext", "understanding"], True),
            ("b_reaction", "afterglow", 1.0, 0.1, "受け止めの余韻", "言葉を受け止めた相手の表情と、引きの二人を置いて余韻にする", ["b_key_line"], -0.55, 0.05, ["speaker_b", "understanding"], False),
        ],
        "mechanisms": [
            ("m_subtext_gap", "curiosity_gap", "サブテクストの情報ギャップ", "言葉と表情のずれが本心の存在を示唆し、読者は行間を読もうとする", "セリフと表情の不一致", "読者が能動的に本心を推測し会話に没入する", "情報ギャップ理論", 0.8, ["b_deflect", "b_pause"], ["subtext"]),
            ("m_rhythm_entrainment", "narrative_transportation", "切り返しリズムへの同調", "一定間隔の切り返しが読みのテンポを作り、間隔の短縮が緊張を体感させる", "肩越しショットの交互配置", "読者の呼吸が会話のテンポに同調する", "物語移入理論", 0.7, ["b_volley1", "b_volley2"], ["speaker_a", "speaker_b"]),
            ("m_axis_break", "prediction_error", "軸越えの予測誤差", "固定されていたイマジナリーラインを越える構図が、関係の変化を無意識に知覚させる", "核心の一言で反転する構図", "言葉以上に関係が変わったと感じられる", "予測符号化理論", 0.75, ["b_key_line"], ["imaginary_line"]),
            ("m_defender_empathy", "empathy_transfer", "本心を隠す者への共感", "隠さざるを得ない事情が示唆されると、読者は防御する側に共感する", "はぐらかしの裏の表情", "核心の一言が読者自身の告白のように響く", "共感の二過程モデル", 0.65, ["b_deflect", "b_key_line"], ["speaker_a"]),
        ],
        "anchor": "はぐらかした直後に一瞬だけ曇る、本心を隠す主体の表情",
        "drivers": ["言葉と表情のずれが示す本心", "詰まっていく切り返しのテンポ", "沈黙のコマに宿る言えない言葉"],
        "emotions": ["穏やか", "探り", "苛立ち", "切迫", "沈黙", "告白", "受容"],
        "template": "{{SPEAKER_A}}と{{SPEAKER_B}}を同一コマに収めて位置関係を確立し、肩越しの切り返しで{{SURFACE_TOPIC}}をめぐる応酬を刻む。{{SUBTEXT}}は表情と間でのみ示唆し、切り返しの間隔を詰めながらショットを寄せる。左ページ末尾に{{SILENT_EXCHANGE}}を置き、めくり直後にイマジナリーラインを越える{{AXIS_BREAK}}の構図で{{KEY_LINE}}を置いて、{{UNDERSTANDING_GOAL}}へ到達させる。",
        "slots": [
            ("SPEAKER_A", "本心を隠して話す主体", "agent", ["本心を独白させない"], ("強がる幼なじみ", "本音を言わない見込み客", "建前を語る当事者")),
            ("SPEAKER_B", "核心へ踏み込む相手", "agent", ["問いで主導権を握る"], ("問い詰める親友", "ヒアリングする担当者", "問いを立てる筆者")),
            ("SURFACE_TOPIC", "表層で交わされる話題", "setting", ["本題を避ける話題にする"], ("進路の話", "予算や機能の話", "表面的な論点")),
            ("SUBTEXT", "言葉にされない本心", "lack", ["セリフで直接言わない"], ("離れたくないという気持ち", "本当は失敗が怖いという不安", "論点の裏にある本当の問題")),
            ("SILENT_EXCHANGE", "セリフのない表情の切り返し", "emotion", ["左ページ末尾", "2コマ以上"], ("見つめ合うだけの二コマ", "問いの後の沈黙を示す一行", "結論の前の空白の段落")),
            ("AXIS_BREAK", "イマジナリーラインを越える構図", "reversal", ["核心の一言と同期させる"], ("背中側から回り込むカメラ", "視点を顧客側に切り替える一文", "論者の立場から読者の立場への転換")),
            ("KEY_LINE", "本心を漏らす核心の一言", "irony", ["短く", "それまでの話題と無関係に見える言葉で"], ("行かないでほしい", "本当は失敗が怖かった", "問題は手法ではなく恐れだった")),
            ("UNDERSTANDING_GOAL", "到達すべき相互理解", "goal", ["言葉で説明しすぎない"], ("二人が同じ方向を見る", "相談が本音の段階に入る", "読者が自分の問題として理解する")),
        ],
        "invariants": [
            "核心の一言までは本心をセリフで直接言わせない",
            "切り返しの間隔は終盤ほど短くし、ショットサイズを寄せていく",
            "核心の一言の直前に必ずセリフのない沈黙のコマを置く",
            "核心の一言はめくり直後に置き、イマジナリーラインを越える構図で描く",
            "フキダシは右から左へ読む順に配置し、話者の切り返しと視線誘導を一致させる",
        ],
        "guide": {
            "manga": "8ページ構成。p1（右）6コマ: 二人を収めたFSで位置関係を確立。p2〜p5は6〜7コマの肩越しショット（MS）の切り返しを基本に、フキダシを右の話者から左の話者へ配置して視線を流す。p3〜p4で切り返しの間隔を詰めショットをMSからCUへ寄せ、はぐらかしの直後に表情ECUを1コマ挟む。p6（左）5コマの最終2コマはセリフなしの表情CUの切り返し（望遠85mm）。めくって p7（右）2コマ: 上段の大ゴマで軸を越えた構図（逆側からのCU）に核心の一言。p8（左）4コマで受け止めの表情と引きのFSで余韻。",
            "marketing": "ヒアリング型の対話コンテンツで、表層の質問（機能・価格）と顧客の回答を交互に置き、回答の裏にある不安を表情描写や短い間で示唆する。沈黙を示す一行の後、視点を顧客側に切り替えて本音の一言を引き出し、その本音に応える提案へ自然に移行する。",
            "article": "インタビュー記事や対話形式の論考で、表層の論点の応酬を切り返し構成で示し、話者の言葉の端に本当の問題を示唆させる。結論の直前に空白の段落を置き、視点を読者自身の立場に転換して核心の一文を置くことで、問題が読者の自分事として着地する。",
        },
        "rhythm": {
            "page_count": 8,
            "panels_per_page": [6, 7, 6, 7, 6, 5, 2, 4],
            "emphasis_pages": [6],
            "page_turn_reveals": [6],
            "spread_pages": [],
            "silent_panels_ratio": 0.15,
            "pacing_notes": [
                "p1（右）: 6コマ。二人を収めたFSでイマジナリーラインを確立する",
                "p2（左）: 7コマ。肩越しショットの切り返しで第一の応酬",
                "p3（右）: 6コマ。はぐらかしで緊張を逃がし、表情ECUで本心を示唆する",
                "p4（左）: 7コマ。切り返しの間隔を詰め、MSからCUへ寄せる",
                "p5（右）: 6コマ。言葉の攻防が尽きる地点まで押し上げる",
                "p6（左）: 5コマ。最終2コマをセリフなしの表情の切り返しにする",
                "p7（右・めくり直後）: 2コマ。軸を越えた構図の大ゴマで核心の一言",
                "p8（左）: 4コマ。受け止めの表情と引きのFSで余韻を作る",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 07: 回想挿入と現在への復帰
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_07_flashback_insert_return",
        "title": "回想挿入と現在への復帰",
        "summary": "現在の危機の極点で時間を止め、象徴物を境に回想へ滑り込み、喪失以前の日常で失われたものの価値を体験させ、核心で約束の意味を開示した後、白枠の無音コマで時制を閉じてめくり直後に現在へ戻り反撃する骨格。",
        "tags": ["回想", "時制", "象徴物", "再起", "めくり"],
        "lineage": ["フラッシュバックによる動機の開示", "円環構造（象徴物の反復）", "危機の極点での時間停止"],
        "waves": {
            "tension": [(0.0, 0.65), (0.1, 0.75), (0.2, 0.45), (0.3, 0.32), (0.4, 0.3), (0.5, 0.42), (0.6, 0.6), (0.7, 0.55), (0.75, 0.95), (0.85, 0.8), (1.0, 0.5)],
            "release": [(0.0, 0.05), (0.2, 0.3), (0.4, 0.45), (0.6, 0.35), (0.7, 0.3), (0.75, 0.4), (0.9, 0.7), (1.0, 0.8)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.3), (0.4, 0.45), (0.6, 0.7), (0.75, 0.85), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.6), (0.2, 0.62), (0.4, 0.65), (0.6, 0.7), (0.75, 0.85), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.6), (0.2, 0.5), (0.4, 0.55), (0.6, 0.6), (0.7, 0.4), (0.75, 0.3), (1.0, 0.35)],
        },
        "nodes": [
            ("present_self", "agent", "present_protagonist", "現在の危機に立つ主体", "追い詰められた現在において、過去の意味を取り戻すことで再起する主体", 0.9, "PRESENT_SELF"),
            ("past_self", "agent", "past_self", "回想の中の主体", "喪失以前の時間を生きる、現在の主体のもう一つの姿", 0.7, "PAST_SELF"),
            ("present_threat", "agent", "present_threat", "現在の危機をもたらす力", "回想の前後で主体を追い詰める現在の圧力", 0.75, "PRESENT_THREAT"),
            ("trigger_object", "symbol", "memory_trigger", "回想の引き金となる象徴物", "現在と過去の両方に登場し、時制を橋渡しする具体物", 0.8, "TRIGGER_OBJECT"),
            ("lost_promise", "lack", "forgotten_meaning", "忘れられていた約束の意味", "読者には伏せられ、回想の核心で初めて開示される過去の意味", 0.85, "PAST_PROMISE"),
            ("resolve", "desire", "renewed_resolve", "過去に報いたいという欲求", "回想によって現在の主体に再点火される動機", 0.8, "RENEWED_RESOLVE"),
        ],
        "edges": [
            ("present_threat", "present_self", "power_gradient", 0.85, 0.8, "現在の危機が主体を圧倒している", 0.0),
            ("present_threat", "present_self", "opposition", 0.8, 0.0, "危機と主体の正面対立", 0.0),
            ("present_self", "lost_promise", "asymmetry", 0.7, 0.6, "主体は約束を知るが読者は知らない", 0.0),
            ("trigger_object", "lost_promise", "embodiment", 0.8, 0.0, "象徴物が約束の意味を宿す", 0.2),
            ("present_self", "past_self", "dependency", 0.65, 0.0, "現在の主体は過去の自分に支えられる", 0.6),
            ("resolve", "lost_promise", "pursuit", 0.75, 0.0, "再起の欲求が約束の回復を目指す", 0.6),
            ("present_self", "present_threat", "power_gradient", 0.8, 0.7, "回想後に主体が勢力を逆転する", 0.75),
        ],
        "beats": [
            ("b_present_crisis", "hook", 0.0, 0.06, "現在の危機", "主体が追い詰められた現在の状況から始め、敗北寸前の緊張を提示する", [], 0.3, 0.0, ["present_self", "present_threat"], False),
            ("b_peak_peril", "crisis", 0.1, 0.05, "敗北寸前の極点", "主体が膝をつく大ゴマで危機の極点を示し、現在の時間を止める", ["b_present_crisis"], 0.15, 0.0, ["present_self", "present_threat"], True),
            ("b_trigger", "transition", 0.2, 0.05, "回想の引き金", "象徴物のECUを境に、枠線の質感を変えて過去の時制へ滑り込む", ["b_peak_peril"], -0.3, 0.05, ["trigger_object"], False),
            ("b_memory_floor", "setup", 0.4, 0.1, "喪失以前の日常", "回想の中の穏やかな日常を描き、失われたものの価値を読者に体験させる", ["b_trigger"], -0.05, 0.1, ["past_self"], True),
            ("b_memory_core", "revelation", 0.6, 0.06, "約束の開示", "回想の核心で、象徴物に込められた約束の意味を初めて開示する", ["b_memory_floor"], 0.3, 0.25, ["lost_promise", "trigger_object", "past_self"], True),
            ("b_boundary", "transition", 0.7, 0.04, "時制の境界", "白枠の無音コマで回想を閉じ、左ページ末尾で現在への帰還を予告する", ["b_memory_core"], -0.08, 0.0, ["trigger_object"], False),
            ("b_return", "costly_breakthrough", 0.75, 0.06, "現在への復帰と反撃", "めくり直後に現在へ戻った主体が、約束の意味を背負って一撃を返す", ["b_boundary", "b_peak_peril"], 0.4, 0.15, ["present_self", "resolve", "present_threat"], True),
            ("b_afterglow", "afterglow", 1.0, 0.1, "過去と現在の重ね合わせ", "象徴物を現在の手の中で映し、過去の自分と現在の自分を一コマで重ねる", ["b_return"], -0.3, 0.1, ["trigger_object", "present_self"], False),
        ],
        "mechanisms": [
            ("m_withheld_past", "curiosity_gap", "伏せられた過去", "主体だけが知る過去の意味を読者に伏せ、回想の核心まで解釈の余白を残す", "現在の主体の不可解な執着", "読者が過去に何があったのかを知りたくなる", "情報ギャップ理論", 0.75, ["b_present_crisis", "b_memory_core"], ["lost_promise"]),
            ("m_value_of_loss", "loss_aversion", "喪失の価値の体験", "失われる前の日常を丁寧に見せることで、喪失の痛みを読者自身の損失として感じさせる", "穏やかな回想の日常", "現在の危機の重みが増し、反撃への願いが強まる", "損失回避（プロスペクト理論）", 0.7, ["b_memory_floor"], ["past_self"]),
            ("m_meaning_reframe", "cognitive_dissonance", "意味の再解釈", "回想の開示によって現在の行動の意味が書き換わり、不協和が解消される", "約束の開示と現在の反撃の接続", "主体の行動が必然として腑に落ちる", "認知的不協和理論", 0.65, ["b_memory_core", "b_return"], ["present_self", "lost_promise"]),
            ("m_catharsis_return", "catharsis", "帰還のカタルシス", "時制の境界で溜めた感情が、現在への復帰と同時に解放される", "白枠の無音コマからめくり直後の反撃", "涙と高揚が同時に訪れる", "カタルシス（緊張解放）", 0.8, ["b_boundary", "b_return"], ["resolve"]),
        ],
        "anchor": "膝をついた現在の主体が、手の中の象徴物を見つめる一瞬",
        "drivers": ["主体だけが知る過去への好奇心", "失われる前の日常の温度", "時制の境界の白いコマ"],
        "emotions": ["切迫", "絶望", "郷愁", "温かさ", "痛み", "決意", "高揚"],
        "template": "{{PRESENT_THREAT}}に追い詰められた{{PRESENT_SELF}}の危機の極点で時間を止め、{{TRIGGER_OBJECT}}を境に過去へ滑り込む。回想では{{PAST_SELF}}の穏やかな日常を描いて喪失の価値を体験させ、核心で{{PAST_PROMISE}}を開示する。白枠の無音コマで時制を閉じ、めくり直後に現在へ戻った主体が{{RENEWED_RESOLVE}}を背負って{{RETURN_ACTION}}を返す。",
        "slots": [
            ("PRESENT_SELF", "現在の危機に立つ主体", "agent", ["回想の前後で同一構図を反復する"], ("膝をつく主人公", "行き詰まった事業の担当者", "壁に当たった実践者")),
            ("PRESENT_THREAT", "現在の危機をもたらす力", "conflict", ["回想中は描かない"], ("勝ち誇る強敵", "迫る期限と競合", "解けない難問")),
            ("TRIGGER_OBJECT", "時制を橋渡しする象徴物", "symbol", ["現在と過去の両方に登場させる", "ECUで提示する"], ("古びたお守り", "創業時の最初の注文書", "最初に読んだ一冊の教科書")),
            ("PAST_SELF", "回想の中の主体", "agent", ["喪失以前の姿で描く"], ("笑っていた幼い日の主人公", "情熱に満ちた創業期の自分", "初学者だった頃の自分")),
            ("PAST_PROMISE", "忘れられていた約束の意味", "lack", ["回想の核心まで伏せる"], ("必ず守ると誓った言葉", "顧客と交わした最初の約束", "学び始めた本当の理由")),
            ("RENEWED_RESOLVE", "再点火された動機", "goal", ["言葉で説明しすぎない"], ("約束を果たすための再起", "原点に立ち返った再挑戦", "原理から考え直す決意")),
            ("RETURN_ACTION", "現在で返す反撃の行動", "reversal", ["めくり直後の大ゴマ", "回想前の構図を反転させる"], ("立ち上がりざまの一撃", "原点に基づく新しい提案", "原理から導いた解法")),
        ],
        "invariants": [
            "回想に入る前に現在の危機の極点で時間を止める",
            "回想の出入口には同一の象徴物を置き、枠線や地の色で時制を区別する",
            "回想の核心で初めて約束の意味を開示し、それまで伏せる",
            "現在への復帰はめくり直後の右ページに置き、回想前の構図を反転させる",
        ],
        "guide": {
            "manga": "9ページ構成。p1（右）5コマ: 現在の危機をMS・ダッチアングルで。p2（左）4コマ: 膝をつく主体の大ゴマ（俯瞰、広角24mm）で極点、最終コマに象徴物のECU。p3〜p5は枠線を細く柔らかくし地をトーンで変えた回想で、p3（右）5コマの穏やかな日常をFS・アイレベル、p4（左）4コマで関係を深め、p5（右）5コマで約束の核心をCU・望遠85mmで開示する。p6（左）3コマ: 最終コマを白枠の無音コマにして時制を閉じる。めくって p7（右）1コマ断ち切り: 現在へ戻った主体の反撃をアオリ・広角18mmで描き、回想前の俯瞰構図を反転させる。p8〜p9で象徴物を現在の手の中に重ねて余韻にする。",
            "marketing": "ブランドストーリー型LPで、顧客が直面する現在の課題を冒頭で示した後、創業期の原体験へ回想として入る。原体験の中で交わされた約束（理念）を核心として開示し、区切りの直後に現在へ戻って、その理念に基づく解決策を提示する。原体験を象徴する具体物を冒頭と結末の両方に置く。",
            "article": "エッセイ型の記事で、書き手が直面している現在の壁から始め、原点となった経験へ遡る。原点の記述の核心で、忘れていた学びの本当の理由を開示し、章を切り替えて現在の壁に戻り、原点から導いた解法を提示する。冒頭と結末に同じ具体物を置いて円環を作る。",
        },
        "rhythm": {
            "page_count": 9,
            "panels_per_page": [5, 4, 5, 4, 5, 3, 1, 3, 4],
            "emphasis_pages": [1, 6],
            "page_turn_reveals": [6],
            "spread_pages": [],
            "silent_panels_ratio": 0.2,
            "pacing_notes": [
                "p1（右）: 5コマ。現在の危機をダッチアングルで切迫させる",
                "p2（左）: 4コマ。膝をつく主体の大ゴマで極点、最終コマに象徴物のECU",
                "p3（右）: 5コマ。枠線と地の色を変えた回想、穏やかな日常",
                "p4（左）: 4コマ。過去の関係を深め喪失の価値を体験させる",
                "p5（右）: 5コマ。回想の核心で約束の意味を開示する",
                "p6（左）: 3コマ。最終コマを白枠の無音コマにして時制を閉じる",
                "p7（右・めくり直後）: 断ち切り1コマ。現在へ戻った主体の反撃",
                "p8（左）: 3コマ。反撃の結果と相手の動揺",
                "p9（右）: 4コマ。象徴物を現在の手に重ね、過去と現在を一つにする",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 08: ギャグの三段オチ（フリ・フリ・オチ）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_08_gag_three_beat",
        "title": "ギャグの三段オチ（フリ・フリ・オチ）",
        "summary": "同型のずれを二度反復して読者の予測を確立し、三度目の直前の無言の溜めで左ページを終え、めくり直後の大ゴマで型そのものを裏切る飛躍を見せ、直後のツッコミで笑いを解放して天丼の気配で余韻を残す骨格。",
        "tags": ["ギャグ", "三段オチ", "ツッコミ", "天丼", "緩急"],
        "lineage": ["三の法則（反復と変化の修辞）", "ボケとツッコミの対話芸", "不一致解消型ユーモア"],
        "waves": {
            "tension": [(0.0, 0.2), (0.15, 0.35), (0.25, 0.3), (0.4, 0.5), (0.5, 0.45), (0.6, 0.62), (0.65, 0.9), (0.75, 0.4), (0.85, 0.5), (1.0, 0.3)],
            "release": [(0.0, 0.2), (0.25, 0.35), (0.5, 0.4), (0.6, 0.2), (0.65, 0.3), (0.75, 0.95), (0.85, 0.7), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.3), (0.25, 0.45), (0.5, 0.6), (0.6, 0.65), (0.65, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.3), (0.25, 0.45), (0.5, 0.6), (0.6, 0.65), (0.65, 0.66), (0.75, 0.9), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.35), (0.25, 0.4), (0.5, 0.45), (0.6, 0.5), (0.65, 0.3), (0.75, 0.25), (1.0, 0.2)],
        },
        "nodes": [
            ("boke", "agent", "boke", "ずれた振る舞いをする主体（ボケ）", "常識の型を真剣に踏み外し、笑いの原因となる主体", 0.85, "BOKE"),
            ("tsukkomi", "agent", "tsukkomi", "ずれを指摘する相手（ツッコミ）", "読者の常識を代弁してずれを言語化し、笑いを解放する主体", 0.8, "TSUKKOMI"),
            ("pattern_expectation", "desire", "pattern_expectation", "繰り返しから生まれる読者の予測", "二度のフリで確立された型が、三度目も同じだろうという予測", 0.8, "ESTABLISHED_PATTERN"),
            ("common_sense", "symbol", "common_sense", "共有された常識の型", "ボケが踏み外し、ツッコミが守る基準線", 0.6, "COMMON_SENSE"),
            ("hidden_twist", "lack", "concealed_punchline", "伏せられたオチの発想", "三度目でのみ開示される、型を裏切る飛躍", 0.85, "PUNCHLINE"),
        ],
        "edges": [
            ("boke", "common_sense", "opposition", 0.7, 0.0, "ボケが常識を踏み外す", 0.15),
            ("tsukkomi", "common_sense", "dependency", 0.7, 0.0, "ツッコミは常識に依拠する", 0.0),
            ("tsukkomi", "boke", "power_gradient", 0.5, 0.4, "ツッコミが場の解釈権を握る", 0.75),
            ("pattern_expectation", "hidden_twist", "pursuit", 0.75, 0.0, "読者の予測がオチを先読みしようとする", 0.4),
            ("boke", "hidden_twist", "asymmetry", 0.8, 0.7, "ボケだけが三度目の飛躍を抱えている", 0.5),
            ("hidden_twist", "pattern_expectation", "betrayal", 0.85, 0.0, "オチが確立された型を裏切る", 0.65),
            ("common_sense", "tsukkomi", "embodiment", 0.55, 0.0, "常識の型がツッコミの言葉に体現される", 0.75),
        ],
        "beats": [
            ("b_furi1", "setup", 0.15, 0.06, "フリ一", "最初のずれを小さく見せ、ボケとツッコミの型を提示する", [], 0.15, 0.05, ["boke", "tsukkomi"], False),
            ("b_lull", "transition", 0.25, 0.04, "一拍の流し", "軽いツッコミで流し、型を日常の中に定着させる", ["b_furi1"], -0.05, 0.05, ["tsukkomi"], False),
            ("b_furi2", "escalation", 0.4, 0.06, "フリ二", "同じ型のずれをやや大きく繰り返し、三度目の予測を確立させる", ["b_lull"], 0.2, 0.05, ["boke", "pattern_expectation"], True),
            ("b_tame", "local_optimization_limit", 0.5, 0.05, "溜めの一拍", "三度目の直前に無言の小ゴマを置き、予測を最大化させたまま左ページを終える", ["b_furi2"], -0.05, 0.0, ["pattern_expectation", "boke"], True),
            ("b_ochi", "structural_twist", 0.65, 0.05, "オチ", "めくり直後の大ゴマで、予測の型そのものを裏切る飛躍を見せる", ["b_tame"], 0.45, 0.3, ["hidden_twist", "boke", "pattern_expectation"], True),
            ("b_tsukkomi", "payoff", 0.75, 0.05, "ツッコミによる解放", "ずれを的確な言葉で言語化し、緊張を笑いとして一挙に解放する", ["b_ochi"], -0.5, 0.25, ["tsukkomi", "common_sense"], True),
            ("b_tendon", "hook", 0.85, 0.05, "天丼の予告", "同じ飛躍がもう一度来そうな気配を小さく置き、笑いを持続させる", ["b_tsukkomi", "b_furi1"], 0.1, 0.0, ["boke"], False),
            ("b_coda", "afterglow", 1.0, 0.06, "余韻のゆるみ", "引きのコマで全員が脱力する絵を置き、次の話題へ流す", ["b_tendon"], -0.2, 0.0, ["boke", "tsukkomi"], False),
        ],
        "mechanisms": [
            ("m_incongruity", "prediction_error", "不一致の解消による笑い", "確立された型と飛躍の不一致が、ツッコミによる再解釈で解消されるとき笑いが生じる", "二度のフリと三度目の飛躍", "驚きが即座に理解へ変わる快", "不一致解消理論（ユーモアの認知理論）", 0.9, ["b_ochi", "b_tsukkomi"], ["hidden_twist", "pattern_expectation"]),
            ("m_rule_of_three", "anticipation_reward", "三の法則", "二度の反復で予測を確立し、三度目でそれを裏切ると最も効率よく驚きが生まれる", "同型のフリの二回反復", "三度目を待つ期待そのものが快になる", "三の法則（反復と変化の修辞学）", 0.8, ["b_furi1", "b_furi2", "b_tame"], ["pattern_expectation"]),
            ("m_relief", "catharsis", "緊張の放出としての笑い", "溜めで高めた緊張がツッコミの言語化によって一挙に放出される", "溜めの一拍とツッコミ", "身体的な脱力と笑い", "放出理論（緊張緩和としての笑い）", 0.7, ["b_tame", "b_tsukkomi"], ["tsukkomi"]),
            ("m_common_ground", "social_proof", "常識の共有", "ツッコミが読者の常識を代弁することで、読者は笑ってよいという合図を受け取る", "読者の内心と一致するツッコミの言葉", "笑いが安心と一体感を伴う", "社会的証明の原理", 0.55, ["b_tsukkomi"], ["common_sense", "tsukkomi"]),
        ],
        "anchor": "真顔でずれた行動を続けるボケの、悪意のない表情",
        "drivers": ["二度の反復で確立された型", "三度目直前の溜め", "読者の内心を代弁するツッコミ"],
        "emotions": ["軽い違和", "期待", "予測", "驚き", "爆笑", "脱力"],
        "template": "{{BOKE}}が{{COMMON_SENSE}}を小さく踏み外すフリ一と、同じ型をやや大きく繰り返すフリ二で{{ESTABLISHED_PATTERN}}を確立する。三度目の直前に{{TAME_PANEL}}を置いて左ページを終え、めくり直後の大ゴマで{{PUNCHLINE}}を見せる。{{TSUKKOMI}}が{{TSUKKOMI_LINE}}でずれを言語化して笑いを解放し、天丼の気配を残して余韻へ流す。",
        "slots": [
            ("BOKE", "ずれた振る舞いをする主体", "agent", ["本人は真剣", "悪意を持たせない"], ("真顔で天然な先輩", "斜め上の機能を勧める営業", "極端な仮説を掲げる論者")),
            ("COMMON_SENSE", "共有された常識の型", "setting", ["読者が即座に理解できる常識"], ("挨拶は普通にするもの", "商品説明は丁寧であるべき", "結論は根拠から導くもの")),
            ("ESTABLISHED_PATTERN", "二度の反復で確立する型", "goal", ["同型を二回だけ", "二回目はやや大きく"], ("毎回挨拶で物を壊す", "毎回少しずつ大げさになる宣伝文", "毎回飛躍する事例紹介")),
            ("TAME_PANEL", "三度目直前の溜め", "emotion", ["無言", "左ページ最終コマ"], ("満を持して扉に手をかける無言コマ", "そして三つ目の特典はという一行", "では三つ目の事例を見てみようという一文")),
            ("PUNCHLINE", "型を裏切る三度目の飛躍", "reversal", ["型そのものを裏切る", "めくり直後の大ゴマ"], ("扉ではなく壁ごと現れる", "三つ目の特典は特典が無いことだった", "三つ目の事例は書き手自身の失敗だった")),
            ("TSUKKOMI", "ずれを言語化する相手", "agent", ["読者の常識を代弁する"], ("冷静な後輩", "購入を迷う読者の声", "反論役の聞き手")),
            ("TSUKKOMI_LINE", "解放をもたらす言語化の一言", "irony", ["短く的確に"], ("そこは扉から入れ", "特典じゃなくて説明だそれは", "それは事例ではなく告白だ")),
        ],
        "invariants": [
            "フリは同型を二回、二回目をやや大きくする",
            "オチの直前に無言の溜めを置き、オチはめくり直後の大ゴマに置く",
            "ツッコミはオチの直後一コマ以内に置き、ずれを短く言語化する",
            "オチは型の延長ではなく、型そのものを裏切る",
        ],
        "guide": {
            "manga": "4ページ構成。p1（右）4コマ: フリ一をMS・アイレベルで小さく描き、ツッコミは小さめのフキダシにする。p2（左）5コマ: フリ二をやや大きいコマで反復し、最終コマはボケの無言のCU（溜め）。めくって p3（右）2コマ: 上段の大ゴマでオチをFS・広角24mm・ダッチアングル10度で描き、下段の小コマで即座にツッコミ（叫びフキダシ）。p4（左）4コマ: 天丼の気配を小さく置き、最終コマを引きのFSで全員の脱力にする。",
            "marketing": "親しみを重視する広告コピーやSNS向けコンテンツで、同型の小さな誇張を二回繰り返して型を作り、三回目で型そのものを裏切る（特典が無いことが特典、など）。直後に読者の内心を代弁するツッコミの一言を置き、笑いと同時に商品の誠実さを印象づける。",
            "article": "硬い論考の息抜きとなる段落で、同じ型の事例を二つ並べ、三つ目で予想を裏切る事例（書き手自身の失敗など）を置く。直後に読者の内心を代弁する短い一文でずれを言語化し、親近感を作ってから本論へ戻る。",
        },
        "rhythm": {
            "page_count": 4,
            "panels_per_page": [4, 5, 2, 4],
            "emphasis_pages": [2],
            "page_turn_reveals": [2],
            "spread_pages": [],
            "silent_panels_ratio": 0.1,
            "pacing_notes": [
                "p1（右）: 4コマ。フリ一を小さく見せ、ボケとツッコミの型を提示する",
                "p2（左）: 5コマ。フリ二をやや大きく反復し、最終コマを無言の溜めにする",
                "p3（右・めくり直後）: 2コマ。大ゴマのオチと直後の小コマのツッコミ",
                "p4（左）: 4コマ。天丼の気配と引きの脱力コマで余韻へ流す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 09: 感情決壊（表情クローズアップ漸進）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_09_emotional_breakdown_closeup",
        "title": "感情決壊（表情クローズアップ漸進）",
        "summary": "平静を装う主体を全身ショットで距離をとって描き、痛みの文脈を読者にだけ知らせ、ショットサイズを一コマずつ寄せて抑圧の圧力を高め、作り笑いの一コマを挟んでめくり直後の大ゴマで決壊させ、受け止める他者と引きの俯瞰で静けさを残す骨格。",
        "tags": ["感情", "決壊", "クローズアップ", "共感", "カタルシス"],
        "lineage": ["ショットサイズ漸進による情動の増幅", "抑圧と解放の劇作法", "受容による回復の物語"],
        "waves": {
            "tension": [(0.0, 0.3), (0.15, 0.36), (0.3, 0.44), (0.45, 0.55), (0.55, 0.62), (0.65, 0.7), (0.7, 0.66), (0.8, 0.96), (0.9, 0.6), (1.0, 0.35)],
            "release": [(0.0, 0.1), (0.3, 0.08), (0.6, 0.05), (0.7, 0.1), (0.8, 0.5), (0.9, 0.8), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.4), (0.2, 0.5), (0.4, 0.6), (0.6, 0.7), (0.8, 0.85), (1.0, 0.9)],
            "character_knowledge": [(0.0, 0.4), (0.2, 0.45), (0.4, 0.5), (0.6, 0.55), (0.8, 0.85), (1.0, 0.9)],
            "cognitive_load": [(0.0, 0.4), (0.3, 0.35), (0.5, 0.3), (0.7, 0.25), (0.8, 0.2), (0.9, 0.25), (1.0, 0.3)],
        },
        "nodes": [
            ("sufferer", "agent", "suppressing_protagonist", "感情を堪える主体", "悲しみや悔しさを抑え込み、平静を装い続ける主体", 0.9, "SUFFERER"),
            ("witness_other", "agent", "accepting_other", "受け止める他者", "主体の決壊を否定せず受け止める存在。決壊の安全地帯となる", 0.7, "ACCEPTING_OTHER"),
            ("suppressed_emotion", "lack", "suppressed_emotion", "抑圧された感情", "言葉にされず、表情の綻びとして漏れ出す本当の感情", 0.9, "SUPPRESSED_EMOTION"),
            ("face", "symbol", "face_closeup", "寄っていく顔のクローズアップ", "ショットサイズの漸進によって感情の圧力を可視化する装置", 0.75, "FACE_PROGRESSION"),
            ("release_need", "desire", "need_for_release", "泣いてよいと許されたい欲求", "主体と読者が共有する、抑圧からの解放への渇望", 0.8, "PERMISSION"),
            ("trigger_word", "symbol", "trigger_word", "決壊を誘う一言", "他者が何気なく発する、抑圧の堤を崩す言葉", 0.7, "TRIGGER_WORD"),
        ],
        "edges": [
            ("sufferer", "suppressed_emotion", "opposition", 0.85, 0.0, "主体が自分の感情を押し殺す", 0.0),
            ("suppressed_emotion", "sufferer", "power_gradient", 0.8, 0.6, "抑えた感情が次第に主体を上回る", 0.65),
            ("sufferer", "witness_other", "dependency", 0.65, 0.0, "主体は受け止める他者を必要としている", 0.75),
            ("release_need", "suppressed_emotion", "pursuit", 0.75, 0.0, "解放への欲求が感情の出口を求める", 0.3),
            ("witness_other", "suppressed_emotion", "asymmetry", 0.6, 0.5, "他者は主体の痛みに気づいている", 0.4),
            ("face", "suppressed_emotion", "embodiment", 0.8, 0.0, "寄っていく顔が抑圧の圧力を体現する", 0.5),
            ("trigger_word", "sufferer", "power_gradient", 0.6, 0.7, "何気ない一言が堤を崩す", 0.7),
        ],
        "beats": [
            ("b_distance", "premise", 0.0, 0.08, "距離のある全身", "FSで平静を装う主体を描き、まだ感情との距離があることを示す", [], 0.0, 0.0, ["sufferer"], False),
            ("b_pain_context", "setup", 0.15, 0.06, "痛みの文脈", "主体が失ったもの・傷ついた理由を短く提示し、読者だけが痛みの大きさを知る", ["b_distance"], 0.06, 0.1, ["suppressed_emotion", "witness_other"], False),
            ("b_approach", "escalation", 0.45, 0.12, "ショットの漸進", "MSからCUへと寄せながら、手の震えや唇の噛みしめを一コマずつ挟む", ["b_pain_context"], 0.2, 0.05, ["face", "sufferer"], False),
            ("b_suppress_limit", "local_optimization_limit", 0.65, 0.05, "堪える限界", "表情のECUで堪える力が限界に達したことを示す", ["b_approach"], 0.08, 0.0, ["face", "suppressed_emotion"], True),
            ("b_forced_smile", "transition", 0.7, 0.04, "作り笑いの一コマ", "左ページ末尾で主体が無理に笑って見せ、緊張をわずかに逸らす", ["b_suppress_limit"], -0.04, 0.0, ["sufferer"], True),
            ("b_break", "climax", 0.8, 0.06, "決壊", "めくり直後の大ゴマで、他者の一言を受けて涙と叫びが溢れ出す", ["b_forced_smile"], 0.3, 0.15, ["trigger_word", "sufferer", "suppressed_emotion"], True),
            ("b_embrace", "payoff", 0.9, 0.06, "受け止める手", "他者の手や肩をCUで描き、決壊が受け入れられたことを示す", ["b_break"], -0.36, 0.05, ["witness_other", "release_need"], False),
            ("b_pullback", "afterglow", 1.0, 0.08, "引きの俯瞰", "二人を俯瞰のELSで小さく置き、感情の嵐が過ぎた静けさを残す", ["b_embrace"], -0.25, 0.05, ["sufferer", "witness_other"], False),
        ],
        "mechanisms": [
            ("m_closeup_intimacy", "empathy_transfer", "接近による共感の増幅", "ショットサイズが寄るほど読者と主体の心理的距離が縮み、感情の伝染が強まる", "FSからECUへの漸進", "読者自身の胸が詰まる感覚", "情動伝染理論", 0.85, ["b_approach", "b_suppress_limit"], ["face", "sufferer"]),
            ("m_pressure_irony", "dramatic_irony", "読者だけが知る痛み", "主体が平静を装う間、読者だけが痛みの大きさを知っている", "痛みの文脈と平静な表情の対置", "早く泣いていいと願う気持ちが募る", "劇的アイロニー", 0.7, ["b_pain_context", "b_forced_smile"], ["suppressed_emotion"]),
            ("m_breakdown_catharsis", "catharsis", "決壊のカタルシス", "抑圧が限界まで高まった後の決壊が、読者の感情も同時に解放する", "作り笑いからめくり直後の涙", "読者も涙ぐむ解放感", "カタルシス（感情の浄化）", 0.9, ["b_break", "b_embrace"], ["release_need"]),
            ("m_safe_base", "reciprocity", "受容による安全基地", "受け止める他者の存在が、決壊を破綻ではなく回復の始まりとして意味づける", "受け止める手のクローズアップ", "悲しみの後に温かさが残る", "愛着理論（安全基地）", 0.6, ["b_embrace", "b_pullback"], ["witness_other"]),
        ],
        "anchor": "平静を装う主体の、噛みしめた唇と震える指先",
        "drivers": ["寄っていくカメラが縮める心理的距離", "読者だけが知る痛みの大きさ", "作り笑いの一コマが生む痛切さ"],
        "emotions": ["平静", "違和", "痛み", "圧迫", "痛切", "決壊", "安堵"],
        "template": "{{SUFFERER}}を全身ショットで距離をとって描き、{{PAIN_CONTEXT}}を読者にだけ知らせる。{{FACE_PROGRESSION}}で主体へ一コマずつ寄りながら{{SUPPRESSED_EMOTION}}の圧力を高め、左ページ末尾に{{FORCED_SMILE}}を置く。めくり直後の大ゴマで{{TRIGGER_WORD}}を受けた決壊を描き、{{ACCEPTING_OTHER}}が受け止めることで{{PERMISSION}}を与え、引きの俯瞰で静けさを残す。",
        "slots": [
            ("SUFFERER", "感情を堪える主体", "agent", ["決壊まで泣かせない"], ("気丈に振る舞う主人公", "弱音を吐けない経営者", "失敗を認められない実践者")),
            ("PAIN_CONTEXT", "主体の痛みの文脈", "lack", ["短く提示し読者にだけ伝える"], ("大切な人との別れ", "長年の努力が報われなかった事実", "信じてきた方法の破綻")),
            ("FACE_PROGRESSION", "顔への段階的な接近", "symbol", ["FS→MS→CU→ECUの順", "細部を一コマずつ挟む"], ("震える手から目元へ寄るカット", "悩みの具体を一つずつ深掘る構成", "抽象から具体へ段階的に寄る記述")),
            ("SUPPRESSED_EMOTION", "抑圧された本当の感情", "emotion", ["言葉にさせない"], ("本当は寂しいという気持ち", "本当は助けてほしいという本音", "本当は悔しいという感情")),
            ("FORCED_SMILE", "決壊直前の作り笑い", "irony", ["左ページ最終コマ", "小さく置く"], ("大丈夫だよと笑う顔", "なんとかなっていますという返事", "まあ想定内ですという強がり")),
            ("TRIGGER_WORD", "決壊を誘う一言", "reversal", ["何気ない言葉にする"], ("もう頑張らなくていい", "一人で抱えなくていいんです", "失敗してもいいという一文")),
            ("ACCEPTING_OTHER", "受け止める他者", "agent", ["否定も説教もしない"], ("黙って肩を抱く友人", "寄り添う相談相手", "共感を示す書き手")),
            ("PERMISSION", "解放の許し", "goal", ["言葉で説明しすぎない"], ("泣いてもいいという許し", "弱音を吐ける関係", "失敗を受け入れる余地")),
        ],
        "invariants": [
            "ショットサイズは決壊に向けて単調に寄せ、決壊後に一気に引く",
            "決壊の直前に作り笑いなど緊張を逸らす一コマを置く",
            "決壊はめくり直後の大ゴマに置く",
            "受け止める他者は否定や説教をしない",
            "決壊前の区間ではフキダシを減らし、表情の細部で圧力を示す",
        ],
        "guide": {
            "manga": "6ページ構成。p1（右）5コマ: FS・アイレベルで平静を装う主体と痛みの文脈。p2（左）4コマ: MSへ寄り手の震えのECUを1コマ。p3（右）4コマ: CUへ寄り唇の噛みしめのECU、フキダシを減らす。p4（左）5コマ: 目元のECU・望遠135mmで堪える限界を示し、最終コマに作り笑いのCUを小さく置く。めくって p5（右）1コマ断ち切り: 他者の一言と涙の決壊をCU〜ECU・アオリ気味で描き、叫びフキダシを添える。p6（左）3コマ: 受け止める手のCU→俯瞰のELSで二人を小さく置く。",
            "marketing": "共感型のLPやストーリー広告で、悩みを抱えながら平気なふりをする顧客像を描き、悩みの具体を段階的に深掘りする。大丈夫ですという強がりを一度見せた直後に、一人で抱えなくていいという一言を置き、受け止める姿勢（相談窓口や寄り添うサービス）を示す。",
            "article": "体験談型の記事で、平静を装っていた時期の記述から始め、抽象から具体へ段階的に寄りながら当時の本当の感情に近づく。強がりの一文を置いた後、章の切れ目で感情を認めた瞬間を描き、読者にも同じ許しを与える結びにする。",
        },
        "rhythm": {
            "page_count": 6,
            "panels_per_page": [5, 4, 4, 5, 1, 3],
            "emphasis_pages": [4],
            "page_turn_reveals": [4],
            "spread_pages": [],
            "silent_panels_ratio": 0.35,
            "pacing_notes": [
                "p1（右）: 5コマ。全身ショットで平静を装う主体と痛みの文脈を示す",
                "p2（左）: 4コマ。MSへ寄り、手の震えのECUを一コマ挟む",
                "p3（右）: 4コマ。CUへ寄り、フキダシを減らして表情で圧力を示す",
                "p4（左）: 5コマ。目元のECUで限界を示し、最終コマに作り笑いを小さく置く",
                "p5（右・めくり直後）: 断ち切り1コマの決壊",
                "p6（左）: 3コマ。受け止める手のCUから俯瞰のELSへ引く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 10: 追跡・チェイスの加速コマ
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_10_chase_accelerating_panels",
        "title": "追跡・チェイスの加速コマ",
        "summary": "発見から逃走を始め、コマを段階的に細く傾けて速度を上げ、追跡者側のカットバックで読者だけに距離の縮まりを見せ、偽の安堵の直後に再捕捉させ、めくり直後の大ゴマで行き止まりを全景で示して代償を伴う跳躍で振り切らせる骨格。",
        "tags": ["追跡", "チェイス", "加速", "カットバック", "代償"],
        "lineage": ["クロスカッティングによる追跡劇", "コマ幅と傾斜による速度表現", "偽の安堵からの再捕捉"],
        "waves": {
            "tension": [(0.0, 0.4), (0.1, 0.5), (0.2, 0.6), (0.25, 0.52), (0.4, 0.7), (0.45, 0.6), (0.6, 0.8), (0.65, 0.72), (0.85, 0.98), (0.95, 0.5), (1.0, 0.55)],
            "release": [(0.0, 0.1), (0.25, 0.25), (0.45, 0.3), (0.65, 0.3), (0.85, 0.05), (0.95, 0.75), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.3), (0.25, 0.4), (0.45, 0.5), (0.65, 0.6), (0.85, 0.75), (0.95, 0.95), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.3), (0.25, 0.35), (0.45, 0.4), (0.65, 0.45), (0.85, 0.75), (0.95, 0.95), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.5), (0.25, 0.55), (0.45, 0.6), (0.65, 0.65), (0.85, 0.5), (0.95, 0.35), (1.0, 0.3)],
        },
        "nodes": [
            ("runner", "agent", "pursued", "追われる主体", "逃げ道を探しながら走り続け、最後に代償を払って境界を越える主体", 0.9, "RUNNER"),
            ("pursuer", "agent", "pursuer", "迫る追跡者", "主体との距離を着実に詰める、止まらない圧力", 0.85, "PURSUER"),
            ("escape_goal", "desire", "escape", "逃げ切りたいという欲求", "主体と読者が共有する、追跡から解放されたいという渇望", 0.8, "ESCAPE_GOAL"),
            ("closing_distance", "lack", "unknown_gap", "主体が知らない距離の縮まり", "カットバックで読者だけに示される、追跡者との距離の真実", 0.75, "CLOSING_GAP"),
            ("speed_lines", "symbol", "speed_frame", "細く傾いていくコマ", "コマ幅の縮小と傾斜で速度と焦燥を可視化する装置", 0.6, "ACCELERATING_FRAMES"),
            ("dead_end", "symbol", "dead_end", "行き止まりの全景", "めくり直後に現れる逃走路の終端。跳躍か捕縛かの二択を突きつける", 0.85, "DEAD_END"),
        ],
        "edges": [
            ("pursuer", "runner", "opposition", 0.9, 0.0, "追う者と追われる者", 0.0),
            ("pursuer", "runner", "power_gradient", 0.75, 0.6, "追跡者が速度で優位に立つ", 0.2),
            ("runner", "escape_goal", "pursuit", 0.9, 0.0, "主体は逃げ切りを目指す", 0.0),
            ("closing_distance", "runner", "asymmetry", 0.7, -0.6, "主体は距離の縮まりを知らない", 0.4),
            ("speed_lines", "closing_distance", "embodiment", 0.6, 0.0, "細く傾くコマが迫る距離を体現する", 0.6),
            ("runner", "speed_lines", "dependency", 0.5, 0.0, "主体の焦燥はコマの速度に支えられる", 0.2),
            ("dead_end", "escape_goal", "opposition", 0.8, -0.5, "行き止まりが逃走の望みを断つ", 0.85),
            ("runner", "pursuer", "power_gradient", 0.7, 0.5, "代償を払った跳躍で主体が追跡を振り切る", 0.95),
        ],
        "beats": [
            ("b_spotted", "hook", 0.0, 0.05, "発見", "主体が追跡者に見つかる瞬間から始め、逃走を強制する", [], 0.3, 0.0, ["runner", "pursuer"], False),
            ("b_first_burst", "escalation", 0.2, 0.06, "最初の加速", "横長のコマから縦長の細いコマへ切り替え、走りの速度を上げる", ["b_spotted"], 0.2, 0.05, ["runner", "speed_lines"], False),
            ("b_shake_off", "transition", 0.25, 0.04, "一瞬の振り切り", "角を曲がって一度距離が開いたように見せる", ["b_first_burst"], -0.08, 0.05, ["runner"], False),
            ("b_cross_cut", "setup", 0.4, 0.06, "追跡者側のカットバック", "読者だけに追跡者の接近を見せ、主体の知らない距離を示す", ["b_shake_off"], 0.18, 0.15, ["pursuer", "closing_distance"], False),
            ("b_false_escape", "false_victory", 0.45, 0.05, "撒いたという錯覚", "主体が物陰で息をつき、逃げ切ったと思う小ゴマを置く", ["b_cross_cut"], -0.1, 0.05, ["runner", "escape_goal"], True),
            ("b_corner", "crisis", 0.6, 0.08, "再捕捉", "追跡者が再び現れ、コマが傾き始めて焦燥が加速する", ["b_false_escape"], 0.2, 0.05, ["pursuer", "speed_lines"], False),
            ("b_dead_end", "revelation", 0.85, 0.05, "めくり直後の行き止まり", "右ページ上段の大ゴマで逃走路の終端を全景で見せる", ["b_corner"], 0.26, 0.15, ["dead_end", "runner"], True),
            ("b_leap", "costly_breakthrough", 0.95, 0.05, "代償を伴う跳躍", "傷や持ち物の喪失と引き換えに境界を越え、追跡を振り切る", ["b_dead_end"], -0.48, 0.2, ["runner", "pursuer", "escape_goal"], True),
            ("b_aftermath", "afterglow", 1.0, 0.04, "息を整える余韻", "失ったものを確かめる主体の小ゴマで、次の追跡の予感を残す", ["b_leap"], 0.05, 0.0, ["runner"], False),
        ],
        "mechanisms": [
            ("m_cross_cut_suspense", "dramatic_irony", "カットバックによるサスペンス", "追う側と追われる側を交互に見せ、読者だけが距離の縮まりを知る", "追跡者側のカットバック", "主体に危険を知らせたい焦燥が生まれる", "サスペンス理論（劇的アイロニー）", 0.85, ["b_cross_cut", "b_false_escape"], ["closing_distance", "pursuer"]),
            ("m_frame_tempo", "narrative_transportation", "コマ形状による速度体験", "コマ幅の縮小と傾斜が読みの速度を上げ、読者の身体に追跡の速度を移す", "細く傾いていくコマの連続", "読む行為そのものが走る感覚に同調する", "物語移入理論", 0.75, ["b_first_burst", "b_corner"], ["speed_lines"]),
            ("m_false_relief", "prediction_error", "偽の安堵の裏切り", "逃げ切ったという錯覚の直後に再捕捉を置き、安堵の予測を裏切る", "物陰で息をつく小ゴマ", "再捕捉の衝撃が倍増する", "予測符号化理論", 0.7, ["b_false_escape", "b_corner"], ["runner"]),
            ("m_loss_cost", "loss_aversion", "代償の重み", "突破に失うものを伴わせると、成功が安易でなく重みのあるものとして記憶される", "跳躍と引き換えの喪失", "安堵の中に痛みが残る", "損失回避（プロスペクト理論）", 0.6, ["b_leap", "b_aftermath"], ["runner"]),
        ],
        "anchor": "振り返る余裕もなく走り続ける主体の、荒い呼吸と足元",
        "drivers": ["読者だけが見ている追跡者の接近", "細く傾いていくコマの速度", "撒いたと思った直後の再捕捉"],
        "emotions": ["驚き", "焦燥", "安堵", "恐怖", "絶望", "決死", "放心"],
        "template": "{{RUNNER}}が{{PURSUER}}に見つかる瞬間から始め、{{ACCELERATING_FRAMES}}でコマを細く傾けて速度を上げる。カットバックで読者だけに{{CLOSING_GAP}}を見せ、{{FALSE_RELIEF}}で一度安堵させた直後に再捕捉させる。めくり直後の大ゴマで{{DEAD_END}}を全景で見せ、{{ESCAPE_GOAL}}のために{{BREAKTHROUGH_COST}}を払う跳躍で振り切らせる。",
        "slots": [
            ("RUNNER", "追われる主体", "agent", ["振り返らせすぎない"], ("路地を駆ける少年", "期限に追われるプロジェクト担当", "時代の変化に追われる業界人")),
            ("PURSUER", "迫る追跡者", "conflict", ["決して止まらない"], ("無表情の追跡者", "迫る競合と期限", "加速する技術の変化")),
            ("ACCELERATING_FRAMES", "細く傾いていくコマの連続", "setting", ["段階的にコマ幅を縮める", "傾斜を少しずつ強める"], ("縦長の細コマの連打", "短い見出しと短文の連続", "段落を短くしていく構成")),
            ("CLOSING_GAP", "主体が知らない距離の縮まり", "lack", ["読者にだけ示す"], ("背後の角を曲がる影", "競合がすでに追いついている数値", "変化が既に足元に来ている兆候")),
            ("FALSE_RELIEF", "逃げ切ったという錯覚", "irony", ["小ゴマで短く"], ("物陰で息をつく横顔", "ひとまず安心という一文", "まだ余裕があるという楽観")),
            ("DEAD_END", "逃走路の終端", "reversal", ["めくり直後の大ゴマ", "全景で見せる"], ("断崖の手前で途切れる路地", "従来手段では間に合わないという事実", "従来の延長線上に解がないという結論")),
            ("ESCAPE_GOAL", "逃げ切りたいという欲求", "goal", ["跳躍の動機として明示する"], ("仲間のもとへ帰ること", "期限内の達成", "変化を生き延びること")),
            ("BREAKTHROUGH_COST", "突破のために払う代償", "cost", ["失うものを具体的に示す"], ("大切な荷物を手放す", "既存の手法を捨てる決断", "過去の成功体験を手放すこと")),
        ],
        "invariants": [
            "コマは追跡の進行に応じて段階的に細く傾けていく",
            "追跡者側のカットバックで読者だけに距離の縮まりを見せる",
            "偽の安堵を一度置いてから再捕捉させる",
            "行き止まりはめくり直後の大ゴマで全景として見せる",
            "突破には必ず具体的な代償を伴わせる",
        ],
        "guide": {
            "manga": "8ページ構成。p1（右）5コマ: 発見の瞬間をMS・アイレベルで。p2〜p3（6コマ）: 横長コマから縦長の細コマへ切り替え、FSの走りと足元ECUを交互に置き、p3で角を曲がる一瞬の振り切りを描く。p4（左）7コマ: 追跡者側のカットバックを俯瞰で挟み、最終コマは物陰で息をつく横顔CU（偽の安堵）。p5〜p6（7〜8コマ）: 再捕捉、コマを5〜15度傾けて斜めのガターで速度を上げ、p6（左）最終コマで前方を見る主体の目ECU。めくって p7（右）2コマ: 上段の大ゴマで行き止まりの全景（ELS・広角18mm・俯瞰）、下段で跳躍の決断CU。p8（左）4コマ: 代償を伴う跳躍と、失ったものを確かめる余韻。",
            "marketing": "危機訴求型のLPで、顧客が変化に追われる状況を短い見出しと短文の連続で加速させ、競合や期限が迫っていることを数値で示す。一度まだ余裕があると思わせた後に再び危機を提示し、従来手段では間に合わないという事実を大きく示した上で、既存の手法を手放す決断を代償として明示し、突破策を提示する。",
            "article": "変化を論じる記事で、段落を次第に短くして変化の加速を体感させ、読者がまだ大丈夫と思いがちな楽観を一度提示してから否定する。従来の延長線上に解がないという結論を章の切れ目の直後に置き、過去の成功体験を手放すことを代償として明確にした上で、新しい方針を示す。",
        },
        "rhythm": {
            "page_count": 8,
            "panels_per_page": [5, 6, 6, 7, 7, 8, 2, 4],
            "emphasis_pages": [6],
            "page_turn_reveals": [6],
            "spread_pages": [],
            "silent_panels_ratio": 0.1,
            "pacing_notes": [
                "p1（右）: 5コマ。発見の瞬間から逃走を始める",
                "p2（左）: 6コマ。横長コマから縦長の細コマへ切り替えて加速する",
                "p3（右）: 6コマ。角を曲がる一瞬の振り切りで距離が開いたように見せる",
                "p4（左）: 7コマ。追跡者側のカットバックを挟み、最終コマを偽の安堵にする",
                "p5（右）: 7コマ。再捕捉。コマを傾けて斜めのガターで速度を上げる",
                "p6（左）: 8コマ。最も細いコマで焦燥を極め、最終コマで前方を見る目",
                "p7（右・めくり直後）: 2コマ。行き止まりの全景の大ゴマと跳躍の決断",
                "p8（左）: 4コマ。代償を伴う跳躍と、失ったものを確かめる余韻",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 11: 新キャラ登場の溜めと全身見せ
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_11_new_character_entrance",
        "title": "新キャラ登場の溜めと全身見せ",
        "summary": "姿より先に噂を届け、影・足元・コマ外の声で部分だけを段階的に示し、周囲が一斉に言葉を失う沈黙の小ゴマを左ページ末尾に置いて、めくり直後の大ゴマでアオリの全身を見せ、力の一端と宙吊りの関係で次へ引く登場の骨格。",
        "tags": ["新キャラ", "登場", "全身見せ", "溜め", "アオリ"],
        "lineage": ["スター登場の遅延演出", "部分提示による期待の構築", "周囲の反応による格付け"],
        "waves": {
            "tension": [(0.0, 0.3), (0.1, 0.4), (0.2, 0.48), (0.3, 0.56), (0.4, 0.64), (0.45, 0.6), (0.5, 0.95), (0.6, 0.75), (0.7, 0.82), (0.85, 0.55), (1.0, 0.6)],
            "release": [(0.0, 0.1), (0.4, 0.05), (0.5, 0.3), (0.6, 0.4), (0.7, 0.3), (0.85, 0.6), (1.0, 0.5)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.2), (0.4, 0.3), (0.45, 0.32), (0.5, 0.7), (0.7, 0.8), (1.0, 0.85)],
            "character_knowledge": [(0.0, 0.3), (0.2, 0.4), (0.4, 0.5), (0.45, 0.55), (0.5, 0.72), (0.7, 0.82), (1.0, 0.85)],
            "cognitive_load": [(0.0, 0.45), (0.2, 0.5), (0.4, 0.55), (0.5, 0.3), (0.7, 0.45), (1.0, 0.4)],
        },
        "nodes": [
            ("newcomer", "agent", "newcomer", "登場する新たな存在", "噂と気配だけで先行し、めくり直後に全身で姿を現す主体", 0.95, "NEWCOMER"),
            ("onlookers", "agent", "onlookers", "反応する周囲の人物たち", "新たな存在を知っており、その反応によって格を伝える群れ", 0.6, "ONLOOKERS"),
            ("protagonist", "agent", "protagonist", "既存の主体", "新たな存在との関係が今後の物語を左右する視点人物", 0.7, "EXISTING_PROTAGONIST"),
            ("reputation", "symbol", "reputation", "先行する噂", "姿より先に届く評判。期待値を膨らませる装置", 0.7, "RUMOR"),
            ("identity_gap", "lack", "unseen_identity", "見えない全貌", "影・足元・声だけが示され、全身が伏せられた状態", 0.85, "PARTIAL_GLIMPSES"),
            ("anticipation", "desire", "reader_anticipation", "全身を見たいという読者の欲求", "部分の積み重ねで膨らむ、姿を確かめたいという期待", 0.8, "FULL_REVEAL"),
        ],
        "edges": [
            ("onlookers", "identity_gap", "asymmetry", 0.75, 0.6, "周囲は正体を知り、読者は知らない", 0.1),
            ("anticipation", "reputation", "dependency", 0.7, 0.0, "読者の期待は噂に依存する", 0.0),
            ("anticipation", "newcomer", "pursuit", 0.85, 0.0, "読者の視線が全身を追う", 0.2),
            ("newcomer", "onlookers", "power_gradient", 0.8, 0.8, "新たな存在が場を支配する", 0.45),
            ("newcomer", "protagonist", "opposition", 0.6, 0.0, "既存の主体との緊張関係", 0.7),
            ("reputation", "newcomer", "embodiment", 0.65, 0.0, "噂が存在の格を体現する", 0.0),
            ("newcomer", "protagonist", "power_gradient", 0.6, 0.5, "初対面で格の差を示す", 0.7),
            ("identity_gap", "anticipation", "opposition", 0.5, -0.4, "伏せられた全貌が期待を焦らす", 0.3),
        ],
        "beats": [
            ("b_rumor", "premise", 0.0, 0.06, "噂の先行", "周囲の会話で新たな存在の評判だけを先に届ける", [], 0.0, 0.05, ["reputation", "onlookers"], False),
            ("b_presence", "setup", 0.2, 0.08, "気配の到来", "影・足音・ざわめきなど部分の描写で接近を示す", ["b_rumor"], 0.18, 0.05, ["identity_gap", "newcomer"], False),
            ("b_voice", "escalation", 0.4, 0.05, "コマ外からの声", "フキダシの尾がコマの外を指す声だけを先に聞かせる", ["b_presence"], 0.16, 0.05, ["newcomer", "identity_gap"], True),
            ("b_hush", "transition", 0.45, 0.04, "場の沈黙", "左ページ末尾で周囲が一斉に振り向き言葉を失う小ゴマを置く", ["b_voice"], -0.04, 0.0, ["onlookers"], True),
            ("b_full_body", "revelation", 0.5, 0.06, "全身見せ", "めくり直後の大ゴマで全身をアオリのFSで見せ、噂の格を視覚で確定させる", ["b_hush"], 0.35, 0.38, ["newcomer", "anticipation"], True),
            ("b_power_glimpse", "escalation", 0.7, 0.06, "力の一端", "能力や人格の一端を一挙動だけ見せ、底知れなさを残す", ["b_full_body"], 0.07, 0.1, ["newcomer", "onlookers"], False),
            ("b_position", "payoff", 0.85, 0.06, "関係図への配置", "既存の主体と視線を交わし、味方か敵か宙吊りのまま関係図に置く", ["b_power_glimpse", "b_rumor"], -0.27, 0.05, ["newcomer", "protagonist"], False),
            ("b_hook_next", "hook", 1.0, 0.04, "次への引き", "新たな存在の意味深な一言で次話への関心を引く", ["b_position"], 0.05, 0.0, ["newcomer"], False),
        ],
        "mechanisms": [
            ("m_delayed_reveal", "curiosity_gap", "部分提示による期待の膨張", "部分だけを積み重ねると、読者は全体像を補完しようとし期待が膨らむ", "影・足音・声の段階的提示", "全身を見た瞬間の満足が増幅される", "情報ギャップ理論", 0.85, ["b_presence", "b_voice"], ["identity_gap", "anticipation"]),
            ("m_social_reaction", "social_proof", "周囲の反応による格付け", "周囲が沈黙し振り向く反応が、存在の格を言葉より強く伝える", "場の沈黙の小ゴマ", "読者が登場人物の重要性を直感的に理解する", "社会的証明の原理", 0.75, ["b_hush", "b_full_body"], ["onlookers"]),
            ("m_authority_pose", "authority", "アオリの全身による威光", "アオリのFSと大ゴマの面積が存在の権威を視覚的に付与する", "めくり直後の全身大ゴマ", "存在に畏怖と魅力を同時に感じる", "権威の原理", 0.7, ["b_full_body"], ["newcomer"]),
            ("m_open_relation", "zeigarnik", "宙吊りの関係", "敵か味方か決めないまま関係図に置くと、未完了の関心が次話へ持ち越される", "既存の主体との視線の交差", "次の登場を待ち望む", "未完了効果（ツァイガルニク効果）", 0.6, ["b_position", "b_hook_next"], ["protagonist", "newcomer"]),
        ],
        "anchor": "噂を聞いて振り向く周囲の人物たちの、息を呑む表情",
        "drivers": ["姿より先に届く噂", "部分だけが示される焦らし", "周囲の沈黙が伝える格"],
        "emotions": ["好奇", "期待", "緊張", "静寂", "畏怖", "魅了", "警戒"],
        "template": "{{RUMOR}}を周囲の会話で先に届け、{{PARTIAL_GLIMPSES}}で{{NEWCOMER}}の接近を部分だけ示す。コマ外からの声を聞かせた後、左ページ末尾で{{ONLOOKERS}}が一斉に言葉を失う沈黙の小ゴマを置き、めくり直後の大ゴマで{{FULL_REVEAL}}を果たす。{{POWER_GLIMPSE}}で底知れなさを残し、{{EXISTING_PROTAGONIST}}との関係を宙吊りのまま関係図に置く。",
        "slots": [
            ("RUMOR", "姿より先に届く噂・評判", "evidence", ["具体的な武勇や評価を一つ含める"], ("百人を一人で退けたという噂", "業界で話題の新サービスという評判", "注目の新理論という前評判")),
            ("PARTIAL_GLIMPSES", "部分だけの提示", "lack", ["影・足元・声・周囲の反応に限定"], ("長い影と重い足音", "利用者の断片的な声", "引用される一節だけ")),
            ("NEWCOMER", "登場する新たな存在", "agent", ["全身はめくり直後まで見せない"], ("転校してきた最強の剣士", "新たに登場する製品", "新たに提示する概念")),
            ("ONLOOKERS", "反応する周囲の人物", "agent", ["一斉に反応させる"], ("ざわめく生徒たち", "注目する業界関係者", "議論を見守る専門家")),
            ("FULL_REVEAL", "全身見せの大ゴマ", "reversal", ["めくり直後", "アオリのFS"], ("校門に立つ全身のアオリ", "製品の全体像を示すメインビジュアル", "概念の全体図")),
            ("POWER_GLIMPSE", "能力・人格の一端", "symbol", ["一挙動だけ", "全てを見せない"], ("指一本で扉を止める", "一つの機能だけを実演する", "一つの応用例だけを示す")),
            ("EXISTING_PROTAGONIST", "関係を結ぶ既存の主体", "agent", ["敵味方を確定させない"], ("視線を返す主人公", "既存の顧客", "従来の理論の支持者")),
        ],
        "invariants": [
            "全身はめくり直後の大ゴマまで見せない",
            "部分提示は影・足元・声・周囲の反応の順に段階的に近づける",
            "全身見せの直前に周囲の沈黙の小ゴマを置く",
            "全身見せはアオリのFSで、視線誘導の終点に顔を置く",
        ],
        "guide": {
            "manga": "5ページ構成。p1（右）5コマ: 噂を語る周囲の会話をMSで。p2（左）6コマ: 長い影（俯瞰）・足元ECU・ざわめく群衆MS、コマ外を指すフキダシの声を置き、最終コマは一斉に振り向く群衆の沈黙CU（無音）。めくって p3（右）1コマ断ち切り: 全身見せをFS・アオリ・広角24mmで描き、右上の背景から左下の顔へ視線を落とす。p4（左）4コマ: 力の一端を一挙動だけ、ECUとMSで。p5（右）4コマ: 既存の主体との視線の交差をCUの切り返しで置き、意味深な一言で引く。",
            "marketing": "新製品のローンチLPで、先行する評判（利用者の声や話題性）を最初に置き、製品の全体像は伏せたまま部分的な特徴や断片的な声を段階的に示す。注目が高まったところでメインビジュアルとして全体像を大きく開示し、一つの機能だけを実演して底知れなさを残し、既存顧客との関係（乗り換えや併用）を示唆して次の行動へつなぐ。",
            "article": "新しい概念を紹介する記事で、前評判や引用される一節を先に示し、概念の全体図は後に回す。部分的な応用例を段階的に提示して関心を高めた後、章の切れ目で概念の全体図を掲げ、一つの応用例だけを深掘りして、従来理論との関係を問いとして残す。",
        },
        "rhythm": {
            "page_count": 5,
            "panels_per_page": [5, 6, 1, 4, 4],
            "emphasis_pages": [2],
            "page_turn_reveals": [2],
            "spread_pages": [],
            "silent_panels_ratio": 0.15,
            "pacing_notes": [
                "p1（右）: 5コマ。周囲の会話で噂を先に届ける",
                "p2（左）: 6コマ。影・足元・声で接近を示し、最終コマを群衆の沈黙にする",
                "p3（右・めくり直後）: 断ち切り1コマの全身見せ。アオリのFS",
                "p4（左）: 4コマ。力の一端を一挙動だけ見せる",
                "p5（右）: 4コマ。既存の主体との視線の交差と意味深な一言で引く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 12: ホラー：違和感の漸増と最終ページの戦慄
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_12_horror_creeping_dread",
        "title": "ホラー：違和感の漸増と最終ページの戦慄",
        "summary": "穏やかな日常の背景の片隅に異常を少しずつ大きく描いて読者だけに気づかせ、主体の合理化と否定できない痕跡を経て偽りの安堵で緊張を抜き、最後のめくり直後に安堵と同構図の背後へ存在の全貌を現して説明も救済もなく終える骨格。",
        "tags": ["ホラー", "違和感", "戦慄", "めくり", "未解決"],
        "lineage": ["背景に潜む恐怖の演出", "偽りの安堵からの戦慄", "未解決の結末による恐怖の持続"],
        "waves": {
            "tension": [(0.0, 0.2), (0.15, 0.28), (0.3, 0.36), (0.4, 0.44), (0.5, 0.4), (0.6, 0.55), (0.7, 0.62), (0.75, 0.4), (0.85, 0.5), (0.9, 1.0), (1.0, 0.85)],
            "release": [(0.0, 0.3), (0.3, 0.2), (0.5, 0.25), (0.7, 0.1), (0.75, 0.55), (0.85, 0.35), (0.9, 0.0), (1.0, 0.05)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.25), (0.4, 0.4), (0.6, 0.55), (0.75, 0.58), (0.9, 0.95), (1.0, 0.98)],
            "character_knowledge": [(0.0, 0.1), (0.2, 0.12), (0.4, 0.18), (0.6, 0.25), (0.75, 0.3), (0.9, 0.9), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.4), (0.2, 0.5), (0.4, 0.6), (0.6, 0.65), (0.75, 0.4), (0.9, 0.35), (1.0, 0.3)],
        },
        "nodes": [
            ("victim", "agent", "unaware_victim", "違和に気づかない主体", "日常を信じ、背景の異常を合理化し続ける視点人物", 0.85, "VICTIM"),
            ("presence", "agent", "uncanny_presence", "正体不明の存在", "背景の片隅に少しずつ姿を現し、最終ページで正体を見せる脅威", 0.9, "UNCANNY_PRESENCE"),
            ("anomaly_trail", "symbol", "anomaly_trail", "背景に紛れた違和の連鎖", "ページを追うごとに少しずつ大きくなる、読者だけが気づく異常", 0.8, "BACKGROUND_ANOMALIES"),
            ("reader_dread", "desire", "reader_dread", "気づいてほしいという読者の願い", "主体に危険を知らせたいが届かない、見る者の焦燥", 0.75, "READER_DREAD"),
            ("hidden_truth", "lack", "hidden_truth", "日常の裏の真相", "最終ページのめくりまで伏せられる、日常そのものが侵食されていたという事実", 0.9, "TERRIFYING_TRUTH"),
        ],
        "edges": [
            ("presence", "victim", "opposition", 0.85, 0.0, "存在が主体の日常を侵食する", 0.6),
            ("presence", "victim", "power_gradient", 0.8, 0.9, "存在は常に主体の背後にいる", 0.9),
            ("reader_dread", "victim", "asymmetry", 0.8, 0.8, "読者だけが違和に気づいている", 0.15),
            ("anomaly_trail", "presence", "embodiment", 0.8, 0.0, "違和の連鎖が存在の接近を体現する", 0.15),
            ("reader_dread", "anomaly_trail", "pursuit", 0.75, 0.0, "読者の視線が背景の異常を探す", 0.2),
            ("victim", "hidden_truth", "dependency", 0.6, 0.0, "主体の安心は真相を知らないことに依存する", 0.0),
            ("hidden_truth", "victim", "betrayal", 0.85, 0.0, "信じていた日常そのものが主体を裏切る", 0.9),
        ],
        "beats": [
            ("b_ordinary", "premise", 0.0, 0.06, "穏やかな日常", "何の変哲もない日常を丁寧に描き、読者に安心の基準を与える", [], 0.0, 0.0, ["victim"], False),
            ("b_anomaly1", "setup", 0.15, 0.06, "最初の違和", "背景の片隅に、説明のつかない小さな異常を一つだけ描く", ["b_ordinary"], 0.08, 0.1, ["anomaly_trail", "reader_dread"], False),
            ("b_anomaly_grows", "escalation", 0.4, 0.1, "違和の漸増", "ページを追うごとに異常を少しずつ大きく、近く描く", ["b_anomaly1"], 0.16, 0.15, ["anomaly_trail", "presence"], False),
            ("b_rationalize", "transition", 0.5, 0.05, "合理化", "主体が違和に気づきかけるが、自分を納得させて日常へ戻る", ["b_anomaly_grows"], -0.04, -0.02, ["victim", "hidden_truth"], False),
            ("b_evidence", "escalation", 0.7, 0.06, "否定できない痕跡", "主体の身の回りに否定しがたい痕跡が現れ、緊張が高まる", ["b_rationalize"], 0.22, 0.1, ["presence", "victim"], False),
            ("b_false_relief", "false_victory", 0.75, 0.05, "偽りの安堵", "痕跡が勘違いだったと説明され、主体も読者もいったん安堵する", ["b_evidence"], -0.22, 0.02, ["victim"], True),
            ("b_turn_reveal", "revelation", 0.9, 0.05, "めくりの戦慄", "最後のめくり直後、安堵の場面のすぐ背後に存在の全貌が現れる", ["b_false_relief", "b_anomaly1"], 0.5, 0.4, ["presence", "hidden_truth", "victim"], True),
            ("b_residue", "afterglow", 1.0, 0.06, "解決されない残像", "説明も救済もないまま、存在の視線だけを残して終える", ["b_turn_reveal"], -0.15, 0.03, ["presence", "reader_dread"], False),
        ],
        "mechanisms": [
            ("m_hidden_in_plain_sight", "dramatic_irony", "背景に潜む異常", "読者だけが背景の異常に気づく構図が、主体の無防備さを痛切にする", "背景の片隅の小さな異常", "読者が気づけと願う焦燥と恐怖", "サスペンス理論（劇的アイロニー）", 0.85, ["b_anomaly1", "b_anomaly_grows"], ["anomaly_trail", "reader_dread"]),
            ("m_uncanny", "prediction_error", "不気味さの漸増", "日常の予測から少しずつ逸脱する描写が、説明できない不快感を蓄積させる", "異常の段階的な拡大", "言語化できない恐怖が募る", "不気味なもの（不気味の谷）仮説", 0.8, ["b_anomaly_grows", "b_evidence"], ["presence"]),
            ("m_false_safety", "contrast_effect", "安堵との対比", "偽りの安堵の直後に戦慄を置くと、恐怖の落差が最大化する", "痕跡の勘違い説明", "防御が解けた瞬間の戦慄", "対比効果", 0.75, ["b_false_relief", "b_turn_reveal"], ["victim"]),
            ("m_unresolved", "zeigarnik", "解決されない恐怖", "説明も救済も与えずに終えることで、恐怖が読後も持続する", "存在の視線だけを残す結末", "読み終えた後も背後が気になる", "未完了効果（ツァイガルニク効果）", 0.7, ["b_residue"], ["presence"]),
            ("m_turn_terror", "anticipation_reward", "めくり行為の恐怖化", "めくりたくないのにめくってしまう予期が、ページ操作そのものを恐怖体験に変える", "最終ページ直前の静かな左ページ", "めくる手が震える体験", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.65, ["b_false_relief", "b_turn_reveal"], ["reader_dread"]),
        ],
        "anchor": "異常に気づかず穏やかに暮らす主体の、無防備な後ろ姿",
        "drivers": ["読者だけが気づく背景の異常", "主体の合理化への苛立ち", "最終ページをめくる手の恐怖"],
        "emotions": ["安心", "違和", "不安", "焦燥", "安堵", "戦慄", "残像"],
        "template": "{{VICTIM}}の穏やかな日常を丁寧に描いた上で、背景の片隅に{{BACKGROUND_ANOMALIES}}を少しずつ大きく描き、{{READER_DREAD}}を募らせる。主体の{{RATIONALIZATION}}で一度日常に戻し、否定できない痕跡の後に{{FALSE_RELIEF}}で安堵させる。最後のめくり直後に、安堵の場面のすぐ背後に{{UNCANNY_PRESENCE}}の全貌を見せ、{{TERRIFYING_TRUTH}}を説明も救済もなく突きつける。",
        "slots": [
            ("VICTIM", "違和に気づかない主体", "agent", ["異常の方向を見させない"], ("一人暮らしを始めた学生", "安定を信じる事業者", "現状維持を信じる読者")),
            ("BACKGROUND_ANOMALIES", "背景に紛れる異常の連鎖", "evidence", ["毎ページ少しずつ大きく近く", "主体には気づかせない"], ("窓の外の人影が毎晩近づく", "少しずつ悪化する解約率", "少しずつ増える例外事例")),
            ("READER_DREAD", "読者の焦燥", "emotion", ["主体と共有させない"], ("早く気づいてという願い", "このままでは危ないという予感", "前提が崩れつつあるという直感")),
            ("RATIONALIZATION", "主体による合理化", "irony", ["もっともらしい説明にする"], ("気のせいだと笑う", "季節要因だろうという分析", "例外は誤差だという解釈")),
            ("FALSE_RELIEF", "偽りの安堵", "irony", ["最後のめくり直前に置く", "読者も安堵させる"], ("物音の正体が猫だったと分かる", "数値が一時的に回復する", "反例が測定ミスだったと説明される")),
            ("UNCANNY_PRESENCE", "正体不明の存在", "conflict", ["最後のめくりまで全貌を見せない"], ("背後に立つ長い影", "構造的な市場の変化", "理論を根底から覆す現象")),
            ("TERRIFYING_TRUTH", "日常の裏の真相", "reversal", ["説明しすぎない", "救済を与えない"], ("最初からずっと部屋の中にいた", "顧客はすでに離れ始めていた", "前提そのものが誤っていた")),
        ],
        "invariants": [
            "異常は背景に描き、ページを追うごとに少しずつ大きく近くする",
            "主体には最終ページまで異常の全貌を見せない",
            "最後のめくりの直前に偽りの安堵を置く",
            "戦慄の開示は最後のめくり直後の右ページに置き、説明も救済も与えない",
            "異常の描写はZ字の視線誘導の終点ではなく、コマの片隅（視線の通過点）に置く",
        ],
        "guide": {
            "manga": "10ページ構成。p1〜p2（5コマ）: 穏やかな日常をFS・アイレベルで丁寧に描く。p3〜p6（5〜6コマ）: 各ページ1コマだけ背景の片隅に異常を置き、ページごとに少しずつ大きく近くする。異常はZ字の視線の通過点に置き、主体は常に逆方向を向かせる。p5で主体の合理化、p6で否定できない痕跡をCUで描く。p7（右）5コマ: 痕跡が勘違いだったと分かる安堵の会話。p8（左）4コマ: 静かな日常に戻る無音のコマで締める。めくって p9（右）1コマ断ち切り: 安堵の場面と同じ構図のまま、主体の背後に存在の全貌をELS・広角18mm・ダッチアングル10度で描く。p10（左）2コマ: 存在の目のECUと、何も知らない主体の後ろ姿で終える。",
            "marketing": "リスク啓発型のコンテンツで、安定して見える現状を丁寧に描いた上で、見過ごされがちな小さな兆候（解約率や問い合わせの変化）を段階的に示す。一時的な回復で読者を安心させた直後に、構造的な変化がすでに進行していた事実を大きく示し、対策の必要性を強く印象づける。",
            "article": "警鐘型の論考で、現状がうまくいっているという記述の中に、少しずつ増える例外事例を挟む。例外を誤差とする合理的な解釈を一度紹介して読者を安心させた後、章の切れ目で前提そのものが誤っていたことを示すデータを提示し、結論を急がず問いを残して終える。",
        },
        "rhythm": {
            "page_count": 10,
            "panels_per_page": [5, 5, 6, 5, 6, 6, 5, 4, 1, 2],
            "emphasis_pages": [8],
            "page_turn_reveals": [8],
            "spread_pages": [],
            "silent_panels_ratio": 0.4,
            "pacing_notes": [
                "p1（右）: 5コマ。穏やかな日常を丁寧に描き安心の基準を作る",
                "p2（左）: 5コマ。日常の続き。背景はまだ正常",
                "p3（右）: 6コマ。背景の片隅に最初の小さな異常を一コマだけ置く",
                "p4（左）: 5コマ。異常を少し大きく近く描く",
                "p5（右）: 6コマ。主体が気づきかけて合理化する",
                "p6（左）: 6コマ。否定できない痕跡をCUで示す",
                "p7（右）: 5コマ。痕跡が勘違いだったと説明され安堵する",
                "p8（左）: 4コマ。静かな日常に戻る無音のコマで締める",
                "p9（右・めくり直後）: 断ち切り1コマ。安堵と同構図の背後に存在の全貌",
                "p10（左）: 2コマ。存在の目と何も知らない主体の後ろ姿で終える",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 13: 競技の時間引き延ばし（瞬間の分解）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_13_sports_time_dilation",
        "title": "競技の時間引き延ばし（瞬間の分解）",
        "summary": "勝負の状況で一瞬の重さを定義し、始動の直後に背景を省略した小ゴマ連打へ切り替えて主観時間を引き延ばし、独白で判断の論理を積み、迷いの割り込みと決断を経て、動作の頂点で結果を伏せたままめくり直後の大ゴマで時間を再び流す骨格。",
        "tags": ["競技", "スポーツ", "時間操作", "独白", "瞬間"],
        "lineage": ["主観時間の引き延ばし（スローモーションの分解）", "内的独白による判断の共有", "動作の頂点での静止"],
        "waves": {
            "tension": [(0.0, 0.5), (0.1, 0.6), (0.2, 0.7), (0.3, 0.75), (0.4, 0.72), (0.5, 0.8), (0.6, 0.78), (0.7, 0.86), (0.8, 0.82), (0.85, 0.99), (1.0, 0.45)],
            "release": [(0.0, 0.1), (0.3, 0.05), (0.5, 0.08), (0.7, 0.05), (0.85, 0.2), (0.95, 0.85), (1.0, 0.9)],
            "audience_knowledge": [(0.0, 0.3), (0.3, 0.45), (0.5, 0.55), (0.7, 0.62), (0.85, 0.9), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.3), (0.3, 0.5), (0.5, 0.62), (0.7, 0.7), (0.85, 0.9), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.5), (0.2, 0.6), (0.4, 0.7), (0.6, 0.75), (0.8, 0.7), (0.85, 0.35), (1.0, 0.3)],
        },
        "nodes": [
            ("athlete", "agent", "athlete", "勝負の一瞬に臨む主体", "一秒にも満たない瞬間の中で、技術と迷いと決断を同時に抱える主体", 0.9, "ATHLETE"),
            ("rival", "agent", "rival_competitor", "同じ瞬間を争う相手", "主体と同じ一瞬を共有し、互いの判断が結果を分ける競争相手", 0.75, "RIVAL"),
            ("frozen_moment", "symbol", "frozen_moment", "引き延ばされた一瞬", "多数のコマに分解され、内的独白と多視点で満たされる時間の断面", 0.85, "FROZEN_MOMENT"),
            ("winning_desire", "desire", "victory", "勝ちたいという欲求", "主体・観客・読者が共有する勝利への渇望", 0.8, "DECISIVE_RESOLVE"),
            ("outcome_unknown", "lack", "unknown_result", "まだ誰も知らない結果", "瞬間の分解が続く間、宙吊りにされた勝敗", 0.8, "RESULT_QUESTION"),
            ("spectators", "agent", "spectators", "見守る観客", "主体の一瞬を外側から見つめ、その反応で瞬間の重さを伝える群れ", 0.5, "SPECTATORS"),
        ],
        "edges": [
            ("athlete", "rival", "opposition", 0.85, 0.0, "同じ一瞬を争う対立", 0.0),
            ("rival", "athlete", "power_gradient", 0.6, 0.3, "相手がわずかに先行している", 0.2),
            ("athlete", "winning_desire", "pursuit", 0.9, 0.0, "主体は勝利を追う", 0.0),
            ("athlete", "outcome_unknown", "asymmetry", 0.55, 0.4, "主体だけが手応えの予感を持つ", 0.7),
            ("frozen_moment", "winning_desire", "embodiment", 0.75, 0.0, "引き延ばされた一瞬が勝利への渇望を体現する", 0.3),
            ("spectators", "athlete", "dependency", 0.55, 0.0, "観客の感情は主体の一瞬に委ねられている", 0.3),
            ("athlete", "rival", "power_gradient", 0.7, 0.6, "最後の判断で主体が逆転する", 0.85),
        ],
        "beats": [
            ("b_situation", "premise", 0.0, 0.06, "勝負の状況", "残り時間・点差・位置関係を簡潔に示し、一瞬の重さを定義する", [], 0.0, 0.05, ["athlete", "rival", "spectators"], False),
            ("b_launch", "escalation", 0.2, 0.06, "始動", "主体が動き出す一コマで、時間が引き延ばされる直前の速度を見せる", ["b_situation"], 0.2, 0.05, ["athlete"], False),
            ("b_freeze", "setup", 0.3, 0.05, "時間停止の宣言", "背景を飛ばした小ゴマ連打に切り替え、一瞬が引き延ばされたことを示す", ["b_launch"], 0.05, 0.05, ["frozen_moment"], False),
            ("b_inner_logic", "escalation", 0.5, 0.1, "内的独白の論理", "主体の判断の根拠を独白で積み上げ、技術と読みを読者に共有する", ["b_freeze"], 0.08, 0.1, ["athlete", "frozen_moment"], False),
            ("b_doubt", "crisis", 0.6, 0.05, "迷いの一瞬", "失敗の記憶や相手の気配が割り込み、判断が揺らぐ", ["b_inner_logic"], -0.02, 0.0, ["athlete", "rival"], True),
            ("b_resolve", "local_optimization_limit", 0.7, 0.05, "決断", "迷いを振り切る決断の言葉で、思考の限界から身体の動作へ移る", ["b_doubt"], 0.08, 0.05, ["athlete", "winning_desire"], False),
            ("b_last_frame", "transition", 0.8, 0.04, "最後のコマ止め", "左ページ末尾で動作の頂点を静止画のように置き、結果を伏せる", ["b_resolve"], -0.04, 0.0, ["frozen_moment", "outcome_unknown"], True),
            ("b_result", "climax", 0.85, 0.05, "結果の大ゴマ", "めくり直後の大ゴマで時間が再び流れ出し、結果が一挙に示される", ["b_last_frame", "b_inner_logic"], 0.17, 0.3, ["athlete", "rival", "outcome_unknown"], True),
            ("b_time_resumes", "afterglow", 1.0, 0.08, "時間の再開", "観客の歓声と音が戻り、引き延ばされていた時間が日常の速度に復帰する", ["b_result"], -0.54, 0.05, ["spectators", "athlete"], False),
        ],
        "mechanisms": [
            ("m_time_dilation", "narrative_transportation", "主観時間の引き延ばし", "一瞬を多数のコマに分解すると、読者の主観時間が引き延ばされ瞬間に没入する", "背景を飛ばした小ゴマ連打", "現実には一秒の出来事を長く体験する", "物語移入理論", 0.85, ["b_freeze", "b_inner_logic"], ["frozen_moment"]),
            ("m_shared_logic", "identity_projection", "判断の共有による自己投影", "主体の判断の論理を独白で共有すると、読者は自分が判断しているように感じる", "内的独白の積み重ね", "結果が自分の判断の結果のように感じられる", "自己投影（同一化）理論", 0.75, ["b_inner_logic", "b_resolve"], ["athlete"]),
            ("m_frozen_result", "zeigarnik", "結果の伏せ", "動作の頂点で静止させて結果を伏せると、未完了の緊張が極大化する", "最後のコマ止め", "めくらずにはいられない", "未完了効果（ツァイガルニク効果）", 0.8, ["b_last_frame", "b_result"], ["outcome_unknown"]),
            ("m_release_tempo", "catharsis", "時間の再開による解放", "引き延ばされた時間が日常の速度に戻る瞬間、溜めた緊張が歓声とともに解放される", "観客の歓声と音の回帰", "全身の力が抜ける解放感", "カタルシス（緊張解放）", 0.7, ["b_result", "b_time_resumes"], ["spectators"]),
        ],
        "anchor": "一瞬の中で迷いを振り切ろうとする主体の、見開いた目",
        "drivers": ["引き延ばされた一瞬の中の独白", "割り込む迷いと失敗の記憶", "動作の頂点で止められた結果"],
        "emotions": ["緊迫", "集中", "迷い", "決意", "息止め", "歓喜", "放心"],
        "template": "{{GAME_SITUATION}}で一瞬の重さを定義し、{{ATHLETE}}の始動の直後に{{FROZEN_MOMENT}}へ切り替えて時間を引き延ばす。独白で判断の論理を積み上げ、{{DOUBT_INTRUSION}}で揺らがせた後、{{DECISIVE_RESOLVE}}で動作へ移る。左ページ末尾で動作の頂点を静止させて{{RESULT_QUESTION}}を伏せ、めくり直後の大ゴマで時間を再び流して{{RIVAL}}との勝敗を示し、{{SPECTATORS}}の歓声で日常の速度へ戻す。",
        "slots": [
            ("GAME_SITUATION", "勝負の状況（時間・点差・位置）", "setting", ["簡潔に数値で示す"], ("残り一秒、一点差", "決算期末まで残り一日", "検証の最終試行")),
            ("ATHLETE", "一瞬に臨む主体", "agent", ["独白で判断を共有する"], ("最後のシュートを打つ選手", "最終判断を下す担当者", "最後の実験を行う研究者")),
            ("FROZEN_MOMENT", "引き延ばされた一瞬", "symbol", ["背景を飛ばした小ゴマ連打", "多視点を挟む"], ("ボールが指を離れる瞬間の分解", "意思決定の一瞬を分解した図解", "一つの判断を細部まで分解した記述")),
            ("DOUBT_INTRUSION", "割り込む迷い", "conflict", ["過去の失敗や相手の気配を使う"], ("前の試合で外した記憶", "前回の失敗事例の記憶", "過去の反証例")),
            ("DECISIVE_RESOLVE", "迷いを振り切る決断", "goal", ["短い言葉で"], ("それでも打つ", "それでも決める", "それでもこの仮説で行く")),
            ("RESULT_QUESTION", "伏せられた勝敗", "lack", ["めくり直後まで伏せる"], ("入るのか外れるのか", "成果は出るのか", "仮説は正しいのか")),
            ("RIVAL", "同じ瞬間を争う相手", "agent", ["主体と同等の格"], ("ブロックに跳ぶ宿敵", "同時に提案する競合", "対立仮説")),
            ("SPECTATORS", "見守る観客", "audience", ["結果の後に音を戻す"], ("総立ちの観客席", "見守る顧客や関係者", "結果を待つ読者")),
        ],
        "invariants": [
            "一瞬の分解区間では背景を省略した小ゴマを多数並べ、主観時間を引き延ばす",
            "分解区間の中に必ず迷いの割り込みを一度置く",
            "結果は左ページ末尾で伏せ、めくり直後の大ゴマで示す",
            "結果の後に音と観客の反応を戻し、時間の速度を日常に復帰させる",
            "分解区間のコマは同じ視線方向（右上から左下）に流し、時間の連続性を保つ",
        ],
        "guide": {
            "manga": "8ページ構成。p1（右）4コマ: 残り時間と点差をスコアのECUと全景ELSで示す。p2（左）5コマ: 始動の動作をFSで描き、最終コマで背景を飛ばしたCUに切り替えて時間停止を宣言する。p3（右）9コマ・p4（左）8コマ: 背景を省略した小ゴマで指先・目・相手の手・ボールをECU中心に分解し、独白で判断の論理を積む。p4中盤に失敗の記憶を枠線を変えた1コマで割り込ませる。p5（右）7コマ: 決断の言葉と身体の動作。p6（左）6コマ: 動作の頂点を最終コマで静止画のように置く（望遠135mm）。めくって p7（右）1コマ断ち切り: 結果の大ゴマをアオリ・広角24mmで。p8（左）4コマ: 歓声の描き文字と観客、主体の放心のCU。",
            "marketing": "導入決定の瞬間を描くケーススタディで、期限や状況を数値で示した後、担当者の意思決定の一瞬を細部まで分解して見せる。過去の失敗の記憶という迷いを一度挟み、決断の言葉の後に結果を伏せたまま区切り、次のセクションで成果を大きく示して、関係者の反応で締める。",
            "article": "意思決定論の記事で、一つの判断の瞬間を細部まで分解して記述し、判断の根拠を順に積み上げる。過去の反証例による迷いを一度挟んだ上で決断を述べ、結果は章の切れ目の後に置く。結果の後には周囲の反応や影響を描き、思考の速度を日常に戻して締める。",
        },
        "rhythm": {
            "page_count": 8,
            "panels_per_page": [4, 5, 9, 8, 7, 6, 1, 4],
            "emphasis_pages": [6],
            "page_turn_reveals": [6],
            "spread_pages": [],
            "silent_panels_ratio": 0.25,
            "pacing_notes": [
                "p1（右）: 4コマ。残り時間と点差を示し一瞬の重さを定義する",
                "p2（左）: 5コマ。始動の動作を描き、最終コマで時間停止を宣言する",
                "p3（右）: 9コマ。背景を省略した小ゴマで一瞬を分解し独白で論理を積む",
                "p4（左）: 8コマ。分解を続け、中盤に失敗の記憶を一コマ割り込ませる",
                "p5（右）: 7コマ。決断の言葉と身体の動作へ移る",
                "p6（左）: 6コマ。最終コマで動作の頂点を静止させ結果を伏せる",
                "p7（右・めくり直後）: 断ち切り1コマ。時間が流れ出し結果を一挙に示す",
                "p8（左）: 4コマ。歓声と観客の反応で時間の速度を日常に戻す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 14: 群像の同時多発カット
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_14_ensemble_simultaneous_cuts",
        "title": "群像の同時多発カット",
        "summary": "場所・時刻・目的の異なる三系統を一コマずつ並べて同時進行の規則を示し、各系統に階段状に高くなる山を作り、読者にだけ系統間の符合を見せながら切り替え間隔を一コマ単位まで縮め、三分割コマを経てめくり直後の見開きで全系統を一点に合流させる骨格。",
        "tags": ["群像", "同時進行", "クロスカット", "合流", "見開き"],
        "lineage": ["並行編集（クロスカッティング）の群像劇", "多視点叙述の収束構造", "ゲシュタルトの閉合を利用した断片提示"],
        "waves": {
            "tension": [(0.0, 0.35), (0.1, 0.45), (0.15, 0.4), (0.25, 0.55), (0.3, 0.5), (0.4, 0.65), (0.45, 0.6), (0.55, 0.75), (0.6, 0.7), (0.75, 0.98), (0.9, 0.6), (1.0, 0.5)],
            "release": [(0.0, 0.15), (0.3, 0.2), (0.6, 0.15), (0.75, 0.1), (0.85, 0.5), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.2), (0.2, 0.35), (0.4, 0.5), (0.6, 0.65), (0.75, 0.9), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.2), (0.2, 0.25), (0.4, 0.3), (0.6, 0.35), (0.75, 0.85), (1.0, 0.92)],
            "cognitive_load": [(0.0, 0.6), (0.2, 0.7), (0.4, 0.8), (0.6, 0.85), (0.75, 0.5), (0.9, 0.4), (1.0, 0.35)],
        },
        "nodes": [
            ("thread_a", "agent", "thread_a_lead", "第一の系統の主体", "それぞれの場所で別の目的を追う群像の一人目", 0.75, "THREAD_A"),
            ("thread_b", "agent", "thread_b_lead", "第二の系統の主体", "第一の系統と無関係に見える別の時間を生きる主体", 0.7, "THREAD_B"),
            ("thread_c", "agent", "thread_c_lead", "第三の系統の主体", "二つの系統と対立しうる立場から同じ地点へ向かう主体", 0.7, "THREAD_C"),
            ("convergence_point", "symbol", "convergence_point", "全系統が交わる一点", "見開きで同時に描かれる合流の場所と瞬間", 0.95, "CONVERGENCE_POINT"),
            ("reader_overview", "desire", "reader_overview", "全体像を掴みたい読者の欲求", "各系統の断片をつなげて全体を把握しようとする欲求", 0.8, "RIPPLE_EFFECTS"),
            ("hidden_connection", "lack", "hidden_connection", "系統間の見えない接続", "各主体は知らないが読者には少しずつ示される因果の糸", 0.85, "HIDDEN_CONNECTION"),
        ],
        "edges": [
            ("thread_a", "convergence_point", "pursuit", 0.75, 0.0, "第一の系統が合流点へ向かう", 0.1),
            ("thread_b", "convergence_point", "pursuit", 0.7, 0.0, "第二の系統が合流点へ向かう", 0.25),
            ("thread_c", "thread_a", "opposition", 0.65, 0.0, "第三の系統は第一の系統と利害が対立する", 0.4),
            ("reader_overview", "hidden_connection", "dependency", 0.7, 0.0, "読者の全体把握は見えない接続に依存する", 0.2),
            ("hidden_connection", "thread_b", "asymmetry", 0.7, -0.6, "各主体は接続を知らない", 0.45),
            ("thread_c", "convergence_point", "power_gradient", 0.6, 0.5, "第三の系統が合流点の主導権を握ろうとする", 0.55),
            ("hidden_connection", "convergence_point", "embodiment", 0.8, 0.0, "見えない接続が合流点で可視化される", 0.75),
            ("thread_b", "thread_a", "betrayal", 0.55, 0.0, "合流点で第二の系統の裏切りが判明する", 0.75),
            ("thread_a", "thread_b", "dependency", 0.5, 0.0, "第一の系統は知らずに第二の系統へ頼っている", 0.3),
        ],
        "beats": [
            ("b_threads", "premise", 0.0, 0.05, "三系統の提示", "場所・時刻・目的の異なる三系統を一コマずつ並べ、同時進行の規則を示す", [], 0.0, 0.05, ["thread_a", "thread_b", "thread_c"], False),
            ("b_thread_a", "escalation", 0.1, 0.05, "第一系統の小さな山", "第一の系統に局所的な緊張の山を作る", ["b_threads"], 0.1, 0.05, ["thread_a"], False),
            ("b_thread_b", "escalation", 0.25, 0.05, "第二系統の小さな山", "第二の系統の山を第一より少し高く作る", ["b_thread_a"], 0.15, 0.05, ["thread_b"], False),
            ("b_thread_c", "escalation", 0.4, 0.05, "第三系統の小さな山", "第三の系統の山をさらに高くし、系統間の高さを階段状に積む", ["b_thread_b"], 0.15, 0.05, ["thread_c"], False),
            ("b_cross_signal", "setup", 0.45, 0.04, "偶然の符合", "異なる系統に同じ物・同じ言葉が現れ、読者だけが接続に気づく", ["b_thread_c", "b_thread_a"], -0.05, 0.15, ["hidden_connection", "reader_overview"], False),
            ("b_acceleration", "local_optimization_limit", 0.55, 0.05, "カット間隔の短縮", "系統の切り替えを一コマ単位まで縮め、同時性を極限まで高める", ["b_cross_signal"], 0.15, 0.05, ["thread_a", "thread_b", "thread_c"], False),
            ("b_last_split", "transition", 0.6, 0.04, "最後の分割コマ", "左ページ末尾で三系統を一コマに縦三分割して並べ、合流の直前を示す", ["b_acceleration"], -0.05, 0.0, ["thread_a", "thread_b", "thread_c"], True),
            ("b_convergence", "climax", 0.75, 0.08, "見開きでの合流", "めくり直後の見開きで三系統が一点で交わり、隠れた接続と裏切りが同時に可視化される", ["b_last_split"], 0.28, 0.4, ["convergence_point", "hidden_connection", "thread_b"], True),
            ("b_ripples", "payoff", 0.9, 0.06, "各系統への波及", "合流の結果が各系統の主体にどう波及したかを一コマずつ示す", ["b_convergence"], -0.38, 0.05, ["thread_a", "thread_c"], False),
            ("b_afterglow", "afterglow", 1.0, 0.06, "群像の余韻", "引きの俯瞰で全員を一枚に収め、新たな関係図を示して終える", ["b_ripples"], -0.1, 0.03, ["reader_overview"], False),
        ],
        "mechanisms": [
            ("m_omniscient_irony", "dramatic_irony", "全知の読者によるアイロニー", "読者だけが全系統を見渡せるため、各主体の知らない接続に気づく", "系統間の偶然の符合", "主体たちに警告したいもどかしさ", "劇的アイロニー", 0.8, ["b_cross_signal", "b_convergence"], ["hidden_connection", "reader_overview"]),
            ("m_pattern_completion", "curiosity_gap", "断片の補完欲求", "系統の断片が揃いきらない状態が、全体像を補完したい欲求を持続させる", "途切れる系統の切り替え", "読者がつながりを探しながら読み進める", "ゲシュタルトの閉合原理", 0.75, ["b_thread_a", "b_thread_b", "b_thread_c"], ["reader_overview"]),
            ("m_rhythm_compression", "anticipation_reward", "切り替え間隔の圧縮", "切り替えの間隔を縮めるほど合流への予期が高まり、見開きで報酬として回収される", "一コマ単位の切り替え", "合流の瞬間に大きな解放が訪れる", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.8, ["b_acceleration", "b_last_split"], ["convergence_point"]),
            ("m_load_relief", "contrast_effect", "高負荷から一枚絵への落差", "多系統を追う高い認知負荷が、見開きの一枚絵で一挙に整理される", "多コマから見開きへの面積の跳躍", "混沌が秩序に変わる爽快感", "認知負荷理論と対比効果", 0.65, ["b_convergence"], ["convergence_point"]),
        ],
        "anchor": "それぞれの場所で、互いを知らずに同じ空を見上げる三人の横顔",
        "drivers": ["読者だけが気づく系統間の符合", "縮まっていく切り替えの間隔", "最後の三分割コマ"],
        "emotions": ["好奇", "混乱", "予感", "焦燥", "驚嘆", "整理", "感慨"],
        "template": "{{THREAD_A}}・{{THREAD_B}}・{{THREAD_C}}の三系統を一コマずつ並べて同時進行の規則を示し、各系統に階段状に高くなる局所の山を作る。異なる系統に{{CROSS_SIGNAL}}を置いて{{HIDDEN_CONNECTION}}を読者にだけ示し、切り替えの間隔を一コマ単位まで縮める。左ページ末尾に{{SPLIT_PANEL}}を置き、めくり直後の見開きで{{CONVERGENCE_POINT}}における合流を描いて、各系統への{{RIPPLE_EFFECTS}}で締める。",
        "slots": [
            ("THREAD_A", "第一の系統の主体", "agent", ["他系統と場所・時刻・目的を変える"], ("駅へ急ぐ少女", "新規顧客を開拓する営業部門", "現象Aを観察する研究")),
            ("THREAD_B", "第二の系統の主体", "agent", ["第一系統と無関係に見せる"], ("取引に向かう男", "製品を改良する開発部門", "現象Bを観察する研究")),
            ("THREAD_C", "第三の系統の主体", "agent", ["第一系統と利害を対立させる"], ("追う側の刑事", "市場を狙う競合", "対立する解釈を示す研究")),
            ("CROSS_SIGNAL", "系統間の偶然の符合", "evidence", ["同じ物・同じ言葉を別系統に置く"], ("三人が同じ時計台の鐘を聞く", "三部門が同じ顧客の声に触れる", "三つの研究に共通する異常値")),
            ("HIDDEN_CONNECTION", "系統間の見えない接続", "lack", ["主体たちには知らせない"], ("三人を結ぶ一つの鞄", "部門をまたぐ一つの課題", "現象をつなぐ共通の原因")),
            ("SPLIT_PANEL", "合流直前の分割コマ", "setting", ["左ページ最終コマ", "縦三分割"], ("三人の足元を縦に三分割", "三部門の状況を並べた比較表", "三つの観察結果を並べた図")),
            ("CONVERGENCE_POINT", "全系統が交わる一点", "symbol", ["見開きで描く", "めくり直後"], ("時計台の下での鉢合わせ", "全部門が集う意思決定会議", "共通原因を示す統合モデル")),
            ("RIPPLE_EFFECTS", "合流が各系統に与える波及", "reversal", ["系統ごとに一コマずつ"], ("それぞれの運命の変化", "各部門の役割の再定義", "各研究の解釈の更新")),
        ],
        "invariants": [
            "三系統は冒頭で一コマずつ並べ、同時進行の規則を明示する",
            "系統の切り替え間隔は合流に向けて単調に短くする",
            "系統間の符合は読者にだけ示し、主体たちには知らせない",
            "合流はめくり直後の見開きで一枚絵として描く",
            "切り替えの際も右上から左下へのZ字の視線誘導を保ち、系統ごとに枠線や地の色で識別させる",
        ],
        "guide": {
            "manga": "12ページ構成。p1（右）5コマ: 三系統を一コマずつ並べ、各コマ右上に場所と時刻の小さな写植を入れる。p2〜p6（6〜7コマ）: 系統をページ単位から段単位へ切り替え、各系統の山を階段状に高くする。p5で別系統に同じ物（鐘・鞄など）をECUで置き符合を示す。p7（右）8コマ・p8（左）7コマ: 一コマ単位の切り替えに圧縮し、p8最終コマを三分割の縦長コマ（足元ECU×3）にする。めくって p9〜p10 見開き: 合流点をELS・広角18mm・俯瞰で描き、三人を右上から左下への対角線上に配置する。p11（右）4コマ: 各系統への波及を一コマずつ。p12（左）3コマ: 引きの俯瞰で新たな関係図を示す。",
            "marketing": "複数のペルソナが登場するLPで、立場の異なる三人の顧客の課題を並行して描き、各課題に共通する兆候を一つずつ示す。セクションの切り替えを次第に短くして同時性を高め、三人の状況を並べた比較表の後に、全員の課題を一つの解決策が同時に解く全体図を大きく示す。",
            "article": "複数の事例を並行して扱う記事で、三つの事例を交互に記述し、各事例に共通する異常値や兆候を少しずつ示す。記述の切り替えを短くしていき、三事例を並べた比較の図の後に、章の切れ目で共通原因を示す統合モデルを提示し、各事例の解釈がどう更新されるかを順に述べる。",
        },
        "rhythm": {
            "page_count": 12,
            "panels_per_page": [5, 6, 6, 7, 6, 7, 8, 7, 1, 1, 4, 3],
            "emphasis_pages": [8, 9],
            "page_turn_reveals": [8],
            "spread_pages": [8],
            "silent_panels_ratio": 0.1,
            "pacing_notes": [
                "p1（右）: 5コマ。三系統を一コマずつ並べ同時進行の規則を示す",
                "p2（左）: 6コマ。第一系統の小さな山を作る",
                "p3（右）: 6コマ。第二系統の山を第一より少し高く作る",
                "p4（左）: 7コマ。第三系統の山をさらに高くする",
                "p5（右）: 6コマ。別系統に同じ物を置き、読者だけに符合を示す",
                "p6（左）: 7コマ。切り替えを段単位に縮める",
                "p7（右）: 8コマ。一コマ単位の切り替えで同時性を極める",
                "p8（左）: 7コマ。最終コマを三系統の縦三分割コマにする",
                "p9（右・めくり直後）: 見開き大ゴマの右半分。合流点に三系統が集まる",
                "p10（左）: 見開き大ゴマの左半分。隠れた接続と裏切りを可視化する",
                "p11（右）: 4コマ。各系統への波及を一コマずつ示す",
                "p12（左）: 3コマ。引きの俯瞰で新たな関係図を示す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 15: 決め台詞の一枚絵
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_15_signature_line_splash",
        "title": "決め台詞の一枚絵",
        "summary": "相手の論理と力で主体を追い詰め、主体のフキダシを一切置かずに言葉を封じ、勝利宣言で沈黙を極めた後に顔を上げる無言の小ゴマを左ページ末尾に置き、めくり直後の1コマで大きなフキダシの決め台詞と主体の姿を一枚絵として示す骨格。",
        "tags": ["決め台詞", "一枚絵", "沈黙", "逆転", "フキダシ"],
        "lineage": ["見得を切る演出（静止と宣言）", "封殺からの反転の修辞", "名台詞の一枚絵化"],
        "waves": {
            "tension": [(0.0, 0.55), (0.1, 0.65), (0.2, 0.72), (0.3, 0.78), (0.4, 0.62), (0.45, 0.7), (0.5, 0.97), (0.6, 0.8), (0.7, 0.6), (0.85, 0.4), (1.0, 0.3)],
            "release": [(0.0, 0.05), (0.3, 0.05), (0.4, 0.15), (0.5, 0.5), (0.6, 0.65), (0.8, 0.75), (1.0, 0.8)],
            "audience_knowledge": [(0.0, 0.4), (0.3, 0.5), (0.4, 0.55), (0.5, 0.8), (0.7, 0.85), (1.0, 0.9)],
            "character_knowledge": [(0.0, 0.5), (0.3, 0.6), (0.4, 0.7), (0.5, 0.85), (0.7, 0.88), (1.0, 0.9)],
            "cognitive_load": [(0.0, 0.55), (0.3, 0.6), (0.4, 0.4), (0.5, 0.2), (0.7, 0.3), (1.0, 0.25)],
        },
        "nodes": [
            ("declarer", "agent", "declarer", "決め台詞を放つ主体", "追い詰められ沈黙を強いられた後、信念の一言で局面を塗り替える主体", 0.9, "DECLARER"),
            ("oppressor", "agent", "oppressor", "主体を追い詰める相手", "論理と力で主体を封じ込め、反論の余地を奪う対手", 0.8, "OPPRESSOR"),
            ("conviction", "desire", "conviction", "譲れない信念", "主体の沈黙の底で固まり、一言として結晶する価値", 0.85, "CONVICTION"),
            ("silenced_voice", "lack", "silenced_voice", "封じられた言葉", "追い詰めの間、主体が口にできずにいる本当の言葉", 0.75, "SILENCED_WORD"),
            ("signature_image", "symbol", "signature_image", "決め台詞の一枚絵", "大きなフキダシと主体の全身・表情が一体となる象徴的な一コマ", 0.95, "SIGNATURE_IMAGE"),
        ],
        "edges": [
            ("oppressor", "declarer", "power_gradient", 0.85, 0.8, "相手が主体を言葉で封じ込める", 0.0),
            ("oppressor", "declarer", "opposition", 0.8, 0.0, "価値観の正面衝突", 0.0),
            ("declarer", "silenced_voice", "asymmetry", 0.6, 0.5, "主体だけが言葉の重さを知っている", 0.2),
            ("declarer", "conviction", "pursuit", 0.8, 0.0, "主体は信念を貫こうとする", 0.4),
            ("signature_image", "conviction", "embodiment", 0.9, 0.0, "一枚絵が信念を体現する", 0.5),
            ("declarer", "oppressor", "power_gradient", 0.85, 0.9, "決め台詞が勢力を反転させる", 0.5),
            ("silenced_voice", "conviction", "dependency", 0.6, 0.0, "封じられた言葉は信念に支えられている", 0.3),
        ],
        "beats": [
            ("b_cornered", "premise", 0.0, 0.06, "追い詰め", "相手の論理と力で主体が追い詰められている状況から始める", [], 0.2, 0.0, ["oppressor", "declarer"], False),
            ("b_pressure", "escalation", 0.2, 0.08, "反論の封殺", "相手の言葉を多コマで畳みかけ、主体のフキダシを一切置かない", ["b_cornered"], 0.07, 0.05, ["oppressor", "silenced_voice"], False),
            ("b_max_pressure", "local_optimization_limit", 0.3, 0.05, "反論不能の極点", "相手が勝利を確信する一言を放ち、主体の沈黙が極まる", ["b_pressure"], 0.06, 0.0, ["oppressor", "declarer"], True),
            ("b_inner_resolve", "transition", 0.4, 0.05, "沈黙の内的決意", "左ページ末尾で主体が顔を上げる無言の小ゴマを置き、内側で信念が固まる", ["b_max_pressure"], -0.16, 0.1, ["declarer", "conviction"], True),
            ("b_signature", "climax", 0.5, 0.06, "決め台詞", "めくり直後の1コマで、大きなフキダシの決め台詞と主体の姿を一枚絵にする", ["b_inner_resolve"], 0.35, 0.25, ["signature_image", "declarer", "conviction"], True),
            ("b_reaction", "payoff", 0.6, 0.06, "相手の表情の崩れ", "相手の表情が崩れるCUで、言葉が局面を塗り替えたことを示す", ["b_signature"], -0.17, 0.05, ["oppressor"], False),
            ("b_quiet", "afterglow", 1.0, 0.1, "静かな余韻", "言葉の後の静けさと、周囲の変化を引きで描く", ["b_reaction"], -0.3, 0.05, ["declarer"], False),
        ],
        "mechanisms": [
            ("m_suppression_release", "catharsis", "封殺からの解放", "主体のフキダシを封じ続けた後の一言が、溜めた感情を一挙に解放する", "主体のフキダシの不在から大フキダシへの転換", "読者が胸のすくような解放を味わう", "カタルシス（緊張解放）", 0.85, ["b_pressure", "b_signature"], ["silenced_voice"]),
            ("m_value_affirmation", "moral_judgment", "信念への道徳的共鳴", "理不尽な論理に対して信念を表明する主体に、読者は道徳的に同一化する", "譲れない価値の一言", "主体を応援する強い感情が生まれる", "道徳基盤理論", 0.75, ["b_signature"], ["conviction", "declarer"]),
            ("m_memorable_line", "peak_end", "一枚絵の記憶化", "決め台詞の一枚絵が作品全体を代表する記憶として定着する", "大フキダシと全身の一体化", "台詞が引用され語り継がれる", "ピーク・エンドの法則", 0.7, ["b_signature", "b_quiet"], ["signature_image"]),
            ("m_silence_tension", "anticipation_reward", "沈黙による予期", "主体が顔を上げる無言のコマが、言葉の到来を予期させる", "左ページ末尾の無言の顔上げ", "めくった瞬間に予期が報われる", "報酬予測誤差（ドーパミン予測誤差仮説）", 0.7, ["b_inner_resolve", "b_signature"], ["declarer"]),
        ],
        "anchor": "反論を封じられながら、拳を握りしめて俯く主体",
        "drivers": ["主体のフキダシが消えた圧迫", "顔を上げる無言のコマ", "大きなフキダシの一枚絵"],
        "emotions": ["圧迫", "屈辱", "沈黙", "決意", "解放", "高揚", "静けさ"],
        "template": "{{OPPRESSOR}}の論理と力で{{DECLARER}}を追い詰め、主体のフキダシを置かずに{{SILENCED_WORD}}を封じる。相手が勝利を確信する{{TRIUMPH_LINE}}で沈黙を極めた後、左ページ末尾で主体が顔を上げる無言の小ゴマを置き、内側で{{CONVICTION}}を固める。めくり直後の1コマで{{SIGNATURE_LINE}}を大きなフキダシとともに{{SIGNATURE_IMAGE}}として描き、相手の表情の崩れと静かな余韻で締める。",
        "slots": [
            ("OPPRESSOR", "主体を追い詰める相手", "conflict", ["論理的に強く見せる"], ("理屈で圧倒する権力者", "安さだけで攻める競合の論理", "反論の余地のない通説")),
            ("DECLARER", "決め台詞を放つ主体", "agent", ["追い詰めの間はフキダシを置かない"], ("俯く主人公", "迷う購入検討者", "反論できずにいる書き手")),
            ("SILENCED_WORD", "封じられた言葉", "lack", ["追い詰めの間は一切言わせない"], ("それでも仲間を見捨てない", "価格ではなく価値で選ぶ", "それでも原則は譲れない")),
            ("TRIUMPH_LINE", "相手が勝利を確信する一言", "irony", ["左ページの中ほどに置く"], ("お前に何ができる", "結局は安い方を選ぶでしょう", "例外は無視してよい")),
            ("CONVICTION", "譲れない信念", "goal", ["独白で説明しすぎない"], ("仲間を守るという誓い", "長く使える品質への信頼", "誠実さという原則")),
            ("SIGNATURE_LINE", "決め台詞", "reversal", ["短く", "大きなフキダシ"], ("俺は一人じゃない", "私は十年後の自分のために選ぶ", "例外こそが真実を語る")),
            ("SIGNATURE_IMAGE", "決め台詞の一枚絵", "symbol", ["めくり直後の1コマ", "全身と表情を一体化"], ("風を受けて立つ全身のアオリ", "キャッチコピーと製品の大写し", "論旨を一文で示す大見出し")),
        ],
        "invariants": [
            "追い詰めの区間では主体のフキダシを一切置かない",
            "決め台詞の直前に主体が顔を上げる無言の小ゴマを置く",
            "決め台詞はめくり直後の1コマに、大きなフキダシとともに置く",
            "決め台詞は短く、それまでの相手の論理に正面から応える内容にする",
        ],
        "guide": {
            "manga": "5ページ構成。p1（右）5コマ: 追い詰める相手をアオリのMSで、主体を俯瞰のFSで小さく描く。p2（左）6コマ: 相手の言葉を多コマで畳みかけ、主体のフキダシは置かない。中ほどに相手の勝利宣言を大きめのコマで置き、最終コマで主体が顔を上げる無言のCU（望遠85mm）。めくって p3（右）1コマ断ち切り: 決め台詞を大きなフキダシで、主体の全身をアオリ・広角24mmで描き、フキダシから顔へ視線を流す。p4（左）4コマ: 相手の表情が崩れるCUと周囲の反応。p5（右）3コマ: 静かな余韻を引きのELSで描く。",
            "marketing": "価格競争に巻き込まれた顧客の迷いを描くLPで、安さを強調する競合の論理を畳みかけて提示し、顧客の本音を一度封じる。迷いの末に顧客が自分の価値基準に気づく一文の溜めを置き、次のセクション冒頭で価値で選ぶというキャッチコピーを大きく掲げて、製品の大写しと一体化させる。",
            "article": "論説で、反論の余地がないように見える通説を論理的に畳みかけて紹介し、書き手の反論を一度保留する。通説の勝利宣言とも言える一文を引いた後に一段落の沈黙を置き、次の章冒頭で書き手の原則を一文の大見出しとして示し、通説の論理に正面から応える。",
        },
        "rhythm": {
            "page_count": 5,
            "panels_per_page": [5, 6, 1, 4, 3],
            "emphasis_pages": [2],
            "page_turn_reveals": [2],
            "spread_pages": [],
            "silent_panels_ratio": 0.3,
            "pacing_notes": [
                "p1（右）: 5コマ。追い詰める相手と小さく描かれる主体",
                "p2（左）: 6コマ。相手の言葉を畳みかけ、最終コマで主体が顔を上げる無言のCU",
                "p3（右・めくり直後）: 断ち切り1コマ。大きなフキダシの決め台詞と主体の全身",
                "p4（左）: 4コマ。相手の表情の崩れと周囲の反応",
                "p5（右）: 3コマ。静かな余韻を引きで描く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 16: 日常パートの間（ま）と余白
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_16_daily_ma_and_whitespace",
        "title": "日常パートの間（ま）と余白",
        "summary": "大きな事件のない日常を横長のコマと人物のいない風景の余白コマでゆっくり刻み、身近な象徴物を繰り返し映しながら低い緊張を保ち、無言の間を経てめくり直後のやや大きなコマで小さな気づきを描き、再び余白へ戻る日常演出の骨格。",
        "tags": ["日常", "間", "余白", "情緒", "気づき"],
        "lineage": ["間（ま）の美学に基づく静的演出", "空ショット（人物のいない風景）の挿入", "小さな気づきの物語"],
        "waves": {
            "tension": [(0.0, 0.15), (0.1, 0.2), (0.2, 0.16), (0.35, 0.24), (0.45, 0.18), (0.6, 0.28), (0.7, 0.22), (0.8, 0.42), (0.9, 0.3), (1.0, 0.2)],
            "release": [(0.0, 0.4), (0.2, 0.5), (0.45, 0.55), (0.7, 0.5), (0.8, 0.45), (0.9, 0.7), (1.0, 0.75)],
            "audience_knowledge": [(0.0, 0.3), (0.2, 0.35), (0.45, 0.4), (0.7, 0.45), (0.8, 0.7), (1.0, 0.75)],
            "character_knowledge": [(0.0, 0.3), (0.2, 0.33), (0.45, 0.36), (0.7, 0.4), (0.8, 0.72), (1.0, 0.78)],
            "cognitive_load": [(0.0, 0.25), (0.2, 0.2), (0.45, 0.18), (0.7, 0.2), (0.8, 0.3), (1.0, 0.2)],
        },
        "nodes": [
            ("dweller", "agent", "everyday_dweller", "日常を生きる主体", "大きな事件のない時間の中で、小さな所作と感覚を積み重ねる主体", 0.8, "DWELLER"),
            ("companion", "agent", "companion", "傍らにいる存在", "言葉少なに同じ時間を共有し、関係の温度を示す相手", 0.6, "COMPANION"),
            ("small_object", "symbol", "everyday_symbol", "日常の小さな象徴物", "繰り返し映され、最後に小さな気づきの鍵となる身近な物", 0.75, "SMALL_OBJECT"),
            ("unnoticed_feeling", "lack", "unnoticed_feeling", "まだ気づかれていない感情", "余白の中に沈み、最後に小さく浮かび上がる感情", 0.8, "UNNOTICED_FEELING"),
            ("calm_desire", "desire", "desire_for_calm", "この時間が続いてほしいという願い", "主体と読者が共有する、穏やかな時間への愛着", 0.7, "CALM_DESIRE"),
        ],
        "edges": [
            ("dweller", "calm_desire", "pursuit", 0.6, 0.0, "主体は穏やかな時間を味わおうとする", 0.0),
            ("dweller", "companion", "dependency", 0.65, 0.0, "主体は傍らの存在に寄りかかっている", 0.0),
            ("small_object", "unnoticed_feeling", "embodiment", 0.75, 0.0, "小さな物が気づかれていない感情を宿す", 0.4),
            ("companion", "unnoticed_feeling", "asymmetry", 0.5, 0.4, "相手は主体より先に感情に気づいている", 0.5),
            ("unnoticed_feeling", "calm_desire", "opposition", 0.35, -0.2, "気づきが穏やかさを揺らす", 0.8),
            ("companion", "dweller", "power_gradient", 0.3, 0.2, "相手がさりげなく主体を導く", 0.6),
        ],
        "beats": [
            ("b_routine", "premise", 0.0, 0.06, "繰り返しの日常", "朝の所作や生活音を横長のコマで描き、時間の流れの遅さを示す", [], 0.0, 0.0, ["dweller"], False),
            ("b_small_motion", "setup", 0.1, 0.05, "小さな所作", "湯気や手元など小さな動きを一コマで切り取る", ["b_routine"], 0.05, 0.02, ["dweller", "small_object"], False),
            ("b_landscape", "transition", 0.2, 0.06, "風景の余白コマ", "人物のいない風景の横長コマで間を置く", ["b_small_motion"], -0.04, 0.0, ["calm_desire"], False),
            ("b_shared_moment", "escalation", 0.35, 0.06, "共有の時間", "傍らの存在と短い言葉を交わし、関係の温度を示す", ["b_landscape"], 0.08, 0.05, ["dweller", "companion"], False),
            ("b_drift", "transition", 0.45, 0.05, "沈黙の漂い", "会話が途切れたままの静かなコマを続ける", ["b_shared_moment"], -0.06, 0.0, ["dweller"], False),
            ("b_detail", "setup", 0.6, 0.05, "象徴物の細部", "繰り返し映された身近な物の細部に光を当てる", ["b_drift", "b_small_motion"], 0.1, 0.05, ["small_object", "unnoticed_feeling"], False),
            ("b_pause", "transition", 0.7, 0.04, "無言の間", "左ページ末尾に人物の無言のCUを置き、気づきの直前で呼吸を止める", ["b_detail"], -0.06, 0.0, ["dweller", "companion"], True),
            ("b_realization", "revelation", 0.8, 0.05, "小さな気づき", "めくり直後のやや大きなコマで、主体が自分の感情に小さく気づく", ["b_pause"], 0.2, 0.25, ["unnoticed_feeling", "dweller", "small_object"], True),
            ("b_afterglow", "afterglow", 1.0, 0.08, "続いていく時間", "再び風景の余白コマに戻り、日常が続いていくことを示す", ["b_realization"], -0.1, 0.03, ["calm_desire", "companion"], False),
        ],
        "mechanisms": [
            ("m_ma", "narrative_transportation", "間による没入", "人物のいない余白コマと無言のコマが、読者自身の時間感覚を作品の時間に同調させる", "風景の余白コマと沈黙の連続", "読者の呼吸がゆっくりになる", "物語移入理論", 0.8, ["b_landscape", "b_drift"], ["calm_desire"]),
            ("m_repeated_detail", "anticipation_reward", "細部への微かな予期", "繰り返し映される身近な物が、意味を持つ瞬間への穏やかな予期を育てる", "象徴物の反復", "気づきの瞬間に小さな満足が訪れる", "期待理論（予期と報酬）", 0.6, ["b_small_motion", "b_detail", "b_realization"], ["small_object"]),
            ("m_quiet_empathy", "empathy_transfer", "静かな共感", "大きな事件がないからこそ、小さな感情の揺れに読者が自分を重ねる", "無言のCUと小さな気づき", "読者が自分の日常の感情を思い出す", "共感の二過程モデル", 0.7, ["b_pause", "b_realization"], ["dweller", "unnoticed_feeling"]),
            ("m_contrast_calm", "contrast_effect", "静と微動の対比", "低い緊張の中では小さな変化が大きく感じられる", "低い緊張度の持続と小さな山", "わずかな感情の動きが鮮やかに感じられる", "対比効果（ウェーバーの法則）", 0.55, ["b_realization"], ["unnoticed_feeling"]),
        ],
        "anchor": "湯気の向こうでぼんやりと窓の外を見る主体の横顔",
        "drivers": ["人物のいない余白コマの静けさ", "繰り返し映される身近な物", "言葉の途切れた間"],
        "emotions": ["穏やか", "まどろみ", "温かさ", "静けさ", "ためらい", "気づき", "安らぎ"],
        "template": "{{DWELLER}}の繰り返しの日常を横長のコマで描き、{{SMALL_OBJECT}}を何度か映す。{{LANDSCAPE_PAUSE}}で人物のいない間を置き、{{COMPANION}}と短い言葉を交わした後の沈黙を続ける。左ページ末尾に{{SILENT_CLOSEUP}}を置き、めくり直後のやや大きなコマで{{UNNOTICED_FEELING}}への小さな気づきを描き、{{CALM_DESIRE}}を確かめるように再び余白へ戻る。",
        "slots": [
            ("DWELLER", "日常を生きる主体", "agent", ["大きな事件を起こさない"], ("一人暮らしの学生", "日々を淡々と過ごす利用者", "日常を観察する書き手")),
            ("SMALL_OBJECT", "繰り返し映される身近な物", "symbol", ["三回以上映す", "最後に意味を持たせる"], ("毎朝使うマグカップ", "毎日使う小さな道具", "机の上の一冊のノート")),
            ("LANDSCAPE_PAUSE", "人物のいない余白コマ", "setting", ["横長のコマ", "セリフなし"], ("夕暮れの川べり", "余白の多いビジュアル", "情景描写だけの段落")),
            ("COMPANION", "傍らにいる存在", "agent", ["言葉少なにする"], ("隣に座る幼なじみ", "寄り添うサービスの担当者", "対話の相手")),
            ("SILENT_CLOSEUP", "気づきの直前の無言のCU", "emotion", ["左ページ最終コマ"], ("何かを言いかけてやめる横顔", "ふと手を止める一文", "考えが途切れる一行")),
            ("UNNOTICED_FEELING", "まだ気づかれていない感情", "lack", ["大げさにしない", "小さく浮かび上がらせる"], ("この時間が好きだという気持ち", "この習慣が支えになっていたという実感", "当たり前の大切さ")),
            ("CALM_DESIRE", "穏やかな時間への愛着", "goal", ["言葉で説明しすぎない"], ("明日も同じ朝が来てほしい", "この使い心地が続いてほしい", "この感覚を大切にしたい")),
        ],
        "invariants": [
            "緊張度は全体を通して低く保ち、ピークは小さな気づき一つに限る",
            "人物のいない余白コマを少なくとも二回置く",
            "身近な象徴物を繰り返し映し、最後の気づきに接続する",
            "気づきはめくり直後のやや大きなコマに置き、大げさな演出をしない",
            "横長のコマを多用し、視線を右から左へゆっくり流す",
        ],
        "guide": {
            "manga": "6ページ構成。p1（右）4コマ: 横長のコマで朝の所作をFS・アイレベルで描き、マグカップをECUで一度映す。p2（左）5コマ: 人物のいない風景の横長コマ（ELS）を一コマ挟み、生活音の小さな描き文字だけを置く。p3（右）3コマ: 傍らの存在と短い会話をMSで。p4（左）5コマ: 沈黙のコマを続け、象徴物の細部をECUで照らし、最終コマに無言の横顔CU（望遠85mm）。めくって p5（右）2コマ: やや大きなコマで小さな気づきをCU・アイレベルで描き、小さなフキダシを一つだけ置く。p6（左）3コマ: 再び風景の余白コマに戻る。",
            "marketing": "暮らしに寄り添う商品のブランドページで、大きな訴求をせずに日常の所作と商品を余白の多いビジュアルで繰り返し見せる。説明を最小限にして読者の時間感覚をゆっくりにした上で、この習慣が支えになっていたという小さな実感を一文で示し、日常が続いていく情景で締める。",
            "article": "エッセイで、出来事の少ない日常を情景描写中心にゆっくり記述し、身近な物を何度か登場させる。考えが途切れる一行の後、章の切れ目でその物が当たり前の大切さを象徴していたという小さな気づきを述べ、結論を押し付けず情景に戻って終える。",
        },
        "rhythm": {
            "page_count": 6,
            "panels_per_page": [4, 5, 3, 5, 2, 3],
            "emphasis_pages": [4],
            "page_turn_reveals": [4],
            "spread_pages": [],
            "silent_panels_ratio": 0.55,
            "pacing_notes": [
                "p1（右）: 4コマ。横長のコマで朝の所作を描き、象徴物を一度映す",
                "p2（左）: 5コマ。人物のいない風景の余白コマを挟み、生活音だけを置く",
                "p3（右）: 3コマ。傍らの存在と短い言葉を交わす",
                "p4（左）: 5コマ。沈黙を続け、象徴物の細部を照らし、最終コマに無言の横顔",
                "p5（右・めくり直後）: 2コマ。やや大きなコマで小さな気づきを描く",
                "p6（左）: 3コマ。再び風景の余白コマに戻り日常が続くことを示す",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 17: 謎解き：手がかり提示→推理連鎖→指摘
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_17_mystery_clue_deduction_accusation",
        "title": "謎解き：手がかり提示→推理連鎖→指摘",
        "summary": "不可解な問いを提示して手がかりを読者にも公平に散りばめ、誤誘導の仮説を新たな証拠で破綻させた後、手がかり同士を結ぶ推理の連鎖で真相へ迫り、名指し直前の無言の溜めを経てめくり直後の大ゴマで指摘し、動機の吐露で余韻を残す骨格。",
        "tags": ["謎解き", "推理", "手がかり", "誤誘導", "指摘"],
        "lineage": ["本格推理のフェアプレイ原則", "誤誘導（ミスディレクション）の技法", "名探偵による一同を集めての指摘"],
        "waves": {
            "tension": [(0.0, 0.3), (0.1, 0.4), (0.2, 0.35), (0.3, 0.45), (0.4, 0.42), (0.5, 0.55), (0.6, 0.62), (0.7, 0.7), (0.8, 0.78), (0.85, 0.74), (0.9, 0.98), (1.0, 0.6)],
            "release": [(0.0, 0.1), (0.4, 0.15), (0.6, 0.2), (0.8, 0.25), (0.85, 0.2), (0.9, 0.45), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.25), (0.4, 0.35), (0.6, 0.45), (0.8, 0.6), (0.9, 0.95), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.5), (0.2, 0.6), (0.4, 0.7), (0.6, 0.8), (0.8, 0.85), (0.9, 0.5), (1.0, 0.4)],
        },
        "nodes": [
            ("detective", "agent", "investigator", "手がかりを読み解く主体", "読者より先に真相へ近づきながら、推理の過程を段階的に開示する主体", 0.9, "INVESTIGATOR"),
            ("culprit", "agent", "concealer", "真相を隠す人物", "読者の前に何度も登場しながら疑いの外に置かれている人物", 0.85, "CONCEALER"),
            ("clues", "symbol", "clue_set", "散りばめられた手がかり", "読者にも公平に提示され、推理の連鎖で意味がつながる証拠群", 0.8, "CLUE_SET"),
            ("truth", "lack", "hidden_truth", "伏せられた真相", "手がかりの背後にあり、指摘の瞬間まで読者に伏せられる事実", 0.9, "HIDDEN_TRUTH"),
            ("reader_solve", "desire", "reader_solving", "自分で解きたいという読者の欲求", "公平に示された手がかりから真相を先回りしたいという欲求", 0.8, "CASE_QUESTION"),
            ("false_lead", "symbol", "misdirection", "誤誘導", "読者の疑いを別方向へ向ける偽の手がかり", 0.6, "MISDIRECTION"),
        ],
        "edges": [
            ("detective", "culprit", "opposition", 0.85, 0.0, "暴く者と隠す者", 0.0),
            ("culprit", "truth", "asymmetry", 0.85, 0.8, "隠す者だけが真相を知る", 0.0),
            ("detective", "truth", "pursuit", 0.9, 0.0, "主体が真相を追う", 0.1),
            ("reader_solve", "clues", "dependency", 0.75, 0.0, "読者の推理は提示された手がかりに依存する", 0.1),
            ("false_lead", "reader_solve", "power_gradient", 0.55, 0.4, "誤誘導が読者の推理を一時的に支配する", 0.2),
            ("clues", "truth", "embodiment", 0.8, 0.0, "手がかりの連鎖が真相を体現する", 0.6),
            ("culprit", "detective", "betrayal", 0.6, 0.0, "協力者を装っていた人物が隠す者だった", 0.9),
            ("detective", "culprit", "power_gradient", 0.8, 0.9, "指摘によって主体が決定的に優位に立つ", 0.9),
        ],
        "beats": [
            ("b_case", "premise", 0.0, 0.05, "事件の提示", "不可解な状況を提示し、解くべき問いを明確にする", [], 0.0, 0.05, ["detective", "truth"], False),
            ("b_clue1", "setup", 0.1, 0.05, "第一の手がかり", "読者にも見える形で最初の手がかりを一コマで提示する", ["b_case"], 0.1, 0.1, ["clues", "reader_solve"], False),
            ("b_misdirect", "transition", 0.2, 0.05, "誤誘導", "疑わしい人物を別に用意し、読者の疑いを逸らす", ["b_clue1"], -0.05, -0.05, ["false_lead", "reader_solve"], False),
            ("b_clue2", "setup", 0.3, 0.05, "第二の手がかり", "第一の手がかりと矛盾するように見える証拠を提示する", ["b_misdirect"], 0.1, 0.1, ["clues"], False),
            ("b_contradiction", "local_optimization_limit", 0.4, 0.05, "既存仮説の破綻", "誤誘導に基づく仮説が新たな証拠で破綻する", ["b_clue2"], -0.03, 0.05, ["false_lead", "detective"], True),
            ("b_chain_start", "escalation", 0.6, 0.08, "推理連鎖の開始", "主体が手がかり同士を結び始め、図解的なコマで論理を積む", ["b_contradiction", "b_clue1"], 0.2, 0.1, ["detective", "clues"], False),
            ("b_chain_peak", "escalation", 0.8, 0.06, "推理連鎖の頂点", "残る一つの可能性へ論理が収束し、名指しの直前まで迫る", ["b_chain_start"], 0.16, 0.1, ["detective", "truth"], False),
            ("b_hesitation", "transition", 0.85, 0.04, "名指し直前の溜め", "左ページ末尾で主体が一同を見回す無言のコマを置く", ["b_chain_peak"], -0.04, 0.0, ["detective"], True),
            ("b_accusation", "revelation", 0.9, 0.05, "指摘", "めくり直後の大ゴマで主体が真相を隠す人物を指し示す", ["b_hesitation", "b_clue2"], 0.24, 0.35, ["detective", "culprit", "truth"], True),
            ("b_motive", "afterglow", 1.0, 0.08, "動機の吐露", "隠す者の動機と、手がかりの意味の回収で余韻を残す", ["b_accusation"], -0.38, 0.05, ["culprit", "clues"], False),
        ],
        "mechanisms": [
            ("m_fair_clues", "self_efficacy", "公平な手がかりによる挑戦", "全ての手がかりを読者にも示すことで、読者自身が解ける挑戦として成立させる", "読者にも見える手がかりのコマ", "読者が能動的に推理に参加する", "自己効力感理論", 0.75, ["b_clue1", "b_clue2"], ["clues", "reader_solve"]),
            ("m_misdirection", "prediction_error", "誤誘導と予測の裏切り", "疑わしい人物へ読者の予測を誘導した上で破綻させ、真相の驚きを増幅する", "別の疑わしい人物の配置", "真相の指摘に強い驚きが伴う", "予測符号化理論", 0.8, ["b_misdirect", "b_contradiction", "b_accusation"], ["false_lead"]),
            ("m_detective_gap", "curiosity_gap", "主体と読者の情報差", "主体が先に真相へ近づく様子を見せ、読者に追いつきたい欲求を生む", "主体の確信めいた表情", "読者がページを急いで読み進める", "情報ギャップ理論", 0.8, ["b_chain_start", "b_chain_peak"], ["detective", "truth"]),
            ("m_coherence", "cognitive_dissonance", "矛盾の解消による納得", "矛盾して見えた手がかりが一つの真相で説明されるとき、強い納得が生まれる", "推理連鎖による矛盾の統合", "全てが腑に落ちる快感", "認知的不協和理論", 0.7, ["b_contradiction", "b_accusation"], ["clues"]),
            ("m_motive_empathy", "moral_judgment", "動機による道徳的揺さぶり", "隠す者の動機を示すことで、単純な断罪でなく道徳的な揺らぎを残す", "動機の吐露", "読者が善悪を考え込む余韻", "道徳基盤理論", 0.55, ["b_motive"], ["culprit"]),
        ],
        "anchor": "一同を見回し、言葉を選ぶ主体の静かな目",
        "drivers": ["公平に示された手がかりへの挑戦心", "誤誘導が崩れる瞬間", "名指し直前の溜め"],
        "emotions": ["好奇", "疑念", "混乱", "推理", "確信", "驚愕", "哀感"],
        "template": "{{CASE_QUESTION}}を提示し、読者にも見える形で{{CLUE_SET}}を散りばめる。{{MISDIRECTION}}で読者の疑いを逸らした後、新たな証拠で既存の仮説を破綻させ、{{INVESTIGATOR}}が手がかり同士を結ぶ推理の連鎖で{{HIDDEN_TRUTH}}へ迫る。左ページ末尾で一同を見回す無言のコマを置き、めくり直後の大ゴマで{{CONCEALER}}を指し示して、{{MOTIVE}}の吐露で余韻を残す。",
        "slots": [
            ("CASE_QUESTION", "解くべき不可解な問い", "goal", ["冒頭で明確に示す"], ("密室から消えた人物の謎", "なぜ売れない商品が突然売れたのか", "なぜ定説と矛盾する結果が出たのか")),
            ("CLUE_SET", "散りばめられた手がかり", "evidence", ["読者にも全て見せる", "三つ以上"], ("濡れた傘・止まった時計・二つの足跡", "三つの顧客データ", "三つの実験結果")),
            ("MISDIRECTION", "誤誘導", "irony", ["もっともらしくする"], ("動機のある別の人物", "季節要因という説明", "測定誤差という解釈")),
            ("INVESTIGATOR", "手がかりを読み解く主体", "agent", ["推理の過程を段階的に開示する"], ("観察眼の鋭い探偵", "データを読む分析者", "仮説を検証する書き手")),
            ("HIDDEN_TRUTH", "伏せられた真相", "lack", ["指摘まで伏せる"], ("時計を止めたのは被害者自身", "購入者は別の目的で買っていた", "前提とした条件自体が違っていた")),
            ("CONCEALER", "真相を隠す人物・要因", "agent", ["序盤から何度も登場させる"], ("親切な隣人", "見過ごされていた顧客層", "見落とされていた変数")),
            ("MOTIVE", "隠された動機・理由", "reversal", ["単純な断罪にしない"], ("家族を守るための嘘", "顧客の本当の困りごと", "現象の背後にある構造")),
        ],
        "invariants": [
            "真相に必要な手がかりは全て指摘より前に読者へ提示する",
            "誤誘導を一つ置き、新たな証拠で破綻させてから真の推理を始める",
            "指摘はめくり直後の大ゴマに置き、直前に無言の溜めを置く",
            "真相を隠す人物は序盤から何度も登場させ、疑いの外に置く",
            "推理連鎖のコマは手がかりの提示順に右上から左下へ並べ、視線の流れで論理をたどらせる",
        ],
        "guide": {
            "manga": "12ページ構成。p1（右）5コマ: 不可解な状況をELSとMSで提示。p2〜p6（5〜6コマ）: 各ページに手がかりを一つずつECUで提示し、p3で別の疑わしい人物を怪しげなアオリのCUで印象づける（誤誘導）。p5で新たな証拠により誤誘導の仮説が破綻する。p7〜p9（6〜7コマ）: 推理連鎖を図解的なコマと手がかりのECUの回想カットで積み、論理を右上から左下へ流す。p10（左）4コマ: 最終コマで一同を見回す主体の目のCU（俯瞰）。めくって p11（右）1コマ断ち切り: 指摘の大ゴマ、指先から人物へ視線を導くFS・広角24mm・ダッチアングル10度。p12（左）4コマ: 動機の吐露をCUで描き、手がかりの意味を回収する。",
            "marketing": "原因分析型のケーススタディで、不可解な成果（または不振）の問いを冒頭で示し、三つのデータを順に公平に提示する。もっともらしい説明（季節要因）を一度提示して否定した後、データ同士を結ぶ分析で真の原因に迫り、区切りの直後に見過ごされていた顧客層を真因として示し、その顧客の本当の困りごとに応える提案で締める。",
            "article": "謎解き型の論考で、定説と矛盾する結果を問いとして提示し、関連する実験結果を全て先に示す。測定誤差という解釈を一度検討して退けた後、結果同士を結ぶ推論で見落とされていた変数を特定し、章の切れ目の直後にそれを指摘して、背後の構造を解説する。",
        },
        "rhythm": {
            "page_count": 12,
            "panels_per_page": [5, 6, 5, 6, 5, 6, 6, 7, 6, 4, 1, 4],
            "emphasis_pages": [10],
            "page_turn_reveals": [10],
            "spread_pages": [],
            "silent_panels_ratio": 0.1,
            "pacing_notes": [
                "p1（右）: 5コマ。不可解な状況を提示し解くべき問いを示す",
                "p2（左）: 6コマ。第一の手がかりを読者にも見える形で提示する",
                "p3（右）: 5コマ。別の疑わしい人物を印象づける誤誘導",
                "p4（左）: 6コマ。第二の手がかりで矛盾を生む",
                "p5（右）: 5コマ。新たな証拠で誤誘導の仮説が破綻する",
                "p6（左）: 6コマ。第三の手がかりと隠す人物の何気ない再登場",
                "p7（右）: 6コマ。推理連鎖の開始。手がかり同士を結ぶ",
                "p8（左）: 7コマ。図解的なコマで論理を積む",
                "p9（右）: 6コマ。残る一つの可能性へ論理を収束させる",
                "p10（左）: 4コマ。最終コマで一同を見回す無言の溜め",
                "p11（右・めくり直後）: 断ち切り1コマ。指摘の大ゴマ",
                "p12（左）: 4コマ。動機の吐露と手がかりの回収",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 18: 引きのラストページ（次回へのクリフハンガー）
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_18_cliffhanger_last_page",
        "title": "引きのラストページ（次回へのクリフハンガー）",
        "summary": "一話の局所的な対立を山場で決着させ、解決と束の間の安堵で緊張を最低まで下げ、安堵の片隅に小さな異物を置いてから、めくり直後の大ゴマでより大きな脅威を出現させ、答えのない問いを最終コマに置いて緊張度の最高値で終える連載の骨格。",
        "tags": ["クリフハンガー", "引き", "連載", "最終ページ", "予兆"],
        "lineage": ["連続活劇のクリフハンガー", "週刊連載のヒキの慣行", "解決と新たな謎の二重構造"],
        "waves": {
            "tension": [(0.0, 0.6), (0.15, 0.7), (0.3, 0.35), (0.4, 0.22), (0.55, 0.25), (0.6, 0.32), (0.7, 0.42), (0.75, 0.36), (0.8, 0.9), (0.9, 0.84), (1.0, 0.97)],
            "release": [(0.0, 0.1), (0.15, 0.2), (0.3, 0.7), (0.4, 0.8), (0.6, 0.6), (0.75, 0.5), (0.8, 0.1), (1.0, 0.0)],
            "audience_knowledge": [(0.0, 0.5), (0.3, 0.65), (0.6, 0.7), (0.75, 0.72), (0.8, 0.85), (1.0, 0.88)],
            "character_knowledge": [(0.0, 0.5), (0.3, 0.65), (0.6, 0.68), (0.75, 0.7), (0.8, 0.75), (1.0, 0.8)],
            "cognitive_load": [(0.0, 0.55), (0.3, 0.4), (0.5, 0.3), (0.75, 0.3), (0.8, 0.55), (1.0, 0.45)],
        },
        "nodes": [
            ("hero", "agent", "episode_protagonist", "一話の問題を解決した主体", "局所的な勝利の直後に、より大きな脅威の到来を知らされる主体", 0.85, "PROTAGONIST"),
            ("local_problem", "agent", "episode_antagonist", "一話の中の障害", "この回の中で解決される局所的な敵や問題", 0.6, "LOCAL_PROBLEM"),
            ("greater_threat", "agent", "greater_threat", "より大きな脅威", "最終ページで初めて姿を見せる、次回以降の物語を牽引する力", 0.95, "GREATER_THREAT"),
            ("peace_desire", "desire", "desire_for_peace", "安堵が続いてほしいという欲求", "解決の後に主体と読者が抱く、平穏への願い", 0.7, "PEACE_MOMENT"),
            ("unresolved_question", "lack", "unresolved_question", "次回へ持ち越される問い", "最終コマで提示され、答えが与えられない問い", 0.9, "CLIFFHANGER_QUESTION"),
            ("omen", "symbol", "omen", "小さな異物（予兆）", "安堵の場面に紛れ込む、脅威の到来を告げる小さな徴", 0.7, "OMEN"),
        ],
        "edges": [
            ("hero", "local_problem", "opposition", 0.75, 0.0, "一話の中の対立", 0.0),
            ("hero", "local_problem", "power_gradient", 0.7, 0.7, "主体が一話の問題を制する", 0.15),
            ("greater_threat", "hero", "power_gradient", 0.9, 0.9, "より大きな脅威が主体を圧倒する", 0.8),
            ("greater_threat", "hero", "opposition", 0.85, 0.0, "次回以降の対立の予告", 0.8),
            ("omen", "greater_threat", "embodiment", 0.75, 0.0, "小さな異物が脅威の到来を体現する", 0.7),
            ("hero", "peace_desire", "pursuit", 0.6, 0.0, "主体は平穏を取り戻そうとする", 0.3),
            ("greater_threat", "unresolved_question", "asymmetry", 0.7, 0.7, "脅威の正体は主体にも読者にも伏せられる", 0.8),
            ("local_problem", "greater_threat", "dependency", 0.55, 0.0, "一話の敵は大きな脅威に操られていた", 0.85),
        ],
        "beats": [
            ("b_local_climax", "climax", 0.15, 0.06, "一話の山場", "この回の局所的な対立を決着させる山場を描く", [], 0.1, 0.05, ["hero", "local_problem"], True),
            ("b_resolution", "payoff", 0.3, 0.06, "解決", "問題が解決し、伏線が回収される", ["b_local_climax"], -0.35, 0.1, ["hero", "local_problem"], False),
            ("b_calm", "transition", 0.4, 0.06, "束の間の安堵", "仲間との穏やかな会話で緊張を最も低くする", ["b_resolution"], -0.13, 0.02, ["hero", "peace_desire"], False),
            ("b_unease", "setup", 0.7, 0.05, "小さな異物", "安堵の場面の片隅に、脅威の到来を告げる小さな徴を置く", ["b_calm"], 0.17, 0.05, ["omen"], False),
            ("b_last_calm", "transition", 0.75, 0.04, "最後の平穏", "左ページ末尾に主体の穏やかな表情の小ゴマを置く", ["b_unease"], -0.06, 0.0, ["hero", "peace_desire"], True),
            ("b_threat_reveal", "revelation", 0.8, 0.05, "脅威の出現", "めくり直後の大ゴマで、より大きな脅威が姿を現す", ["b_last_calm", "b_unease"], 0.54, 0.15, ["greater_threat", "omen"], True),
            ("b_reaction", "crisis", 0.9, 0.04, "反応の凍結", "主体と仲間が凍りつく反応のコマで一拍置く", ["b_threat_reveal"], -0.06, 0.0, ["hero"], False),
            ("b_to_be_continued", "hook", 1.0, 0.04, "引きの最終コマ", "左ページ最終コマで答えのない問いを突きつけて終える", ["b_reaction", "b_resolution"], 0.13, 0.05, ["greater_threat", "unresolved_question", "local_problem"], True),
        ],
        "mechanisms": [
            ("m_cliffhanger", "zeigarnik", "未完了による持ち越し", "答えのない問いで終えることで、未完了の緊張が次回まで持続する", "最終コマの問い", "次回を読まずにはいられない", "未完了効果（ツァイガルニク効果）", 0.9, ["b_to_be_continued"], ["unresolved_question"]),
            ("m_relief_contrast", "contrast_effect", "安堵との落差", "解決と安堵で緊張を最低まで下げた後に脅威を出すと、落差で衝撃が最大化する", "束の間の安堵からの脅威出現", "安心しきった心に強い衝撃が走る", "対比効果", 0.8, ["b_calm", "b_threat_reveal"], ["peace_desire"]),
            ("m_episode_reward", "anticipation_reward", "一話の満足と次回への予期", "局所的な決着で満足を与えつつ、より大きな脅威で次の報酬を予期させる", "一話の山場と最終ページの脅威", "満足と渇望が同時に残る", "期待理論（予期と報酬）", 0.75, ["b_local_climax", "b_threat_reveal"], ["hero", "greater_threat"]),
            ("m_omen_irony", "dramatic_irony", "予兆の先行", "安堵の場面に読者だけが気づく小さな異物を置き、不穏さを先取りさせる", "片隅の小さな徴", "安堵の中に落ち着かなさが混ざる", "劇的アイロニー", 0.6, ["b_unease"], ["omen"]),
        ],
        "anchor": "戦いを終えて仲間と笑い合う主体の、無防備な笑顔",
        "drivers": ["解決後の束の間の安堵", "安堵に紛れた小さな異物", "答えのない最終コマ"],
        "emotions": ["高揚", "達成", "安堵", "微かな不穏", "戦慄", "渇望"],
        "template": "{{PROTAGONIST}}が{{LOCAL_PROBLEM}}を一話の山場で決着させ、解決と{{PEACE_MOMENT}}で緊張を最低まで下げる。安堵の場面の片隅に{{OMEN}}を置き、左ページ末尾に最後の平穏の小ゴマを置く。めくり直後の大ゴマで{{GREATER_THREAT}}を出現させ、凍りつく反応の一拍の後、左ページ最終コマで{{CLIFFHANGER_QUESTION}}を突きつけて終える。",
        "slots": [
            ("PROTAGONIST", "一話の問題を解決した主体", "agent", ["解決直後は無防備にする"], ("初勝利を挙げた主人公", "最初の成果を出したチーム", "最初の問いに答えた書き手")),
            ("LOCAL_PROBLEM", "一話の中の障害", "conflict", ["この回で必ず解決する"], ("町を荒らす怪物", "目先の売上不振", "最初の疑問")),
            ("PEACE_MOMENT", "束の間の安堵", "emotion", ["緊張度を全体で最低にする"], ("仲間との祝いの食卓", "成果を祝う社内の一場面", "一つの結論に満足する一文")),
            ("OMEN", "安堵に紛れる小さな異物", "symbol", ["片隅に小さく置く"], ("窓の外を横切る黒い羽", "見慣れない競合の名前", "説明できない小さな例外")),
            ("GREATER_THREAT", "より大きな脅威", "reversal", ["めくり直後の大ゴマ", "正体の核心は伏せる"], ("空を覆う巨大な影", "市場そのものを変える新勢力", "結論を覆しかねない新事実")),
            ("CLIFFHANGER_QUESTION", "次回へ持ち越される問い", "lack", ["最終コマに置く", "答えを与えない"], ("あれは一体何者なのか", "この変化にどう備えるのか", "この新事実は何を意味するのか")),
        ],
        "invariants": [
            "一話の局所的な対立は必ずこの回の中で決着させる",
            "脅威の出現の前に、緊張度を全体で最も低い地点まで下げる",
            "脅威はめくり直後の大ゴマで出し、正体の核心は伏せる",
            "最終コマは答えのない問いで終え、緊張度を全体の最高値にする",
        ],
        "guide": {
            "manga": "6ページ構成。p1（右）5コマ: 一話の山場をダッチアングルと広角24mmで描く。p2（左）6コマ: 解決と伏線の回収をMSで。p3（右）4コマ: 仲間との穏やかな会話をアイレベルのFSで描き、緊張を最低にする。p4（左）5コマ: 安堵の場面の片隅に小さな異物をECUで一コマ置き、最終コマに主体の穏やかな笑顔CUを小さく置く。めくって p5（右）1コマ断ち切り: より大きな脅威をELS・アオリ・広角18mmで出現させる。p6（左）2コマ: 凍りつく主体のCU（望遠135mm）と、答えのない問いを置いた最終コマで終える。",
            "marketing": "連載型のメールマガジンや動画シリーズで、一回ごとに読者の小さな課題を解決して満足を与えた後、次回予告として業界を変える大きな変化の兆しを示す。安心させた直後に兆しを大きく提示し、答えを与えずに問いで締めて次回の開封を促す。",
            "article": "連載記事の各回で、最初の問いにきちんと答えて満足させた上で、結論に説明できない小さな例外を添える。章の切れ目の後で結論を覆しかねない新事実を提示し、答えを出さずに次回への問いで締めくくる。",
        },
        "rhythm": {
            "page_count": 6,
            "panels_per_page": [5, 6, 4, 5, 1, 2],
            "emphasis_pages": [4, 5],
            "page_turn_reveals": [4],
            "spread_pages": [],
            "silent_panels_ratio": 0.2,
            "pacing_notes": [
                "p1（右）: 5コマ。一話の山場で局所的な対立を決着させる",
                "p2（左）: 6コマ。解決と伏線の回収",
                "p3（右）: 4コマ。束の間の安堵で緊張を最も低くする",
                "p4（左）: 5コマ。片隅に小さな異物を置き、最終コマに穏やかな笑顔",
                "p5（右・めくり直後）: 断ち切り1コマ。より大きな脅威の出現",
                "p6（左）: 2コマ。凍りつく反応と、答えのない問いを置いた最終コマ",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 19: 告白・関係変化の沈黙ゴマ
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_19_confession_silent_panels",
        "title": "告白・関係変化の沈黙ゴマ",
        "summary": "二人きりの場で主体が言いかけては呑み込む小ゴマを左ページ末尾に置き、めくり直後に告白の一言で関係の前提を反転させ、セリフのない沈黙のコマで応答を宙吊りにし、応答直前の口元のECUを経て、めくり直後に応答を示して関係の変化を確定させる骨格。",
        "tags": ["告白", "沈黙", "関係変化", "めくり", "間"],
        "lineage": ["告白場面の沈黙演出", "言いかけて呑む修辞（言いさし）", "二段のめくりによる宙吊り"],
        "waves": {
            "tension": [(0.0, 0.3), (0.15, 0.42), (0.25, 0.55), (0.3, 0.48), (0.35, 0.8), (0.45, 0.74), (0.55, 0.86), (0.6, 0.8), (0.65, 0.96), (0.8, 0.5), (1.0, 0.3)],
            "release": [(0.0, 0.2), (0.3, 0.1), (0.35, 0.15), (0.6, 0.05), (0.65, 0.4), (0.8, 0.8), (1.0, 0.85)],
            "audience_knowledge": [(0.0, 0.3), (0.3, 0.35), (0.35, 0.6), (0.6, 0.62), (0.65, 0.9), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.45), (0.3, 0.5), (0.35, 0.6), (0.6, 0.62), (0.65, 0.9), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.4), (0.3, 0.35), (0.35, 0.3), (0.6, 0.15), (0.65, 0.2), (0.8, 0.25), (1.0, 0.2)],
        },
        "nodes": [
            ("confessor", "agent", "confessor", "想いを告げる主体", "言いかけては呑み込み、ついに関係を変える一言を口にする主体", 0.9, "CONFESSOR"),
            ("beloved", "agent", "recipient", "想いを告げられる相手", "告白を受け止め、沈黙の後に応答する相手", 0.85, "RECIPIENT"),
            ("feeling", "lack", "unspoken_feeling", "言葉にされていない想い", "主体が抱え続け、告白の瞬間に初めて言葉になる感情", 0.9, "UNSPOKEN_FEELING"),
            ("relationship", "desire", "new_relationship", "関係が変わってほしいという願い", "主体と読者が共有する、新しい関係への期待と恐れ", 0.8, "NEW_RELATION"),
            ("silence", "symbol", "silent_panels", "沈黙のコマ", "告白と応答の間に置かれる、風や手元だけのセリフのないコマ群", 0.75, "SILENCE_SEQUENCE"),
        ],
        "edges": [
            ("confessor", "feeling", "asymmetry", 0.8, 0.7, "主体だけが想いを知っている", 0.0),
            ("confessor", "relationship", "pursuit", 0.85, 0.0, "主体は関係の変化を求める", 0.2),
            ("confessor", "beloved", "dependency", 0.7, 0.0, "主体の想いは相手の応答に委ねられる", 0.35),
            ("beloved", "confessor", "power_gradient", 0.65, 0.6, "応答の主導権は相手にある", 0.35),
            ("silence", "feeling", "embodiment", 0.75, 0.0, "沈黙のコマが言葉にならない想いを体現する", 0.45),
            ("feeling", "relationship", "opposition", 0.5, -0.3, "言葉にすれば今の関係が壊れるかもしれない", 0.25),
            ("beloved", "confessor", "opposition", 0.3, 0.0, "応答までのわずかな拒絶の可能性", 0.5),
        ],
        "beats": [
            ("b_setting", "premise", 0.0, 0.06, "二人きりの場", "夕暮れや帰り道など二人きりになる場を引きのコマで示す", [], 0.0, 0.0, ["confessor", "beloved"], False),
            ("b_approach", "escalation", 0.25, 0.06, "言い出しの直前", "主体が何かを言おうとして相手を呼び止める", ["b_setting"], 0.25, 0.05, ["confessor"], False),
            ("b_hesitate", "transition", 0.3, 0.04, "言いかけて呑む", "左ページ末尾で言葉を呑み込む主体の小ゴマを置く", ["b_approach"], -0.07, 0.0, ["confessor", "feeling"], True),
            ("b_confession", "structural_twist", 0.35, 0.05, "告白", "めくり直後に告白の一言を置き、関係の前提を反転させる", ["b_hesitate"], 0.32, 0.25, ["confessor", "feeling", "relationship"], True),
            ("b_silence", "crisis", 0.45, 0.06, "沈黙ゴマの連打", "風・髪・手元・足元などセリフのないコマを続け、応答を宙吊りにする", ["b_confession"], -0.06, 0.0, ["silence", "beloved"], False),
            ("b_held_breath", "escalation", 0.55, 0.05, "息を詰める", "主体の表情に寄り、拒絶の可能性への恐れを高める", ["b_silence"], 0.12, 0.0, ["confessor", "silence"], False),
            ("b_last_silence", "transition", 0.6, 0.04, "最後の沈黙", "左ページ末尾に相手の口元のECUを置き、応答の直前で止める", ["b_held_breath"], -0.06, 0.0, ["beloved"], False),
            ("b_answer", "revelation", 0.65, 0.05, "応答", "めくり直後のコマで相手の応答を示し、関係の変化を確定させる", ["b_last_silence", "b_confession"], 0.16, 0.28, ["beloved", "relationship"], True),
            ("b_new_relation", "afterglow", 1.0, 0.1, "新しい関係の余韻", "二人の距離が変わったことを引きの構図と並んだ影で示す", ["b_answer"], -0.66, 0.05, ["confessor", "beloved", "relationship"], False),
        ],
        "mechanisms": [
            ("m_withheld_words", "zeigarnik", "言いかけの未完了", "言いかけて呑み込む描写が、言葉の完了への強い欲求を生む", "言いかけて呑む小ゴマ", "読者が告白の瞬間を待ちわびる", "未完了効果（ツァイガルニク効果）", 0.8, ["b_approach", "b_hesitate"], ["feeling"]),
            ("m_silence_suspense", "loss_aversion", "沈黙による拒絶への恐れ", "応答の前の沈黙が、関係を失うかもしれない恐れを最大化する", "セリフのないコマの連打", "読者が主体と同じように息を詰める", "損失回避（プロスペクト理論）", 0.8, ["b_silence", "b_held_breath"], ["relationship", "silence"]),
            ("m_identity", "identity_projection", "告白する者への自己投影", "言えない想いを抱えた経験が、読者を主体に重ね合わせる", "言葉を探す主体の表情", "読者が自分の記憶を重ねて胸を締めつけられる", "自己投影（同一化）理論", 0.75, ["b_confession", "b_held_breath"], ["confessor"]),
            ("m_relief", "catharsis", "応答による解放", "沈黙で溜めた緊張が応答の瞬間に一挙に解ける", "めくり直後の応答のコマ", "安堵と喜びが同時に押し寄せる", "カタルシス（緊張解放）", 0.7, ["b_answer", "b_new_relation"], ["relationship"]),
        ],
        "anchor": "相手を呼び止めたまま言葉を探す主体の、握りしめた鞄の紐",
        "drivers": ["言いかけて呑み込む未完了", "応答を宙吊りにする沈黙のコマ", "関係を失うかもしれない恐れ"],
        "emotions": ["ときめき", "ためらい", "勇気", "沈黙", "恐れ", "安堵", "温もり"],
        "template": "{{CONFESSOR}}と{{RECIPIENT}}が二人きりになる場を引きで示し、主体が言いかけては呑み込む小ゴマを左ページ末尾に置く。めくり直後に{{CONFESSION_LINE}}で{{UNSPOKEN_FEELING}}を言葉にし、{{SILENCE_SEQUENCE}}で応答を宙吊りにする。左ページ末尾に応答直前の{{LAST_SILENT_DETAIL}}を置き、めくり直後に{{ANSWER}}を示して{{NEW_RELATION}}への変化を確定させる。",
        "slots": [
            ("CONFESSOR", "想いを告げる主体", "agent", ["一度は言いかけて呑み込ませる"], ("幼なじみに想いを寄せる少年", "長く迷い続けた見込み客", "本音を語れずにいた書き手")),
            ("RECIPIENT", "想いを告げられる相手", "agent", ["応答まで表情の全体を見せない"], ("背を向けて歩く幼なじみ", "提案を受ける担当者", "語りかけられる読者")),
            ("CONFESSION_LINE", "告白の一言", "reversal", ["短く", "めくり直後に置く"], ("ずっと好きだった", "本当は助けてほしかった", "私はこの方法で失敗した")),
            ("UNSPOKEN_FEELING", "言葉にされていない想い", "lack", ["告白まで言葉にしない"], ("隠し続けた恋心", "言い出せなかった悩み", "認められなかった失敗")),
            ("SILENCE_SEQUENCE", "応答前の沈黙のコマ群", "setting", ["セリフなし", "風・手元・足元など"], ("揺れる髪と夕日のコマ", "返答を待つ数行の余白", "読者に考えさせる空白の段落")),
            ("LAST_SILENT_DETAIL", "応答直前の細部", "emotion", ["左ページ最終コマ", "ECU"], ("開きかけた相手の口元", "返信を打つ指先", "結論の直前の一呼吸")),
            ("ANSWER", "相手の応答", "evidence", ["めくり直後のコマ"], ("私も、と小さく頷く", "一緒に解決しましょうという返答", "それで良いのだという肯定")),
            ("NEW_RELATION", "変化した関係", "goal", ["言葉で説明しすぎない"], ("並んで歩く二つの影", "伴走する関係の始まり", "書き手と読者の信頼")),
        ],
        "invariants": [
            "告白の前に一度、言いかけて呑み込む小ゴマを置く",
            "告白と応答はそれぞれめくり直後に置き、間に沈黙のコマを挟む",
            "沈黙のコマにはセリフを置かず、風・髪・手元などの細部で時間を刻む",
            "応答直前の左ページ最終コマは相手の口元や手元のECUにする",
            "沈黙のコマは右上から左下へ等間隔に並べ、時間の経過を一定の速度で流す",
        ],
        "guide": {
            "manga": "7ページ構成。p1（右）4コマ: 二人きりの帰り道をELSとFSで。p2（左）6コマ: 呼び止める主体のMS、相手の背中を描き、最終コマに言葉を呑む主体の口元CU（無音）。めくって p3（右）2コマ: 上段の大ゴマで告白の一言をCU・望遠85mmで、下段で相手の驚きの横顔。p4（左）6コマ: 揺れる髪・夕日・手元・足元のセリフのない小ゴマを等間隔に並べ、最終コマを相手の口元ECUにする。めくって p5（右）2コマ: 応答の一言を大きめのコマでCUに、下段で主体の表情。p6（左）4コマ: 互いの表情の切り返し。p7（右）3コマ: 並んだ二つの影を引きのELSで描く。",
            "marketing": "相談型サービスのLPで、悩みを抱えながら言い出せない顧客像を描き、一度ためらう様子を示してから本音を言葉にさせる。返答を待つ余白の後に、一緒に解決しましょうという応答を示し、伴走する関係が始まる情景で申込みへつなぐ。",
            "article": "書き手が自分の失敗を打ち明ける記事で、一度言い淀む記述を置いてから、章の冒頭で失敗を率直に告白する。読者に考えさせる空白の段落を挟んだ後、それで良いのだという肯定を示し、書き手と読者の信頼関係の上に本論を展開する。",
        },
        "rhythm": {
            "page_count": 7,
            "panels_per_page": [4, 6, 2, 6, 2, 4, 3],
            "emphasis_pages": [2, 4],
            "page_turn_reveals": [2, 4],
            "spread_pages": [],
            "silent_panels_ratio": 0.5,
            "pacing_notes": [
                "p1（右）: 4コマ。二人きりになる場を引きで示す",
                "p2（左）: 6コマ。呼び止めて言いかけ、最終コマで言葉を呑み込む",
                "p3（右・めくり直後）: 2コマ。告白の一言の大ゴマと相手の驚き",
                "p4（左）: 6コマ。セリフのない沈黙のコマを等間隔に並べ、最終コマを相手の口元にする",
                "p5（右・めくり直後）: 2コマ。応答の一言と主体の表情",
                "p6（左）: 4コマ。互いの表情の切り返し",
                "p7（右）: 3コマ。並んだ二つの影を引きで描く",
            ],
        },
    }
)

# ---------------------------------------------------------------------------
# 20: 覚醒・段階的開示
# ---------------------------------------------------------------------------
PATTERNS.append(
    {
        "id": "mr_20_awakening_staged_reveal",
        "title": "覚醒・段階的開示",
        "summary": "圧倒的な敵の一撃で主体を完全に打ち倒し、守るべきものを見た主体に覚醒の兆しを三段階で灯し、部分的覚醒の暴走でどん底へ落とした後、めくり直後の見開きで守る意志と結びついた完全覚醒を描き、力の誇示の後に代償を示して締める骨格。",
        "tags": ["覚醒", "逆転", "見開き", "段階的開示", "代償"],
        "lineage": ["英雄の潜在力の覚醒譚", "段階的パワーアップの演出", "どん底からの反転（ペリペテイア）"],
        "waves": {
            "tension": [(0.0, 0.7), (0.1, 0.82), (0.15, 0.6), (0.25, 0.68), (0.3, 0.62), (0.45, 0.78), (0.55, 0.5), (0.7, 0.99), (0.8, 0.84), (0.85, 0.88), (0.9, 0.6), (1.0, 0.45)],
            "release": [(0.0, 0.0), (0.15, 0.05), (0.3, 0.1), (0.45, 0.2), (0.55, 0.05), (0.7, 0.4), (0.85, 0.55), (1.0, 0.6)],
            "audience_knowledge": [(0.0, 0.2), (0.25, 0.35), (0.45, 0.5), (0.55, 0.52), (0.7, 0.85), (0.85, 0.9), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.1), (0.25, 0.2), (0.45, 0.35), (0.55, 0.35), (0.7, 0.85), (0.85, 0.92), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.5), (0.25, 0.55), (0.45, 0.6), (0.55, 0.5), (0.7, 0.4), (0.85, 0.45), (1.0, 0.35)],
        },
        "nodes": [
            ("awakener", "agent", "awakening_protagonist", "覚醒する主体", "敗北の底から、内に眠る力を段階的に引き出していく主体", 0.9, "AWAKENER"),
            ("overwhelming_foe", "agent", "overwhelming_foe", "圧倒的な敵", "主体を一度は完全に打ち倒す、既存の力では届かない相手", 0.85, "OVERWHELMING_FOE"),
            ("dormant_power", "lack", "dormant_power", "眠っている力", "主体自身も理解していない内なる力。段階的に兆しを見せる", 0.9, "DORMANT_POWER"),
            ("will_to_protect", "desire", "will_to_protect", "守りたいという意志", "力の解放の鍵となる、主体の根源的な動機", 0.85, "TRIGGER_MOTIVE"),
            ("aura", "symbol", "awakening_sign", "覚醒の兆し（光・紋様）", "段階ごとに強まる視覚的な徴。読者には先に見える", 0.75, "AWAKENING_SIGNS"),
            ("price", "symbol", "price_of_power", "力の代償", "覚醒と引き換えに主体が失うもの・背負うもの", 0.7, "POWER_COST"),
        ],
        "edges": [
            ("overwhelming_foe", "awakener", "power_gradient", 0.95, 0.9, "敵が主体を圧倒している", 0.0),
            ("overwhelming_foe", "awakener", "opposition", 0.9, 0.0, "主体と敵の生死をかけた対立", 0.0),
            ("aura", "dormant_power", "embodiment", 0.8, 0.0, "覚醒の兆しが眠る力を体現する", 0.25),
            ("awakener", "dormant_power", "asymmetry", 0.65, -0.6, "主体は自分の力を理解していない", 0.25),
            ("awakener", "will_to_protect", "pursuit", 0.85, 0.0, "主体は守るために立ち上がろうとする", 0.15),
            ("dormant_power", "will_to_protect", "dependency", 0.75, 0.0, "力の解放は守る意志に依存する", 0.55),
            ("awakener", "overwhelming_foe", "power_gradient", 0.9, 0.85, "覚醒によって勢力が逆転する", 0.7),
            ("price", "awakener", "opposition", 0.55, -0.4, "力の代償が主体を蝕む", 0.95),
        ],
        "beats": [
            ("b_defeat", "crisis", 0.1, 0.05, "敗北の一撃", "圧倒的な敵の一撃で主体が完全に打ち倒される", [], 0.12, 0.0, ["overwhelming_foe", "awakener"], True),
            ("b_fallen", "transition", 0.15, 0.04, "倒れ伏す", "倒れた主体を俯瞰で小さく描き、絶望を示す", ["b_defeat"], -0.22, 0.0, ["awakener"], False),
            ("b_first_sign", "setup", 0.25, 0.05, "最初の兆し", "守るべきものを見た主体の指先に、かすかな光が灯る", ["b_fallen"], 0.08, 0.1, ["aura", "will_to_protect"], False),
            ("b_partial", "escalation", 0.45, 0.06, "部分的覚醒", "力が半ば解放され一時的に押し返すが、制御しきれない", ["b_first_sign"], 0.16, 0.1, ["dormant_power", "awakener"], False),
            ("b_failure", "crisis", 0.55, 0.05, "制御不能のどん底", "力が暴走して自滅しかけ、左ページ末尾で再び倒れる", ["b_partial"], -0.28, 0.0, ["awakener", "dormant_power"], True),
            ("b_awakening", "climax", 0.7, 0.08, "見開きの覚醒", "めくり直後の見開きで、守る意志と結びついた力が完全に解放される", ["b_failure", "b_first_sign"], 0.49, 0.35, ["awakener", "dormant_power", "will_to_protect", "aura"], True),
            ("b_power", "payoff", 0.85, 0.05, "力の誇示", "覚醒した力で敵を圧倒し、兆しの意味を回収する", ["b_awakening"], 0.04, 0.05, ["awakener", "overwhelming_foe"], False),
            ("b_turn_tide", "transition", 0.9, 0.04, "決着の余波", "敵が退き、静けさが戻る", ["b_power"], -0.28, 0.02, ["overwhelming_foe"], False),
            ("b_cost", "costly_breakthrough", 1.0, 0.06, "代償の提示", "覚醒の代償として主体が失ったもの・背負ったものを示す", ["b_turn_tide", "b_awakening"], -0.15, 0.05, ["price", "awakener"], False),
        ],
        "mechanisms": [
            ("m_staged_reveal", "anticipation_reward", "段階的開示による予期", "兆しを段階的に強めることで、完全覚醒への予期と期待を積み上げる", "最初の兆しと部分的覚醒", "見開きの覚醒が予期の頂点で報われる", "期待理論（予期と報酬）", 0.85, ["b_first_sign", "b_partial", "b_awakening"], ["aura", "dormant_power"]),
            ("m_bottom_rebound", "contrast_effect", "どん底からの反転", "部分的覚醒の失敗で一度どん底に落とすと、完全覚醒の高揚が対比で増幅される", "制御不能の失敗", "反転の高揚が最大化する", "対比効果", 0.8, ["b_failure", "b_awakening"], ["awakener"]),
            ("m_protective_motive", "moral_judgment", "守る意志への共鳴", "力の解放が自己のためでなく誰かを守る意志と結びつくと、読者は道徳的に強く共鳴する", "守るべきものを見る主体の目", "覚醒が正しく美しいものとして受け取られる", "道徳基盤理論（ケアの基盤）", 0.7, ["b_first_sign", "b_awakening"], ["will_to_protect"]),
            ("m_self_efficacy", "self_efficacy", "潜在力の確信", "主体が自分の力を理解していく過程が、読者に自分にも眠る力があるという感覚を与える", "主体自身が驚く覚醒の表情", "読者が勇気づけられる", "自己効力感理論", 0.6, ["b_partial", "b_awakening", "b_cost"], ["awakener"]),
        ],
        "anchor": "倒れ伏しながら、守るべきものへ手を伸ばす主体の指先",
        "drivers": ["読者にだけ先に見える覚醒の兆し", "部分的覚醒の挫折", "守る意志と結びつく力"],
        "emotions": ["絶望", "微かな希望", "期待", "挫折", "高揚", "圧倒", "哀切"],
        "template": "{{OVERWHELMING_FOE}}の一撃で{{AWAKENER}}を完全に打ち倒し、守るべきものを見た主体に{{AWAKENING_SIGNS}}を段階的に灯す。{{DORMANT_POWER}}の部分的な解放で一度押し返すが制御できずに再び倒れ、左ページ末尾でどん底を示す。めくり直後の見開きで{{TRIGGER_MOTIVE}}と結びついた{{AWAKENING_IMAGE}}を描き、力の誇示の後に{{POWER_COST}}を示して締める。",
        "slots": [
            ("OVERWHELMING_FOE", "圧倒的な敵", "conflict", ["既存の力では届かない格にする"], ("城壁を砕く巨大な敵", "圧倒的な資本を持つ競合", "既存手法では解けない難題")),
            ("AWAKENER", "覚醒する主体", "agent", ["一度完全に敗北させる"], ("倒れた少年剣士", "撤退寸前の小さなチーム", "諦めかけた学習者")),
            ("AWAKENING_SIGNS", "段階的に強まる覚醒の兆し", "symbol", ["三段階で強める", "読者に先に見せる"], ("指先の光から全身の紋様へ", "小さな成功の兆しの積み重ね", "部分的な理解の手応え")),
            ("DORMANT_POWER", "眠っている力", "lack", ["主体自身にも理解させない"], ("血に眠る古い力", "チームに眠る独自の強み", "自分でも気づいていない才能")),
            ("TRIGGER_MOTIVE", "解放の鍵となる動機", "goal", ["自己のためでなく誰かのため"], ("仲間を守るという誓い", "顧客を守るという使命", "誰かに教えたいという願い")),
            ("AWAKENING_IMAGE", "見開きの覚醒の一枚絵", "reversal", ["めくり直後の見開き", "アオリの全身"], ("光をまとって立ち上がる全身", "独自の強みが市場を変える全体図", "理解が一気につながる統合図")),
            ("POWER_COST", "覚醒の代償", "cost", ["具体的に示す"], ("失われた片目の光", "手放した既存事業", "費やした膨大な時間")),
        ],
        "invariants": [
            "覚醒の前に主体を一度完全に敗北させる",
            "覚醒の兆しは三段階で強め、読者に先に見せる",
            "部分的覚醒は必ず失敗させ、完全覚醒の直前にどん底を置く",
            "完全覚醒はめくり直後の見開きに置き、守る意志と結びつける",
            "覚醒の後に必ず代償を示す",
        ],
        "guide": {
            "manga": "10ページ構成。p1（右）5コマ: 敵の一撃をダッチアングル・広角24mmで描く。p2（左）4コマ: 倒れ伏す主体を俯瞰のELSで小さく。p3（右）4コマ: 守るべきものを見る主体の目のECUと、指先のかすかな光のECU。p4（左）5コマ: 部分的覚醒で押し返すFSのアクション。p5（右）3コマ: 力の暴走をダッチアングルで。p6（左）5コマ: 再び倒れ、最終コマで守るべきものへ伸ばす手のECU（無音）。めくって p7〜p8 見開き: 光をまとって立ち上がる全身をアオリ・広角18mmで描き、右上の守るべきものから左下の主体へ視線を流す。p9（右）3コマ: 力の誇示と敵の後退。p10（左）4コマ: 代償をCUで静かに示す。",
            "marketing": "逆転劇型のブランドストーリーで、巨大な競合の前で撤退寸前に追い込まれた状況から始め、小さな成功の兆しを段階的に積み上げる。一度は拡大に失敗するどん底を示した後、顧客を守るという使命と結びついた独自の強みが市場を変える全体図を大きく示し、手放したものも誠実に語って信頼を得る。",
            "article": "成長体験の記事で、既存手法では解けない難題に挫折した経験から始め、部分的な理解の手応えを段階的に述べる。一度は理解したつもりで失敗した経験を挟み、章の切れ目で誰かに教えたいという動機が理解を統合した瞬間を図解し、そのために費やした時間という代償にも触れて締める。",
        },
        "rhythm": {
            "page_count": 10,
            "panels_per_page": [5, 4, 4, 5, 3, 5, 1, 1, 3, 4],
            "emphasis_pages": [6, 7],
            "page_turn_reveals": [6],
            "spread_pages": [6],
            "silent_panels_ratio": 0.2,
            "pacing_notes": [
                "p1（右）: 5コマ。圧倒的な敵の一撃で主体を打ち倒す",
                "p2（左）: 4コマ。倒れ伏す主体を俯瞰で小さく描く",
                "p3（右）: 4コマ。守るべきものを見た主体の指先に最初の兆し",
                "p4（左）: 5コマ。部分的覚醒で一時的に押し返す",
                "p5（右）: 3コマ。力が暴走し制御できなくなる",
                "p6（左）: 5コマ。再び倒れ、最終コマで守るべきものへ手を伸ばす",
                "p7（右・めくり直後）: 見開き大ゴマの右半分。守るべきものを右上に置く",
                "p8（左）: 見開き大ゴマの左半分。光をまとって立ち上がる全身",
                "p9（右）: 3コマ。覚醒した力の誇示と敵の後退",
                "p10（左）: 4コマ。覚醒の代償を静かに示す",
            ],
        },
    }
)


if __name__ == "__main__":
    raise SystemExit(main())
