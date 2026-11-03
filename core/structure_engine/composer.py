"""構造代数演算器（Structure Algebra）。

## 合成 ``merge_structures(A, B, weight_topology, weight_wave)``
構造Aの「対立トポロジー」と構造Bの「感情・情報開示波形」を補間合成する。

- ``weight_topology`` = 位相（力学グラフ）における A の比率（1.0 = 純A、0.0 = Bの位相をAのノードへ写像）
- ``weight_wave``     = 波形における B の比率（1.0 = 純B、0.0 = 純A）

1. **ノード整列**: コスト ``c_ij = [種別不一致]·1.0 + (1 − J_ij)·0.6 + |C_i − C_j|·0.4``
   （J: 役割/ラベルの文字 bigram Jaccard の大きい方、C: 強度加重次数中心性）を
   純 Python のハンガリアン法（矩形はパディング）で最小化し、コスト > 1.2 の対は未対応とする。
2. **位相補間**: 対応付け後の同一キー（source, target, kind）ごとに
   ``強度 = w_A·I_A + (1−w_A)·I_B``、``勾配 = w_A·G_A + (1−w_A)·G_B``。片側のみのエッジは強度を比率倍し、
   0.15 未満を除去。未対応Bノードは接続強度×(1−w_A) ≥ 0.15 のとき ``b_`` 接頭辞で追加。
3. **論理修復**: 裏切りエッジには逆向きの依存（信頼）を補完、主体消失時はAの主役を復元、
   非連結成分は主役へ弱い関係で接続。
4. **波形補間**: 両波形を N = max(len, 21) 点へ再標本化し、優勢側（weight_wave ≥ 0.5 なら B）を
   基準時間軸として、他方を緊張度の DTW 最適経路でワープしてから凸結合する。
5. **ビート再配置**: A のビートを DTW 経路由来の単調時間写像で合成時間軸へ移し、
   tension_delta / information_shift を合成波形から再計算。大域ピーク ±0.1 に転換点が無ければ
   最寄りの転換点を移動（不可能なら頂点ビートを挿入）。
6. **メカニズム・ブループリント・転用ガイド**の和集合合成（スロットはノード整列で同一視、衝突は ``B_``）。

## 変異 ``mutate_structure(s, mutation_type, intensity, seed)``
7種の構造変異（倒叙化・権力勾配反転・偽クライマックス挿入・時系列逆転・裏切り挿入・視点交換・中盤圧縮）。
intensity ∈ [0,1] は元波形との補間量（0 で波形不変）。seed により決定論的。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.common import clamp, stable_seed, truncate, utc_now_iso
from core.structure_engine.consistency import audit_structure, has_errors
from core.structure_engine.schemas import (
    SLOT_PLACEHOLDER_RE,
    TURNING_POINT_TYPES,
    WAVE_CHANNELS,
    WAVE_MIN_POINTS,
    BeatType,
    DynamicTensionGraph,
    EmotionalInformationWave,
    UniversalStructure,
    new_structure_id,
)
from core.structure_engine.similarity import dtw_path

__all__ = [
    "MutationType",
    "MUTATION_DESCRIPTIONS",
    "MUTATION_LABELS_JA",
    "MergeReport",
    "hungarian",
    "hungarian_assignment",
    "align_nodes",
    "interpolate_waves",
    "merge_structures",
    "merge_structures_with_report",
    "mutate_structure",
    "recompute_beat_dynamics",
]

_MATCH_MAX_COST = 1.2
_EDGE_KEEP_THRESHOLD = 0.15
_DYNAMICS_WINDOW = 0.06
_PEAK_TOLERANCE = 0.1


class MutationType(str, Enum):
    """構造変異の種別。"""

    INVERT_DISCLOSURE = "invert_disclosure"
    INVERT_POWER_GRADIENT = "invert_power_gradient"
    INSERT_FALSE_CLIMAX = "insert_false_climax"
    REVERSE_CHRONOLOGY = "reverse_chronology"
    ADD_BETRAYAL = "add_betrayal"
    PERSPECTIVE_SWAP = "perspective_swap"
    COMPRESS_MIDDLE = "compress_middle"


MUTATION_LABELS_JA: dict[str, str] = {
    MutationType.INVERT_DISCLOSURE.value: "倒叙化",
    MutationType.INVERT_POWER_GRADIENT.value: "権力勾配反転",
    MutationType.INSERT_FALSE_CLIMAX.value: "偽クライマックス挿入",
    MutationType.REVERSE_CHRONOLOGY.value: "時系列逆転",
    MutationType.ADD_BETRAYAL.value: "裏切り挿入",
    MutationType.PERSPECTIVE_SWAP.value: "視点交換",
    MutationType.COMPRESS_MIDDLE.value: "中盤圧縮",
}

MUTATION_DESCRIPTIONS: dict[str, str] = {
    MutationType.INVERT_DISCLOSURE.value: (
        "情報開示順序の逆転（倒叙化）。読者の認知量を冒頭20%で前倒しに引き上げ、作中人物との非対称性を"
        "サスペンス（劇的アイロニー）へ転換する。反転ビートは「人物側の気づき」として再解釈される。"
    ),
    MutationType.INVERT_POWER_GRADIENT.value: (
        "権力勾配の反転。権力勾配・情報非対称性エッジの向き（または勾配の符号）を反転し、"
        "優位と劣位を入れ替えて秩序を劣位の側から描き直す。"
    ),
    MutationType.INSERT_FALSE_CLIMAX.value: (
        "偽のクライマックスの挿入。t≈0.55±0.1 に緊張度の隆起と解放のスパイク（偽の勝利）を置き、"
        "直後に急落（崩壊・危機）を挿入して予測誤差を最大化する。"
    ),
    MutationType.REVERSE_CHRONOLOGY.value: (
        "時系列の逆転（結末先出し）。波形を時間反転して終端を整形し、結末から原因へ遡る順にビートを並べ替え、"
        "因果を「問いの連鎖」として再構成する。"
    ),
    MutationType.ADD_BETRAYAL.value: (
        "裏切りの挿入。最も強い依存関係の逆向きに t≈0.68 で顕在化する裏切りを追加し、"
        "露見（開示）と依存先喪失（危機）のビートを挿入する。"
    ),
    MutationType.PERSPECTIVE_SWAP.value: (
        "視点交換。主役と対立者の役割・スロット・重要度・共感の錨を入れ替え、両者間の勾配を反転する。"
    ),
    MutationType.COMPRESS_MIDDLE.value: (
        "中盤圧縮。単調な時間写像で中盤（t=0.25〜0.75）を圧縮し、終盤（反転以降）に紙幅を割り当てる。"
    ),
}


class MergeReport(BaseModel):
    """構造合成の報告。node_mapping は「Bのノード → 合成後ノードID（未採用は None）」。"""

    node_mapping: dict[str, str | None]
    weight_topology: float
    weight_wave: float
    dtw_alignment_cost: float
    fixes_applied: list[str] = Field(default_factory=list)
    issues_remaining: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 汎用ヘルパー
# ---------------------------------------------------------------------------


def _unique_id(base: str, taken: set[str], max_len: int = 64) -> str:
    """``^[a-z][a-z0-9_]{0,63}$`` を満たす一意ID。"""
    clean = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in base.lower()).strip("_")
    if not clean or not clean[0].isalpha():
        clean = f"n_{clean}" if clean else "node"
    clean = clean[:max_len]
    candidate = clean
    counter = 2
    while candidate in taken:
        suffix = f"_{counter}"
        candidate = clean[: max_len - len(suffix)] + suffix
        counter += 1
    taken.add(candidate)
    return candidate


def _unique_slot(base: str, taken: set[str]) -> str:
    clean = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in base.upper()).strip("_")
    if not clean or not clean[0].isalpha():
        clean = f"S_{clean}" if clean else "SLOT"
    clean = clean[:64]
    candidate = clean
    counter = 2
    while candidate in taken:
        suffix = f"_{counter}"
        candidate = clean[: 64 - len(suffix)] + suffix
        counter += 1
    taken.add(candidate)
    return candidate


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _gauss(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _wave_arrays(wave: EmotionalInformationWave, n: int) -> dict[str, np.ndarray]:
    arrays = wave.resample_arrays(n, WAVE_CHANNELS)
    return {name: np.asarray(arrays[name], dtype=float) for name in WAVE_CHANNELS}


def _wave_from_arrays(arrays: dict[str, np.ndarray]) -> EmotionalInformationWave:
    n = len(arrays["tension"])
    grid = np.linspace(0.0, 1.0, n)
    payload = {name: [round(float(v), 4) for v in np.clip(arrays[name], 0.0, 1.0)] for name in WAVE_CHANNELS}
    return EmotionalInformationWave(t=[round(float(v), 6) for v in grid], **payload)


def _wave_payload(wave: EmotionalInformationWave) -> dict[str, Any]:
    data = wave.model_dump(mode="json")
    data.pop("information_asymmetry", None)
    return data


def _protagonist_id(graph: DynamicTensionGraph) -> str:
    centrality = graph.degree_centrality()
    order = {node.id: i for i, node in enumerate(graph.nodes)}
    agents = graph.nodes_of_kind("agent")
    pool = agents if agents else list(graph.nodes)
    best = sorted(pool, key=lambda n: (-n.weight, -centrality.get(n.id, 0.0), order[n.id]))[0]
    return best.id


def _protagonist_from_dicts(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> str | None:
    if not nodes:
        return None
    degree = {node_id: 0.0 for node_id in nodes}
    for edge in edges:
        if edge["source"] in degree:
            degree[edge["source"]] += edge["intensity"]
        if edge["target"] in degree:
            degree[edge["target"]] += edge["intensity"]
    order = {node_id: i for i, node_id in enumerate(nodes)}
    agents = [nid for nid, n in nodes.items() if n["kind"] == "agent"]
    pool = agents if agents else list(nodes)
    return sorted(pool, key=lambda nid: (-float(nodes[nid]["weight"]), -degree[nid], order[nid]))[0]


def recompute_beat_dynamics(beats: list[dict[str, Any]], wave: EmotionalInformationWave) -> None:
    """ビート dict 群の tension_delta / information_shift を波形から再計算する（窓 h = max(0.06, span)）。"""
    for beat in beats:
        t = float(beat["t"])
        h = max(_DYNAMICS_WINDOW, float(beat.get("span", 0.05)))
        lo, hi = max(0.0, t - h), min(1.0, t + h)
        beat["tension_delta"] = round(clamp(wave.sample("tension", hi) - wave.sample("tension", lo), -1.0, 1.0), 4)
        beat["information_shift"] = round(
            clamp(wave.sample("information_asymmetry", hi) - wave.sample("information_asymmetry", lo), -1.0, 1.0), 4
        )


def _is_pivotal(beat: dict[str, Any]) -> bool:
    return bool(beat.get("is_turning_point")) or BeatType(beat["beat_type"]) in TURNING_POINT_TYPES


def _insert_beat(beats: list[dict[str, Any]], beat: dict[str, Any], *, link_next: bool = True) -> int:
    """t の順序を保って挿入し、直前ビートを因果に設定。直後ビートの因果へ自身を追加する。"""
    index = len(beats)
    for i, existing in enumerate(beats):
        if float(existing["t"]) > float(beat["t"]):
            index = i
            break
    if not beat.get("causes") and index > 0:
        beat["causes"] = [beats[index - 1]["id"]]
    beats.insert(index, beat)
    if link_next and index + 1 < len(beats):
        following = beats[index + 1]
        if beat["id"] not in following["causes"]:
            following["causes"] = [*following["causes"], beat["id"]]
    return index


def _ensure_peak_pivot(
    beats: list[dict[str, Any]],
    wave: EmotionalInformationWave,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    fixes: list[str],
) -> None:
    tension = wave.channel("tension")
    peak_t = float(wave.t[int(np.argmax(tension))])
    pivotal = [i for i, beat in enumerate(beats) if _is_pivotal(beat)]
    if any(abs(float(beats[i]["t"]) - peak_t) <= _PEAK_TOLERANCE + 1e-9 for i in pivotal):
        return
    if pivotal:
        nearest = min(pivotal, key=lambda k: abs(float(beats[k]["t"]) - peak_t))
        lo = float(beats[nearest - 1]["t"]) if nearest > 0 else 0.0
        hi = float(beats[nearest + 1]["t"]) if nearest + 1 < len(beats) else 1.0
        moved = clamp(peak_t, lo, hi)
        if abs(moved - peak_t) <= _PEAK_TOLERANCE:
            fixes.append(
                f"転換点 {beats[nearest]['id']} を t={float(beats[nearest]['t']):.2f} から緊張度ピーク t≈{moved:.2f} へ移動"
            )
            beats[nearest]["t"] = round(moved, 4)
            return
    taken = {beat["id"] for beat in beats}
    climax_id = _unique_id("climax_peak", taken)
    protagonist = _protagonist_from_dicts(nodes, edges)
    involved = [protagonist] if protagonist else []
    agents = [nid for nid, n in nodes.items() if n["kind"] == "agent" and nid != protagonist]
    if agents:
        involved.append(agents[0])
    _insert_beat(
        beats,
        {
            "id": climax_id,
            "beat_type": BeatType.CLIMAX.value,
            "t": round(peak_t, 4),
            "span": 0.04,
            "label": "合成波形の頂点",
            "function": "合成された緊張度の頂点で、力学上の対立と代償が最大化する。",
            "causes": [],
            "tension_delta": 0.0,
            "information_shift": 0.0,
            "involved_nodes": involved,
            "is_turning_point": True,
        },
    )
    fixes.append(f"緊張度ピーク t≈{peak_t:.2f} に頂点ビート {climax_id} を挿入")


# ---------------------------------------------------------------------------
# ハンガリアン法・ノード整列
# ---------------------------------------------------------------------------


def hungarian(cost: Sequence[Sequence[float]]) -> list[int]:
    """正方コスト行列の最小コスト完全割当（行 i → 列 assignment[i]）。O(n³) ポテンシャル法。"""
    n = len(cost)
    if n == 0:
        return []
    if any(len(row) != n for row in cost):
        raise ValueError("hungarian: 正方行列が必要です")
    inf = math.inf
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = float(cost[i0 - 1][j - 1]) - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def hungarian_assignment(cost_matrix: Sequence[Sequence[float]] | np.ndarray, pad_value: float | None = None) -> list[tuple[int, int]]:
    """矩形コスト行列の最小コスト割当。パディングで正方化し、実在する行列の対のみ返す。"""
    matrix = np.asarray(cost_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("hungarian_assignment: 2次元行列が必要です")
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return []
    size = max(rows, cols)
    pad = float(pad_value) if pad_value is not None else float(np.max(matrix)) + 1.0
    square = np.full((size, size), pad, dtype=float)
    square[:rows, :cols] = matrix
    assignment = hungarian(square.tolist())
    return [(i, j) for i, j in enumerate(assignment) if i < rows and j < cols]


def _bigrams(text: str) -> set[str]:
    compact = "".join(text.lower().split())
    if len(compact) >= 2:
        return {compact[i : i + 2] for i in range(len(compact) - 1)}
    return {compact} if compact else set()


def _jaccard(a: str, b: str) -> float:
    ga, gb = _bigrams(a), _bigrams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def align_nodes(
    graph_a: DynamicTensionGraph, graph_b: DynamicTensionGraph, *, max_cost: float = _MATCH_MAX_COST
) -> tuple[dict[str, str | None], dict[str, float]]:
    """B のノードを A のノードへ整列する。戻り値: (B id → A id | None, B id → 整列コスト)。"""
    ca = graph_a.degree_centrality()
    cb = graph_b.degree_centrality()
    matrix = np.zeros((len(graph_a.nodes), len(graph_b.nodes)), dtype=float)
    for i, na in enumerate(graph_a.nodes):
        for j, nb in enumerate(graph_b.nodes):
            kind_cost = 0.0 if na.kind == nb.kind else 1.0
            similarity = max(_jaccard(na.role, nb.role), _jaccard(na.label, nb.label))
            matrix[i, j] = kind_cost + (1.0 - similarity) * 0.6 + abs(ca[na.id] - cb[nb.id]) * 0.4
    mapping: dict[str, str | None] = {nb.id: None for nb in graph_b.nodes}
    costs: dict[str, float] = {}
    for i, j in hungarian_assignment(matrix, pad_value=max_cost + 1.0):
        cost = float(matrix[i, j])
        nb_id = graph_b.nodes[j].id
        costs[nb_id] = round(cost, 6)
        if cost <= max_cost:
            mapping[nb_id] = graph_a.nodes[i].id
    return mapping, costs


# ---------------------------------------------------------------------------
# 波形補間（DTW整列）
# ---------------------------------------------------------------------------


def _identity(t: float) -> float:
    return float(t)


def _interpolate_detailed(
    wave_a: EmotionalInformationWave,
    wave_b: EmotionalInformationWave,
    weight_b: float,
    *,
    n: int | None = None,
    align: bool = True,
) -> tuple[EmotionalInformationWave, Callable[[float], float], Callable[[float], float], float, str]:
    """合成波形・A/B時間軸→合成時間軸の写像・DTWコスト・基準側を返す。"""
    w = clamp(weight_b)
    size = max(int(n or 0), len(wave_a.t), len(wave_b.t), WAVE_MIN_POINTS)
    arrays_a = _wave_arrays(wave_a, size)
    arrays_b = _wave_arrays(wave_b, size)
    if not align:
        merged = {name: (1.0 - w) * arrays_a[name] + w * arrays_b[name] for name in WAVE_CHANNELS}
        cost, _ = dtw_path(arrays_a["tension"], arrays_b["tension"])
        return _wave_from_arrays(merged), _identity, _identity, float(cost), "none"
    reference = "b" if w >= 0.5 else "a"
    ref, other = (arrays_b, arrays_a) if reference == "b" else (arrays_a, arrays_b)
    cost, path = dtw_path(ref["tension"], other["tension"])
    ref_buckets: list[list[int]] = [[] for _ in range(size)]
    other_buckets: list[list[int]] = [[] for _ in range(size)]
    for i, j in path:
        ref_buckets[i].append(j)
        other_buckets[j].append(i)
    warped = {
        name: np.array([float(np.mean(other[name][bucket])) for bucket in ref_buckets], dtype=float)
        for name in WAVE_CHANNELS
    }
    if reference == "b":
        merged = {name: (1.0 - w) * warped[name] + w * arrays_b[name] for name in WAVE_CHANNELS}
    else:
        merged = {name: (1.0 - w) * arrays_a[name] + w * warped[name] for name in WAVE_CHANNELS}
    mean_ref_index = np.array([float(np.mean(bucket)) for bucket in other_buckets], dtype=float)
    mean_ref_index = np.maximum.accumulate(mean_ref_index)
    grid_index = np.arange(size, dtype=float)

    def map_other(t: float) -> float:
        position = clamp(t) * (size - 1)
        return float(np.interp(position, grid_index, mean_ref_index) / (size - 1))

    map_a = _identity if reference == "a" else map_other
    map_b = _identity if reference == "b" else map_other
    return _wave_from_arrays(merged), map_a, map_b, float(cost), reference


def interpolate_waves(
    wave_a: EmotionalInformationWave,
    wave_b: EmotionalInformationWave,
    weight_b: float,
    *,
    n: int | None = None,
    align: bool = True,
) -> EmotionalInformationWave:
    """波形 A・B の補間（weight_b = B の比率）。align=True で優勢側の時間軸へ DTW 整列してから凸結合する。"""
    wave, _, _, _, _ = _interpolate_detailed(wave_a, wave_b, weight_b, n=n, align=align)
    return wave


# ---------------------------------------------------------------------------
# 合成
# ---------------------------------------------------------------------------


def _merge_topology(
    a: UniversalStructure,
    b: UniversalStructure,
    w_a: float,
    mapping: dict[str, str | None],
    map_a: Callable[[float], float],
    map_b: Callable[[float], float],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str | None], str]:
    ga, gb = a.tension_graph, b.tension_graph
    w_b = 1.0 - w_a
    strength_a = {n.id: max((e.intensity for e in ga.incident_edges(n.id)), default=0.0) for n in ga.nodes}
    strength_b = {n.id: max((e.intensity for e in gb.incident_edges(n.id)), default=0.0) for n in gb.nodes}
    matched_a = {a_id: b_id for b_id, a_id in mapping.items() if a_id is not None}
    protagonist_a = _protagonist_id(ga)
    nodes_b = {n.id: n for n in gb.nodes}

    nodes: dict[str, dict[str, Any]] = {}
    for node in ga.nodes:
        keep = node.id in matched_a or w_a * strength_a[node.id] >= _EDGE_KEEP_THRESHOLD or node.id == protagonist_a
        if not keep:
            continue
        data = node.model_dump(mode="json")
        if node.id in matched_a:
            partner = nodes_b[matched_a[node.id]]
            data["weight"] = round(clamp(w_a * node.weight + w_b * partner.weight), 4)
        nodes[node.id] = data

    taken = {n.id for n in ga.nodes}
    merged_id_of_b: dict[str, str | None] = {}
    for node in gb.nodes:
        a_id = mapping.get(node.id)
        if a_id is not None and a_id in nodes:
            merged_id_of_b[node.id] = a_id
        elif w_b * strength_b[node.id] >= _EDGE_KEEP_THRESHOLD:
            new_node_id = _unique_id(f"b_{node.id}", taken)
            data = node.model_dump(mode="json")
            data["id"] = new_node_id
            data["weight"] = round(clamp(node.weight * (0.5 + 0.5 * w_b)), 4)
            nodes[new_node_id] = data
            merged_id_of_b[node.id] = new_node_id
        else:
            merged_id_of_b[node.id] = None

    pairs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in ga.edges:
        if edge.source in nodes and edge.target in nodes:
            pairs.setdefault((edge.source, edge.target, edge.kind.value), {"a": None, "b": None})["a"] = edge
    for edge in gb.edges:
        source = merged_id_of_b.get(edge.source)
        target = merged_id_of_b.get(edge.target)
        if source and target and source != target:
            pairs.setdefault((source, target, edge.kind.value), {"a": None, "b": None})["b"] = edge

    edges: list[dict[str, Any]] = []
    for (source, target, kind), pair in pairs.items():
        ea, eb = pair["a"], pair["b"]
        ta = map_a(ea.activation_t) if ea is not None and ea.activation_t is not None else None
        tb = map_b(eb.activation_t) if eb is not None and eb.activation_t is not None else None
        if ea is not None and eb is not None:
            intensity = w_a * ea.intensity + w_b * eb.intensity
            gradient = w_a * ea.gradient + w_b * eb.gradient
            if ta is not None and tb is not None:
                activation = w_a * ta + w_b * tb
            else:
                activation = ta if ta is not None else tb
            label = ea.label if w_a >= 0.5 else eb.label
        elif ea is not None:
            intensity, gradient, activation, label = w_a * ea.intensity, ea.gradient, ta, ea.label
        else:
            intensity, gradient, activation, label = w_b * eb.intensity, eb.gradient, tb, eb.label
        if intensity < _EDGE_KEEP_THRESHOLD:
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "kind": kind,
                "intensity": round(clamp(intensity), 4),
                "gradient": round(clamp(gradient, -1.0, 1.0), 4),
                "label": label,
                "activation_t": round(clamp(activation), 4) if activation is not None else None,
            }
        )
    return nodes, edges, merged_id_of_b, protagonist_a


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (edge["source"], edge["target"], edge["kind"])


def _components(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    for edge in edges:
        if edge["source"] in adjacency and edge["target"] in adjacency:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])
    seen: set[str] = set()
    components: list[list[str]] = []
    for node_id in nodes:
        if node_id in seen:
            continue
        stack, component = [node_id], []
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _repair_topology(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    fixes: list[str],
    protagonist_hint: str,
    a: UniversalStructure,
) -> str:
    """論理修復を行い、主役ノードIDを返す。"""
    if not any(node["kind"] == "agent" for node in nodes.values()):
        source_node = a.tension_graph.get_node(_protagonist_id(a.tension_graph))
        if source_node is not None:
            data = source_node.model_dump(mode="json")
            if data["id"] in nodes:
                data["id"] = _unique_id(f"{data['id']}_restored", set(nodes))
            data["kind"] = "agent"
            nodes[data["id"]] = data
            fixes.append(f"主体ノードが消失したため構造Aの主役 {data['id']} を復元")
    if len(nodes) < 2:
        for node in a.tension_graph.nodes:
            if node.id not in nodes:
                nodes[node.id] = node.model_dump(mode="json")
                fixes.append(f"ノード数を確保するため構造Aの {node.id} を復元")
            if len(nodes) >= 2:
                break
    protagonist = protagonist_hint if protagonist_hint in nodes else _protagonist_from_dicts(nodes, edges)
    if protagonist is None:
        raise ValueError("合成構造の主役ノードを決定できません")

    keys = {_edge_key(edge) for edge in edges}
    for edge in list(edges):
        if edge["kind"] != "betrayal":
            continue
        pair = {edge["source"], edge["target"]}
        if any(o["kind"] == "dependency" and {o["source"], o["target"]} == pair for o in edges):
            continue
        activation = edge.get("activation_t")
        dependency = {
            "source": edge["target"],
            "target": edge["source"],
            "kind": "dependency",
            "intensity": round(max(0.3, 0.6 * float(edge["intensity"])), 4),
            "gradient": 0.0,
            "label": "裏切りの前提となる信頼",
            "activation_t": round(max(0.0, float(activation) - 0.3), 4) if activation is not None else None,
        }
        if _edge_key(dependency) not in keys:
            edges.append(dependency)
            keys.add(_edge_key(dependency))
            fixes.append(f"裏切り {edge['source']}->{edge['target']} の前提として依存関係を補完")

    if not edges:
        other = next(node_id for node_id in nodes if node_id != protagonist)
        edges.append(
            {
                "source": protagonist,
                "target": other,
                "kind": "dependency",
                "intensity": 0.3,
                "gradient": 0.0,
                "label": "合成時の接続補完",
                "activation_t": None,
            }
        )
        keys.add(_edge_key(edges[-1]))
        fixes.append("エッジが消失したため主役と他ノードを接続")

    components = _components(nodes, edges)
    if len(components) > 1:
        for component in components:
            if protagonist in component:
                continue
            anchor = max(component, key=lambda nid: float(nodes[nid]["weight"]))
            kind = nodes[anchor]["kind"]
            if kind == "desire":
                link = (protagonist, anchor, "pursuit")
            elif kind == "symbol":
                link = (anchor, protagonist, "embodiment")
            else:
                link = (protagonist, anchor, "dependency")
            if link in keys:
                continue
            edges.append(
                {
                    "source": link[0],
                    "target": link[1],
                    "kind": link[2],
                    "intensity": 0.2,
                    "gradient": 0.0,
                    "label": "合成時の接続補完",
                    "activation_t": None,
                }
            )
            keys.add(link)
            fixes.append(f"非連結成分 {{{', '.join(component)}}} を主役 {protagonist} へ接続")
    return protagonist


def _merge_blueprint(
    a: UniversalStructure,
    b: UniversalStructure,
    merged_id_of_b: dict[str, str | None],
    peak_t: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    a_slot_names = {slot.name for slot in a.blueprint.slots}
    rename: dict[str, str] = {}
    for node in b.tension_graph.nodes:
        if not node.slot:
            continue
        merged_id = merged_id_of_b.get(node.id)
        a_node = a.tension_graph.get_node(merged_id) if merged_id else None
        if a_node is not None and a_node.slot and a_node.slot in a_slot_names:
            rename[node.slot] = a_node.slot
    slots = [slot.model_dump(mode="json") for slot in a.blueprint.slots]
    taken = set(a_slot_names)
    for slot in b.blueprint.slots:
        if slot.name in rename:
            continue
        if slot.name in taken:
            new_name = _unique_slot(f"B_{slot.name}", taken)
        else:
            new_name = slot.name
            taken.add(new_name)
        rename[slot.name] = new_name
        data = slot.model_dump(mode="json")
        data["name"] = new_name
        slots.append(data)

    def _rename(match: Any) -> str:
        return "{{" + rename.get(match.group(1), match.group(1)) + "}}"

    b_template = SLOT_PLACEHOLDER_RE.sub(_rename, b.blueprint.template).strip()
    template = f"{a.blueprint.template.rstrip()}\n{b_template}"
    invariants = _dedupe([*a.blueprint.invariants, *b.blueprint.invariants])
    invariants.append(f"緊張度のピークは t≈{peak_t:.2f} に置く（合成波形）")
    return {"template": template, "slots": slots, "invariants": invariants}, rename


def _merge_beats(
    a: UniversalStructure,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    map_a: Callable[[float], float],
    wave: EmotionalInformationWave,
    fixes: list[str],
) -> list[dict[str, Any]]:
    beats: list[dict[str, Any]] = []
    previous = 0.0
    for beat in a.beats.beats:
        data = beat.model_dump(mode="json")
        mapped = round(max(previous, clamp(map_a(beat.t))), 4)
        data["t"] = mapped
        data["involved_nodes"] = [node_id for node_id in beat.involved_nodes if node_id in nodes]
        beats.append(data)
        previous = mapped
    _ensure_peak_pivot(beats, wave, nodes, edges, fixes)
    recompute_beat_dynamics(beats, wave)
    return beats


def _merge_mechanics(
    a: UniversalStructure,
    b: UniversalStructure,
    beats: list[dict[str, Any]],
    node_ids: set[str],
    merged_id_of_b: dict[str, str | None],
    weight_topology: float,
    weight_wave: float,
) -> dict[str, Any]:
    beat_ids = {beat["id"] for beat in beats}
    first_beat_of_type: dict[str, str] = {}
    for beat in beats:
        first_beat_of_type.setdefault(beat["beat_type"], beat["id"])
    mechanisms: list[dict[str, Any]] = []
    taken: set[str] = set()
    categories: set[str] = set()
    for mechanism in a.mechanics.mechanisms:
        data = mechanism.model_dump(mode="json")
        data["linked_beats"] = [x for x in mechanism.linked_beats if x in beat_ids]
        data["linked_nodes"] = [x for x in mechanism.linked_nodes if x in node_ids]
        taken.add(data["id"])
        categories.add(mechanism.category.value)
        mechanisms.append(data)
    for mechanism in b.mechanics.mechanisms:
        if mechanism.category.value in categories:
            continue
        data = mechanism.model_dump(mode="json")
        data["id"] = _unique_id(data["id"] if data["id"] not in taken else f"b_{data['id']}", taken)
        linked_beats: list[str] = []
        for beat_id in mechanism.linked_beats:
            source_beat = b.beats.get(beat_id)
            if source_beat is None:
                continue
            target = first_beat_of_type.get(source_beat.beat_type.value)
            if target is not None and target not in linked_beats:
                linked_beats.append(target)
        data["linked_beats"] = linked_beats
        data["linked_nodes"] = _dedupe(
            merged_id_of_b[x] for x in mechanism.linked_nodes if merged_id_of_b.get(x) in node_ids
        )
        data["strength"] = round(clamp(mechanism.strength * (0.5 + 0.5 * weight_wave)), 4)
        categories.add(mechanism.category.value)
        mechanisms.append(data)
    return {
        "mechanisms": mechanisms,
        "empathy_anchor": a.mechanics.empathy_anchor if weight_topology >= 0.5 else b.mechanics.empathy_anchor,
        "immersion_drivers": _dedupe([*a.mechanics.immersion_drivers, *b.mechanics.immersion_drivers]),
        "emotion_sequence": list(
            b.mechanics.emotion_sequence if weight_wave >= 0.5 else a.mechanics.emotion_sequence
        ),
    }


def _combine_guides(a: UniversalStructure, b: UniversalStructure) -> dict[str, str]:
    combined: dict[str, str] = {}
    for field in ("manga", "marketing", "article"):
        text_a = getattr(a.transfer_guide, field).strip()
        text_b = getattr(b.transfer_guide, field).strip()
        if text_a and text_b:
            text = f"{text_a}\n［波形設計の参照：{b.metadata.title}］{text_b}"
        else:
            text = text_a or text_b
        combined[field] = truncate(text, 4000)
    return combined


def merge_structures_with_report(
    struct_a: UniversalStructure,
    struct_b: UniversalStructure,
    weight_topology: float = 0.7,
    weight_wave: float = 0.7,
    *,
    new_id: str | None = None,
    title: str | None = None,
) -> tuple[UniversalStructure, MergeReport]:
    """構造Aの対立トポロジーと構造Bの感情・情報開示波形を補間合成し、報告とともに返す。"""
    w_topology = clamp(weight_topology)
    w_wave = clamp(weight_wave)
    fixes: list[str] = []
    wave, map_a, map_b, cost, _reference = _interpolate_detailed(struct_a.wave, struct_b.wave, w_wave)
    mapping, _costs = align_nodes(struct_a.tension_graph, struct_b.tension_graph)
    nodes, edges, merged_id_of_b, protagonist_hint = _merge_topology(
        struct_a, struct_b, w_topology, mapping, map_a, map_b
    )
    peak_t = float(wave.t[int(np.argmax(wave.channel("tension")))])
    blueprint, slot_rename = _merge_blueprint(struct_a, struct_b, merged_id_of_b, peak_t)
    slot_names = {slot["name"] for slot in blueprint["slots"]}
    for merged_id in merged_id_of_b.values():
        if merged_id is None or merged_id not in nodes or struct_a.tension_graph.get_node(merged_id) is not None:
            continue
        slot = nodes[merged_id].get("slot")
        if slot:
            renamed = slot_rename.get(slot, slot)
            nodes[merged_id]["slot"] = renamed if renamed in slot_names else None
    protagonist = _repair_topology(nodes, edges, fixes, protagonist_hint, struct_a)
    beats = _merge_beats(struct_a, nodes, edges, map_a, wave, fixes)
    mechanics = _merge_mechanics(
        struct_a, struct_b, beats, set(nodes), merged_id_of_b, w_topology, w_wave
    )
    dominant, secondary = (struct_b, struct_a) if w_wave >= 0.5 else (struct_a, struct_b)
    rhythm = dominant.rhythm_profile or secondary.rhythm_profile
    origin = (
        struct_a.metadata.origin_domain
        if struct_a.metadata.origin_domain == struct_b.metadata.origin_domain
        else "hybrid"
    )
    merged_title = title or f"{struct_a.metadata.title} × {struct_b.metadata.title}"
    summary = (
        f"構造「{struct_a.metadata.title}」の対立トポロジー（A比率 {w_topology:.2f}）と、"
        f"構造「{struct_b.metadata.title}」の感情・情報開示波形（B比率 {w_wave:.2f}）を DTW 整列のうえ補間合成した純粋骨格。"
        f"主役ノードは {protagonist}、緊張度ピークは t≈{peak_t:.2f}。"
    )
    payload: dict[str, Any] = {
        "id": new_id or new_structure_id("mg"),
        "metadata": {
            "title": truncate(merged_title, 200),
            "summary": truncate(summary, 4000),
            "category": "custom",
            "source_kind": "merged",
            "tags": _dedupe([*struct_a.metadata.tags, *struct_b.metadata.tags, "構造合成"])[:16],
            "origin_domain": origin,
            "lineage": _dedupe([*struct_a.metadata.lineage, *struct_b.metadata.lineage]),
            "parent_ids": [struct_a.id, struct_b.id],
            "language": struct_a.metadata.language,
            "version": 1,
            "created_at": utc_now_iso(),
            "notes": truncate("合成時の修正: " + (" / ".join(fixes) if fixes else "なし"), 4000),
        },
        "tension_graph": {"nodes": list(nodes.values()), "edges": edges},
        "wave": _wave_payload(wave),
        "beats": {"beats": beats},
        "mechanics": mechanics,
        "blueprint": blueprint,
        "transfer_guide": _combine_guides(struct_a, struct_b),
        "rhythm_profile": rhythm.model_dump(mode="json") if rhythm is not None else None,
    }
    structure = UniversalStructure.model_validate(payload)
    issues = audit_structure(structure)
    if has_errors(issues):
        details = "; ".join(issue.message for issue in issues if issue.severity == "error")
        raise ValueError(f"合成構造に解消できない整合性エラーがあります: {details}")
    report = MergeReport(
        node_mapping=merged_id_of_b,
        weight_topology=w_topology,
        weight_wave=w_wave,
        dtw_alignment_cost=round(cost, 6),
        fixes_applied=fixes,
        issues_remaining=[f"{issue.code}: {issue.message}" for issue in issues],
    )
    return structure, report


def merge_structures(
    struct_a: UniversalStructure,
    struct_b: UniversalStructure,
    weight_topology: float = 0.7,
    weight_wave: float = 0.7,
    *,
    new_id: str | None = None,
    title: str | None = None,
) -> UniversalStructure:
    """構造Aの対立トポロジー × 構造Bの感情・情報開示波形の補間合成。"""
    structure, _ = merge_structures_with_report(
        struct_a, struct_b, weight_topology, weight_wave, new_id=new_id, title=title
    )
    return structure


# ---------------------------------------------------------------------------
# 変異
# ---------------------------------------------------------------------------


def _arrays_from_payload(wave: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t = np.asarray(wave["t"], dtype=float)
    return t, {name: np.asarray(wave[name], dtype=float) for name in WAVE_CHANNELS}


def _store_arrays(data: dict[str, Any], arrays: dict[str, np.ndarray]) -> EmotionalInformationWave:
    wave = _wave_from_arrays(arrays)
    data["wave"] = _wave_payload(wave)
    return wave


def _blend(original: dict[str, np.ndarray], mutated: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    return {
        name: np.clip((1.0 - alpha) * original[name] + alpha * mutated[name], 0.0, 1.0) for name in WAVE_CHANNELS
    }


def _graph_from_payload(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes = {node["id"]: node for node in data["tension_graph"]["nodes"]}
    return nodes, data["tension_graph"]["edges"]


def _add_or_strengthen_mechanism(
    data: dict[str, Any],
    category: str,
    template: dict[str, Any],
    link_beats: Sequence[str],
    link_nodes: Sequence[str],
    strength: float,
) -> None:
    mechanisms = data["mechanics"]["mechanisms"]
    for mechanism in mechanisms:
        if mechanism["category"] == category:
            mechanism["strength"] = round(max(float(mechanism["strength"]), strength), 4)
            mechanism["linked_beats"] = _dedupe([*mechanism["linked_beats"], *link_beats])
            mechanism["linked_nodes"] = _dedupe([*mechanism["linked_nodes"], *link_nodes])
            return
    taken = {m["id"] for m in mechanisms}
    entry = dict(template)
    entry["id"] = _unique_id(entry["id"], taken)
    entry["category"] = category
    entry["strength"] = round(strength, 4)
    entry["linked_beats"] = list(link_beats)
    entry["linked_nodes"] = list(link_nodes)
    mechanisms.append(entry)


def _append_invariant(data: dict[str, Any], text: str) -> None:
    invariants = data["blueprint"]["invariants"]
    if text not in invariants:
        invariants.append(text)


def _protagonist_in_payload(data: dict[str, Any]) -> str:
    nodes, edges = _graph_from_payload(data)
    protagonist = _protagonist_from_dicts(nodes, edges)
    if protagonist is None:
        raise ValueError("変異対象の構造に主役ノードがありません")
    return protagonist


def _mutate_invert_disclosure(data: dict[str, Any], alpha: float, rng: random.Random) -> list[str]:
    t, original = _arrays_from_payload(data["wave"])
    audience = original["audience_knowledge"]
    character = original["character_knowledge"]
    peak = max(float(audience.max()), float(character.max()), 0.85)
    early_level = 0.92 * peak
    ramp = _smoothstep(t / 0.2)
    floor = audience[0] + (early_level - audience[0]) * ramp
    audience_m = np.maximum.accumulate(np.maximum(audience, floor))
    suspense = np.clip(audience_m - character, 0.0, None)
    mutated = dict(original)
    mutated["audience_knowledge"] = audience_m
    mutated["tension"] = np.clip(original["tension"] + 0.12 * suspense * (1.0 - original["tension"]), 0.0, 1.0)
    wave = _store_arrays(data, _blend(original, mutated, alpha))

    beats = data["beats"]["beats"]
    protagonist = _protagonist_in_payload(data)
    taken = {beat["id"] for beat in beats}
    irony_id = _unique_id("irony_setup", taken)
    first_t = float(beats[0]["t"]) if beats else 0.02
    for beat in beats:
        if beat["beat_type"] in (BeatType.STRUCTURAL_TWIST.value, BeatType.REVELATION.value):
            beat["label"] = truncate("人物側の気づき：" + beat["label"], 200)
            beat["function"] = truncate("読者が既に知る真相に作中人物が到達する。" + beat["function"], 2000)
    irony_beat = {
        "id": irony_id,
        "beat_type": BeatType.REVELATION.value,
        "t": round(min(0.02, first_t), 4),
        "span": 0.04,
        "label": "真相の先行開示",
        "function": "作中人物より先に読者へ真相を開示し、以降の全場面を劇的アイロニーで二重化する。",
        "causes": [],
        "tension_delta": 0.0,
        "information_shift": 0.0,
        "involved_nodes": [protagonist],
        "is_turning_point": False,
    }
    beats.insert(0, irony_beat)
    if len(beats) > 1 and irony_id not in beats[1]["causes"]:
        beats[1]["causes"] = [irony_id, *beats[1]["causes"]]
    recompute_beat_dynamics(beats, wave)
    _add_or_strengthen_mechanism(
        data,
        "dramatic_irony",
        {
            "id": "inverted_irony",
            "name": "倒叙による劇的アイロニー",
            "description": "読者だけが真相を知るため、人物の言動すべてが二重の意味を帯びる。",
            "trigger": "冒頭での真相の先行開示",
            "effect": "露見の瞬間を待つ持続的なサスペンス",
            "theory": "劇的アイロニー／爆弾理論",
        },
        [irony_id],
        [protagonist],
        0.6 + 0.3 * alpha,
    )
    _append_invariant(data, "真相は冒頭20%以内に読者へ開示する（倒叙化）")
    return ["読者認知量を前倒しし、劇的アイロニーのビートを冒頭に挿入"]


def _mutate_invert_power_gradient(data: dict[str, Any], alpha: float, rng: random.Random) -> list[str]:
    nodes, edges = _graph_from_payload(data)
    graph_before = DynamicTensionGraph.model_validate(data["tension_graph"])
    balance = graph_before.power_balance()
    for edge in edges:
        if edge["kind"] not in ("power_gradient", "asymmetry"):
            continue
        if alpha >= 0.999:
            edge["source"], edge["target"] = edge["target"], edge["source"]
        else:
            edge["gradient"] = round(float(edge["gradient"]) * (1.0 - 2.0 * alpha), 4)
    ranked = sorted(balance, key=lambda nid: -abs(balance[nid]))[:2]
    for node_id in ranked:
        if abs(balance[node_id]) > 1e-9:
            node = nodes[node_id]
            node["description"] = truncate(
                (node.get("description") or "") + "（権力勾配の反転により優位と劣位が入れ替わる）", 2000
            )
    _add_or_strengthen_mechanism(
        data,
        "moral_judgment",
        {
            "id": "power_inversion_judgment",
            "name": "立場逆転への道徳的判断",
            "description": "優位にあった者の失墜と劣位の者の台頭が、受け手の公正感覚を強く刺激する。",
            "trigger": "権力勾配の反転",
            "effect": "溜飲が下がる感覚、または驕りへの戒め",
            "theory": "公正世界信念",
        },
        [],
        ranked,
        0.5 + 0.3 * alpha,
    )
    _append_invariant(data, "権力勾配を反転させ、劣位の側から秩序を描く")
    return ["権力勾配・情報非対称性エッジを反転"]


def _mutate_insert_false_climax(data: dict[str, Any], alpha: float, rng: random.Random) -> list[str]:
    t, original = _arrays_from_payload(data["wave"])
    n = len(t)
    raw_fc = 0.55 + (rng.random() - 0.5) * 0.2
    t_fc = round(round(raw_fc * (n - 1)) / (n - 1), 4)
    t_fc = min(max(t_fc, 0.45), 0.65)
    t_drop = round(min(0.9, t_fc + 0.1), 4)
    mutated = dict(original)
    mutated["tension"] = np.clip(
        original["tension"] + 0.35 * _gauss(t, t_fc, 0.03) - 0.32 * _gauss(t, t_drop, 0.035), 0.02, 1.0
    )
    mutated["release"] = np.clip(
        original["release"] + 0.45 * _gauss(t, t_fc, 0.03) - 0.25 * _gauss(t, t_drop, 0.04), 0.0, 1.0
    )
    mutated["audience_knowledge"] = np.maximum.accumulate(
        np.clip(original["audience_knowledge"] + 0.15 * _sigmoid((t - t_drop) / 0.02), 0.0, 1.0)
    )
    mutated["character_knowledge"] = np.clip(
        original["character_knowledge"] + 0.12 * _sigmoid((t - t_drop) / 0.02), 0.0, 1.0
    )
    wave = _store_arrays(data, _blend(original, mutated, alpha))

    beats = data["beats"]["beats"]
    protagonist = _protagonist_in_payload(data)
    taken = {beat["id"] for beat in beats}
    fv_id = _unique_id("false_victory", taken)
    collapse_id = _unique_id("false_victory_collapse", taken)
    _insert_beat(
        beats,
        {
            "id": fv_id,
            "beat_type": BeatType.FALSE_VICTORY.value,
            "t": t_fc,
            "span": 0.04,
            "label": "偽の勝利",
            "function": "目標達成に見える一時的な勝利で緊張を解放させ、受け手の予測を固定する。",
            "causes": [],
            "tension_delta": 0.0,
            "information_shift": 0.0,
            "involved_nodes": [protagonist],
            "is_turning_point": True,
        },
    )
    _insert_beat(
        beats,
        {
            "id": collapse_id,
            "beat_type": BeatType.CRISIS.value,
            "t": t_drop,
            "span": 0.04,
            "label": "勝利の崩壊",
            "function": "偽の勝利が誤認だったと判明し、獲得したはずのものが反転して失われる。",
            "causes": [fv_id],
            "tension_delta": 0.0,
            "information_shift": 0.0,
            "involved_nodes": [protagonist],
            "is_turning_point": True,
        },
    )
    recompute_beat_dynamics(beats, wave)
    _add_or_strengthen_mechanism(
        data,
        "prediction_error",
        {
            "id": "false_climax_error",
            "name": "偽のクライマックスによる予測誤差",
            "description": "解決したと信じた直後の崩壊が、受け手の予測モデルを破壊する。",
            "trigger": "一時的な勝利と直後の反転",
            "effect": "驚愕と、結末への不確実性の再燃",
            "theory": "予測符号化理論",
        },
        [fv_id, collapse_id],
        [protagonist],
        0.6 + 0.3 * alpha,
    )
    _append_invariant(data, f"偽のクライマックス t≈{t_fc:.2f} の直後（t≈{t_drop:.2f}）に崩壊を置く")
    return [f"t={t_fc:.2f} に偽の勝利、t={t_drop:.2f} に崩壊を挿入"]


def _mutate_reverse_chronology(data: dict[str, Any], alpha: float, rng: random.Random) -> list[str]:
    t, original = _arrays_from_payload(data["wave"])
    mutated = {name: original[name][::-1].copy() for name in WAVE_CHANNELS}
    final_knowledge = max(float(original["audience_knowledge"][-1]), 0.8)
    mutated["audience_knowledge"] = np.clip(
        0.55 * final_knowledge + (1.0 - 0.55 * final_knowledge) * _smoothstep(t), 0.0, 1.0
    )
    tension = mutated["tension"]
    tension[0] = max(float(tension[0]), 0.45)
    tension = np.clip(tension + 0.25 * _gauss(t, 0.88, 0.05), 0.0, 1.0)
    tension[-1] = min(float(tension[-1]), 0.3)
    mutated["tension"] = tension
    release = mutated["release"]
    release[-1] = max(float(release[-1]), 0.7)
    mutated["release"] = release
    wave = _store_arrays(data, _blend(original, mutated, alpha))

    beats = list(reversed(data["beats"]["beats"]))
    for index, beat in enumerate(beats):
        beat["t"] = round(clamp(1.0 - float(beat["t"])), 4)
        beat["causes"] = [beats[index - 1]["id"]] if index > 0 else []
    for index in range(1, len(beats)):
        if beats[index]["t"] < beats[index - 1]["t"]:
            beats[index]["t"] = beats[index - 1]["t"]
    if beats:
        beats[0]["beat_type"] = BeatType.HOOK.value
        beats[0]["label"] = truncate("結末の先出し：" + beats[0]["label"], 200)
    if len(beats) > 1:
        beats[-1]["beat_type"] = BeatType.PAYOFF.value
        beats[-1]["is_turning_point"] = True
        beats[-1]["label"] = truncate("起点の開示：" + beats[-1]["label"], 200)
    data["beats"]["beats"] = beats
    recompute_beat_dynamics(beats, wave)
    for edge in data["tension_graph"]["edges"]:
        if edge.get("activation_t") is not None:
            edge["activation_t"] = round(clamp(1.0 - float(edge["activation_t"])), 4)
    protagonist = _protagonist_in_payload(data)
    _add_or_strengthen_mechanism(
        data,
        "curiosity_gap",
        {
            "id": "reverse_chronology_gap",
            "name": "結末から原因への問い",
            "description": "結末を先に知ることで「なぜそうなったのか」という問いが全編を牽引する。",
            "trigger": "冒頭での結末の提示",
            "effect": "原因を遡って知りたいという持続的な好奇心",
            "theory": "情報ギャップ理論",
        },
        [beats[0]["id"]] if beats else [],
        [protagonist],
        0.6 + 0.3 * alpha,
    )
    _append_invariant(data, "結末を冒頭で示し、原因を遡って開示する（時系列逆転）")
    return ["波形を時間反転しビート順と因果を再構成"]


def _mutate_add_betrayal(data: dict[str, Any], alpha: float, rng: random.Random) -> list[str]:
    nodes, edges = _graph_from_payload(data)
    keys = {_edge_key(edge) for edge in edges}
    dependencies = sorted(
        (edge for edge in edges if edge["kind"] == "dependency"), key=lambda e: -float(e["intensity"])
    )
    notes: list[str] = []
    if dependencies:
        dependent, betrayer = dependencies[0]["source"], dependencies[0]["target"]
    else:
        protagonist = _protagonist_in_payload(data)
        others = sorted(
            (nid for nid in nodes if nid != protagonist),
            key=lambda nid: (0 if nodes[nid]["kind"] == "agent" else 1, -float(nodes[nid]["weight"])),
        )
        dependent, betrayer = protagonist, others[0]
        dependency = {
            "source": dependent,
            "target": betrayer,
            "kind": "dependency",
            "intensity": 0.7,
            "gradient": 0.0,
            "label": "信頼に基づく依存",
            "activation_t": 0.2,
        }
        edges.append(dependency)
        keys.add(_edge_key(dependency))
        notes.append(f"{dependent} -> {betrayer} の依存関係を新設")
    betrayal_key = (betrayer, dependent, "betrayal")
    strength = round(0.6 + 0.3 * alpha, 4)
    if betrayal_key in keys:
        for edge in edges:
            if _edge_key(edge) == betrayal_key:
                edge["intensity"] = round(max(float(edge["intensity"]), strength), 4)
                edge["activation_t"] = 0.68
    else:
        edges.append(
            {
                "source": betrayer,
                "target": dependent,
                "kind": "betrayal",
                "intensity": strength,
                "gradient": 0.0,
                "label": "信頼の裏切り",
                "activation_t": 0.68,
            }
        )
    t, original = _arrays_from_payload(data["wave"])
    mutated = dict(original)
    mutated["tension"] = np.clip(original["tension"] + 0.25 * _gauss(t, 0.72, 0.05), 0.0, 1.0)
    mutated["release"] = np.clip(original["release"] - 0.15 * _gauss(t, 0.72, 0.05), 0.0, 1.0)
    mutated["audience_knowledge"] = np.maximum.accumulate(
        np.clip(original["audience_knowledge"] + 0.2 * _sigmoid((t - 0.68) / 0.02), 0.0, 1.0)
    )
    mutated["character_knowledge"] = np.clip(
        original["character_knowledge"] + 0.25 * _sigmoid((t - 0.7) / 0.02), 0.0, 1.0
    )
    wave = _store_arrays(data, _blend(original, mutated, alpha))
    beats = data["beats"]["beats"]
    taken = {beat["id"] for beat in beats}
    reveal_id = _unique_id("betrayal_reveal", taken)
    loss_id = _unique_id("dependency_loss", taken)
    _insert_beat(
        beats,
        {
            "id": reveal_id,
            "beat_type": BeatType.REVELATION.value,
            "t": 0.68,
            "span": 0.04,
            "label": "裏切りの露見",
            "function": "依存していた相手が背後で逆の目的を持っていたと露見し、関係の前提が崩れる。",
            "causes": [],
            "tension_delta": 0.0,
            "information_shift": 0.0,
            "involved_nodes": [betrayer, dependent],
            "is_turning_point": True,
        },
    )
    _insert_beat(
        beats,
        {
            "id": loss_id,
            "beat_type": BeatType.CRISIS.value,
            "t": 0.76,
            "span": 0.04,
            "label": "依存先の喪失",
            "function": "拠り所を失った主体が、自分自身の判断だけで次の一手を選ばざるを得なくなる。",
            "causes": [reveal_id],
            "tension_delta": 0.0,
            "information_shift": 0.0,
            "involved_nodes": [dependent],
            "is_turning_point": True,
        },
    )
    recompute_beat_dynamics(beats, wave)
    _add_or_strengthen_mechanism(
        data,
        "moral_judgment",
        {
            "id": "betrayal_judgment",
            "name": "裏切りへの道徳的怒り",
            "description": "信頼の裏切りが受け手の公正感覚を刺激し、主体への共感と対立者への怒りを強める。",
            "trigger": "依存関係の逆向きの裏切り",
            "effect": "主体の反撃への強い期待",
            "theory": "道徳基盤理論",
        },
        [reveal_id],
        [betrayer, dependent],
        0.6 + 0.3 * alpha,
    )
    _append_invariant(data, "裏切りの前に依存（信頼）を描き、t≈0.68 で露見させる")
    notes.append(f"{betrayer} -> {dependent} の裏切りを t=0.68 に挿入")
    return notes


def _mutate_perspective_swap(data: dict[str, Any], alpha: float, rng: random.Random) -> list[str]:
    nodes, edges = _graph_from_payload(data)
    protagonist = _protagonist_in_payload(data)

    def link_strength(node_id: str, kinds: tuple[str, ...]) -> float:
        return max(
            (
                float(edge["intensity"])
                for edge in edges
                if edge["kind"] in kinds and {edge["source"], edge["target"]} == {protagonist, node_id}
            ),
            default=0.0,
        )

    candidates = [nid for nid in nodes if nid != protagonist]
    agents = [nid for nid in candidates if nodes[nid]["kind"] == "agent"]
    pool = agents if agents else candidates
    antagonist = max(
        pool,
        key=lambda nid: (
            link_strength(nid, ("opposition", "betrayal")),
            link_strength(nid, ("power_gradient", "asymmetry", "dependency", "pursuit", "embodiment")),
            float(nodes[nid]["weight"]),
        ),
    )
    first, second = nodes[protagonist], nodes[antagonist]
    for field in ("role", "slot", "weight"):
        first[field], second[field] = second[field], first[field]
    for edge in edges:
        if edge["kind"] in ("power_gradient", "asymmetry") and {edge["source"], edge["target"]} == {protagonist, antagonist}:
            edge["gradient"] = round(-float(edge["gradient"]) if alpha >= 0.999 else float(edge["gradient"]) * (1.0 - 2.0 * alpha), 4)
    mechanics = data["mechanics"]
    old_anchor = mechanics["empathy_anchor"]
    mechanics["empathy_anchor"] = truncate(
        f"視点交換：{second['label']}の側に立ち、{first['label']}を外側から見る。"
        f"元の共感点（{old_anchor}）は対立の動機として再解釈される。",
        1000,
    )
    for mechanism in mechanics["mechanisms"]:
        if mechanism["category"] in ("empathy_transfer", "identity_projection"):
            swapped = []
            for node_id in mechanism["linked_nodes"]:
                if node_id == protagonist:
                    swapped.append(antagonist)
                elif node_id == antagonist:
                    swapped.append(protagonist)
                else:
                    swapped.append(node_id)
            mechanism["linked_nodes"] = _dedupe(swapped)
    _append_invariant(data, "視点を対立者の側へ移し、元の主人公を外部から描く")
    return [f"{protagonist} と {antagonist} の視点・役割を交換"]


def _mutate_compress_middle(data: dict[str, Any], alpha: float, rng: random.Random) -> list[str]:
    t, original = _arrays_from_payload(data["wave"])
    middle_end = 0.75 - 0.2 * alpha
    knots_x = np.array([0.0, 0.25, 0.75, 1.0])
    knots_y = np.array([0.0, 0.25, middle_end, 1.0])

    def forward(value: float) -> float:
        return float(np.interp(value, knots_x, knots_y))

    inverse_positions = np.interp(t, knots_y, knots_x)
    warped = {name: np.interp(inverse_positions, t, original[name]) for name in WAVE_CHANNELS}
    wave = _store_arrays(data, warped)
    beats = data["beats"]["beats"]
    previous = 0.0
    for beat in beats:
        old_t = float(beat["t"])
        new_t = round(max(previous, clamp(forward(old_t))), 4)
        slope = (forward(min(1.0, old_t + 0.01)) - forward(max(0.0, old_t - 0.01))) / max(
            1e-6, min(1.0, old_t + 0.01) - max(0.0, old_t - 0.01)
        )
        beat["t"] = new_t
        beat["span"] = round(clamp(float(beat["span"]) * slope), 4)
        previous = new_t
    recompute_beat_dynamics(beats, wave)
    for edge in data["tension_graph"]["edges"]:
        if edge.get("activation_t") is not None:
            edge["activation_t"] = round(clamp(forward(float(edge["activation_t"]))), 4)
    _append_invariant(data, "中盤を圧縮し、終盤（反転以降）に紙幅を割く")
    return [f"中盤 t∈[0.25, 0.75] を [0.25, {middle_end:.2f}] へ圧縮"]


_MUTATORS: dict[MutationType, Callable[[dict[str, Any], float, random.Random], list[str]]] = {
    MutationType.INVERT_DISCLOSURE: _mutate_invert_disclosure,
    MutationType.INVERT_POWER_GRADIENT: _mutate_invert_power_gradient,
    MutationType.INSERT_FALSE_CLIMAX: _mutate_insert_false_climax,
    MutationType.REVERSE_CHRONOLOGY: _mutate_reverse_chronology,
    MutationType.ADD_BETRAYAL: _mutate_add_betrayal,
    MutationType.PERSPECTIVE_SWAP: _mutate_perspective_swap,
    MutationType.COMPRESS_MIDDLE: _mutate_compress_middle,
}


def mutate_structure(
    struct: UniversalStructure,
    mutation_type: MutationType | str,
    *,
    intensity: float = 1.0,
    seed: int = 0,
    new_id: str | None = None,
) -> UniversalStructure:
    """構造変異を適用した新しい構造を返す（元構造は変更しない）。"""
    kind = MutationType(mutation_type)
    alpha = clamp(intensity)
    rng = random.Random(stable_seed("mutate", struct.id, kind.value, int(seed)))
    data = struct.model_dump(mode="json")
    data["wave"].pop("information_asymmetry", None)
    notes = _MUTATORS[kind](data, alpha, rng)
    label = MUTATION_LABELS_JA[kind.value]
    meta = data["metadata"]
    meta["title"] = truncate(f"{struct.metadata.title}〔{label}〕", 200)
    meta["source_kind"] = "mutated"
    meta["parent_ids"] = [struct.id]
    meta["tags"] = _dedupe([*struct.metadata.tags, label])
    meta["version"] = struct.metadata.version + 1
    meta["created_at"] = utc_now_iso()
    meta["notes"] = truncate(f"変異: {label}（強度 {alpha:.2f}, seed {seed}）: " + " / ".join(notes), 4000)
    data["id"] = new_id or new_structure_id("mu")
    return UniversalStructure.model_validate(data)
