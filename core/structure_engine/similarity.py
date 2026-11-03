"""構造類似度計算（コサイン類似度・多変量DTW・グラフ類似度）。

外部ライブラリに依存せず NumPy と標準ライブラリのみで実装する。

- ``cosine_similarity``: ベクトル間のコサイン類似度（ゼロベクトルを含む場合は 0.0）。
- ``dtw_path`` / ``dtw_distance``: Sakoe-Chiba 窓付き多変量 DTW。局所コストはユークリッド距離、
  累積コストを最適経路長で正規化した値を距離とする。
- ``wave_similarity``: 感情・情報波形を等間隔 n 点へ再標本化し、
  チャネル平均中心化コサイン（形状類似）と DTW 類似度 ``exp(-d/τ)``（τ=0.12、時間伸縮に頑健）を算出。
  hybrid は両者の平均 ``0.5·dtw_sim + 0.5·(cos+1)/2``。
- ``graph_similarity``: エッジ種別ヒストグラム・ノード種別分布・緊張エネルギー比の加重和。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from core.structure_engine.schemas import (
    DEFAULT_CHANNELS,
    DynamicTensionGraph,
    EdgeKind,
    EmotionalInformationWave,
    NodeKind,
    UniversalStructure,
)

__all__ = [
    "DTW_TEMPERATURE",
    "SimilarityMethod",
    "cosine_similarity",
    "default_window",
    "dtw_path",
    "dtw_distance",
    "wave_feature_matrix",
    "wave_similarity",
    "graph_similarity",
    "structure_similarity",
]

DTW_TEMPERATURE = 0.12
SimilarityMethod = Literal["dtw", "cosine", "hybrid"]
_EPS = 1e-12


def cosine_similarity(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """コサイン類似度 ∈ [-1, 1]。どちらかがゼロベクトルなら 0.0。"""
    va = np.asarray(a, dtype=float).ravel()
    vb = np.asarray(b, dtype=float).ravel()
    if va.shape != vb.shape:
        raise ValueError(f"cosine_similarity: 形状が一致しません {va.shape} vs {vb.shape}")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))


def default_window(n: int) -> int:
    """Sakoe-Chiba 窓幅の既定値 ``max(2, n // 5)``。"""
    return max(2, int(n) // 5)


def _as_2d(values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("DTW の入力は 1 次元系列または (長さ, 次元) の 2 次元配列である必要があります")
    return arr


def dtw_path(
    a: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    b: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    window: int | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """多変量 DTW の正規化距離と最適整列経路（0始まりインデックス対の列）を返す。

    窓は長さ比でスケールした対角線を中心に ``|j − i·(m−1)/(n−1)| ≤ w``。
    経路が存在するよう窓幅は長さの差に応じて自動拡張される。
    """
    xa = _as_2d(a)
    xb = _as_2d(b)
    if xa.shape[1] != xb.shape[1]:
        raise ValueError(f"DTW: 次元数が一致しません {xa.shape[1]} vs {xb.shape[1]}")
    n, m = xa.shape[0], xb.shape[0]
    if n == 0 or m == 0:
        raise ValueError("DTW: 空の系列は比較できません")
    base_window = default_window(max(n, m)) if window is None else max(0, int(window))
    ratio = (m - 1) / (n - 1) if n > 1 else 0.0
    if n == m:
        effective = base_window
    else:
        effective = max(base_window, int(math.ceil(max(ratio, 1.0))), 1)
    local = np.sqrt(((xa[:, None, :] - xb[None, :, :]) ** 2).sum(axis=2))
    inf = math.inf
    cost = np.full((n + 1, m + 1), inf, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        center = (i - 1) * ratio + 1 if n > 1 else 1.0
        j_start = max(1, int(math.floor(center - effective)))
        j_end = min(m, int(math.ceil(center + effective)))
        if n == 1:
            j_start, j_end = 1, m
        for j in range(j_start, j_end + 1):
            best = min(cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            if best < inf:
                cost[i, j] = local[i - 1, j - 1] + best
    if not math.isfinite(cost[n, m]):
        return dtw_path(a, b, window=max(n, m))
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        if i == 1 and j == 1:
            break
        candidates = (
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        )
        best_value = min(c[0] for c in candidates)
        for value, ni, nj in candidates:
            if value == best_value and ni >= 0 and nj >= 0:
                i, j = ni, nj
                break
    path.reverse()
    distance = float(cost[n, m] / len(path))
    return distance, path


def dtw_distance(
    a: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    b: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    window: int | None = None,
) -> float:
    """多変量 DTW の正規化距離（経路長で割った累積コスト）。"""
    return dtw_path(a, b, window)[0]


def wave_feature_matrix(
    wave: EmotionalInformationWave, channels: Sequence[str] = DEFAULT_CHANNELS, n: int = 32
) -> np.ndarray:
    """波形を等間隔 n 点へ再標本化した (n, チャネル数) 行列。"""
    arrays = wave.resample_arrays(n, channels)
    return np.column_stack([arrays[name] for name in channels])


def wave_similarity(
    w1: EmotionalInformationWave,
    w2: EmotionalInformationWave,
    *,
    method: SimilarityMethod = "hybrid",
    channels: Sequence[str] = DEFAULT_CHANNELS,
    n: int = 32,
    window: int | None = None,
) -> dict[str, Any]:
    """波形類似度。戻り値: ``{"score", "dtw_distance", "dtw_similarity", "cosine", "method"}``。"""
    if method not in ("dtw", "cosine", "hybrid"):
        raise ValueError(f"未知の類似度手法です: {method}")
    x = wave_feature_matrix(w1, channels, n)
    y = wave_feature_matrix(w2, channels, n)
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    cosine = cosine_similarity(xc.ravel(), yc.ravel())
    distance = dtw_distance(x, y, window)
    dtw_sim = math.exp(-distance / DTW_TEMPERATURE)
    cosine_score = (cosine + 1.0) / 2.0
    if method == "cosine":
        score = cosine_score
    elif method == "dtw":
        score = dtw_sim
    else:
        score = 0.5 * dtw_sim + 0.5 * cosine_score
    return {
        "score": float(min(1.0, max(0.0, score))),
        "dtw_distance": float(distance),
        "dtw_similarity": float(dtw_sim),
        "cosine": float(cosine),
        "method": method,
    }


def _kind_distribution(graph: DynamicTensionGraph) -> np.ndarray:
    counts = np.zeros(len(NodeKind), dtype=float)
    order = {kind.value: i for i, kind in enumerate(NodeKind)}
    for node in graph.nodes:
        counts[order[node.kind.value]] += 1.0
    total = counts.sum()
    return counts / total if total > 0 else counts


def graph_similarity(g1: DynamicTensionGraph, g2: DynamicTensionGraph) -> float:
    """力学グラフの類似度 ∈ [0, 1]。

    ``0.5·cos(エッジ種別ヒストグラム) + 0.3·cos(ノード種別分布) + 0.2·min(E1,E2)/max(E1,E2)``
    """
    h1 = g1.edge_kind_histogram()
    h2 = g2.edge_kind_histogram()
    kinds = [kind.value for kind in EdgeKind]
    edge_cos = max(0.0, cosine_similarity([h1[k] for k in kinds], [h2[k] for k in kinds]))
    node_cos = max(0.0, cosine_similarity(_kind_distribution(g1), _kind_distribution(g2)))
    e1, e2 = g1.tension_energy(), g2.tension_energy()
    if max(e1, e2) <= _EPS:
        ratio = 1.0
    else:
        ratio = min(e1, e2) / max(e1, e2)
    return float(min(1.0, max(0.0, 0.5 * edge_cos + 0.3 * node_cos + 0.2 * ratio)))


def structure_similarity(
    a: UniversalStructure,
    b: UniversalStructure,
    *,
    wave_weight: float = 0.7,
    method: SimilarityMethod = "hybrid",
) -> float:
    """波形類似度とグラフ類似度の加重和による構造全体の類似度 ∈ [0, 1]。"""
    weight = min(1.0, max(0.0, float(wave_weight)))
    wave_score = wave_similarity(a.wave, b.wave, method=method)["score"]
    graph_score = graph_similarity(a.tension_graph, b.tension_graph)
    return float(weight * wave_score + (1.0 - weight) * graph_score)
