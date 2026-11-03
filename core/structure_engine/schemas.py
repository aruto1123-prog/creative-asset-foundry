"""5階層・普遍的構造オントロジー（Universal Structure Ontology）。

あらゆる物語・漫画・映画・ビジネス企画から表層のフレーバー（固有名詞・世界観）を剥ぎ取った
「純粋骨格」を、次の5階層で表現する。

- Layer 1 ``DynamicTensionGraph``: 主体・欲求・欠落・象徴のノードと、対立・依存・非対称性・
  権力勾配・裏切り（＋追求・体現）の有向エッジからなる力学トポロジー。
- Layer 2 ``EmotionalInformationWave``: 正規化時間 t∈[0,1]（21点＝20分割以上）上の緊張度・
  カタルシス解放度・読者認知量・作中人物認知量・認知負荷の波形。情報非対称性
  （読者認知量−作中人物認知量）は派生チャネルとして常に整合的に算出される。
- Layer 3 ``StructuralBeatChain``: フック〜余韻までのビートと因果DAG。
- Layer 4 ``CognitiveEmpathyMechanics``: 没入・感情移入を生む認知心理学的メカニズム。
- Layer 5 ``DomainAgnosticBlueprint``: ``{{SLOT}}`` 抽象変数を持つ分野非依存テンプレート。

LLM出力の頑健な受理のため未知キーは無視（``extra="ignore"``）し、層を横断する参照整合性は
``UniversalStructure`` / ``StructureLayers`` のモデル検証で強制する。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from core.common import new_id, utc_now_iso

__all__ = [
    "WAVE_MIN_POINTS",
    "WAVE_CHANNELS",
    "DEFAULT_CHANNELS",
    "ALL_CHANNELS",
    "NODE_ID_PATTERN",
    "SLOT_NAME_PATTERN",
    "STRUCTURE_ID_PATTERN",
    "SLOT_PLACEHOLDER_RE",
    "NodeKind",
    "EdgeKind",
    "NODE_KIND_LABELS_JA",
    "EDGE_KIND_LABELS_JA",
    "EDGE_TENSION_WEIGHTS",
    "TensionNode",
    "TensionEdge",
    "DynamicTensionGraph",
    "EmotionalInformationWave",
    "pchip_interpolate",
    "BeatType",
    "BEAT_TYPE_LABELS_JA",
    "TURNING_POINT_TYPES",
    "StructuralBeat",
    "StructuralBeatChain",
    "MechanismCategory",
    "MECHANISM_LABELS_JA",
    "CognitiveMechanism",
    "CognitiveEmpathyMechanics",
    "SlotCategory",
    "BlueprintSlot",
    "DomainAgnosticBlueprint",
    "StructureCategory",
    "STRUCTURE_CATEGORY_LABELS_JA",
    "SourceKind",
    "StructureMetadata",
    "PanelRhythmProfile",
    "TransferGuide",
    "StructureLayers",
    "UniversalStructure",
    "new_structure_id",
]

WAVE_MIN_POINTS = 21
WAVE_CHANNELS: tuple[str, ...] = (
    "tension",
    "release",
    "audience_knowledge",
    "character_knowledge",
    "cognitive_load",
)
DEFAULT_CHANNELS: tuple[str, ...] = ("tension", "release", "information_asymmetry", "cognitive_load")
ALL_CHANNELS: tuple[str, ...] = WAVE_CHANNELS + ("information_asymmetry",)

NODE_ID_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
SLOT_NAME_PATTERN = r"^[A-Z][A-Z0-9_]{1,63}$"
STRUCTURE_ID_PATTERN = r"^[a-z0-9][a-z0-9_\-]{2,95}$"
SLOT_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Z][A-Z0-9_]*)\s*\}\}")

_EPS = 1e-6


class _Base(BaseModel):
    """全スキーマ共通設定（未知キーは無視し、LLM出力を頑健に受理する）。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------------------
# Layer 1: Dynamic Tension Graph
# ---------------------------------------------------------------------------


class NodeKind(str, Enum):
    """力学グラフのノード種別。"""

    AGENT = "agent"
    DESIRE = "desire"
    LACK = "lack"
    SYMBOL = "symbol"


class EdgeKind(str, Enum):
    """力学グラフの有向エッジ種別。"""

    OPPOSITION = "opposition"
    DEPENDENCY = "dependency"
    ASYMMETRY = "asymmetry"
    POWER_GRADIENT = "power_gradient"
    BETRAYAL = "betrayal"
    PURSUIT = "pursuit"
    EMBODIMENT = "embodiment"


NODE_KIND_LABELS_JA: dict[str, str] = {
    NodeKind.AGENT.value: "主体",
    NodeKind.DESIRE.value: "欲求",
    NodeKind.LACK.value: "欠落",
    NodeKind.SYMBOL.value: "象徴",
}

EDGE_KIND_LABELS_JA: dict[str, str] = {
    EdgeKind.OPPOSITION.value: "対立",
    EdgeKind.DEPENDENCY.value: "依存",
    EdgeKind.ASYMMETRY.value: "非対称性",
    EdgeKind.POWER_GRADIENT.value: "権力勾配",
    EdgeKind.BETRAYAL.value: "裏切り",
    EdgeKind.PURSUIT.value: "追求",
    EdgeKind.EMBODIMENT.value: "体現",
}

# 力学的緊張エネルギーへの寄与係数（対立・裏切りが最大、体現は最小）
EDGE_TENSION_WEIGHTS: dict[str, float] = {
    EdgeKind.OPPOSITION.value: 1.0,
    EdgeKind.BETRAYAL.value: 1.2,
    EdgeKind.POWER_GRADIENT.value: 0.8,
    EdgeKind.ASYMMETRY.value: 0.6,
    EdgeKind.PURSUIT.value: 0.4,
    EdgeKind.DEPENDENCY.value: 0.3,
    EdgeKind.EMBODIMENT.value: 0.1,
}


class TensionNode(_Base):
    """力学グラフのノード（固有名詞を含まない抽象的役割）。"""

    id: str = Field(..., pattern=NODE_ID_PATTERN, description="ノードID（英小文字スネークケース）")
    kind: NodeKind = Field(..., description="agent=主体 / desire=欲求 / lack=欠落 / symbol=象徴")
    role: str = Field(..., min_length=1, max_length=80, description="抽象役割名（例: protagonist, gatekeeper）")
    label: str = Field(..., min_length=1, max_length=200, description="抽象的な説明ラベル（固有名詞禁止）")
    description: str = Field("", max_length=2000, description="力学上の機能の詳細")
    weight: float = Field(0.5, ge=0.0, le=1.0, description="構造上の重要度")
    slot: str | None = Field(None, pattern=r"^[A-Z][A-Z0-9_]*$", description="対応するブループリントのスロット名")


class TensionEdge(_Base):
    """力学グラフの有向エッジ。gradient は source から見た優位性（正=source優位）。"""

    source: str = Field(..., pattern=NODE_ID_PATTERN)
    target: str = Field(..., pattern=NODE_ID_PATTERN)
    kind: EdgeKind
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="関係の強度")
    gradient: float = Field(0.0, ge=-1.0, le=1.0, description="権力・情報の勾配（正=source優位）")
    label: str = Field("", max_length=300)
    activation_t: float | None = Field(None, ge=0.0, le=1.0, description="関係が顕在化する正規化時刻")

    @model_validator(mode="after")
    def _forbid_self_loop(self) -> "TensionEdge":
        if self.source == self.target:
            raise ValueError(f"自己ループのエッジは禁止です: {self.source}")
        return self


class DynamicTensionGraph(_Base):
    """Layer 1: 力学トポロジー構造。"""

    nodes: list[TensionNode] = Field(..., min_length=2)
    edges: list[TensionEdge] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_references(self) -> "DynamicTensionGraph":
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                raise ValueError(f"ノードIDが重複しています: {node.id}")
            seen.add(node.id)
        edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            if edge.source not in seen or edge.target not in seen:
                raise ValueError(f"エッジが未定義ノードを参照しています: {edge.source} -> {edge.target}")
            key = (edge.source, edge.target, edge.kind.value)
            if key in edge_keys:
                raise ValueError(f"重複エッジです: {edge.source} -[{edge.kind.value}]-> {edge.target}")
            edge_keys.add(key)
        return self

    def node_ids(self) -> list[str]:
        return [node.id for node in self.nodes]

    def get_node(self, node_id: str) -> TensionNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def nodes_of_kind(self, kind: NodeKind | str) -> list[TensionNode]:
        value = kind.value if isinstance(kind, NodeKind) else str(kind)
        return [node for node in self.nodes if node.kind.value == value]

    def edges_of_kind(self, kind: EdgeKind | str) -> list[TensionEdge]:
        value = kind.value if isinstance(kind, EdgeKind) else str(kind)
        return [edge for edge in self.edges if edge.kind.value == value]

    def incident_edges(self, node_id: str) -> list[TensionEdge]:
        return [edge for edge in self.edges if node_id in (edge.source, edge.target)]

    def adjacency(self, kind: EdgeKind | str | None = None, *, signed_gradient: bool = False) -> np.ndarray:
        """ノード順 ``node_ids()`` の隣接行列。signed_gradient=True で勾配×強度（符号付き）。"""
        ids = self.node_ids()
        index = {node_id: i for i, node_id in enumerate(ids)}
        matrix = np.zeros((len(ids), len(ids)), dtype=float)
        kind_value = None if kind is None else (kind.value if isinstance(kind, EdgeKind) else str(kind))
        for edge in self.edges:
            if kind_value is not None and edge.kind.value != kind_value:
                continue
            i, j = index[edge.source], index[edge.target]
            if signed_gradient:
                matrix[i, j] += edge.gradient * edge.intensity
            else:
                matrix[i, j] = max(matrix[i, j], edge.intensity)
        return matrix

    def power_balance(self) -> dict[str, float]:
        """各ノードの正味の権力・情報優位（権力勾配・非対称性エッジの勾配×強度の流出−流入）。"""
        balance = {node_id: 0.0 for node_id in self.node_ids()}
        for edge in self.edges:
            if edge.kind in (EdgeKind.POWER_GRADIENT, EdgeKind.ASYMMETRY):
                flow = edge.gradient * edge.intensity
                balance[edge.source] += flow
                balance[edge.target] -= flow
        return {key: round(value, 6) for key, value in balance.items()}

    def tension_energy(self) -> float:
        """力学的緊張エネルギー（エッジ種別係数×強度の総和。勾配系は |gradient| で増幅）。"""
        total = 0.0
        for edge in self.edges:
            weight = EDGE_TENSION_WEIGHTS[edge.kind.value]
            if edge.kind in (EdgeKind.POWER_GRADIENT, EdgeKind.ASYMMETRY):
                total += weight * edge.intensity * (0.5 + 0.5 * abs(edge.gradient))
            else:
                total += weight * edge.intensity
        return round(total, 6)

    def edge_kind_histogram(self) -> dict[str, float]:
        """エッジ種別ごとの強度合計（全種別をキーに持つ）。"""
        histogram = {kind.value: 0.0 for kind in EdgeKind}
        for edge in self.edges:
            histogram[edge.kind.value] += edge.intensity
        return {key: round(value, 6) for key, value in histogram.items()}

    def degree_centrality(self) -> dict[str, float]:
        """強度加重の次数中心性（最大値で正規化）。"""
        degree = {node_id: 0.0 for node_id in self.node_ids()}
        for edge in self.edges:
            degree[edge.source] += edge.intensity
            degree[edge.target] += edge.intensity
        peak = max(degree.values()) if degree else 0.0
        if peak <= 0:
            return {key: 0.0 for key in degree}
        return {key: round(value / peak, 6) for key, value in degree.items()}

    def to_dict_graph(self) -> dict[str, Any]:
        """UI描画用のノードリンク形式。"""
        return {
            "nodes": [node.model_dump(mode="json") for node in self.nodes],
            "links": [edge.model_dump(mode="json") for edge in self.edges],
        }


# ---------------------------------------------------------------------------
# Layer 2: Emotional & Information Wave
# ---------------------------------------------------------------------------


def _pchip_end_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    """PCHIP端点の形状保存3点推定。"""
    slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
    if np.sign(slope) != np.sign(delta0):
        return 0.0
    if np.sign(delta0) != np.sign(delta1) and abs(slope) > abs(3.0 * delta0):
        return 3.0 * delta0
    return float(slope)


def pchip_interpolate(xk: Sequence[float], yk: Sequence[float], x: Sequence[float] | np.ndarray) -> np.ndarray:
    """Fritsch–Carlson 法による区分3次エルミート単調補間（オーバーシュートしない）。"""
    xs = np.asarray(xk, dtype=float)
    ys = np.asarray(yk, dtype=float)
    query = np.asarray(x, dtype=float)
    if xs.ndim != 1 or xs.size != ys.size or xs.size == 0:
        raise ValueError("pchip_interpolate: キーフレームの形状が不正です")
    if xs.size == 1:
        return np.full_like(query, ys[0], dtype=float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError("pchip_interpolate: キーフレームの t は狭義単調増加である必要があります")
    h = np.diff(xs)
    delta = np.diff(ys) / h
    n = xs.size
    slopes = np.zeros(n, dtype=float)
    if n == 2:
        slopes[:] = delta[0]
    else:
        for k in range(1, n - 1):
            if delta[k - 1] * delta[k] <= 0:
                slopes[k] = 0.0
            else:
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                slopes[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        slopes[0] = _pchip_end_slope(h[0], h[1], delta[0], delta[1])
        slopes[-1] = _pchip_end_slope(h[-1], h[-2], delta[-1], delta[-2])
    clipped = np.clip(query, xs[0], xs[-1])
    idx = np.clip(np.searchsorted(xs, clipped, side="right") - 1, 0, n - 2)
    local = (clipped - xs[idx]) / h[idx]
    h00 = (1 + 2 * local) * (1 - local) ** 2
    h10 = local * (1 - local) ** 2
    h01 = local**2 * (3 - 2 * local)
    h11 = local**2 * (local - 1)
    return h00 * ys[idx] + h10 * h[idx] * slopes[idx] + h01 * ys[idx + 1] + h11 * h[idx] * slopes[idx + 1]


def _clean_unit_values(values: Any, name: str) -> list[float]:
    """[0,1] チャネル値の正規化（浮動小数誤差の微小超過のみ丸め、明確な範囲外は拒否）。"""
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} は数値配列である必要があります")
    cleaned: list[float] = []
    for i, raw in enumerate(list(values)):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{i}] が数値ではありません: {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name}[{i}] が有限値ではありません")
        if value < -_EPS or value > 1.0 + _EPS:
            raise ValueError(f"{name}[{i}]={value} は [0, 1] の範囲外です")
        cleaned.append(min(1.0, max(0.0, value)))
    return cleaned


class EmotionalInformationWave(_Base):
    """Layer 2: 正規化時間軸上の感情・情報波形。

    情報非対称性 A(t) = audience_knowledge(t) − character_knowledge(t) ∈ [−1, 1]
    （正: 読者が作中人物より多くを知る＝サスペンス、負: 作中人物/語りが情報を秘匿＝ミステリー・サプライズの溜め）。
    """

    t: list[float] = Field(..., description="正規化時間 0.0〜1.0（21点以上・狭義単調増加）")
    tension: list[float] = Field(..., description="緊張度 [0,1]")
    release: list[float] = Field(..., description="カタルシス解放度 [0,1]")
    audience_knowledge: list[float] = Field(..., description="読者/顧客の認知量 [0,1]")
    character_knowledge: list[float] = Field(..., description="作中人物の認知量 [0,1]")
    cognitive_load: list[float] = Field(..., description="認知負荷 [0,1]")

    @field_validator("tension", "release", "audience_knowledge", "character_knowledge", "cognitive_load", mode="before")
    @classmethod
    def _validate_channel(cls, value: Any, info: Any) -> list[float]:
        return _clean_unit_values(value, info.field_name)

    @field_validator("t", mode="before")
    @classmethod
    def _validate_time(cls, value: Any) -> list[float]:
        times = _clean_unit_values(value, "t")
        if times and abs(times[0]) <= _EPS:
            times[0] = 0.0
        if times and abs(times[-1] - 1.0) <= _EPS:
            times[-1] = 1.0
        return times

    @model_validator(mode="after")
    def _validate_shape(self) -> "EmotionalInformationWave":
        n = len(self.t)
        if n < WAVE_MIN_POINTS:
            raise ValueError(f"波形は {WAVE_MIN_POINTS} 点以上（20分割以上）が必要です（現在 {n} 点）")
        for name in WAVE_CHANNELS:
            length = len(getattr(self, name))
            if length != n:
                raise ValueError(f"チャネル {name} の長さ {length} が t の長さ {n} と一致しません")
        if self.t[0] != 0.0 or self.t[-1] != 1.0:
            raise ValueError("t は 0.0 で始まり 1.0 で終わる必要があります")
        for i in range(1, n):
            if self.t[i] <= self.t[i - 1]:
                raise ValueError(f"t は狭義単調増加である必要があります（index {i}）")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def information_asymmetry(self) -> list[float]:
        """派生チャネル: 読者認知量 − 作中人物認知量。"""
        return [round(a - c, 6) for a, c in zip(self.audience_knowledge, self.character_knowledge)]

    @property
    def n_points(self) -> int:
        return len(self.t)

    def channel(self, name: str) -> np.ndarray:
        """チャネル配列（``information_asymmetry`` を含む）。"""
        if name == "information_asymmetry":
            return np.asarray(self.audience_knowledge, dtype=float) - np.asarray(self.character_knowledge, dtype=float)
        if name == "t":
            return np.asarray(self.t, dtype=float)
        if name not in WAVE_CHANNELS:
            raise KeyError(f"未知のチャネルです: {name}")
        return np.asarray(getattr(self, name), dtype=float)

    def as_matrix(self, channels: Sequence[str] = DEFAULT_CHANNELS) -> np.ndarray:
        """形状 (チャネル数, 点数) の行列。"""
        return np.vstack([self.channel(name) for name in channels])

    def resample_arrays(self, n: int, channels: Iterable[str] = ALL_CHANNELS) -> dict[str, np.ndarray]:
        """等間隔 n 点へ線形補間した配列群（n>=2、内部計算用）。"""
        if n < 2:
            raise ValueError("resample_arrays: n は 2 以上が必要です")
        grid = np.linspace(0.0, 1.0, n)
        source_t = np.asarray(self.t, dtype=float)
        result: dict[str, np.ndarray] = {"t": grid}
        for name in channels:
            result[name] = np.interp(grid, source_t, self.channel(name))
        return result

    def resample(self, n: int = WAVE_MIN_POINTS) -> "EmotionalInformationWave":
        """等間隔 n 点（>=21）の波形へ再標本化する。"""
        if n < WAVE_MIN_POINTS:
            raise ValueError(f"resample: n は {WAVE_MIN_POINTS} 以上が必要です")
        arrays = self.resample_arrays(n, WAVE_CHANNELS)
        payload = {name: [round(float(v), 6) for v in np.clip(arrays[name], 0.0, 1.0)] for name in WAVE_CHANNELS}
        return EmotionalInformationWave(t=[round(float(v), 6) for v in arrays["t"]], **payload)

    def sample(self, channel: str, t: float) -> float:
        """任意時刻 t における線形補間値。"""
        position = min(1.0, max(0.0, float(t)))
        return float(np.interp(position, np.asarray(self.t, dtype=float), self.channel(channel)))

    def peak_index(self, channel: str = "tension") -> int:
        return int(np.argmax(self.channel(channel)))

    def peak_t(self, channel: str = "tension") -> float:
        return float(self.t[self.peak_index(channel)])

    def suspense_coefficient(self) -> float:
        """サスペンス係数 = mean( max(A,0) × tension )。読者だけが知る危機の持続量。"""
        asym = self.channel("information_asymmetry")
        tension = self.channel("tension")
        return float(np.clip(np.mean(np.clip(asym, 0.0, None) * tension), 0.0, 1.0))

    def surprise_coefficient(self) -> float:
        """サプライズ係数 = clip(2·max_i[Δaud⁺_i · (0.5 + 0.5·秘匿度_i)], 0, 1)。

        秘匿度_i = max(−A_i, 0)。最大の「一挙開示」の瞬間を、直前の秘匿で増幅して評価する。
        """
        audience = self.channel("audience_knowledge")
        asym = self.channel("information_asymmetry")
        jumps = np.clip(np.diff(audience), 0.0, None)
        if jumps.size == 0:
            return 0.0
        concealment = np.clip(-asym[:-1], 0.0, None)
        return float(np.clip(2.0 * np.max(jumps * (0.5 + 0.5 * concealment)), 0.0, 1.0))

    def catharsis_index(self) -> float:
        """カタルシス指数 = Σ(緊張の下降量 × 直後の解放度) / 緊張ピーク。"""
        tension = self.channel("tension")
        release = self.channel("release")
        drops = np.clip(tension[:-1] - tension[1:], 0.0, None)
        peak = float(np.max(tension))
        if peak <= _EPS:
            return 0.0
        return float(np.clip(np.sum(drops * release[1:]) / peak, 0.0, 1.0))

    def summary(self) -> dict[str, float]:
        """波形の要約統計。"""
        tension = self.channel("tension")
        return {
            "n_points": float(self.n_points),
            "peak_t": round(self.peak_t("tension"), 6),
            "peak_tension": round(float(np.max(tension)), 6),
            "mean_tension": round(float(np.mean(tension)), 6),
            "mean_release": round(float(np.mean(self.channel("release"))), 6),
            "mean_cognitive_load": round(float(np.mean(self.channel("cognitive_load"))), 6),
            "final_audience_knowledge": round(float(self.audience_knowledge[-1]), 6),
            "suspense": round(self.suspense_coefficient(), 6),
            "surprise": round(self.surprise_coefficient(), 6),
            "catharsis": round(self.catharsis_index(), 6),
        }

    @classmethod
    def from_keyframes(
        cls,
        keyframes: Mapping[str, Sequence[Sequence[float]]],
        n: int = WAVE_MIN_POINTS,
    ) -> "EmotionalInformationWave":
        """キーフレーム（チャネル→[(t, 値), ...]）から PCHIP 単調補間で n 点の波形を生成する。

        欠落チャネルの既定導出:
        - release: 緊張の下降量から ``clip(3·max(0, T[i−1]−T[i]) + 0.05)``
        - audience_knowledge: 0.1→1.0 の線形
        - character_knowledge: audience_knowledge と同一（非対称性なし）
        - cognitive_load: ``0.35 + 0.3·tension``
        """
        if n < WAVE_MIN_POINTS:
            raise ValueError(f"from_keyframes: n は {WAVE_MIN_POINTS} 以上が必要です")
        grid = np.linspace(0.0, 1.0, n)

        def _interp(points: Sequence[Sequence[float]]) -> np.ndarray:
            pairs = sorted((float(p[0]), float(p[1])) for p in points)
            if not pairs:
                raise ValueError("キーフレームが空です")
            dedup: list[tuple[float, float]] = []
            for t_value, y_value in pairs:
                if dedup and abs(dedup[-1][0] - t_value) <= _EPS:
                    dedup[-1] = (t_value, y_value)
                else:
                    dedup.append((t_value, y_value))
            if dedup[0][0] > 0.0:
                dedup.insert(0, (0.0, dedup[0][1]))
            if dedup[-1][0] < 1.0:
                dedup.append((1.0, dedup[-1][1]))
            xs = [p[0] for p in dedup]
            ys = [p[1] for p in dedup]
            return np.clip(pchip_interpolate(xs, ys, grid), 0.0, 1.0)

        if "tension" not in keyframes:
            raise ValueError("from_keyframes: tension のキーフレームは必須です")
        tension = _interp(keyframes["tension"])
        if "release" in keyframes:
            release = _interp(keyframes["release"])
        else:
            drops = np.concatenate([[0.0], np.clip(tension[:-1] - tension[1:], 0.0, None)])
            release = np.clip(3.0 * drops + 0.05, 0.0, 1.0)
        audience = _interp(keyframes["audience_knowledge"]) if "audience_knowledge" in keyframes else np.linspace(0.1, 1.0, n)
        character = _interp(keyframes["character_knowledge"]) if "character_knowledge" in keyframes else audience.copy()
        load = _interp(keyframes["cognitive_load"]) if "cognitive_load" in keyframes else np.clip(0.35 + 0.3 * tension, 0.0, 1.0)

        def _round(values: np.ndarray) -> list[float]:
            return [round(float(v), 4) for v in values]

        return cls(
            t=[round(float(v), 6) for v in grid],
            tension=_round(tension),
            release=_round(release),
            audience_knowledge=_round(audience),
            character_knowledge=_round(character),
            cognitive_load=_round(load),
        )


# ---------------------------------------------------------------------------
# Layer 3: Structural Beats & Turning Points
# ---------------------------------------------------------------------------


class BeatType(str, Enum):
    """構造ビート種別。"""

    HOOK = "hook"
    PREMISE = "premise"
    SETUP = "setup"
    ESCALATION = "escalation"
    LOCAL_OPTIMIZATION_LIMIT = "local_optimization_limit"
    FALSE_VICTORY = "false_victory"
    CRISIS = "crisis"
    REVELATION = "revelation"
    STRUCTURAL_TWIST = "structural_twist"
    COSTLY_BREAKTHROUGH = "costly_breakthrough"
    CLIMAX = "climax"
    PAYOFF = "payoff"
    AFTERGLOW = "afterglow"
    TRANSITION = "transition"


BEAT_TYPE_LABELS_JA: dict[str, str] = {
    BeatType.HOOK.value: "フック",
    BeatType.PREMISE.value: "前提の提示",
    BeatType.SETUP.value: "仕込み",
    BeatType.ESCALATION.value: "段階的上昇",
    BeatType.LOCAL_OPTIMIZATION_LIMIT.value: "局所的最適化の限界",
    BeatType.FALSE_VICTORY.value: "偽の勝利",
    BeatType.CRISIS.value: "危機・どん底",
    BeatType.REVELATION.value: "真相の開示",
    BeatType.STRUCTURAL_TWIST.value: "構造的反転",
    BeatType.COSTLY_BREAKTHROUGH.value: "代償を伴う突破",
    BeatType.CLIMAX.value: "クライマックス",
    BeatType.PAYOFF.value: "回収",
    BeatType.AFTERGLOW.value: "余韻",
    BeatType.TRANSITION.value: "転換",
}

TURNING_POINT_TYPES: frozenset[BeatType] = frozenset(
    {
        BeatType.FALSE_VICTORY,
        BeatType.CRISIS,
        BeatType.REVELATION,
        BeatType.STRUCTURAL_TWIST,
        BeatType.COSTLY_BREAKTHROUGH,
        BeatType.CLIMAX,
    }
)


class StructuralBeat(_Base):
    """因果チェーン上の1ビート。causes は先行ビートIDのみを参照できる。"""

    id: str = Field(..., pattern=NODE_ID_PATTERN)
    beat_type: BeatType
    t: float = Field(..., ge=0.0, le=1.0, description="正規化時刻")
    span: float = Field(0.05, ge=0.0, le=1.0, description="持続幅")
    label: str = Field(..., min_length=1, max_length=200)
    function: str = Field(..., min_length=1, max_length=2000, description="抽象的な構造機能")
    causes: list[str] = Field(default_factory=list, description="因果的に先行するビートID")
    tension_delta: float = Field(0.0, ge=-1.0, le=1.0)
    information_shift: float = Field(0.0, ge=-1.0, le=1.0, description="情報非対称性の変化量")
    involved_nodes: list[str] = Field(default_factory=list)
    is_turning_point: bool = False


class StructuralBeatChain(_Base):
    """Layer 3: ビート列（t 非減少）と因果DAG。"""

    beats: list[StructuralBeat] = Field(..., min_length=3)

    @model_validator(mode="after")
    def _validate_chain(self) -> "StructuralBeatChain":
        seen: set[str] = set()
        previous_t = -1.0
        for index, beat in enumerate(self.beats):
            if beat.id in seen:
                raise ValueError(f"ビートIDが重複しています: {beat.id}")
            if beat.t < previous_t - _EPS:
                raise ValueError(f"ビートは t の非減少順に並べる必要があります（index {index}: {beat.id}）")
            for cause in beat.causes:
                if cause not in seen:
                    raise ValueError(f"ビート {beat.id} の因果 {cause} は先行ビートを参照していません")
            seen.add(beat.id)
            previous_t = beat.t
        return self

    def ids(self) -> list[str]:
        return [beat.id for beat in self.beats]

    def get(self, beat_id: str) -> StructuralBeat | None:
        for beat in self.beats:
            if beat.id == beat_id:
                return beat
        return None

    def index_of(self, beat_id: str) -> int:
        for index, beat in enumerate(self.beats):
            if beat.id == beat_id:
                return index
        raise KeyError(beat_id)

    def turning_points(self) -> list[StructuralBeat]:
        """明示的に転換点フラグを持つビート。"""
        return [beat for beat in self.beats if beat.is_turning_point]

    def pivotal_beats(self) -> list[StructuralBeat]:
        """転換点フラグまたは転換点種別のビート。"""
        return [beat for beat in self.beats if beat.is_turning_point or beat.beat_type in TURNING_POINT_TYPES]

    def by_type(self, beat_type: BeatType | str) -> list[StructuralBeat]:
        value = beat_type.value if isinstance(beat_type, BeatType) else str(beat_type)
        return [beat for beat in self.beats if beat.beat_type.value == value]

    def causal_edges(self) -> list[tuple[str, str]]:
        return [(cause, beat.id) for beat in self.beats for cause in beat.causes]

    def causal_depth(self) -> int:
        """最長因果パス上のビート数。"""
        depth: dict[str, int] = {}
        for beat in self.beats:
            depth[beat.id] = 1 + max((depth[c] for c in beat.causes), default=0)
        return max(depth.values()) if depth else 0


# ---------------------------------------------------------------------------
# Layer 4: Cognitive & Empathy Mechanics
# ---------------------------------------------------------------------------


class MechanismCategory(str, Enum):
    """認知心理学的・物語論的メカニズム分類。"""

    CURIOSITY_GAP = "curiosity_gap"
    DRAMATIC_IRONY = "dramatic_irony"
    EMPATHY_TRANSFER = "empathy_transfer"
    PREDICTION_ERROR = "prediction_error"
    COGNITIVE_DISSONANCE = "cognitive_dissonance"
    IDENTITY_PROJECTION = "identity_projection"
    MORAL_JUDGMENT = "moral_judgment"
    LOSS_AVERSION = "loss_aversion"
    SOCIAL_PROOF = "social_proof"
    ZEIGARNIK = "zeigarnik"
    PEAK_END = "peak_end"
    CATHARSIS = "catharsis"
    SCARCITY = "scarcity"
    RECIPROCITY = "reciprocity"
    AUTHORITY = "authority"
    COMMITMENT_CONSISTENCY = "commitment_consistency"
    NARRATIVE_TRANSPORTATION = "narrative_transportation"
    ANTICIPATION_REWARD = "anticipation_reward"
    CONTRAST_EFFECT = "contrast_effect"
    SELF_EFFICACY = "self_efficacy"


MECHANISM_LABELS_JA: dict[str, str] = {
    MechanismCategory.CURIOSITY_GAP.value: "情報ギャップ（好奇心）",
    MechanismCategory.DRAMATIC_IRONY.value: "劇的アイロニー",
    MechanismCategory.EMPATHY_TRANSFER.value: "共感移入",
    MechanismCategory.PREDICTION_ERROR.value: "予測誤差（スキーマ違反）",
    MechanismCategory.COGNITIVE_DISSONANCE.value: "認知的不協和",
    MechanismCategory.IDENTITY_PROJECTION.value: "自己投影",
    MechanismCategory.MORAL_JUDGMENT.value: "道徳的判断",
    MechanismCategory.LOSS_AVERSION.value: "損失回避",
    MechanismCategory.SOCIAL_PROOF.value: "社会的証明",
    MechanismCategory.ZEIGARNIK.value: "未完了効果",
    MechanismCategory.PEAK_END.value: "ピーク・エンドの法則",
    MechanismCategory.CATHARSIS.value: "カタルシス",
    MechanismCategory.SCARCITY.value: "希少性",
    MechanismCategory.RECIPROCITY.value: "返報性",
    MechanismCategory.AUTHORITY.value: "権威",
    MechanismCategory.COMMITMENT_CONSISTENCY.value: "コミットメントと一貫性",
    MechanismCategory.NARRATIVE_TRANSPORTATION.value: "物語移入",
    MechanismCategory.ANTICIPATION_REWARD.value: "予期と報酬",
    MechanismCategory.CONTRAST_EFFECT.value: "対比効果",
    MechanismCategory.SELF_EFFICACY.value: "自己効力感",
}


class CognitiveMechanism(_Base):
    """人が構造に没入し感情を動かされる理由を定義するメカニズム。"""

    id: str = Field(..., pattern=NODE_ID_PATTERN)
    category: MechanismCategory
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1, max_length=2000)
    trigger: str = Field(..., min_length=1, max_length=1000, description="発火させる構造的特徴")
    effect: str = Field(..., min_length=1, max_length=1000, description="受け手に生じる反応")
    theory: str = Field("", max_length=300, description="依拠する理論（URL禁止）")
    strength: float = Field(0.5, ge=0.0, le=1.0)
    linked_beats: list[str] = Field(default_factory=list)
    linked_nodes: list[str] = Field(default_factory=list)


class CognitiveEmpathyMechanics(_Base):
    """Layer 4: 認知・共感メカニズム群。"""

    mechanisms: list[CognitiveMechanism] = Field(..., min_length=1)
    empathy_anchor: str = Field(..., min_length=1, max_length=1000, description="共感が錨を下ろす地点")
    immersion_drivers: list[str] = Field(..., min_length=1)
    emotion_sequence: list[str] = Field(..., min_length=2, description="受け手の感情遷移列")

    @model_validator(mode="after")
    def _unique_ids(self) -> "CognitiveEmpathyMechanics":
        seen: set[str] = set()
        for mechanism in self.mechanisms:
            if mechanism.id in seen:
                raise ValueError(f"メカニズムIDが重複しています: {mechanism.id}")
            seen.add(mechanism.id)
        return self

    def by_category(self, category: MechanismCategory | str) -> list[CognitiveMechanism]:
        value = category.value if isinstance(category, MechanismCategory) else str(category)
        return [m for m in self.mechanisms if m.category.value == value]

    def dominant(self) -> CognitiveMechanism:
        return max(self.mechanisms, key=lambda m: m.strength)


# ---------------------------------------------------------------------------
# Layer 5: Domain-Agnostic Blueprint
# ---------------------------------------------------------------------------


class SlotCategory(str, Enum):
    """ブループリントの抽象変数スロット分類。"""

    AGENT = "agent"
    GOAL = "goal"
    LACK = "lack"
    CONFLICT = "conflict"
    IRONY = "irony"
    REVERSAL = "reversal"
    COST = "cost"
    SYMBOL = "symbol"
    SETTING = "setting"
    AUDIENCE = "audience"
    OFFER = "offer"
    EVIDENCE = "evidence"
    EMOTION = "emotion"
    OTHER = "other"


class BlueprintSlot(_Base):
    """抽象変数スロット（例: PROTAGONIST_OBSESSIVE_GOAL）。"""

    name: str = Field(..., pattern=SLOT_NAME_PATTERN)
    description: str = Field(..., min_length=1, max_length=1000)
    category: SlotCategory = SlotCategory.OTHER
    constraints: list[str] = Field(default_factory=list)
    examples: dict[str, str] = Field(default_factory=dict, description="領域別の充填例（manga/marketing/article）")
    required: bool = True


class DomainAgnosticBlueprint(_Base):
    """Layer 5: 任意分野へ注入可能な純粋骨格テンプレート。"""

    template: str = Field(..., min_length=20)
    slots: list[BlueprintSlot] = Field(..., min_length=1)
    invariants: list[str] = Field(..., min_length=1, description="転用時に必ず保存すべき構造的不変条件")

    @model_validator(mode="after")
    def _validate_slots(self) -> "DomainAgnosticBlueprint":
        names: set[str] = set()
        for slot in self.slots:
            if slot.name in names:
                raise ValueError(f"スロット名が重複しています: {slot.name}")
            names.add(slot.name)
        used = set(SLOT_PLACEHOLDER_RE.findall(self.template))
        undefined = sorted(used - names)
        if undefined:
            raise ValueError(f"テンプレートが未定義スロットを参照しています: {', '.join(undefined)}")
        missing = sorted(slot.name for slot in self.slots if slot.required and slot.name not in used)
        if missing:
            raise ValueError(f"必須スロットがテンプレートに現れません: {', '.join(missing)}")
        return self

    def placeholders(self) -> list[str]:
        """テンプレート内の出現順（重複除去）のスロット名。"""
        ordered: list[str] = []
        for name in SLOT_PLACEHOLDER_RE.findall(self.template):
            if name not in ordered:
                ordered.append(name)
        return ordered

    def slot(self, name: str) -> BlueprintSlot | None:
        for slot in self.slots:
            if slot.name == name:
                return slot
        return None

    def missing_bindings(self, bindings: Mapping[str, str]) -> list[str]:
        return [slot.name for slot in self.slots if slot.required and not str(bindings.get(slot.name, "")).strip()]

    def render(self, bindings: Mapping[str, str], strict: bool = True) -> str:
        """スロットを束縛して具体化する。strict=False では未束縛スロットを説明文で補う。"""
        missing = self.missing_bindings(bindings)
        if strict and missing:
            raise ValueError(f"未束縛の必須スロットがあります: {', '.join(missing)}")

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            value = str(bindings.get(name, "")).strip()
            if value:
                return value
            slot = self.slot(name)
            return f"〔{slot.description if slot else name}〕"

        return SLOT_PLACEHOLDER_RE.sub(_replace, self.template)


# ---------------------------------------------------------------------------
# Metadata / Top-level
# ---------------------------------------------------------------------------


class StructureCategory(str, Enum):
    NARRATIVE_ARCHETYPE = "narrative_archetype"
    MANGA_PANEL_RHYTHM = "manga_panel_rhythm"
    PERSUASION_MARKETING = "persuasion_marketing"
    ADVANCED_NARRATIVE = "advanced_narrative"
    MANGA_DIRECTION = "manga_direction"
    EMOTIONAL_WAVE = "emotional_wave"
    CUSTOM = "custom"


STRUCTURE_CATEGORY_LABELS_JA: dict[str, str] = {
    StructureCategory.NARRATIVE_ARCHETYPE.value: "物語原型",
    StructureCategory.MANGA_PANEL_RHYTHM.value: "漫画リズム",
    StructureCategory.PERSUASION_MARKETING.value: "説得・マーケティング",
    StructureCategory.ADVANCED_NARRATIVE.value: "高度な物語構造",
    StructureCategory.MANGA_DIRECTION.value: "漫画演出",
    StructureCategory.EMOTIONAL_WAVE.value: "感情波形プリセット",
    StructureCategory.CUSTOM.value: "カスタム",
}


class SourceKind(str, Enum):
    SEED = "seed"
    EXTRACTED = "extracted"
    MERGED = "merged"
    MUTATED = "mutated"
    MANUAL = "manual"


class StructureMetadata(_Base):
    """構造のメタデータ。lineage のみ物語論上の原型名（系譜）を保持でき、脱フレーバー検査の対象外。"""

    title: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1, max_length=4000)
    category: StructureCategory = StructureCategory.CUSTOM
    source_kind: SourceKind = SourceKind.MANUAL
    tags: list[str] = Field(default_factory=list)
    origin_domain: str = Field("generic", max_length=80)
    lineage: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    language: str = Field("ja", max_length=16)
    version: int = Field(1, ge=1)
    created_at: str = Field(default_factory=utc_now_iso)
    notes: str = Field("", max_length=4000)


class PanelRhythmProfile(_Base):
    """漫画演出リズム骨格（ページごとのコマ数・強調・めくり・見開き）。インデックスは0始まり。"""

    page_count: int = Field(..., ge=1, le=64)
    panels_per_page: list[int]
    emphasis_pages: list[int] = Field(default_factory=list)
    page_turn_reveals: list[int] = Field(default_factory=list)
    spread_pages: list[int] = Field(default_factory=list, description="見開きの右ページ側インデックス")
    silent_panels_ratio: float = Field(0.0, ge=0.0, le=1.0)
    pacing_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_profile(self) -> "PanelRhythmProfile":
        if len(self.panels_per_page) != self.page_count:
            raise ValueError("panels_per_page の長さは page_count と一致する必要があります")
        for count in self.panels_per_page:
            if count < 1 or count > 12:
                raise ValueError(f"1ページのコマ数は1〜12である必要があります: {count}")
        for name in ("emphasis_pages", "page_turn_reveals"):
            for index in getattr(self, name):
                if index < 0 or index >= self.page_count:
                    raise ValueError(f"{name} のインデックス {index} が範囲外です")
        for index in self.spread_pages:
            if index < 0 or index + 1 >= self.page_count:
                raise ValueError(f"spread_pages のインデックス {index} は見開き2ページ分の範囲に収まりません")
        return self


class TransferGuide(_Base):
    """各領域への転用ガイド。"""

    manga: str = Field("", max_length=4000)
    marketing: str = Field("", max_length=4000)
    article: str = Field("", max_length=4000)


def _validate_cross_layer(
    graph: DynamicTensionGraph,
    beats: StructuralBeatChain,
    mechanics: CognitiveEmpathyMechanics,
    blueprint: DomainAgnosticBlueprint,
) -> None:
    node_ids = set(graph.node_ids())
    beat_ids = set(beats.ids())
    slot_names = {slot.name for slot in blueprint.slots}
    for beat in beats.beats:
        unknown = [n for n in beat.involved_nodes if n not in node_ids]
        if unknown:
            raise ValueError(f"ビート {beat.id} が未定義ノードを参照しています: {', '.join(unknown)}")
    for mechanism in mechanics.mechanisms:
        unknown_beats = [b for b in mechanism.linked_beats if b not in beat_ids]
        if unknown_beats:
            raise ValueError(f"メカニズム {mechanism.id} が未定義ビートを参照しています: {', '.join(unknown_beats)}")
        unknown_nodes = [n for n in mechanism.linked_nodes if n not in node_ids]
        if unknown_nodes:
            raise ValueError(f"メカニズム {mechanism.id} が未定義ノードを参照しています: {', '.join(unknown_nodes)}")
    for node in graph.nodes:
        if node.slot is not None and node.slot not in slot_names:
            raise ValueError(f"ノード {node.id} のスロット {node.slot} がブループリントに定義されていません")


class StructureLayers(_Base):
    """LLM抽出出力用: IDと完全なメタデータを除いた5階層本体。"""

    title: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=list)
    tension_graph: DynamicTensionGraph
    wave: EmotionalInformationWave
    beats: StructuralBeatChain
    mechanics: CognitiveEmpathyMechanics
    blueprint: DomainAgnosticBlueprint
    transfer_guide: TransferGuide = Field(default_factory=TransferGuide)

    @model_validator(mode="after")
    def _cross_layer(self) -> "StructureLayers":
        _validate_cross_layer(self.tension_graph, self.beats, self.mechanics, self.blueprint)
        return self


class UniversalStructure(_Base):
    """5階層の普遍的構造（純粋骨格）。"""

    id: str = Field(..., pattern=STRUCTURE_ID_PATTERN)
    metadata: StructureMetadata
    tension_graph: DynamicTensionGraph
    wave: EmotionalInformationWave
    beats: StructuralBeatChain
    mechanics: CognitiveEmpathyMechanics
    blueprint: DomainAgnosticBlueprint
    transfer_guide: TransferGuide = Field(default_factory=TransferGuide)
    rhythm_profile: PanelRhythmProfile | None = None

    @model_validator(mode="after")
    def _cross_layer(self) -> "UniversalStructure":
        _validate_cross_layer(self.tension_graph, self.beats, self.mechanics, self.blueprint)
        return self

    @property
    def title(self) -> str:
        return self.metadata.title

    def layer_summary(self) -> dict[str, Any]:
        """各層の規模と主要指標の要約。"""
        return {
            "id": self.id,
            "title": self.metadata.title,
            "category": self.metadata.category.value,
            "source_kind": self.metadata.source_kind.value,
            "nodes": len(self.tension_graph.nodes),
            "edges": len(self.tension_graph.edges),
            "tension_energy": self.tension_graph.tension_energy(),
            "beats": len(self.beats.beats),
            "turning_points": len(self.beats.pivotal_beats()),
            "causal_depth": self.beats.causal_depth(),
            "mechanisms": len(self.mechanics.mechanisms),
            "slots": len(self.blueprint.slots),
            "wave": self.wave.summary(),
        }

    def text_fields(self) -> list[str]:
        """脱フレーバー検査対象となる全可読文字列（metadata.lineage は除外）。"""
        fields: list[str] = [self.metadata.title, self.metadata.summary, self.metadata.notes]
        fields.extend(self.metadata.tags)
        for node in self.tension_graph.nodes:
            fields.extend([node.role, node.label, node.description])
        fields.extend(edge.label for edge in self.tension_graph.edges)
        for beat in self.beats.beats:
            fields.extend([beat.label, beat.function])
        for mechanism in self.mechanics.mechanisms:
            fields.extend([mechanism.name, mechanism.description, mechanism.trigger, mechanism.effect, mechanism.theory])
        fields.append(self.mechanics.empathy_anchor)
        fields.extend(self.mechanics.immersion_drivers)
        fields.extend(self.mechanics.emotion_sequence)
        fields.append(self.blueprint.template)
        for slot in self.blueprint.slots:
            fields.append(slot.description)
            fields.extend(slot.constraints)
            fields.extend(slot.examples.values())
        fields.extend(self.blueprint.invariants)
        fields.extend([self.transfer_guide.manga, self.transfer_guide.marketing, self.transfer_guide.article])
        if self.rhythm_profile is not None:
            fields.extend(self.rhythm_profile.pacing_notes)
        return [text for text in fields if text]

    def to_layers(self) -> StructureLayers:
        return StructureLayers(
            title=self.metadata.title,
            summary=self.metadata.summary,
            tags=list(self.metadata.tags),
            tension_graph=self.tension_graph,
            wave=self.wave,
            beats=self.beats,
            mechanics=self.mechanics,
            blueprint=self.blueprint,
            transfer_guide=self.transfer_guide,
        )

    @classmethod
    def from_layers(
        cls,
        layers: StructureLayers,
        *,
        id: str | None = None,  # noqa: A002 - スキーマのフィールド名に合わせる
        metadata: StructureMetadata | None = None,
        rhythm_profile: PanelRhythmProfile | None = None,
        **metadata_overrides: Any,
    ) -> "UniversalStructure":
        """StructureLayers から UniversalStructure を構築する。"""
        if metadata is None:
            meta_payload: dict[str, Any] = {
                "title": layers.title,
                "summary": layers.summary,
                "tags": list(layers.tags),
            }
            meta_payload.update(metadata_overrides)
            metadata = StructureMetadata.model_validate(meta_payload)
        elif metadata_overrides:
            metadata = metadata.model_copy(update=metadata_overrides)
        return cls(
            id=id or new_structure_id(),
            metadata=metadata,
            tension_graph=layers.tension_graph,
            wave=layers.wave,
            beats=layers.beats,
            mechanics=layers.mechanics,
            blueprint=layers.blueprint,
            transfer_guide=layers.transfer_guide,
            rhythm_profile=rhythm_profile,
        )


def new_structure_id(prefix: str = "st") -> str:
    """構造ID（``st_xxxxxxxxxxxx``）を生成する。"""
    return new_id(prefix)
