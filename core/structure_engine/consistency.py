"""構造整合性監査（Structure Consistency Audit）。

スキーマ検証（型・範囲・参照整合）を通過した構造に対し、物語論的・力学的な整合性を検査する。

| コード | 重大度 | 内容 |
|---|---|---|
| ``no_agent`` | error | 主体（agent）ノードが存在しない |
| ``disconnected_graph`` | warning | 力学グラフが非連結（無向） |
| ``betrayal_without_dependency`` | warning | 裏切りエッジの前提となる依存関係が無い |
| ``orphan_beat`` | warning | 先頭以外のビートに因果（causes）が無い |
| ``peak_without_pivot`` | warning | 緊張度の大域ピーク ±0.1 に転換点ビートが無い |
| ``turning_point_flat`` | warning | 転換点ビートの周辺で緊張度・情報量がほぼ変化しない |
| ``twist_info_sign`` | warning | 反転/開示ビートの information_shift の符号が波形の非対称性変化と逆 |
| ``missing_resolution`` | warning | 終盤（t≥0.8）の解決系ビート（突破・頂点・回収・余韻）が無い |
| ``slot_unused`` | warning | 任意スロットがテンプレートに現れない |
| ``flat_wave`` | warning | 緊張度波形がほぼ平坦（標準偏差 < 0.02） |
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from core.structure_engine.schemas import (
    BeatType,
    EdgeKind,
    NodeKind,
    SLOT_PLACEHOLDER_RE,
    UniversalStructure,
)

__all__ = [
    "ConsistencyIssue",
    "ISSUE_CODES",
    "RESOLUTION_BEAT_TYPES",
    "audit_structure",
    "has_errors",
    "issue_codes",
    "graph_components",
]

ISSUE_CODES: tuple[str, ...] = (
    "no_agent",
    "disconnected_graph",
    "betrayal_without_dependency",
    "orphan_beat",
    "peak_without_pivot",
    "turning_point_flat",
    "twist_info_sign",
    "missing_resolution",
    "slot_unused",
    "flat_wave",
)

RESOLUTION_BEAT_TYPES: frozenset[BeatType] = frozenset(
    {BeatType.COSTLY_BREAKTHROUGH, BeatType.CLIMAX, BeatType.PAYOFF, BeatType.AFTERGLOW}
)
_TWIST_TYPES: frozenset[BeatType] = frozenset({BeatType.STRUCTURAL_TWIST, BeatType.REVELATION})


class ConsistencyIssue(BaseModel):
    """監査で検出された1件の問題。"""

    code: str
    severity: Literal["error", "warning"]
    message: str
    location: str = Field("", description="問題箇所（ノード/ビート/スロットID等）")


def graph_components(structure: UniversalStructure) -> list[list[str]]:
    """力学グラフの無向連結成分（ノードID列のリスト、ノード出現順）。"""
    graph = structure.tension_graph
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in graph.node_ids()}
    for edge in graph.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node_id in graph.node_ids():
        if node_id in seen:
            continue
        stack = [node_id]
        component: list[str] = []
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        order = {nid: i for i, nid in enumerate(graph.node_ids())}
        components.append(sorted(component, key=lambda nid: order[nid]))
    return components


def _local_change(structure: UniversalStructure, channel: str, t: float, half_window: float) -> float:
    wave = structure.wave
    before = wave.sample(channel, max(0.0, t - half_window))
    center = wave.sample(channel, t)
    after = wave.sample(channel, min(1.0, t + half_window))
    return max(abs(center - before), abs(after - center), abs(after - before))


def audit_structure(structure: UniversalStructure, *, peak_tolerance: float = 0.1) -> list[ConsistencyIssue]:
    """構造の物語論的・力学的整合性を監査し、問題一覧を返す（問題が無ければ空リスト）。"""
    issues: list[ConsistencyIssue] = []
    graph = structure.tension_graph
    beats = structure.beats.beats

    # 1. 主体ノードの存在
    if not graph.nodes_of_kind(NodeKind.AGENT):
        issues.append(
            ConsistencyIssue(
                code="no_agent",
                severity="error",
                message="主体（agent）ノードが存在しません。力学の担い手を最低1つ定義してください。",
                location="tension_graph.nodes",
            )
        )

    # 2. グラフの連結性
    components = graph_components(structure)
    if len(components) > 1:
        detail = " / ".join("{" + ", ".join(c) + "}" for c in components)
        issues.append(
            ConsistencyIssue(
                code="disconnected_graph",
                severity="warning",
                message=f"力学グラフが {len(components)} 個の成分に分断されています: {detail}",
                location="tension_graph",
            )
        )

    # 3. 裏切りの前提となる依存
    for edge in graph.edges_of_kind(EdgeKind.BETRAYAL):
        has_dependency = any(
            other.kind == EdgeKind.DEPENDENCY and {other.source, other.target} == {edge.source, edge.target}
            for other in graph.edges
        )
        if not has_dependency:
            issues.append(
                ConsistencyIssue(
                    code="betrayal_without_dependency",
                    severity="warning",
                    message=f"裏切り {edge.source} -> {edge.target} の前提となる依存関係（信頼）が描かれていません。",
                    location=f"{edge.source}->{edge.target}",
                )
            )

    # 4. 因果の孤立
    for index, beat in enumerate(beats):
        if index > 0 and not beat.causes:
            issues.append(
                ConsistencyIssue(
                    code="orphan_beat",
                    severity="warning",
                    message=f"ビート {beat.id} に先行ビートからの因果がありません。",
                    location=beat.id,
                )
            )

    # 5. 緊張度ピークと転換点
    tension = structure.wave.channel("tension")
    peak_t = float(structure.wave.t[int(np.argmax(tension))])
    pivotal = structure.beats.pivotal_beats()
    if not any(abs(beat.t - peak_t) <= peak_tolerance + 1e-9 for beat in pivotal):
        issues.append(
            ConsistencyIssue(
                code="peak_without_pivot",
                severity="warning",
                message=f"緊張度の大域ピーク t≈{peak_t:.2f} の ±{peak_tolerance:.2f} に転換点ビートがありません。",
                location=f"t={peak_t:.2f}",
            )
        )

    # 6. 平坦な転換点
    for beat in structure.beats.turning_points():
        window = max(0.05, beat.span)
        tension_change = _local_change(structure, "tension", beat.t, window)
        info_change = _local_change(structure, "information_asymmetry", beat.t, window)
        if tension_change < 0.03 and info_change < 0.03 and abs(beat.information_shift) < 0.05:
            issues.append(
                ConsistencyIssue(
                    code="turning_point_flat",
                    severity="warning",
                    message=f"転換点 {beat.id} の周辺で緊張度・情報量がほとんど変化しません（Δ緊張 {tension_change:.3f}）。",
                    location=beat.id,
                )
            )

    # 7. 反転/開示の情報変化の符号
    for beat in beats:
        if beat.beat_type not in _TWIST_TYPES:
            continue
        window = max(0.06, beat.span)
        before = structure.wave.sample("information_asymmetry", max(0.0, beat.t - window))
        after = structure.wave.sample("information_asymmetry", min(1.0, beat.t + window))
        observed = after - before
        if abs(observed) > 0.05 and abs(beat.information_shift) > 0.05 and np.sign(observed) != np.sign(beat.information_shift):
            issues.append(
                ConsistencyIssue(
                    code="twist_info_sign",
                    severity="warning",
                    message=(
                        f"ビート {beat.id} の information_shift={beat.information_shift:+.2f} が"
                        f"波形の非対称性変化 {observed:+.2f} と逆向きです。"
                    ),
                    location=beat.id,
                )
            )

    # 8. 解決の有無
    if not any(beat.beat_type in RESOLUTION_BEAT_TYPES and beat.t >= 0.8 - 1e-9 for beat in beats):
        issues.append(
            ConsistencyIssue(
                code="missing_resolution",
                severity="warning",
                message="終盤（t≥0.8）に突破・頂点・回収・余韻のいずれのビートもありません。",
                location="beats",
            )
        )

    # 9. 未使用スロット
    used = set(SLOT_PLACEHOLDER_RE.findall(structure.blueprint.template))
    for slot in structure.blueprint.slots:
        if not slot.required and slot.name not in used:
            issues.append(
                ConsistencyIssue(
                    code="slot_unused",
                    severity="warning",
                    message=f"任意スロット {slot.name} がテンプレートで使われていません。",
                    location=slot.name,
                )
            )

    # 10. 平坦な波形
    if float(np.std(tension)) < 0.02:
        issues.append(
            ConsistencyIssue(
                code="flat_wave",
                severity="warning",
                message="緊張度波形がほぼ平坦です（標準偏差 < 0.02）。",
                location="wave.tension",
            )
        )
    return issues


def has_errors(issues: Iterable[ConsistencyIssue]) -> bool:
    """error 重大度の問題を含むか。"""
    return any(issue.severity == "error" for issue in issues)


def issue_codes(issues: Iterable[ConsistencyIssue]) -> set[str]:
    """問題コードの集合。"""
    return {issue.code for issue in issues}
