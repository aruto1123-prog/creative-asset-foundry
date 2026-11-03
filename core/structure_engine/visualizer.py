"""構造の可視化（Mermaid 記法・SVG 波形・Markdown レポート）。

- ``graph_to_mermaid``: 力学グラフを Mermaid ``flowchart LR`` へ。ノード形状は種別で区別
  （主体 ``["…"]`` / 欲求 ``(["…"])`` / 欠落 ``{{"…"}}`` / 象徴 ``[("…")]``）、
  エッジは種別の日本語名と強度ラベル付き（裏切り ``-.->``、対立 ``==>``、その他 ``-->``）。
- ``beats_to_mermaid``: ビートの因果 DAG を ``flowchart TD`` へ（転換点を強調）。
- ``wave_to_svg``: 感情・情報波形の SVG（グリッド・軸・凡例・チャネル別 polyline・ビートマーカー・比較重ね描き）。
  情報非対称性 [-1, 1] は表示上 [0, 1] へ線形写像する。
- ``structure_report_markdown``: 5階層の要約 Markdown。
"""

from __future__ import annotations

from collections.abc import Sequence
from xml.sax.saxutils import escape, quoteattr

from core.structure_engine.schemas import (
    BEAT_TYPE_LABELS_JA,
    DEFAULT_CHANNELS,
    EDGE_KIND_LABELS_JA,
    MECHANISM_LABELS_JA,
    NODE_KIND_LABELS_JA,
    DynamicTensionGraph,
    EmotionalInformationWave,
    StructuralBeatChain,
    UniversalStructure,
)

__all__ = [
    "CHANNEL_COLORS",
    "CHANNEL_LABELS_JA",
    "graph_to_mermaid",
    "beats_to_mermaid",
    "wave_to_svg",
    "structure_report_markdown",
]

CHANNEL_COLORS: dict[str, str] = {
    "tension": "#f87171",
    "release": "#4ade80",
    "information_asymmetry": "#60a5fa",
    "cognitive_load": "#facc15",
    "audience_knowledge": "#a78bfa",
    "character_knowledge": "#f472b6",
}

CHANNEL_LABELS_JA: dict[str, str] = {
    "tension": "緊張度",
    "release": "カタルシス解放度",
    "information_asymmetry": "情報非対称性",
    "cognitive_load": "認知負荷",
    "audience_knowledge": "読者認知量",
    "character_knowledge": "人物認知量",
}

_NODE_SHAPES: dict[str, tuple[str, str]] = {
    "agent": ('["', '"]'),
    "desire": ('(["', '"])'),
    "lack": ('{{"', '"}}'),
    "symbol": ('[("', '")]'),
}

_EDGE_ARROWS: dict[str, str] = {
    "betrayal": "-.->",
    "opposition": "==>",
}


def _mermaid_text(text: str) -> str:
    """Mermaid ラベル用エスケープ（二重引用符・改行・角括弧の衝突回避）。"""
    cleaned = " ".join(str(text).split())
    return cleaned.replace('"', "#quot;").replace("|", "／")


def _mermaid_node_id(node_id: str) -> str:
    return f"n_{node_id}"


def graph_to_mermaid(structure_or_graph: UniversalStructure | DynamicTensionGraph) -> str:
    """力学グラフを Mermaid flowchart 記法へ変換する。"""
    graph = structure_or_graph.tension_graph if isinstance(structure_or_graph, UniversalStructure) else structure_or_graph
    lines = [
        "flowchart LR",
        "  classDef agent fill:#1f2a44,stroke:#7c9cff,color:#e6e9f3",
        "  classDef desire fill:#163a2e,stroke:#4ade80,color:#e6e9f3",
        "  classDef lack fill:#3a1f24,stroke:#f87171,color:#e6e9f3",
        "  classDef symbol fill:#3a321a,stroke:#facc15,color:#e6e9f3",
    ]
    for node in graph.nodes:
        left, right = _NODE_SHAPES[node.kind.value]
        label = f"{NODE_KIND_LABELS_JA[node.kind.value]}: {node.label}"
        lines.append(f"  {_mermaid_node_id(node.id)}{left}{_mermaid_text(label)}{right}:::{node.kind.value}")
    for edge in graph.edges:
        arrow = _EDGE_ARROWS.get(edge.kind.value, "-->")
        label = f"{EDGE_KIND_LABELS_JA[edge.kind.value]} {edge.intensity:.2f}"
        if edge.gradient:
            label += f" 勾配{edge.gradient:+.2f}"
        if edge.activation_t is not None:
            label += f" t={edge.activation_t:.2f}"
        lines.append(
            f'  {_mermaid_node_id(edge.source)} {arrow}|"{_mermaid_text(label)}"| {_mermaid_node_id(edge.target)}'
        )
    return "\n".join(lines)


def beats_to_mermaid(structure_or_beats: UniversalStructure | StructuralBeatChain) -> str:
    """ビート因果チェーンを Mermaid flowchart TD へ変換する（転換点は turning クラスで強調）。"""
    chain = structure_or_beats.beats if isinstance(structure_or_beats, UniversalStructure) else structure_or_beats
    lines = [
        "flowchart TD",
        "  classDef beat fill:#1b2030,stroke:#6b7493,color:#e6e9f3",
        "  classDef turning fill:#3b2a10,stroke:#f59e0b,stroke-width:3px,color:#ffffff",
    ]
    pivotal = {beat.id for beat in chain.pivotal_beats()}
    for beat in chain.beats:
        type_label = BEAT_TYPE_LABELS_JA.get(beat.beat_type.value, beat.beat_type.value)
        label = f"t={beat.t:.2f} {type_label}<br/>{_mermaid_text(beat.label)}"
        css = "turning" if beat.id in pivotal else "beat"
        lines.append(f'  b_{beat.id}["{label}"]:::{css}')
    for cause, effect in chain.causal_edges():
        lines.append(f"  b_{cause} --> b_{effect}")
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def wave_to_svg(
    wave: EmotionalInformationWave,
    *,
    width: int = 800,
    height: int = 320,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    beats: StructuralBeatChain | None = None,
    title: str | None = None,
    compare: Sequence[tuple[str, EmotionalInformationWave]] | None = None,
) -> str:
    """感情・情報波形の SVG 文字列を生成する（XML として妥当、テキストはエスケープ済み）。"""
    width = max(240, int(width))
    height = max(160, int(height))
    margin_left, margin_right, margin_top, margin_bottom = 46.0, 16.0, 34.0 if title else 18.0, 52.0
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_of(t: float) -> float:
        return margin_left + float(t) * plot_w

    def y_of(value: float, channel: str) -> float:
        shown = (float(value) + 1.0) / 2.0 if channel == "information_asymmetry" else float(value)
        shown = min(1.0, max(0.0, shown))
        return margin_top + (1.0 - shown) * plot_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" font-family="Inter, Noto Sans JP, sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0d0f16" rx="8"/>',
    ]
    if title:
        parts.append(
            f'<text x="{_fmt(margin_left)}" y="22" fill="#e6e9f3" font-size="14" font-weight="600">{escape(title)}</text>'
        )
    parts.append('<g class="grid" stroke="#2f3549" stroke-width="1">')
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + (1.0 - level) * plot_h
        parts.append(f'<line x1="{_fmt(margin_left)}" y1="{_fmt(y)}" x2="{_fmt(margin_left + plot_w)}" y2="{_fmt(y)}"/>')
    for step in range(11):
        x = x_of(step / 10)
        parts.append(f'<line x1="{_fmt(x)}" y1="{_fmt(margin_top)}" x2="{_fmt(x)}" y2="{_fmt(margin_top + plot_h)}" stroke-opacity="0.5"/>')
    parts.append("</g>")
    parts.append('<g class="axis" fill="#99a1bd" font-size="10">')
    for level in (0.0, 0.5, 1.0):
        y = margin_top + (1.0 - level) * plot_h
        parts.append(f'<text x="{_fmt(margin_left - 6)}" y="{_fmt(y + 3)}" text-anchor="end">{level:.1f}</text>')
    for step in range(0, 11, 2):
        x = x_of(step / 10)
        parts.append(f'<text x="{_fmt(x)}" y="{_fmt(margin_top + plot_h + 14)}" text-anchor="middle">{step / 10:.1f}</text>')
    parts.append(
        f'<text x="{_fmt(margin_left + plot_w / 2)}" y="{_fmt(margin_top + plot_h + 28)}" text-anchor="middle">正規化時間 t</text>'
    )
    parts.append("</g>")

    if compare:
        for compare_index, (label, other) in enumerate(compare):
            for name in channels:
                values = other.channel(name)
                points = " ".join(f"{_fmt(x_of(t))},{_fmt(y_of(v, name))}" for t, v in zip(other.t, values))
                parts.append(
                    f'<polyline class="compare" data-compare={quoteattr(str(label))} data-channel="{escape(name)}" '
                    f'points="{points}" fill="none" stroke="{CHANNEL_COLORS.get(name, "#cccccc")}" '
                    f'stroke-width="1.5" stroke-dasharray="{4 + 2 * compare_index},3" stroke-opacity="0.55"/>'
                )

    for name in channels:
        values = wave.channel(name)
        points = " ".join(f"{_fmt(x_of(t))},{_fmt(y_of(v, name))}" for t, v in zip(wave.t, values))
        parts.append(
            f'<polyline class="channel" data-channel="{escape(name)}" points="{points}" fill="none" '
            f'stroke="{CHANNEL_COLORS.get(name, "#cccccc")}" stroke-width="2.2" stroke-linejoin="round"/>'
        )

    if beats is not None:
        pivotal = {beat.id for beat in beats.pivotal_beats()}
        parts.append('<g class="beats">')
        for beat in beats.beats:
            x = x_of(beat.t)
            y = y_of(wave.sample("tension", beat.t), "tension")
            is_pivot = beat.id in pivotal
            stroke = "#f59e0b" if is_pivot else "#6b7493"
            type_label = BEAT_TYPE_LABELS_JA.get(beat.beat_type.value, beat.beat_type.value)
            parts.append(
                f'<g class="beat-marker" data-beat={quoteattr(beat.id)}>'
                f'<line x1="{_fmt(x)}" y1="{_fmt(margin_top)}" x2="{_fmt(x)}" y2="{_fmt(margin_top + plot_h)}" '
                f'stroke="{stroke}" stroke-width="{2 if is_pivot else 1}" stroke-dasharray="3,3"/>'
                f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{4 if is_pivot else 3}" fill="{stroke}"/>'
                f"<title>{escape(f't={beat.t:.2f} {type_label}: {beat.label}')}</title>"
                "</g>"
            )
        parts.append("</g>")

    legend_entries = [(CHANNEL_LABELS_JA.get(name, name), CHANNEL_COLORS.get(name, "#cccccc"), False) for name in channels]
    if compare:
        legend_entries.extend((f"比較: {label}", "#99a1bd", True) for label, _ in compare)
    legend_y = height - 12.0
    cursor = margin_left
    parts.append('<g class="legend" font-size="10" fill="#c5cbe0">')
    for text, color, dashed in legend_entries:
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        parts.append(
            f'<line x1="{_fmt(cursor)}" y1="{_fmt(legend_y - 3)}" x2="{_fmt(cursor + 16)}" y2="{_fmt(legend_y - 3)}" '
            f'stroke="{color}" stroke-width="2"{dash}/>'
        )
        parts.append(f'<text x="{_fmt(cursor + 20)}" y="{_fmt(legend_y)}">{escape(text)}</text>')
        cursor += 28 + 11 * len(text)
    parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def structure_report_markdown(structure: UniversalStructure) -> str:
    """5階層の要約を Markdown で返す（Mermaid 図・数表を含む）。"""
    summary = structure.layer_summary()
    wave_stats = summary["wave"]
    lines: list[str] = [
        f"# {structure.metadata.title}",
        "",
        structure.metadata.summary,
        "",
        f"- ID: `{structure.id}` / カテゴリ: {structure.metadata.category.value} / 由来: {structure.metadata.source_kind.value}",
        f"- タグ: {', '.join(structure.metadata.tags) if structure.metadata.tags else '（なし）'}",
        "",
        "## Layer 1: 力学グラフ",
        "",
        f"ノード {summary['nodes']} / エッジ {summary['edges']} / 緊張エネルギー {summary['tension_energy']:.3f}",
        "",
        "```mermaid",
        graph_to_mermaid(structure),
        "```",
        "",
        "## Layer 2: 感情・情報波形",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        f"| 緊張度ピーク | t={wave_stats['peak_t']:.2f}（{wave_stats['peak_tension']:.2f}） |",
        f"| 平均緊張度 | {wave_stats['mean_tension']:.3f} |",
        f"| サスペンス係数 | {wave_stats['suspense']:.3f} |",
        f"| サプライズ係数 | {wave_stats['surprise']:.3f} |",
        f"| カタルシス指数 | {wave_stats['catharsis']:.3f} |",
        "",
        "## Layer 3: ビートと転換点",
        "",
        "| t | 種別 | ラベル | 機能 |",
        "|---|---|---|---|",
    ]
    for beat in structure.beats.beats:
        marker = "★" if beat.is_turning_point else ""
        type_label = BEAT_TYPE_LABELS_JA.get(beat.beat_type.value, beat.beat_type.value)
        lines.append(f"| {beat.t:.2f} | {marker}{type_label} | {beat.label} | {beat.function} |")
    lines.extend(["", "```mermaid", beats_to_mermaid(structure), "```", "", "## Layer 4: 認知・共感メカニズム", ""])
    for mechanism in structure.mechanics.mechanisms:
        category = MECHANISM_LABELS_JA.get(mechanism.category.value, mechanism.category.value)
        lines.append(f"- **{mechanism.name}**（{category}・強度 {mechanism.strength:.2f}）: {mechanism.description}")
    lines.extend(
        [
            f"- 共感の錨: {structure.mechanics.empathy_anchor}",
            f"- 感情遷移: {' → '.join(structure.mechanics.emotion_sequence)}",
            "",
            "## Layer 5: 分野非依存ブループリント",
            "",
            "```text",
            structure.blueprint.template,
            "```",
            "",
            "| スロット | 分類 | 説明 |",
            "|---|---|---|",
        ]
    )
    for slot in structure.blueprint.slots:
        lines.append(f"| `{slot.name}` | {slot.category.value} | {slot.description} |")
    lines.extend(["", "### 不変条件", ""])
    lines.extend(f"- {inv}" for inv in structure.blueprint.invariants)
    guide = structure.transfer_guide
    lines.extend(["", "## 転用ガイド", ""])
    for label, text in (("漫画", guide.manga), ("マーケティング", guide.marketing), ("記事", guide.article)):
        if text:
            lines.append(f"- **{label}**: {text}")
    return "\n".join(lines) + "\n"
