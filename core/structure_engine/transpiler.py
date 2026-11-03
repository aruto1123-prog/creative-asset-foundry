"""異分野トランスパイラ（Domain Transpiler）。

純粋骨格（UniversalStructure）を、元の力学（Layer 1）と感情・情報波形（Layer 2）を保存したまま
① オリジナル漫画プロット ② 高CVR LP / セールスファネル ③ 体系的考察・解説記事 の具体的設計書へ変換する。

- 決定論的変換 ``transpile_deterministic``: ビートごとに1セクション。セクション境界は隣接ビートの中点、
  緊張度・解放度の目標値は波形のビート時刻での標本値、情報開示方針は非対称性と読者認知量の変化から文章化する。
- 忠実度 ``FidelityReport``: セクション代表点（ビート時刻）の緊張度目標を区分線形補間した21点波形と
  元の緊張度波形のピアソン相関、ビート被覆率、主体ノードの写像保存から
  ``score = 0.5·max(相関, 0) + 0.3·被覆率 + 0.2·位相保存`` を算出する。
- LLM 増幅 ``StructureTranspiler.transpile``: 文章（見出し・本文・感情目標・情報開示）のみをハーネスで増幅し、
  数値（目標値・区間）は決定論的設計を保持して統合する。
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.common import new_id, truncate, utc_now_iso
from core.llm_client import HarnessConfig, HarnessError, LLMClient, MultiStageReasoningHarness
from core.structure_engine.prompts import TRANSPILE_OBJECTIVES, TRANSPILE_RUBRIC, TRANSPILE_SYSTEM_PROMPTS
from core.structure_engine.visualizer import beats_to_mermaid
from core.structure_engine.schemas import (
    BEAT_TYPE_LABELS_JA,
    MECHANISM_LABELS_JA,
    WAVE_MIN_POINTS,
    NodeKind,
    StructuralBeat,
    UniversalStructure,
)

__all__ = [
    "TargetDomain",
    "DOMAIN_LABELS_JA",
    "DesignSection",
    "FidelityReport",
    "TranspiledDesign",
    "TranspiledSectionContent",
    "TranspiledContent",
    "transpile_deterministic",
    "compute_fidelity",
    "StructureTranspiler",
]

logger = logging.getLogger(__name__)


class TargetDomain(str, Enum):
    MANGA_PLOT = "manga_plot"
    MARKETING_LP = "marketing_lp"
    ARTICLE = "article"


DOMAIN_LABELS_JA: dict[str, str] = {
    TargetDomain.MANGA_PLOT.value: "オリジナル漫画プロット",
    TargetDomain.MARKETING_LP.value: "高CVRランディングページ／セールスファネル",
    TargetDomain.ARTICLE.value: "体系的考察・解説記事",
}
_EXAMPLE_KEYS = {
    TargetDomain.MANGA_PLOT: "manga",
    TargetDomain.MARKETING_LP: "marketing",
    TargetDomain.ARTICLE: "article",
}


class DesignSection(BaseModel):
    """ビート1つに対応する設計書セクション。"""

    index: int
    beat_id: str
    beat_type: str
    t_start: float
    t_end: float
    heading: str
    content: str
    tension_target: float
    release_target: float
    information_reveal: str
    emotional_goal: str
    mechanisms: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FidelityReport(BaseModel):
    """元構造に対する忠実度。"""

    wave_correlation: float
    beat_coverage: float
    topology_preserved: bool
    score: float


class TranspiledDesign(BaseModel):
    """対象領域の具体的設計書。"""

    id: str
    structure_id: str
    domain: TargetDomain
    title: str
    logline: str
    theme_variables: dict[str, str] = Field(default_factory=dict)
    slot_bindings: dict[str, str] = Field(default_factory=dict)
    rendered_blueprint: str
    sections: list[DesignSection]
    domain_payload: dict[str, Any] = Field(default_factory=dict)
    fidelity: FidelityReport
    created_at: str = Field(default_factory=utc_now_iso)


class TranspiledSectionContent(BaseModel):
    """LLM が増幅する文章部分（1セクション）。"""

    index: int = Field(..., ge=0)
    heading: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=4000)
    emotional_goal: str = Field(..., min_length=1, max_length=400)
    information_reveal: str = Field(..., min_length=1, max_length=600)


class TranspiledContent(BaseModel):
    """LLM 増幅の出力スキーマ（数値は含まない）。"""

    title: str = Field(..., min_length=1, max_length=200)
    logline: str = Field(..., min_length=1, max_length=600)
    sections: list[TranspiledSectionContent] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# 領域別の見出し・本文テンプレート
# ---------------------------------------------------------------------------

_LP_HEADINGS: dict[str, str] = {
    "hook": "ファーストビュー：逆説的なヘッドライン",
    "premise": "共感：読者の現状と信じている方法の言語化",
    "setup": "前提の共有：判断基準のすり合わせ",
    "escalation": "問題の拡大：放置した場合の損失",
    "local_optimization_limit": "問題の深掘り：既存手段が頭打ちになる理由",
    "false_victory": "よくある誤解：一時的に効く施策の落とし穴",
    "crisis": "損失の明確化：このままでは失うもの",
    "revelation": "核心の開示：見落とされていた真因",
    "structural_twist": "パラダイムシフト：前提そのものを入れ替える新しいメカニズム",
    "costly_breakthrough": "オファーと代償：導入に必要なものを誠実に提示",
    "climax": "決断の後押し：比較と保証",
    "payoff": "証拠とベネフィット：再現された成果",
    "afterglow": "CTAと未来像：最初の一歩",
    "transition": "橋渡し：次の論点への移行",
}
_ARTICLE_HEADINGS: dict[str, str] = {
    "hook": "導入：常識への問い",
    "premise": "既存の常識の提示",
    "setup": "前提の整理",
    "escalation": "常識に沿った実践とその成果",
    "local_optimization_limit": "構造的パラドックス：努力が頭打ちになる理由",
    "false_victory": "一見うまくいく反例の検討",
    "crisis": "パラドックスが生む損失",
    "revelation": "独自仮説：見落とされていた変数",
    "structural_twist": "独自仮説と証拠：前提の反転",
    "costly_breakthrough": "普遍的フレームワーク化：代償を含めた新しい方法",
    "climax": "フレームワークの適用例",
    "payoff": "検証結果と再現性",
    "afterglow": "実践アクション：今日からできること",
    "transition": "論点の転換",
}
_CVR_LEVERS: dict[str, list[str]] = {
    "hook": ["好奇心ギャップ", "パターン中断"],
    "premise": ["共感", "自己認識"],
    "local_optimization_limit": ["損失回避", "認知的不協和"],
    "structural_twist": ["新規性", "権威ある証拠"],
    "revelation": ["新規性", "納得感"],
    "costly_breakthrough": ["価値の対比", "誠実な代償提示"],
    "crisis": ["損失回避"],
    "afterglow": ["自己効力感", "行動の簡易化"],
    "payoff": ["社会的証明"],
    "climax": ["コミットメントと一貫性"],
}


def _pacing_hint(tension: float) -> str:
    if tension >= 0.75:
        return "コマ数を絞り、大ゴマ・見開き・断ち切りで瞬間を見せる"
    if tension >= 0.45:
        return "表情のクローズアップと会話のカットバックで駆け引きを進める"
    return "余白と間を取り、日常や内面の静けさを描く"


def _intensity_word(tension: float) -> str:
    if tension >= 0.8:
        return "最高潮"
    if tension >= 0.6:
        return "高い"
    if tension >= 0.4:
        return "中程度の"
    return "穏やかな"


def _information_reveal(structure: UniversalStructure, t_start: float, t_end: float) -> str:
    wave = structure.wave
    asym_delta = wave.sample("information_asymmetry", t_end) - wave.sample("information_asymmetry", t_start)
    audience_delta = wave.sample("audience_knowledge", t_end) - wave.sample("audience_knowledge", t_start)
    asym_level = wave.sample("information_asymmetry", (t_start + t_end) / 2.0)
    if audience_delta >= 0.15:
        return f"核心情報を一挙に開示する（受け手の認知量 +{audience_delta:.2f}）。"
    if asym_level >= 0.15:
        return f"受け手だけが知る情報を保ったまま進め、サスペンスを維持する（非対称性 {asym_level:+.2f}）。"
    if asym_level <= -0.15 or asym_delta <= -0.1:
        return f"人物・送り手側の秘匿を深め、謎として保留する（非対称性 {asym_level:+.2f}）。"
    if asym_delta >= 0.1:
        return f"受け手の理解を人物より一歩先へ進める（非対称性の変化 {asym_delta:+.2f}）。"
    return "情報量は維持し、既知の手がかりの意味づけを深める。"


def _emotional_goal(structure: UniversalStructure, index: int, total: int, tension: float, release: float) -> str:
    sequence = structure.mechanics.emotion_sequence
    position = 0 if total <= 1 else round(index * (len(sequence) - 1) / (total - 1))
    emotion = sequence[min(len(sequence) - 1, max(0, position))]
    if release >= 0.6:
        return f"「{emotion}」と解放（緊張 {tension:.2f} / 解放 {release:.2f}）"
    return f"「{emotion}」を{_intensity_word(tension)}強度で（緊張 {tension:.2f}）"


def _slot_bindings(structure: UniversalStructure, domain: TargetDomain, theme: dict[str, str]) -> dict[str, str]:
    theme_word = (theme.get("THEME") or theme.get("theme") or theme.get("テーマ") or "").strip()
    key = _EXAMPLE_KEYS[domain]
    bindings: dict[str, str] = {}
    for slot in structure.blueprint.slots:
        value = (theme.get(slot.name) or "").strip()
        if not value:
            example = (slot.examples.get(key) or "").strip()
            if example:
                value = f"{example}（{theme_word}）" if theme_word else example
            elif theme_word:
                value = f"{theme_word}における{slot.description}"
            else:
                value = slot.description
        bindings[slot.name] = value
    return bindings


def _nodes_bound(structure: UniversalStructure, bindings: dict[str, str]) -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = []
    for node in structure.tension_graph.nodes:
        agents.append(
            {
                "node_id": node.id,
                "kind": node.kind.value,
                "role": node.role,
                "label": node.label,
                "slot": node.slot,
                "binding": bindings.get(node.slot, node.label) if node.slot else node.label,
                "weight": node.weight,
            }
        )
    return agents


def _section_text(
    domain: TargetDomain,
    beat: StructuralBeat,
    tension: float,
    bindings: dict[str, str],
    structure: UniversalStructure,
) -> tuple[str, str]:
    type_label = BEAT_TYPE_LABELS_JA.get(beat.beat_type.value, beat.beat_type.value)
    involved = [structure.tension_graph.get_node(n) for n in beat.involved_nodes]
    actors = "・".join(
        bindings.get(node.slot, node.label) if node.slot else node.label for node in involved if node is not None
    ) or "主要な主体"
    if domain == TargetDomain.MANGA_PLOT:
        heading = f"{type_label}：{beat.label}"
        content = (
            f"{beat.function} 中心となるのは {actors}。演出方針: {_pacing_hint(tension)}。"
            f"この幕の緊張度は{_intensity_word(tension)}水準（{tension:.2f}）。"
        )
    elif domain == TargetDomain.MARKETING_LP:
        heading = _LP_HEADINGS.get(beat.beat_type.value, f"{type_label}：{beat.label}")
        levers = "・".join(_CVR_LEVERS.get(beat.beat_type.value, ["一貫性"]))
        content = (
            f"構造上の機能: {beat.function} 読者側の当事者は {actors}。"
            f"コピー方針: {_intensity_word(tension)}熱量で「{beat.label}」を伝える。CVRレバー: {levers}。"
        )
    else:
        heading = _ARTICLE_HEADINGS.get(beat.beat_type.value, f"{type_label}：{beat.label}")
        content = (
            f"論点: {beat.function} 扱う主体は {actors}。"
            f"抽象（原理）と具体（事例・数値）を交互に示し、{_intensity_word(tension)}強度で読者の前提に揺さぶりをかける。"
        )
    return heading, content


def _allocate_pages(weights: list[float], pages: int) -> list[int]:
    """最大剰余法でページ数を配分（各セクション最低1ページ、ページ不足時は重み順）。"""
    count = len(weights)
    if count == 0:
        return []
    if pages <= count:
        order = sorted(range(count), key=lambda i: -weights[i])
        allocation = [0] * count
        for i in order[:pages]:
            allocation[i] = 1
        return allocation
    remaining = pages - count
    total = sum(weights) or 1.0
    raw = [w / total * remaining for w in weights]
    allocation = [1 + int(np.floor(r)) for r in raw]
    leftover = pages - sum(allocation)
    order = sorted(range(count), key=lambda i: -(raw[i] - np.floor(raw[i])))
    for i in order[:leftover]:
        allocation[i] += 1
    return allocation


def _domain_payload(
    structure: UniversalStructure,
    domain: TargetDomain,
    sections: list[DesignSection],
    bindings: dict[str, str],
    theme: dict[str, str],
) -> dict[str, Any]:
    agents = _nodes_bound(structure, bindings)
    agent_entries = [entry for entry in agents if entry["kind"] == NodeKind.AGENT.value]
    pivotal_ids = {beat.id for beat in structure.beats.pivotal_beats()}
    if domain == TargetDomain.MANGA_PLOT:
        try:
            pages_total = max(4, min(64, int(theme.get("PAGES", "16"))))
        except ValueError:
            pages_total = 16
        weights = [(s.t_end - s.t_start) * (0.6 + 0.8 * s.tension_target) for s in sections]
        allocation = _allocate_pages(weights, pages_total)
        page_plan: list[dict[str, Any]] = []
        cursor = 1
        for section, pages in zip(sections, allocation):
            density = "low" if section.tension_target >= 0.75 else "medium" if section.tension_target >= 0.45 else "high"
            page_plan.append(
                {
                    "section_index": section.index,
                    "beat_id": section.beat_id,
                    "start_page": cursor if pages > 0 else max(1, cursor - 1),
                    "pages": pages,
                    "panel_density": density,
                }
            )
            cursor += pages
        reveals = [p["start_page"] for p in page_plan if p["beat_id"] in pivotal_ids]
        spreads = [p["start_page"] for p, s in zip(page_plan, sections) if s.tension_target >= 0.8 and p["pages"] >= 2]
        return {
            "agents": agent_entries,
            "characters": [
                {"node_id": a["node_id"], "role": a["role"], "description": a["binding"], "weight": a["weight"]}
                for a in agent_entries
            ],
            "motifs": [a for a in agents if a["kind"] != NodeKind.AGENT.value],
            "page_plan": page_plan,
            "pages_total": pages_total,
            "page_turn_reveals": reveals,
            "spread_candidates": spreads,
        }
    if domain == TargetDomain.MARKETING_LP:
        stage_bounds = [
            ("attention", 0.0, 0.2),
            ("interest", 0.2, 0.45),
            ("desire", 0.45, 0.7),
            ("conviction", 0.7, 0.88),
            ("action", 0.88, 1.01),
        ]
        funnel: list[dict[str, Any]] = []
        for stage, lo, hi in stage_bounds:
            members = [s.index for s in sections if lo <= (s.t_start + s.t_end) / 2.0 < hi]
            funnel.append({"stage": stage, "section_indices": members})
        objections = [
            {
                "section_index": s.index,
                "objection": f"「{s.heading}」の段階で読者が抱く疑念（{s.beat_type}）",
                "handling": "根拠となる事例・数値と、代償を含めた誠実な比較で応答する",
            }
            for s in sections
            if s.beat_type in ("local_optimization_limit", "crisis", "costly_breakthrough", "false_victory")
        ]
        last = sections[-1]
        return {
            "agents": agent_entries,
            "personas": [{"node_id": a["node_id"], "role": a["role"], "description": a["binding"]} for a in agent_entries],
            "funnel_stages": funnel,
            "sections": [
                {
                    "section_index": s.index,
                    "headline_idea": s.heading,
                    "copy_direction": s.content,
                    "cvr_levers": _CVR_LEVERS.get(s.beat_type, ["一貫性"]),
                }
                for s in sections
            ],
            "cta": {"after_section": last.index, "copy": f"{bindings.get('PROTAGONIST', '読者')}が今日踏み出せる最初の一歩"},
            "objections": objections,
        }
    return {
        "agents": agent_entries,
        "actors": [{"node_id": a["node_id"], "role": a["role"], "description": a["binding"]} for a in agent_entries],
        "thesis": structure.blueprint.render(bindings, strict=False).split("。")[0] + "。",
        "outline": [{"section_index": s.index, "heading": s.heading, "beat_type": s.beat_type} for s in sections],
        "figures": [
            {"kind": "mermaid", "caption": "論証の因果構造", "code": beats_to_mermaid(structure)},
            {
                "kind": "chart",
                "caption": "読者の緊張度と理解度の推移",
                "series": {
                    "t": [round(float(v), 3) for v in structure.wave.t],
                    "tension": [round(float(v), 3) for v in structure.wave.tension],
                    "audience_knowledge": [round(float(v), 3) for v in structure.wave.audience_knowledge],
                },
            },
        ],
        "evidence_plan": [
            f"{MECHANISM_LABELS_JA.get(m.category.value, m.category.value)}: {m.trigger} → {m.effect}"
            + (f"（依拠: {m.theory}）" if m.theory else "")
            for m in structure.mechanics.mechanisms
        ],
    }


def compute_fidelity(structure: UniversalStructure, design_sections: list[DesignSection], domain_payload: dict[str, Any]) -> FidelityReport:
    """設計書の元構造に対する忠実度を算出する。"""
    beat_times = {beat.id: beat.t for beat in structure.beats.beats}
    knots = sorted((beat_times.get(s.beat_id, (s.t_start + s.t_end) / 2.0), s.tension_target) for s in design_sections)
    grid = np.linspace(0.0, 1.0, WAVE_MIN_POINTS)
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    reconstructed = np.interp(grid, xs, ys)
    original = np.asarray(structure.wave.resample_arrays(WAVE_MIN_POINTS, ("tension",))["tension"], dtype=float)
    if np.std(reconstructed) < 1e-9 or np.std(original) < 1e-9:
        correlation = 1.0 if np.allclose(reconstructed, original, atol=0.05) else 0.0
    else:
        correlation = float(np.corrcoef(reconstructed, original)[0, 1])
    covered = {s.beat_id for s in design_sections}
    coverage = len(covered & set(beat_times)) / max(1, len(beat_times))
    agent_ids = {node.id for node in structure.tension_graph.nodes_of_kind(NodeKind.AGENT)}
    mapped = {entry["node_id"] for entry in domain_payload.get("agents", [])}
    topology_preserved = agent_ids <= mapped
    score = 0.5 * max(correlation, 0.0) + 0.3 * coverage + 0.2 * (1.0 if topology_preserved else 0.0)
    return FidelityReport(
        wave_correlation=round(correlation, 6),
        beat_coverage=round(coverage, 6),
        topology_preserved=topology_preserved,
        score=round(score, 6),
    )


def transpile_deterministic(
    structure: UniversalStructure,
    domain: TargetDomain | str,
    theme_variables: dict[str, str] | None = None,
) -> TranspiledDesign:
    """LLM 不要の決定論的トランスパイル。"""
    target = TargetDomain(domain)
    theme = {str(k): str(v) for k, v in (theme_variables or {}).items()}
    bindings = _slot_bindings(structure, target, theme)
    rendered = structure.blueprint.render(bindings, strict=False)
    beats = structure.beats.beats
    mechanisms_by_beat: dict[str, list[str]] = {}
    for mechanism in structure.mechanics.mechanisms:
        for beat_id in mechanism.linked_beats:
            mechanisms_by_beat.setdefault(beat_id, []).append(mechanism.name)
    sections: list[DesignSection] = []
    for index, beat in enumerate(beats):
        t_start = 0.0 if index == 0 else (beats[index - 1].t + beat.t) / 2.0
        t_end = 1.0 if index == len(beats) - 1 else (beat.t + beats[index + 1].t) / 2.0
        tension = structure.wave.sample("tension", beat.t)
        release = structure.wave.sample("release", beat.t)
        heading, content = _section_text(target, beat, tension, bindings, structure)
        notes = [f"t={beat.t:.2f}・{BEAT_TYPE_LABELS_JA.get(beat.beat_type.value, beat.beat_type.value)}"]
        if beat.is_turning_point:
            notes.append("転換点")
        sections.append(
            DesignSection(
                index=index,
                beat_id=beat.id,
                beat_type=beat.beat_type.value,
                t_start=round(t_start, 4),
                t_end=round(t_end, 4),
                heading=heading,
                content=content,
                tension_target=round(tension, 4),
                release_target=round(release, 4),
                information_reveal=_information_reveal(structure, t_start, t_end),
                emotional_goal=_emotional_goal(structure, index, len(beats), tension, release),
                mechanisms=mechanisms_by_beat.get(beat.id, []),
                notes=notes,
            )
        )
    payload = _domain_payload(structure, target, sections, bindings, theme)
    theme_word = (theme.get("THEME") or theme.get("theme") or theme.get("テーマ") or "").strip()
    if target == TargetDomain.MANGA_PLOT:
        title = f"{theme_word}——{structure.metadata.title}" if theme_word else f"漫画化：{structure.metadata.title}"
    elif target == TargetDomain.MARKETING_LP:
        title = f"{theme_word}の常識を覆すLP設計" if theme_word else f"LP設計：{structure.metadata.title}"
    else:
        title = f"{theme_word}を構造で読み解く" if theme_word else f"構造で読み解く：{structure.metadata.title}"
    logline = truncate(rendered.replace("\n", " "), 600)
    return TranspiledDesign(
        id=new_id("tp"),
        structure_id=structure.id,
        domain=target,
        title=truncate(title, 200),
        logline=logline,
        theme_variables=theme,
        slot_bindings=bindings,
        rendered_blueprint=rendered,
        sections=sections,
        domain_payload=payload,
        fidelity=compute_fidelity(structure, sections, payload),
    )


class StructureTranspiler:
    """LLM 増幅付きトランスパイラ（LLM 無し・失敗時は決定論的設計）。"""

    def __init__(self, llm_client: LLMClient | None = None, harness_config: HarnessConfig | None = None) -> None:
        self.llm_client = llm_client
        self.harness_config = harness_config

    async def transpile(
        self,
        structure: UniversalStructure,
        domain: TargetDomain | str,
        theme_variables: dict[str, str] | None = None,
    ) -> TranspiledDesign:
        baseline = transpile_deterministic(structure, domain, theme_variables)
        if self.llm_client is None:
            return baseline
        target = baseline.domain
        harness = MultiStageReasoningHarness(self.llm_client, self.harness_config or HarnessConfig())
        try:
            result = await harness.run(
                task="structure.transpile",
                objective=TRANSPILE_OBJECTIVES[target.value],
                payload={
                    "domain": target.value,
                    "theme_variables": baseline.theme_variables,
                    "slot_bindings": baseline.slot_bindings,
                    "rendered_blueprint": baseline.rendered_blueprint,
                    "sections": [
                        {
                            "index": s.index,
                            "beat_type": s.beat_type,
                            "t_start": s.t_start,
                            "t_end": s.t_end,
                            "tension_target": s.tension_target,
                            "release_target": s.release_target,
                            "information_reveal": s.information_reveal,
                            "emotional_goal": s.emotional_goal,
                            "mechanisms": s.mechanisms,
                            "draft_heading": s.heading,
                            "draft_content": s.content,
                        }
                        for s in baseline.sections
                    ],
                },
                output_model=TranspiledContent,
                system_prompt=TRANSPILE_SYSTEM_PROMPTS[target.value],
                rubric=TRANSPILE_RUBRIC,
                context={"baseline": baseline.model_dump(mode="json"), "domain": target.value},
            )
        except HarnessError as exc:
            logger.warning("トランスパイルのLLM増幅に失敗したため決定論的設計を返します: %s", exc)
            return baseline
        content = result.output
        by_index = {section.index: section for section in content.sections}
        merged_sections: list[DesignSection] = []
        for section in baseline.sections:
            enriched = by_index.get(section.index)
            if enriched is None:
                merged_sections.append(section)
                continue
            merged_sections.append(
                section.model_copy(
                    update={
                        "heading": enriched.heading.strip() or section.heading,
                        "content": enriched.content.strip() or section.content,
                        "emotional_goal": enriched.emotional_goal.strip() or section.emotional_goal,
                        "information_reveal": enriched.information_reveal.strip() or section.information_reveal,
                        "notes": [*section.notes, "LLM増幅"],
                    }
                )
            )
        payload = dict(baseline.domain_payload)
        payload["harness_run_id"] = result.run_id
        if result.critique is not None:
            payload["critique_score"] = result.critique.overall_score
        return baseline.model_copy(
            update={
                "title": content.title.strip() or baseline.title,
                "logline": content.logline.strip() or baseline.logline,
                "sections": merged_sections,
                "domain_payload": payload,
                "fidelity": compute_fidelity(structure, merged_sections, payload),
            }
        )
