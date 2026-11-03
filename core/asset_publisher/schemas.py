"""Asset Publisher スキーマ。

研究ログ（ResearchLog）→ 3軸コンテンツ企画（ContentMatrix）→ 5段階執筆
（ArticleBlueprint → ArticleDraft → サニタイズ → 批評 → 推敲）→ 公開記事（PublishedArticle）
までの全データ契約を定義する。LLM出力を頑健に受理するため未知キーは無視する。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import new_id, utc_now_iso

__all__ = [
    "ContentAxis",
    "AXIS_ORDER",
    "AXIS_LABELS_JA",
    "AXIS_READERS_JA",
    "Finding",
    "ResearchLog",
    "OutlineSection",
    "ContentPlan",
    "ContentMatrix",
    "BlueprintRole",
    "BLUEPRINT_ROLE_ORDER",
    "BLUEPRINT_ROLE_LABELS_JA",
    "FigureSpec",
    "BlueprintSection",
    "ArticleBlueprint",
    "DraftSection",
    "ArticleDraft",
    "Replacement",
    "RemovedURL",
    "SanitizationReport",
    "SectionScore",
    "CritiqueScore",
    "PipelineStageRecord",
    "PublishedArticle",
    "ArticleSummary",
    "count_prose_chars",
    "split_markdown_segments",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ContentAxis(str, Enum):
    """コンテンツ展開の3軸。"""

    CREATOR_THEORY = "creator_theory"
    AI_ENGINEERING = "ai_engineering"
    BUSINESS_MARKETING = "business_marketing"


AXIS_ORDER: tuple[ContentAxis, ...] = (
    ContentAxis.CREATOR_THEORY,
    ContentAxis.AI_ENGINEERING,
    ContentAxis.BUSINESS_MARKETING,
)

AXIS_LABELS_JA: dict[str, str] = {
    ContentAxis.CREATOR_THEORY.value: "物語構造・漫画演出理論",
    ContentAxis.AI_ENGINEERING.value: "モデル調合・特徴量制御・データセット設計論",
    ContentAxis.BUSINESS_MARKETING.value: "企画・LP・マーケティング構造論",
}

AXIS_READERS_JA: dict[str, str] = {
    ContentAxis.CREATOR_THEORY.value: "オリジナルの漫画や物語を設計するクリエイター",
    ContentAxis.AI_ENGINEERING.value: "画像生成モデルの調合やLoRA学習に取り組むAI開発者",
    ContentAxis.BUSINESS_MARKETING.value: "企画・LP・セールスファネルを設計するビジネス担当者",
}


class Finding(_Base):
    """研究ログ内の1つの発見（主張・根拠・数値指標）。"""

    claim: str = Field(..., min_length=1, max_length=2000)
    evidence: str = Field("", max_length=4000)
    metrics: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(0.7, ge=0.0, le=1.0)


class ResearchLog(_Base):
    """構造研究・モデル実験・データセット分析・漫画制作から得た知見の記録。"""

    id: str = Field(default_factory=lambda: new_id("log"))
    kind: Literal["structure_discovery", "model_experiment", "dataset_analysis", "manga_production", "mixed"] = "mixed"
    title: str = Field(..., min_length=1, max_length=300)
    summary: str = Field(..., min_length=1, max_length=6000)
    findings: list[Finding] = Field(..., min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)

    def strongest_finding(self) -> Finding:
        """確信度が最大の発見（同点は先頭）。"""
        return max(self.findings, key=lambda f: f.confidence)

    def all_metrics(self) -> dict[str, float]:
        """全発見の数値指標を統合した辞書（キー衝突時は先勝ち）。"""
        merged: dict[str, float] = {}
        for finding in self.findings:
            for key, value in finding.metrics.items():
                merged.setdefault(key, float(value))
        return merged


class OutlineSection(_Base):
    heading: str = Field(..., min_length=1, max_length=300)
    purpose: str = Field(..., min_length=1, max_length=1000)
    key_points: list[str] = Field(default_factory=list)
    figure_idea: str = Field("", max_length=1000)


class ContentPlan(_Base):
    """1軸分のコンテンツ企画・構成案。"""

    axis: ContentAxis
    title: str = Field(..., min_length=1, max_length=300)
    subtitle: str = Field("", max_length=500)
    target_reader: str = Field(..., min_length=1, max_length=500)
    reader_pain: str = Field(..., min_length=1, max_length=1000)
    core_insight: str = Field(..., min_length=1, max_length=2000)
    angle: str = Field(..., min_length=1, max_length=500)
    outline: list[OutlineSection] = Field(..., min_length=3)
    keywords: list[str] = Field(default_factory=list)
    hook_line: str = Field(..., min_length=1, max_length=1000)
    call_to_action: str = Field(..., min_length=1, max_length=1000)
    estimated_chars: int = Field(6000, ge=0, le=200000)


class ContentMatrix(_Base):
    """1つの研究ログから3軸へ同時展開した企画マトリクス。"""

    id: str = Field(default_factory=lambda: new_id("matrix"))
    source_id: str = Field(..., min_length=1)
    source_title: str = Field(..., min_length=1, max_length=300)
    shared_core: str = Field(..., min_length=1, max_length=4000)
    plans: list[ContentPlan] = Field(..., min_length=3, max_length=3)
    cross_links: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _one_plan_per_axis(self) -> "ContentMatrix":
        axes = [plan.axis for plan in self.plans]
        if sorted(a.value for a in axes) != sorted(a.value for a in AXIS_ORDER):
            raise ValueError("plans は3軸（creator_theory / ai_engineering / business_marketing）を各1件ずつ含む必要があります")
        return self

    def plan_for(self, axis: ContentAxis | str) -> ContentPlan:
        value = axis.value if isinstance(axis, ContentAxis) else str(axis)
        for plan in self.plans:
            if plan.axis.value == value:
                return plan
        raise KeyError(value)


class BlueprintRole(str, Enum):
    """記事5部構成の役割。"""

    CONVENTIONAL_WISDOM = "conventional_wisdom"
    STRUCTURAL_PARADOX = "structural_paradox"
    HYPOTHESIS_EVIDENCE = "hypothesis_evidence"
    UNIVERSAL_FRAMEWORK = "universal_framework"
    PRACTICAL_ACTION = "practical_action"


BLUEPRINT_ROLE_ORDER: tuple[BlueprintRole, ...] = tuple(BlueprintRole)

BLUEPRINT_ROLE_LABELS_JA: dict[str, str] = {
    BlueprintRole.CONVENTIONAL_WISDOM.value: "既存の常識の提示",
    BlueprintRole.STRUCTURAL_PARADOX.value: "構造的パラドックスの指摘",
    BlueprintRole.HYPOTHESIS_EVIDENCE.value: "独自仮説と実験データ",
    BlueprintRole.UNIVERSAL_FRAMEWORK.value: "普遍的フレームワーク化",
    BlueprintRole.PRACTICAL_ACTION.value: "実践アクション",
}


class FigureSpec(_Base):
    """章に挿入する図解の仕様。"""

    kind: Literal["mermaid", "table", "none"] = "none"
    caption: str = Field("", max_length=300)
    mermaid_code: str = Field("", max_length=8000)

    @model_validator(mode="after")
    def _mermaid_requires_code(self) -> "FigureSpec":
        if self.kind == "mermaid" and not self.mermaid_code.strip():
            raise ValueError("kind=mermaid の図解には mermaid_code が必要です")
        return self


class BlueprintSection(_Base):
    role: BlueprintRole
    heading: str = Field(..., min_length=1, max_length=300)
    intent: str = Field(..., min_length=1, max_length=2000)
    key_claims: list[str] = Field(..., min_length=1)
    evidence: list[str] = Field(default_factory=list)
    figure: FigureSpec = Field(default_factory=FigureSpec)


def _check_role_order(sections: list[Any]) -> None:
    roles = [section.role for section in sections]
    if roles != list(BLUEPRINT_ROLE_ORDER):
        expected = ", ".join(r.value for r in BLUEPRINT_ROLE_ORDER)
        raise ValueError(f"sections は5役割を正準順（{expected}）で含む必要があります")


class ArticleBlueprint(_Base):
    """記事の設計図（5部構成）。"""

    title: str = Field(..., min_length=1, max_length=300)
    thesis: str = Field(..., min_length=1, max_length=2000)
    target_reader: str = Field(..., min_length=1, max_length=500)
    axis: ContentAxis
    sections: list[BlueprintSection] = Field(..., min_length=5, max_length=5)

    @model_validator(mode="after")
    def _canonical_order(self) -> "ArticleBlueprint":
        _check_role_order(self.sections)
        return self

    def section(self, role: BlueprintRole | str) -> BlueprintSection:
        value = role.value if isinstance(role, BlueprintRole) else str(role)
        for section in self.sections:
            if section.role.value == value:
                return section
        raise KeyError(value)


_FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[ \t]*([^\s`]*)")


def split_markdown_segments(text: str) -> list[tuple[str, str]]:
    """Markdown を ``("prose" | "code" | "mermaid", 断片)`` の列へ行単位で分割する。

    フェンス（``` / ~~~、3文字以上）の開始行と終了行はコード断片側に含める。
    閉じられていないフェンスは末尾までをコード断片とみなす。断片を連結すると元の文字列に戻る。
    """
    segments: list[tuple[str, str]] = []
    buffer: list[str] = []
    fence_char = ""
    fence_len = 0
    kind = "prose"
    for line in (text or "").splitlines(keepends=True):
        if not fence_char:
            match = _FENCE_OPEN_RE.match(line)
            if match:
                if buffer:
                    segments.append(("prose", "".join(buffer)))
                    buffer = []
                fence = match.group(1)
                fence_char, fence_len = fence[0], len(fence)
                kind = "mermaid" if match.group(2).lower() == "mermaid" else "code"
            buffer.append(line)
            continue
        buffer.append(line)
        stripped = line.strip()
        if stripped and set(stripped) == {fence_char} and len(stripped) >= fence_len:
            segments.append((kind, "".join(buffer)))
            buffer = []
            fence_char, fence_len, kind = "", 0, "prose"
    if buffer:
        segments.append((kind if fence_char else "prose", "".join(buffer)))
    return segments


def count_prose_chars(markdown: str) -> int:
    """コードブロック（Mermaid含む）を除いた本文の文字数（空白を除く）。"""
    prose = "".join(chunk for kind, chunk in split_markdown_segments(markdown or "") if kind == "prose")
    return len(re.sub(r"\s+", "", prose))


class DraftSection(_Base):
    """執筆済みの1章。char_count は本文（コード除く）から自動算出される。"""

    role: BlueprintRole
    heading: str = Field(..., min_length=1, max_length=300)
    body_markdown: str = Field(..., min_length=1, max_length=40000)
    char_count: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _recount(self) -> "DraftSection":
        self.char_count = count_prose_chars(self.body_markdown)
        return self


class ArticleDraft(_Base):
    """5章構成の記事ドラフト。"""

    title: str = Field(..., min_length=1, max_length=300)
    lead: str = Field(..., min_length=1, max_length=6000)
    sections: list[DraftSection] = Field(..., min_length=5, max_length=5)
    closing: str = Field(..., min_length=1, max_length=6000)

    @model_validator(mode="after")
    def _canonical_order(self) -> "ArticleDraft":
        _check_role_order(self.sections)
        return self

    def to_markdown(self) -> str:
        """タイトル・リード・5章・結語を Markdown へ整形する。"""
        parts = [f"# {self.title.strip()}", "", self.lead.strip(), ""]
        for section in self.sections:
            parts.extend([f"## {section.heading.strip()}", "", section.body_markdown.strip(), ""])
        parts.extend(["## むすび", "", self.closing.strip(), ""])
        return "\n".join(parts).rstrip() + "\n"

    @property
    def total_chars(self) -> int:
        return count_prose_chars(self.lead) + sum(s.char_count for s in self.sections) + count_prose_chars(self.closing)


class Replacement(_Base):
    """一人称統一で行った置換の記録。"""

    original: str
    replacement: str
    index: int = Field(..., ge=0)
    context: str = ""


class RemovedURL(_Base):
    """除去したURL（またはリンク構文）の記録。"""

    kind: Literal[
        "bare_url",
        "markdown_link",
        "markdown_image",
        "reference_definition",
        "reference_link",
        "html_anchor",
        "html_media",
        "autolink",
        "mermaid_click",
        "defanged_url",
        "domain_like",
    ]
    original: str
    kept_text: str = ""


class SanitizationReport(_Base):
    """Compliance & Style Sanitizer の結果。"""

    first_person_replacements: list[Replacement] = Field(default_factory=list)
    urls_removed: list[RemovedURL] = Field(default_factory=list)
    first_person_unified: bool = True
    url_free: bool = True
    residual_first_person: list[str] = Field(default_factory=list)
    residual_urls: list[str] = Field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.first_person_replacements) + len(self.urls_removed)

    def merge(self, other: "SanitizationReport") -> "SanitizationReport":
        """2つの報告を統合する（残存状況は後者＝より新しい検査結果を採用）。"""
        return SanitizationReport(
            first_person_replacements=[*self.first_person_replacements, *other.first_person_replacements],
            urls_removed=[*self.urls_removed, *other.urls_removed],
            first_person_unified=other.first_person_unified,
            url_free=other.url_free,
            residual_first_person=list(other.residual_first_person),
            residual_urls=list(other.residual_urls),
        )


class SectionScore(_Base):
    """1章の批評スコア（4観点×25点）。total は4観点の合計として常に再計算される。"""

    role: BlueprintRole
    heading: str = ""
    logic_coherence: float = Field(..., ge=0.0, le=25.0)
    abstraction_balance: float = Field(..., ge=0.0, le=25.0)
    persuasiveness: float = Field(..., ge=0.0, le=25.0)
    evidence_density: float = Field(..., ge=0.0, le=25.0)
    total: float = Field(0.0, ge=0.0, le=100.0)
    feedback: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _recompute_total(self) -> "SectionScore":
        self.total = round(
            self.logic_coherence + self.abstraction_balance + self.persuasiveness + self.evidence_density, 2
        )
        return self


class CritiqueScore(_Base):
    """記事全体の批評結果。"""

    sections: list[SectionScore] = Field(..., min_length=1)
    overall: float = Field(..., ge=0.0, le=100.0)
    passed: bool = False
    threshold: float = Field(80.0, ge=0.0, le=100.0)

    def section(self, role: BlueprintRole | str) -> SectionScore | None:
        value = role.value if isinstance(role, BlueprintRole) else str(role)
        for section in self.sections:
            if section.role.value == value:
                return section
        return None

    def weak_sections(self) -> list[SectionScore]:
        return [section for section in self.sections if section.total < self.threshold]


class PipelineStageRecord(_Base):
    stage: str
    iteration: int = 0
    duration_ms: float = 0.0
    score: float | None = None
    notes: str = ""


class PublishedArticle(_Base):
    """5段階パイプラインを経た公開可能な記事。"""

    id: str = Field(default_factory=lambda: new_id("article"))
    title: str = Field(..., min_length=1, max_length=300)
    axis: ContentAxis
    source_id: str = ""
    markdown: str = Field(..., min_length=1)
    blueprint: ArticleBlueprint
    draft: ArticleDraft
    sanitization: SanitizationReport
    critique_history: list[CritiqueScore] = Field(default_factory=list)
    final_score: float = Field(0.0, ge=0.0, le=100.0)
    iterations: int = Field(0, ge=0)
    stages: list[PipelineStageRecord] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class ArticleSummary(_Base):
    """一覧表示用の記事要約。"""

    id: str
    title: str
    axis: ContentAxis
    source_id: str = ""
    final_score: float = 0.0
    tags: list[str] = Field(default_factory=list)
    char_count: int = 0
    created_at: str = ""
