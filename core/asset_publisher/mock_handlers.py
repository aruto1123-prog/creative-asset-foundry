"""Asset Publisher 用の高精度モックLLMハンドラ。

各タスクは決定論的ベースライン（matrix_transpiler / writer_pipeline）を、ハーネスが渡す
``pub_*`` コンテキストから再構成して返す。これにより、モックLLM経由でも実LLMと同じ
スキーマ・同じ品質基準を満たす応答が決定論的に得られる。
"""

from __future__ import annotations

from typing import Any

from core.asset_publisher.matrix_transpiler import expand_matrix_deterministic
from core.asset_publisher.schemas import (
    ArticleBlueprint,
    ArticleDraft,
    BlueprintRole,
    ContentPlan,
    DraftSection,
    ResearchLog,
    SectionScore,
)
from core.asset_publisher.writer_pipeline import (
    WriterConfig,
    build_blueprint_deterministic,
    compose_section,
    critique_deterministic,
    polish_section_deterministic,
)
from core.llm_client import MockContext, register_mock_handler

__all__ = [
    "mock_publisher_matrix",
    "mock_publisher_blueprint",
    "mock_publisher_section_draft",
    "mock_publisher_critique",
    "mock_publisher_polish",
]

_STAGES = ("synthesize", "refine")


def _log(ctx: MockContext) -> ResearchLog | None:
    payload = ctx.context.get("pub_research_log")
    return ResearchLog.model_validate(payload) if isinstance(payload, dict) else None


def _plan(ctx: MockContext) -> ContentPlan:
    return ContentPlan.model_validate(ctx.context["pub_plan"])


def _config(ctx: MockContext) -> WriterConfig:
    payload = ctx.context.get("pub_config")
    return WriterConfig.model_validate(payload) if isinstance(payload, dict) else WriterConfig()


@register_mock_handler("publisher.matrix", stages=_STAGES)
def mock_publisher_matrix(ctx: MockContext) -> dict[str, Any] | None:
    payload = ctx.context.get("pub_research_log")
    if not isinstance(payload, dict):
        return None
    return expand_matrix_deterministic(ResearchLog.model_validate(payload)).model_dump(mode="json")


@register_mock_handler("publisher.blueprint", stages=_STAGES)
def mock_publisher_blueprint(ctx: MockContext) -> dict[str, Any] | None:
    if "pub_plan" not in ctx.context:
        return None
    return build_blueprint_deterministic(_plan(ctx), _log(ctx)).model_dump(mode="json")


@register_mock_handler("publisher.section_draft", stages=_STAGES)
def mock_publisher_section_draft(ctx: MockContext) -> dict[str, Any] | None:
    if "pub_blueprint" not in ctx.context or "pub_role" not in ctx.context:
        return None
    blueprint = ArticleBlueprint.model_validate(ctx.context["pub_blueprint"])
    section = compose_section(blueprint, BlueprintRole(ctx.context["pub_role"]), _plan(ctx), _log(ctx), _config(ctx))
    return section.model_dump(mode="json")


@register_mock_handler("publisher.critique", stages=_STAGES)
def mock_publisher_critique(ctx: MockContext) -> dict[str, Any] | None:
    payload = ctx.context.get("pub_article_draft")
    if not isinstance(payload, dict):
        return None
    draft = ArticleDraft.model_validate(payload)
    blueprint_payload = ctx.context.get("pub_blueprint")
    blueprint = ArticleBlueprint.model_validate(blueprint_payload) if isinstance(blueprint_payload, dict) else None
    threshold = float(ctx.context.get("pub_threshold", 80.0))
    return critique_deterministic(draft, blueprint, threshold, _config(ctx)).model_dump(mode="json")


@register_mock_handler("publisher.polish", stages=_STAGES)
def mock_publisher_polish(ctx: MockContext) -> dict[str, Any] | None:
    if "pub_section" not in ctx.context or "pub_blueprint" not in ctx.context:
        return None
    section = DraftSection.model_validate(ctx.context["pub_section"])
    score_payload = ctx.context.get("pub_section_score")
    score = SectionScore.model_validate(score_payload) if isinstance(score_payload, dict) else None
    blueprint = ArticleBlueprint.model_validate(ctx.context["pub_blueprint"])
    polished = polish_section_deterministic(
        section,
        score,
        blueprint.section(section.role),
        _plan(ctx),
        _log(ctx),
        _config(ctx),
        int(ctx.context.get("pub_round", 1)),
    )
    return polished.model_dump(mode="json")
