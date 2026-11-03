"""R&D知見の自動体系化・マルチチャネル資産変換器（Asset Publisher）。

``schemas`` と ``sanitizer`` のみ即時 import し、その他のサブモジュール（LLM 連携を含む）は
属性アクセス時に遅延 import する。
"""

from __future__ import annotations

import importlib
from typing import Any

from core.asset_publisher.sanitizer import (  # noqa: F401 - 再エクスポート
    FIRST_PERSON_CANONICAL,
    ComplianceSanitizer,
    detect_first_person_variants,
    find_urls,
    remove_urls,
    sanitize_text,
    unify_first_person,
)
from core.asset_publisher.schemas import (  # noqa: F401 - 再エクスポート
    AXIS_LABELS_JA,
    AXIS_ORDER,
    BLUEPRINT_ROLE_ORDER,
    ArticleBlueprint,
    ArticleDraft,
    BlueprintRole,
    BlueprintSection,
    ContentAxis,
    ContentMatrix,
    ContentPlan,
    CritiqueScore,
    DraftSection,
    Finding,
    FigureSpec,
    OutlineSection,
    PipelineStageRecord,
    PublishedArticle,
    RemovedURL,
    Replacement,
    ResearchLog,
    SanitizationReport,
    SectionScore,
)

_LAZY_SUBMODULES = (
    "matrix_transpiler",
    "writer_pipeline",
    "knowledge_graph",
    "exporter",
    "repository",
    "prompts",
    "mock_handlers",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"core.asset_publisher.{name}")
    raise AttributeError(f"module 'core.asset_publisher' has no attribute {name!r}")
