"""構造駆動型・漫画ネーム＆ComfyUI生成パイプライン（Manga Pipeline）。

``schemas`` のみ即時 import し、その他のサブモジュールは属性アクセス時に遅延 import する。
"""

from __future__ import annotations

import importlib
from typing import Any

from core.manga_pipeline.schemas import (  # noqa: F401 - 再エクスポート
    Balloon,
    BalloonKind,
    CameraAngle,
    CameraWork,
    CharacterAction,
    CharacterProfile,
    CompositionRule,
    GazeTransition,
    GrammarReport,
    GrammarViolation,
    Page,
    PageGeometry,
    PageSide,
    PageTurnRole,
    Panel,
    PanelShape,
    Point,
    Polygon,
    ShotSize,
    SoundEffect,
    Spread,
    Storyboard,
)

_LAZY_SUBMODULES = (
    "geometry",
    "text_layout",
    "storyboard_generator",
    "grammar_validator",
    "comfy_compiler",
    "comfy_client",
    "comfy_mock_server",
    "layout_renderer",
    "prompts",
    "mock_handlers",
    "diagnostics_schemas",
    "diagnostics",
    "diagnostics_overlay",
    "page_vision",
    "png_codec",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"core.manga_pipeline.{name}")
    raise AttributeError(f"module 'core.manga_pipeline' has no attribute {name!r}")
