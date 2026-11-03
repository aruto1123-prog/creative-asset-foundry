"""極限絵柄開発・モデル調合＆LoRA研究ラボ（Style Alchemy）。

共有契約である ``schemas`` と ``comfy_export`` を即時 import し、その他のサブモジュールは
属性アクセス時に遅延 import する。
"""

from __future__ import annotations

import importlib
from typing import Any

from core.style_alchemy.comfy_export import (  # noqa: F401 - 再エクスポート
    ComfyGraphBuilder,
    ModelStackRefs,
    adapt_prompt_to_object_info,
    build_model_stack,
    build_txt2img_workflow,
    snap_resolution,
)
from core.style_alchemy.schemas import (  # noqa: F401 - 再エクスポート
    ARCHITECTURE_BLOCKS,
    LBW_BLOCKS,
    BlockMergeRecipe,
    CheckpointRef,
    LoRASpec,
    MergeMethod,
    ModelArchitecture,
    SamplerSettings,
    StyleRecipe,
    block_groups,
    expand_group_weights,
)

_LAZY_SUBMODULES = (
    "preference_engine",
    "model_optimizer",
    "dataset_curator",
    "grid_laboratory",
    "repository",
    "mock_handlers",
    "prompts",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"core.style_alchemy.{name}")
    raise AttributeError(f"module 'core.style_alchemy' has no attribute {name!r}")
