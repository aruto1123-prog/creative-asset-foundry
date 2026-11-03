"""普遍的構造抽出・構造代数演算・異分野トランスパイラ（Structure Engine）。

``schemas`` のみ即時 import し、その他のサブモジュールは属性アクセス時に遅延 import する
（スキーマだけを使う利用者が重い依存を読み込まずに済むようにするため）。
"""

from __future__ import annotations

import importlib
from typing import Any

from core.structure_engine.schemas import (  # noqa: F401 - 再エクスポート
    DEFAULT_CHANNELS,
    WAVE_CHANNELS,
    WAVE_MIN_POINTS,
    BeatType,
    BlueprintSlot,
    CognitiveEmpathyMechanics,
    CognitiveMechanism,
    DomainAgnosticBlueprint,
    DynamicTensionGraph,
    EdgeKind,
    EmotionalInformationWave,
    MechanismCategory,
    NodeKind,
    PanelRhythmProfile,
    SlotCategory,
    SourceKind,
    StructuralBeat,
    StructuralBeatChain,
    StructureCategory,
    StructureLayers,
    StructureMetadata,
    TensionEdge,
    TensionNode,
    TransferGuide,
    UniversalStructure,
    new_structure_id,
)

_LAZY_SUBMODULES = (
    "consistency",
    "composer",
    "extractor",
    "mock_handlers",
    "prompts",
    "repository",
    "similarity",
    "transpiler",
    "visualizer",
    "seeds",
    "samples",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"core.structure_engine.{name}")
    raise AttributeError(f"module 'core.structure_engine' has no attribute {name!r}")
