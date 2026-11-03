"""REST API のリクエストボディ定義。

重い処理のリクエストは ``"async": true`` を受け付け、ジョブとして非同期実行する
（Python の予約語を避けるためフィールド名は ``run_async``、JSON 上のキーは ``async``）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.asset_publisher.schemas import ContentAxis, ContentPlan, ResearchLog
from core.structure_engine.composer import MutationType
from core.structure_engine.schemas import EmotionalInformationWave, StructureCategory
from core.structure_engine.transpiler import TargetDomain
from core.style_alchemy.dataset_curator import CaptionRecord
from core.style_alchemy.grid_laboratory import GridAxis
from core.style_alchemy.schemas import CheckpointRef, LoRASpec, ModelArchitecture, StyleRecipe

__all__ = [
    "AsyncRequest",
    "ExtractRequest",
    "MergeRequest",
    "MutateRequest",
    "TranspileRequest",
    "SearchRequest",
    "SeedLoadRequest",
    "CandidateCreateRequest",
    "BootstrapRequest",
    "ComparisonRequest",
    "SuggestRequest",
    "RecipeExportRequest",
    "DatasetAnalyzeRequest",
    "GridCreateRequest",
    "GridResultRequest",
    "GridComfyRequest",
    "StoryboardCreateRequest",
    "ComfyCompileRequest",
    "ComfyQueueRequest",
    "MatrixRequest",
    "ArticleRequest",
    "TextRequest",
    "SeriesRequest",
]

Grouping = Literal["grouped", "per_block"]


class _Body(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class AsyncRequest(_Body):
    """``async: true`` でジョブ化できるリクエストの基底。"""

    run_async: bool = Field(False, alias="async")


# -- Structure Lab ---------------------------------------------------------------


class ExtractRequest(AsyncRequest):
    text: str = Field(..., min_length=1)
    source_kind: str = "synopsis"
    title_hint: str | None = None
    category: StructureCategory = StructureCategory.CUSTOM
    save: bool = True


class MergeRequest(_Body):
    struct_a_id: str = Field(..., min_length=1)
    struct_b_id: str = Field(..., min_length=1)
    weight_topology: float = Field(0.7, ge=0.0, le=1.0)
    weight_wave: float = Field(0.7, ge=0.0, le=1.0)
    title: str | None = None
    save: bool = True


class MutateRequest(_Body):
    structure_id: str = Field(..., min_length=1)
    mutation_type: MutationType
    intensity: float = Field(1.0, ge=0.0, le=1.0)
    seed: int = 0
    save: bool = True


class TranspileRequest(_Body):
    domain: TargetDomain
    theme_variables: dict[str, str] = Field(default_factory=dict)


class SearchRequest(_Body):
    structure_id: str | None = None
    wave: EmotionalInformationWave | None = None
    method: Literal["dtw", "cosine", "hybrid"] = "hybrid"
    top_k: int = Field(5, ge=1, le=100)
    category: str | None = None


class SeedLoadRequest(_Body):
    overwrite: bool = False


# -- Style Alchemy ---------------------------------------------------------------


class CandidateCreateRequest(_Body):
    label: str = ""
    params: dict[str, float] = Field(default_factory=dict)
    recipe: StyleRecipe | None = None
    image_ref: str | None = None
    source: str = "manual"


class BootstrapRequest(_Body):
    n: int = Field(8, ge=2, le=64)
    architecture: ModelArchitecture = ModelArchitecture.SDXL
    lora_names: list[str] = Field(default_factory=list)
    grouping: Grouping = "grouped"
    seed: int = 0


class ComparisonRequest(_Body):
    left_id: str = Field(..., min_length=1)
    right_id: str = Field(..., min_length=1)
    outcome: Literal["left", "right", "tie"]
    response_time_ms: float | None = Field(None, ge=0.0)
    session_id: str | None = None


class SuggestRequest(_Body):
    n: int = Field(3, ge=1, le=12)
    strategy: Literal["gp", "cmaes", "genetic", "hybrid"] = "hybrid"
    acquisition: Literal["ei", "ucb", "pi"] = "ei"
    architecture: ModelArchitecture = ModelArchitecture.SDXL
    lora_names: list[str] = Field(default_factory=list)
    grouping: Grouping = "grouped"
    register_candidates: bool = Field(False, alias="register")
    seed: int = 0


def _default_model_a() -> CheckpointRef:
    return CheckpointRef(name="caf_model_a.safetensors")


def _default_model_b() -> CheckpointRef:
    return CheckpointRef(name="caf_model_b.safetensors")


class RecipeExportRequest(_Body):
    params: dict[str, float] = Field(default_factory=dict)
    architecture: ModelArchitecture = ModelArchitecture.SDXL
    model_a: CheckpointRef = Field(default_factory=_default_model_a)
    model_b: CheckpointRef = Field(default_factory=_default_model_b)
    model_c: CheckpointRef | None = None
    loras: list[LoRASpec] = Field(default_factory=list)
    name: str | None = None
    prompt: str = "1girl, manga panel, dynamic pose, detailed background"
    negative: str = "lowres, bad anatomy, blurry, watermark"
    seed: int = 0
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(1024, ge=64, le=4096)


class DatasetAnalyzeRequest(_Body):
    captions_text: str | None = None
    records: list[CaptionRecord] | None = None
    trigger_token: str = "caf_style"
    use_llm: bool = False
    save: bool = True
    name: str = "dataset analysis"


class GridCreateRequest(_Body):
    name: str = Field(..., min_length=1, max_length=200)
    base_prompt: str = ""
    negative_prompt: str = ""
    x_axis: GridAxis
    y_axis: GridAxis | None = None
    z_axis: GridAxis | None = None
    base_params: dict[str, float] = Field(default_factory=dict)
    seed: int = 0


class GridResultRequest(_Body):
    cell_id: str = Field(..., min_length=1)
    score: float
    image_ref: str | None = None
    notes: str = ""


class GridComfyRequest(_Body):
    architecture: ModelArchitecture = ModelArchitecture.SDXL
    model_a: CheckpointRef = Field(default_factory=_default_model_a)
    model_b: CheckpointRef | None = None
    loras: list[LoRASpec] = Field(default_factory=list)
    width: int | None = Field(None, ge=64, le=4096)
    height: int | None = Field(None, ge=64, le=4096)


# -- Manga Studio ----------------------------------------------------------------


class StoryboardCreateRequest(AsyncRequest):
    structure_id: str = Field(..., min_length=1)
    theme: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    title: str | None = Field(None, max_length=200)


class ComfyCompileRequest(_Body):
    recipe: StyleRecipe | None = None
    recipe_params: dict[str, float] | None = None
    architecture: ModelArchitecture = ModelArchitecture.SDXL
    model_a: CheckpointRef | None = None
    model_b: CheckpointRef | None = None
    loras: list[LoRASpec] = Field(default_factory=list)
    panel_ids: list[str] | None = None
    base_resolution: int | None = Field(None, ge=256, le=2048)


class ComfyQueueRequest(ComfyCompileRequest):
    max_panels: int = Field(8, ge=1, le=500)


# -- Asset Publisher -------------------------------------------------------------


class MatrixRequest(AsyncRequest):
    structure_id: str | None = None
    research_log: ResearchLog | None = None
    experiment: dict[str, Any] | None = None
    save: bool = True


class ArticleRequest(AsyncRequest):
    plan: ContentPlan | None = None
    matrix_id: str | None = None
    axis: ContentAxis | None = None
    research_log: ResearchLog | None = None
    structure_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class TextRequest(_Body):
    text: str


class SeriesRequest(_Body):
    article_ids: list[str] = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    format: Literal["note", "obsidian"] = "obsidian"
