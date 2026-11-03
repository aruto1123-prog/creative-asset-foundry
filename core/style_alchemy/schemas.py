"""Style Alchemy 共有スキーマ: モデルアーキテクチャ別ブロック定義と StyleRecipe。

``StyleRecipe`` は Style Alchemy（最適化・エクスポート）と Manga Pipeline（ComfyUIコンパイル）の
共有契約であり、「どのベースモデルを、どの階層比率でマージし、どの LoRA をどの階層ウェイトで適用し、
どのサンプラー設定で生成するか」を完全に記述する。

ブロック命名（本システムの正準名）:
- SDXL : BASE, IN00..IN08, MID, OUT00..OUT08（20）
- SD1.5: BASE, IN00..IN11, MID, OUT00..OUT11（26）
- FLUX : BASE, DB00..DB18（Double Blocks）, SB00..SB37（Single Blocks）（58）
- ANIMA: BASE, B00..B27（Cosmos-Predict2 2B 系の汎用DiTを想定した 28 ブロック。深さはチェックポイントに応じて読み替える）
BASE はテキストエンコーダおよび埋め込み層・最終層などブロック外の重みを表す。

``BlockMergeRecipe.block_weights`` の値は **model_b の比率**（0=純A, 1=純B）。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import new_id, utc_now_iso

__all__ = [
    "ModelArchitecture",
    "ARCHITECTURE_BLOCKS",
    "LBW_BLOCKS",
    "ARCHITECTURE_PROFILES",
    "block_groups",
    "expand_group_weights",
    "architecture_family",
    "CheckpointRef",
    "MergeMethod",
    "BlockMergeRecipe",
    "LoRASpec",
    "SamplerSettings",
    "StyleRecipe",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ModelArchitecture(str, Enum):
    SDXL = "sdxl"
    SD15 = "sd15"
    FLUX = "flux"
    ANIMA = "anima"


def _seq(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(count)]


ARCHITECTURE_BLOCKS: dict[ModelArchitecture, list[str]] = {
    ModelArchitecture.SDXL: ["BASE", *_seq("IN", 9), "MID", *_seq("OUT", 9)],
    ModelArchitecture.SD15: ["BASE", *_seq("IN", 12), "MID", *_seq("OUT", 12)],
    ModelArchitecture.FLUX: ["BASE", *_seq("DB", 19), *_seq("SB", 38)],
    ModelArchitecture.ANIMA: ["BASE", *_seq("B", 28)],
}

# LoRA Block Weight の慣行的な評価点（Attention を含むブロックのみ）
LBW_BLOCKS: dict[ModelArchitecture, list[str]] = {
    ModelArchitecture.SD15: [
        "BASE", "IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "MID",
        "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11",
    ],
    ModelArchitecture.SDXL: [
        "BASE", "IN04", "IN05", "IN07", "IN08", "MID",
        "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05",
    ],
    ModelArchitecture.FLUX: list(ARCHITECTURE_BLOCKS[ModelArchitecture.FLUX]),
    ModelArchitecture.ANIMA: list(ARCHITECTURE_BLOCKS[ModelArchitecture.ANIMA]),
}

# ComfyUI 上の構成プロファイル（ローダー種別・潜在ノード・既定ファイル名）
ARCHITECTURE_PROFILES: dict[ModelArchitecture, dict[str, Any]] = {
    ModelArchitecture.SDXL: {
        "loader": "checkpoint",
        "latent_node": "EmptyLatentImage",
        "merge_node": "ModelMergeSDXL",
        "native_resolution": 1024,
        "guidance_node": False,
    },
    ModelArchitecture.SD15: {
        "loader": "checkpoint",
        "latent_node": "EmptyLatentImage",
        "merge_node": "ModelMergeSD1",
        "native_resolution": 512,
        "guidance_node": False,
    },
    ModelArchitecture.FLUX: {
        "loader": "unet",
        "latent_node": "EmptySD3LatentImage",
        "merge_node": "ModelMergeFlux1",
        "clip_loader": "DualCLIPLoader",
        "clip_type": "flux",
        "default_text_encoders": ["clip_l.safetensors", "t5xxl_fp16.safetensors"],
        "default_vae": "ae.safetensors",
        "native_resolution": 1024,
        "guidance_node": True,
    },
    ModelArchitecture.ANIMA: {
        "loader": "unet",
        "latent_node": "EmptyLatentImage",
        "merge_node": "ModelMergeCosmosPredict2_2B",
        "clip_loader": "CLIPLoader",
        "clip_type": "stable_diffusion",
        "default_text_encoders": ["text_encoder.safetensors"],
        "default_vae": "vae.safetensors",
        "native_resolution": 1024,
        "guidance_node": False,
    },
}


def architecture_family(arch: ModelArchitecture | str) -> str:
    """``unet``（SDXL/SD1.5）または ``dit``（FLUX/ANIMA）。"""
    value = ModelArchitecture(arch)
    return "unet" if value in (ModelArchitecture.SDXL, ModelArchitecture.SD15) else "dit"


def block_groups(arch: ModelArchitecture | str) -> dict[str, list[str]]:
    """階層グループ（UNet: BASE/IN/MID/OUT、FLUX: BASE/DOUBLE/SINGLE、ANIMA: BASE/EARLY/MIDDLE/LATE）。"""
    value = ModelArchitecture(arch)
    blocks = ARCHITECTURE_BLOCKS[value]
    if value in (ModelArchitecture.SDXL, ModelArchitecture.SD15):
        return {
            "BASE": ["BASE"],
            "IN": [b for b in blocks if b.startswith("IN")],
            "MID": ["MID"],
            "OUT": [b for b in blocks if b.startswith("OUT")],
        }
    if value == ModelArchitecture.FLUX:
        return {
            "BASE": ["BASE"],
            "DOUBLE": [b for b in blocks if b.startswith("DB")],
            "SINGLE": [b for b in blocks if b.startswith("SB")],
        }
    body = [b for b in blocks if b != "BASE"]
    size, remainder = divmod(len(body), 3)
    sizes = [size + (1 if i < remainder else 0) for i in range(3)]
    early = body[: sizes[0]]
    middle = body[sizes[0] : sizes[0] + sizes[1]]
    late = body[sizes[0] + sizes[1] :]
    return {"BASE": ["BASE"], "EARLY": early, "MIDDLE": middle, "LATE": late}


def expand_group_weights(
    arch: ModelArchitecture | str, group_weights: dict[str, float], default: float = 0.5
) -> dict[str, float]:
    """グループ単位の重みを全ブロックへ展開する（個別ブロック指定があればそれを優先）。"""
    value = ModelArchitecture(arch)
    groups = block_groups(value)
    expanded: dict[str, float] = {}
    for group, members in groups.items():
        weight = float(group_weights.get(group, default))
        for block in members:
            expanded[block] = weight
    for key, weight in group_weights.items():
        if key in ARCHITECTURE_BLOCKS[value]:
            expanded[key] = float(weight)
    return expanded


class CheckpointRef(_Base):
    """チェックポイント（ComfyUI の models フォルダ内ファイル名）参照。"""

    name: str = Field(..., min_length=1, max_length=300)
    path: str = Field("", max_length=1000)
    architecture: ModelArchitecture = ModelArchitecture.SDXL


class MergeMethod(str, Enum):
    WEIGHTED_SUM = "weighted_sum"
    ADD_DIFFERENCE = "add_difference"


class BlockMergeRecipe(_Base):
    """階層別ブロックウェイトマージのレシピ。block_weights は model_b の比率。"""

    architecture: ModelArchitecture
    method: MergeMethod = MergeMethod.WEIGHTED_SUM
    model_a: CheckpointRef
    model_b: CheckpointRef
    model_c: CheckpointRef | None = None
    block_weights: dict[str, float] = Field(default_factory=dict)
    default_weight: float = Field(0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_recipe(self) -> "BlockMergeRecipe":
        valid = set(ARCHITECTURE_BLOCKS[self.architecture])
        unknown = sorted(set(self.block_weights) - valid)
        if unknown:
            raise ValueError(f"{self.architecture.value} に存在しないブロックです: {', '.join(unknown)}")
        for block, weight in self.block_weights.items():
            if not 0.0 <= float(weight) <= 1.0:
                raise ValueError(f"ブロック {block} の比率 {weight} は [0,1] の範囲外です")
        if self.method == MergeMethod.ADD_DIFFERENCE and self.model_c is None:
            raise ValueError("add_difference には model_c（差分の基準モデル）が必要です")
        return self

    def weight_for(self, block: str) -> float:
        return float(self.block_weights.get(block, self.default_weight))

    def full_vector(self) -> list[float]:
        """正準ブロック順の比率ベクトル。"""
        return [self.weight_for(block) for block in ARCHITECTURE_BLOCKS[self.architecture]]

    def to_mbw_string(self, precision: int = 3) -> str:
        """SuperMerger 等の MBW 形式（カンマ区切り、正準順）。"""
        return ",".join(f"{round(w, precision):g}" for w in self.full_vector())

    def group_means(self) -> dict[str, float]:
        groups = block_groups(self.architecture)
        return {
            group: round(sum(self.weight_for(b) for b in members) / len(members), 6)
            for group, members in groups.items()
        }


class LoRASpec(_Base):
    """LoRA 適用仕様。block_weights は LBW 乗数（未指定ブロックは 1.0）。"""

    name: str = Field(..., min_length=1, max_length=300)
    path: str = Field("", max_length=1000)
    strength_model: float = Field(1.0, ge=-2.0, le=2.0)
    strength_clip: float = Field(1.0, ge=-2.0, le=2.0)
    block_weights: dict[str, float] | None = None
    trigger_words: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_weights(self) -> "LoRASpec":
        if self.block_weights:
            for block, weight in self.block_weights.items():
                if not -2.0 <= float(weight) <= 2.0:
                    raise ValueError(f"LoRA {self.name} のブロック {block} の乗数 {weight} は [-2,2] の範囲外です")
        return self

    def lbw_vector(self, arch: ModelArchitecture | str) -> list[float]:
        blocks = LBW_BLOCKS[ModelArchitecture(arch)]
        weights = self.block_weights or {}
        return [float(weights.get(block, 1.0)) for block in blocks]

    def lbw_string(self, arch: ModelArchitecture | str, precision: int = 3) -> str:
        return ",".join(f"{round(w, precision):g}" for w in self.lbw_vector(arch))

    def has_block_weights(self) -> bool:
        """非一様な LBW 指定があるか。"""
        if not self.block_weights:
            return False
        return any(abs(float(w) - 1.0) > 1e-9 for w in self.block_weights.values())


class SamplerSettings(_Base):
    """サンプラー設定（ComfyUI KSampler 準拠）。"""

    sampler_name: str = "dpmpp_2m"
    scheduler: str = "karras"
    steps: int = Field(28, ge=1, le=200)
    cfg: float = Field(6.0, ge=0.0, le=30.0)
    denoise: float = Field(1.0, ge=0.0, le=1.0)
    clip_skip: int = Field(1, ge=1, le=12)


class StyleRecipe(_Base):
    """絵柄レシピ（モデル調合・LoRA配合・生成設定）の完全記述。"""

    id: str = Field(default_factory=lambda: new_id("recipe"))
    name: str = Field("untitled recipe", min_length=1, max_length=200)
    architecture: ModelArchitecture = ModelArchitecture.SDXL
    base_checkpoint: CheckpointRef
    merge: BlockMergeRecipe | None = None
    loras: list[LoRASpec] = Field(default_factory=list)
    sampler: SamplerSettings = Field(default_factory=SamplerSettings)
    vae: str | None = None
    text_encoders: list[str] = Field(default_factory=list)
    text_encoder_type: str | None = None
    style_tokens: list[str] = Field(default_factory=list)
    negative_tokens: list[str] = Field(default_factory=list)
    params: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    notes: str = Field("", max_length=4000)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "StyleRecipe":
        if self.merge is not None and self.merge.architecture != self.architecture:
            raise ValueError("merge.architecture は recipe.architecture と一致する必要があります")
        valid_blocks = set(ARCHITECTURE_BLOCKS[self.architecture]) | set(LBW_BLOCKS[self.architecture])
        for lora in self.loras:
            if lora.block_weights:
                unknown = sorted(set(lora.block_weights) - valid_blocks)
                if unknown:
                    raise ValueError(
                        f"LoRA {lora.name} に {self.architecture.value} で無効なブロックがあります: {', '.join(unknown)}"
                    )
        return self

    @property
    def profile(self) -> dict[str, Any]:
        return ARCHITECTURE_PROFILES[self.architecture]

    def resolved_text_encoders(self) -> list[str]:
        if self.text_encoders:
            return list(self.text_encoders)
        return list(self.profile.get("default_text_encoders", []))

    def resolved_vae(self) -> str | None:
        if self.vae:
            return self.vae
        return self.profile.get("default_vae")

    def positive_prefix(self) -> str:
        """スタイルトークン＋LoRAトリガーワード。"""
        tokens: list[str] = []
        for token in [*self.style_tokens, *(w for lora in self.loras for w in lora.trigger_words)]:
            token = token.strip()
            if token and token not in tokens:
                tokens.append(token)
        return ", ".join(tokens)

    def negative_text(self) -> str:
        return ", ".join(t.strip() for t in self.negative_tokens if t.strip())
