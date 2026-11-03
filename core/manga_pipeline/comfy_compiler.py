"""ComfyUI 動的ワークフロー・コンパイラ。

各コマの演出パラメータ（ショット・アングル・レンズ・構図・人物の演技）と Style Alchemy の
``StyleRecipe``（階層マージ・LoRA 階層ウェイト・サンプラー）から、ComfyUI API 形式の
ノードグラフを動的にコンパイルする。

- モデルスタック: ``core.style_alchemy.comfy_export.build_model_stack``（ローダー→マージ→LoRA→CLIP Skip）
- 領域分割（Attention Couple 相当）: 人物ごとに CLIPTextEncode → SolidMask/MaskComposite でマスクを作り
  ConditioningSetMask で領域条件化し、全体条件（弱め）と ConditioningCombine で統合する。
  人物領域はコマ内正規化座標とスケールから求め、重なりは中心間の中点で縦帯に分割、8 の倍数に丸める。
- ControlNet（OpenPose / Lineart / Depth）: LoadImage → ControlNetLoader → ControlNetApplyAdvanced を
  positive / negative の両方へ連鎖させる。
- シード管理: ``SeedManager`` がネームのシードとコマIDから決定論的シードを導出し、固定（lock）も可能。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.common import stable_seed, utc_now_iso
from core.manga_pipeline.schemas import CameraAngle, CompositionRule, Panel, ShotSize, Storyboard
from core.style_alchemy.comfy_export import (
    ComfyGraphBuilder,
    build_model_stack,
    snap_resolution,
    validate_prompt,
)
from core.style_alchemy.schemas import StyleRecipe

__all__ = [
    "ControlNetKind",
    "DEFAULT_CONTROLNET_MODELS",
    "DEFAULT_NEGATIVE_PROMPT",
    "ControlGuide",
    "CompilerConfig",
    "RegionMask",
    "CompiledPanelWorkflow",
    "SeedManager",
    "ComfyWorkflowCompiler",
    "SHOT_PHRASES",
    "ANGLE_PHRASES",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ControlNetKind(str, Enum):
    OPENPOSE = "openpose"
    LINEART = "lineart"
    DEPTH = "depth"


DEFAULT_CONTROLNET_MODELS: dict[str, str] = {
    ControlNetKind.OPENPOSE.value: "controlnet_openpose_sdxl.safetensors",
    ControlNetKind.LINEART.value: "controlnet_lineart_sdxl.safetensors",
    ControlNetKind.DEPTH.value: "controlnet_depth_sdxl.safetensors",
}

DEFAULT_NEGATIVE_PROMPT = (
    "color, colorful, lowres, bad anatomy, bad hands, extra fingers, missing fingers, deformed face, "
    "text, speech bubble text, watermark, signature, blurry, jpeg artifacts, worst quality, low quality"
)

SHOT_PHRASES: dict[str, str] = {
    ShotSize.ECU.value: "extreme close-up",
    ShotSize.CU.value: "close-up",
    ShotSize.MS.value: "medium shot",
    ShotSize.FS.value: "full body shot",
    ShotSize.ELS.value: "extreme long shot, wide establishing shot",
}

ANGLE_PHRASES: dict[str, str] = {
    CameraAngle.EYE_LEVEL.value: "eye level",
    CameraAngle.HIGH.value: "high angle, from above",
    CameraAngle.LOW.value: "low angle, from below",
    CameraAngle.DUTCH.value: "dutch angle, tilted frame",
}

COMPOSITION_PHRASES: dict[str, str] = {
    CompositionRule.RULE_OF_THIRDS.value: "rule of thirds composition",
    CompositionRule.CENTERED.value: "centered composition",
    CompositionRule.DIAGONAL.value: "dynamic diagonal composition",
    CompositionRule.GOLDEN_SPIRAL.value: "golden spiral composition",
    CompositionRule.SYMMETRY.value: "symmetrical composition, face-off",
    CompositionRule.FRAME_IN_FRAME.value: "frame within a frame",
    CompositionRule.LEADING_LINES.value: "leading lines",
    CompositionRule.NEGATIVE_SPACE.value: "large negative space",
}


class ControlGuide(_Base):
    """ControlNet ガイド（ComfyUI の input フォルダ内の画像名）。"""

    kind: ControlNetKind
    image_name: str = Field(..., min_length=1)
    strength: float = Field(0.8, ge=0.0, le=2.0)
    start_percent: float = Field(0.0, ge=0.0, le=1.0)
    end_percent: float = Field(0.85, ge=0.0, le=1.0)
    model_name: str | None = None


class CompilerConfig(_Base):
    """コンパイラ設定。"""

    base_resolution: int = Field(1024, ge=256, le=2048)
    controlnet_models: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_CONTROLNET_MODELS))
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    filename_prefix: str = "caf"
    seed_base: int = 0
    region_feather: int = Field(32, ge=0, le=256)
    region_strength: float = Field(1.0, ge=0.0, le=2.0)
    global_strength: float = Field(0.6, ge=0.0, le=2.0)
    monochrome: bool = True


class RegionMask(_Base):
    """人物領域（ピクセル、8の倍数）。"""

    character_id: str
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., ge=8)
    height: int = Field(..., ge=8)
    strength: float = 1.0


class CompiledPanelWorkflow(_Base):
    """1コマ分のコンパイル済みワークフロー。"""

    panel_id: str
    page_index: int
    prompt: dict[str, Any]
    seed: int
    width: int
    height: int
    regions: list[RegionMask] = Field(default_factory=list)
    positive_text: str
    negative_text: str
    crop: Literal["full", "right_half", "left_half"] = "full"
    filename_prefix: str = ""
    warnings: list[str] = Field(default_factory=list)


class SeedManager:
    """コマごとのシード管理（決定論的導出・固定・履歴）。"""

    def __init__(self, base_seed: int = 0) -> None:
        self.base_seed = int(base_seed)
        self._locks: dict[str, int] = {}
        self.history: list[dict[str, Any]] = []

    def derive(self, storyboard_seed: int, panel_id: str) -> int:
        if panel_id in self._locks:
            seed = self._locks[panel_id]
            locked = True
        else:
            seed = stable_seed("caf.comfy.seed", self.base_seed, storyboard_seed, panel_id)
            locked = False
        self.history.append({"panel_id": panel_id, "seed": seed, "locked": locked, "at": utc_now_iso()})
        return seed

    def lock(self, panel_id: str, seed: int) -> None:
        self._locks[panel_id] = int(seed)

    def unlock(self, panel_id: str) -> None:
        self._locks.pop(panel_id, None)

    def is_locked(self, panel_id: str) -> bool:
        return panel_id in self._locks

    def locked(self) -> dict[str, int]:
        return dict(self._locks)


def _snap8(value: float) -> int:
    return max(0, int(round(value / 8.0)) * 8)


def _join(*parts: str) -> str:
    return ", ".join(p.strip().strip(",") for p in parts if p and p.strip().strip(","))


class ComfyWorkflowCompiler:
    """ネーム（Storyboard）→ コマごとの ComfyUI API ワークフロー。"""

    def __init__(self, recipe: StyleRecipe, config: CompilerConfig | None = None, seed_manager: SeedManager | None = None) -> None:
        self.recipe = recipe
        self.config = config or CompilerConfig()
        self.seed_manager = seed_manager or SeedManager(self.config.seed_base)

    # -- 解像度・領域 -----------------------------------------------------------

    def panel_resolution(self, panel: Panel) -> tuple[int, int]:
        """コマの外接矩形のアスペクトを保ち、base_resolution² 相当の画素数・64の倍数へ丸める。"""
        x0, y0, x1, y1 = panel.polygon.bbox()
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        if panel.spread_role in ("main", "continuation"):
            width *= 2.0
        return snap_resolution(width, height, target=self.config.base_resolution, multiple=64)

    def compute_regions(self, panel: Panel, width: int, height: int) -> list[RegionMask]:
        """人物ごとの領域マスク（重なりは中心間の中点で縦帯に分割）。"""
        if len(panel.characters) < 2:
            return []
        x0, y0, x1, y1 = panel.polygon.bbox()
        aspect = max(1e-6, (y1 - y0) / max(1e-6, x1 - x0))
        boxes: list[list[float]] = []
        for character in panel.characters:
            box_h = min(1.0, max(0.18, character.scale * 1.05))
            box_w = min(1.0, max(0.16, character.scale * 0.62 * aspect))
            cx, cy = character.position.x, character.position.y
            boxes.append([max(0.0, cx - box_w / 2), max(0.0, cy - box_h / 2), min(1.0, cx + box_w / 2), min(1.0, cy + box_h / 2)])
        order = sorted(range(len(boxes)), key=lambda i: panel.characters[i].position.x)
        for a, b in zip(order, order[1:]):
            left, right = boxes[a], boxes[b]
            if left[2] > right[0]:
                middle = (panel.characters[a].position.x + panel.characters[b].position.x) / 2.0
                left[2] = min(left[2], middle)
                right[0] = max(right[0], middle)
        regions: list[RegionMask] = []
        for index, box in enumerate(boxes):
            px0 = min(width - 8, _snap8(box[0] * width))
            py0 = min(height - 8, _snap8(box[1] * height))
            px1 = max(px0 + 64, _snap8(box[2] * width))
            py1 = max(py0 + 64, _snap8(box[3] * height))
            px1 = min(width, px1)
            py1 = min(height, py1)
            regions.append(
                RegionMask(
                    character_id=panel.characters[index].character_id,
                    x=px0,
                    y=py0,
                    width=max(8, _snap8(px1 - px0)),
                    height=max(8, _snap8(py1 - py0)),
                    strength=self.config.region_strength,
                )
            )
        return regions

    # -- プロンプト ---------------------------------------------------------------

    def _character_phrase(self, storyboard: Storyboard, panel: Panel, index: int) -> str:
        action = panel.characters[index]
        profile = storyboard.character(action.character_id)
        visual = profile.visual_description if profile else "a person"
        return _join(visual, f"{action.expression} expression", action.pose)

    def build_prompt_text(self, storyboard: Storyboard, panel: Panel) -> tuple[str, str]:
        camera = panel.camera
        lens = ""
        if camera.lens_mm <= 28:
            lens = "wide-angle lens, strong perspective"
        elif camera.lens_mm >= 85:
            lens = "telephoto lens, compressed perspective"
        style = "monochrome manga panel, black and white, ink lineart, screentone shading" if self.config.monochrome else "manga illustration"
        characters = " AND ".join(self._character_phrase(storyboard, panel, i) for i in range(len(panel.characters)))
        if len(panel.characters) > 1:
            characters = f"{len(panel.characters)} people, " + characters
        positive = _join(
            self.recipe.positive_prefix(),
            style,
            SHOT_PHRASES[camera.shot.value],
            ANGLE_PHRASES[camera.angle.value],
            lens,
            COMPOSITION_PHRASES[camera.composition.value],
            ", ".join(panel.prompt_hints[:4]) if panel.prompt_hints else "",
            "dramatic lighting, high contrast, heavy black fill" if panel.tension >= 0.75 else "",
            characters,
        )
        negative = _join(self.config.negative_prompt, self.recipe.negative_text())
        return positive, negative

    # -- コンパイル -------------------------------------------------------------

    def compile_panel(self, storyboard: Storyboard, panel: Panel, controls: list[ControlGuide] | None = None) -> CompiledPanelWorkflow:
        recipe = self.recipe
        width, height = self.panel_resolution(panel)
        seed = self.seed_manager.derive(storyboard.seed, panel.beat_id if panel.spread_role == "continuation" and panel.beat_id else panel.id)
        positive_text, negative_text = self.build_prompt_text(storyboard, panel)
        builder = ComfyGraphBuilder()
        stack = build_model_stack(builder, recipe)
        warnings = list(stack.notes)
        positive_node = builder.add("CLIPTextEncode", {"text": positive_text, "clip": stack.clip}, title=f"Positive {panel.id}")
        negative_node = builder.add("CLIPTextEncode", {"text": negative_text, "clip": stack.clip}, title="Negative")
        positive_ref = builder.ref(positive_node, "CONDITIONING")
        negative_ref = builder.ref(negative_node, "CONDITIONING")
        regions = self.compute_regions(panel, width, height)
        if regions:
            full_mask = builder.add("SolidMask", {"value": 1.0, "width": width, "height": height}, title="Global Mask")
            combined = builder.ref(
                builder.add(
                    "ConditioningSetMask",
                    {
                        "conditioning": positive_ref,
                        "mask": builder.ref(full_mask, "MASK"),
                        "strength": self.config.global_strength,
                        "set_cond_area": "default",
                    },
                    title="Global Condition (weak)",
                ),
                "CONDITIONING",
            )
            for index, region in enumerate(regions):
                text = _join(
                    self.recipe.positive_prefix(),
                    SHOT_PHRASES[panel.camera.shot.value],
                    self._character_phrase(storyboard, panel, index),
                )
                encoded = builder.add("CLIPTextEncode", {"text": text, "clip": stack.clip}, title=f"Region {region.character_id}")
                base = builder.add("SolidMask", {"value": 0.0, "width": width, "height": height}, title=f"Mask Base {index}")
                area = builder.add("SolidMask", {"value": 1.0, "width": region.width, "height": region.height}, title=f"Mask Area {index}")
                composite = builder.add(
                    "MaskComposite",
                    {
                        "destination": builder.ref(base, "MASK"),
                        "source": builder.ref(area, "MASK"),
                        "x": region.x,
                        "y": region.y,
                        "operation": "add",
                    },
                    title=f"Region Mask {region.character_id}",
                )
                mask_ref = builder.ref(composite, "MASK")
                if self.config.region_feather > 0:
                    feather = self.config.region_feather
                    feathered = builder.add(
                        "FeatherMask",
                        {"mask": mask_ref, "left": feather, "top": feather, "right": feather, "bottom": feather},
                        title=f"Feather {index}",
                    )
                    mask_ref = builder.ref(feathered, "MASK")
                masked = builder.add(
                    "ConditioningSetMask",
                    {
                        "conditioning": builder.ref(encoded, "CONDITIONING"),
                        "mask": mask_ref,
                        "strength": region.strength,
                        "set_cond_area": "default",
                    },
                    title=f"Regional Condition {region.character_id}",
                )
                merged = builder.add(
                    "ConditioningCombine",
                    {"conditioning_1": combined, "conditioning_2": builder.ref(masked, "CONDITIONING")},
                    title=f"Combine {index}",
                )
                combined = builder.ref(merged, "CONDITIONING")
            positive_ref = combined
        for index, guide in enumerate(controls or []):
            image = builder.add("LoadImage", {"image": guide.image_name}, title=f"Control Image {guide.kind.value}")
            model_name = guide.model_name or self.config.controlnet_models.get(guide.kind.value) or DEFAULT_CONTROLNET_MODELS[guide.kind.value]
            control = builder.add("ControlNetLoader", {"control_net_name": model_name}, title=f"ControlNet {guide.kind.value}")
            applied = builder.add(
                "ControlNetApplyAdvanced",
                {
                    "positive": positive_ref,
                    "negative": negative_ref,
                    "control_net": builder.ref(control, "CONTROL_NET"),
                    "image": builder.ref(image, "IMAGE"),
                    "strength": guide.strength,
                    "start_percent": guide.start_percent,
                    "end_percent": guide.end_percent,
                },
                title=f"Apply ControlNet #{index + 1} ({guide.kind.value})",
            )
            positive_ref = builder.ref(applied, "positive")
            negative_ref = builder.ref(applied, "negative")
        cfg_value = recipe.sampler.cfg
        if recipe.profile.get("guidance_node"):
            guidance = builder.add(
                "FluxGuidance", {"conditioning": positive_ref, "guidance": cfg_value if cfg_value > 1.5 else 3.5}, title="Flux Guidance"
            )
            positive_ref = builder.ref(guidance, "CONDITIONING")
            cfg_value = 1.0
        latent = builder.add(recipe.profile["latent_node"], {"width": width, "height": height, "batch_size": 1}, title="Latent")
        sampler = builder.add(
            "KSampler",
            {
                "model": stack.model,
                "seed": seed,
                "steps": recipe.sampler.steps,
                "cfg": cfg_value,
                "sampler_name": recipe.sampler.sampler_name,
                "scheduler": recipe.sampler.scheduler,
                "positive": positive_ref,
                "negative": negative_ref,
                "latent_image": builder.ref(latent, "LATENT"),
                "denoise": recipe.sampler.denoise,
            },
            title="KSampler",
        )
        decoded = builder.add("VAEDecode", {"samples": builder.ref(sampler, "LATENT"), "vae": stack.vae}, title="VAE Decode")
        prefix = f"{self.config.filename_prefix}/{storyboard.id}/p{panel.page_index:02d}_k{panel.index_in_page:02d}"
        builder.add("SaveImage", {"images": builder.ref(decoded, "IMAGE"), "filename_prefix": prefix}, title=f"Save {panel.id}")
        prompt = builder.to_prompt()
        problems = validate_prompt(prompt)
        if problems:
            raise ValueError(f"コマ {panel.id} のワークフロー検証に失敗しました: {'; '.join(problems)}")
        crop: Literal["full", "right_half", "left_half"] = "full"
        if panel.spread_role == "main":
            crop = "right_half"
        elif panel.spread_role == "continuation":
            crop = "left_half"
        return CompiledPanelWorkflow(
            panel_id=panel.id,
            page_index=panel.page_index,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            regions=regions,
            positive_text=positive_text,
            negative_text=negative_text,
            crop=crop,
            filename_prefix=prefix,
            warnings=warnings,
        )

    def compile_storyboard(
        self, storyboard: Storyboard, controls_by_panel: dict[str, list[ControlGuide]] | None = None
    ) -> list[CompiledPanelWorkflow]:
        controls_by_panel = controls_by_panel or {}
        return [
            self.compile_panel(storyboard, panel, controls_by_panel.get(panel.id))
            for panel in storyboard.panels_in_reading_order()
        ]
