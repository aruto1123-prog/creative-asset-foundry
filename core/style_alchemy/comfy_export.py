"""ComfyUI API 形式ノードグラフの構築基盤。

ComfyUI の ``POST /prompt`` が受け付ける API 形式::

    {"<node_id>": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": 1, ...}, "_meta": {"title": "..."}}}

リンクは ``[node_id, output_index]``。本モジュールは

- ``ComfyGraphBuilder``: ノード追加・出力名による参照解決・検証（未定義参照/出力範囲/循環）
- ``build_model_stack``: StyleRecipe → ローダー → 階層ブロックマージ → LoRA チェーン → CLIP Skip
- ``build_txt2img_workflow``: 単体検証用 txt2img ワークフロー
- ``adapt_prompt_to_object_info``: 実機 ``/object_info`` に基づく入力名の差異吸収と未導入ノードのフォールバック

を提供する。ComfyUI の ``ModelMerge*`` 系の比率は model1 の重み（1.0 で純 model1）なので、
本システムの「model_b の比率 w」は ``1 - w`` に変換して出力する。
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field

from core.style_alchemy.schemas import (
    ARCHITECTURE_BLOCKS,
    BlockMergeRecipe,
    MergeMethod,
    ModelArchitecture,
    StyleRecipe,
)

__all__ = [
    "NODE_OUTPUTS",
    "LBW_NODE_CLASS",
    "ComfyGraphBuilder",
    "ModelStackRefs",
    "comfy_merge_inputs",
    "build_model_stack",
    "build_txt2img_workflow",
    "adapt_prompt_to_object_info",
    "snap_resolution",
]

LBW_NODE_CLASS = "LoraLoaderBlockWeight //Inspire"

NODE_OUTPUTS: dict[str, list[str]] = {
    "CheckpointLoaderSimple": ["MODEL", "CLIP", "VAE"],
    "UNETLoader": ["MODEL"],
    "DualCLIPLoader": ["CLIP"],
    "CLIPLoader": ["CLIP"],
    "VAELoader": ["VAE"],
    "ModelMergeSimple": ["MODEL"],
    "ModelMergeBlocks": ["MODEL"],
    "ModelMergeSDXL": ["MODEL"],
    "ModelMergeSD1": ["MODEL"],
    "ModelMergeFlux1": ["MODEL"],
    "ModelMergeCosmosPredict2_2B": ["MODEL"],
    "ModelMergeSubtract": ["MODEL"],
    "ModelMergeAdd": ["MODEL"],
    "CLIPMergeSimple": ["CLIP"],
    "CLIPMergeSubtract": ["CLIP"],
    "CLIPMergeAdd": ["CLIP"],
    "LoraLoader": ["MODEL", "CLIP"],
    "LoraLoaderModelOnly": ["MODEL"],
    LBW_NODE_CLASS: ["MODEL", "CLIP", "STRING"],
    "CLIPSetLastLayer": ["CLIP"],
    "CLIPTextEncode": ["CONDITIONING"],
    "FluxGuidance": ["CONDITIONING"],
    "ConditioningCombine": ["CONDITIONING"],
    "ConditioningSetMask": ["CONDITIONING"],
    "ConditioningSetArea": ["CONDITIONING"],
    "ConditioningSetAreaStrength": ["CONDITIONING"],
    "ConditioningZeroOut": ["CONDITIONING"],
    "SolidMask": ["MASK"],
    "MaskComposite": ["MASK"],
    "FeatherMask": ["MASK"],
    "InvertMask": ["MASK"],
    "ControlNetLoader": ["CONTROL_NET"],
    "ControlNetApplyAdvanced": ["positive", "negative"],
    "LoadImage": ["IMAGE", "MASK"],
    "EmptyLatentImage": ["LATENT"],
    "EmptySD3LatentImage": ["LATENT"],
    "KSampler": ["LATENT"],
    "KSamplerAdvanced": ["LATENT"],
    "VAEDecode": ["IMAGE"],
    "VAEEncode": ["LATENT"],
    "ImageScale": ["IMAGE"],
    "SaveImage": [],
    "PreviewImage": [],
}

_MERGE_NODE_CLASSES = {
    "ModelMergeSDXL",
    "ModelMergeSD1",
    "ModelMergeFlux1",
    "ModelMergeCosmosPredict2_2B",
    "ModelMergeBlocks",
}


class ComfyGraphBuilder:
    """ComfyUI API 形式のノードグラフを組み立てるビルダー。"""

    def __init__(self, start_id: int = 1) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._next_id = start_id

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def add(
        self,
        class_type: str,
        inputs: dict[str, Any],
        title: str | None = None,
        node_id: str | None = None,
    ) -> str:
        """ノードを追加し、そのIDを返す。"""
        if node_id is None:
            while str(self._next_id) in self._nodes:
                self._next_id += 1
            node_id = str(self._next_id)
            self._next_id += 1
        elif node_id in self._nodes:
            raise ValueError(f"ノードIDが重複しています: {node_id}")
        node: dict[str, Any] = {"class_type": class_type, "inputs": dict(inputs)}
        if title:
            node["_meta"] = {"title": title}
        self._nodes[node_id] = node
        return node_id

    def node(self, node_id: str) -> dict[str, Any]:
        return self._nodes[node_id]

    def nodes(self) -> dict[str, dict[str, Any]]:
        return self._nodes

    def set_input(self, node_id: str, name: str, value: Any) -> None:
        self._nodes[node_id]["inputs"][name] = value

    def ref(self, node_id: str, output: str | int = 0) -> list[Any]:
        """出力名または出力インデックスからリンク ``[node_id, index]`` を作る。"""
        if node_id not in self._nodes:
            raise KeyError(f"未定義ノードへの参照です: {node_id}")
        if isinstance(output, int):
            return [node_id, output]
        class_type = self._nodes[node_id]["class_type"]
        outputs = NODE_OUTPUTS.get(class_type)
        if outputs is None:
            raise KeyError(f"出力名を解決できない未知ノードです: {class_type}（インデックスで指定してください）")
        if output not in outputs:
            raise KeyError(f"{class_type} に出力 {output} はありません（候補: {outputs}）")
        return [node_id, outputs.index(output)]

    def find(self, class_type: str) -> list[str]:
        return [node_id for node_id, node in self._nodes.items() if node["class_type"] == class_type]

    @staticmethod
    def is_link(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
        )

    def validate(self) -> list[str]:
        """未定義参照・出力インデックス範囲外・循環参照を検出して問題一覧を返す。"""
        problems: list[str] = []
        for node_id, node in self._nodes.items():
            for name, value in node["inputs"].items():
                if not self.is_link(value):
                    continue
                target_id, index = value[0], value[1]
                if target_id not in self._nodes:
                    problems.append(f"ノード {node_id}.{name} が未定義ノード {target_id} を参照しています")
                    continue
                outputs = NODE_OUTPUTS.get(self._nodes[target_id]["class_type"])
                if outputs is not None and not 0 <= index < len(outputs):
                    problems.append(
                        f"ノード {node_id}.{name} が {target_id}({self._nodes[target_id]['class_type']}) の"
                        f"存在しない出力 {index} を参照しています"
                    )
        state: dict[str, int] = {}

        def _visit(current: str, trail: list[str]) -> None:
            state[current] = 1
            for value in self._nodes[current]["inputs"].values():
                if not self.is_link(value) or value[0] not in self._nodes:
                    continue
                dependency = value[0]
                if state.get(dependency) == 1:
                    problems.append("循環参照: " + " -> ".join([*trail, current, dependency]))
                elif state.get(dependency) is None:
                    _visit(dependency, [*trail, current])
            state[current] = 2

        for node_id in self._nodes:
            if state.get(node_id) is None:
                _visit(node_id, [])
        return problems

    def to_prompt(self) -> dict[str, Any]:
        return copy.deepcopy(self._nodes)


class ModelStackRefs(BaseModel):
    """モデルスタック構築結果（後続ノードが接続する MODEL / CLIP / VAE のリンク）。"""

    model: list[Any]
    clip: list[Any]
    vae: list[Any]
    notes: list[str] = Field(default_factory=list)


def _ratio(weight_b: float) -> float:
    """model_b 比率 → ComfyUI ratio（model1 の重み）。"""
    return round(1.0 - float(weight_b), 6)


def comfy_merge_inputs(merge: BlockMergeRecipe) -> tuple[str, dict[str, float], list[str]]:
    """BlockMergeRecipe を ComfyUI のアーキテクチャ別マージノード入力へ変換する。"""
    arch = merge.architecture
    w = merge.weight_for
    notes: list[str] = []
    inputs: dict[str, float] = {}
    if arch == ModelArchitecture.SDXL:
        node = "ModelMergeSDXL"
        inputs["time_embed."] = _ratio(w("BASE"))
        inputs["label_emb."] = _ratio(w("BASE"))
        for i in range(9):
            inputs[f"input_blocks.{i}"] = _ratio(w(f"IN{i:02d}"))
        for i in range(3):
            inputs[f"middle_block.{i}"] = _ratio(w("MID"))
        for i in range(9):
            inputs[f"output_blocks.{i}"] = _ratio(w(f"OUT{i:02d}"))
        inputs["out."] = _ratio(w("BASE"))
    elif arch == ModelArchitecture.SD15:
        node = "ModelMergeSD1"
        inputs["time_embed."] = _ratio(w("BASE"))
        inputs["label_emb."] = _ratio(w("BASE"))
        for i in range(12):
            inputs[f"input_blocks.{i}."] = _ratio(w(f"IN{i:02d}"))
        for i in range(3):
            inputs[f"middle_block.{i}."] = _ratio(w("MID"))
        for i in range(12):
            inputs[f"output_blocks.{i}."] = _ratio(w(f"OUT{i:02d}"))
        inputs["out."] = _ratio(w("BASE"))
    elif arch == ModelArchitecture.FLUX:
        node = "ModelMergeFlux1"
        for key in ("img_in.", "time_in.", "guidance_in", "vector_in.", "txt_in."):
            inputs[key] = _ratio(w("BASE"))
        for i in range(19):
            inputs[f"double_blocks.{i}."] = _ratio(w(f"DB{i:02d}"))
        for i in range(38):
            inputs[f"single_blocks.{i}."] = _ratio(w(f"SB{i:02d}"))
        inputs["final_layer."] = _ratio(w("BASE"))
    else:
        node = "ModelMergeCosmosPredict2_2B"
        for key in ("pos_embedder.", "x_embedder.", "t_embedder.", "t_embedding_norm."):
            inputs[key] = _ratio(w("BASE"))
        depth = len(ARCHITECTURE_BLOCKS[ModelArchitecture.ANIMA]) - 1
        for i in range(depth):
            inputs[f"blocks.{i}."] = _ratio(w(f"B{i:02d}"))
        inputs["final_layer."] = _ratio(w("BASE"))
        notes.append(
            "ANIMA は Cosmos-Predict2 2B 系 DiT を想定した per-block マージノードを使用。"
            "実機に当該ノードが無い場合は adapt_prompt_to_object_info が ModelMergeSimple（平均比率）へ自動置換する。"
        )
    return node, inputs, notes


def _ckpt_loader(builder: ComfyGraphBuilder, name: str, title: str) -> str:
    return builder.add("CheckpointLoaderSimple", {"ckpt_name": name}, title=title)


def _unet_loader(builder: ComfyGraphBuilder, name: str, title: str) -> str:
    return builder.add("UNETLoader", {"unet_name": name, "weight_dtype": "default"}, title=title)


def build_model_stack(builder: ComfyGraphBuilder, recipe: StyleRecipe) -> ModelStackRefs:
    """StyleRecipe からモデルスタック（ローダー→マージ→LoRA→CLIP Skip）を構築する。"""
    arch = recipe.architecture
    profile = recipe.profile
    notes: list[str] = []
    merge = recipe.merge

    if profile["loader"] == "checkpoint":
        primary_name = merge.model_a.name if merge is not None else recipe.base_checkpoint.name
        loader_a = _ckpt_loader(builder, primary_name, "Checkpoint A")
        model = builder.ref(loader_a, "MODEL")
        clip = builder.ref(loader_a, "CLIP")
        vae = builder.ref(loader_a, "VAE")
        if merge is not None:
            loader_b = _ckpt_loader(builder, merge.model_b.name, "Checkpoint B")
            target_model = builder.ref(loader_b, "MODEL")
            target_clip = builder.ref(loader_b, "CLIP")
            if merge.method == MergeMethod.ADD_DIFFERENCE and merge.model_c is not None:
                loader_c = _ckpt_loader(builder, merge.model_c.name, "Checkpoint C (difference base)")
                diff = builder.add(
                    "ModelMergeSubtract",
                    {"model1": target_model, "model2": builder.ref(loader_c, "MODEL"), "multiplier": 1.0},
                    title="B - C",
                )
                added = builder.add(
                    "ModelMergeAdd", {"model1": model, "model2": builder.ref(diff, "MODEL")}, title="A + (B - C)"
                )
                target_model = builder.ref(added, "MODEL")
                clip_diff = builder.add(
                    "CLIPMergeSubtract",
                    {"clip1": target_clip, "clip2": builder.ref(loader_c, "CLIP"), "multiplier": 1.0},
                    title="CLIP B - C",
                )
                clip_added = builder.add(
                    "CLIPMergeAdd", {"clip1": clip, "clip2": builder.ref(clip_diff, "CLIP")}, title="CLIP A + (B - C)"
                )
                target_clip = builder.ref(clip_added, "CLIP")
            node_class, merge_inputs, merge_notes = comfy_merge_inputs(merge)
            notes.extend(merge_notes)
            merged = builder.add(
                node_class,
                {"model1": model, "model2": target_model, **merge_inputs},
                title=f"Block Merge ({arch.value})",
            )
            model = builder.ref(merged, "MODEL")
            base_weight = merge.weight_for("BASE")
            if base_weight > 0.0:
                clip_merge = builder.add(
                    "CLIPMergeSimple",
                    {"clip1": clip, "clip2": target_clip, "ratio": _ratio(base_weight)},
                    title="Text Encoder Merge (BASE)",
                )
                clip = builder.ref(clip_merge, "CLIP")
        if recipe.vae:
            vae_loader = builder.add("VAELoader", {"vae_name": recipe.vae}, title="VAE")
            vae = builder.ref(vae_loader, "VAE")
    else:
        primary_name = merge.model_a.name if merge is not None else recipe.base_checkpoint.name
        unet_a = _unet_loader(builder, primary_name, "Diffusion Model A")
        model = builder.ref(unet_a, "MODEL")
        encoders = recipe.resolved_text_encoders()
        clip_type = recipe.text_encoder_type or profile.get("clip_type", "stable_diffusion")
        if profile.get("clip_loader") == "DualCLIPLoader":
            first = encoders[0] if encoders else "clip_l.safetensors"
            second = encoders[1] if len(encoders) > 1 else "t5xxl_fp16.safetensors"
            clip_loader = builder.add(
                "DualCLIPLoader", {"clip_name1": first, "clip_name2": second, "type": clip_type}, title="Text Encoders"
            )
        else:
            clip_loader = builder.add(
                "CLIPLoader",
                {"clip_name": encoders[0] if encoders else "text_encoder.safetensors", "type": clip_type},
                title="Text Encoder",
            )
        clip = builder.ref(clip_loader, "CLIP")
        vae_loader = builder.add("VAELoader", {"vae_name": recipe.resolved_vae() or "vae.safetensors"}, title="VAE")
        vae = builder.ref(vae_loader, "VAE")
        if merge is not None:
            unet_b = _unet_loader(builder, merge.model_b.name, "Diffusion Model B")
            target_model = builder.ref(unet_b, "MODEL")
            if merge.method == MergeMethod.ADD_DIFFERENCE and merge.model_c is not None:
                unet_c = _unet_loader(builder, merge.model_c.name, "Diffusion Model C (difference base)")
                diff = builder.add(
                    "ModelMergeSubtract",
                    {"model1": target_model, "model2": builder.ref(unet_c, "MODEL"), "multiplier": 1.0},
                    title="B - C",
                )
                added = builder.add(
                    "ModelMergeAdd", {"model1": model, "model2": builder.ref(diff, "MODEL")}, title="A + (B - C)"
                )
                target_model = builder.ref(added, "MODEL")
            node_class, merge_inputs, merge_notes = comfy_merge_inputs(merge)
            notes.extend(merge_notes)
            merged = builder.add(
                node_class,
                {"model1": model, "model2": target_model, **merge_inputs},
                title=f"Block Merge ({arch.value})",
            )
            model = builder.ref(merged, "MODEL")

    for index, lora in enumerate(recipe.loras):
        if lora.has_block_weights():
            node_id = builder.add(
                LBW_NODE_CLASS,
                {
                    "model": model,
                    "clip": clip,
                    "category_filter": "All",
                    "lora_name": lora.name,
                    "strength_model": lora.strength_model,
                    "strength_clip": lora.strength_clip,
                    "inverse": False,
                    "seed": 0,
                    "A": 4.0,
                    "B": 1.0,
                    "preset": "Preset",
                    "block_vector": lora.lbw_string(arch),
                    "bypass": False,
                },
                title=f"LoRA Block Weight #{index + 1}: {lora.name}",
            )
            notes.append(f"LoRA {lora.name} は LBW ノード（ComfyUI-Inspire-Pack）で階層ウェイトを適用")
        else:
            node_id = builder.add(
                "LoraLoader",
                {
                    "model": model,
                    "clip": clip,
                    "lora_name": lora.name,
                    "strength_model": lora.strength_model,
                    "strength_clip": lora.strength_clip,
                },
                title=f"LoRA #{index + 1}: {lora.name}",
            )
        model = builder.ref(node_id, "MODEL")
        clip = builder.ref(node_id, "CLIP")

    if profile["loader"] == "checkpoint" and recipe.sampler.clip_skip > 1:
        skip = builder.add(
            "CLIPSetLastLayer", {"clip": clip, "stop_at_clip_layer": -int(recipe.sampler.clip_skip)}, title="CLIP Skip"
        )
        clip = builder.ref(skip, "CLIP")
    return ModelStackRefs(model=model, clip=clip, vae=vae, notes=notes)


def snap_resolution(width: float, height: float, *, target: int = 1024, multiple: int = 64) -> tuple[int, int]:
    """アスペクト比を保ちつつ総画素数 ≈ target² 、各辺を multiple の倍数へ丸める。"""
    if width <= 0 or height <= 0:
        return target, target
    aspect = float(width) / float(height)
    area = float(target * target)
    out_h = (area / aspect) ** 0.5
    out_w = out_h * aspect
    snapped_w = max(multiple, int(round(out_w / multiple)) * multiple)
    snapped_h = max(multiple, int(round(out_h / multiple)) * multiple)
    return snapped_w, snapped_h


def _join_prompt(*parts: str) -> str:
    return ", ".join(p.strip().strip(",") for p in parts if p and p.strip().strip(","))


def build_txt2img_workflow(
    recipe: StyleRecipe,
    positive: str,
    negative: str,
    *,
    seed: int,
    width: int,
    height: int,
    filename_prefix: str = "caf",
    batch_size: int = 1,
) -> dict[str, Any]:
    """レシピ単体検証用の txt2img ワークフロー（API 形式）。"""
    builder = ComfyGraphBuilder()
    stack = build_model_stack(builder, recipe)
    positive_text = _join_prompt(recipe.positive_prefix(), positive)
    negative_text = _join_prompt(negative, recipe.negative_text())
    pos = builder.add("CLIPTextEncode", {"text": positive_text, "clip": stack.clip}, title="Positive")
    neg = builder.add("CLIPTextEncode", {"text": negative_text, "clip": stack.clip}, title="Negative")
    positive_ref = builder.ref(pos, "CONDITIONING")
    cfg = recipe.sampler.cfg
    if recipe.profile.get("guidance_node"):
        guidance = builder.add(
            "FluxGuidance",
            {"conditioning": positive_ref, "guidance": cfg if cfg > 1.5 else 3.5},
            title="Flux Guidance",
        )
        positive_ref = builder.ref(guidance, "CONDITIONING")
        cfg = 1.0
    latent = builder.add(
        recipe.profile["latent_node"],
        {"width": int(width), "height": int(height), "batch_size": int(batch_size)},
        title="Latent",
    )
    sampler = builder.add(
        "KSampler",
        {
            "model": stack.model,
            "seed": int(seed),
            "steps": recipe.sampler.steps,
            "cfg": cfg,
            "sampler_name": recipe.sampler.sampler_name,
            "scheduler": recipe.sampler.scheduler,
            "positive": positive_ref,
            "negative": builder.ref(neg, "CONDITIONING"),
            "latent_image": builder.ref(latent, "LATENT"),
            "denoise": recipe.sampler.denoise,
        },
        title="KSampler",
    )
    decoded = builder.add(
        "VAEDecode", {"samples": builder.ref(sampler, "LATENT"), "vae": stack.vae}, title="VAE Decode"
    )
    builder.add("SaveImage", {"images": builder.ref(decoded, "IMAGE"), "filename_prefix": filename_prefix}, title="Save")
    problems = builder.validate()
    if problems:
        raise ValueError("ワークフロー検証エラー: " + "; ".join(problems))
    return builder.to_prompt()


def _input_names(spec: dict[str, Any]) -> list[str]:
    names: list[str] = []
    inputs = spec.get("input", {}) if isinstance(spec, dict) else {}
    for section in ("required", "optional"):
        block = inputs.get(section, {})
        if isinstance(block, dict):
            names.extend(block.keys())
    return names


def adapt_prompt_to_object_info(
    prompt: dict[str, Any], object_info: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """実機 ComfyUI の ``/object_info`` に合わせてワークフローを適合させる。

    - 入力名の表記揺れ（末尾ドットの有無）を実機の入力名へ改名する。
    - 実機に存在しないアーキテクチャ別マージノードは ``ModelMergeSimple``（比率平均）へ置換する。
    - 実機に存在しないその他のノードは警告として報告する。
    """
    adapted = copy.deepcopy(prompt)
    warnings: list[str] = []
    for node_id, node in adapted.items():
        class_type = node.get("class_type", "")
        spec = object_info.get(class_type)
        if spec is None:
            if class_type in _MERGE_NODE_CLASSES:
                inputs = node.get("inputs", {})
                ratios = [float(v) for k, v in inputs.items() if k not in ("model1", "model2") and isinstance(v, (int, float))]
                mean_ratio = round(sum(ratios) / len(ratios), 6) if ratios else 0.5
                node["class_type"] = "ModelMergeSimple"
                node["inputs"] = {"model1": inputs.get("model1"), "model2": inputs.get("model2"), "ratio": mean_ratio}
                warnings.append(
                    f"ノード {node_id}: {class_type} が未導入のため ModelMergeSimple(ratio={mean_ratio}) へ置換しました"
                )
            else:
                warnings.append(f"ノード {node_id}: {class_type} は接続先 ComfyUI に存在しません（カスタムノードの導入が必要）")
            continue
        valid_names = _input_names(spec)
        if not valid_names:
            continue
        normalized = {name.rstrip("."): name for name in valid_names}
        renamed: dict[str, Any] = {}
        for key, value in node.get("inputs", {}).items():
            if key in valid_names:
                renamed[key] = value
                continue
            candidate = normalized.get(key.rstrip("."))
            if candidate is not None:
                renamed[candidate] = value
                warnings.append(f"ノード {node_id}: 入力名 {key!r} を {candidate!r} に適合させました")
            else:
                renamed[key] = value
                warnings.append(f"ノード {node_id}: 入力 {key!r} は {class_type} の既知入力にありません")
        node["inputs"] = renamed
    return adapted, warnings
