#!/usr/bin/env python3
"""ComfyUI 実機の互換性チェック（ノード有無・入力名の適合）と実生成テスト。

使い方（リポジトリ直下で仮想環境を有効化した状態、ComfyUI 起動済み）:
  python local_tools/check_comfy.py                                   # 接続確認と利用可能モデル一覧
  python local_tools/check_comfy.py --checkpoint <ckpt名>              # ワークフロー互換性チェック
  python local_tools/check_comfy.py --checkpoint <ckpt名> --generate   # 実際に画像を生成
  python local_tools/check_comfy.py --checkpoint <A> --checkpoint-b <B> # 階層マージノードの互換性も確認
FLUX / ANIMA は --architecture flux|anima と --text-encoder・--vae（・--text-encoder-type）を指定する。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.manga_pipeline.comfy_client import ComfyClient, ComfyProgressEvent  # noqa: E402
from core.manga_pipeline.comfy_compiler import ComfyWorkflowCompiler  # noqa: E402
from core.manga_pipeline.storyboard_generator import StoryboardConfig, StoryboardGenerator  # noqa: E402
from core.structure_engine.samples import sample_structure  # noqa: E402
from core.style_alchemy.comfy_export import (  # noqa: E402
    adapt_prompt_to_object_info,
    build_txt2img_workflow,
    validate_prompt,
)
from core.style_alchemy.schemas import (  # noqa: E402
    BlockMergeRecipe,
    CheckpointRef,
    LoRASpec,
    ModelArchitecture,
    StyleRecipe,
)

MODEL_FIELDS = (
    ("CheckpointLoaderSimple", "ckpt_name", "チェックポイント（models/checkpoints）"),
    ("UNETLoader", "unet_name", "拡散モデル（models/diffusion_models・unet）"),
    ("LoraLoader", "lora_name", "LoRA（models/loras）"),
    ("VAELoader", "vae_name", "VAE（models/vae）"),
    ("CLIPLoader", "clip_name", "テキストエンコーダ（models/text_encoders・clip）"),
    ("ControlNetLoader", "control_net_name", "ControlNet（models/controlnet）"),
)


def options_of(info: dict[str, Any], node: str, field: str) -> list[str]:
    """object_info から選択肢一覧を取り出す（旧形式 [[...], {...}] と新形式 ["COMBO", {"options": [...]}] の両対応）。"""
    spec = info.get(node, {}).get("input", {})
    for section in ("required", "optional"):
        value = spec.get(section, {}).get(field)
        if not isinstance(value, list) or not value:
            continue
        if isinstance(value[0], list):
            return [str(item) for item in value[0]]
        if value[0] == "COMBO" and len(value) > 1 and isinstance(value[1], dict):
            return [str(item) for item in value[1].get("options", [])]
    return []


def build_recipe(args: argparse.Namespace) -> StyleRecipe:
    arch = ModelArchitecture(args.architecture)
    base = CheckpointRef(name=args.checkpoint, architecture=arch)
    merge = None
    if args.checkpoint_b:
        merge = BlockMergeRecipe(
            architecture=arch,
            model_a=base,
            model_b=CheckpointRef(name=args.checkpoint_b, architecture=arch),
            default_weight=0.3,
        )
    return StyleRecipe(
        name="local-compat-check",
        architecture=arch,
        base_checkpoint=base,
        merge=merge,
        loras=[LoRASpec(name=name, strength_model=0.8, strength_clip=0.8) for name in args.lora],
        vae=args.vae,
        text_encoders=list(args.text_encoder),
        text_encoder_type=args.text_encoder_type,
        style_tokens=["monochrome", "manga", "lineart", "screentone"],
        negative_tokens=["color", "lowres", "blurry"],
    )


def progress_printer(event: ComfyProgressEvent) -> None:
    if event.type == "progress" and event.max:
        print(f"      進捗 {event.value}/{event.max}", end="\r", flush=True)
    elif event.type in {"execution_start", "execution_success", "execution_error", "execution_interrupted"}:
        print(f"\n      {event.type}" if event.type != "execution_start" else f"      {event.type}")


def view_url(base_url: str, image: Any) -> str:
    return f"{base_url.rstrip('/')}/view?filename={quote(image.filename)}&subfolder={quote(image.subfolder)}&type={quote(image.type)}"


async def main_async(args: argparse.Namespace) -> int:
    client = ComfyClient(args.url, timeout_s=args.timeout)
    stats = await client.system_stats()
    devices = [f"{d.get('name')} ({d.get('type')})" for d in stats.get("devices", [])]
    print(f"接続OK: {args.url} / デバイス: {', '.join(devices) or '不明'}")
    info = await client.get_object_info()
    print(f"登録ノード数: {len(info)}")
    for node, field, label in MODEL_FIELDS:
        options = options_of(info, node, field)
        shown = ", ".join(options[:12]) + (" …" if len(options) > 12 else "")
        print(f"  {label}: {len(options)} 件 {shown}")
    if not args.checkpoint:
        print("\n--checkpoint に上の一覧の実在ファイル名を指定すると、ワークフロー互換性チェックを行います。")
        return 0

    recipe = build_recipe(args)
    txt2img = build_txt2img_workflow(
        recipe, args.prompt, args.negative, seed=args.seed, width=args.width, height=args.height, filename_prefix="caf/compat"
    )
    storyboard = StoryboardGenerator(StoryboardConfig(target_pages=8, seed=1)).generate_validated_sync(
        sample_structure("reversal"), theme={"THEME": "失われた故郷"}
    )
    compiled = ComfyWorkflowCompiler(recipe).compile_storyboard(storyboard)[: max(0, args.panels)]
    workflows: list[tuple[str, dict[str, Any]]] = [("txt2img", txt2img)] + [(f"ネームのコマ {c.panel_id}", c.prompt) for c in compiled]

    blocking = 0
    adapted_workflows: list[tuple[str, dict[str, Any]]] = []
    for label, prompt in workflows:
        problems = validate_prompt(prompt)
        adapted, warnings = adapt_prompt_to_object_info(prompt, info)
        missing = sorted({node["class_type"] for node in adapted.values() if node["class_type"] not in info})
        blocking += len(problems) + len(missing)
        print(f"\n[{label}] ノード {len(prompt)} 個 / 構造エラー {len(problems)} / 未導入ノード {missing or 'なし'}")
        for warning in warnings:
            print(f"   - {warning}")
        adapted_workflows.append((label, adapted))

    if blocking:
        print("\n未導入ノードまたは構造エラーがあります。カスタムノードの導入・モデル名を確認してから --generate してください。")
        return 1
    if not args.generate:
        print("\n互換性チェック合格。--generate で実際に生成できます。")
        return 0
    failures = 0
    for label, adapted in adapted_workflows[: 1 + max(0, args.generate_panels)]:
        print(f"\n[{label}] を ComfyUI に投入します…")
        result = await client.run_workflow(adapted, on_progress=progress_printer)
        print(f"      結果: {result.status}（{result.duration_s:.1f} 秒）{result.error or ''}")
        for images in result.outputs.values():
            for image in images:
                print(f"      出力: {image.subfolder}/{image.filename} → {view_url(args.url, image)}")
        failures += 0 if result.status == "success" else 1
    return 0 if failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ComfyUI 実機互換性チェック")
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--architecture", choices=[a.value for a in ModelArchitecture], default="sdxl")
    parser.add_argument("--checkpoint", help="ベースのチェックポイント／拡散モデルのファイル名")
    parser.add_argument("--checkpoint-b", help="階層マージ相手のファイル名（マージノード互換性の確認用）")
    parser.add_argument("--lora", action="append", default=[], help="LoRA ファイル名（複数指定可）")
    parser.add_argument("--vae", default=None)
    parser.add_argument("--text-encoder", action="append", default=[], help="FLUX/ANIMA 用テキストエンコーダ")
    parser.add_argument("--text-encoder-type", default=None, help="CLIPLoader の type（ANIMA 等で必要に応じて）")
    parser.add_argument("--prompt", default="1girl, standing, city street, dramatic lighting")
    parser.add_argument("--negative", default="lowres, bad anatomy, watermark")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--panels", type=int, default=2, help="互換性チェックするネームのコマ数")
    parser.add_argument("--generate", action="store_true", help="実際に生成する")
    parser.add_argument("--generate-panels", type=int, default=1, help="txt2img に加えて生成するコマ数")
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
