#!/usr/bin/env python3
"""Style Alchemy Arena 用に、実際の ComfyUI 生成画像つきの候補を一括登録する。

ラテン超方格でマージ比率・LoRA 重み・絵柄パラメータを散らした N 件のレシピを作り、
全候補を同一シード・同一プロンプトで生成して（差がレシピだけに由来するように）、
生成画像の /view URL を image_ref として Arena に登録する。

使い方（スタジオと ComfyUI を起動した状態）:
  python local_tools/generate_arena_candidates.py --model-a <ckptA> --model-b <ckptB> --lora <style.safetensors> -n 8
  python local_tools/generate_arena_candidates.py --model-a <ckpt> -n 6     # マージ相手が無い場合は同一モデル
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.manga_pipeline.comfy_client import ComfyClient  # noqa: E402
from core.style_alchemy.comfy_export import adapt_prompt_to_object_info, build_txt2img_workflow  # noqa: E402
from core.style_alchemy.model_optimizer import ParameterSpace, RecipeExporter  # noqa: E402
from core.style_alchemy.schemas import CheckpointRef, LoRASpec, ModelArchitecture  # noqa: E402


async def main_async(args: argparse.Namespace) -> int:
    arch = ModelArchitecture(args.architecture)
    lora_specs = [LoRASpec(name=name) for name in args.lora]
    space = ParameterSpace.for_style_lab(arch, lora_names=[Path(name).stem for name in args.lora], grouping="grouped")
    samples = space.sample_lhs(args.n, seed=args.seed)
    exporter = RecipeExporter()
    comfy = ComfyClient(args.comfy, timeout_s=args.timeout)
    info = await comfy.get_object_info()
    model_a = CheckpointRef(name=args.model_a, architecture=arch)
    model_b = CheckpointRef(name=args.model_b or args.model_a, architecture=arch)
    registered = 0
    async with httpx.AsyncClient(base_url=args.studio, timeout=120.0, trust_env=False) as studio:
        health = await studio.get("/api/health")
        health.raise_for_status()
        for index, params in enumerate(samples, start=1):
            recipe = exporter.params_to_recipe(
                params, architecture=arch, model_a=model_a, model_b=model_b, loras=lora_specs, name=f"arena-{index:02d}"
            )
            workflow = build_txt2img_workflow(
                recipe, args.prompt, args.negative, seed=args.image_seed, width=args.width, height=args.height,
                filename_prefix="caf/arena",
            )
            adapted, warnings = adapt_prompt_to_object_info(workflow, info)
            for warning in warnings:
                print(f"  注意: {warning}")
            print(f"[{index}/{len(samples)}] 生成中…")
            result = await comfy.run_workflow(adapted)
            images = [image for group in result.outputs.values() for image in group]
            if result.status != "success" or not images:
                print(f"  生成失敗: {result.status} {result.error or ''}")
                continue
            image = images[0]
            image_ref = (
                f"{args.comfy.rstrip('/')}/view?filename={quote(image.filename)}"
                f"&subfolder={quote(image.subfolder)}&type={quote(image.type)}"
            )
            response = await studio.post(
                "/api/style/candidates",
                json={
                    "label": f"実画像候補 {index:02d}",
                    "params": params,
                    "recipe": recipe.model_dump(mode="json"),
                    "image_ref": image_ref,
                    "source": "comfyui",
                },
            )
            response.raise_for_status()
            registered += 1
            print(f"  登録: {image_ref}")
    print(f"\n{registered} 件を Arena に登録しました。Style Alchemy Arena の A/B 判定で比較できます。")
    return 0 if registered == len(samples) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arena 用の実画像候補を一括生成・登録する")
    parser.add_argument("--studio", default="http://127.0.0.1:8765")
    parser.add_argument("--comfy", default="http://127.0.0.1:8188")
    parser.add_argument("--architecture", choices=[a.value for a in ModelArchitecture], default="sdxl")
    parser.add_argument("--model-a", required=True, help="ベースのチェックポイント名")
    parser.add_argument("--model-b", default=None, help="マージ相手のチェックポイント名（省略時は model-a）")
    parser.add_argument("--lora", action="append", default=[], help="LoRA ファイル名（複数指定可）")
    parser.add_argument("-n", type=int, default=8, help="候補数")
    parser.add_argument("--seed", type=int, default=0, help="パラメータ標本化のシード")
    parser.add_argument("--image-seed", type=int, default=20261104, help="全候補で共通の生成シード")
    parser.add_argument("--prompt", default="1girl, upper body, looking at viewer, city street at dusk")
    parser.add_argument("--negative", default="lowres, bad anatomy, watermark, text")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
