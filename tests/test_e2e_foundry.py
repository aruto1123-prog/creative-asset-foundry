"""Phase 8: Creative Asset Foundry 一気通貫 E2E 統合テスト。

シード60件ロード → 新規テキストから構造抽出 → シードと構造合成 → 合成構造から漫画ネーム生成＋漫画文法バリデータ通過
→ SVG原稿とComfyUIワークフローJSON出力（モックComfyUIでキュー実行・WebSocket進捗監視）
→ Elo/Bradley-Terry選好学習とベイズ最適化で次のLoRA/マージ重みを算出
→ 一連の研究データから解説記事（一人称「私」・URL完全除去）を生成、までを自動実行する。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from core.config import LLMSettings
from core.db import Database
from core.llm_client import create_llm_client


def _simulate_comparisons(candidates, utility, n: int, seed: int):
    from core.style_alchemy.preference_engine import Comparison

    rng = np.random.default_rng(seed)
    comparisons = []
    for k in range(n):
        i, j = rng.choice(len(candidates), size=2, replace=False)
        diff = utility(candidates[i].params) - utility(candidates[j].params)
        p_left = 1.0 / (1.0 + math.exp(-4.0 * diff))
        outcome = "left" if rng.random() < p_left else "right"
        comparisons.append(
            Comparison(id=f"cmp_{k:03d}", left_id=candidates[i].id, right_id=candidates[j].id, outcome=outcome)
        )
    return comparisons


@pytest.mark.asyncio
async def test_full_foundry_pipeline(tmp_path) -> None:
    from core.asset_publisher.matrix_transpiler import (
        ContentMatrixTranspiler,
        research_log_from_experiment,
        research_log_from_structure,
    )
    from core.asset_publisher.sanitizer import detect_first_person_variants, find_urls
    from core.asset_publisher.schemas import ContentAxis
    from core.asset_publisher.writer_pipeline import ArticleWriterPipeline
    from core.manga_pipeline.comfy_client import ComfyClient
    from core.manga_pipeline.comfy_compiler import ComfyWorkflowCompiler
    from core.manga_pipeline.comfy_mock_server import MockComfyServer
    from core.manga_pipeline.layout_renderer import RenderOptions, SVGLayoutRenderer
    from core.manga_pipeline.storyboard_generator import StoryboardConfig, StoryboardGenerator
    from core.structure_engine.composer import merge_structures_with_report
    from core.structure_engine.consistency import audit_structure
    from core.structure_engine.extractor import StructureExtractor, find_flavor_leaks
    from core.structure_engine.repository import StructureRepository
    from core.structure_engine.samples import sample_text_ja
    from core.structure_engine.seeds.loader import seed_database
    from core.style_alchemy.comfy_export import validate_prompt
    from core.style_alchemy.model_optimizer import BayesianStyleOptimizer, ParameterSpace, RecipeExporter
    from core.style_alchemy.preference_engine import PreferenceAnalyzer, StyleCandidate, rank_candidates
    from core.style_alchemy.schemas import CheckpointRef, LoRASpec, ModelArchitecture

    db = Database(tmp_path / "e2e_foundry.sqlite3")
    client = create_llm_client(LLMSettings(provider="mock"), db=db)

    # 1. 初期シード60パターンをDBにロード
    repo = StructureRepository(db)
    seed_report = seed_database(repo)
    assert seed_report.total == 60
    assert repo.count() == 60

    # 2. 新規テキストから脱フレーバー構造抽出（4段階ハーネス経由）
    extraction = await StructureExtractor(client).extract(sample_text_ja(), source_kind="synopsis")
    extracted = extraction.structure
    assert extraction.used_llm
    assert extraction.harness_run_id
    assert not find_flavor_leaks(extracted, extraction.deflavor.tokens)
    joined = "\n".join(extracted.text_fields())
    for proper_noun in ("レオン", "グレゴール", "アルカディア", "ミスト", "セリス"):
        assert proper_noun not in joined
    repo.save(extracted)

    # 3. 既存シードと構造合成（Aの対立トポロジー × Bの感情・情報開示波形）
    narrative_seeds = repo.list(category="narrative_archetype")
    assert len(narrative_seeds) == 20
    seed_structure = repo.get(narrative_seeds[0].id)
    assert seed_structure is not None
    merged, merge_report = merge_structures_with_report(extracted, seed_structure, weight_topology=0.65, weight_wave=0.7)
    assert merged.metadata.parent_ids == [extracted.id, seed_structure.id]
    assert not [issue for issue in audit_structure(merged) if issue.severity == "error"]
    assert 0.0 <= merge_report.dtw_alignment_cost
    repo.save(merged)

    # 4. 合成構造から漫画ネームを生成し、漫画文法バリデータをパスさせる
    generator = StoryboardGenerator(StoryboardConfig(target_pages=12, seed=11), llm_client=client)
    storyboard = await generator.generate_validated(merged, theme={"THEME": "失われた居場所"})
    assert storyboard.grammar is not None
    assert storyboard.grammar.passed
    assert storyboard.grammar.score >= 80.0
    assert storyboard.page_count >= 4
    panels = storyboard.panels_in_reading_order()
    assert len(panels) >= storyboard.page_count

    # 5. SVG原稿のレンダリング（視線誘導オーバーレイ付き）
    renderer = SVGLayoutRenderer(RenderOptions(show_gaze_vectors=True))
    svgs = renderer.render_all(storyboard)
    assert len(svgs) == storyboard.page_count
    for svg in svgs:
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")
    layers = renderer.layers(storyboard)
    assert len(layers["pages"]) == storyboard.page_count

    # 6. Elo / Bradley-Terry 選好学習 → 美的選好ベクトル → ベイズ最適化 → レシピ
    space = ParameterSpace.for_style_lab(ModelArchitecture.SDXL, lora_names=["ink_style"], grouping="grouped")
    candidates = [
        StyleCandidate(id=f"cand_{i:02d}", label=f"候補{i:02d}", params=params)
        for i, params in enumerate(space.sample_lhs(12, seed=5))
    ]

    def utility(params: dict[str, float]) -> float:
        return 2.0 * params["style.black_fill_ratio"] - 1.5 * params["style.line_taper"]

    comparisons = _simulate_comparisons(candidates, utility, n=160, seed=21)
    ratings = rank_candidates(candidates, comparisons)
    assert [r.rank for r in ratings] == list(range(1, len(candidates) + 1))
    vector = PreferenceAnalyzer(n_bootstrap=40, seed=0).analyze(candidates, comparisons)
    features = {f.name: f for f in vector.features}
    assert features["style.black_fill_ratio"].direction == "prefer_higher"
    assert features["style.line_taper"].direction == "prefer_lower"

    optimizer = BayesianStyleOptimizer(space, seed=0)
    by_id = {c.id: c for c in candidates}
    for rating in ratings:
        optimizer.add_observation(by_id[rating.candidate_id].params, rating.bt_elo_scale, rating.candidate_id)
    suggestions = optimizer.suggest(n=3, strategy="hybrid")
    assert len(suggestions) == 3
    bounds = {dim.name: (dim.low, dim.high) for dim in space.dims}
    for suggestion in suggestions:
        for name, value in suggestion.params.items():
            low, high = bounds[name]
            assert low - 1e-9 <= value <= high + 1e-9

    recipe = RecipeExporter().params_to_recipe(
        suggestions[0].params,
        architecture=ModelArchitecture.SDXL,
        model_a=CheckpointRef(name="caf_base_a.safetensors"),
        model_b=CheckpointRef(name="caf_base_b.safetensors"),
        loras=[LoRASpec(name="ink_style.safetensors", trigger_words=["caf_ink"])],
        name="E2E 最適レシピ",
    )
    assert recipe.merge is not None
    assert recipe.loras and recipe.loras[0].name == "ink_style.safetensors"

    # 7. ComfyUIワークフローJSONのコンパイルとモックComfyUIでのキュー実行・進捗監視
    compiled = ComfyWorkflowCompiler(recipe).compile_storyboard(storyboard)
    assert len(compiled) == len(panels)
    for workflow in compiled:
        assert validate_prompt(workflow.prompt) == []
        assert workflow.width % 8 == 0 and workflow.height % 8 == 0
    events = []
    with MockComfyServer() as server:
        comfy = ComfyClient(server.base_url)
        execution = await comfy.run_workflow(compiled[0].prompt, on_progress=events.append)
    assert execution.status == "success"
    assert execution.outputs
    assert any(event.type == "progress" for event in events)

    # 8. 研究データから3軸企画 → 5段階執筆（一人称「私」統一・URL完全除去）
    log = research_log_from_structure(merged)
    experiment_log = research_log_from_experiment(
        {
            "preference_vector": vector.model_dump(mode="json"),
            "ratings": [r.model_dump(mode="json") for r in ratings[:5]],
            "suggestions": [s.model_dump(mode="json") for s in suggestions],
        }
    )
    assert experiment_log.findings
    matrix = await ContentMatrixTranspiler(client).expand(log)
    assert {plan.axis for plan in matrix.plans} == set(ContentAxis)
    article = await ArticleWriterPipeline(client).run(matrix.plans[0], log)
    assert article.final_score >= 80.0
    assert article.sanitization.first_person_unified
    assert article.sanitization.url_free
    assert detect_first_person_variants(article.markdown) == []
    assert find_urls(article.markdown) == []
    assert "私" in article.markdown
    assert "```mermaid" in article.markdown

    # 9. テレメトリ（トークン・レイテンシ・思考ログ）が記録されている
    stats = client.telemetry.stats()
    assert stats["totals"]["calls"] > 0
    assert stats["totals"]["prompt_tokens"] > 0
