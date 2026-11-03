"""Phase 3: 極限絵柄開発・モデル調合＆LoRA研究ラボ（Style Alchemy）のテスト。"""

from __future__ import annotations

import itertools
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from core.style_alchemy.comfy_export import LBW_NODE_CLASS, validate_prompt
from core.style_alchemy.dataset_curator import (
    CONTROLLABLE_CATEGORIES,
    CaptionRecord,
    DatasetCurator,
    TagCategory,
    TagTaxonomy,
)
from core.style_alchemy.grid_laboratory import GridAxis, GridLaboratory, TokenDictionary, TokenEntry
from core.style_alchemy.model_optimizer import (
    CMAES,
    STYLE_PARAMS,
    BayesianStyleOptimizer,
    GaussianProcess,
    GeneticOptimizer,
    Observation,
    ParameterSpace,
    RBFKernel,
    RecipeExporter,
    expected_improvement,
    norm_cdf,
    probability_of_improvement,
    upper_confidence_bound,
)
from core.style_alchemy.preference_engine import (
    LAMBDA_GRID,
    BradleyTerryModel,
    Comparison,
    EloEngine,
    PreferenceAnalyzer,
    StyleCandidate,
    rank_candidates,
    ridge_fit,
    ridge_loocv_mse,
    spearman_correlation,
    suggest_next_pair,
)
from core.style_alchemy.preview import render_candidate_preview_svg
from core.style_alchemy.repository import StyleLabRepository
from core.style_alchemy.schemas import (
    ARCHITECTURE_BLOCKS,
    LBW_BLOCKS,
    CheckpointRef,
    LoRASpec,
    ModelArchitecture,
    StyleRecipe,
)

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _true_utility(params: dict[str, float]) -> float:
    return 2.0 * params["style.black_fill_ratio"] - 1.5 * params["style.line_taper"]


def _simulate(candidates: list[StyleCandidate], utility, n: int, seed: int, scale: float = 4.0) -> list[Comparison]:
    rng = np.random.default_rng(seed)
    comparisons: list[Comparison] = []
    for k in range(n):
        i, j = rng.choice(len(candidates), size=2, replace=False)
        diff = utility(candidates[i].params) - utility(candidates[j].params)
        p_left = 1.0 / (1.0 + math.exp(-scale * diff))
        outcome = "left" if rng.random() < p_left else "right"
        comparisons.append(Comparison(id=f"cmp_{k:04d}", left_id=candidates[i].id, right_id=candidates[j].id, outcome=outcome))
    return comparisons


def _style_candidates(n: int, seed: int) -> list[StyleCandidate]:
    space = ParameterSpace.for_style_lab(ModelArchitecture.SDXL, grouping="grouped")
    return [
        StyleCandidate(id=f"cand_{i:02d}", label=f"候補{i:02d}", params=params)
        for i, params in enumerate(space.sample_lhs(n, seed=seed))
    ]


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def test_elo_expected_score_and_symmetry() -> None:
    elo = EloEngine()
    assert elo.expected_score(1500, 1500) == pytest.approx(0.5)
    assert elo.expected_score(1900, 1500) == pytest.approx(1.0 / 1.1)
    for ra, rb in [(1400, 1650), (1720, 1310), (1500, 1501)]:
        assert elo.expected_score(ra, rb) + elo.expected_score(rb, ra) == pytest.approx(1.0)


def test_elo_k_decay_and_zero_sum_update() -> None:
    elo = EloEngine(k_factor=32, k_decay_games=30)
    assert elo.k_for(0) == pytest.approx(32.0)
    assert elo.k_for(30) == pytest.approx(16.0)
    assert EloEngine(k_decay_games=None).k_for(500) == pytest.approx(32.0)
    new_a, new_b = elo.update(1500, 1500, 1.0)
    assert (new_a, new_b) == (pytest.approx(1516.0), pytest.approx(1484.0))
    a2, b2 = elo.update(1600, 1450, 0.0)
    assert a2 + b2 == pytest.approx(1600 + 1450)
    tie_a, tie_b = elo.update(1500, 1500, 0.5)
    assert (tie_a, tie_b) == (pytest.approx(1500.0), pytest.approx(1500.0))


def test_elo_compute_sequential_ignores_unknown_ids() -> None:
    comparisons = [Comparison(left_id="a", right_id=other, outcome="left") for other in ("b", "c", "b", "c")]
    comparisons.append(Comparison(left_id="a", right_id="ghost", outcome="right"))
    ratings = EloEngine().compute(["a", "b", "c"], comparisons)
    assert ratings["a"] > 1500 > ratings["b"]
    assert set(ratings) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Bradley-Terry
# ---------------------------------------------------------------------------


def test_bradley_terry_recovers_true_ranking() -> None:
    rng = np.random.default_rng(0)
    ids = [f"c{i:02d}" for i in range(12)]
    theta = np.linspace(-1.5, 1.5, 12)
    rng.shuffle(theta)
    comparisons = []
    for k in range(300):
        i, j = rng.choice(12, size=2, replace=False)
        p_left = 1.0 / (1.0 + math.exp(-(theta[i] - theta[j])))
        comparisons.append(Comparison(id=f"m{k}", left_id=ids[i], right_id=ids[j], outcome="left" if rng.random() < p_left else "right"))
    result = BradleyTerryModel().fit(ids, comparisons)
    assert result.converged
    assert spearman_correlation(theta, [result.log_strengths[c] for c in ids]) > 0.85
    assert all(0.0 < se < 5.0 for se in result.std_errors.values())
    assert math.exp(np.mean(np.log(list(result.strengths.values())))) == pytest.approx(1.0)
    assert result.log_likelihood < 0
    p = BradleyTerryModel.predict_proba(result, ids[0], ids[1])
    assert 0.0 < p < 1.0
    assert p + BradleyTerryModel.predict_proba(result, ids[1], ids[0]) == pytest.approx(1.0)
    for cid in ids:
        assert result.elo_scale[cid] == pytest.approx(1500 + 400 * math.log10(result.strengths[cid]))


def test_bradley_terry_all_wins_converges_to_finite_values() -> None:
    comparisons = [Comparison(left_id="a", right_id=b, outcome="left") for b in ("b", "c", "d") for _ in range(5)]
    result = BradleyTerryModel().fit(["a", "b", "c", "d"], comparisons)
    assert result.converged
    assert all(math.isfinite(v) and v > 0 for v in result.strengths.values())
    assert max(result.strengths, key=result.strengths.get) == "a"


def test_bradley_terry_newton_polish_matches_pure_mm() -> None:
    rng = np.random.default_rng(3)
    ids = [f"p{i}" for i in range(6)]
    comparisons = [
        Comparison(left_id=ids[i], right_id=ids[j], outcome=("left" if rng.random() < 0.3 + 0.1 * i else "right"))
        for _ in range(60)
        for i, j in [tuple(rng.choice(6, size=2, replace=False))]
    ]

    class PureMM(BradleyTerryModel):
        _MM_HANDOFF = 1e-13

    fast = BradleyTerryModel().fit(ids, comparisons)
    slow = PureMM(max_iter=200000, tol=1e-12).fit(ids, comparisons)
    for cid in ids:
        assert fast.log_strengths[cid] == pytest.approx(slow.log_strengths[cid], abs=1e-8)


def test_bradley_terry_ties_and_empty_inputs() -> None:
    tie_only = [Comparison(left_id="x", right_id="y", outcome="tie") for _ in range(4)]
    result = BradleyTerryModel().fit(["x", "y"], tie_only)
    assert result.strengths["x"] == pytest.approx(result.strengths["y"])
    empty = BradleyTerryModel().fit([], [])
    assert empty.strengths == {} and empty.converged


# ---------------------------------------------------------------------------
# 美的選好ベクトル
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pairwise_logistic", "ridge_bt"])
def test_preference_vector_recovers_utility_signs(method: str) -> None:
    candidates = _style_candidates(16, seed=7)
    comparisons = _simulate(candidates, _true_utility, n=260, seed=11)
    vector = PreferenceAnalyzer(n_bootstrap=40, seed=0, method=method).analyze(candidates, comparisons)
    features = {f.name: f for f in vector.features}
    assert set(features) >= set(STYLE_PARAMS)
    assert features["style.black_fill_ratio"].direction == "prefer_higher"
    assert features["style.line_taper"].direction == "prefer_lower"
    top_two = {f.name for f in sorted(vector.features, key=lambda f: -f.importance)[:2]}
    assert top_two == {"style.black_fill_ratio", "style.line_taper"}
    assert sum(f.importance for f in vector.features) == pytest.approx(1.0)
    assert vector.fit_accuracy > 0.7
    assert vector.regularization in LAMBDA_GRID
    assert vector.n_comparisons == 260 and vector.n_candidates == 16
    for feature in vector.features:
        assert feature.ci_low <= feature.ci_high
    assert features["style.black_fill_ratio"].ci_low > 0
    assert features["style.line_taper"].ci_high < 0
    assert features["style.black_fill_ratio"].permutation_importance > 0
    if method == "ridge_bt":
        assert vector.r2 is not None and vector.r2 > 0.5
    else:
        assert vector.r2 is None


def test_ridge_loocv_matches_bruteforce() -> None:
    rng = np.random.default_rng(5)
    z = rng.normal(size=(14, 3))
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    y = z @ np.array([1.0, -0.5, 0.2]) + rng.normal(scale=0.3, size=14)
    for lam in (0.01, 0.5, 4.0):
        errors = []
        for i in range(14):
            mask = np.arange(14) != i
            weights, intercept = ridge_fit(z[mask], y[mask], lam)
            errors.append((y[i] - (z[i] @ weights + intercept)) ** 2)
        assert ridge_loocv_mse(z, y, lam) == pytest.approx(float(np.mean(errors)), rel=1e-6)


def test_preference_analyzer_neutral_without_data() -> None:
    candidates = _style_candidates(4, seed=1)
    vector = PreferenceAnalyzer().analyze(candidates, [])
    assert vector.n_comparisons == 0
    assert all(f.direction == "neutral" and f.importance == 0.0 for f in vector.features)
    with pytest.raises(ValueError):
        PreferenceAnalyzer(method="unknown")  # type: ignore[arg-type]


def test_rank_candidates_counts_and_order() -> None:
    candidates = _style_candidates(10, seed=3)
    comparisons = _simulate(candidates, _true_utility, n=150, seed=2)
    comparisons.append(Comparison(left_id=candidates[0].id, right_id=candidates[1].id, outcome="tie"))
    ratings = rank_candidates(candidates, comparisons)
    assert [r.rank for r in ratings] == list(range(1, 11))
    assert [r.bt_strength for r in ratings] == sorted((r.bt_strength for r in ratings), reverse=True)
    assert sum(r.games for r in ratings) == 2 * len(comparisons)
    assert sum(r.wins for r in ratings) == sum(r.losses for r in ratings)
    assert sum(r.ties for r in ratings) == 2
    best = max(candidates, key=lambda c: _true_utility(c.params))
    assert best.id in {r.candidate_id for r in ratings[:3]}


def test_suggest_next_pair_behaviour() -> None:
    candidates = _style_candidates(5, seed=4)
    first = suggest_next_pair(candidates, [], seed=3)
    assert first is not None and first[0] != first[1]
    assert suggest_next_pair(candidates, [], seed=3) == first
    a, b = candidates[0].id, candidates[1].id
    heavy = [Comparison(left_id=a, right_id=b, outcome="left" if k % 2 else "right") for k in range(12)]
    pair = suggest_next_pair(candidates, heavy, seed=3)
    assert pair is not None and set(pair) != {a, b}
    assert suggest_next_pair(candidates[:1], [], seed=0) is None


# ---------------------------------------------------------------------------
# パラメータ空間
# ---------------------------------------------------------------------------


def test_parameter_space_for_style_lab_dimensions() -> None:
    space = ParameterSpace.for_style_lab(ModelArchitecture.SDXL, lora_names=["ink_style.safetensors"], grouping="grouped")
    names = space.names()
    for expected in [*STYLE_PARAMS, "merge.BASE", "merge.IN", "merge.MID", "merge.OUT", "lora.ink_style.strength", "lora.ink_style.OUT"]:
        assert expected in names
    assert space.dim("lora.ink_style.strength").high == pytest.approx(1.5)
    assert space.dim("merge.IN").default == pytest.approx(0.5)
    per_block = ParameterSpace.for_style_lab(ModelArchitecture.SDXL, lora_names=["a"], grouping="per_block")
    merge_dims = [n for n in per_block.names() if n.startswith("merge.")]
    lora_dims = [n for n in per_block.names() if n.startswith("lora.a.") and not n.endswith("strength")]
    assert len(merge_dims) == len(ARCHITECTURE_BLOCKS[ModelArchitecture.SDXL])
    assert len(lora_dims) == len(LBW_BLOCKS[ModelArchitecture.SDXL])
    flux = ParameterSpace.for_style_lab("flux", grouping="grouped", include_style_params=False)
    assert flux.names() == ["merge.BASE", "merge.DOUBLE", "merge.SINGLE"]
    with pytest.raises(ValueError):
        ParameterSpace.for_style_lab("sdxl", grouping="unknown")  # type: ignore[arg-type]


def test_parameter_space_lhs_stratified_and_roundtrip() -> None:
    space = ParameterSpace.for_style_lab(ModelArchitecture.SDXL, lora_names=["ink"], grouping="grouped")
    unit = space.sample_lhs_unit(10, seed=1)
    for j in range(space.dim_count):
        assert sorted(np.floor(unit[:, j] * 10).astype(int).tolist()) == list(range(10))
    samples = space.sample_lhs(6, seed=2)
    for params in samples:
        restored = space.from_unit(space.to_unit(params))
        for name in space.names():
            assert restored[name] == pytest.approx(params[name], abs=1e-6)
            dim = space.dim(name)
            assert dim.low <= params[name] <= dim.high
    clipped = space.clip_params({"style.line_taper": 9.0, "unknown": 1.0})
    assert clipped["style.line_taper"] == 1.0 and "unknown" not in clipped
    assert clipped["merge.IN"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ガウス過程・獲得関数
# ---------------------------------------------------------------------------


def test_gaussian_process_fits_1d_function() -> None:
    x = np.linspace(0.0, 1.0, 8)[:, None]
    y = np.sin(6.0 * x[:, 0])
    gp = GaussianProcess(RBFKernel(0.3), noise=1e-6).fit(x, y)
    info = gp.optimize_hyperparameters(n_restarts=4, seed=0)
    grid = np.linspace(0.0, 1.0, 101)[:, None]
    mean, std = gp.predict(grid)
    assert float(np.max(np.abs(mean - np.sin(6.0 * grid[:, 0])))) < 0.05
    assert info["log_marginal_likelihood"] == pytest.approx(gp.log_marginal_likelihood(), rel=1e-6)
    _, std_train = gp.predict(x)
    assert float(np.max(std_train)) < float(np.max(std)) + 1e-12
    far_gp = GaussianProcess(RBFKernel(0.1)).fit(np.array([[0.0], [0.1]]), np.array([0.0, 1.0]))
    _, std_far = far_gp.predict(np.array([[0.05], [0.9]]))
    assert std_far[1] > std_far[0]


def test_expected_improvement_maximum_near_optimum() -> None:
    x = np.array([[0.0], [0.25], [0.5], [1.0]])
    y = -((x[:, 0] - 0.7) ** 2)
    gp = GaussianProcess(RBFKernel(0.3)).fit(x, y)
    gp.optimize_hyperparameters(n_restarts=4, seed=0)
    grid = np.linspace(0.0, 1.0, 201)[:, None]
    mean, std = gp.predict(grid)
    ei = expected_improvement(mean, std, float(y.max()))
    assert abs(float(grid[int(np.argmax(ei)), 0]) - 0.7) < 0.15


def test_acquisition_function_properties() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert expected_improvement(1.0, 0.0, best=0.5, xi=0.0) == pytest.approx(0.5)
    assert expected_improvement(0.0, 0.0, best=0.5, xi=0.0) == pytest.approx(0.0)
    values = expected_improvement(np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.2, 0.2]), best=0.5)
    assert np.all(values >= 0) and values[2] > values[1] > values[0]
    assert upper_confidence_bound(1.0, 0.5, kappa=2.0) == pytest.approx(2.0)
    assert probability_of_improvement(0.5, 0.1, best=0.5, xi=0.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CMA-ES / GA
# ---------------------------------------------------------------------------


def test_cmaes_sphere_and_ellipsoid() -> None:
    es = CMAES(mean=np.full(5, 0.8), sigma=0.3, seed=0)
    best_x, best_value = es.optimize(lambda pts: -np.sum((pts - 0.3) ** 2, axis=1), generations=80)
    assert best_value > -1e-5
    assert np.allclose(best_x, 0.3, atol=5e-3)
    weights = 10 ** (np.arange(4) * 4 / 3)
    es2 = CMAES(mean=np.full(4, 0.1), sigma=0.3, seed=1)
    _, value2 = es2.optimize(lambda pts: -np.sum(weights * (pts - 0.5) ** 2, axis=1), generations=200)
    assert value2 > -1e-6
    bounded = CMAES(mean=np.full(3, 0.5), sigma=0.5, seed=2)
    candidates = bounded.ask()
    assert np.all(candidates >= 0.0) and np.all(candidates <= 1.0)


def test_genetic_optimizer_improves() -> None:
    ga = GeneticOptimizer(6, seed=0)
    objective = lambda pts: -np.sum((pts - 0.6) ** 2, axis=1)  # noqa: E731
    initial = ga.ask()
    ga.tell(initial, objective(initial))
    initial_best = ga.best_value
    for _ in range(30):
        children = ga.ask()
        assert np.all(children >= 0.0) and np.all(children <= 1.0)
        ga.tell(children, objective(children))
    assert ga.best_value > initial_best
    assert ga.best_value > -0.01
    assert ga.population.shape == (24, 6)


# ---------------------------------------------------------------------------
# ベイズ最適化
# ---------------------------------------------------------------------------


def test_bayesian_optimizer_beats_random_search() -> None:
    space = ParameterSpace.for_style_lab(ModelArchitecture.SDXL, grouping="grouped")
    center = np.random.default_rng(42).random(space.dim_count)

    def objective(params: dict[str, float]) -> float:
        return -float(np.sum((space.to_unit(params) - center) ** 2))

    optimizer = BayesianStyleOptimizer(space, seed=0)
    for params in space.sample_lhs(5, seed=3):
        optimizer.add_observation(params, objective(params))
    for _ in range(20):
        suggestion = optimizer.suggest(n=1, strategy="gp")[0]
        optimizer.add_observation(suggestion.params, objective(suggestion.params))
    best = optimizer.best()
    assert best is not None
    random_bests = []
    for seed in range(7):
        rng = np.random.default_rng(100 + seed)
        random_bests.append(max(objective(space.from_unit(rng.random(space.dim_count))) for _ in range(25)))
    assert best.score > float(np.median(random_bests))


@pytest.mark.parametrize("strategy", ["gp", "cmaes", "genetic", "hybrid"])
def test_bayesian_optimizer_batch_suggestions(strategy: str) -> None:
    space = ParameterSpace.for_style_lab(ModelArchitecture.SDXL, lora_names=["ink_style"], grouping="grouped")
    optimizer = BayesianStyleOptimizer(space, seed=1, acquisition="ei")
    for k, params in enumerate(space.sample_lhs(8, seed=9)):
        optimizer.add_observation(params, 1500.0 + 300.0 * _true_utility(params), candidate_id=f"c{k}")
    suggestions = optimizer.suggest(n=3, strategy=strategy)  # type: ignore[arg-type]
    assert [s.rank for s in suggestions] == [1, 2, 3]
    units = []
    for suggestion in suggestions:
        assert set(suggestion.params) == set(space.names())
        for name, value in suggestion.params.items():
            dim = space.dim(name)
            assert dim.low - 1e-9 <= value <= dim.high + 1e-9
        assert suggestion.rationale and "予測スコア" in suggestion.rationale
        assert suggestion.predicted_std >= 0
        units.append(space.to_unit(suggestion.params))
    min_gap = 0.04 * math.sqrt(space.dim_count) * 0.99
    for a, b in itertools.combinations(units, 2):
        assert float(np.linalg.norm(a - b)) >= min_gap
    if strategy != "hybrid":
        assert all(s.strategy in (strategy, "gp") for s in suggestions)


def test_bayesian_optimizer_initial_lhs_posterior_and_sensitivity() -> None:
    space = ParameterSpace.for_style_lab(ModelArchitecture.SD15, grouping="grouped")
    optimizer = BayesianStyleOptimizer(space, seed=2)
    initial = optimizer.suggest(n=4)
    assert all(s.strategy == "lhs" for s in initial)
    assert optimizer.posterior(space.default_params()) == (0.0, 1.0)
    for params in space.sample_lhs(6, seed=4):
        optimizer.add_observation(params, _true_utility(params))
    mean, std = optimizer.posterior(space.default_params())
    assert math.isfinite(mean) and std > 0
    sensitivity = optimizer.sensitivity()
    assert sum(sensitivity.values()) == pytest.approx(1.0)
    assert optimizer.best() is not None
    with pytest.raises(ValueError):
        optimizer.add_observation(space.default_params(), float("nan"))
    assert isinstance(optimizer.observations[0], Observation)


# ---------------------------------------------------------------------------
# レシピエクスポート
# ---------------------------------------------------------------------------


def test_recipe_exporter_builds_valid_recipe_and_workflow() -> None:
    space = ParameterSpace.for_style_lab(ModelArchitecture.SDXL, lora_names=["ink_style.safetensors"], grouping="grouped")
    params = space.default_params()
    params.update(
        {
            "style.black_fill_ratio": 0.82,
            "style.line_taper": 0.75,
            "merge.IN": 0.2,
            "merge.OUT": 0.9,
            "lora.ink_style.strength": 1.1,
            "lora.ink_style.OUT": 0.3,
        }
    )
    exporter = RecipeExporter()
    recipe = exporter.params_to_recipe(
        params,
        architecture=ModelArchitecture.SDXL,
        model_a=CheckpointRef(name="base_a.safetensors"),
        model_b=CheckpointRef(name="base_b.safetensors"),
        loras=[LoRASpec(name="ink_style.safetensors", trigger_words=["caf_ink"])],
        name="テストレシピ",
    )
    assert recipe.merge is not None
    assert recipe.merge.weight_for("IN04") == pytest.approx(0.2)
    assert recipe.merge.weight_for("OUT08") == pytest.approx(0.9)
    lora = recipe.loras[0]
    assert lora.strength_model == pytest.approx(1.1) and lora.strength_clip == pytest.approx(1.1)
    assert lora.block_weights is not None and lora.block_weights["OUT03"] == pytest.approx(0.3)
    assert "heavy black fill, strong contrast" in recipe.style_tokens
    assert "tapered ink lines, variable line weight" in recipe.style_tokens
    config = exporter.to_merge_config(recipe)
    assert len(config["mbw_string"].split(",")) == len(ARCHITECTURE_BLOCKS[ModelArchitecture.SDXL])
    assert list(config["block_weights"]) == ARCHITECTURE_BLOCKS[ModelArchitecture.SDXL]
    assert config["loras"][0]["lbw_string"].count(",") == len(LBW_BLOCKS[ModelArchitecture.SDXL]) - 1
    workflow = exporter.to_comfy_workflow(recipe, "1girl, standing", "blurry", seed=7, width=832, height=1216)
    assert validate_prompt(workflow) == []
    classes = {node["class_type"] for node in workflow.values()}
    assert {"ModelMergeSDXL", LBW_NODE_CLASS, "KSampler", "SaveImage"} <= classes
    merge_node = next(node for node in workflow.values() if node["class_type"] == "ModelMergeSDXL")
    assert merge_node["inputs"]["input_blocks.4"] == pytest.approx(0.8)
    script = exporter.to_merge_script(recipe)
    compile(script, "<merge>", "exec")
    assert "safetensors" in script and "base_a.safetensors" in script


@pytest.mark.parametrize("arch", list(ModelArchitecture))
def test_merge_script_and_workflow_for_all_architectures(arch: ModelArchitecture) -> None:
    space = ParameterSpace.for_style_lab(arch, lora_names=["detail"], grouping="grouped")
    params = space.sample_lhs(1, seed=5)[0]
    exporter = RecipeExporter()
    recipe = exporter.params_to_recipe(
        params,
        architecture=arch,
        model_a=CheckpointRef(name="a.safetensors"),
        model_b=CheckpointRef(name="b.safetensors"),
        model_c=CheckpointRef(name="c.safetensors"),
        loras=[LoRASpec(name="detail.safetensors")],
    )
    assert recipe.merge is not None and recipe.merge.method.value == "add_difference"
    script = exporter.to_merge_script(recipe)
    compile(script, f"<merge-{arch.value}>", "exec")
    workflow = exporter.to_comfy_workflow(recipe, "test", "", seed=1, width=1024, height=1024)
    assert validate_prompt(workflow) == []


def test_apply_params_overrides_recipe() -> None:
    exporter = RecipeExporter()
    base = exporter.params_to_recipe(
        {"merge.IN": 0.5},
        architecture="sdxl",
        model_a=CheckpointRef(name="a.safetensors"),
        model_b=CheckpointRef(name="b.safetensors"),
        loras=[LoRASpec(name="ink.safetensors", strength_model=0.6)],
    )
    derived = exporter.apply_params(base, {"merge.OUT": 0.1, "lora.ink.strength": 1.3, "style.black_fill_ratio": 0.9}, {"cfg": 4.5, "steps": 20})
    assert derived.merge is not None and derived.merge.weight_for("OUT02") == pytest.approx(0.1)
    assert derived.merge.weight_for("IN02") == pytest.approx(0.5)
    assert derived.loras[0].strength_model == pytest.approx(1.3)
    assert derived.sampler.cfg == pytest.approx(4.5) and derived.sampler.steps == 20
    assert "heavy black fill, strong contrast" in derived.style_tokens
    no_merge = StyleRecipe(base_checkpoint=CheckpointRef(name="x.safetensors"))
    with pytest.raises(ValueError):
        exporter.to_merge_script(no_merge)
    assert exporter.to_merge_config(no_merge)["method"] == "none"


# ---------------------------------------------------------------------------
# データセット精製
# ---------------------------------------------------------------------------


def _biased_captions() -> str:
    lines = []
    for i in range(8):
        lines.append(f"img{i:03d}: 1girl, crying, tears, from above, upper body, standing, simple background, monochrome, screentones, masterpiece")
    for i in range(8, 16):
        lines.append(f"img{i:03d}: 1girl, smile, eye level, cowboy shot, sitting, classroom, monochrome, best quality, signature")
    for i in range(16, 24):
        lines.append(f"img{i:03d}: 1girl, angry, from below, full body, running, cityscape, rim light, lineart")
    for i in range(24, 30):
        lines.append(f"img{i:03d}: 1girl, smile, from below, upper body, standing, cityscape, soft lighting")
    return "\n".join(lines)


def test_taxonomy_normalization_and_classification() -> None:
    taxonomy = TagTaxonomy()
    assert len(taxonomy) >= 300
    cases = {
        "(smile:1.2)": ("smile", TagCategory.EXPRESSION),
        "((masterpiece))": ("masterpiece", TagCategory.QUALITY),
        "Upper_Body": ("upper body", TagCategory.COMPOSITION),
        "俯瞰": ("from above", TagCategory.CAMERA_ANGLE),
        "逆光": ("backlighting", TagCategory.LIGHTING),
        "Backlit": ("backlighting", TagCategory.LIGHTING),
        "score_9": ("score 9", TagCategory.QUALITY),
        "スクリーントーン": ("screentones", TagCategory.STYLE),
        "白背景": ("white background", TagCategory.BACKGROUND),
    }
    for raw, (normalized, category) in cases.items():
        assert taxonomy.normalize(raw) == normalized
        assert taxonomy.classify(raw) == category
    assert taxonomy.normalize("lucy \\(cyberpunk\\)") == "lucy (cyberpunk)"
    assert taxonomy.classify("purple hair") == TagCategory.SUBJECT
    assert taxonomy.extract_known_terms("a girl smiling from above under dramatic lighting") == ["dramatic lighting", "from above", "smile"]
    taxonomy.add("ink splash", TagCategory.STYLE)
    assert taxonomy.classify("ink splash") == TagCategory.STYLE


def test_parse_caption_lines_formats() -> None:
    records = DatasetCurator.parse_caption_lines("0001.png: crying, from above\n# コメント\nsmile:1.2, standing\n\nimg_02: 笑顔, 全身")
    assert [r.image_id for r in records] == ["0001.png", "img_0002", "img_02"]
    assert records[1].caption == "smile:1.2, standing"


def test_cleanse_removes_style_quality_meta_and_prefixes_trigger() -> None:
    curator = DatasetCurator(trigger_token="ink_style")
    record = CaptionRecord(image_id="a1", caption="masterpiece, 1girl, (crying:1.2), 俯瞰, monochrome, screentones, signature, upper body, ink_style")
    cleansed = curator.cleanse([record])[0]
    assert cleansed.caption.split(", ")[0] == "ink_style"
    assert cleansed.caption.count("ink_style") == 1
    for removed in ("monochrome", "screentones", "masterpiece", "signature"):
        assert removed not in cleansed.caption.split(", ")
    assert set(cleansed.removed_style_tags) == {"monochrome", "screentones"}
    assert set(cleansed.removed_noise_tags) == {"masterpiece", "signature"}
    assert cleansed.controllable["expression"] == ["crying"]
    assert cleansed.controllable["camera_angle"] == ["from above"]
    assert cleansed.subject_tags == ["1girl"]
    assert set(cleansed.missing_categories) == {"pose", "lighting", "background"}


def test_dataset_analysis_detects_entanglement_and_gaps() -> None:
    curator = DatasetCurator(trigger_token="ink_style")
    report = curator.analyze(curator.parse_caption_lines(_biased_captions()))
    assert report.total_images == 30
    pairs = {(e.tag_a, e.tag_b): e for e in report.entanglements}
    entangled = pairs.get(("crying", "from above")) or pairs.get(("from above", "crying"))
    assert entangled is not None
    assert entangled.severity == "high" and entangled.conditional_prob == pytest.approx(1.0)
    backlighting = [d for d in report.deficits if d.tag == "backlighting"]
    assert backlighting and backlighting[0].current_count == 0 and backlighting[0].required_additional > 0
    assert any(shot.conditions.get("lighting") == "backlighting" for shot in report.recommended_shots)
    assert any("癒着の解消" in shot.reason for shot in report.recommended_shots)
    assert report.total_additional_images >= max(d.required_additional for d in report.deficits)
    assert 0.0 <= report.balance_score <= 100.0
    matrix = np.array(report.cooccurrence.counts)
    assert np.array_equal(matrix, matrix.T)
    assert "monochrome" not in report.tag_counts
    for category in CONTROLLABLE_CATEGORIES:
        assert 0.0 <= report.normalized_entropy[category.value] <= 1.0
    balanced = curator.analyze(
        curator.parse_caption_lines(
            "\n".join(
                f"b{i:03d}: {expr}, {angle}, {shot}"
                for i, (expr, angle, shot) in enumerate(
                    itertools.product(["smile", "crying", "angry", "surprised"], ["from above", "from below", "eye level"], ["close-up", "full body"])
                )
            )
        )
    )
    assert not [e for e in balanced.entanglements if {e.category_a, e.category_b} == {"expression", "camera_angle"}]
    assert curator.analyze([]).warnings


@pytest.mark.asyncio
async def test_cleanse_with_llm_matches_rule_based(mock_llm_client) -> None:
    curator = DatasetCurator(trigger_token="ink_style", llm_client=mock_llm_client)
    records = curator.parse_caption_lines(_biased_captions())[:4]
    via_llm = await curator.cleanse_with_llm(records)
    rule = curator.cleanse(records)
    assert [c.caption for c in via_llm] == [c.caption for c in rule]
    assert [c.image_id for c in via_llm] == [r.image_id for r in records]
    stats = mock_llm_client.telemetry.stats()
    assert stats


def test_mock_handler_registered() -> None:
    from core.llm_client import get_mock_handler

    import core.style_alchemy.mock_handlers  # noqa: F401

    assert get_mock_handler("style.caption_cleanse", "synthesize") is not None


# ---------------------------------------------------------------------------
# グリッド実験
# ---------------------------------------------------------------------------


def test_token_dictionary_render_and_roundtrip() -> None:
    dictionary = TokenDictionary()
    assert set(TokenDictionary.DEFAULT_CATEGORIES) <= set(dictionary.categories())
    for category in TokenDictionary.DEFAULT_CATEGORIES:
        assert len(dictionary.list(category)) >= 8
    assert dictionary.render("tapered ink lines") == "tapered ink lines"
    assert dictionary.render("tapered ink lines", 1.2) == "(tapered ink lines:1.20)"
    dictionary.add("ink splatter", "texture", weight=1.3, description="インクの飛沫")
    assert dictionary.render("ink splatter") == "(ink splatter:1.30)"
    restored = TokenDictionary.from_dict(dictionary.to_dict())
    assert len(restored) == len(dictionary)
    assert restored.remove("ink splatter") and restored.get("ink splatter") is None


def test_grid_generates_cells_and_anova() -> None:
    lab = GridLaboratory()
    experiment = lab.create_experiment(
        "線質×ベタ比率",
        "1girl, standing, {x}",
        GridAxis(name="線画", kind="token", target="line_art", values=["clean lineart", "tapered ink lines", "rough sketch lines"]),
        GridAxis(name="ベタ", kind="param", target="style.black_fill_ratio", values=[0.2, 0.5, 0.8]),
        GridAxis(name="シード", kind="seed", values=[11, 22]),
        negative_prompt="blurry",
    )
    cells = lab.cells(experiment.id)
    assert len(cells) == 18 == experiment.cell_count()
    assert {c.seed for c in cells} == {11, 22}
    assert all("{x}" not in c.prompt for c in cells)
    assert any("tapered ink lines" in c.prompt for c in cells)
    x_effect = [0.0, 6.0, 2.0]
    for cell in cells:
        score = 50.0 + x_effect[cell.x_index] + 0.5 * cell.y_index + 0.1 * cell.z_index
        lab.record_result(cell.id, score, image_ref=f"img/{cell.id}.png")
    analysis = lab.analyze(experiment.id)
    assert analysis.n_scored == 18
    assert analysis.anova["ss_x"] > analysis.anova["ss_y"] > 0
    assert analysis.anova["ss_total"] == pytest.approx(
        analysis.anova["ss_x"] + analysis.anova["ss_y"] + analysis.anova["ss_xy"] + analysis.anova["ss_residual"]
    )
    best = next(c for c in lab.cells(experiment.id) if c.id == analysis.best_cell_id)
    assert best.x_index == 1 and best.y_index == 2
    assert analysis.interaction_xy is not None and len(analysis.interaction_xy) == 3
    means = [entry["mean"] for entry in analysis.main_effects["線画"]]
    assert max(range(3), key=lambda i: means[i]) == 1
    assert lab.get_experiment(experiment.id).status == "completed"


def test_grid_axis_kinds_and_token_weight_sweep() -> None:
    lab = GridLaboratory()
    experiment = lab.create_experiment(
        "重み掃引",
        "portrait",
        GridAxis(name="線の重み", kind="token", target="tapered ink lines", values=[0.8, 1.0, 1.2]),
        GridAxis(name="CFG", kind="cfg", values=[4.0, 7.0]),
        GridAxis(name="LoRA", kind="lora_weight", target="ink_style", values=[0.5, 1.0]),
    )
    cells = lab.cells(experiment.id)
    prompts = {c.prompt for c in cells}
    assert "portrait, (tapered ink lines:0.80)" in prompts and "portrait, tapered ink lines" in prompts
    assert {c.sampler_overrides["cfg"] for c in cells} == {4.0, 7.0}
    assert {c.params["lora.ink_style.strength"] for c in cells} == {0.5, 1.0}
    with pytest.raises(ValueError):
        GridAxis(name="bad", kind="param", values=[0.1])
    with pytest.raises(ValueError):
        GridAxis(name="bad", kind="cfg", values=["high"])


def test_grid_to_comfy_batch_is_valid(tmp_db) -> None:
    repo = StyleLabRepository(tmp_db)
    lab = GridLaboratory(repo=repo)
    experiment = lab.create_experiment(
        "マージ比率",
        "1girl",
        GridAxis(name="OUT", kind="merge_weight", target="OUT", values=[0.2, 0.8]),
        GridAxis(name="steps", kind="steps", values=[20, 30]),
    )
    recipe = RecipeExporter().params_to_recipe(
        {}, architecture="sdxl", model_a=CheckpointRef(name="a.safetensors"), model_b=CheckpointRef(name="b.safetensors"),
        loras=[LoRASpec(name="ink_style.safetensors")],
    )
    batch = lab.to_comfy_batch(experiment.id, recipe)
    assert len(batch) == 4
    for item in batch:
        assert validate_prompt(item["workflow"]) == []
        sampler = next(n for n in item["workflow"].values() if n["class_type"] == "KSampler")
        assert sampler["inputs"]["steps"] in (20, 30)
    merge_inputs = [next(n for n in item["workflow"].values() if n["class_type"] == "ModelMergeSDXL")["inputs"] for item in batch]
    assert {round(m["output_blocks.0"], 3) for m in merge_inputs} == {0.8, 0.2}
    reloaded = GridLaboratory(repo=repo)
    assert len(reloaded.cells(experiment.id)) == 4


# ---------------------------------------------------------------------------
# リポジトリ
# ---------------------------------------------------------------------------


def test_repository_crud(tmp_db) -> None:
    repo = StyleLabRepository(tmp_db)
    recipe = StyleRecipe(name="保存テスト", base_checkpoint=CheckpointRef(name="x.safetensors"))
    candidates = [StyleCandidate(id=f"c{i}", label=f"候補{i}", params={"style.line_taper": i / 4}, recipe=recipe if i == 0 else None) for i in range(3)]
    for candidate in candidates:
        repo.save_candidate(candidate)
    assert repo.count_candidates() == 3
    loaded = repo.get_candidate("c0")
    assert loaded is not None and loaded.recipe is not None and loaded.recipe.name == "保存テスト"
    repo.save_comparison(Comparison(id="m1", left_id="c0", right_id="c1", outcome="left", response_time_ms=420.0))
    repo.save_comparison(Comparison(id="m2", left_id="c1", right_id="c2", outcome="tie", session_id="s1"))
    assert [c.id for c in repo.list_comparisons()] == ["m1", "m2"]
    assert [c.id for c in repo.list_comparisons(session_id="s1")] == ["m2"]
    repo.add_observation(Observation(params={"style.line_taper": 0.2}, score=1510.0, candidate_id="c0"))
    assert repo.list_observations()[0].score == pytest.approx(1510.0)
    assert repo.delete_candidate("c0")
    assert repo.get_candidate("c0") is None
    assert [c.id for c in repo.list_comparisons()] == ["m2"]
    assert repo.list_observations() == []
    repo.save_recipe(recipe)
    assert repo.get_recipe(recipe.id).name == "保存テスト"
    assert [r.id for r in repo.list_recipes()] == [recipe.id]
    assert repo.delete_recipe(recipe.id) and repo.get_recipe(recipe.id) is None
    dictionary = repo.load_dictionary()
    assert len(dictionary) >= 48
    repo.save_token(TokenEntry(token="ink splatter", category="texture", weight=1.4))
    assert any(t.token == "ink splatter" for t in repo.list_tokens("texture"))
    assert repo.delete_token("ink splatter", "texture")
    report = DatasetCurator().analyze(DatasetCurator.parse_caption_lines(_biased_captions()))
    report_id = repo.save_dataset_report(report, name="偏りテスト")
    assert repo.get_dataset_report(report_id).total_images == 30
    assert repo.list_dataset_reports()[0]["id"] == report_id
    assert repo.delete_dataset_report(report_id)


# ---------------------------------------------------------------------------
# プレビュー
# ---------------------------------------------------------------------------


def test_preview_svg_valid_deterministic_and_responsive() -> None:
    heavy = {
        "style.black_fill_ratio": 0.9,
        "style.line_taper": 0.9,
        "style.shadow_contrast": 0.85,
        "style.screentone_density": 0.7,
        "style.perspective_distortion": 0.8,
        "merge.IN": 0.2,
        "lora.ink_style.strength": 1.2,
    }
    light = {"style.black_fill_ratio": 0.1, "style.line_taper": 0.1, "style.shadow_contrast": 0.2}
    svg_a = render_candidate_preview_svg(heavy, seed=1, label="A&B<")
    svg_b = render_candidate_preview_svg(light, seed=1, label="B")
    root = ET.fromstring(svg_a)
    assert root.tag.endswith("svg")
    ET.fromstring(svg_b)
    assert svg_a == render_candidate_preview_svg(heavy, seed=1, label="A&B<")
    assert svg_a != svg_b
    assert render_candidate_preview_svg(heavy, seed=2) != render_candidate_preview_svg(heavy, seed=1)
    assert "A&amp;B&lt;" in svg_a
    small = render_candidate_preview_svg({}, width=128, height=96)
    assert 'width="128"' in small and 'height="96"' in small
