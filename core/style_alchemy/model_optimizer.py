"""階層別ブロックウェイトマージ＆LoRA重み ベイズ最適化エンジン。

美的選好スコア（Elo / Bradley-Terry）を目的関数として、

- 多次元連続パラメータ空間（絵柄パラメータ・階層マージ比率・LoRA 強度・LoRA Block Weight）
- ガウス過程回帰（RBF/ARD カーネル、Cholesky、対数周辺尤度最大化によるハイパーパラメータ推定）
- 獲得関数（Expected Improvement / Upper Confidence Bound / Probability of Improvement）
- 進化戦略（CMA-ES）と遺伝的アルゴリズム（トーナメント選択・BLX-α 交叉・ガウス突然変異・エリート保存）

を組み合わせ、「次にどのレシピを生成して比較すれば最小の試行回数で理想の絵柄に到達できるか」を提案する。
最適レシピはマージ実行スクリプト用設定・ローカル実行用 Python スクリプト・ComfyUI ノードグラフへ出力する。
NumPy と標準ライブラリのみで完結する。
"""

from __future__ import annotations

import json
import math
import string
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import slugify, stable_hash, utc_now_iso
from core.style_alchemy.comfy_export import build_txt2img_workflow
from core.style_alchemy.schemas import (
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

__all__ = [
    "STYLE_PARAMS",
    "STYLE_PARAM_DESCRIPTIONS",
    "ParameterDim",
    "ParameterSpace",
    "latin_hypercube",
    "RBFKernel",
    "GaussianProcess",
    "norm_cdf",
    "norm_pdf",
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "CMAES",
    "GeneticOptimizer",
    "Observation",
    "OptimizationSuggestion",
    "BayesianStyleOptimizer",
    "RecipeExporter",
    "lora_key",
]

STYLE_PARAMS: tuple[str, ...] = (
    "style.line_taper",
    "style.black_fill_ratio",
    "style.shadow_contrast",
    "style.screentone_density",
    "style.perspective_distortion",
)

STYLE_PARAM_DESCRIPTIONS: dict[str, str] = {
    "style.line_taper": "線の入り抜き強度（0=均一な線幅 / 1=強い入り抜き）",
    "style.black_fill_ratio": "ベタ比率（画面に占める黒ベタ面積の割合）",
    "style.shadow_contrast": "影のコントラスト（0=柔らかい階調 / 1=硬いアニメ影）",
    "style.screentone_density": "スクリーントーン密度（網点の量と細かさ）",
    "style.perspective_distortion": "パース歪み（広角的な誇張の強さ）",
}


def lora_key(name: str) -> str:
    """LoRA 名からパラメータキー（拡張子を除いたファイル名）を得る。"""
    stem = Path(name).stem
    return stem or name


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------------------
# パラメータ空間
# ---------------------------------------------------------------------------


class ParameterDim(_Base):
    """連続パラメータ1次元。"""

    name: str = Field(..., min_length=1)
    low: float
    high: float
    default: float
    group: str = "style"
    description: str = ""

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ParameterDim":
        if not self.high > self.low:
            raise ValueError(f"{self.name}: high は low より大きい必要があります")
        if not self.low - 1e-12 <= self.default <= self.high + 1e-12:
            raise ValueError(f"{self.name}: default {self.default} が範囲 [{self.low}, {self.high}] 外です")
        return self


def latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """単位超立方体上のラテン超方格サンプル（各次元を n 層に1点ずつ）。"""
    if n <= 0:
        return np.zeros((0, dim))
    samples = np.empty((n, dim), dtype=float)
    for j in range(dim):
        samples[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return samples


class ParameterSpace(_Base):
    """最適化対象の多次元連続パラメータ空間。"""

    dims: list[ParameterDim] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> "ParameterSpace":
        names = [dim.name for dim in self.dims]
        if len(names) != len(set(names)):
            raise ValueError("パラメータ名が重複しています")
        return self

    @property
    def dim_count(self) -> int:
        return len(self.dims)

    def names(self) -> list[str]:
        return [dim.name for dim in self.dims]

    def dim(self, name: str) -> ParameterDim:
        for dim in self.dims:
            if dim.name == name:
                return dim
        raise KeyError(name)

    def index(self, name: str) -> int:
        for k, dim in enumerate(self.dims):
            if dim.name == name:
                return k
        raise KeyError(name)

    def bounds(self) -> np.ndarray:
        return np.array([[dim.low, dim.high] for dim in self.dims], dtype=float)

    def groups(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for dim in self.dims:
            grouped.setdefault(dim.group, []).append(dim.name)
        return grouped

    def _lows_spans(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = self.bounds()
        return bounds[:, 0], bounds[:, 1] - bounds[:, 0]

    def to_unit(self, x: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
        """実値 → 単位空間 [0,1]^d（dict は欠落キーを既定値で補完）。"""
        lows, spans = self._lows_spans()
        if isinstance(x, Mapping):
            arr = np.array([float(x.get(dim.name, dim.default)) for dim in self.dims], dtype=float)
        else:
            arr = np.asarray(x, dtype=float)
        return np.clip((arr - lows) / spans, 0.0, 1.0)

    def from_unit(self, u: Sequence[float] | np.ndarray) -> dict[str, float]:
        """単位空間 → 実値 dict。"""
        lows, spans = self._lows_spans()
        values = lows + np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * spans
        return {dim.name: round(float(values[k]), 6) for k, dim in enumerate(self.dims)}

    def clip_params(self, params: Mapping[str, float]) -> dict[str, float]:
        """欠落を既定値で補い、範囲外を境界へ丸めた dict（未知キーは除外）。"""
        result: dict[str, float] = {}
        for dim in self.dims:
            value = float(params.get(dim.name, dim.default))
            if not math.isfinite(value):
                value = dim.default
            result[dim.name] = round(min(max(value, dim.low), dim.high), 6)
        return result

    def sample_lhs_unit(self, n: int, seed: int = 0) -> np.ndarray:
        return latin_hypercube(n, self.dim_count, np.random.default_rng(seed))

    def sample_lhs(self, n: int, seed: int = 0) -> list[dict[str, float]]:
        """ラテン超方格による n 点の実値サンプル。"""
        return [self.from_unit(row) for row in self.sample_lhs_unit(n, seed)]

    def default_params(self) -> dict[str, float]:
        return {dim.name: float(dim.default) for dim in self.dims}

    def default_unit(self) -> np.ndarray:
        return self.to_unit(self.default_params())

    @classmethod
    def for_style_lab(
        cls,
        architecture: ModelArchitecture | str,
        lora_names: Iterable[str] = (),
        grouping: Literal["grouped", "per_block"] = "grouped",
        include_style_params: bool = True,
    ) -> "ParameterSpace":
        """絵柄研究ラボ標準の空間（絵柄5次元＋階層マージ＋LoRA 強度・階層ウェイト）。"""
        arch = ModelArchitecture(architecture)
        if grouping not in ("grouped", "per_block"):
            raise ValueError(f"未知の grouping です: {grouping}")
        dims: list[ParameterDim] = []
        if include_style_params:
            for name in STYLE_PARAMS:
                dims.append(
                    ParameterDim(name=name, low=0.0, high=1.0, default=0.5, group="style", description=STYLE_PARAM_DESCRIPTIONS[name])
                )
        if grouping == "grouped":
            for group, members in block_groups(arch).items():
                dims.append(
                    ParameterDim(
                        name=f"merge.{group}", low=0.0, high=1.0, default=0.5, group="merge",
                        description=f"{group} 階層（{len(members)}ブロック）における model_b の比率",
                    )
                )
        else:
            for block in ARCHITECTURE_BLOCKS[arch]:
                dims.append(
                    ParameterDim(
                        name=f"merge.{block}", low=0.0, high=1.0, default=0.5, group="merge",
                        description=f"ブロック {block} における model_b の比率",
                    )
                )
        seen_keys: set[str] = set()
        for raw_name in lora_names:
            key = lora_key(raw_name)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            dims.append(
                ParameterDim(
                    name=f"lora.{key}.strength", low=0.0, high=1.5, default=0.8, group="lora",
                    description=f"LoRA {key} の適用強度",
                )
            )
            if grouping == "grouped":
                lbw = set(LBW_BLOCKS[arch])
                for group, members in block_groups(arch).items():
                    if not lbw.intersection(members):
                        continue
                    dims.append(
                        ParameterDim(
                            name=f"lora.{key}.{group}", low=0.0, high=1.5, default=1.0, group="lora",
                            description=f"LoRA {key} の {group} 階層ウェイト（LBW）",
                        )
                    )
            else:
                for block in LBW_BLOCKS[arch]:
                    dims.append(
                        ParameterDim(
                            name=f"lora.{key}.{block}", low=0.0, high=1.5, default=1.0, group="lora",
                            description=f"LoRA {key} のブロック {block} ウェイト（LBW）",
                        )
                    )
        return cls(dims=dims)


# ---------------------------------------------------------------------------
# ガウス過程回帰
# ---------------------------------------------------------------------------


class RBFKernel:
    """RBF（二乗指数）カーネル k(a,b) = σ² exp(−½ Σ_k ((a_k−b_k)/ℓ_k)²)。ℓ がベクトルなら ARD。"""

    def __init__(self, length_scales: float | Sequence[float] | np.ndarray = 0.3, variance: float = 1.0) -> None:
        self.length_scales = np.atleast_1d(np.asarray(length_scales, dtype=float)).copy()
        if np.any(self.length_scales <= 0):
            raise ValueError("length_scales は正である必要があります")
        self.variance = float(variance)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(np.asarray(a, dtype=float)) / self.length_scales
        b = np.atleast_2d(np.asarray(b, dtype=float)) / self.length_scales
        sq = np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2.0 * a @ b.T
        return self.variance * np.exp(-0.5 * np.clip(sq, 0.0, None))

    def expanded(self, dim: int) -> "RBFKernel":
        """等方カーネルを dim 次元の ARD へ展開したコピー。"""
        scales = self.length_scales if self.length_scales.size == dim else np.full(dim, float(self.length_scales[0]))
        return RBFKernel(scales, self.variance)

    def copy(self) -> "RBFKernel":
        return RBFKernel(self.length_scales.copy(), self.variance)


class GaussianProcess:
    """ガウス過程回帰（ゼロ平均、y 正規化、Cholesky 分解、ジッタ自動増加）。"""

    def __init__(self, kernel: RBFKernel | None = None, noise: float = 1e-4, normalize_y: bool = True) -> None:
        self.kernel = kernel or RBFKernel()
        self.noise = float(noise)
        self.normalize_y = normalize_y
        self._x: np.ndarray | None = None
        self._y_raw: np.ndarray | None = None
        self._y_norm: np.ndarray | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._chol: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._alpha is not None

    @property
    def y_scale(self) -> float:
        return self._y_std

    @staticmethod
    def _cholesky(matrix: np.ndarray) -> np.ndarray:
        identity = np.eye(matrix.shape[0])
        jitter = 0.0
        for _ in range(10):
            try:
                return np.linalg.cholesky(matrix + jitter * identity)
            except np.linalg.LinAlgError:
                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
        repaired = (vectors * np.clip(values, 1e-9, None)) @ vectors.T
        return np.linalg.cholesky(repaired + 1e-9 * identity)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if x.shape[0] != y.shape[0] or x.shape[0] == 0:
            raise ValueError("GaussianProcess.fit: X と y の件数が不正です")
        if self.kernel.length_scales.size not in (1, x.shape[1]):
            raise ValueError("カーネルの長さスケール次元が入力次元と一致しません")
        self._x = x
        self._y_raw = y
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            std = float(np.std(y))
            self._y_std = std if std > 1e-12 else 1.0
        else:
            self._y_mean, self._y_std = 0.0, 1.0
        self._y_norm = (y - self._y_mean) / self._y_std
        gram = self.kernel(x, x) + self.noise * np.eye(x.shape[0])
        self._chol = self._cholesky(gram)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, self._y_norm))
        return self

    def predict(self, xs: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        if self._x is None or self._alpha is None or self._chol is None:
            raise RuntimeError("GaussianProcess は未学習です")
        xs = np.atleast_2d(np.asarray(xs, dtype=float))
        cross = self.kernel(xs, self._x)
        mean = cross @ self._alpha * self._y_std + self._y_mean
        if not return_std:
            return mean
        v = np.linalg.solve(self._chol, cross.T)
        variance = np.clip(self.kernel.variance - np.sum(v**2, axis=0), 1e-12, None)
        return mean, np.sqrt(variance) * self._y_std

    @staticmethod
    def _lml(x: np.ndarray, y_norm: np.ndarray, kernel: RBFKernel, noise: float) -> float:
        gram = kernel(x, x) + noise * np.eye(x.shape[0])
        try:
            chol = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            return -math.inf
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_norm))
        return float(-0.5 * y_norm @ alpha - np.sum(np.log(np.diag(chol))) - 0.5 * x.shape[0] * math.log(2.0 * math.pi))

    def log_marginal_likelihood(self) -> float:
        if self._x is None or self._y_norm is None:
            raise RuntimeError("GaussianProcess は未学習です")
        return self._lml(self._x, self._y_norm, self.kernel, self.noise)

    def optimize_hyperparameters(
        self,
        n_restarts: int = 8,
        seed: int = 0,
        length_bounds: tuple[float, float] = (0.03, 5.0),
        variance_bounds: tuple[float, float] = (0.05, 20.0),
        noise_bounds: tuple[float, float] = (1e-6, 0.5),
        max_evaluations: int = 300,
    ) -> dict[str, Any]:
        """対数空間の乱択多始点＋座標探索で対数周辺尤度を最大化し、再学習する。"""
        if self._x is None or self._y_norm is None:
            raise RuntimeError("optimize_hyperparameters の前に fit が必要です")
        x, y_norm = self._x, self._y_norm
        dim = x.shape[1]
        kernel = self.kernel.expanded(dim)
        lower = np.array([math.log(length_bounds[0])] * dim + [math.log(variance_bounds[0]), math.log(noise_bounds[0])])
        upper = np.array([math.log(length_bounds[1])] * dim + [math.log(variance_bounds[1]), math.log(noise_bounds[1])])

        def objective(theta: np.ndarray) -> float:
            candidate = RBFKernel(np.exp(theta[:dim]), math.exp(theta[dim]))
            return self._lml(x, y_norm, candidate, math.exp(theta[dim + 1]))

        current = np.clip(
            np.concatenate([np.log(kernel.length_scales), [math.log(kernel.variance), math.log(max(self.noise, 1e-9))]]),
            lower,
            upper,
        )
        rng = np.random.default_rng(seed)
        starts = [current] + [lower + rng.random(lower.size) * (upper - lower) for _ in range(max(0, n_restarts))]
        best_theta, best_value = current, objective(current)
        for start in starts:
            theta = start.copy()
            value = objective(theta)
            step = 0.6
            evaluations = 0
            while step > 0.03 and evaluations < max_evaluations:
                improved = False
                for k in range(theta.size):
                    for sign in (1.0, -1.0):
                        trial = theta.copy()
                        trial[k] = min(max(trial[k] + sign * step, lower[k]), upper[k])
                        if trial[k] == theta[k]:
                            continue
                        evaluations += 1
                        trial_value = objective(trial)
                        if trial_value > value + 1e-9:
                            theta, value = trial, trial_value
                            improved = True
                            break
                if not improved:
                    step *= 0.5
            if value > best_value:
                best_theta, best_value = theta, value
        self.kernel = RBFKernel(np.exp(best_theta[:dim]), math.exp(best_theta[dim]))
        self.noise = math.exp(best_theta[dim + 1])
        self.fit(x, self._y_raw if self._y_raw is not None else y_norm)
        return {
            "length_scales": [float(v) for v in self.kernel.length_scales],
            "variance": float(self.kernel.variance),
            "noise": float(self.noise),
            "log_marginal_likelihood": float(best_value),
        }


# ---------------------------------------------------------------------------
# 獲得関数
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_ERF = np.vectorize(math.erf, otypes=[float])


def _as_output(values: np.ndarray, scalar: bool) -> np.ndarray | float:
    return float(values.reshape(-1)[0]) if scalar else values


def norm_cdf(z: np.ndarray | float) -> np.ndarray | float:
    """標準正規分布の累積分布関数 Φ(z) = ½(1 + erf(z/√2))。"""
    arr = np.asarray(z, dtype=float)
    return _as_output(0.5 * (1.0 + _ERF(arr / _SQRT2)), arr.ndim == 0)


def norm_pdf(z: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(z, dtype=float)
    return _as_output(np.exp(-0.5 * arr**2) / _SQRT2PI, arr.ndim == 0)


def expected_improvement(mu: np.ndarray | float, sigma: np.ndarray | float, best: float, xi: float = 0.01) -> np.ndarray | float:
    """EI = (μ−f*−ξ)Φ(z) + σφ(z), z=(μ−f*−ξ)/σ（σ→0 では max(μ−f*−ξ, 0)）。"""
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    scalar = mu_arr.ndim == 0 and sigma_arr.ndim == 0
    mu_arr, sigma_arr = np.broadcast_arrays(np.atleast_1d(mu_arr), np.atleast_1d(sigma_arr))
    improvement = mu_arr - best - xi
    result = np.maximum(improvement, 0.0).astype(float)
    positive = sigma_arr > 1e-12
    if np.any(positive):
        z = improvement[positive] / sigma_arr[positive]
        cdf = 0.5 * (1.0 + _ERF(z / _SQRT2))
        pdf = np.exp(-0.5 * z**2) / _SQRT2PI
        result[positive] = improvement[positive] * cdf + sigma_arr[positive] * pdf
    return _as_output(np.maximum(result, 0.0), scalar)


def upper_confidence_bound(mu: np.ndarray | float, sigma: np.ndarray | float, kappa: float = 2.0) -> np.ndarray | float:
    """UCB = μ + κσ。"""
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    scalar = mu_arr.ndim == 0 and sigma_arr.ndim == 0
    return _as_output(np.atleast_1d(mu_arr + kappa * sigma_arr), scalar)


def probability_of_improvement(mu: np.ndarray | float, sigma: np.ndarray | float, best: float, xi: float = 0.01) -> np.ndarray | float:
    """PI = Φ((μ−f*−ξ)/σ)。"""
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    scalar = mu_arr.ndim == 0 and sigma_arr.ndim == 0
    mu_arr, sigma_arr = np.broadcast_arrays(np.atleast_1d(mu_arr), np.atleast_1d(sigma_arr))
    improvement = mu_arr - best - xi
    result = (improvement > 0).astype(float)
    positive = sigma_arr > 1e-12
    if np.any(positive):
        result[positive] = 0.5 * (1.0 + _ERF(improvement[positive] / sigma_arr[positive] / _SQRT2))
    return _as_output(result, scalar)


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------


class CMAES:
    """(μ/μ_w, λ)-CMA-ES（Hansen の標準形）。tell は値を最大化する。境界はクリップ＋ペナルティ。"""

    def __init__(
        self,
        mean: Sequence[float] | np.ndarray,
        sigma: float = 0.3,
        popsize: int | None = None,
        seed: int = 0,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> None:
        self.lower = float(lower)
        self.upper = float(upper)
        self.mean = np.clip(np.asarray(mean, dtype=float).copy(), self.lower, self.upper)
        self.dim = self.mean.size
        if self.dim == 0:
            raise ValueError("CMAES: 次元が0です")
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        n = self.dim
        self.popsize = int(popsize) if popsize else 4 + int(3 * math.log(n))
        self.mu = self.popsize // 2
        raw_weights = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()
        self.mueff = 1.0 / float(np.sum(self.weights**2))
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.inv_sqrt_c = np.eye(n)
        self.generation = 0
        self.best_x = self.mean.copy()
        self.best_value = -math.inf
        self._raw: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        """λ 個の候補（境界内へクリップ済み）を生成する。"""
        z = self.rng.standard_normal((self.popsize, self.dim))
        y = z @ (self.B * self.D).T
        raw = self.mean + self.sigma * y
        self._raw = raw
        return np.clip(raw, self.lower, self.upper)

    def tell(self, solutions: np.ndarray, values: Sequence[float] | np.ndarray) -> None:
        """評価値（大きいほど良い）で分布を更新する。"""
        solutions = np.atleast_2d(np.asarray(solutions, dtype=float))
        values = np.asarray(values, dtype=float).ravel()
        if solutions.shape[0] != values.size:
            raise ValueError("CMAES.tell: 解と評価値の件数が一致しません")
        if self._raw is not None and self._raw.shape == solutions.shape and np.allclose(
            np.clip(self._raw, self.lower, self.upper), solutions
        ):
            raw = self._raw
        else:
            raw = solutions
        clipped = np.clip(raw, self.lower, self.upper)
        distance = np.sum((raw - clipped) ** 2, axis=1)
        spread = float(np.ptp(values)) if values.size > 1 else 1.0
        penalized = values - (spread + 1e-9) * 10.0 * distance
        best_index = int(np.argmax(values))
        if values[best_index] > self.best_value:
            self.best_value = float(values[best_index])
            self.best_x = clipped[best_index].copy()
        order = np.argsort(-penalized)
        selected = raw[order[: self.mu]]
        old_mean = self.mean.copy()
        new_mean = self.weights @ selected
        shift = (new_mean - old_mean) / self.sigma
        n = self.dim
        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (self.inv_sqrt_c @ shift)
        norm_ps = float(np.linalg.norm(self.ps))
        hsig = norm_ps / math.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self.generation + 1))) / self.chi_n < 1.4 + 2.0 / (n + 1.0)
        self.pc = (1.0 - self.cc) * self.pc + (1.0 if hsig else 0.0) * math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * shift
        artmp = (selected - old_mean) / self.sigma
        rank_mu = artmp.T @ (artmp * self.weights[:, None])
        self.C = (
            (1.0 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (0.0 if hsig else 1.0) * self.cc * (2.0 - self.cc) * self.C)
            + self.cmu * rank_mu
        )
        self.sigma = max(1e-12, self.sigma * math.exp((self.cs / self.damps) * (norm_ps / self.chi_n - 1.0)))
        self.C = (self.C + self.C.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.clip(eigenvalues, 1e-20, None)
        self.D = np.sqrt(eigenvalues)
        self.B = eigenvectors
        self.inv_sqrt_c = (eigenvectors / self.D) @ eigenvectors.T
        self.mean = np.clip(new_mean, self.lower, self.upper)
        self.generation += 1
        self._raw = None

    def should_stop(self, tol: float = 1e-9) -> bool:
        return bool(self.sigma * float(np.max(self.D)) < tol or float(np.max(self.D) / np.min(self.D)) > 1e7)

    def optimize(self, func: Callable[[np.ndarray], np.ndarray], generations: int) -> tuple[np.ndarray, float]:
        """バッチ評価関数 func（(λ,d)→(λ,)）を generations 世代だけ最大化する。"""
        for _ in range(int(generations)):
            candidates = self.ask()
            self.tell(candidates, func(candidates))
            if self.should_stop():
                break
        return self.best_x.copy(), float(self.best_value)


# ---------------------------------------------------------------------------
# 遺伝的アルゴリズム
# ---------------------------------------------------------------------------


class GeneticOptimizer:
    """単位超立方体上の実数値GA（トーナメント選択・BLX-α 交叉・ガウス突然変異・エリート保存）。"""

    def __init__(
        self,
        dim: int,
        pop_size: int = 24,
        alpha: float = 0.3,
        mutation_rate: float = 0.15,
        mutation_sigma: float = 0.1,
        elite: int = 2,
        seed: int = 0,
        tournament_size: int = 3,
        initial: np.ndarray | None = None,
    ) -> None:
        if dim <= 0 or pop_size < 2:
            raise ValueError("GeneticOptimizer: dim>0 かつ pop_size>=2 が必要です")
        self.dim = int(dim)
        self.pop_size = int(pop_size)
        self.alpha = float(alpha)
        self.mutation_rate = float(mutation_rate)
        self.mutation_sigma = float(mutation_sigma)
        self.elite = max(0, min(int(elite), self.pop_size))
        self.tournament_size = max(2, int(tournament_size))
        self.rng = np.random.default_rng(seed)
        population = latin_hypercube(self.pop_size, self.dim, self.rng)
        if initial is not None:
            seeds = np.clip(np.atleast_2d(np.asarray(initial, dtype=float)), 0.0, 1.0)[: self.pop_size]
            population[: seeds.shape[0]] = seeds
        self.population = population
        self.fitness: np.ndarray | None = None
        self.best_x = population[0].copy()
        self.best_value = -math.inf
        self.generation = 0

    def _tournament(self) -> np.ndarray:
        assert self.fitness is not None
        size = min(self.tournament_size, self.pop_size)
        contenders = self.rng.choice(self.pop_size, size=size, replace=False)
        winner = contenders[int(np.argmax(self.fitness[contenders]))]
        return self.population[winner]

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        low = np.minimum(a, b)
        high = np.maximum(a, b)
        span = high - low
        return self.rng.uniform(low - self.alpha * span, high + self.alpha * span)

    def _mutate(self, child: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.dim) < self.mutation_rate
        if np.any(mask):
            child = child.copy()
            child[mask] += self.rng.normal(0.0, self.mutation_sigma, int(mask.sum()))
        return child

    def ask(self, n: int | None = None) -> np.ndarray:
        """評価すべき個体群（初回は初期集団、以降は子個体）を返す。"""
        count = int(n) if n else self.pop_size
        if self.fitness is None:
            if count <= self.pop_size:
                return self.population[:count].copy()
            extra = latin_hypercube(count - self.pop_size, self.dim, self.rng)
            return np.vstack([self.population, extra])
        children = [
            np.clip(self._mutate(self._crossover(self._tournament(), self._tournament())), 0.0, 1.0)
            for _ in range(count)
        ]
        return np.array(children)

    def tell(self, x: np.ndarray, y: Sequence[float] | np.ndarray) -> None:
        """評価済み個体を取り込み、エリート＋トーナメントで次世代集団を選抜する。"""
        x = np.clip(np.atleast_2d(np.asarray(x, dtype=float)), 0.0, 1.0)
        y = np.asarray(y, dtype=float).ravel()
        if self.fitness is None:
            pool_x, pool_y = x, y
        else:
            pool_x = np.vstack([self.population, x])
            pool_y = np.concatenate([self.fitness, y])
        order = list(np.argsort(-pool_y))
        selected = order[: self.elite]
        remaining = order[self.elite :]
        while len(selected) < min(self.pop_size, len(pool_y)) and remaining:
            size = min(self.tournament_size, len(remaining))
            positions = self.rng.choice(len(remaining), size=size, replace=False)
            winner = max(positions, key=lambda pos: pool_y[remaining[pos]])
            selected.append(remaining.pop(int(winner)))
        self.population = pool_x[selected]
        self.fitness = pool_y[selected]
        if self.population.shape[0] < self.pop_size:
            pad = self.pop_size - self.population.shape[0]
            self.population = np.vstack([self.population, latin_hypercube(pad, self.dim, self.rng)])
            self.fitness = np.concatenate([self.fitness, np.full(pad, -math.inf)])
        top = int(np.argmax(pool_y))
        if pool_y[top] > self.best_value:
            self.best_value = float(pool_y[top])
            self.best_x = pool_x[top].copy()
        self.generation += 1

    def optimize(self, func: Callable[[np.ndarray], np.ndarray], generations: int) -> tuple[np.ndarray, float]:
        """バッチ評価関数 func を generations 世代だけ最大化する。"""
        initial = self.ask()
        self.tell(initial, func(initial))
        for _ in range(int(generations)):
            children = self.ask()
            self.tell(children, func(children))
        return self.best_x.copy(), float(self.best_value)


# ---------------------------------------------------------------------------
# ベイズ最適化
# ---------------------------------------------------------------------------


class Observation(_Base):
    """観測（レシピのパラメータと選好スコア）。"""

    params: dict[str, float]
    score: float
    candidate_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class OptimizationSuggestion(_Base):
    """次に生成・比較すべきレシピ候補。"""

    params: dict[str, float]
    predicted_mean: float
    predicted_std: float
    acquisition: float
    strategy: str
    rationale: str
    rank: int


_STRATEGY_LABELS = {
    "gp": "GP＋獲得関数の多始点局所探索",
    "cmaes": "CMA-ES による GP-UCB 最大化",
    "genetic": "GP 代理モデル上の遺伝的アルゴリズム",
    "lhs": "ラテン超方格による初期探索",
}


class BayesianStyleOptimizer:
    """美的選好スコアを目的関数とするベイズ最適化器。"""

    def __init__(
        self,
        space: ParameterSpace,
        seed: int = 0,
        acquisition: Literal["ei", "ucb", "pi"] = "ei",
        xi: float = 0.01,
        kappa: float = 2.0,
        hyperopt_restarts: int = 3,
    ) -> None:
        if acquisition not in ("ei", "ucb", "pi"):
            raise ValueError(f"未知の獲得関数です: {acquisition}")
        self.space = space
        self.seed = int(seed)
        self.acquisition = acquisition
        self.xi = float(xi)
        self.kappa = float(kappa)
        self.hyperopt_restarts = int(hyperopt_restarts)
        self._observations: list[Observation] = []
        self._model: GaussianProcess | None = None
        self._model_size = -1

    # -- 観測 ----------------------------------------------------------------

    def add_observation(self, params: Mapping[str, float], score: float, candidate_id: str | None = None) -> Observation:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("score は有限値である必要があります")
        observation = Observation(params=self.space.clip_params(params), score=value, candidate_id=candidate_id)
        self._observations.append(observation)
        self._model = None
        return observation

    @property
    def observations(self) -> list[Observation]:
        return list(self._observations)

    def best(self) -> Observation | None:
        if not self._observations:
            return None
        return max(self._observations, key=lambda o: o.score)

    def _training(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([self.space.to_unit(o.params) for o in self._observations], dtype=float)
        y = np.array([o.score for o in self._observations], dtype=float)
        return x, y

    def fit_model(self) -> GaussianProcess:
        """観測に GP を学習（観測3件以上でハイパーパラメータ最適化）。"""
        if self._model is not None and self._model_size == len(self._observations):
            return self._model
        if not self._observations:
            raise RuntimeError("観測がありません")
        x, y = self._training()
        model = GaussianProcess(RBFKernel(np.full(self.space.dim_count, 0.35), 1.0), noise=1e-3, normalize_y=True)
        model.fit(x, y)
        if len(self._observations) >= 3:
            model.optimize_hyperparameters(n_restarts=self.hyperopt_restarts, seed=self.seed + len(self._observations))
        self._model = model
        self._model_size = len(self._observations)
        return model

    def posterior(self, params: Mapping[str, float]) -> tuple[float, float]:
        """事後平均と標準偏差（観測0件では (0, 1)）。"""
        if not self._observations:
            return 0.0, 1.0
        model = self.fit_model()
        mean, std = model.predict(self.space.to_unit(params)[None, :])
        return float(mean[0]), float(std[0])

    def sensitivity(self) -> dict[str, float]:
        """各次元の感度（GP 長さスケールの逆数を正規化）。"""
        names = self.space.names()
        if len(self._observations) < 2:
            return {name: 1.0 / len(names) for name in names}
        model = self.fit_model()
        scales = model.kernel.expanded(len(names)).length_scales
        inverse = 1.0 / scales
        total = float(inverse.sum())
        return {name: float(inverse[k] / total) for k, name in enumerate(names)}

    # -- 獲得関数 ------------------------------------------------------------

    def _acquire(self, model: GaussianProcess, points: np.ndarray, best: float) -> np.ndarray:
        mean, std = model.predict(points)
        if self.acquisition == "ucb":
            return np.asarray(upper_confidence_bound(mean, std, self.kappa), dtype=float)
        xi = self.xi * model.y_scale
        if self.acquisition == "pi":
            return np.asarray(probability_of_improvement(mean, std, best, xi), dtype=float)
        return np.asarray(expected_improvement(mean, std, best, xi), dtype=float)

    def _ucb_surrogate(self, model: GaussianProcess) -> Callable[[np.ndarray], np.ndarray]:
        def surrogate(points: np.ndarray) -> np.ndarray:
            mean, std = model.predict(points)
            return mean + self.kappa * std

        return surrogate

    # -- 提案 ----------------------------------------------------------------

    def _propose_gp(self, model: GaussianProcess, best: float, rng: np.random.Generator) -> list[np.ndarray]:
        dim = self.space.dim_count
        pool = latin_hypercube(512, dim, rng)
        values = self._acquire(model, pool, best)
        starts = pool[np.argsort(-values)[:5]]
        proposals: list[np.ndarray] = []
        for start in starts:
            point = start.copy()
            value = float(self._acquire(model, point[None, :], best)[0])
            step = 0.1
            for _ in range(40):
                if step < 0.0125:
                    break
                neighbours = np.repeat(point[None, :], 2 * dim, axis=0)
                for k in range(dim):
                    neighbours[2 * k, k] += step
                    neighbours[2 * k + 1, k] -= step
                neighbours = np.clip(neighbours, 0.0, 1.0)
                scores = self._acquire(model, neighbours, best)
                top = int(np.argmax(scores))
                if scores[top] > value + 1e-12:
                    point, value = neighbours[top], float(scores[top])
                else:
                    step /= 2.0
            proposals.append(point)
        return proposals

    def _propose_cmaes(self, model: GaussianProcess, start: np.ndarray, seed: int) -> list[np.ndarray]:
        es = CMAES(mean=start, sigma=0.25, seed=seed)
        best_x, _ = es.optimize(self._ucb_surrogate(model), generations=25)
        return [best_x, np.clip(es.mean, 0.0, 1.0)]

    def _propose_genetic(self, model: GaussianProcess, best: float, seeds: np.ndarray, seed: int) -> list[np.ndarray]:
        ga = GeneticOptimizer(self.space.dim_count, pop_size=24, seed=seed, initial=seeds)
        best_x, _ = ga.optimize(lambda pts: self._acquire(model, pts, best), generations=12)
        return [best_x]

    def _rationale(self, strategy: str, params: dict[str, float], mean: float, std: float, acquisition: float, best: Observation | None) -> str:
        label = _STRATEGY_LABELS.get(strategy, strategy)
        acq_name = {"ei": "期待改善量EI", "ucb": "上側信頼限界UCB", "pi": "改善確率PI"}[self.acquisition]
        text = f"{label}による提案。{acq_name}={acquisition:.4g}、予測スコア {mean:.1f} ± {std:.1f}"
        if best is None:
            return text + "。"
        text += f"（現在の最良 {best.score:.1f}）"
        deltas = sorted(
            ((name, params[name] - best.params.get(name, params[name])) for name in params),
            key=lambda item: -abs(item[1]),
        )[:3]
        changes = [f"{name} {best.params.get(name, 0.0):.2f}→{params[name]:.2f}" for name, delta in deltas if abs(delta) > 1e-3]
        if changes:
            text += "。最良観測からの主な変更: " + "、".join(changes)
        sensitivity = self.sensitivity()
        top_sensitive = sorted(sensitivity, key=lambda name: -sensitivity[name])[:2]
        return text + f"。感度上位: {'、'.join(top_sensitive)}。"

    def suggest(self, n: int = 3, strategy: Literal["gp", "cmaes", "genetic", "hybrid"] = "hybrid") -> list[OptimizationSuggestion]:
        """次に試すべき n 件のパラメータを提案する（n>1 は Kriging believer で多様化）。"""
        if n < 1:
            raise ValueError("n は1以上が必要です")
        if strategy not in ("gp", "cmaes", "genetic", "hybrid"):
            raise ValueError(f"未知の戦略です: {strategy}")
        best_obs = self.best()
        if len(self._observations) < 2:
            unit = self.space.sample_lhs_unit(n, seed=self.seed + len(self._observations))
            base_mean = float(np.mean([o.score for o in self._observations])) if self._observations else 0.0
            return [
                OptimizationSuggestion(
                    params=self.space.from_unit(row),
                    predicted_mean=base_mean,
                    predicted_std=0.0,
                    acquisition=0.0,
                    strategy="lhs",
                    rationale="観測が2件未満のため、ラテン超方格でパラメータ空間を均等に初期探索します。",
                    rank=k + 1,
                )
                for k, row in enumerate(unit)
            ]
        model = self.fit_model()
        x_train, y_train = self._training()
        best_value = float(y_train.max())
        rng = np.random.default_rng(stable_hash(self.seed, len(self._observations), strategy) % (2**32))
        believer = model
        believer_x, believer_y = x_train.copy(), y_train.copy()
        chosen: list[tuple[np.ndarray, str, float]] = []
        min_distance = 0.04 * math.sqrt(self.space.dim_count)
        for k in range(n):
            pool: list[tuple[np.ndarray, str]] = []
            start = believer_x[int(np.argmax(believer_y))]
            if strategy in ("gp", "hybrid"):
                pool.extend((p, "gp") for p in self._propose_gp(believer, best_value, rng))
            if strategy in ("cmaes", "hybrid"):
                pool.extend((p, "cmaes") for p in self._propose_cmaes(believer, start, self.seed + 31 * k))
            if strategy in ("genetic", "hybrid"):
                top_seeds = believer_x[np.argsort(-believer_y)[:6]]
                pool.extend((p, "genetic") for p in self._propose_genetic(believer, best_value, top_seeds, self.seed + 57 * k))
            points = np.array([p for p, _ in pool])
            scores = self._acquire(believer, points, best_value)
            existing = np.vstack([believer_x] + [c[0][None, :] for c in chosen]) if chosen else believer_x
            distances = np.min(np.linalg.norm(points[:, None, :] - existing[None, :, :], axis=2), axis=1)
            order = list(np.argsort(-scores))
            pick = next((i for i in order if distances[i] >= min_distance), None)
            if pick is None:
                fallback = latin_hypercube(256, self.space.dim_count, rng)
                fb_dist = np.min(np.linalg.norm(fallback[:, None, :] - existing[None, :, :], axis=2), axis=1)
                fb_scores = self._acquire(believer, fallback, best_value)
                feasible = np.where(fb_dist >= min_distance)[0]
                index = int(feasible[np.argmax(fb_scores[feasible])]) if feasible.size else int(np.argmax(fb_dist))
                point, source, value = fallback[index], "gp", float(fb_scores[index])
            else:
                point, source, value = points[pick], pool[pick][1], float(scores[pick])
            chosen.append((point, source, value))
            fantasy = float(np.asarray(believer.predict(point[None, :], return_std=False)).ravel()[0])
            believer_x = np.vstack([believer_x, point[None, :]])
            believer_y = np.append(believer_y, fantasy)
            believer = GaussianProcess(model.kernel.copy(), noise=model.noise, normalize_y=True).fit(believer_x, believer_y)
        suggestions: list[OptimizationSuggestion] = []
        means, stds = model.predict(np.array([c[0] for c in chosen]))
        for rank, ((point, source, value), mean, std) in enumerate(zip(chosen, means, stds), start=1):
            params = self.space.from_unit(point)
            suggestions.append(
                OptimizationSuggestion(
                    params=params,
                    predicted_mean=float(mean),
                    predicted_std=float(std),
                    acquisition=float(value),
                    strategy=source,
                    rationale=self._rationale(source, params, float(mean), float(std), float(value), best_obs),
                    rank=rank,
                )
            )
        return suggestions


# ---------------------------------------------------------------------------
# レシピエクスポーター
# ---------------------------------------------------------------------------

_DEFAULT_NEGATIVE = ["lowres", "blurry", "jpeg artifacts", "watermark", "signature", "bad anatomy", "extra fingers"]

_MERGE_SCRIPT_TEMPLATE = string.Template(
    '''#!/usr/bin/env python3
"""Creative Asset Foundry 自動生成: 階層別ブロックマージ実行スクリプト。

レシピ: $recipe_name
アーキテクチャ: $architecture
方式: $method
MBW: $mbw

使い方:
    python merge_recipe.py --output merged.safetensors --dtype fp16

必要パッケージ: torch, safetensors
LoRA は焼き込まず、同時にエクスポートされる ComfyUI ワークフローで推論時に適用する。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CONFIG = json.loads($config_literal)

ARCHITECTURE = CONFIG["architecture"]
KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "net.", "model.")
TEXT_ENCODER_PREFIXES = ("conditioner.", "cond_stage_model.", "text_encoders.", "clip.", "te.")
VAE_PREFIXES = ("first_stage_model.", "vae.")


def strip_prefix(key: str) -> str:
    for prefix in KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def block_of(key: str) -> str:
    """state_dict のキーを正準ブロック名へ写像する。"""
    if key.startswith(VAE_PREFIXES):
        return "VAE"
    if key.startswith(TEXT_ENCODER_PREFIXES):
        return "BASE"
    body = strip_prefix(key)
    if ARCHITECTURE in ("sdxl", "sd15"):
        match = re.match(r"input_blocks\\.(\\d+)\\.", body)
        if match:
            return f"IN{int(match.group(1)):02d}"
        if body.startswith("middle_block."):
            return "MID"
        match = re.match(r"output_blocks\\.(\\d+)\\.", body)
        if match:
            return f"OUT{int(match.group(1)):02d}"
        return "BASE"
    if ARCHITECTURE == "flux":
        match = re.match(r"double_blocks\\.(\\d+)\\.", body)
        if match:
            return f"DB{int(match.group(1)):02d}"
        match = re.match(r"single_blocks\\.(\\d+)\\.", body)
        if match:
            return f"SB{int(match.group(1)):02d}"
        return "BASE"
    match = re.match(r"blocks\\.(\\d+)\\.", body)
    if match:
        return f"B{int(match.group(1)):02d}"
    return "BASE"


def weight_for(block: str) -> float:
    """model_b の比率（VAE は常に model_a を保持）。"""
    if block == "VAE":
        return 0.0
    return float(CONFIG["block_weights"].get(block, CONFIG["default_weight"]))


def resolve(model: dict) -> str:
    return str(Path(model.get("path") or model["name"]).expanduser())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CAF 階層別ブロックマージ")
    parser.add_argument("--output", default=CONFIG["output_name"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args(argv)
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError:
        print("torch と safetensors が必要です: python -m pip install torch safetensors", file=sys.stderr)
        return 1
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    models = CONFIG["models"]
    state_a = load_file(resolve(models["model_a"]), device=args.device)
    state_b = load_file(resolve(models["model_b"]), device=args.device)
    state_c = load_file(resolve(models["model_c"]), device=args.device) if models.get("model_c") else None
    merged = {}
    touched = {}
    for key, tensor_a in state_a.items():
        block = block_of(key)
        weight = weight_for(block)
        tensor_b = state_b.get(key)
        if not tensor_a.is_floating_point():
            merged[key] = tensor_a
            continue
        if tensor_b is None or tensor_b.shape != tensor_a.shape or weight == 0.0:
            merged[key] = tensor_a.to(dtype)
            continue
        if CONFIG["method"] == "add_difference" and state_c is not None and key in state_c and state_c[key].shape == tensor_a.shape:
            result = tensor_a.float() + weight * (tensor_b.float() - state_c[key].float())
        else:
            result = (1.0 - weight) * tensor_a.float() + weight * tensor_b.float()
        merged[key] = result.to(dtype)
        touched[block] = touched.get(block, 0) + 1
    metadata = {"caf_recipe": CONFIG["recipe_name"], "caf_mbw": CONFIG["mbw_string"], "caf_method": CONFIG["method"]}
    save_file(merged, args.output, metadata=metadata)
    summary = ", ".join(f"{block}:{count}" for block, count in sorted(touched.items()))
    print(f"保存しました: {args.output}（マージしたテンソル数 {summary}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
)


class RecipeExporter:
    """最適化パラメータ ⇄ StyleRecipe、マージ設定・スクリプト・ComfyUI ワークフローへの出力。"""

    def __init__(self, default_sampler: SamplerSettings | None = None, negative_tokens: Sequence[str] | None = None) -> None:
        self.default_sampler = default_sampler or SamplerSettings()
        self.negative_tokens = list(negative_tokens) if negative_tokens is not None else list(_DEFAULT_NEGATIVE)

    # -- パラメータ → レシピ ---------------------------------------------------

    @staticmethod
    def style_tokens_for(params: Mapping[str, float]) -> list[str]:
        """絵柄パラメータから生成プロンプト用スタイルトークンを導出する。"""
        tokens = ["monochrome", "manga"]
        taper = params.get("style.line_taper")
        if taper is not None:
            tokens.append("tapered ink lines, variable line weight" if taper >= 0.6 else "uniform clean line weight" if taper <= 0.3 else "inked lineart")
        black = params.get("style.black_fill_ratio")
        if black is not None:
            if black >= 0.6:
                tokens.append("heavy black fill, strong contrast")
            elif black <= 0.25:
                tokens.append("light linework, minimal black fill")
        shadow = params.get("style.shadow_contrast")
        if shadow is not None:
            if shadow >= 0.6:
                tokens.append("hard cel shadows, high contrast lighting")
            elif shadow <= 0.3:
                tokens.append("soft gradient shading")
        tone = params.get("style.screentone_density")
        if tone is not None and tone >= 0.5:
            tokens.append("screentone shading, halftone dots")
        perspective = params.get("style.perspective_distortion")
        if perspective is not None:
            if perspective >= 0.6:
                tokens.append("dramatic perspective, foreshortening")
            elif perspective <= 0.2:
                tokens.append("flat orthographic composition")
        return tokens

    @staticmethod
    def _merge_weights(arch: ModelArchitecture, params: Mapping[str, float], default: float) -> dict[str, float]:
        raw = {key[len("merge."):]: float(value) for key, value in params.items() if key.startswith("merge.")}
        expanded = expand_group_weights(arch, raw, default=default)
        return {block: round(min(max(weight, 0.0), 1.0), 6) for block, weight in expanded.items()}

    @staticmethod
    def _lora_from_params(spec: LoRASpec, arch: ModelArchitecture, params: Mapping[str, float]) -> LoRASpec:
        key = lora_key(spec.name)
        prefix = f"lora.{key}."
        relevant = {k[len(prefix):]: float(v) for k, v in params.items() if k.startswith(prefix)}
        update: dict[str, Any] = {}
        if "strength" in relevant:
            strength = round(min(max(relevant.pop("strength"), -2.0), 2.0), 6)
            update["strength_model"] = strength
            update["strength_clip"] = strength
        if relevant:
            lbw = set(LBW_BLOCKS[arch])
            weights: dict[str, float] = dict(spec.block_weights or {})
            groups = block_groups(arch)
            for name, value in relevant.items():
                if name in groups:
                    for block in groups[name]:
                        if block in lbw:
                            weights[block] = value
            for name, value in relevant.items():
                if name in lbw:
                    weights[name] = value
            if weights:
                update["block_weights"] = {block: round(min(max(w, -2.0), 2.0), 6) for block, w in weights.items()}
        return spec.model_copy(update=update) if update else spec.model_copy()

    def params_to_recipe(
        self,
        params: Mapping[str, float],
        *,
        architecture: ModelArchitecture | str,
        model_a: CheckpointRef,
        model_b: CheckpointRef,
        loras: Sequence[LoRASpec] = (),
        name: str | None = None,
        model_c: CheckpointRef | None = None,
        default_merge_weight: float = 0.5,
        sampler: SamplerSettings | None = None,
        vae: str | None = None,
    ) -> StyleRecipe:
        """最適化パラメータから StyleRecipe を構築する。"""
        arch = ModelArchitecture(architecture)
        ref_a = model_a.model_copy(update={"architecture": arch})
        ref_b = model_b.model_copy(update={"architecture": arch})
        ref_c = model_c.model_copy(update={"architecture": arch}) if model_c is not None else None
        merge = BlockMergeRecipe(
            architecture=arch,
            method=MergeMethod.ADD_DIFFERENCE if ref_c is not None else MergeMethod.WEIGHTED_SUM,
            model_a=ref_a,
            model_b=ref_b,
            model_c=ref_c,
            block_weights=self._merge_weights(arch, params, default_merge_weight),
            default_weight=default_merge_weight,
        )
        lora_specs = [self._lora_from_params(spec, arch, params) for spec in loras]
        numeric = {key: round(float(value), 6) for key, value in params.items() if isinstance(value, (int, float))}
        digest = stable_hash(json.dumps(numeric, sort_keys=True)) % 16**6
        return StyleRecipe(
            name=name or f"optimized recipe {digest:06x}",
            architecture=arch,
            base_checkpoint=ref_a,
            merge=merge,
            loras=lora_specs,
            sampler=(sampler or self.default_sampler).model_copy(),
            vae=vae,
            style_tokens=self.style_tokens_for(params),
            negative_tokens=list(self.negative_tokens),
            params=numeric,
            notes="BayesianStyleOptimizer の提案パラメータから生成",
        )

    def apply_params(
        self,
        recipe: StyleRecipe,
        params: Mapping[str, float],
        sampler_overrides: Mapping[str, Any] | None = None,
    ) -> StyleRecipe:
        """既存レシピにパラメータ（merge./lora./style.）とサンプラー上書きを適用した派生レシピ。"""
        arch = recipe.architecture
        update: dict[str, Any] = {}
        if recipe.merge is not None and any(key.startswith("merge.") for key in params):
            weights = dict(recipe.merge.block_weights)
            weights.update(self._merge_weights(arch, {k: v for k, v in params.items() if k.startswith("merge.")}, recipe.merge.default_weight))
            update["merge"] = recipe.merge.model_copy(update={"block_weights": weights})
        if recipe.loras:
            update["loras"] = [self._lora_from_params(spec, arch, params) for spec in recipe.loras]
        if any(key.startswith("style.") for key in params):
            merged_params = {**recipe.params, **{k: float(v) for k, v in params.items()}}
            update["params"] = merged_params
            update["style_tokens"] = self.style_tokens_for(merged_params)
        if sampler_overrides:
            allowed = {k: v for k, v in sampler_overrides.items() if k in SamplerSettings.model_fields}
            if allowed:
                update["sampler"] = SamplerSettings.model_validate({**recipe.sampler.model_dump(), **allowed})
        derived = recipe.model_copy(update=update, deep=True)
        return StyleRecipe.model_validate(derived.model_dump())

    # -- 出力 ------------------------------------------------------------------

    def to_merge_config(self, recipe: StyleRecipe) -> dict[str, Any]:
        """マージ実行スクリプト用設定（正準順ブロック比率・MBW 文字列・LoRA LBW 文字列）。"""
        arch = recipe.architecture
        order = ARCHITECTURE_BLOCKS[arch]
        merge = recipe.merge

        def _model(ref: CheckpointRef | None) -> dict[str, str] | None:
            return None if ref is None else {"name": ref.name, "path": ref.path}

        config: dict[str, Any] = {
            "tool": "caf-merge",
            "version": 1,
            "recipe_id": recipe.id,
            "recipe_name": recipe.name,
            "architecture": arch.value,
            "block_order": list(order),
            "comfy_ratio_convention": "ComfyUI の ModelMerge 系 ratio は model1 の重み。本設定の比率 w は model_b の比率であり ComfyUI では 1-w を指定する。",
            "output_name": f"{slugify(recipe.name)}.safetensors",
            "loras": [
                {
                    "name": lora.name,
                    "path": lora.path,
                    "strength_model": lora.strength_model,
                    "strength_clip": lora.strength_clip,
                    "lbw_blocks": list(LBW_BLOCKS[arch]),
                    "lbw_string": lora.lbw_string(arch),
                    "trigger_words": list(lora.trigger_words),
                }
                for lora in recipe.loras
            ],
            "created_at": utc_now_iso(),
        }
        if merge is None:
            config.update(
                {
                    "method": "none",
                    "models": {"model_a": _model(recipe.base_checkpoint), "model_b": None, "model_c": None},
                    "block_weights": {},
                    "default_weight": 0.0,
                    "mbw_string": "",
                    "base_alpha": 0.0,
                    "group_means": {},
                }
            )
            return config
        config.update(
            {
                "method": merge.method.value,
                "models": {"model_a": _model(merge.model_a), "model_b": _model(merge.model_b), "model_c": _model(merge.model_c)},
                "block_weights": {block: merge.weight_for(block) for block in order},
                "default_weight": merge.default_weight,
                "mbw_string": merge.to_mbw_string(),
                "base_alpha": merge.weight_for("BASE"),
                "group_means": merge.group_means(),
            }
        )
        return config

    def to_merge_script(self, recipe: StyleRecipe) -> str:
        """ローカル実行用の階層別ブロックマージ Python スクリプト（torch + safetensors）。"""
        if recipe.merge is None:
            raise ValueError("このレシピにはマージ設定がありません")
        config = self.to_merge_config(recipe)
        script = _MERGE_SCRIPT_TEMPLATE.substitute(
            recipe_name=recipe.name.replace("\\", "/").replace('"""', "'''"),
            architecture=recipe.architecture.value,
            method=recipe.merge.method.value,
            mbw=config["mbw_string"],
            config_literal=repr(json.dumps(config, ensure_ascii=False)),
        )
        compile(script, "<caf-merge-script>", "exec")
        return script

    def to_comfy_workflow(
        self,
        recipe: StyleRecipe,
        positive: str,
        negative: str,
        seed: int,
        width: int,
        height: int,
        filename_prefix: str = "caf/recipe",
    ) -> dict[str, Any]:
        """ComfyUI API 形式の txt2img 検証ワークフロー。"""
        return build_txt2img_workflow(
            recipe, positive, negative, seed=seed, width=width, height=height, filename_prefix=filename_prefix
        )
