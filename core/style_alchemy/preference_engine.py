"""美的選好学習エンジン（Elo / Bradley-Terry / 美的選好ベクトル）。

生成画像の1対1ブラインド比較（A/Bテスト）の勝敗履歴から、

- Elo レーティング（K 減衰・引分対応、比較を時系列順に逐次適用）
- Bradley-Terry モデル強度（Hunter 2004 の MM アルゴリズムによる最尤推定。仮想平均プレイヤーとの
  事前仮想試合で正則化し、全勝・未勝利の候補が居ても有限値へ収束。標準誤差は Fisher 情報行列から算出）
- 美的選好ベクトル（実験パラメータを説明変数とした L2 正則化ペアワイズ・ロジスティック回帰、
  または BT 対数強度へのリッジ回帰。標準化係数・Permutation 重要度・ブートストラップ信頼区間）

を算出する。外部の機械学習フレームワークには依存せず、NumPy のみで完結する。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import new_id, stable_hash, utc_now_iso
from core.style_alchemy.schemas import StyleRecipe

__all__ = [
    "Outcome",
    "StyleCandidate",
    "Comparison",
    "RatingEntry",
    "EloEngine",
    "BradleyTerryResult",
    "BradleyTerryModel",
    "FeatureWeight",
    "AestheticPreferenceVector",
    "PreferenceAnalyzer",
    "LAMBDA_GRID",
    "ridge_fit",
    "ridge_loocv_mse",
    "logistic_fit",
    "rank_candidates",
    "suggest_next_pair",
    "spearman_correlation",
    "ratings_by_id",
]

Outcome = Literal["left", "right", "tie"]

LAMBDA_GRID: tuple[float, ...] = tuple(float(v) for v in np.logspace(-3, 2, 11))
_NEUTRAL_THRESHOLD = 0.05


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class StyleCandidate(_Base):
    """A/B 比較の対象となる絵柄候補（実験パラメータ・レシピ・生成画像参照）。"""

    id: str = Field(default_factory=lambda: new_id("cand"))
    label: str = ""
    params: dict[str, float] = Field(default_factory=dict)
    recipe: StyleRecipe | None = None
    image_ref: str | None = None
    source: str = "manual"
    created_at: str = Field(default_factory=utc_now_iso)


class Comparison(_Base):
    """1対1ブラインド比較の判定結果。"""

    id: str = Field(default_factory=lambda: new_id("cmp"))
    left_id: str
    right_id: str
    outcome: Outcome
    response_time_ms: float | None = None
    session_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _distinct(self) -> "Comparison":
        if self.left_id == self.right_id:
            raise ValueError("同一候補同士の比較はできません")
        return self

    def score_left(self) -> float:
        """左側の得点（勝ち=1, 負け=0, 引分=0.5）。"""
        if self.outcome == "left":
            return 1.0
        if self.outcome == "right":
            return 0.0
        return 0.5


class RatingEntry(_Base):
    """ランキング表の1行。"""

    candidate_id: str
    label: str = ""
    elo: float
    bt_strength: float
    bt_elo_scale: float
    bt_std_error: float
    games: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    rank: int


def _unique_ids(candidate_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in candidate_ids:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def _chronological(comparisons: Iterable[Comparison]) -> list[Comparison]:
    """created_at による安定ソート（同時刻は入力順を保持）。"""
    return sorted(list(comparisons), key=lambda c: c.created_at)


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


class EloEngine:
    """Elo レーティング。K_eff = K / (1 + games / k_decay_games)。"""

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
        scale: float = 400.0,
        k_decay_games: float | None = 30.0,
    ) -> None:
        if k_factor <= 0 or scale <= 0:
            raise ValueError("k_factor と scale は正である必要があります")
        self.k_factor = float(k_factor)
        self.initial_rating = float(initial_rating)
        self.scale = float(scale)
        self.k_decay_games = k_decay_games

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """A の期待得点 1 / (1 + 10^((rb − ra)/scale))。"""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / self.scale))

    def k_for(self, games: int) -> float:
        if not self.k_decay_games or self.k_decay_games <= 0:
            return self.k_factor
        return self.k_factor / (1.0 + max(0, games) / float(self.k_decay_games))

    def update(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float,
        games_a: int = 0,
        games_b: int = 0,
    ) -> tuple[float, float]:
        """1試合分の更新。score_a は A の得点（1/0.5/0）。"""
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        new_a = rating_a + self.k_for(games_a) * (score_a - expected_a)
        new_b = rating_b + self.k_for(games_b) * ((1.0 - score_a) - expected_b)
        return new_a, new_b

    def compute(self, candidate_ids: Iterable[str], comparisons: Iterable[Comparison]) -> dict[str, float]:
        """比較を時系列順に逐次適用したレーティング（未知IDを含む比較は無視）。"""
        ids = _unique_ids(candidate_ids)
        ratings = {cid: self.initial_rating for cid in ids}
        games = {cid: 0 for cid in ids}
        for comparison in _chronological(comparisons):
            left, right = comparison.left_id, comparison.right_id
            if left not in ratings or right not in ratings:
                continue
            ratings[left], ratings[right] = self.update(
                ratings[left], ratings[right], comparison.score_left(), games[left], games[right]
            )
            games[left] += 1
            games[right] += 1
        return {cid: round(value, 6) for cid, value in ratings.items()}


# ---------------------------------------------------------------------------
# Bradley-Terry
# ---------------------------------------------------------------------------


class BradleyTerryResult(_Base):
    """Bradley-Terry 推定結果（strengths は幾何平均1に正規化）。"""

    strengths: dict[str, float]
    log_strengths: dict[str, float]
    std_errors: dict[str, float]
    elo_scale: dict[str, float]
    log_likelihood: float
    iterations: int
    converged: bool


def _pair_tables(ids: Sequence[str], comparisons: Iterable[Comparison]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    """対戦数行列 N・勝点ベクトル W・(i, j, 左得点) のリスト。"""
    index = {cid: k for k, cid in enumerate(ids)}
    n = len(ids)
    games = np.zeros((n, n), dtype=float)
    wins = np.zeros(n, dtype=float)
    records: list[tuple[int, int, float]] = []
    for comparison in comparisons:
        i = index.get(comparison.left_id)
        j = index.get(comparison.right_id)
        if i is None or j is None:
            continue
        score = comparison.score_left()
        games[i, j] += 1.0
        games[j, i] += 1.0
        wins[i] += score
        wins[j] += 1.0 - score
        records.append((i, j, score))
    return games, wins, records


class BradleyTerryModel:
    """Bradley-Terry モデルの MM 最尤推定（仮想平均プレイヤーによる正則化付き）。

    Hunter (2004) の MM 更新 π_i ← W_i / Σ_j n_ij/(π_i+π_j) に、強度1の仮想平均プレイヤーとの
    ``prior_games`` 試合（勝敗半々）を加える。MM は線形収束のため、近傍（|Δlogπ|<1e-5）へ到達した後は
    同一目的関数（正則化対数尤度・厳密凹）の Newton 法で仕上げ、二次収束で tol まで詰める。
    """

    _MM_HANDOFF = 1e-5

    def __init__(self, max_iter: int = 2000, tol: float = 1e-10, prior_games: float = 1.0) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.prior_games = float(prior_games)

    @staticmethod
    def _information(pi: np.ndarray, games: np.ndarray, prior: float) -> np.ndarray:
        """log 強度に関する Fisher 情報行列（= 負のヘッセ行列）。"""
        prob = pi[:, None] / (pi[:, None] + pi[None, :])
        weight = games * prob * (1.0 - prob)
        info = -weight.copy()
        np.fill_diagonal(info, weight.sum(axis=1))
        if prior > 0:
            p_virtual = pi / (pi + 1.0)
            info[np.diag_indices_from(info)] += prior * p_virtual * (1.0 - p_virtual)
        return info

    def fit(self, candidate_ids: Iterable[str], comparisons: Iterable[Comparison]) -> BradleyTerryResult:
        ids = _unique_ids(candidate_ids)
        if not ids:
            return BradleyTerryResult(
                strengths={}, log_strengths={}, std_errors={}, elo_scale={},
                log_likelihood=0.0, iterations=0, converged=True,
            )
        games, wins, records = _pair_tables(ids, list(comparisons))
        prior = max(0.0, self.prior_games)
        pi = np.ones(len(ids), dtype=float)
        converged = False
        iterations = 0
        numerator = wins + prior / 2.0
        handoff = max(self.tol, self._MM_HANDOFF) if prior > 0 else self.tol
        for iterations in range(1, self.max_iter + 1):
            denominator = (games / (pi[:, None] + pi[None, :])).sum(axis=1) + prior / (pi + 1.0)
            updated = pi.copy()
            active = denominator > 0
            updated[active] = numerator[active] / denominator[active]
            updated = np.maximum(updated, 1e-12)
            if prior <= 0:
                updated /= math.exp(float(np.mean(np.log(updated))))
            delta = float(np.max(np.abs(np.log(updated) - np.log(pi))))
            pi = updated
            if delta < handoff:
                converged = delta < self.tol
                break
        if prior > 0 and not converged:
            theta = np.log(pi)
            for _ in range(100):
                iterations += 1
                current = np.exp(theta)
                prob = current[:, None] / (current[:, None] + current[None, :])
                p_virtual = current / (current + 1.0)
                gradient = wins - (games * prob).sum(axis=1) + prior * (0.5 - p_virtual)
                step = np.linalg.solve(self._information(current, games, prior), gradient)
                theta = theta + step
                if float(np.max(np.abs(step))) < self.tol:
                    converged = True
                    break
            pi = np.exp(theta)
        log_pi = np.log(pi)
        covariance = np.linalg.pinv(self._information(pi, games, prior))
        std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        # 対数尤度（データのみ）
        log_likelihood = 0.0
        for i, j, score in records:
            p_left = pi[i] / (pi[i] + pi[j])
            p_left = min(max(p_left, 1e-300), 1.0 - 1e-16)
            log_likelihood += score * math.log(p_left) + (1.0 - score) * math.log(1.0 - p_left)
        centered = log_pi - float(np.mean(log_pi))
        normalized = np.exp(centered)
        return BradleyTerryResult(
            strengths={cid: float(normalized[k]) for k, cid in enumerate(ids)},
            log_strengths={cid: float(centered[k]) for k, cid in enumerate(ids)},
            std_errors={cid: float(std_errors[k]) for k, cid in enumerate(ids)},
            elo_scale={cid: float(1500.0 + 400.0 * math.log10(normalized[k])) for k, cid in enumerate(ids)},
            log_likelihood=float(log_likelihood),
            iterations=int(iterations),
            converged=converged,
        )

    @staticmethod
    def predict_proba(result: BradleyTerryResult, left_id: str, right_id: str) -> float:
        """left が right に勝つ確率 π_l / (π_l + π_r)。"""
        left = result.strengths[left_id]
        right = result.strengths[right_id]
        return float(left / (left + right))


# ---------------------------------------------------------------------------
# 美的選好ベクトル
# ---------------------------------------------------------------------------


class FeatureWeight(_Base):
    """1つの実験パラメータに対する選好の重み。CI は標準化係数の単位。"""

    name: str
    coefficient: float
    std_coefficient: float
    importance: float = Field(..., ge=0.0, le=1.0)
    permutation_importance: float
    ci_low: float
    ci_high: float
    direction: Literal["prefer_higher", "prefer_lower", "neutral"]


class AestheticPreferenceVector(_Base):
    """無意識の美的選好を数値化したベクトル。"""

    features: list[FeatureWeight]
    intercept: float
    method: Literal["ridge_bt", "pairwise_logistic"]
    regularization: float
    fit_accuracy: float
    r2: float | None = None
    n_comparisons: int
    n_candidates: int
    created_at: str = Field(default_factory=utc_now_iso)

    def as_dict(self) -> dict[str, float]:
        """特徴量名 → 標準化係数。"""
        return {feature.name: feature.std_coefficient for feature in self.features}


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    constant = std < 1e-12
    safe = np.where(constant, 1.0, std)
    z = (matrix - mean) / safe
    z[:, constant] = 0.0
    return z, mean, safe, constant


def ridge_fit(z: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """切片を罰則化しないリッジ回帰 (w, intercept)。"""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    z_mean = z.mean(axis=0)
    zc = z - z_mean
    y_mean = float(y.mean())
    system = zc.T @ zc + lam * np.eye(z.shape[1])
    weights = np.linalg.solve(system, zc.T @ (y - y_mean))
    intercept = y_mean - float(z_mean @ weights)
    return weights, intercept


def ridge_loocv_mse(z: np.ndarray, y: np.ndarray, lam: float) -> float:
    """hat 行列による閉形式 leave-one-out 平均二乗誤差（切片非罰則）。"""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    n = z.shape[0]
    zc = z - z.mean(axis=0)
    system = zc.T @ zc + lam * np.eye(z.shape[1])
    hat = zc @ np.linalg.solve(system, zc.T) + 1.0 / n
    fitted = hat @ y
    residual = y - fitted
    leverage = np.clip(np.diag(hat), None, 1.0 - 1e-9)
    return float(np.mean((residual / (1.0 - leverage)) ** 2))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logistic_fit(design: np.ndarray, target: np.ndarray, lam: float, max_iter: int = 100, tol: float = 1e-9) -> np.ndarray:
    """L2 正則化ロジスティック回帰（切片なし）の Newton-IRLS。target は [0,1] の得点。"""
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    dim = design.shape[1]
    weights = np.zeros(dim, dtype=float)
    identity = np.eye(dim)
    for _ in range(max_iter):
        prob = _sigmoid(design @ weights)
        gradient = design.T @ (prob - target) + lam * weights
        curvature = prob * (1.0 - prob)
        hessian = design.T @ (design * curvature[:, None]) + lam * identity
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights = weights - step
        if float(np.linalg.norm(step)) < tol * (1.0 + float(np.linalg.norm(weights))):
            break
    return weights


def _logistic_loglik(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    logits = design @ weights
    log_p = -np.logaddexp(0.0, -logits)
    log_q = -np.logaddexp(0.0, logits)
    return float(np.sum(target * log_p + (1.0 - target) * log_q))


def _pairwise_accuracy(utilities: np.ndarray, pairs: np.ndarray, scores: np.ndarray) -> float:
    """非引分の比較について、効用差の符号が勝敗と一致する割合（効用同値は0.5）。"""
    mask = scores != 0.5
    if not np.any(mask):
        return 0.5
    diff = utilities[pairs[mask, 0]] - utilities[pairs[mask, 1]]
    observed = scores[mask]
    correct = np.where(diff == 0, 0.5, ((diff > 0) == (observed > 0.5)).astype(float))
    return float(np.mean(correct))


def spearman_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """順位相関係数（同順位は平均順位）。"""

    def _ranks(values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(len(arr), dtype=float)
        ranks[order] = np.arange(len(arr), dtype=float)
        for value in np.unique(arr):
            tie = arr == value
            if tie.sum() > 1:
                ranks[tie] = ranks[tie].mean()
        return ranks

    ra, rb = _ranks(a), _ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


class PreferenceAnalyzer:
    """実験パラメータと勝敗から美的選好ベクトルを推定する。"""

    def __init__(
        self,
        regularization: float | None = None,
        n_bootstrap: int = 200,
        seed: int = 0,
        method: Literal["ridge_bt", "pairwise_logistic"] = "pairwise_logistic",
        permutation_repeats: int = 5,
    ) -> None:
        if method not in ("ridge_bt", "pairwise_logistic"):
            raise ValueError(f"未知の推定法です: {method}")
        self.regularization = regularization
        self.n_bootstrap = max(0, int(n_bootstrap))
        self.seed = int(seed)
        self.method = method
        self.permutation_repeats = max(1, int(permutation_repeats))

    # -- 公開 API ----------------------------------------------------------

    def analyze(
        self,
        candidates: Sequence[StyleCandidate],
        comparisons: Sequence[Comparison],
        feature_names: Sequence[str] | None = None,
    ) -> AestheticPreferenceVector:
        unique: dict[str, StyleCandidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.id, candidate)
        cand_list = list(unique.values())
        names = list(feature_names) if feature_names is not None else self._common_features(cand_list)
        index = {c.id: k for k, c in enumerate(cand_list)}
        usable = [c for c in comparisons if c.left_id in index and c.right_id in index]
        if not names or not usable or len(cand_list) < 2:
            return self._neutral(names, len(usable), len(cand_list))
        raw = np.array([[float(c.params.get(name, 0.0)) for name in names] for c in cand_list], dtype=float)
        z, _, scale, constant = _standardize(raw)
        pairs = np.array([[index[c.left_id], index[c.right_id]] for c in usable], dtype=int)
        scores = np.array([c.score_left() for c in usable], dtype=float)
        rng = np.random.default_rng(self.seed)
        ids = [c.id for c in cand_list]

        if self.method == "pairwise_logistic":
            lam = self._select_lambda_logistic(z, pairs, scores, rng)
            weights = self._fit_logistic(z, pairs, scores, lam)
            intercept = 0.0
            r2: float | None = None
        else:
            target = self._bt_target(ids, usable)
            lam = self._select_lambda_ridge(z, target)
            weights, intercept = ridge_fit(z, target, lam)
            fitted = z @ weights + intercept
            total = float(np.sum((target - target.mean()) ** 2))
            r2 = float(1.0 - np.sum((target - fitted) ** 2) / total) if total > 1e-12 else 0.0
        weights = np.where(constant, 0.0, weights)
        utilities = z @ weights
        accuracy = _pairwise_accuracy(utilities, pairs, scores)
        permutation = self._permutation_importance(z, weights, pairs, scores, accuracy, rng)
        boot = self._bootstrap(z, pairs, scores, ids, usable, lam, constant, rng)
        magnitude = np.abs(weights)
        total_magnitude = float(magnitude.sum())
        features: list[FeatureWeight] = []
        for k, name in enumerate(names):
            std_coef = float(weights[k])
            if boot.size:
                ci_low = float(np.percentile(boot[:, k], 2.5))
                ci_high = float(np.percentile(boot[:, k], 97.5))
            else:
                ci_low = ci_high = std_coef
            if abs(std_coef) < _NEUTRAL_THRESHOLD:
                direction: Literal["prefer_higher", "prefer_lower", "neutral"] = "neutral"
            else:
                direction = "prefer_higher" if std_coef > 0 else "prefer_lower"
            features.append(
                FeatureWeight(
                    name=name,
                    coefficient=float(std_coef / scale[k]) if not constant[k] else 0.0,
                    std_coefficient=std_coef,
                    importance=float(magnitude[k] / total_magnitude) if total_magnitude > 0 else 0.0,
                    permutation_importance=float(permutation[k]),
                    ci_low=min(ci_low, ci_high),
                    ci_high=max(ci_low, ci_high),
                    direction=direction,
                )
            )
        features.sort(key=lambda f: (-f.importance, f.name))
        return AestheticPreferenceVector(
            features=features,
            intercept=float(intercept),
            method=self.method,  # type: ignore[arg-type]
            regularization=float(lam),
            fit_accuracy=float(accuracy),
            r2=r2,
            n_comparisons=len(usable),
            n_candidates=len(cand_list),
        )

    # -- 内部 --------------------------------------------------------------

    @staticmethod
    def _common_features(candidates: Sequence[StyleCandidate]) -> list[str]:
        if not candidates:
            return []
        common = set(candidates[0].params)
        for candidate in candidates[1:]:
            common &= set(candidate.params)
        return sorted(common)

    def _neutral(self, names: Sequence[str], n_comparisons: int, n_candidates: int) -> AestheticPreferenceVector:
        lam = float(self.regularization) if self.regularization is not None else 1.0
        return AestheticPreferenceVector(
            features=[
                FeatureWeight(
                    name=name, coefficient=0.0, std_coefficient=0.0, importance=0.0,
                    permutation_importance=0.0, ci_low=0.0, ci_high=0.0, direction="neutral",
                )
                for name in names
            ],
            intercept=0.0,
            method=self.method,  # type: ignore[arg-type]
            regularization=lam,
            fit_accuracy=0.5,
            r2=None if self.method == "pairwise_logistic" else 0.0,
            n_comparisons=n_comparisons,
            n_candidates=n_candidates,
        )

    @staticmethod
    def _design(z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        return z[pairs[:, 0]] - z[pairs[:, 1]]

    def _fit_logistic(self, z: np.ndarray, pairs: np.ndarray, scores: np.ndarray, lam: float) -> np.ndarray:
        return logistic_fit(self._design(z, pairs), scores, lam)

    def _select_lambda_logistic(self, z: np.ndarray, pairs: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> float:
        if self.regularization is not None:
            return float(self.regularization)
        m = len(scores)
        if m < 10:
            return 1.0
        design = self._design(z, pairs)
        folds = np.array_split(rng.permutation(m), 5)
        best_lam, best_score = 1.0, -math.inf
        for lam in LAMBDA_GRID:
            total = 0.0
            for fold in folds:
                train = np.setdiff1d(np.arange(m), fold)
                weights = logistic_fit(design[train], scores[train], lam)
                total += _logistic_loglik(design[fold], scores[fold], weights)
            mean_ll = total / m
            if mean_ll > best_score + 1e-12 or (abs(mean_ll - best_score) <= 1e-12 and lam > best_lam):
                best_lam, best_score = lam, mean_ll
        return float(best_lam)

    def _bt_target(self, ids: Sequence[str], comparisons: Sequence[Comparison]) -> np.ndarray:
        result = BradleyTerryModel().fit(ids, comparisons)
        return np.array([result.log_strengths[cid] for cid in ids], dtype=float)

    def _select_lambda_ridge(self, z: np.ndarray, target: np.ndarray) -> float:
        if self.regularization is not None:
            return float(self.regularization)
        if z.shape[0] < 3:
            return 1.0
        scores = [(ridge_loocv_mse(z, target, lam), lam) for lam in LAMBDA_GRID]
        best_mse = min(score for score, _ in scores)
        return float(max(lam for score, lam in scores if score <= best_mse + 1e-12))

    def _permutation_importance(
        self,
        z: np.ndarray,
        weights: np.ndarray,
        pairs: np.ndarray,
        scores: np.ndarray,
        baseline: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        drops = np.zeros(z.shape[1], dtype=float)
        for k in range(z.shape[1]):
            if weights[k] == 0:
                continue
            accuracies = []
            for _ in range(self.permutation_repeats):
                shuffled = z.copy()
                shuffled[:, k] = rng.permutation(shuffled[:, k])
                accuracies.append(_pairwise_accuracy(shuffled @ weights, pairs, scores))
            drops[k] = baseline - float(np.mean(accuracies))
        return drops

    def _bootstrap(
        self,
        z: np.ndarray,
        pairs: np.ndarray,
        scores: np.ndarray,
        ids: Sequence[str],
        comparisons: Sequence[Comparison],
        lam: float,
        constant: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.n_bootstrap <= 0:
            return np.zeros((0, z.shape[1]))
        m = len(scores)
        samples = np.zeros((self.n_bootstrap, z.shape[1]), dtype=float)
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, m, size=m)
            if self.method == "pairwise_logistic":
                weights = self._fit_logistic(z, pairs[idx], scores[idx], lam)
            else:
                target = self._bt_target(ids, [comparisons[i] for i in idx])
                weights, _ = ridge_fit(z, target, lam)
            samples[b] = np.where(constant, 0.0, weights)
        return samples


# ---------------------------------------------------------------------------
# ランキング・能動学習
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: Sequence[StyleCandidate],
    comparisons: Sequence[Comparison],
    *,
    elo: EloEngine | None = None,
    bradley_terry: BradleyTerryModel | None = None,
) -> list[RatingEntry]:
    """Elo と Bradley-Terry を併算し、BT 強度降順で順位付けする。"""
    unique: dict[str, StyleCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.id, candidate)
    ids = list(unique)
    elo_ratings = (elo or EloEngine()).compute(ids, comparisons)
    bt = (bradley_terry or BradleyTerryModel()).fit(ids, comparisons)
    tallies = {cid: {"games": 0, "wins": 0, "losses": 0, "ties": 0} for cid in ids}
    for comparison in comparisons:
        left, right = comparison.left_id, comparison.right_id
        if left not in tallies or right not in tallies:
            continue
        tallies[left]["games"] += 1
        tallies[right]["games"] += 1
        if comparison.outcome == "left":
            tallies[left]["wins"] += 1
            tallies[right]["losses"] += 1
        elif comparison.outcome == "right":
            tallies[right]["wins"] += 1
            tallies[left]["losses"] += 1
        else:
            tallies[left]["ties"] += 1
            tallies[right]["ties"] += 1
    ordered = sorted(ids, key=lambda cid: (-bt.strengths[cid], -elo_ratings[cid], cid))
    entries: list[RatingEntry] = []
    for rank, cid in enumerate(ordered, start=1):
        tally = tallies[cid]
        games = tally["games"]
        win_rate = (tally["wins"] + 0.5 * tally["ties"]) / games if games else 0.0
        entries.append(
            RatingEntry(
                candidate_id=cid,
                label=unique[cid].label,
                elo=elo_ratings[cid],
                bt_strength=bt.strengths[cid],
                bt_elo_scale=bt.elo_scale[cid],
                bt_std_error=bt.std_errors[cid],
                games=games,
                wins=tally["wins"],
                losses=tally["losses"],
                ties=tally["ties"],
                win_rate=round(win_rate, 6),
                rank=rank,
            )
        )
    return entries


def suggest_next_pair(
    candidates: Sequence[StyleCandidate],
    comparisons: Sequence[Comparison],
    seed: int = 0,
) -> tuple[str, str] | None:
    """次に判定すべきペア（勝率≈0.5 × 不確実性 × 対戦の少なさ が最大）。左右は決定論的にランダム化。"""
    ids = _unique_ids(c.id for c in candidates)
    if len(ids) < 2:
        return None
    usable = [c for c in comparisons if c.left_id in set(ids) and c.right_id in set(ids)]
    bt = BradleyTerryModel().fit(ids, usable)
    games, _, _ = _pair_tables(ids, usable)
    best_key: tuple[float, int] | None = None
    best_pair: tuple[str, str] | None = None
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pa = bt.strengths[ids[a]]
            pb = bt.strengths[ids[b]]
            p = pa / (pa + pb)
            closeness = 1.0 - abs(2.0 * p - 1.0)
            uncertainty = bt.std_errors[ids[a]] + bt.std_errors[ids[b]]
            novelty = 1.0 / (1.0 + games[a, b])
            score = round(closeness * uncertainty * novelty, 12)
            tiebreak = stable_hash(seed, ids[a], ids[b]) % 1_000_003
            key = (score, tiebreak)
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (ids[a], ids[b])
    if best_pair is None:
        return None
    if stable_hash(seed, best_pair[0], best_pair[1], len(usable)) % 2 == 1:
        return best_pair[1], best_pair[0]
    return best_pair


def ratings_by_id(entries: Iterable[RatingEntry]) -> Mapping[str, RatingEntry]:
    """RatingEntry の ID 索引。"""
    return {entry.candidate_id: entry for entry in entries}
