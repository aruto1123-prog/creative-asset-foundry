"""Manga Grammar Validator — 漫画文法の数学的・論理的自動監査。

検査規則（rule_id / 既定重大度）:

| rule_id | 重大度 | 内容 |
|---|---|---|
| MONOTONOUS_SHOT | warning | 読み順で同一ショットサイズが3コマ以上連続（単調な画面） |
| MONOTONOUS_ANGLE | warning | 読み順で同一カメラアングルが3コマ以上連続 |
| PEAK_EMPHASIS | error | 緊張最大ビート・全転換点ビートのコマが 大ゴマ/splash/見開き・面積比≥0.4・めくり直後ページ先頭 のいずれでもない |
| BALLOON_OVERLOAD | warning / error | 1フキダシの文字数超過（warning）、1コマ合計が上限×2.2超（error）、フキダシ数超過（warning） |
| GAZE_BACKTRACK | warning | 視線誘導ベクトルの逆行（同段で右向き・段跨ぎで上向き）、または自然な読み順とコマ順の不一致 |
| GAZE_CROSSING | warning | 視線遷移線分の交差 |
| DENSITY_CONTRAST | warning / error | 連続3ページの情報密度の変動係数が閾値未満（緩急不足、連続区間は1件に集約）／上限超過（error） |
| PAGE_TURN_HOOK | info | めくり直後に転換点を置くのに、直前ページ終端に引きが無い |
| BALLOON_ORDER | info | コマ内フキダシの読み順（右→左・上→下）と配置の不一致 |

スコア = 100 − Σ penalty（error 12 / warning 4 / info 1）。合格 = スコア ≥ 閾値 かつ error 0件。
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from core.manga_pipeline.geometry import compute_gaze_transitions, compute_reading_order
from core.manga_pipeline.schemas import (
    GrammarReport,
    GrammarViolation,
    Page,
    PageTurnRole,
    Panel,
    Storyboard,
)

__all__ = [
    "SEVERITY_PENALTY",
    "RULE_DESCRIPTIONS",
    "identify_peak_beat",
    "emphasis_beat_ids",
    "panel_area_ratio",
    "page_information_density",
    "MangaGrammarValidator",
    "StoryboardFixer",
]

SEVERITY_PENALTY: dict[str, float] = {"error": 12.0, "warning": 4.0, "info": 1.0}

RULE_DESCRIPTIONS: dict[str, str] = {
    "MONOTONOUS_SHOT": "同一ショットサイズの3コマ以上連続",
    "MONOTONOUS_ANGLE": "同一カメラアングルの3コマ以上連続",
    "PEAK_EMPHASIS": "緊張ピーク・転換点の強調不足",
    "BALLOON_OVERLOAD": "フキダシ文字数・数の過多",
    "GAZE_BACKTRACK": "視線誘導の逆行・読み順不一致",
    "GAZE_CROSSING": "視線遷移の交差",
    "DENSITY_CONTRAST": "情報密度の緩急不足・過密",
    "PAGE_TURN_HOOK": "めくり前の引き不足",
    "BALLOON_ORDER": "フキダシ読み順の不一致",
}

_EMPHASIS_LABELS = {"large", "splash", "spread"}


def identify_peak_beat(structure: Any) -> Any | None:
    """大域緊張最大時刻に最も近いビート（同距離なら転換点を優先）。"""
    if structure is None or not structure.beats.beats:
        return None
    peak_t = structure.wave.peak_t("tension")
    pivotal = {beat.id for beat in structure.beats.pivotal_beats()}
    return min(
        structure.beats.beats,
        key=lambda beat: (abs(beat.t - peak_t) - (0.02 if beat.id in pivotal else 0.0), beat.t),
    )


def emphasis_beat_ids(structure: Any) -> list[str]:
    """強調必須ビートID（全転換点＋大域ピーク）を時間順で返す。"""
    if structure is None:
        return []
    ids = {beat.id for beat in structure.beats.pivotal_beats()}
    peak = identify_peak_beat(structure)
    if peak is not None:
        ids.add(peak.id)
    return [beat.id for beat in structure.beats.beats if beat.id in ids]


def panel_area_ratio(panel: Panel, storyboard: Storyboard, page: Page) -> float:
    """コマ面積 / 基本枠面積。"""
    frame_area = storyboard.geometry.frame_area(page.side)
    if frame_area <= 0:
        return 0.0
    return panel.polygon.area() / frame_area


def page_information_density(page: Page) -> float:
    """ページ情報密度 = コマ数 + 文字数/25 + 0.5×フキダシ数 + 0.3×オノマトペ数。"""
    chars = sum(panel.text_char_count() for panel in page.panels)
    balloons = sum(len(panel.balloons) for panel in page.panels)
    sfx = sum(len(panel.sfx) for panel in page.panels)
    return round(len(page.panels) + chars / 25.0 + 0.5 * balloons + 0.3 * sfx, 4)


def _violation(rule_id: str, severity: str, message: str, fix: str, *, page_index: int | None = None, panel_ids: list[str] | None = None) -> GrammarViolation:
    return GrammarViolation(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        page_index=page_index,
        panel_ids=list(panel_ids or []),
        message=message,
        fix_instruction=fix,
        penalty=SEVERITY_PENALTY[severity],
    )


class MangaGrammarValidator:
    """漫画文法バリデータ。config は StoryboardConfig 互換（max_chars_per_balloon 等）。"""

    def __init__(
        self,
        config: Any | None = None,
        *,
        density_cv_threshold: float = 0.12,
        density_upper_limit: float = 26.0,
        large_area_ratio: float = 0.4,
        monotony_run: int = 3,
        gaze_tolerance: float = 20.0,
    ) -> None:
        if config is None:
            module = importlib.import_module("core.manga_pipeline.storyboard_generator")
            config = module.StoryboardConfig()
        self.config = config
        self.density_cv_threshold = density_cv_threshold
        self.density_upper_limit = density_upper_limit
        self.large_area_ratio = large_area_ratio
        self.monotony_run = monotony_run
        self.gaze_tolerance = gaze_tolerance

    # -- 個別規則 ------------------------------------------------------------

    def _check_monotony(self, storyboard: Storyboard) -> list[GrammarViolation]:
        panels = [p for p in storyboard.panels_in_reading_order() if p.spread_role != "continuation"]
        violations: list[GrammarViolation] = []
        for attribute, rule_id, label in (
            ("shot", "MONOTONOUS_SHOT", "ショットサイズ"),
            ("angle", "MONOTONOUS_ANGLE", "カメラアングル"),
        ):
            run: list[Panel] = []
            for panel in panels + [None]:  # type: ignore[list-item]
                value = getattr(panel.camera, attribute).value if panel is not None else None
                if run and value == getattr(run[-1].camera, attribute).value:
                    run.append(panel)
                    continue
                if len(run) >= self.monotony_run:
                    run_value = getattr(run[0].camera, attribute).value
                    middle = run[len(run) // 2]
                    violations.append(
                        _violation(
                            rule_id,
                            "warning",
                            f"同一{label}（{run_value}）が {len(run)} コマ連続し画面が単調です",
                            f"コマ {middle.id} の{label}を前後と異なる値に変更し、連続を2コマ以下に分断してください",
                            page_index=run[0].page_index,
                            panel_ids=[p.id for p in run],
                        )
                    )
                run = [panel] if panel is not None else []
        return violations

    def _check_peak_emphasis(self, storyboard: Storyboard, structure: Any) -> tuple[list[GrammarViolation], float]:
        violations: list[GrammarViolation] = []
        if structure is not None:
            targets = emphasis_beat_ids(structure)
        else:
            panels = storyboard.panels_in_reading_order()
            targets = []
            if panels:
                peak_panel = max(panels, key=lambda p: p.tension)
                targets = [peak_panel.beat_id] if peak_panel.beat_id else []
                if not targets:
                    page = storyboard.page_of_panel(peak_panel.id)
                    if page is not None and not self._panel_emphasized(peak_panel, storyboard, page):
                        violations.append(
                            _violation(
                                "PEAK_EMPHASIS",
                                "error",
                                f"緊張最大のコマ {peak_panel.id} が強調されていません",
                                "このコマを大ゴマ化（面積比0.4以上）するか、めくり直後の右ページ先頭へ移してください",
                                page_index=page.index,
                                panel_ids=[peak_panel.id],
                            )
                        )
                    return violations, 0.0 if violations else 1.0
        satisfied = 0
        for beat_id in targets:
            holders = [
                (panel, page)
                for page in storyboard.pages
                for panel in page.panels
                if beat_id in panel.beat_ids()
            ]
            if not holders:
                violations.append(
                    _violation(
                        "PEAK_EMPHASIS",
                        "error",
                        f"転換点ビート {beat_id} を描くコマがありません",
                        f"ビート {beat_id} に専用のコマを割り当て、大ゴマとして配置してください",
                    )
                )
                continue
            if any(self._panel_emphasized(panel, storyboard, page) for panel, page in holders):
                satisfied += 1
                continue
            panel, page = holders[0]
            violations.append(
                _violation(
                    "PEAK_EMPHASIS",
                    "error",
                    f"転換点ビート {beat_id} のコマ {panel.id} が強調されていません（面積比 {panel_area_ratio(panel, storyboard, page):.2f}）",
                    f"コマ {panel.id} を単独段の大ゴマ（面積比{self.large_area_ratio:.1f}以上）にするか、めくり直後の右ページ先頭へ移してください",
                    page_index=page.index,
                    panel_ids=[panel.id],
                )
            )
        coverage = satisfied / len(targets) if targets else 1.0
        return violations, coverage

    def _panel_emphasized(self, panel: Panel, storyboard: Storyboard, page: Page) -> bool:
        if panel.emphasis in _EMPHASIS_LABELS:
            return True
        if panel_area_ratio(panel, storyboard, page) >= self.large_area_ratio:
            return True
        return page.page_turn_role == PageTurnRole.REVEAL and panel.index_in_page == 0

    def _check_balloons(self, storyboard: Storyboard) -> list[GrammarViolation]:
        max_chars = int(self.config.max_chars_per_balloon)
        max_balloons = int(self.config.max_balloons_per_panel)
        violations: list[GrammarViolation] = []
        for panel in storyboard.panels_in_reading_order():
            for balloon in panel.balloons:
                if balloon.char_count > max_chars:
                    violations.append(
                        _violation(
                            "BALLOON_OVERLOAD",
                            "warning",
                            f"フキダシ {balloon.id} が {balloon.char_count} 字で上限 {max_chars} 字を超えています",
                            f"台詞を句点・読点で分割して {max_chars} 字以下のフキダシ複数に分けるか、要点だけに削ってください",
                            page_index=panel.page_index,
                            panel_ids=[panel.id],
                        )
                    )
            total = panel.text_char_count()
            if total > 2.2 * max_chars:
                violations.append(
                    _violation(
                        "BALLOON_OVERLOAD",
                        "error",
                        f"コマ {panel.id} の総文字数 {total} 字が上限 {int(2.2 * max_chars)} 字を超えています",
                        "台詞の一部を前後のコマへ送るか、コマを分割して1コマの文字量を減らしてください",
                        page_index=panel.page_index,
                        panel_ids=[panel.id],
                    )
                )
            if len(panel.balloons) > max_balloons:
                violations.append(
                    _violation(
                        "BALLOON_OVERLOAD",
                        "warning",
                        f"コマ {panel.id} のフキダシ数 {len(panel.balloons)} が上限 {max_balloons} を超えています",
                        f"フキダシを {max_balloons} 個以下に統合するか、隣接コマへ送ってください",
                        page_index=panel.page_index,
                        panel_ids=[panel.id],
                    )
                )
        return violations

    def _check_gaze(self, storyboard: Storyboard) -> tuple[list[GrammarViolation], int, int]:
        violations: list[GrammarViolation] = []
        backtracks = 0
        crossings = 0
        for page in storyboard.pages:
            ordered = page.ordered_panels()
            if len(ordered) < 2:
                continue
            transitions = compute_gaze_transitions(page, tol_x=self.gaze_tolerance, tol_y=self.gaze_tolerance)
            back = [t for t in transitions if t.is_backtrack]
            natural = compute_reading_order(ordered)
            mismatch = natural != [p.id for p in ordered]
            if back or mismatch:
                backtracks += len(back) + (1 if mismatch else 0)
                ids = sorted({t.from_panel for t in back} | {t.to_panel for t in back}) if back else [p.id for p in ordered]
                reason = "視線誘導ベクトルが逆行しています" if back else "自然な読み順（右上→左下）とコマ順が一致しません"
                violations.append(
                    _violation(
                        "GAZE_BACKTRACK",
                        "warning",
                        f"ページ {page.number}: {reason}",
                        "段を水平に揃えた矩形コマ割りへ組み直し、コマ順を右→左・上→下の自然な読み順に一致させてください",
                        page_index=page.index,
                        panel_ids=ids,
                    )
                )
            crossing = [t for t in transitions if t.crosses]
            if crossing:
                crossings += len(crossing)
                violations.append(
                    _violation(
                        "GAZE_CROSSING",
                        "warning",
                        f"ページ {page.number}: 視線遷移が {len(crossing)} 箇所で交差しています",
                        "交差する遷移のコマを同一段へまとめるか、段構成を単純化して視線の流れを一筆書きにしてください",
                        page_index=page.index,
                        panel_ids=sorted({t.from_panel for t in crossing}),
                    )
                )
        return violations, backtracks, crossings

    def _check_density(self, storyboard: Storyboard) -> tuple[list[GrammarViolation], list[float]]:
        pages = sorted(storyboard.pages, key=lambda p: p.index)
        densities = [page_information_density(page) for page in pages]
        violations: list[GrammarViolation] = []
        for page, density in zip(pages, densities):
            if density > self.density_upper_limit:
                violations.append(
                    _violation(
                        "DENSITY_CONTRAST",
                        "error",
                        f"ページ {page.number} の情報密度 {density:.1f} が上限 {self.density_upper_limit:.0f} を超えています",
                        "コマ数か台詞量を減らし、このページの情報を前後のページへ分散してください",
                        page_index=page.index,
                        panel_ids=[p.id for p in page.panels],
                    )
                )
        flagged: list[int] = []
        for i in range(len(pages) - 2):
            window = np.asarray(densities[i : i + 3], dtype=float)
            mean = float(np.mean(window))
            cv = float(np.std(window) / mean) if mean > 1e-9 else 0.0
            if cv < self.density_cv_threshold:
                flagged.append(i)
        runs: list[list[int]] = []
        for start in flagged:
            if runs and start <= runs[-1][-1] + 1:
                runs[-1].append(start)
            else:
                runs.append([start])
        for run in runs:
            page_indices = list(range(run[0], run[-1] + 3))
            numbers = "・".join(str(pages[i].number) for i in page_indices)
            violations.append(
                _violation(
                    "DENSITY_CONTRAST",
                    "warning",
                    f"ページ {numbers} の情報密度が平板で緩急がありません",
                    "中間ページのコマ数を±2程度変える（間のコマを抜く／会話コマを足す）か、台詞の無い溜めページを作って密度差をつけてください",
                    page_index=pages[run[0] + 1].index,
                    panel_ids=[],
                )
            )
        return violations, densities

    def _check_page_turn_hooks(self, storyboard: Storyboard, structure: Any) -> list[GrammarViolation]:
        violations: list[GrammarViolation] = []
        targets = set(emphasis_beat_ids(structure)) if structure is not None else set()
        pages = sorted(storyboard.pages, key=lambda p: p.index)
        for previous, page in zip(pages, pages[1:]):
            if page.page_turn_role != PageTurnRole.REVEAL or not page.panels:
                continue
            first = page.ordered_panels()[0]
            if targets and not (set(first.beat_ids()) & targets):
                continue
            if not targets and first.emphasis not in _EMPHASIS_LABELS:
                continue
            ordered = previous.ordered_panels()
            if not ordered:
                continue
            last = ordered[-1]
            hooked = "page_turn_hook" in last.prompt_hints or last.tension >= previous.tension_mean
            if not hooked:
                violations.append(
                    _violation(
                        "PAGE_TURN_HOOK",
                        "info",
                        f"ページ {page.number} のめくり直後の開示に対し、ページ {previous.number} の終端コマ {last.id} に引きがありません",
                        f"コマ {last.id} を緊張を高める引きのコマ（寄りの表情・意味深な台詞・無音の溜め）にしてください",
                        page_index=previous.index,
                        panel_ids=[last.id],
                    )
                )
        return violations

    def _check_balloon_order(self, storyboard: Storyboard) -> list[GrammarViolation]:
        violations: list[GrammarViolation] = []
        for panel in storyboard.panels_in_reading_order():
            ordered = sorted(panel.balloons, key=lambda b: b.reading_order)
            for a, b in zip(ordered, ordered[1:]):
                to_the_right = b.center.x > a.center.x + 8.0
                not_lower = b.center.y < a.center.y + 0.5 * min(a.height, b.height)
                if to_the_right and not_lower:
                    violations.append(
                        _violation(
                            "BALLOON_ORDER",
                            "info",
                            f"コマ {panel.id} のフキダシ {b.id} が先行する {a.id} より右上にあり読み順が逆転しています",
                            "フキダシを読み順どおり右から左・上から下へ並べ直してください",
                            page_index=panel.page_index,
                            panel_ids=[panel.id],
                        )
                    )
                    break
        return violations

    # -- 統合 -----------------------------------------------------------------

    def validate(self, storyboard: Storyboard, structure: Any | None = None) -> GrammarReport:
        """全規則を検査してスコア付きレポートを返す。"""
        violations: list[GrammarViolation] = []
        violations.extend(self._check_monotony(storyboard))
        peak_violations, coverage = self._check_peak_emphasis(storyboard, structure)
        violations.extend(peak_violations)
        violations.extend(self._check_balloons(storyboard))
        gaze_violations, backtracks, crossings = self._check_gaze(storyboard)
        violations.extend(gaze_violations)
        density_violations, densities = self._check_density(storyboard)
        violations.extend(density_violations)
        violations.extend(self._check_page_turn_hooks(storyboard, structure))
        violations.extend(self._check_balloon_order(storyboard))
        penalty = sum(v.penalty for v in violations)
        score = max(0.0, 100.0 - penalty)
        threshold = float(self.config.grammar_threshold)
        errors = sum(1 for v in violations if v.severity == "error")
        panels = storyboard.panels_in_reading_order()
        balloons = [b for p in panels for b in p.balloons]
        shots = {p.camera.shot.value for p in panels}
        angles = {p.camera.angle.value for p in panels}
        density_array = np.asarray(densities, dtype=float) if densities else np.zeros(1)
        metrics = {
            "pages": float(storyboard.page_count),
            "panels": float(len(panels)),
            "balloons": float(len(balloons)),
            "avg_chars_per_balloon": round(float(np.mean([b.char_count for b in balloons])) if balloons else 0.0, 3),
            "max_chars_per_balloon": float(max((b.char_count for b in balloons), default=0)),
            "density_mean": round(float(np.mean(density_array)), 4),
            "density_cv": round(float(np.std(density_array) / np.mean(density_array)) if float(np.mean(density_array)) > 0 else 0.0, 4),
            "emphasis_coverage": round(coverage, 4),
            "gaze_backtracks": float(backtracks),
            "gaze_crossings": float(crossings),
            "shot_variety": round(len(shots) / 5.0, 4),
            "angle_variety": round(len(angles) / 4.0, 4),
            "errors": float(errors),
            "warnings": float(sum(1 for v in violations if v.severity == "warning")),
            "infos": float(sum(1 for v in violations if v.severity == "info")),
        }
        return GrammarReport(
            score=round(score, 2),
            passed=score >= threshold and errors == 0,
            violations=violations,
            metrics=metrics,
        )


def __getattr__(name: str) -> Any:
    if name == "StoryboardFixer":
        module = importlib.import_module("core.manga_pipeline.storyboard_generator")
        return module.StoryboardFixer
    raise AttributeError(f"module 'core.manga_pipeline.grammar_validator' has no attribute {name!r}")
