"""多軸グリッド実験台帳（Grid Laboratory）。

- 構造化トークン辞書: 線画タッチ・白黒トーン設計・パース歪み・構図・照明・質感のトークンを重み付きで管理
- X / Y / Z の3次元グリッド: トークン差し替え（または重み掃引）・数値パラメータ・LoRA 強度・マージ比率・
  CFG・ステップ数・サンプラー・シードを軸に直積バッチを生成
- 結果追跡と分析: 主効果、X×Y 交互作用表、二元配置分散分析（平方和分解）、最良セル
- ComfyUI バッチ出力: セルごとの派生レシピから txt2img ワークフローを生成
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import new_id, utc_now_iso
from core.style_alchemy.comfy_export import build_txt2img_workflow
from core.style_alchemy.schemas import ARCHITECTURE_PROFILES, StyleRecipe

if TYPE_CHECKING:
    from core.style_alchemy.repository import StyleLabRepository

__all__ = [
    "DEFAULT_TOKEN_CATEGORIES",
    "DEFAULT_TOKENS",
    "TokenEntry",
    "TokenDictionary",
    "AxisKind",
    "GridAxis",
    "GridExperiment",
    "GridCell",
    "GridAnalysis",
    "GridLaboratory",
]

DEFAULT_TOKEN_CATEGORIES: dict[str, str] = {
    "line_art": "線画タッチ",
    "tone_design": "白黒トーン設計",
    "perspective": "パース歪み",
    "composition": "構図",
    "lighting": "照明",
    "texture": "質感",
}

DEFAULT_TOKENS: dict[str, tuple[tuple[str, str], ...]] = {
    "line_art": (
        ("clean lineart", "均一で整った主線。デジタル清書のような安定感"),
        ("variable line weight", "強弱のある主線。輪郭の重みで立体を示す"),
        ("tapered ink lines", "入り抜きの効いたペン線。筆圧の抑揚が出る"),
        ("rough sketch lines", "ラフな鉛筆線。勢いと未完成の空気感"),
        ("thick outlines", "太い外形線。シルエットを強調する"),
        ("fine hatching", "細密なハッチング。陰影を線の密度で描く"),
        ("brush ink strokes", "筆ペンの払い。墨の濃淡と速度感"),
        ("g-pen lines", "Gペン特有の鋭い抑揚"),
        ("technical pen lines", "製図ペンの均一線。メカや背景向き"),
    ),
    "tone_design": (
        ("screentone shading", "網点トーンによる陰影"),
        ("halftone gradient", "グラデーショントーンで空気遠近を出す"),
        ("heavy black fill", "大胆なベタで画面を締める"),
        ("high contrast black and white", "白黒の強いコントラスト"),
        ("sand tone texture", "砂目トーンのざらつき"),
        ("white highlights on black", "ベタ上のホワイトで光を描く"),
        ("minimal tone", "トーンを最小限に抑えた余白の多い画面"),
        ("cross hatching shadows", "カケアミ風の重ね線で影を作る"),
        ("speed line tone", "流線トーンで動きを出す"),
    ),
    "perspective": (
        ("dramatic perspective", "誇張パースによる迫力"),
        ("fisheye lens", "魚眼による強い歪み"),
        ("wide angle distortion", "広角レンズの周辺歪み"),
        ("two-point perspective", "二点透視の安定した空間"),
        ("three-point perspective", "三点透視による見上げ・見下ろし"),
        ("foreshortening", "短縮法で手前の部位を大きく"),
        ("telephoto compression", "望遠圧縮で背景を引き寄せる"),
        ("low horizon line", "低い地平線で人物を大きく見せる"),
        ("isometric view", "等角投影の俯瞰図"),
    ),
    "composition": (
        ("rule of thirds", "三分割法"),
        ("centered composition", "日の丸構図で正面性を出す"),
        ("diagonal composition", "対角線構図で動きを出す"),
        ("negative space", "余白で間を作る"),
        ("frame within frame", "画中の枠で視線を誘導する"),
        ("leading lines", "線で視線を主題へ導く"),
        ("symmetrical composition", "対称構図で静けさを出す"),
        ("close-up framing", "寄りの画角で感情を伝える"),
        ("establishing shot", "状況説明の引きの画"),
    ),
    "lighting": (
        ("rim light", "輪郭光で人物を背景から分離"),
        ("backlighting", "逆光でシルエットを強調"),
        ("chiaroscuro", "明暗法による劇的な陰影"),
        ("hard shadows", "硬い影でコントラストを上げる"),
        ("soft ambient light", "柔らかな環境光"),
        ("spotlight", "スポットライトで主題を浮かせる"),
        ("silhouette", "シルエットで象徴化する"),
        ("volumetric light rays", "光芒で空気を見せる"),
        ("low-key lighting", "暗部の多いローキー照明"),
    ),
    "texture": (
        ("paper texture", "紙の目のテクスチャ"),
        ("ink bleed", "インクのにじみ"),
        ("dry brush texture", "かすれた筆致"),
        ("film grain", "粒状感"),
        ("watercolor wash", "水彩のにじみ"),
        ("cross hatching texture", "線の重なりによる質感"),
        ("stippling", "点描による階調"),
        ("smudged graphite", "鉛筆のこすれ"),
        ("clean digital finish", "均質なデジタル仕上げ"),
    ),
}


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TokenEntry(_Base):
    """構造化トークン辞書の1語。"""

    token: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=80)
    weight: float = Field(1.0, ge=0.0, le=3.0)
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class TokenDictionary:
    """カテゴリ別の重み付きプロンプトトークン辞書。"""

    DEFAULT_CATEGORIES = DEFAULT_TOKEN_CATEGORIES

    def __init__(self, entries: Iterable[TokenEntry] | None = None, include_defaults: bool = True) -> None:
        self._entries: dict[tuple[str, str], TokenEntry] = {}
        if include_defaults:
            for category, tokens in DEFAULT_TOKENS.items():
                for token, description in tokens:
                    self._entries[(category, token)] = TokenEntry(token=token, category=category, description=description)
        for entry in entries or ():
            self._entries[(entry.category, entry.token)] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        entry: TokenEntry | str,
        category: str | None = None,
        weight: float = 1.0,
        description: str = "",
        tags: Sequence[str] = (),
    ) -> TokenEntry:
        """トークンを追加（同カテゴリ同名は置換）。"""
        if isinstance(entry, TokenEntry):
            item = entry
        else:
            if category is None:
                raise ValueError("文字列で追加する場合は category が必要です")
            item = TokenEntry(token=entry, category=category, weight=weight, description=description, tags=list(tags))
        self._entries[(item.category, item.token)] = item
        return item

    def remove(self, token: str, category: str | None = None) -> bool:
        keys = [key for key in self._entries if key[1] == token and (category is None or key[0] == category)]
        for key in keys:
            del self._entries[key]
        return bool(keys)

    def get(self, token: str, category: str | None = None) -> TokenEntry | None:
        for (cat, tok), entry in self._entries.items():
            if tok == token and (category is None or cat == category):
                return entry
        return None

    def list(self, category: str | None = None) -> list[TokenEntry]:
        return [entry for (cat, _), entry in sorted(self._entries.items()) if category is None or cat == category]

    def categories(self) -> list[str]:
        return sorted({cat for cat, _ in self._entries})

    def render(self, token: str, weight: float | None = None) -> str:
        """``(token:1.20)`` 形式（重み1.0は素のトークン）。"""
        if weight is None:
            entry = self.get(token)
            weight = entry.weight if entry is not None else 1.0
        if abs(float(weight) - 1.0) < 1e-9:
            return token
        return f"({token}:{float(weight):.2f})"

    def to_dict(self) -> dict[str, Any]:
        categories: dict[str, list[dict[str, Any]]] = {}
        for entry in self.list():
            categories.setdefault(entry.category, []).append(entry.model_dump(mode="json"))
        return {"categories": categories, "labels": dict(DEFAULT_TOKEN_CATEGORIES)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenDictionary":
        entries = [
            TokenEntry.model_validate({**item, "category": item.get("category", category)})
            for category, items in (data.get("categories") or {}).items()
            for item in items
        ]
        return cls(entries=entries, include_defaults=False)


AxisKind = Literal["token", "param", "lora_weight", "merge_weight", "cfg", "steps", "seed", "sampler"]


class GridAxis(_Base):
    """グリッドの1軸。"""

    name: str = Field(..., min_length=1, max_length=80)
    kind: AxisKind
    target: str = ""
    values: list[str | float | int] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_axis(self) -> "GridAxis":
        numeric_kinds = {"param", "lora_weight", "merge_weight", "cfg", "steps", "seed"}
        if self.kind in numeric_kinds:
            for value in self.values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"軸 {self.name}（{self.kind}）の値は数値である必要があります: {value!r}") from exc
        if self.kind in {"param", "lora_weight", "merge_weight"} and not self.target:
            raise ValueError(f"軸 {self.name}（{self.kind}）には target が必要です")
        return self

    def label(self, index: int) -> str:
        value = self.values[index]
        return f"{value:g}" if isinstance(value, float) else str(value)


class GridExperiment(_Base):
    """X/Y/Z グリッド実験の定義。"""

    id: str = Field(default_factory=lambda: new_id("grid"))
    name: str = Field(..., min_length=1, max_length=200)
    base_prompt: str = ""
    negative_prompt: str = ""
    x_axis: GridAxis
    y_axis: GridAxis | None = None
    z_axis: GridAxis | None = None
    base_params: dict[str, float] = Field(default_factory=dict)
    seed: int = 0
    status: Literal["draft", "running", "completed"] = "draft"
    created_at: str = Field(default_factory=utc_now_iso)

    def axes(self) -> list[tuple[str, GridAxis]]:
        result: list[tuple[str, GridAxis]] = [("x", self.x_axis)]
        if self.y_axis is not None:
            result.append(("y", self.y_axis))
        if self.z_axis is not None:
            result.append(("z", self.z_axis))
        return result

    def cell_count(self) -> int:
        count = 1
        for _, axis in self.axes():
            count *= len(axis.values)
        return count


class GridCell(_Base):
    """グリッドの1セル（1生成条件）と結果。"""

    id: str
    experiment_id: str
    x_index: int
    y_index: int = 0
    z_index: int = 0
    prompt: str
    negative_prompt: str = ""
    params: dict[str, float] = Field(default_factory=dict)
    seed: int = 0
    sampler_overrides: dict[str, Any] = Field(default_factory=dict)
    image_ref: str | None = None
    score: float | None = None
    notes: str = ""


class GridAnalysis(_Base):
    """グリッド実験の分析結果。"""

    experiment_id: str
    n_scored: int
    main_effects: dict[str, list[dict[str, Any]]]
    interaction_xy: list[list[float | None]] | None = None
    best_cell_id: str | None = None
    anova: dict[str, float] = Field(default_factory=dict)


def _apply_axis(
    axis: GridAxis,
    value: str | float | int,
    prompt_parts: list[str],
    prompt: str,
    params: dict[str, float],
    sampler: dict[str, Any],
    dictionary: TokenDictionary,
    slot: str,
) -> tuple[str, int | None]:
    """軸の値をセルへ適用し、(更新後プロンプト, シード上書き) を返す。"""
    seed_override: int | None = None
    if axis.kind == "token":
        if isinstance(value, (int, float)) and not isinstance(value, bool) and axis.target:
            rendered = dictionary.render(axis.target, float(value))
        else:
            rendered = dictionary.render(str(value))
        placeholders = ("{" + axis.name + "}", "{" + slot + "}")
        replaced = False
        for placeholder in placeholders:
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, rendered)
                replaced = True
        if not replaced:
            prompt_parts.append(rendered)
    elif axis.kind == "param":
        params[axis.target] = float(value)
    elif axis.kind == "lora_weight":
        params[f"lora.{axis.target}.strength"] = float(value)
    elif axis.kind == "merge_weight":
        params[f"merge.{axis.target}"] = float(value)
    elif axis.kind == "cfg":
        sampler["cfg"] = float(value)
    elif axis.kind == "steps":
        sampler["steps"] = int(float(value))
    elif axis.kind == "sampler":
        sampler["sampler_name"] = str(value)
    elif axis.kind == "seed":
        seed_override = int(float(value))
    return prompt, seed_override


class GridLaboratory:
    """グリッド実験の作成・セル生成・結果記録・分析・ComfyUI バッチ化。"""

    def __init__(self, repo: "StyleLabRepository | None" = None, dictionary: TokenDictionary | None = None) -> None:
        self.repo = repo
        if dictionary is not None:
            self.dictionary = dictionary
        elif repo is not None:
            self.dictionary = repo.load_dictionary()
        else:
            self.dictionary = TokenDictionary()
        self._experiments: dict[str, GridExperiment] = {}
        self._cells: dict[str, dict[str, GridCell]] = {}

    # -- 作成 ----------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        base_prompt: str,
        x_axis: GridAxis,
        y_axis: GridAxis | None = None,
        z_axis: GridAxis | None = None,
        *,
        negative_prompt: str = "",
        base_params: Mapping[str, float] | None = None,
        seed: int = 0,
    ) -> GridExperiment:
        experiment = GridExperiment(
            name=name,
            base_prompt=base_prompt,
            negative_prompt=negative_prompt,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            base_params=dict(base_params or {}),
            seed=seed,
        )
        cells = self.generate_cells(experiment)
        self._experiments[experiment.id] = experiment
        self._cells[experiment.id] = {cell.id: cell for cell in cells}
        if self.repo is not None:
            self.repo.save_experiment(experiment)
            self.repo.save_cells(cells)
        return experiment

    def generate_cells(self, experiment: GridExperiment) -> list[GridCell]:
        """X×Y×Z の直積セルを生成する。"""
        axes = experiment.axes()
        ranges = [range(len(axis.values)) for _, axis in axes]
        cells: list[GridCell] = []
        for indices in itertools.product(*ranges):
            prompt = experiment.base_prompt
            extra_parts: list[str] = []
            params = dict(experiment.base_params)
            sampler: dict[str, Any] = {}
            seed = experiment.seed
            position = {"x": 0, "y": 0, "z": 0}
            for (slot, axis), index in zip(axes, indices):
                position[slot] = index
                prompt, seed_override = _apply_axis(axis, axis.values[index], extra_parts, prompt, params, sampler, self.dictionary, slot)
                if seed_override is not None:
                    seed = seed_override
            full_prompt = ", ".join(part for part in [prompt.strip().strip(","), *extra_parts] if part)
            cells.append(
                GridCell(
                    id=f"{experiment.id}_x{position['x']}_y{position['y']}_z{position['z']}",
                    experiment_id=experiment.id,
                    x_index=position["x"],
                    y_index=position["y"],
                    z_index=position["z"],
                    prompt=full_prompt,
                    negative_prompt=experiment.negative_prompt,
                    params=params,
                    seed=seed,
                    sampler_overrides=sampler,
                )
            )
        return cells

    # -- 取得・記録 ----------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> GridExperiment | None:
        if experiment_id in self._experiments:
            return self._experiments[experiment_id]
        if self.repo is not None:
            experiment = self.repo.get_experiment(experiment_id)
            if experiment is not None:
                self._experiments[experiment_id] = experiment
                self._cells[experiment_id] = {cell.id: cell for cell in self.repo.list_cells(experiment_id)}
            return experiment
        return None

    def cells(self, experiment_id: str) -> list[GridCell]:
        if self.get_experiment(experiment_id) is None:
            raise KeyError(f"実験が見つかりません: {experiment_id}")
        return sorted(self._cells.get(experiment_id, {}).values(), key=lambda c: (c.z_index, c.y_index, c.x_index))

    def record_result(self, cell_id: str, score: float, image_ref: str | None = None, notes: str = "") -> GridCell:
        """セルに評価結果を記録する（全セル記録で実験を completed に更新）。"""
        experiment_id = cell_id.rsplit("_x", 1)[0]
        if self.get_experiment(experiment_id) is None or cell_id not in self._cells.get(experiment_id, {}):
            raise KeyError(f"セルが見つかりません: {cell_id}")
        cell = self._cells[experiment_id][cell_id]
        updated = cell.model_copy(update={"score": float(score), "image_ref": image_ref or cell.image_ref, "notes": notes or cell.notes})
        self._cells[experiment_id][cell_id] = updated
        experiment = self._experiments[experiment_id]
        all_scored = all(c.score is not None for c in self._cells[experiment_id].values())
        status = "completed" if all_scored else "running"
        if experiment.status != status:
            experiment = experiment.model_copy(update={"status": status})
            self._experiments[experiment_id] = experiment
            if self.repo is not None:
                self.repo.save_experiment(experiment)
        if self.repo is not None:
            self.repo.update_cell(updated)
        return updated

    # -- 分析 ----------------------------------------------------------------

    def analyze(self, experiment_id: str) -> GridAnalysis:
        """主効果・X×Y 交互作用・二元配置分散分析・最良セル。"""
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"実験が見つかりません: {experiment_id}")
        scored = [cell for cell in self.cells(experiment_id) if cell.score is not None]
        main_effects: dict[str, list[dict[str, Any]]] = {}
        for slot, axis in experiment.axes():
            entries: list[dict[str, Any]] = []
            for index in range(len(axis.values)):
                values = [c.score for c in scored if getattr(c, f"{slot}_index") == index and c.score is not None]
                entries.append(
                    {
                        "index": index,
                        "value": axis.values[index],
                        "label": axis.label(index),
                        "mean": float(np.mean(values)) if values else None,
                        "count": len(values),
                    }
                )
            main_effects[axis.name] = entries
        if not scored:
            return GridAnalysis(experiment_id=experiment_id, n_scored=0, main_effects=main_effects)
        scores = np.array([c.score for c in scored], dtype=float)
        grand = float(scores.mean())
        ss_total = float(np.sum((scores - grand) ** 2))
        x_idx = np.array([c.x_index for c in scored])
        x_means = {i: float(scores[x_idx == i].mean()) for i in np.unique(x_idx)}
        ss_x = float(sum((x_idx == i).sum() * (m - grand) ** 2 for i, m in x_means.items()))
        anova: dict[str, float] = {"grand_mean": grand, "ss_total": ss_total, "ss_x": ss_x}
        interaction: list[list[float | None]] | None = None
        if experiment.y_axis is not None:
            y_idx = np.array([c.y_index for c in scored])
            y_means = {j: float(scores[y_idx == j].mean()) for j in np.unique(y_idx)}
            ss_y = float(sum((y_idx == j).sum() * (m - grand) ** 2 for j, m in y_means.items()))
            ss_xy = 0.0
            interaction = []
            for i in range(len(experiment.x_axis.values)):
                row: list[float | None] = []
                for j in range(len(experiment.y_axis.values)):
                    mask = (x_idx == i) & (y_idx == j)
                    if mask.any():
                        cell_mean = float(scores[mask].mean())
                        row.append(cell_mean)
                        if i in x_means and j in y_means:
                            ss_xy += float(mask.sum()) * (cell_mean - x_means[i] - y_means[j] + grand) ** 2
                    else:
                        row.append(None)
                interaction.append(row)
            anova.update({"ss_y": ss_y, "ss_xy": ss_xy, "ss_residual": max(0.0, ss_total - ss_x - ss_y - ss_xy)})
            if ss_total > 0:
                anova.update({"eta2_x": ss_x / ss_total, "eta2_y": ss_y / ss_total, "eta2_xy": ss_xy / ss_total})
        else:
            anova["ss_residual"] = max(0.0, ss_total - ss_x)
            if ss_total > 0:
                anova["eta2_x"] = ss_x / ss_total
        best = max(scored, key=lambda c: (c.score, -c.x_index, -c.y_index, -c.z_index))
        return GridAnalysis(
            experiment_id=experiment_id,
            n_scored=len(scored),
            main_effects=main_effects,
            interaction_xy=interaction,
            best_cell_id=best.id,
            anova={key: round(float(value), 8) for key, value in anova.items()},
        )

    # -- ComfyUI バッチ --------------------------------------------------------

    def to_comfy_batch(
        self,
        experiment_id: str,
        recipe: StyleRecipe,
        width: int | None = None,
        height: int | None = None,
    ) -> list[dict[str, Any]]:
        """各セルの派生レシピから txt2img ワークフローを生成する。

        戻り値の各要素は ``{"cell_id", "x_index", "y_index", "z_index", "workflow"}``。
        """
        from core.style_alchemy.model_optimizer import RecipeExporter

        exporter = RecipeExporter()
        resolution = int(ARCHITECTURE_PROFILES[recipe.architecture]["native_resolution"])
        out_w = int(width or resolution)
        out_h = int(height or resolution)
        batch: list[dict[str, Any]] = []
        for cell in self.cells(experiment_id):
            variant = exporter.apply_params(recipe, cell.params, cell.sampler_overrides)
            workflow = build_txt2img_workflow(
                variant,
                cell.prompt,
                cell.negative_prompt,
                seed=cell.seed,
                width=out_w,
                height=out_h,
                filename_prefix=f"caf/grid/{experiment_id}/{cell.id}",
            )
            batch.append(
                {"cell_id": cell.id, "x_index": cell.x_index, "y_index": cell.y_index, "z_index": cell.z_index, "workflow": workflow}
            )
        return batch
