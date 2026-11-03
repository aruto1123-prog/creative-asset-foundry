"""1対多コンテンツ・マトリクス変換器。

1つの構造発見またはモデル検証ログ（ResearchLog）から、

① クリエイター向け「物語構造・漫画演出理論」
② AI開発者向け「モデル調合・特徴量制御・データセット設計論」
③ ビジネス層向け「人の心を動かす企画・LP・マーケティング構造論」

の3軸の企画・構成案を同時に展開する。決定論的ベースライン（``expand_deterministic``）が常に完全な
ContentMatrix を生成し、LLM 接続時は多段階ハーネス（task ``publisher.matrix``）で増幅する。
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from core.asset_publisher.schemas import (
    AXIS_LABELS_JA,
    AXIS_ORDER,
    AXIS_READERS_JA,
    BLUEPRINT_ROLE_ORDER,
    ContentAxis,
    ContentMatrix,
    ContentPlan,
    Finding,
    OutlineSection,
    ResearchLog,
)
from core.common import stable_hash, truncate
from core.structure_engine.schemas import (
    BEAT_TYPE_LABELS_JA,
    EDGE_KIND_LABELS_JA,
    MECHANISM_LABELS_JA,
    UniversalStructure,
)

logger = logging.getLogger(__name__)

__all__ = [
    "METRIC_LABELS_JA",
    "AXIS_FRAMES",
    "metric_label",
    "format_metric",
    "research_log_from_structure",
    "research_log_from_experiment",
    "expand_matrix_deterministic",
    "ContentMatrixTranspiler",
]

METRIC_LABELS_JA: dict[str, str] = {
    "peak_t": "緊張度ピークの位置",
    "peak_tension": "緊張度のピーク値",
    "mean_tension": "平均緊張度",
    "mean_release": "平均解放度",
    "catharsis": "カタルシス指数",
    "suspense": "サスペンス係数",
    "surprise": "サプライズ係数",
    "turning_points": "転換点の数",
    "causal_depth": "因果の深さ",
    "first_turn_t": "最初の転換点の位置",
    "tension_energy": "緊張エネルギー",
    "nodes": "力学ノード数",
    "edges": "力学エッジ数",
    "dominant_strength": "主要メカニズムの強度",
    "mechanisms": "認知メカニズム数",
    "slots": "抽象スロット数",
    "invariants": "不変条件の数",
    "fit_accuracy": "選好予測精度",
    "n_comparisons": "比較判定数",
    "n_candidates": "候補数",
    "top_importance": "最重要特徴量の重要度",
    "top_coefficient": "最重要特徴量の係数",
    "elo_top": "首位候補のElo",
    "elo_spread": "Eloの上位下位差",
    "top_win_rate": "首位候補の勝率",
    "predicted_mean": "提案レシピの予測スコア",
    "predicted_std": "予測の不確実性",
    "acquisition": "獲得関数値",
    "total_images": "学習画像数",
    "balance_score": "データセット均衡スコア",
    "entanglements": "概念癒着の数",
    "total_additional_images": "追加推奨枚数",
    "max_npmi": "最大NPMI",
    "grammar_score": "漫画文法スコア",
    "violations": "文法違反の数",
    "pages": "ページ数",
    "panels": "コマ数",
    "best_score": "グリッド最高スコア",
}


def metric_label(key: str) -> str:
    return METRIC_LABELS_JA.get(key, key.replace("_", " "))


def format_metric(value: float) -> str:
    """整数値は整数表記、それ以外は小数第2位までで表記する。"""
    number = float(value)
    if abs(number - round(number)) < 1e-9 and abs(number) >= 1.0:
        return str(int(round(number)))
    return f"{number:.2f}"


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            seen.append(value)
    return seen


# ---------------------------------------------------------------------------
# ResearchLog 生成
# ---------------------------------------------------------------------------


def research_log_from_structure(structure: UniversalStructure) -> ResearchLog:
    """構造の波形統計・転換点・力学・メカニズム・ブループリントから研究ログを構成する。"""
    wave = structure.wave
    summary = wave.summary()
    pivotal = structure.beats.pivotal_beats()
    graph = structure.tension_graph
    histogram = graph.edge_kind_histogram()
    dominant_edge = max(histogram.items(), key=lambda item: item[1])[0]
    mechanisms = sorted(structure.mechanics.mechanisms, key=lambda m: -m.strength)
    dominant_mechanism = mechanisms[0]
    suspense = summary["suspense"]
    surprise = summary["surprise"]
    if suspense >= 0.08 and suspense >= surprise * 0.5:
        info_label = "読者が作中人物より先に真相へ近づくサスペンス優位"
    else:
        info_label = "情報を伏せて一挙に開示するサプライズ優位"
    turn_positions = "、".join(f"t={beat.t:.2f}" for beat in pivotal[:4]) or "なし"
    first_turn = pivotal[0].t if pivotal else summary["peak_t"]

    findings = [
        Finding(
            claim=(
                f"緊張度のピークは正規化時間 t={summary['peak_t']:.2f} にあり、ピーク値は {summary['peak_tension']:.2f}、"
                f"カタルシス指数は {summary['catharsis']:.2f} である"
            ),
            evidence=(
                f"{int(summary['n_points'])}点の感情波形から算出した（平均緊張度 {summary['mean_tension']:.2f}、"
                f"平均解放度 {summary['mean_release']:.2f}）"
            ),
            metrics={
                "peak_t": summary["peak_t"],
                "peak_tension": summary["peak_tension"],
                "catharsis": summary["catharsis"],
                "mean_tension": summary["mean_tension"],
            },
            confidence=0.85,
        ),
        Finding(
            claim=f"情報設計は{info_label}であり、サスペンス係数は {suspense:.2f}、サプライズ係数は {surprise:.2f} を示す",
            evidence="読者の認知量と作中人物の認知量の差分（情報非対称性）の推移から算出した",
            metrics={"suspense": suspense, "surprise": surprise},
            confidence=0.8,
        ),
        Finding(
            claim=(
                f"転換点は {len(pivotal)} 箇所（{turn_positions}）に置かれ、因果の連鎖は {structure.beats.causal_depth()} 段に及ぶ"
            ),
            evidence="ビート間の因果リンクを先行関係として追跡し、最長経路を測定した",
            metrics={
                "turning_points": float(len(pivotal)),
                "causal_depth": float(structure.beats.causal_depth()),
                "first_turn_t": round(first_turn, 4),
            },
            confidence=0.75,
        ),
        Finding(
            claim=(
                f"力学の中核は「{EDGE_KIND_LABELS_JA[dominant_edge]}」の関係であり、緊張エネルギーは "
                f"{graph.tension_energy():.2f} に達する"
            ),
            evidence=f"{len(graph.nodes)} 個のノードと {len(graph.edges)} 本の有向エッジの強度を種別ごとに集計した",
            metrics={
                "tension_energy": graph.tension_energy(),
                "nodes": float(len(graph.nodes)),
                "edges": float(len(graph.edges)),
            },
            confidence=0.7,
        ),
        Finding(
            claim=(
                f"没入を最も強く駆動するのは「{dominant_mechanism.name}」"
                f"（{MECHANISM_LABELS_JA[dominant_mechanism.category.value]}、強度 {dominant_mechanism.strength:.2f}）である"
            ),
            evidence=f"{len(mechanisms)} 種類の認知メカニズムを強度順に比較した",
            metrics={
                "dominant_strength": dominant_mechanism.strength,
                "mechanisms": float(len(mechanisms)),
            },
            confidence=0.7,
        ),
        Finding(
            claim=(
                f"{len(structure.blueprint.slots)} 個の抽象スロットと {len(structure.blueprint.invariants)} 個の不変条件により、"
                "漫画・LP・記事のいずれにも力学を保ったまま転用できる"
            ),
            evidence="ブループリントのスロット定義と不変条件を領域横断で検査した",
            metrics={
                "slots": float(len(structure.blueprint.slots)),
                "invariants": float(len(structure.blueprint.invariants)),
            },
            confidence=0.65,
        ),
    ]
    data: dict[str, Any] = {
        "structure_id": structure.id,
        "title": structure.metadata.title,
        "category": structure.metadata.category.value,
        "wave_summary": summary,
        "wave": {
            "t": [round(v, 4) for v in wave.t],
            "tension": [round(v, 4) for v in wave.tension],
            "release": [round(v, 4) for v in wave.release],
            "information_asymmetry": [round(v, 4) for v in wave.information_asymmetry],
        },
        "beats": [
            {
                "id": beat.id,
                "type": beat.beat_type.value,
                "type_ja": BEAT_TYPE_LABELS_JA[beat.beat_type.value],
                "t": beat.t,
                "label": beat.label,
                "pivotal": beat in pivotal,
                "causes": list(beat.causes),
            }
            for beat in structure.beats.beats
        ],
        "mechanisms": [
            {"name": m.name, "category": m.category.value, "category_ja": MECHANISM_LABELS_JA[m.category.value], "strength": m.strength}
            for m in mechanisms
        ],
        "invariants": list(structure.blueprint.invariants),
        "slots": [slot.name for slot in structure.blueprint.slots],
        "transfer_guide": structure.transfer_guide.model_dump(mode="json"),
    }
    tags = _unique(
        [
            *structure.metadata.tags,
            *(MECHANISM_LABELS_JA[m.category.value] for m in mechanisms[:3]),
            *(BEAT_TYPE_LABELS_JA[b.beat_type.value] for b in pivotal[:2]),
        ]
    )
    return ResearchLog(
        id=f"log_{structure.id}",
        kind="structure_discovery",
        title=f"{structure.metadata.title}の構造分析",
        summary=structure.metadata.summary,
        findings=findings,
        data=data,
        tags=tags,
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _findings_from_preference(vector: dict[str, Any]) -> list[Finding]:
    features = [f for f in vector.get("features", []) if isinstance(f, dict) and f.get("name")]
    if not features:
        return []
    ranked = sorted(features, key=lambda f: -_as_float(f.get("importance")))
    top = ranked[0]
    direction_ja = {"prefer_higher": "高いほど好まれる", "prefer_lower": "低いほど好まれる", "neutral": "影響が小さい"}
    descriptions = [
        f"{f['name']}（{direction_ja.get(str(f.get('direction')), '方向不明')}、重要度 {_as_float(f.get('importance')):.2f}）"
        for f in ranked[:3]
    ]
    accuracy = _as_float(vector.get("fit_accuracy"))
    comparisons = _as_float(vector.get("n_comparisons"))
    return [
        Finding(
            claim=f"美的選好ベクトルの最重要特徴量は {top['name']} であり、{direction_ja.get(str(top.get('direction')), '方向不明')}",
            evidence="ブラインドA/B判定の勝敗を目的変数とした正則化回帰の標準化係数と順列重要度から推定した",
            metrics={
                "top_importance": _as_float(top.get("importance")),
                "top_coefficient": _as_float(top.get("coefficient")),
                "fit_accuracy": accuracy,
                "n_comparisons": comparisons,
            },
            confidence=0.8 if accuracy >= 0.6 else 0.6,
        ),
        Finding(
            claim="上位の選好要因は " + "、".join(descriptions) + " である",
            evidence="ブートストラップ信頼区間で係数の符号の安定性を確認した",
            metrics={"n_candidates": _as_float(vector.get("n_candidates"))},
            confidence=0.7,
        ),
    ]


def _findings_from_ratings(ratings: list[Any]) -> list[Finding]:
    rows = [r for r in ratings if isinstance(r, dict)]
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: _as_float(r.get("rank"), 1e9))
    top = ordered[0]
    elos = [_as_float(r.get("elo"), 1500.0) for r in rows]
    label = str(top.get("label") or top.get("candidate_id") or "首位候補")
    return [
        Finding(
            claim=f"Eloレーティングの首位は「{label}」で、上位と下位の差は {max(elos) - min(elos):.0f} 点に開いた",
            evidence="一対比較の勝敗履歴から Elo と Bradley-Terry 強度を算出して順位付けした",
            metrics={
                "elo_top": _as_float(top.get("elo"), 1500.0),
                "elo_spread": max(elos) - min(elos),
                "top_win_rate": _as_float(top.get("win_rate")),
            },
            confidence=0.75,
        )
    ]


def _findings_from_suggestions(suggestions: list[Any]) -> list[Finding]:
    rows = [s for s in suggestions if isinstance(s, dict)]
    if not rows:
        return []
    best = max(rows, key=lambda s: _as_float(s.get("predicted_mean")))
    params = best.get("params") if isinstance(best.get("params"), dict) else {}
    key_params = sorted(params.items(), key=lambda kv: -abs(_as_float(kv[1]) - 0.5))[:3]
    described = "、".join(f"{name}={_as_float(value):.2f}" for name, value in key_params) or "既定値付近"
    return [
        Finding(
            claim=f"ベイズ最適化は次の試行として {described} を中心とする配合を提案した",
            evidence=f"ガウス過程の事後分布と獲得関数（戦略: {best.get('strategy', 'hybrid')}）から {len(rows)} 件の候補を算出した",
            metrics={
                "predicted_mean": _as_float(best.get("predicted_mean")),
                "predicted_std": _as_float(best.get("predicted_std")),
                "acquisition": _as_float(best.get("acquisition")),
            },
            confidence=0.65,
        )
    ]


def _findings_from_gap_report(report: dict[str, Any]) -> list[Finding]:
    if not report:
        return []
    entanglements = [e for e in report.get("entanglements", []) if isinstance(e, dict)]
    deficits = [d for d in report.get("deficits", []) if isinstance(d, dict)]
    findings: list[Finding] = []
    if entanglements:
        worst = max(entanglements, key=lambda e: _as_float(e.get("npmi")))
        findings.append(
            Finding(
                claim=f"キャプションでは「{worst.get('tag_a')}」と「{worst.get('tag_b')}」が癒着しており、概念の分離が不十分である",
                evidence="タグ共起行列から正規化自己相互情報量（NPMI）を算出して検出した",
                metrics={"entanglements": float(len(entanglements)), "max_npmi": _as_float(worst.get("npmi"))},
                confidence=0.75,
            )
        )
    findings.append(
        Finding(
            claim=(
                f"データセットの均衡スコアは {_as_float(report.get('balance_score')):.1f} で、"
                f"追加で {int(_as_float(report.get('total_additional_images')))} 枚の学習画像が推奨される"
            ),
            evidence=f"{len(deficits)} 件の不足条件をカテゴリ別エントロピーと目標比率から算出した",
            metrics={
                "total_images": _as_float(report.get("total_images")),
                "balance_score": _as_float(report.get("balance_score")),
                "total_additional_images": _as_float(report.get("total_additional_images")),
            },
            confidence=0.7,
        )
    )
    return findings


def _findings_from_grid(grid: dict[str, Any]) -> list[Finding]:
    if not grid:
        return []
    effects = grid.get("main_effects") if isinstance(grid.get("main_effects"), dict) else {}
    best_axis = ""
    best_range = 0.0
    for axis_name, rows in effects.items():
        means = [_as_float(r.get("mean")) for r in rows if isinstance(r, dict) and r.get("mean") is not None]
        if len(means) >= 2 and max(means) - min(means) > best_range:
            best_range = max(means) - min(means)
            best_axis = str(axis_name)
    return [
        Finding(
            claim=f"グリッド実験では軸「{best_axis or '主効果'}」の主効果が最も大きく、平均スコアの幅は {best_range:.2f} だった",
            evidence="X/Y/Z 軸の水準ごとの平均と分散分解（主効果・交互作用）から判定した",
            metrics={"best_score": _as_float(grid.get("best_score"))},
            confidence=0.6,
        )
    ]


def _findings_from_storyboard(board: dict[str, Any]) -> list[Finding]:
    if not board:
        return []
    grammar = board.get("grammar") if isinstance(board.get("grammar"), dict) else board
    score = _as_float(grammar.get("score"))
    violations = grammar.get("violations", [])
    return [
        Finding(
            claim=f"構造駆動ネームの漫画文法スコアは {score:.1f} 点で、残存違反は {len(violations)} 件だった",
            evidence="単調な画面の連続・ピークの大ゴマ配置・視線誘導の逆行と交差・情報密度の緩急を自動監査した",
            metrics={
                "grammar_score": score,
                "violations": float(len(violations)),
                "pages": _as_float(board.get("pages") or board.get("page_count")),
            },
            confidence=0.7,
        )
    ]


def research_log_from_experiment(payload: dict[str, Any]) -> ResearchLog:
    """Style Alchemy / Manga Pipeline の実験結果（dict）を研究ログへ変換する。"""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("実験ログが空です。preference_vector / ratings / suggestions / gap_report / grid / storyboard のいずれかが必要です")
    findings: list[Finding] = []
    categories: list[str] = []
    if isinstance(payload.get("preference_vector"), dict):
        found = _findings_from_preference(payload["preference_vector"])
        findings.extend(found)
        if found:
            categories.append("model_experiment")
    if isinstance(payload.get("ratings"), list):
        found = _findings_from_ratings(payload["ratings"])
        findings.extend(found)
        if found:
            categories.append("model_experiment")
    if isinstance(payload.get("suggestions"), list):
        found = _findings_from_suggestions(payload["suggestions"])
        findings.extend(found)
        if found:
            categories.append("model_experiment")
    if isinstance(payload.get("gap_report"), dict):
        found = _findings_from_gap_report(payload["gap_report"])
        findings.extend(found)
        if found:
            categories.append("dataset_analysis")
    if isinstance(payload.get("grid"), dict):
        found = _findings_from_grid(payload["grid"])
        findings.extend(found)
        if found:
            categories.append("model_experiment")
    if isinstance(payload.get("storyboard"), dict):
        found = _findings_from_storyboard(payload["storyboard"])
        findings.extend(found)
        if found:
            categories.append("manga_production")
    if not findings:
        numeric = {k: _as_float(v) for k, v in payload.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        keys = "、".join(sorted(payload.keys())[:6])
        findings.append(
            Finding(
                claim=f"実験ログには {keys} の記録が含まれ、{len(numeric)} 個の数値指標が得られた",
                evidence="ログのトップレベル指標をそのまま集計した",
                metrics=dict(list(numeric.items())[:8]),
                confidence=0.5,
            )
        )
        categories.append("mixed")
    unique_categories = _unique(categories)
    kind = unique_categories[0] if len(unique_categories) == 1 else "mixed"
    titles = {
        "model_experiment": "美的選好学習とモデル配合最適化の実験ログ",
        "dataset_analysis": "LoRA学習データセットの偏り分析ログ",
        "manga_production": "構造駆動ネーム生成の制作ログ",
        "mixed": "創作R&Dの統合実験ログ",
    }
    title = str(payload.get("title") or titles[kind])
    summary = str(
        payload.get("summary")
        or f"{title}。{len(findings)} 件の発見を、選好・最適化・データセット・制作の各観点から数値とともに記録した。"
    )
    tags = _unique([*(payload.get("tags") or []), "美的選好" if "model_experiment" in categories else "", "データ駆動"])
    log_id = str(payload.get("id") or f"log_exp_{stable_hash(title, len(findings), summary) % 16**12:012x}")
    return ResearchLog(id=log_id, kind=kind, title=title, summary=summary, findings=findings, data=dict(payload), tags=tags)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3軸展開（決定論ベースライン）
# ---------------------------------------------------------------------------

AXIS_FRAMES: dict[str, dict[str, Any]] = {
    ContentAxis.CREATOR_THEORY.value: {
        "title": "{topic}はなぜ読者を動かすのか――物語構造・漫画演出理論で解く",
        "reader_pain": "感覚に頼った構成では、狙ったページで読者の感情が動かず、手応えと反応がずれ続ける",
        "principle": "読者の感情が動く瞬間は才能の偶然ではなく、緊張と情報開示の配置によって設計できる",
        "headings": [
            "「展開は感性で決まる」という常識",
            "上手い構成ほど読者が離れていく逆説",
            "感情曲線と転換点の検証データ",
            "どの作品にも写せる5段階の構造設計",
            "次のネームで試す3つの行動",
        ],
        "keywords": ["物語構造", "漫画演出", "感情曲線", "コマ割り", "めくり"],
        "hook_line": "名作の「ゾクッとする瞬間」は、偶然ではなくページ番号で設計できる。",
        "call_to_action": "次のネームで、転換点を感情曲線のピークに合わせて1ページだけ動かしてみてほしい",
        "figure": "感情曲線とビート配置の重ね合わせ図",
    },
    ContentAxis.AI_ENGINEERING.value: {
        "title": "{topic}を数値で制御する――モデル調合・特徴量制御・データセット設計論",
        "reader_pain": "好みの絵柄を言語化できないまま試行回数だけが増え、マージ比率やLoRA重みの判断が属人化している",
        "principle": "美的選好は言語化できなくても一対比較で測定でき、階層比率と重みの最適化問題として扱える",
        "headings": [
            "「プロンプトで好みは再現できる」という常識",
            "試行を重ねるほど理想から遠ざかる理由",
            "選好データと最適化の検証結果",
            "選好を数値化する5段階フレームワーク",
            "次の実験で試す3つの行動",
        ],
        "keywords": ["美的選好", "Bradley-Terry", "ベイズ最適化", "LoRA", "データセット設計"],
        "hook_line": "理想の絵柄は、言葉ではなく2枚の比較から見つかる。",
        "call_to_action": "まずは30回のブラインド比較を記録し、選好ベクトルを1本引いてみてほしい",
        "figure": "選好スコアと特徴量重要度の棒グラフ",
    },
    ContentAxis.BUSINESS_MARKETING.value: {
        "title": "{topic}に学ぶ人を動かす設計――企画・LP・マーケティング構造論",
        "reader_pain": "訴求を強め施策を重ねても成約率が頭打ちになり、次に何を変えるべきか判断できない",
        "principle": "人が行動する瞬間は訴求の強さではなく、情報を開示する順序と構造的反転の位置で決まる",
        "headings": [
            "「訴求を強めれば売れる」という常識",
            "改善を重ねるほど成約率が頭打ちになる逆説",
            "情報開示の順序と反応の検証データ",
            "人が動く構造の5段階フレームワーク",
            "次のLP改善で試す3つの行動",
        ],
        "keywords": ["LP設計", "セールスファネル", "情報開示", "行動変容", "構造的反転"],
        "hook_line": "成約率を決めるのは、言葉の強さではなく、情報を開く順番だ。",
        "call_to_action": "次のLPで、構造的反転を置く位置を1か所だけ変えてA/Bテストしてみてほしい",
        "figure": "ファネル段階と離脱率の対応図",
    },
}

_OUTLINE_PURPOSES = {
    BLUEPRINT_ROLE_ORDER[0]: "読者が当然と考える常識を言語化し、共感と問題意識を同時に立ち上げる",
    BLUEPRINT_ROLE_ORDER[1]: "常識に忠実であるほど成果が頭打ちになる構造的パラドックスを指摘する",
    BLUEPRINT_ROLE_ORDER[2]: "独自仮説を提示し、実験データと図解で検証する",
    BLUEPRINT_ROLE_ORDER[3]: "検証結果を再現可能な普遍的フレームワークへ抽象化する",
    BLUEPRINT_ROLE_ORDER[4]: "明日から実行できる具体的な行動へ落とし込む",
}


def _topic_from_log(log: ResearchLog) -> str:
    topic = log.title
    for suffix in ("の構造分析", "の実験ログ", "の分析ログ", "の制作ログ", "ログ"):
        if topic.endswith(suffix) and len(topic) > len(suffix) + 2:
            topic = topic[: -len(suffix)]
            break
    return truncate(topic, 30)


def _distribute_points(findings: list[Finding], index: int) -> list[str]:
    """発見を章へ分配する（各章2点、循環的に割当て）。"""
    if not findings:
        return []
    first = findings[index % len(findings)].claim
    second = findings[(index + 2) % len(findings)].claim
    return _unique([truncate(first, 120), truncate(second, 120)])


def _build_plan(log: ResearchLog, axis: ContentAxis) -> ContentPlan:
    frame = AXIS_FRAMES[axis.value]
    topic = _topic_from_log(log)
    strongest = log.strongest_finding()
    outline = [
        OutlineSection(
            heading=frame["headings"][i],
            purpose=_OUTLINE_PURPOSES[role],
            key_points=_distribute_points(log.findings, i) or [strongest.claim],
            figure_idea=frame["figure"] if i == 2 else f"{frame['headings'][i]}の構造図",
        )
        for i, role in enumerate(BLUEPRINT_ROLE_ORDER)
    ]
    keywords = _unique([*frame["keywords"], *log.tags[:5]])
    return ContentPlan(
        axis=axis,
        title=frame["title"].format(topic=topic),
        subtitle=truncate(f"{strongest.claim}という発見から", 140),
        target_reader=AXIS_READERS_JA[axis.value],
        reader_pain=frame["reader_pain"],
        core_insight=f"{frame['principle']}。根拠となる発見は「{truncate(strongest.claim, 120)}」である",
        angle=AXIS_LABELS_JA[axis.value],
        outline=outline,
        keywords=keywords,
        hook_line=frame["hook_line"],
        call_to_action=frame["call_to_action"],
        estimated_chars=5 * 1200 + 800,
    )


def expand_matrix_deterministic(log: ResearchLog) -> ContentMatrix:
    """研究ログから3軸の企画マトリクスを決定論的に生成する。"""
    strongest = log.strongest_finding()
    shared_core = truncate(f"{log.summary} 中核の発見は「{strongest.claim}」である。", 1200)
    plans = [_build_plan(log, axis) for axis in AXIS_ORDER]
    labels = [AXIS_LABELS_JA[a.value] for a in AXIS_ORDER]
    cross_links = [
        f"{labels[0]}の検証データ（第3章）は、{labels[1]}の第4章で数値化の前提として再利用できる",
        f"{labels[1]}の最適化手順（第4章）は、{labels[2]}のA/Bテスト設計（第5章）へ写像できる",
        f"{labels[2]}の情報開示の順序（第2章）は、{labels[0]}のめくり設計（第4章）と同型である",
    ]
    matrix_id = f"matrix_{stable_hash(log.id, 'matrix') % 16**12:012x}"
    return ContentMatrix(
        id=matrix_id,
        source_id=log.id,
        source_title=log.title,
        shared_core=shared_core,
        plans=plans,
        cross_links=cross_links,
    )


class ContentMatrixTranspiler:
    """研究ログ → 3軸コンテンツ企画マトリクスの変換器。"""

    def __init__(self, llm_client: Any | None = None, harness_config: Any | None = None) -> None:
        self.llm_client = llm_client
        self.harness_config = harness_config

    def expand_deterministic(self, log: ResearchLog) -> ContentMatrix:
        return expand_matrix_deterministic(log)

    async def expand(self, log: ResearchLog) -> ContentMatrix:
        """LLM 接続時は多段階ハーネスで増幅し、失敗時は決定論ベースラインを返す。"""
        baseline = expand_matrix_deterministic(log)
        if self.llm_client is None:
            return baseline
        from core.asset_publisher.prompts import MATRIX_SYSTEM_PROMPT, publisher_rubric
        from core.asset_publisher.sanitizer import find_urls
        from core.llm_client import HarnessConfig, LLMError, MultiStageReasoningHarness

        harness = MultiStageReasoningHarness(self.llm_client, self.harness_config or HarnessConfig())

        def _post_validate(matrix: ContentMatrix) -> list[str]:
            issues: list[str] = []
            for plan in matrix.plans:
                if len(plan.outline) < 5:
                    issues.append(f"{plan.axis.value} の構成は5章（Blueprint 5役割）にしてください")
                texts = [plan.title, plan.subtitle, plan.core_insight, plan.hook_line, plan.call_to_action]
                if any(find_urls(text) for text in texts):
                    issues.append(f"{plan.axis.value} の企画文にURLを含めないでください")
            return issues

        try:
            result = await harness.run(
                task="publisher.matrix",
                objective="1つの研究ログを、クリエイター・AI開発者・ビジネスの3軸の記事企画へ同時展開する",
                payload={"research_log": log.model_dump(mode="json"), "baseline": baseline.model_dump(mode="json")},
                output_model=ContentMatrix,
                system_prompt=MATRIX_SYSTEM_PROMPT,
                rubric=publisher_rubric("matrix"),
                context={"pub_research_log": log.model_dump(mode="json")},
                post_validate=_post_validate,
            )
        except (LLMError, ValidationError, ValueError) as exc:
            logger.warning("3軸展開の LLM 経路が失敗したため決定論ベースラインを使用します: %s", exc)
            return baseline
        matrix = result.output
        return matrix.model_copy(update={"id": baseline.id, "source_id": log.id, "source_title": log.title})
