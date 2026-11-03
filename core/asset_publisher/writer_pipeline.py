"""5段階・長文執筆＆厳格サニタイズパイプライン。

1. ``Blueprint``            : 常識の提示 → 構造的パラドックス → 独自仮説と実験データ/図解 → 普遍的フレームワーク → 実践アクション
2. ``Deep Section Drafting``: 章ごとの高密度執筆と Mermaid 図解の自動挿入
3. ``Compliance & Style Sanitizer``: 一人称「私」統一・URL完全除去（常時）
4. ``Critique Scoring``     : 論理・抽象と具体・納得度・根拠密度 各25点の100点満点採点
5. ``Iterative Polish``     : 基準点未満の章のみ再推敲し、採点が下がらない改稿だけを採用

すべての段は決定論的ベースラインを持ち（LLM なしで完全動作）、LLM 接続時は多段階ハーネスで増幅する。
LLM 経路が失敗した場合は自動的にベースラインへ退避する。
"""

from __future__ import annotations

import inspect
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from core.asset_publisher.matrix_transpiler import format_metric, metric_label
from core.asset_publisher.sanitizer import ComplianceSanitizer, detect_first_person_variants, find_urls, sanitize_text
from core.asset_publisher.schemas import (
    AXIS_LABELS_JA,
    BLUEPRINT_ROLE_LABELS_JA,
    BLUEPRINT_ROLE_ORDER,
    ArticleBlueprint,
    ArticleDraft,
    BlueprintRole,
    BlueprintSection,
    ContentAxis,
    ContentPlan,
    CritiqueScore,
    DraftSection,
    FigureSpec,
    Finding,
    PipelineStageRecord,
    PublishedArticle,
    ResearchLog,
    SanitizationReport,
    SectionScore,
    count_prose_chars,
    split_markdown_segments,
)
from core.common import stable_hash, stable_seed, truncate

logger = logging.getLogger(__name__)

__all__ = [
    "WriterConfig",
    "AXIS_PROFILES",
    "ArticleWriterPipeline",
    "build_blueprint_deterministic",
    "compose_section",
    "compose_lead",
    "compose_closing",
    "draft_article_deterministic",
    "score_section",
    "critique_deterministic",
    "polish_section_deterministic",
    "log_from_plan",
]

ProgressCallback = Callable[[str, float, str], Any]


class WriterConfig(BaseModel):
    """執筆パイプライン設定。"""

    quality_threshold: float = Field(80.0, ge=0.0, le=100.0)
    max_polish_rounds: int = Field(3, ge=0, le=10)
    min_section_chars: int = Field(600, ge=100, le=20000)
    target_section_chars: int = Field(1200, ge=200, le=40000)
    include_mermaid: bool = True


# ---------------------------------------------------------------------------
# 軸別プロファイル
# ---------------------------------------------------------------------------

AXIS_PROFILES: dict[str, dict[str, Any]] = {
    ContentAxis.CREATOR_THEORY.value: {
        "domain": "物語と漫画の制作",
        "practitioner": "作り手",
        "belief": "読者の心を動かす展開は、才能と感覚から生まれる",
        "belief_short": "展開は感覚で決まる",
        "belief_reason": "名作の多くが作者の直感から生まれたと語られてきた",
        "assumption": "感情の動きは設計できない",
        "assumption_short": "描き手の手応え",
        "unit": "ページ",
        "artifact": "ネーム",
        "metric_word": "感情曲線",
        "scene": "見開きのめくり直後に真相を置く場面",
        "objection": "構造で考えると作品が型にはまって退屈になる",
        "steps": [
            ("前提の可視化", "読者と主人公が共有している前提を1文で書き出し、どのページで共有させるかを決める"),
            ("限界点の特定", "同じ方法では越えられない壁が現れるページを、感情曲線の最初の頭打ちに合わせて置く"),
            ("構造的反転の配置", "前提を覆す反転を、めくり直後の右ページの大ゴマに置く"),
            ("代償の設計", "突破に必要な代償を、主人公の欠落と同じ根から生やす"),
            ("余韻の設計", "最後の3コマで、欠落の意味そのものを書き換える"),
        ],
        "actions": [
            "次のネームで、転換点を感情曲線のピークに合わせて1ページだけ前後に動かし、2案を並べて読み比べる",
            "見開きのめくり位置を2か所だけ決め、その直前の左ページ終端に必ず引きを置く",
            "3人の読者に読んでもらい、驚きが起きたページ番号と理由を1行ずつ記録する",
        ],
        "application": (
            "裏切りの発覚をめくり直後の右ページに置き、直前の左ページ終端に庇護者の微笑を描くだけで、"
            "同じ台詞でも読者の驚きは大きく変わる"
        ),
    },
    ContentAxis.AI_ENGINEERING.value: {
        "domain": "画像生成モデルの研究",
        "practitioner": "開発者",
        "belief": "理想の絵柄は、プロンプトを工夫し続ければいつか再現できる",
        "belief_short": "好みはプロンプトで再現できる",
        "belief_reason": "プロンプトの微調整で出力が劇的に変わる経験を誰もがしている",
        "assumption": "好みは言葉で説明できる",
        "assumption_short": "言葉にできる好み",
        "unit": "試行",
        "artifact": "レシピ",
        "metric_word": "選好スコア",
        "scene": "線の入り抜きだけを変えた2枚を並べて判定する場面",
        "objection": "美しさを数値にすると、感性の微妙な差が失われる",
        "steps": [
            ("選好の観測", "2枚の画像を並べるブラインド判定を30回以上記録し、判定にかかった時間も残す"),
            ("選好の数値化", "Eloレーティングと Bradley-Terry モデルで各候補の強さを推定する"),
            ("特徴量の分解", "正則化回帰で線の入り抜き・ベタ比率・ブロックマージ比率の寄与を分解する"),
            ("次の実験の提案", "ガウス過程と期待改善量で、次に比較すべき配合を最小の試行数で選ぶ"),
            ("レシピの固定", "最良の階層マージ比率と LoRA 重みをワークフローとして書き出し、再現性を確保する"),
        ],
        "actions": [
            "まず30回のブラインド比較を記録し、Elo の上位3候補を確認する",
            "上位候補のパラメータ差から、最も寄与の大きい特徴量を1つ選んで単独で振る",
            "ベイズ最適化の提案を3件だけ生成し、次の比較セッションで検証する",
        ],
        "application": (
            "ベタ比率とブロックマージの出力側比率だけを動かした12候補を総当たりで比べると、"
            "言葉では説明できなかった好みが係数の符号として現れる"
        ),
    },
    ContentAxis.BUSINESS_MARKETING.value: {
        "domain": "企画とマーケティング",
        "practitioner": "担当者",
        "belief": "成約率は訴求の強さと露出量で決まる",
        "belief_short": "訴求を強めれば売れる",
        "belief_reason": "強いコピーに差し替えた直後に数字が跳ねる経験が多い",
        "assumption": "顧客はすでに課題を自覚している",
        "assumption_short": "訴求の強さ",
        "unit": "セクション",
        "artifact": "LP",
        "metric_word": "反応率",
        "scene": "ファーストビュー直後ではなく中盤に常識の反転を置く場面",
        "objection": "構造の話は理屈っぽく、現場の売上には直結しない",
        "steps": [
            ("前提の共有", "顧客が信じている常識を、否定せずに言語化して共感を得る"),
            ("限界の提示", "常識どおりの努力が頭打ちになる構造を、事例と数値で示す"),
            ("構造的反転", "常識そのものが停滞の原因だという反転を、証拠とともに中盤に置く"),
            ("代償の明示", "乗り換えに伴う手間や費用を隠さず示し、比較で判断を委ねる"),
            ("最小行動の提示", "今日できる最小の一歩を行動喚起として示し、心理的な負担を下げる"),
        ],
        "actions": [
            "次のLPで、構造的反転を置く位置を1か所だけ変えた2案をA/Bテストする",
            "ファーストビューから行動喚起までの各セクションの離脱率を記録し、最大の離脱点を特定する",
            "乗り換えの代償を明示した版と明示しない版で、問い合わせの質を比較する",
        ],
        "application": (
            "業界の常識に沿った施策の限界を先に共有し、その常識自体が原因だという反転を中盤に置いたLPは、"
            "反論を先回りできるため行動喚起の直前での離脱が減りやすい"
        ),
    },
}

_ORDINALS = ("第一", "第二", "第三", "第四", "第五")
_FALLBACK_METRIC_PHRASES = (
    "検証した5つの層のうち2つで判断が停滞すること",
    "3回の比較で傾向が見え始めること",
    "10件の記録から共通する構造が現れること",
)


# ---------------------------------------------------------------------------
# 執筆コンテキスト
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    plan: ContentPlan
    log: ResearchLog
    axis: str
    profile: dict[str, Any]
    topic: str
    insight: str
    insight_short: str
    reader: str
    pain: str
    axis_label: str
    findings: list[Finding]
    metric_phrases: list[str]
    metric_items: list[tuple[str, float]]
    rng: random.Random = field(repr=False, default_factory=random.Random)

    def claim(self, index: int) -> str:
        return _strip_period(self.findings[index % len(self.findings)].claim)

    def metric(self, index: int) -> str:
        return self.metric_phrases[index % len(self.metric_phrases)]


def _strip_period(text: str) -> str:
    return text.strip().rstrip("。．.")


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            seen.append(value)
    return seen


def log_from_plan(plan: ContentPlan) -> ResearchLog:
    """研究ログが無い場合に、企画の構成要点から最小の研究ログを組み立てる。"""
    points = [p for section in plan.outline for p in section.key_points if p.strip()]
    findings = [Finding(claim=truncate(point, 300), evidence="企画の構成要点から引用", confidence=0.6) for point in points[:6]]
    if not findings:
        findings = [Finding(claim=truncate(plan.core_insight, 300), evidence="企画の中核洞察から引用", confidence=0.6)]
    return ResearchLog(
        id=f"log_plan_{stable_hash(plan.title, plan.axis.value) % 16**12:012x}",
        kind="mixed",
        title=plan.title,
        summary=plan.core_insight,
        findings=findings,
        tags=list(plan.keywords),
    )


def _topic(log: ResearchLog, plan: ContentPlan) -> str:
    title = log.title or plan.title
    for suffix in ("の構造分析", "の実験ログ", "の分析ログ", "の制作ログ"):
        if title.endswith(suffix) and len(title) > len(suffix) + 2:
            title = title[: -len(suffix)]
            break
    return truncate(title, 32)


def _context(plan: ContentPlan, log: ResearchLog | None) -> _Ctx:
    research = log or log_from_plan(plan)
    axis = plan.axis.value
    metrics = list(research.all_metrics().items())
    phrases = [f"{metric_label(key)}が{format_metric(value)}" for key, value in metrics[:8]]
    for fallback in _FALLBACK_METRIC_PHRASES:
        if len(phrases) >= 3:
            break
        phrases.append(fallback)
    insight = _strip_period(_strip_period(plan.core_insight).split("。")[0])
    insight_short = truncate(insight, 70)
    return _Ctx(
        plan=plan,
        log=research,
        axis=axis,
        profile=AXIS_PROFILES[axis],
        topic=_topic(research, plan),
        insight=truncate(insight, 200),
        insight_short=insight_short,
        reader=plan.target_reader,
        pain=truncate(_strip_period(plan.reader_pain), 90),
        axis_label=AXIS_LABELS_JA[axis],
        findings=list(research.findings),
        metric_phrases=phrases,
        metric_items=metrics,
        rng=random.Random(stable_seed(plan.title, axis, research.id)),
    )


# ---------------------------------------------------------------------------
# Mermaid 図解
# ---------------------------------------------------------------------------

_MERMAID_UNSAFE_RE = re.compile(r"[\"\[\]{}()<>|#;`\n\r]")


def _mm(text: str, limit: int = 26) -> str:
    """Mermaid の二重引用符ラベルとして安全な文字列へ整形する。"""
    cleaned = _MERMAID_UNSAFE_RE.sub(" ", text or "").strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return truncate(cleaned, limit) or "要素"


def _figure_conventional(ctx: _Ctx) -> FigureSpec:
    p = ctx.profile
    code = "\n".join(
        [
            "flowchart LR",
            f'    A["常識: {_mm(p["belief_short"])}"] --> B["同じ基準での努力"]',
            '    B --> C["局所的な改善"]',
            '    C --> D{"成果は伸びたか"}',
            "    D -->|はい| B",
            '    D -->|いいえ| E["感覚の不足と解釈"]',
            "    E --> B",
        ]
    )
    return FigureSpec(kind="mermaid", caption="常識に従う努力が同じ場所を回る循環", mermaid_code=code)


def _figure_paradox(ctx: _Ctx) -> FigureSpec:
    p = ctx.profile
    code = "\n".join(
        [
            "flowchart TD",
            f'    A["前提: {_mm(p["assumption"])}"] --> B["同じ基準での微調整"]',
            '    B --> C["局所的な最適化"]',
            '    C --> D["成果の頭打ち"]',
            '    D --> E["前提への依存の強化"]',
            "    E --> B",
            '    D -.-> F["構造的パラドックスの自覚"]',
            '    F --> G["前提そのものを測定する"]',
        ]
    )
    return FigureSpec(kind="mermaid", caption="局所最適化が前提への依存を強める構造", mermaid_code=code)


def _wave_series(ctx: _Ctx) -> tuple[list[float], list[float]] | None:
    wave = ctx.log.data.get("wave") if isinstance(ctx.log.data, dict) else None
    if not isinstance(wave, dict):
        return None
    tension = wave.get("tension")
    release = wave.get("release")
    if not isinstance(tension, list) or len(tension) < 5:
        return None
    try:
        tension_values = [round(float(v), 2) for v in tension]
        release_values = [round(float(v), 2) for v in release] if isinstance(release, list) and len(release) == len(tension) else []
    except (TypeError, ValueError):
        return None
    return tension_values, release_values


def _figure_hypothesis(ctx: _Ctx) -> FigureSpec:
    series = _wave_series(ctx)
    if series is not None:
        tension, release = series
        lines = [
            "xychart-beta",
            '    title "緊張度と解放度の推移"',
            '    x-axis "正規化時間" 0 --> 1',
            '    y-axis "強度" 0 --> 1',
            "    line [" + ", ".join(f"{v:.2f}" for v in tension) + "]",
        ]
        if release:
            lines.append("    line [" + ", ".join(f"{v:.2f}" for v in release) + "]")
        return FigureSpec(kind="mermaid", caption="感情波形（上: 緊張度、下: 解放度）", mermaid_code="\n".join(lines))
    unit_metrics = [(k, v) for k, v in ctx.metric_items if 0.0 <= v <= 1.0][:6]
    chosen = unit_metrics if len(unit_metrics) >= 2 else ctx.metric_items[:6]
    if len(chosen) >= 2:
        top = max(abs(v) for _, v in chosen) or 1.0
        upper = 1 if top <= 1.0 else round(top * 1.2, 2)
        labels = ", ".join(f'"{_mm(metric_label(k), 12)}"' for k, _ in chosen)
        values = ", ".join(f"{max(0.0, v):.2f}" for _, v in chosen)
        code = "\n".join(
            [
                "xychart-beta",
                '    title "主要指標の比較"',
                f"    x-axis [{labels}]",
                f'    y-axis "値" 0 --> {upper}',
                f"    bar [{values}]",
            ]
        )
        return FigureSpec(kind="mermaid", caption="検証で得た主要指標", mermaid_code=code)
    findings = ctx.findings[:4]
    lines = ["flowchart LR", f'    H["仮説: {_mm(ctx.insight_short)}"]']
    for i, finding in enumerate(findings, start=1):
        lines.append(f'    H --> F{i}["{_mm(finding.claim, 22)}"]')
    return FigureSpec(kind="mermaid", caption="仮説と検証結果の対応", mermaid_code="\n".join(lines))


def _figure_framework(ctx: _Ctx) -> FigureSpec:
    steps = ctx.profile["steps"]
    lines = ["flowchart LR"]
    for i, (name, _) in enumerate(steps, start=1):
        lines.append(f'    S{i}["{i} {_mm(name, 14)}"]')
    for i in range(1, len(steps)):
        lines.append(f"    S{i} --> S{i + 1}")
    lines.append(f"    S{len(steps)} -.->|次の題材へ| S1")
    return FigureSpec(kind="mermaid", caption="5段階フレームワークの因果連鎖", mermaid_code="\n".join(lines))


def _figure_action(ctx: _Ctx) -> FigureSpec:
    code = "\n".join(
        [
            "flowchart LR",
            '    R["記録"] --> C["比較"]',
            '    C --> H["仮説"]',
            '    H --> X["実験"]',
            '    X --> E["評価"]',
            "    E -->|基準を固定して反復| R",
        ]
    )
    return FigureSpec(kind="mermaid", caption="明日から回す検証ループ", mermaid_code=code)


_FIGURE_BUILDERS: dict[BlueprintRole, Callable[[_Ctx], FigureSpec]] = {
    BlueprintRole.CONVENTIONAL_WISDOM: _figure_conventional,
    BlueprintRole.STRUCTURAL_PARADOX: _figure_paradox,
    BlueprintRole.HYPOTHESIS_EVIDENCE: _figure_hypothesis,
    BlueprintRole.UNIVERSAL_FRAMEWORK: _figure_framework,
    BlueprintRole.PRACTICAL_ACTION: _figure_action,
}


def _mermaid_block(figure: FigureSpec) -> str:
    caption = f"*図: {figure.caption}*\n\n" if figure.caption else ""
    return f"{caption}```mermaid\n{figure.mermaid_code.strip()}\n```"


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------


def _role_claims(role: BlueprintRole, ctx: _Ctx) -> list[str]:
    p = ctx.profile
    if role == BlueprintRole.CONVENTIONAL_WISDOM:
        return [
            f"{truncate(ctx.reader, 40)}の多くは「{p['belief']}」という常識に立っている",
            f"この常識には「{p['assumption']}」という暗黙の前提がある",
        ]
    if role == BlueprintRole.STRUCTURAL_PARADOX:
        return [
            f"常識に忠実であるほど、{p['metric_word']}は同じ局所解の周囲で頭打ちになる",
            truncate(ctx.claim(1), 120),
        ]
    if role == BlueprintRole.HYPOTHESIS_EVIDENCE:
        return [
            f"「{ctx.insight_short}」という仮説は研究ログの数値で検証できる",
            truncate(_strip_period(ctx.log.strongest_finding().claim), 120),
        ]
    if role == BlueprintRole.UNIVERSAL_FRAMEWORK:
        steps = p["steps"]
        return [
            f"検証結果は「{steps[0][0]}」から「{steps[-1][0]}」までの5段階の枠組みとして再現できる",
            "各段階は前の段階の結果を前提とする因果の連鎖になっている",
        ]
    return [
        "明日から実行できる行動は3つに絞れる",
        f"行動の効果は{p['metric_word']}の変化として計測できる",
    ]


def _role_evidence(role: BlueprintRole, ctx: _Ctx) -> list[str]:
    items = [f"{metric_label(k)}: {format_metric(v)}" for k, v in ctx.metric_items]
    if not items:
        items = [truncate(_strip_period(f.claim), 80) for f in ctx.findings]
    slices = {
        BlueprintRole.CONVENTIONAL_WISDOM: items[0:2],
        BlueprintRole.STRUCTURAL_PARADOX: items[2:5] or items[0:2],
        BlueprintRole.HYPOTHESIS_EVIDENCE: items[0:6],
        BlueprintRole.UNIVERSAL_FRAMEWORK: items[-3:],
        BlueprintRole.PRACTICAL_ACTION: items[-2:],
    }
    return slices[role] or items[:2]


def build_blueprint_deterministic(plan: ContentPlan, log: ResearchLog | None = None) -> ArticleBlueprint:
    """企画と研究ログから5部構成の設計図を決定論的に構築する。"""
    ctx = _context(plan, log)
    sections: list[BlueprintSection] = []
    for index, role in enumerate(BLUEPRINT_ROLE_ORDER):
        outline = plan.outline[index] if index < len(plan.outline) else None
        heading = outline.heading if outline is not None else BLUEPRINT_ROLE_LABELS_JA[role.value]
        intent = outline.purpose if outline is not None else BLUEPRINT_ROLE_LABELS_JA[role.value]
        sections.append(
            BlueprintSection(
                role=role,
                heading=heading,
                intent=intent,
                key_claims=_role_claims(role, ctx),
                evidence=_role_evidence(role, ctx),
                figure=_FIGURE_BUILDERS[role](ctx),
            )
        )
    return ArticleBlueprint(
        title=plan.title,
        thesis=truncate(ctx.insight, 400),
        target_reader=plan.target_reader,
        axis=plan.axis,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Deep Section Drafting（決定論）
# ---------------------------------------------------------------------------


def _claims_paragraph(section: BlueprintSection) -> str:
    sentences = [f"{_ORDINALS[i]}に、{_strip_period(claim)}。" for i, claim in enumerate(section.key_claims[:5])]
    return "この章の要点を先に示しておく。" + "".join(sentences)


def _evidence_block(section: BlueprintSection) -> str:
    if not section.evidence:
        return ""
    bullets = "\n".join(f"- {_strip_period(item)}" for item in section.evidence[:6])
    return f"根拠として参照するデータは次のとおりである。\n\n{bullets}"


def _compose_conventional(ctx: _Ctx) -> list[str]:
    p = ctx.profile
    opening = ctx.rng.choice(["現場では", "多くの議論では"])
    return [
        f"{p['domain']}の{opening}、「{p['belief']}」と語られることが多い。{ctx.reader}の多くも、程度の差はあれ"
        "この前提に立っているはずだ。正直に告白すると、私自身も長くそう信じてきた。",
        f"確かに、この常識には一定の合理性がある。なぜなら、{p['belief_reason']}からだ。実際、経験を積んだ"
        f"{p['practitioner']}ほど、言葉にできない判断を数多く下している。その結果として、感覚こそが本質だという理解が"
        "広く共有されてきた。",
        f"例えば、私が{ctx.topic}を分析し始めたときも、最初は同じ前提で作業を進めた。具体的には、{ctx.metric(0)}という"
        f"データが手元にあったにもかかわらず、私はそれを{p['practitioner']}の勘の産物だと解釈していた。"
        f"{len(ctx.findings)}件の発見を並べても、当初はそれが構造の痕跡だとは考えなかったのである。",
        f"しかし、この常識には見落とされがちな前提が潜んでいる。それは「{p['assumption']}」という前提だ。この前提を"
        f"疑わない限り、{ctx.pain}という状態は構造的に再生産される。つまり、努力の量を増やしても、努力の向きそのものは"
        "変わらない。そのため、改善の手応えと実際の成果との差は、時間とともに広がっていく。",
        "もちろん、感覚が無意味だと言いたいのではない。ただし、感覚が働く理由を説明できないままでは、成功を再現する"
        "ことも、失敗の原因を特定することもできない。では、その前提が崩れたとき、私たちの手元には何が残るのだろうか？"
        "あなたの現場にも、うまくいった理由を説明できない成功が1つや2つはあるはずだ。",
        f"本稿では、その説明できなさの正体を{ctx.axis_label}の観点から解き明かす。{ctx.metric(1)}という指標が示すように、"
        "手がかりはすでにデータの中にある。",
    ]


def _compose_paradox(ctx: _Ctx) -> list[str]:
    p = ctx.profile
    return [
        f"私が{ctx.topic}を検証して見えてきたのは、常識に忠実であるほど成果が頭打ちになるという構造的パラドックスだ。"
        "なぜこのような逆説が生じるのか。結論から言えば、問題は個人の技量ではなく、繰り返される判断の構造の側にある。",
        f"具体的には、{ctx.claim(1)}。さらに、{ctx.claim(2)}。これらの観察を並べると、{p['practitioner']}が良かれと"
        "思って重ねる微調整が、同じ局所解の周囲を回り続ける力学として働いていることが分かる。実際に、"
        f"{ctx.metric(2)}というデータは、その停滞が偶然ではないことを示している。",
        f"その原因は、評価の基準が{p['assumption_short']}に固定されていることにある。基準が固定されている限り、改善は"
        "常に同じ方向へのみ進む。したがって、試行回数を増やすほど確信は深まる一方で、構造の外側にある解は視界から"
        "消えていく。言い換えれば、努力が報われない理由は努力の不足ではなく、努力を測る物差しの側にあるのだ。",
        f"もちろん、この指摘には反論もあるだろう。確かに、感覚だけで壁を越える{p['practitioner']}も存在する。ただし、"
        "その成功を再現しようとした瞬間に、私たちは再び同じ構造へ引き戻される。とはいえ、これは感覚を否定する話では"
        "ない。感覚が働く土台を構造として理解する話である。",
        f"要するに、{ctx.pain}という状態は努力不足の症状ではなく、前提が生み出す必然的な帰結なのだ。この逆説を受け入れた"
        "とき、ようやく次の問いが立つ。私たちは何を測れば、構造そのものを動かせるのだろうか？あなたなら、まずどの数値に"
        "目を向けるだろうか？",
    ]


def _compose_hypothesis(ctx: _Ctx) -> list[str]:
    p = ctx.profile
    bullets = "\n".join(
        f"- {_strip_period(f.claim)}（確信度 {f.confidence:.2f}）" for f in ctx.findings[:4]
    )
    return [
        f"そこで私は、「{ctx.insight}」という仮説を立てた。この仮説が正しければ、成果の差は才能の差としてではなく、"
        "測定可能な構造の差として現れるはずだ。なぜなら、構造は観測でき、観測できるものは比較できるからである。",
        f"検証のために、私は{ctx.topic}を5つの層に分解し、それぞれを数値として記録した。主な結果は次のとおりである。"
        f"\n\n{bullets}",
        f"この結果から言えるのは、{_strip_period(ctx.log.strongest_finding().claim)}ということだ。つまり、感情や判断の"
        "動きは偶然の産物ではなく、配置と順序によって説明できる。実際に、この指標を基準に構成を見直すと、どこで集中が"
        "途切れるのかを具体的に指摘できるようになった。その結果、私の検証は「なんとなく良い」から「どこを動かせば"
        "良くなるか」という問いへ移った。",
        f"ただし、この検証には限界もある。データは{len(ctx.findings)}件の発見にもとづくものであり、すべての"
        f"{p['domain']}に当てはまるとは限らない。とはいえ、数値で語れる部分が確実に存在するという事実だけでも、"
        f"{p['practitioner']}の判断を大きく変えるには十分だと私は考えている。",
        f"あなたなら、{ctx.metric(0)}という結果と、{ctx.metric(1)}という結果のどちらから検証を始めるだろうか？",
    ]


def _compose_framework(ctx: _Ctx) -> list[str]:
    p = ctx.profile
    steps = p["steps"]
    step_lines = "\n".join(f"{i}. **{name}**：{desc}" for i, (name, desc) in enumerate(steps, start=1))
    return [
        f"以上の検証を踏まえ、私はこの知見を{ctx.axis_label}の普遍的な枠組みとして整理した。枠組みの本質は、感覚に"
        "頼っていた判断を、順序のある5つの段階へ分解することにある。",
        step_lines,
        f"この枠組みの要点は、各段階が前の段階の結果を前提として成立する因果の連鎖になっている点だ。例えば、"
        f"「{steps[0][0]}」を飛ばして「{steps[2][0]}」に着手すると、何を反転させるべきかが定まらない。したがって、"
        "段階の順序そのものが品質を決める。",
        f"具体的には、{p['application']}。言い換えれば、この枠組みは題材ではなく力学を扱っているため、漫画のネーム、"
        f"モデルの調合、LPの構成のいずれにも写し替えられる。実際に、{ctx.metric(len(ctx.metric_phrases) - 1)}という"
        "構造データは、1つの骨格を複数の領域へ展開できることを裏づけている。",
        "一見すると、5つの段階は手間が増えるだけに見えるかもしれない。しかし実際には、迷いの時間が減るため、総作業量は"
        "むしろ小さくなる。もちろん、すべての案件で全段階を厳密に回す必要はない。ただし、成果が頭打ちになったときこそ、"
        "この順序に立ち返る価値がある。あなたの直近のプロジェクトでは、どの段階が最も手薄だっただろうか？",
    ]


def _compose_action(ctx: _Ctx) -> list[str]:
    p = ctx.profile
    actions = "\n".join(f"{i}. {action}" for i, action in enumerate(p["actions"], start=1))
    return [
        f"最後に、あなたが明日から実践できる行動を3つに絞って示したい。いずれも{p['unit']}単位で小さく始められ、"
        "結果を数値で確認できるものだ。",
        actions,
        "まず、最初の1週間は記録に徹してほしい。具体的には、判断を下した瞬間と、その理由を1行ずつ残す。次に、記録が"
        f"10件を超えたら、その中で{p['metric_word']}が最も動いた場面を3つ選ぶ。最後に、その3つに共通する構造を"
        "言語化する。この3段階を回すだけで、感覚の中に埋もれていた法則が見え始める。",
        "もちろん、最初から完璧な分析は必要ない。ただし、比較の基準だけは固定しておくことが重要だ。なぜなら、基準が"
        "揺れると、改善なのか偶然なのかを区別できなくなるからである。その結果、私たちは再び感覚の議論へ戻ってしまう。",
        f"私自身、この手順を回し始めてから、{ctx.insight_short}という確信を持てるようになった。データは少なくても"
        f"構わない。{ctx.metric(0)}のような1つの指標から始めればよい。重要なのは、構造を測るという習慣そのものだ。"
        "あなたは、どの行動から始めるだろうか？",
    ]


_ROLE_COMPOSERS: dict[BlueprintRole, Callable[[_Ctx], list[str]]] = {
    BlueprintRole.CONVENTIONAL_WISDOM: _compose_conventional,
    BlueprintRole.STRUCTURAL_PARADOX: _compose_paradox,
    BlueprintRole.HYPOTHESIS_EVIDENCE: _compose_hypothesis,
    BlueprintRole.UNIVERSAL_FRAMEWORK: _compose_framework,
    BlueprintRole.PRACTICAL_ACTION: _compose_action,
}


def _elaborations(ctx: _Ctx) -> list[str]:
    p = ctx.profile
    return [
        f"補足しておくと、ここで扱う構造は特定の題材に依存しない。つまり、{p['domain']}で得た知見は、別の領域へも"
        f"そのまま持ち運べる。例えば、{p['scene']}は、他の領域では情報を開示する順序の問題として現れる。",
        f"さらに言えば、構造を測る作業は一度で終わるものではない。{ctx.metric(len(ctx.metric_phrases) - 1)}のような指標は、"
        "題材や受け手が変われば当然変わる。したがって、私は検証を1回の結論ではなく、更新され続ける記録として扱っている。",
        f"この観点に立つと、{p['practitioner']}の仕事は「正解を当てること」から「仮説を速く検証すること」へ変わる。"
        "その結果、失敗は損失ではなく、次の判断を改善するためのデータになる。",
    ]


def _insert_before_figure(body: str, addition: str) -> str:
    """図解（キャプション＋Mermaid）の直前へ段落を挿入する。図解が無ければ末尾へ追記する。"""
    marker = body.find("*図: ")
    if marker < 0:
        marker = body.find("```mermaid")
    if marker < 0:
        return f"{body.rstrip()}\n\n{addition}"
    return f"{body[:marker].rstrip()}\n\n{addition}\n\n{body[marker:]}"


def compose_section(
    blueprint: ArticleBlueprint,
    role: BlueprintRole | str,
    plan: ContentPlan,
    log: ResearchLog | None = None,
    config: WriterConfig | None = None,
) -> DraftSection:
    """設計図の1章を決定論的に執筆する（要点 → 本文 → 根拠データ → 図解）。"""
    cfg = config or WriterConfig()
    role_value = BlueprintRole(role)
    ctx = _context(plan, log)
    section = blueprint.section(role_value)
    paragraphs = [_claims_paragraph(section), *_ROLE_COMPOSERS[role_value](ctx), _evidence_block(section)]
    body = "\n\n".join(p for p in paragraphs if p)
    for extra in _elaborations(ctx):
        if count_prose_chars(body) >= cfg.min_section_chars:
            break
        body = f"{body}\n\n{extra}"
    if cfg.include_mermaid and section.figure.kind == "mermaid" and section.figure.mermaid_code.strip():
        body = f"{body}\n\n{_mermaid_block(section.figure)}"
    return DraftSection(role=role_value, heading=section.heading, body_markdown=body)


def compose_lead(blueprint: ArticleBlueprint, plan: ContentPlan, log: ResearchLog | None = None) -> str:
    ctx = _context(plan, log)
    return (
        f"{plan.hook_line}\n\n私は{ctx.topic}の分析を通じて、「{ctx.insight}」という結論に至った。本稿は{plan.target_reader}"
        f"に向けて、{ctx.axis_label}の視点からその根拠と実践方法をまとめたものである。常識の再確認から始め、構造的"
        "パラドックス、仮説と検証データ、普遍的な枠組み、そして明日からの行動までを順に示す。"
    )


def compose_closing(blueprint: ArticleBlueprint, plan: ContentPlan, log: ResearchLog | None = None) -> str:
    ctx = _context(plan, log)
    return (
        f"本稿では、{ctx.topic}の分析をもとに、{ctx.pain}という状態が努力不足ではなく構造の帰結であることを示し、"
        "それを動かすための5段階の枠組みを提示した。要するに、感覚は否定すべきものではなく、構造という土台の上で"
        f"こそ再現可能な力になる。{_strip_period(plan.call_to_action)}。私はこれからも検証を続け、その結果を共有していきたい。"
    )


def draft_article_deterministic(
    blueprint: ArticleBlueprint,
    plan: ContentPlan,
    log: ResearchLog | None = None,
    config: WriterConfig | None = None,
) -> ArticleDraft:
    """設計図から記事全体を決定論的に執筆する。"""
    sections = [compose_section(blueprint, role, plan, log, config) for role in BLUEPRINT_ROLE_ORDER]
    return ArticleDraft(
        title=blueprint.title,
        lead=compose_lead(blueprint, plan, log),
        sections=sections,
        closing=compose_closing(blueprint, plan, log),
    )


# ---------------------------------------------------------------------------
# Critique Scoring（決定論スコアラー）
# ---------------------------------------------------------------------------

_CONNECTIVES = (
    "つまり", "なぜなら", "したがって", "だから", "そのため", "その結果", "一方で", "しかし", "ただし", "実際",
    "例えば", "たとえば", "具体的には", "要するに", "第一に", "第二に", "第三に", "次に", "最後に", "まず", "よって",
    "ゆえに", "これにより", "言い換えれば", "逆に", "にもかかわらず", "とはいえ", "さらに", "加えて", "このように",
    "以上から", "結果として", "裏を返せば",
)
_CAUSAL = ("ため", "から", "ので", "結果", "原因", "理由", "によって", "により", "ゆえ", "おかげで", "せいで")
_ABSTRACT = (
    "構造", "本質", "原理", "枠組み", "フレームワーク", "普遍", "抽象", "概念", "法則", "仮説", "モデル", "体系",
    "力学", "メカニズム", "パターン", "原則", "前提", "パラダイム", "理論",
)
_CONCRETE = ("例えば", "たとえば", "具体的には", "実際", "事例", "ケース", "場面", "具体例", "実例", "手順")
_DATA = ("データ", "指標", "係数", "スコア", "測定", "実験", "ログ", "比較", "検証", "数値", "統計", "サンプル", "観測", "計測", "推定", "分析")
_COUNTER = ("反論", "確かに", "もちろん", "ただし", "とはいえ", "一見", "異論", "批判", "限界", "例外", "とは限らない", "注意が必要")
_READER = ("あなた", "読者", "皆さん")
_QUESTION_RE = re.compile(r"[？?]|のだろうか|ではないだろうか|だろうか")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_KEYWORD_RE = re.compile(r"[一-龥々〆ヵヶ]{2,}|[ァ-ヺー]{2,}|[A-Za-z][A-Za-z0-9\-]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n]+")

_CONNECTIVE_TARGET = 0.30
_CAUSAL_TARGET = 0.30
_FEEDBACK_THRESHOLD = 21.25


def _prose_of(markdown: str) -> str:
    return "".join(chunk for kind, chunk in split_markdown_segments(markdown) if kind == "prose")


def _count(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def _sentences(prose: str) -> list[str]:
    return [s.strip(" \t-*0123456789.、") for s in _SENTENCE_SPLIT_RE.split(prose) if len(s.strip()) >= 4]


def _claim_covered(claim: str, body: str) -> bool:
    keywords = _unique(_KEYWORD_RE.findall(claim))
    if not keywords:
        return True
    hits = sum(1 for keyword in keywords if keyword in body)
    return hits / len(keywords) >= 0.6


def score_section(
    section: DraftSection,
    blueprint_section: BlueprintSection | None = None,
    config: WriterConfig | None = None,
) -> SectionScore:
    """論理・抽象と具体・納得度・根拠密度を各25点で採点する（決定論）。"""
    cfg = config or WriterConfig()
    prose = _prose_of(section.body_markdown)
    sentences = _sentences(prose)
    n_sentences = max(1, len(sentences))
    n_chars = count_prose_chars(section.body_markdown)

    connective_rate = _count(prose, _CONNECTIVES) / n_sentences
    causal_rate = _count(prose, _CAUSAL) / n_sentences
    if blueprint_section is not None and blueprint_section.key_claims:
        coverage = sum(1 for c in blueprint_section.key_claims if _claim_covered(c, prose)) / len(blueprint_section.key_claims)
    else:
        coverage = 1.0
    logic = 25.0 * (
        0.40 * min(1.0, connective_rate / _CONNECTIVE_TARGET)
        + 0.25 * min(1.0, causal_rate / _CAUSAL_TARGET)
        + 0.35 * coverage
    )

    numbers = len(_NUMBER_RE.findall(prose))
    abstract = _count(prose, _ABSTRACT)
    concrete = _count(prose, _CONCRETE) + min(numbers, 6)
    if abstract == 0 or concrete == 0:
        balance = 0.0
    else:
        ratio = concrete / (abstract + concrete)
        balance = 1.0 - min(1.0, abs(ratio - 0.5) / 0.5)
    richness = min(1.0, min(abstract, concrete) / 4.0)
    abstraction = 25.0 * (0.55 * balance + 0.45 * richness)

    evidence_sentences = sum(1 for s in sentences if _NUMBER_RE.search(s) or any(term in s for term in _DATA))
    counter = _count(prose, _COUNTER)
    questions = len(_QUESTION_RE.findall(prose))
    reader = _count(prose, _READER)
    persuasion = 25.0 * (
        0.40 * min(1.0, evidence_sentences / 4.0)
        + 0.30 * min(1.0, counter / 2.0)
        + 0.15 * min(1.0, float(questions))
        + 0.15 * min(1.0, float(reader))
    )

    numbers_per_400 = numbers / max(1.0, n_chars / 400.0)
    data_refs = _count(prose, _DATA)
    has_figure = 1.0 if "```mermaid" in section.body_markdown else 0.0
    length_ratio = min(1.0, n_chars / float(cfg.min_section_chars))
    evidence = 25.0 * (
        0.35 * min(1.0, numbers_per_400 / 2.0)
        + 0.30 * min(1.0, data_refs / 4.0)
        + 0.15 * has_figure
        + 0.20 * length_ratio
    )

    feedback: list[str] = []
    if logic < _FEEDBACK_THRESHOLD:
        feedback.append("【論理】接続詞と因果の標識を増やし、主張と根拠の対応を「第一に」「したがって」で明示する")
    if abstraction < _FEEDBACK_THRESHOLD:
        side = "具体（数値・場面・手順）" if concrete <= abstract else "抽象（構造・原理・前提）"
        feedback.append(f"【抽象と具体】不足している{side}を補う段落を加える")
    if persuasion < _FEEDBACK_THRESHOLD:
        feedback.append("【納得度】反論の先回り（確かに・もちろん・ただし）と読者への問いかけを加える")
    if evidence < _FEEDBACK_THRESHOLD:
        feedback.append("【根拠】研究ログの数値とデータ参照を増やし、図解を挿入する")
    if n_chars < cfg.min_section_chars:
        feedback.append(f"【分量】本文が{n_chars}字と短い。{cfg.min_section_chars}字以上へ拡充する")

    return SectionScore(
        role=section.role,
        heading=section.heading,
        logic_coherence=round(min(25.0, logic), 2),
        abstraction_balance=round(min(25.0, abstraction), 2),
        persuasiveness=round(min(25.0, persuasion), 2),
        evidence_density=round(min(25.0, evidence), 2),
        feedback=feedback,
    )


def _assemble_critique(sections: list[SectionScore], threshold: float) -> CritiqueScore:
    overall = round(sum(s.total for s in sections) / len(sections), 2)
    passed = overall >= threshold and all(s.total >= threshold for s in sections)
    return CritiqueScore(sections=sections, overall=overall, passed=passed, threshold=threshold)


def critique_deterministic(
    draft: ArticleDraft,
    blueprint: ArticleBlueprint | None = None,
    threshold: float = 80.0,
    config: WriterConfig | None = None,
) -> CritiqueScore:
    """記事ドラフト全章を決定論的に採点する。"""
    scores: list[SectionScore] = []
    for section in draft.sections:
        blueprint_section = blueprint.section(section.role) if blueprint is not None else None
        scores.append(score_section(section, blueprint_section, config))
    return _assemble_critique(scores, threshold)


# ---------------------------------------------------------------------------
# Iterative Polish（決定論）
# ---------------------------------------------------------------------------


def _polish_logic(ctx: _Ctx, section: BlueprintSection | None, round_index: int) -> str:
    claims = [_strip_period(c) for c in (section.key_claims if section else [])] or [ctx.claim(0), ctx.claim(1)]
    second = claims[1] if len(claims) > 1 else ctx.claim(round_index)
    lead = ("ここで論理の筋道を整理しておく。", "改めて推論の順序を確認しておきたい。", "議論の骨格をもう一度たどっておく。")[
        round_index % 3
    ]
    return (
        f"{lead}第一に、{claims[0]}。第二に、{second}。したがって、{ctx.insight_short}と言える。なぜなら、前提が"
        "変われば判断の基準が変わり、その結果として成果の分布そのものが変わるからだ。つまり、この章の主張は"
        "観察と因果の両方によって支えられている。"
    )


def _polish_concrete(ctx: _Ctx, round_index: int) -> str:
    p = ctx.profile
    return (
        f"具体例を{round_index}つ挙げよう。例えば、{p['scene']}を考えてほしい。実際に、{ctx.metric(round_index)}という"
        f"データは、この場面で何が起きているかを数値で示している。具体的には、{p['application']}。"
    )


def _polish_abstract(ctx: _Ctx, round_index: int) -> str:
    return (
        "この現象の本質は、個別の技法ではなく構造の側にある。つまり、前提・限界・反転・代償・余韻という5つの要素が、"
        f"どの題材でも同じ力学として働くという原理である。{ctx.axis_label}の枠組みとして捉え直すと、"
        f"{round_index}回目の検証で得た指標の意味がより明確になる。"
    )


def _polish_persuasion(ctx: _Ctx, round_index: int) -> str:
    p = ctx.profile
    closing = ("あなたはどちらの立場で次の一手を選ぶだろうか？", "あなたの現場では、この反論はどこまで当てはまるだろうか？")[
        round_index % 2
    ]
    return (
        f"確かに、「{p['objection']}」という反論もあるだろう。しかし、構造は表現を縛る檻ではなく、表現が届く確率を"
        "上げる土台である。もちろん、すべてを構造で説明できるわけではない。とはいえ、説明できる部分を説明しないまま"
        f"にしておく理由もない。{closing}"
    )


def _polish_evidence(ctx: _Ctx) -> str:
    bullets = "\n".join(f"- {phrase}" for phrase in ctx.metric_phrases[:6])
    return (
        f"数値で確認しておこう。\n\n{bullets}\n\nこれらのデータは、{len(ctx.findings)}件の検証から得られた指標であり、"
        "比較の基準として機能する。したがって、次の判断はこの数値を起点に行えばよい。"
    )


def polish_section_deterministic(
    section: DraftSection,
    score: SectionScore | None,
    blueprint_section: BlueprintSection | None,
    plan: ContentPlan,
    log: ResearchLog | None = None,
    config: WriterConfig | None = None,
    round_index: int = 1,
) -> DraftSection:
    """弱い観点を補強する段落を追加して章を再推敲する（採点が下がる改稿は採用しない）。"""
    cfg = config or WriterConfig()
    ctx = _context(plan, log)
    current = score_section(section, blueprint_section, cfg)
    body = section.body_markdown
    additions: list[str] = []
    if current.logic_coherence < _FEEDBACK_THRESHOLD:
        additions.append(_polish_logic(ctx, blueprint_section, round_index))
    if current.abstraction_balance < _FEEDBACK_THRESHOLD:
        prose = _prose_of(body)
        numbers = len(_NUMBER_RE.findall(prose))
        concrete = _count(prose, _CONCRETE) + min(numbers, 6)
        abstract = _count(prose, _ABSTRACT)
        additions.append(_polish_concrete(ctx, round_index) if concrete <= abstract else _polish_abstract(ctx, round_index))
    if current.persuasiveness < _FEEDBACK_THRESHOLD:
        additions.append(_polish_persuasion(ctx, round_index))
    if current.evidence_density < _FEEDBACK_THRESHOLD:
        additions.append(_polish_evidence(ctx))
    elaborations = _elaborations(ctx)
    position = 0
    while count_prose_chars("\n\n".join([body, *additions])) < cfg.min_section_chars and position < len(elaborations):
        additions.append(elaborations[position])
        position += 1
    if not additions:
        return section
    for addition in additions:
        body = _insert_before_figure(body, addition)
    if cfg.include_mermaid and "```mermaid" not in body and blueprint_section is not None:
        figure = blueprint_section.figure
        if figure.kind == "mermaid" and figure.mermaid_code.strip():
            body = f"{body}\n\n{_mermaid_block(figure)}"
    candidate = DraftSection(role=section.role, heading=section.heading, body_markdown=body)
    if score_section(candidate, blueprint_section, cfg).total + 1e-9 < current.total:
        return section
    return candidate


# ---------------------------------------------------------------------------
# パイプライン
# ---------------------------------------------------------------------------


async def _notify(callback: ProgressCallback | None, stage: str, progress: float, message: str) -> None:
    if callback is None:
        return
    result = callback(stage, round(min(1.0, max(0.0, progress)), 4), message)
    if inspect.isawaitable(result):
        await result


class ArticleWriterPipeline:
    """5段階の長文執筆パイプライン。"""

    def __init__(
        self,
        llm_client: Any | None = None,
        config: WriterConfig | None = None,
        harness_config: Any | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or WriterConfig()
        self.harness_config = harness_config
        self.sanitizer = ComplianceSanitizer()
        self._harness: Any | None = None
        if llm_client is not None:
            from core.llm_client import HarnessConfig, MultiStageReasoningHarness

            self._harness = MultiStageReasoningHarness(llm_client, harness_config or HarnessConfig())

    @property
    def uses_llm(self) -> bool:
        return self._harness is not None

    async def _harness_output(self, *, rubric_stage: str, **kwargs: Any) -> Any | None:
        """ハーネスを実行し、失敗時は None を返す（呼び出し側が決定論ベースラインへ退避する）。"""
        if self._harness is None:
            return None
        from core.asset_publisher.prompts import publisher_rubric
        from core.llm_client import LLMError

        try:
            result = await self._harness.run(rubric=publisher_rubric(rubric_stage), **kwargs)
        except (LLMError, ValidationError, ValueError) as exc:
            logger.warning("Asset Publisher の LLM 経路（%s）が失敗したため決定論ベースラインを使用します: %s", kwargs.get("task"), exc)
            return None
        return result.output

    def _base_context(self, plan: ContentPlan, log: ResearchLog | None) -> dict[str, Any]:
        return {
            "pub_plan": plan.model_dump(mode="json"),
            "pub_research_log": log.model_dump(mode="json") if log is not None else None,
            "pub_config": self.config.model_dump(mode="json"),
        }

    # Stage 1 -------------------------------------------------------------
    async def build_blueprint(self, plan: ContentPlan, log: ResearchLog | None = None) -> ArticleBlueprint:
        baseline = build_blueprint_deterministic(plan, log)
        if self._harness is None:
            return baseline
        from core.asset_publisher.prompts import BLUEPRINT_SYSTEM_PROMPT

        def _validate(blueprint: ArticleBlueprint) -> list[str]:
            issues: list[str] = []
            for section in blueprint.sections:
                if section.figure.kind != "mermaid":
                    issues.append(f"{section.role.value} の figure は kind=mermaid にしてください")
                if find_urls(" ".join([section.heading, section.intent, *section.key_claims, *section.evidence])):
                    issues.append(f"{section.role.value} にURLを含めないでください")
            return issues

        output = await self._harness_output(
            rubric_stage="blueprint",
            task="publisher.blueprint",
            objective="企画と研究ログから、常識→パラドックス→仮説と検証→フレームワーク→実践の5部構成の設計図を作る",
            payload={"plan": plan.model_dump(mode="json"), "research_log": log.model_dump(mode="json") if log else None},
            output_model=ArticleBlueprint,
            system_prompt=BLUEPRINT_SYSTEM_PROMPT,
            context=self._base_context(plan, log),
            post_validate=_validate,
        )
        if output is None:
            return baseline
        return output.model_copy(update={"axis": plan.axis})

    # Stage 2 -------------------------------------------------------------
    async def draft_section(
        self, blueprint: ArticleBlueprint, role: BlueprintRole, plan: ContentPlan, log: ResearchLog | None = None
    ) -> DraftSection:
        baseline = compose_section(blueprint, role, plan, log, self.config)
        if self._harness is None:
            return baseline
        from core.asset_publisher.prompts import SECTION_DRAFT_SYSTEM_PROMPT

        section = blueprint.section(role)
        cfg = self.config

        def _validate(draft: DraftSection) -> list[str]:
            issues: list[str] = []
            if draft.char_count < cfg.min_section_chars:
                issues.append(f"本文が{draft.char_count}字です。{cfg.min_section_chars}字以上に拡充してください")
            if cfg.include_mermaid and "```mermaid" not in draft.body_markdown:
                issues.append("章末に ```mermaid の図解を挿入してください")
            if find_urls(draft.body_markdown):
                issues.append("URLやリンク構文を削除してください")
            if detect_first_person_variants(draft.body_markdown):
                issues.append("一人称は「私」に統一してください")
            return issues

        output = await self._harness_output(
            rubric_stage="section",
            task="publisher.section_draft",
            objective=f"記事の第{BLUEPRINT_ROLE_ORDER.index(role) + 1}章（{BLUEPRINT_ROLE_LABELS_JA[role.value]}）を高密度に執筆する",
            payload={
                "blueprint_section": section.model_dump(mode="json"),
                "thesis": blueprint.thesis,
                "target_reader": blueprint.target_reader,
                "findings": [f.model_dump(mode="json") for f in (log.findings if log else [])],
            },
            output_model=DraftSection,
            system_prompt=SECTION_DRAFT_SYSTEM_PROMPT,
            context={**self._base_context(plan, log), "pub_blueprint": blueprint.model_dump(mode="json"), "pub_role": role.value},
            post_validate=_validate,
        )
        if output is None:
            return baseline
        candidate = DraftSection(role=role, heading=output.heading or section.heading, body_markdown=output.body_markdown)
        if candidate.char_count < cfg.min_section_chars or (cfg.include_mermaid and "```mermaid" not in candidate.body_markdown):
            return baseline
        return candidate

    async def draft_sections(
        self,
        blueprint: ArticleBlueprint,
        plan: ContentPlan,
        log: ResearchLog | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ArticleDraft:
        sections: list[DraftSection] = []
        for index, role in enumerate(BLUEPRINT_ROLE_ORDER):
            await _notify(on_progress, "draft", 0.15 + 0.08 * index, f"第{index + 1}章「{blueprint.section(role).heading}」を執筆中")
            sections.append(await self.draft_section(blueprint, role, plan, log))
        return ArticleDraft(
            title=blueprint.title,
            lead=compose_lead(blueprint, plan, log),
            sections=sections,
            closing=compose_closing(blueprint, plan, log),
        )

    # Stage 3 -------------------------------------------------------------
    def sanitize(self, draft: ArticleDraft) -> tuple[ArticleDraft, SanitizationReport]:
        return self.sanitizer.sanitize_draft(draft)

    # Stage 4 -------------------------------------------------------------
    async def critique(self, draft: ArticleDraft, blueprint: ArticleBlueprint | None = None) -> CritiqueScore:
        threshold = self.config.quality_threshold
        deterministic = critique_deterministic(draft, blueprint, threshold, self.config)
        if self._harness is None:
            return deterministic
        from core.asset_publisher.prompts import CRITIQUE_SYSTEM_PROMPT

        output = await self._harness_output(
            rubric_stage="critique",
            task="publisher.critique",
            objective="記事の各章を論理・抽象と具体・納得度・根拠密度の4観点×25点で厳格に採点する",
            payload={"article_markdown": draft.to_markdown(), "threshold": threshold},
            output_model=CritiqueScore,
            system_prompt=CRITIQUE_SYSTEM_PROMPT,
            context={
                "pub_article_draft": draft.model_dump(mode="json"),
                "pub_blueprint": blueprint.model_dump(mode="json") if blueprint is not None else None,
                "pub_threshold": threshold,
                "pub_config": self.config.model_dump(mode="json"),
            },
        )
        if output is None:
            return deterministic
        merged: list[SectionScore] = []
        for base in deterministic.sections:
            llm = output.section(base.role)
            if llm is None:
                merged.append(base)
                continue
            merged.append(
                SectionScore(
                    role=base.role,
                    heading=base.heading,
                    logic_coherence=round((base.logic_coherence + llm.logic_coherence) / 2.0, 2),
                    abstraction_balance=round((base.abstraction_balance + llm.abstraction_balance) / 2.0, 2),
                    persuasiveness=round((base.persuasiveness + llm.persuasiveness) / 2.0, 2),
                    evidence_density=round((base.evidence_density + llm.evidence_density) / 2.0, 2),
                    feedback=_unique([*base.feedback, *llm.feedback]),
                )
            )
        return _assemble_critique(merged, threshold)

    # Stage 5 -------------------------------------------------------------
    async def polish_section(
        self,
        section: DraftSection,
        score: SectionScore,
        blueprint: ArticleBlueprint,
        plan: ContentPlan,
        log: ResearchLog | None = None,
        round_index: int = 1,
    ) -> DraftSection:
        blueprint_section = blueprint.section(section.role)
        baseline = polish_section_deterministic(section, score, blueprint_section, plan, log, self.config, round_index)
        if self._harness is None:
            return baseline
        from core.asset_publisher.prompts import POLISH_SYSTEM_PROMPT

        cfg = self.config

        def _validate(draft: DraftSection) -> list[str]:
            issues: list[str] = []
            if draft.char_count < cfg.min_section_chars:
                issues.append(f"本文を{cfg.min_section_chars}字以上にしてください")
            if cfg.include_mermaid and "```mermaid" not in draft.body_markdown:
                issues.append("図解（```mermaid）を保持してください")
            return issues

        output = await self._harness_output(
            rubric_stage="polish",
            task="publisher.polish",
            objective=f"基準点未満の章（{BLUEPRINT_ROLE_LABELS_JA[section.role.value]}）を採点の指摘に沿って再推敲する",
            payload={"section": section.model_dump(mode="json"), "score": score.model_dump(mode="json")},
            output_model=DraftSection,
            system_prompt=POLISH_SYSTEM_PROMPT,
            context={
                **self._base_context(plan, log),
                "pub_section": section.model_dump(mode="json"),
                "pub_section_score": score.model_dump(mode="json"),
                "pub_blueprint": blueprint.model_dump(mode="json"),
                "pub_round": round_index,
            },
            post_validate=_validate,
        )
        if output is None:
            return baseline
        candidate = DraftSection(role=section.role, heading=section.heading, body_markdown=output.body_markdown)
        candidate_total = score_section(candidate, blueprint_section, cfg).total
        baseline_total = score_section(baseline, blueprint_section, cfg).total
        return candidate if candidate_total >= baseline_total else baseline

    async def polish(
        self,
        draft: ArticleDraft,
        critique: CritiqueScore,
        blueprint: ArticleBlueprint,
        plan: ContentPlan,
        log: ResearchLog | None = None,
        round_index: int = 1,
    ) -> ArticleDraft:
        """基準点未満の章のみを再推敲した新しいドラフトを返す。"""
        weak = {score.role for score in critique.weak_sections()}
        sections: list[DraftSection] = []
        for section in draft.sections:
            if section.role in weak:
                score = critique.section(section.role) or score_section(section, blueprint.section(section.role), self.config)
                sections.append(await self.polish_section(section, score, blueprint, plan, log, round_index))
            else:
                sections.append(section)
        return draft.model_copy(update={"sections": sections})

    # Orchestration -------------------------------------------------------
    async def run(
        self,
        plan: ContentPlan,
        log: ResearchLog | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> PublishedArticle:
        """5段階パイプラインを実行し、公開可能な記事を返す。"""
        stages: list[PipelineStageRecord] = []

        def record(stage: str, iteration: int, started: float, score: float | None = None, notes: str = "") -> None:
            stages.append(
                PipelineStageRecord(
                    stage=stage,
                    iteration=iteration,
                    duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                    score=score,
                    notes=notes,
                )
            )

        await _notify(on_progress, "blueprint", 0.05, "記事設計図（5部構成）を作成中")
        started = time.perf_counter()
        blueprint = await self.build_blueprint(plan, log)
        record("blueprint", 0, started, notes=f"{len(blueprint.sections)}章の設計図")

        started = time.perf_counter()
        draft = await self.draft_sections(blueprint, plan, log, on_progress)
        record("draft", 0, started, notes=f"本文 {draft.total_chars} 字")

        await _notify(on_progress, "sanitize", 0.6, "一人称の統一とURLの完全除去を実行中")
        started = time.perf_counter()
        draft, report = self.sanitize(draft)
        record(
            "sanitize",
            0,
            started,
            notes=f"一人称置換 {len(report.first_person_replacements)} 件 / URL除去 {len(report.urls_removed)} 件",
        )

        await _notify(on_progress, "critique", 0.7, "100点満点で章別に採点中")
        started = time.perf_counter()
        critique = await self.critique(draft, blueprint)
        record("critique", 0, started, score=critique.overall, notes="合格" if critique.passed else "基準未満の章あり")
        history = [critique]
        rounds = 0
        while not critique.passed and rounds < self.config.max_polish_rounds:
            rounds += 1
            await _notify(on_progress, "polish", 0.72 + 0.06 * rounds, f"基準未満の章を再推敲中（{rounds}回目）")
            started = time.perf_counter()
            candidate = await self.polish(draft, critique, blueprint, plan, log, rounds)
            candidate, polish_report = self.sanitize(candidate)
            new_critique = await self.critique(candidate, blueprint)
            if candidate.to_markdown() == draft.to_markdown():
                record("polish", rounds, started, score=critique.overall, notes="改善可能な箇所が無いため終了")
                break
            if new_critique.overall + 1e-9 < critique.overall:
                record("polish", rounds, started, score=critique.overall, notes="推敲案は採点が低下したため棄却")
                break
            report = report.merge(polish_report)
            draft, critique = candidate, new_critique
            history.append(new_critique)
            record("polish", rounds, started, score=new_critique.overall, notes="合格" if new_critique.passed else "継続")

        await _notify(on_progress, "finalize", 0.95, "最終サニタイズと検証を実行中")
        started = time.perf_counter()
        markdown, final_report = sanitize_text(draft.to_markdown())
        report = report.merge(final_report)
        record(
            "finalize",
            rounds,
            started,
            score=critique.overall,
            notes=f"一人称統一={report.first_person_unified} / URLゼロ={report.url_free}",
        )
        research = log or log_from_plan(plan)
        article = PublishedArticle(
            title=draft.title,
            axis=plan.axis,
            source_id=research.id,
            markdown=markdown,
            blueprint=blueprint,
            draft=draft,
            sanitization=report,
            critique_history=history,
            final_score=critique.overall,
            iterations=rounds,
            stages=stages,
            tags=_unique([*plan.keywords, *research.tags])[:12],
        )
        await _notify(on_progress, "complete", 1.0, f"記事を生成しました（{critique.overall:.1f}点）")
        return article
