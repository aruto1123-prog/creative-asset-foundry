"""Phase 5: Asset Publisher（3軸展開・5段階執筆・サニタイズ・ナレッジグラフ・編纂）のテスト。"""

from __future__ import annotations

import asyncio
import math
import re

import pytest

from core.asset_publisher.exporter import (
    SeriesCompiler,
    export_note_markdown,
    export_obsidian_markdown,
    export_series_markdown,
    mermaid_to_text,
)
from core.asset_publisher.knowledge_graph import KnowledgeGraph
from core.asset_publisher.matrix_transpiler import (
    ContentMatrixTranspiler,
    expand_matrix_deterministic,
    research_log_from_experiment,
    research_log_from_structure,
)
from core.asset_publisher.repository import ArticleRepository
from core.asset_publisher.sanitizer import (
    ComplianceSanitizer,
    detect_first_person_variants,
    find_urls,
    remove_urls,
    sanitize_text,
    unify_first_person,
)
from core.asset_publisher.schemas import (
    AXIS_ORDER,
    BLUEPRINT_ROLE_ORDER,
    ArticleDraft,
    BlueprintRole,
    ContentAxis,
    DraftSection,
    PublishedArticle,
)
from core.asset_publisher.writer_pipeline import (
    ArticleWriterPipeline,
    WriterConfig,
    build_blueprint_deterministic,
    compose_section,
    critique_deterministic,
    draft_article_deterministic,
    score_section,
)
from core.structure_engine.samples import SAMPLE_KINDS, sample_structure, sample_structures

# ---------------------------------------------------------------------------
# 共通フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def logs():
    return {kind: research_log_from_structure(sample_structure(kind)) for kind in SAMPLE_KINDS}


@pytest.fixture(scope="module")
def matrices(logs):
    return {kind: expand_matrix_deterministic(log) for kind, log in logs.items()}


@pytest.fixture(scope="module")
def published(logs, matrices) -> dict[tuple[str, str], PublishedArticle]:
    async def build() -> dict[tuple[str, str], PublishedArticle]:
        results: dict[tuple[str, str], PublishedArticle] = {}
        for kind in SAMPLE_KINDS:
            for plan in matrices[kind].plans:
                results[(kind, plan.axis.value)] = await ArticleWriterPipeline().run(plan, logs[kind])
        return results

    return asyncio.run(build())


# ---------------------------------------------------------------------------
# サニタイザー: 一人称
# ---------------------------------------------------------------------------

_SINGULAR_CASES = [
    ("僕は考えた。", "私は考えた。"),
    ("ぼくは考えた。", "私は考えた。"),
    ("ボクが試した。", "私が試した。"),
    ("俺の仮説だ。", "私の仮説だ。"),
    ("おれは走った。", "私は走った。"),
    ("オレも驚いた。", "私も驚いた。"),
    ("わたしは書く。", "私は書く。"),
    ("ワタシの結論。", "私の結論。"),
    ("わたくしは申し上げる。", "私は申し上げる。"),
    ("あたしは見た。", "私は見た。"),
    ("アタシが選んだ。", "私が選んだ。"),
    ("わしは知らん。", "私は知らん。"),
    ("ワシが行く。", "私が行く。"),
    ("拙者は参る。", "私は参る。"),
    ("小生は考える。", "私は考える。"),
    ("吾輩は猫である。", "私は猫である。"),
    ("我輩は語る。", "私は語る。"),
    ("筆者は検証した。", "私は検証した。"),
    ("当方では対応する。", "私では対応する。"),
    ("俺様が決める。", "私が決める。"),
    ("おいらは行く。", "私は行く。"),
    ("オイラが作った。", "私が作った。"),
]

_PLURAL_CASES = [
    ("僕たちは進む。", "私たちは進む。"),
    ("僕らの結論。", "私たちの結論。"),
    ("俺たちが勝つ。", "私たちが勝つ。"),
    ("俺らは走る。", "私たちは走る。"),
    ("我々は前進する。", "私たちは前進する。"),
    ("われわれは考える。", "私たちは考える。"),
    ("私達は検証した。", "私たちは検証した。"),
    ("わたしたちは書く。", "私たちは書く。"),
    ("僕達の挑戦。", "私たちの挑戦。"),
    ("俺達が作る。", "私たちが作る。"),
    ("筆者らは比較した。", "私たちは比較した。"),
    ("ぼくらが選ぶ。", "私たちが選ぶ。"),
]


@pytest.mark.parametrize(("source", "expected"), _SINGULAR_CASES + _PLURAL_CASES)
def test_first_person_variants_are_unified(source: str, expected: str) -> None:
    unified, replacements = unify_first_person(source)
    assert unified == expected
    assert replacements
    assert detect_first_person_variants(unified) == []


@pytest.mark.parametrize(
    "text",
    [
        "下僕と公僕と従僕と家僕と忠僕は職業名である。",
        "僕婢という古語がある。",
        "お礼（おれい）を伝える。",
        "オレンジとボクシングとワシントンとカフェオレ。",
        "小生意気な態度。",
        "原著の筆者は別の研究者だ。",
        "読者は、自分の経験に照らして判断する。",
        "自分自身で考えることが大切だ。",
        "ご自分の判断で選んでほしい。",
        "あなたが自分で試すとよい。",
    ],
)
def test_first_person_false_positives_are_preserved(text: str) -> None:
    unified, replacements = unify_first_person(text)
    assert unified == text
    assert replacements == []
    assert detect_first_person_variants(text) == []


def test_jibun_subject_usage_is_replaced() -> None:
    text = "自分は長くそう信じてきた。そのとき、自分が間違っていたと気づいた。自分たちの方法を見直す。"
    unified, replacements = unify_first_person(text)
    assert unified == "私は長くそう信じてきた。そのとき、私が間違っていたと気づいた。私たちの方法を見直す。"
    assert [r.original for r in replacements] == ["自分", "自分", "自分たち"]


def test_first_person_respects_code_and_targets_mermaid() -> None:
    text = (
        "本文で僕は語る。`僕`はインラインコード。\n\n"
        "```python\n# 僕のコード\nprint('俺')\n```\n\n"
        "```mermaid\nflowchart LR\n    A[\"僕の仮説\"] --> B[\"検証\"]\n```\n"
    )
    unified, replacements = unify_first_person(text)
    assert "本文で私は語る。" in unified
    assert "`僕`" in unified
    assert "# 僕のコード" in unified and "print('俺')" in unified
    assert 'A["私の仮説"]' in unified
    assert len(replacements) == 2


def test_replacement_records_point_to_original_positions() -> None:
    text = "前置き。僕は書いた。そして俺たちは考えた。"
    _, replacements = unify_first_person(text)
    for replacement in replacements:
        assert text[replacement.index : replacement.index + len(replacement.original)] == replacement.original
        assert replacement.original in replacement.context


# ---------------------------------------------------------------------------
# サニタイザー: URL
# ---------------------------------------------------------------------------

_URL_CASES = [
    ("詳細はhttps://example.com/path?a=1をご覧ください。", "bare_url", "詳細はをご覧ください。"),
    ("旧来の http://example.org/x と ftp://files.example.net/a.zip", "bare_url", None),
    ("file:///home/user/secret.txt を参照", "bare_url", None),
    ("全角 ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ／ｐａｔｈ です", "bare_url", "全角 です"),
    ("全角 ｗｗｗ．ｅｘａｍｐｌｅ．ｃｏｍ です", "bare_url", "全角 です"),
    ("訪問先は www.example.com/page です", "bare_url", None),
    ("hxxps://evil.example.com/x を避ける", "defanged_url", None),
    ("ttps://bad.example.net に注意", "defanged_url", None),
    ("h**ps://masked.example.com も危険", "defanged_url", None),
    ("http[:]//bracket.example.com も", "defanged_url", None),
    ("example[.]com と sub(.)example(.)org", "defanged_url", None),
    ("example dot com へ", "defanged_url", None),
    ("[リンク文字](https://example.com \"title\") を読む", "markdown_link", "リンク文字 を読む"),
    ("![画像](https://img.example.com/a.png)の説明", "markdown_image", "の説明"),
    ("<https://auto.example.com> を開く", "autolink", "を開く"),
    ("<a href='https://x.example.com' target=_blank>アンカー</a>", "html_anchor", "アンカー"),
    ('<A target="x" HREF="https://y.example.com">大文字</A>', "html_anchor", "大文字"),
    ('<img src="https://i.example.com/a.png" alt="x">画像', "html_media", "画像"),
    ('<iframe src="https://v.example.com/embed"></iframe>動画', "html_media", "動画"),
    ('<script src="https://cdn.example.com/a.js"></script>本文', "html_media", "本文"),
    ("mailto:someone@example.com へ連絡", "bare_url", None),
    ("javascript:alert(1) は禁止", "bare_url", None),
    ("data:text/html;base64,AAAA を埋め込む", "bare_url", None),
    ("//cdn.example.com/lib.js を読み込む", "bare_url", None),
    ("公式サイト example.com を見る", "domain_like", None),
    ("sub.example.co.jp/path が資料", "domain_like", None),
    ("github.io/foo にある", "domain_like", None),
]


@pytest.mark.parametrize(("source", "kind", "expected"), _URL_CASES)
def test_url_forms_are_removed(source: str, kind: str, expected: str | None) -> None:
    assert find_urls(source), f"検出漏れ: {source}"
    cleaned, removed = remove_urls(source)
    assert find_urls(cleaned) == []
    assert kind in {r.kind for r in removed}
    if expected is not None:
        assert cleaned == expected


@pytest.mark.parametrize(
    "host",
    [
        "bit.ly/3abc", "amzn.to/xyz", "a.r10.to/hA", "t.co/abc", "goo.gl/x", "tinyurl.com", "ow.ly/y",
        "buff.ly/z", "lnkd.in/q", "rebrand.ly/w", "is.gd/x", "cutt.ly/k", "x.gd/y", "a.co/d",
    ],
)
def test_shortener_and_affiliate_hosts_are_removed(host: str) -> None:
    text = f"おすすめは {host} から"
    cleaned, removed = remove_urls(text)
    assert host.split("/")[0] not in cleaned
    assert removed and removed[0].kind == "domain_like"
    assert find_urls(cleaned) == []


@pytest.mark.parametrize(
    "token",
    ["main.py", "README.md", "Node.js", "config.json", "model.safetensors", "index.html", "Socket.io",
     "model.to(device)", "e.g.", "v1.2.3", "0.85", "train.sh", "setup.cfg", "data.csv"],
)
def test_file_names_and_technical_terms_are_not_misdetected(token: str) -> None:
    text = f"ここで {token} を使う。"
    assert find_urls(text) == []
    cleaned, removed = remove_urls(text)
    assert cleaned == text
    assert removed == []


def test_reference_links_and_definitions() -> None:
    text = "参照リンク[資料][ref1]と[ref2]を読む。\n\n[ref1]: https://a.example.com\n[ref2]: <https://b.example.com> \"T\"\n脚注[^1]\n\n[^1]: 脚注本文\n"
    cleaned, removed = remove_urls(text)
    kinds = [r.kind for r in removed]
    assert kinds.count("reference_definition") == 2
    assert kinds.count("reference_link") == 2
    assert "参照リンク資料とref2を読む。" in cleaned
    assert "[^1]: 脚注本文" in cleaned
    assert find_urls(cleaned) == []


def test_mermaid_click_and_href_lines_are_removed_and_code_urls_too() -> None:
    text = (
        "```mermaid\nflowchart LR\n    A[\"仮説\"] --> B[\"検証\"]\n    click A \"https://evil.example.com\"\n"
        "    click B href \"https://x.example.com\" _blank\n```\n\n```python\nrequests.get(\"https://api.example.com/v1\")\n```\n"
    )
    cleaned, removed = remove_urls(text)
    assert [r.kind for r in removed].count("mermaid_click") == 2
    assert "click" not in cleaned
    assert 'A["仮説"] --> B["検証"]' in cleaned
    assert 'requests.get("")' in cleaned
    assert find_urls(cleaned) == []


def test_cleanup_removes_empty_artifacts() -> None:
    text = "本文（https://d.example.com）終わり。\n- [x](https://a.example.com)\n- https://b.example.com\n出典：https://c.example.com\n参照:\n値は  1 です  。"
    cleaned, _ = remove_urls(text)
    assert "（）" not in cleaned and "()" not in cleaned
    assert "出典" not in cleaned and "参照" not in cleaned
    assert not re.search(r"^\s*-\s*$", cleaned, re.MULTILINE)
    assert "本文終わり。" in cleaned
    assert "値は 1 です。" in cleaned


def test_sanitize_is_idempotent_and_reports() -> None:
    text = "僕は https://example.com を見た。俺たちは [資料](https://x.example.com) を読み、筆者らは bit.ly/abc を避けた。"
    first, report = sanitize_text(text)
    assert report.first_person_unified and report.url_free
    assert len(report.first_person_replacements) == 3
    assert len(report.urls_removed) == 3
    second, second_report = sanitize_text(first)
    assert second == first
    assert second_report.total_changes == 0
    assert detect_first_person_variants(first) == [] and find_urls(first) == []


def test_find_urls_detects_every_kind_in_one_document() -> None:
    document = "\n".join(source for source, _, _ in _URL_CASES)
    detected = find_urls(document)
    assert len(detected) >= len(_URL_CASES)
    cleaned, removed = remove_urls(document)
    assert find_urls(cleaned) == []
    assert {kind for _, kind, _ in _URL_CASES} <= {r.kind for r in removed}


def test_compliance_sanitizer_verify_and_draft(logs, matrices) -> None:
    sanitizer = ComplianceSanitizer()
    report = sanitizer.verify("僕は https://example.com を見た。")
    assert not report.first_person_unified and not report.url_free
    assert report.residual_first_person == ["僕"]
    plan = matrices["reversal"].plans[0]
    blueprint = build_blueprint_deterministic(plan, logs["reversal"])
    draft = draft_article_deterministic(blueprint, plan, logs["reversal"])
    dirty_sections = list(draft.sections)
    dirty_sections[0] = DraftSection(
        role=dirty_sections[0].role,
        heading="僕の見出し https://h.example.com",
        body_markdown=dirty_sections[0].body_markdown + "\n\n俺は [ここ](https://x.example.com) を見た。",
    )
    dirty = ArticleDraft(
        title="僕のタイトル www.example.com",
        lead="筆者は bit.ly/abc で読んだ。",
        sections=dirty_sections,
        closing="我々は hxxps://evil.example.com を避ける。",
    )
    cleaned, draft_report = sanitizer.sanitize_draft(dirty)
    assert draft_report.first_person_unified and draft_report.url_free
    markdown = cleaned.to_markdown()
    assert find_urls(markdown) == [] and detect_first_person_variants(markdown) == []
    assert cleaned.title.startswith("私のタイトル")
    assert cleaned.sections[0].heading.startswith("私の見出し")
    assert len(draft_report.first_person_replacements) >= 5
    assert len(draft_report.urls_removed) >= 5


# ---------------------------------------------------------------------------
# 研究ログ・3軸マトリクス
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", SAMPLE_KINDS)
def test_research_log_from_structure(kind: str) -> None:
    structure = sample_structure(kind)
    log = research_log_from_structure(structure)
    assert log.kind == "structure_discovery"
    assert 4 <= len(log.findings) <= 6
    assert all(f.metrics for f in log.findings)
    assert log.data["structure_id"] == structure.id
    assert len(log.data["wave"]["tension"]) == structure.wave.n_points
    assert log.tags
    assert research_log_from_structure(structure).findings == log.findings


def test_research_log_from_experiment_variants() -> None:
    experiment = research_log_from_experiment(
        {
            "preference_vector": {
                "features": [
                    {"name": "style.black_fill_ratio", "coefficient": 1.8, "importance": 0.6, "direction": "prefer_higher"},
                    {"name": "style.line_taper", "coefficient": -1.2, "importance": 0.3, "direction": "prefer_lower"},
                ],
                "fit_accuracy": 0.82,
                "n_comparisons": 160,
                "n_candidates": 12,
            },
            "ratings": [
                {"candidate_id": "c1", "label": "候補1", "elo": 1640, "rank": 1, "win_rate": 0.8},
                {"candidate_id": "c2", "label": "候補2", "elo": 1420, "rank": 2, "win_rate": 0.3},
            ],
            "suggestions": [{"params": {"style.black_fill_ratio": 0.9}, "predicted_mean": 1.2, "predicted_std": 0.3, "acquisition": 0.4, "strategy": "hybrid"}],
        }
    )
    assert experiment.kind == "model_experiment"
    assert len(experiment.findings) >= 4
    assert "style.black_fill_ratio" in experiment.findings[0].claim
    dataset = research_log_from_experiment(
        {
            "gap_report": {
                "total_images": 40,
                "balance_score": 62.5,
                "total_additional_images": 18,
                "entanglements": [{"tag_a": "crying", "tag_b": "from above", "npmi": 0.71}],
                "deficits": [{"tag": "backlighting"}],
            }
        }
    )
    assert dataset.kind == "dataset_analysis"
    assert any("crying" in f.claim for f in dataset.findings)
    manga = research_log_from_experiment({"storyboard": {"grammar": {"score": 91.0, "violations": []}, "page_count": 12}})
    assert manga.kind == "manga_production"
    mixed = research_log_from_experiment(
        {"storyboard": {"grammar": {"score": 88.0, "violations": [1]}}, "gap_report": {"balance_score": 70, "total_additional_images": 3}}
    )
    assert mixed.kind == "mixed"
    generic = research_log_from_experiment({"loss": 0.12, "epochs": 30})
    assert generic.findings[0].metrics["epochs"] == 30.0
    with pytest.raises(ValueError):
        research_log_from_experiment({})


@pytest.mark.parametrize("kind", SAMPLE_KINDS)
def test_matrix_deterministic_three_axes(kind: str, logs) -> None:
    matrix = expand_matrix_deterministic(logs[kind])
    assert [plan.axis for plan in matrix.plans] == list(AXIS_ORDER)
    for plan in matrix.plans:
        assert len(plan.outline) == 5
        assert plan.keywords and plan.hook_line and plan.call_to_action
        for text in (plan.title, plan.subtitle, plan.core_insight, plan.hook_line, plan.call_to_action):
            assert find_urls(text) == []
    again = expand_matrix_deterministic(logs[kind])
    assert again.id == matrix.id
    assert [p.model_dump() for p in again.plans] == [p.model_dump() for p in matrix.plans]
    assert len(matrix.cross_links) >= 3


async def test_matrix_llm_path_and_fallback(mock_llm_client, logs) -> None:
    log = logs["persuasion"]
    matrix = await ContentMatrixTranspiler(mock_llm_client).expand(log)
    assert {plan.axis for plan in matrix.plans} == set(ContentAxis)
    assert matrix.id == expand_matrix_deterministic(log).id
    runs = mock_llm_client.telemetry.recent_runs(10)
    assert any(run["task"] == "publisher.matrix" for run in runs)

    from core.llm_client import LLMClient, MockLLMAdapter

    failing = LLMClient(MockLLMAdapter(failure_plan={"publisher.matrix:synthesize": ["garbage"] * 20}), max_repair_attempts=1)
    fallback = await ContentMatrixTranspiler(failing).expand(log)
    assert fallback.model_dump(exclude={"created_at"}) == expand_matrix_deterministic(log).model_dump(exclude={"created_at"})


# ---------------------------------------------------------------------------
# 5段階執筆パイプライン
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", SAMPLE_KINDS)
@pytest.mark.parametrize("axis", [a.value for a in AXIS_ORDER])
def test_writer_reaches_threshold_with_compliance(published, kind: str, axis: str) -> None:
    article = published[(kind, axis)]
    assert article.final_score >= 80.0
    assert article.critique_history[-1].passed
    assert article.sanitization.first_person_unified and article.sanitization.url_free
    assert detect_first_person_variants(article.markdown) == []
    assert find_urls(article.markdown) == []
    assert "私" in article.markdown
    assert article.markdown.count("```mermaid") >= 5
    assert [s.role for s in article.draft.sections] == list(BLUEPRINT_ROLE_ORDER)
    assert all(s.char_count >= 600 for s in article.draft.sections)
    stages = [record.stage for record in article.stages]
    for stage in ("blueprint", "draft", "sanitize", "critique", "finalize"):
        assert stage in stages
    assert article.markdown.startswith(f"# {article.title}")


async def test_writer_progress_and_llm_path(mock_llm_client, logs, matrices) -> None:
    events: list[tuple[str, float, str]] = []

    async def on_progress(stage: str, progress: float, message: str) -> None:
        events.append((stage, progress, message))

    plan = matrices["inverted"].plans[1]
    article = await ArticleWriterPipeline(mock_llm_client).run(plan, logs["inverted"], on_progress=on_progress)
    assert article.final_score >= 80.0
    assert find_urls(article.markdown) == [] and detect_first_person_variants(article.markdown) == []
    stages_seen = [stage for stage, _, _ in events]
    for stage in ("blueprint", "draft", "sanitize", "critique", "finalize", "complete"):
        assert stage in stages_seen
    assert events[-1][0] == "complete" and events[-1][1] == 1.0
    progresses = [progress for _, progress, _ in events]
    assert progresses == sorted(progresses)
    tasks = {run["task"] for run in mock_llm_client.telemetry.recent_runs(50)}
    assert {"publisher.blueprint", "publisher.section_draft", "publisher.critique"} <= tasks


class _WeakFirstSectionPipeline(ArticleWriterPipeline):
    async def draft_section(self, blueprint, role, plan, log=None):
        section = await super().draft_section(blueprint, role, plan, log)
        if role == BlueprintRole.STRUCTURAL_PARADOX:
            weak = "構造には問題がある。前提が強い。これは本質的な問題だ。構造を変える必要がある。"
            return DraftSection(role=role, heading=section.heading, body_markdown=weak)
        return section


async def test_polish_is_monotonic_and_recovers_weak_section(logs, matrices) -> None:
    plan = matrices["reversal"].plans[0]
    article = await _WeakFirstSectionPipeline().run(plan, logs["reversal"])
    assert article.iterations >= 1
    overall = [critique.overall for critique in article.critique_history]
    assert overall == sorted(overall)
    assert article.critique_history[0].section(BlueprintRole.STRUCTURAL_PARADOX).total < 80.0
    assert article.final_score >= 80.0
    assert any(record.stage == "polish" for record in article.stages)
    assert find_urls(article.markdown) == [] and detect_first_person_variants(article.markdown) == []


async def test_polish_round_never_lowers_section_scores(logs, matrices) -> None:
    plan = matrices["persuasion"].plans[2]
    pipeline = ArticleWriterPipeline(config=WriterConfig(quality_threshold=99.5, max_polish_rounds=2))
    blueprint = build_blueprint_deterministic(plan, logs["persuasion"])
    draft = draft_article_deterministic(blueprint, plan, logs["persuasion"])
    before = await pipeline.critique(draft, blueprint)
    polished = await pipeline.polish(draft, before, blueprint, plan, logs["persuasion"], round_index=1)
    after = await pipeline.critique(polished, blueprint)
    for old, new in zip(before.sections, after.sections):
        assert new.total + 1e-9 >= old.total
    assert after.overall + 1e-9 >= before.overall


def test_critique_is_sensitive_to_evidence(logs, matrices) -> None:
    plan = matrices["reversal"].plans[1]
    blueprint = build_blueprint_deterministic(plan, logs["reversal"])
    section = compose_section(blueprint, BlueprintRole.HYPOTHESIS_EVIDENCE, plan, logs["reversal"])
    full = score_section(section, blueprint.section(section.role))
    stripped_body = re.sub(r"\d+(?:\.\d+)?", "", section.body_markdown)
    for term in ("データ", "指標", "検証", "比較", "分析", "数値", "実験"):
        stripped_body = stripped_body.replace(term, "")
    stripped_body = re.sub(r"```mermaid.*?```", "", stripped_body, flags=re.DOTALL)
    weaker = score_section(
        DraftSection(role=section.role, heading=section.heading, body_markdown=stripped_body), blueprint.section(section.role)
    )
    assert weaker.evidence_density < full.evidence_density
    assert weaker.total < full.total
    assert any(item.startswith("【根拠】") for item in weaker.feedback)


def test_critique_scores_are_bounded_and_consistent(logs, matrices) -> None:
    plan = matrices["inverted"].plans[0]
    blueprint = build_blueprint_deterministic(plan, logs["inverted"])
    draft = draft_article_deterministic(blueprint, plan, logs["inverted"])
    critique = critique_deterministic(draft, blueprint, 80.0)
    for section in critique.sections:
        for value in (section.logic_coherence, section.abstraction_balance, section.persuasiveness, section.evidence_density):
            assert 0.0 <= value <= 25.0
        assert math.isclose(
            section.total,
            section.logic_coherence + section.abstraction_balance + section.persuasiveness + section.evidence_density,
            abs_tol=0.02,
        )
    assert math.isclose(critique.overall, sum(s.total for s in critique.sections) / 5.0, abs_tol=0.02)


def test_pipeline_sanitize_stage_reports_changes(logs, matrices) -> None:
    plan = matrices["reversal"].plans[2]
    blueprint = build_blueprint_deterministic(plan, logs["reversal"])
    draft = draft_article_deterministic(blueprint, plan, logs["reversal"])
    sections = list(draft.sections)
    sections[4] = DraftSection(
        role=sections[4].role,
        heading=sections[4].heading,
        body_markdown=sections[4].body_markdown + "\n\n僕のおすすめは https://aff.example.com/?id=1 と amzn.to/abc だ。",
    )
    dirty = draft.model_copy(update={"sections": sections})
    cleaned, report = ArticleWriterPipeline().sanitize(dirty)
    assert report.url_free and report.first_person_unified
    assert len(report.urls_removed) == 2
    assert [r.original for r in report.first_person_replacements] == ["僕"]
    assert "私のおすすめは" in cleaned.sections[4].body_markdown


# ---------------------------------------------------------------------------
# ナレッジグラフ
# ---------------------------------------------------------------------------


def test_knowledge_graph_pagerank_communities_and_gaps(published) -> None:
    graph = KnowledgeGraph()
    for structure in sample_structures():
        graph.add_structure(structure)
        graph.add_research_log(research_log_from_structure(structure))
    rank = graph.pagerank()
    assert math.isclose(sum(rank.values()), 1.0, abs_tol=1e-9)
    assert all(value > 0 for value in rank.values())
    communities = graph.communities()
    concept_labels = {row["label"] for row in graph.concepts()}
    assert set(communities) == concept_labels
    gaps_before = graph.detect_gaps(top_k=5)
    assert 1 <= len(gaps_before) <= 5
    scores = [gap.gap_score for gap in gaps_before]
    assert scores == sorted(scores, reverse=True)
    target = next(gap for gap in gaps_before if len(gap.concepts) == 1)
    assert target.coverage == 0.0
    article = published[("reversal", ContentAxis.CREATOR_THEORY.value)]
    covering = article.model_copy(update={"id": "article_cover", "tags": [*article.tags, target.concepts[0]]})
    graph.add_article(covering)
    after = {tuple(gap.concepts): gap.gap_score for gap in graph.detect_gaps(top_k=50)}
    assert after.get(tuple(target.concepts), 0.0) < target.gap_score
    mermaid = graph.to_mermaid()
    assert mermaid.startswith("flowchart LR")
    assert find_urls(mermaid) == []
    stats = graph.stats()
    assert stats["structures"] == 3 and stats["logs"] == 3 and stats["articles"] == 1
    assert stats["communities"] >= 1


def test_knowledge_graph_bridge_gap_and_db_build(tmp_db, published) -> None:
    repo = ArticleRepository(tmp_db)
    for key in (("reversal", "creator_theory"), ("persuasion", "business_marketing")):
        repo.save(published[key])
    graph = KnowledgeGraph(tmp_db).build_from_db()
    assert graph.stats()["articles"] == 2
    structure_graph = KnowledgeGraph()
    for structure in sample_structures():
        structure_graph.add_structure(structure)
    gaps = structure_graph.detect_gaps(top_k=30)
    assert gaps
    assert all(gap.suggested_title and len(gap.suggested_outline) == 5 for gap in gaps)
    assert all(find_urls(gap.suggested_title) == [] for gap in gaps)


# ---------------------------------------------------------------------------
# エクスポート・編纂
# ---------------------------------------------------------------------------


def test_mermaid_to_text_variants() -> None:
    flow = 'flowchart LR\n    A["常識"] --> B["努力"]\n    B -->|はい| C{"成果"}\n    C -.-> D["自覚"]'
    bullets = mermaid_to_text(flow)
    assert "・常識 → 努力" in bullets
    assert "・努力 → 成果（はい）" in bullets
    assert "・成果 ⇢ 自覚" in bullets
    xy = 'xychart-beta\n    title "推移"\n    x-axis "t" 0 --> 1\n    y-axis "v" 0 --> 1\n    line [0.1, 0.9, 0.2]'
    xy_bullets = mermaid_to_text(xy)
    assert xy_bullets[0] == "・推移"
    assert "最大 0.90" in xy_bullets[1]
    bar = 'xychart-beta\n    x-axis ["A", "B"]\n    y-axis "値" 0 --> 1\n    bar [0.3, 0.7]'
    assert mermaid_to_text(bar) == ["・A: 0.3", "・B: 0.7"]
    quadrant = "quadrantChart\n    title 図\n    Point A: [0.3, 0.6]"
    assert "・Point A（x=0.3, y=0.6）" in mermaid_to_text(quadrant)


def test_note_export(published) -> None:
    article = published[("inverted", ContentAxis.AI_ENGINEERING.value)]
    note = export_note_markdown(article)
    assert "```mermaid" not in note
    assert note.count("【図解】") >= 5
    assert "・" in note
    headings = [line for line in note.splitlines() if line.startswith("#")]
    assert headings
    assert all(re.match(r"^#{2,3} ", line) for line in headings)
    assert find_urls(note) == [] and detect_first_person_variants(note) == []


def test_obsidian_export(published) -> None:
    article = published[("persuasion", ContentAxis.BUSINESS_MARKETING.value)]
    exported = export_obsidian_markdown(article, related_titles=["関連する記事A", "関連する記事B"])
    assert exported.startswith("---\ntitle: ")
    header = exported.split("---")[1]
    for key in ("title:", "aliases:", "tags:", "created:", "axis:", "score:", "source:"):
        assert key in header
    assert "```mermaid" in exported
    assert "[[関連する記事A]]" in exported
    assert re.search(r"\[\[[^\]]+\]\]", exported.split("## 関連記事")[0])
    assert find_urls(exported) == [] and detect_first_person_variants(exported) == []


def _article_with(article: PublishedArticle, *, article_id: str, created: str, tags: list[str], extra: str) -> PublishedArticle:
    return article.model_copy(
        update={
            "id": article_id,
            "title": f"{article.title}（{article_id}）",
            "created_at": created,
            "tags": tags,
            "markdown": article.markdown + f"\n\n{extra}\n",
        }
    )


def test_series_compiler_orders_by_concept_dependency(published) -> None:
    base = published[("reversal", ContentAxis.CREATOR_THEORY.value)]
    dependent = _article_with(base, article_id="art_dependent", created="2026-01-01T00:00:00+00:00", tags=["応用編"], extra="本稿は基礎概念Xを前提とする。")
    foundation = _article_with(base, article_id="art_foundation", created="2026-02-01T00:00:00+00:00", tags=["基礎概念X"], extra="基礎概念Xを導入する。")
    tail = _article_with(base, article_id="art_tail", created="2026-03-01T00:00:00+00:00", tags=["発展編"], extra="応用編の議論を踏まえる。")
    compiler = SeriesCompiler()
    ordered = [a.id for a in compiler.order([dependent, foundation, tail])]
    assert ordered.index("art_foundation") < ordered.index("art_dependent") < ordered.index("art_tail")
    volume, markdown = compiler.compile([tail, dependent, foundation], title="構造論シリーズ")
    assert sorted(volume.ordered_article_ids()) == sorted(["art_dependent", "art_foundation", "art_tail"])
    assert len(volume.ordered_article_ids()) == 3
    assert volume.glossary and "基礎概念X" in volume.glossary
    assert "## 目次" in markdown and "## 用語集" in markdown
    assert find_urls(markdown) == [] and detect_first_person_variants(markdown) == []
    note_version = export_series_markdown(volume, [tail, dependent, foundation], "note")
    assert "```mermaid" not in note_version
    assert all(re.match(r"^#{2,3} ", line) for line in note_version.splitlines() if line.startswith("#"))
    with pytest.raises(ValueError):
        compiler.compile([], title="空")


# ---------------------------------------------------------------------------
# リポジトリ・モックハンドラ
# ---------------------------------------------------------------------------


def test_article_repository_crud(tmp_db, published, matrices) -> None:
    repo = ArticleRepository(tmp_db)
    first = published[("reversal", "creator_theory")]
    second = published[("inverted", "ai_engineering")]
    repo.save(first)
    repo.save(second)
    assert repo.count() == 2
    assert repo.count(axis="ai_engineering") == 1
    loaded = repo.get(first.id)
    assert loaded is not None and loaded.markdown == first.markdown
    summaries = repo.list(axis="creator_theory")
    assert [s.id for s in summaries] == [first.id]
    assert summaries[0].char_count > 0
    repo.save(first.model_copy(update={"final_score": 99.0}))
    assert repo.get(first.id).final_score == 99.0
    assert repo.delete(first.id) and not repo.delete(first.id)
    assert repo.get(first.id) is None
    matrix = matrices["persuasion"]
    repo.save_matrix(matrix)
    assert repo.get_matrix(matrix.id).plans[0].title == matrix.plans[0].title
    assert repo.list_matrices()[0]["id"] == matrix.id
    assert repo.delete_matrix(matrix.id)
    assert repo.get_matrix(matrix.id) is None


def test_mock_handlers_are_registered() -> None:
    import core.asset_publisher.mock_handlers as handlers
    from core.llm_client import get_mock_handler

    expected = {
        "publisher.matrix": handlers.mock_publisher_matrix,
        "publisher.blueprint": handlers.mock_publisher_blueprint,
        "publisher.section_draft": handlers.mock_publisher_section_draft,
        "publisher.critique": handlers.mock_publisher_critique,
        "publisher.polish": handlers.mock_publisher_polish,
    }
    for task, handler in expected.items():
        assert get_mock_handler(task, "synthesize") is handler
        assert get_mock_handler(task, "refine") is handler


# ---------------------------------------------------------------------------
# 回帰・互換性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[[劇的アイロニー]]（読者だけが知る）を使う。", "[[劇的アイロニー]]（読者だけが知る）を使う。"),
        ("注[1]（補足）は残す。", "注[1]（補足）は残す。"),
        ("［資料］（ｈｔｔｐｓ：／／ｘ．ｃｏｍ／ａ）を見る。", "［資料］を見る。"),
    ],
)
def test_full_width_brackets_are_not_markdown_syntax(source: str, expected: str) -> None:
    cleaned, _ = sanitize_text(source)
    assert cleaned == expected
    assert find_urls(cleaned) == []


def _pep701_violations(source: str) -> list[str]:
    """Python 3.12 未満で構文エラーになる f-string 式（バックスラッシュ・外側と同じ引用符）を検出する。"""
    import io
    import tokenize

    if not hasattr(tokenize, "FSTRING_START"):
        return []
    problems: list[str] = []
    stack: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.FSTRING_START:
            quote = token.string.lstrip("rRfFbBuU")
            if stack and len(stack[-1]) == 1 and quote[:1] == stack[-1]:
                problems.append(f"{token.start[0]}行: 同種引用符の入れ子 f-string")
            stack.append(quote)
        elif token.type == tokenize.FSTRING_END:
            stack.pop()
        elif stack and token.type == tokenize.STRING:
            body = token.string.lstrip("rRbBuU")
            if "\\" in token.string:
                problems.append(f"{token.start[0]}行: f-string 式内のバックスラッシュ")
            if len(stack[-1]) == 1 and body[:1] == stack[-1]:
                problems.append(f"{token.start[0]}行: f-string 式内で外側と同じ引用符")
    return problems


def test_publisher_sources_are_python310_compatible() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "core" / "asset_publisher").glob("*.py")) + [Path(__file__).resolve()]
    problems: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path), feature_version=(3, 10))
        problems.extend(f"{path.name}:{issue}" for issue in _pep701_violations(source))
    assert problems == []
