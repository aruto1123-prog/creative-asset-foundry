"""体系的シリーズ編纂器と各種 Markdown エクスポーター。

- ``export_note_markdown``: note 向け（見出しは ##/### のみ、Mermaid は「【図解】」＋ノード/エッジのテキストへ変換）
- ``export_obsidian_markdown``: Obsidian 向け（YAML frontmatter、概念語の [[wikilink]] 化、関連記事、Mermaid 保持）
- ``SeriesCompiler``: 概念の初出依存による位相順序とコミュニティによる章立てで、複数記事を1冊の教本へ編纂
- ``export_series_markdown``: 編纂結果を note / Obsidian 形式で出力

すべての出力は最後に Compliance Sanitizer を通し、URL ゼロ・一人称「私」を保証する。
"""

from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.asset_publisher.sanitizer import sanitize_text
from core.asset_publisher.schemas import AXIS_LABELS_JA, AXIS_ORDER, PublishedArticle, split_markdown_segments
from core.common import new_id, truncate, utc_now_iso
from core.structure_engine.schemas import MECHANISM_LABELS_JA

__all__ = [
    "SeriesChapter",
    "SeriesVolume",
    "SeriesCompiler",
    "mermaid_to_text",
    "export_note_markdown",
    "export_obsidian_markdown",
    "export_series_markdown",
]

_GLOSSARY_DEFAULTS: dict[str, str] = {
    MECHANISM_LABELS_JA["curiosity_gap"]: "知っていることと知りたいことの差が生む、読み進める動機。",
    MECHANISM_LABELS_JA["dramatic_irony"]: "受け手だけが真相を知ることで、登場人物の言動が二重の意味を帯びる効果。",
    MECHANISM_LABELS_JA["empathy_transfer"]: "受け手が自分の経験を主体の欠落に重ね、選択を自分事として体験する過程。",
    MECHANISM_LABELS_JA["prediction_error"]: "予測と結果のずれが注意と記憶を強める現象。構造的反転の中核。",
    MECHANISM_LABELS_JA["cognitive_dissonance"]: "信念と事実の矛盾が、解消を求める強い動機を生む状態。",
    MECHANISM_LABELS_JA["peak_end"]: "体験全体の評価が、ピークと終わりの印象でほぼ決まるという法則。",
    MECHANISM_LABELS_JA["social_proof"]: "他者の選択や評価が、自分の判断の根拠として働く傾向。",
    MECHANISM_LABELS_JA["loss_aversion"]: "同じ大きさの利得より損失を強く感じる傾向。",
    "構造的反転": "物語や論説の前提そのものが覆り、それまでの情報が再解釈される転換点。",
    "感情曲線": "正規化時間に沿って緊張と解放の推移を描いた波形。",
    "美的選好": "言語化しにくい好みを、一対比較の勝敗から数値として推定したもの。",
}


class SeriesChapter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = Field(..., ge=1)
    title: str
    article_ids: list[str] = Field(default_factory=list)
    summary: str = ""


class SeriesVolume(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: new_id("series"))
    title: str
    description: str = ""
    chapters: list[SeriesChapter] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

    def ordered_article_ids(self) -> list[str]:
        return [article_id for chapter in self.chapters for article_id in chapter.article_ids]


# ---------------------------------------------------------------------------
# Mermaid → テキスト
# ---------------------------------------------------------------------------

_NODE_DEF_RE = re.compile(
    r"(?P<id>[A-Za-z_][\w-]*)\s*(?:\[\[|\[\(|\(\[|\(\(|\{\{|\[|\(|\{|>)\s*(?P<label>\"[^\"]*\"|[^\]\)\}\"]+?)\s*(?:\]\]|\)\]|\]\)|\)\)|\}\}|\]|\)|\})"
)
_EDGE_RE = re.compile(
    r"(?P<a>[A-Za-z_][\w-]*)(?:\s*(?:\[[^\]]*\]|\([^)]*\)|\{[^}]*\}))*\s*(?P<arrow>-\.->|-->|==>|---|-\.-|--)\s*(?:\|(?P<label>[^|]*)\|)?\s*(?P<b>[A-Za-z_][\w-]*)"
)
_XY_TITLE_RE = re.compile(r'^\s*title\s+"?(?P<title>[^"]*)"?\s*$')
_XY_SERIES_RE = re.compile(r"^\s*(?P<kind>line|bar)\s*\[(?P<values>[^\]]*)\]")
_XY_AXIS_CATS_RE = re.compile(r"^\s*x-axis\s*\[(?P<cats>[^\]]*)\]")
_QUADRANT_POINT_RE = re.compile(r"^\s*(?P<name>[^:\n]+):\s*\[(?P<x>[0-9.]+)\s*,\s*(?P<y>[0-9.]+)\]")
_MERMAID_SYNTAX_RE = re.compile(r"[\[\](){}\"|]")


def _mermaid_code(chunk: str) -> str:
    lines = chunk.splitlines()
    if lines and lines[0].lstrip().startswith(("```", "~~~")):
        lines = lines[1:]
    if lines and lines[-1].strip() and set(lines[-1].strip()) <= {"`", "~"}:
        lines = lines[:-1]
    return "\n".join(lines)


def mermaid_to_text(code: str) -> list[str]:
    """Mermaid 図解を note 等で読める箇条書きテキストへ変換する。"""
    lines = [line for line in code.strip().splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].strip().split()[0].lower()
    body = lines[1:]
    bullets: list[str] = []
    if header in ("flowchart", "graph"):
        labels: dict[str, str] = {}
        for line in body:
            for match in _NODE_DEF_RE.finditer(line):
                labels.setdefault(match.group("id"), match.group("label").strip().strip('"').strip())
        for line in body:
            position = 0
            while True:
                match = _EDGE_RE.search(line, position)
                if match is None:
                    break
                a, b = match.group("a"), match.group("b")
                edge_label = (match.group("label") or "").strip()
                arrow = "⇢" if "." in match.group("arrow") else "→"
                text = f"・{labels.get(a, a)} {arrow} {labels.get(b, b)}"
                if edge_label:
                    text += f"（{edge_label}）"
                bullets.append(text)
                position = match.start("b")
        if not bullets:
            bullets = [f"・{label}" for label in labels.values()]
        return bullets
    if header == "xychart-beta":
        title = ""
        categories: list[str] = []
        for line in body:
            title_match = _XY_TITLE_RE.match(line)
            if title_match:
                title = title_match.group("title").strip()
            cats_match = _XY_AXIS_CATS_RE.match(line)
            if cats_match:
                categories = [c.strip().strip('"') for c in cats_match.group("cats").split(",") if c.strip()]
        if title:
            bullets.append(f"・{title}")
        series_index = 0
        for line in body:
            series_match = _XY_SERIES_RE.match(line)
            if not series_match:
                continue
            values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", series_match.group("values"))]
            if not values:
                continue
            series_index += 1
            if series_match.group("kind") == "bar" and categories and len(categories) == len(values):
                bullets.extend(f"・{cat}: {value:g}" for cat, value in zip(categories, values))
            else:
                peak = max(values)
                bullets.append(
                    f"・系列{series_index}: 開始 {values[0]:.2f} → 最大 {peak:.2f}（{values.index(peak) + 1}点目） → 終了 {values[-1]:.2f}（全{len(values)}点）"
                )
        return bullets
    if header == "quadrantchart":
        for line in body:
            point = _QUADRANT_POINT_RE.match(line)
            if point:
                bullets.append(f"・{point.group('name').strip()}（x={point.group('x')}, y={point.group('y')}）")
        return bullets or ["・" + line.strip() for line in body]
    return ["・" + _MERMAID_SYNTAX_RE.sub(" ", line).strip() for line in body]


# ---------------------------------------------------------------------------
# note / Obsidian
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<text>.+?)[ \t]*#*[ \t]*$")


def _shift_headings(prose: str, *, note_mode: bool, shift: int = 1) -> str:
    out: list[str] = []
    for line in prose.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        newline = "\n" if line.endswith("\n") else ""
        match = _HEADING_RE.match(stripped)
        if not match:
            out.append(line)
            continue
        level = len(match.group("hashes")) + shift
        text = match.group("text").strip()
        if note_mode and level > 3:
            out.append(f"**{text}**{newline}")
        else:
            out.append(f"{'#' * min(6, level)} {text}{newline}")
    return "".join(out)


def _tables_to_bullets(prose: str) -> str:
    out: list[str] = []
    for line in prose.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            out.append("・" + " / ".join(c for c in cells if c) + ("\n" if line.endswith("\n") else ""))
        else:
            out.append(line)
    return "".join(out)


def _note_body(markdown: str, heading_shift: int = 1) -> str:
    pieces: list[str] = []
    for kind, chunk in split_markdown_segments(markdown):
        if kind == "mermaid":
            bullets = mermaid_to_text(_mermaid_code(chunk))
            pieces.append("【図解】\n" + "\n".join(bullets) + "\n")
        elif kind == "code":
            pieces.append(chunk)
        else:
            pieces.append(_tables_to_bullets(_shift_headings(chunk, note_mode=True, shift=heading_shift)))
    return "".join(pieces)


def export_note_markdown(article: PublishedArticle) -> str:
    """note 向け Markdown（見出しは ##/### のみ、Mermaid はテキスト化）を返す。"""
    converted = _note_body(article.markdown, heading_shift=1)
    cleaned, _ = sanitize_text(converted)
    return cleaned.strip() + "\n"


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value}"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{text}"'


def _obsidian_tag(tag: str) -> str:
    cleaned = re.sub(r"[\s#,.:;!?/\\\[\]{}()「」『』、。・]+", "_", tag.strip()).strip("_")
    return cleaned or "caf"


def _wikilink_concepts(prose: str, concepts: Iterable[str], linked: set[str]) -> str:
    result = prose
    for concept in sorted({c for c in concepts if c and len(c) >= 2}, key=len, reverse=True):
        if concept in linked:
            continue
        pattern = re.compile(r"(?<!\[\[)" + re.escape(concept) + r"(?!\]\])")
        lines = result.splitlines(keepends=True)
        done = False
        for index, line in enumerate(lines):
            if done or line.lstrip().startswith("#") or "[[" in line and concept in line:
                continue
            match = pattern.search(line)
            if match:
                lines[index] = line[: match.start()] + f"[[{concept}]]" + line[match.end() :]
                done = True
        if done:
            linked.add(concept)
            result = "".join(lines)
    return result


def export_obsidian_markdown(article: PublishedArticle, related_titles: Sequence[str] = ()) -> str:
    """Obsidian 向け Markdown（YAML frontmatter・wikilink・関連記事・Mermaid 保持）を返す。"""
    alias = truncate(article.title.split("――")[0], 40)
    tags = [_obsidian_tag(tag) for tag in article.tags] or ["caf"]
    frontmatter_lines = [
        "---",
        f"title: {_yaml_scalar(article.title)}",
        f"aliases: [{_yaml_scalar(alias)}]",
        "tags: [" + ", ".join(_yaml_scalar(tag) for tag in dict.fromkeys(tags)) + "]",
        f"created: {_yaml_scalar(article.created_at)}",
        f"axis: {_yaml_scalar(article.axis.value)}",
        f"axis_label: {_yaml_scalar(AXIS_LABELS_JA[article.axis.value])}",
        f"score: {round(article.final_score, 2)}",
        f"iterations: {article.iterations}",
        f"source: {_yaml_scalar(article.source_id)}",
        "---",
        "",
    ]
    linked: set[str] = set()
    pieces: list[str] = []
    for kind, chunk in split_markdown_segments(article.markdown):
        if kind == "prose":
            pieces.append(_wikilink_concepts(chunk, article.tags, linked))
        else:
            pieces.append(chunk)
    body = "".join(pieces).rstrip() + "\n"
    if related_titles:
        body += "\n## 関連記事\n\n" + "\n".join(f"- [[{title}]]" for title in related_titles) + "\n"
    cleaned, _ = sanitize_text("\n".join(frontmatter_lines) + body)
    return cleaned


# ---------------------------------------------------------------------------
# シリーズ編纂
# ---------------------------------------------------------------------------


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SeriesCompiler:
    """複数記事を、概念の初出依存（位相順）とコミュニティ（章立て）で1冊へ編纂する。"""

    def _concepts(self, articles: list[PublishedArticle]) -> dict[str, set[str]]:
        vocabulary = {tag for article in articles for tag in article.tags if len(tag) >= 2}
        return {
            article.id: set(article.tags) | {term for term in vocabulary if term in article.markdown}
            for article in articles
        }

    def order(self, articles: list[PublishedArticle]) -> list[PublishedArticle]:
        """概念を最初に導入する記事が、それを使う記事より前に来る位相順。"""
        by_id = {a.id: a for a in articles}
        concepts = self._concepts(articles)
        axis_rank = {axis.value: index for index, axis in enumerate(AXIS_ORDER)}

        def key(article: PublishedArticle) -> tuple[int, str, str, str]:
            return (axis_rank.get(article.axis.value, 9), article.created_at, article.title, article.id)

        introducer: dict[str, str] = {}
        for article in sorted(articles, key=lambda a: (a.created_at, a.id)):
            for tag in article.tags:
                introducer.setdefault(tag, article.id)
        successors: dict[str, set[str]] = defaultdict(set)
        indegree = {a.id: 0 for a in articles}
        for article in articles:
            for concept in concepts[article.id]:
                source = introducer.get(concept)
                if source and source != article.id and article.id not in successors[source]:
                    successors[source].add(article.id)
                    indegree[article.id] += 1
        heap = [(key(by_id[a]), a) for a, d in indegree.items() if d == 0]
        heapq.heapify(heap)
        ordered: list[str] = []
        while heap:
            _, current = heapq.heappop(heap)
            ordered.append(current)
            for nxt in sorted(successors[current]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(heap, (key(by_id[nxt]), nxt))
        remaining = sorted((by_id[a] for a in indegree if a not in ordered), key=key)
        ordered.extend(a.id for a in remaining)
        return [by_id[a] for a in ordered]

    def _communities(self, articles: list[PublishedArticle], concepts: dict[str, set[str]]) -> dict[str, str]:
        ids = sorted(a.id for a in articles)
        labels = {a: a for a in ids}
        weights = {
            (a, b): _jaccard(concepts[a], concepts[b]) for i, a in enumerate(ids) for b in ids[i + 1 :]
        }
        for _ in range(30):
            changed = False
            for node in ids:
                votes: dict[str, float] = defaultdict(float)
                for other in ids:
                    if other == node:
                        continue
                    weight = weights.get((node, other)) or weights.get((other, node)) or 0.0
                    if weight >= 0.1:
                        votes[labels[other]] += weight
                if not votes:
                    continue
                top = max(votes.values())
                candidates = sorted(label for label, value in votes.items() if abs(value - top) <= 1e-12)
                choice = labels[node] if labels[node] in candidates else candidates[0]
                if choice != labels[node]:
                    labels[node] = choice
                    changed = True
            if not changed:
                break
        return labels

    def compile(
        self, articles: Sequence[PublishedArticle], title: str, description: str | None = None
    ) -> tuple[SeriesVolume, str]:
        items = list(articles)
        if not items:
            raise ValueError("編纂する記事がありません")
        ordered = self.order(items)
        position = {a.id: i for i, a in enumerate(ordered)}
        concepts = self._concepts(items)
        communities = self._communities(items, concepts)
        groups: dict[str, list[PublishedArticle]] = defaultdict(list)
        for article in ordered:
            groups[communities[article.id]].append(article)
        chapter_groups = sorted(groups.values(), key=lambda members: min(position[m.id] for m in members))
        chapters: list[SeriesChapter] = []
        for index, members in enumerate(chapter_groups, start=1):
            members.sort(key=lambda m: position[m.id])
            counter = Counter(tag for member in members for tag in member.tags)
            dominant = counter.most_common(1)[0][0] if counter else members[0].title
            axes = {m.axis.value for m in members}
            axis_label = AXIS_LABELS_JA[next(iter(axes))] if len(axes) == 1 else "領域横断"
            chapters.append(
                SeriesChapter(
                    index=index,
                    title=f"第{index}章 {dominant}と{axis_label}",
                    article_ids=[m.id for m in members],
                    summary=f"{len(members)}本の記事で「{dominant}」を中心に{axis_label}を扱う。",
                )
            )
        introducer: dict[str, PublishedArticle] = {}
        for article in ordered:
            for tag in article.tags:
                introducer.setdefault(tag, article)
        glossary = {
            concept: _GLOSSARY_DEFAULTS.get(concept, f"本シリーズの中核概念。初出は「{truncate(source.title, 40)}」。")
            for concept, source in sorted(introducer.items())
        }
        volume = SeriesVolume(
            title=title,
            description=description
            or f"私がこれまでに書いた{len(items)}本の記事を、概念の依存関係に沿って体系的に編み直した教本である。",
            chapters=chapters,
            glossary=glossary,
        )
        return volume, export_series_markdown(volume, items, "obsidian")


def export_series_markdown(
    volume: SeriesVolume, articles: Sequence[PublishedArticle], fmt: Literal["note", "obsidian"] = "obsidian"
) -> str:
    """編纂結果を note / Obsidian 形式の1本の Markdown にする。"""
    by_id = {a.id: a for a in articles}
    lines: list[str] = []
    if fmt == "obsidian":
        lines.extend(
            [
                "---",
                f"title: {_yaml_scalar(volume.title)}",
                'tags: ["caf_series"]',
                f"created: {_yaml_scalar(volume.created_at)}",
                f"chapters: {len(volume.chapters)}",
                "---",
                "",
                f"# {volume.title}",
            ]
        )
    else:
        lines.append(f"## {volume.title}")
    lines.extend(["", volume.description, ""])
    lines.append("## 目次" if fmt == "obsidian" else "### 目次")
    lines.append("")
    for chapter in volume.chapters:
        lines.append(f"- {chapter.title}")
        for article_id in chapter.article_ids:
            article = by_id.get(article_id)
            if article is not None:
                link = f"[[{article.title}]]" if fmt == "obsidian" else article.title
                lines.append(f"  - {link}")
    lines.extend(["", "## 序文" if fmt == "obsidian" else "### 序文", ""])
    lines.append(
        "本書では、各記事を「どの概念を前提とし、どの概念を新たに導入するか」という依存関係で並べ替えた。"
        "前の章で導入した概念を次の章が使うため、先頭から順に読めば、私が検証で得た構造の知見を無理なく積み上げられる。"
    )
    for chapter in volume.chapters:
        lines.extend(["", f"# {chapter.title}" if fmt == "obsidian" else f"## {chapter.title}", "", chapter.summary, ""])
        for article_id in chapter.article_ids:
            article = by_id.get(article_id)
            if article is None:
                continue
            if fmt == "obsidian":
                body = "".join(
                    _shift_headings(chunk, note_mode=False, shift=1) if kind == "prose" else chunk
                    for kind, chunk in split_markdown_segments(article.markdown)
                )
            else:
                body = _note_body(article.markdown, heading_shift=2)
            lines.extend([body.strip(), ""])
    lines.extend(["## 用語集" if fmt == "obsidian" else "### 用語集", ""])
    for concept, definition in volume.glossary.items():
        term = f"[[{concept}]]" if fmt == "obsidian" else concept
        lines.append(f"- **{term}**: {definition}")
    cleaned, _ = sanitize_text("\n".join(lines).rstrip() + "\n")
    return cleaned
