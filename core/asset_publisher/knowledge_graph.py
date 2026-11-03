"""体系的ナレッジグラフ: 記事・構造・研究ログと概念の関係から、未開拓のテーマ空白を検出する。

- ノード: 概念（タグ・キーワード・メカニズム種別・転換点ビート種別・力学の中核関係）、記事、構造、研究ログ
- エッジ: 文書–概念（重み=出現強度）、概念–概念（同一文書内の共起）
- PageRank（べき乗法、減衰率 d=0.85、ダングリングノードは一様再分配）
- ラベル伝播法による概念コミュニティ（決定論的な走査順と同点処理）
- コンテンツ空白: 需要（構造・ログ側での PageRank と頻度）× (1 − 記事被覆率) × 新規性、
  および記事が橋渡ししていないコミュニティ間の概念ペア（構造的空隙）
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.asset_publisher.schemas import (
    AXIS_LABELS_JA,
    BLUEPRINT_ROLE_LABELS_JA,
    BLUEPRINT_ROLE_ORDER,
    ContentAxis,
    PublishedArticle,
    ResearchLog,
)
from core.structure_engine.schemas import (
    BEAT_TYPE_LABELS_JA,
    EDGE_KIND_LABELS_JA,
    MECHANISM_LABELS_JA,
    UniversalStructure,
)

__all__ = ["ContentGap", "KnowledgeGraph"]

_CREATOR_HINTS = ("漫画", "コマ", "めくり", "見開き", "ネーム", "物語", "感情曲線", "ビート", "伏線", "演出", "キャラクター", "読者", "転換", "反転", "クライマックス", "余韻", "フック")
_AI_HINTS = ("LoRA", "マージ", "データセット", "選好", "最適化", "モデル", "キャプション", "特徴量", "Elo", "Bradley", "ベイズ", "学習", "ブロック", "画像生成", "絵柄")
_BUSINESS_HINTS = ("LP", "ファネル", "マーケティング", "成約", "顧客", "社会的証明", "希少性", "返報性", "権威", "コミットメント", "損失回避", "行動変容", "企画", "セールス", "論説", "説得")


class ContentGap(BaseModel):
    """次に書くべき高価値テーマ（コンテンツ空白）。"""

    model_config = ConfigDict(extra="ignore")

    concepts: list[str] = Field(..., min_length=1)
    axis: ContentAxis | None = None
    gap_score: float = Field(..., ge=0.0)
    demand: float = Field(..., ge=0.0)
    coverage: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    suggested_title: str
    suggested_outline: list[str] = Field(default_factory=list)


def _normalize(label: str) -> str:
    return re.sub(r"\s+", " ", str(label)).strip()


def _infer_axis(concept: str) -> ContentAxis:
    scores = {
        ContentAxis.CREATOR_THEORY: sum(1 for h in _CREATOR_HINTS if h.lower() in concept.lower()),
        ContentAxis.AI_ENGINEERING: sum(1 for h in _AI_HINTS if h.lower() in concept.lower()),
        ContentAxis.BUSINESS_MARKETING: sum(1 for h in _BUSINESS_HINTS if h.lower() in concept.lower()),
    }
    best = max(scores.items(), key=lambda item: (item[1], -list(scores).index(item[0])))
    return best[0] if best[1] > 0 else ContentAxis.CREATOR_THEORY


class KnowledgeGraph:
    """記事・構造・研究ログと概念の重み付き無向グラフ。"""

    def __init__(self, db: Any | None = None) -> None:
        self.db = db
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str], float] = defaultdict(float)
        self._doc_concepts: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ build
    def _add_node(self, node_id: str, node_type: str, label: str, **attrs: Any) -> None:
        node = self._nodes.setdefault(node_id, {"type": node_type, "label": label})
        node.update(attrs)

    def _add_edge(self, a: str, b: str, weight: float) -> None:
        if a == b or weight <= 0:
            return
        key = (a, b) if a < b else (b, a)
        self._edges[key] += float(weight)

    def _concept_id(self, label: str) -> str:
        normalized = _normalize(label)
        node_id = f"c:{normalized}"
        self._add_node(node_id, "concept", normalized)
        return node_id

    def _link_document(self, doc_id: str, concepts: dict[str, float]) -> None:
        concept_ids: list[str] = []
        for label, weight in concepts.items():
            if not _normalize(label):
                continue
            concept_id = self._concept_id(label)
            concept_ids.append(concept_id)
            self._add_edge(doc_id, concept_id, weight)
        unique_ids = sorted(set(concept_ids))
        self._doc_concepts[doc_id] = set(unique_ids)
        if len(unique_ids) > 1:
            pair_weight = 1.0 / (len(unique_ids) - 1)
            for i, a in enumerate(unique_ids):
                for b in unique_ids[i + 1 :]:
                    self._add_edge(a, b, pair_weight)

    def add_structure(self, structure: UniversalStructure) -> str:
        doc_id = f"s:{structure.id}"
        self._add_node(doc_id, "structure", structure.metadata.title)
        concepts: dict[str, float] = {}
        for tag in structure.metadata.tags:
            concepts[tag] = max(concepts.get(tag, 0.0), 1.0)
        for mechanism in structure.mechanics.mechanisms:
            label = MECHANISM_LABELS_JA[mechanism.category.value]
            concepts[label] = max(concepts.get(label, 0.0), 0.5 + mechanism.strength)
        for beat in structure.beats.pivotal_beats():
            label = BEAT_TYPE_LABELS_JA[beat.beat_type.value]
            concepts[label] = max(concepts.get(label, 0.0), 0.8)
        histogram = structure.tension_graph.edge_kind_histogram()
        dominant = max(histogram.items(), key=lambda item: item[1])[0]
        concepts[EDGE_KIND_LABELS_JA[dominant]] = max(concepts.get(EDGE_KIND_LABELS_JA[dominant], 0.0), 0.7)
        self._link_document(doc_id, concepts)
        return doc_id

    def add_research_log(self, log: ResearchLog) -> str:
        doc_id = f"l:{log.id}"
        self._add_node(doc_id, "log", log.title)
        concepts = {tag: 1.0 for tag in log.tags}
        known = [n["label"] for n in self._nodes.values() if n["type"] == "concept"]
        text = " ".join([log.summary, *(f.claim for f in log.findings)])
        for label in known:
            if label in text:
                concepts.setdefault(label, 0.6)
        self._link_document(doc_id, concepts)
        return doc_id

    def add_article(self, article: PublishedArticle) -> str:
        doc_id = f"a:{article.id}"
        self._add_node(doc_id, "article", article.title, axis=article.axis.value)
        concepts = {tag: 1.0 for tag in article.tags}
        known = [n["label"] for n in self._nodes.values() if n["type"] == "concept"]
        for label in known:
            if label in article.markdown:
                concepts.setdefault(label, 0.6)
        self._link_document(doc_id, concepts)
        return doc_id

    def build_from_db(self, db: Any | None = None) -> "KnowledgeGraph":
        """DB 上の記事（ArticleRepository）と構造（StructureRepository）からグラフを構築する。"""
        database = db or self.db
        if database is None:
            raise ValueError("build_from_db には Database が必要です")
        from core.asset_publisher.repository import ArticleRepository

        try:
            from core.structure_engine.repository import StructureRepository
        except ImportError:
            structures: list[UniversalStructure] = []
        else:
            structures = StructureRepository(database).all()
        for structure in structures:
            self.add_structure(structure)
        for article in ArticleRepository(database).all():
            self.add_article(article)
        return self

    # --------------------------------------------------------------- queries
    def _neighbors(self) -> dict[str, dict[str, float]]:
        adjacency: dict[str, dict[str, float]] = {node_id: {} for node_id in self._nodes}
        for (a, b), weight in self._edges.items():
            adjacency[a][b] = weight
            adjacency[b][a] = weight
        return adjacency

    def pagerank(self, damping: float = 0.85, max_iter: int = 200, tol: float = 1e-12) -> dict[str, float]:
        """重み付き PageRank（総和1）。"""
        nodes = sorted(self._nodes)
        if not nodes:
            return {}
        adjacency = self._neighbors()
        n = len(nodes)
        rank = {node: 1.0 / n for node in nodes}
        out_weight = {node: sum(adjacency[node].values()) for node in nodes}
        for _ in range(max_iter):
            dangling = sum(rank[node] for node in nodes if out_weight[node] <= 0)
            updated: dict[str, float] = {}
            for node in nodes:
                incoming = sum(rank[nb] * w / out_weight[nb] for nb, w in adjacency[node].items() if out_weight[nb] > 0)
                updated[node] = (1.0 - damping) / n + damping * (incoming + dangling / n)
            delta = sum(abs(updated[node] - rank[node]) for node in nodes)
            rank = updated
            if delta < tol:
                break
        total = sum(rank.values())
        return {node: value / total for node, value in rank.items()}

    def communities(self, max_iter: int = 50) -> dict[str, int]:
        """概念ノードのラベル伝播コミュニティ（概念ラベル → コミュニティ番号）。"""
        adjacency = self._neighbors()
        concepts = sorted(node for node, data in self._nodes.items() if data["type"] == "concept")
        labels = {node: node for node in concepts}
        for _ in range(max_iter):
            changed = False
            for node in concepts:
                votes: dict[str, float] = defaultdict(float)
                for neighbor, weight in adjacency[node].items():
                    if neighbor in labels:
                        votes[labels[neighbor]] += weight
                if not votes:
                    continue
                top_weight = max(votes.values())
                candidates = sorted(label for label, weight in votes.items() if abs(weight - top_weight) <= 1e-12)
                choice = candidates[0]
                if labels[node] in candidates:
                    choice = labels[node]
                if choice != labels[node]:
                    labels[node] = choice
                    changed = True
            if not changed:
                break
        ordered = sorted(set(labels.values()))
        mapping = {label: index for index, label in enumerate(ordered)}
        return {self._nodes[node]["label"]: mapping[labels[node]] for node in concepts}

    def concepts(self) -> list[dict[str, Any]]:
        """概念ごとの統計（PageRank・文書種別ごとの出現数・コミュニティ）。"""
        rank = self.pagerank()
        community = self.communities()
        adjacency = self._neighbors()
        rows: list[dict[str, Any]] = []
        for node_id, data in sorted(self._nodes.items()):
            if data["type"] != "concept":
                continue
            counts = {"structure": 0, "article": 0, "log": 0}
            for neighbor in adjacency[node_id]:
                kind = self._nodes[neighbor]["type"]
                if kind in counts:
                    counts[kind] += 1
            rows.append(
                {
                    "label": data["label"],
                    "pagerank": round(rank.get(node_id, 0.0), 8),
                    "structure_count": counts["structure"],
                    "article_count": counts["article"],
                    "log_count": counts["log"],
                    "degree": len(adjacency[node_id]),
                    "community": community.get(data["label"], -1),
                }
            )
        rows.sort(key=lambda row: (-row["pagerank"], row["label"]))
        return rows

    def detect_gaps(self, top_k: int = 5) -> list[ContentGap]:
        """記事化されていない高需要の概念と、記事が橋渡ししていないコミュニティ間ペアを検出する。"""
        rows = self.concepts()
        if not rows:
            return []
        max_rank = max(row["pagerank"] for row in rows) or 1.0
        max_freq = max(row["structure_count"] + row["log_count"] for row in rows) or 1
        by_label = {row["label"]: row for row in rows}
        adjacency = self._neighbors()
        covered = {label for label, row in by_label.items() if row["article_count"] > 0}
        gaps: list[ContentGap] = []
        for row in rows:
            source_frequency = row["structure_count"] + row["log_count"]
            if source_frequency == 0:
                continue
            demand = 0.6 * (row["pagerank"] / max_rank) + 0.4 * (source_frequency / max_freq)
            coverage = 1.0 - math.exp(-row["article_count"])
            concept_neighbors = [
                self._nodes[n]["label"] for n in adjacency[f"c:{row['label']}"] if self._nodes[n]["type"] == "concept"
            ]
            covered_ratio = (
                sum(1 for label in concept_neighbors if label in covered) / len(concept_neighbors) if concept_neighbors else 0.0
            )
            novelty = 1.0 / (1.0 + covered_ratio)
            score = demand * (1.0 - coverage) * (0.5 + 0.5 * novelty)
            if score <= 0:
                continue
            axis = _infer_axis(row["label"])
            gaps.append(
                ContentGap(
                    concepts=[row["label"]],
                    axis=axis,
                    gap_score=round(score, 6),
                    demand=round(demand, 6),
                    coverage=round(coverage, 6),
                    rationale=(
                        f"「{row['label']}」は構造・研究ログで {source_frequency} 回参照され PageRank 上位にあるが、"
                        f"記事での扱いは {row['article_count']} 件にとどまる"
                    ),
                    suggested_title=f"「{row['label']}」を設計する――{AXIS_LABELS_JA[axis.value]}の未踏領域",
                    suggested_outline=[
                        f"{BLUEPRINT_ROLE_LABELS_JA[role.value]}: {row['label']}" for role in BLUEPRINT_ROLE_ORDER
                    ],
                )
            )
        community = self.communities()
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["structure_count"] + row["log_count"] > 0:
                groups[community.get(row["label"], -1)].append(row)
        article_sets = [concepts for doc, concepts in self._doc_concepts.items() if doc.startswith("a:")]
        group_ids = sorted(groups)
        for i, first in enumerate(group_ids):
            for second in group_ids[i + 1 :]:
                top_a = max(groups[first], key=lambda r: r["pagerank"])
                top_b = max(groups[second], key=lambda r: r["pagerank"])
                id_a, id_b = f"c:{top_a['label']}", f"c:{top_b['label']}"
                bridged = any(id_a in concepts and id_b in concepts for concepts in article_sets)
                if bridged:
                    continue
                demand = 0.5 * (top_a["pagerank"] + top_b["pagerank"]) / max_rank
                score = 0.9 * demand
                gaps.append(
                    ContentGap(
                        concepts=[top_a["label"], top_b["label"]],
                        axis=None,
                        gap_score=round(score, 6),
                        demand=round(demand, 6),
                        coverage=0.0,
                        rationale=(
                            f"「{top_a['label']}」と「{top_b['label']}」は別々の概念群の中心にあるが、"
                            "両者を橋渡しする記事が存在しない（構造的空隙）"
                        ),
                        suggested_title=f"「{top_a['label']}」×「{top_b['label']}」――領域横断の構造論",
                        suggested_outline=[
                            f"{BLUEPRINT_ROLE_LABELS_JA[role.value]}: {top_a['label']}と{top_b['label']}の接点"
                            for role in BLUEPRINT_ROLE_ORDER
                        ],
                    )
                )
        gaps.sort(key=lambda gap: (-gap.gap_score, gap.concepts))
        unique: list[ContentGap] = []
        seen: set[tuple[str, ...]] = set()
        for gap in gaps:
            key = tuple(sorted(gap.concepts))
            if key in seen:
                continue
            seen.add(key)
            unique.append(gap)
            if len(unique) >= top_k:
                break
        return unique

    def to_mermaid(self, max_concepts: int = 24) -> str:
        """上位概念とその共起を Mermaid flowchart として出力する。"""
        rows = self.concepts()[:max_concepts]
        index = {row["label"]: f"C{i}" for i, row in enumerate(rows)}
        lines = ["flowchart LR"]
        for row in rows:
            label = re.sub(r"[\"\[\]{}()<>|#;`]", " ", row["label"]).strip() or "概念"
            lines.append(f'    {index[row["label"]]}["{label}"]')
        for (a, b), weight in sorted(self._edges.items()):
            if not (a.startswith("c:") and b.startswith("c:")):
                continue
            label_a, label_b = self._nodes[a]["label"], self._nodes[b]["label"]
            if label_a in index and label_b in index:
                lines.append(f"    {index[label_a]} ---|{weight:.2f}| {index[label_b]}")
        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        counts = defaultdict(int)
        for data in self._nodes.values():
            counts[data["type"]] += 1
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "concepts": counts["concept"],
            "articles": counts["article"],
            "structures": counts["structure"],
            "logs": counts["log"],
            "communities": len(set(self.communities().values())),
        }
