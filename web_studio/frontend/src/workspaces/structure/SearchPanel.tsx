// 波形類似度検索（DTW / コサイン / ハイブリッド）。

import { useState } from "react";
import { api } from "../../api/client";
import type { SimilarityHit, UniversalStructure } from "../../api/types";
import { Badge, Button, Card, EmptyState, Field, NumberInput, Select } from "../../components/ui";
import { useAction } from "../../hooks/useAsync";
import { CATEGORY_LABELS, labelOf } from "../../lib/labels";

type Method = "dtw" | "cosine" | "hybrid";

const METHODS: Array<{ value: Method; label: string }> = [
  { value: "hybrid", label: "ハイブリッド（DTW＋コサイン）" },
  { value: "dtw", label: "DTW（動的時間伸縮）" },
  { value: "cosine", label: "コサイン類似度" },
];

const CATEGORY_OPTIONS = [
  { value: "", label: "全カテゴリ" },
  { value: "narrative_archetype", label: "物語原型" },
  { value: "manga_panel_rhythm", label: "漫画演出リズム" },
  { value: "persuasion_marketing", label: "説得・マーケティング" },
  { value: "advanced_narrative", label: "高度な物語構造" },
  { value: "manga_direction", label: "漫画演出" },
  { value: "emotional_wave", label: "感情波形プリセット" },
  { value: "custom", label: "カスタム" },
];

export function SearchPanel({ structure, onOpen }: { structure: UniversalStructure | null; onOpen: (id: string) => void }) {
  const [method, setMethod] = useState<Method>("hybrid");
  const [topK, setTopK] = useState(8);
  const [category, setCategory] = useState("");
  const [hits, setHits] = useState<SimilarityHit[] | null>(null);

  const search = useAction(async () => {
    if (!structure) {
      throw new Error("構造が選択されていません");
    }
    const response = await api.structures.search({ structure_id: structure.id, method, top_k: topK, category: category || null });
    setHits(response.hits.filter((hit) => hit.id !== structure.id));
    return response;
  });

  if (!structure) {
    return <EmptyState title="左の一覧から検索の基準となる構造を選択してください" />;
  }

  return (
    <Card title={`類似構造検索: ${structure.metadata.title}`} subtitle="感情・情報開示波形（緊張・解放・情報非対称性・認知負荷）の形状類似度で検索します">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
        <Field label="類似度手法">
          <Select value={method} options={METHODS} onChange={setMethod} ariaLabel="類似度手法" />
        </Field>
        <Field label="カテゴリ">
          <Select value={category} options={CATEGORY_OPTIONS} onChange={setCategory} ariaLabel="検索カテゴリ" />
        </Field>
        <Field label="上位件数">
          <NumberInput value={topK} onChange={setTopK} min={1} max={30} ariaLabel="上位件数" />
        </Field>
        <div className="flex items-end">
          <Button variant="primary" loading={search.running} onClick={() => void search.run()}>
            検索
          </Button>
        </div>
      </div>
      {hits && (
        <div className="mt-4 flex flex-col gap-2">
          {hits.length === 0 && <EmptyState title="類似構造が見つかりませんでした" />}
          {hits.map((hit, index) => (
            <button
              key={hit.id}
              type="button"
              onClick={() => onOpen(hit.id)}
              className="flex items-center gap-3 rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-left hover:border-accent/60"
            >
              <span className="w-6 text-right font-mono text-xs text-ink-500">{index + 1}</span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm text-ink-100">{hit.title}</span>
                  <Badge>{labelOf(CATEGORY_LABELS, hit.category)}</Badge>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink-700">
                  <div className="h-full rounded-full bg-accent" style={{ width: `${Math.round(hit.score * 100)}%` }} />
                </div>
              </div>
              <div className="w-40 text-right font-mono text-[11px] text-ink-300">
                <div>score {hit.score.toFixed(3)}</div>
                <div className="text-ink-500">
                  DTW {hit.dtw_distance === null ? "-" : hit.dtw_distance.toFixed(3)} / cos {hit.cosine === null ? "-" : hit.cosine.toFixed(3)}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </Card>
  );
}
