// Workspace 1: Structure Lab — 抽出・プリセット骨格ブラウザー・5層詳細・合成/変異・異分野変換・類似検索。

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, errorMessage } from "../api/client";
import type { ExtractionResult, StructureSummary, UniversalStructure } from "../api/types";
import { Badge, Card, EmptyState, ErrorNote, Spinner, Tabs, TextInput, cx } from "../components/ui";
import { CATEGORY_LABELS, SOURCE_KIND_LABELS, labelOf } from "../lib/labels";
import { ComposePanel } from "./structure/ComposePanel";
import { ExtractPanel } from "./structure/ExtractPanel";
import { SearchPanel } from "./structure/SearchPanel";
import { StructureDetail } from "./structure/StructureDetail";
import { TranspilePanel } from "./structure/TranspilePanel";

type CategoryKey =
  | "all"
  | "narrative_archetype"
  | "manga_panel_rhythm"
  | "persuasion_marketing"
  | "advanced_narrative"
  | "manga_direction"
  | "emotional_wave"
  | "custom";
type ModeKey = "detail" | "extract" | "compose" | "transpile" | "search";

const CATEGORY_TABS: Array<{ key: CategoryKey; label: string }> = [
  { key: "all", label: "全て" },
  { key: "narrative_archetype", label: "物語原型" },
  { key: "manga_panel_rhythm", label: "漫画リズム" },
  { key: "persuasion_marketing", label: "説得・マーケ" },
  { key: "advanced_narrative", label: "高度な物語構造" },
  { key: "manga_direction", label: "漫画演出" },
  { key: "emotional_wave", label: "感情波形" },
  { key: "custom", label: "カスタム" },
];

const MODE_TABS: Array<{ key: ModeKey; label: string }> = [
  { key: "detail", label: "5階層詳細" },
  { key: "extract", label: "テキストから抽出" },
  { key: "compose", label: "合成・変異" },
  { key: "transpile", label: "異分野変換" },
  { key: "search", label: "類似検索" },
];

export function StructureLab({ dataVersion, onDataChanged }: { dataVersion: number; onDataChanged: () => void }) {
  const [category, setCategory] = useState<CategoryKey>("all");
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<StructureSummary[]>([]);
  const [allItems, setAllItems] = useState<StructureSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [listVersion, setListVersion] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [structure, setStructure] = useState<UniversalStructure | null>(null);
  const [structureLoading, setStructureLoading] = useState(false);
  const [structureError, setStructureError] = useState<string | null>(null);
  const [mode, setMode] = useState<ModeKey>("detail");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const handle = window.setTimeout(() => {
      api.structures
        .list({ category: category === "all" ? null : category, q: query.trim() || null, limit: 300 })
        .then((result) => {
          if (cancelled) {
            return;
          }
          setItems(result.items);
          setTotal(result.total);
          setListError(null);
        })
        .catch((reason: unknown) => {
          if (!cancelled) {
            setListError(errorMessage(reason));
          }
        })
        .finally(() => {
          if (!cancelled) {
            setLoading(false);
          }
        });
    }, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [category, query, listVersion, dataVersion]);

  useEffect(() => {
    api.structures
      .list({ limit: 500 })
      .then((result) => setAllItems(result.items))
      .catch(() => setAllItems([]));
  }, [listVersion, dataVersion]);

  useEffect(() => {
    if (!selectedId && items.length > 0) {
      setSelectedId(items[0].id);
    }
  }, [items, selectedId]);

  useEffect(() => {
    if (!selectedId) {
      setStructure(null);
      return;
    }
    let cancelled = false;
    setStructureLoading(true);
    api.structures
      .get(selectedId)
      .then((result) => {
        if (!cancelled) {
          setStructure(result);
          setStructureError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setStructureError(errorMessage(reason));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setStructureLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const refreshLists = useCallback(() => {
    setListVersion((value) => value + 1);
    onDataChanged();
  }, [onDataChanged]);

  const open = useCallback((id: string) => {
    setSelectedId(id);
    setMode("detail");
  }, []);

  const handleExtracted = useCallback(
    (result: ExtractionResult) => {
      refreshLists();
      setSelectedId(result.structure.id);
      setStructure(result.structure);
    },
    [refreshLists],
  );

  const grouped = useMemo(() => items, [items]);

  return (
    <div className="grid grid-cols-1 gap-4 p-5 xl:grid-cols-[340px_minmax(0,1fr)]">
      <Card
        title="プリセット骨格ブラウザー"
        subtitle={`${total} 件（シード骨格＋抽出・合成・変異）`}
        className="xl:sticky xl:top-4 xl:max-h-[calc(100vh-7rem)] xl:overflow-hidden"
        bodyClassName="flex flex-col gap-3 xl:h-[calc(100vh-12rem)]"
      >
        <Tabs tabs={CATEGORY_TABS} active={category} onChange={setCategory} size="sm" />
        <TextInput value={query} onChange={setQuery} placeholder="タイトル・要約・タグで検索" type="search" ariaLabel="構造検索" />
        <ErrorNote message={listError} />
        <div className="flex-1 overflow-y-auto pr-1">
          {loading && items.length === 0 && (
            <div className="flex justify-center py-6">
              <Spinner />
            </div>
          )}
          {!loading && grouped.length === 0 && <EmptyState title="構造がありません" hint="System 画面で初期シードを投入するか、テキストから抽出してください。" />}
          <ul className="flex flex-col gap-1.5">
            {grouped.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => open(item.id)}
                  className={cx(
                    "w-full rounded-lg border px-3 py-2 text-left transition-colors",
                    item.id === selectedId ? "border-accent bg-accent-soft/40" : "border-ink-700 bg-ink-850 hover:border-ink-500",
                  )}
                >
                  <div className="flex items-center gap-1.5">
                    <Badge tone={item.source_kind === "seed" ? "neutral" : "accent"}>{labelOf(SOURCE_KIND_LABELS, item.source_kind)}</Badge>
                    <span className="truncate text-[11px] text-ink-400">{labelOf(CATEGORY_LABELS, item.category)}</span>
                  </div>
                  <div className="mt-1 line-clamp-2 text-sm font-medium text-ink-100">{item.title}</div>
                  <div className="mt-1 flex gap-3 font-mono text-[10px] text-ink-500">
                    <span>peak {item.peak_t.toFixed(2)}</span>
                    <span>susp {item.suspense.toFixed(2)}</span>
                    <span>surp {item.surprise.toFixed(2)}</span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </Card>

      <div className="flex min-w-0 flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-ink-100">Structure Lab</h1>
            <p className="text-xs text-ink-400">普遍的構造の抽出・構造代数・異分野転用</p>
          </div>
          <Tabs tabs={MODE_TABS} active={mode} onChange={setMode} />
        </div>
        {mode === "detail" && (
          <>
            {structureLoading && !structure && <Spinner />}
            <ErrorNote message={structureError} />
            {structure ? (
              <StructureDetail
                structure={structure}
                onSaved={(updated) => {
                  setStructure(updated);
                  refreshLists();
                }}
                onDeleted={() => {
                  setSelectedId(null);
                  setStructure(null);
                  refreshLists();
                }}
              />
            ) : (
              !structureLoading && <EmptyState title="構造を選択してください" />
            )}
          </>
        )}
        {mode === "extract" && <ExtractPanel onExtracted={handleExtracted} />}
        {mode === "compose" && <ComposePanel items={allItems} current={structure} onCreated={refreshLists} onOpen={open} />}
        {mode === "transpile" && <TranspilePanel structure={structure} />}
        {mode === "search" && <SearchPanel structure={structure} onOpen={open} />}
      </div>
    </div>
  );
}
