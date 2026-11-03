// 構造詳細: 5層タブ（力学グラフ・波形編集・ビート・メカニズム・ブループリント）＋転用ガイド・Mermaid。

import { useEffect, useMemo, useState } from "react";
import { api, downloadJson } from "../../api/client";
import type { EmotionalInformationWave, UniversalStructure } from "../../api/types";
import { BeatChain } from "../../components/charts/BeatChain";
import { TensionGraphView } from "../../components/charts/TensionGraphView";
import { WaveChart } from "../../components/charts/WaveChart";
import { Badge, Button, Card, CopyButton, EmptyState, ScoreBar, SectionTitle, Spinner, Stat, Tabs, TagList } from "../../components/ui";
import { useAction, useResource } from "../../hooks/useAsync";
import { fmt, summarizeWave } from "../../lib/format";
import { BEAT_TYPE_LABELS, CATEGORY_LABELS, MECHANISM_LABELS, NODE_KIND_LABELS, SOURCE_KIND_LABELS, labelOf } from "../../lib/labels";

type LayerKey = "graph" | "wave" | "beats" | "mechanics" | "blueprint" | "transfer" | "mermaid";

const LAYER_TABS: Array<{ key: LayerKey; label: string }> = [
  { key: "graph", label: "L1 力学グラフ" },
  { key: "wave", label: "L2 感情・情報波形" },
  { key: "beats", label: "L3 ビート" },
  { key: "mechanics", label: "L4 認知メカニズム" },
  { key: "blueprint", label: "L5 ブループリント" },
  { key: "transfer", label: "転用ガイド" },
  { key: "mermaid", label: "Mermaid" },
];

function stripComputed(wave: EmotionalInformationWave): EmotionalInformationWave {
  return {
    t: wave.t,
    tension: wave.tension,
    release: wave.release,
    audience_knowledge: wave.audience_knowledge,
    character_knowledge: wave.character_knowledge,
    cognitive_load: wave.cognitive_load,
  };
}

function TemplateView({ template }: { template: string }) {
  const parts = template.split(/(\{\{\s*[A-Z][A-Z0-9_]*\s*\}\})/g);
  return (
    <p className="whitespace-pre-wrap rounded-lg border border-ink-700 bg-ink-850 p-3 text-sm leading-relaxed text-ink-100">
      {parts.map((part, index) =>
        /^\{\{/.test(part) ? (
          <span key={index} className="mx-0.5 rounded bg-accent-soft px-1 font-mono text-xs text-accent-strong">
            {part}
          </span>
        ) : (
          <span key={index}>{part}</span>
        ),
      )}
    </p>
  );
}

export function StructureDetail({
  structure,
  onSaved,
  onDeleted,
}: {
  structure: UniversalStructure;
  onSaved: (structure: UniversalStructure) => void;
  onDeleted: (id: string) => void;
}) {
  const [layer, setLayer] = useState<LayerKey>("graph");
  const [draftWave, setDraftWave] = useState<EmotionalInformationWave>(structure.wave);
  const [selectedBeat, setSelectedBeat] = useState<string | null>(null);

  useEffect(() => {
    setDraftWave(structure.wave);
    setSelectedBeat(null);
  }, [structure]);

  const dirty = draftWave !== structure.wave;
  const summary = useMemo(() => summarizeWave(draftWave), [draftWave]);
  const beat = structure.beats.beats.find((item) => item.id === selectedBeat) ?? null;
  const nodeLabel = (id: string) => structure.tension_graph.nodes.find((node) => node.id === id)?.label ?? id;

  const saveWave = useAction(
    async () => {
      const updated = await api.structures.updateWave(structure.id, stripComputed(draftWave));
      onSaved(updated);
      return updated;
    },
    { successMessage: "波形を保存しました" },
  );
  const remove = useAction(
    async () => {
      await api.structures.remove(structure.id);
      onDeleted(structure.id);
    },
    { successMessage: "構造を削除しました" },
  );
  const mermaid = useResource(() => api.structures.mermaid(structure.id), [structure.id, layer === "mermaid"], layer === "mermaid");

  const meta = structure.metadata;
  return (
    <div className="flex flex-col gap-4">
      <Card
        title={meta.title}
        subtitle={<span className="font-mono">{structure.id}</span>}
        actions={
          <>
            <Badge tone="accent">{labelOf(CATEGORY_LABELS, meta.category)}</Badge>
            <Badge>{labelOf(SOURCE_KIND_LABELS, meta.source_kind)}</Badge>
            <Button size="sm" variant="ghost" onClick={() => downloadJson(`${structure.id}.json`, structure)}>
              JSON
            </Button>
            {meta.source_kind !== "seed" && (
              <Button
                size="sm"
                variant="danger"
                loading={remove.running}
                onClick={() => {
                  if (window.confirm(`「${meta.title}」を削除しますか？`)) {
                    void remove.run();
                  }
                }}
              >
                削除
              </Button>
            )}
          </>
        }
      >
        <p className="text-sm leading-relaxed text-ink-200">{meta.summary}</p>
        <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-6">
          <Stat label="ノード / エッジ" value={`${structure.tension_graph.nodes.length} / ${structure.tension_graph.edges.length}`} />
          <Stat label="ビート" value={structure.beats.beats.length} hint={`転換点 ${structure.beats.beats.filter((b) => b.is_turning_point).length}`} />
          <Stat label="緊張ピーク t" value={fmt(summary.peakT)} hint={`最大 ${fmt(summary.peakTension)}`} />
          <Stat label="サスペンス係数" value={fmt(summary.suspense, 3)} />
          <Stat label="サプライズ係数" value={fmt(summary.surprise, 3)} />
          <Stat label="カタルシス指数" value={fmt(summary.catharsis, 3)} />
        </div>
        <div className="mt-3 flex flex-col gap-1.5 text-xs text-ink-300">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-ink-500">タグ</span>
            <TagList items={meta.tags} />
          </div>
          {meta.lineage.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-ink-500">系譜</span>
              <TagList items={meta.lineage} tone="info" />
            </div>
          )}
          {meta.parent_ids.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-ink-500">親構造</span>
              <TagList items={meta.parent_ids} tone="accent" />
            </div>
          )}
        </div>
      </Card>

      <Tabs tabs={LAYER_TABS} active={layer} onChange={setLayer} size="sm" />

      {layer === "graph" && (
        <Card title="Layer 1: Dynamic Tension Graph" subtitle="主体・欲求・欠落・象徴の力学トポロジー（線種=関係種別、太さ=強度）">
          <TensionGraphView graph={structure.tension_graph} />
        </Card>
      )}

      {layer === "wave" && (
        <Card
          title="Layer 2: Emotional & Information Wave"
          subtitle={`正規化時間 ${draftWave.t.length} 点。情報非対称性 = 読者認知量 − 作中人物認知量`}
          actions={
            <>
              {dirty && <Badge tone="warning">未保存の編集あり</Badge>}
              <Button size="sm" variant="ghost" disabled={!dirty} onClick={() => setDraftWave(structure.wave)}>
                元に戻す
              </Button>
              <Button size="sm" variant="primary" disabled={!dirty} loading={saveWave.running} onClick={() => void saveWave.run()}>
                波形を保存
              </Button>
            </>
          }
        >
          <WaveChart wave={draftWave} beats={structure.beats.beats} editable onChange={setDraftWave} selectedBeatId={selectedBeat} onBeatSelect={setSelectedBeat} height={320} />
        </Card>
      )}

      {layer === "beats" && (
        <Card title="Layer 3: Structural Beats & Turning Points" subtitle="弧は因果（原因→結果）。菱形は転換点。クリックで詳細表示。">
          <BeatChain beats={structure.beats.beats} selectedId={selectedBeat} onSelect={setSelectedBeat} />
          <div className="mt-3 grid grid-cols-1 gap-2 lg:grid-cols-2">
            {structure.beats.beats.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setSelectedBeat(item.id)}
                className={`rounded-lg border px-3 py-2 text-left text-xs transition-colors ${item.id === selectedBeat ? "border-accent bg-accent-soft/40" : "border-ink-700 bg-ink-850 hover:border-ink-500"}`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={item.is_turning_point ? "accent" : "neutral"}>{BEAT_TYPE_LABELS[item.beat_type] ?? item.beat_type}</Badge>
                  <span className="font-mono text-ink-400">t={item.t.toFixed(2)}</span>
                  <span className="font-medium text-ink-100">{item.label}</span>
                </div>
                <p className="mt-1 text-ink-300">{item.function}</p>
              </button>
            ))}
          </div>
          {beat && (
            <div className="mt-3 rounded-lg border border-accent/40 bg-ink-850 p-3 text-xs text-ink-200">
              <div className="mb-1 text-sm font-semibold text-accent-strong">{beat.label}</div>
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                <span>緊張変化: {fmt(beat.tension_delta)}</span>
                <span>情報シフト: {fmt(beat.information_shift)}</span>
                <span>持続幅: {fmt(beat.span)}</span>
                <span>転換点: {beat.is_turning_point ? "はい" : "いいえ"}</span>
              </div>
              <div className="mt-2">原因ビート: {beat.causes.length > 0 ? beat.causes.join("、") : "なし（起点）"}</div>
              <div className="mt-1">関与ノード: {beat.involved_nodes.map(nodeLabel).join("、") || "なし"}</div>
            </div>
          )}
        </Card>
      )}

      {layer === "mechanics" && (
        <Card title="Layer 4: Cognitive & Empathy Mechanics" subtitle={`共感の錨: ${structure.mechanics.empathy_anchor}`}>
          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
            <span className="text-ink-500">感情遷移</span>
            {structure.mechanics.emotion_sequence.map((emotion, index) => (
              <span key={`${emotion}-${index}`} className="flex items-center gap-2">
                {index > 0 && <span className="text-ink-600">→</span>}
                <Badge tone="info">{emotion}</Badge>
              </span>
            ))}
          </div>
          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
            <span className="text-ink-500">没入ドライバー</span>
            <TagList items={structure.mechanics.immersion_drivers} tone="accent" />
          </div>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            {structure.mechanics.mechanisms.map((mechanism) => (
              <div key={mechanism.id} className="rounded-lg border border-ink-700 bg-ink-850 p-3 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-semibold text-ink-100">{mechanism.name}</span>
                  <Badge tone="accent">{MECHANISM_LABELS[mechanism.category] ?? mechanism.category}</Badge>
                </div>
                <p className="mt-1 text-ink-300">{mechanism.description}</p>
                <div className="mt-2 grid gap-1 text-ink-300">
                  <span>
                    <span className="text-ink-500">発火条件:</span> {mechanism.trigger}
                  </span>
                  <span>
                    <span className="text-ink-500">効果:</span> {mechanism.effect}
                  </span>
                  {mechanism.theory && (
                    <span>
                      <span className="text-ink-500">理論:</span> {mechanism.theory}
                    </span>
                  )}
                </div>
                <div className="mt-2">
                  <ScoreBar value={mechanism.strength} label="強度" />
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {layer === "blueprint" && (
        <Card title="Layer 5: Domain-Agnostic Blueprint" subtitle="抽象変数スロットを持つ分野非依存の純粋骨格">
          <TemplateView template={structure.blueprint.template} />
          <SectionTitle>スロット</SectionTitle>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-ink-400">
                <tr>
                  <th className="py-1 pr-3">スロット</th>
                  <th className="py-1 pr-3">説明</th>
                  <th className="py-1 pr-3">漫画</th>
                  <th className="py-1 pr-3">マーケ</th>
                  <th className="py-1">記事</th>
                </tr>
              </thead>
              <tbody>
                {structure.blueprint.slots.map((slot) => (
                  <tr key={slot.name} className="border-t border-ink-800 align-top text-ink-200">
                    <td className="py-1.5 pr-3 font-mono text-accent-strong">{slot.name}</td>
                    <td className="py-1.5 pr-3">{slot.description}</td>
                    <td className="py-1.5 pr-3 text-ink-300">{slot.examples.manga ?? "-"}</td>
                    <td className="py-1.5 pr-3 text-ink-300">{slot.examples.marketing ?? "-"}</td>
                    <td className="py-1.5 text-ink-300">{slot.examples.article ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4">
            <SectionTitle>構造的不変条件</SectionTitle>
            <ul className="list-disc space-y-1 pl-5 text-xs text-ink-200">
              {structure.blueprint.invariants.map((invariant, index) => (
                <li key={index}>{invariant}</li>
              ))}
            </ul>
          </div>
          <div className="mt-4">
            <SectionTitle>ノードとスロットの対応</SectionTitle>
            <div className="flex flex-wrap gap-2 text-xs">
              {structure.tension_graph.nodes
                .filter((node) => node.slot)
                .map((node) => (
                  <Badge key={node.id} tone="neutral">
                    {NODE_KIND_LABELS[node.kind]}「{node.label}」→ {node.slot}
                  </Badge>
                ))}
            </div>
          </div>
        </Card>
      )}

      {layer === "transfer" && (
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          {(
            [
              ["manga", "漫画への転用"],
              ["marketing", "マーケティングへの転用"],
              ["article", "記事への転用"],
            ] as const
          ).map(([key, label]) => (
            <Card key={key} title={label}>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink-200">{structure.transfer_guide[key] || "記述なし"}</p>
            </Card>
          ))}
          {structure.rhythm_profile && (
            <Card title="漫画演出リズム骨格" className="xl:col-span-3">
              <div className="flex items-end gap-1">
                {structure.rhythm_profile.panels_per_page.map((count, index) => {
                  const profile = structure.rhythm_profile;
                  const reveal = profile?.page_turn_reveals.includes(index);
                  const emphasis = profile?.emphasis_pages.includes(index);
                  const spread = profile?.spread_pages.includes(index);
                  return (
                    <div key={index} className="flex flex-col items-center gap-1">
                      <div
                        className={`w-7 rounded-t ${emphasis ? "bg-amber-400" : spread ? "bg-violet-400" : "bg-accent"}`}
                        style={{ height: `${12 + count * 12}px` }}
                        title={`${index + 1}ページ: ${count}コマ`}
                      />
                      <span className="text-[10px] text-ink-400">{index + 1}</span>
                      {reveal && <span className="text-[9px] text-amber-300">めくり</span>}
                    </div>
                  );
                })}
              </div>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-ink-300">
                {structure.rhythm_profile.pacing_notes.map((note, index) => (
                  <li key={index}>{note}</li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}

      {layer === "mermaid" && (
        <Card title="Mermaid 記法" subtitle="Obsidian や Mermaid 対応エディタへ貼り付けて図解として利用できます">
          {mermaid.loading && <Spinner />}
          {mermaid.error && <EmptyState title="Mermaid を取得できませんでした" hint={mermaid.error} />}
          {mermaid.data && (
            <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
              {(
                [
                  ["graph", "力学グラフ（flowchart）"],
                  ["beats", "ビート因果（flowchart）"],
                ] as const
              ).map(([key, label]) => (
                <div key={key} className="rounded-lg border border-ink-700 bg-ink-950">
                  <div className="flex items-center justify-between border-b border-ink-700 px-3 py-1.5 text-xs text-ink-400">
                    {label}
                    <CopyButton text={mermaid.data?.[key] ?? ""} />
                  </div>
                  <pre className="max-h-96 overflow-auto p-3 font-mono text-[11px] text-ink-200">{mermaid.data?.[key]}</pre>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
