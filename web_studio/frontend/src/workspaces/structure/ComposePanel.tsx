// 構造代数: 2構造の合成（Merge）と構造変異（Mutate）。

import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { MergeReport, MutationInfo, StructureSummary, UniversalStructure } from "../../api/types";
import { WaveChart } from "../../components/charts/WaveChart";
import { Badge, Button, Card, EmptyState, Field, NumberInput, Select, Slider, Stat } from "../../components/ui";
import { useAction, useResource } from "../../hooks/useAsync";
import { fmt } from "../../lib/format";
import { CATEGORY_LABELS, labelOf } from "../../lib/labels";

interface MergeOutcome {
  merged: UniversalStructure;
  report: MergeReport;
  a: UniversalStructure;
  b: UniversalStructure;
}

interface MutateOutcome {
  mutated: UniversalStructure;
  original: UniversalStructure;
  mutation: string;
}

function optionsOf(items: StructureSummary[]) {
  return items.map((item) => ({ value: item.id, label: `［${labelOf(CATEGORY_LABELS, item.category)}］${item.title}` }));
}

export function ComposePanel({
  items,
  current,
  onCreated,
  onOpen,
}: {
  items: StructureSummary[];
  current: UniversalStructure | null;
  onCreated: () => void;
  onOpen: (id: string) => void;
}) {
  const options = useMemo(() => optionsOf(items), [items]);
  const [aId, setAId] = useState("");
  const [bId, setBId] = useState("");
  const [weightTopology, setWeightTopology] = useState(0.7);
  const [weightWave, setWeightWave] = useState(0.7);
  const [mergeOutcome, setMergeOutcome] = useState<MergeOutcome | null>(null);

  const [mutateId, setMutateId] = useState("");
  const [mutation, setMutation] = useState("invert_disclosure");
  const [intensity, setIntensity] = useState(1);
  const [seed, setSeed] = useState(0);
  const [mutateOutcome, setMutateOutcome] = useState<MutateOutcome | null>(null);

  const mutations = useResource<MutationInfo[]>(() => api.structures.mutations(), []);

  useEffect(() => {
    if (items.length === 0) {
      return;
    }
    const fallbackA = current?.id ?? items[0].id;
    setAId((value) => value || fallbackA);
    setBId((value) => value || (items.find((item) => item.id !== fallbackA)?.id ?? items[0].id));
    setMutateId((value) => value || fallbackA);
  }, [items, current]);

  const merge = useAction(
    async () => {
      const [a, b] = await Promise.all([api.structures.get(aId), api.structures.get(bId)]);
      const response = await api.structures.merge({ struct_a_id: aId, struct_b_id: bId, weight_topology: weightTopology, weight_wave: weightWave, save: true });
      setMergeOutcome({ merged: response.structure, report: response.report, a, b });
      onCreated();
      return response;
    },
    { successMessage: (r) => `合成しました: ${r.structure.metadata.title}` },
  );

  const mutate = useAction(
    async () => {
      const original = await api.structures.get(mutateId);
      const response = await api.structures.mutate({ structure_id: mutateId, mutation_type: mutation, intensity, seed, save: true });
      setMutateOutcome({ mutated: response.structure, original, mutation });
      onCreated();
      return response;
    },
    { successMessage: (r) => `変異構造を生成しました: ${r.structure.metadata.title}` },
  );

  if (items.length < 2) {
    return <EmptyState title="合成には2つ以上の構造が必要です" hint="System 画面から初期シードを投入するか、テキストから構造を抽出してください。" />;
  }

  const mutationOptions = (mutations.data ?? []).map((item) => ({ value: item.type, label: item.label || item.type }));
  const mutationInfo = (mutations.data ?? []).find((item) => item.type === mutation);

  return (
    <div className="flex flex-col gap-4">
      <Card title="構造合成（Merge）" subtitle="構造Aの対立トポロジーと構造Bの感情・情報開示波形を補間合成し、矛盾を自動修復した新しい骨格を生成します">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Field label="構造A（力学トポロジーの供給元）">
            <Select value={aId} options={options} onChange={setAId} ariaLabel="構造A" />
          </Field>
          <Field label="構造B（感情・情報開示波形の供給元）">
            <Select value={bId} options={options} onChange={setBId} ariaLabel="構造B" />
          </Field>
          <Slider label="位相合成: 構造Aの比率（weight_topology）" value={weightTopology} onChange={setWeightTopology} hint="1.0 で A の対立トポロジーをそのまま採用" />
          <Slider label="波形合成: 構造Bの比率（weight_wave）" value={weightWave} onChange={setWeightWave} hint="1.0 で B の感情・情報開示波形をそのまま採用（DTW 整列）" />
        </div>
        <div className="mt-3">
          <Button variant="primary" loading={merge.running} disabled={!aId || !bId || aId === bId} onClick={() => void merge.run()}>
            合成を実行
          </Button>
          {aId === bId && <span className="ml-3 text-xs text-amber-300">異なる構造を選択してください</span>}
        </div>
        {mergeOutcome && (
          <div className="mt-4 flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold text-ink-100">{mergeOutcome.merged.metadata.title}</span>
              <Button size="sm" onClick={() => onOpen(mergeOutcome.merged.id)}>
                合成結果を開く
              </Button>
            </div>
            <WaveChart
              wave={mergeOutcome.merged.wave}
              beats={mergeOutcome.merged.beats.beats}
              overlays={[
                { label: `A: ${mergeOutcome.a.metadata.title}`, wave: mergeOutcome.a.wave, color: "#94a3b8", dash: "6 4" },
                { label: `B: ${mergeOutcome.b.metadata.title}`, wave: mergeOutcome.b.wave, color: "#38bdf8", dash: "2 3" },
              ]}
              initialChannels={["tension", "information_asymmetry"]}
              title="合成波形（実線）と元波形（破線）"
            />
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              <Stat label="DTW 整列コスト" value={fmt(mergeOutcome.report.dtw_alignment_cost, 4)} />
              <Stat label="位相: A比率" value={fmt(mergeOutcome.report.weight_topology)} />
              <Stat label="波形: B比率" value={fmt(mergeOutcome.report.weight_wave)} />
              <Stat label="自動修正" value={mergeOutcome.report.fixes_applied.length} />
            </div>
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <div className="rounded-lg border border-ink-700 bg-ink-850 p-3 text-xs">
                <div className="mb-1 font-semibold text-ink-300">適用した修正</div>
                {mergeOutcome.report.fixes_applied.length === 0 ? (
                  <span className="text-ink-500">修正は不要でした</span>
                ) : (
                  <ul className="list-disc space-y-0.5 pl-4 text-ink-200">
                    {mergeOutcome.report.fixes_applied.map((fix, index) => (
                      <li key={index}>{fix}</li>
                    ))}
                  </ul>
                )}
                {mergeOutcome.report.issues_remaining.length > 0 && (
                  <>
                    <div className="mb-1 mt-2 font-semibold text-amber-300">残存する注意点</div>
                    <ul className="list-disc space-y-0.5 pl-4 text-amber-200">
                      {mergeOutcome.report.issues_remaining.map((issue, index) => (
                        <li key={index}>{issue}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
              <div className="rounded-lg border border-ink-700 bg-ink-850 p-3 text-xs">
                <div className="mb-1 font-semibold text-ink-300">ノード整列（B → 合成後）</div>
                <table className="w-full text-left">
                  <tbody>
                    {Object.entries(mergeOutcome.report.node_mapping).map(([from, to]) => (
                      <tr key={from} className="border-t border-ink-800">
                        <td className="py-0.5 pr-2 font-mono text-ink-300">{from}</td>
                        <td className="py-0.5 text-ink-200">{to === null ? <Badge>未使用</Badge> : <span className="font-mono">{to}</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </Card>

      <Card title="構造変異（Mutate）" subtitle="情報開示順序の逆転（倒叙化）・権力勾配の反転・偽のクライマックス挿入などの構造変異演算">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Field label="変異させる構造">
            <Select value={mutateId} options={options} onChange={setMutateId} ariaLabel="変異させる構造" />
          </Field>
          <Field label="変異種別" hint={mutationInfo?.description}>
            <Select value={mutation} options={mutationOptions.length > 0 ? mutationOptions : [{ value: mutation, label: mutation }]} onChange={setMutation} ariaLabel="変異種別" />
          </Field>
          <Slider label="変異強度（intensity）" value={intensity} onChange={setIntensity} />
          <Field label="乱数シード">
            <NumberInput value={seed} onChange={setSeed} min={0} ariaLabel="乱数シード" />
          </Field>
        </div>
        <div className="mt-3">
          <Button variant="primary" loading={mutate.running} disabled={!mutateId} onClick={() => void mutate.run()}>
            変異を実行
          </Button>
        </div>
        {mutateOutcome && (
          <div className="mt-4 flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="accent">{mutateOutcome.mutation}</Badge>
              <span className="text-sm font-semibold text-ink-100">{mutateOutcome.mutated.metadata.title}</span>
              <Button size="sm" onClick={() => onOpen(mutateOutcome.mutated.id)}>
                変異結果を開く
              </Button>
            </div>
            <WaveChart
              wave={mutateOutcome.mutated.wave}
              beats={mutateOutcome.mutated.beats.beats}
              overlays={[{ label: `元: ${mutateOutcome.original.metadata.title}`, wave: mutateOutcome.original.wave, color: "#94a3b8" }]}
              initialChannels={["tension", "audience_knowledge", "character_knowledge"]}
              overlayChannel="tension"
              title="変異後（実線）と変異前の緊張度（破線）"
            />
          </div>
        )}
      </Card>
    </div>
  );
}
