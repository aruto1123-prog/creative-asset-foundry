// 多軸グリッド実験台帳（X/Y/Z 軸定義 → セル生成 → スコア入力 → 主効果・分散分析）と構造化トークン辞書。

import { useEffect, useMemo, useState } from "react";
import { api, downloadJson } from "../../api/client";
import type { GridAxis, GridAxisKind, GridCell, GridDetail, TokenEntry } from "../../api/types";
import { Badge, Button, Card, EmptyState, Field, NumberInput, Select, Stat, TextInput, Toggle } from "../../components/ui";
import { useAction, useResource } from "../../hooks/useAsync";
import { fmt, parseAxisValues } from "../../lib/format";

const AXIS_KINDS: Array<{ value: GridAxisKind; label: string }> = [
  { value: "token", label: "トークン（プロンプト挿入）" },
  { value: "param", label: "絵柄パラメータ" },
  { value: "lora_weight", label: "LoRA 重み" },
  { value: "merge_weight", label: "マージ比率" },
  { value: "cfg", label: "CFG" },
  { value: "steps", label: "ステップ数" },
  { value: "seed", label: "シード" },
  { value: "sampler", label: "サンプラー" },
];

interface AxisDraft {
  enabled: boolean;
  name: string;
  kind: GridAxisKind;
  target: string;
  values: string;
}

function toAxis(draft: AxisDraft): GridAxis {
  return { name: draft.name || draft.target || draft.kind, kind: draft.kind, target: draft.target, values: parseAxisValues(draft.values) };
}

function AxisEditor({ label, draft, onChange, optional }: { label: string; draft: AxisDraft; onChange: (draft: AxisDraft) => void; optional: boolean }) {
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-850 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold text-ink-200">{label}</span>
        {optional && <Toggle checked={draft.enabled} onChange={(enabled) => onChange({ ...draft, enabled })} label="使用する" />}
      </div>
      {(draft.enabled || !optional) && (
        <div className="grid grid-cols-1 gap-2 md:grid-cols-4">
          <Field label="軸名">
            <TextInput value={draft.name} onChange={(name) => onChange({ ...draft, name })} ariaLabel={`${label} 軸名`} />
          </Field>
          <Field label="種別">
            <Select value={draft.kind} options={AXIS_KINDS} onChange={(kind) => onChange({ ...draft, kind })} ariaLabel={`${label} 種別`} />
          </Field>
          <Field label="対象（パラメータ名・LoRA名・トークンカテゴリ）">
            <TextInput value={draft.target} onChange={(target) => onChange({ ...draft, target })} ariaLabel={`${label} 対象`} />
          </Field>
          <Field label="値（カンマ区切り）">
            <TextInput value={draft.values} onChange={(values) => onChange({ ...draft, values })} ariaLabel={`${label} 値`} />
          </Field>
        </div>
      )}
    </div>
  );
}

function axisValue(axis: GridAxis | null, index: number): string {
  if (!axis) {
    return "-";
  }
  const value = axis.values[index];
  return value === undefined ? String(index) : String(value);
}

export function GridPanel() {
  const [name, setName] = useState("線画タッチ × ベタ比率");
  const [basePrompt, setBasePrompt] = useState("1girl, upper body, manga panel, monochrome");
  const [negative, setNegative] = useState("lowres, blurry");
  const [seed, setSeed] = useState(42);
  const [x, setX] = useState<AxisDraft>({ enabled: true, name: "線画タッチ", kind: "token", target: "line_art", values: "clean lineart, rough sketch lines, variable line weight" });
  const [y, setY] = useState<AxisDraft>({ enabled: true, name: "ベタ比率", kind: "param", target: "style.black_fill_ratio", values: "0.2, 0.5, 0.8" });
  const [z, setZ] = useState<AxisDraft>({ enabled: false, name: "CFG", kind: "cfg", target: "cfg", values: "5, 7" });
  const [detail, setDetail] = useState<GridDetail | null>(null);
  const [scores, setScores] = useState<Record<string, number>>({});
  const [listVersion, setListVersion] = useState(0);
  const experiments = useResource(() => api.style.grids(), [listVersion]);
  const tokens = useResource<TokenEntry[]>(() => api.style.tokens(), [listVersion]);
  const [tokenDraft, setTokenDraft] = useState<TokenEntry>({ token: "", category: "line_art", weight: 1, description: "", tags: [] });

  useEffect(() => {
    if (!detail) {
      return;
    }
    const initial: Record<string, number> = {};
    for (const cell of detail.cells) {
      if (cell.score !== null) {
        initial[cell.id] = cell.score;
      }
    }
    setScores(initial);
  }, [detail]);

  const create = useAction(
    async () => {
      const created = await api.style.createGrid({
        name,
        base_prompt: basePrompt,
        negative_prompt: negative,
        x_axis: toAxis(x),
        y_axis: y.enabled ? toAxis(y) : null,
        z_axis: z.enabled ? toAxis(z) : null,
        seed,
      });
      const full = await api.style.grid(created.experiment.id);
      setDetail(full);
      setListVersion((value) => value + 1);
      return full;
    },
    { successMessage: (d) => `${d.cells.length} セルの実験を作成しました` },
  );

  const open = useAction(async (id: string) => {
    const full = await api.style.grid(id);
    setDetail(full);
    return full;
  });

  const record = useAction(
    async (cell: GridCell) => {
      if (!detail) {
        return null;
      }
      await api.style.recordGridResult(detail.experiment.id, { cell_id: cell.id, score: scores[cell.id] ?? 0 });
      const full = await api.style.grid(detail.experiment.id);
      setDetail(full);
      return full;
    },
    { successMessage: "スコアを記録しました" },
  );

  const addToken = useAction(
    async () => {
      await api.style.addToken(tokenDraft);
      setTokenDraft({ ...tokenDraft, token: "", description: "" });
      setListVersion((value) => value + 1);
    },
    { successMessage: "トークンを辞書に追加しました" },
  );

  const removeToken = useAction(
    async (entry: TokenEntry) => {
      await api.style.deleteToken(entry.token, entry.category);
      setListVersion((value) => value + 1);
    },
    { successMessage: "トークンを削除しました" },
  );

  const tokenGroups = useMemo(() => {
    const groups = new Map<string, TokenEntry[]>();
    for (const entry of tokens.data ?? []) {
      groups.set(entry.category, [...(groups.get(entry.category) ?? []), entry]);
    }
    return Array.from(groups.entries());
  }, [tokens.data]);

  const analysis = detail?.analysis ?? null;
  const experiment = detail?.experiment ?? null;

  return (
    <div className="flex flex-col gap-4">
      <Card title="多軸グリッド実験の作成" subtitle="線画タッチ・白黒トーン設計・パース歪み等の構造化トークンやパラメータを X/Y/Z 軸に割り当て、直積バッチを生成します">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <Field label="実験名">
            <TextInput value={name} onChange={setName} ariaLabel="実験名" />
          </Field>
          <Field label="ベースプロンプト" className="md:col-span-2">
            <TextInput value={basePrompt} onChange={setBasePrompt} ariaLabel="ベースプロンプト" />
          </Field>
          <Field label="シード">
            <NumberInput value={seed} onChange={setSeed} ariaLabel="グリッドシード" />
          </Field>
          <Field label="ネガティブプロンプト" className="md:col-span-4">
            <TextInput value={negative} onChange={setNegative} ariaLabel="ネガティブプロンプト" />
          </Field>
        </div>
        <div className="mt-3 flex flex-col gap-2">
          <AxisEditor label="X 軸" draft={x} onChange={setX} optional={false} />
          <AxisEditor label="Y 軸" draft={y} onChange={setY} optional />
          <AxisEditor label="Z 軸" draft={z} onChange={setZ} optional />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button variant="primary" loading={create.running} onClick={() => void create.run()}>
            実験を作成
          </Button>
          {(experiments.data ?? []).length > 0 && (
            <span className="flex flex-wrap items-center gap-1 text-xs text-ink-400">
              既存の実験:
              {(experiments.data ?? []).slice(0, 8).map((item) => (
                <Button key={item.id} size="sm" variant="ghost" onClick={() => void open.run(item.id)}>
                  {item.name}
                </Button>
              ))}
            </span>
          )}
        </div>
      </Card>

      {detail && experiment && (
        <Card
          title={`実験: ${experiment.name}`}
          subtitle={<span className="font-mono">{experiment.id}</span>}
          actions={
            <Button size="sm" onClick={() => downloadJson(`${experiment.id}_cells.json`, detail.cells)}>
              セル一覧 JSON
            </Button>
          }
        >
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <Stat label="セル数" value={detail.cells.length} />
            <Stat label="採点済み" value={analysis?.n_scored ?? detail.cells.filter((cell) => cell.score !== null).length} />
            <Stat label="最良セル" value={analysis?.best_cell_id ? analysis.best_cell_id.slice(-6) : "-"} />
            <Stat label="状態" value={experiment.status} />
          </div>
          <div className="mt-3 max-h-[420px] overflow-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-ink-900 text-ink-400">
                <tr>
                  <th className="py-1 pr-2">{experiment.x_axis.name}</th>
                  {experiment.y_axis && <th className="py-1 pr-2">{experiment.y_axis.name}</th>}
                  {experiment.z_axis && <th className="py-1 pr-2">{experiment.z_axis.name}</th>}
                  <th className="py-1 pr-2">プロンプト</th>
                  <th className="py-1 pr-2">シード</th>
                  <th className="py-1">スコア</th>
                </tr>
              </thead>
              <tbody>
                {detail.cells.map((cell) => (
                  <tr key={cell.id} className={`border-t border-ink-800 align-top text-ink-200 ${analysis?.best_cell_id === cell.id ? "bg-emerald-950/30" : ""}`}>
                    <td className="py-1 pr-2">{axisValue(experiment.x_axis, cell.x_index)}</td>
                    {experiment.y_axis && <td className="py-1 pr-2">{axisValue(experiment.y_axis, cell.y_index)}</td>}
                    {experiment.z_axis && <td className="py-1 pr-2">{axisValue(experiment.z_axis, cell.z_index)}</td>}
                    <td className="max-w-md py-1 pr-2 text-ink-300">{cell.prompt}</td>
                    <td className="py-1 pr-2 font-mono">{cell.seed}</td>
                    <td className="py-1">
                      <div className="flex items-center gap-1">
                        <NumberInput
                          value={scores[cell.id] ?? cell.score ?? 0}
                          onChange={(value) => setScores((current) => ({ ...current, [cell.id]: value }))}
                          step={0.5}
                          className="w-20 py-1 text-xs"
                          ariaLabel={`セル ${cell.id} のスコア`}
                        />
                        <Button size="sm" onClick={() => void record.run(cell)} loading={record.running}>
                          記録
                        </Button>
                        {cell.score !== null && <Badge tone="success">{cell.score}</Badge>}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {analysis && analysis.n_scored > 0 && (
            <div className="mt-4 grid grid-cols-1 gap-3 xl:grid-cols-2">
              <div className="rounded-lg border border-ink-700 bg-ink-850 p-3">
                <div className="mb-2 text-xs font-semibold text-ink-300">主効果（軸の値ごとの平均スコア）</div>
                {Object.entries(analysis.main_effects).map(([axisName, effects]) => {
                  const means = effects.map((effect) => Number(effect.mean ?? 0));
                  const maxMean = Math.max(1e-6, ...means.map((value) => Math.abs(value)));
                  return (
                    <div key={axisName} className="mb-3">
                      <div className="mb-1 text-[11px] text-ink-400">{axisName}</div>
                      {effects.map((effect, index) => (
                        <div key={index} className="flex items-center gap-2 text-[11px]">
                          <span className="w-40 truncate text-ink-300">{String(effect.value ?? index)}</span>
                          <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-700">
                            <div className="h-full rounded-full bg-accent" style={{ width: `${(Math.abs(Number(effect.mean ?? 0)) / maxMean) * 100}%` }} />
                          </div>
                          <span className="w-24 text-right font-mono text-ink-200">
                            {fmt(Number(effect.mean ?? 0))}（n={String(effect.count ?? 0)}）
                          </span>
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
              <div className="rounded-lg border border-ink-700 bg-ink-850 p-3">
                <div className="mb-2 text-xs font-semibold text-ink-300">二元配置分散分析（平方和）</div>
                <table className="w-full text-left text-xs">
                  <tbody>
                    {Object.entries(analysis.anova).map(([key, value]) => (
                      <tr key={key} className="border-t border-ink-800">
                        <td className="py-1 pr-2 font-mono text-ink-300">{key}</td>
                        <td className="py-1 text-right font-mono text-ink-100">{fmt(value, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {analysis.interaction_xy && (
                  <>
                    <div className="mb-1 mt-3 text-xs font-semibold text-ink-300">X × Y 交互作用（セル平均）</div>
                    <table className="text-xs">
                      <tbody>
                        {analysis.interaction_xy.map((row, i) => (
                          <tr key={i}>
                            {row.map((value, j) => (
                              <td key={j} className="border border-ink-800 px-2 py-1 text-center font-mono text-ink-200">
                                {value === null ? "-" : value.toFixed(2)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}
              </div>
            </div>
          )}
        </Card>
      )}

      <Card title="構造化トークン辞書" subtitle="線画タッチ・白黒トーン・パース歪み等の再利用可能なプロンプト語彙">
        <div className="grid grid-cols-1 gap-2 md:grid-cols-5">
          <Field label="トークン">
            <TextInput value={tokenDraft.token} onChange={(token) => setTokenDraft({ ...tokenDraft, token })} ariaLabel="トークン" />
          </Field>
          <Field label="カテゴリ">
            <TextInput value={tokenDraft.category} onChange={(category) => setTokenDraft({ ...tokenDraft, category })} ariaLabel="トークンカテゴリ" />
          </Field>
          <Field label="重み">
            <NumberInput value={tokenDraft.weight} onChange={(weight) => setTokenDraft({ ...tokenDraft, weight })} step={0.05} ariaLabel="トークン重み" />
          </Field>
          <Field label="説明">
            <TextInput value={tokenDraft.description} onChange={(description) => setTokenDraft({ ...tokenDraft, description })} ariaLabel="トークン説明" />
          </Field>
          <div className="flex items-end">
            <Button variant="primary" disabled={!tokenDraft.token.trim()} loading={addToken.running} onClick={() => void addToken.run()}>
              追加
            </Button>
          </div>
        </div>
        <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-3">
          {tokenGroups.length === 0 && <EmptyState title="トークン辞書は空です" />}
          {tokenGroups.map(([category, entries]) => (
            <div key={category} className="rounded-lg border border-ink-700 bg-ink-850 p-3">
              <div className="mb-2 text-xs font-semibold text-ink-300">{category}</div>
              <div className="flex flex-col gap-1">
                {entries.map((entry) => (
                  <div key={`${entry.category}-${entry.token}`} className="flex items-center gap-2 text-[11px]">
                    <span className="font-mono text-ink-100">{entry.weight === 1 ? entry.token : `(${entry.token}:${entry.weight.toFixed(2)})`}</span>
                    <span className="truncate text-ink-500">{entry.description}</span>
                    <button type="button" className="ml-auto text-ink-500 hover:text-rose-300" onClick={() => void removeToken.run(entry)} aria-label={`${entry.token} を削除`}>
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
