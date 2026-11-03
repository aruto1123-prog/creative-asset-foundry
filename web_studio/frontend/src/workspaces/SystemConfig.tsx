// Workspace 5: System & Local AI Config — DeepSeek / ComfyUI 接続設定・思考段階設定・DB 統計とバックアップ・テレメトリ。

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { BackupInfo, ComfySettings, ConnectionTestResult, DbStats, HarnessSettings, LLMProvider, LLMSettings, SeedSummary, StructuredOutputMode, TelemetryResponse } from "../api/types";
import { JsonViewer } from "../components/JsonViewer";
import { Badge, Button, Card, EmptyState, ErrorNote, Field, NumberInput, Select, Slider, Spinner, Stat, TextInput, Toggle } from "../components/ui";
import { useAction, useResource } from "../hooks/useAsync";
import { bytes, dateTime, fmt, pct } from "../lib/format";
import { CATEGORY_LABELS, labelOf } from "../lib/labels";

const PROVIDERS: Array<{ value: LLMProvider; label: string }> = [
  { value: "auto", label: "auto（OpenAI互換 → Ollama → モックの順に自動選択）" },
  { value: "openai_compatible", label: "OpenAI 互換 API（vLLM / SGLang / llama.cpp server 等）" },
  { value: "ollama", label: "Ollama" },
  { value: "mock", label: "高精度モック（オフライン・決定論）" },
];

const STRUCTURED_MODES: Array<{ value: StructuredOutputMode; label: string }> = [
  { value: "json_object", label: "json_object（JSON モード）" },
  { value: "json_schema", label: "json_schema（スキーマ制約付き生成）" },
  { value: "prompt_only", label: "prompt_only（プロンプト指示のみ）" },
];

const STAGE_KEYS = ["decompose", "synthesize", "critique", "refine"] as const;
const STAGE_LABELS: Record<(typeof STAGE_KEYS)[number], string> = {
  decompose: "Decompose（分解）",
  synthesize: "Synthesize（合成）",
  critique: "Critique（批評）",
  refine: "Refine（精錬）",
};

function TestResult({ result }: { result: ConnectionTestResult | null }) {
  if (!result) {
    return null;
  }
  return (
    <div className={`mt-3 rounded-lg border px-3 py-2 text-xs ${result.ok ? "border-emerald-500/40 bg-emerald-950/30 text-emerald-100" : "border-rose-500/40 bg-rose-950/30 text-rose-100"}`}>
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={result.ok ? "success" : "danger"}>{result.ok ? "接続成功" : "接続失敗"}</Badge>
        {result.provider && <span>プロバイダ: {result.provider}</span>}
        {result.model && <span>モデル: {result.model}</span>}
        {typeof result.latency_ms === "number" && <span>応答 {result.latency_ms.toFixed(0)} ms</span>}
        {result.fallback_active && <Badge tone="warning">フォールバック中</Badge>}
      </div>
      {result.detail && <p className="mt-1 opacity-90">{result.detail}</p>}
      {result.models && result.models.length > 0 && <p className="mt-1 opacity-80">利用可能モデル: {result.models.join(", ")}</p>}
    </div>
  );
}

export function SystemConfig({ onDataChanged }: { onDataChanged: () => void }) {
  const config = useResource(() => api.system.getConfig(), []);
  const [llm, setLlm] = useState<LLMSettings | null>(null);
  const [comfy, setComfy] = useState<ComfySettings | null>(null);
  const [llmTest, setLlmTest] = useState<ConnectionTestResult | null>(null);
  const [comfyTest, setComfyTest] = useState<ConnectionTestResult | null>(null);
  const [dbVersion, setDbVersion] = useState(0);
  const stats = useResource<DbStats>(() => api.system.dbStats(), [dbVersion]);
  const backups = useResource<BackupInfo[]>(() => api.system.backups(), [dbVersion]);
  const seeds = useResource<SeedSummary>(() => api.seeds.summary(), [dbVersion]);
  const [telemetryVersion, setTelemetryVersion] = useState(0);
  const telemetry = useResource<TelemetryResponse>(() => api.system.telemetry(60), [telemetryVersion]);

  useEffect(() => {
    if (config.data) {
      setLlm(structuredClone(config.data.llm));
      setComfy(structuredClone(config.data.comfy));
    }
  }, [config.data]);

  const saveLlm = useAction(
    async () => {
      if (!llm) {
        return null;
      }
      const updated = await api.system.updateConfig({ llm });
      config.setData(updated);
      onDataChanged();
      return updated;
    },
    { successMessage: "LLM 設定を保存し、推論クライアントを再構築しました" },
  );

  const saveComfy = useAction(
    async () => {
      if (!comfy) {
        return null;
      }
      const updated = await api.system.updateConfig({ comfy });
      config.setData(updated);
      onDataChanged();
      return updated;
    },
    { successMessage: "ComfyUI 設定を保存しました" },
  );

  const testLlm = useAction(async () => {
    const result = await api.system.testLlm();
    setLlmTest(result);
    return result;
  });

  const testComfy = useAction(async () => {
    const result = await api.system.testComfy();
    setComfyTest(result);
    return result;
  });

  const backup = useAction(
    async () => {
      const result = await api.system.backup();
      setDbVersion((value) => value + 1);
      return result;
    },
    { successMessage: (r) => `バックアップを作成しました: ${r.path ?? r.filename ?? r.name ?? ""}` },
  );

  const loadSeeds = useAction(
    async (overwrite: boolean) => {
      const report = await api.seeds.load(overwrite);
      setDbVersion((value) => value + 1);
      onDataChanged();
      return report;
    },
    { successMessage: (r) => `シード投入: 追加 ${r.inserted}・更新 ${r.updated}・既存 ${r.skipped}（全 ${r.total}）` },
  );

  const updateHarness = (patch: Partial<HarnessSettings>) => {
    if (llm) {
      setLlm({ ...llm, harness: { ...llm.harness, ...patch } });
    }
  };

  const totals = telemetry.data?.stats.totals;

  return (
    <div className="flex flex-col gap-4 p-5">
      <div>
        <h1 className="text-xl font-semibold text-ink-100">System &amp; Local AI Config</h1>
        <p className="text-xs text-ink-400">
          ローカル DeepSeek-V4.1-Flash・ComfyUI の接続、4段階思考ハーネスの設定、SQLite の統計とバックアップ{config.data ? `（データ: ${config.data.data_dir}）` : ""}
        </p>
      </div>
      {config.loading && !config.data && <Spinner />}
      <ErrorNote message={config.error} />

      {llm && (
        <Card
          title="ローカル LLM（DeepSeek-V4.1-Flash）"
          subtitle="OpenAI 互換 API（既定 http://localhost:8000/v1）または Ollama（既定 http://localhost:11434/v1）"
          actions={
            <>
              <Button size="sm" loading={testLlm.running} onClick={() => void testLlm.run()}>
                接続テスト
              </Button>
              <Button size="sm" variant="primary" loading={saveLlm.running} onClick={() => void saveLlm.run()}>
                保存
              </Button>
            </>
          }
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            <Field label="プロバイダ">
              <Select value={llm.provider} options={PROVIDERS} onChange={(provider) => setLlm({ ...llm, provider })} ariaLabel="LLM プロバイダ" />
            </Field>
            <Field label="OpenAI 互換エンドポイント URL">
              <TextInput value={llm.base_url} onChange={(base_url) => setLlm({ ...llm, base_url })} type="url" ariaLabel="OpenAI 互換 URL" />
            </Field>
            <Field label="モデル名（OpenAI 互換）">
              <TextInput value={llm.model} onChange={(model) => setLlm({ ...llm, model })} ariaLabel="モデル名" />
            </Field>
            <Field label="Ollama エンドポイント URL">
              <TextInput value={llm.ollama_base_url} onChange={(ollama_base_url) => setLlm({ ...llm, ollama_base_url })} type="url" ariaLabel="Ollama URL" />
            </Field>
            <Field label="モデル名（Ollama）">
              <TextInput value={llm.ollama_model} onChange={(ollama_model) => setLlm({ ...llm, ollama_model })} ariaLabel="Ollama モデル名" />
            </Field>
            <Field label="API キー" hint="マスク表示のまま保存すると既存のキーを保持します">
              <TextInput value={llm.api_key} onChange={(api_key) => setLlm({ ...llm, api_key })} type="password" ariaLabel="API キー" />
            </Field>
            <Field label="構造化出力モード">
              <Select value={llm.structured_output} options={STRUCTURED_MODES} onChange={(structured_output) => setLlm({ ...llm, structured_output })} ariaLabel="構造化出力モード" />
            </Field>
            <Field label="最大トークン数">
              <NumberInput value={llm.max_tokens} onChange={(max_tokens) => setLlm({ ...llm, max_tokens })} min={256} step={256} ariaLabel="最大トークン数" />
            </Field>
            <Field label="タイムアウト（秒）">
              <NumberInput value={llm.timeout_s} onChange={(timeout_s) => setLlm({ ...llm, timeout_s })} min={1} ariaLabel="タイムアウト" />
            </Field>
            <Slider label="既定 temperature" value={llm.temperature} min={0} max={1.5} step={0.05} onChange={(temperature) => setLlm({ ...llm, temperature })} />
          </div>

          <div className="mt-5 rounded-lg border border-ink-700 bg-ink-850 p-4">
            <div className="mb-3 text-xs font-semibold text-ink-200">思考段階設定（Multi-Stage Reasoning Harness）</div>
            <div className="flex flex-wrap gap-5">
              <Toggle checked={llm.harness.enable_decompose} onChange={(enable_decompose) => updateHarness({ enable_decompose })} label="Step 1 Decompose" />
              <span className="flex items-center gap-2 text-xs text-ink-400">
                <Badge tone="accent">Step 2 Synthesize（常時）</Badge>
              </span>
              <Toggle checked={llm.harness.enable_critique} onChange={(enable_critique) => updateHarness({ enable_critique })} label="Step 3 Critique" />
              <Toggle checked={llm.harness.enable_refine} onChange={(enable_refine) => updateHarness({ enable_refine })} label="Step 4 Refine" />
              <Toggle checked={llm.harness.skip_refine_when_accepted} onChange={(skip_refine_when_accepted) => updateHarness({ skip_refine_when_accepted })} label="合格時は Refine を省略" />
            </div>
            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
              <Slider label="批評の合格基準点" value={llm.harness.critique_accept_score} min={0} max={100} step={1} onChange={(critique_accept_score) => updateHarness({ critique_accept_score })} format={(value) => `${value.toFixed(0)} 点`} />
              <Field label="自己修復リトライ回数">
                <NumberInput value={llm.harness.max_repair_attempts} onChange={(max_repair_attempts) => updateHarness({ max_repair_attempts })} min={0} max={10} ariaLabel="自己修復リトライ回数" />
              </Field>
              <Field label="追加精錬ラウンド上限">
                <NumberInput value={llm.harness.max_refine_rounds} onChange={(max_refine_rounds) => updateHarness({ max_refine_rounds })} min={0} max={6} ariaLabel="追加精錬ラウンド上限" />
              </Field>
            </div>
            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-4">
              {STAGE_KEYS.map((stage) => (
                <Slider
                  key={stage}
                  label={`${STAGE_LABELS[stage]} 温度`}
                  value={llm.harness.stage_temperatures[stage] ?? llm.temperature}
                  min={0}
                  max={1.5}
                  step={0.05}
                  onChange={(value) => updateHarness({ stage_temperatures: { ...llm.harness.stage_temperatures, [stage]: value } })}
                />
              ))}
            </div>
          </div>
          <TestResult result={llmTest} />
        </Card>
      )}

      {comfy && (
        <Card
          title="ComfyUI"
          subtitle="API（/prompt・/history・/view・/ws）で非同期キュー投入と進捗監視を行います"
          actions={
            <>
              <Button size="sm" loading={testComfy.running} onClick={() => void testComfy.run()}>
                接続テスト
              </Button>
              <Button size="sm" variant="primary" loading={saveComfy.running} onClick={() => void saveComfy.run()}>
                保存
              </Button>
            </>
          }
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
            <Field label="エンドポイント URL" className="md:col-span-2">
              <TextInput value={comfy.base_url} onChange={(base_url) => setComfy({ ...comfy, base_url })} type="url" ariaLabel="ComfyUI URL" />
            </Field>
            <Field label="タイムアウト（秒）">
              <NumberInput value={comfy.timeout_s} onChange={(timeout_s) => setComfy({ ...comfy, timeout_s })} min={1} ariaLabel="ComfyUI タイムアウト" />
            </Field>
            <Field label="出力ファイル接頭辞">
              <TextInput value={comfy.output_prefix} onChange={(output_prefix) => setComfy({ ...comfy, output_prefix })} ariaLabel="出力ファイル接頭辞" />
            </Field>
          </div>
          <div className="mt-3">
            <Toggle checked={comfy.use_mock} onChange={(use_mock) => setComfy({ ...comfy, use_mock })} label="内蔵モック ComfyUI を使う（実機なしで投入・進捗監視を検証）" />
          </div>
          <TestResult result={comfyTest} />
        </Card>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card
          title="SQLite データベース"
          subtitle={stats.data?.path}
          actions={
            <>
              <Button size="sm" variant="ghost" onClick={() => setDbVersion((value) => value + 1)}>
                更新
              </Button>
              <Button size="sm" variant="primary" loading={backup.running} onClick={() => void backup.run()}>
                バックアップ作成
              </Button>
            </>
          }
        >
          {stats.error && <ErrorNote message={stats.error} />}
          {stats.data && (
            <>
              <div className="grid grid-cols-2 gap-2">
                <Stat label="ファイルサイズ（WAL 含む）" value={bytes(stats.data.size_bytes)} />
                <Stat label="テーブル数" value={Object.keys(stats.data.tables).length} />
              </div>
              <div className="mt-3 max-h-64 overflow-y-auto">
                <table className="w-full text-left text-xs">
                  <tbody>
                    {Object.entries(stats.data.tables).map(([table, rows]) => (
                      <tr key={table} className="border-t border-ink-800 text-ink-200">
                        <td className="py-1 pr-2 font-mono">{table}</td>
                        <td className="py-1 text-right font-mono">{rows.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          <div className="mt-4">
            <div className="mb-1 text-xs font-semibold text-ink-300">バックアップ</div>
            {(backups.data ?? []).length === 0 ? (
              <p className="text-xs text-ink-500">バックアップはまだありません。</p>
            ) : (
              <ul className="flex max-h-40 flex-col gap-1 overflow-y-auto text-[11px] text-ink-300">
                {(backups.data ?? []).map((item, index) => (
                  <li key={`${item.path ?? item.name ?? index}`} className="flex justify-between gap-2 font-mono">
                    <span className="truncate">{item.name ?? item.filename ?? item.path}</span>
                    <span className="shrink-0 text-ink-500">
                      {bytes(item.size_bytes)} {item.created_at ? dateTime(item.created_at) : ""}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Card>

        <Card
          title="初期シード（普遍的構造データベース）"
          subtitle="物語原型・漫画演出リズム・説得/マーケティング構造 各20パターン"
          actions={
            <>
              <Button size="sm" loading={loadSeeds.running} onClick={() => void loadSeeds.run(false)}>
                未投入分を投入
              </Button>
              <Button size="sm" variant="danger" loading={loadSeeds.running} onClick={() => void loadSeeds.run(true)}>
                上書き再投入
              </Button>
            </>
          }
        >
          {seeds.error && <ErrorNote message={seeds.error} />}
          {seeds.data && (
            <div className="flex flex-col gap-2">
              <Stat label="シード構造数" value={seeds.data.total} />
              {Object.entries(seeds.data.by_category).map(([category, count]) => (
                <div key={category} className="flex items-center justify-between rounded-lg border border-ink-700 bg-ink-850 px-3 py-1.5 text-xs">
                  <span className="text-ink-200">{labelOf(CATEGORY_LABELS, category)}</span>
                  <span className="font-mono text-ink-100">{count}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <Card
        title="推論テレメトリ"
        subtitle="推論トークン数・レイテンシ・成功率・ハーネス実行履歴（SQLite に自動記録）"
        actions={
          <Button size="sm" variant="ghost" onClick={() => setTelemetryVersion((value) => value + 1)} loading={telemetry.loading}>
            更新
          </Button>
        }
      >
        {telemetry.error && <ErrorNote message={telemetry.error} />}
        {telemetry.data && totals && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-2 md:grid-cols-6">
              <Stat label="呼び出し" value={totals.calls} />
              <Stat label="成功率" value={pct(totals.success_rate ?? 0, 1)} />
              <Stat label="入力トークン" value={(totals.prompt_tokens ?? 0).toLocaleString()} />
              <Stat label="出力トークン" value={(totals.completion_tokens ?? 0).toLocaleString()} />
              <Stat label="思考トークン" value={(totals.reasoning_tokens ?? 0).toLocaleString()} />
              <Stat label="平均レイテンシ" value={`${fmt(totals.avg_latency_ms ?? 0, 0)} ms`} />
            </div>
            {telemetry.data.calls.length === 0 ? (
              <EmptyState title="まだ LLM 呼び出しはありません" hint="構造抽出や記事生成を実行すると記録されます。" />
            ) : (
              <div className="max-h-80 overflow-auto">
                <table className="w-full text-left text-[11px]">
                  <thead className="sticky top-0 bg-ink-900 text-ink-400">
                    <tr>
                      <th className="py-1 pr-2">時刻</th>
                      <th className="py-1 pr-2">タスク</th>
                      <th className="py-1 pr-2">段階</th>
                      <th className="py-1 pr-2">プロバイダ</th>
                      <th className="py-1 pr-2">試行</th>
                      <th className="py-1 pr-2">結果</th>
                      <th className="py-1 pr-2">入力/出力/思考</th>
                      <th className="py-1">レイテンシ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {telemetry.data.calls.map((call, index) => (
                      <tr key={call.id ?? index} className="border-t border-ink-800 text-ink-200">
                        <td className="py-1 pr-2 font-mono">{dateTime(call.created_at)}</td>
                        <td className="py-1 pr-2 font-mono">{call.task}</td>
                        <td className="py-1 pr-2">{call.stage}</td>
                        <td className="py-1 pr-2">{call.provider ?? "-"}</td>
                        <td className="py-1 pr-2 font-mono">{call.attempt ?? 1}</td>
                        <td className="py-1 pr-2">
                          <Badge tone={call.success ? "success" : "danger"}>{call.success ? "成功" : "失敗"}</Badge>
                        </td>
                        <td className="py-1 pr-2 font-mono">
                          {call.prompt_tokens ?? 0}/{call.completion_tokens ?? 0}/{call.reasoning_tokens ?? 0}
                        </td>
                        <td className="py-1 font-mono">{fmt(call.latency_ms ?? 0, 0)} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {telemetry.data.runs.length > 0 && (
              <div>
                <div className="mb-1 text-xs font-semibold text-ink-300">ハーネス実行履歴</div>
                <div className="max-h-60 overflow-auto">
                  <table className="w-full text-left text-[11px]">
                    <thead className="text-ink-400">
                      <tr>
                        <th className="py-1 pr-2">開始</th>
                        <th className="py-1 pr-2">タスク</th>
                        <th className="py-1 pr-2">状態</th>
                        <th className="py-1 pr-2">批評スコア</th>
                        <th className="py-1">合計レイテンシ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {telemetry.data.runs.map((run) => (
                        <tr key={run.run_id} className="border-t border-ink-800 text-ink-200">
                          <td className="py-1 pr-2 font-mono">{dateTime(run.started_at)}</td>
                          <td className="py-1 pr-2 font-mono">{run.task}</td>
                          <td className="py-1 pr-2">{run.status}</td>
                          <td className="py-1 pr-2 font-mono">{run.critique_score === null || run.critique_score === undefined ? "-" : run.critique_score.toFixed(1)}</td>
                          <td className="py-1 font-mono">{fmt(run.total_latency_ms ?? 0, 0)} ms</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {telemetry.data.stats.by_task_stage && telemetry.data.stats.by_task_stage.length > 0 && (
              <JsonViewer data={telemetry.data.stats.by_task_stage} title="タスク×段階別統計" maxHeight={240} />
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
