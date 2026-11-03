// ベイズ最適化（GP + EI/UCB/PI、CMA-ES、GA）による次レシピ提案とレシピ／ComfyUI ワークフロー出力。

import { useState } from "react";
import { api, downloadJson, downloadText } from "../../api/client";
import type { BlockGrouping, ModelArchitecture, OptimizationSuggestion, ParameterDim, RecipeExportResponse, SuggestResponse } from "../../api/types";
import { JsonViewer } from "../../components/JsonViewer";
import { ParamBars } from "../../components/charts/ParamBars";
import { Badge, Button, Card, CopyButton, EmptyState, Field, NumberInput, ScoreBar, Select, Stat, TextInput } from "../../components/ui";
import { useAction } from "../../hooks/useAsync";
import { fmt, parseList } from "../../lib/format";
import { ARCHITECTURE_OPTIONS, GROUPING_OPTIONS } from "./ArenaPanel";

type Strategy = "hybrid" | "gp" | "cmaes" | "genetic";
type Acquisition = "ei" | "ucb" | "pi";

const STRATEGIES: Array<{ value: Strategy; label: string }> = [
  { value: "hybrid", label: "ハイブリッド（全戦略統合）" },
  { value: "gp", label: "ガウス過程＋獲得関数" },
  { value: "cmaes", label: "CMA-ES（GP 代理）" },
  { value: "genetic", label: "遺伝的アルゴリズム（GP 代理）" },
];

const ACQUISITIONS: Array<{ value: Acquisition; label: string }> = [
  { value: "ei", label: "Expected Improvement" },
  { value: "ucb", label: "Upper Confidence Bound" },
  { value: "pi", label: "Probability of Improvement" },
];

export function OptimizerPanel({ onCandidatesChanged }: { onCandidatesChanged: () => void }) {
  const [architecture, setArchitecture] = useState<ModelArchitecture>("sdxl");
  const [loraNames, setLoraNames] = useState("ink_style");
  const [grouping, setGrouping] = useState<BlockGrouping>("grouped");
  const [strategy, setStrategy] = useState<Strategy>("hybrid");
  const [acquisition, setAcquisition] = useState<Acquisition>("ei");
  const [n, setN] = useState(3);
  const [modelA, setModelA] = useState("base_model_a.safetensors");
  const [modelB, setModelB] = useState("base_model_b.safetensors");
  const [prompt, setPrompt] = useState("1girl, standing, city street, manga panel");
  const [response, setResponse] = useState<SuggestResponse | null>(null);
  const [dims, setDims] = useState<ParameterDim[]>([]);
  const [exported, setExported] = useState<{ suggestion: OptimizationSuggestion; result: RecipeExportResponse } | null>(null);

  const suggest = useAction(
    async () => {
      const loras = parseList(loraNames);
      const result = await api.style.suggest({ n, strategy, acquisition, architecture, lora_names: loras, grouping });
      const space = result.space && Array.isArray(result.space.dims) ? result.space : await api.style.space(architecture, loras, grouping);
      setResponse(result);
      setDims(space.dims);
      setExported(null);
      return result;
    },
    { successMessage: (r) => `${r.suggestions.length} 件の次レシピを提案しました` },
  );

  const register = useAction(
    async (suggestion: OptimizationSuggestion) => {
      const created = await api.style.createCandidate({ label: `提案#${suggestion.rank}（${suggestion.strategy}）`, params: suggestion.params });
      onCandidatesChanged();
      return created;
    },
    { successMessage: (c) => `候補「${c.label}」を登録しました` },
  );

  const registerAll = useAction(
    async () => {
      const suggestions = response?.suggestions ?? [];
      for (const suggestion of suggestions) {
        await api.style.createCandidate({ label: `提案#${suggestion.rank}（${suggestion.strategy}）`, params: suggestion.params });
      }
      onCandidatesChanged();
      return suggestions.length;
    },
    { successMessage: (count) => `${count} 件の提案を候補として登録しました` },
  );

  const exportRecipe = useAction(
    async (suggestion: OptimizationSuggestion) => {
      const result = await api.style.exportRecipe({
        params: suggestion.params,
        architecture,
        model_a: { name: modelA, architecture },
        model_b: { name: modelB, architecture },
        loras: parseList(loraNames).map((name) => ({ name: name.includes(".") ? name : `${name}.safetensors` })),
        name: `提案#${suggestion.rank} レシピ`,
        prompt,
        negative: "lowres, blurry, bad anatomy",
        seed: 1000 + suggestion.rank,
        width: 1024,
        height: 1024,
      });
      setExported({ suggestion, result });
      return result;
    },
    { successMessage: "レシピとワークフローを出力しました" },
  );

  const observationCount = typeof response?.observations === "number" ? response.observations : response?.observations.length ?? 0;

  return (
    <div className="flex flex-col gap-4">
      <Card title="ベイズ最適化による次レシピ提案" subtitle="判定から推定した Bradley-Terry 強度（Elo 換算）を目的関数とし、最小の試行回数で理想の絵柄へ到達する次の配合を算出します">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Field label="アーキテクチャ">
            <Select value={architecture} options={ARCHITECTURE_OPTIONS} onChange={setArchitecture} ariaLabel="最適化アーキテクチャ" />
          </Field>
          <Field label="LoRA 名">
            <TextInput value={loraNames} onChange={setLoraNames} ariaLabel="最適化 LoRA 名" />
          </Field>
          <Field label="ブロック粒度">
            <Select value={grouping} options={GROUPING_OPTIONS} onChange={setGrouping} ariaLabel="最適化ブロック粒度" />
          </Field>
          <Field label="戦略">
            <Select value={strategy} options={STRATEGIES} onChange={setStrategy} ariaLabel="最適化戦略" />
          </Field>
          <Field label="獲得関数">
            <Select value={acquisition} options={ACQUISITIONS} onChange={setAcquisition} ariaLabel="獲得関数" />
          </Field>
          <Field label="提案数">
            <NumberInput value={n} onChange={setN} min={1} max={12} ariaLabel="提案数" />
          </Field>
        </div>
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
          <Field label="マージ元モデル A">
            <TextInput value={modelA} onChange={setModelA} ariaLabel="モデル A" />
          </Field>
          <Field label="マージ先モデル B">
            <TextInput value={modelB} onChange={setModelB} ariaLabel="モデル B" />
          </Field>
          <Field label="検証用プロンプト">
            <TextInput value={prompt} onChange={setPrompt} ariaLabel="検証用プロンプト" />
          </Field>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <Button variant="primary" loading={suggest.running} onClick={() => void suggest.run()}>
            次のレシピを提案
          </Button>
          {response && response.suggestions.length > 0 && (
            <Button loading={registerAll.running} onClick={() => void registerAll.run()}>
              全提案を候補登録
            </Button>
          )}
        </div>
      </Card>

      {response && (
        <Card title="提案レシピ" subtitle={`観測 ${observationCount} 件に基づく提案`}>
          {response.best && (
            <div className="mb-3 rounded-lg border border-emerald-500/30 bg-emerald-950/30 px-3 py-2 text-xs text-emerald-100">
              現在の最良観測: スコア {fmt(response.best.score, 1)}
              {response.best.candidate_id ? `（候補 ${response.best.candidate_id}）` : ""}
            </div>
          )}
          {response.sensitivity && Object.keys(response.sensitivity).length > 0 && (
            <div className="mb-3 rounded-lg border border-ink-700 bg-ink-850 p-3">
              <div className="mb-2 text-xs font-semibold text-ink-300">次元感度（ガウス過程の ARD 長さスケールの逆数・正規化）</div>
              <div className="grid grid-cols-1 gap-1 md:grid-cols-2">
                {Object.entries(response.sensitivity)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 10)
                  .map(([name, value]) => (
                    <ScoreBar key={name} label={name} value={value} max={Math.max(1e-6, ...Object.values(response.sensitivity ?? {}))} color="#a78bfa" />
                  ))}
              </div>
            </div>
          )}
          {response.suggestions.length === 0 && <EmptyState title="提案がありません" />}
          <div className="flex flex-col gap-3">
            {response.suggestions.map((suggestion) => (
              <div key={suggestion.rank} className="rounded-lg border border-ink-700 bg-ink-850 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="accent">#{suggestion.rank}</Badge>
                  <Badge>{suggestion.strategy}</Badge>
                  <span className="font-mono text-xs text-ink-300">
                    予測 {fmt(suggestion.predicted_mean, 1)} ± {fmt(suggestion.predicted_std, 1)} ／ 獲得値 {fmt(suggestion.acquisition, 4)}
                  </span>
                  <span className="ml-auto flex gap-2">
                    <Button size="sm" loading={register.running} onClick={() => void register.run(suggestion)}>
                      候補として登録
                    </Button>
                    <Button size="sm" variant="primary" loading={exportRecipe.running} onClick={() => void exportRecipe.run(suggestion)}>
                      レシピ／ComfyUI 出力
                    </Button>
                  </span>
                </div>
                <p className="mt-2 text-xs leading-relaxed text-ink-300">{suggestion.rationale}</p>
                <div className="mt-3">
                  <ParamBars params={suggestion.params} dims={dims} />
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {exported && (
        <Card
          title={`出力: ${exported.result.recipe.name}`}
          subtitle="マージ実行スクリプト用設定・ローカル実行用マージスクリプト・ComfyUI API ワークフロー"
          actions={
            <>
              <Button size="sm" onClick={() => downloadJson(`${exported.result.recipe.id}_recipe.json`, exported.result.recipe)}>
                レシピ JSON
              </Button>
              <Button size="sm" onClick={() => downloadJson(`${exported.result.recipe.id}_merge_config.json`, exported.result.merge_config)}>
                マージ設定
              </Button>
              <Button size="sm" onClick={() => downloadText(`${exported.result.recipe.id}_merge.py`, exported.result.merge_script, "text/x-python;charset=utf-8")}>
                マージスクリプト
              </Button>
              <Button size="sm" variant="primary" onClick={() => downloadJson(`${exported.result.recipe.id}_comfyui_workflow.json`, exported.result.comfy_workflow)}>
                ComfyUI ワークフロー
              </Button>
            </>
          }
        >
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <Stat label="アーキテクチャ" value={exported.result.recipe.architecture} />
            <Stat label="LoRA" value={exported.result.recipe.loras.length} />
            <Stat label="ComfyUI ノード数" value={Object.keys(exported.result.comfy_workflow).length} />
            <Stat label="スタイルトークン" value={exported.result.recipe.style_tokens.length} />
          </div>
          <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
            <JsonViewer data={exported.result.merge_config} title="マージ設定" maxHeight={300} />
            <JsonViewer data={exported.result.comfy_workflow} title="ComfyUI API ワークフロー" maxHeight={300} />
          </div>
          <div className="mt-3 overflow-hidden rounded-lg border border-ink-700 bg-ink-950">
            <div className="flex items-center justify-between border-b border-ink-700 px-3 py-1.5 text-xs text-ink-400">
              マージスクリプト（ローカル実行用 Python）
              <CopyButton text={exported.result.merge_script} />
            </div>
            <pre className="max-h-72 overflow-auto p-3 font-mono text-[11px] text-ink-200">{exported.result.merge_script}</pre>
          </div>
        </Card>
      )}
    </div>
  );
}
