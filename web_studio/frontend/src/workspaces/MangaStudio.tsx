// Workspace 3: Manga Studio — 構造駆動ネーム生成・タイムライン・SVG プレビュー・漫画文法監査・ComfyUI ワークフロー。

import { useEffect, useMemo, useState } from "react";
import { api, downloadJson, errorMessage } from "../api/client";
import type { CompiledPanelWorkflow, GrammarViolation, Job, ModelArchitecture, Storyboard, StoryboardSummary } from "../api/types";
import { JobProgress } from "../components/JobProgress";
import { JsonViewer } from "../components/JsonViewer";
import { MangaTimeline } from "../components/charts/MangaTimeline";
import { ScoreGauge } from "../components/charts/ScoreGauge";
import { Badge, Button, Card, EmptyState, ErrorNote, Field, KeyValueEditor, NumberInput, Select, Slider, Stat, Tabs, TextInput, Toggle, cx } from "../components/ui";
import { useAction, useResource } from "../hooks/useAsync";
import { useStructureOptions } from "../hooks/useStructures";
import { parseList } from "../lib/format";
import { ANGLE_LABELS, ARCHITECTURE_LABELS, BALLOON_LABELS, CATEGORY_LABELS, SEVERITY_TONES, SHOT_LABELS, labelOf } from "../lib/labels";

type PreviewMode = "single" | "spread";

const ARCHITECTURES: Array<{ value: ModelArchitecture; label: string }> = (["sdxl", "sd15", "flux", "anima"] as ModelArchitecture[]).map((value) => ({
  value,
  label: ARCHITECTURE_LABELS[value],
}));

function spreadPair(storyboard: Storyboard, index: number): { right: number; left: number } | null {
  const page = storyboard.pages[index];
  if (!page) {
    return null;
  }
  if (page.side === "right") {
    const next = storyboard.pages[index + 1];
    return next && next.side === "left" ? { right: index, left: index + 1 } : null;
  }
  const previous = storyboard.pages[index - 1];
  return previous && previous.side === "right" ? { right: index - 1, left: index } : null;
}

function ViolationItem({ violation, onJump }: { violation: GrammarViolation; onJump: (page: number) => void }) {
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={SEVERITY_TONES[violation.severity] ?? "neutral"}>{violation.severity}</Badge>
        <span className="font-mono text-ink-300">{violation.rule_id}</span>
        <span className="text-ink-500">−{violation.penalty.toFixed(1)}点</span>
        {violation.page_index !== null && (
          <Button size="sm" variant="ghost" onClick={() => onJump(violation.page_index ?? 0)}>
            {violation.page_index + 1}ページへ
          </Button>
        )}
      </div>
      <p className="mt-1 text-ink-200">{violation.message}</p>
      <p className="mt-0.5 text-ink-400">修正指示: {violation.fix_instruction}</p>
    </div>
  );
}

export function MangaStudio({ dataVersion }: { dataVersion: number }) {
  const structures = useStructureOptions(dataVersion);
  const [structureId, setStructureId] = useState("");
  const [title, setTitle] = useState("");
  const [theme, setTheme] = useState<Array<[string, string]>>([["THEME", ""]]);
  const [targetPages, setTargetPages] = useState(16);
  const [seed, setSeed] = useState(0);
  const [useLlmDialogue, setUseLlmDialogue] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [listVersion, setListVersion] = useState(0);
  const storyboards = useResource<StoryboardSummary[]>(() => api.manga.list(), [listVersion]);

  const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
  const [page, setPage] = useState(0);
  const [mode, setMode] = useState<PreviewMode>("single");
  const [gaze, setGaze] = useState(true);
  const [labels, setLabels] = useState(true);
  const [guides, setGuides] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [imageError, setImageError] = useState<string | null>(null);

  const [architecture, setArchitecture] = useState<ModelArchitecture>("sdxl");
  const [checkpoint, setCheckpoint] = useState("sd_xl_base_1.0.safetensors");
  const [loraNames, setLoraNames] = useState("");
  const [workflows, setWorkflows] = useState<CompiledPanelWorkflow[]>([]);
  const [viewWorkflow, setViewWorkflow] = useState<CompiledPanelWorkflow | null>(null);
  const [queueJob, setQueueJob] = useState<Job | null>(null);
  const [queueResult, setQueueResult] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!structureId && structures.items.length > 0) {
      setStructureId(structures.items[0].id);
    }
  }, [structures.items, structureId]);

  useEffect(() => {
    setImageError(null);
  }, [storyboard?.id, page, mode, gaze, labels, guides]);

  const structureOptions = useMemo(
    () => structures.items.map((item) => ({ value: item.id, label: `［${labelOf(CATEGORY_LABELS, item.category)}］${item.title}` })),
    [structures.items],
  );

  const recipe = useMemo(
    () => ({
      name: "Manga Studio レシピ",
      architecture,
      base_checkpoint: { name: checkpoint, architecture },
      loras: parseList(loraNames).map((name) => ({ name: name.includes(".") ? name : `${name}.safetensors`, strength_model: 0.8, strength_clip: 0.8 })),
      style_tokens: ["monochrome", "manga", "lineart", "screentone"],
      negative_tokens: ["color", "lowres", "blurry"],
    }),
    [architecture, checkpoint, loraNames],
  );

  const loadStoryboard = (next: Storyboard) => {
    setStoryboard(next);
    setPage(0);
    setWorkflows([]);
    setViewWorkflow(null);
    setQueueResult(null);
  };

  const generate = useAction(
    async () => {
      const themeRecord: Record<string, string> = {};
      for (const [key, value] of theme) {
        if (key.trim() && value.trim()) {
          themeRecord[key.trim()] = value.trim();
        }
      }
      const created = await api.manga.create(
        {
          structure_id: structureId,
          title: title.trim() || undefined,
          theme: themeRecord,
          config: { target_pages: targetPages, seed, use_llm_dialogue: useLlmDialogue },
        },
        setJob,
      );
      loadStoryboard(created);
      setListVersion((value) => value + 1);
      return created;
    },
    { successMessage: (s) => `ネームを生成しました（${s.pages.length}ページ・文法スコア ${s.grammar?.score.toFixed(1) ?? "-"}）` },
  );

  const open = useAction(async (id: string) => {
    const loaded = await api.manga.get(id);
    loadStoryboard(loaded);
    return loaded;
  });

  const validate = useAction(
    async () => {
      if (!storyboard) {
        return null;
      }
      const report = await api.manga.validate(storyboard.id);
      setStoryboard({ ...storyboard, grammar: report });
      return report;
    },
    { successMessage: (r) => (r ? `再検証しました（${r.score.toFixed(1)}点）` : "再検証しました") },
  );

  const autofix = useAction(
    async () => {
      if (!storyboard) {
        return null;
      }
      const fixed = await api.manga.autofix(storyboard.id);
      setStoryboard(fixed);
      setListVersion((value) => value + 1);
      return fixed;
    },
    { successMessage: "自動修正して再検証しました" },
  );

  const remove = useAction(
    async () => {
      if (!storyboard) {
        return;
      }
      await api.manga.remove(storyboard.id);
      setStoryboard(null);
      setListVersion((value) => value + 1);
    },
    { successMessage: "ネームを削除しました" },
  );

  const exportLayers = useAction(async () => {
    if (!storyboard) {
      return;
    }
    const layers = await api.manga.layers(storyboard.id);
    downloadJson(`${storyboard.id}_layers.json`, layers);
  });

  const compile = useAction(
    async () => {
      if (!storyboard) {
        return [];
      }
      const response = await api.manga.compileComfy(storyboard.id, { architecture, recipe });
      setWorkflows(response.workflows);
      setViewWorkflow(response.workflows[0] ?? null);
      return response.workflows;
    },
    { successMessage: (list) => `${list.length} コマ分の ComfyUI ワークフローを生成しました` },
  );

  const queue = useAction(
    async () => {
      if (!storyboard) {
        return null;
      }
      setQueueResult(null);
      const panelIds = storyboard.pages[page]?.panels.map((panel) => panel.id) ?? [];
      const result = await api.manga.queueComfy(storyboard.id, { architecture, recipe, panel_ids: panelIds }, setQueueJob);
      setQueueResult(result);
      return result;
    },
    { successMessage: "ComfyUI への投入が完了しました" },
  );

  const pair = storyboard ? spreadPair(storyboard, page) : null;
  const showSpread = mode === "spread" && pair !== null;
  const previewOptions = { gaze, labels, guides };
  const currentPage = storyboard?.pages[page] ?? null;
  const grammar = storyboard?.grammar ?? null;
  const totalPanels = storyboard?.pages.reduce((sum, item) => sum + item.panels.length, 0) ?? 0;

  return (
    <div className="flex flex-col gap-4 p-5">
      <div>
        <h1 className="text-xl font-semibold text-ink-100">Manga Studio</h1>
        <p className="text-xs text-ink-400">構造の感情曲線・情報開示曲線・ビートから、ページ割り→コマ割り→カメラ→セリフを展開し、漫画文法を監査して ComfyUI へ</p>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Card title="ネーム生成" subtitle="右ページ開始・左ページ終端の見開き慣行で、転換点をめくり直後へ配置します">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <Field label="構造">
              <Select value={structureId} options={structureOptions.length > 0 ? structureOptions : [{ value: "", label: "構造がありません" }]} onChange={setStructureId} ariaLabel="ネームの構造" />
            </Field>
            <Field label="タイトル（任意）">
              <TextInput value={title} onChange={setTitle} placeholder="未指定なら構造名から生成" ariaLabel="ネームタイトル" />
            </Field>
            <Slider label="目標ページ数" value={targetPages} min={4} max={48} step={1} onChange={setTargetPages} format={(value) => `${value} ページ`} />
            <Field label="シード">
              <NumberInput value={seed} onChange={setSeed} min={0} ariaLabel="ネームシード" />
            </Field>
            <Field label="テーマ変数（台詞・プロンプトの語彙）" className="md:col-span-2">
              <KeyValueEditor entries={theme} onChange={setTheme} keyPlaceholder="変数名" valuePlaceholder="具体的なテーマ" />
            </Field>
            <Toggle checked={useLlmDialogue} onChange={setUseLlmDialogue} label="DeepSeek で台詞を増幅する（manga.dialogue）" />
          </div>
          <div className="mt-3 flex items-center gap-3">
            <Button variant="primary" loading={generate.running} disabled={!structureId} onClick={() => void generate.run()}>
              ネームを生成（文法監査ループ込み）
            </Button>
          </div>
          {generate.running && (
            <div className="mt-3">
              <JobProgress job={job} />
            </div>
          )}
        </Card>
        <Card title="保存済みネーム" actions={<Button size="sm" variant="ghost" onClick={storyboards.reload}>更新</Button>}>
          <div className="flex max-h-80 flex-col gap-1.5 overflow-y-auto">
            {(storyboards.data ?? []).length === 0 && <EmptyState title="まだネームがありません" />}
            {(storyboards.data ?? []).map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => void open.run(item.id)}
                className={cx("rounded-lg border px-3 py-2 text-left text-xs", storyboard?.id === item.id ? "border-accent bg-accent-soft/40" : "border-ink-700 bg-ink-850 hover:border-ink-500")}
              >
                <div className="truncate font-medium text-ink-100">{item.title}</div>
                <div className="mt-0.5 flex gap-2 text-ink-400">
                  <span>{item.page_count}ページ</span>
                  {item.score !== null && <span>文法 {item.score.toFixed(1)}</span>}
                  {item.passed !== null && <Badge tone={item.passed ? "success" : "danger"}>{item.passed ? "合格" : "要修正"}</Badge>}
                </div>
              </button>
            ))}
          </div>
        </Card>
      </div>

      {!storyboard && <EmptyState title="ネームを生成するか、保存済みネームを開いてください" />}

      {storyboard && (
        <>
          <Card
            title={storyboard.title}
            subtitle={<span className="font-mono">{storyboard.id}</span>}
            actions={
              <>
                <Button size="sm" loading={validate.running} onClick={() => void validate.run()}>
                  再検証
                </Button>
                <Button size="sm" variant="primary" loading={autofix.running} onClick={() => void autofix.run()}>
                  自動修正
                </Button>
                <Button size="sm" loading={exportLayers.running} onClick={() => void exportLayers.run()}>
                  レイヤー JSON
                </Button>
                <Button size="sm" onClick={() => downloadJson(`${storyboard.id}.json`, storyboard)}>
                  ネーム JSON
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  loading={remove.running}
                  onClick={() => {
                    if (window.confirm("このネームを削除しますか？")) {
                      void remove.run();
                    }
                  }}
                >
                  削除
                </Button>
              </>
            }
          >
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-[180px_minmax(0,1fr)]">
              <div className="flex flex-col items-center">
                <ScoreGauge score={grammar?.score ?? 0} threshold={80} label="漫画文法スコア" />
                {grammar && <Badge tone={grammar.passed ? "success" : "danger"}>{grammar.passed ? "文法監査 合格" : "文法監査 不合格"}</Badge>}
              </div>
              <div className="flex flex-col gap-3">
                <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
                  <Stat label="ページ" value={storyboard.pages.length} />
                  <Stat label="コマ" value={totalPanels} />
                  <Stat label="見開き" value={storyboard.spreads.length} />
                  <Stat label="登場人物" value={storyboard.characters.length} />
                  <Stat label="違反" value={grammar?.violations.length ?? 0} />
                </div>
                <div className="flex flex-wrap gap-1.5 text-xs">
                  {storyboard.characters.map((character) => (
                    <Badge key={character.id} tone="info" title={character.visual_description}>
                      {character.name_placeholder}（{character.role}）
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
            <div className="mt-4">
              <MangaTimeline storyboard={storyboard} selectedPage={page} onSelectPage={setPage} />
            </div>
          </Card>

          <div className="grid grid-cols-1 gap-4 2xl:grid-cols-[minmax(0,1fr)_420px]">
            <Card
              title="SVG ネームプレビュー"
              subtitle={currentPage ? `${currentPage.number}ページ（${currentPage.side === "right" ? "右" : "左"}）・${currentPage.panels.length}コマ・役割 ${currentPage.page_turn_role}` : undefined}
              actions={
                <>
                  <Tabs
                    tabs={[
                      { key: "single", label: "単ページ" },
                      { key: "spread", label: "見開き" },
                    ]}
                    active={mode}
                    onChange={setMode}
                    size="sm"
                  />
                  <Button size="sm" variant="ghost" disabled={page <= 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>
                    ← 前
                  </Button>
                  <Button size="sm" variant="ghost" disabled={page >= storyboard.pages.length - 1} onClick={() => setPage((value) => Math.min(storyboard.pages.length - 1, value + 1))}>
                    次 →
                  </Button>
                </>
              }
            >
              <div className="mb-3 flex flex-wrap items-center gap-4">
                <Toggle checked={gaze} onChange={setGaze} label="視線誘導ベクトル" />
                <Toggle checked={labels} onChange={setLabels} label="コマ番号・ショット表記" />
                <Toggle checked={guides} onChange={setGuides} label="断ち切り・仕上がりガイド" />
                <div className="w-56">
                  <Slider label="ズーム" value={zoom} min={0.4} max={2} step={0.05} onChange={setZoom} format={(value) => `${Math.round(value * 100)}%`} />
                </div>
              </div>
              {mode === "spread" && pair === null && <p className="mb-2 text-xs text-amber-300">このページは見開きの組を持たないため単ページで表示しています。</p>}
              <ErrorNote message={imageError} />
              <div className="overflow-auto rounded-lg border border-ink-700 bg-ink-800 p-3">
                <img
                  key={`${storyboard.id}-${page}-${mode}-${gaze}-${labels}-${guides}`}
                  src={showSpread && pair ? api.manga.spreadSvgUrl(storyboard.id, pair.right, pair.left, previewOptions) : api.manga.pageSvgUrl(storyboard.id, page, previewOptions)}
                  alt={`${page + 1}ページのネーム`}
                  style={{ width: `${(showSpread ? 900 : 520) * zoom}px`, maxWidth: "none" }}
                  className="mx-auto block bg-white"
                  onError={() => setImageError("SVG を取得できませんでした（ネームが削除されたか、バックエンドが応答していません）")}
                />
              </div>
            </Card>

            <div className="flex flex-col gap-4">
              <Card title="漫画文法アラート" subtitle="単調な画面・ピークの強調不足・フキダシ過多・視線の逆行/交差・緩急不足を検査">
                {!grammar && <EmptyState title="未検証です" action={<Button size="sm" onClick={() => void validate.run()}>検証する</Button>} />}
                {grammar && grammar.violations.length === 0 && <p className="text-xs text-emerald-300">違反はありません。</p>}
                <div className="flex max-h-[420px] flex-col gap-2 overflow-y-auto">
                  {grammar?.violations.map((violation, index) => (
                    <ViolationItem key={`${violation.rule_id}-${index}`} violation={violation} onJump={setPage} />
                  ))}
                </div>
                {grammar && Object.keys(grammar.metrics).length > 0 && (
                  <div className="mt-3 grid grid-cols-2 gap-1 text-[11px] text-ink-400">
                    {Object.entries(grammar.metrics)
                      .slice(0, 12)
                      .map(([key, value]) => (
                        <span key={key} className="truncate font-mono">
                          {key}: {typeof value === "number" ? value.toFixed(3) : String(value)}
                        </span>
                      ))}
                  </div>
                )}
              </Card>
              {currentPage && (
                <Card title={`${currentPage.number}ページのコマ構成`}>
                  <div className="flex max-h-[360px] flex-col gap-2 overflow-y-auto">
                    {currentPage.panels.map((panel) => (
                      <div key={panel.id} className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-[11px] text-ink-300">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <Badge tone="accent">#{panel.index_in_page + 1}</Badge>
                          <Badge>{labelOf(SHOT_LABELS, panel.camera.shot)}</Badge>
                          <Badge>{labelOf(ANGLE_LABELS, panel.camera.angle)}</Badge>
                          <Badge>{panel.camera.lens_mm}mm</Badge>
                          {panel.emphasis !== "normal" && <Badge tone="warning">{panel.emphasis}</Badge>}
                          {panel.is_silent && <Badge tone="info">無音</Badge>}
                          {panel.bleed_sides.length > 0 && <Badge tone="info">断ち切り</Badge>}
                          {panel.breakout && <Badge tone="info">ぶち抜き</Badge>}
                          <span className="ml-auto font-mono text-ink-500">緊張 {panel.tension.toFixed(2)}</span>
                        </div>
                        {panel.description && <p className="mt-1 text-ink-300">{panel.description}</p>}
                        {panel.balloons.map((balloon) => (
                          <p key={balloon.id} className="mt-1 text-ink-100">
                            <span className="text-ink-500">［{labelOf(BALLOON_LABELS, balloon.kind)}］</span>
                            {balloon.text}
                          </p>
                        ))}
                        {panel.sfx.length > 0 && <p className="mt-1 text-amber-200">描き文字: {panel.sfx.map((sfx) => sfx.text).join("、")}</p>}
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </div>
          </div>

          <Card
            title="ComfyUI ワークフロー"
            subtitle="各コマの演出パラメータとレシピから API 形式ノードグラフをコンパイル（複数キャラは領域マスク、シードはコマ単位で固定）"
            actions={
              workflows.length > 0 ? (
                <Button size="sm" onClick={() => downloadJson(`${storyboard.id}_comfyui_workflows.json`, Object.fromEntries(workflows.map((workflow) => [workflow.panel_id, workflow.prompt])))}>
                  一括ダウンロード
                </Button>
              ) : undefined
            }
          >
            <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
              <Field label="アーキテクチャ">
                <Select value={architecture} options={ARCHITECTURES} onChange={setArchitecture} ariaLabel="ComfyUI アーキテクチャ" />
              </Field>
              <Field label="チェックポイント">
                <TextInput value={checkpoint} onChange={setCheckpoint} ariaLabel="チェックポイント" />
              </Field>
              <Field label="LoRA（カンマ区切り）">
                <TextInput value={loraNames} onChange={setLoraNames} placeholder="例: ink_style" ariaLabel="LoRA" />
              </Field>
              <div className="flex items-end gap-2">
                <Button variant="primary" loading={compile.running} onClick={() => void compile.run()}>
                  ワークフロー生成
                </Button>
                <Button loading={queue.running} onClick={() => void queue.run()}>
                  表示中ページを投入
                </Button>
              </div>
            </div>
            {queue.running && (
              <div className="mt-3">
                <JobProgress job={queueJob} />
              </div>
            )}
            {queueResult && (
              <div className="mt-3">
                <JsonViewer data={queueResult} title="ComfyUI 実行結果" maxHeight={220} />
              </div>
            )}
            {workflows.length > 0 && (
              <div className="mt-4 grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                <div className="max-h-[480px] overflow-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-ink-900 text-ink-400">
                      <tr>
                        <th className="py-1 pr-2">コマ</th>
                        <th className="py-1 pr-2">ページ</th>
                        <th className="py-1 pr-2">解像度</th>
                        <th className="py-1 pr-2">シード</th>
                        <th className="py-1 pr-2">領域</th>
                        <th className="py-1" />
                      </tr>
                    </thead>
                    <tbody>
                      {workflows.map((workflow) => (
                        <tr key={workflow.panel_id} className={cx("border-t border-ink-800 text-ink-200", viewWorkflow?.panel_id === workflow.panel_id && "bg-accent-soft/30")}>
                          <td className="py-1 pr-2 font-mono">{workflow.panel_id}</td>
                          <td className="py-1 pr-2">{workflow.page_index + 1}</td>
                          <td className="py-1 pr-2 font-mono">
                            {workflow.width}×{workflow.height}
                          </td>
                          <td className="py-1 pr-2 font-mono">{workflow.seed}</td>
                          <td className="py-1 pr-2">{workflow.regions.length}</td>
                          <td className="py-1">
                            <div className="flex gap-1">
                              <Button size="sm" variant="ghost" onClick={() => setViewWorkflow(workflow)}>
                                表示
                              </Button>
                              <Button size="sm" variant="ghost" onClick={() => downloadJson(`${workflow.panel_id}_workflow.json`, workflow.prompt)}>
                                保存
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {viewWorkflow && (
                  <div className="flex flex-col gap-2">
                    <div className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-[11px] text-ink-300">
                      <div>
                        <span className="text-ink-500">Positive:</span> {viewWorkflow.positive_text}
                      </div>
                      <div className="mt-1">
                        <span className="text-ink-500">Negative:</span> {viewWorkflow.negative_text}
                      </div>
                      {(viewWorkflow.warnings ?? []).length > 0 && <div className="mt-1 text-amber-200">{(viewWorkflow.warnings ?? []).join(" / ")}</div>}
                    </div>
                    <JsonViewer data={viewWorkflow.prompt} filename={`${viewWorkflow.panel_id}_workflow.json`} title={`API ワークフロー: ${viewWorkflow.panel_id}`} maxHeight={380} />
                  </div>
                )}
              </div>
            )}
          </Card>
        </>
      )}
      {structures.error && <ErrorNote message={errorMessage(structures.error)} />}
    </div>
  );
}
