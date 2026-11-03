// Workspace 4: Asset Publisher — 3軸企画展開・5段階執筆・コンプライアンス検証・エクスポート・空白テーマ・シリーズ編纂。

import { useEffect, useMemo, useState } from "react";
import { api, downloadText } from "../api/client";
import type { ArticleSummary, ContentGap, ContentMatrix, ContentPlan, Job, PublishedArticle, ResearchLog, SanitizationReport, SanitizeResponse, SeriesResponse } from "../api/types";
import { JobProgress } from "../components/JobProgress";
import { MarkdownView } from "../components/MarkdownView";
import { ScoreGauge, Sparkline } from "../components/charts/ScoreGauge";
import { Badge, Button, Card, CopyButton, EmptyState, ErrorNote, Field, NumberInput, Select, Stat, Tabs, TagList, TextArea, TextInput, cx } from "../components/ui";
import { useAction, useResource } from "../hooks/useAsync";
import { useProgressSocket } from "../hooks/useProgressSocket";
import { useStructureOptions } from "../hooks/useStructures";
import { dateTime, fmt } from "../lib/format";
import { AXIS_LABELS, AXIS_SHORT_LABELS, CATEGORY_LABELS, labelOf } from "../lib/labels";

type SourceMode = "structure" | "research_log" | "experiment";

const STAGES = [
  { key: "blueprint", label: "Blueprint", hint: "常識→パラドックス→仮説と証拠→枠組み→実践" },
  { key: "draft", label: "Deep Drafting", hint: "章ごとの高密度執筆と Mermaid 図解" },
  { key: "sanitize", label: "Sanitizer", hint: "一人称「私」統一・URL 完全除去" },
  { key: "critique", label: "Critique", hint: "100点満点の章別採点" },
  { key: "polish", label: "Polish", hint: "基準未満の章を再推敲" },
];

function stageIndex(stage: string | undefined): number {
  if (!stage) {
    return -1;
  }
  if (stage.includes("blueprint")) {
    return 0;
  }
  if (stage.includes("draft")) {
    return 1;
  }
  if (stage.includes("sanit")) {
    return 2;
  }
  if (stage.includes("critique")) {
    return 3;
  }
  if (stage.includes("polish")) {
    return 4;
  }
  if (stage.includes("final") || stage.includes("complete") || stage.includes("done")) {
    return 5;
  }
  return -1;
}

function ComplianceBadges({ report }: { report: SanitizationReport }) {
  return (
    <div className="flex flex-wrap gap-2">
      <Badge tone={report.first_person_unified ? "success" : "danger"}>
        {report.first_person_unified ? "✓" : "✕"} 一人称「私」統一（置換 {report.first_person_replacements.length} 件・残存 {report.residual_first_person.length}）
      </Badge>
      <Badge tone={report.url_free ? "success" : "danger"}>
        {report.url_free ? "✓" : "✕"} URL 完全除去（除去 {report.urls_removed.length} 件・残存 {report.residual_urls.length}）
      </Badge>
    </div>
  );
}

function PlanCard({ plan, onWrite, writing }: { plan: ContentPlan; onWrite: (plan: ContentPlan) => void; writing: boolean }) {
  return (
    <div className="flex flex-col rounded-xl border border-ink-700 bg-ink-850 p-4">
      <Badge tone="accent">{AXIS_SHORT_LABELS[plan.axis] ?? plan.axis}</Badge>
      <h4 className="mt-2 text-sm font-semibold leading-snug text-ink-100">{plan.title}</h4>
      {plan.subtitle && <p className="mt-1 text-xs text-ink-400">{plan.subtitle}</p>}
      <p className="mt-2 text-xs text-ink-300">
        <span className="text-ink-500">読者:</span> {plan.target_reader}
      </p>
      <p className="mt-1 text-xs text-ink-300">
        <span className="text-ink-500">核心:</span> {plan.core_insight}
      </p>
      <p className="mt-1 text-xs italic text-ink-300">「{plan.hook_line}」</p>
      <ol className="mt-2 list-decimal space-y-0.5 pl-5 text-[11px] text-ink-300">
        {plan.outline.map((section, index) => (
          <li key={index}>{section.heading}</li>
        ))}
      </ol>
      <div className="mt-2">
        <TagList items={plan.keywords.slice(0, 8)} />
      </div>
      <div className="mt-auto pt-3">
        <Button variant="primary" className="w-full" loading={writing} onClick={() => onWrite(plan)}>
          このプランで執筆
        </Button>
      </div>
    </div>
  );
}

export function AssetPublisher({ dataVersion }: { dataVersion: number }) {
  const structures = useStructureOptions(dataVersion);
  const { lastEvent } = useProgressSocket();
  const [sourceMode, setSourceMode] = useState<SourceMode>("structure");
  const [structureId, setStructureId] = useState("");
  const [logJson, setLogJson] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [matrix, setMatrix] = useState<ContentMatrix | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [article, setArticle] = useState<PublishedArticle | null>(null);
  const [view, setView] = useState<"preview" | "markdown">("preview");
  const [verifyReport, setVerifyReport] = useState<SanitizationReport | null>(null);
  const [listVersion, setListVersion] = useState(0);
  const articles = useResource<ArticleSummary[]>(() => api.publisher.articles(), [listVersion]);
  const [gapTopK, setGapTopK] = useState(6);
  const gaps = useResource<ContentGap[]>(() => api.publisher.gaps(gapTopK), [listVersion, gapTopK]);
  const [sanitizeInput, setSanitizeInput] = useState("");
  const [sanitizeOutput, setSanitizeOutput] = useState<SanitizeResponse | null>(null);
  const [seriesTitle, setSeriesTitle] = useState("構造で設計する創作と資産化");
  const [seriesSelection, setSeriesSelection] = useState<string[]>([]);
  const [series, setSeries] = useState<SeriesResponse | null>(null);

  useEffect(() => {
    if (!structureId && structures.items.length > 0) {
      setStructureId(structures.items[0].id);
    }
  }, [structures.items, structureId]);

  const structureOptions = useMemo(
    () => structures.items.map((item) => ({ value: item.id, label: `［${labelOf(CATEGORY_LABELS, item.category)}］${item.title}` })),
    [structures.items],
  );

  const parseJson = (): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(logJson) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setJsonError("JSON オブジェクトを入力してください");
        return null;
      }
      setJsonError(null);
      return parsed as Record<string, unknown>;
    } catch (reason) {
      setJsonError(`JSON の解析に失敗しました: ${reason instanceof Error ? reason.message : String(reason)}`);
      return null;
    }
  };

  const expand = useAction(
    async () => {
      let body: Parameters<typeof api.publisher.matrix>[0];
      if (sourceMode === "structure") {
        body = { structure_id: structureId };
      } else {
        const parsed = parseJson();
        if (!parsed) {
          throw new Error(jsonError ?? "JSON を確認してください");
        }
        body = sourceMode === "research_log" ? { research_log: parsed as unknown as ResearchLog } : { experiment: parsed };
      }
      const result = await api.publisher.matrix(body);
      setMatrix(result);
      return result;
    },
    { successMessage: "3軸の企画を展開しました" },
  );

  const write = useAction(
    async (plan: ContentPlan) => {
      setArticle(null);
      setVerifyReport(null);
      const body: Parameters<typeof api.publisher.createArticle>[0] = { plan };
      if (sourceMode === "structure" && structureId) {
        body.structure_id = structureId;
      } else if (sourceMode === "research_log") {
        const parsed = parseJson();
        if (parsed) {
          body.research_log = parsed as unknown as ResearchLog;
        }
      } else if (matrix) {
        body.matrix_id = matrix.id;
        body.axis = plan.axis;
      }
      const created = await api.publisher.createArticle(body, setJob);
      setArticle(created);
      setListVersion((value) => value + 1);
      return created;
    },
    { successMessage: (a) => `記事を生成しました（${a.final_score.toFixed(1)}点）` },
  );

  const open = useAction(async (id: string) => {
    const loaded = await api.publisher.article(id);
    setArticle(loaded);
    setVerifyReport(null);
    return loaded;
  });

  const verify = useAction(
    async () => {
      if (!article) {
        return null;
      }
      const report = await api.publisher.verify(article.markdown);
      setVerifyReport(report);
      return report;
    },
    { successMessage: (r) => (r && r.first_person_unified && r.url_free ? "再検証: 一人称・URL ともに準拠しています" : "再検証が完了しました") },
  );

  const exportArticle = useAction(async (format: "note" | "obsidian") => {
    if (!article) {
      return;
    }
    const text = await api.publisher.exportArticle(article.id, format);
    downloadText(`${article.id}_${format}.md`, text, "text/markdown;charset=utf-8");
  });

  const removeArticle = useAction(
    async (id: string) => {
      await api.publisher.removeArticle(id);
      if (article?.id === id) {
        setArticle(null);
      }
      setListVersion((value) => value + 1);
    },
    { successMessage: "記事を削除しました" },
  );

  const sanitize = useAction(async () => {
    const result = await api.publisher.sanitize(sanitizeInput);
    setSanitizeOutput(result);
    return result;
  });

  const compileSeries = useAction(
    async () => {
      const result = await api.publisher.series(seriesSelection, seriesTitle);
      setSeries(result);
      return result;
    },
    { successMessage: (r) => `シリーズ「${r.volume.title}」を編纂しました（${r.volume.chapters.length}章）` },
  );

  const liveStage = write.running && lastEvent && job && lastEvent.job_id === job.id ? lastEvent.stage : job?.stage;
  const currentStage = write.running ? stageIndex(liveStage) : article ? 5 : -1;
  const scoreHistory = article?.critique_history.map((critique) => critique.overall) ?? [];
  const compliance = verifyReport ?? article?.sanitization ?? null;

  return (
    <div className="flex flex-col gap-4 p-5">
      <div>
        <h1 className="text-xl font-semibold text-ink-100">Asset Publisher</h1>
        <p className="text-xs text-ink-400">構造研究・モデル実験の知見を、3軸の高付加価値記事へ自動体系化（一人称「私」・URL 完全除去）</p>
      </div>

      <Card title="ソースと3軸企画展開" subtitle="1つの構造発見または検証ログを、クリエイター／AI開発者／ビジネスの3つの読者文脈へ同時展開します">
        <Tabs
          tabs={[
            { key: "structure", label: "構造から" },
            { key: "research_log", label: "研究ログ JSON" },
            { key: "experiment", label: "実験ログ JSON" },
          ]}
          active={sourceMode}
          onChange={setSourceMode}
          size="sm"
        />
        <div className="mt-3">
          {sourceMode === "structure" ? (
            <Field label="構造">
              <Select value={structureId} options={structureOptions.length > 0 ? structureOptions : [{ value: "", label: "構造がありません" }]} onChange={setStructureId} ariaLabel="記事化する構造" />
            </Field>
          ) : (
            <Field
              label={sourceMode === "research_log" ? "研究ログ（ResearchLog: title / summary / findings[]）" : "実験ログ（preference_vector / ratings / suggestions / gap_report 等）"}
              hint="Style Alchemy の出力 JSON などをそのまま貼り付けられます"
            >
              <TextArea value={logJson} onChange={setLogJson} rows={8} monospace placeholder='{"title": "…", "summary": "…", "findings": [{"claim": "…", "evidence": "…", "metrics": {"score": 0.8}}]}' ariaLabel="ログ JSON" />
            </Field>
          )}
          <ErrorNote message={jsonError} />
          <div className="mt-3">
            <Button variant="primary" loading={expand.running} disabled={sourceMode === "structure" ? !structureId : logJson.trim().length === 0} onClick={() => void expand.run()}>
              3軸企画を展開
            </Button>
          </div>
        </div>
        {matrix && (
          <div className="mt-4">
            <div className="mb-3 rounded-lg border border-accent/30 bg-accent-soft/20 px-3 py-2 text-xs text-ink-200">
              <span className="text-ink-400">共通の核心洞察:</span> {matrix.shared_core}
            </div>
            <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
              {matrix.plans.map((plan) => (
                <PlanCard key={plan.axis} plan={plan} writing={write.running} onWrite={(selected) => void write.run(selected)} />
              ))}
            </div>
          </div>
        )}
      </Card>

      <Card title="5段階執筆パイプライン" subtitle="Blueprint → Deep Section Drafting → Compliance & Style Sanitizer → Critique Scoring → Iterative Polish">
        <ol className="grid grid-cols-1 gap-2 md:grid-cols-5">
          {STAGES.map((stage, index) => {
            const done = currentStage > index;
            const activeStage = currentStage === index;
            return (
              <li
                key={stage.key}
                className={cx(
                  "rounded-lg border px-3 py-2 text-xs",
                  done ? "border-emerald-500/40 bg-emerald-950/30 text-emerald-100" : activeStage ? "border-accent bg-accent-soft/40 text-accent-strong" : "border-ink-700 bg-ink-850 text-ink-400",
                )}
              >
                <div className="font-semibold">
                  {index + 1}. {stage.label} {done ? "✓" : ""}
                </div>
                <div className="mt-0.5 text-[11px] opacity-80">{stage.hint}</div>
              </li>
            );
          })}
        </ol>
        {write.running && (
          <div className="mt-3">
            <JobProgress job={job} />
          </div>
        )}
        {!write.running && !article && <p className="mt-3 text-xs text-ink-500">企画カードの「このプランで執筆」から開始します。</p>}
      </Card>

      {article && (
        <Card
          title={article.title}
          subtitle={`${AXIS_LABELS[article.axis] ?? article.axis} ／ ${dateTime(article.created_at)}`}
          actions={
            <>
              <Button size="sm" loading={verify.running} onClick={() => void verify.run()}>
                再検証
              </Button>
              <Button size="sm" loading={exportArticle.running} onClick={() => void exportArticle.run("note")}>
                note 用 Markdown
              </Button>
              <Button size="sm" variant="primary" loading={exportArticle.running} onClick={() => void exportArticle.run("obsidian")}>
                Obsidian 用 Markdown
              </Button>
            </>
          }
        >
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[200px_minmax(0,1fr)]">
            <div className="flex flex-col items-center gap-2">
              <ScoreGauge score={article.final_score} threshold={80} label="批評スコア" />
              <Sparkline values={scoreHistory} />
              <span className="text-[11px] text-ink-500">推敲 {article.iterations} 回</span>
            </div>
            <div className="flex flex-col gap-3">
              {compliance && <ComplianceBadges report={compliance} />}
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                <Stat label="章数" value={article.draft.sections.length} />
                <Stat label="文字数" value={article.markdown.length.toLocaleString()} />
                <Stat label="Mermaid 図解" value={(article.markdown.match(/```mermaid/g) ?? []).length} />
                <Stat label="最終スコア" value={fmt(article.final_score, 1)} />
              </div>
              {article.critique_history.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-[11px]">
                    <thead className="text-ink-400">
                      <tr>
                        <th className="py-1 pr-2">章</th>
                        <th className="py-1 pr-2">論理</th>
                        <th className="py-1 pr-2">抽象と具体</th>
                        <th className="py-1 pr-2">納得度</th>
                        <th className="py-1 pr-2">根拠密度</th>
                        <th className="py-1">合計</th>
                      </tr>
                    </thead>
                    <tbody>
                      {article.critique_history[article.critique_history.length - 1].sections.map((section) => (
                        <tr key={section.role} className="border-t border-ink-800 text-ink-200">
                          <td className="py-1 pr-2">{section.heading || section.role}</td>
                          <td className="py-1 pr-2 font-mono">{section.logic_coherence.toFixed(1)}</td>
                          <td className="py-1 pr-2 font-mono">{section.abstraction_balance.toFixed(1)}</td>
                          <td className="py-1 pr-2 font-mono">{section.persuasiveness.toFixed(1)}</td>
                          <td className="py-1 pr-2 font-mono">{section.evidence_density.toFixed(1)}</td>
                          <td className="py-1 font-mono text-accent-strong">{section.total.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {article.stages.length > 0 && (
                <div className="flex flex-wrap gap-1.5 text-[11px]">
                  {article.stages.map((stage, index) => (
                    <Badge key={`${stage.stage}-${index}`} tone="neutral" title={stage.notes}>
                      {stage.stage}#{stage.iteration} {stage.duration_ms.toFixed(0)}ms{stage.score !== null ? ` ${stage.score.toFixed(1)}点` : ""}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div className="mt-4 flex items-center justify-between">
            <Tabs
              tabs={[
                { key: "preview", label: "プレビュー" },
                { key: "markdown", label: "Markdown 原文" },
              ]}
              active={view}
              onChange={setView}
              size="sm"
            />
            <CopyButton text={article.markdown} label="Markdown をコピー" />
          </div>
          <div className="mt-3 max-h-[720px] overflow-y-auto rounded-lg border border-ink-700 bg-ink-950 p-5">
            {view === "preview" ? <MarkdownView text={article.markdown} /> : <pre className="whitespace-pre-wrap font-mono text-xs text-ink-200">{article.markdown}</pre>}
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="記事アーカイブ" actions={<Button size="sm" variant="ghost" onClick={articles.reload}>更新</Button>}>
          {(articles.data ?? []).length === 0 && <EmptyState title="記事はまだありません" />}
          <div className="flex max-h-96 flex-col gap-1.5 overflow-y-auto">
            {(articles.data ?? []).map((item) => (
              <div key={item.id} className={cx("flex items-center gap-2 rounded-lg border px-3 py-2 text-xs", article?.id === item.id ? "border-accent bg-accent-soft/30" : "border-ink-700 bg-ink-850")}>
                <input
                  type="checkbox"
                  checked={seriesSelection.includes(item.id)}
                  onChange={(event) => setSeriesSelection((current) => (event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id)))}
                  aria-label={`${item.title} をシリーズに含める`}
                  className="accent-[#7c9cff]"
                />
                <button type="button" className="min-w-0 flex-1 text-left" onClick={() => void open.run(item.id)}>
                  <div className="truncate font-medium text-ink-100">{item.title}</div>
                  <div className="mt-0.5 flex gap-2 text-ink-400">
                    <Badge>{AXIS_SHORT_LABELS[item.axis] ?? item.axis}</Badge>
                    <span>{item.final_score.toFixed(1)}点</span>
                    <span>{item.char_count.toLocaleString()}字</span>
                  </div>
                </button>
                <Button size="sm" variant="ghost" onClick={() => void removeArticle.run(item.id)} aria-label="記事を削除">
                  ✕
                </Button>
              </div>
            ))}
          </div>
          <div className="mt-3 flex flex-wrap items-end gap-2">
            <Field label="シリーズ名" className="flex-1">
              <TextInput value={seriesTitle} onChange={setSeriesTitle} ariaLabel="シリーズ名" />
            </Field>
            <Button variant="primary" disabled={seriesSelection.length === 0} loading={compileSeries.running} onClick={() => void compileSeries.run()}>
              選択記事を教本に編纂（{seriesSelection.length}）
            </Button>
          </div>
        </Card>

        <Card
          title="コンテンツ空白（次に書くべきテーマ）"
          subtitle="記事群と構造データのナレッジグラフから、需要が高く未開拓の概念を検出"
          actions={
            <div className="flex items-center gap-2">
              <NumberInput value={gapTopK} onChange={setGapTopK} min={1} max={20} className="w-16 py-1 text-xs" ariaLabel="提案件数" />
              <Button size="sm" variant="ghost" onClick={gaps.reload} loading={gaps.loading}>
                再計算
              </Button>
            </div>
          }
        >
          {gaps.error && <ErrorNote message={gaps.error} />}
          {(gaps.data ?? []).length === 0 && !gaps.loading && <EmptyState title="空白テーマは検出されませんでした" />}
          <div className="flex max-h-[480px] flex-col gap-2 overflow-y-auto">
            {(gaps.data ?? []).map((gap, index) => (
              <div key={index} className="rounded-lg border border-ink-700 bg-ink-850 p-3 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="accent">空白度 {gap.gap_score.toFixed(3)}</Badge>
                  {gap.axis && <Badge>{AXIS_SHORT_LABELS[gap.axis] ?? gap.axis}</Badge>}
                  <span className="font-mono text-ink-500">
                    需要 {gap.demand.toFixed(2)} ／ 被覆 {gap.coverage.toFixed(2)}
                  </span>
                </div>
                <div className="mt-1.5 text-sm font-medium text-ink-100">{gap.suggested_title}</div>
                <p className="mt-1 text-ink-400">{gap.rationale}</p>
                <div className="mt-1.5">
                  <TagList items={gap.concepts} tone="info" />
                </div>
                {gap.suggested_outline.length > 0 && (
                  <ol className="mt-1.5 list-decimal space-y-0.5 pl-5 text-ink-300">
                    {gap.suggested_outline.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ol>
                )}
              </div>
            ))}
          </div>
        </Card>
      </div>

      {series && (
        <Card
          title={`教本: ${series.volume.title}`}
          subtitle={series.volume.description}
          actions={
            <Button size="sm" variant="primary" onClick={() => downloadText(`${series.volume.id}.md`, series.markdown, "text/markdown;charset=utf-8")}>
              Markdown をダウンロード
            </Button>
          }
        >
          <ol className="mb-3 list-decimal space-y-1 pl-5 text-xs text-ink-200">
            {series.volume.chapters.map((chapter) => (
              <li key={chapter.index}>
                <span className="font-medium">{chapter.title}</span>
                <span className="text-ink-500">（{chapter.article_ids.length}記事）</span>
                {chapter.summary && <span className="text-ink-400"> — {chapter.summary}</span>}
              </li>
            ))}
          </ol>
          <div className="max-h-[520px] overflow-y-auto rounded-lg border border-ink-700 bg-ink-950 p-5">
            <MarkdownView text={series.markdown} />
          </div>
        </Card>
      )}

      <Card title="サニタイズ試験" subtitle="任意のテキストに対して、一人称の「私」への統一と URL（Markdown/HTML リンク・短縮URL・難読化表記を含む）の完全除去を試せます">
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <TextArea value={sanitizeInput} onChange={setSanitizeInput} rows={8} placeholder="僕はこの方法を試した。詳細は https://… や [リンク](https://…) を参照。" ariaLabel="サニタイズ対象テキスト" />
          <div className="flex flex-col gap-2">
            <Button variant="primary" disabled={sanitizeInput.trim().length === 0} loading={sanitize.running} onClick={() => void sanitize.run()}>
              サニタイズを実行
            </Button>
            {sanitizeOutput && (
              <>
                <ComplianceBadges report={sanitizeOutput.report} />
                <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-lg border border-ink-700 bg-ink-950 p-3 text-xs text-ink-100">{sanitizeOutput.text}</pre>
                {sanitizeOutput.report.first_person_replacements.length > 0 && (
                  <div className="text-[11px] text-ink-300">
                    一人称置換: {sanitizeOutput.report.first_person_replacements.map((item) => `${item.original}→${item.replacement}`).join("、")}
                  </div>
                )}
                {sanitizeOutput.report.urls_removed.length > 0 && (
                  <div className="text-[11px] text-ink-300">
                    除去した URL 種別: {sanitizeOutput.report.urls_removed.map((item) => item.kind).join("、")}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
