// キャプション精製と共起バイアス・ギャップ分析（絵柄と被写体制御要素の分離）。

import { useMemo, useState } from "react";
import { api, downloadJson } from "../../api/client";
import type { DatasetAnalyzeResponse } from "../../api/types";
import { Heatmap } from "../../components/charts/Heatmap";
import { ScoreGauge } from "../../components/charts/ScoreGauge";
import { Badge, Button, Card, Field, ScoreBar, SectionTitle, Stat, TagList, TextArea, TextInput, Toggle } from "../../components/ui";
import { useAction } from "../../hooks/useAsync";
import { pct } from "../../lib/format";
import { SEVERITY_TONES, TAG_CATEGORY_LABELS, labelOf } from "../../lib/labels";

export function DatasetPanel() {
  const [captions, setCaptions] = useState("");
  const [trigger, setTrigger] = useState("caf_style");
  const [useLlm, setUseLlm] = useState(false);
  const [result, setResult] = useState<DatasetAnalyzeResponse | null>(null);

  const analyze = useAction(
    async () => {
      const response = await api.style.analyzeDataset({ captions_text: captions, trigger_token: trigger, use_llm: useLlm });
      setResult(response);
      return response;
    },
    { successMessage: (r) => `${r.report.total_images} 枚のキャプションを分析しました` },
  );

  const heatmap = useMemo(() => {
    if (!result) {
      return null;
    }
    const { tags, npmi, counts } = result.report.cooccurrence;
    const order = tags
      .map((tag, index) => ({ tag, index, count: result.report.tag_counts[tag] ?? 0 }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 24);
    return {
      labels: order.map((item) => item.tag),
      values: order.map((row) => order.map((column) => npmi[row.index]?.[column.index] ?? 0)),
      counts: order.map((row) => order.map((column) => counts[row.index]?.[column.index] ?? 0)),
    };
  }, [result]);

  const report = result?.report;
  return (
    <div className="flex flex-col gap-4">
      <Card title="キャプション共起バイアス分析" subtitle="固定して覚えさせたい絵柄（タッチ・空気感）と、プロンプトで制御したい要素（構図・アングル・ポーズ・表情・照明・背景）を分離し、癒着と不足を検出します">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_260px]">
          <Field label="キャプション（1行1画像。「画像ID: タグ, タグ, …」または「タグ, タグ, …」）">
            <TextArea value={captions} onChange={setCaptions} rows={10} monospace placeholder="img_0001: 1girl, crying, from above, monochrome, screentone, simple background" ariaLabel="キャプション" />
          </Field>
          <div className="flex flex-col gap-3">
            <Field label="トリガートークン">
              <TextInput value={trigger} onChange={setTrigger} ariaLabel="トリガートークン" />
            </Field>
            <Toggle checked={useLlm} onChange={setUseLlm} label="DeepSeek でキャプションを精製する" />
            <Button variant="primary" loading={analyze.running} disabled={captions.trim().length === 0} onClick={() => void analyze.run()}>
              分析を実行
            </Button>
            {result && (
              <Button onClick={() => downloadJson("cleansed_captions.json", result.cleansed)}>精製キャプションを保存</Button>
            )}
          </div>
        </div>
      </Card>

      {report && (
        <>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[260px_minmax(0,1fr)]">
            <Card title="バランススコア">
              <div className="flex flex-col items-center">
                <ScoreGauge score={report.balance_score} threshold={70} label="データセット均衡度" />
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <Stat label="画像数" value={report.total_images} />
                <Stat label="追加推奨枚数" value={report.total_additional_images} />
              </div>
            </Card>
            <Card title="カテゴリ別エントロピー（正規化）" subtitle="1.0 に近いほど偏りなく分布。低い値は特定の値への偏り（学習時に癒着しやすい）">
              <div className="flex flex-col gap-2">
                {Object.entries(report.normalized_entropy).map(([category, value]) => (
                  <ScoreBar key={category} label={labelOf(TAG_CATEGORY_LABELS, category)} value={value} color={value < 0.5 ? "#fb7185" : value < 0.75 ? "#fbbf24" : "#34d399"} />
                ))}
              </div>
              {report.warnings.length > 0 && (
                <ul className="mt-3 list-disc space-y-0.5 pl-5 text-xs text-amber-200">
                  {report.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              )}
            </Card>
          </div>

          <Card title="概念の癒着（Entanglement）" subtitle="異カテゴリ間で常に共起するタグの組。LoRA が両者を一体として覚え、プロンプトで分離制御できなくなるリスク">
            {report.entanglements.length === 0 ? (
              <p className="text-xs text-emerald-300">重大な癒着は検出されませんでした。</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-ink-400">
                    <tr>
                      <th className="py-1 pr-3">重大度</th>
                      <th className="py-1 pr-3">タグA</th>
                      <th className="py-1 pr-3">タグB</th>
                      <th className="py-1 pr-3">NPMI</th>
                      <th className="py-1 pr-3">P(B|A)</th>
                      <th className="py-1 pr-3">リフト</th>
                      <th className="py-1">共起数</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.entanglements.map((item, index) => (
                      <tr key={index} className="border-t border-ink-800 text-ink-200">
                        <td className="py-1 pr-3">
                          <Badge tone={SEVERITY_TONES[item.severity] ?? "neutral"}>{item.severity}</Badge>
                        </td>
                        <td className="py-1 pr-3">
                          {item.tag_a} <span className="text-ink-500">（{labelOf(TAG_CATEGORY_LABELS, item.category_a)}）</span>
                        </td>
                        <td className="py-1 pr-3">
                          {item.tag_b} <span className="text-ink-500">（{labelOf(TAG_CATEGORY_LABELS, item.category_b)}）</span>
                        </td>
                        <td className="py-1 pr-3 font-mono">{item.npmi.toFixed(3)}</td>
                        <td className="py-1 pr-3 font-mono">{pct(item.conditional_prob)}</td>
                        <td className="py-1 pr-3 font-mono">{item.lift.toFixed(2)}</td>
                        <td className="py-1 font-mono">{item.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card title="不足タグ（ギャップ）" subtitle="目標シェアに達するために追加すべき枚数">
              {report.deficits.length === 0 ? (
                <p className="text-xs text-emerald-300">不足しているタグはありません。</p>
              ) : (
                <div className="max-h-80 overflow-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-ink-400">
                      <tr>
                        <th className="py-1 pr-3">カテゴリ</th>
                        <th className="py-1 pr-3">タグ</th>
                        <th className="py-1 pr-3">現在</th>
                        <th className="py-1 pr-3">シェア / 目標</th>
                        <th className="py-1">追加枚数</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.deficits.map((deficit, index) => (
                        <tr key={index} className="border-t border-ink-800 text-ink-200">
                          <td className="py-1 pr-3">{labelOf(TAG_CATEGORY_LABELS, deficit.category)}</td>
                          <td className="py-1 pr-3">{deficit.tag}</td>
                          <td className="py-1 pr-3 font-mono">{deficit.current_count}</td>
                          <td className="py-1 pr-3 font-mono">
                            {pct(deficit.current_share)} / {pct(deficit.target_share)}
                          </td>
                          <td className="py-1 font-mono text-amber-200">+{deficit.required_additional}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
            <Card title="推奨撮影・生成条件" subtitle="不足の解消と癒着の分離を同時に満たす追加学習画像の条件">
              {report.recommended_shots.length === 0 ? (
                <p className="text-xs text-emerald-300">追加の撮影条件は不要です。</p>
              ) : (
                <div className="flex max-h-80 flex-col gap-2 overflow-auto">
                  {report.recommended_shots.map((shot, index) => (
                    <div key={index} className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-xs">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <Badge tone="accent">{shot.count} 枚</Badge>
                        {Object.entries(shot.conditions).map(([category, tag]) => (
                          <Badge key={category}>
                            {labelOf(TAG_CATEGORY_LABELS, category)}: {tag}
                          </Badge>
                        ))}
                        <span className="ml-auto font-mono text-ink-500">優先度 {shot.priority.toFixed(2)}</span>
                      </div>
                      <p className="mt-1 text-ink-300">{shot.reason}</p>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          </div>

          {heatmap && heatmap.labels.length > 1 && (
            <Card title="共起 NPMI ヒートマップ（出現上位タグ）">
              <Heatmap labels={heatmap.labels} values={heatmap.values} counts={heatmap.counts} />
            </Card>
          )}

          <Card title="精製後キャプション" subtitle="絵柄タグ・品質タグ・メタタグを除去し、トリガートークンと制御要素を整理した学習用キャプション">
            <div className="max-h-96 overflow-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-ink-900 text-ink-400">
                  <tr>
                    <th className="py-1 pr-3">画像</th>
                    <th className="py-1 pr-3">精製キャプション</th>
                    <th className="py-1 pr-3">除去（絵柄）</th>
                    <th className="py-1">欠落カテゴリ</th>
                  </tr>
                </thead>
                <tbody>
                  {result?.cleansed.slice(0, 200).map((item) => (
                    <tr key={item.image_id} className="border-t border-ink-800 align-top text-ink-200">
                      <td className="py-1 pr-3 font-mono">{item.image_id}</td>
                      <td className="py-1 pr-3">{item.caption}</td>
                      <td className="py-1 pr-3">
                        <TagList items={item.removed_style_tags} tone="warning" />
                      </td>
                      <td className="py-1">
                        <TagList items={item.missing_categories.map((category) => labelOf(TAG_CATEGORY_LABELS, category))} tone="danger" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <SectionTitle>タグ頻度（上位）</SectionTitle>
            <div className="flex flex-wrap gap-1">
              {Object.entries(report.tag_counts)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 40)
                .map(([tag, count]) => (
                  <Badge key={tag}>
                    {tag} × {count}
                  </Badge>
                ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
