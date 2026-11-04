// テキストからの5階層構造抽出（脱フレーバー → 4段階ハーネス）。

import { useState } from "react";
import { api } from "../../api/client";
import type { ExtractionResult, Job } from "../../api/types";
import { JobProgress } from "../../components/JobProgress";
import { Badge, Button, Card, Field, Select, Stat, TextArea, TextInput } from "../../components/ui";
import { useAction } from "../../hooks/useAsync";
import { fmt } from "../../lib/format";

const SOURCE_KINDS = [
  { value: "synopsis", label: "あらすじ・プロット" },
  { value: "storyboard", label: "漫画ネーム・脚本" },
  { value: "article", label: "記事・論説" },
  { value: "lp", label: "LP・セールス原稿" },
];

const CATEGORIES = [
  { value: "custom", label: "カスタム" },
  { value: "narrative_archetype", label: "物語原型" },
  { value: "manga_panel_rhythm", label: "漫画演出リズム" },
  { value: "persuasion_marketing", label: "説得・マーケティング" },
  { value: "advanced_narrative", label: "高度な物語構造" },
  { value: "manga_direction", label: "漫画演出" },
  { value: "emotional_wave", label: "感情波形プリセット" },
];

const TOKEN_CATEGORY_LABELS: Record<string, string> = {
  person: "人物",
  place: "場所",
  organization: "組織",
  artifact: "固有物",
  genre_term: "ジャンル装飾",
  title: "作品名",
  other: "その他",
};

export function ExtractPanel({ onExtracted }: { onExtracted: (result: ExtractionResult) => void }) {
  const [text, setText] = useState("");
  const [sourceKind, setSourceKind] = useState("synopsis");
  const [category, setCategory] = useState("custom");
  const [titleHint, setTitleHint] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<ExtractionResult | null>(null);

  const extract = useAction(
    async () => {
      setResult(null);
      const extracted = await api.structures.extract(
        { text, source_kind: sourceKind, category, title_hint: titleHint.trim() || null, save: true },
        setJob,
      );
      setResult(extracted);
      onExtracted(extracted);
      return extracted;
    },
    { successMessage: (r) => `構造を抽出しました: ${r.structure.metadata.title}` },
  );

  return (
    <div className="flex flex-col gap-4">
      <Card title="5階層構造の抽出" subtitle="固有名詞・ジャンル装飾を剥ぎ取り、力学・波形・ビート・認知メカニズム・ブループリントへ逆コンパイルします">
        <div className="flex flex-col gap-3">
          <Field label="入力テキスト（あらすじ・ネーム・記事・LP原稿）">
            <TextArea
              value={text}
              onChange={setText}
              rows={10}
              placeholder="例: 主人公が何を失い、何を目指し、誰と対立し、どこで前提が反転し、何を代償に突破するのかが分かる文章を貼り付けてください。固有名詞はそのままで構いません（自動で抽象化されます）。"
              ariaLabel="抽出対象テキスト"
            />
          </Field>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <Field label="ソース種別">
              <Select value={sourceKind} options={SOURCE_KINDS} onChange={setSourceKind} ariaLabel="ソース種別" />
            </Field>
            <Field label="保存カテゴリ">
              <Select value={category} options={CATEGORIES} onChange={setCategory} ariaLabel="保存カテゴリ" />
            </Field>
            <Field label="タイトルのヒント（任意）">
              <TextInput value={titleHint} onChange={setTitleHint} placeholder="抽象的な題名" ariaLabel="タイトルのヒント" />
            </Field>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="primary" loading={extract.running} disabled={text.trim().length === 0} onClick={() => void extract.run()}>
              抽出を実行（Decompose → Synthesize → Critique → Refine）
            </Button>
            <span className="text-xs text-ink-500">{text.length} 文字</span>
          </div>
          {extract.running && <JobProgress job={job} />}
        </div>
      </Card>

      {result && (
        <Card title={`抽出結果: ${result.structure.metadata.title}`} subtitle={<span className="font-mono">{result.structure.id}</span>}>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <Stat label="LLM 使用" value={result.used_llm ? "はい" : "いいえ（決定論）"} />
            <Stat label="自己批評スコア" value={result.critique_score === null ? "-" : fmt(result.critique_score, 1)} />
            <Stat label="除去した固有表現" value={result.deflavor.tokens.length} />
            <Stat label="漏洩を自動修正" value={result.leaked_tokens_fixed.length} />
          </div>
          {result.deflavor.tokens.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-ink-400">
                  <tr>
                    <th className="py-1 pr-3">種別</th>
                    <th className="py-1 pr-3">抽象化後</th>
                    <th className="py-1">出現数</th>
                  </tr>
                </thead>
                <tbody>
                  {result.deflavor.tokens.map((token) => (
                    <tr key={`${token.placeholder}-${token.surface}`} className="border-t border-ink-800 text-ink-200">
                      <td className="py-1 pr-3">
                        <Badge>{TOKEN_CATEGORY_LABELS[token.category] ?? token.category}</Badge>
                      </td>
                      <td className="py-1 pr-3 font-mono text-accent-strong">{token.placeholder}</td>
                      <td className="py-1">{token.occurrences}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {Object.keys(result.deflavor.genre_signals).length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              <span className="text-ink-500">ジャンル信号</span>
              {Object.entries(result.deflavor.genre_signals)
                .filter(([, value]) => value > 0)
                .sort(([, a], [, b]) => b - a)
                .map(([genre, value]) => (
                  <Badge key={genre} tone="info">
                    {genre} {value.toFixed(2)}
                  </Badge>
                ))}
            </div>
          )}
          <details className="mt-3 text-xs text-ink-300">
            <summary className="cursor-pointer text-ink-400">抽象化後テキスト</summary>
            <p className="mt-2 whitespace-pre-wrap rounded-lg border border-ink-700 bg-ink-850 p-3 leading-relaxed">{result.deflavor.abstracted_text}</p>
          </details>
        </Card>
      )}
    </div>
  );
}
