// 異分野トランスパイラ: 漫画プロット / LP・セールスファネル / 解説記事 へのワンクリック変換。

import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { TargetDomain, TranspiledDesign, UniversalStructure } from "../../api/types";
import { JsonViewer } from "../../components/JsonViewer";
import { Badge, Button, Card, EmptyState, Field, KeyValueEditor, ScoreBar, Stat } from "../../components/ui";
import { useAction } from "../../hooks/useAsync";
import { fmt, pct } from "../../lib/format";
import { BEAT_TYPE_LABELS, DOMAIN_LABELS } from "../../lib/labels";

const DOMAINS: TargetDomain[] = ["manga_plot", "marketing_lp", "article"];

export function TranspilePanel({ structure }: { structure: UniversalStructure | null }) {
  const [entries, setEntries] = useState<Array<[string, string]>>([]);
  const [design, setDesign] = useState<TranspiledDesign | null>(null);

  useEffect(() => {
    if (!structure) {
      return;
    }
    setEntries([["THEME", ""], ...structure.blueprint.slots.map((slot) => [slot.name, ""] as [string, string])]);
    setDesign(null);
  }, [structure]);

  const transpile = useAction(
    async (domain: TargetDomain) => {
      if (!structure) {
        throw new Error("構造が選択されていません");
      }
      const themeVariables: Record<string, string> = {};
      for (const [key, value] of entries) {
        if (key.trim() && value.trim()) {
          themeVariables[key.trim()] = value.trim();
        }
      }
      const result = await api.structures.transpile(structure.id, domain, themeVariables);
      setDesign(result);
      return result;
    },
    { successMessage: (r) => `${DOMAIN_LABELS[r.domain] ?? r.domain} へ変換しました` },
  );

  if (!structure) {
    return <EmptyState title="左の一覧から変換する構造を選択してください" />;
  }

  return (
    <div className="flex flex-col gap-4">
      <Card title={`異分野変換: ${structure.metadata.title}`} subtitle="元の力学と感情波形を保ったまま、対象領域の具体的設計書へ変換します。空欄のスロットは転用例から補完されます。">
        <Field label="テーマ変数（スロット名 → 具体値）">
          <KeyValueEditor entries={entries} onChange={setEntries} keyPlaceholder="スロット名" valuePlaceholder="この企画での具体値" />
        </Field>
        <div className="mt-3 flex flex-wrap gap-2">
          {DOMAINS.map((domain) => (
            <Button key={domain} variant="primary" loading={transpile.running} onClick={() => void transpile.run(domain)}>
              {DOMAIN_LABELS[domain]} へ変換
            </Button>
          ))}
        </div>
      </Card>

      {design && (
        <Card
          title={design.title}
          subtitle={design.logline}
          actions={<Badge tone="accent">{DOMAIN_LABELS[design.domain] ?? design.domain}</Badge>}
        >
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <Stat label="忠実度スコア" value={fmt(design.fidelity.score, 3)} />
            <Stat label="波形相関" value={fmt(design.fidelity.wave_correlation, 3)} />
            <Stat label="ビート被覆率" value={pct(design.fidelity.beat_coverage)} />
            <Stat label="力学保存" value={design.fidelity.topology_preserved ? "保存" : "欠落あり"} />
          </div>
          <div className="mt-4 rounded-lg border border-ink-700 bg-ink-850 p-3 text-sm leading-relaxed text-ink-100">
            <div className="mb-1 text-xs font-semibold text-ink-400">具体化されたブループリント</div>
            <p className="whitespace-pre-wrap">{design.rendered_blueprint}</p>
          </div>
          <div className="mt-4 flex flex-col gap-2">
            {design.sections.map((section) => (
              <div key={`${section.index}-${section.beat_id}`} className="rounded-lg border border-ink-700 bg-ink-850 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="accent">{section.index + 1}</Badge>
                  <Badge>{BEAT_TYPE_LABELS[section.beat_type] ?? section.beat_type}</Badge>
                  <span className="font-mono text-[11px] text-ink-400">
                    t {section.t_start.toFixed(2)}–{section.t_end.toFixed(2)}
                  </span>
                  <span className="text-sm font-semibold text-ink-100">{section.heading}</span>
                </div>
                <p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-ink-200">{section.content}</p>
                <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2">
                  <ScoreBar label="目標緊張度" value={section.tension_target} color="#f472b6" />
                  <ScoreBar label="目標解放度" value={section.release_target} color="#34d399" />
                </div>
                <div className="mt-2 grid gap-1 text-[11px] text-ink-300">
                  <span>
                    <span className="text-ink-500">情報開示:</span> {section.information_reveal}
                  </span>
                  <span>
                    <span className="text-ink-500">感情目標:</span> {section.emotional_goal}
                  </span>
                  {section.mechanisms.length > 0 && (
                    <span>
                      <span className="text-ink-500">メカニズム:</span> {section.mechanisms.join("、")}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4">
            <JsonViewer data={design.domain_payload} filename={`${design.id}_${design.domain}.json`} title="領域別ペイロード（domain_payload）" maxHeight={320} />
          </div>
        </Card>
      )}
    </div>
  );
}
