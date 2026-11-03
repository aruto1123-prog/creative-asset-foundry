// ブラインド A/B 判定アリーナ（←/→ で勝者、↓・Space で引分）と候補ブートストラップ。

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorMessage } from "../../api/client";
import type { ArenaPair, BlockGrouping, ComparisonOutcome, ModelArchitecture, StyleCandidate } from "../../api/types";
import { Badge, Button, Card, EmptyState, Field, NumberInput, Select, Spinner, TextInput } from "../../components/ui";
import { useAction } from "../../hooks/useAsync";
import { useToast } from "../../hooks/useToast";
import { parseList, randomSessionId } from "../../lib/format";
import { ARCHITECTURE_LABELS } from "../../lib/labels";

export const ARCHITECTURE_OPTIONS: Array<{ value: ModelArchitecture; label: string }> = (["sdxl", "sd15", "flux", "anima"] as ModelArchitecture[]).map((value) => ({
  value,
  label: ARCHITECTURE_LABELS[value],
}));

export const GROUPING_OPTIONS: Array<{ value: BlockGrouping; label: string }> = [
  { value: "grouped", label: "グループ単位（IN/MID/OUT 等）" },
  { value: "per_block", label: "ブロック単位（全ブロック個別）" },
];

export function candidateImage(candidate: StyleCandidate): string {
  const ref = candidate.image_ref ?? "";
  if (/^(https?:|data:|\/)/.test(ref)) {
    return ref;
  }
  return api.style.previewUrl(candidate.id);
}

interface LastJudgement {
  pair: ArenaPair;
  outcome: ComparisonOutcome;
  ms: number;
}

const OUTCOME_LABELS: Record<ComparisonOutcome, string> = { left: "左の勝ち", right: "右の勝ち", tie: "引分" };

export function ArenaPanel({
  active,
  candidateVersion,
  onJudged,
  onCandidatesChanged,
}: {
  active: boolean;
  candidateVersion: number;
  onJudged: () => void;
  onCandidatesChanged: () => void;
}) {
  const { push } = useToast();
  const [pair, setPair] = useState<ArenaPair | null>(null);
  const [pairError, setPairError] = useState<string | null>(null);
  const [loadingPair, setLoadingPair] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [last, setLast] = useState<LastJudgement | null>(null);
  const [count, setCount] = useState(0);
  const shownAt = useRef(performance.now());
  const sessionId = useMemo(() => randomSessionId(), []);

  const [architecture, setArchitecture] = useState<ModelArchitecture>("sdxl");
  const [loraNames, setLoraNames] = useState("ink_style");
  const [grouping, setGrouping] = useState<BlockGrouping>("grouped");
  const [n, setN] = useState(12);
  const [seed, setSeed] = useState(7);

  const loadPair = useCallback(async () => {
    setLoadingPair(true);
    try {
      const candidates = await api.style.candidates();
      if (candidates.length < 2) {
        setPair(null);
        setPairError(`登録済み候補が ${candidates.length} 件です。A/B 判定には2件以上が必要です`);
        return;
      }
      const next = await api.style.nextPair();
      setPair(next);
      setPairError(null);
      shownAt.current = performance.now();
    } catch (reason) {
      setPair(null);
      setPairError(errorMessage(reason));
    } finally {
      setLoadingPair(false);
    }
  }, []);

  useEffect(() => {
    void loadPair();
  }, [loadPair, candidateVersion]);

  const judge = useCallback(
    async (outcome: ComparisonOutcome) => {
      if (!pair || submitting) {
        return;
      }
      const ms = Math.round(performance.now() - shownAt.current);
      setSubmitting(true);
      try {
        await api.style.compare({ left_id: pair.left.id, right_id: pair.right.id, outcome, response_time_ms: ms, session_id: sessionId });
        setLast({ pair, outcome, ms });
        setCount((value) => value + 1);
        onJudged();
        await loadPair();
      } catch (reason) {
        push("error", errorMessage(reason));
      } finally {
        setSubmitting(false);
      }
    },
    [pair, submitting, sessionId, onJudged, loadPair, push],
  );

  useEffect(() => {
    if (!active) {
      return undefined;
    }
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable)) {
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        void judge("left");
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        void judge("right");
      } else if (event.key === "ArrowDown" || event.key === " ") {
        event.preventDefault();
        void judge("tie");
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [active, judge]);

  const bootstrap = useAction(
    async () => {
      const created = await api.style.bootstrap({ n, architecture, lora_names: parseList(loraNames), grouping, seed });
      onCandidatesChanged();
      return created;
    },
    { successMessage: (created) => `候補を ${created.length} 件登録しました（ラテン超方格サンプリング）` },
  );

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="ブラインド A/B 判定"
        subtitle="直感で好きな方を選んでください。← 左 ／ → 右 ／ ↓ または Space で引分。判定後すぐに次の対戦が表示されます。"
        actions={
          <>
            <Badge tone="accent">このセッション {count} 判定</Badge>
            <Button size="sm" variant="ghost" onClick={() => void loadPair()} loading={loadingPair}>
              ペアを再取得
            </Button>
          </>
        }
      >
        {pairError && !pair && (
          <EmptyState title="対戦ペアを用意できません" hint={`${pairError} — 下の「候補ブートストラップ」で候補を登録してください。`} />
        )}
        {pair && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {(
              [
                ["left", pair.left, "←"],
                ["right", pair.right, "→"],
              ] as const
            ).map(([side, candidate, key]) => (
              <button
                key={side}
                type="button"
                disabled={submitting}
                onClick={() => void judge(side)}
                className="group relative overflow-hidden rounded-xl border-2 border-ink-700 bg-ink-950 transition-colors hover:border-accent disabled:opacity-60"
                aria-label={`${side === "left" ? "左" : "右"}を選ぶ`}
              >
                <img src={candidateImage(candidate)} alt={side === "left" ? "候補A" : "候補B"} className="aspect-square w-full object-contain" draggable={false} />
                <span className="absolute left-3 top-3 rounded-md bg-ink-950/80 px-2 py-1 font-mono text-sm text-ink-100">
                  {side === "left" ? "A" : "B"} <span className="text-accent-strong">{key}</span>
                </span>
              </button>
            ))}
          </div>
        )}
        {pair && (
          <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
            <Button onClick={() => void judge("left")} disabled={submitting}>
              ← 左が良い
            </Button>
            <Button onClick={() => void judge("tie")} disabled={submitting}>
              ↓ 引分
            </Button>
            <Button onClick={() => void judge("right")} disabled={submitting}>
              右が良い →
            </Button>
            {submitting && <Spinner small />}
          </div>
        )}
        {last && (
          <div className="mt-3 rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-xs text-ink-300">
            前回の判定（ラベル開示）: A = <strong className="text-ink-100">{last.pair.left.label}</strong> ／ B = <strong className="text-ink-100">{last.pair.right.label}</strong> →{" "}
            <Badge tone="accent">{OUTCOME_LABELS[last.outcome]}</Badge> 応答時間 {(last.ms / 1000).toFixed(2)} 秒
          </div>
        )}
      </Card>

      <Card title="候補ブートストラップ" subtitle="パラメータ空間（ブロックマージ比率・LoRA 重み・絵柄特徴量）からラテン超方格で初期候補を生成します">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
          <Field label="アーキテクチャ">
            <Select value={architecture} options={ARCHITECTURE_OPTIONS} onChange={setArchitecture} ariaLabel="アーキテクチャ" />
          </Field>
          <Field label="LoRA 名（カンマ区切り）">
            <TextInput value={loraNames} onChange={setLoraNames} ariaLabel="LoRA 名" />
          </Field>
          <Field label="ブロック粒度">
            <Select value={grouping} options={GROUPING_OPTIONS} onChange={setGrouping} ariaLabel="ブロック粒度" />
          </Field>
          <Field label="件数">
            <NumberInput value={n} onChange={setN} min={2} max={64} ariaLabel="件数" />
          </Field>
          <Field label="シード">
            <NumberInput value={seed} onChange={setSeed} min={0} ariaLabel="シード" />
          </Field>
        </div>
        <div className="mt-3">
          <Button variant="primary" loading={bootstrap.running} onClick={() => void bootstrap.run()}>
            候補を生成して登録
          </Button>
        </div>
      </Card>
    </div>
  );
}
