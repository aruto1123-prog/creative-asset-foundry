// ジョブ進捗表示（WebSocket のライブイベントを優先し、ポーリング結果で補完する）。

import type { Job } from "../api/types";
import { useJobProgress } from "../hooks/useProgressSocket";
import { Badge, ProgressBar } from "./ui";

const STATUS_LABELS: Record<string, string> = {
  queued: "待機中",
  running: "実行中",
  succeeded: "完了",
  failed: "失敗",
};

export function JobProgress({ job }: { job: Job | null }) {
  const live = useJobProgress(job?.id ?? null);
  if (!job) {
    return null;
  }
  const useLive = live !== null && (live.progress >= job.progress || live.status !== job.status);
  const progress = useLive ? live.progress : job.progress;
  const status = useLive ? live.status : job.status;
  const stage = useLive ? live.stage : job.stage;
  const message = useLive ? live.message : job.message;
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-ink-700 bg-ink-850 px-3 py-2">
      <div className="flex items-center gap-2 text-xs">
        <Badge tone={status === "failed" ? "danger" : status === "succeeded" ? "success" : "accent"}>{STATUS_LABELS[status] ?? status}</Badge>
        <span className="font-mono text-ink-400">{stage}</span>
        <span className="truncate text-ink-300">{message}</span>
      </div>
      <ProgressBar value={progress} tone={status === "failed" ? "danger" : status === "succeeded" ? "success" : "accent"} />
    </div>
  );
}
