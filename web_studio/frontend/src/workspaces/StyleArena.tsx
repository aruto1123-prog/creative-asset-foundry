// Workspace 2: Style Alchemy Arena — A/B 判定・ランキング・選好ベクトル・ベイズ最適化・データセット分析・グリッド実験。

import { useCallback, useState } from "react";
import { Tabs } from "../components/ui";
import { ArenaPanel } from "./style/ArenaPanel";
import { DatasetPanel } from "./style/DatasetPanel";
import { GridPanel } from "./style/GridPanel";
import { OptimizerPanel } from "./style/OptimizerPanel";
import { RankingPanel } from "./style/RankingPanel";

type TabKey = "arena" | "ranking" | "optimizer" | "dataset" | "grid";

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: "arena", label: "A/B アリーナ" },
  { key: "ranking", label: "ランキング・選好ベクトル" },
  { key: "optimizer", label: "ベイズ最適化" },
  { key: "dataset", label: "データセット分析" },
  { key: "grid", label: "グリッド実験・トークン辞書" },
];

export function StyleArena({ active }: { active: boolean }) {
  const [tab, setTab] = useState<TabKey>("arena");
  const [judgeVersion, setJudgeVersion] = useState(0);
  const [candidateVersion, setCandidateVersion] = useState(0);
  const onJudged = useCallback(() => setJudgeVersion((value) => value + 1), []);
  const onCandidatesChanged = useCallback(() => setCandidateVersion((value) => value + 1), []);

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-ink-100">Style Alchemy Arena</h1>
          <p className="text-xs text-ink-400">美的選好の測定（Elo / Bradley-Terry）→ 選好ベクトル → 階層マージ・LoRA 配合のベイズ最適化</p>
        </div>
        <Tabs tabs={TABS} active={tab} onChange={setTab} />
      </div>
      <div hidden={tab !== "arena"}>
        <ArenaPanel active={active && tab === "arena"} candidateVersion={candidateVersion} onJudged={onJudged} onCandidatesChanged={onCandidatesChanged} />
      </div>
      {tab === "ranking" && <RankingPanel version={judgeVersion + candidateVersion} />}
      {tab === "optimizer" && <OptimizerPanel onCandidatesChanged={onCandidatesChanged} />}
      {tab === "dataset" && <DatasetPanel />}
      {tab === "grid" && <GridPanel />}
    </div>
  );
}
