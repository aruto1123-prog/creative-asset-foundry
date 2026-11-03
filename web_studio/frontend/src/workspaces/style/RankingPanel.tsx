// Elo / Bradley-Terry ランキングと美的選好ベクトル。

import { api } from "../../api/client";
import type { AestheticPreferenceVector, RatingEntry } from "../../api/types";
import { PreferenceBars } from "../../components/charts/PreferenceBars";
import { Badge, Button, Card, EmptyState, Spinner, Stat } from "../../components/ui";
import { useResource } from "../../hooks/useAsync";
import { fmt, pct } from "../../lib/format";

const MIN_COMPARISONS_FOR_VECTOR = 3;

interface RankingBundle {
  ratings: RatingEntry[];
  nCandidates: number;
  nComparisons: number;
  vector: AestheticPreferenceVector | null;
}

export function RankingPanel({ version }: { version: number }) {
  const bundle = useResource<RankingBundle>(async () => {
    const ranking = await api.style.rankingsFull();
    const ready = ranking.n_candidates >= 2 && ranking.n_comparisons >= MIN_COMPARISONS_FOR_VECTOR;
    const vectorData = ready ? await api.style.preferenceVector() : null;
    return { ratings: ranking.ratings, nCandidates: ranking.n_candidates, nComparisons: ranking.n_comparisons, vector: vectorData };
  }, [version]);
  const rankings = { data: bundle.data?.ratings ?? null, loading: bundle.loading, error: bundle.error, reload: bundle.reload };
  const vector = {
    data: bundle.data?.vector ?? null,
    loading: bundle.loading,
    error:
      bundle.error ??
      (bundle.data && bundle.data.vector === null
        ? `比較が ${bundle.data.nComparisons} 件です（候補2件以上・比較${MIN_COMPARISONS_FOR_VECTOR}件以上で推定可能）`
        : null),
    reload: bundle.reload,
  };
  const maxElo = Math.max(1600, ...(rankings.data ?? []).map((entry) => entry.bt_elo_scale));
  const minElo = Math.min(1400, ...(rankings.data ?? []).map((entry) => entry.bt_elo_scale));

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Elo / Bradley-Terry ランキング"
        subtitle="Bradley-Terry 最尤推定（MM 法＋正則化）の強度順。Elo は判定の時系列逐次更新。"
        actions={
          <Button size="sm" variant="ghost" onClick={rankings.reload} loading={rankings.loading}>
            更新
          </Button>
        }
      >
        {rankings.loading && !rankings.data && <Spinner />}
        {rankings.error && <EmptyState title="ランキングを取得できません" hint={rankings.error} />}
        {rankings.data && rankings.data.length === 0 && <EmptyState title="候補がありません" hint="アリーナで候補を登録・判定してください。" />}
        {rankings.data && rankings.data.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-ink-400">
                <tr>
                  <th className="py-1.5 pr-3">順位</th>
                  <th className="py-1.5 pr-3">候補</th>
                  <th className="py-1.5 pr-3">BT Elo 換算 ± SE</th>
                  <th className="py-1.5 pr-3">BT 強度</th>
                  <th className="py-1.5 pr-3">Elo</th>
                  <th className="py-1.5 pr-3">対戦</th>
                  <th className="py-1.5 pr-3">勝/敗/分</th>
                  <th className="py-1.5">勝率</th>
                </tr>
              </thead>
              <tbody>
                {rankings.data.map((entry) => {
                  const ratio = (entry.bt_elo_scale - minElo) / Math.max(1, maxElo - minElo);
                  return (
                    <tr key={entry.candidate_id} className="border-t border-ink-800 text-ink-200">
                      <td className="py-1.5 pr-3 font-mono">{entry.rank}</td>
                      <td className="py-1.5 pr-3">{entry.label}</td>
                      <td className="py-1.5 pr-3">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-ink-700">
                            <div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(2, ratio * 100)}%` }} />
                          </div>
                          <span className="font-mono">
                            {entry.bt_elo_scale.toFixed(0)} ± {(400 * entry.bt_std_error / Math.LN10).toFixed(0)}
                          </span>
                        </div>
                      </td>
                      <td className="py-1.5 pr-3 font-mono">{fmt(entry.bt_strength, 3)}</td>
                      <td className="py-1.5 pr-3 font-mono">{entry.elo.toFixed(0)}</td>
                      <td className="py-1.5 pr-3 font-mono">{entry.games}</td>
                      <td className="py-1.5 pr-3 font-mono">
                        {entry.wins}/{entry.losses}/{entry.ties}
                      </td>
                      <td className="py-1.5 font-mono">{pct(entry.win_rate)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        title="美的選好ベクトル（Aesthetic Preference Vector）"
        subtitle="実験パラメータを説明変数、判定結果を目的変数とした正則化回帰の標準化係数。ひげは 95% ブートストラップ信頼区間。"
        actions={
          <Button size="sm" variant="ghost" onClick={vector.reload} loading={vector.loading}>
            再推定
          </Button>
        }
      >
        {vector.loading && !vector.data && <Spinner />}
        {vector.error && <EmptyState title="選好ベクトルはまだ推定できません" hint={`${vector.error} — 候補間の判定を増やしてください。`} />}
        {vector.data && (
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
              <Stat label="推定法" value={vector.data.method === "pairwise_logistic" ? "ペアワイズ・ロジスティック" : "BT リッジ"} />
              <Stat label="正則化 λ" value={fmt(vector.data.regularization, 4)} />
              <Stat label="ペア予測一致率" value={pct(vector.data.fit_accuracy, 1)} />
              <Stat label="比較数" value={vector.data.n_comparisons} />
              <Stat label="候補数" value={vector.data.n_candidates} />
            </div>
            <PreferenceBars features={vector.data.features} />
            <div className="flex flex-wrap gap-2 text-xs">
              {vector.data.features
                .filter((feature) => feature.direction !== "neutral")
                .sort((a, b) => b.importance - a.importance)
                .slice(0, 6)
                .map((feature) => (
                  <Badge key={feature.name} tone={feature.direction === "prefer_higher" ? "success" : "danger"}>
                    {feature.name}: {feature.direction === "prefer_higher" ? "高い値を好む" : "低い値を好む"}
                  </Badge>
                ))}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
