// 美的選好ベクトル: 標準化係数の発散型横棒＋95%信頼区間ひげ・方向バッジ・重要度。

import type { FeatureWeight } from "../../api/types";

const ROW = 30;
const LABEL_W = 250;
const WIDTH = 900;
const BAR_W = 460;

function groupColor(name: string): string {
  if (name.startsWith("style.")) {
    return "#f472b6";
  }
  if (name.startsWith("merge.")) {
    return "#7c9cff";
  }
  if (name.startsWith("lora.")) {
    return "#34d399";
  }
  return "#fbbf24";
}

export function PreferenceBars({ features }: { features: FeatureWeight[] }) {
  const sorted = [...features].sort((a, b) => b.importance - a.importance);
  const extent = Math.max(0.05, ...sorted.map((f) => Math.max(Math.abs(f.ci_low), Math.abs(f.ci_high), Math.abs(f.std_coefficient)))) * 1.1;
  const height = sorted.length * ROW + 36;
  const center = LABEL_W + BAR_W / 2;
  const xOf = (value: number) => center + (value / extent) * (BAR_W / 2);
  return (
    <svg viewBox={`0 0 ${WIDTH} ${height}`} className="w-full rounded-lg border border-ink-700 bg-ink-950" role="img" aria-label="美的選好ベクトル">
      <line x1={center} x2={center} y1={14} y2={height - 18} stroke="#454d66" />
      <text x={LABEL_W} y={12} fontSize={10} fill="#6b7493">
        ← 低い値を好む
      </text>
      <text x={LABEL_W + BAR_W} y={12} fontSize={10} fill="#6b7493" textAnchor="end">
        高い値を好む →
      </text>
      {sorted.map((feature, index) => {
        const y = 24 + index * ROW;
        const value = feature.std_coefficient;
        const color = groupColor(feature.name);
        const x0 = xOf(Math.min(0, value));
        const x1 = xOf(Math.max(0, value));
        return (
          <g key={feature.name}>
            <text x={LABEL_W - 10} y={y + 13} textAnchor="end" fontSize={11} fill="#c5cbe0" fontFamily="monospace">
              {feature.name.length > 32 ? `${feature.name.slice(0, 32)}…` : feature.name}
            </text>
            <rect x={x0} y={y + 3} width={Math.max(1, x1 - x0)} height={14} rx={3} fill={color} fillOpacity={feature.direction === "neutral" ? 0.3 : 0.75} />
            <line x1={xOf(feature.ci_low)} x2={xOf(feature.ci_high)} y1={y + 10} y2={y + 10} stroke="#e6e9f3" strokeWidth={1.4} />
            <line x1={xOf(feature.ci_low)} x2={xOf(feature.ci_low)} y1={y + 5} y2={y + 15} stroke="#e6e9f3" strokeWidth={1.4} />
            <line x1={xOf(feature.ci_high)} x2={xOf(feature.ci_high)} y1={y + 5} y2={y + 15} stroke="#e6e9f3" strokeWidth={1.4} />
            <text x={LABEL_W + BAR_W + 14} y={y + 13} fontSize={10.5} fill={feature.direction === "prefer_higher" ? "#6ee7b7" : feature.direction === "prefer_lower" ? "#fda4af" : "#99a1bd"}>
              {feature.direction === "prefer_higher" ? "高を好む" : feature.direction === "prefer_lower" ? "低を好む" : "中立"}
            </text>
            <text x={LABEL_W + BAR_W + 80} y={y + 13} fontSize={10.5} fill="#99a1bd" fontFamily="monospace">
              β={value.toFixed(2)}  重要度 {(feature.importance * 100).toFixed(1)}%
            </text>
            <title>{`${feature.name}: 標準化係数 ${value.toFixed(3)}（95%CI ${feature.ci_low.toFixed(3)}〜${feature.ci_high.toFixed(3)}）, 置換重要度 ${feature.permutation_importance.toFixed(3)}`}</title>
          </g>
        );
      })}
    </svg>
  );
}
