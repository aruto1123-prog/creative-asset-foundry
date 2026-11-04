// ページ別のスコア・情報密度・推定強度をヒートセルで並べる帯（クリックでページ選択）。

import type { PageMetrics } from "../../api/types";
import { DENSITY_LEVEL_LABELS, labelOf } from "../../lib/labels";

const CELL = 34;
const ROW_H = 22;
const LABEL_W = 92;
const TOP = 18;

function scoreColor(score: number): string {
  if (score >= 85) {
    return "#047857";
  }
  if (score >= 70) {
    return "#2563eb";
  }
  if (score >= 55) {
    return "#b45309";
  }
  return "#be123c";
}

function rampColor(value: number, hue: number): string {
  const v = Math.max(0, Math.min(1, value));
  const lightness = 14 + v * 46;
  return `hsl(${hue}, 72%, ${lightness.toFixed(0)}%)`;
}

const DENSITY_COLORS: Record<string, string> = {
  sparse: "#1e3a5f",
  balanced: "#166534",
  dense: "#a16207",
  overloaded: "#9f1239",
};

export function PageHeatStrip({
  pages,
  selected,
  onSelect,
}: {
  pages: PageMetrics[];
  selected: number;
  onSelect: (page: number) => void;
}) {
  const width = LABEL_W + Math.max(1, pages.length) * CELL + 8;
  const height = TOP + ROW_H * 4 + 10;
  const rows: Array<{ key: string; label: string }> = [
    { key: "score", label: "ページ得点" },
    { key: "density", label: "情報密度" },
    { key: "intensity", label: "平均強度" },
    { key: "peak", label: "最大強度" },
  ];
  return (
    <div className="overflow-x-auto">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="ページ別ヒートストリップ">
        {rows.map((row, rowIndex) => (
          <text key={row.key} x={LABEL_W - 8} y={TOP + rowIndex * ROW_H + 15} textAnchor="end" fontSize={10} fill="#99a1bd">
            {row.label}
          </text>
        ))}
        {pages.map((page, index) => {
          const x = LABEL_W + index * CELL;
          const isSelected = page.page_index === selected;
          const role = page.page_turn_role === "reveal" ? "め" : page.page_turn_role === "hook" ? "引" : "";
          return (
            <g key={page.page_index} onClick={() => onSelect(page.page_index)} className="cursor-pointer">
              <text x={x + CELL / 2} y={TOP - 5} textAnchor="middle" fontSize={10} fill={isSelected ? "#a9bcff" : "#6b7493"}>
                {page.page_index + 1}
                {page.side === "right" ? "右" : "左"}
                {role}
              </text>
              <rect x={x + 1} y={TOP} width={CELL - 2} height={ROW_H - 2} rx={3} fill={scoreColor(page.score)} />
              <text x={x + CELL / 2} y={TOP + 14} textAnchor="middle" fontSize={10} fill="#f8fafc" fontFamily="monospace">
                {page.score.toFixed(0)}
              </text>
              <rect x={x + 1} y={TOP + ROW_H} width={CELL - 2} height={ROW_H - 2} rx={3} fill={DENSITY_COLORS[page.density_level] ?? "#334155"} />
              <text x={x + CELL / 2} y={TOP + ROW_H + 14} textAnchor="middle" fontSize={10} fill="#f8fafc">
                {labelOf(DENSITY_LEVEL_LABELS, page.density_level)}
              </text>
              <rect x={x + 1} y={TOP + ROW_H * 2} width={CELL - 2} height={ROW_H - 2} rx={3} fill={rampColor(page.mean_intensity, 330)} />
              <rect x={x + 1} y={TOP + ROW_H * 3} width={CELL - 2} height={ROW_H - 2} rx={3} fill={rampColor(page.peak_intensity, 280)} />
              {isSelected && <rect x={x} y={TOP - 1} width={CELL} height={ROW_H * 4} rx={4} fill="none" stroke="#a9bcff" strokeWidth={2} />}
              <title>
                {`${page.page_index + 1}ページ（${page.side === "right" ? "右" : "左"}・${page.page_turn_role}）: 得点 ${page.score.toFixed(1)} / ${page.panel_count}コマ / ${page.char_count}字 / 密度 ${labelOf(DENSITY_LEVEL_LABELS, page.density_level)} / 平均強度 ${page.mean_intensity.toFixed(2)} / 逆行 ${page.backtracks} / 交差 ${page.crossings}`}
              </title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
