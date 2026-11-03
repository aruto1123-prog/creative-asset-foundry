// 感情曲線と同期したページ／コマ・タイムライン（めくり・引き・見開きの表示、クリックでページ選択）。

import type { Storyboard } from "../../api/types";

const WIDTH = 920;
const HEIGHT = 226;
const PAD_LEFT = 40;
const PAD_RIGHT = 14;
const TOP = 54;
const BOTTOM = 40;

export function MangaTimeline({
  storyboard,
  selectedPage,
  onSelectPage,
}: {
  storyboard: Storyboard;
  selectedPage: number;
  onSelectPage: (index: number) => void;
}) {
  const pages = storyboard.pages;
  const count = Math.max(1, pages.length);
  const plotW = WIDTH - PAD_LEFT - PAD_RIGHT;
  const plotH = HEIGHT - TOP - BOTTOM;
  const xOf = (t: number) => PAD_LEFT + Math.min(1, Math.max(0, t)) * plotW;
  const yOf = (v: number) => TOP + (1 - Math.min(1, Math.max(0, v))) * plotH;
  const bands = pages.map((page, index) => {
    const start = typeof page.t_start === "number" ? page.t_start : index / count;
    const end = typeof page.t_end === "number" && page.t_end > start ? page.t_end : (index + 1) / count;
    return { page, start, end };
  });
  const panels = pages.flatMap((page) => page.panels.map((panel) => ({ panel, page })));
  const sortedPanels = [...panels].sort((a, b) => a.panel.t - b.panel.t);
  const curve = sortedPanels.map(({ panel }, i) => `${i === 0 ? "M" : "L"}${xOf(panel.t).toFixed(1)},${yOf(panel.tension).toFixed(1)}`).join(" ");
  const spreadPages = new Set<number>();
  for (const spread of storyboard.spreads) {
    spreadPages.add(spread.right_page);
    spreadPages.add(spread.left_page);
  }
  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full rounded-lg border border-ink-700 bg-ink-950" role="img" aria-label="ページ・コマタイムライン">
      {bands.map(({ page, start, end }) => {
        const selected = page.index === selectedPage;
        const x = xOf(start);
        const w = Math.max(2, xOf(end) - xOf(start));
        return (
          <g key={page.index} className="cursor-pointer" onClick={() => onSelectPage(page.index)}>
            <rect x={x} y={TOP} width={w} height={plotH} fill={selected ? "#2a3767" : page.side === "right" ? "#12151f" : "#0d0f16"} stroke={selected ? "#7c9cff" : "#232838"} />
            <text x={x + w / 2} y={HEIGHT - BOTTOM + 14} textAnchor="middle" fontSize={10} fill={selected ? "#e6e9f3" : "#99a1bd"}>
              {page.number}
            </text>
            <text x={x + w / 2} y={HEIGHT - BOTTOM + 26} textAnchor="middle" fontSize={8.5} fill="#6b7493">
              {page.side === "right" ? "右" : "左"}
            </text>
            {page.page_turn_role === "reveal" && (
              <text x={x + w / 2} y={TOP - 18} textAnchor="middle" fontSize={9} fill="#fbbf24">
                めくり
              </text>
            )}
            {page.page_turn_role === "hook" && (
              <text x={x + w / 2} y={TOP - 18} textAnchor="middle" fontSize={9} fill="#f472b6">
                引き
              </text>
            )}
            {spreadPages.has(page.index) && <rect x={x} y={TOP - 10} width={w} height={4} fill="#a78bfa" rx={2} />}
            <title>{`${page.number}ページ（${page.side === "right" ? "右" : "左"}）: コマ ${page.panels.length}／情報密度 ${page.information_density.toFixed(2)}／平均緊張 ${page.tension_mean.toFixed(2)}`}</title>
          </g>
        );
      })}
      {[0, 0.5, 1].map((tick) => (
        <text key={tick} x={PAD_LEFT - 6} y={yOf(tick) + 3} textAnchor="end" fontSize={9} fill="#6b7493">
          {tick.toFixed(1)}
        </text>
      ))}
      <path d={curve} fill="none" stroke="#f472b6" strokeWidth={2.2} pointerEvents="none" />
      {sortedPanels.map(({ panel }) => (
        <circle
          key={panel.id}
          cx={xOf(panel.t)}
          cy={yOf(panel.tension)}
          r={panel.emphasis === "normal" ? 2.6 : 4.5}
          fill={panel.emphasis === "normal" ? "#f9a8d4" : "#fbbf24"}
          pointerEvents="none"
        />
      ))}
      <text x={PAD_LEFT} y={16} fontSize={10} fill="#99a1bd">
        緊張度（コマ単位）とページ帯 ― 紫帯: 見開き ／ 黄点: 大ゴマ・見開きゴマ
      </text>
    </svg>
  );
}
