// 推定劇的強度（視覚的手がかりから逆解析）と目標緊張曲線の比較チャート。ページ境界とコマ位置を表示する。

import type { IntensityCurve, PanelMetrics } from "../../api/types";

const WIDTH = 920;
const HEIGHT = 240;
const PAD_LEFT = 42;
const PAD_RIGHT = 16;
const TOP = 22;
const BOTTOM = 42;

function pathOf(ts: number[], values: number[], xOf: (t: number) => number, yOf: (v: number) => number): string {
  return values.map((value, index) => `${index === 0 ? "M" : "L"}${xOf(ts[index] ?? 0).toFixed(1)},${yOf(value).toFixed(1)}`).join(" ");
}

export function IntensityCurveChart({
  curve,
  panels,
  selectedPage,
  onSelectPage,
}: {
  curve: IntensityCurve;
  panels: PanelMetrics[];
  selectedPage?: number;
  onSelectPage?: (page: number) => void;
}) {
  const plotW = WIDTH - PAD_LEFT - PAD_RIGHT;
  const plotH = HEIGHT - TOP - BOTTOM;
  const xOf = (t: number) => PAD_LEFT + Math.min(1, Math.max(0, t)) * plotW;
  const yOf = (v: number) => TOP + (1 - Math.min(1, Math.max(0, v))) * plotH;
  const pageOfPanel = new Map(panels.map((panel) => [panel.panel_id, panel.page_index]));
  const ordered = [...panels].sort((a, b) => a.t - b.t);
  const pageBands: Array<{ page: number; start: number; end: number }> = [];
  for (const panel of ordered) {
    const last = pageBands[pageBands.length - 1];
    if (last && last.page === panel.page_index) {
      last.end = panel.t;
    } else {
      pageBands.push({ page: panel.page_index, start: panel.t, end: panel.t });
    }
  }
  const bands = pageBands.map((band, index) => {
    const previous = pageBands[index - 1];
    const next = pageBands[index + 1];
    const start = previous ? (previous.end + band.start) / 2 : 0;
    const end = next ? (band.end + next.start) / 2 : 1;
    return { page: band.page, start, end };
  });
  const inferredPath = pathOf(curve.t, curve.inferred, xOf, yOf);
  const targetPath = curve.target ? pathOf(curve.t, curve.target, xOf, yOf) : null;
  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full" role="img" aria-label="推定強度と目標曲線の比較">
      <rect x={0} y={0} width={WIDTH} height={HEIGHT} rx={10} fill="#0d0f16" />
      {bands.map((band) => (
        <g
          key={`band-${band.page}-${band.start}`}
          onClick={() => onSelectPage?.(band.page)}
          className={onSelectPage ? "cursor-pointer" : undefined}
        >
          <rect
            x={xOf(band.start)}
            y={TOP}
            width={Math.max(1, xOf(band.end) - xOf(band.start))}
            height={plotH}
            fill={band.page === selectedPage ? "rgba(124,156,255,0.16)" : band.page % 2 === 0 ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0)"}
          />
          <text x={(xOf(band.start) + xOf(band.end)) / 2} y={HEIGHT - BOTTOM + 16} textAnchor="middle" fontSize={10} fill={band.page === selectedPage ? "#a9bcff" : "#6b7493"}>
            p{band.page + 1}
          </text>
          <title>{`${band.page + 1}ページ（クリックで表示）`}</title>
        </g>
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={`grid-${v}`}>
          <line x1={PAD_LEFT} x2={WIDTH - PAD_RIGHT} y1={yOf(v)} y2={yOf(v)} stroke="#232838" strokeDasharray={v === 0 || v === 1 ? undefined : "3 4"} />
          <text x={PAD_LEFT - 6} y={yOf(v) + 3} textAnchor="end" fontSize={9} fill="#6b7493">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      {targetPath && <path d={targetPath} fill="none" stroke="#38bdf8" strokeWidth={2} strokeDasharray="7 5" />}
      <path d={inferredPath} fill="none" stroke="#f472b6" strokeWidth={2.2} />
      {curve.inferred.map((value, index) => {
        const panelId = curve.panel_ids[index];
        const page = panelId !== undefined ? pageOfPanel.get(panelId) : undefined;
        return (
          <circle
            key={`pt-${index}`}
            cx={xOf(curve.t[index] ?? 0)}
            cy={yOf(value)}
            r={page === selectedPage ? 3.6 : 2.6}
            fill={page === selectedPage ? "#fde68a" : "#f472b6"}
            onClick={() => (page !== undefined ? onSelectPage?.(page) : undefined)}
            className={onSelectPage ? "cursor-pointer" : undefined}
          >
            <title>{`${panelId ?? `#${index + 1}`}: 推定強度 ${value.toFixed(2)}${curve.target ? ` / 目標 ${(curve.target[index] ?? 0).toFixed(2)}` : ""}`}</title>
          </circle>
        );
      })}
      <g transform={`translate(${PAD_LEFT + 8}, ${HEIGHT - 12})`}>
        <line x1={0} x2={22} y1={0} y2={0} stroke="#f472b6" strokeWidth={2.2} />
        <text x={28} y={3} fontSize={10} fill="#c5cbe0">
          推定劇的強度（コマの大きさ・ショット・ベタ・描き文字などから逆解析）
        </text>
        {targetPath && (
          <>
            <line x1={430} x2={452} y1={0} y2={0} stroke="#38bdf8" strokeWidth={2} strokeDasharray="7 5" />
            <text x={458} y={3} fontSize={10} fill="#c5cbe0">
              目標緊張曲線
            </text>
          </>
        )}
      </g>
    </svg>
  );
}
