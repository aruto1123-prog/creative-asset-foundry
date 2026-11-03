// 共起 NPMI ヒートマップ（発散配色・ホバーで値と共起数）。

import { useState } from "react";

function colorFor(value: number): string {
  const v = Math.max(-1, Math.min(1, value));
  if (v >= 0) {
    const alpha = 0.12 + v * 0.88;
    return `rgba(244, 114, 182, ${alpha.toFixed(3)})`;
  }
  const alpha = 0.12 + -v * 0.88;
  return `rgba(56, 189, 248, ${alpha.toFixed(3)})`;
}

export function Heatmap({ labels, values, counts }: { labels: string[]; values: number[][]; counts?: number[][] }) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const n = labels.length;
  const cell = n > 20 ? 18 : n > 12 ? 24 : 30;
  const labelSpace = 130;
  const width = labelSpace + n * cell + 10;
  const height = labelSpace + n * cell + 10;
  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-auto rounded-lg border border-ink-700 bg-ink-950">
        <svg width={width} height={height} role="img" aria-label="共起ヒートマップ">
          {labels.map((label, j) => (
            <text
              key={`col-${label}`}
              x={labelSpace + j * cell + cell / 2}
              y={labelSpace - 6}
              fontSize={10}
              fill={hover?.j === j ? "#e6e9f3" : "#99a1bd"}
              transform={`rotate(-50 ${labelSpace + j * cell + cell / 2} ${labelSpace - 6})`}
            >
              {label.length > 18 ? `${label.slice(0, 18)}…` : label}
            </text>
          ))}
          {labels.map((label, i) => (
            <text key={`row-${label}`} x={labelSpace - 6} y={labelSpace + i * cell + cell / 2 + 3} fontSize={10} textAnchor="end" fill={hover?.i === i ? "#e6e9f3" : "#99a1bd"}>
              {label.length > 18 ? `${label.slice(0, 18)}…` : label}
            </text>
          ))}
          {values.map((row, i) =>
            row.map((value, j) => (
              <rect
                key={`${i}-${j}`}
                x={labelSpace + j * cell}
                y={labelSpace + i * cell}
                width={cell - 1}
                height={cell - 1}
                fill={i === j ? "#181c28" : colorFor(value)}
                stroke={hover && hover.i === i && hover.j === j ? "#e6e9f3" : "none"}
                onMouseEnter={() => setHover({ i, j })}
                onMouseLeave={() => setHover(null)}
              />
            )),
          )}
        </svg>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-[11px] text-ink-400">
        <span className="flex items-center gap-1">
          <span className="h-3 w-6 rounded" style={{ background: colorFor(-1) }} />負の相関
        </span>
        <span className="flex items-center gap-1">
          <span className="h-3 w-6 rounded" style={{ background: colorFor(1) }} />正の相関（癒着の疑い）
        </span>
        {hover && hover.i !== hover.j && (
          <span className="font-mono text-ink-200">
            {labels[hover.i]} × {labels[hover.j]}: NPMI {values[hover.i][hover.j].toFixed(3)}
            {counts && ` ／ 共起 ${counts[hover.i]?.[hover.j] ?? 0} 枚`}
          </span>
        )}
      </div>
    </div>
  );
}
