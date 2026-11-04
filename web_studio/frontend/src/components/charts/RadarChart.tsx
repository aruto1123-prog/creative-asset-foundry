// 多軸レーダーチャート（0〜100点）。比較系列（処方適用前など）を破線で重ね描きできる。

export interface RadarAxis {
  key: string;
  label: string;
  value: number;
  color?: string;
}

const GRID_LEVELS = [20, 40, 60, 80, 100];

export function RadarChart({
  axes,
  compare,
  compareLabel = "比較",
  size = 300,
  max = 100,
}: {
  axes: RadarAxis[];
  compare?: Record<string, number> | null;
  compareLabel?: string;
  size?: number;
  max?: number;
}) {
  const count = Math.max(3, axes.length);
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.34;
  const angleOf = (index: number) => -Math.PI / 2 + (index / count) * Math.PI * 2;
  const point = (index: number, value: number) => {
    const ratio = Math.max(0, Math.min(1, value / max));
    const angle = angleOf(index);
    return { x: cx + Math.cos(angle) * radius * ratio, y: cy + Math.sin(angle) * radius * ratio };
  };
  const polygon = (values: number[]) =>
    values
      .map((value, index) => {
        const p = point(index, value);
        return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      })
      .join(" ");
  const mainValues = axes.map((axis) => axis.value);
  const compareValues = compare ? axes.map((axis) => compare[axis.key] ?? 0) : null;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="5軸スコアのレーダーチャート">
      {GRID_LEVELS.map((level) => (
        <polygon
          key={level}
          points={polygon(axes.map(() => (level / 100) * max))}
          fill="none"
          stroke={level === 100 ? "#454d66" : "#232838"}
          strokeWidth={level === 100 ? 1.2 : 1}
        />
      ))}
      {axes.map((axis, index) => {
        const outer = point(index, max);
        return <line key={`spoke-${axis.key}`} x1={cx} y1={cy} x2={outer.x} y2={outer.y} stroke="#232838" />;
      })}
      {GRID_LEVELS.slice(0, 4).map((level) => {
        const p = point(0, (level / 100) * max);
        return (
          <text key={`tick-${level}`} x={p.x + 4} y={p.y + 3} fontSize={9} fill="#6b7493">
            {level}
          </text>
        );
      })}
      {compareValues && (
        <polygon points={polygon(compareValues)} fill="rgba(153,161,189,0.12)" stroke="#99a1bd" strokeWidth={1.5} strokeDasharray="5 4">
          <title>{compareLabel}</title>
        </polygon>
      )}
      <polygon points={polygon(mainValues)} fill="rgba(124,156,255,0.22)" stroke="#7c9cff" strokeWidth={2} />
      {axes.map((axis, index) => {
        const p = point(index, axis.value);
        const label = point(index, max * 1.2);
        const anchor = Math.abs(label.x - cx) < 6 ? "middle" : label.x > cx ? "start" : "end";
        return (
          <g key={axis.key}>
            <circle cx={p.x} cy={p.y} r={4} fill={axis.color ?? "#7c9cff"} stroke="#07080c" strokeWidth={1.5}>
              <title>{`${axis.label}: ${axis.value.toFixed(1)}点`}</title>
            </circle>
            <text x={label.x} y={label.y - 4} textAnchor={anchor} fontSize={11} fontWeight={600} fill={axis.color ?? "#c5cbe0"}>
              {axis.label}
            </text>
            <text x={label.x} y={label.y + 10} textAnchor={anchor} fontSize={10} fill="#99a1bd" fontFamily="monospace">
              {axis.value.toFixed(1)}
              {compareValues ? `（${compareLabel} ${(compare?.[axis.key] ?? 0).toFixed(1)}）` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
