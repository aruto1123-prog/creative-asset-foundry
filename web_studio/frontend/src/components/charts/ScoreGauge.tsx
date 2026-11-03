// 0〜100 点の半円ゲージ（閾値マーカー付き）とスコア推移スパークライン。

export function ScoreGauge({ score, threshold = 80, label, size = 160 }: { score: number; threshold?: number; label?: string; size?: number }) {
  const value = Math.max(0, Math.min(100, score));
  const r = size * 0.38;
  const cx = size / 2;
  const cy = size * 0.55;
  const polar = (v: number) => {
    const angle = Math.PI * (1 - v / 100);
    return { x: cx + r * Math.cos(angle), y: cy - r * Math.sin(angle) };
  };
  const start = polar(0);
  const end = polar(value);
  const largeArc = value > 50 ? 1 : 0;
  const color = value >= threshold ? "#34d399" : value >= threshold - 15 ? "#fbbf24" : "#fb7185";
  const tick = polar(threshold);
  const tickInner = { x: cx + (tick.x - cx) * 0.82, y: cy + (tick.y - cy) * 0.82 };
  return (
    <svg width={size} height={size * 0.68} viewBox={`0 0 ${size} ${size * 0.68}`} role="img" aria-label={`${label ?? "スコア"} ${value.toFixed(1)}点`}>
      <path d={`M${start.x},${start.y} A${r},${r} 0 1 1 ${polar(100).x},${polar(100).y}`} fill="none" stroke="#232838" strokeWidth={size * 0.07} strokeLinecap="round" />
      {value > 0 && (
        <path d={`M${start.x},${start.y} A${r},${r} 0 ${largeArc} 1 ${end.x},${end.y}`} fill="none" stroke={color} strokeWidth={size * 0.07} strokeLinecap="round" />
      )}
      <line x1={tickInner.x} y1={tickInner.y} x2={tick.x + (tick.x - cx) * 0.12} y2={tick.y + (tick.y - cy) * 0.12} stroke="#e6e9f3" strokeWidth={2} />
      <text x={cx} y={cy - 4} textAnchor="middle" fontSize={size * 0.17} fontWeight={700} fill="#e6e9f3">
        {value.toFixed(1)}
      </text>
      <text x={cx} y={cy + size * 0.1} textAnchor="middle" fontSize={size * 0.07} fill="#99a1bd">
        {label ?? "スコア"}（基準 {threshold}）
      </text>
    </svg>
  );
}

export function Sparkline({ values, width = 220, height = 56, color = "#7c9cff", min = 0, max = 100 }: { values: number[]; width?: number; height?: number; color?: string; min?: number; max?: number }) {
  if (values.length === 0) {
    return null;
  }
  const xOf = (i: number) => 6 + (values.length === 1 ? (width - 12) / 2 : (i / (values.length - 1)) * (width - 12));
  const yOf = (v: number) => 6 + (1 - (v - min) / Math.max(1e-6, max - min)) * (height - 12);
  const d = values.map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  return (
    <svg width={width} height={height} role="img" aria-label="スコア推移">
      <rect x={0} y={0} width={width} height={height} rx={6} fill="#0d0f16" stroke="#232838" />
      <path d={d} fill="none" stroke={color} strokeWidth={2} />
      {values.map((v, i) => (
        <g key={i}>
          <circle cx={xOf(i)} cy={yOf(v)} r={3} fill={color} />
          <title>{`${i + 1}回目: ${v.toFixed(1)}`}</title>
        </g>
      ))}
    </svg>
  );
}
