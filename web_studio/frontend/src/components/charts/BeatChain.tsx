// Layer 3 ビート因果チェーン（時間軸上の配置・因果アーク・転換点強調）。

import type { StructuralBeat } from "../../api/types";
import { BEAT_TYPE_LABELS } from "../../lib/labels";

const WIDTH = 920;
const HEIGHT = 190;
const PAD = 40;
const BASELINE = 112;

export function BeatChain({
  beats,
  selectedId,
  onSelect,
}: {
  beats: StructuralBeat[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const xOf = (t: number) => PAD + Math.min(1, Math.max(0, t)) * (WIDTH - PAD * 2);
  const byId = new Map(beats.map((beat) => [beat.id, beat]));
  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full rounded-lg border border-ink-700 bg-ink-950" role="img" aria-label="ビート因果チェーン">
      <line x1={PAD} x2={WIDTH - PAD} y1={BASELINE} y2={BASELINE} stroke="#2f3549" strokeWidth={2} />
      {beats.flatMap((beat) =>
        beat.causes.map((causeId) => {
          const cause = byId.get(causeId);
          if (!cause) {
            return null;
          }
          const x1 = xOf(cause.t);
          const x2 = xOf(beat.t);
          const lift = Math.min(90, 18 + Math.abs(x2 - x1) * 0.35);
          return (
            <path
              key={`${causeId}-${beat.id}`}
              d={`M${x1},${BASELINE - 8} Q${(x1 + x2) / 2},${BASELINE - 8 - lift} ${x2},${BASELINE - 8}`}
              fill="none"
              stroke={selectedId === beat.id || selectedId === causeId ? "#a9bcff" : "#454d66"}
              strokeWidth={1.5}
              markerEnd="url(#beat-arrow)"
            />
          );
        }),
      )}
      <defs>
        <marker id="beat-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#6b7493" />
        </marker>
      </defs>
      {beats.map((beat, index) => {
        const x = xOf(beat.t);
        const selected = beat.id === selectedId;
        const color = beat.is_turning_point ? "#a9bcff" : "#99a1bd";
        const label = BEAT_TYPE_LABELS[beat.beat_type] ?? beat.beat_type;
        const below = index % 2 === 0;
        return (
          <g key={beat.id} className="cursor-pointer" onClick={() => onSelect(beat.id)}>
            {beat.is_turning_point ? (
              <rect x={x - 7} y={BASELINE - 7} width={14} height={14} transform={`rotate(45 ${x} ${BASELINE})`} fill={selected ? "#7c9cff" : "#2a3767"} stroke={color} strokeWidth={2} />
            ) : (
              <circle cx={x} cy={BASELINE} r={selected ? 7 : 5.5} fill={selected ? "#7c9cff" : "#181c28"} stroke={color} strokeWidth={2} />
            )}
            <text x={x} y={below ? BASELINE + 26 : BASELINE + 42} textAnchor="middle" fontSize={10} fill={selected ? "#e6e9f3" : color}>
              {label}
            </text>
            <text x={x} y={below ? BASELINE + 38 : BASELINE + 54} textAnchor="middle" fontSize={9} fill="#6b7493">
              t={beat.t.toFixed(2)}
            </text>
            <title>{`${label}: ${beat.label}\n${beat.function}`}</title>
          </g>
        );
      })}
    </svg>
  );
}
