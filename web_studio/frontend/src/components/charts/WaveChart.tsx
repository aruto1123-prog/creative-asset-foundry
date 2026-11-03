// 感情・情報開示曲線のインタラクティブ SVG チャート（チャネル切替・ドラッグ編集・ビートマーカー・重ね描き）。

import { useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { BaseWaveChannel, EmotionalInformationWave, StructuralBeat, WaveChannel } from "../../api/types";
import { ALL_WAVE_CHANNELS, BASE_WAVE_CHANNELS, BEAT_TYPE_LABELS, WAVE_CHANNEL_COLORS, WAVE_CHANNEL_LABELS } from "../../lib/labels";
import { channelValues, clamp } from "../../lib/format";
import { cx } from "../ui";

export interface WaveOverlay {
  label: string;
  wave: EmotionalInformationWave;
  color: string;
  dash?: string;
}

interface WaveChartProps {
  wave: EmotionalInformationWave;
  beats?: StructuralBeat[];
  editable?: boolean;
  onChange?: (wave: EmotionalInformationWave) => void;
  overlays?: WaveOverlay[];
  overlayChannel?: WaveChannel;
  height?: number;
  initialChannels?: WaveChannel[];
  selectedBeatId?: string | null;
  onBeatSelect?: (beatId: string) => void;
  title?: string;
}

const WIDTH = 920;
const PAD_LEFT = 46;
const PAD_RIGHT = 18;
const PAD_TOP = 26;
const PAD_BOTTOM = 32;

function makeScales(height: number) {
  const plotW = WIDTH - PAD_LEFT - PAD_RIGHT;
  const plotH = height - PAD_TOP - PAD_BOTTOM;
  const xOf = (t: number) => PAD_LEFT + clamp(t) * plotW;
  const yOfUnit = (v: number) => PAD_TOP + (1 - clamp(v)) * plotH;
  const yOf = (channel: WaveChannel, v: number) => (channel === "information_asymmetry" ? yOfUnit((v + 1) / 2) : yOfUnit(v));
  return { plotW, plotH, xOf, yOfUnit, yOf };
}

function linePath(wave: EmotionalInformationWave, channel: WaveChannel, height: number): string {
  const { xOf, yOf } = makeScales(height);
  const values = channelValues(wave, channel);
  return values
    .map((value, index) => `${index === 0 ? "M" : "L"}${xOf(wave.t[index] ?? index / Math.max(1, values.length - 1)).toFixed(1)},${yOf(channel, value).toFixed(1)}`)
    .join(" ");
}

export function WaveChart({
  wave,
  beats = [],
  editable = false,
  onChange,
  overlays = [],
  overlayChannel = "tension",
  height = 300,
  initialChannels = ["tension", "release", "information_asymmetry", "cognitive_load"],
  selectedBeatId = null,
  onBeatSelect,
  title,
}: WaveChartProps) {
  const [visible, setVisible] = useState<WaveChannel[]>(initialChannels);
  const [editChannel, setEditChannel] = useState<BaseWaveChannel>("tension");
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { plotW, plotH, xOf, yOfUnit } = makeScales(height);

  const paths = useMemo(
    () => ALL_WAVE_CHANNELS.map((channel) => ({ channel, d: linePath(wave, channel, height) })),
    [wave, height],
  );

  const overlayPaths = useMemo(
    () => overlays.map((overlay) => ({ ...overlay, d: linePath(overlay.wave, overlayChannel, height) })),
    [overlays, overlayChannel, height],
  );

  const toSvgPoint = (event: ReactPointerEvent<SVGElement>) => {
    const svg = svgRef.current;
    if (!svg) {
      return null;
    }
    const matrix = svg.getScreenCTM();
    if (!matrix) {
      return null;
    }
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(matrix.inverse());
  };

  const updateValue = (index: number, svgY: number) => {
    if (!onChange) {
      return;
    }
    const value = Math.round(clamp(1 - (svgY - PAD_TOP) / plotH) * 1000) / 1000;
    const nextValues = [...wave[editChannel]];
    nextValues[index] = value;
    const next: EmotionalInformationWave = { ...wave, [editChannel]: nextValues };
    next.information_asymmetry = next.audience_knowledge.map((a, i) => a - next.character_knowledge[i]);
    onChange(next);
  };

  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const point = toSvgPoint(event);
    if (!point) {
      return;
    }
    const n = wave.t.length;
    const tValue = clamp((point.x - PAD_LEFT) / plotW);
    let nearest = 0;
    for (let i = 1; i < n; i += 1) {
      if (Math.abs(wave.t[i] - tValue) < Math.abs(wave.t[nearest] - tValue)) {
        nearest = i;
      }
    }
    setHoverIndex(nearest);
    if (dragIndex !== null) {
      updateValue(dragIndex, point.y);
    }
  };

  const toggleChannel = (channel: WaveChannel) => {
    setVisible((current) => (current.includes(channel) ? current.filter((item) => item !== channel) : [...current, channel]));
  };

  const xTicks = Array.from({ length: 11 }, (_, i) => i / 10);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const editValues = wave[editChannel];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {title && <span className="mr-2 text-xs font-semibold text-ink-300">{title}</span>}
        {ALL_WAVE_CHANNELS.map((channel) => (
          <button
            key={channel}
            type="button"
            onClick={() => toggleChannel(channel)}
            className={cx(
              "flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] transition-colors",
              visible.includes(channel) ? "border-ink-500 bg-ink-800 text-ink-100" : "border-ink-700 text-ink-500",
            )}
          >
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: WAVE_CHANNEL_COLORS[channel] }} />
            {WAVE_CHANNEL_LABELS[channel]}
          </button>
        ))}
        {editable && (
          <span className="ml-auto flex items-center gap-1 text-[11px] text-ink-400">
            編集チャネル:
            {BASE_WAVE_CHANNELS.map((channel) => (
              <button
                key={channel}
                type="button"
                onClick={() => {
                  setEditChannel(channel);
                  setVisible((current) => (current.includes(channel) ? current : [...current, channel]));
                }}
                className={cx(
                  "rounded px-1.5 py-0.5",
                  editChannel === channel ? "bg-accent-soft text-accent-strong" : "text-ink-400 hover:text-ink-200",
                )}
              >
                {WAVE_CHANNEL_LABELS[channel]}
              </button>
            ))}
          </span>
        )}
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${height}`}
        className={cx("w-full select-none rounded-lg border border-ink-700 bg-ink-950", dragIndex !== null && "cursor-ns-resize")}
        onPointerMove={handlePointerMove}
        onPointerUp={() => setDragIndex(null)}
        onPointerLeave={() => {
          setHoverIndex(null);
          setDragIndex(null);
        }}
        role="img"
        aria-label="感情・情報開示曲線"
      >
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line x1={PAD_LEFT} x2={WIDTH - PAD_RIGHT} y1={yOfUnit(tick)} y2={yOfUnit(tick)} stroke="#232838" strokeWidth={1} />
            <text x={PAD_LEFT - 8} y={yOfUnit(tick) + 3} textAnchor="end" fontSize={10} fill="#6b7493">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line x1={xOf(tick)} x2={xOf(tick)} y1={PAD_TOP} y2={height - PAD_BOTTOM} stroke="#181c28" strokeWidth={1} />
            <text x={xOf(tick)} y={height - PAD_BOTTOM + 14} textAnchor="middle" fontSize={10} fill="#6b7493">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={WIDTH - PAD_RIGHT} y={height - 4} textAnchor="end" fontSize={10} fill="#6b7493">
          正規化時間 t
        </text>
        {visible.includes("information_asymmetry") && (
          <line x1={PAD_LEFT} x2={WIDTH - PAD_RIGHT} y1={yOfUnit(0.5)} y2={yOfUnit(0.5)} stroke="#38bdf8" strokeOpacity={0.25} strokeDasharray="4 4" />
        )}
        {beats.map((beat) => {
          const selected = beat.id === selectedBeatId;
          const label = BEAT_TYPE_LABELS[beat.beat_type] ?? beat.beat_type;
          return (
            <g key={beat.id} className="cursor-pointer" onClick={() => onBeatSelect?.(beat.id)}>
              <line
                x1={xOf(beat.t)}
                x2={xOf(beat.t)}
                y1={PAD_TOP - 4}
                y2={height - PAD_BOTTOM}
                stroke={beat.is_turning_point ? "#a9bcff" : "#454d66"}
                strokeWidth={selected ? 2 : 1}
                strokeDasharray={beat.is_turning_point ? "" : "3 4"}
              />
              <text x={xOf(beat.t) + 3} y={PAD_TOP - 10} fontSize={9} fill={beat.is_turning_point ? "#a9bcff" : "#99a1bd"}>
                {label.length > 7 ? `${label.slice(0, 7)}…` : label}
              </text>
              <title>{`${label}（t=${beat.t.toFixed(2)}）: ${beat.label}`}</title>
            </g>
          );
        })}
        {overlayPaths.map((overlay) => (
          <path key={overlay.label} d={overlay.d} fill="none" stroke={overlay.color} strokeWidth={1.6} strokeDasharray={overlay.dash ?? "6 4"} strokeOpacity={0.85} />
        ))}
        {paths
          .filter((path) => visible.includes(path.channel))
          .map((path) => (
            <path
              key={path.channel}
              d={path.d}
              fill="none"
              stroke={WAVE_CHANNEL_COLORS[path.channel]}
              strokeWidth={path.channel === "tension" ? 2.6 : 1.8}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
        {editable &&
          editValues.map((value, index) => (
            <circle
              key={`handle-${index}`}
              cx={xOf(wave.t[index])}
              cy={yOfUnit(value)}
              r={dragIndex === index ? 6.5 : 4.5}
              fill={WAVE_CHANNEL_COLORS[editChannel]}
              stroke="#07080c"
              strokeWidth={1.5}
              className="cursor-ns-resize"
              onPointerDown={(event) => {
                event.preventDefault();
                (event.currentTarget.ownerSVGElement ?? event.currentTarget).setPointerCapture?.(event.pointerId);
                setDragIndex(index);
              }}
            />
          ))}
        {hoverIndex !== null && (
          <g pointerEvents="none">
            <line x1={xOf(wave.t[hoverIndex])} x2={xOf(wave.t[hoverIndex])} y1={PAD_TOP} y2={height - PAD_BOTTOM} stroke="#e6e9f3" strokeOpacity={0.25} />
            <rect x={Math.min(xOf(wave.t[hoverIndex]) + 8, WIDTH - 190)} y={PAD_TOP + 4} width={180} height={14 + visible.length * 13} rx={6} fill="#0d0f16" stroke="#2f3549" />
            <text x={Math.min(xOf(wave.t[hoverIndex]) + 16, WIDTH - 182)} y={PAD_TOP + 16} fontSize={10} fill="#c5cbe0">
              t = {wave.t[hoverIndex].toFixed(2)}
            </text>
            {visible.map((channel, i) => (
              <text key={channel} x={Math.min(xOf(wave.t[hoverIndex]) + 16, WIDTH - 182)} y={PAD_TOP + 29 + i * 13} fontSize={10} fill={WAVE_CHANNEL_COLORS[channel]}>
                {WAVE_CHANNEL_LABELS[channel]}: {channelValues(wave, channel)[hoverIndex].toFixed(3)}
              </text>
            ))}
          </g>
        )}
      </svg>
      {overlays.length > 0 && (
        <div className="flex flex-wrap gap-3 text-[11px] text-ink-300">
          <span className="text-ink-500">重ね描き（{WAVE_CHANNEL_LABELS[overlayChannel]}）:</span>
          {overlays.map((overlay) => (
            <span key={overlay.label} className="flex items-center gap-1.5">
              <svg width="22" height="6" aria-hidden="true">
                <line x1="0" x2="22" y1="3" y2="3" stroke={overlay.color} strokeWidth="2" strokeDasharray={overlay.dash ?? "6 4"} />
              </svg>
              {overlay.label}
            </span>
          ))}
        </div>
      )}
      {editable && <p className="text-[11px] text-ink-500">点をドラッグして {WAVE_CHANNEL_LABELS[editChannel]} を編集できます（読者/作中人物の認知量を編集すると情報非対称性が再計算されます）。</p>}
    </div>
  );
}
