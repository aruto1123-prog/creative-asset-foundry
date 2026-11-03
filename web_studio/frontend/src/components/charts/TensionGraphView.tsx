// Layer 1 力学グラフの SVG 描画（Fruchterman–Reingold 決定論レイアウト・種別配色・ホバー詳細）。

import { useMemo, useState } from "react";
import type { DynamicTensionGraph, EdgeKind, NodeKind, TensionEdge, TensionNode } from "../../api/types";
import { EDGE_KIND_LABELS, EDGE_KIND_STYLES, NODE_KIND_COLORS, NODE_KIND_LABELS } from "../../lib/labels";

const WIDTH = 680;
const HEIGHT = 440;
const KIND_ORDER: NodeKind[] = ["agent", "desire", "lack", "symbol"];

interface Positioned {
  node: TensionNode;
  x: number;
  y: number;
  r: number;
}

function layout(graph: DynamicTensionGraph): Map<string, Positioned> {
  const nodes = [...graph.nodes].sort((a, b) => KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind) || b.weight - a.weight);
  const n = nodes.length;
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2;
  const positions = nodes.map((node, index) => {
    const angle = (2 * Math.PI * index) / Math.max(1, n) - Math.PI / 2;
    return { node, x: cx + Math.cos(angle) * 150, y: cy + Math.sin(angle) * 130 };
  });
  const index = new Map(positions.map((p, i) => [p.node.id, i]));
  const area = (WIDTH - 120) * (HEIGHT - 120);
  const k = Math.sqrt(area / Math.max(1, n)) * 1.05;
  let temperature = 60;
  for (let iteration = 0; iteration < 260; iteration += 1) {
    const disp = positions.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const dx = positions[i].x - positions[j].x;
        const dy = positions[i].y - positions[j].y;
        const dist = Math.max(0.01, Math.hypot(dx, dy));
        const force = (k * k) / dist;
        disp[i].x += (dx / dist) * force;
        disp[i].y += (dy / dist) * force;
        disp[j].x -= (dx / dist) * force;
        disp[j].y -= (dy / dist) * force;
      }
    }
    for (const edge of graph.edges) {
      const a = index.get(edge.source);
      const b = index.get(edge.target);
      if (a === undefined || b === undefined) {
        continue;
      }
      const dx = positions[a].x - positions[b].x;
      const dy = positions[a].y - positions[b].y;
      const dist = Math.max(0.01, Math.hypot(dx, dy));
      const force = ((dist * dist) / k) * (0.4 + edge.intensity * 0.6);
      disp[a].x -= (dx / dist) * force;
      disp[a].y -= (dy / dist) * force;
      disp[b].x += (dx / dist) * force;
      disp[b].y += (dy / dist) * force;
    }
    for (let i = 0; i < n; i += 1) {
      disp[i].x += (cx - positions[i].x) * 0.02;
      disp[i].y += (cy - positions[i].y) * 0.02;
      const length = Math.max(0.01, Math.hypot(disp[i].x, disp[i].y));
      positions[i].x += (disp[i].x / length) * Math.min(length, temperature);
      positions[i].y += (disp[i].y / length) * Math.min(length, temperature);
      positions[i].x = Math.min(WIDTH - 70, Math.max(70, positions[i].x));
      positions[i].y = Math.min(HEIGHT - 50, Math.max(40, positions[i].y));
    }
    temperature = Math.max(1.5, temperature * 0.975);
  }
  // 衝突解消: ノード半径とラベル領域を考慮した最小距離を保証する
  const radius = (node: TensionNode) => 11 + node.weight * 13;
  for (let pass = 0; pass < 80; pass += 1) {
    let moved = false;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const dx = positions[j].x - positions[i].x;
        const dy = positions[j].y - positions[i].y;
        const dist = Math.max(0.01, Math.hypot(dx, dy));
        const minDist = radius(positions[i].node) + radius(positions[j].node) + 58;
        if (dist < minDist) {
          const push = (minDist - dist) / 2;
          const ux = dist > 0.02 ? dx / dist : Math.cos(i + j);
          const uy = dist > 0.02 ? dy / dist : Math.sin(i + j);
          positions[i].x -= ux * push;
          positions[i].y -= uy * push;
          positions[j].x += ux * push;
          positions[j].y += uy * push;
          moved = true;
        }
      }
    }
    for (const p of positions) {
      p.x = Math.min(WIDTH - 70, Math.max(70, p.x));
      p.y = Math.min(HEIGHT - 50, Math.max(40, p.y));
    }
    if (!moved) {
      break;
    }
  }
  return new Map(positions.map((p) => [p.node.id, { node: p.node, x: p.x, y: p.y, r: radius(p.node) }]));
}

function edgePath(source: Positioned, target: Positioned, curvature: number): { d: string; mx: number; my: number } {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const dist = Math.max(1, Math.hypot(dx, dy));
  const nx = -dy / dist;
  const ny = dx / dist;
  const mx = (source.x + target.x) / 2 + nx * curvature;
  const my = (source.y + target.y) / 2 + ny * curvature;
  const ex = target.x - ((target.x - mx) / Math.max(1, Math.hypot(target.x - mx, target.y - my))) * (target.r + 4);
  const ey = target.y - ((target.y - my) / Math.max(1, Math.hypot(target.x - mx, target.y - my))) * (target.r + 4);
  const sx = source.x + ((mx - source.x) / Math.max(1, Math.hypot(mx - source.x, my - source.y))) * source.r;
  const sy = source.y + ((my - source.y) / Math.max(1, Math.hypot(mx - source.x, my - source.y))) * source.r;
  return { d: `M${sx.toFixed(1)},${sy.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`, mx, my };
}

type Hover = { type: "node"; node: TensionNode } | { type: "edge"; edge: TensionEdge } | null;

export function TensionGraphView({ graph }: { graph: DynamicTensionGraph }) {
  const positions = useMemo(() => layout(graph), [graph]);
  const [hover, setHover] = useState<Hover>(null);

  const edges = useMemo(() => {
    const totals = new Map<string, number>();
    for (const edge of graph.edges) {
      const key = [edge.source, edge.target].sort().join("|");
      totals.set(key, (totals.get(key) ?? 0) + 1);
    }
    const pairCount = new Map<string, number>();
    return graph.edges.map((edge) => {
      const key = [edge.source, edge.target].sort().join("|");
      const order = pairCount.get(key) ?? 0;
      pairCount.set(key, order + 1);
      const direction = edge.source < edge.target ? 1 : -1;
      const curvature = (18 + order * 30) * direction;
      return { edge, curvature, parallel: totals.get(key) ?? 1 };
    });
  }, [graph]);

  const usedKinds = Array.from(new Set(graph.edges.map((edge) => edge.kind))) as EdgeKind[];

  return (
    <div className="flex flex-col gap-2">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full rounded-lg border border-ink-700 bg-ink-950" role="img" aria-label="力学グラフ">
        <defs>
          {(Object.keys(EDGE_KIND_STYLES) as EdgeKind[]).map((kind) => (
            <marker key={kind} id={`arrow-${kind}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill={EDGE_KIND_STYLES[kind].color} />
            </marker>
          ))}
        </defs>
        {edges.map(({ edge, curvature, parallel }, index) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) {
            return null;
          }
          const style = EDGE_KIND_STYLES[edge.kind];
          const { d, mx, my } = edgePath(source, target, curvature);
          const active = hover?.type === "edge" && hover.edge === edge;
          return (
            <g key={`${edge.source}-${edge.target}-${edge.kind}-${index}`} onMouseEnter={() => setHover({ type: "edge", edge })} onMouseLeave={() => setHover(null)} className="cursor-pointer">
              <path d={d} fill="none" stroke="transparent" strokeWidth={12} />
              <path
                d={d}
                fill="none"
                stroke={style.color}
                strokeOpacity={active ? 1 : 0.78}
                strokeWidth={1 + edge.intensity * (edge.kind === "opposition" ? 4.5 : 3)}
                strokeDasharray={style.dash}
                markerEnd={`url(#arrow-${edge.kind})`}
              />
              {(parallel <= 2 || active) && (
                <text x={mx} y={my - 4} textAnchor="middle" fontSize={9} fill={style.color} opacity={active ? 1 : 0.75}>
                  {EDGE_KIND_LABELS[edge.kind]}
                </text>
              )}
            </g>
          );
        })}
        {Array.from(positions.values()).map((p) => {
          const active = hover?.type === "node" && hover.node.id === p.node.id;
          return (
            <g key={p.node.id} onMouseEnter={() => setHover({ type: "node", node: p.node })} onMouseLeave={() => setHover(null)} className="cursor-pointer">
              <circle cx={p.x} cy={p.y} r={p.r + (active ? 3 : 0)} fill={NODE_KIND_COLORS[p.node.kind]} fillOpacity={0.22} stroke={NODE_KIND_COLORS[p.node.kind]} strokeWidth={active ? 3 : 2} />
              <text x={p.x} y={p.y + 3.5} textAnchor="middle" fontSize={10} fontWeight={600} fill="#e6e9f3">
                {NODE_KIND_LABELS[p.node.kind]}
              </text>
              <text x={p.x} y={p.y + p.r + 14} textAnchor="middle" fontSize={10.5} fill="#c5cbe0">
                {p.node.label.length > 16 ? `${p.node.label.slice(0, 16)}…` : p.node.label}
              </text>
              {p.node.slot && (
                <text x={p.x} y={p.y + p.r + 26} textAnchor="middle" fontSize={8.5} fill="#7c9cff" fontFamily="monospace">
                  {`{{${p.node.slot}}}`}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="min-h-[44px] rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-xs text-ink-200">
        {hover === null && <span className="text-ink-500">ノードやエッジにカーソルを合わせると詳細を表示します。</span>}
        {hover?.type === "node" && (
          <span>
            <strong style={{ color: NODE_KIND_COLORS[hover.node.kind] }}>{NODE_KIND_LABELS[hover.node.kind]}</strong>（{hover.node.role}・重要度 {hover.node.weight.toFixed(2)}）: {hover.node.label}
            {hover.node.description && <span className="text-ink-400"> — {hover.node.description}</span>}
          </span>
        )}
        {hover?.type === "edge" && (
          <span>
            <strong style={{ color: EDGE_KIND_STYLES[hover.edge.kind].color }}>{EDGE_KIND_LABELS[hover.edge.kind]}</strong>: {positions.get(hover.edge.source)?.node.label} → {positions.get(hover.edge.target)?.node.label}
            ／強度 {hover.edge.intensity.toFixed(2)}・勾配 {hover.edge.gradient.toFixed(2)}
            {hover.edge.activation_t !== null && `・顕在化 t=${hover.edge.activation_t.toFixed(2)}`}
            {hover.edge.label && <span className="text-ink-400"> — {hover.edge.label}</span>}
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-400">
        {KIND_ORDER.map((kind) => (
          <span key={kind} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NODE_KIND_COLORS[kind] }} />
            {NODE_KIND_LABELS[kind]}
          </span>
        ))}
        <span className="text-ink-600">|</span>
        {usedKinds.map((kind) => (
          <span key={kind} className="flex items-center gap-1.5">
            <svg width="20" height="6" aria-hidden="true">
              <line x1="0" x2="20" y1="3" y2="3" stroke={EDGE_KIND_STYLES[kind].color} strokeWidth="2" strokeDasharray={EDGE_KIND_STYLES[kind].dash} />
            </svg>
            {EDGE_KIND_LABELS[kind]}
          </span>
        ))}
      </div>
    </div>
  );
}
