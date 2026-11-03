// レシピパラメータ（ブロックマージ比率・LoRA 重み・スタイル特徴量）の横棒表示。

import type { ParameterDim } from "../../api/types";

function groupOf(name: string): { group: string; color: string } {
  if (name.startsWith("merge.")) {
    return { group: "ブロックマージ比率（Bの比率）", color: "#7c9cff" };
  }
  if (name.startsWith("lora.")) {
    return { group: "LoRA 重み・階層ウェイト", color: "#34d399" };
  }
  if (name.startsWith("style.")) {
    return { group: "絵柄特徴量", color: "#f472b6" };
  }
  return { group: "その他", color: "#fbbf24" };
}

export function ParamBars({ params, dims }: { params: Record<string, number>; dims?: ParameterDim[] }) {
  const bounds = new Map((dims ?? []).map((dim) => [dim.name, dim]));
  const groups = new Map<string, Array<[string, number]>>();
  for (const [name, value] of Object.entries(params).sort(([a], [b]) => a.localeCompare(b))) {
    const { group } = groupOf(name);
    const list = groups.get(group) ?? [];
    list.push([name, value]);
    groups.set(group, list);
  }
  return (
    <div className="flex flex-col gap-3">
      {Array.from(groups.entries()).map(([group, entries]) => (
        <div key={group}>
          <div className="mb-1 text-[11px] font-semibold text-ink-400">{group}</div>
          <div className="grid grid-cols-1 gap-1 md:grid-cols-2">
            {entries.map(([name, value]) => {
              const dim = bounds.get(name);
              const low = dim?.low ?? 0;
              const high = dim?.high ?? (name.startsWith("lora.") ? 1.5 : 1);
              const ratio = Math.min(1, Math.max(0, (value - low) / Math.max(1e-6, high - low)));
              const { color } = groupOf(name);
              return (
                <div key={name} className="flex items-center gap-2" title={dim?.description ?? name}>
                  <span className="w-40 shrink-0 truncate font-mono text-[11px] text-ink-300">{name.replace(/^(merge|lora|style)\./, "")}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-700">
                    <div className="h-full rounded-full" style={{ width: `${ratio * 100}%`, backgroundColor: color }} />
                  </div>
                  <span className="w-12 text-right font-mono text-[11px] text-ink-200">{value.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
