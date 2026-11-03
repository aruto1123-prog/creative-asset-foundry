// JSON ビューア（シンタックスハイライト・コピー・ダウンロード）。

import { useMemo } from "react";
import { downloadJson } from "../api/client";
import { Button, CopyButton } from "./ui";

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlight(json: string): string {
  const escaped = escapeHtml(json);
  return escaped.replace(
    /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      let cls = "text-amber-300";
      if (match.startsWith('"')) {
        cls = match.endsWith(":") ? "text-sky-300" : "text-emerald-300";
      } else if (match === "true" || match === "false") {
        cls = "text-violet-300";
      } else if (match === "null") {
        cls = "text-ink-400";
      }
      return `<span class="${cls}">${match}</span>`;
    },
  );
}

export function JsonViewer({
  data,
  filename,
  maxHeight = 420,
  title,
}: {
  data: unknown;
  filename?: string;
  maxHeight?: number;
  title?: string;
}) {
  const text = useMemo(() => JSON.stringify(data, null, 2) ?? "null", [data]);
  const html = useMemo(() => highlight(text.length > 400000 ? text.slice(0, 400000) : text), [text]);
  return (
    <div className="overflow-hidden rounded-lg border border-ink-700 bg-ink-950">
      <div className="flex items-center justify-between gap-2 border-b border-ink-700 px-3 py-1.5">
        <span className="truncate font-mono text-[11px] text-ink-400">{title ?? filename ?? "JSON"}</span>
        <div className="flex gap-1">
          <CopyButton text={text} />
          {filename && (
            <Button size="sm" variant="ghost" onClick={() => downloadJson(filename, data)}>
              ダウンロード
            </Button>
          )}
        </div>
      </div>
      <pre className="overflow-auto p-3 font-mono text-[11px] leading-relaxed text-ink-200" style={{ maxHeight }} dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}
