// Markdown プレビュー（marked）。生 HTML はエスケープし、Mermaid ブロックはコードとして表示する。

import { useMemo } from "react";
import { Marked } from "marked";
import type { Tokens } from "marked";

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const markdown = new Marked({ gfm: true, breaks: false });
markdown.use({
  renderer: {
    html(token: Tokens.HTML | Tokens.Tag): string {
      return escapeHtml(token.text);
    },
    code(token: Tokens.Code): string {
      const lang = (token.lang ?? "").trim();
      const isMermaid = lang === "mermaid";
      const label = isMermaid ? "Mermaid 図解" : lang || "code";
      return `<div class="md-codeblock${isMermaid ? " md-mermaid" : ""}"><div class="md-codeblock-label">${escapeHtml(label)}</div><pre><code>${escapeHtml(token.text)}</code></pre></div>`;
    },
    link(token: Tokens.Link): string {
      return `<span class="md-link">${escapeHtml(token.text)}</span>`;
    },
    image(token: Tokens.Image): string {
      return `<span class="md-image">［画像: ${escapeHtml(token.text || "無題")}］</span>`;
    },
  },
});

export function MarkdownView({ text, className }: { text: string; className?: string }) {
  const html = useMemo(() => markdown.parse(text, { async: false }) as string, [text]);
  return <article className={`md-body ${className ?? ""}`} dangerouslySetInnerHTML={{ __html: html }} />;
}
