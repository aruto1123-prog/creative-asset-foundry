// Creative Asset Foundry 統合スタジオ: サイドバー・ステータスバー・5ワークスペース。

import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, errorMessage } from "./api/client";
import type { Health } from "./api/types";
import { Badge, cx } from "./components/ui";
import { ProgressSocketProvider, useProgressSocket } from "./hooks/useProgressSocket";
import { ToastProvider } from "./hooks/useToast";
import { AssetPublisher } from "./workspaces/AssetPublisher";
import { MangaStudio } from "./workspaces/MangaStudio";
import { StructureLab } from "./workspaces/StructureLab";
import { StyleArena } from "./workspaces/StyleArena";
import { SystemConfig } from "./workspaces/SystemConfig";

type WorkspaceKey = "structure" | "style" | "manga" | "publisher" | "system";

interface WorkspaceDef {
  key: WorkspaceKey;
  label: string;
  caption: string;
  icon: ReactNode;
}

const ICON_PROPS = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

const WORKSPACES: WorkspaceDef[] = [
  {
    key: "structure",
    label: "Structure Lab",
    caption: "構造抽出・合成・転用",
    icon: (
      <svg {...ICON_PROPS} aria-hidden="true">
        <circle cx="6" cy="6" r="2.5" />
        <circle cx="18" cy="7" r="2.5" />
        <circle cx="12" cy="18" r="2.5" />
        <path d="M8.3 7l7.4.2M7.3 8.2l3.6 7.6M16.9 9.3l-3.7 6.6" />
      </svg>
    ),
  },
  {
    key: "style",
    label: "Style Alchemy Arena",
    caption: "選好学習・ベイズ最適化",
    icon: (
      <svg {...ICON_PROPS} aria-hidden="true">
        <rect x="3" y="5" width="7.5" height="14" rx="1.5" />
        <rect x="13.5" y="5" width="7.5" height="14" rx="1.5" />
        <path d="M6.7 2.5v2.5M17.3 2.5v2.5" />
      </svg>
    ),
  },
  {
    key: "manga",
    label: "Manga Studio",
    caption: "ネーム・文法監査・ComfyUI",
    icon: (
      <svg {...ICON_PROPS} aria-hidden="true">
        <path d="M4 3h16v18H4z" />
        <path d="M4 10h9l3-7M13 10v11M13 15h7" />
      </svg>
    ),
  },
  {
    key: "publisher",
    label: "Asset Publisher",
    caption: "3軸企画・5段階執筆",
    icon: (
      <svg {...ICON_PROPS} aria-hidden="true">
        <path d="M6 3h9l4 4v14H6z" />
        <path d="M15 3v4h4M9 12h7M9 16h7" />
      </svg>
    ),
  },
  {
    key: "system",
    label: "System & Local AI",
    caption: "DeepSeek・ComfyUI・DB",
    icon: (
      <svg {...ICON_PROPS} aria-hidden="true">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" />
      </svg>
    ),
  },
];

function readInitialWorkspace(): WorkspaceKey {
  const fromHash = window.location.hash.replace("#", "") as WorkspaceKey;
  if (WORKSPACES.some((w) => w.key === fromHash)) {
    return fromHash;
  }
  try {
    const stored = window.localStorage.getItem("caf.workspace") as WorkspaceKey | null;
    if (stored && WORKSPACES.some((w) => w.key === stored)) {
      return stored;
    }
  } catch {
    return "structure";
  }
  return "structure";
}

function StatusBar({ health, healthError, onRefresh }: { health: Health | null; healthError: string | null; onRefresh: () => void }) {
  const { connected } = useProgressSocket();
  const provider = health?.llm_provider ?? "-";
  return (
    <header className="flex flex-wrap items-center gap-2 border-b border-ink-800 bg-ink-900/80 px-5 py-2.5 text-xs backdrop-blur">
      <span className="mr-1 text-ink-400">LLM</span>
      <Badge tone={provider === "mock" ? "warning" : "success"} title={health?.llm_model}>
        {provider}
        {health?.llm_model ? ` · ${health.llm_model}` : ""}
      </Badge>
      {health?.llm_fallback_active && <Badge tone="warning">モックへフォールバック中</Badge>}
      <span className="ml-3 mr-1 text-ink-400">ComfyUI</span>
      <Badge tone={health?.comfy_mock ? "warning" : "info"} title={health?.comfy_base_url}>
        {health?.comfy_mock ? "内蔵モック" : health?.comfy_base_url ?? "-"}
      </Badge>
      <span className="ml-3 mr-1 text-ink-400">構造DB</span>
      <Badge tone="accent">
        {health?.structures ?? 0} 件（シード {health?.seeds_loaded ?? 0}）
      </Badge>
      <span className="ml-auto flex items-center gap-2">
        {healthError && <Badge tone="danger">{healthError}</Badge>}
        <span className="flex items-center gap-1.5 text-ink-300">
          <span className={cx("h-2 w-2 rounded-full", connected ? "bg-emerald-400" : "bg-rose-400")} />
          {connected ? "WebSocket 接続中" : "WebSocket 再接続中"}
        </span>
        <button type="button" onClick={onRefresh} className="rounded px-2 py-0.5 text-ink-400 hover:bg-ink-800 hover:text-ink-100">
          更新
        </button>
      </span>
    </header>
  );
}

function Shell() {
  const [active, setActive] = useState<WorkspaceKey>(readInitialWorkspace);
  const [visited, setVisited] = useState<Set<WorkspaceKey>>(() => new Set([readInitialWorkspace()]));
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [dataVersion, setDataVersion] = useState(0);

  const refreshHealth = useCallback(() => {
    api.system
      .health()
      .then((result) => {
        setHealth(result);
        setHealthError(null);
      })
      .catch((reason: unknown) => setHealthError(errorMessage(reason)));
  }, []);

  useEffect(() => {
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 15000);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  const select = (key: WorkspaceKey) => {
    setActive(key);
    setVisited((current) => new Set(current).add(key));
    window.location.hash = key;
    try {
      window.localStorage.setItem("caf.workspace", key);
    } catch {
      return;
    }
  };

  const notifyDataChanged = useCallback(() => {
    setDataVersion((value) => value + 1);
    refreshHealth();
  }, [refreshHealth]);

  return (
    <div className="flex h-full min-h-screen bg-ink-950 text-ink-100">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-ink-800 bg-ink-900 md:flex">
        <div className="border-b border-ink-800 px-5 py-4">
          <div className="text-lg font-bold tracking-tight text-accent-strong">Creative Asset Foundry</div>
          <div className="mt-0.5 text-[11px] text-ink-400">ローカルLLM × 画像生成 統合スタジオ</div>
        </div>
        <nav className="flex flex-1 flex-col gap-1 p-3" aria-label="ワークスペース">
          {WORKSPACES.map((workspace, index) => (
            <button
              key={workspace.key}
              type="button"
              onClick={() => select(workspace.key)}
              className={cx(
                "flex items-start gap-3 rounded-lg px-3 py-2.5 text-left transition-colors",
                active === workspace.key ? "bg-accent-soft text-accent-strong" : "text-ink-300 hover:bg-ink-800 hover:text-ink-100",
              )}
            >
              <span className="mt-0.5">{workspace.icon}</span>
              <span>
                <span className="block text-sm font-medium">
                  {index + 1}. {workspace.label}
                </span>
                <span className="block text-[11px] text-ink-400">{workspace.caption}</span>
              </span>
            </button>
          ))}
        </nav>
        <div className="border-t border-ink-800 px-5 py-3 text-[11px] text-ink-500">v{health?.version ?? "1.0.0"}</div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <nav className="flex gap-1 overflow-x-auto border-b border-ink-800 bg-ink-900 p-2 md:hidden" aria-label="ワークスペース（モバイル）">
          {WORKSPACES.map((workspace) => (
            <button
              key={workspace.key}
              type="button"
              onClick={() => select(workspace.key)}
              className={cx("whitespace-nowrap rounded-md px-3 py-1.5 text-xs", active === workspace.key ? "bg-accent-soft text-accent-strong" : "text-ink-300")}
            >
              {workspace.label}
            </button>
          ))}
        </nav>
        <StatusBar health={health} healthError={healthError} onRefresh={refreshHealth} />
        <main className="min-w-0 flex-1 overflow-y-auto">
          {visited.has("structure") && (
            <div hidden={active !== "structure"}>
              <StructureLab dataVersion={dataVersion} onDataChanged={notifyDataChanged} />
            </div>
          )}
          {visited.has("style") && (
            <div hidden={active !== "style"}>
              <StyleArena active={active === "style"} />
            </div>
          )}
          {visited.has("manga") && (
            <div hidden={active !== "manga"}>
              <MangaStudio dataVersion={dataVersion} />
            </div>
          )}
          {visited.has("publisher") && (
            <div hidden={active !== "publisher"}>
              <AssetPublisher dataVersion={dataVersion} />
            </div>
          )}
          {visited.has("system") && (
            <div hidden={active !== "system"}>
              <SystemConfig onDataChanged={notifyDataChanged} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <ProgressSocketProvider>
        <Shell />
      </ProgressSocketProvider>
    </ToastProvider>
  );
}
