// 型付き API クライアント。fetch ラッパー・エラー整形・ジョブ待機・ダウンロード補助を提供する。

import type {
  AestheticPreferenceVector,
  ArenaPair,
  ArticleSummary,
  BackupInfo,
  BlockGrouping,
  CaptionRecord,
  CheckpointRef,
  ComfyCompileResponse,
  Comparison,
  ComparisonOutcome,
  ConnectionTestResult,
  ContentAxis,
  ContentGap,
  ContentMatrix,
  ContentPlan,
  DatasetAnalyzeResponse,
  DbStats,
  DiagnosticApplyResponse,
  DiagnosticReport,
  DiagnosticSummary,
  EmotionalInformationWave,
  ExtractionResult,
  GrammarReport,
  GridAxis,
  GridDetail,
  GridExperiment,
  Health,
  ImageDetectionResult,
  Job,
  JobRef,
  LoRASpec,
  MergeResponse,
  MermaidResponse,
  ModelArchitecture,
  PageAnnotation,
  PageSide,
  MutateResponse,
  MutationInfo,
  ParameterSpace,
  PublicConfig,
  PublishedArticle,
  RatingEntry,
  RecipeExportResponse,
  ResearchLog,
  SanitizationReport,
  SanitizeResponse,
  SearchResponse,
  SeedReport,
  SeedSummary,
  SeriesResponse,
  Storyboard,
  StoryboardConfig,
  StoryboardSummary,
  StructureList,
  StyleCandidate,
  StyleRecipe,
  SuggestResponse,
  TargetDomain,
  TelemetryResponse,
  TokenEntry,
  TranspiledDesign,
  UniversalStructure,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail ? `${detail}（HTTP ${status}）` : `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

type QueryValue = string | number | boolean | null | undefined;

export function buildQuery(params: Record<string, QueryValue>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

function formatDetail(payload: unknown): string {
  if (payload === null || payload === undefined) {
    return "";
  }
  if (typeof payload === "string") {
    return payload;
  }
  if (typeof payload === "object") {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item && typeof item === "object") {
            const entry = item as { loc?: unknown[]; msg?: string };
            const loc = Array.isArray(entry.loc) ? entry.loc.join(".") : "";
            return loc ? `${loc}: ${entry.msg ?? ""}` : entry.msg ?? JSON.stringify(item);
          }
          return String(item);
        })
        .join(" / ");
    }
    if (detail !== undefined) {
      return JSON.stringify(detail);
    }
    return JSON.stringify(payload);
  }
  return String(payload);
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: { Accept: "application/json" } };
  if (body !== undefined) {
    init.headers = { ...init.headers, "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (reason) {
    const message = reason instanceof Error ? reason.message : String(reason);
    throw new ApiError(0, `バックエンドに接続できません: ${message}`);
  }
  const contentType = response.headers.get("content-type") ?? "";
  const isJson = contentType.includes("application/json");
  if (!response.ok) {
    let payload: unknown = null;
    try {
      payload = isJson ? await response.json() : await response.text();
    } catch {
      payload = response.statusText;
    }
    throw new ApiError(response.status, formatDetail(payload) || response.statusText);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  if (isJson) {
    return (await response.json()) as T;
  }
  return (await response.text()) as unknown as T;
}

export async function getText(path: string): Promise<string> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch (reason) {
    const message = reason instanceof Error ? reason.message : String(reason);
    throw new ApiError(0, `バックエンドに接続できません: ${message}`);
  }
  const text = await response.text();
  if (!response.ok) {
    let detail = text;
    try {
      detail = formatDetail(JSON.parse(text));
    } catch {
      detail = text || response.statusText;
    }
    throw new ApiError(response.status, detail);
  }
  return text;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body?: unknown) => request<T>("POST", path, body ?? {});
const put = <T>(path: string, body?: unknown) => request<T>("PUT", path, body ?? {});
const del = <T>(path: string) => request<T>("DELETE", path);

/** 配列、または {items|results|...: 配列} 形式の応答を配列へ正規化する。 */
export function asList<T>(value: unknown, keys: string[] = []): T[] {
  if (Array.isArray(value)) {
    return value as T[];
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of [...keys, "items", "results", "data", "list"]) {
      if (Array.isArray(record[key])) {
        return record[key] as T[];
      }
    }
  }
  return [];
}

export function isJobRef(value: unknown): value is JobRef {
  return Boolean(value) && typeof value === "object" && typeof (value as JobRef).job_id === "string";
}

export function errorMessage(reason: unknown): string {
  if (reason instanceof ApiError) {
    return reason.message;
  }
  if (reason instanceof Error) {
    return reason.message;
  }
  return String(reason);
}

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

/** ジョブ完了までポーリングし、結果を返す（進捗は WebSocket で別途配信される）。 */
export async function waitForJob<T>(jobId: string, onUpdate?: (job: Job) => void, timeoutMs = 15 * 60 * 1000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  let delay = 300;
  while (Date.now() < deadline) {
    const job = await get<Job>(`/api/jobs/${encodeURIComponent(jobId)}`);
    onUpdate?.(job);
    if (job.status === "succeeded") {
      return job.result as T;
    }
    if (job.status === "failed") {
      throw new ApiError(500, job.error || job.message || "ジョブが失敗しました");
    }
    await sleep(delay);
    delay = Math.min(1200, Math.round(delay * 1.4));
  }
  throw new ApiError(408, "ジョブの完了待ちがタイムアウトしました");
}

/** async: true でジョブ投入し、完了結果を返す。同期応答が返った場合はそのまま返す。 */
export async function runJob<T>(path: string, body: Record<string, unknown>, onUpdate?: (job: Job) => void): Promise<T> {
  const response = await post<T | JobRef>(path, { ...body, async: true });
  if (isJobRef(response)) {
    onUpdate?.({
      id: response.job_id,
      kind: path,
      status: "queued",
      progress: 0,
      stage: "queued",
      message: "ジョブを受け付けました",
    });
    return waitForJob<T>(response.job_id, onUpdate);
  }
  return response as T;
}

// ---------------------------------------------------------------------------
// ダウンロード補助
// ---------------------------------------------------------------------------

export function downloadText(filename: string, text: string, mime = "text/plain;charset=utf-8"): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadJson(filename: string, data: unknown): void {
  downloadText(filename, JSON.stringify(data, null, 2), "application/json;charset=utf-8");
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

// ---------------------------------------------------------------------------
// エンドポイント
// ---------------------------------------------------------------------------

const enc = encodeURIComponent;

export interface ExtractRequest {
  text: string;
  source_kind: string;
  title_hint?: string | null;
  category?: string;
  save?: boolean;
}

export interface MergeRequest {
  struct_a_id: string;
  struct_b_id: string;
  weight_topology: number;
  weight_wave: number;
  save?: boolean;
}

export interface MutateRequest {
  structure_id: string;
  mutation_type: string;
  intensity: number;
  seed: number;
  save?: boolean;
}

export interface SearchRequest {
  structure_id?: string;
  wave?: EmotionalInformationWave;
  method: "dtw" | "cosine" | "hybrid";
  top_k: number;
  category?: string | null;
}

export interface BootstrapRequest {
  n: number;
  architecture: ModelArchitecture;
  lora_names: string[];
  grouping: BlockGrouping;
  seed: number;
}

export interface SuggestRequest {
  n: number;
  strategy: "gp" | "cmaes" | "genetic" | "hybrid";
  acquisition: "ei" | "ucb" | "pi";
  architecture: ModelArchitecture;
  lora_names: string[];
  grouping: BlockGrouping;
  register?: boolean;
}

export interface RecipeExportRequest {
  params: Record<string, number>;
  architecture: ModelArchitecture;
  model_a: CheckpointRef;
  model_b: CheckpointRef;
  model_c?: CheckpointRef | null;
  loras: LoRASpec[];
  name: string;
  prompt: string;
  negative: string;
  seed: number;
  width: number;
  height: number;
}

export interface GridCreateRequest {
  name: string;
  base_prompt: string;
  negative_prompt: string;
  x_axis: GridAxis;
  y_axis?: GridAxis | null;
  z_axis?: GridAxis | null;
  base_params?: Record<string, number>;
  seed: number;
}

export interface StoryboardCreateRequest {
  structure_id: string;
  title?: string;
  theme: Record<string, string>;
  config: Partial<StoryboardConfig>;
}

export interface ComfyCompileRequest {
  architecture: ModelArchitecture;
  recipe?: Partial<StyleRecipe> | null;
  recipe_params?: Record<string, number> | null;
  panel_ids?: string[] | null;
}

export interface DiagnosticRequest {
  storyboard_id?: string;
  storyboard?: Storyboard;
  annotation?: PageAnnotation;
  images?: string[];
  first_page_side?: PageSide;
  target_structure_id?: string | null;
  use_llm?: boolean;
  simulate?: boolean;
  max_prescriptions?: number;
  title?: string;
}

export interface DiagnosticOverlayQuery {
  heat: boolean;
  gaze: boolean;
  pins: boolean;
  order: boolean;
  image: boolean;
}

export interface MatrixRequest {
  structure_id?: string;
  research_log?: ResearchLog;
  experiment?: Record<string, unknown>;
}

export interface ArticleRequest {
  plan?: ContentPlan;
  matrix_id?: string;
  axis?: ContentAxis;
  research_log?: ResearchLog;
  structure_id?: string;
}

export const api = {
  system: {
    health: () => get<Health>("/api/health"),
    getConfig: () => get<PublicConfig>("/api/system/config"),
    updateConfig: (update: Record<string, unknown>) => put<PublicConfig>("/api/system/config", update),
    testLlm: () => post<ConnectionTestResult>("/api/system/llm/test"),
    testComfy: () => post<ConnectionTestResult>("/api/system/comfy/test"),
    dbStats: () => get<DbStats>("/api/system/db/stats"),
    backup: () => post<BackupInfo>("/api/system/db/backup"),
    backups: async () => asList<BackupInfo>(await get<unknown>("/api/system/db/backups"), ["backups"]),
    telemetry: (limit = 50) => get<TelemetryResponse>(`/api/system/telemetry${buildQuery({ limit })}`),
  },
  jobs: {
    list: async () => asList<Job>(await get<unknown>("/api/jobs"), ["jobs"]),
    get: (id: string) => get<Job>(`/api/jobs/${enc(id)}`),
  },
  structures: {
    list: (params: { category?: string | null; source_kind?: string | null; q?: string | null; limit?: number; offset?: number }) =>
      get<StructureList>(`/api/structures${buildQuery(params)}`),
    get: (id: string) => get<UniversalStructure>(`/api/structures/${enc(id)}`),
    save: (structure: UniversalStructure) => post<UniversalStructure>("/api/structures", structure),
    remove: (id: string) => del<unknown>(`/api/structures/${enc(id)}`),
    updateWave: (id: string, wave: EmotionalInformationWave) => put<UniversalStructure>(`/api/structures/${enc(id)}/wave`, wave),
    extract: (body: ExtractRequest, onUpdate?: (job: Job) => void) =>
      runJob<ExtractionResult>("/api/structures/extract", { ...body }, onUpdate),
    merge: (body: MergeRequest) => post<MergeResponse>("/api/structures/merge", body),
    mutate: (body: MutateRequest) => post<MutateResponse>("/api/structures/mutate", body),
    mutations: async () => asList<MutationInfo>(await get<unknown>("/api/structures/mutations"), ["mutations"]),
    transpile: (id: string, domain: TargetDomain, theme_variables: Record<string, string>) =>
      post<TranspiledDesign>(`/api/structures/${enc(id)}/transpile`, { domain, theme_variables }),
    search: (body: SearchRequest) => post<SearchResponse>("/api/structures/search", body),
    mermaid: (id: string) => get<MermaidResponse>(`/api/structures/${enc(id)}/mermaid`),
    waveSvgUrl: (id: string, compareId?: string) => `/api/structures/${enc(id)}/wave.svg${buildQuery({ compare_id: compareId })}`,
  },
  seeds: {
    load: (overwrite: boolean) => post<SeedReport>("/api/seeds/load", { overwrite }),
    summary: () => get<SeedSummary>("/api/seeds/summary"),
  },
  style: {
    candidates: async () => asList<StyleCandidate>(await get<unknown>("/api/style/candidates"), ["candidates"]),
    createCandidate: (body: { label: string; params: Record<string, number>; recipe?: StyleRecipe | null; image_ref?: string | null }) =>
      post<StyleCandidate>("/api/style/candidates", body),
    deleteCandidate: (id: string) => del<unknown>(`/api/style/candidates/${enc(id)}`),
    bootstrap: async (body: BootstrapRequest) => asList<StyleCandidate>(await post<unknown>("/api/style/candidates/bootstrap", body), ["candidates", "created"]),
    previewUrl: (id: string) => `/api/style/candidates/${enc(id)}/preview.svg`,
    nextPair: () => get<ArenaPair>("/api/style/arena/next"),
    comparisons: async () => asList<Comparison>(await get<unknown>("/api/style/comparisons"), ["comparisons"]),
    compare: (body: { left_id: string; right_id: string; outcome: ComparisonOutcome; response_time_ms: number; session_id: string }) =>
      post<{ comparison: Comparison; rankings?: unknown }>("/api/style/comparisons", body),
    rankings: async () => asList<RatingEntry>(await get<unknown>("/api/style/rankings"), ["rankings", "ratings"]),
    rankingsFull: async () => {
      const response = await get<{ ratings?: RatingEntry[]; n_candidates?: number; n_comparisons?: number }>("/api/style/rankings");
      return {
        ratings: asList<RatingEntry>(response, ["ratings", "rankings"]),
        n_candidates: response.n_candidates ?? 0,
        n_comparisons: response.n_comparisons ?? 0,
      };
    },
    preferenceVector: () => get<AestheticPreferenceVector>("/api/style/preference-vector"),
    space: async (architecture: ModelArchitecture, loraNames: string[], grouping: BlockGrouping) => {
      const search = new URLSearchParams({ architecture, grouping });
      for (const name of loraNames) {
        search.append("lora_names", name);
      }
      const response = await get<{ space?: ParameterSpace; dims?: unknown }>(`/api/style/optimizer/space?${search.toString()}`);
      if (response.space && Array.isArray(response.space.dims)) {
        return response.space;
      }
      return { dims: Array.isArray(response.dims) ? (response.dims as ParameterSpace["dims"]) : [] };
    },
    suggest: (body: SuggestRequest) => post<SuggestResponse>("/api/style/optimizer/suggest", body),
    exportRecipe: (body: RecipeExportRequest) => post<RecipeExportResponse>("/api/style/recipes/export", body),
    analyzeDataset: (body: { captions_text?: string; records?: CaptionRecord[]; trigger_token: string; use_llm: boolean }) =>
      post<DatasetAnalyzeResponse>("/api/style/dataset/analyze", body),
    tokens: async () => asList<TokenEntry>(await get<unknown>("/api/style/tokens"), ["tokens"]),
    addToken: (entry: TokenEntry) => post<unknown>("/api/style/tokens", entry),
    deleteToken: (token: string, category: string) => del<unknown>(`/api/style/tokens${buildQuery({ token, category })}`),
    grids: async () => asList<GridExperiment>(await get<unknown>("/api/style/grid"), ["experiments"]),
    createGrid: (body: GridCreateRequest) => post<GridDetail>("/api/style/grid", body),
    grid: (id: string) => get<GridDetail>(`/api/style/grid/${enc(id)}`),
    recordGridResult: (id: string, body: { cell_id: string; score: number; image_ref?: string | null; notes?: string }) =>
      post<unknown>(`/api/style/grid/${enc(id)}/results`, body),
  },
  manga: {
    list: async () => asList<StoryboardSummary>(await get<unknown>("/api/manga/storyboards"), ["storyboards"]),
    create: (body: StoryboardCreateRequest, onUpdate?: (job: Job) => void) =>
      runJob<Storyboard>("/api/manga/storyboards", { ...body }, onUpdate),
    get: (id: string) => get<Storyboard>(`/api/manga/storyboards/${enc(id)}`),
    remove: (id: string) => del<unknown>(`/api/manga/storyboards/${enc(id)}`),
    validate: (id: string) => post<GrammarReport>(`/api/manga/storyboards/${enc(id)}/validate`),
    autofix: (id: string) => post<Storyboard>(`/api/manga/storyboards/${enc(id)}/autofix`),
    pageSvgUrl: (id: string, page: number, options: { gaze: boolean; labels: boolean; guides: boolean }) =>
      `/api/manga/storyboards/${enc(id)}/pages/${page}/svg${buildQuery({
        gaze: options.gaze,
        labels: options.labels,
        guides: options.guides,
      })}`,
    spreadSvgUrl: (id: string, right: number, left: number, options: { gaze: boolean; labels: boolean; guides: boolean }) =>
      `/api/manga/storyboards/${enc(id)}/spreads/${right}/${left}/svg${buildQuery({
        gaze: options.gaze,
        labels: options.labels,
        guides: options.guides,
      })}`,
    layers: (id: string) => get<Record<string, unknown>>(`/api/manga/storyboards/${enc(id)}/layers`),
    compileComfy: (id: string, body: ComfyCompileRequest) => post<ComfyCompileResponse>(`/api/manga/storyboards/${enc(id)}/comfy`, body),
    queueComfy: (id: string, body: ComfyCompileRequest, onUpdate?: (job: Job) => void) =>
      runJob<Record<string, unknown>>(`/api/manga/storyboards/${enc(id)}/comfy/queue`, { ...body }, onUpdate),
  },
  diagnostics: {
    run: (body: DiagnosticRequest, onUpdate?: (job: Job) => void) =>
      runJob<DiagnosticReport>("/api/manga/diagnostics", { ...body }, onUpdate),
    list: async () => asList<DiagnosticSummary>(await get<unknown>("/api/manga/diagnostics"), ["diagnostics", "reports"]),
    get: (id: string) => get<DiagnosticReport>(`/api/manga/diagnostics/${enc(id)}`),
    remove: (id: string) => del<unknown>(`/api/manga/diagnostics/${enc(id)}`),
    overlaySvgUrl: (id: string, page: number, options: DiagnosticOverlayQuery) =>
      `/api/manga/diagnostics/${enc(id)}/pages/${page}/svg${buildQuery({
        heat: options.heat,
        gaze: options.gaze,
        pins: options.pins,
        order: options.order,
        image: options.image,
      })}`,
    apply: (id: string, body: { prescription_ids: string[]; save_storyboard: boolean }) =>
      post<DiagnosticApplyResponse>(`/api/manga/diagnostics/${enc(id)}/apply`, body),
    detect: (body: { images: string[]; first_page_side: PageSide; title?: string }) =>
      post<ImageDetectionResult>("/api/manga/diagnostics/detect", body),
    sampleImageUrl: (seed: number, variant: string) => `/api/manga/diagnostics/sample-image.png${buildQuery({ seed, variant })}`,
    forStoryboard: (id: string) => get<DiagnosticReport>(`/api/manga/storyboards/${enc(id)}/diagnostics`),
  },
  publisher: {
    matrix: (body: MatrixRequest) => post<ContentMatrix>("/api/publisher/matrix", body),
    articles: async () => asList<ArticleSummary>(await get<unknown>("/api/publisher/articles"), ["articles"]),
    createArticle: (body: ArticleRequest, onUpdate?: (job: Job) => void) =>
      runJob<PublishedArticle>("/api/publisher/articles", { ...body }, onUpdate),
    article: (id: string) => get<PublishedArticle>(`/api/publisher/articles/${enc(id)}`),
    removeArticle: (id: string) => del<unknown>(`/api/publisher/articles/${enc(id)}`),
    exportArticle: (id: string, format: "note" | "obsidian") => getText(`/api/publisher/articles/${enc(id)}/export${buildQuery({ format })}`),
    sanitize: (text: string) => post<SanitizeResponse>("/api/publisher/sanitize", { text }),
    verify: (text: string) => post<SanitizationReport>("/api/publisher/verify", { text }),
    gaps: async (topK = 6) => asList<ContentGap>(await get<unknown>(`/api/publisher/gaps${buildQuery({ top_k: topK })}`), ["gaps"]),
    series: (articleIds: string[], title: string) => post<SeriesResponse>("/api/publisher/series", { article_ids: articleIds, title }),
  },
};
