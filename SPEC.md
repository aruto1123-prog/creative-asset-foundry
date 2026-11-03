# Creative Asset Foundry (CAF) — 統合仕様書 `SPEC.md`

- 仕様バージョン: 1.0.0
- 対象ランタイム: Python 3.10 以上（開発・検証 3.13） / Node.js 18 以上（開発・検証 22）
- 本書の位置づけ: ユーザー指示書（Phase 0〜8）の全要件を構造化し、さらに **モジュール間インターフェース契約（Contract）** を確定させた唯一の正本。各モジュールの実装者（人間・サブエージェント問わず）は本書の契約に厳密に従う。

---

## 0. ビジョン

ローカルLLM（DeepSeek-V4.1-Flash）とローカル画像生成環境（ComfyUI / LoRA / モデルマージ）を統合し、次の4つの創作研究ループを永続的・自律的に回す統合プラットフォームを構築する。

| ループ | 内容 | 主担当モジュール |
|---|---|---|
| L1 構造ループ | 普遍的構造の抽出 → 構造代数（合成・変異）→ 異分野転用 | `core/structure_engine` |
| L2 美学ループ | 美的選好（Elo / Bradley-Terry）→ 選好ベクトル → ベイズ最適化 → レシピ出力 | `core/style_alchemy` |
| L3 制作ループ | 構造 → ネーム（コマ割り・カメラ・セリフ）→ 漫画文法監査 → ComfyUI → SVG原稿 | `core/manga_pipeline` |
| L4 資産化ループ | 研究ログ → 3軸企画 → 5段階執筆 → サニタイズ → ナレッジグラフ → 教本編纂 | `core/asset_publisher` |

全ループは `core/llm_client`（多段階推論ハーネス）と SQLite を共有し、`web_studio`（FastAPI + React）から操作・可視化される。

---

## 1. 絶対遵守ルール（全実装者共通）

| ID | ルール |
|---|---|
| R-01 | **Zero Placeholder Policy**: `TODO` / `FIXME` / `XXX` / `NotImplementedError` / 単独の `pass` 文 / プレースホルダ用 `...` 本体 / 「以下同様」「ダミー実装」「（省略）」等の省略表現をコード中に一切書かない。例外クラス等の空本体は docstring で表現する。`tests/test_quality_gates.py` が機械的に検査する。 |
| R-02 | **Deterministic Core + LLM Amplification**: LLMを使う全機能は「LLMなしで完全に動作する決定論的アルゴリズム（ベースライン）」を持ち、LLMはその出力を多段階ハーネスで増幅・精緻化する。LLM接続不能時もシステムは劣化動作で機能し続ける。 |
| R-03 | **Dependency Injection**: LLMアダプタ・ComfyUIトランスポート・DBは全てコンストラクタ注入。テストは高精度モック（`MockLLMAdapter` / `MockComfyServer`）で外部ネットワークに一切依存せず決定論的に完走する。 |
| R-04 | **脱フレーバー保証**: 構造データ（Layer 1〜5）に固有名詞・ジャンル固有装飾を残さない。抽出時に除去したトークンは再走査し、漏洩を自動置換する。 |
| R-05 | **記事コンプライアンス**: Asset Publisher の出力記事は一人称を「私」に統一し、本文中のURL（http/https、Markdownリンク、HTMLアンカー、短縮URL、難読化URL）を100%除去する。最終出力の直前に必ずサニタイザーを通す。 |
| R-06 | **数値計算の自前実装**: 最適化・統計・類似度計算（GP、CMA-ES、GA、Bradley-Terry、DTW、ハンガリアン法、PageRank等）は標準ライブラリ＋NumPyのみで実装する（SciPy・scikit-learn・PyTorch等に依存しない）。 |
| R-07 | **Python 3.10 互換**: `from __future__ import annotations` を使用。`StrEnum` / `typing.Self` / `ExceptionGroup` 等 3.11+ 専用機能は使わない。列挙型は `class X(str, Enum)` とし、文字列化は必ず `.value` を使う（3.12+ の f-string 仕様差異を回避）。 |
| R-08 | **永続化**: すべての永続データは `core.db.Database`（SQLite, WAL）経由。各モジュールは `Database.ensure_schema(name, ddl)` で自分のテーブルを冪等に作成する。 |
| R-09 | **テスト**: 各Phaseのテストは `pytest` で全件グリーン。1ファイルあたり概ね60秒以内。乱数は必ずシード固定。 |

---

## 2. ディレクトリ構成

```
creative-asset-foundry/
├── SPEC.md / PROGRESS.md / README.md / pyproject.toml / run_studio.py
├── core/
│   ├── __init__.py            # __version__
│   ├── common.py              # 共通ユーティリティ（ID・時刻・安定ハッシュ・clamp・slugify）
│   ├── config.py              # FoundrySettings（LLM/ComfyUI/パス設定・環境変数上書き・JSON永続化）
│   ├── db.py                  # Database（SQLite WAL・スキーマ登録・統計・バックアップ）
│   ├── llm_client/            # Phase 1: DeepSeek-V4.1-Flash 特化 多段階推論ハーネス
│   ├── structure_engine/      # Phase 2: 普遍的構造抽出・構造代数・異分野トランスパイラ
│   │   └── seeds/             # Phase 6: 初期プリセット純粋骨格 60パターン（JSON + ローダー + 著作スクリプト）
│   ├── style_alchemy/         # Phase 3: 選好学習・階層マージ/LoRAベイズ最適化・データセット精製・グリッド実験
│   ├── manga_pipeline/        # Phase 4: ネーム生成・漫画文法検証・ComfyUIコンパイラ・SVGレンダラー
│   └── asset_publisher/       # Phase 5: 3軸展開・5段階執筆・サニタイズ・ナレッジグラフ・編纂
├── web_studio/
│   ├── backend/               # Phase 7: FastAPI + WebSocket + SQLite
│   └── frontend/              # Phase 7: React 18 + TypeScript + Vite + Tailwind CSS
├── tests/                     # 全単体テスト + E2E
└── docs/                      # ARCHITECTURE.md / MASTER_GUIDE.md
```

---

## 3. 技術スタック

- Python: `pydantic>=2.6`, `numpy>=1.26`, `httpx>=0.27`, `fastapi>=0.110`, `uvicorn[standard]>=0.29`, `websockets>=12`
- テスト: `pytest>=8`, `pytest-asyncio>=0.23`（`asyncio_mode = "auto"`）
- フロントエンド: React 18.3 / TypeScript 5.x / Vite 5.x / Tailwind CSS 3.4 / `marked`（Markdownプレビュー）
- 既定ポート: バックエンド `8765`、Vite 開発サーバ `5173`（`/api` と `/ws` をバックエンドへプロキシ）
- 既定外部エンドポイント: OpenAI互換 `http://localhost:8000/v1`、Ollama `http://localhost:11434/v1`、ComfyUI `http://127.0.0.1:8188`

---

## 4. 共通基盤契約

### 4.1 `core/common.py`
```python
def utc_now_iso() -> str                      # "2026-11-03T12:00:00+00:00"
def new_id(prefix: str) -> str                # f"{prefix}_{uuid4().hex[:12]}"
def stable_hash(*parts: object) -> int        # SHA-256 由来の64bit整数（プロセス間で不変）
def stable_seed(*parts: object) -> int        # 0 <= seed < 2**31-1
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float
def slugify(text: str, max_len: int = 48) -> str   # ASCII slug。和文のみ等で空ならハッシュ由来
def truncate(text: str, limit: int) -> str
```

### 4.2 `core/config.py`
```python
class HarnessSettings(BaseModel):
    enable_decompose: bool = True; enable_critique: bool = True; enable_refine: bool = True
    critique_accept_score: float = 85.0; skip_refine_when_accepted: bool = False
    max_repair_attempts: int = 3; max_refine_rounds: int = 2
    stage_temperatures: dict[str, float] = {"decompose":0.4,"synthesize":0.3,"critique":0.1,"refine":0.2}
class LLMSettings(BaseModel):
    provider: Literal["auto","mock","openai_compatible","ollama"] = "auto"
    base_url: str = "http://localhost:8000/v1"; ollama_base_url: str = "http://localhost:11434/v1"
    model: str = "DeepSeek-V4.1-Flash"; ollama_model: str = "deepseek-v4.1-flash"; api_key: str = "EMPTY"
    timeout_s: float = 300.0; temperature: float = 0.3; max_tokens: int = 8192
    structured_output: Literal["json_schema","json_object","prompt_only"] = "json_object"
    mock_seed: int = 20260101; harness: HarnessSettings = HarnessSettings()
class ComfySettings(BaseModel):
    base_url: str = "http://127.0.0.1:8188"; timeout_s: float = 600.0; client_id_prefix: str = "caf"
    output_prefix: str = "caf"; use_mock: bool = False
class FoundrySettings(BaseModel):
    data_dir: Path; llm: LLMSettings; comfy: ComfySettings
    db_path / backup_dir / export_dir / config_path: Path   # プロパティ（data_dir 配下）
    @classmethod def load(cls, data_dir: Path | None = None) -> FoundrySettings   # config.json + 環境変数
    def save(self) -> Path
    def public_dict(self) -> dict                                   # api_key をマスク
```
環境変数上書き: `CAF_DATA_DIR`, `CAF_LLM_PROVIDER`, `CAF_LLM_BASE_URL`, `CAF_LLM_MODEL`, `CAF_LLM_API_KEY`, `CAF_OLLAMA_BASE_URL`, `CAF_COMFY_URL`, `CAF_COMFY_MOCK`。

### 4.3 `core/db.py`
```python
class Database:
    def __init__(self, path: str | Path)              # 親ディレクトリ自動作成, WAL, _schema_registry 作成
    path: Path
    def connect(self) -> ContextManager[sqlite3.Connection]   # row_factory=Row, commit/rollback 自動
    def execute(self, sql: str, params: Sequence | Mapping = ()) -> ExecResult  # .lastrowid / .rowcount
    def executemany(self, sql: str, seq: Iterable) -> int
    def query(self, sql: str, params=()) -> list[dict[str, Any]]
    def query_one(self, sql: str, params=()) -> dict[str, Any] | None
    def scalar(self, sql: str, params=()) -> Any
    def ensure_schema(self, name: str, ddl: str) -> None       # 冪等。登録済みならスキップ
    def tables(self) -> list[str]
    def table_stats(self) -> dict[str, int]
    def size_bytes(self) -> int
    def backup(self, dest: str | Path) -> Path                 # sqlite3 オンラインバックアップAPI
```

---

## 5. Phase 1 — `core/llm_client/`：DeepSeek-V4.1-Flash 多段階推論基盤

### 5.1 要件
| ID | 要件 |
|---|---|
| P1-01 | OpenAI互換API（`http://localhost:8000/v1`：vLLM / SGLang / llama.cpp server / LM Studio 等）と Ollama API（`http://localhost:11434/v1`）の両方に対応する非同期クライアント。 |
| P1-02 | **4段階思考増幅パイプライン**: Step1 `Decompose`（本質要素・制約分解）→ Step2 `Synthesize`（Pydantic v2 JSON Schema 厳密準拠ドラフト）→ Step3 `Critique`（論理矛盾・構造の浅さ・抽象化不足のルーブリック自己批評）→ Step4 `Refine`（批評反映の最終構造化データ）。 |
| P1-03 | JSON抽出・スキーマ検証失敗時の **自動自己修復リトライ**（検証エラーをフィードバックした再生成）。 |
| P1-04 | 推論トークン数（prompt/completion/reasoning）・レイテンシ・思考ログ（reasoning_content / `<think>`）の **SQLite自動記録**。 |
| P1-05 | 全Pydanticスキーマに完全準拠した **決定論的かつリアルな高精度モックLLMアダプタ**（DI切替、タスク別ハンドラ登録、JSON Schema合成器、障害注入）。 |
| P1-06 | `auto` プロバイダ: 実エンドポイント（OpenAI互換→Ollama）を順に試行し、到達不能ならモックへ自動フォールバック（状態を公開）。 |

### 5.2 ファイル構成と責務
| ファイル | 責務 |
|---|---|
| `types.py` | `ChatMessage`, `HarnessStage`, `LLMRequest`, `TokenUsage`, `LLMResponse`, `StructuredResult[T]` |
| `errors.py` | `LLMError` ← `LLMConnectionError`, `LLMResponseError`, `JSONExtractionError`, `StructuredOutputError`, `HarnessError` |
| `tokens.py` | `estimate_tokens(text)`（ASCII≈4文字/トークン、CJK≈1文字/トークンの混合推定） |
| `json_repair.py` | `strip_reasoning_tags`, `extract_json_block`, `repair_json_text`, `parse_json_lenient`, `format_validation_errors` |
| `adapters.py` | `BaseLLMAdapter`, `OpenAICompatibleAdapter`, `OllamaAdapter`, `AutoFallbackAdapter` |
| `mock.py` | `MockLLMAdapter`, `MockContext`, `register_mock_handler`, `get_mock_handler`, `JsonSchemaSynthesizer`, 既定 decompose/critique ハンドラ |
| `telemetry.py` | `TelemetryStore`（`llm_calls` / `harness_runs` / `harness_stage_logs`） |
| `client.py` | `LLMClient`（chat / generate_structured / 自己修復 / テレメトリ） |
| `prompts.py` | DeepSeek向け長文システムプロンプト群（ペルソナ、Decompose、Synthesize、Critique、Refine、スキーマ指示、修復指示） |
| `harness.py` | `RubricCriterion`, `DEFAULT_RUBRIC`, `DecompositionResult`, `CritiqueReport`, `HarnessConfig`, `StageLog`, `HarnessResult`, `MultiStageReasoningHarness` |
| `factory.py` | `create_adapter(settings)`, `create_llm_client(settings=None, db=None, adapter=None)` |

### 5.3 公開API契約（`from core.llm_client import ...`）
```python
class ChatMessage(BaseModel): role: Literal["system","user","assistant"]; content: str
def system_msg(text) -> ChatMessage; def user_msg(text) -> ChatMessage; def assistant_msg(text) -> ChatMessage
class HarnessStage(str, Enum): DECOMPOSE="decompose"; SYNTHESIZE="synthesize"; CRITIQUE="critique"; REFINE="refine"; REPAIR="repair"; SINGLE="single"
class LLMRequest(BaseModel):
    messages: list[ChatMessage]; model: str | None = None; temperature: float = 0.3; max_tokens: int = 4096
    response_schema: dict | None = None; schema_name: str | None = None
    task: str = "generic"; stage: HarnessStage = HarnessStage.SINGLE
    context: dict[str, Any] = {}      # モックハンドラ用の構造化コンテキスト（実LLMには送らない）
    run_id: str | None = None; attempt: int = 1; json_mode: bool = False
class TokenUsage(BaseModel): prompt_tokens: int = 0; completion_tokens: int = 0; reasoning_tokens: int = 0; estimated: bool = False
    total_tokens: int  (property) ; __add__
class LLMResponse(BaseModel): content: str; reasoning: str = ""; model: str; provider: str; usage: TokenUsage; latency_ms: float; finish_reason: str = "stop"; raw: dict | None = None
class StructuredResult(BaseModel, Generic[T]): value: T; response: LLMResponse; attempts: int; raw_text: str; syntax_repaired: bool

class BaseLLMAdapter(ABC):
    provider: str; model: str
    async def complete(self, request: LLMRequest) -> LLMResponse          # abstract
    async def health_check(self) -> dict   # {"ok": bool, "provider": str, "model": str, "detail": str, "models": list[str]}
    async def aclose(self) -> None
class OpenAICompatibleAdapter(BaseLLMAdapter):
    def __init__(self, base_url="http://localhost:8000/v1", model="DeepSeek-V4.1-Flash", api_key="EMPTY",
                 timeout_s=300.0, structured_output="json_object", extra_body: dict | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, max_transport_retries: int = 2,
                 retry_backoff_s: float = 0.5)
    # POST {base_url}/chat/completions。structured_output="json_schema" → response_format={"type":"json_schema","json_schema":{"name","schema","strict":False}}
    # "json_object" → {"type":"json_object"}、"prompt_only" → response_format なし。
    # message.reasoning_content / message.reasoning を思考ログとして取得。content 内 <think>…</think> は除去して思考ログへ。
    # usage.completion_tokens_details.reasoning_tokens を取得。usage欠落時は推定（estimated=True）。
    # 429/5xx/TransportError は指数バックオフで再試行、最終的に LLMConnectionError。
    async def list_models(self) -> list[str]                               # GET {base_url}/models
class OllamaAdapter(OpenAICompatibleAdapter):  # 既定 base_url="http://localhost:11434/v1", model="deepseek-v4.1-flash", api_key="ollama"
    # health_check: GET {root}/api/tags（root は /v1 を除いたURL）→ 失敗時 /v1/models
class AutoFallbackAdapter(BaseLLMAdapter):
    def __init__(self, primaries: list[BaseLLMAdapter], fallback: BaseLLMAdapter, probe_timeout_s: float = 2.0)
    fallback_active: bool; selected_provider: str     # 初回リクエスト時に health_check で選択し以後キャッシュ

class MockContext:  # dataclass
    request: LLMRequest; task: str; stage: str; schema: dict | None; schema_name: str | None
    context: dict; seed: int; rng: random.Random; attempt: int
    def synthesize(self, schema: dict | None = None) -> Any   # JSON Schema から決定論的インスタンス生成
    draft: Any (property = context.get("draft")); critique: Any (property = context.get("critique"))
MockHandler = Callable[[MockContext], Any]   # dict/list を返す。None を返すと汎用合成へフォールバック
def register_mock_handler(task: str, stages: str | Iterable[str] = "*") -> Callable[[MockHandler], MockHandler]
def get_mock_handler(task: str, stage: str) -> MockHandler | None
# 解決順序: (task, stage) → (task, "*") → ("*", stage) → 汎用 JsonSchemaSynthesizer
# 既定登録: ("*","decompose") と ("*","critique") は mock.py 内蔵の解析的ハンドラ。
MOCK_HANDLER_MODULES = ("core.structure_engine.mock_handlers", "core.style_alchemy.mock_handlers",
                        "core.manga_pipeline.mock_handlers", "core.asset_publisher.mock_handlers")
# 初回 complete() 時に遅延 import（当該モジュール自身の ModuleNotFoundError のみ無視、それ以外の例外は送出）

class MockLLMAdapter(BaseLLMAdapter):
    provider = "mock"
    def __init__(self, seed: int = 20260101, model: str = "mock-deepseek-v4.1-flash",
                 failure_plan: dict[str, list[str]] | None = None, emit_think_tags: bool = True,
                 simulated_latency_ms: float = 0.0)
    # failure_plan のキー: "task:stage" | "task:*" | "*:*"。値は呼び出し毎に消費される挙動キュー:
    #   "ok" | "garbage"（JSONなし）| "schema_violation"（必須欠落JSON）| "repairable_json"（末尾カンマ等の修復可能JSON）
    #   | "connection_error"（LLMConnectionError送出）| "empty"（空文字）
    # 正常応答の content: ("<think>…</think>\n" if emit_think_tags) + "```json\n{...}\n```"、reasoning にも思考ログ。
    # 乱数シード = stable_seed(seed, task, stage, schema_name, attempt, sha256(全メッセージ))
    calls: list[LLMRequest]       # 受信リクエスト履歴（テスト検証用）

class JsonSchemaSynthesizer:
    def __init__(self, schema: dict, seed: int = 0)
    def generate(self) -> Any
    # $ref/$defs, anyOf/oneOf/allOf, enum/const, type配列, min/maxItems, uniqueItems, minimum/maximum/exclusive*,
    # multipleOf, min/maxLength, pattern（既知パターン＋候補検証）, format(date-time/uuid/date), default に対応。
    # プロパティ名ヒューリスティック（title/name/label/description/summary/id/tags/score 等）で日本語のリアルな値を生成。
    # URLは生成しない。

class TelemetryStore:
    def __init__(self, db: Database)
    def record_call(self, *, run_id, task, stage, provider, model, attempt, success, error, usage: TokenUsage,
                    latency_ms, request_excerpt, response_excerpt, reasoning_log) -> int
    def start_run(self, run_id, task, objective, stages: list[str]) -> None
    def record_stage(self, run_id, log: StageLog) -> None
    def finish_run(self, run_id, *, status, usage: TokenUsage, latency_ms, critique_score, error=None) -> None
    def recent_calls(self, limit=50) -> list[dict]; def recent_runs(self, limit=20) -> list[dict]
    def get_run(self, run_id) -> dict | None      # stages と calls を内包
    def stats(self) -> dict    # 総計 + task/stage 別（件数・成功率・平均レイテンシ・トークン合計）

class LLMClient:
    def __init__(self, adapter: BaseLLMAdapter, telemetry: TelemetryStore | None = None,
                 default_model: str | None = None, max_repair_attempts: int = 3)
    adapter; telemetry; provider (property); model (property)
    async def chat(self, messages, *, task="generic", stage=HarnessStage.SINGLE, temperature=0.3,
                   max_tokens=4096, run_id=None, context=None, json_mode=False) -> LLMResponse
    async def generate_structured(self, output_model: type[T], messages: list[ChatMessage], *, task: str,
                   stage=HarnessStage.SINGLE, temperature=0.2, max_tokens=8192, run_id=None,
                   context: dict | None = None, max_repair_attempts: int | None = None) -> StructuredResult[T]
    # スキーマ指示（prompts.build_schema_instruction）を system に付与 → 応答を parse_json_lenient → model_validate。
    # 失敗時は assistant(失敗出力) + user(修復指示: 検証エラー一覧 + スキーマ名) を追記して再試行（attempt+1, 同stage, context["repair"]=True）。
    # 全試行失敗で StructuredOutputError(raw_text, validation_errors, attempts)。全呼び出しをテレメトリ記録。
    async def health_check(self) -> dict; async def aclose(self) -> None

class RubricCriterion(BaseModel): key: str; description: str; weight: float = 1.0
DEFAULT_RUBRIC: list[RubricCriterion]  # logical_consistency, structural_depth, abstraction_quality, schema_fidelity, actionability
class DecompositionResult(BaseModel):
    essence: str; core_elements: list[ElementItem] (min1); constraints: list[ConstraintItem] (min1)
    hidden_assumptions: list[str]; success_criteria: list[str] (min1); abstraction_targets: list[str]; risks: list[str]
    # ElementItem{name, description, importance∈[0,1]} / ConstraintItem{name, description, kind: "hard"|"soft"}
class CriterionScore(BaseModel): key: str; score: float (0..10); rationale: str
class CritiqueReport(BaseModel):
    scores: list[CriterionScore] (min1); logical_contradictions: list[str]; shallow_points: list[str]
    abstraction_gaps: list[str]; must_fix: list[str]; overall_score: float (0..100)
    verdict: Literal["accept","revise"]; summary: str
class HarnessConfig(BaseModel):  # HarnessSettings と同フィールド + @classmethod from_settings(HarnessSettings)
class StageLog(BaseModel): stage: str; ordinal: int; latency_ms: float; usage: TokenUsage; attempts: int; success: bool; summary: str; thought_excerpt: str
class HarnessResult(BaseModel, Generic[T]):
    run_id: str; task: str; output: T; draft: T | None; decomposition: DecompositionResult | None
    critique: CritiqueReport | None; stage_logs: list[StageLog]; usage: TokenUsage; total_latency_ms: float
    refined: bool; refine_rounds: int
class MultiStageReasoningHarness:
    def __init__(self, client: LLMClient, config: HarnessConfig | None = None)
    async def run(self, *, task: str, objective: str, payload: Any, output_model: type[T], system_prompt: str,
                  rubric: list[RubricCriterion] | None = None, context: dict | None = None,
                  config: HarnessConfig | None = None,
                  post_validate: Callable[[T], list[str]] | None = None,
                  on_stage: Callable[[StageLog], Any] | None = None) -> HarnessResult[T]
    # モックへ渡す context キー: 全段共通 {**context, "objective", "payload"}
    #   synthesize: +{"decomposition"} / critique: +{"draft","rubric","accept_score"} / refine: +{"draft","critique","must_fix"}
    # Decompose/Critique 失敗は警告ログを残して続行。Synthesize 失敗は HarnessError。Refine 失敗はドラフトを採用。
    # post_validate が問題を返した場合 max_refine_rounds まで追加 Refine（must_fix に問題を注入）。
def create_adapter(settings: LLMSettings) -> BaseLLMAdapter
def create_llm_client(settings: LLMSettings | None = None, db: Database | None = None,
                      adapter: BaseLLMAdapter | None = None) -> LLMClient   # CAF_LLM_PROVIDER が最優先
```

### 5.4 LLM連携機能の実装パターン（全モジュール共通）
```python
# 1) 決定論的ベースライン（必須）
def heuristic_task(inputs) -> OutputModel: ...
# 2) <module>/mock_handlers.py にタスク別モックハンドラを登録（ベースラインからリアルな応答を合成）
@register_mock_handler("module.task", stages=("synthesize", "refine"))
def _mock_task(ctx: MockContext) -> dict:
    return heuristic_task(ctx.context["inputs"]).model_dump(mode="json")
# 3) 本番経路: ハーネスで増幅
result = await harness.run(task="module.task", objective=..., payload=..., output_model=OutputModel,
                           system_prompt=MODULE_PROMPT, context={"inputs": inputs_dict})
```
登録済みタスク名一覧: `structure.extract`, `structure.transpile`, `style.caption_cleanse`, `manga.dialogue`, `publisher.matrix`, `publisher.blueprint`, `publisher.section_draft`, `publisher.critique`, `publisher.polish`。

---

## 6. Phase 2 — `core/structure_engine/`：普遍的構造抽出＆異分野転用エンジン

### 6.1 要件
| ID | 要件 |
|---|---|
| P2-01 | 5階層オントロジー（`schemas.py`）: L1 Dynamic Tension Graph / L2 Emotional & Information Wave（t∈[0,1]、21点=20分割以上）/ L3 Structural Beats & Turning Points / L4 Cognitive & Empathy Mechanics / L5 Domain-Agnostic Blueprint（`{{SLOT}}` 抽象変数） |
| P2-02 | 脱フレーバー抽出器（`extractor.py`）: 固有名詞・ジャンル装飾を除去し5階層JSONへ逆コンパイル。構造主義物語論に基づくDeepSeek用システムプロンプト群（`prompts.py`）。 |
| P2-03 | 構造代数（`composer.py`）: `merge_structures(a, b, weight_topology, weight_wave)`（Aの対立トポロジー×Bの感情・情報開示波形の補間合成、矛盾自動修正）と `mutate_structure(s, mutation_type)`（倒叙化・権力勾配反転・偽クライマックス挿入 等7種）。 |
| P2-04 | 異分野トランスパイラ（`transpiler.py`）: ①漫画プロット ②高CVR LP/セールスファネル ③体系的考察・解説記事 へ、力学と感情波形を保存したまま変換（忠実度スコア付き）。 |
| P2-05 | 類似度検索＆可視化（`similarity.py`, `repository.py`, `visualizer.py`）: SQLite保存・検索、コサイン類似度・DTW（Sakoe-Chiba窓付き、多変量）、Mermaid記法、SVG波形。 |

### 6.2 `schemas.py` 契約（正本は実装ファイル。要点のみ）
```python
WAVE_MIN_POINTS = 21
DEFAULT_CHANNELS = ("tension", "release", "information_asymmetry", "cognitive_load")
class NodeKind(str, Enum): AGENT="agent"(主体); DESIRE="desire"(欲求); LACK="lack"(欠落); SYMBOL="symbol"(象徴)
class EdgeKind(str, Enum): OPPOSITION="opposition"; DEPENDENCY="dependency"; ASYMMETRY="asymmetry";
                           POWER_GRADIENT="power_gradient"; BETRAYAL="betrayal"; PURSUIT="pursuit"; EMBODIMENT="embodiment"
class TensionNode: id(^[a-z][a-z0-9_]{0,63}$), kind, role, label, description="", weight∈[0,1]=0.5, slot: str|None(^[A-Z][A-Z0-9_]*$)
class TensionEdge: source, target, kind, intensity∈[0,1]=0.5, gradient∈[-1,1]=0.0(正=source優位), label="", activation_t∈[0,1]|None
class DynamicTensionGraph: nodes(min2), edges(min1)  # 参照整合・重複禁止・自己ループ禁止
    node_ids(), get_node(id), nodes_of_kind(k), edges_of_kind(k), adjacency(kind=None)->ndarray,
    power_balance()->dict, tension_energy()->float, edge_kind_histogram()->dict, degree_centrality()->dict
class EmotionalInformationWave:  # 全チャネル長さ一致・len>=21・t[0]=0,t[-1]=1 狭義単調増加・値∈[0,1]
    t, tension, release, audience_knowledge, character_knowledge, cognitive_load: list[float]
    information_asymmetry: list[float]   # computed_field = audience_knowledge - character_knowledge ∈[-1,1]
    channel(name)->ndarray, as_matrix(channels=DEFAULT_CHANNELS)->ndarray(C×N), resample(n>=21)->Wave,
    resample_arrays(n)->dict[str,ndarray], sample(channel, t)->float, peak_index(channel="tension")->int,
    suspense_coefficient()->float, surprise_coefficient()->float, catharsis_index()->float, summary()->dict
    @classmethod from_keyframes(keyframes: dict[str, list[tuple[float,float]]], n=21) -> Wave  # PCHIP単調補間
class BeatType(str, Enum): HOOK, PREMISE, SETUP, ESCALATION, LOCAL_OPTIMIZATION_LIMIT, FALSE_VICTORY, CRISIS,
                           REVELATION, STRUCTURAL_TWIST, COSTLY_BREAKTHROUGH, CLIMAX, PAYOFF, AFTERGLOW, TRANSITION
class StructuralBeat: id, beat_type, t∈[0,1], span∈[0,1]=0.05, label, function, causes: list[str](先行ビートのみ),
                      tension_delta∈[-1,1], information_shift∈[-1,1], involved_nodes: list[str], is_turning_point: bool
class StructuralBeatChain: beats(min3)  # id一意・t非減少・因果は先行ビート参照（DAG）
    get(id), turning_points(), by_type(t), causal_edges(), causal_depth()
class MechanismCategory(str, Enum): curiosity_gap, dramatic_irony, empathy_transfer, prediction_error, cognitive_dissonance,
    identity_projection, moral_judgment, loss_aversion, social_proof, zeigarnik, peak_end, catharsis, scarcity,
    reciprocity, authority, commitment_consistency, narrative_transportation, anticipation_reward, contrast_effect, self_efficacy
class CognitiveMechanism: id, category, name, description, trigger, effect, theory="", strength∈[0,1], linked_beats, linked_nodes
class CognitiveEmpathyMechanics: mechanisms(min1), empathy_anchor, immersion_drivers(min1), emotion_sequence(min2)
class SlotCategory(str, Enum): agent, goal, lack, conflict, irony, reversal, cost, symbol, setting, audience, offer, evidence, emotion, other
class BlueprintSlot: name(^[A-Z][A-Z0-9_]{1,63}$), description, category, constraints=[], examples: dict[str,str]={}, required=True
class DomainAgnosticBlueprint: template(min20, `{{SLOT}}` を含む), slots(min1), invariants(min1)
    placeholders()->list[str], missing_bindings(bindings)->list[str], render(bindings, strict=True)->str
class StructureCategory(str, Enum): NARRATIVE_ARCHETYPE="narrative_archetype"; MANGA_PANEL_RHYTHM="manga_panel_rhythm";
                                    PERSUASION_MARKETING="persuasion_marketing"; CUSTOM="custom"
class SourceKind(str, Enum): SEED, EXTRACTED, MERGED, MUTATED, MANUAL
class StructureMetadata: title, summary, category=CUSTOM, source_kind=MANUAL, tags=[], origin_domain="generic",
                         lineage: list[str]=[](物語論上の系譜。原型名可・脱フレーバー検査対象外), parent_ids=[], language="ja",
                         version=1, created_at, notes=""
class PanelRhythmProfile: page_count∈[1,64], panels_per_page(len=page_count, 各1..12), emphasis_pages, page_turn_reveals,
                          spread_pages, silent_panels_ratio∈[0,1], pacing_notes
class TransferGuide: manga="", marketing="", article=""
class StructureLayers: title, summary, tags, tension_graph, wave, beats, mechanics, blueprint, transfer_guide  # LLM抽出出力用
class UniversalStructure: id(^[a-z0-9][a-z0-9_\-]{2,95}$), metadata, tension_graph, wave, beats, mechanics, blueprint,
                          transfer_guide=TransferGuide(), rhythm_profile: PanelRhythmProfile | None = None
    # 横断検証: ビートの involved_nodes ⊆ ノード、メカニズムの linked_* ⊆ ビート/ノード、node.slot ⊆ スロット名
    layer_summary()->dict, text_fields()->list[str](lineage除く全可読文字列), @classmethod from_layers(layers, id, metadata)
def new_structure_id(prefix: str = "st") -> str
```

### 6.3 その他ファイル契約
```python
# extractor.py
class FlavorToken(BaseModel): surface: str; category: Literal["person","place","organization","artifact","genre_term","title","other"]; placeholder: str; occurrences: int
class DeflavorResult(BaseModel): abstracted_text: str; tokens: list[FlavorToken]; genre_signals: dict[str, float]
def deflavor_text(text: str) -> DeflavorResult
def heuristic_extract(text: str, *, source_kind: str = "synopsis", title_hint: str | None = None) -> StructureLayers
def find_flavor_leaks(structure: UniversalStructure, tokens: list[FlavorToken]) -> list[str]
def scrub_flavor(structure: UniversalStructure, tokens: list[FlavorToken]) -> UniversalStructure
class ExtractionResult(BaseModel): structure: UniversalStructure; deflavor: DeflavorResult; harness_run_id: str | None; used_llm: bool; leaked_tokens_fixed: list[str]; critique_score: float | None
class StructureExtractor:
    def __init__(self, llm_client: LLMClient | None = None, harness_config: HarnessConfig | None = None)
    async def extract(self, text: str, *, source_kind: str = "synopsis", title_hint: str | None = None,
                      category: StructureCategory = StructureCategory.CUSTOM) -> ExtractionResult
# consistency.py
class ConsistencyIssue(BaseModel): code: str; severity: Literal["error","warning"]; message: str; location: str
def audit_structure(s: UniversalStructure) -> list[ConsistencyIssue]
# composer.py   weight_topology = 位相におけるAの比率(1.0=純A) / weight_wave = 波形におけるBの比率(1.0=純B)
class MutationType(str, Enum): INVERT_DISCLOSURE="invert_disclosure"; INVERT_POWER_GRADIENT="invert_power_gradient";
    INSERT_FALSE_CLIMAX="insert_false_climax"; REVERSE_CHRONOLOGY="reverse_chronology"; ADD_BETRAYAL="add_betrayal";
    PERSPECTIVE_SWAP="perspective_swap"; COMPRESS_MIDDLE="compress_middle"
MUTATION_DESCRIPTIONS: dict[str, str]
class MergeReport(BaseModel): node_mapping: dict[str, str | None]; weight_topology: float; weight_wave: float; dtw_alignment_cost: float; fixes_applied: list[str]; issues_remaining: list[str]
def interpolate_waves(wave_a, wave_b, weight_b: float, *, n: int | None = None, align: bool = True) -> EmotionalInformationWave
def merge_structures(struct_a, struct_b, weight_topology: float = 0.7, weight_wave: float = 0.7, *, new_id=None, title=None) -> UniversalStructure
def merge_structures_with_report(...same...) -> tuple[UniversalStructure, MergeReport]
def mutate_structure(struct, mutation_type: MutationType | str, *, intensity: float = 1.0, seed: int = 0, new_id=None) -> UniversalStructure
# transpiler.py
class TargetDomain(str, Enum): MANGA_PLOT="manga_plot"; MARKETING_LP="marketing_lp"; ARTICLE="article"
class DesignSection(BaseModel): index; beat_id; beat_type; t_start; t_end; heading; content; tension_target; release_target; information_reveal; emotional_goal; mechanisms: list[str]; notes: list[str]
class FidelityReport(BaseModel): wave_correlation: float; beat_coverage: float; topology_preserved: bool; score: float
class TranspiledDesign(BaseModel): id; structure_id; domain; title; logline; theme_variables; slot_bindings; rendered_blueprint; sections: list[DesignSection]; domain_payload: dict; fidelity: FidelityReport; created_at
def transpile_deterministic(structure, domain, theme_variables=None) -> TranspiledDesign
class StructureTranspiler: __init__(llm_client=None, harness_config=None); async transpile(structure, domain, theme_variables=None) -> TranspiledDesign
# similarity.py
def cosine_similarity(a, b) -> float; def dtw_distance(a, b, window: int | None = None) -> float
def dtw_path(a, b, window=None) -> tuple[float, list[tuple[int,int]]]
def wave_similarity(w1, w2, *, method: Literal["dtw","cosine","hybrid"]="hybrid", channels=DEFAULT_CHANNELS, n=32) -> dict  # {"score","dtw_distance","cosine"}
def graph_similarity(g1, g2) -> float
# repository.py
class StructureSummary(BaseModel): id, title, summary, category, source_kind, tags, created_at, updated_at, peak_t, suspense, surprise
class SimilarityHit(BaseModel): id, title, category, score∈[0,1], dtw_distance: float|None, cosine: float|None
class StructureRepository:
    def __init__(self, db: Database); save(s)->str(upsert); get(id); list(*, category=None, source_kind=None, tag=None, query=None, limit=200, offset=0)->list[StructureSummary]
    delete(id)->bool; count(category=None)->int; all()->list[UniversalStructure]
    search_similar(query: UniversalStructure | EmotionalInformationWave, *, method="hybrid", top_k=5, category=None, exclude_ids=())->list[SimilarityHit]
# visualizer.py
def graph_to_mermaid(structure_or_graph) -> str; def beats_to_mermaid(structure) -> str
def wave_to_svg(wave, *, width=800, height=320, channels=DEFAULT_CHANNELS, beats=None, title=None, compare=None) -> str
def structure_report_markdown(structure) -> str
# seeds/loader.py
SEED_FILES = {"narrative_archetype": "narrative_archetypes.json", "manga_panel_rhythm": "manga_panel_rhythm_patterns.json", "persuasion_marketing": "persuasion_and_marketing_structures.json"}
def seed_dir() -> Path; def load_seed_file(filename: str) -> list[UniversalStructure]; def load_all_seeds() -> list[UniversalStructure]
class SeedReport(BaseModel): inserted: int; updated: int; skipped: int; total: int; by_category: dict[str, int]
def seed_database(repo: StructureRepository, *, overwrite: bool = False) -> SeedReport
```

---

## 7. Phase 3 — `core/style_alchemy/`：極限絵柄開発・モデル調合＆LoRA研究ラボ

### 7.1 要件
| ID | 要件 |
|---|---|
| P3-01 | Elo レーティング（K減衰・引分対応）と Bradley-Terry 最尤推定（MMアルゴリズム、事前仮想試合による正則化、Fisher情報による標準誤差）。 |
| P3-02 | 実験パラメータ（線の入り抜き・ベタ比率・影コントラスト・ブロックマージ比率・LoRA重み等）を説明変数とするリッジ回帰（LOOCVでλ自動選択）＋ペアワイズ・ロジスティック回帰、標準化係数・Permutation重要度・ブートストラップ信頼区間による **美的選好ベクトル** 抽出。 |
| P3-03 | 階層別ブロックマージ（SDXL/SD1.5: BASE/IN/MID/OUT、FLUX: Double/Single Blocks、Anima等の汎用DiT）× LoRA Block Weight の多次元連続空間定義。 |
| P3-04 | ガウス過程回帰（RBF/ARD、ハイパラ最適化）＋獲得関数（EI/UCB/PI）＋ CMA-ES ＋ 遺伝的アルゴリズム（BLX-α交叉・ガウス突然変異）で「次に試すべきレシピ」を提案（NumPyのみ）。 |
| P3-05 | 最適レシピのエクスポート: マージ実行スクリプト用設定（JSON・MBW文字列・Pythonスクリプト）と ComfyUI API ノードグラフ。 |
| P3-06 | データセット精製: 絵柄（固定学習）と被写体制御要素（構図・アングル・ポーズ・表情・照明・背景密度）の分離キャプションクレンジング（ルール＋DeepSeek連携）。 |
| P3-07 | 共起行列・NPMI・エントロピー・条件付き確率による概念癒着検出と、不足条件（例: 俯瞰×逆光）の追加枚数を算出するギャップ分析。 |
| P3-08 | 多軸グリッド実験台帳: 構造化トークン辞書（線画タッチ・白黒トーン・パース歪み等）、X/Y/Z 3次元バッチ生成、結果追跡、主効果・交互作用分析。 |

### 7.2 共有スキーマ契約（`schemas.py` と `comfy_export.py` は統合基盤として先行実装）
```python
class ModelArchitecture(str, Enum): SDXL="sdxl"; SD15="sd15"; FLUX="flux"; ANIMA="anima"
ARCHITECTURE_BLOCKS: dict[ModelArchitecture, list[str]]
#  SDXL: BASE, IN00..IN08, MID, OUT00..OUT08 (20) / SD15: BASE, IN00..IN11, MID, OUT00..OUT11 (26)
#  FLUX: BASE, DB00..DB18, SB00..SB37 (58) / ANIMA(汎用DiT): BASE, B00..B27 (29)
LBW_BLOCKS: dict[ModelArchitecture, list[str]]   # LoRA Block Weight 慣行（SDXL 12点, SD15 17点 等）
def block_groups(arch) -> dict[str, list[str]]   # SDXL/SD15: BASE/IN/MID/OUT, FLUX: BASE/DOUBLE/SINGLE, ANIMA: BASE/EARLY/MIDDLE/LATE
class CheckpointRef: name, path="", architecture
class MergeMethod(str, Enum): WEIGHTED_SUM="weighted_sum"; ADD_DIFFERENCE="add_difference"
class BlockMergeRecipe: architecture, method, model_a, model_b, model_c=None, block_weights: dict[str,float](=Bの比率∈[0,1]), default_weight=0.5
    full_vector()->list[float]（正準順）, to_mbw_string()->str
class LoRASpec: name, path="", strength_model∈[-2,2]=1.0, strength_clip∈[-2,2]=1.0, block_weights: dict[str,float]|None, trigger_words=[]
    lbw_vector(arch)->list[float]
class SamplerSettings: sampler_name="dpmpp_2m", scheduler="karras", steps=28, cfg=6.0, denoise=1.0, clip_skip=1
class StyleRecipe: id, name, architecture, base_checkpoint: CheckpointRef, merge: BlockMergeRecipe|None=None, loras: list[LoRASpec]=[],
                   sampler=SamplerSettings(), vae: str|None=None, text_encoders: list[str]=[], style_tokens=[], negative_tokens=[],
                   params: dict[str,float]={}, created_at, notes=""
# comfy_export.py（ComfyUI API形式 {"<node_id>": {"class_type": str, "inputs": {...}, "_meta": {"title": str}}}、リンクは [node_id, output_index]）
class ComfyGraphBuilder: add(class_type, inputs, title=None)->str; ref(node_id, output_name_or_index)->list; to_prompt()->dict; validate()->list[str]
NODE_OUTPUTS: dict[str, list[str]]   # 既知ノードの出力名（CheckpointLoaderSimple→["MODEL","CLIP","VAE"] 等）
class ModelStackRefs(BaseModel): model: list; clip: list; vae: list
def build_model_stack(builder, recipe: StyleRecipe) -> ModelStackRefs   # ローダー→ブロックマージ→LoRAチェーン→CLIPSetLastLayer
def build_txt2img_workflow(recipe, positive: str, negative: str, *, seed: int, width: int, height: int, filename_prefix="caf") -> dict
```
注: ComfyUI の `ModelMergeBlocks` 系ノードの比率は **model1 の重み**（ratio=1.0 で純model1）。本システムの `block_weights` は **model_b の比率** であり、エクスポート時に `1 - w` へ変換する。

### 7.3 その他ファイル契約
```python
# preference_engine.py
class StyleCandidate: id, label, params: dict[str,float], recipe: StyleRecipe|None=None, image_ref: str|None=None, source: str="manual", created_at
class Comparison: id, left_id, right_id, outcome: Literal["left","right","tie"], response_time_ms: float|None=None, session_id: str|None=None, created_at
class RatingEntry: candidate_id, label, elo, bt_strength, bt_elo_scale, bt_std_error, games, wins, losses, ties, win_rate, rank
class EloEngine: __init__(k_factor=32.0, initial_rating=1500.0, scale=400.0, k_decay_games: float|None=30.0); expected_score(ra, rb); update(ra, rb, score_a, games_a=0, games_b=0)->tuple; compute(candidate_ids, comparisons)->dict[str,float]
class BradleyTerryResult: strengths, log_strengths, std_errors, elo_scale: dict[str,float]; log_likelihood: float; iterations: int; converged: bool
class BradleyTerryModel: __init__(max_iter=2000, tol=1e-10, prior_games=1.0); fit(candidate_ids, comparisons)->BradleyTerryResult; predict_proba(result, i, j)->float
class FeatureWeight: name, coefficient, std_coefficient, importance∈[0,1], permutation_importance, ci_low, ci_high, direction: Literal["prefer_higher","prefer_lower","neutral"]
class AestheticPreferenceVector: features: list[FeatureWeight], intercept, method: Literal["ridge_bt","pairwise_logistic"], regularization, fit_accuracy, r2: float|None, n_comparisons, n_candidates, created_at
class PreferenceAnalyzer: __init__(regularization: float|None=None, n_bootstrap=200, seed=0, method="pairwise_logistic"); analyze(candidates, comparisons, feature_names=None)->AestheticPreferenceVector
def rank_candidates(candidates, comparisons) -> list[RatingEntry]
def suggest_next_pair(candidates, comparisons, seed=0) -> tuple[str, str] | None   # 情報量最大（p≈0.5×不確実性）
# model_optimizer.py
class ParameterDim: name, low, high, default, group: str, description=""
class ParameterSpace: dims; names(); bounds()->ndarray; to_unit(x)->ndarray; from_unit(u)->dict; sample_lhs(n, seed)->list[dict]; default_params()->dict
    @classmethod for_style_lab(architecture, lora_names=(), grouping: Literal["grouped","per_block"]="grouped", include_style_params=True)
STYLE_PARAMS = ("style.line_taper", "style.black_fill_ratio", "style.shadow_contrast", "style.screentone_density", "style.perspective_distortion")
class RBFKernel; class GaussianProcess: fit(X,y); predict(Xs, return_std=True); log_marginal_likelihood(); optimize_hyperparameters(n_restarts=8, seed=0)
def expected_improvement(mu, sigma, best, xi=0.01); def upper_confidence_bound(mu, sigma, kappa=2.0); def probability_of_improvement(mu, sigma, best, xi=0.01)
class CMAES: __init__(mean, sigma=0.3, popsize=None, seed=0, lower=0.0, upper=1.0); ask()->ndarray; tell(solutions, values)（values は最大化）; best_x; best_value; should_stop()
class GeneticOptimizer: __init__(dim, pop_size=24, alpha=0.3, mutation_rate=0.15, mutation_sigma=0.1, elite=2, seed=0); ask(n)->ndarray; tell(X, y)
class Observation: params: dict[str,float]; score: float; candidate_id: str|None=None
class OptimizationSuggestion: params, predicted_mean, predicted_std, acquisition, strategy, rationale, rank
class BayesianStyleOptimizer: __init__(space, seed=0, acquisition: Literal["ei","ucb","pi"]="ei", xi=0.01, kappa=2.0)
    add_observation(params, score, candidate_id=None); observations; suggest(n=3, strategy: Literal["gp","cmaes","genetic","hybrid"]="hybrid")->list[OptimizationSuggestion]
    best()->Observation|None; posterior(params)->tuple[float,float]; sensitivity()->dict[str,float]
class RecipeExporter: params_to_recipe(params, *, architecture, model_a, model_b, loras=(), name=None, model_c=None)->StyleRecipe
    to_merge_config(recipe)->dict; to_merge_script(recipe)->str（ローカル実行用Python。compile可能）; to_comfy_workflow(recipe, positive, negative, seed, width, height)->dict
# dataset_curator.py
class TagCategory(str, Enum): STYLE, COMPOSITION, CAMERA_ANGLE, POSE, EXPRESSION, LIGHTING, BACKGROUND, SUBJECT, QUALITY, META, OTHER
CONTROLLABLE_CATEGORIES = (COMPOSITION, CAMERA_ANGLE, POSE, EXPRESSION, LIGHTING, BACKGROUND)
class CaptionRecord: image_id, caption, path=""
class CleansedCaption: image_id, trigger_token, caption, controllable: dict[str, list[str]], subject_tags, removed_style_tags, removed_noise_tags, missing_categories
class TagTaxonomy: classify(tag)->TagCategory; normalize(tag)->str; add(tag, category)
class Entanglement: tag_a, category_a, tag_b, category_b, support, npmi, conditional_prob, lift, severity: Literal["low","medium","high"]
class TagDeficit: category, tag, current_count, current_share, target_share, required_additional
class ShotRequirement: conditions: dict[str,str], count, reason, priority
class CooccurrenceMatrix: tags: list[str], counts: list[list[int]], npmi: list[list[float]]
class DatasetGapReport: total_images, tag_counts, category_distributions, category_entropy, normalized_entropy, cooccurrence, entanglements, deficits, recommended_shots, total_additional_images, balance_score∈[0,100], warnings
class DatasetCurator: __init__(trigger_token="caf_style", taxonomy=None, llm_client=None, target_min_share=0.12, min_count=4)
    parse_caption_lines(text)->list[CaptionRecord]; cleanse(records)->list[CleansedCaption]; async cleanse_with_llm(records)->list[CleansedCaption]; analyze(records_or_cleansed)->DatasetGapReport
# grid_laboratory.py
class TokenEntry: token, category, weight=1.0, description="", tags=[]
class TokenDictionary: DEFAULT_CATEGORIES(line_art, tone_design, perspective, composition, lighting, texture); add/remove/list/render(token, weight)->"(token:1.20)"/to_dict/from_dict
class GridAxis: name, kind: Literal["token","param","lora_weight","merge_weight","cfg","steps","seed","sampler"], target: str, values: list
class GridExperiment: id, name, base_prompt, negative_prompt, x_axis, y_axis=None, z_axis=None, base_params, seed, status, created_at
class GridCell: id, experiment_id, x_index, y_index, z_index, prompt, negative_prompt, params, seed, sampler_overrides, image_ref, score, notes
class GridAnalysis: main_effects: dict[str, list[dict]], interaction_xy: list[list[float|None]]|None, best_cell_id, anova: dict[str,float]
class GridLaboratory: __init__(repo=None, dictionary=None); create_experiment(...)->GridExperiment; generate_cells(exp)->list[GridCell]
    record_result(cell_id, score, image_ref=None, notes=""); analyze(exp_id)->GridAnalysis; to_comfy_batch(exp_id, recipe)->list[dict]
# repository.py
class StyleLabRepository: __init__(db); candidates / comparisons / observations / experiments / cells / tokens / recipes / dataset_reports のCRUD
```

---

## 8. Phase 4 — `core/manga_pipeline/`：構造駆動型ネーム＆ComfyUI生成パイプライン

### 8.1 要件
| ID | 要件 |
|---|---|
| P4-01 | 見開き構成（右ページ開始・左ページ終端の「めくり」フック）、ページ情報密度・白黒バランス。 |
| P4-02 | 右上→左下のZ字視線誘導（再帰XYカット）、矩形・斜め割り・変形ゴマ・断ち切り・ぶち抜き対応のポリゴン座標、視線移動ベクトル。 |
| P4-03 | カメラワーク（ECU/CU/MS/FS/ELS、俯瞰/アオリ/アイレベル/ダッチ、18〜135mm）、構図ルール、演技・表情・視線方向。 |
| P4-04 | フキダシ（通常・叫び・モノローグ・フラッシュ）の座標・形状、縦書きセリフの文節改行＋禁則処理、オノマトペ配置ベクトル。 |
| P4-05 | 構造（感情曲線・情報開示曲線・ビート）→ページ割り→コマ割り→カメラ・セリフ自動展開。 |
| P4-06 | **Manga Grammar Validator**: 同一ショット/アングル3連続禁止、ピーク（Tension最大・反転ビート）の大ゴマ/見開き/めくり直後配置の強制、フキダシ文字数過多、視線ベクトル逆行・交差、ページ内情報密度の緩急適正化。違反スコアと修正指示 → 自動再構成ループ。 |
| P4-07 | ComfyUI動的コンパイラ: コマ演出＋最適レシピ→APIノードグラフ。複数キャラの領域分割（マスク座標計算）、ControlNet（OpenPose/Lineart/Depth）、シード管理、非同期キュー投入＆WebSocket進捗監視（モックサーバー付き）。 |
| P4-08 | SVGネーム/原稿レンダラー: コマ枠（太さ・断ち切り）、画像レイヤー、ベクターフキダシ、縦書きテキスト、視線誘導矢印（切替）＋レイヤー構造JSON。 |

### 8.2 契約
```python
# schemas.py（座標系: ページ幅1000×高さ1414の仕上がり座標、断ち切り幅30）
class PageGeometry: width=1000.0, height=1414.0, bleed=30.0, margin_top=90.0, margin_bottom=110.0, margin_outer=70.0, margin_inner=90.0, gutter_x=16.0, gutter_y=30.0
    frame_rect(side)->tuple[x0,y0,x1,y1]（右ページはノド=左辺、左ページはノド=右辺）; bleed_rect()->tuple
class Point: x, y; class Polygon: points(min3); area(), centroid(), bbox(), contains(p), edges()
class ShotSize(str, Enum): ECU, CU, MS, FS, ELS
class CameraAngle(str, Enum): EYE_LEVEL="eye_level"; HIGH="high"(俯瞰); LOW="low"(アオリ); DUTCH="dutch"
class CompositionRule(str, Enum): rule_of_thirds, centered, diagonal, golden_spiral, symmetry, frame_in_frame, leading_lines, negative_space
class CameraWork: shot, angle, lens_mm∈[18,135], composition, dutch_degrees∈[-45,45]=0, notes=""; horizontal_fov_deg (property)
class CharacterProfile: id, role, name_placeholder, visual_description, seed: int|None=None
class CharacterAction: character_id, position: Point(コマ内正規化[0,1]), scale∈[0.05,1], pose, expression, gaze: Point(単位ベクトル), action
class BalloonKind(str, Enum): NORMAL="normal"; SHOUT="shout"; MONOLOGUE="monologue"; FLASH="flash"
class Balloon: id, kind, speaker_id, text, columns: list[str](右→左の縦書き列), center: Point, width, height, tail_target: Point|None, reading_order
class SoundEffect: id, text, position: Point, direction: Point, scale, rotation_deg, style: Literal["impact","motion","ambient","emotion"]
class PanelShape(str, Enum): RECT, SLANTED, IRREGULAR, BORDERLESS
class Panel: id, page_index, index_in_page, reading_order, polygon, shape, bleed_sides: list[Literal["top","bottom","left","right"]], breakout: bool,
             camera, characters, balloons, sfx, description, beat_id, t, tension, emphasis: Literal["normal","large","splash","spread"], is_silent, prompt_hints
class PageSide(str, Enum): RIGHT="right"; LEFT="left"
class PageTurnRole(str, Enum): NONE="none"; HOOK="hook"(めくり前の左ページ終端の引き); REVEAL="reveal"(めくり直後の右ページ)
class Page: index, number, side, panels, spread_id, page_turn_role, information_density, black_ratio_target, tension_mean, notes
class Spread: id, right_page, left_page, kind: Literal["facing","double_page_panel"]
class GazeTransition: from_panel, to_panel, start, end, vector, is_backtrack, crosses: list[int]
class GrammarViolation: rule_id, severity: Literal["error","warning","info"], page_index, panel_ids, message, fix_instruction, penalty
class GrammarReport: score∈[0,100], passed, violations, metrics: dict[str,float], iteration
class Storyboard: id, title, structure_id, theme, geometry, first_page_side, pages, spreads, characters, seed, grammar: GrammarReport|None, revision_log, created_at
    panels_in_reading_order(), panel(id), page_count (property)
# storyboard_generator.py
class StoryboardConfig: target_pages=16 [4..64], min_panels_per_page=2, max_panels_per_page=7, first_page_side=LEFT, max_chars_per_balloon=40,
                        max_chars_per_column=7, max_balloons_per_panel=3, grammar_threshold=80.0, max_fix_iterations=6, seed=0, use_llm_dialogue=False
class StoryboardGenerator: __init__(config=None, llm_client=None); generate(structure, theme=None, characters=None)->Storyboard
    generate_validated_sync(structure, theme=None, characters=None)->Storyboard; async generate_validated(...)->Storyboard
class MangaGrammarValidator: __init__(config=None); validate(storyboard, structure=None)->GrammarReport
class StoryboardFixer: apply(storyboard, report, structure=None)->Storyboard
def compute_reading_order(panels: list[Panel]) -> list[str]; def compute_gaze_transitions(page: Page) -> list[GazeTransition]
def break_japanese_text(text: str, max_chars_per_column: int = 7) -> list[str]   # 文節改行＋禁則
# comfy_compiler.py
class ControlNetKind(str, Enum): OPENPOSE="openpose"; LINEART="lineart"; DEPTH="depth"
class ControlGuide: kind, image_name, strength=0.8, start_percent=0.0, end_percent=0.85, model_name: str|None=None
class CompilerConfig: base_resolution=1024, controlnet_models: dict[str,str], negative_prompt, filename_prefix="caf", seed_base=0, region_feather=32
class RegionMask: character_id, x, y, width, height（ピクセル、8の倍数）, strength
class CompiledPanelWorkflow: panel_id, page_index, prompt: dict, seed, width, height, regions: list[RegionMask], positive_text, negative_text
class SeedManager: derive(storyboard_seed, panel_id)->int; lock(panel_id, seed); unlock(panel_id); history
class ComfyWorkflowCompiler: __init__(recipe: StyleRecipe, config=None, seed_manager=None); panel_resolution(panel)->tuple[int,int]
    compute_regions(panel, width, height)->list[RegionMask]; compile_panel(storyboard, panel, controls=None)->CompiledPanelWorkflow; compile_storyboard(storyboard, controls_by_panel=None)->list[CompiledPanelWorkflow]
# comfy_client.py
class ComfyProgressEvent: type, prompt_id, node, value, max, progress, raw
class ComfyImageRef: filename, subfolder, type
class ComfyExecutionResult: prompt_id, status: Literal["success","error","interrupted","timeout"], outputs: dict[str, list[ComfyImageRef]], events, error, duration_s
class ComfyClient: __init__(base_url="http://127.0.0.1:8188", client_id=None, timeout_s=600.0, transport=None)
    async queue_prompt(prompt)->str; get_history(prompt_id)->dict; get_object_info(node_class=None)->dict; view_image(filename, subfolder="", type="output")->bytes
    interrupt(); system_stats()->dict; wait_for_completion(prompt_id, on_progress=None, timeout_s=None)->ComfyExecutionResult; run_workflow(prompt, on_progress=None)->ComfyExecutionResult
# comfy_mock_server.py
def create_mock_comfy_app(step_delay_s: float = 0.0) -> FastAPI   # /prompt /history/{id} /view /object_info /system_stats /interrupt /ws
class MockComfyServer: コンテキストマネージャ（uvicorn をスレッド起動、空きポート）; base_url
# layout_renderer.py
class RenderOptions: show_gaze_vectors=False, show_panel_labels=True, show_bleed_guides=True, frame_stroke=4.0, draft_sketch=True, image_hrefs: dict[str,str]={}, font_family
class SVGLayoutRenderer: __init__(options=None); render_page(storyboard, page_index)->str; render_spread(storyboard, right_index, left_index)->str; render_all(storyboard)->list[str]; layers(storyboard)->dict
```

---

## 9. Phase 5 — `core/asset_publisher/`：R&D知見の自動体系化・マルチチャネル資産変換器

### 9.1 要件
| ID | 要件 |
|---|---|
| P5-01 | 1つの構造発見/モデル検証ログ → ①クリエイター向け「物語構造・漫画演出理論」②AI開発者向け「モデル調合・特徴量制御・データセット設計論」③ビジネス層向け「企画・LP・マーケティング構造論」の3軸企画へ同時展開。 |
| P5-02 | 5段階長文執筆: `Blueprint`（常識の提示→構造的パラドックス→独自仮説と実験データ/図解→普遍的フレームワーク化→実践アクション）→ `Deep Section Drafting`（章ごと高密度執筆＋Mermaid図解自動挿入）→ `Compliance & Style Sanitizer`（一人称「私」統一・URL完全除去）→ `Critique Scoring`（100点満点）→ `Iterative Polish`（基準未満章の再推敲）。 |
| P5-03 | ナレッジグラフ（PageRank・ラベル伝播コミュニティ）によるコンテンツ空白検出と次テーマ提案。 |
| P5-04 | 複数記事の教本/マガジン編纂、note用Markdown / Obsidian用Frontmatter付きMarkdown出力。 |

### 9.2 契約
```python
# schemas.py
class ContentAxis(str, Enum): CREATOR_THEORY="creator_theory"; AI_ENGINEERING="ai_engineering"; BUSINESS_MARKETING="business_marketing"
class Finding: claim, evidence, metrics: dict[str,float]={}, confidence∈[0,1]=0.7
class ResearchLog: id, kind: Literal["structure_discovery","model_experiment","dataset_analysis","manga_production","mixed"], title, summary, findings(min1), data: dict={}, tags=[], created_at
class OutlineSection: heading, purpose, key_points: list[str], figure_idea=""
class ContentPlan: axis, title, subtitle, target_reader, reader_pain, core_insight, angle, outline(min3), keywords, hook_line, call_to_action, estimated_chars
class ContentMatrix: id, source_id, source_title, shared_core, plans(3軸各1), cross_links, created_at
class BlueprintRole(str, Enum): CONVENTIONAL_WISDOM, STRUCTURAL_PARADOX, HYPOTHESIS_EVIDENCE, UNIVERSAL_FRAMEWORK, PRACTICAL_ACTION
class FigureSpec: kind: Literal["mermaid","table","none"], caption, mermaid_code=""
class BlueprintSection: role, heading, intent, key_claims(min1), evidence, figure
class ArticleBlueprint: title, thesis, target_reader, axis, sections(5, 正準順)
class DraftSection: role, heading, body_markdown, char_count
class ArticleDraft: title, lead, sections(5), closing; to_markdown()->str
class Replacement: original, replacement, index, context
class RemovedURL: kind: Literal["bare_url","markdown_link","markdown_image","reference_definition","reference_link","html_anchor","html_media","autolink","mermaid_click","defanged_url","domain_like"], original, kept_text
class SanitizationReport: first_person_replacements, urls_removed, first_person_unified: bool, url_free: bool, residual_first_person, residual_urls
class SectionScore: role, heading, logic_coherence∈[0,25], abstraction_balance∈[0,25], persuasiveness∈[0,25], evidence_density∈[0,25], total∈[0,100], feedback
class CritiqueScore: sections, overall, passed, threshold
class PipelineStageRecord: stage, iteration, duration_ms, score: float|None, notes
class PublishedArticle: id, title, axis, source_id, markdown, blueprint, draft, sanitization, critique_history, final_score, iterations, stages, tags, created_at
# sanitizer.py
def unify_first_person(text) -> tuple[str, list[Replacement]]; def detect_first_person_variants(text) -> list[str]
def remove_urls(text) -> tuple[str, list[RemovedURL]]; def find_urls(text) -> list[str]
def sanitize_text(text) -> tuple[str, SanitizationReport]
class ComplianceSanitizer: sanitize(text)->tuple[str, SanitizationReport]; sanitize_draft(draft)->tuple[ArticleDraft, SanitizationReport]; verify(text)->SanitizationReport
# matrix_transpiler.py
def research_log_from_structure(structure) -> ResearchLog; def research_log_from_experiment(payload: dict) -> ResearchLog
class ContentMatrixTranspiler: __init__(llm_client=None, harness_config=None); expand_deterministic(log)->ContentMatrix; async expand(log)->ContentMatrix
# writer_pipeline.py
class WriterConfig: quality_threshold=80.0, max_polish_rounds=3, min_section_chars=600, target_section_chars=1200, include_mermaid=True
class ArticleWriterPipeline: __init__(llm_client=None, config=None, harness_config=None)
    async run(plan, log=None, on_progress=None)->PublishedArticle   # on_progress(stage: str, progress: float, message: str)
    build_blueprint / draft_sections / sanitize / critique / polish（各段を個別呼び出し可能）
# knowledge_graph.py
class ContentGap: concepts, axis: ContentAxis|None, gap_score, demand, coverage, rationale, suggested_title, suggested_outline
class KnowledgeGraph: add_article(a); add_structure(s); add_research_log(l); concepts(); pagerank(); communities(); detect_gaps(top_k=5)->list[ContentGap]; to_mermaid(); stats()
class ArticleRepository: __init__(db); save/get/list/delete（PublishedArticle）; save_matrix/get_matrix/list_matrices
# exporter.py
class SeriesChapter: index, title, article_ids, summary; class SeriesVolume: id, title, description, chapters, glossary, created_at
class SeriesCompiler: compile(articles, title, description=None)->tuple[SeriesVolume, str]
def export_note_markdown(article) -> str; def export_obsidian_markdown(article, related_titles=()) -> str
def export_series_markdown(volume, articles, fmt: Literal["note","obsidian"]="obsidian") -> str
```

---

## 10. Phase 6 — 初期プリセット「普遍的構造データベース」（`core/structure_engine/seeds/`）

| ID | 要件 |
|---|---|
| P6-01 | `narrative_archetypes.json`（20）: 古今東西の神話・古典劇・映画・漫画に通底する対立構造・情報非対称性・逆転劇の純粋骨格。各21点の緻密な感情・情報波形、力学グラフ、漫画転用ガイド付き。 |
| P6-02 | `manga_panel_rhythm_patterns.json`（20）: 第1話冒頭3ページの掴み、バトル/心理戦の駆け引き、静寂→爆発の緩急、見開きめくりフック等。`rhythm_profile` 必須。 |
| P6-03 | `persuasion_and_marketing_structures.json`（20）: 固定観念を覆す論説、PASONA/ストーリーブランド進化型LP、技術知見の体系化フレームワーク等。 |
| P6-04 | 全60件が `UniversalStructure` に完全準拠（ID: `na_XX_*` / `mr_XX_*` / `pm_XX_*`）。固有名詞を層内に含めない（系譜は `metadata.lineage` のみ）。 |
| P6-05 | SQLite初期シードローダー（`seed_database`）とJSON再生成用の著作スクリプト（`seeds/authoring/`）。 |

JSONファイル形式: `{"schema_version": "1.0", "category": "<StructureCategory>", "description": "...", "patterns": [<UniversalStructure>, ... 20件]}`

各パターンの品質基準: ノード4〜8、エッジ5〜12（7種のエッジ種別を適切に使い分け）、ビート6〜10（転換点を2以上、因果チェーンを明示）、メカニズム3〜5、スロット6〜10、不変条件3〜6、転用ガイド3領域すべて記述、21点波形は6〜12個のキーフレームから設計し各パターンで固有の動態を持つ。

---

## 11. Phase 7 — `web_studio/`：統合ローカルWebスタジオ

### 11.1 バックエンド（FastAPI）
- `web_studio/backend/app.py`: `create_app(settings: FoundrySettings | None = None, *, services: FoundryServices | None = None) -> FastAPI`
- `web_studio/backend/services.py`: `FoundryServices`（settings, db, llm_client, harness_config, 各リポジトリ・エンジン、ジョブマネージャ、WebSocketハブ）。設定更新時にLLMクライアントを再構築。
- ジョブ: 重い処理は `{"async": true}` でジョブID返却、`/ws/progress` に進捗イベント `{"type":"job","job_id","kind","status","progress","stage","message"}` を配信。既定は同期実行で結果を直接返す。
- `web_studio/frontend/dist` が存在すれば `/` で SPA として配信。
- 開発時CORS: `http://localhost:5173`, `http://127.0.0.1:5173`。

| メソッド | パス | 概要 |
|---|---|---|
| GET | `/api/health` | 稼働状態・LLMプロバイダ・フォールバック状態・シード件数 |
| GET/PUT | `/api/system/config` | LLM（URL・モデル名・思考段階設定）/ ComfyUI 設定の取得・更新（永続化） |
| POST | `/api/system/llm/test` | LLM接続テスト |
| POST | `/api/system/comfy/test` | ComfyUI接続テスト |
| GET | `/api/system/db/stats` | DBファイルサイズ・テーブル別件数 |
| POST/GET | `/api/system/db/backup` / `/api/system/db/backups` | バックアップ作成・一覧 |
| GET | `/api/system/telemetry` | LLM呼び出し履歴・統計・ハーネス実行履歴 |
| GET | `/api/jobs` / `/api/jobs/{id}` | ジョブ一覧・状態 |
| WS | `/ws/progress` | 進捗配信（ping/pong対応） |
| GET/POST | `/api/structures` | 一覧（category/source_kind/q/limit/offset）・保存 |
| GET/DELETE | `/api/structures/{id}` | 取得・削除 |
| PUT | `/api/structures/{id}/wave` | 波形編集の保存 |
| POST | `/api/structures/extract` | テキスト→5階層構造抽出 |
| POST | `/api/structures/merge` | 構造合成（weight_topology / weight_wave） |
| POST | `/api/structures/mutate` | 構造変異 |
| GET | `/api/structures/mutations` | 変異種別一覧 |
| POST | `/api/structures/{id}/transpile` | 漫画/LP/記事への変換 |
| POST | `/api/structures/search` | 波形類似度検索（DTW/コサイン/ハイブリッド） |
| GET | `/api/structures/{id}/mermaid` / `/api/structures/{id}/wave.svg` | Mermaid記法・SVG波形 |
| POST/GET | `/api/seeds/load` / `/api/seeds/summary` | シード投入・集計 |
| GET/POST | `/api/style/candidates` (+ `/bootstrap`, `/{id}`, `/{id}/preview.svg`) | 候補管理・LHS初期候補生成・手続き的プレビュー |
| GET | `/api/style/arena/next` | 次のA/B対戦ペア（能動学習） |
| GET/POST | `/api/style/comparisons` | 判定履歴・判定登録 |
| GET | `/api/style/rankings` | Elo / Bradley-Terry ランキング |
| GET | `/api/style/preference-vector` | 美的選好ベクトル |
| GET | `/api/style/optimizer/space` | パラメータ空間 |
| POST | `/api/style/optimizer/suggest` | ベイズ最適化による次レシピ提案 |
| POST | `/api/style/recipes/export` | マージ設定・スクリプト・ComfyUIワークフロー出力 |
| POST | `/api/style/dataset/analyze` | キャプションクレンジング＋共起バイアス・ギャップ分析 |
| GET/POST/DELETE | `/api/style/tokens` | 構造化トークン辞書 |
| GET/POST | `/api/style/grid` (+ `/{id}`, `/{id}/results`) | グリッド実験 |
| GET/POST | `/api/manga/storyboards` | ネーム一覧・生成（文法検証ループ込み） |
| GET/DELETE | `/api/manga/storyboards/{id}` | 取得・削除 |
| POST | `/api/manga/storyboards/{id}/validate` / `/autofix` | 漫画文法検証・自動修正 |
| GET | `/api/manga/storyboards/{id}/pages/{page_index}/svg` | SVGネーム（`gaze`, `labels` クエリ） |
| GET | `/api/manga/storyboards/{id}/layers` | レイヤー構造JSON |
| POST | `/api/manga/storyboards/{id}/comfy` | ComfyUIワークフローJSONコンパイル |
| POST | `/api/manga/storyboards/{id}/comfy/queue` | ComfyUIへ非同期投入（ジョブ） |
| POST | `/api/publisher/matrix` | 3軸企画展開 |
| GET/POST | `/api/publisher/articles` | 記事一覧・5段階執筆実行 |
| GET/DELETE | `/api/publisher/articles/{id}` | 取得・削除 |
| GET | `/api/publisher/articles/{id}/export` | `format=note|obsidian` |
| POST | `/api/publisher/sanitize` / `/api/publisher/verify` | サニタイズ実行・検証 |
| GET | `/api/publisher/gaps` | コンテンツ空白・次テーマ提案 |
| POST | `/api/publisher/series` | シリーズ編纂 |

### 11.2 フロントエンド（React 18 + TypeScript + Vite + Tailwind CSS、ダークモード基調）
| Workspace | 機能 |
|---|---|
| 1 Structure Lab | テキスト抽出、60種プリセットブラウザー、感情・情報開示曲線のインタラクティブSVG編集（ドラッグ）、Merge/Mutateスライダー、漫画/LP/記事ワンクリック変換、類似検索 |
| 2 Style Alchemy Arena | ←/→キーによるブラインドA/B判定（↓/Spaceで引分）、Eloランキング表、選好ベクトル棒グラフ、ベイズ最適化提案レシピ、キャプション共起バイアス分析、グリッド実験 |
| 3 Manga Studio | 感情曲線同期のページ/コマ・タイムライン、SVGネームプレビュー（視線誘導トグル）、漫画文法スコア＆改善アラート、ComfyUIワークフローJSON生成・ダウンロード |
| 4 Asset Publisher | 3軸企画展開、5段階執筆（進捗表示）、一人称「私」統一＆URL完全除去の検証バッジ、Markdownプレビュー＆エクスポート、空白テーマ提案 |
| 5 System & Local AI Config | DeepSeekエンドポイント・モデル名・思考段階設定、ComfyUIエンドポイント、DB統計＆バックアップ、テレメトリ |

---

## 12. Phase 8 — 統合検証・ドキュメント

| ID | 要件 |
|---|---|
| P8-01 | `tests/test_e2e_foundry.py`: シード60件ロード → テキスト抽出 → シードとMerge → ネーム生成＋文法バリデータ通過 → SVG＆ComfyUI JSON出力（モックComfyUIでキュー実行）→ Elo/BT選好学習＋ベイズ最適化で次レシピ算出 → 研究データから解説記事（一人称「私」・URL除去済み）生成、を一気通貫で自動実行。 |
| P8-02 | `tests/test_backend_api.py`: FastAPI `TestClient` による全エンドポイント・WebSocketテスト。 |
| P8-03 | `web_studio/frontend` で `npm install` → `npx tsc --noEmit` → `npm run build` がエラー0件。 |
| P8-04 | `python run_studio.py` ワンコマンド起動（DB初期化・シード投入・必要ならフロントエンドビルド・バックエンド起動・ブラウザオープン、Windows/macOS/Linux対応）。 |
| P8-05 | `docs/MASTER_GUIDE.md`: 全アルゴリズムの数理解説、DeepSeek-V4.1-Flash / ComfyUI 接続手順、研究・制作・資産化の運用ワークフロー。`docs/ARCHITECTURE.md`: 構成・データフロー。 |
| P8-06 | 全テスト・ビルド完全パス後、`PROGRESS.md` 全項目 `[x]` で最終コミット。 |

---

## 13. 品質ゲート
1. `python -m pytest` 全件パス（外部ネットワーク不要）。
2. `tests/test_quality_gates.py`: 禁止パターン検査（R-01）・全モジュールimport可能性・シード60件検証。
3. `npx tsc --noEmit` / `npm run build` エラー0件。
4. 記事出力は `find_urls()==[]` かつ `detect_first_person_variants()==[]`。
5. 全ネーム出力は `GrammarReport.passed` かつ `score >= grammar_threshold`。
