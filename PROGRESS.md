# PROGRESS — Creative Asset Foundry 構築進捗管理表

凡例: `[x]` 完了 / `[ ]` 未完了。各Phase完了ごとに更新してコミットする。

## Phase 0：仕様書・進捗管理表・基盤
- [x] ディレクトリ構成の作成（core / web_studio / tests / docs）
- [x] `SPEC.md`（全要件の構造化＋モジュール間インターフェース契約）
- [x] `PROGRESS.md`（本ファイル）
- [x] `pyproject.toml` / `.gitignore` / `README.md`
- [x] 共通基盤 `core/common.py` / `core/config.py` / `core/db.py`
- [x] `tests/conftest.py` / `tests/test_quality_gates.py`（禁止パターン機械検査）
- [x] git 初期化・Phase 0 コミット

## Phase 1：ローカルDeepSeek-V4.1-Flash 多段階推論基盤（`core/llm_client/`）
- [x] 型定義（ChatMessage / LLMRequest / LLMResponse / TokenUsage / StructuredResult）と例外階層
- [x] JSON抽出・修復（`<think>`除去、フェンス抽出、末尾カンマ・コメント・単引用・未引用キー・途中切断の修復）
- [x] OpenAI互換アダプタ（`http://localhost:8000/v1`、json_schema/json_object、reasoning_content、再試行）
- [x] Ollamaアダプタ（`http://localhost:11434/v1`、`/api/tags` ヘルスチェック）
- [x] AutoFallbackアダプタ（実機→モック自動切替）
- [x] 高精度モックLLMアダプタ（タスク別ハンドラ登録・JSON Schema合成器・障害注入・決定論性）
- [x] テレメトリ（トークン・レイテンシ・思考ログのSQLite記録、統計）
- [x] LLMClient（構造化生成＋自己修復リトライ）
- [x] 4段階ハーネス（Decompose → Synthesize → Critique → Refine、post_validate追加精錬）
- [x] DeepSeek向け長文システムプロンプト群
- [x] ファクトリ／DI（mock / openai_compatible / ollama / auto、環境変数上書き）
- [x] `tests/test_llm_client.py` 全件パス

## Phase 2：普遍的構造抽出＆異分野転用エンジン（`core/structure_engine/`）
- [x] 5階層オントロジー `schemas.py`（L1 力学グラフ / L2 感情・情報波形 / L3 ビート / L4 認知メカニズム / L5 ブループリント）
- [x] 脱フレーバー抽出器 `extractor.py`（固有名詞・装飾除去、ヒューリスティック逆コンパイル、漏洩再走査）
- [x] 構造主義物語論に基づくDeepSeek用システムプロンプト群 `prompts.py`
- [x] 構造整合性監査 `consistency.py`
- [x] 構造代数 `composer.py`（merge_structures：ノード整列＋DTW整列補間、mutate_structure：7種）
- [x] 異分野トランスパイラ `transpiler.py`（漫画プロット / LP・ファネル / 解説記事、忠実度スコア）
- [x] 類似度 `similarity.py`（コサイン・多変量DTW・グラフ類似）
- [x] リポジトリ `repository.py`（SQLite保存・検索・類似検索）
- [x] 可視化 `visualizer.py`（Mermaid・SVG波形・Markdownレポート）
- [x] モックハンドラ `mock_handlers.py`
- [x] `tests/test_structure_engine.py` 全件パス

## Phase 3：極限絵柄開発・モデル調合＆LoRA研究ラボ（`core/style_alchemy/`）
- [x] 共有スキーマ `schemas.py`（アーキテクチャ別ブロック定義・StyleRecipe）
- [x] ComfyUIグラフ基盤 `comfy_export.py`（モデルスタック・txt2img）
- [x] 選好学習 `preference_engine.py`（Elo・Bradley-Terry MM・リッジ/ロジスティック・重要度・CI・能動学習ペア）
- [x] 最適化 `model_optimizer.py`（パラメータ空間・GP・EI/UCB/PI・CMA-ES・GA・提案・エクスポーター）
- [x] データセット精製 `dataset_curator.py`（クレンジング・共起/NPMI/エントロピー・癒着検出・ギャップ分析・LLM連携）
- [x] グリッド実験 `grid_laboratory.py`（トークン辞書・XYZバッチ・分析）
- [x] リポジトリ `repository.py` / モックハンドラ `mock_handlers.py`
- [x] `tests/test_style_alchemy.py` 全件パス

## Phase 4：構造駆動型・漫画ネーム＆ComfyUI生成パイプライン（`core/manga_pipeline/`）
- [ ] 漫画演出スキーマ `schemas.py`（ページ幾何・コマポリゴン・カメラ・フキダシ・オノマトペ・見開き）
- [ ] 幾何・視線誘導（再帰XYカット読み順、視線ベクトル、逆行・交差判定）
- [ ] 縦書き文節改行＋禁則処理
- [ ] 構造駆動ネーム生成 `storyboard_generator.py`（ページ割り→コマ割り→カメラ→セリフ）
- [ ] Manga Grammar Validator＋自動修正ループ
- [ ] ComfyUIコンパイラ `comfy_compiler.py`（領域マスク・ControlNet・シード管理）
- [ ] ComfyUIクライアント `comfy_client.py`（非同期キュー・WebSocket進捗）＋ `comfy_mock_server.py`
- [ ] SVGレンダラー `layout_renderer.py`（断ち切り・フキダシ・縦書き・視線矢印・レイヤーJSON）
- [ ] `tests/test_manga_pipeline.py` 全件パス

## Phase 5：R&D知見の自動体系化・マルチチャネル資産変換器（`core/asset_publisher/`）
- [x] スキーマ `schemas.py`
- [x] サニタイザー `sanitizer.py`（一人称「私」統一・URL完全除去・検証）
- [x] 3軸マトリクス `matrix_transpiler.py`
- [x] 5段階執筆 `writer_pipeline.py`（Blueprint → Drafting(Mermaid) → Sanitize → Critique → Polish）
- [x] ナレッジグラフ `knowledge_graph.py`（PageRank・コミュニティ・空白検出）
- [x] 編纂・出力 `exporter.py`（note / Obsidian / シリーズ）
- [x] リポジトリ・モックハンドラ
- [x] `tests/test_asset_publisher.py` 全件パス

## Phase 6：初期プリセット「普遍的構造データベース」（60パターン）
- [ ] `narrative_archetypes.json`（20）
- [ ] `manga_panel_rhythm_patterns.json`（20）
- [ ] `persuasion_and_marketing_structures.json`（20）
- [ ] 著作スクリプト `seeds/authoring/`
- [x] SQLite初期シードローダー `seeds/loader.py`
- [ ] `tests/test_seeds.py` 全件パス

## Phase 7：統合ローカルWebスタジオUI（`web_studio/`）
- [ ] FastAPI アプリファクトリ・サービスコンテナ・ジョブ管理・WebSocketハブ
- [ ] 全RESTエンドポイント（system / structures / seeds / style / manga / publisher / jobs）
- [ ] `tests/test_backend_api.py` 全件パス
- [ ] フロントエンド基盤（Vite + React 18 + TypeScript + Tailwind、APIクライアント、WSフック）
- [ ] Workspace 1 Structure Lab
- [ ] Workspace 2 Style Alchemy Arena
- [ ] Workspace 3 Manga Studio
- [ ] Workspace 4 Asset Publisher
- [ ] Workspace 5 System & Local AI Config

## Phase 8：結合・E2E・ビルド検証・ドキュメント
- [ ] `tests/test_e2e_foundry.py`（一気通貫E2E）全件パス
- [ ] `pytest` 全件グリーン
- [ ] `npx tsc --noEmit` エラー0件
- [ ] `npm run build` 成功
- [ ] `run_studio.py`（ワンコマンド起動）
- [ ] `docs/MASTER_GUIDE.md`（数理解説・接続手順・運用ワークフロー）
- [ ] `docs/ARCHITECTURE.md`
- [ ] 最終コミット
