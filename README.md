# Creative Asset Foundry (CAF)

ローカルLLM（**DeepSeek-V4.1-Flash**）とローカル画像生成環境（**ComfyUI / LoRA / モデルマージ**）を統合し、

1. 普遍的構造の抽出・構造代数・異分野転用（Structure Engine）
2. 美的選好に基づく絵柄・モデル配合の数理最適化（Style Alchemy）
3. 構造駆動型のオリジナル漫画ネーム生成と ComfyUI 生成（Manga Pipeline）
4. 研究開発知見の自動コンテンツ資産化（Asset Publisher）

を1つのローカルスタジオで循環させる統合プラットフォームです。

## クイックスタート

```bash
pip install -e ".[dev]"
```

```bash
python run_studio.py
```

ブラウザで `http://127.0.0.1:8765` が開きます。ローカルの DeepSeek-V4.1-Flash（OpenAI互換 `http://localhost:8000/v1` または Ollama `http://localhost:11434/v1`）や ComfyUI（`http://127.0.0.1:8188`）が起動していない場合でも、高精度モックへ自動フォールバックしてすべての機能を試せます。

## ドキュメント

- [SPEC.md](SPEC.md) — 全要件とモジュール間インターフェース契約
- [PROGRESS.md](PROGRESS.md) — 構築進捗
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — アーキテクチャ設計書
- [docs/MASTER_GUIDE.md](docs/MASTER_GUIDE.md) — 数理解説・接続手順・運用ワークフロー

## 主な構成

| ワークスペース | 機能 |
|---|---|
| Structure Lab | テキストからの5階層構造抽出、60種のプリセット骨格、波形のドラッグ編集、構造合成（Merge）・変異（Mutate）、漫画/LP/記事への変換、DTW類似検索 |
| Style Alchemy Arena | ←/→ キーのブラインドA/B判定、Elo/Bradley-Terry ランキング、美的選好ベクトル、ベイズ最適化による次のマージ比率・LoRA重み提案、キャプション共起分析、グリッド実験 |
| Manga Studio | 構造駆動ネーム生成、漫画文法バリデータと自動修正、SVGネーム（視線誘導表示）、ComfyUI ワークフロー生成・投入 |
| Asset Publisher | 3軸企画展開、5段階執筆、一人称「私」統一・URL完全除去の検証、note/Obsidian 出力、空白テーマ提案、シリーズ編纂 |
| System & Local AI Config | DeepSeek-V4.1-Flash / ComfyUI 接続設定、思考段階設定、DB統計・バックアップ、テレメトリ |

## テストと検証

```bash
python -m pytest
```

```bash
cd web_studio/frontend && npm install && npx tsc --noEmit && npm run build
```

UI スモークテスト（バックエンドを mock で起動した状態で Playwright により5ワークスペースを巡回）:

```bash
cd web_studio/frontend && CAF_UI_URL=http://127.0.0.1:8799 node scripts/ui-smoke.mjs
```
