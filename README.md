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

## テスト

```bash
python -m pytest
```
