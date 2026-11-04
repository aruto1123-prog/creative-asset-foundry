# Creative Asset Foundry ローカル導入指示書（Claude Code 実行用）

この文書は、利用者のPC上で動く **Claude Code** に渡し、Creative Asset Foundry（CAF）の導入を「初期設定」から「実用」まで自律的に完了させるための指示書です。

## 使い方（利用者向け）

1. ローカルPCでターミナルを開き、作業用フォルダで Claude Code を起動します。
2. 次の文をそのまま貼り付けます。

```text
https://github.com/aruto1123-prog/creative-asset-foundry をクローンし、リポジトリ内の docs/LOCAL_SETUP_WITH_CLAUDE_CODE.md（main ブランチ）の指示書に従って、初期設定から実用まで完了させてください。システムへのインストールや大容量ダウンロードの前には必ず確認してください。
```

- リポジトリが非公開の場合は、事前に `gh auth login` などで GitHub の認証を済ませてください。
- ローカル LLM（DeepSeek-V4.1-Flash）や ComfyUI がまだ無くても開始できます。その場合は、モック構成で動作確認したあとに導入手順へ進みます。

---

## 0. Claude Code への指示

あなたはこのPCに CAF を導入するセットアップ担当エンジニアです。ローカルの DeepSeek-V4.1-Flash と ComfyUI に接続し、利用者が日常的に使える状態まで仕上げてください。Step 1〜10 を順に実行し、各 Step の「完了条件」を満たしてから次へ進みます。

### 厳守事項
1. **OS 判定**: 最初に OS（Windows / macOS / Linux）とシェルを判定し、以降はその OS 用のコマンドを使う。Windows は PowerShell を前提とする。
2. **事前承認**: システム全体に影響する操作は、実行前に内容を説明して利用者の承認を得る。対象は Python・Node.js・Git・Ollama・ComfyUI・GPU ドライバのインストール、PATH の変更、管理者権限や sudo の使用。
3. **大容量ダウンロード**: LLM や画像生成モデルは、ファイル名・サイズ・保存先・空き容量・ライセンスを提示し、承認を得てから取得する。配布元やファイル名・タグを推測で決めず、利用者に確認するか公式情報で確かめる。
4. **既存資産の保護**: 既存のファイル・ComfyUI・モデル・データを削除したり上書きしたりしない。
5. **外部非公開**: 起動するサーバはすべて `127.0.0.1`（localhost）にだけバインドする。
6. **コードを改変しない**: テストを通すためにテストやアプリのコードを書き換えない。失敗したら原因を調べ、環境側で解決できないものは報告する。
7. **記録**: 各 Step の結果（実行コマンド・バージョン・結果・問題と対処）を `data/setup_report.md`（Git 管理外）に追記する。
8. **質問**: 判断に迷って作業が止まる場合は、選択肢と推奨を示して利用者に質問する。

### 完了の定義（Definition of Done）
- [ ] リポジトリを取得し、使用するバージョンを確定した
- [ ] Python 仮想環境と依存を導入し、`python -m pytest` が全件パスした
- [ ] フロントエンドで `npm ci` → `npx tsc --noEmit` → `npm run build` が成功した
- [ ] モック構成でスタジオが起動し、受け入れテスト（`--allow-mock`）に合格した
- [ ] ローカル LLM に接続し、`check_llm.py` で構造化出力モードを確定して設定を保存した
- [ ] ComfyUI に接続し、`check_comfy.py` の互換性チェックに合格して実画像を1枚以上生成した
- [ ] 実機構成の受け入れテスト（`--checkpoint` 付き）に合格した
- [ ] 起動スクリプトを作成し、バックアップを確認して `data/setup_report.md` を完成させた

---

## Step 1. 事前確認

1. 環境情報を取得し、記録する。
   - **OS / CPU / メモリ**
     - macOS: `sw_vers`、`sysctl -n machdep.cpu.brand_string hw.memsize`
     - Linux: `uname -a`、`free -h`
     - Windows: `Get-ComputerInfo | Select-Object OsName,OsVersion,CsTotalPhysicalMemory`
   - **GPU**
     - NVIDIA: `nvidia-smi`
     - macOS: `system_profiler SPDisplaysDataType`
   - **ディスク空き容量**: 最低 10GB。モデルを置く場合はその分を加算する。
2. 必要なツールのバージョンを確認する。

| ツール | 要件 | 確認コマンド |
|---|---|---|
| Git | 任意の最近の版 | `git --version` |
| Python | 3.10 以上（推奨 3.11〜3.13） | `python3 --version`（Windows は `py -3 --version`） |
| Node.js / npm | Node 18 以上（推奨 22）・npm 9 以上 | `node --version`・`npm --version` |

   足りないものは、OS に応じた導入方法を提示して承認を得てから入れる。
   - Windows: `winget install Python.Python.3.12` / `winget install OpenJS.NodeJS.LTS` / `winget install Git.Git`
   - macOS: `brew install python@3.12 node git`
   - Linux: ディストリビューションのパッケージ、または公式配布物
3. 使用ポートが空いているか確認する。

| ポート | 用途 |
|---|---|
| 8765 | CAF |
| 5173 | Vite 開発サーバ（任意） |
| 8000 | OpenAI 互換 LLM |
| 11434 | Ollama |
| 8188 | ComfyUI |

   - macOS / Linux: `lsof -i :8765`
   - Windows: `Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue`

**完了条件**: 要件を満たしていること、または満たすための導入を利用者が承認したこと。

---

## Step 2. 取得とバージョン選択

```bash
git clone https://github.com/aruto1123-prog/creative-asset-foundry.git
cd creative-asset-foundry
git log --oneline -8
```

**バージョン選択の規則**
- **既定は安定版のコミット `25f7668`**。全テスト588件・型チェック・ビルド・UIスモークを検証済みで、初期シードは60件。
- main の最新コミットのメッセージが `WIP` で始まらない場合は、拡張版（シード100件・ネーム構造ドクター）が完成済みとみなす。そのうえで Step 3〜4 のテスト・ビルドがすべて通れば main を使ってよい。通らなければ安定版へ戻す。

安定版を使う場合は、作業ブランチを作り、本書と補助ツールを main から取り出す。

```bash
git switch -c local-stable 25f7668
git checkout origin/main -- local_tools docs/LOCAL_SETUP_WITH_CLAUDE_CODE.md
```

- `local_tools/` に4つの補助ツールが入る（付録A）。これらは安定版でも動作することを確認済み。
- `data/` は Git 管理外の実行データ置き場で、DB・設定・バックアップ・書き出しが入る。

**完了条件**: 使用するコミットを決め、`setup_report.md` に記録したこと。

---

## Step 3. Python 環境とテスト

以降のコマンドはリポジトリ直下で実行する。

**macOS / Linux**
```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

**Windows（PowerShell）**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

- 仮想環境は有効化せず、`.venv` 内の python を直接呼ぶ。有効化する場合、Windows で実行ポリシーにより拒否されたら、`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を提案して承認を得る。
- テストは外部ネットワークに一切依存しない（LLM と ComfyUI はモックで検証される）。
- 本書では、以降 `python` と書いたら `.venv` の python を指す。

**完了条件**: `python -m pytest` が全件パスしたこと（安定版は 588 passed）。失敗した場合は原因を調べて記録する（厳守事項6）。

---

## Step 4. フロントエンドのビルド

```bash
cd web_studio/frontend
npm ci
npx tsc --noEmit
npm run build
cd ../..
```

`npm ci` は同梱の `package-lock.json` どおりに再現インストールする。社内プロキシ等で失敗した場合は、npm のプロキシ設定を利用者に確認する。

**完了条件**: 型エラーが0件で、`web_studio/frontend/dist/index.html` が生成されたこと。

---

## Step 5. モック構成での起動と受け入れテスト

実機の LLM や ComfyUI が無くても全機能を確認できる構成で、配線を検証する。

1. スタジオを起動する（別ターミナルまたはバックグラウンド）。
   ```bash
   python run_studio.py --provider mock --comfy-mock --no-browser
   ```
   - `--provider` と `--comfy-mock` は一時的な上書きで、`data/config.json` には保存されない。
   - 初回起動時は DB を初期化し、シードを投入する。
2. 受け入れテストを実行する。
   ```bash
   python local_tools/acceptance_test.py --allow-mock
   ```
   全行が `[OK]` になり、末尾に「受け入れテスト: 合格」と出ればよい。
3. 利用者にブラウザで `http://127.0.0.1:8765` を開いてもらい、5つのワークスペースが表示されることを確認してもらう。
4. 任意: UI スモークテスト（Playwright で5ワークスペースを巡回。コンソールエラーが0件なら合格）
   ```bash
   cd web_studio/frontend
   npm install --no-save playwright
   npx playwright install chromium
   CAF_UI_URL=http://127.0.0.1:8765 node scripts/ui-smoke.mjs
   cd ../..
   ```
   Windows では `$env:CAF_UI_URL="http://127.0.0.1:8765"; node scripts/ui-smoke.mjs` とする。
5. 確認できたらスタジオを停止する（Ctrl+C）。

**完了条件**: 受け入れテスト（`--allow-mock`）に合格したこと。

---

## Step 6. ローカル LLM（DeepSeek-V4.1-Flash）の接続

### 6.1 方式の選択
利用者のハードウェアと既存環境から方式を提案し、承認を得る。

| 環境 | 推奨方式 | エンドポイント |
|---|---|---|
| NVIDIA GPU ＋ Linux（Windows なら WSL2） | vLLM または SGLang | `http://localhost:8000/v1` |
| Windows ネイティブ / macOS / VRAM が少ない | Ollama | `http://localhost:11434/v1` |
| GUI で管理したい / GGUF を使う | LM Studio または llama.cpp server | `http://localhost:8000/v1`（ポートは起動時に指定） |

**事前確認**
- 利用者が DeepSeek-V4.1-Flash をどの形式（重みの配布形式・量子化・Ollama のタグ等）で持っているか、または入手予定かを確認する。
- モデルカードで必要な VRAM / メモリ量を確かめ、搭載量と照合する。足りなければ量子化版やオフロードを提案する。
- 既にサーバが起動しているかを確認する。
  - OpenAI 互換: `curl http://localhost:8000/v1/models`
  - Ollama: `curl http://localhost:11434/api/tags`

### 6.2 サーバ起動の例
`<...>` は利用者の環境に合わせて置き換える。オプションはモデルカードと GPU に合わせて調整する。

```bash
# vLLM（Linux / WSL2）
vllm serve <MODEL_PATH_OR_ID> --host 127.0.0.1 --port 8000 --served-model-name DeepSeek-V4.1-Flash --max-model-len 32768
# SGLang（Linux / WSL2）
python -m sglang.launch_server --model-path <MODEL_PATH_OR_ID> --host 127.0.0.1 --port 8000 --served-model-name DeepSeek-V4.1-Flash
# llama.cpp server（GGUF）
llama-server -m <GGUF_PATH> --host 127.0.0.1 --port 8000 -c 32768
# Ollama（モデルタグは公式ライブラリ等で確認したものを使う）
ollama pull <MODEL_TAG>
ollama list
```

- LM Studio の場合は、GUI でモデルを読み込み、Local Server をポート 8000 で起動する。
- WSL2 で LLM を動かし、CAF を Windows 側で動かす場合でも、通常は `localhost` で到達できる。
- **コンテキスト長**: CAF のプロンプトには JSON Schema（構造抽出で約14KB）が含まれる。そのため 16K トークン以上、推奨 32K を確保する。

### 6.3 実機検証と設定の保存
`--model` には、`/v1/models`（Ollama は `ollama list`）が返す名前を**正確に**指定する。

```bash
# OpenAI 互換（vLLM / SGLang / llama.cpp / LM Studio）
python local_tools/check_llm.py --provider openai_compatible --base-url http://localhost:8000/v1 --model DeepSeek-V4.1-Flash --save
# Ollama
python local_tools/check_llm.py --provider ollama --base-url http://localhost:11434/v1 --model <MODEL_TAG> --save
```

- 3つの構造化出力モード（`json_schema` → `json_object` → `prompt_only`）を順に試す。そのうえで、4段階ハーネス（分解→生成→批評→修正）が成功し、自己修復リトライが最も少ないモードを推奨する。
- `--save` を付けると、推奨設定を `data/config.json` に保存する。
- 大型モデルで時間切れになる場合は `--timeout 1200` などに延ばす。
- 環境変数 `CAF_LLM_PROVIDER` 等が設定されていると、その値が優先される。検証前に unset する。

**完了条件**: いずれかのモードで `[OK]` になり、推奨設定が保存されたこと。全モードで NG なら、サーバのログ・モデル名・コンテキスト長・タイムアウトを調べて報告する。

---

## Step 7. ComfyUI の接続

### 7.1 用意と起動
- **既に ComfyUI がある場合**: 場所と起動方法を利用者に確認し、`127.0.0.1:8188` で起動する（例: `python main.py --listen 127.0.0.1 --port 8188`）。
- **無い場合**: ComfyUI の公式手順を確認する（デスクトップ版・ポータブル版・手動インストールのうち環境に合うもの）。PyTorch は GPU（CUDA / MPS / CPU）に合う公式の導入方法を使う。導入計画を提示し、承認を得てから導入する。

### 7.2 モデル配置
利用者が保有するモデルを確認し、ComfyUI の所定フォルダに置く。ダウンロードは厳守事項3に従う。

| 種類 | 置き場所 |
|---|---|
| チェックポイント | `models/checkpoints` |
| LoRA | `models/loras` |
| VAE | `models/vae` |
| ControlNet | `models/controlnet` |
| FLUX 等の拡散モデル | `models/diffusion_models`（旧版は `models/unet`） |
| テキストエンコーダ | `models/text_encoders`（旧版は `models/clip`） |

- **任意**: LoRA の階層ウェイト（LBW）を使う場合は、ComfyUI-Inspire-Pack を ComfyUI-Manager などから導入する。未導入でも CAF は通常の LoraLoader を使う構成で動く。

### 7.3 互換性チェックと実生成

```bash
# 1) 接続確認と、利用可能なモデル名の一覧表示
python local_tools/check_comfy.py
# 2) 実在するチェックポイント名でワークフロー互換性をチェック（未導入ノード・入力名の差異を表示）
python local_tools/check_comfy.py --checkpoint <ckpt名.safetensors>
# 3) 実際に生成（txt2img と、ネームの1コマ）
python local_tools/check_comfy.py --checkpoint <ckpt名.safetensors> --generate
# 4) 任意: 階層マージノードの互換性（2つ目のチェックポイントがある場合）
python local_tools/check_comfy.py --checkpoint <A> --checkpoint-b <B>
```

- FLUX / ANIMA は `--architecture flux|anima` に加え、`--text-encoder`（複数可）・`--vae` を指定する。ANIMA は必要に応じて `--text-encoder-type` も指定する（MASTER_GUIDE 3.5節）。
- **確認ポイント**: `ModelMergeSDXL` の入力名などが実機と違っても、CAF の `adapt_prompt_to_object_info` が実機の `/object_info` に合わせて名前を直す。マージ用ノードが無ければ `ModelMergeSimple` に自動で置き換える。チェック結果の「入力名を適合させました」等の行を `setup_report.md` に記録する。

### 7.4 CAF の ComfyUI 設定を保存

```bash
python -c "from core.config import FoundrySettings; s=FoundrySettings.load(); s.comfy.base_url='http://127.0.0.1:8188'; s.comfy.use_mock=False; print(s.save())"
```

（UI の 5 System & Local AI Config 画面でも設定できる。）

**完了条件**: 互換性チェックで「互換性チェック合格」と出て、`--generate` で `結果: success` が1件以上出たこと。生成画像は ComfyUI の `output/caf/` 配下にある。

---

## Step 8. 実機構成での起動と受け入れテスト

1. 起動順は ① LLM サーバ → ② ComfyUI → ③ CAF。CAF は CLI の一時上書き無しで起動し、`data/config.json` の設定を使う。
   ```bash
   python run_studio.py --no-browser
   ```
2. `http://127.0.0.1:8765/api/health` を確認する。`llm_provider` がモック以外で、`llm_fallback_active` が `false` であること。
3. 受け入れテストを実行する（LLM の速度によっては数十分かかる）。
   ```bash
   python local_tools/acceptance_test.py --checkpoint <ckpt名.safetensors>
   ```
   次を一気通貫で検証する。
   - 構造抽出（実機 LLM・4段階）
   - シードとの合成
   - ネーム生成と漫画文法の検証
   - SVG 出力
   - ComfyUI での実生成（先頭2コマ）
   - 3軸企画と5段階執筆
   - 一人称「私」統一と URL 除去
   - note 出力
   - DB バックアップ
   - テレメトリ
4. Style Alchemy Arena 用に、実画像つきの候補を登録する。マージ相手が無ければ `--model-b` は省略してよい。
   ```bash
   python local_tools/generate_arena_candidates.py --model-a <ckptA> --model-b <ckptB> --lora <LoRA名> -n 8
   ```
   全候補を同じシード・同じプロンプトで生成するので、画像の違いはレシピ（マージ比率・LoRA 重み・絵柄パラメータ）だけに由来する。

**完了条件**: 受け入れテストで「合格」と出て、Arena に実画像候補が登録されたこと。

---

## Step 9. 利用者と行う初回の実用ワークフロー

利用者にブラウザを操作してもらいながら、次を一緒に1周する。各画面の詳細は `docs/MASTER_GUIDE.md` 5章を参照。

1. **Structure Lab**: 利用者自身のあらすじやメモから5階層構造を抽出し、プリセット骨格と合成（Merge）して保存する。
2. **Manga Studio**:
   - 合成した構造からネームを生成する（例: 16ページ）。
   - 漫画文法スコアと SVG ネームを確認する。
   - ComfyUI 欄のチェックポイント名を実在ファイル名にしてワークフローを生成し、キューへ投入する。
3. **Style Alchemy Arena**:
   - 実画像候補で ←/→ キーの A/B 判定を10回以上行う（↓ か Space で引分）。
   - ランキングと美的選好ベクトルを確認する。
   - ベイズ最適化の提案レシピを書き出す。
   - 書き出したレシピを `generate_arena_candidates.py` 相当の手順、または ComfyUI で生成し、再び判定する（このループを繰り返す）。
4. **Asset Publisher**:
   - 3軸企画を展開し、5段階執筆を実行する。
   - 「私」統一と URL 除去の検証バッジを確認する。
   - note 用と Obsidian 用に書き出す。
5. **System & Local AI Config**:
   - 思考段階設定（速度重視なら critique / refine を無効化）を確認する。
   - テレメトリ（トークン・レイテンシ・修復回数）を確認する。
   - DB バックアップを作成する。

**完了条件**: 利用者が4つのループを一通り操作でき、成果物（ネーム SVG・生成画像・記事 Markdown）を確認できたこと。

---

## Step 10. 常用化

1. **起動スクリプト**をリポジトリ直下に作り、Git の管理から外す（`.git/info/exclude` に追記）。
   - `start_caf.sh`（macOS / Linux）
     ```bash
     #!/usr/bin/env bash
     cd "$(dirname "$0")"
     exec .venv/bin/python run_studio.py "$@"
     ```
     作成後に `chmod +x start_caf.sh` を実行する。
   - `start_caf.ps1`（Windows）
     ```powershell
     Set-Location $PSScriptRoot
     & .\.venv\Scripts\python.exe run_studio.py @args
     ```
   LLM サーバと ComfyUI の起動手順も、同じ場所に `README_LOCAL.md` として書き残す。
2. **更新手順**（本書のバージョン選択規則に従う）
   ```bash
   git fetch origin
   ```
   - 新しい版へ切り替えたら、`python -m pip install -e ".[dev]"` と `cd web_studio/frontend && npm ci && npm run build` を実行する。
   - 次回の `run_studio.py` 起動時に、追加されたシードが既存 DB へ自動で追加される。
3. **バックアップ**
   - UI の System 画面、または `POST /api/system/db/backup` で `data/backups/` に作成する。
   - `data/config.json` はバックアップに含まれないため、別途コピーする。
   - 復元は、CAF を停止してバックアップファイルを `data/foundry.sqlite3` に置き換える。
4. **最終報告**: `data/setup_report.md` に次をまとめ、利用者に提示する。
   - 完了の定義チェックリストの結果
   - 確定した LLM 設定（provider・モデル名・structured_output・timeout）
   - ComfyUI 設定と互換性チェックの警告
   - 未解決事項

---

## 付録A. 補助ツール（`local_tools/`）

| ツール | 用途 | 主な引数 |
|---|---|---|
| `check_llm.py` | LLM 接続・構造化出力・4段階ハーネスの実機検証と推奨設定の保存 | `--provider` `--base-url` `--model` `--mode` `--timeout` `--save` |
| `check_comfy.py` | ComfyUI の接続、モデル一覧、ワークフロー互換性（未導入ノード・入力名の適合）、実生成 | `--url` `--checkpoint` `--checkpoint-b` `--lora` `--architecture` `--generate` |
| `generate_arena_candidates.py` | 実画像つき Arena 候補の一括生成・登録（同一シードでレシピ差だけを比較） | `--model-a` `--model-b` `--lora` `-n` `--prompt` |
| `acceptance_test.py` | 起動中スタジオに対する一気通貫の受け入れテスト | `--allow-mock` `--checkpoint` `--pages` |

いずれも、安定版 `25f7668` でモック LLM・モック ComfyUI を相手に動作確認済み。実機での挙動は Step 6〜8 で確認する。

## 付録B. 設定（`data/config.json`）
優先順位は「環境変数 ＞ `data/config.json` ＞ 既定値」。Python から安全に変更するには次のようにする。

```python
from core.config import FoundrySettings
s = FoundrySettings.load()
s.llm.provider = "openai_compatible"      # "ollama" / "auto" / "mock"
s.llm.base_url = "http://localhost:8000/v1"
s.llm.model = "DeepSeek-V4.1-Flash"       # /v1/models が返す名前と一致させる
s.llm.structured_output = "json_object"   # check_llm.py の推奨値
s.llm.timeout_s = 600
s.comfy.base_url = "http://127.0.0.1:8188"
s.comfy.use_mock = False
print(s.save())
```

- 主な環境変数: `CAF_DATA_DIR` / `CAF_LLM_PROVIDER` / `CAF_LLM_BASE_URL` / `CAF_LLM_MODEL` / `CAF_LLM_API_KEY` / `CAF_OLLAMA_BASE_URL` / `CAF_COMFY_URL` / `CAF_COMFY_MOCK`
- 起動中のスタジオに即時反映したい場合は、UI の System 画面、または `PUT /api/system/config` を使う。

## 付録C. よくある問題と対処

| 症状 | 対処 |
|---|---|
| health の `llm_fallback_active` が true | LLM サーバ未起動、または URL・モデル名の不一致。`check_llm.py` で再検証する。 |
| 抽出が遅い・時間切れになる | `timeout_s` を延ばす。System 画面で critique / refine を無効化し、速度を優先する。 |
| 構造化出力の修復リトライが多い | `check_llm.py --mode` で別モードを試す。サーバ側のガイド付き生成（JSON Schema 対応）の有無を確認する。 |
| ComfyUI でノード不足と表示される | 該当カスタムノードを導入する。LBW が不要なら LoRA の階層ウェイトを使わない構成にする。 |
| ComfyUI で入力名の警告が出る | 自動適合済みなら生成は可能。結果を `setup_report.md` に記録する。 |
| ポートが競合する | `run_studio.py --port 8766` などで変更する（ComfyUI・LLM 側のポートも合わせて設定する）。 |
| Windows で npm が見つからない | Node.js を導入してシェルを再起動する（`run_studio.py` は `npm.cmd` を自動検出する）。 |
| `pytest` が失敗する | テストは書き換えない。Python のバージョンと依存のバージョンを記録し、エラー全文とともに報告する。 |
