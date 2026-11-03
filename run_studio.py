#!/usr/bin/env python3
"""Creative Asset Foundry — ワンコマンド起動ランチャー。

    python run_studio.py

を実行すると、次を自動で行う（Windows / macOS / Linux 対応）。

1. Python バージョン・依存パッケージの検査（``--install-deps`` で不足分を自動インストール）
2. データディレクトリ・SQLite データベースの初期化と初期シード60パターンの投入
3. フロントエンド（React + Vite）の依存インストールと本番ビルド（dist が無い場合のみ。Node.js が必要）
4. FastAPI バックエンドの起動（ビルド済み UI を同一オリジンで配信）とブラウザの自動オープン

``--dev`` を付けると Vite 開発サーバ（ホットリロード、ポート5173）とバックエンドを同時起動する。
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "web_studio" / "frontend"
DIST_INDEX = FRONTEND_DIR / "dist" / "index.html"
REQUIRED_MODULES = {
    "pydantic": "pydantic>=2.6,<3",
    "numpy": "numpy>=1.26",
    "httpx": "httpx>=0.27",
    "fastapi": "fastapi>=0.110",
    "uvicorn": "uvicorn[standard]>=0.29",
    "websockets": "websockets>=12",
}
MIN_PYTHON = (3, 10)


def _print(message: str) -> None:
    print(f"[CAF] {message}", flush=True)


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 以上が必要です（現在 {sys.version.split()[0]}）。"
        )


def missing_modules() -> list[str]:
    missing: list[str] = []
    for module, requirement in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(requirement)
    return missing


def ensure_dependencies(install: bool) -> None:
    missing = missing_modules()
    if not missing:
        return
    if not install:
        joined = " ".join(f'"{req}"' for req in missing)
        raise SystemExit(
            "不足している Python パッケージがあります: "
            + ", ".join(missing)
            + f"\n  python -m pip install {joined}\nを実行するか、--install-deps を付けて再実行してください。"
        )
    _print("不足パッケージをインストールします: " + ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    importlib.invalidate_caches()


def npm_executable() -> str | None:
    candidates = ["npm.cmd", "npm"] if os.name == "nt" else ["npm"]
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def build_frontend(force: bool) -> bool:
    """フロントエンドをビルドする。ビルド済みまたは成功なら True。"""
    if DIST_INDEX.exists() and not force:
        return True
    npm = npm_executable()
    if npm is None:
        _print("Node.js / npm が見つからないため UI のビルドをスキップします（API は利用可能です）。")
        return False
    if not (FRONTEND_DIR / "node_modules").exists():
        _print("フロントエンド依存をインストールしています（初回のみ）…")
        subprocess.check_call([npm, "install", "--no-audit", "--no-fund"], cwd=FRONTEND_DIR)
    _print("フロントエンドを本番ビルドしています…")
    subprocess.check_call([npm, "run", "build"], cwd=FRONTEND_DIR)
    return DIST_INDEX.exists()


def apply_environment(args: argparse.Namespace) -> None:
    """CLI 引数を環境変数へ反映（バックエンドは core.config の環境変数上書きで読み取る）。"""
    if args.data_dir:
        os.environ["CAF_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())
    if args.provider:
        os.environ["CAF_LLM_PROVIDER"] = args.provider
    if args.llm_url:
        os.environ["CAF_LLM_BASE_URL"] = args.llm_url
    if args.model:
        os.environ["CAF_LLM_MODEL"] = args.model
    if args.ollama_url:
        os.environ["CAF_OLLAMA_BASE_URL"] = args.ollama_url
    if args.comfy_url:
        os.environ["CAF_COMFY_URL"] = args.comfy_url
    if args.comfy_mock:
        os.environ["CAF_COMFY_MOCK"] = "1"


def initialize_foundry() -> dict[str, object]:
    """設定読み込み・DB初期化・初期シード投入を行い、要約を返す。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from core.config import FoundrySettings
    from core.db import Database
    from core.structure_engine.repository import StructureRepository
    from core.structure_engine.seeds.loader import seed_database

    settings = FoundrySettings.load()
    settings.ensure_dirs()
    if not settings.config_path.exists():
        settings.save()
    db = Database(settings.db_path)
    repo = StructureRepository(db)
    report = seed_database(repo, overwrite=False)
    return {
        "data_dir": str(settings.data_dir),
        "db_path": str(settings.db_path),
        "llm_provider": settings.llm.provider,
        "llm_base_url": settings.llm.base_url,
        "comfy_url": settings.comfy.base_url,
        "seeds_total": report.total,
        "seeds_inserted": report.inserted,
    }


def wait_and_open(url: str, health_url: str, open_browser: bool, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.status == 200:
                    _print(f"スタジオ起動完了: {url}")
                    if open_browser:
                        webbrowser.open(url)
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    _print(f"ヘルスチェックがタイムアウトしました（{health_url}）。ログを確認してください。")


def start_vite_dev_server() -> subprocess.Popen[bytes] | None:
    npm = npm_executable()
    if npm is None:
        _print("npm が見つからないため Vite 開発サーバを起動できません。")
        return None
    if not (FRONTEND_DIR / "node_modules").exists():
        subprocess.check_call([npm, "install", "--no-audit", "--no-fund"], cwd=FRONTEND_DIR)
    _print("Vite 開発サーバを起動します: http://127.0.0.1:5173")
    return subprocess.Popen([npm, "run", "dev", "--", "--host", "127.0.0.1"], cwd=FRONTEND_DIR)


def run_server(args: argparse.Namespace) -> None:
    import uvicorn

    host_for_url = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    api_url = f"http://{host_for_url}:{args.port}"
    ui_url = "http://127.0.0.1:5173" if args.dev else api_url
    threading.Thread(
        target=wait_and_open,
        args=(ui_url, f"{api_url}/api/health", not args.no_browser),
        daemon=True,
    ).start()
    uvicorn.run(
        "web_studio.backend.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creative Asset Foundry ローカルスタジオ起動")
    parser.add_argument("--host", default="127.0.0.1", help="バックエンドの待受ホスト（既定 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="バックエンドのポート（既定 8765）")
    parser.add_argument("--data-dir", help="データディレクトリ（既定 ./data）")
    parser.add_argument(
        "--provider",
        choices=["auto", "mock", "openai_compatible", "ollama"],
        help="LLM プロバイダ（既定 auto: OpenAI互換→Ollama→モックの順に自動選択）",
    )
    parser.add_argument("--llm-url", help="OpenAI互換エンドポイント（既定 http://localhost:8000/v1）")
    parser.add_argument("--ollama-url", help="Ollama エンドポイント（既定 http://localhost:11434/v1）")
    parser.add_argument("--model", help="LLM モデル名（既定 DeepSeek-V4.1-Flash）")
    parser.add_argument("--comfy-url", help="ComfyUI エンドポイント（既定 http://127.0.0.1:8188）")
    parser.add_argument("--comfy-mock", action="store_true", help="ComfyUI を内蔵モックサーバで代替する")
    parser.add_argument("--dev", action="store_true", help="Vite 開発サーバ（ホットリロード）を同時起動する")
    parser.add_argument("--skip-frontend", action="store_true", help="フロントエンドのビルドを行わない")
    parser.add_argument("--rebuild-frontend", action="store_true", help="dist があっても再ビルドする")
    parser.add_argument("--install-deps", action="store_true", help="不足 Python パッケージを自動インストールする")
    parser.add_argument("--init-only", action="store_true", help="DB初期化とシード投入のみ行い終了する")
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    check_python()
    ensure_dependencies(args.install_deps)
    apply_environment(args)
    os.chdir(ROOT)
    summary = initialize_foundry()
    _print(
        f"データ: {summary['data_dir']} / DB: {summary['db_path']} / "
        f"シード: {summary['seeds_total']}件（今回投入 {summary['seeds_inserted']}件）"
    )
    _print(f"LLM: {summary['llm_provider']} ({summary['llm_base_url']}) / ComfyUI: {summary['comfy_url']}")
    if args.init_only:
        _print("初期化のみ完了しました。")
        return 0
    vite_process: subprocess.Popen[bytes] | None = None
    if args.dev:
        vite_process = start_vite_dev_server()
    elif not args.skip_frontend:
        try:
            build_frontend(force=args.rebuild_frontend)
        except subprocess.CalledProcessError as exc:
            _print(f"フロントエンドのビルドに失敗しました（{exc}）。API のみで起動します。")
    try:
        run_server(args)
    except KeyboardInterrupt:
        _print("停止しました。")
    finally:
        if vite_process is not None and vite_process.poll() is None:
            vite_process.terminate()
            try:
                vite_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vite_process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
