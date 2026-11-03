"""テスト共通設定。

- 外部ネットワークへ一切依存しないよう、LLMプロバイダを強制的に ``mock`` に固定する。
- テスト中に生成される SQLite / エクスポートはすべて一時ディレクトリへ隔離する。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CAF_LLM_PROVIDER"] = "mock"
os.environ.setdefault("CAF_COMFY_MOCK", "1")
os.environ.setdefault("CAF_DATA_DIR", tempfile.mkdtemp(prefix="caf_test_data_"))


@pytest.fixture
def tmp_db(tmp_path: Path):
    """テストごとに独立した SQLite データベース。"""
    from core.db import Database

    return Database(tmp_path / "caf_test.sqlite3")


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """一時データディレクトリを指す FoundrySettings（LLMはモック固定）。"""
    from core.config import FoundrySettings

    data_dir = tmp_path / "data"
    monkeypatch.setenv("CAF_DATA_DIR", str(data_dir))
    settings = FoundrySettings.load(data_dir)
    settings.llm.provider = "mock"
    settings.comfy.use_mock = True
    settings.ensure_dirs()
    return settings


@pytest.fixture
def mock_llm_client(tmp_db):
    """テレメトリ付きのモックLLMクライアント。"""
    from core.config import LLMSettings
    from core.llm_client import create_llm_client

    return create_llm_client(LLMSettings(provider="mock"), db=tmp_db)
