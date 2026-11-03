"""Foundry 全体設定（LLM / ComfyUI / データパス）。

優先順位: 環境変数 > ``<data_dir>/config.json`` > 既定値。
Web Studio の System 画面から更新された設定は ``save()`` で ``config.json`` に永続化される。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "HarnessSettings",
    "LLMSettings",
    "ComfySettings",
    "FoundrySettings",
    "get_default_data_dir",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parents[1]

_TRUE_VALUES = {"1", "true", "yes", "on"}


class HarnessSettings(BaseModel):
    """4段階思考増幅ハーネスの段階設定。"""

    enable_decompose: bool = True
    enable_critique: bool = True
    enable_refine: bool = True
    critique_accept_score: float = Field(85.0, ge=0.0, le=100.0)
    skip_refine_when_accepted: bool = False
    max_repair_attempts: int = Field(3, ge=0, le=10)
    max_refine_rounds: int = Field(2, ge=0, le=6)
    stage_temperatures: dict[str, float] = Field(
        default_factory=lambda: {"decompose": 0.4, "synthesize": 0.3, "critique": 0.1, "refine": 0.2}
    )


class LLMSettings(BaseModel):
    """ローカル DeepSeek-V4.1-Flash 接続設定。"""

    provider: Literal["auto", "mock", "openai_compatible", "ollama"] = "auto"
    base_url: str = "http://localhost:8000/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    model: str = "DeepSeek-V4.1-Flash"
    ollama_model: str = "deepseek-v4.1-flash"
    api_key: str = "EMPTY"
    timeout_s: float = Field(300.0, gt=0.0)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(8192, ge=256)
    structured_output: Literal["json_schema", "json_object", "prompt_only"] = "json_object"
    mock_seed: int = 20260101
    harness: HarnessSettings = Field(default_factory=HarnessSettings)


class ComfySettings(BaseModel):
    """ローカル ComfyUI 接続設定。"""

    base_url: str = "http://127.0.0.1:8188"
    timeout_s: float = Field(600.0, gt=0.0)
    client_id_prefix: str = "caf"
    output_prefix: str = "caf"
    use_mock: bool = False


def get_default_data_dir() -> Path:
    """既定のデータディレクトリ（``CAF_DATA_DIR`` または ``<repo>/data``）。"""
    env = os.environ.get("CAF_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "data").resolve()


class FoundrySettings(BaseModel):
    """Foundry 全体設定のルート。"""

    data_dir: Path = Field(default_factory=get_default_data_dir)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    comfy: ComfySettings = Field(default_factory=ComfySettings)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "foundry.sqlite3"

    @property
    def backup_dir(self) -> Path:
        return self.data_dir / "backups"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"

    def ensure_dirs(self) -> None:
        """データ・バックアップ・エクスポート用ディレクトリを作成する。"""
        for path in (self.data_dir, self.backup_dir, self.export_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, data_dir: Path | str | None = None) -> "FoundrySettings":
        """``config.json`` と環境変数から設定を構築する。"""
        base_dir = Path(data_dir).expanduser().resolve() if data_dir is not None else get_default_data_dir()
        payload: dict[str, Any] = {}
        config_file = base_dir / "config.json"
        if config_file.exists():
            try:
                loaded = json.loads(config_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                payload = loaded
        payload["data_dir"] = str(base_dir)
        settings = cls.model_validate(payload)
        settings.apply_env_overrides()
        return settings

    def apply_env_overrides(self) -> None:
        """環境変数による上書きを適用する（テスト・クラウド実行での DI 切替用）。"""
        env = os.environ
        provider = env.get("CAF_LLM_PROVIDER")
        if provider in {"auto", "mock", "openai_compatible", "ollama"}:
            self.llm.provider = provider  # type: ignore[assignment]
        if env.get("CAF_LLM_BASE_URL"):
            self.llm.base_url = env["CAF_LLM_BASE_URL"]
        if env.get("CAF_OLLAMA_BASE_URL"):
            self.llm.ollama_base_url = env["CAF_OLLAMA_BASE_URL"]
        if env.get("CAF_LLM_MODEL"):
            self.llm.model = env["CAF_LLM_MODEL"]
        if env.get("CAF_LLM_API_KEY"):
            self.llm.api_key = env["CAF_LLM_API_KEY"]
        if env.get("CAF_COMFY_URL"):
            self.comfy.base_url = env["CAF_COMFY_URL"]
        if env.get("CAF_COMFY_MOCK") is not None:
            self.comfy.use_mock = env["CAF_COMFY_MOCK"].strip().lower() in _TRUE_VALUES

    def save(self) -> Path:
        """設定を ``config.json`` に保存し、そのパスを返す。"""
        self.ensure_dirs()
        data = self.model_dump(mode="json")
        data.pop("data_dir", None)
        self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.config_path

    def public_dict(self) -> dict[str, Any]:
        """API 応答用（APIキーをマスク）。"""
        data = self.model_dump(mode="json")
        key = data["llm"].get("api_key") or ""
        data["llm"]["api_key"] = (key[:2] + "***") if len(key) > 4 else ("***" if key else "")
        data["db_path"] = str(self.db_path)
        return data
