"""LLM アダプタ／クライアントのファクトリ（DI 切替点）。

環境変数 ``CAF_LLM_PROVIDER``（``auto`` / ``mock`` / ``openai_compatible`` / ``ollama``）が設定されていれば
``LLMSettings.provider`` より優先する。テスト・クラウド VM ではこれを ``mock`` に固定する。
"""

from __future__ import annotations

import os

from core.config import LLMSettings
from core.db import Database
from core.llm_client.adapters import AutoFallbackAdapter, BaseLLMAdapter, OllamaAdapter, OpenAICompatibleAdapter
from core.llm_client.client import LLMClient
from core.llm_client.harness import HarnessConfig, MultiStageReasoningHarness
from core.llm_client.mock import MockLLMAdapter
from core.llm_client.telemetry import TelemetryStore

__all__ = ["resolve_provider", "create_adapter", "create_llm_client", "create_harness", "VALID_PROVIDERS"]

VALID_PROVIDERS = ("auto", "mock", "openai_compatible", "ollama")


def resolve_provider(settings: LLMSettings) -> str:
    """環境変数を考慮した実効プロバイダ名。"""
    env_value = os.environ.get("CAF_LLM_PROVIDER", "").strip().lower()
    if env_value in VALID_PROVIDERS:
        return env_value
    return settings.provider


def create_adapter(settings: LLMSettings | None = None) -> BaseLLMAdapter:
    """設定からアダプタを生成する。"""
    settings = settings or LLMSettings()
    provider = resolve_provider(settings)
    if provider == "mock":
        return MockLLMAdapter(seed=settings.mock_seed)
    openai_adapter = OpenAICompatibleAdapter(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        timeout_s=settings.timeout_s,
        structured_output=settings.structured_output,
    )
    if provider == "openai_compatible":
        return openai_adapter
    ollama_adapter = OllamaAdapter(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        api_key="ollama",
        timeout_s=settings.timeout_s,
        structured_output=settings.structured_output,
    )
    if provider == "ollama":
        return ollama_adapter
    return AutoFallbackAdapter([openai_adapter, ollama_adapter], MockLLMAdapter(seed=settings.mock_seed))


def create_llm_client(
    settings: LLMSettings | None = None,
    db: Database | None = None,
    adapter: BaseLLMAdapter | None = None,
) -> LLMClient:
    """アダプタ・テレメトリ・自己修復回数を設定した ``LLMClient`` を生成する。"""
    settings = settings or LLMSettings()
    chosen = adapter if adapter is not None else create_adapter(settings)
    telemetry = TelemetryStore(db) if db is not None else None
    return LLMClient(chosen, telemetry=telemetry, max_repair_attempts=settings.harness.max_repair_attempts)


def create_harness(
    settings: LLMSettings | None = None,
    db: Database | None = None,
    adapter: BaseLLMAdapter | None = None,
    client: LLMClient | None = None,
) -> MultiStageReasoningHarness:
    """設定の段階構成を反映した ``MultiStageReasoningHarness`` を生成する。"""
    settings = settings or LLMSettings()
    llm_client = client if client is not None else create_llm_client(settings, db=db, adapter=adapter)
    return MultiStageReasoningHarness(llm_client, HarnessConfig.from_settings(settings.harness))
