"""LLM アダプタ群（DI 対象）。

- ``OpenAICompatibleAdapter``: vLLM / SGLang / llama.cpp server / LM Studio 等の OpenAI 互換 API
  （既定 ``http://localhost:8000/v1``）。``response_format`` の3モード、``reasoning_content`` /
  ``<think>`` 思考ログ、``usage.completion_tokens_details.reasoning_tokens`` に対応。
- ``OllamaAdapter``: Ollama の OpenAI 互換エンドポイント（既定 ``http://localhost:11434/v1``）。
  ヘルスチェックはネイティブ ``/api/tags`` を優先する。
- ``AutoFallbackAdapter``: 実機エンドポイントを順に試し、到達不能ならモックへ自動フォールバックする。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from core.common import truncate
from core.llm_client.errors import LLMConnectionError, LLMError, LLMResponseError
from core.llm_client.json_repair import strip_reasoning_tags
from core.llm_client.tokens import estimate_messages_tokens, estimate_tokens
from core.llm_client.types import HarnessStage, LLMRequest, LLMResponse, TokenUsage, user_msg

__all__ = [
    "BaseLLMAdapter",
    "OpenAICompatibleAdapter",
    "OllamaAdapter",
    "AutoFallbackAdapter",
    "STRUCTURED_OUTPUT_MODES",
]

logger = logging.getLogger(__name__)

STRUCTURED_OUTPUT_MODES = ("json_schema", "json_object", "prompt_only")
_SCHEMA_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-]+")
_THINK_MARKER_RE = re.compile(r"<\s*/?\s*think", re.IGNORECASE)


class BaseLLMAdapter(ABC):
    """全アダプタの共通インターフェース。"""

    provider: str = "base"
    model: str = ""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """正規化済みリクエストを実行し、応答を返す。"""

    async def health_check(self) -> dict[str, Any]:
        """最小のチャット補完（ping）を送って疎通を確認する。サブクラスはより軽量な方法で上書きする。"""
        request = LLMRequest(
            messages=[user_msg("ping")],
            max_tokens=1,
            task="system.health",
            stage=HarnessStage.SINGLE,
        )
        started = time.perf_counter()
        try:
            await self.complete(request)
        except LLMError as exc:
            return {"ok": False, "provider": self.provider, "model": self.model, "detail": str(exc), "models": []}
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": True,
            "provider": self.provider,
            "model": self.model,
            "detail": f"ping 応答 {latency_ms:.0f} ms",
            "models": [self.model] if self.model else [],
        }

    async def aclose(self) -> None:
        """保持している接続資源を解放する（既定では解放対象なし）。"""
        return None


def _sanitize_schema_name(name: str | None) -> str:
    cleaned = _SCHEMA_NAME_RE.sub("_", name or "structured_output").strip("_") or "structured_output"
    return cleaned[:64]


class OpenAICompatibleAdapter(BaseLLMAdapter):
    """OpenAI 互換 Chat Completions API アダプタ。"""

    provider = "openai_compatible"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "DeepSeek-V4.1-Flash",
        api_key: str = "EMPTY",
        timeout_s: float = 300.0,
        structured_output: str = "json_object",
        extra_body: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        max_transport_retries: int = 2,
        retry_backoff_s: float = 0.5,
    ) -> None:
        if structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(f"structured_output は {STRUCTURED_OUTPUT_MODES} のいずれか: {structured_output!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.structured_output = structured_output
        self.extra_body: dict[str, Any] = dict(extra_body or {})
        self._transport = transport
        self.max_transport_retries = max(0, int(max_transport_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    # -- HTTP 基盤 ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        """イベントループごとに AsyncClient を遅延生成する（別ループからの再利用による不具合を防ぐ）。"""
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            timeout = httpx.Timeout(self.timeout_s, connect=min(10.0, self.timeout_s))
            self._client = httpx.AsyncClient(timeout=timeout, transport=self._transport)
            self._client_loop = loop
        return self._client

    async def aclose(self) -> None:
        """HTTP クライアントを閉じる。"""
        client = self._client
        self._client = None
        if client is None or client.is_closed:
            return None
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if self._client_loop is current:
            await client.aclose()
        self._client_loop = None
        return None

    async def _request_json(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """一時障害（429/5xx/通信断）を指数バックオフで再試行しつつ JSON 応答を得る。"""
        last_error: BaseException | None = None
        attempts = self.max_transport_retries + 1
        for attempt in range(attempts):
            try:
                client = self._get_client()
                if method == "GET":
                    response = await client.get(url, headers=self._headers())
                else:
                    response = await client.post(url, json=payload, headers=self._headers())
            except httpx.TransportError as exc:
                last_error = exc
            else:
                status = response.status_code
                if status == 429 or status >= 500:
                    last_error = LLMResponseError(
                        f"HTTP {status}: {truncate(response.text, 300)}", status_code=status, body=response.text
                    )
                elif status >= 400:
                    raise LLMResponseError(
                        f"{url} が HTTP {status} を返しました: {truncate(response.text, 500)}",
                        status_code=status,
                        body=response.text,
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise LLMResponseError(
                            f"{url} の応答が JSON ではありません: {truncate(response.text, 300)}",
                            status_code=status,
                            body=response.text,
                        ) from exc
                    if not isinstance(data, dict):
                        raise LLMResponseError(f"{url} の応答がオブジェクトではありません", status_code=status)
                    return data
            if attempt < attempts - 1:
                delay = self.retry_backoff_s * (2**attempt)
                logger.warning("LLM エンドポイント一時障害（%s/%s）: %s", attempt + 1, attempts, last_error)
                if delay > 0:
                    await asyncio.sleep(delay)
        raise LLMConnectionError(f"{url} への要求が {attempts} 回失敗しました: {last_error}") from last_error

    # -- ペイロード -------------------------------------------------------

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """Chat Completions のリクエストボディを組み立てる。"""
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        wants_json = request.response_schema is not None or request.json_mode
        if wants_json and self.structured_output != "prompt_only":
            if self.structured_output == "json_schema" and request.response_schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": _sanitize_schema_name(request.schema_name),
                        "schema": request.response_schema,
                        "strict": False,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        for key, value in self.extra_body.items():
            payload[key] = value
        return payload

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """``_build_payload`` の公開エイリアス（テスト・デバッグ用）。"""
        return self._build_payload(request)

    # -- 応答解析 ---------------------------------------------------------

    def _parse_completion(self, data: dict[str, Any], request: LLMRequest, latency_ms: float) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise LLMResponseError(
                "choices が空の応答を受信しました", body=truncate(json.dumps(data, ensure_ascii=False), 2000)
            )
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        content = str(content)
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not isinstance(reasoning, str):
            reasoning = json.dumps(reasoning, ensure_ascii=False)
        if _THINK_MARKER_RE.search(content):
            content, think_text = strip_reasoning_tags(content)
            if think_text and think_text not in reasoning:
                reasoning = f"{reasoning}\n{think_text}".strip() if reasoning else think_text
        usage_raw = data.get("usage") or {}
        if isinstance(usage_raw, dict) and usage_raw:
            details = usage_raw.get("completion_tokens_details") or {}
            reasoning_tokens = int(details.get("reasoning_tokens") or usage_raw.get("reasoning_tokens") or 0)
            estimated = False
            if reasoning_tokens == 0 and reasoning:
                reasoning_tokens = estimate_tokens(reasoning)
                estimated = True
            usage = TokenUsage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
                reasoning_tokens=reasoning_tokens,
                estimated=estimated,
            )
        else:
            reasoning_tokens = estimate_tokens(reasoning)
            usage = TokenUsage(
                prompt_tokens=estimate_messages_tokens(request.messages),
                completion_tokens=estimate_tokens(content) + reasoning_tokens,
                reasoning_tokens=reasoning_tokens,
                estimated=True,
            )
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            model=str(data.get("model") or request.model or self.model),
            provider=self.provider,
            usage=usage,
            latency_ms=max(0.0, latency_ms),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            raw=data,
        )

    # -- 公開 API ---------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Chat Completions を実行する。"""
        payload = self._build_payload(request)
        started = time.perf_counter()
        data = await self._request_json("POST", f"{self.base_url}/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._parse_completion(data, request, latency_ms)

    async def list_models(self) -> list[str]:
        """``GET {base_url}/models`` でモデル ID 一覧を取得する。"""
        data = await self._request_json("GET", f"{self.base_url}/models")
        items = data.get("data") or data.get("models") or []
        names: list[str] = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("id") or item.get("name") or item.get("model")
                if name:
                    names.append(str(name))
            elif isinstance(item, str):
                names.append(item)
        return names

    async def health_check(self) -> dict[str, Any]:
        """モデル一覧取得で疎通とモデルの存在を確認する。"""
        try:
            models = await self.list_models()
        except LLMError as exc:
            return {"ok": False, "provider": self.provider, "model": self.model, "detail": str(exc), "models": []}
        available = _model_listed(self.model, models) if models else None
        if available is False:
            detail = f"接続成功。ただしモデル {self.model} は一覧（{len(models)} 件）にありません"
        else:
            detail = "接続成功"
        return {
            "ok": True,
            "provider": self.provider,
            "model": self.model,
            "detail": detail,
            "models": models,
            "model_available": available,
        }


def _model_listed(model: str, names: list[str]) -> bool:
    target = model.strip().lower()
    target_base = target.split(":", 1)[0]
    for name in names:
        candidate = name.strip().lower()
        if candidate == target or candidate.split(":", 1)[0] == target_base:
            return True
        if candidate.rsplit("/", 1)[-1] == target.rsplit("/", 1)[-1]:
            return True
    return False


class OllamaAdapter(OpenAICompatibleAdapter):
    """Ollama（OpenAI 互換 ``/v1``）アダプタ。"""

    provider = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "deepseek-v4.1-flash",
        api_key: str = "ollama",
        timeout_s: float = 300.0,
        structured_output: str = "json_object",
        extra_body: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        max_transport_retries: int = 2,
        retry_backoff_s: float = 0.5,
        keep_alive: str | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            structured_output=structured_output,
            extra_body=extra_body,
            transport=transport,
            max_transport_retries=max_transport_retries,
            retry_backoff_s=retry_backoff_s,
        )
        self.keep_alive = keep_alive

    @property
    def root_url(self) -> str:
        """``/v1`` を除いた Ollama のルート URL。"""
        url = self.base_url
        if url.endswith("/v1"):
            url = url[: -len("/v1")]
        return url.rstrip("/")

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload = super()._build_payload(request)
        if self.keep_alive:
            payload.setdefault("keep_alive", self.keep_alive)
        return payload

    async def health_check(self) -> dict[str, Any]:
        """ネイティブ ``/api/tags`` を優先し、失敗時は ``/v1/models`` で確認する。"""
        url = f"{self.root_url}/api/tags"
        try:
            response = await self._get_client().get(url, headers=self._headers())
        except httpx.TransportError as exc:
            native_error: str | None = f"{url}: {exc}"
        else:
            native_error = None
            if response.status_code < 400:
                try:
                    data = response.json()
                except ValueError:
                    data = {}
                models = [
                    str(item.get("name") or item.get("model"))
                    for item in (data.get("models") or [])
                    if isinstance(item, dict) and (item.get("name") or item.get("model"))
                ]
                available = _model_listed(self.model, models) if models else None
                detail = "Ollama 接続成功" if available is not False else f"Ollama 接続成功。ただしモデル {self.model} は未取得です"
                return {
                    "ok": True,
                    "provider": self.provider,
                    "model": self.model,
                    "detail": detail,
                    "models": models,
                    "model_available": available,
                }
            native_error = f"{url}: HTTP {response.status_code}"
        fallback = await super().health_check()
        if not fallback.get("ok") and native_error:
            fallback["detail"] = f"{native_error} / {fallback.get('detail', '')}"
        return fallback


class AutoFallbackAdapter(BaseLLMAdapter):
    """実機アダプタを順に試し、全滅ならフォールバック（通常はモック）を使う自動選択アダプタ。"""

    provider = "auto"

    def __init__(
        self,
        primaries: list[BaseLLMAdapter],
        fallback: BaseLLMAdapter,
        probe_timeout_s: float = 2.0,
    ) -> None:
        self.primaries = list(primaries)
        self.fallback = fallback
        self.probe_timeout_s = float(probe_timeout_s)
        self._selected: BaseLLMAdapter | None = None
        self.fallback_active = False
        self.probe_results: list[dict[str, Any]] = []

    @property
    def model(self) -> str:  # type: ignore[override]
        target = self._selected or (self.primaries[0] if self.primaries else self.fallback)
        return target.model

    @property
    def selected(self) -> BaseLLMAdapter | None:
        """選択済みアダプタ（未選択なら None）。"""
        return self._selected

    @property
    def selected_provider(self) -> str:
        """選択済みアダプタのプロバイダ名（未選択なら ``undetermined``）。"""
        return self._selected.provider if self._selected is not None else "undetermined"

    async def _probe(self, adapter: BaseLLMAdapter) -> dict[str, Any]:
        try:
            info = await asyncio.wait_for(adapter.health_check(), timeout=self.probe_timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            info = {"ok": False, "provider": adapter.provider, "model": adapter.model, "detail": "probe timeout"}
        except Exception as exc:  # 疎通確認はいかなる例外でも選択処理を止めない
            info = {"ok": False, "provider": adapter.provider, "model": adapter.model, "detail": str(exc)}
        info.setdefault("provider", adapter.provider)
        return info

    async def select(self, *, force: bool = False) -> BaseLLMAdapter:
        """疎通確認でアダプタを選択する（選択結果はキャッシュ。``force`` で再探索）。"""
        if self._selected is not None and not force:
            return self._selected
        results: list[dict[str, Any]] = []
        chosen: BaseLLMAdapter | None = None
        for adapter in self.primaries:
            info = await self._probe(adapter)
            results.append(info)
            if info.get("ok"):
                chosen = adapter
                break
        self.probe_results = results
        if chosen is None:
            self._selected = self.fallback
            self.fallback_active = True
            logger.warning("実機 LLM に到達できないためフォールバック（%s）を使用します", self.fallback.provider)
        else:
            self._selected = chosen
            self.fallback_active = False
        return self._selected

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """選択済みアダプタで実行し、実機が接続障害を起こした場合はフォールバックで再実行する。"""
        adapter = await self.select()
        try:
            return await adapter.complete(request)
        except LLMConnectionError:
            if adapter is self.fallback:
                raise
            logger.warning("%s で接続障害が発生したためフォールバックへ切り替えます", adapter.provider)
            self._selected = self.fallback
            self.fallback_active = True
            return await self.fallback.complete(request)

    async def health_check(self) -> dict[str, Any]:
        """実機を再探索し、選択状態を含む疎通情報を返す（フォールバックは常に利用可能）。"""
        adapter = await self.select(force=True)
        info = await adapter.health_check() if adapter is self.fallback else dict(self.probe_results[-1])
        return {
            "ok": bool(info.get("ok", True)),
            "provider": self.provider,
            "selected_provider": adapter.provider,
            "fallback_active": self.fallback_active,
            "model": adapter.model,
            "detail": info.get("detail", ""),
            "models": list(info.get("models", [])),
            "probes": self.probe_results,
        }

    async def aclose(self) -> None:
        """配下の全アダプタを閉じる。"""
        for adapter in [*self.primaries, self.fallback]:
            await adapter.aclose()
        return None
