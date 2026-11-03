"""ComfyUI 非同期クライアント（キュー投入・WebSocket 進捗監視・出力取得）。

ComfyUI のサーバー API:

- ``POST /prompt`` ``{"prompt": <API形式>, "client_id": <id>}`` → ``{"prompt_id", "number", "node_errors"}``
- ``GET /history/{prompt_id}`` → ``{prompt_id: {"outputs": {node_id: {"images": [...]}}, "status": {...}}}``
- ``GET /view?filename=&subfolder=&type=`` → 画像バイト列
- ``GET /object_info[/{class}]`` / ``GET /system_stats`` / ``POST /interrupt``
- ``WS /ws?clientId=<id>`` → ``status`` / ``execution_start`` / ``execution_cached`` / ``executing`` /
  ``progress`` / ``executed`` / ``execution_success`` / ``execution_error`` / ``execution_interrupted``

``run_workflow`` は投入前に WebSocket を接続してイベント取りこぼしを防ぎ、完了（``executing`` で
node=None、または ``execution_success``）を検知したら履歴から出力画像を収集する。WebSocket が使えない
場合や沈黙が続く場合は履歴ポーリングで補完する。ループバック宛の通信はプロキシを経由しない。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ComfyError",
    "ComfyConnectionError",
    "ComfyProgressEvent",
    "ComfyImageRef",
    "ComfyExecutionResult",
    "ComfyClient",
    "ProgressCallback",
]

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class ComfyError(Exception):
    """ComfyUI 呼び出しの失敗（ノード検証エラー等）。"""

    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class ComfyConnectionError(ComfyError):
    """ComfyUI へ接続できない。"""


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ComfyProgressEvent(_Base):
    """WebSocket イベントの正規化表現。"""

    type: str
    prompt_id: str | None = None
    node: str | None = None
    value: int | None = None
    max: int | None = None
    progress: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)


class ComfyImageRef(_Base):
    filename: str
    subfolder: str = ""
    type: str = "output"


class ComfyExecutionResult(_Base):
    prompt_id: str
    status: Literal["success", "error", "interrupted", "timeout"]
    outputs: dict[str, list[ComfyImageRef]] = Field(default_factory=dict)
    events: int = 0
    error: str | None = None
    duration_s: float = 0.0


ProgressCallback = Callable[[ComfyProgressEvent], Any]


def _ws_errors() -> tuple[type[BaseException], ...]:
    """WebSocket 接続・通信失敗として履歴ポーリングへ切り替える例外群。"""
    errors: list[type[BaseException]] = [OSError, asyncio.TimeoutError]
    try:
        from websockets.exceptions import WebSocketException

        errors.append(WebSocketException)
    except ImportError:
        return tuple(errors)
    return tuple(errors)


def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


class ComfyClient:
    """ComfyUI の HTTP / WebSocket API クライアント。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        client_id: str | None = None,
        timeout_s: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
        ws_connect: Callable[..., Any] | None = None,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or f"caf-{uuid.uuid4().hex[:12]}"
        self.timeout_s = float(timeout_s)
        self._transport = transport
        self._ws_connect = ws_connect
        self.poll_interval_s = float(poll_interval_s)

    # -- HTTP -----------------------------------------------------------------

    @property
    def ws_url(self) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/ws?clientId={self.client_id}"

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            transport=self._transport,
            timeout=httpx.Timeout(min(self.timeout_s, 120.0)),
            trust_env=not _is_loopback(self.base_url),
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            async with self._client() as client:
                response = await client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise ComfyConnectionError(f"ComfyUI に接続できません（{self.base_url}）: {exc}") from exc
        return response

    async def queue_prompt(self, prompt: dict[str, Any], *, extra_data: dict[str, Any] | None = None) -> str:
        """ワークフローをキューへ投入し prompt_id を返す。"""
        body: dict[str, Any] = {"prompt": prompt, "client_id": self.client_id}
        if extra_data:
            body["extra_data"] = extra_data
        response = await self._request("POST", "/prompt", json=body)
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"raw": response.text}
        if response.status_code >= 400:
            message = payload.get("error", {}).get("message") if isinstance(payload, dict) and isinstance(payload.get("error"), dict) else None
            raise ComfyError(
                f"ComfyUI がワークフローを拒否しました（HTTP {response.status_code}）: {message or payload}",
                status_code=response.status_code,
                payload=payload,
            )
        prompt_id = payload.get("prompt_id") if isinstance(payload, dict) else None
        if not prompt_id:
            raise ComfyError("ComfyUI の応答に prompt_id がありません", status_code=response.status_code, payload=payload)
        return str(prompt_id)

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        """prompt_id の履歴エントリ（未完了なら空 dict）。"""
        response = await self._request("GET", f"/history/{prompt_id}")
        if response.status_code >= 400:
            raise ComfyError(f"履歴の取得に失敗しました（HTTP {response.status_code}）", status_code=response.status_code)
        payload = response.json()
        if isinstance(payload, dict):
            entry = payload.get(prompt_id)
            if isinstance(entry, dict):
                return entry
        return {}

    async def get_object_info(self, node_class: str | None = None) -> dict[str, Any]:
        path = f"/object_info/{node_class}" if node_class else "/object_info"
        response = await self._request("GET", path)
        if response.status_code >= 400:
            raise ComfyError(f"object_info の取得に失敗しました（HTTP {response.status_code}）", status_code=response.status_code)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    async def view_image(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:  # noqa: A002 - API 名に合わせる
        response = await self._request("GET", "/view", params={"filename": filename, "subfolder": subfolder, "type": type})
        if response.status_code >= 400:
            raise ComfyError(f"画像 {filename} の取得に失敗しました（HTTP {response.status_code}）", status_code=response.status_code)
        return response.content

    async def interrupt(self) -> None:
        response = await self._request("POST", "/interrupt")
        if response.status_code >= 400:
            raise ComfyError(f"中断要求に失敗しました（HTTP {response.status_code}）", status_code=response.status_code)

    async def system_stats(self) -> dict[str, Any]:
        response = await self._request("GET", "/system_stats")
        if response.status_code >= 400:
            raise ComfyError(f"system_stats の取得に失敗しました（HTTP {response.status_code}）", status_code=response.status_code)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    # -- WebSocket ----------------------------------------------------------------

    def _connect_ws(self) -> Any:
        if self._ws_connect is not None:
            return self._ws_connect(self.ws_url)
        import websockets

        kwargs: dict[str, Any] = {"max_size": None, "open_timeout": 10}
        if _is_loopback(self.base_url):
            kwargs["proxy"] = None
        try:
            return websockets.connect(self.ws_url, **kwargs)
        except TypeError:
            kwargs.pop("proxy", None)
            return websockets.connect(self.ws_url, **kwargs)

    @staticmethod
    def parse_event(message: dict[str, Any]) -> ComfyProgressEvent:
        """WebSocket メッセージ（JSON）を ComfyProgressEvent へ正規化する。"""
        kind = str(message.get("type", "unknown"))
        data = message.get("data") if isinstance(message.get("data"), dict) else {}
        value = data.get("value")
        maximum = data.get("max")
        progress = 0.0
        if isinstance(value, (int, float)) and isinstance(maximum, (int, float)) and maximum:
            progress = max(0.0, min(1.0, float(value) / float(maximum)))
        if kind in ("execution_success",):
            progress = 1.0
        node = data.get("node")
        return ComfyProgressEvent(
            type=kind,
            prompt_id=data.get("prompt_id"),
            node=str(node) if node is not None else None,
            value=int(value) if isinstance(value, (int, float)) else None,
            max=int(maximum) if isinstance(maximum, (int, float)) else None,
            progress=progress,
            raw=message,
        )

    @staticmethod
    async def _emit(callback: ProgressCallback | None, event: ComfyProgressEvent) -> None:
        if callback is None:
            return
        outcome = callback(event)
        if inspect.isawaitable(outcome):
            await outcome

    async def _collect_outputs(self, prompt_id: str) -> tuple[dict[str, list[ComfyImageRef]], str | None, bool]:
        """履歴から出力画像を収集する。戻り値: (outputs, エラー文, 完了済みか)。"""
        entry = await self.get_history(prompt_id)
        if not entry:
            return {}, None, False
        outputs: dict[str, list[ComfyImageRef]] = {}
        for node_id, node_output in (entry.get("outputs") or {}).items():
            images = node_output.get("images") if isinstance(node_output, dict) else None
            if images:
                outputs[str(node_id)] = [ComfyImageRef.model_validate(image) for image in images]
        status = entry.get("status") or {}
        error = None
        if isinstance(status, dict) and status.get("status_str") == "error":
            error = "ComfyUI の実行がエラーで終了しました"
        completed = bool(status.get("completed", True)) if isinstance(status, dict) else True
        return outputs, error, completed

    async def _consume(
        self,
        ws: Any,
        prompt_id: str,
        on_progress: ProgressCallback | None,
        deadline: float,
        started: float,
    ) -> ComfyExecutionResult:
        events = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ComfyExecutionResult(
                    prompt_id=prompt_id, status="timeout", events=events, error="完了待ちがタイムアウトしました",
                    duration_s=time.monotonic() - started,
                )
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=min(remaining, max(self.poll_interval_s, 0.05)))
            except asyncio.TimeoutError:
                outputs, error, done = await self._collect_outputs(prompt_id)
                if done:
                    return ComfyExecutionResult(
                        prompt_id=prompt_id, status="error" if error else "success", outputs=outputs, events=events,
                        error=error, duration_s=time.monotonic() - started,
                    )
                continue
            if isinstance(message, (bytes, bytearray)):
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            event = self.parse_event(payload)
            if event.prompt_id not in (None, prompt_id):
                continue
            events += 1
            await self._emit(on_progress, event)
            finished = (event.type == "executing" and event.node is None and event.prompt_id == prompt_id) or event.type == "execution_success"
            if event.type == "execution_error":
                data = payload.get("data") or {}
                return ComfyExecutionResult(
                    prompt_id=prompt_id, status="error", events=events,
                    error=str(data.get("exception_message") or data.get("error") or "execution_error"),
                    duration_s=time.monotonic() - started,
                )
            if event.type == "execution_interrupted":
                return ComfyExecutionResult(
                    prompt_id=prompt_id, status="interrupted", events=events, error="実行が中断されました",
                    duration_s=time.monotonic() - started,
                )
            if finished:
                outputs, error, _ = await self._collect_outputs(prompt_id)
                return ComfyExecutionResult(
                    prompt_id=prompt_id, status="error" if error else "success", outputs=outputs, events=events,
                    error=error, duration_s=time.monotonic() - started,
                )

    async def wait_for_completion(
        self, prompt_id: str, on_progress: ProgressCallback | None = None, timeout_s: float | None = None
    ) -> ComfyExecutionResult:
        """投入済み prompt の完了を待つ（既に完了済みなら即座に結果を返す）。"""
        started = time.monotonic()
        deadline = started + (timeout_s if timeout_s is not None else self.timeout_s)
        outputs, error, done = await self._collect_outputs(prompt_id)
        if done:
            return ComfyExecutionResult(
                prompt_id=prompt_id, status="error" if error else "success", outputs=outputs, error=error,
                duration_s=time.monotonic() - started,
            )
        try:
            async with self._connect_ws() as ws:
                return await self._consume(ws, prompt_id, on_progress, deadline, started)
        except _ws_errors() as exc:
            return await self._poll_until_done(prompt_id, deadline, started, reason=str(exc))

    async def _poll_until_done(self, prompt_id: str, deadline: float, started: float, reason: str) -> ComfyExecutionResult:
        while time.monotonic() < deadline:
            outputs, error, done = await self._collect_outputs(prompt_id)
            if done:
                return ComfyExecutionResult(
                    prompt_id=prompt_id, status="error" if error else "success", outputs=outputs, error=error,
                    duration_s=time.monotonic() - started,
                )
            await asyncio.sleep(self.poll_interval_s)
        return ComfyExecutionResult(
            prompt_id=prompt_id, status="timeout", error=f"完了待ちがタイムアウトしました（WebSocket 不可: {reason}）",
            duration_s=time.monotonic() - started,
        )

    async def run_workflow(
        self, prompt: dict[str, Any], on_progress: ProgressCallback | None = None, timeout_s: float | None = None
    ) -> ComfyExecutionResult:
        """WebSocket 接続 → キュー投入 → 進捗監視 → 出力収集 を一括で行う。"""
        started = time.monotonic()
        deadline = started + (timeout_s if timeout_s is not None else self.timeout_s)
        prompt_id: str | None = None
        try:
            async with self._connect_ws() as ws:
                prompt_id = await self.queue_prompt(prompt)
                return await self._consume(ws, prompt_id, on_progress, deadline, started)
        except _ws_errors() as exc:
            if prompt_id is None:
                prompt_id = await self.queue_prompt(prompt)
            return await self._poll_until_done(prompt_id, deadline, started, reason=str(exc))

    async def run_many(
        self, prompts: list[dict[str, Any]], on_progress: ProgressCallback | None = None, timeout_s: float | None = None
    ) -> list[ComfyExecutionResult]:
        """複数ワークフローを順に実行する（1接続を再利用せず、各ワークフローの完了を個別に待つ）。"""
        results: list[ComfyExecutionResult] = []
        for prompt in prompts:
            results.append(await self.run_workflow(prompt, on_progress=on_progress, timeout_s=timeout_s))
        return results
