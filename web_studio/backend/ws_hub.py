"""WebSocket 進捗配信ハブ（``/ws/progress``）。

- 接続時に ``{"type": "hello"}`` を送る。
- クライアントからの ``{"type": "ping"}`` に ``{"type": "pong"}`` で応答する。
- ジョブマネージャからのイベントを全接続へブロードキャストする（送信はソケット単位で直列化）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from core.common import utc_now_iso

__all__ = ["WebSocketHub"]

logger = logging.getLogger("caf.backend.ws")


class WebSocketHub:
    """接続中の WebSocket クライアント集合と配信ロジック。"""

    def __init__(self) -> None:
        self._clients: dict[WebSocket, asyncio.Lock] = {}
        self.sent_messages = 0

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def _send(self, websocket: WebSocket, message: dict[str, Any]) -> bool:
        lock = self._clients.get(websocket)
        if lock is None:
            return False
        try:
            async with lock:
                await websocket.send_json(message)
        except (WebSocketDisconnect, RuntimeError, OSError) as exc:
            logger.debug("WebSocket 送信に失敗したため切断扱いにします: %s", exc)
            self._clients.pop(websocket, None)
            return False
        self.sent_messages += 1
        return True

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients[websocket] = asyncio.Lock()
        await self._send(
            websocket,
            {"type": "hello", "clients": len(self._clients), "server_time": utc_now_iso()},
        )

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.pop(websocket, None)

    async def broadcast(self, message: dict[str, Any]) -> int:
        """全クライアントへ送信し、配信できた件数を返す。"""
        delivered = 0
        for websocket in list(self._clients):
            if await self._send(websocket, message):
                delivered += 1
        return delivered

    async def serve(self, websocket: WebSocket) -> None:
        """1接続分の受信ループ（ping/pong と未知メッセージへのエラー応答）。"""
        await self.connect(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    payload: Any = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {"type": raw.strip()}
                kind = payload.get("type") if isinstance(payload, dict) else None
                if kind == "ping":
                    await self._send(websocket, {"type": "pong", "server_time": utc_now_iso()})
                else:
                    await self._send(
                        websocket,
                        {"type": "error", "detail": f"未知のメッセージ種別です: {kind!r}（ping のみ対応）"},
                    )
        except WebSocketDisconnect:
            logger.debug("WebSocket クライアントが切断しました")
        finally:
            self.disconnect(websocket)
