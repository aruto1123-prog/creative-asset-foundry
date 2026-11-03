"""クラウドテスト用の模擬 ComfyUI サーバー。

実 ComfyUI と同じ HTTP / WebSocket プロトコル（``/prompt``・``/history``・``/view``・``/object_info``・
``/system_stats``・``/interrupt``・``/queue``・``/ws``）を FastAPI で再現する。

- 投入時にワークフローを構造検証（``validate_prompt``）し、不正なら 400 と ``node_errors`` を返す。
- 実行はトポロジカル順に ``executing`` を送出し、KSampler ではステップ数ぶん ``progress`` を送る。
- SaveImage / PreviewImage では zlib による小さな有効 PNG（グレースケールの決定論的グラデーション）を生成し、
  ``executed`` と履歴に出力を記録、``/view`` で返す。
- ``MockComfyServer`` は空きポートで uvicorn をバックグラウンドスレッド起動する同期コンテキストマネージャ。
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import struct
import threading
import time
import uuid
import zlib
from typing import Any

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from core.common import stable_seed
from core.style_alchemy.comfy_export import NODE_OUTPUTS, ComfyGraphBuilder, validate_prompt

__all__ = ["make_png", "create_mock_comfy_app", "MockComfyServer"]


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def make_png(width: int, height: int, seed: int = 0) -> bytes:
    """8bit グレースケールの有効な PNG を生成する（seed で模様が変わる）。"""
    width = max(1, int(width))
    height = max(1, int(height))
    phase = seed % 97
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            value = (x * 255 // max(1, width - 1) + y * 128 // max(1, height - 1) + phase * 3) % 256
            if ((x + phase) // 8 + y // 8) % 2 == 0:
                value = 255 - value
            rows.append(value)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + _png_chunk(b"IEND", b"")


def _topological_order(prompt: dict[str, Any]) -> list[str]:
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(node_id: str) -> None:
        if state.get(node_id) == 2:
            return
        state[node_id] = 1
        for value in prompt[node_id].get("inputs", {}).values():
            if ComfyGraphBuilder.is_link(value) and value[0] in prompt and state.get(value[0]) != 1:
                visit(value[0])
        state[node_id] = 2
        order.append(node_id)

    for node_id in sorted(prompt, key=lambda k: (len(k), k)):
        visit(node_id)
    return order


class _MockState:
    def __init__(self, step_delay_s: float) -> None:
        self.step_delay_s = float(step_delay_s)
        self.history: dict[str, dict[str, Any]] = {}
        self.images: dict[tuple[str, str], bytes] = {}
        self.sockets: dict[str, list[WebSocket]] = {}
        self.counter = 0
        self.queue_running: list[str] = []
        self.interrupted: set[str] = set()
        self.lock = asyncio.Lock()

    async def send(self, client_id: str, message: dict[str, Any]) -> None:
        for websocket in list(self.sockets.get(client_id, [])):
            try:
                await websocket.send_json(message)
            except (RuntimeError, WebSocketDisconnect):
                self.sockets.get(client_id, []).remove(websocket)


def _latent_size(prompt: dict[str, Any]) -> tuple[int, int]:
    for node in prompt.values():
        if node.get("class_type") in ("EmptyLatentImage", "EmptySD3LatentImage"):
            inputs = node.get("inputs", {})
            return int(inputs.get("width", 512)), int(inputs.get("height", 512))
    return 512, 512


def create_mock_comfy_app(step_delay_s: float = 0.0) -> FastAPI:
    """模擬 ComfyUI の FastAPI アプリを生成する。"""
    app = FastAPI(title="Mock ComfyUI (Creative Asset Foundry)")
    state = _MockState(step_delay_s)
    app.state.mock = state

    async def execute(prompt_id: str, prompt: dict[str, Any], client_id: str, number: int) -> None:
        await asyncio.sleep(0.05)
        started = time.time()
        await state.send(client_id, {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 1}}}})
        await state.send(client_id, {"type": "execution_start", "data": {"prompt_id": prompt_id, "timestamp": int(started * 1000)}})
        await state.send(client_id, {"type": "execution_cached", "data": {"nodes": [], "prompt_id": prompt_id, "timestamp": int(started * 1000)}})
        outputs: dict[str, Any] = {}
        width, height = _latent_size(prompt)
        seed = 0
        for node_id in _topological_order(prompt):
            if prompt_id in state.interrupted:
                await state.send(client_id, {"type": "execution_interrupted", "data": {"prompt_id": prompt_id, "node_id": node_id}})
                state.history[prompt_id] = {
                    "prompt": [number, prompt_id, prompt, {}, []],
                    "outputs": outputs,
                    "status": {"status_str": "error", "completed": True, "messages": [["execution_interrupted", {}]]},
                }
                return
            node = prompt[node_id]
            class_type = node.get("class_type")
            await state.send(client_id, {"type": "executing", "data": {"node": node_id, "display_node": node_id, "prompt_id": prompt_id}})
            if class_type in ("KSampler", "KSamplerAdvanced"):
                seed = int(node.get("inputs", {}).get("seed", 0))
                steps = max(1, min(200, int(node.get("inputs", {}).get("steps", 20))))
                for step in range(1, steps + 1):
                    await state.send(
                        client_id,
                        {"type": "progress", "data": {"value": step, "max": steps, "prompt_id": prompt_id, "node": node_id}},
                    )
                    if state.step_delay_s:
                        await asyncio.sleep(state.step_delay_s)
            if class_type in ("SaveImage", "PreviewImage"):
                prefix = str(node.get("inputs", {}).get("filename_prefix", "ComfyUI"))
                subfolder, _, stem = prefix.rpartition("/")
                filename = f"{stem or 'ComfyUI'}_{number:05d}_.png"
                image_type = "output" if class_type == "SaveImage" else "temp"
                scale = max(1, max(width, height) // 64)
                data = make_png(max(8, width // scale), max(8, height // scale), seed=stable_seed(prompt_id, seed))
                state.images[(subfolder, filename)] = data
                image_ref = {"filename": filename, "subfolder": subfolder, "type": image_type}
                outputs[node_id] = {"images": [image_ref]}
                await state.send(
                    client_id,
                    {"type": "executed", "data": {"node": node_id, "display_node": node_id, "output": {"images": [image_ref]}, "prompt_id": prompt_id}},
                )
        state.history[prompt_id] = {
            "prompt": [number, prompt_id, prompt, {"client_id": client_id}, list(outputs)],
            "outputs": outputs,
            "status": {"status_str": "success", "completed": True, "messages": [["execution_success", {"prompt_id": prompt_id}]]},
        }
        await state.send(client_id, {"type": "execution_success", "data": {"prompt_id": prompt_id, "timestamp": int(time.time() * 1000)}})
        await state.send(client_id, {"type": "executing", "data": {"node": None, "prompt_id": prompt_id}})
        await state.send(client_id, {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}}})
        if prompt_id in state.queue_running:
            state.queue_running.remove(prompt_id)

    @app.post("/prompt")
    async def post_prompt(request: Request) -> JSONResponse:
        body = await request.json()
        prompt = body.get("prompt") if isinstance(body, dict) else None
        if not isinstance(prompt, dict) or not prompt:
            return JSONResponse(
                {"error": {"type": "invalid_prompt", "message": "prompt がありません", "details": "", "extra_info": {}}, "node_errors": {}},
                status_code=400,
            )
        try:
            problems = validate_prompt(prompt)
        except ValueError as exc:
            problems = [str(exc)]
        if problems:
            return JSONResponse(
                {
                    "error": {"type": "prompt_outputs_failed_validation", "message": "; ".join(problems), "details": "", "extra_info": {}},
                    "node_errors": {"graph": {"errors": problems}},
                },
                status_code=400,
            )
        client_id = str(body.get("client_id") or "anonymous")
        async with state.lock:
            state.counter += 1
            number = state.counter
        prompt_id = uuid.uuid4().hex
        state.queue_running.append(prompt_id)
        asyncio.get_running_loop().create_task(execute(prompt_id, prompt, client_id, number))
        return JSONResponse({"prompt_id": prompt_id, "number": number, "node_errors": {}})

    @app.get("/history/{prompt_id}")
    async def get_history(prompt_id: str) -> JSONResponse:
        entry = state.history.get(prompt_id)
        return JSONResponse({prompt_id: entry} if entry else {})

    @app.get("/history")
    async def get_all_history() -> JSONResponse:
        return JSONResponse(state.history)

    @app.get("/view")
    async def view(filename: str = Query(...), subfolder: str = Query(""), type: str = Query("output")) -> Response:  # noqa: A002
        data = state.images.get((subfolder, filename))
        if data is None:
            return Response(status_code=404, content=b"not found")
        return Response(content=data, media_type="image/png")

    def _node_spec(class_type: str) -> dict[str, Any]:
        outputs = NODE_OUTPUTS.get(class_type, [])
        return {
            "input": {"required": {}, "optional": {}},
            "output": list(outputs),
            "output_name": list(outputs),
            "output_is_list": [False] * len(outputs),
            "name": class_type,
            "display_name": class_type,
            "category": "mock",
            "output_node": class_type in ("SaveImage", "PreviewImage"),
        }

    @app.get("/object_info")
    async def object_info() -> JSONResponse:
        return JSONResponse({name: _node_spec(name) for name in NODE_OUTPUTS})

    @app.get("/object_info/{node_class}")
    async def object_info_one(node_class: str) -> JSONResponse:
        if node_class not in NODE_OUTPUTS:
            return JSONResponse({})
        return JSONResponse({node_class: _node_spec(node_class)})

    @app.get("/system_stats")
    async def system_stats() -> JSONResponse:
        return JSONResponse(
            {
                "system": {"os": "mock", "comfyui_version": "mock-1.0", "python_version": "mock", "embedded_python": False},
                "devices": [{"name": "mock-gpu", "type": "cuda", "index": 0, "vram_total": 24 * 1024**3, "vram_free": 20 * 1024**3}],
            }
        )

    @app.get("/queue")
    async def queue() -> JSONResponse:
        return JSONResponse({"queue_running": [[0, pid] for pid in state.queue_running], "queue_pending": []})

    @app.post("/interrupt")
    async def interrupt() -> JSONResponse:
        state.interrupted.update(state.queue_running)
        return JSONResponse({"interrupted": list(state.queue_running)})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, clientId: str = Query("anonymous")) -> None:  # noqa: N803 - ComfyUI のパラメータ名
        await websocket.accept()
        state.sockets.setdefault(clientId, []).append(websocket)
        await websocket.send_json({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": len(state.queue_running)}}, "sid": clientId}})
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            sockets = state.sockets.get(clientId, [])
            if websocket in sockets:
                sockets.remove(websocket)

    return app


class MockComfyServer:
    """模擬 ComfyUI を空きポートの uvicorn スレッドで起動する同期コンテキストマネージャ。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, step_delay_s: float = 0.0, startup_timeout_s: float = 15.0) -> None:
        self.host = host
        self.requested_port = port
        self.step_delay_s = step_delay_s
        self.startup_timeout_s = startup_timeout_s
        self.port: int | None = None
        self.app = create_mock_comfy_app(step_delay_s=step_delay_s)
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise RuntimeError("MockComfyServer はまだ起動していません")
        return f"http://{self.host}:{self.port}"

    def start(self) -> "MockComfyServer":
        import uvicorn

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.requested_port))
        sock.listen(64)
        self._socket = sock
        self.port = int(sock.getsockname()[1])
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", lifespan="off", ws="auto")
        server = uvicorn.Server(config)
        self._server = server
        self._thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name="mock-comfyui", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while not server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                self.stop()
                raise RuntimeError("MockComfyServer の起動に失敗しました")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.close()
        self._server = None
        self._thread = None
        self._socket = None

    def __enter__(self) -> "MockComfyServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    async def __aenter__(self) -> "MockComfyServer":
        return await asyncio.to_thread(self.start)

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self.stop)
