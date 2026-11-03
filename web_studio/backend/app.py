"""Creative Asset Foundry Web Studio — FastAPI アプリケーションファクトリ。

起動方法（run_studio.py が利用）::

    uvicorn.run("web_studio.backend.app:create_app", factory=True, host="127.0.0.1", port=8765)

引数なしの ``create_app()`` は環境変数（``CAF_DATA_DIR`` / ``CAF_LLM_PROVIDER`` 等）と
``<data_dir>/config.json`` から設定を読み込む。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import __version__
from core.config import REPO_ROOT, FoundrySettings
from web_studio.backend.routers import ALL_ROUTERS
from web_studio.backend.services import FoundryServices

__all__ = ["create_app", "DEFAULT_FRONTEND_DIST", "DEV_ORIGINS"]

logger = logging.getLogger("caf.backend")

DEFAULT_FRONTEND_DIST = REPO_ROOT / "web_studio" / "frontend" / "dist"
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_RESERVED_PREFIXES = ("api", "ws")


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def _key_error(_request: Request, exc: KeyError) -> JSONResponse:
        detail = exc.args[0] if exc.args else "指定されたリソースが見つかりません"
        return JSONResponse(status_code=404, content={"detail": str(detail)})

    @app.exception_handler(FileNotFoundError)
    async def _file_not_found(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})


def _mount_frontend(app: FastAPI, dist: Path) -> bool:
    """ビルド済み SPA を配信する（index.html が無ければ何もしない）。"""
    index = dist / "index.html"
    if not index.is_file():
        return False
    root = dist.resolve()
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", include_in_schema=False)
    async def _spa_index() -> FileResponse:
        return FileResponse(str(index))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str) -> FileResponse:
        head = full_path.split("/", 1)[0]
        if head in _RESERVED_PREFIXES:
            raise HTTPException(status_code=404, detail=f"Not Found: /{full_path}")
        candidate = (root / full_path).resolve()
        if candidate.is_file() and (candidate == root or root in candidate.parents):
            return FileResponse(str(candidate))
        return FileResponse(str(index))

    return True


def create_app(
    settings: FoundrySettings | None = None,
    *,
    services: FoundryServices | None = None,
    frontend_dist: Path | str | None = None,
) -> FastAPI:
    """Web Studio の FastAPI アプリを構築する。"""
    if services is None:
        services = FoundryServices(settings or FoundrySettings.load())
    dist = Path(frontend_dist) if frontend_dist is not None else DEFAULT_FRONTEND_DIST

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(services.initialize)
        try:
            yield
        finally:
            await services.shutdown()

    app = FastAPI(
        title="Creative Asset Foundry Studio API",
        version=__version__,
        description="構造研究・美的選好最適化・漫画制作・知見の資産化を統合するローカルスタジオ API",
        lifespan=lifespan,
    )
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _install_exception_handlers(app)
    for router in ALL_ROUTERS:
        app.include_router(router)

    @app.websocket("/ws/progress")
    async def ws_progress(websocket: WebSocket) -> None:
        await services.hub.serve(websocket)

    app.state.frontend_mounted = _mount_frontend(app, dist)
    return app
