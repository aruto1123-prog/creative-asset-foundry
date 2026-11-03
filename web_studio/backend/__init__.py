"""Creative Asset Foundry Web Studio バックエンド（FastAPI + WebSocket + SQLite）。

``create_app`` は ``web_studio.backend.app`` から提供する（重い依存の読み込みは import 時に行う）。
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def create_app(*args: Any, **kwargs: Any) -> Any:
    """``web_studio.backend.app.create_app`` への委譲（パッケージ import を軽量に保つ）。"""
    from web_studio.backend.app import create_app as _create_app

    return _create_app(*args, **kwargs)
