"""Web Studio REST ルーター群。"""

from __future__ import annotations

from web_studio.backend.routers import diagnostics, jobs, manga, publisher, structures, style, system

ALL_ROUTERS = (
    system.router,
    jobs.router,
    structures.router,
    style.router,
    manga.router,
    diagnostics.router,
    publisher.router,
)

__all__ = ["ALL_ROUTERS"]
