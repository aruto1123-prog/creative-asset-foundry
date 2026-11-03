"""ジョブ参照 API。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from web_studio.backend.deps import get_services, not_found
from web_studio.backend.services import FoundryServices

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(
    limit: int = Query(50, ge=1, le=500),
    services: FoundryServices = Depends(get_services),
) -> dict[str, Any]:
    records = services.jobs.list(limit)
    return {"jobs": [record.model_dump(mode="json") for record in records], "running": services.jobs.running_count()}


@router.get("/{job_id}")
def get_job(job_id: str, services: FoundryServices = Depends(get_services)) -> dict[str, Any]:
    record = services.jobs.get(job_id)
    if record is None:
        raise not_found("ジョブ", job_id)
    return record.model_dump(mode="json")
