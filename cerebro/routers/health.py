from fastapi import APIRouter, Request
import asyncio

from cerebro.config.settings import settings
from cerebro.config.model import registry
from cerebro.services.health_service import HealthService

router = APIRouter()

@router.get("/health")
async def health(request_obj: Request):
    models = registry.all()
    rag_service = request_obj.app.state.rag_service

    qdrant_task = HealthService.check_qdrant(rag_service)
    langfuse_task = HealthService.check_langfuse(settings.langfuse_url)
    llm_task = HealthService.check_llm_models(models)

    qdrant_status, langfuse_status, llm_status = await asyncio.gather(
        qdrant_task,
        langfuse_task,
        llm_task
    )

    status = {
        "Qdrant": qdrant_status,
        "Langfuse": langfuse_status,
        "llm_models": llm_status
    }

    llm_ok = all(v == "ok" for v in llm_status.values())

    if qdrant_status != "ok":
        overall = "down"
    elif not llm_ok or langfuse_status != "ok":
        overall = "degraded"
    else:
        overall = "ok"

    return {
        "status": overall,
        "services": status
    }