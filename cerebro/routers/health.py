from fastapi import APIRouter
import httpx
from cerebro.services.rag_service import _qdrant_aclient
from cerebro.config.settings import settings

router = APIRouter()

@router.get("/health")
async def health():
    status = {"Qdrant": "error", "vllm": "error"}

    try:
        await _qdrant_aclient.get_collections()
        status["Qdrant"] = "ok"
    except Exception as err:
        print(f"Qdrant error: {err}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.vllm_url}/models", timeout=2.0)
            if response.status_code == 200:
                status["vllm"] = "ok"
    except Exception as err:
        print(f"vLLM error: {err}")

    overall_status = "ok" if all(service == "ok" for service in status.values()) else "degraded"
    return {"status": overall_status, "services": status}