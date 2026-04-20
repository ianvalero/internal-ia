import time
from fastapi import APIRouter, HTTPException

from cerebro.config.model import registry

router = APIRouter(prefix="/v1")

@router.get("/models", tags=["model"], response_model=dict, summary="Models list")
async def list_models():
    models = registry.all()

    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "name": model.name,
                "collection": model.collection,
                "provider": model.llm.provider,
                "context_window": model.context_window,
                "temperature": model.llm.temperature,
                "chat_template": model.chat_template,
                "created": int(time.time()),
                "owned_by": "cerebro"
            }
            for name, model in models.items()
        ]
    }

@router.get("/models/{model_name}", tags=["model"], response_model=dict, summary="Model data")
async def get_model(model_name: str):
    try:
        model = registry.get(name=model_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return {
        "object": "model",
        "data": {
            "id": model_name,
            "object": "model",
            "name": model.name,
            "collection": model.collection,
            "provider": model.llm.provider,
            "context_window": model.context_window,
            "temperature": model.llm.temperature,
            "chat_template": model.chat_template,
            "created": int(time.time()),
            "owned_by": "cerebro"
        }
    }
