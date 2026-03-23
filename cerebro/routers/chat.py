import time
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from cerebro.models.schema import ChatRequest
from cerebro.services.rag_service import get_rag_response, CONFIGS

router = APIRouter(prefix="/v1")

async def stream_response(response_text: str, model: str):
    """Genera la respuesta en formato SSE (Server-Sent Events) que espera LibreChat."""

    chunk_size = 10
    for i in range(0, len(response_text), chunk_size):
        chunk = response_text[i:i + chunk_size]
        data = {
            "id": "dt-chat-rag",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": chunk
                },
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"

    final = {
        "id": "dt-chat-rag",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

@router.post("/chat/completions", tags=["chat"], response_model=dict, summary="Chat completions")
async def chat(request: ChatRequest):
    if request.model not in CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available: {list(CONFIGS.keys())}"
        )

    response_text = await get_rag_response(
        model_name=request.model,
        messages=request.messages,
    )

    # LibreChat siempre espera streaming
    return StreamingResponse(
        stream_response(response_text, request.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # importante para Nginx
        }
    )

@router.get("/models", tags=["models"], response_model=dict, summary="List models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": model_name, "object": "model"}
            for model_name in CONFIGS.keys()
        ]
    }