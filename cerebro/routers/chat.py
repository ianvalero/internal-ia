import time
import json
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from langfuse import propagate_attributes

from cerebro.config.model import registry
from cerebro.models.schema import ChatRequest
from cerebro.services.llm_service import LLMService
from cerebro.services.rag_service import RAGService

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
async def chat(
        request: ChatRequest,
        request_obj: Request,
        x_user_id: str = Header(None),
        x_user_name: str = Header(None),
        x_user_email: str = Header(None)
):
    model = registry.get(name=request.model)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found in registry. Available: {list(registry.all().keys())}"
        )

    langfuse = request_obj.app.state.langfuse
    rag_service: RAGService = request_obj.app.state.rag_service
    llm_service: LLMService = request_obj.app.state.llm_service
    user_id = x_user_id or "anonymous"
    query = request.messages[-1].content if request.messages else ""

    with langfuse.start_as_current_observation(as_type="span", name="chat-request") as root:
        with propagate_attributes(
            user_id=user_id,
            session_id=request.session_id,
            metadata={
                "user_name": x_user_name,
                "user_email": x_user_email,
            },
            tags=[model.collection, x_user_name]
        ):
            root.update(
                tags=["env:production", f"model:{model.llm.model}", f"collection:{model.collection}"]
            )

            with langfuse.start_as_current_observation(as_type="span", name="rag-retrieval") as rag_span:
                nodes = await rag_service.retrieve(model_name=request.model, query=query)
                rag_span.update(
                    input={"query": query},
                    output={
                        "chunks": len(nodes),
                        "sources": [
                            {
                                "file_name": node.metadata.get("file_name"),
                                "score": node.score,
                                "file_path": node.metadata.get("file_path"),
                                "last_modified_date": node.metadata.get("last_modified_date")
                            }
                            for node in nodes
                        ]
                    }
                )

            with langfuse.start_as_current_observation(as_type="span", name="llm-generation") as llm_span:
                response_text = await llm_service.generate(
                    model_name=request.model,
                    query=query,
                    nodes=nodes,
                    history=""
                )
                llm_span.update(
                    input={"query": query, "chunks_used": len(nodes), "model_name": model.llm.model},
                    output={"response": response_text}
                )

            root.set_trace_io(
                input={"query": query},
                output={
                    "response": str(response_text),
                    "chunks_used": len(nodes),
                    "collection": model.collection,
                    "model_name": model.llm.model
                }
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