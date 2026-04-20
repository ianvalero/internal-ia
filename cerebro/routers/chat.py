from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse

from cerebro.config.model import registry
from cerebro.models.schema import ChatRequest
from cerebro.services.llm_service import LLMService
from cerebro.services.rag_service import RAGService
from cerebro.services.chat_service import handle_chat, ChatContext
from cerebro.services.title_service import handle_title, TitleContext
from cerebro.utils.streaming import stream_response

router = APIRouter(prefix="/v1")

@router.post("/chat/completions", tags=["chat"], response_model=dict, summary="Chat completions")
async def chat(
        request: ChatRequest,
        request_obj: Request,
        x_user_id: str = Header(None),
        x_user_name: str = Header(None),
        x_user_email: str = Header(None),
        x_conversation_id: str = Header(None)
):
    model = registry.get(name=request.model)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found in registry. Available: {list(registry.all().keys())}"
        )

    print(dict(request_obj.headers))
    print(x_conversation_id)

    langfuse = request_obj.app.state.langfuse
    rag_service: RAGService = request_obj.app.state.rag_service
    llm_service: LLMService = request_obj.app.state.llm_service
    user_id = x_user_id or "anonymous"
    query = request.messages[-1].content if request.messages else ""

    is_title_request = "Provide a concise title" in query or "5-word-or-less title" in query
    if is_title_request:
        title_context = TitleContext(
            query=query,
            model_name=request.model,
            model=model,
            user_id=user_id
        )
        response_text = await handle_title(title_context, llm_service, langfuse)
    else:
        history = request.messages[:-1]
        chat_context = ChatContext(
            query=query,
            history=history,
            model_name=request.model,
            model=model,
            user_id=user_id,
            user_name=x_user_name,
            user_email=x_user_email,
            session_id=x_conversation_id
        )
        response_text = await handle_chat(chat_context, rag_service, llm_service, langfuse)

    # LibreChat siempre espera streaming
    return StreamingResponse(
        stream_response(response_text, model.llm.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # importante para Nginx
        }
    )