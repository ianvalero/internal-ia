from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse

from cerebro.config.model import registry
from cerebro.models.schema import ChatRequest
from cerebro.services.llm_service import LLMService
from cerebro.services.rag_service import RAGService
from cerebro.services.chat_service import handle_chat_stream, ChatContext
from cerebro.services.title_service import handle_title, TitleContext
from cerebro.utils.streaming import stream_llm_response

router = APIRouter(prefix="/v1")

@router.post("/chat/completions", tags=["chat"])
async def chat(
    request: ChatRequest,
    request_obj: Request,
    x_user_id: str = Header(None),
    x_user_name: str = Header(None),
    x_user_email: str = Header(None),
    x_conversation_id: str = Header(None),
):
    model = registry.get(name=request.model)

    langfuse = request_obj.app.state.langfuse
    rag_service: RAGService = request_obj.app.state.rag_service
    llm_service: LLMService = request_obj.app.state.llm_service

    query = request.messages[-1].content if request.messages else ""
    history = request.messages[:-1]

    chat_context = ChatContext(
        query=query,
        history=history,
        model_name=request.model,
        model=model,
        user_id=x_user_id or "anonymous",
        user_name=x_user_name or "anonymous",
        user_email=x_user_email or "",
        session_id=x_conversation_id or request.session_id or "",
    )

    token_stream = handle_chat_stream(
        chat_context=chat_context,
        rag_service=rag_service,
        llm_service=llm_service,
        langfuse=langfuse,
        max_tokens=min(request.max_tokens, 300),
    )

    return StreamingResponse(
        stream_llm_response(
            token_stream=token_stream,
            model=model.llm.model,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )