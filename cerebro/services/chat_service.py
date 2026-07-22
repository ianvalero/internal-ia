from dataclasses import dataclass
from langfuse import propagate_attributes
from collections.abc import AsyncGenerator

from cerebro.config.model import Model
from cerebro.services.rag_service import RAGService
from cerebro.services.llm_service import LLMService


@dataclass
class ChatContext:
    query: str
    history: list
    model_name: str
    model: Model
    user_id: str
    user_name: str
    user_email: str
    session_id: str

async def handle_chat_stream(
    chat_context: ChatContext,
    rag_service: RAGService,
    llm_service: LLMService,
    langfuse,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    with langfuse.start_as_current_observation(as_type="span", name="chat-request") as root:
        with propagate_attributes(
            user_id=chat_context.user_id,
            session_id=chat_context.session_id,
            metadata={
                "user_name": chat_context.user_name,
                "user_email": chat_context.user_email,
            },
            tags=[
                chat_context.model.collection,
                chat_context.user_name,
            ],
        ):
            with langfuse.start_as_current_observation(as_type="span", name="rag-retrieval") as rag_span:
                nodes = await rag_service.retrieve(
                    model_name=chat_context.model_name,
                    query=chat_context.query,
                )

                rag_span.update(
                    input={"query": chat_context.query},
                    output={
                        "chunks": len(nodes),
                        "sources": [
                            {
                                "file_name": node.metadata.get("file_name"),
                                "score": node.score,
                                "file_path": node.metadata.get("file_path"),
                                "last_modified_date": node.metadata.get(
                                    "last_modified_date"
                                ),
                            }
                            for node in nodes
                        ],
                    },
                )

            chunks: list[str] = []

            with langfuse.start_as_current_observation(
                as_type="generation",
                name="llm-generation",
                model=chat_context.model.llm.model
            ) as generation:
                async for token in llm_service.stream_generate(
                    model_name=chat_context.model_name,
                    query=chat_context.query,
                    nodes=nodes,
                    history=chat_context.history,
                    max_tokens=max_tokens,
                ):
                    chunks.append(token)
                    yield token

                response_text = "".join(chunks)

                generation.update(
                    input={
                        "query": chat_context.query,
                        "chunks_used": len(nodes),
                    },
                    output=response_text,
                )

            root.set_trace_io(
                input={"query": chat_context.query},
                output={
                    "response": response_text,
                    "chunks_used": len(nodes),
                    "collection": chat_context.model.collection,
                    "model_name": chat_context.model.llm.model,
                },
            )