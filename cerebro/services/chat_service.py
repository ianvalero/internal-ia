from dataclasses import dataclass
from langfuse import propagate_attributes

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

async def handle_chat(chat_context: ChatContext, rag_service: RAGService, llm_service: LLMService, langfuse) -> str:
    with langfuse.start_as_current_observation(as_type="span", name="chat-request") as root:
        with propagate_attributes(
            user_id=chat_context.user_id,
            session_id=chat_context.session_id,
            metadata={
                "user_name": chat_context.user_name,
                "user_email": chat_context.user_email,
            },
            tags=[chat_context.model.collection, chat_context.user_name]
        ):

            with langfuse.start_as_current_observation(as_type="span", name="rag-retrieval") as rag_span:
                nodes = await rag_service.retrieve(model_name=chat_context.model_name, query=chat_context.query)
                rag_span.update(
                    input={"query": chat_context.query},
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
                    model_name=chat_context.model_name,
                    query=chat_context.query,
                    nodes=nodes,
                    history=chat_context.history
                )
                llm_span.update(
                    input={
                        "query": chat_context.query,
                        "chunks_used": len(nodes),
                        "model_name": chat_context.model.llm.model
                    },
                    output={"response": response_text}
                )

            root.set_trace_io(
                input={"query": chat_context.query},
                output={
                    "response": str(response_text),
                    "chunks_used": len(nodes),
                    "collection": chat_context.model.collection,
                    "model_name": chat_context.model.llm.model
                }
            )

    return str(response_text)