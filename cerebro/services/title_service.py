from dataclasses import dataclass
from langfuse import propagate_attributes

from cerebro.config.model import Model
from cerebro.services.llm_service import LLMService


@dataclass
class TitleContext:
    query: str
    model_name: str
    model: Model
    user_id: str
    
async def handle_title(title_context: TitleContext, llm_service: LLMService, langfuse) -> str:
    with langfuse.start_as_current_observation(as_type="span", name="title-request") as root:
        with propagate_attributes(
            user_id=title_context.user_id,
            tags=["LibreChat", "Generate title"]
        ):

            with langfuse.start_as_current_observation(as_type="span", name="llm-generation") as llm_span:
                response_text = await llm_service.generate(
                    model_name=title_context.model_name,
                    query=title_context.query,
                )
                llm_span.update(
                    input={
                        "query": title_context.query,
                        "model_name": title_context.model.llm.model
                    },
                    output={"response": response_text}
                )

            root.set_trace_io(
                input={"query": title_context.query},
                output={
                    "response": str(response_text),
                    "model_name": title_context.model.llm.model
                }
            )

    return str(response_text)