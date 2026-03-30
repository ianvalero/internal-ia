from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate

from cerebro.config.model import registry

class LLMService:
    def __init__(self):
        self._llm_cache: dict = {}

    async def generate(self, model_name: str, query: str, nodes: list = None, history: str = "") -> str:
        if nodes is None:
            nodes = []

        context = "\n\n".join([node.get_content() for node in nodes])

        template = self._get_qa_template()
        prompt = template.format(
            context_str=context,
            history=history,
            query_str=query
        )

        llm = self._get_llm(model_name)
        response = await llm.acomplete(prompt)
        return str(response)

    def _get_llm(self, model_name: str):
        if model_name not in self._llm_cache:
            model = registry.get(name=model_name)

            self._llm_cache[model_name] = OpenAILike(
            model=model.llm.model,
            api_base=model.llm.base_url,
            api_key=model.llm.api_key,
            context_window=model.context_window,
            is_chat_model=True,
            timeout=model.llm.timeout,
            max_retries=model.llm.max_retries,
            temperature=model.llm.temperature
        )
        return self._llm_cache[model_name]

    def _get_qa_template(self):
        return PromptTemplate(
            "Usa el siguiente contexto para responder la pregunta.\n"
            "Recuerda el historial de la conversación si es relevante.\n\n"
            "Contexto:\n{context_str}\n\n"
            "Historial:\n{history}\n\n"
            "Pregunta: {query_str}\n"
            "Respuesta:"
        )