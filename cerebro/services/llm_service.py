from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate

from cerebro.config.model import registry

class LLMService:
    def __init__(self):
        self._llm_cache: dict = {}

    async def generate(self, model_name: str, query: str, nodes: list = None, history: list = None) -> str:
        if nodes is None:
            nodes = []
        if history is None:
            history = []

        context = "\n\n".join([node.get_content() for node in nodes])

        history_text = "\n".join([
            f"{'user' if msg.role == 'user' else 'assistant'}: {msg.content}"
            for msg in history
        ]) if history else ""

        template = self._get_qa_template(chat_template=registry.get(model_name).chat_template)
        prompt = template.format(
            context_str=context,
            history_str=history_text,
            query_str=query
        )

        llm = self._get_llm(model_name)
        response = await llm.acomplete(prompt)
        return str(response)

    def _get_llm(self, model_name: str) -> OpenAILike:
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

    def _get_qa_template(self, chat_template: str) -> PromptTemplate:
        templates = {
            "tinyllama": PromptTemplate(
                "<|system|>\nEres un asistente que responde basándose en el contexto.</s>\n"
                "<|user|>\nContexto:\n{context_str}\n\nHistorial:\n{history_str}\n\nPregunta: {query_str}</s>\n"
                "<|assistant|>\n"
            ),
            "qwen": PromptTemplate(
                "<|im_start|>system\nEres un asistente que responde basándose en el contexto.<|im_end|>\n"
                "<|im_start|>user\nContexto:\n{context_str}\n\nHistorial:\n{history_str}\n\nPregunta: {query_str}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "default": PromptTemplate(
                "Contexto:\n{context_str}\n\n"
                "Historial:\n{history_str}\n\n"
                "Pregunta: {query_str}\n"
                "Respuesta:"
            ),
        }
        return templates.get(chat_template, templates["default"])