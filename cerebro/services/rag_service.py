from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.llms.openai_like import OpenAILike

import asyncio
from qdrant_client import AsyncQdrantClient, QdrantClient
from cerebro.config.settings import settings

CONFIGS = {
    "cuentos-tinyllama": {
        "collection": "cuentos",
        "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "context_window": 2048,
    }
}

QA_TEMPLATE = PromptTemplate(
"""Usa el siguiente contexto para responder la pregunta.
Recuerda el historial de la conversación si es relevante.

Contexto:
{context_str}

Historial:
{history}

Pregunta: {query_str}
Respuesta:"""
)

_qdrant_client = QdrantClient(url=settings.qdrant_url)
_qdrant_aclient = AsyncQdrantClient(url=settings.qdrant_url)

_embedding = HuggingFaceEmbedding(
    model_name=settings.embedding_model_path,
    cache_folder="/root/.cache/huggingface/hub",
    local_files_only=True,
    device="cpu",
)

_llm_cache: dict = {}
_engine_cache: dict = {}

async def get_rag_response(model_name: str, messages: list) -> str:
    if model_name not in CONFIGS:
        return f"LLM model '{model_name}' not found. Available models: {list(CONFIGS.keys())}"

    config = CONFIGS[model_name]
    llm = _get_llm(model_name)
    user_query = _extract_text(messages[-1].content)
    history = _build_history(messages)

    print(f"[1] Construyendo query engine para colección: {config['collection']}")
    query_engine = await asyncio.to_thread(
        _get_query_engine, config["collection"], llm
    )

    retriever = query_engine.retriever
    nodes = await retriever.aretrieve(user_query)

    print(f"\n{'=' * 50}")
    print(f"Query: {user_query}")
    print(f"Nodos recuperados de Qdrant: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"\n--- Chunk {i + 1} | Score: {node.score:.4f} ---")
        print(f"Fichero: {node.metadata.get('file_name', 'desconocido')}")
        print(f"Texto: {node.text[:200]}...")
    print(f"{'=' * 50}\n")


    print("[2] Query engine listo, lanzando query...")
    response = await query_engine.aquery(
        f"{user_query}\n\n[Historial previo:\n{history}]"
        if history else user_query
    )

    print(f"[3] Respuesta recibida: {str(response)[:100]}")
    return response.response.strip()

def _get_query_engine(collection: str, llm: OpenAILike):
    cache_key = f"{collection}"
    if cache_key not in _engine_cache:
        vector_store = QdrantVectorStore(
            client=_qdrant_client,
            aclient=_qdrant_aclient,
            collection_name=collection,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=_embedding,
            use_async=True,
        )
        query_engine = index.as_query_engine(
            similarity_top_k=10,
            use_async=True,
            llm=llm,
            response_mode="compact",
            node_postprocessors=[],
        )
        query_engine.update_prompts({
            "response_synthesizer:text_qa_template": QA_TEMPLATE
        })
        _engine_cache[cache_key] = query_engine
    return _engine_cache[cache_key]

def _get_llm(model_name: str) -> OpenAILike:
    if model_name not in _llm_cache:
        config = CONFIGS[model_name]
        _llm_cache[model_name] = OpenAILike(
            model=config["llm_model"],
            api_base=settings.vllm_url,
            api_key=settings.vllm_api_key,
            context_window=config["context_window"],
            is_chat_model=True,
            timeout=300,
            max_retries=2,
        )
    return _llm_cache[model_name]

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content

    return " ".join(
        part.text for part in content
        if part.type == "text" and part.text
    )

def _build_history(messages: list) -> str:
    lines = []
    for m in messages[:-1]:
        role = "Usuario" if m.role == "user" else "Asistente"
        lines.append(f"{role}: {_extract_text(m.content)}")

    return "\n".join(lines)