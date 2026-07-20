from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models import CollectionsResponse

from cerebro.config.settings import settings
from cerebro.config.model import registry


class RAGService:
    def __init__(self):
        self._qdrant_client = QdrantClient(url=settings.qdrant_url)
        self._qdrant_aclient = AsyncQdrantClient(url=settings.qdrant_url)

        self._embedding = OpenAIEmbedding(
            model_name=settings.embedding_model_name,
            api_base=settings.embedding_base_url,
            api_key="local",
            embed_batch_size=32,
        )

        self._index_cache: dict = {}

    async def retrieve(self, model_name: str, query: str, min_score: float = 0.5) -> list:
        index = self._get_index(model_name=model_name)
        retriever = index.as_retriever(similarity_top_k=10)

        nodes = await retriever.aretrieve(query)
        return [node for node in nodes if node.score >= min_score]

    async def get_collections(self) -> CollectionsResponse:
        return await self._qdrant_aclient.get_collections()

    async def close(self):
        await self._qdrant_aclient.close()
        self._qdrant_client.close()

    def _get_index(self, model_name: str) -> VectorStoreIndex:
        if model_name not in self._index_cache:
            model = registry.get(model_name)
            vector_store = QdrantVectorStore(
                client=self._qdrant_client,
                aclient=self._qdrant_aclient,
                collection_name=model.collection,
            )

            self._index_cache[model_name] = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self._embedding,
                use_async=True,
            )

        return self._index_cache[model_name]

