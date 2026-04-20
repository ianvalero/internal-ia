from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from langfuse import get_client

from cerebro.config.loader import load_models
from cerebro.config.model import registry
from cerebro.routers import chat, health, model
from cerebro.services.rag_service import RAGService
from cerebro.services.llm_service import LLMService

@asynccontextmanager
async def lifespan(app: FastAPI):
    models = load_models()
    registry.load(models)
    print(f"Loaded models: {list(models.keys())}")

    LlamaIndexInstrumentor().instrument()
    app.state.langfuse = get_client()

    app.state.llm_service = LLMService()
    app.state.rag_service = RAGService()

    yield
    # Cerrar servicios al finalizar la aplicación
    await app.state.rag_service.close()
app = FastAPI(
    title="RAG Middleware API",
    description="API for RAG Middleware",
    version="0.1.0",
    lifespan=lifespan,
)

# TODO Ahora permitimos todo, cen producción hay que dejar acceso solo al frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

LlamaIndexInstrumentor().instrument()
langfuse = get_client()
app.include_router(chat.router)
app.include_router(health.router)
app.include_router(model.router)