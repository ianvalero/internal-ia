from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from cerebro.routers import chat, health
from cerebro.services.rag_service import _qdrant_aclient, _qdrant_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Al apagar la app cierra las conexiones limpiamente
    await _qdrant_aclient.close()
    _qdrant_client.close()
app = FastAPI(
    title="RAG Middleware API",
    description="API for RAG Middleware",
    version="0.1.0",
)

# TODO Ahora permitimos todo, cen producción hay que dejar acceso solo al frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(chat.router)
app.include_router(health.router)