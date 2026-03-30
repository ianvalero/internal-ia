from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    proxy_url: str | None = None
    qdrant_url: str
    langfuse_url: str
    embedding_model_path: str
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    class Config:
        env_file = "cerebro/.env"
        env_file_encoding = "utf-8"


settings = Settings()