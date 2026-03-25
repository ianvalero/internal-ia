from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    proxy_url: str | None = None
    qdrant_url: str
    vllm_url: str
    vllm_api_key: str
    embedding_model_path: str

    class Config:
        env_file = "cerebro/.env"
        env_file_encoding = "utf-8"


settings = Settings()