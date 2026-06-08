from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    proxy_url: str | None = None
    qdrant_url: str
    langfuse_url: str
    embedding_model_name: str
    embedding_base_url: str

    class Config:
        env_file = "cerebro/.env"
        env_file_encoding = "utf-8"


settings = Settings()