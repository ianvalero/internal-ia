import os
from pydantic import BaseModel, model_validator

class LLMConfig(BaseModel):
    provider: str
    base_url: str
    model: str
    temperature: float
    timeout: int = 60
    max_retries: int = 1

    api_key_env: str | None = None
    api_key: str = "EMPTY"

    @model_validator(mode="after")
    def resolve_api_key(self):
        if self.api_key_env:
            resolved = os.getenv(self.api_key_env)
            if not resolved:
                raise ValueError(f"Missing env var: {self.api_key_env}")
            self.api_key = resolved
        return self

class Model(BaseModel):
    name: str
    collection: str
    context_window: int
    chat_template: str = "default"
    llm: LLMConfig

class ModelRegistry:
    def __init__(self):
        self._models: dict[str, Model] = {}

    def load(self, models: dict[str, Model]) -> None:
        self._models = models

    def get(self, name: str) -> Model:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]

    def all(self) -> dict[str, Model]:
        return self._models


registry = ModelRegistry()