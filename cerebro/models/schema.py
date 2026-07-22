from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict | None = None

class Message(BaseModel):
    role: str
    content: str | list[ContentPart]

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    session_id: str | None = None
    user: str | None = None