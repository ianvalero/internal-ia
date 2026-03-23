from pydantic import BaseModel

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
    temperature: float = 0.7
    max_tokens: int = 1024