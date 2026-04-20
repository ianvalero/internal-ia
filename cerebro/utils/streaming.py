import time
import json

async def stream_response(response_text: str, model: str):
    """Genera la respuesta en formato SSE (Server-Sent Events) que espera LibreChat."""

    chunk_size = 10
    for i in range(0, len(response_text), chunk_size):
        chunk = response_text[i:i + chunk_size]
        data = {
            "id": "dt-chat-rag",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": chunk
                },
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"

    final = {
        "id": "dt-chat-rag",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"