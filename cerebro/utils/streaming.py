import json
import time
import uuid
from collections.abc import AsyncIterator

async def stream_llm_response(
    token_stream: AsyncIterator[str],
    model: str,
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    initial = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "",
                },
                "finish_reason": None,
            }
        ],
    }

    yield f"data: {json.dumps(initial, ensure_ascii=False)}\n\n"

    async for token in token_stream:
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": token,
                    },
                    "finish_reason": None,
                }
            ],
        }

        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }

    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"