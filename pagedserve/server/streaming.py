"""Server-Sent Events (SSE) formatting utilities for streaming generation."""

import json
from pagedserve.engine.engine import StreamEvent


def format_sse_event(data: dict, event: str = "data") -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def format_sse_done() -> str:
    """SSE stream termination sentinel per OpenAI convention."""
    return "data: [DONE]\n\n"


def stream_event_to_completion_chunk(event: StreamEvent, model: str, request_id: str) -> dict:
    """Convert a StreamEvent into an OpenAI-compatible completion chunk."""
    return {
        "id": request_id,
        "object": "text_completion",
        "model": model,
        "choices": [
            {
                "text": event.text_delta,
                "index": 0,
                "finish_reason": event.finish_reason,
            }
        ],
    }


def stream_event_to_chat_chunk(event: StreamEvent, model: str, request_id: str) -> dict:
    """Convert a StreamEvent into an OpenAI-compatible chat completion chunk."""
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": event.text_delta},
                "finish_reason": event.finish_reason,
            }
        ],
    }
