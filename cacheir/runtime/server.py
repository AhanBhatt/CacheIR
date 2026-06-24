from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from cacheir.runtime.cpu import Runtime


def create_app(artifact_path: str | Path):
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Serving requires optional dependencies: pip install cacheir[server]") from exc

    runtime = Runtime(artifact_path)
    app = FastAPI(title="CacheIR local server")

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "cacheir-local", "object": "model"}]}

    @app.post("/v1/completions")
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", ""))
        max_tokens = int(payload.get("max_tokens", 16))
        text = "".join(runtime.generate(prompt, max_new_tokens=max_tokens))
        prompt_tokens = len(runtime.tokenizer.encode(prompt))
        return {
            "id": "cacheir-cmpl",
            "object": "text_completion",
            "model": "cacheir-local",
            "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens, "total_tokens": prompt_tokens + max_tokens},
        }

    @app.post("/v1/chat/completions")
    def chat(payload: dict[str, Any]):
        messages = payload.get("messages", [])
        prompt = "\n".join(str(item.get("content", "")) for item in messages)
        max_tokens = int(payload.get("max_tokens", 16))
        stream = bool(payload.get("stream", False))
        if not stream:
            text = "".join(runtime.generate(prompt, max_new_tokens=max_tokens))
            prompt_tokens = len(runtime.tokenizer.encode(prompt))
            return {
                "id": "cacheir-chat",
                "object": "chat.completion",
                "model": "cacheir-local",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens, "total_tokens": prompt_tokens + max_tokens},
            }

        def events():
            for token in runtime.generate(prompt, max_new_tokens=max_tokens):
                chunk = {
                    "id": "cacheir-chat",
                    "object": "chat.completion.chunk",
                    "model": "cacheir-local",
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done = {
                "id": "cacheir-chat",
                "object": "chat.completion.chunk",
                "model": "cacheir-local",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
