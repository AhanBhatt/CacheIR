from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from cacheir.runtime import create_runtime
from cacheir.runtime.scheduler import ContinuousBatchScheduler, GenerationRequest


def create_app(
    artifact_path: str | Path,
    *,
    max_batch_size: int = 4,
    max_queue_size: int | None = None,
    backend: str = "auto",
):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Serving requires optional dependencies: pip install cacheir[server]") from exc

    runtime = create_runtime(artifact_path, backend=backend)
    scheduler = ContinuousBatchScheduler(runtime, max_batch_size=max_batch_size, use_prefix_cache=True, max_queue_size=max_queue_size)
    started_at = time.time()
    counters = {
        "requests_total": 0,
        "stream_requests_total": 0,
        "tokens_generated_total": 0,
    }
    app = FastAPI(title="CacheIR local server")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": "cacheir-local",
            "uptime_s": time.time() - started_at,
            "target": runtime.artifact.target,
            "quant": runtime.artifact.quant,
            "scheduler": scheduler.stats(),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        stats = scheduler.stats()
        prefix = stats["prefix_cache"]
        assert isinstance(prefix, dict)
        lines = [
            "# HELP cacheir_requests_total Total OpenAI-compatible requests handled.",
            "# TYPE cacheir_requests_total counter",
            f"cacheir_requests_total {counters['requests_total']}",
            "# HELP cacheir_stream_requests_total Streaming requests handled.",
            "# TYPE cacheir_stream_requests_total counter",
            f"cacheir_stream_requests_total {counters['stream_requests_total']}",
            "# HELP cacheir_tokens_generated_total Completion tokens generated.",
            "# TYPE cacheir_tokens_generated_total counter",
            f"cacheir_tokens_generated_total {counters['tokens_generated_total']}",
            "# HELP cacheir_prefix_cache_hits Prefix-cache hits.",
            "# TYPE cacheir_prefix_cache_hits counter",
            f"cacheir_prefix_cache_hits {prefix.get('hits', 0)}",
            "# HELP cacheir_prefix_cache_entries Prefix-cache entries resident.",
            "# TYPE cacheir_prefix_cache_entries gauge",
            f"cacheir_prefix_cache_entries {prefix.get('entries', 0)}",
            "# HELP cacheir_scheduler_completed_requests Requests completed by the scheduler.",
            "# TYPE cacheir_scheduler_completed_requests counter",
            f"cacheir_scheduler_completed_requests {stats['scheduler']['completed_requests']}",
            "# HELP cacheir_scheduler_rejected_requests Requests rejected by scheduler admission control.",
            "# TYPE cacheir_scheduler_rejected_requests counter",
            f"cacheir_scheduler_rejected_requests {stats['scheduler'].get('rejected_requests', 0)}",
            "# HELP cacheir_scheduler_cancelled_requests Requests cancelled before completion.",
            "# TYPE cacheir_scheduler_cancelled_requests counter",
            f"cacheir_scheduler_cancelled_requests {stats['scheduler'].get('cancelled_requests', 0)}",
            "# HELP cacheir_scheduler_batched_prefill_rounds Batched prefill rounds executed by the scheduler.",
            "# TYPE cacheir_scheduler_batched_prefill_rounds counter",
            f"cacheir_scheduler_batched_prefill_rounds {stats['scheduler'].get('batched_prefill_rounds', 0)}",
            "# HELP cacheir_scheduler_batched_prefill_padded_tokens Padding tokens processed by variable-length prefill batches.",
            "# TYPE cacheir_scheduler_batched_prefill_padded_tokens counter",
            f"cacheir_scheduler_batched_prefill_padded_tokens {stats['scheduler'].get('batched_prefill_padded_tokens', 0)}",
            "# HELP cacheir_scheduler_batched_decode_rounds Batched decode rounds executed by the scheduler.",
            "# TYPE cacheir_scheduler_batched_decode_rounds counter",
            f"cacheir_scheduler_batched_decode_rounds {stats['scheduler'].get('batched_decode_rounds', 0)}",
            "# HELP cacheir_scheduler_batched_decode_tokens Tokens decoded through scheduler batch rounds.",
            "# TYPE cacheir_scheduler_batched_decode_tokens counter",
            f"cacheir_scheduler_batched_decode_tokens {stats['scheduler'].get('batched_decode_tokens', 0)}",
        ]
        return "\n".join(lines) + "\n"

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "cacheir-local", "object": "model"}]}

    @app.post("/v1/completions")
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", ""))
        max_tokens = int(payload.get("max_tokens", 16))
        text = "".join(runtime.generate(prompt, max_new_tokens=max_tokens, use_prefix_cache=True))
        prompt_tokens = len(runtime.tokenizer.encode(prompt))
        counters["requests_total"] += 1
        counters["tokens_generated_total"] += max_tokens
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
            text = "".join(runtime.generate(prompt, max_new_tokens=max_tokens, use_prefix_cache=True))
            prompt_tokens = len(runtime.tokenizer.encode(prompt))
            counters["requests_total"] += 1
            counters["tokens_generated_total"] += max_tokens
            return {
                "id": "cacheir-chat",
                "object": "chat.completion",
                "model": "cacheir-local",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens, "total_tokens": prompt_tokens + max_tokens},
            }

        def events():
            emitted = 0
            counters["requests_total"] += 1
            counters["stream_requests_total"] += 1
            for token in runtime.generate(prompt, max_new_tokens=max_tokens, use_prefix_cache=True):
                emitted += 1
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
            counters["tokens_generated_total"] += emitted

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/cacheir/batch_completions")
    def batch_completions(payload: dict[str, Any]) -> dict[str, Any]:
        prompts = payload.get("prompts", [])
        if not isinstance(prompts, list):
            prompts = [str(prompts)]
        prompts = [str(prompt) for prompt in prompts]
        max_tokens = int(payload.get("max_tokens", 16))
        priorities = payload.get("priorities", [])
        if not isinstance(priorities, list):
            priorities = []
        requests = [
            GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_tokens,
                priority=int(priorities[idx]) if idx < len(priorities) else 0,
            )
            for idx, prompt in enumerate(prompts)
        ]
        try:
            scheduler.submit_many(requests)
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        results = scheduler.run_until_complete()
        counters["requests_total"] += len(results)
        counters["tokens_generated_total"] += sum(len(result.generated_ids) for result in results)
        return {
            "object": "list",
            "data": [
                {
                    "id": result.request_id,
                    "object": "text_completion",
                    "model": "cacheir-local",
                    "choices": [{"index": 0, "text": result.text, "finish_reason": result.finish_reason}],
                    "cacheir": {
                        "prefill_ms": result.prefill_ms,
                        "decode_ms": result.decode_ms,
                        "reused_prefix_tokens": result.reused_prefix_tokens,
                    },
                }
                for result in results
            ],
            "scheduler": scheduler.stats(),
        }

    return app
