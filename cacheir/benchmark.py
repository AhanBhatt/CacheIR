from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cacheir.runtime import create_runtime
from cacheir.runtime.artifact import CompileArtifact


@dataclass
class BenchmarkResult:
    prompt_tokens: int
    decode_tokens: int
    prefill_ms_avg: float
    decode_ms_avg: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    kv_cache: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_benchmark(
    artifact: CompileArtifact | str | Path,
    *,
    prompt: str = "CacheIR benchmark prompt",
    decode_tokens: int = 16,
    repeats: int = 3,
    backend: str = "cpu",
    warmup: int = 0,
    runtime_kwargs: dict[str, Any] | None = None,
) -> BenchmarkResult:
    runtime = create_runtime(artifact, backend=backend, **(runtime_kwargs or {}))
    token_ids = runtime.tokenizer.encode(prompt)
    prefill_times = []
    decode_times = []

    total_iterations = max(1, repeats) + max(0, warmup)
    for iteration in range(total_iterations):
        record = iteration >= max(0, warmup)
        start = time.perf_counter()
        logits = runtime.run([token_ids], mode="prefill")
        _synchronize(runtime)
        if record:
            prefill_times.append(time.perf_counter() - start)
        next_id = _argmax_token(logits)
        for _decode in range(decode_tokens):
            start = time.perf_counter()
            logits = runtime.run([[next_id]], mode="decode")
            _synchronize(runtime)
            if record:
                decode_times.append(time.perf_counter() - start)
            next_id = _argmax_token(logits)

    prefill_avg = statistics.mean(prefill_times)
    decode_avg = statistics.mean(decode_times) if decode_times else 0.0
    return BenchmarkResult(
        prompt_tokens=len(token_ids),
        decode_tokens=decode_tokens,
        prefill_ms_avg=prefill_avg * 1000.0,
        decode_ms_avg=decode_avg * 1000.0,
        prefill_tokens_per_s=(len(token_ids) / prefill_avg) if prefill_avg else 0.0,
        decode_tokens_per_s=(1.0 / decode_avg) if decode_avg else 0.0,
        kv_cache=runtime.cache_stats(),
    )


def save_benchmark(result: BenchmarkResult, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return out


def _synchronize(runtime: object) -> None:
    sync = getattr(runtime, "synchronize", None)
    if callable(sync):
        sync()


def _argmax_token(logits: object) -> int:
    tensor_slice = logits[0, -1]  # type: ignore[index]
    argmax = getattr(tensor_slice, "argmax", None)
    value = argmax() if callable(argmax) else tensor_slice.argmax()  # type: ignore[union-attr]
    item = getattr(value, "item", None)
    return int(item() if callable(item) else value)
