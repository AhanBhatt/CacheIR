from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from cacheir.runtime import Runtime
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
) -> BenchmarkResult:
    runtime = Runtime(artifact)
    token_ids = runtime.tokenizer.encode(prompt)
    prefill_times = []
    decode_times = []

    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        logits = runtime.run([token_ids], mode="prefill")
        prefill_times.append(time.perf_counter() - start)
        next_id = int(logits[0, -1].argmax())
        for _decode in range(decode_tokens):
            start = time.perf_counter()
            logits = runtime.run([[next_id]], mode="decode")
            decode_times.append(time.perf_counter() - start)
            next_id = int(logits[0, -1].argmax())

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
