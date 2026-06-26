from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cacheir import ContinuousBatchScheduler, compile_model
from cacheir.importers import create_tiny_model
from cacheir.runtime.artifact import CompileArtifact


DEFAULT_PROMPTS = [
    "CacheIR production serving",
    "CacheIR production serving with prefix reuse",
    "CacheIR production serving with prefix reuse and decode",
    "A different prompt for queue fairness",
]


def _load_or_create_artifact(path: str | None, workdir: Path) -> CompileArtifact:
    if path:
        return CompileArtifact.load(path)
    model = create_tiny_model(
        workdir / "scheduler_model",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=96)


def run_scheduler_benchmark(
    *,
    artifact_path: str | None = None,
    prompts: list[str] | None = None,
    max_batch_size: int = 4,
    max_new_tokens: int = 8,
    repeats: int = 3,
) -> dict[str, object]:
    prompts = prompts or DEFAULT_PROMPTS
    samples: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="cacheir-scheduler-bench-") as tmp:
        artifact = _load_or_create_artifact(artifact_path, Path(tmp))
        for _ in range(max(1, repeats)):
            scheduler = ContinuousBatchScheduler(artifact, max_batch_size=max_batch_size, use_prefix_cache=True)
            start = time.perf_counter()
            results = scheduler.generate_batch(prompts, max_new_tokens=max_new_tokens)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            samples.append(
                {
                    "elapsed_ms": elapsed_ms,
                    "requests": len(results),
                    "generated_tokens": sum(len(result.generated_ids) for result in results),
                    "reused_prefix_tokens": sum(result.reused_prefix_tokens for result in results),
                    "scheduler": scheduler.stats(),
                }
            )

    elapsed_values = [float(sample["elapsed_ms"]) for sample in samples]
    generated_tokens = [int(sample["generated_tokens"]) for sample in samples]
    request_count = len(prompts)
    median_elapsed_ms = statistics.median(elapsed_values)
    return {
        "prompts": prompts,
        "max_batch_size": max_batch_size,
        "max_new_tokens": max_new_tokens,
        "repeats": repeats,
        "median_elapsed_ms": median_elapsed_ms,
        "mean_elapsed_ms": statistics.fmean(elapsed_values),
        "requests_per_s": request_count / (median_elapsed_ms / 1000.0),
        "generated_tokens_per_s": statistics.median(generated_tokens) / (median_elapsed_ms / 1000.0),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CacheIR continuous-batch scheduler")
    parser.add_argument("--artifact", default=None, help="optional CacheIR artifact path or bundle")
    parser.add_argument("--output", default=".tmp/reports/scheduler_benchmark.json")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", action="append", default=None, help="prompt; may be repeated")
    args = parser.parse_args()

    result = run_scheduler_benchmark(
        artifact_path=args.artifact,
        prompts=args.prompt,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
