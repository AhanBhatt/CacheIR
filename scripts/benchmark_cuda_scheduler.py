from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from cacheir import ContinuousBatchScheduler, CudaRuntime, compile_model, cuda_runtime_available
from cacheir.importers import create_tiny_model


DEFAULT_PROMPTS = [
    "CacheIR CUDA scheduler request 0",
    "CacheIR CUDA scheduler request 11",
    "CacheIR CUDA scheduler request 222",
    "CacheIR CUDA scheduler request 3333",
]


def run_cuda_scheduler_benchmark(
    *,
    workdir: Path,
    output: Path | None = None,
    prompts: list[str] | None = None,
    max_batch_size: int = 4,
    max_new_tokens: int = 16,
    repeats: int = 3,
    warmup: int = 1,
    hidden_size: int = 1024,
    num_layers: int = 4,
    intermediate_size: int | None = None,
    cuda_dtype: str = "float16",
    use_triton_attention: bool = False,
) -> dict[str, object]:
    prompts = prompts or DEFAULT_PROMPTS[:max_batch_size]
    workdir.mkdir(parents=True, exist_ok=True)
    intermediate_size = intermediate_size or hidden_size * 4
    result: dict[str, object] = {
        "cuda_available": cuda_runtime_available(),
        "prompts": prompts,
        "max_batch_size": max_batch_size,
        "max_new_tokens": max_new_tokens,
        "repeats": repeats,
        "warmup": warmup,
        "cuda_dtype": cuda_dtype,
        "use_triton_attention": use_triton_attention,
        "model": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_layers": num_layers,
        },
    }
    if not cuda_runtime_available():
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    run_dir = workdir / f"h{hidden_size}_l{num_layers}_b{max_batch_size}"
    model = create_tiny_model(
        run_dir / "model",
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=max(1, hidden_size // 64),
        num_key_value_heads=max(1, hidden_size // 128),
    )
    artifact = compile_model(model, target="cuda", mode=["prefill", "decode"], max_seq=128, output=run_dir / "artifact")
    result["artifact"] = str(run_dir / "artifact")

    sequential_samples = []
    batched_samples = []
    for iteration in range(max(0, warmup) + max(1, repeats)):
        record = iteration >= max(0, warmup)
        sequential = _run_scheduler_once(
            artifact,
            prompts,
            max_batch_size=1,
            max_new_tokens=max_new_tokens,
            cuda_dtype=cuda_dtype,
            use_triton_attention=use_triton_attention,
        )
        batched = _run_scheduler_once(
            artifact,
            prompts,
            max_batch_size=max_batch_size,
            max_new_tokens=max_new_tokens,
            cuda_dtype=cuda_dtype,
            use_triton_attention=use_triton_attention,
        )
        if record:
            sequential_samples.append(sequential)
            batched_samples.append(batched)

    result["sequential"] = _summarize_samples(sequential_samples)
    result["batched"] = _summarize_samples(batched_samples)
    seq_ms = float(result["sequential"]["median_elapsed_ms"])  # type: ignore[index]
    batched_ms = float(result["batched"]["median_elapsed_ms"])  # type: ignore[index]
    result["speedup"] = {
        "latency": seq_ms / batched_ms if batched_ms else 0.0,
        "generated_tokens_per_s": float(result["batched"]["generated_tokens_per_s"]) / float(result["sequential"]["generated_tokens_per_s"]),  # type: ignore[index]
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        output.with_suffix(".md").write_text(_format_markdown(result), encoding="utf-8")
    return result


def _run_scheduler_once(
    artifact: object,
    prompts: list[str],
    *,
    max_batch_size: int,
    max_new_tokens: int,
    cuda_dtype: str,
    use_triton_attention: bool,
) -> dict[str, object]:
    runtime = CudaRuntime(artifact, dtype=cuda_dtype, use_triton_attention=use_triton_attention)
    scheduler = ContinuousBatchScheduler(runtime, max_batch_size=max_batch_size, use_prefix_cache=False)
    start = time.perf_counter()
    results = scheduler.generate_batch(prompts, max_new_tokens=max_new_tokens)
    runtime.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    generated_tokens = sum(len(result.generated_ids) for result in results)
    return {
        "elapsed_ms": elapsed_ms,
        "requests": len(results),
        "generated_tokens": generated_tokens,
        "generated_tokens_per_s": generated_tokens / (elapsed_ms / 1000.0),
        "scheduler": scheduler.stats(),
    }


def _summarize_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    elapsed = [float(sample["elapsed_ms"]) for sample in samples]
    tokens_per_s = [float(sample["generated_tokens_per_s"]) for sample in samples]
    scheduler = samples[-1]["scheduler"] if samples else {}
    return {
        "median_elapsed_ms": statistics.median(elapsed) if elapsed else 0.0,
        "mean_elapsed_ms": statistics.fmean(elapsed) if elapsed else 0.0,
        "generated_tokens_per_s": statistics.median(tokens_per_s) if tokens_per_s else 0.0,
        "samples": samples,
        "last_scheduler": scheduler,
    }


def _format_markdown(result: dict[str, object]) -> str:
    sequential = result.get("sequential", {})
    batched = result.get("batched", {})
    speedup = result.get("speedup", {})
    lines = [
        "# CacheIR CUDA Scheduler Benchmark",
        "",
        f"CUDA available: `{result['cuda_available']}`",
        "",
        "| Mode | Median elapsed ms | Generated tok/s |",
        "|---|---:|---:|",
    ]
    if isinstance(sequential, dict):
        lines.append(f"| Sequential CUDA sessions | {float(sequential['median_elapsed_ms']):.3f} | {float(sequential['generated_tokens_per_s']):.1f} |")
    if isinstance(batched, dict):
        lines.append(f"| Batched CUDA scheduler | {float(batched['median_elapsed_ms']):.3f} | {float(batched['generated_tokens_per_s']):.1f} |")
    if isinstance(speedup, dict):
        lines.extend(
            [
                "",
                f"Latency speedup: `{float(speedup['latency']):.2f}x`",
                f"Throughput speedup: `{float(speedup['generated_tokens_per_s']):.2f}x`",
            ]
        )
    if isinstance(batched, dict):
        last = batched.get("last_scheduler", {})
        if isinstance(last, dict):
            sched = last.get("scheduler", {})
            if isinstance(sched, dict):
                lines.extend(
                    [
                        "",
                        "## Scheduler Evidence",
                        "",
                        f"- Batched prefill rounds: `{sched.get('batched_prefill_rounds', 0)}`",
                        f"- Batched prefill tokens: `{sched.get('batched_prefill_tokens', 0)}`",
                        f"- Batched prefill padding tokens: `{sched.get('batched_prefill_padded_tokens', 0)}`",
                        f"- Batched decode rounds: `{sched.get('batched_decode_rounds', 0)}`",
                        f"- Batched decode tokens: `{sched.get('batched_decode_tokens', 0)}`",
                    ]
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sequential vs batched CUDA scheduler decode")
    parser.add_argument("--workdir", type=Path, default=Path(".tmp/reports/cuda_scheduler"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/reports/cuda_scheduler_benchmark_latest.json"))
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--cuda-dtype", default="float16", choices=["float16", "float32", "auto"])
    parser.add_argument("--use-triton-attention", action="store_true")
    parser.add_argument("--prompt", action="append", default=None)
    args = parser.parse_args()
    result = run_cuda_scheduler_benchmark(
        workdir=args.workdir,
        output=args.output,
        prompts=args.prompt,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        cuda_dtype=args.cuda_dtype,
        use_triton_attention=args.use_triton_attention,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
