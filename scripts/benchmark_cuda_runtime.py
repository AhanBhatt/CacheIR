from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cacheir import compile_model
from cacheir.benchmark import run_benchmark
from cacheir.importers import create_tiny_model
from cacheir.runtime import create_runtime, cuda_runtime_available


def run_cuda_runtime_benchmark(
    *,
    workdir: Path,
    output: Path | None = None,
    repeats: int = 5,
    warmup: int = 2,
    decode_tokens: int = 16,
    hidden_size: int = 128,
    num_layers: int = 2,
    intermediate_size: int | None = None,
    cuda_dtype: str = "float16",
) -> dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    intermediate_size = intermediate_size or hidden_size * 4
    run_dir = workdir / f"h{hidden_size}_l{num_layers}_i{intermediate_size}"
    model = create_tiny_model(
        run_dir / "tiny_cuda_model",
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=max(1, hidden_size // 64),
        num_key_value_heads=max(1, hidden_size // 128),
    )
    artifact_dir = run_dir / "tiny_cuda_artifact"
    artifact = compile_model(model, target="cuda", mode=["prefill", "decode"], max_seq=128, output=artifact_dir)
    prompt = "CacheIR CUDA runtime benchmark prompt"
    result: dict[str, object] = {
        "created_at_unix": time.time(),
        "model": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_layers": num_layers,
            "artifact": str(artifact_dir),
        },
        "prompt": prompt,
        "decode_tokens": decode_tokens,
        "repeats": repeats,
        "warmup": warmup,
        "cuda_available": cuda_runtime_available(),
        "cuda_dtype": cuda_dtype,
    }
    cpu = run_benchmark(artifact, prompt=prompt, decode_tokens=decode_tokens, repeats=repeats, warmup=warmup, backend="cpu").to_dict()
    result["cpu"] = cpu
    if cuda_runtime_available():
        cuda = run_benchmark(
            artifact,
            prompt=prompt,
            decode_tokens=decode_tokens,
            repeats=repeats,
            warmup=warmup,
            backend="cuda",
            runtime_kwargs={"dtype": cuda_dtype},
        ).to_dict()
        runtime = create_runtime(artifact, backend="cuda", dtype=cuda_dtype)
        list(runtime.generate(prompt, max_new_tokens=2))
        runtime.synchronize()
        cuda["kernel_counts_after_smoke"] = runtime.cache_stats().get("kernel_counts", {})
        result["cuda"] = cuda
        result["speedup"] = {
            "prefill_latency": _ratio(cpu["prefill_ms_avg"], cuda["prefill_ms_avg"]),
            "decode_latency": _ratio(cpu["decode_ms_avg"], cuda["decode_ms_avg"]),
            "prefill_throughput": _ratio(cuda["prefill_tokens_per_s"], cpu["prefill_tokens_per_s"]),
            "decode_throughput": _ratio(cuda["decode_tokens_per_s"], cpu["decode_tokens_per_s"]),
        }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        output.with_suffix(".md").write_text(_format_markdown(result), encoding="utf-8")
    return result


def _ratio(numerator: object, denominator: object) -> float:
    n = float(numerator)
    d = float(denominator)
    return n / d if d else 0.0


def _format_markdown(result: dict[str, object]) -> str:
    lines = [
        "# CacheIR CUDA Runtime Benchmark",
        "",
        f"CUDA available: `{result['cuda_available']}`",
        "",
        "| Backend | Prefill ms | Decode ms/token | Prefill tok/s | Decode tok/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["cpu", "cuda"]:
        entry = result.get(name)
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"| {name} | {float(entry['prefill_ms_avg']):.3f} | {float(entry['decode_ms_avg']):.3f} | "
            f"{float(entry['prefill_tokens_per_s']):.1f} | {float(entry['decode_tokens_per_s']):.1f} |"
        )
    speedup = result.get("speedup")
    if isinstance(speedup, dict):
        lines.extend(
            [
                "",
                "## Speedup",
                "",
                f"- Prefill latency ratio CPU/CUDA: `{float(speedup['prefill_latency']):.2f}x`",
                f"- Decode latency ratio CPU/CUDA: `{float(speedup['decode_latency']):.2f}x`",
                f"- Prefill throughput ratio CUDA/CPU: `{float(speedup['prefill_throughput']):.2f}x`",
                f"- Decode throughput ratio CUDA/CPU: `{float(speedup['decode_throughput']):.2f}x`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CacheIR CPU vs CUDA runtime backends")
    parser.add_argument("--workdir", type=Path, default=Path(".tmp/reports/cuda_runtime"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/reports/cuda_runtime_benchmark_latest.json"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--cuda-dtype", default="float16", choices=["float16", "float32", "auto"])
    args = parser.parse_args()
    result = run_cuda_runtime_benchmark(
        workdir=args.workdir,
        output=args.output,
        repeats=args.repeats,
        warmup=args.warmup,
        decode_tokens=args.decode_tokens,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        cuda_dtype=args.cuda_dtype,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
