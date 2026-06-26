from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Callable


def _time_cuda(fn: Callable[[], object], *, warmup: int, iters: int) -> dict[str, float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run_gpu_kernel_benchmarks(*, warmup: int = 10, iters: int = 50) -> dict[str, object]:
    os.environ.setdefault("TRITON_CACHE_DIR", str((Path(".tmp") / "triton-cache").resolve()))
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    import torch
    import cacheir.backends.triton_kernels as tk

    result: dict[str, object] = {
        "available": bool(tk.triton_available() and torch.cuda.is_available()),
        "warmup": warmup,
        "iters": iters,
    }
    if not result["available"]:
        result["reason"] = "Triton or torch CUDA is unavailable"
        return result

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    result["device"] = {
        "name": props.name,
        "capability": [props.major, props.minor],
        "total_memory_mb": int(props.total_memory // (1024 * 1024)),
        "torch": torch.__version__,
    }

    rows, hidden = 512, 1024
    x = torch.randn(rows, hidden, device=device, dtype=torch.float32)
    norm_weight = torch.ones(hidden, device=device, dtype=torch.float32)
    rms_out = torch.empty_like(x)
    rms_grid = (rows,)
    rms = _time_cuda(
        lambda: tk.rms_norm_kernel[rms_grid](x, norm_weight, rms_out, hidden=hidden, eps=1e-6, BLOCK=1024),
        warmup=warmup,
        iters=iters,
    )
    rms_bytes = rows * hidden * 4 * 3
    rms["approx_gbps"] = rms_bytes / (rms["median_ms"] / 1000.0) / 1e9

    total = 1 << 20
    gate = torch.randn(total, device=device, dtype=torch.float32)
    up = torch.randn(total, device=device, dtype=torch.float32)
    silu_out = torch.empty_like(gate)
    silu = _time_cuda(
        lambda: tk.silu_mul_kernel[((total + 255) // 256,)](gate, up, silu_out, total=total, BLOCK=256),
        warmup=warmup,
        iters=iters,
    )
    silu_bytes = total * 4 * 3
    silu["approx_gbps"] = silu_bytes / (silu["median_ms"] / 1000.0) / 1e9

    m, k_dim, n = 512, 1024, 512
    a = torch.randn(m, k_dim, device=device, dtype=torch.float16)
    b = torch.randn(k_dim, n, device=device, dtype=torch.float16)
    mm_out = torch.empty((m, n), device=device, dtype=torch.float32)
    matmul = _time_cuda(
        lambda: tk.matmul_f16_kernel[((m + 15) // 16, (n + 15) // 16)](
            a,
            b,
            mm_out,
            m=m,
            n=n,
            k=k_dim,
            stride_am=a.stride(0),
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            stride_bn=b.stride(1),
            stride_om=mm_out.stride(0),
            stride_on=mm_out.stride(1),
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=32,
        ),
        warmup=warmup,
        iters=iters,
    )
    matmul["approx_tflops"] = (2.0 * m * n * k_dim) / (matmul["median_ms"] / 1000.0) / 1e12
    torch_matmul = _time_cuda(lambda: torch.matmul(a, b), warmup=warmup, iters=iters)
    torch_matmul["approx_tflops"] = (2.0 * m * n * k_dim) / (torch_matmul["median_ms"] / 1000.0) / 1e12

    seq_len, head_dim = 128, 64
    q = torch.randn(1, head_dim, device=device, dtype=torch.float32)
    k_cache = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    v_cache = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    attn_out = torch.empty_like(q)
    decode = _time_cuda(
        lambda: tk.paged_attention_decode_kernel[(1,)](
            q, k_cache, v_cache, attn_out, seq_len=seq_len, head_dim=head_dim, BLOCK=64
        ),
        warmup=warmup,
        iters=iters,
    )
    decode["tokens_per_s"] = 1000.0 / decode["median_ms"]

    batch, num_heads, num_kv_heads, page_size, max_pages, head_dim = 4, 8, 4, 16, 8, 64
    q_batch = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.float32)
    k_pages = torch.randn(64, num_kv_heads, page_size, head_dim, device=device, dtype=torch.float32)
    v_pages = torch.randn(64, num_kv_heads, page_size, head_dim, device=device, dtype=torch.float32)
    page_table = torch.arange(batch * max_pages, device=device, dtype=torch.int32).reshape(batch, max_pages)
    seq_lens = torch.full((batch,), page_size * max_pages, device=device, dtype=torch.int32)
    batch_out = torch.empty_like(q_batch)
    batch_decode = _time_cuda(
        lambda: tk.paged_attention_decode_batch_kernel[(num_heads, batch)](
            q_batch,
            k_pages,
            v_pages,
            page_table,
            seq_lens,
            batch_out,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_pages_per_seq=max_pages,
            page_size=page_size,
            head_dim=head_dim,
            BLOCK=64,
        ),
        warmup=warmup,
        iters=iters,
    )
    batch_decode["tokens_per_s"] = (batch * num_heads) * 1000.0 / batch_decode["median_ms"]

    result["benchmarks"] = {
        "triton_rms_norm": {"shape": [rows, hidden], **rms},
        "triton_silu_mul": {"elements": total, **silu},
        "triton_matmul_f16": {"shape": [m, k_dim, n], **matmul},
        "torch_matmul_f16": {"shape": [m, k_dim, n], **torch_matmul},
        "triton_decode_attention": {"seq_len": seq_len, "head_dim": head_dim, **decode},
        "triton_batch_decode_attention": {
            "batch": batch,
            "num_heads": num_heads,
            "seq_len": page_size * max_pages,
            "head_dim": head_dim,
            **batch_decode,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CacheIR Triton CUDA kernels")
    parser.add_argument("--output", default=".tmp/reports/gpu_kernel_benchmarks.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    result = run_gpu_kernel_benchmarks(warmup=args.warmup, iters=args.iters)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
