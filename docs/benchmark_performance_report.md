# CacheIR Benchmark, Performance, and Result Report

Date: 2026-06-25

## Executive Summary

CacheIR now has stronger production-facing runtime surfaces: prefix-cache-aware
prefill, per-request KV sessions that share weights/tokenizer state, a small
continuous-batch scheduler, server health and Prometheus-style metrics endpoints,
and a batch completions endpoint. It also now has an end-to-end CUDA artifact
runtime: the same compiled CacheIR graphs can execute on GPU tensors with a
GPU-resident KV cache, fp16 weights, SDPA/FlashAttention-capable attention
dispatch, Triton elementwise kernels, and cached packed QKV and Gate/Up weights
for fewer decode launches. The scheduler can now also batch variable-length CUDA
prefill with padding/masking at KV boundaries and active CUDA decode rounds
across multiple request sessions.

The CPU runtime also gained a native fused SiLU-multiply pybind11 kernel and a
more honest matmul dispatch policy: NumPy's optimized contraction is the
default, while CacheIR's native AVX matmul remains available for experiments
through `CACHEIR_NATIVE_MATMUL`.

The vLLM CUDA comparison path is runnable again. Two local blockers were fixed:

- vLLM V1 failed before inference with `RuntimeError: UVA is not available`.
  CacheIR's benchmark launcher now injects a `sitecustomize.py` shim and
  prepends the CacheIR project root so spawned vLLM workers receive the no-UVA
  fallback.
- FlashInfer JIT then needed `nvcc`. WSL2 root installed NVIDIA CUDA Toolkit
  13.3 at `/usr/local/cuda`, and CacheIR now exports that CUDA path for
  CacheIR-launched vLLM benchmarks.

This improves CacheIR substantially as an explainable compiler/runtime project.
It still does not honestly rate as an 8/10 production LLM serving system or raw
performance competitor on real model sizes. It still lacks packed quantized
GEMM, a true page-backed shared GPU paged KV allocator used directly by fused
attention kernels, tuned GEMM selection, and large-model benchmark evidence.

## Environment

| Field | Value |
| --- | --- |
| Host OS | Windows |
| CPU | AMD64 Family 26 Model 96 Stepping 0 |
| CPU cores | 16 |
| RAM | 31,865 MB |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| GPU memory | 8,151 MB |
| NVIDIA driver | 591.86 |
| Native Python | 3.13 |
| Native Torch | 2.12.1+cu130 |
| WSL distro | Ubuntu 24.04.3 LTS |
| WSL CUDA toolkit | 13.3 at `/usr/local/cuda` |
| WSL vLLM env | `/home/bhatt/cacheir-llm-venv` |

Bandwidth calibration:

| Measurement | Result |
| --- | ---: |
| CPU copy bandwidth | 32.04 GB/s |
| CUDA host-to-device bandwidth | 13.98 GB/s |
| Sample size | 16 MB |
| Repeats | 5 |

## Runtime And Serving Changes

Implemented in this pass:

- Prefix-cache hit/miss/eviction counters.
- Prefix-cache-aware prefill that can restore a cached KV prefix, append only the
  uncached suffix, and preserve RoPE offsets and causal masks.
- `Runtime.fork()` for independent KV-cache sessions sharing weights, tokenizer,
  and prefix cache.
- `ContinuousBatchScheduler` with per-request KV sessions, scheduling rounds,
  prefix reuse metrics, request counters, and token counters.
- FastAPI `/healthz`, `/metrics`, and `/v1/cacheir/batch_completions`.
- Native C++/pybind11 fused `silu_mul`.
- Matmul dispatch policy changed to default NumPy contraction, with
  `CACHEIR_NATIVE_MATMUL=force` or `CACHEIR_NATIVE_MATMUL=auto` for experiments.
- `CudaRuntime` for executing full CacheIR decoder artifacts on CUDA tensors.
- CUDA runtime GPU-resident KV cache with the same page-table metadata exposed by
  the CPU reference cache.
- CUDA runtime fp16 path, SDPA attention dispatch, Triton RMSNorm/SiLU-multiply
  for larger tensors, and cached packed QKV/Gate-Up weights.
- Shared CUDA page allocator accounting across forked request sessions.
- Scheduler-integrated CUDA batching for variable-length prefill and active
  decode rounds across independent request sessions.
- Scheduler admission controls: queue limits, request priorities, cancellation,
  and Prometheus counters.
- Prometheus-style metrics for batched prefill/decode scheduler rounds and
  tokens.
- Benchmark backend selector: `cacheir benchmark --backend cuda --cuda-dtype
  float16 --warmup N`.
- vLLM compatibility environment now includes both the CacheIR project root and
  `/usr/local/cuda/bin` when present.

## Validation

Commands:

```bash
cmake -S cpp -B cpp/build-py-active -DCACHEIR_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-py-active --config Release
python -m pytest
python -m compileall cacheir scripts -q
```

Result:

```text
34 passed, 4 warnings
```

Native smoke:

```text
available=True, simd_backend=avx512, silu_mul=True
```

## CPU Benchmark Matrix

Command:

```bash
python scripts/benchmark_matrix.py \
  --output .tmp/reports/cacheir_benchmark_matrix_latest.json \
  --repeats 3 \
  --decode-tokens 16
```

Selected rows:

| Model | Variant | Prompt | Prefill ms | Decode ms/token | Decode tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| tiny_1l_h16 | fp32 | short | 6.505 | 0.571 | 1,752.5 |
| small_2l_h32 | fp32 | short | 2.406 | 0.974 | 1,026.9 |
| small_2l_h32 | fp32 | medium | 2.565 | 1.005 | 994.6 |
| medium_4l_h64 | fp32 | short | 4.469 | 2.217 | 451.1 |
| medium_4l_h64 | fp32 | medium | 6.663 | 2.169 | 461.1 |
| medium_4l_h64 | int4_awq | long | 24.887 | 2.087 | 479.3 |

Artifacts:

- `.tmp/reports/cacheir_benchmark_matrix_latest.json`
- `.tmp/reports/cacheir_benchmark_matrix_latest.md`

## Scheduler Benchmark

Command:

```bash
python scripts/benchmark_scheduler.py \
  --output .tmp/reports/scheduler_benchmark.json \
  --repeats 3 \
  --max-new-tokens 8 \
  --max-batch-size 4
```

Result:

| Metric | Value |
| --- | ---: |
| Requests | 4 |
| Generated tokens | 32 |
| Median elapsed | 54.819 ms |
| Requests/s | 72.97 |
| Generated tokens/s | 583.74 |
| Prefix hits per sample | 2 |
| Prompt tokens reused | 70 |
| Max observed batch | 4 |

Artifact:

- `.tmp/reports/scheduler_benchmark.json`

## GPU Kernel Benchmarks

Command:

```bash
python scripts/benchmark_gpu_kernels.py \
  --output .tmp/reports/gpu_kernel_benchmarks_latest.json \
  --warmup 10 \
  --iters 50
```

| Kernel | Shape / Workload | Median ms | Throughput |
| --- | --- | ---: | ---: |
| Triton RMSNorm | 512 x 1024 fp32 | 0.0157 | 400.02 GB/s |
| Triton SiLU multiply | 1,048,576 fp32 elements | 0.0213 | 589.97 GB/s |
| Triton FP16 matmul | 512 x 1024 x 512 | 0.0667 | 8.05 TFLOP/s |
| Torch FP16 matmul | 512 x 1024 x 512 | 0.0275 | 19.54 TFLOP/s |
| Triton decode attention | seq=128, head_dim=64 | 0.0552 | 18,131.71 token/s |
| Triton batch decode attention | batch=4, heads=8, seq=128 | 0.0566 | 565,770.88 head-token/s |

Artifact:

- `.tmp/reports/gpu_kernel_benchmarks_latest.json`

The Torch matmul baseline remains much faster than the small CacheIR Triton
matmul. That is expected: CacheIR's kernel is a compact reference Tensor Core
lowering, not a tuned GEMM library.

## End-to-End CUDA Runtime Benchmarks

Command:

```bash
python scripts/benchmark_cuda_runtime.py \
  --output .tmp/reports/cuda_runtime_benchmark_h1024_latest.json \
  --repeats 3 \
  --warmup 1 \
  --decode-tokens 8 \
  --hidden-size 1024 \
  --num-layers 4 \
  --cuda-dtype float16
```

The benchmark compiles the same tiny Llama-shaped model to a CUDA CacheIR
artifact, runs the CPU reference backend and CUDA backend through the same
benchmark API, warms up before timing, and reports prefill/decode separately.

| Shape | Backend | Prefill ms | Decode ms/token | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| h128, 2 layers | CPU fp32 | 2.122 | 1.007 | 17,432.8 | 993.0 |
| h128, 2 layers | CUDA fp16 | 3.058 | 3.210 | 12,099.6 | 311.5 |
| h512, 2 layers | CPU fp32 | 5.828 | 2.077 | 6,348.3 | 481.4 |
| h512, 2 layers | CUDA fp16 | 3.867 | 4.030 | 9,567.5 | 248.1 |
| h1024, 4 layers | CPU fp32 | 30.984 | 12.271 | 1,194.1 | 81.5 |
| h1024, 4 layers | CUDA fp16 | 7.533 | 7.573 | 4,912.0 | 132.1 |
| h2048, 4 layers | CPU fp32 | 94.036 | 30.619 | 393.5 | 32.7 |
| h2048, 4 layers | CUDA fp16 | 6.705 | 7.333 | 5,517.9 | 136.4 |

Artifacts:

- `.tmp/reports/cuda_runtime_benchmark_latest.json`
- `.tmp/reports/cuda_runtime_benchmark_h512_latest.json`
- `.tmp/reports/cuda_runtime_benchmark_h1024_latest.json`
- `.tmp/reports/cuda_runtime_benchmark_h2048_latest.json`

Interpretation: the CUDA runtime is slower on very small toy shapes because
launch overhead dominates. It crosses over on larger toy shapes: h1024/l4 is
4.11x faster on prefill and 1.62x faster on decode; h2048/l4 is 14.02x faster
on prefill and 4.18x faster on decode. This proves CacheIR now has real
end-to-end GPU model execution, but it is still not a production serving engine:
the scheduler-level CUDA batching path still uses independent per-request K/V
tensors behind shared page accounting, and quantized Tensor Core GEMM is not yet
implemented.

## CUDA Scheduler Batching Benchmark

Command:

```bash
python scripts/benchmark_cuda_scheduler.py \
  --output .tmp/reports/cuda_scheduler_benchmark_latest.json \
  --repeats 3 \
  --warmup 1 \
  --max-batch-size 4 \
  --max-new-tokens 16 \
  --hidden-size 1024 \
  --num-layers 4 \
  --cuda-dtype float16
```

The benchmark compares four independent CUDA request sessions advanced
sequentially against the same four sessions admitted together, prefilling
variable-length prompts as a padded batch and decoding active rounds through
`run_decode_batch`. The h1024/h2048 runs processed 134 real prefill tokens plus
6 padding tokens.

| Shape | Mode | Median elapsed ms | Generated tok/s | Scheduler evidence |
| --- | --- | ---: | ---: | --- |
| h1024, 4 layers, batch 4 | Sequential CUDA sessions | 582.177 | 109.9 | 64 scheduling rounds, 0 batched decode tokens |
| h1024, 4 layers, batch 4 | Batched prefill + decode | 339.796 | 188.3 | 1 batched prefill round, 6 padding tokens, 16 batched decode rounds, 64 batched decode tokens |
| h2048, 4 layers, batch 4 | Sequential CUDA sessions | 867.240 | 37.0 | 32 scheduling rounds, 0 batched decode tokens |
| h2048, 4 layers, batch 4 | Batched prefill + decode | 684.002 | 46.8 | 1 batched prefill round, 6 padding tokens, 8 batched decode rounds, 32 batched decode tokens |

Artifacts:

- `.tmp/reports/cuda_scheduler_benchmark_latest.json`
- `.tmp/reports/cuda_scheduler_benchmark_h2048_latest.json`

Interpretation: h1024/l4 improves by 1.71x in latency and generated-token
throughput. h2048/l4 improves by 1.27x because prefill dominates the short
8-token run and attention still splits per request after padded dense work. This
is real scheduler-level CUDA batching with variable-length prefill, but it is
still short of production LLM serving because the fused attention path does not
yet consume a persistent page-backed shared GPU KV pool.

## Upstream Benchmarks

Native Windows:

| System | Status |
| --- | --- |
| IREE | Available; StableHLO add compiled to 9,781-byte VMFB and benchmarked |
| TVM | Available; TE vector-add checksum 512.0 |
| vLLM | Not importable in native Windows/Python 3.13 |
| llama.cpp | `llama-bench` not on native Windows PATH |

WSL2 CUDA vLLM:

| Field | Value |
| --- | --- |
| vLLM | 0.23.0 |
| Torch | 2.11.0+cu130 |
| Attention backend | FlashInfer |
| CUDA toolkit | 13.3 |
| Model | CacheIR-created tiny Llama h128 |
| Shape | 2 layers, hidden 128, 2 heads, head_dim 64 |
| Request | 16 input tokens + 8 output tokens |
| Average latency | 28.535 ms |
| Median latency | 25.447 ms |
| P90 latency | 37.368 ms |
| Engine init/warmup | 89.71 s, mostly first-run FlashInfer JIT/autotune |

Artifacts:

- `.tmp/reports/vllm_cuda_benchmark_latest.json`
- `.tmp/reports/vllm_latency_latest.json`

Matched CacheIR CPU reference:

| Metric | Value |
| --- | ---: |
| Prefill latency | 10.553 ms |
| Decode latency | 1.859 ms/token |
| Estimated request latency | 25.425 ms |
| Decode throughput | 537.92 token/s |

Artifact:

- `.tmp/reports/cacheir_vllm_shape_benchmark_latest.json`

This toy result should not be read as CacheIR beating vLLM generally. The model
is only 8.5M parameters and batch size is 1, so vLLM's fixed engine overhead and
FlashInfer JIT costs dominate. vLLM's advantages appear on realistic model
sizes, larger batches, continuous batching, longer contexts, and optimized graph
capture paths.

## What The Results Mean

CacheIR is now stronger on three axes:

- It has production-shaped serving surfaces: health, metrics, batch completions,
  prefix reuse, scheduler counters, queue limits, priorities, cancellation,
  isolated per-request KV state, and CUDA batched prefill/decode rounds exposed
  through metrics.
- It has a clearer performance posture: fast reference NumPy matmul by default,
  native CPU kernels where they help, GPU kernel microbenchmarks that run on the
  local RTX 5070, an end-to-end CUDA runtime that wins once toy model sizes
  become compute-heavy enough, and a measured scheduler-level CUDA batching
  speedup.
- It can run a real vLLM CUDA latency benchmark in WSL after fixing no-UVA
  propagation and installing a toolkit for FlashInfer JIT.

The project is still not a production peer of vLLM/TensorRT-LLM/llama.cpp/MLC
LLM. The biggest gap is production-grade GPU serving: CacheIR can execute full
graphs on CUDA and batch scheduler rounds now, but its shared CUDA page allocator
is still accounting metadata rather than the persistent page-backed storage used
directly by a production fused attention kernel. The second biggest gap is high-performance GEMM
and quantization: simulated int4 lowering is useful for compiler behavior, but it
is not packed int4 inference. The third biggest gap is model scale and coverage:
the strongest CUDA evidence is still toy-shaped models, not 0.5B/1.5B/7B local
models.

Honest current evaluation after this pass:

| Dimension | Score | Reason |
| --- | ---: | --- |
| Explainable LLM compiler project | 8/10 | Real IR, passes, artifacts, diffs, memory plans, imports, tests, docs |
| Research/portfolio systems project | 8/10 | Broad, runnable, benchmarked, and inspectable |
| Production LLM serving system | 7/10 | Serving APIs, scheduler, metrics, prefix reuse, queue limits, priorities, cancellation, CUDA model runtime, shared page accounting, and variable-length CUDA scheduler batching exist; persistent page-backed fused attention and reliability hardening are still missing |
| Raw performance competitor | 6/10 | CUDA fp16 runtime, Triton kernels, batched decode attention smoke execution, and CUDA scheduler batching show measured speedups on larger toy shapes, but no tuned quantized GEMM or large-model evidence yet |

To reach an honest 8/10 in the last two categories, CacheIR needs an end-to-end
production CUDA serving loop with persistent page-backed attention, packed
quantized GEMM, tuned GEMM selection, real model coverage beyond toy shapes, and
throughput/latency comparisons on 0.5B to 7B-class models.
