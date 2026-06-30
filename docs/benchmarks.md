# Benchmarks

CacheIR benchmarks report prefill and decode separately:

```bash
cacheir benchmark examples/tiny_artifact --prompt "CacheIR benchmark" --decode-tokens 32 --repeats 3
```

Output fields:

- `prompt_tokens`
- `decode_tokens`
- `prefill_ms_avg`
- `decode_ms_avg`
- `prefill_tokens_per_s`
- `decode_tokens_per_s`
- `kv_cache`

The detailed benchmark and performance report for the latest run is
`docs/benchmark_performance_report.md`.

The first backend is a CPU runtime that always has NumPy reference coverage.
Native RMSNorm and fused SiLU-multiply dispatch through `_cacheir_native` when
the pybind11 module is built. Native matmul is intentionally opt-in through
`CACHEIR_NATIVE_MATMUL=force` or `CACHEIR_NATIVE_MATMUL=auto` because the local
AVX dot-loop kernel is not consistently faster than NumPy's optimized
contraction on decode-heavy shapes. These numbers are for regression tracking,
pass attribution, and native-kernel smoke testing. CUDA kernels use the same
artifact and benchmark surfaces when a compatible CUDA host toolchain is
available.

The CUDA artifact runtime can be benchmarked directly with the shared benchmark
API or the dedicated CPU/CUDA comparison script:

```bash
cacheir benchmark ARTIFACT_DIR --backend cuda --cuda-dtype float16 --warmup 2

python scripts/benchmark_cuda_runtime.py \
  --output .tmp/reports/cuda_runtime_benchmark_h1024_latest.json \
  --repeats 3 \
  --warmup 1 \
  --decode-tokens 8 \
  --hidden-size 1024 \
  --num-layers 4 \
  --cuda-dtype float16
```

The continuous-batch scheduler benchmark exercises per-request KV sessions that
share weights, tokenizer state, and prefix-cache snapshots:

```bash
python scripts/benchmark_scheduler.py \
  --output .tmp/reports/scheduler_benchmark.json \
  --repeats 3 \
  --max-new-tokens 8 \
  --max-batch-size 4
```

The CUDA scheduler benchmark compares sequential CUDA request sessions with
variable-length batched prefill and active-request batched decode:

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

CacheIR also ships a comparison harness for external systems. It always records a
CacheIR run, then runs user-supplied commands for tools available on the machine:

```bash
python scripts/compare_external_benchmarks.py examples/tiny_artifact \
  --vllm-command "python bench_vllm.py" \
  --llama-command "llama-bench -m model.gguf" \
  --iree-command "iree-benchmark-module --module=model.vmfb" \
  --tvm-command "python bench_tvm.py" \
  --output benchmark_comparison.json
```

The harness does not vendor or wrap those projects. It captures command metadata,
return codes, elapsed wall time, and output tails so local vLLM, llama.cpp, IREE,
and TVM comparisons can be attached to the same prefill/decode benchmark record.
When the IREE and TVM wheels are installed, the harness can also run built-in
upstream smoke benchmarks without a user-supplied command:

```bash
python scripts/compare_external_benchmarks.py examples/tiny_artifact \
  --run-installed-smoke \
  --output benchmark_comparison.json
```

The same installed-system checks are available from the CLI:

```bash
cacheir external --benchmark --workdir .tmp/upstream
```

When local model artifacts are available, the CLI and harness can also launch
model-aware vLLM and llama.cpp benchmark helpers:

```bash
cacheir external --benchmark --workdir .tmp/upstream \
  --vllm-model /path/to/hf_model \
  --llama-model /path/to/model.gguf

python scripts/compare_external_benchmarks.py examples/tiny_artifact \
  --run-installed-smoke \
  --vllm-model /path/to/hf_model \
  --llama-model /path/to/model.gguf \
  --output benchmark_comparison.json
```

## Local Native Experiment Snapshot

Latest local verification on 2026-06-29:

- Full test suite passed with `python -m pytest -q`: 38 passed, 6 warnings in
  17.47s.
- `python -m compileall cacheir scripts -q` passed after the production-serving
  changes.
- CUDA is available on the RTX 5070 Laptop GPU. The WSL2 CUDA environment at
  `/home/bhatt/cacheir-llm-venv` contains vLLM 0.23.0, Torch 2.11.0+cu130,
  FlashInfer, Triton, and llama-cpp-python.
- `CudaRuntime` now uses persistent page-backed GPU KV pools. The fused Triton
  decode path consumes the shared page table directly instead of repacking
  per-request K/V tensors for the scheduler's batch decode.
- Packed int4 and int8 weight objects store compressed uint8 payloads, per-row
  scales, affine zero points, and shape metadata. CPU and CUDA loaders route
  quantized matmul, QKV, and SwiGLU paths through those packed objects.
- Scheduler hardening covers blocking backpressure, fairness aging,
  preemption, per-request token limits, bounded latency counters, and queue soak
  metrics.
- Persistent CUDA scheduler benchmark on h512/l2 with `--use-triton-attention`
  improved median latency from 176.225 ms to 78.491 ms and generated-token
  throughput from 184.0 tok/s to 407.8 tok/s.
- CUDA runtime benchmark on h512/l2 measured 3.515 ms prefill versus CPU
  8.800 ms. Decode was 3.264 ms/token on CUDA versus 3.139 ms/token on CPU for
  that small shape.
- Triton GPU kernels executed on CUDA and matched Torch references for RMSNorm,
  SwiGLU, FP16 matmul, single-query decode attention, and multi-batch persistent
  page-table decode attention.
- Qwen2.5-0.5B-Instruct was downloaded locally in WSL2, compiled through the HF
  importer, and executed on CUDA with bf16 safetensors shape discovery and
  persistent paged decode.
- Qwen2.5-0.5B apples-to-apples latency: CacheIR averaged 233.774 ms/request
  for 16 input + 8 output tokens; vLLM averaged 43.697 ms/request in the same
  WSL2 CUDA environment after warmup.
- A CUDA llama.cpp build ran `llama-bench` against a locally converted tiny GGUF
  Llama model: 6,120.22 prompt tok/s for 16 prompt tokens and 2,044.44
  generation tok/s for 8 generated tokens.
- `cacheir profile --calibrate --sample-mb 16 --repeats 5` measured CPU copy at
  32.04 GB/s and CUDA H2D at 13.98 GB/s on this host.
- `nvidia-cutlass 4.2.0.0` installed successfully; CacheIR detects its
  `cutlass_cppgen` Python surface through the CUTLASS adapter probe.
- FlashInfer executed through CacheIR's adapter smoke test. FlashAttention,
  TensorRT-LLM, and MLC LLM remain guarded optional integrations on this host
  because binary-only PyPI checks found no compatible wheels and source install
  attempts were not stable in the local WSL2 image.
- `cacheir external --benchmark --workdir .tmp/upstream` compiled StableHLO
  through IREE to a 9,781-byte VMFB and ran `iree-benchmark-module`; TVM built
  and ran a TE vector-add benchmark with checksum 512.0.
- The comparison harness now probes and reports vLLM, llama.cpp, TensorRT-LLM,
  MLC LLM, IREE, and TVM when those systems are installed.
