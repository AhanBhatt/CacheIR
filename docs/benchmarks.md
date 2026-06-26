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

Latest local verification on 2026-06-25:

- Full optional project install passed with
  `python -m pip install -e ".[dev,server,importers,benchmark,native,gpu]"`.
- PyPI packages installed include `pybind11`, `triton-windows`, ONNX/importer
  dependencies, server dependencies, and benchmark dependencies.
- MSVC Build Tools 2022 installed at `C:\BuildTools`; `cl.exe` resolves from the
  user PATH.
- CUDA Torch installed: `torch 2.12.1+cu130`; CUDA is available on the RTX 5070
  Laptop GPU.
- `_cacheir_native.pyd` built with CMake, PyPI pybind11, and the active Python
  3.13 interpreter.
- Native smoke reported `simd_backend() == "avx512"`.
- C++ RMSNorm, matmul, and fused SiLU-multiply matched NumPy within float32 tolerance.
- Runtime matmul dispatch defaults to NumPy's optimized contraction; native matmul is available as an opt-in kernel experiment.
- CUDA C++ target built with MSVC/NVCC: `cacheir_cuda_kernels.lib`.
- CUDA C++ FP16 WMMA Tensor Core matmul and reduced paged-attention decode paths
  built with MSVC/NVCC.
- End-to-end `CudaRuntime` executed full decoder artifacts with fp16 weights,
  GPU-resident KV state, SDPA attention, Triton elementwise dispatch gates, and
  cached packed QKV/Gate-Up weights.
- CUDA runtime benchmark crossover results: h1024/l4 CUDA fp16 measured
  7.533 ms prefill and 7.573 ms/token decode versus CPU fp32 30.984 ms prefill
  and 12.271 ms/token decode; h2048/l4 CUDA fp16 measured 6.705 ms prefill and
  7.333 ms/token decode versus CPU fp32 94.036 ms prefill and
  30.619 ms/token decode.
- CUDA scheduler benchmark result: h1024/l4 batch-4 variable-length batched
  prefill plus active decode batching processed 134 real prefill tokens plus 6
  padding tokens, improving median latency from 582.177 ms to 339.796 ms and
  generated-token throughput from 109.9 tok/s to 188.3 tok/s. h2048/l4 batch-4
  improved median latency from 867.240 ms to 684.002 ms and throughput from
  37.0 tok/s to 46.8 tok/s.
- Triton batched decode attention smoke test executed through the scheduler path
  with `--use-triton-attention` on h128/l1 and showed a 3.36x speedup over
  sequential CUDA sessions for that tiny smoke shape.
- Triton GPU kernels executed on CUDA and matched Torch references for RMSNorm,
  SwiGLU, FP16 matmul, single-query decode attention, and multi-batch
  page-table decode attention.
- `cacheir profile --calibrate --sample-mb 16 --repeats 5` measured CPU copy at
  32.04 GB/s and CUDA H2D at 13.98 GB/s on this host.
- `nvidia-cutlass 4.2.0.0` installed successfully; CacheIR detects its
  `cutlass_cppgen` Python surface through the CUTLASS adapter probe.
- `iree-base-compiler 3.11.0`, `iree-base-runtime 3.11.0`, and
  `apache-tvm 0.25.0` installed from PyPI.
- `cacheir external --benchmark --workdir .tmp/upstream` compiled StableHLO
  through IREE to a 9,781-byte VMFB and ran `iree-benchmark-module`; TVM built
  and ran a TE vector-add benchmark with checksum 512.0.
- WSL2 CUDA environments validated direct FlashAttention prefill and FlashInfer
  decode execution through CacheIR's adapter wrappers.
- A CUDA llama.cpp build ran a real `llama-bench` model benchmark against a
  locally converted GGUF tiny Llama model; the JSON result is in
  `.tmp/upstream/llama_cpp_tiny_benchmark.json`.
- The vLLM model benchmark runner installs a process-local no-UVA fallback shim
  before vLLM workers initialize and prepends both the CacheIR project root and
  `/usr/local/cuda/bin` to the benchmark environment.
- WSL2 root installed NVIDIA CUDA Toolkit 13.3 at `/usr/local/cuda`; this gives
  FlashInfer JIT a real `nvcc` without installing a Linux NVIDIA driver package.
- A WSL2 CUDA vLLM environment at `/home/bhatt/cacheir-llm-venv` ran vLLM
  0.23.0 with Torch 2.11.0+cu130 and FlashInfer against a CacheIR-created tiny
  Llama h128 model. Latest result: 28.535 ms average request latency for
  16 input + 8 output tokens.
- Matched CacheIR h128 CPU reference result: 10.553 ms prefill and
  1.859 ms/token decode, about 25.425 ms estimated full-request latency for the
  same toy request.
- Continuous-batch scheduler result: 4 requests, 32 generated tokens, median
  54.819 ms, 72.97 requests/s, 583.74 generated tokens/s, and 70 prompt tokens
  reused through prefix cache.
- Benchmark matrix ran 18 rows: 3 tiny model sizes x fp32/int4_awq x
  short/medium/long prompts, 3 repeats, 16 decode tokens.
- The comparison harness ran installed IREE/TVM smoke benchmarks and now exposes
  model-aware vLLM and llama.cpp helpers when model paths are supplied.
