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

The first backend is a CPU runtime that always has NumPy reference coverage and
can dispatch RMSNorm and out-by-in matmul through `_cacheir_native` when the
pybind11 module is built. These numbers are for regression tracking, pass
attribution, and native-kernel smoke testing. CUDA kernels use the same artifact
and benchmark surfaces when a compatible CUDA host toolchain is available.

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

Latest local verification on 2026-06-24:

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
- C++ RMSNorm and matmul matched NumPy within float32 tolerance.
- CUDA C++ target built with MSVC/NVCC: `cacheir_cuda_kernels.lib`.
- CUDA C++ FP16 WMMA Tensor Core matmul and reduced paged-attention decode paths
  built with MSVC/NVCC.
- Triton GPU kernels executed on CUDA and matched Torch references for RMSNorm,
  SwiGLU, FP16 matmul, single-query decode attention, and multi-batch
  page-table decode attention.
- `cacheir profile --calibrate --sample-mb 1 --repeats 1` measured CPU copy at
  8.05 GB/s and CUDA H2D at 2.85 GB/s on this host.
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
- The vLLM model benchmark runner launched against the same tiny HF model, but
  the local WSL vLLM 0.23.0 CUDA worker failed during initialization with
  `RuntimeError: UVA is not available`; the captured benchmark metadata is in
  `.tmp/upstream/vllm_tiny_benchmark.json`.
- Benchmark matrix ran 18 rows: 3 tiny model sizes x fp32/int4_awq x
  short/medium/long prompts, 3 repeats, 16 decode tokens.
- The comparison harness ran installed IREE/TVM smoke benchmarks and now exposes
  model-aware vLLM and llama.cpp helpers when model paths are supplied.
