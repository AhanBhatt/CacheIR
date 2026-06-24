# CacheIR

CacheIR is a narrow, inspectable compiler and runtime for decoder-only transformer
inference. It imports a Llama/Mistral/Qwen-style model, lowers it into its own IR,
runs transformer-specific optimization passes, plans memory and execution, selects
backend kernels, and executes inference through a reference runtime.

It is not a PyTorch wrapper and it is not a vLLM clone. The default runtime backend
is a NumPy CPU correctness backend with native acceleration hooks. Native work is
present as optional C++20/OpenMP AVX2/AVX512 kernels exposed through pybind11,
guarded Triton kernels, optional CUDA fused-kernel sources, optional accelerator
adapters, CUDA graph capture plans, calibrated KV spillover cost models, and experimental
import/export surfaces for GGUF, StableHLO, and MLIR-style CacheIR.

## Why CacheIR Exists

Most ML compiler stacks are powerful but broad and opaque. LLM serving has a very
specific shape: prefill is compute-heavy, decode is KV-cache and memory-bandwidth
heavy, and transformer optimizations need to be visible to be trusted.

CacheIR focuses on that smaller problem:

- compile prefill and decode as separate graph modes
- lower attention into explicit KV-cache-aware operations
- expose every pass as a graph diff
- emit memory plans and execution schedules
- run a real end-to-end reference path without PyTorch execution
- benchmark prefill and decode separately

## Current Capabilities

| Layer | Status |
| --- | --- |
| Model import | Hugging Face config + NPZ/safetensors metadata, ONNX graph skeleton, GGUF metadata plus dense F32/F16/BF16/I8/I16/I32/I64/F64, classic quant reads, and optional reference K/IQ/TQ/NV/MX dequantization, broader StableHLO text/region subset |
| IR | CacheIR JSON/text graph format, tensor types, weight specs, attrs, pass traces |
| Compiler passes | shape inference, constant folding, QKV fusion, RMSNorm+QKV+RoPE fusion, SwiGLU fusion, prefill/decode specialization, layout conversion, quant-aware lowering, hardware hints, kernel selection, scheduling, memory planning |
| Runtime | NumPy CPU backend, tokenizer bridge, paged KV cache, prefix-cache reuse, calibrated CPU/GPU spillover policy experiments, greedy streaming generation |
| Serving | OpenAI-compatible local FastAPI server |
| Tooling | CLI, artifact bundles, graph HTML/DOT/MLIR export, benchmark runner, external comparison harness, hardware profiler with bandwidth calibration |
| Native backend | C++20/OpenMP library with AVX2/AVX512 dispatch, optional pybind11 bridge, guarded Triton RMSNorm/SwiGLU/QKV/RoPE/decode-attention kernels, Triton FP16 matmul, multi-batch page-table Triton decode attention, optional CUDA fused-kernel, FP16 WMMA Tensor Core matmul, reduced paged-attention, and CUDA graph planning targets |

## Complete Tech Stack

| Area | Stack |
| --- | --- |
| Primary languages | Python 3.10+, C++20 |
| Python packaging | `pyproject.toml`, setuptools, editable installs, optional dependency groups |
| Core numerical runtime | NumPy reference kernels |
| Compiler IR | Custom CacheIR graph IR, JSON artifacts, text IR dumps, pass diffs |
| Model import | Hugging Face `config.json`, NPZ reference weights, safetensors optional, ONNX optional, GGUF metadata and dense F32/F16/BF16/I8/I16/I32/I64/F64 plus Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 native subset, optional `gguf` reference dequantization for supported K/IQ/TQ/NV/MX formats, StableHLO textual/region subset |
| Transformer architecture | Llama/Mistral/Qwen-style decoder-only blocks, RMSNorm, RoPE, grouped-query attention, SwiGLU, residual streams |
| Compiler passes | Shape inference, constant folding, QKV fusion, RMSNorm+QKV+RoPE fusion, SwiGLU fusion, prefill/decode specialization, layout conversion, quant-aware lowering, DCE, hardware hints, kernel selection, execution scheduling, static memory planning |
| Runtime systems | Weight loader, tokenizer bridge, paged KV cache, prefix-cache snapshots, calibrated spillover policy hooks, backend dispatcher, greedy streaming decode loop |
| CPU backend | NumPy executable backend; C++20/OpenMP native library with scalar, AVX2/FMA, and AVX512 dispatch; optional pybind11 module `_cacheir_native`; runtime dispatches to native kernels when available |
| GPU backend surface | CUDA/Triton target naming, schedule generation, guarded Triton RMSNorm, SwiGLU, FP16 matmul, fused RMSNorm/QKV/RoPE, single-query decode attention, and multi-batch page-table decode attention kernels; optional CUDA C++ fused-kernel, FP16 WMMA Tensor Core matmul, reduced paged-attention, and CUDA graph capture planning target |
| Accelerator adapters | Optional CUTLASS, FlashAttention, and FlashInfer probes/dispatch contracts plus guarded direct execution wrappers for prefill, single decode, and batch paged decode; CUTLASS detects the `nvidia-cutlass`/`cutlass_cppgen` wheel when installed |
| Quantization | int4/int8-style graph lowering plus CPU-side quantize/dequantize simulation |
| Serving | FastAPI and Uvicorn optional dependencies, OpenAI-compatible `/v1/models`, `/v1/completions`, `/v1/chat/completions` |
| Benchmarks | Built-in benchmark CLI, prefill/decode split metrics, benchmark matrix script, comparison harness for vLLM, llama.cpp, IREE, and TVM commands, plus installed IREE/TVM smoke benchmark execution |
| Visualization | HTML export, Graphviz DOT export, text IR export, MLIR-style CacheIR dialect export, parser round trip, and verifier |
| Native build | CMake, Ninja, OpenMP, pybind11 optional |
| Testing | pytest, Python bytecode compilation checks, CLI smoke tests, CMake build checks |
| Infrastructure | Dockerfile, Makefile, GitHub Actions |
| Documentation | Markdown docs and full LaTeX project documentation in `docs/latex/` |
| External context | vLLM, llama.cpp, IREE, TVM, StableHLO, MLIR, CUTLASS, FlashAttention/FlashInfer are comparison or optional integration surfaces, not default runtime dependencies |

## Install

```bash
python -m pip install -e ".[dev,server]"
```

Optional importer dependencies:

```bash
python -m pip install -e ".[importers,benchmark]"
```

Optional native/GPU dependency groups:

```bash
python -m pip install -e ".[native,gpu]"
cmake -S cpp -B cpp/build -DCACHEIR_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release
```

Optional accelerator adapter probes:

```bash
python -m pip install -e ".[accelerators]"
```

Optional upstream compiler/runtime comparison tools:

```bash
python -m pip install -e ".[upstream]"
```

On native Windows/Python 3.13, IREE and TVM install from wheels while vLLM,
FlashAttention, FlashInfer, and `llama-cpp-python` do not all provide compatible
wheels for this host. The CUDA validation path for those packages is WSL2:
`/home/bhatt/cacheir-llm-venv` has vLLM, FlashInfer, Triton, Torch, and
`llama-cpp-python`; `/home/bhatt/flash-attn-venv` has FlashAttention; and
`/home/bhatt/cacheir-tools/llama.cpp/build-cuda13/bin` has a CUDA llama.cpp
build. CacheIR keeps these adapters and benchmark runners guarded so compatible
Linux/CUDA environments can execute them directly.

If CMake discovers the wrong Python on Windows, pin the active interpreter:

```bash
cmake -S cpp -B cpp/build-py-active \
  -DCACHEIR_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-py-active --config Release
```

CUDA sources are enabled with `-DCACHEIR_BUILD_CUDA=ON`. On Windows, NVIDIA CUDA
requires the MSVC host compiler (`cl.exe`) on `PATH`; Linux builds can use the
normal CUDA host compiler flow. This workspace has Build Tools installed at
`C:\BuildTools` and `cl.exe` on the user PATH.

## Quickstart

Create a tiny Llama-shaped model, compile it, inspect the lowered graph, run
generation, and benchmark prefill/decode:

```bash
cacheir make-tiny examples/tiny_model
cacheir compile examples/tiny_model --output examples/tiny_artifact
cacheir inspect examples/tiny_artifact --mode decode
cacheir run examples/tiny_artifact --prompt "CacheIR" --max-new-tokens 16
cacheir benchmark examples/tiny_artifact --decode-tokens 32 --repeats 3
cacheir export examples/tiny_artifact examples/decode.html --mode decode
cacheir export examples/tiny_artifact examples/decode.mlir --mode decode --format mlir
```

The artifact directory contains:

```text
artifact.json              full machine-readable compiler artifact
graphs/prefill.cir         optimized prefill IR
graphs/decode.cir          optimized decode IR
schedules/*.json           runtime kernel schedule by mode
passes/*.diff              pass-by-pass graph diffs
```

## Python API

```python
from cacheir import Runtime, compile_model

artifact = compile_model(
    model_path="examples/tiny_model",
    target="cpu",
    quant=None,
    mode=["prefill", "decode"],
    max_seq=128,
)

rt = Runtime(artifact)
for token in rt.generate("Explain MLIR in simple terms", max_new_tokens=16):
    print(token, end="")
```

## Compiler Example

Logical transformer IR starts as separate operations:

```text
%attn_norm = rms_norm(%hidden, %attn_norm_w)
%q = matmul(%attn_norm, %q_proj_w)
%k = matmul(%attn_norm, %k_proj_w)
%v = matmul(%attn_norm, %v_proj_w)
%q_rope, %k_rope = rope(%q, %k)
%attn = grouped_query_attention(%q_rope, %k_rope, %v)
```

The decode-specialized graph lowers that into scheduled runtime calls:

```text
%q_rope, %k_rope, %v = fused_rmsnorm_qkv_rope(...)
%attn = paged_attention_decode(%q_rope, %k_rope, %v)
```

Every pass records before/after text and a unified diff:

```bash
cacheir inspect examples/tiny_artifact \
  --mode decode \
  --pass-name prefill_decode_specialization
```

## CLI

```bash
cacheir profile
cacheir profile --calibrate --sample-mb 16 --repeats 5
cacheir make-tiny MODEL_DIR
cacheir compile MODEL_DIR --target cpu --quant int4_awq --output ARTIFACT_DIR
cacheir compile model.stablehlo --target cpu --output stablehlo_artifact
cacheir inspect ARTIFACT_DIR --mode decode
cacheir export ARTIFACT_DIR graph.dot --format dot
cacheir export ARTIFACT_DIR graph.mlir --format mlir
cacheir benchmark ARTIFACT_DIR --prompt "hello" --decode-tokens 64 --repeats 5
cacheir external --benchmark --workdir .tmp/upstream
cacheir run ARTIFACT_DIR --prompt "hello"
cacheir serve ARTIFACT_DIR --host 127.0.0.1 --port 8000
```

## Benchmarks

CacheIR reports prefill and decode separately because they stress different parts
of the system:

```bash
python scripts/benchmark_matrix.py --output benchmark_results.json --repeats 3 --decode-tokens 16
```

```json
{
  "prompt_tokens": 24,
  "decode_tokens": 32,
  "prefill_ms_avg": 4.2,
  "decode_ms_avg": 0.3,
  "prefill_tokens_per_s": 5714.0,
  "decode_tokens_per_s": 3333.0,
  "kv_cache": {"page_size": 16, "layers": {...}}
}
```

The CPU runtime keeps NumPy correctness coverage and dispatches to native C++
kernels when `_cacheir_native` is available. CUDA kernels plug into the same
artifact and benchmark surfaces when built with a compatible CUDA host toolchain.

For apples-to-apples local comparisons, provide explicit external commands for
the tools installed on the machine:

```bash
python scripts/compare_external_benchmarks.py examples/tiny_artifact \
  --vllm-command "python bench_vllm.py" \
  --llama-command "llama-bench -m model.gguf" \
  --iree-command "iree-benchmark-module --module=model.vmfb" \
  --tvm-command "python bench_tvm.py" \
  --output benchmark_comparison.json
```

When IREE and TVM are installed but no explicit commands are supplied, CacheIR can
run built-in upstream smoke benchmarks:

```bash
python scripts/compare_external_benchmarks.py examples/tiny_artifact \
  --run-installed-smoke \
  --output benchmark_comparison.json
```

Latest local native validation on 2026-06-24:

- Full optional project install passed with `python -m pip install -e ".[dev,server,importers,benchmark,native,gpu]"`.
- PyPI packages installed: `pybind11`, `triton-windows`, ONNX/importer deps, server deps, benchmark deps.
- MSVC Build Tools 2022 installed at `C:\BuildTools`; `cl.exe` resolves from the user PATH.
- CUDA Torch installed: `torch 2.12.1+cu130`; CUDA is available on the RTX 5070 Laptop GPU.
- `_cacheir_native.pyd` built with CMake and the active Python 3.13 interpreter.
- Native SIMD probe reported `avx512`.
- Native RMSNorm and matmul matched NumPy within float32 tolerance.
- CUDA C++ target built with MSVC/NVCC: `cacheir_cuda_kernels.lib`.
- CUDA C++ FP16 WMMA Tensor Core matmul and reduced paged-attention decode paths built with MSVC/NVCC.
- Triton GPU kernels executed on CUDA and matched Torch references for RMSNorm, SwiGLU, FP16 matmul, single-query decode attention, and multi-batch page-table decode attention.
- `cacheir profile --calibrate --sample-mb 1 --repeats 1` measured CPU copy at 8.05 GB/s and CUDA H2D at 2.85 GB/s on this host.
- `nvidia-cutlass 4.2.0.0` installed and the CUTLASS adapter probe detects `cutlass_cppgen`.
- `iree-base-compiler 3.11.0`, `iree-base-runtime 3.11.0`, and `apache-tvm 0.25.0` installed from PyPI.
- `cacheir external --benchmark --workdir .tmp/upstream` compiled a StableHLO add module through IREE to a 9,781-byte VMFB and ran `iree-benchmark-module`; TVM built and ran a TE vector-add benchmark with checksum 512.0.
- WSL2 CUDA environments validated direct FlashAttention prefill and FlashInfer decode execution through CacheIR's adapter wrappers.
- A CUDA llama.cpp build ran a real `llama-bench` model benchmark against a locally converted GGUF tiny Llama model; the JSON result is in `.tmp/upstream/llama_cpp_tiny_benchmark.json`.
- The vLLM model benchmark runner launched against the same tiny HF model, but the local WSL vLLM 0.23.0 CUDA worker failed during initialization with `RuntimeError: UVA is not available`; the failure metadata is captured in `.tmp/upstream/vllm_tiny_benchmark.json`.
- Benchmark matrix ran 18 rows with 3 repeats and 16 decode tokens.
- The comparison harness can now run installed IREE/TVM smoke benchmarks plus model-aware vLLM and llama.cpp benchmark helpers when model paths are supplied.

## Development

```bash
python -m pytest -q
python -m compileall cacheir -q
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release
```

Full project documentation is available as LaTeX source at
`docs/latex/cacheir_project_documentation.tex`. Generated PDF/ODF outputs and
LaTeX build products are ignored by `.gitignore`.

Project layout:

```text
cacheir/
  ir.py                  CacheIR graph IR
  compiler.py            importer selection and pass pipeline
  importers/             HF, ONNX, GGUF, StableHLO, tiny model support
  passes/                compiler passes
  runtime/               artifact, tokenizer, KV cache, CPU runtime, server
  backends/              backend registry, optional native bridge, Triton kernels, upstream probes
cpp/                     C++20/OpenMP AVX backend and optional pybind11 module
docs/                    architecture, IR, runtime, benchmark notes
examples/                tiny model compile demo
tests/                   pytest coverage
```

## Honest Boundaries

Implemented and tested:

- CPU reference execution without PyTorch
- prefill/decode graph specialization
- paged KV-cache reference behavior
- prefix-cache reuse and spillover policy experiments
- artifact bundles, pass diffs, graph export, benchmarks, server entrypoint
- quantization-aware lowering with CPU-side quantize/dequantize simulation
- C++20 backend library build with scalar, AVX2/FMA, and AVX512 dispatch
- optional pybind11 bridge for native RMSNorm and matmul kernels, with runtime dispatch when `_cacheir_native` is importable
- guarded Triton RMSNorm, SiLU/SwiGLU, fused RMSNorm/QKV/RoPE, and single-query decode attention kernels
- guarded Triton FP16 matmul kernel using `tl.dot`/Tensor Core lowering where available
- guarded Triton multi-batch page-table decode attention with GQA mapping
- optional CUDA C++ fused kernels for RMSNorm, SwiGLU, RMSNorm/QKV/RoPE, FP16 WMMA Tensor Core matmul, batched paged-attention ABI, and reduced paged-attention decode
- CUDA graph capture planning for decode replay loops
- native GGUF dense F32/F16/BF16/I8/I16/I32/I64/F64 plus Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 tensor reads
- optional reference GGUF dequantization for K/IQ/TQ/NV/MX formats exposed by the `gguf` package
- StableHLO text importer with arithmetic, shape, broadcast, slice, reduce-region, and cast coverage
- MLIR-style CacheIR dialect emitter, parser round trip, and verifier
- optional upstream MLIR C++ dialect registration skeleton through `CACHEIR_BUILD_MLIR=ON`
- CUTLASS, FlashAttention, and FlashInfer adapter probes, dispatch contracts, and guarded direct wrappers for prefill, single decode, and batch paged decode
- direct FlashAttention and FlashInfer smoke execution on WSL2 CUDA environments with compatible wheels
- calibrated low-VRAM KV spillover cost model with measured bandwidth calibration, transfer estimates, and resident-page budgeting
- IREE StableHLO compile/runtime benchmark integration through upstream IREE wheels
- TVM TE/TIR runtime benchmark integration through the upstream TVM wheel
- external benchmark comparison harness for vLLM, llama.cpp, IREE, and TVM, including installed IREE/TVM smoke runs and model-aware vLLM/llama.cpp helpers

No active in-repository milestone remains open. The only remaining caveat is
environment-specific: this WSL2 vLLM install is present and the real model
benchmark runner executes, but the local CUDA worker fails before inference
because UVA is unavailable on this host.

## License

Apache-2.0.
