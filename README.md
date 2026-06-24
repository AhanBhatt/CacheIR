# CacheIR

CacheIR is a narrow, inspectable compiler and runtime for decoder-only transformer
inference. It imports a Llama/Mistral/Qwen-style model, lowers it into its own IR,
runs transformer-specific optimization passes, plans memory and execution, selects
backend kernels, and executes inference through a reference runtime.

It is not a PyTorch wrapper and it is not a vLLM clone. The current runtime backend
is a NumPy CPU correctness backend, with C++20/OpenMP and Triton/CUDA target surfaces
laid out for native kernels.

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
| Model import | Hugging Face config + NPZ/safetensors metadata, ONNX graph skeleton, GGUF metadata subset |
| IR | CacheIR JSON/text graph format, tensor types, weight specs, attrs, pass traces |
| Compiler passes | shape inference, constant folding, QKV fusion, RMSNorm+QKV+RoPE fusion, SwiGLU fusion, prefill/decode specialization, layout conversion, quant-aware lowering, hardware hints, kernel selection, scheduling, memory planning |
| Runtime | NumPy CPU backend, tokenizer bridge, paged KV cache, greedy streaming generation |
| Serving | OpenAI-compatible local FastAPI server |
| Tooling | CLI, artifact bundles, graph HTML/DOT export, benchmark runner, hardware profiler |
| Native backend | C++20/OpenMP skeleton builds; Triton target metadata is present |

## Complete Tech Stack

| Area | Stack |
| --- | --- |
| Primary languages | Python 3.10+, C++20 |
| Python packaging | `pyproject.toml`, setuptools, editable installs, optional dependency groups |
| Core numerical runtime | NumPy reference kernels |
| Compiler IR | Custom CacheIR graph IR, JSON artifacts, text IR dumps, pass diffs |
| Model import | Hugging Face `config.json`, NPZ reference weights, safetensors optional, ONNX optional, GGUF metadata subset |
| Transformer architecture | Llama/Mistral/Qwen-style decoder-only blocks, RMSNorm, RoPE, grouped-query attention, SwiGLU, residual streams |
| Compiler passes | Shape inference, constant folding, QKV fusion, RMSNorm+QKV+RoPE fusion, SwiGLU fusion, prefill/decode specialization, layout conversion, quant-aware lowering, DCE, hardware hints, kernel selection, execution scheduling, static memory planning |
| Runtime systems | Weight loader, tokenizer bridge, paged KV cache, backend dispatcher, greedy streaming decode loop |
| CPU backend | NumPy executable backend today; C++20/OpenMP static-library skeleton for native kernels |
| GPU backend surface | CUDA/Triton target naming, kernel metadata, schedule generation; executable Triton/CUDA kernels are planned |
| Quantization | int4/int8-style graph lowering plus CPU-side quantize/dequantize simulation |
| Serving | FastAPI and Uvicorn optional dependencies, OpenAI-compatible `/v1/models`, `/v1/completions`, `/v1/chat/completions` |
| Benchmarks | Built-in benchmark CLI, prefill/decode split metrics, benchmark matrix script |
| Visualization | HTML export, Graphviz DOT export, text IR export |
| Native build | CMake, Ninja, OpenMP |
| Testing | pytest, Python bytecode compilation checks, CLI smoke tests, CMake build checks |
| Infrastructure | Dockerfile, Makefile, GitHub Actions |
| Documentation | Markdown docs and full LaTeX project documentation in `docs/latex/` |
| Future integrations | pybind11, AVX2/AVX512 kernels, CUTLASS, FlashAttention/FlashInfer, StableHLO, MLIR, IREE, TVM comparison |

## Install

```bash
python -m pip install -e ".[dev,server]"
```

Optional importer dependencies:

```bash
python -m pip install -e ".[importers,benchmark]"
```

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
cacheir make-tiny MODEL_DIR
cacheir compile MODEL_DIR --target cpu --quant int4_awq --output ARTIFACT_DIR
cacheir inspect ARTIFACT_DIR --mode decode
cacheir export ARTIFACT_DIR graph.dot --format dot
cacheir benchmark ARTIFACT_DIR --prompt "hello" --decode-tokens 64 --repeats 5
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

The NumPy backend is for correctness and regression tracking. Native C++/CUDA
kernels should plug into the same artifact and benchmark surfaces.

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
  importers/             HF, ONNX, GGUF, tiny model support
  passes/                compiler passes
  runtime/               artifact, tokenizer, KV cache, CPU runtime, server
  backends/              backend registry and target metadata
cpp/                     C++20 backend skeleton
docs/                    architecture, IR, runtime, benchmark notes
examples/                tiny model compile demo
tests/                   pytest coverage
```

## Honest Boundaries

Implemented and tested:

- CPU reference execution without PyTorch
- prefill/decode graph specialization
- paged KV-cache reference behavior
- artifact bundles, pass diffs, graph export, benchmarks, server entrypoint
- quantization-aware lowering with CPU-side quantize/dequantize simulation
- C++20 backend library build

Planned native-kernel work:

- pybind11 bridge for C++ CPU kernels
- AVX2/AVX512 optimized kernels
- Triton/CUDA fused kernels
- native GGUF tensor execution
- StableHLO and MLIR dialect experiments

## License

Apache-2.0.
