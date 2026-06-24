# CacheIR Architecture

CacheIR has five layers.

1. Importers recover transformer metadata and build a logical graph.
2. The IR stores logical ops, tensor types, layouts, memory plans, and schedules.
3. Passes optimize the graph and specialize prefill/decode.
4. Backends select kernels for CPU or GPU-oriented targets.
5. The runtime executes the compiled graph and owns tokenizer, KV-cache, and server loops.

The reference execution backend is NumPy CPU. It is intentionally simple so pass
behavior is easy to validate. Native experiments live beside it: a C++20/OpenMP
backend with scalar, AVX2/FMA, and AVX512 dispatch; an optional pybind11 module;
guarded Triton kernels for RMSNorm, SwiGLU, fused RMSNorm/QKV/RoPE, and
FP16 matmul, single-query plus multi-batch page-table decode attention; optional CUDA C++
fused-kernel, FP16 WMMA Tensor Core matmul, reduced paged-attention, and CUDA graph planning sources; and
guarded CUTLASS, FlashAttention, and FlashInfer adapter contracts and wrappers.

## Decode-First Design

Prefill and decode are compiled as separate graphs. The decode graph marks attention
as `paged_attention_decode`, uses smaller KV-cache pages, and labels attention as
KV-bandwidth-bound in the schedule. Prefill keeps full-sequence causal attention and
is optimized as a compute-heavy graph.

## Artifact Layout

`cacheir compile MODEL --output artifact_dir` creates a bundle:

- `artifact.json`: full machine-readable compiler artifact
- `graphs/*.cir`: optimized IR text
- `schedules/*.json`: runtime schedule with kernel and cost estimates
- `passes/*.diff`: pass-by-pass IR diffs

Graphs can also be exported as HTML, Graphviz DOT, text IR, or an experimental
MLIR-style CacheIR dialect:

```bash
cacheir export artifact_dir graph.mlir --mode decode --format mlir
```

For upstream MLIR experiments, `cpp/mlir/` also contains a minimal C++ dialect
registration library. It is optional and builds only when CMake is configured
with `-DCACHEIR_BUILD_MLIR=ON` and `MLIR_DIR` points at an upstream
`MLIRConfig.cmake`.

## Import and Lowering Experiments

The main path remains Hugging Face-style decoder-only transformer import. CacheIR
also includes deliberately narrow experiments for systems work:

- GGUF metadata and native dense F32/F16/BF16/I8/I16/I32/I64/F64 plus Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 tensor reads for direct tensor execution.
- GGUF K/IQ/TQ/NV/MX reference dequantization through the optional `gguf` package when upstream exposes a dequantizer.
- StableHLO textual import for arithmetic, shape, broadcast, slice, reduce-region, and cast experiments.
- MLIR-style CacheIR dialect emission, parser round trip, verifier, and optional upstream C++ dialect registration skeleton for inspecting lowered graphs.
- IREE StableHLO compilation/runtime benchmarking through installed upstream IREE wheels.
- TVM TE/TIR runtime benchmarking through the installed upstream TVM wheel.
- Calibrated low-VRAM spillover policy experiments that budget resident pages from free memory, page bytes, measured bandwidth, and transfer costs.
- External benchmark comparison hooks for vLLM, llama.cpp, IREE, and TVM runs, including model-aware vLLM/llama.cpp helpers when local artifacts are available.
