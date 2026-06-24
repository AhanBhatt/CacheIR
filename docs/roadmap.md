# Roadmap

Implemented:

- HF config + NPZ/safetensors shape import
- ONNX graph import skeleton
- GGUF metadata parser subset and native dense F32/F16/BF16/I8/I16/I32/I64/F64 plus Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 tensor reads
- GGUF K/IQ/TQ/NV/MX reference dequantization through the optional `gguf` package where upstream exposes a dequantizer
- StableHLO textual importer for arithmetic, shape, broadcast, slice, reduce-region, and cast experiments
- CacheIR JSON/text IR
- MLIR-style CacheIR dialect export, parser round trip, and verifier
- pass traces and diffs
- prefill/decode graph specialization
- QKV, RMSNorm+QKV+RoPE, and SwiGLU fusion
- memory planning and execution schedules
- paged KV-cache reference runtime
- prefix-cache reuse, calibrated spillover policy experiments, and measured bandwidth calibration
- OpenAI-compatible local server
- benchmark and graph export CLI
- external benchmark comparison harness for vLLM, llama.cpp, IREE, and TVM commands plus installed IREE/TVM smoke runs
- C++20/OpenMP backend with scalar, AVX2/FMA, and AVX512 dispatch
- optional pybind11 bridge for C++ RMSNorm and matmul kernels
- guarded Triton RMSNorm, SiLU/SwiGLU, FP16 matmul, fused RMSNorm/QKV/RoPE, single-query decode attention, and multi-batch page-table decode attention kernels
- optional CUDA C++ fused kernels for RMSNorm, SwiGLU, RMSNorm/QKV/RoPE, FP16 WMMA Tensor Core matmul, batched paged-attention ABI, and reduced block-level paged-attention decode path
- CUDA graph capture planning for decode replay loops
- CUTLASS, FlashAttention, and FlashInfer adapter probes plus guarded direct execution wrappers for prefill, single decode, and batch paged decode
- direct FlashAttention and FlashInfer smoke execution on WSL2 CUDA environments with compatible wheels
- `cacheir profile --calibrate` bandwidth measurement for spillover cost models
- IREE StableHLO compile/runtime benchmark integration through `iree-base-compiler` and `iree-base-runtime`
- TVM TE/TIR runtime benchmark integration through `apache-tvm`
- model-aware vLLM and llama.cpp benchmark helpers for installed systems and local model artifacts
- real llama.cpp CUDA `llama-bench` model benchmark against a locally converted GGUF tiny Llama model
- upstream MLIR C++ dialect registration skeleton behind `CACHEIR_BUILD_MLIR=ON`

No active in-repository milestone remains open. The one caveat is
environment-specific rather than missing project work: this WSL2 vLLM install is
present and the real model benchmark helper launches, but the local CUDA worker
fails before inference with `RuntimeError: UVA is not available`.
