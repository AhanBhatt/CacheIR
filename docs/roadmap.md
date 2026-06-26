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
- prefix-cache reuse with hit/miss counters, calibrated spillover policy experiments, and measured bandwidth calibration
- forked per-request KV sessions sharing weights/tokenizer state
- continuous-batch scheduler with prefix reuse metrics, queue limits, priorities, cancellation, server metrics, CPU scheduler benchmark, and CUDA scheduler benchmark
- OpenAI-compatible local server with `/healthz`, `/metrics`, streaming chat completions, and CacheIR batch completions
- benchmark, scheduler benchmark, CUDA runtime benchmark, GPU kernel benchmark, and graph export CLI
- external benchmark comparison harness for vLLM, llama.cpp, IREE, and TVM commands plus installed IREE/TVM smoke runs
- C++20/OpenMP backend with scalar, AVX2/FMA, and AVX512 dispatch
- optional pybind11 bridge for C++ RMSNorm, matmul, and fused SiLU-multiply kernels
- native matmul runtime policy with NumPy default and opt-in native force/auto modes
- guarded Triton RMSNorm, SiLU/SwiGLU, FP16 matmul, fused RMSNorm/QKV/RoPE, single-query decode attention, and multi-batch page-table decode attention kernels
- end-to-end CUDA runtime execution for full decoder graphs with fp16 weights, GPU-resident KV state, SDPA attention dispatch, Triton elementwise dispatch gates, and cached packed QKV/Gate-Up weights
- shared CUDA page allocator accounting across forked request sessions
- scheduler-integrated CUDA batching for variable-length padded prefill and active decode rounds across independent request sessions
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
- vLLM no-UVA CUDA worker fallback for CacheIR-launched vLLM benchmarks, including project-root propagation into spawned workers
- CUDA Toolkit 13.3 WSL install and CacheIR launcher CUDA path export for FlashInfer JIT
- repeatable GPU kernel benchmark script with workspace-local Triton cache
- repeatable CUDA runtime benchmark script showing CPU/CUDA prefill and decode crossover points
- repeatable CUDA scheduler benchmark script showing sequential-session versus batched-prefill/decode throughput
- runnable WSL2 CUDA vLLM comparison with vLLM 0.23.0, Torch 2.11.0+cu130, FlashInfer, CUDA Toolkit 13.3, and a CacheIR-created tiny Llama h128 model

Remaining work for an honest 8/10 production-serving and raw-performance score:

- Persistent page-backed shared GPU KV storage used directly by the scheduler's
  fused decode kernels, beyond the current shared allocator accounting.
- Fused multi-request paged attention using that persistent shared GPU page pool
  instead of repacking per-request K/V tensors for reference kernels.
- Packed int4/int8 quantized GEMM with real compression, scales, zero points, and
  model-loader integration.
- Tuned GEMM paths through CUTLASS/cuBLASLt/Triton autotuning and shape-specific
  kernel selection.
- Production continuous batching hardening: streaming backpressure, preemption,
  fairness policies, bounded latency tests, and long-running queue soak tests.
- Large-model coverage and benchmarks on 0.5B, 1.5B, and 7B-class local models.
- Real apples-to-apples throughput and latency comparisons against vLLM,
  llama.cpp, TensorRT-LLM, MLC LLM, IREE, and TVM where those systems are
  installed.
- Reliability hardening: memory limits, OOM recovery, tokenizer edge cases,
  model-family conformance, and long-context soak tests.
