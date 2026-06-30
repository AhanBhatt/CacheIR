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
- direct FlashInfer smoke execution on WSL2 CUDA, with FlashAttention kept as a
  guarded adapter for environments that provide a compatible wheel
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
- persistent page-backed shared GPU KV storage used directly by CacheIR's Triton
  single-request and multi-request paged decode kernels
- fused multi-request paged attention over the persistent shared GPU page pool
  without per-request K/V repacking
- compact per-layer physical KV page slots behind globally unique logical page IDs
- packed int4/int8 quantized weights with real `uint8` compression, per-row
  scales, affine zero points, model-loader integration, and CPU/CUDA
  dequant-at-GEMM-boundary execution
- shape-specific GEMM plan recording with cuBLASLt-through-Torch default dispatch
  and guarded opt-in Triton Tensor Core matmul
- continuous batching hardening with blocking backpressure attempts, fairness
  aging, resumable preemption, request token limits, queue-wait accounting, and
  bounded-latency counters
- CUDA memory-limit checks, OOM recovery cleanup, tokenizer empty/invalid-ID edge
  handling, and BF16 safetensors loading for Qwen-style models
- Qwen2.5-0.5B-Instruct compile and CUDA inference smoke with persistent paged
  decode attention on the local RTX 5070 Laptop GPU
- apples-to-apples Qwen2.5-0.5B CacheIR/vLLM latency run for 16 input + 8 output
  tokens in WSL2 CUDA
- real llama.cpp CUDA `llama-bench` JSON comparison against the local tiny GGUF
  model
- TensorRT-LLM and MLC LLM probe surfaces in the external comparison harness

Environment-limited coverage:

- Qwen2.5-1.5B and 7B-class weights were not already local. The implemented
  importer/runtime path is model-family compatible, and the benchmark harness can
  run them when local artifacts are present, but they were not downloaded in this
  pass after the 0.5B checkpoint download took several minutes unauthenticated.
- FlashInfer is installed and executed in WSL2 CUDA. FlashAttention,
  TensorRT-LLM, and MLC LLM remain guarded optional integrations in this
  environment because binary-only PyPI checks did not expose compatible wheels;
  source-build attempts for FlashAttention/TensorRT-LLM were not retained after
  destabilizing WSL.
