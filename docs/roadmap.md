# Roadmap

Implemented:

- HF config + NPZ/safetensors shape import
- ONNX graph import skeleton
- GGUF metadata parser subset
- CacheIR JSON/text IR
- pass traces and diffs
- prefill/decode graph specialization
- QKV, RMSNorm+QKV+RoPE, and SwiGLU fusion
- memory planning and execution schedules
- paged KV-cache reference runtime
- OpenAI-compatible local server
- benchmark and graph export CLI
- C++20 backend skeleton

Next native-kernel milestones:

- pybind11 bridge for C++ CPU kernels
- AVX2/AVX512 matmul and RMSNorm kernels
- Triton decode attention
- CUDA graph capture for decode loops
- prefix-cache reuse policy benchmarks
- StableHLO importer experiments
