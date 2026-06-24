# Runtime

The runtime loads a `CompileArtifact`, creates a tokenizer bridge, loads weights,
allocates a paged KV cache, and dispatches each scheduled node to reference CPU
kernels.

The CPU backend is a correctness backend. It uses NumPy kernels and simulates int4
or int8 weight quantization by quantizing and dequantizing weights once on load.

The KV cache exposes page metadata even though the reference arrays are contiguous.
Prefix-cache snapshots and spillover policy markers let decode scheduling
experiments reuse prefixes, mark resident pages, and model CPU/GPU spill behavior.
The spillover policy can now derive a resident-page budget from page bytes, free
GPU memory, safety margin, measured PCIe/CPU bandwidth, and transfer latency. CUDA and Triton
kernels consume the same page-table contract for single-query and multi-batch
decode attention experiments.

Native execution hooks are optional. If `_cacheir_native` is built with
`CACHEIR_BUILD_PYTHON=ON`, `cacheir.backends.native` exposes C++ RMSNorm and
out-by-in matmul kernels and reports the selected SIMD path (`scalar`, `avx2`, or
`avx512`). GGUF runtime loading can read dense F32/F16/BF16/I8/I16/I32/I64/F64
plus Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 tensors directly from `.gguf` files. It can
also delegate K/IQ/TQ/NV/MX dequantization to the optional `gguf` reference
package for formats upstream supports.

Run locally:

```bash
cacheir make-tiny examples/tiny_model
cacheir compile examples/tiny_model --output examples/tiny_artifact
cacheir run examples/tiny_artifact --prompt "CacheIR" --max-new-tokens 16
cacheir serve examples/tiny_artifact
```
