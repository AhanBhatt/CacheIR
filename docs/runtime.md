# Runtime

The runtime loads a `CompileArtifact`, creates a tokenizer bridge, loads weights,
allocates a paged KV cache, and dispatches each scheduled node to reference CPU
kernels.

The CPU backend is a correctness backend. It uses NumPy kernels and simulates int4
or int8 weight quantization by quantizing and dequantizing weights once on load.

The KV cache exposes page metadata even though the reference arrays are contiguous.
Future CUDA kernels can consume the same page-table contract.

Run locally:

```bash
cacheir make-tiny examples/tiny_model
cacheir compile examples/tiny_model --output examples/tiny_artifact
cacheir run examples/tiny_artifact --prompt "CacheIR" --max-new-tokens 16
cacheir serve examples/tiny_artifact
```
