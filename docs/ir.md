# CacheIR IR

The IR is deliberately small. It represents decoder-only transformer inference
without importing a general-purpose tensor framework.

Core ops include:

- `token_embedding`
- `rms_norm`
- `matmul`
- `qkv_projection`
- `rope`
- `grouped_query_attention`
- `paged_attention_prefill`
- `paged_attention_decode`
- `fused_rmsnorm_qkv_rope`
- `fused_swiglu`
- `quantized_matmul`

Each `Graph` stores logical inputs and outputs, tensor types, weight specs, nodes,
pass traces, memory plans, and execution schedules.

The text format is meant for humans and diffs. The JSON artifact is the stable
machine-readable format. An experimental MLIR-style CacheIR dialect can be emitted
for compiler experiments:

```bash
cacheir export artifact_dir graph.mlir --mode decode --format mlir
```

The textual dialect emitter is intentionally descriptive. It preserves op names,
operands, results, attrs, and graph outputs so passes can be inspected in a
familiar compiler syntax while CacheIR keeps its own small JSON artifact as the
source of truth. For upstream MLIR experiments, `cpp/mlir/` adds a minimal C++
dialect registration skeleton behind `CACHEIR_BUILD_MLIR=ON`; it lets an MLIR
tooling build register the `cacheir` namespace while still allowing unknown ops
and types during early dialect iteration.
