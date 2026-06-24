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
machine-readable format.
