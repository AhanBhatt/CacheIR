# CacheIR Architecture

CacheIR has five layers.

1. Importers recover transformer metadata and build a logical graph.
2. The IR stores logical ops, tensor types, layouts, memory plans, and schedules.
3. Passes optimize the graph and specialize prefill/decode.
4. Backends select kernels for CPU or GPU-oriented targets.
5. The runtime executes the compiled graph and owns tokenizer, KV-cache, and server loops.

The reference execution backend is NumPy CPU. It is intentionally simple so pass
behavior is easy to validate. The C++20 backend skeleton and Triton target metadata
are present for native kernels.

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
