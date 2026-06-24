# Benchmarks

CacheIR benchmarks report prefill and decode separately:

```bash
cacheir benchmark examples/tiny_artifact --prompt "CacheIR benchmark" --decode-tokens 32 --repeats 3
```

Output fields:

- `prompt_tokens`
- `decode_tokens`
- `prefill_ms_avg`
- `decode_ms_avg`
- `prefill_tokens_per_s`
- `decode_tokens_per_s`
- `kv_cache`

The first backend is a reference CPU backend, so these numbers are for regression
tracking and pass attribution. Native C++/CUDA kernels should be benchmarked against
the same artifact and CLI surface.
