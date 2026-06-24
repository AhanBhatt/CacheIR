from cacheir.backends.adapters import (
    AdapterStatus,
    AdapterSmokeResult,
    available_adapters,
    flash_attention_prefill,
    flashinfer_batch_paged_decode,
    flashinfer_paged_decode,
    probe_adapters,
    run_adapter_smokes,
    run_flash_attention_smoke,
    run_flashinfer_decode_smoke,
    select_attention_backend,
)
from cacheir.backends.cuda_graphs import CudaGraphCapturePlan, cuda_graph_capture_available, plan_decode_cuda_graph
from cacheir.backends.registry import Backend, BackendRegistry, default_registry
from cacheir.backends.upstream import (
    compile_stablehlo_with_iree,
    probe_external_systems,
    run_installed_upstream_benchmarks,
    run_llama_cpp_benchmark,
    run_vllm_latency_benchmark,
)
from cacheir.backends.native import available as native_available, simd_backend

__all__ = [
    "AdapterStatus",
    "AdapterSmokeResult",
    "Backend",
    "BackendRegistry",
    "CudaGraphCapturePlan",
    "available_adapters",
    "cuda_graph_capture_available",
    "default_registry",
    "flash_attention_prefill",
    "flashinfer_batch_paged_decode",
    "flashinfer_paged_decode",
    "compile_stablehlo_with_iree",
    "native_available",
    "plan_decode_cuda_graph",
    "probe_adapters",
    "probe_external_systems",
    "run_adapter_smokes",
    "run_flash_attention_smoke",
    "run_flashinfer_decode_smoke",
    "run_installed_upstream_benchmarks",
    "run_llama_cpp_benchmark",
    "run_vllm_latency_benchmark",
    "select_attention_backend",
    "simd_backend",
]
