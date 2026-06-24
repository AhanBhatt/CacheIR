from __future__ import annotations

import importlib
import importlib.util
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    package: str
    available: bool
    capability: str
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "package": self.package,
            "available": self.available,
            "capability": self.capability,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AdapterSmokeResult:
    name: str
    available: bool
    executed: bool
    elapsed_s: float | None = None
    output_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "executed": self.executed,
            "elapsed_s": self.elapsed_s,
            "output_shape": self.output_shape,
            "dtype": self.dtype,
            "reason": self.reason,
        }


def _has_module(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def probe_adapters() -> dict[str, AdapterStatus]:
    """Return optional accelerator adapters CacheIR can dispatch to.

    These adapters are intentionally thin. CacheIR still owns the IR, memory
    plan, and scheduling decision; when a package is present, the runtime can
    lower a compatible scheduled op into that library's native call.
    """

    cutlass_available = _has_module("cutlass") or _has_module("nvidia.cutlass") or _has_module("cutlass_cppgen") or _has_module("cutlass_library")
    return {
        "cutlass": AdapterStatus(
            name="cutlass",
            package="nvidia-cutlass",
            available=cutlass_available,
            capability="GEMM and epilogue kernels for CUDA lowering experiments",
            notes="Selected for matmul-like schedules when Python CUTLASS bindings are importable.",
        ),
        "flash_attention": AdapterStatus(
            name="flash_attention",
            package="flash-attn",
            available=_has_module("flash_attn"),
            capability="prefill attention adapter for contiguous Q/K/V tensors",
            notes="Useful for benchmark comparisons; CacheIR keeps its own page-table decode kernel.",
        ),
        "flashinfer": AdapterStatus(
            name="flashinfer",
            package="flashinfer-python",
            available=_has_module("flashinfer"),
            capability="paged decode/prefill attention adapter for serving-style experiments",
            notes="Preferred external adapter for production paged attention when present.",
        ),
    }


def available_adapters() -> list[str]:
    return [name for name, status in probe_adapters().items() if status.available]


def select_attention_backend(*, mode: str = "decode", prefer_external: bool = True) -> str:
    adapters = probe_adapters()
    if prefer_external and mode == "decode" and adapters["flashinfer"].available:
        return "flashinfer"
    if prefer_external and mode == "prefill" and adapters["flash_attention"].available:
        return "flash_attention"
    return "cacheir_triton"


def flash_attention_prefill(q: Any, k: Any, v: Any, *, causal: bool = True, softmax_scale: float | None = None) -> Any:
    """Execute FlashAttention prefill when the optional package is installed."""

    module = require_adapter("flash_attention")
    fn = getattr(module, "flash_attn_func", None)
    if fn is None:
        interface = importlib.import_module("flash_attn.flash_attn_interface")
        fn = getattr(interface, "flash_attn_func")
    return fn(q, k, v, dropout_p=0.0, softmax_scale=softmax_scale, causal=causal)


def run_flash_attention_smoke(
    *,
    batch: int = 1,
    seq_len: int = 16,
    heads: int = 4,
    head_dim: int = 64,
    dtype: str = "float16",
) -> AdapterSmokeResult:
    """Run a tiny real FlashAttention prefill call when torch/CUDA are present."""

    if not probe_adapters()["flash_attention"].available:
        return AdapterSmokeResult("flash_attention", available=False, executed=False, reason="flash-attn is not importable")
    try:
        import torch
    except ImportError as exc:
        return AdapterSmokeResult("flash_attention", available=True, executed=False, reason=f"torch is not importable: {exc}")
    if not torch.cuda.is_available():
        return AdapterSmokeResult("flash_attention", available=True, executed=False, reason="torch CUDA is not available")
    torch_dtype = getattr(torch, dtype)
    q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch_dtype)
    k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch_dtype)
    v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch_dtype)
    start = time.perf_counter()
    try:
        out = flash_attention_prefill(q, k, v, causal=True)
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - depends on optional CUDA package/runtime
        return AdapterSmokeResult("flash_attention", available=True, executed=False, reason=str(exc))
    return AdapterSmokeResult(
        "flash_attention",
        available=True,
        executed=True,
        elapsed_s=time.perf_counter() - start,
        output_shape=tuple(int(dim) for dim in out.shape),
        dtype=str(out.dtype),
    )


def flashinfer_paged_decode(q: Any, k_cache: Any, v_cache: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute a FlashInfer decode entry point when available.

    FlashInfer has moved APIs across releases, so this wrapper checks the common
    single-decode locations and forwards user-supplied page-table arguments to
    whichever callable exists.
    """

    module = require_adapter("flashinfer")
    candidates = [
        getattr(module, "single_decode_with_kv_cache", None),
        getattr(getattr(module, "decode", None), "single_decode_with_kv_cache", None),
        getattr(module, "single_decode_with_kv_cache_return_lse", None),
        getattr(getattr(module, "decode", None), "single_decode_with_kv_cache_return_lse", None),
    ]
    for fn in candidates:
        if fn is not None:
            return fn(q, k_cache, v_cache, *args, **kwargs)
    raise RuntimeError("Installed flashinfer package does not expose a supported decode entry point")


def run_flashinfer_decode_smoke(
    *,
    seq_len: int = 5,
    heads: int = 2,
    head_dim: int = 64,
    dtype: str = "float16",
) -> AdapterSmokeResult:
    """Run a tiny real FlashInfer single-token decode call when installed.

    Some FlashInfer wheels dispatch to JIT kernels and need a discoverable CUDA
    toolkit/driver stub at runtime. This helper reports that setup issue instead
    of hiding it behind a generic import probe.
    """

    if not probe_adapters()["flashinfer"].available:
        return AdapterSmokeResult("flashinfer", available=False, executed=False, reason="flashinfer is not importable")
    try:
        import torch
    except ImportError as exc:
        return AdapterSmokeResult("flashinfer", available=True, executed=False, reason=f"torch is not importable: {exc}")
    if not torch.cuda.is_available():
        return AdapterSmokeResult("flashinfer", available=True, executed=False, reason="torch CUDA is not available")
    torch_dtype = getattr(torch, dtype)
    q = torch.randn(heads, head_dim, device="cuda", dtype=torch_dtype)
    k = torch.randn(seq_len, heads, head_dim, device="cuda", dtype=torch_dtype)
    v = torch.randn(seq_len, heads, head_dim, device="cuda", dtype=torch_dtype)
    start = time.perf_counter()
    try:
        out = flashinfer_paged_decode(q, k, v, kv_layout="NHD")
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - depends on optional CUDA package/runtime
        return AdapterSmokeResult("flashinfer", available=True, executed=False, reason=str(exc))
    return AdapterSmokeResult(
        "flashinfer",
        available=True,
        executed=True,
        elapsed_s=time.perf_counter() - start,
        output_shape=tuple(int(dim) for dim in out.shape),
        dtype=str(out.dtype),
    )


def run_adapter_smokes() -> dict[str, AdapterSmokeResult]:
    return {
        "flash_attention": run_flash_attention_smoke(),
        "flashinfer": run_flashinfer_decode_smoke(),
    }


def flashinfer_batch_paged_decode(q: Any, k_cache: Any, v_cache: Any, workspace: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute FlashInfer's batch paged decode wrapper when installed.

    Different FlashInfer releases expose slightly different wrapper class names.
    CacheIR treats all of them as optional external execution targets and keeps
    page-table construction in its own runtime/compiler layer.
    """

    module = require_adapter("flashinfer")
    decode_module = getattr(module, "decode", None)
    plan_kwargs = kwargs.pop("plan_kwargs", None)
    wrappers = [
        getattr(module, "BatchDecodeWithPagedKVCacheWrapper", None),
        getattr(module, "BatchDecodeWithPagedKVCachePyTorchWrapper", None),
        getattr(decode_module, "BatchDecodeWithPagedKVCacheWrapper", None) if decode_module is not None else None,
        getattr(decode_module, "BatchDecodeWithPagedKVCachePyTorchWrapper", None) if decode_module is not None else None,
    ]
    for wrapper_type in wrappers:
        if wrapper_type is None:
            continue
        wrapper = wrapper_type(workspace, *args, **kwargs)
        plan = getattr(wrapper, "plan", None)
        if callable(plan):
            if plan_kwargs:
                plan(**plan_kwargs)
        run = getattr(wrapper, "run", None)
        if callable(run):
            return run(q, k_cache, v_cache)
    fallback = getattr(module, "batch_decode_with_paged_kv_cache", None)
    if fallback is not None:
        return fallback(q, k_cache, v_cache, workspace, *args, **kwargs)
    raise RuntimeError("Installed flashinfer package does not expose a supported batch paged decode wrapper")


def require_adapter(name: str) -> Any:
    adapters = probe_adapters()
    if name not in adapters:
        raise KeyError(f"Unknown adapter {name!r}")
    status = adapters[name]
    if not status.available:
        raise RuntimeError(f"{name} adapter requires optional package {status.package!r}")
    module_name = {"cutlass": "cutlass", "flash_attention": "flash_attn", "flashinfer": "flashinfer"}[name]
    if name == "cutlass":
        for candidate in ("cutlass", "nvidia.cutlass", "cutlass_cppgen", "cutlass_library"):
            if _has_module(candidate):
                module_name = candidate
                break
    return importlib.import_module(module_name)
