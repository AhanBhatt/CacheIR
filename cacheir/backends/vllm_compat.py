from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any


def install_no_uva_fallback(*, force: bool = False) -> dict[str, object]:
    """Patch vLLM's staged write buffers to avoid hard-requiring CUDA UVA.

    vLLM V1 uses UVA-backed CPU buffers for low-latency metadata staging. Some
    local CUDA environments, notably WSL configurations, expose a CUDA device but
    do not expose UVA. In that case vLLM raises before inference starts. This
    compatibility patch keeps the same public buffer methods but stages through
    regular CUDA tensors instead. It trades a little staging efficiency for the
    ability to run the benchmark.
    """

    try:
        torch = importlib.import_module("torch")
        buffer_utils = importlib.import_module("vllm.v1.worker.gpu.buffer_utils")
    except ImportError as exc:
        return {"installed": False, "reason": f"import failed: {exc}"}

    is_uva_available = getattr(buffer_utils, "is_uva_available", None)
    if callable(is_uva_available) and is_uva_available() and not force:
        return {"installed": False, "reason": "UVA is available; fallback not needed"}
    if getattr(buffer_utils, "_CACHEIR_NO_UVA_FALLBACK", False):
        return {"installed": True, "reason": "fallback already installed"}

    def _cuda_device() -> Any:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cuda")

    def _length(x: Any) -> int:
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            return len(x)
        return int(getattr(x, "shape", [len(x)])[0])

    def _as_cpu_tensor(x: Any, dtype: Any) -> Any:
        if hasattr(torch, "Tensor") and isinstance(x, torch.Tensor):
            return x.detach().to(device="cpu", dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device="cpu")

    class NoUvaBuffer:
        def __init__(self, size: int | Sequence[int], dtype: Any):
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
            self.np = self.cpu.numpy()
            self.uva = torch.zeros(size, dtype=dtype, device=_cuda_device())

    class NoUvaBufferPool:
        def __init__(
            self,
            size: int | Sequence[int],
            dtype: Any,
            max_concurrency: int | None = None,
        ):
            if max_concurrency is None:
                max_concurrency = getattr(buffer_utils, "_DEFAULT_MAX_CONCURRENCY", 2)
            self.size = size
            self.dtype = dtype
            self.max_concurrency = max(2, int(max_concurrency))
            self._gpu_bufs = [torch.empty(size, dtype=dtype, device=_cuda_device()) for _ in range(self.max_concurrency)]
            self._curr = 0

        def copy_to_uva(self, x: Any) -> Any:
            self._curr = (self._curr + 1) % self.max_concurrency
            out = self._gpu_bufs[self._curr]
            n = _length(x)
            source = _as_cpu_tensor(x, self.dtype)
            out[:n].copy_(source[:n], non_blocking=True)
            return out[:n]

        def copy_to_gpu(self, x: Any, out: Any | None = None) -> Any:
            staged = self.copy_to_uva(x)
            return staged.clone() if out is None else out.copy_(staged, non_blocking=True)

    class NoUvaBackedTensor:
        def __init__(
            self,
            size: int | Sequence[int],
            dtype: Any,
            max_concurrency: int | None = None,
        ):
            self.dtype = dtype
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
            self.np = self.cpu.numpy()
            self.pool = NoUvaBufferPool(size, dtype, max_concurrency)
            self.gpu = self.pool.copy_to_uva(self.np)

        def copy_to_uva(self, n: int | None = None) -> Any:
            self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
            return self.gpu

    buffer_utils.UvaBuffer = NoUvaBuffer
    buffer_utils.UvaBufferPool = NoUvaBufferPool
    if hasattr(buffer_utils, "UvaBackedTensor"):
        buffer_utils.UvaBackedTensor = NoUvaBackedTensor
    buffer_utils._CACHEIR_NO_UVA_FALLBACK = True
    return {"installed": True, "reason": "UVA unavailable; using CUDA tensor staging"}
