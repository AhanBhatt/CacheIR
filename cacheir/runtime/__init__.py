from pathlib import Path
from typing import Literal

from cacheir.runtime.artifact import CompileArtifact
from cacheir.runtime.cpu import Runtime
from cacheir.runtime.cuda import CudaRuntime, cuda_runtime_available
from cacheir.runtime.scheduler import ContinuousBatchScheduler, GenerationRequest, GenerationResult


def create_runtime(
    artifact: CompileArtifact | str | Path,
    *,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    **kwargs,
):
    loaded = CompileArtifact.load(artifact) if isinstance(artifact, (str, Path)) else artifact
    selected = backend
    if selected == "auto":
        selected = "cuda" if loaded.target in {"cuda", "triton"} and cuda_runtime_available() else "cpu"
    if selected == "cuda":
        return CudaRuntime(loaded, **kwargs)
    if selected == "cpu":
        return Runtime(loaded, **kwargs)
    raise ValueError(f"Unknown CacheIR runtime backend {backend!r}")


__all__ = [
    "ContinuousBatchScheduler",
    "CudaRuntime",
    "GenerationRequest",
    "GenerationResult",
    "Runtime",
    "create_runtime",
    "cuda_runtime_available",
]
