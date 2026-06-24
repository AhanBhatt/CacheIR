from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    name: str
    devices: tuple[str, ...]
    kernels: tuple[str, ...]
    notes: str = ""


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}

    def register(self, backend: Backend) -> None:
        self._backends[backend.name] = backend

    def get(self, name: str) -> Backend:
        return self._backends[name]

    def names(self) -> list[str]:
        return sorted(self._backends)


def default_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        Backend(
            name="cpu",
            devices=("x86_64", "arm64"),
            kernels=(
                "embedding_gather",
                "rms_norm",
                "matmul_avx",
                "fused_rmsnorm_qkv_rope",
                "paged_attention_prefill",
                "paged_attention_decode",
                "fused_swiglu",
            ),
            notes="NumPy reference runtime today; C++20 AVX/OpenMP skeleton in cpp/.",
        )
    )
    registry.register(
        Backend(
            name="cuda",
            devices=("sm80+", "consumer_gpu"),
            kernels=(
                "matmul_tensorcore",
                "fused_rmsnorm_qkv_rope",
                "paged_attention_prefill",
                "paged_attention_decode",
                "awq_int4_tensorcore",
            ),
            notes="Compiler target is modeled; Triton/CUDA kernels are the next backend milestone.",
        )
    )
    return registry
