from __future__ import annotations


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def require_triton() -> None:
    if not triton_available():
        raise RuntimeError("Triton kernels require the optional 'triton' package and a CUDA-capable environment")


def describe_kernels() -> dict[str, str]:
    return {
        "triton.fused_rmsnorm_qkv_rope": "planned fused RMSNorm/QKV/RoPE kernel",
        "triton.paged_attention_prefill": "planned block/paged prefill attention kernel",
        "triton.paged_attention_decode": "planned token-by-token paged decode attention kernel",
        "triton.awq_int4_tensorcore": "planned int4 AWQ Tensor Core matmul kernel",
    }
