from __future__ import annotations

from dataclasses import dataclass

from cacheir.ir import Graph


@dataclass(frozen=True)
class CudaGraphCapturePlan:
    enabled: bool
    mode: str
    batch_size: int
    decode_tokens_per_replay: int
    static_inputs: tuple[str, ...]
    captured_ops: tuple[str, ...]
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "batch_size": self.batch_size,
            "decode_tokens_per_replay": self.decode_tokens_per_replay,
            "static_inputs": list(self.static_inputs),
            "captured_ops": list(self.captured_ops),
            "reason": self.reason,
        }


def cuda_graph_capture_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(getattr(torch, "cuda", None) and torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph"))


def plan_decode_cuda_graph(graph: Graph, *, batch_size: int = 1, decode_tokens_per_replay: int = 1) -> CudaGraphCapturePlan:
    if graph.mode != "decode":
        return CudaGraphCapturePlan(False, graph.mode, batch_size, decode_tokens_per_replay, (), (), "only decode graphs are capture candidates")
    dynamic_ops = {"token_embedding", "paged_attention_decode", "fused_rmsnorm_qkv_rope", "quantized_fused_rmsnorm_qkv_rope"}
    captured = [node.op for node in graph.nodes if node.op in dynamic_ops or node.op.startswith("fused_") or node.op in {"add", "matmul", "quantized_matmul"}]
    if not captured:
        return CudaGraphCapturePlan(False, graph.mode, batch_size, decode_tokens_per_replay, (), (), "graph has no decode kernels to capture")
    if batch_size < 1 or decode_tokens_per_replay < 1:
        return CudaGraphCapturePlan(False, graph.mode, batch_size, decode_tokens_per_replay, (), (), "batch and replay counts must be positive")
    return CudaGraphCapturePlan(
        cuda_graph_capture_available(),
        graph.mode,
        batch_size,
        decode_tokens_per_replay,
        ("input_ids", "kv_cache_page_table", "seq_lens"),
        tuple(captured),
        "" if cuda_graph_capture_available() else "torch CUDA graph support is unavailable in this environment",
    )
