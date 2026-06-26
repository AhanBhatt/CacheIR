from __future__ import annotations

from pathlib import Path
from typing import Iterable
import os

import numpy as np

import cacheir.backends.native as native
from cacheir.importers.gguf import GGUFReader
from cacheir.quantization import quantize_dequantize
from cacheir.ir import Graph, Node, WeightSpec
from cacheir.runtime.artifact import CompileArtifact
from cacheir.runtime.kv_cache import PagedKVCache, PrefixCache, SpilloverCostModel, SpilloverPolicy
from cacheir.runtime.tokenizer import TokenizerBridge


class WeightStore:
    def __init__(self, model_path: str | Path, specs: dict[str, WeightSpec], quant: str | None = None):
        self.model_path = Path(model_path)
        self.specs = specs
        self.quant = quant
        self._npz: dict[str, np.ndarray] | None = None
        self._gguf: GGUFReader | None = None
        self._cache: dict[str, np.ndarray] = {}

    def get(self, value_name: str) -> np.ndarray:
        spec = self.specs[value_name]
        if value_name in self._cache:
            return self._cache[value_name]
        tensor = self._load_by_key(spec)
        tensor = tensor.astype(np.float32, copy=False)
        if self.quant and tensor.ndim == 2:
            tensor = quantize_dequantize(tensor, self.quant)
        self._cache[value_name] = tensor
        return self._cache[value_name]

    def _load_by_key(self, spec: WeightSpec) -> np.ndarray:
        if self.model_path.is_file() and self.model_path.suffix == ".npz":
            return np.load(self.model_path)[spec.key]
        if self.model_path.is_file() and self.model_path.suffix == ".safetensors":
            try:
                from safetensors import safe_open
            except ImportError as exc:
                raise RuntimeError("safetensors runtime loading requires the optional 'safetensors' dependency") from exc
            with safe_open(self.model_path, framework="np") as handle:
                return handle.get_tensor(spec.key)
        if self.model_path.is_file() and self.model_path.suffix == ".gguf":
            if self._gguf is None:
                self._gguf = GGUFReader(self.model_path)
            return self._gguf.read_tensor(spec.key)
        if self.model_path.is_dir():
            npz_path = self.model_path / (spec.file or "weights.npz")
            if npz_path.exists():
                if self._npz is None:
                    with np.load(npz_path) as data:
                        self._npz = {key: data[key] for key in data.files}
                return self._npz[spec.key]
            safetensor_path = self.model_path / spec.file if spec.file else None
            if safetensor_path and safetensor_path.exists():
                try:
                    from safetensors import safe_open
                except ImportError as exc:
                    raise RuntimeError("safetensors runtime loading requires the optional 'safetensors' dependency") from exc
                with safe_open(safetensor_path, framework="np") as handle:
                    return handle.get_tensor(spec.key)
        raise FileNotFoundError(f"Could not load weight {spec.key!r} from {self.model_path}")


class Runtime:
    """Execute compiled CacheIR graphs with a NumPy CPU backend."""

    def __init__(
        self,
        artifact: CompileArtifact | str | Path,
        *,
        weights: WeightStore | None = None,
        tokenizer: TokenizerBridge | None = None,
        prefix_cache: PrefixCache | None = None,
    ):
        self.artifact = CompileArtifact.load(artifact) if isinstance(artifact, (str, Path)) else artifact
        graph = self.artifact.graph("decode")
        self.weights = weights or WeightStore(self.artifact.model_path, graph.weights, self.artifact.quant)
        self.tokenizer = tokenizer or TokenizerBridge(self.artifact.model_path, self.artifact.config.vocab_size)
        page_size = int(graph.attrs.get("execution_hints", {}).get("page_size", 16))
        self.kv_cache = PagedKVCache(page_size=page_size)
        self.prefix_cache = prefix_cache or PrefixCache(capacity=8)

    def fork(self) -> "Runtime":
        """Create an independent KV-cache session sharing weights and tokenizer."""

        return Runtime(
            self.artifact,
            weights=self.weights,
            tokenizer=self.tokenizer,
            prefix_cache=self.prefix_cache,
        )

    def run(
        self,
        input_ids: Iterable[Iterable[int]] | np.ndarray,
        *,
        mode: str = "prefill",
        reset_kv: bool | None = None,
    ) -> np.ndarray:
        graph = self.artifact.graph(mode)
        if reset_kv is None:
            reset_kv = mode == "prefill"
        if mode == "prefill" and reset_kv:
            self.kv_cache.clear()
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        env: dict[str, np.ndarray] = {"input_ids": ids}
        for name in graph.weights:
            env[name] = self.weights.get(name)
        for name, value in graph.constants.items():
            env[name] = np.asarray(value)

        for node in graph.nodes:
            outputs = self._execute_node(graph, node, env, mode)
            for name, value in zip(node.outputs, outputs):
                env[name] = value
        return env[graph.outputs[0]]

    def prefill_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        use_prefix_cache: bool = False,
        remember_prefix: bool = False,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        ids = [int(token) for token in token_ids]
        reused_prefix: tuple[int, ...] = ()
        logits: np.ndarray
        if use_prefix_cache and ids:
            prefix, snapshot = self.prefix_cache.longest_prefix(ids)
            if snapshot is not None and 0 < len(prefix) < len(ids):
                self.kv_cache.load_prefix(snapshot)
                reused_prefix = prefix
                logits = self.run([ids[len(prefix) :]], mode="prefill", reset_kv=False)
            else:
                logits = self.run([ids], mode="prefill", reset_kv=True)
        else:
            logits = self.run([ids], mode="prefill", reset_kv=True)

        if remember_prefix and ids:
            self.remember_prefix(ids)
        return logits, reused_prefix

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        use_prefix_cache: bool = False,
        remember_prefix: bool = True,
    ) -> Iterable[str]:
        token_ids = self.tokenizer.encode(prompt)
        logits, _ = self.prefill_tokens(
            token_ids,
            use_prefix_cache=use_prefix_cache,
            remember_prefix=remember_prefix and use_prefix_cache,
        )
        for _ in range(max_new_tokens):
            next_id = int(np.argmax(logits[0, -1])) % self.artifact.config.vocab_size
            yield self.tokenizer.decode([next_id])
            logits = self.run([[next_id]], mode="decode")

    def cache_stats(self) -> dict[str, object]:
        stats = self.kv_cache.stats()
        stats["prefix_cache"] = self.prefix_cache.stats()
        return stats

    def enable_spillover(
        self,
        max_resident_pages: int | None = None,
        *,
        target: str = "cpu",
        gpu_free_memory_mb: float | None = None,
        page_bytes: int | None = None,
        safety_margin_mb: float = 512.0,
        pcie_bandwidth_gbps: float = 12.0,
    ) -> None:
        observed_page_bytes = page_bytes or self.kv_cache.estimated_page_bytes() or 0
        cost_model = None
        if observed_page_bytes or gpu_free_memory_mb is not None:
            cost_model = SpilloverCostModel(
                page_bytes=int(observed_page_bytes),
                gpu_free_memory_mb=gpu_free_memory_mb,
                safety_margin_mb=safety_margin_mb,
                pcie_bandwidth_gbps=pcie_bandwidth_gbps,
            )
        self.kv_cache.spillover_policy = SpilloverPolicy(max_resident_pages=max_resident_pages, target=target, cost_model=cost_model)

    def remember_prefix(self, token_ids: list[int] | tuple[int, ...]) -> None:
        self.prefix_cache.put(token_ids, self.kv_cache.snapshot_prefix(len(token_ids)))

    def load_longest_prefix(self, token_ids: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        prefix, snapshot = self.prefix_cache.longest_prefix(token_ids)
        if snapshot is not None:
            self.kv_cache.load_prefix(snapshot)
        return prefix

    def _execute_node(self, graph: Graph, node: Node, env: dict[str, np.ndarray], mode: str) -> list[np.ndarray]:
        op = node.op
        if op == "token_embedding":
            return [env[node.inputs[1]][env[node.inputs[0]]]]
        if op == "rms_norm":
            return [self._rms_norm(env[node.inputs[0]], env[node.inputs[1]], float(node.attrs.get("eps", 1e-6)))]
        if op in {"matmul", "quantized_matmul"}:
            return [self._matmul(env[node.inputs[0]], env[node.inputs[1]])]
        if op in {"qkv_projection"}:
            x = env[node.inputs[0]]
            return [self._matmul(x, env[node.inputs[1]]), self._matmul(x, env[node.inputs[2]]), self._matmul(x, env[node.inputs[3]])]
        if op in {"fused_rmsnorm_qkv_rope", "quantized_fused_rmsnorm_qkv_rope"}:
            x = self._rms_norm(env[node.inputs[0]], env[node.inputs[1]], float(node.attrs.get("eps", 1e-6)))
            q = self._matmul(x, env[node.inputs[2]])
            k = self._matmul(x, env[node.inputs[3]])
            v = self._matmul(x, env[node.inputs[4]])
            layer = int(node.attrs.get("layer", 0))
            q, k = self._rope(q, k, node.attrs, self._position_offset(layer, mode))
            return [q, k, v]
        if op == "rope":
            layer = int(node.attrs.get("layer", 0))
            q, k = self._rope(env[node.inputs[0]], env[node.inputs[1]], node.attrs, self._position_offset(layer, mode))
            return [q, k]
        if op in {"paged_attention_prefill", "paged_attention_decode", "grouped_query_attention"}:
            return [self._attention(env[node.inputs[0]], env[node.inputs[1]], env[node.inputs[2]], node.attrs, mode)]
        if op == "add":
            return [env[node.inputs[0]] + env[node.inputs[1]]]
        if op == "silu":
            x = env[node.inputs[0]]
            return [x / (1.0 + np.exp(-x))]
        if op == "sigmoid":
            x = env[node.inputs[0]]
            return [1.0 / (1.0 + np.exp(-x))]
        if op == "elementwise_mul":
            return [env[node.inputs[0]] * env[node.inputs[1]]]
        if op in {"fused_swiglu", "quantized_fused_swiglu"}:
            x = env[node.inputs[0]]
            gate = self._matmul(x, env[node.inputs[1]])
            up = self._matmul(x, env[node.inputs[2]])
            if native.available() and hasattr(native, "silu_mul") and gate.dtype == np.float32 and up.dtype == np.float32:
                return [native.silu_mul(gate, up)]
            return [(gate / (1.0 + np.exp(-gate))) * up]
        if op == "softmax":
            return [self._softmax(env[node.inputs[0]], axis=-1)]
        raise NotImplementedError(f"CPU runtime does not implement op {op}")

    @staticmethod
    def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        if native.available() and x.ndim == 3 and x.dtype == np.float32 and weight.dtype == np.float32:
            return native.rms_norm(x, weight, eps)
        scale = np.rsqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps) if hasattr(np, "rsqrt") else 1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
        return (x * scale) * weight

    @staticmethod
    def _matmul(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        policy = os.environ.get("CACHEIR_NATIVE_MATMUL", "numpy").lower()
        use_native = policy in {"1", "true", "force"}
        if policy == "auto" and x.ndim == 3:
            rows = int(np.prod(x.shape[:-1]))
            in_features = int(x.shape[-1])
            out_features = int(weight.shape[0]) if weight.ndim == 2 else 0
            use_native = rows >= 8 and in_features >= 256 and out_features <= 1024
        if use_native and native.available() and x.ndim == 3 and x.dtype == np.float32 and weight.dtype == np.float32:
            return native.matmul_out_in(x, weight)
        return np.einsum("...i,oi->...o", x, weight, optimize=True)

    def _position_offset(self, layer: int, mode: str) -> int:
        if layer in self.kv_cache:
            return self.kv_cache.length(layer)
        return 0

    def _rope(self, q: np.ndarray, k: np.ndarray, attrs: dict[str, object], offset: int) -> tuple[np.ndarray, np.ndarray]:
        head_dim = int(attrs["head_dim"])
        theta = float(attrs.get("rope_theta", 10000.0))
        return self._rope_one(q, head_dim, theta, offset), self._rope_one(k, head_dim, theta, offset)

    @staticmethod
    def _rope_one(x: np.ndarray, head_dim: int, theta: float, offset: int) -> np.ndarray:
        batch, seq, flat = x.shape
        heads = flat // head_dim
        view = x.reshape(batch, seq, heads, head_dim)
        half = head_dim // 2
        freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))
        pos = (np.arange(seq, dtype=np.float32) + float(offset))[:, None]
        angles = pos * freq[None, :]
        cos = np.cos(angles)[None, :, None, :]
        sin = np.sin(angles)[None, :, None, :]
        even = view[..., 0::2]
        odd = view[..., 1::2]
        rotated = np.empty_like(view)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos
        return rotated.reshape(batch, seq, flat)

    def _attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, attrs: dict[str, object], mode: str) -> np.ndarray:
        layer = int(attrs.get("layer", 0))
        heads = int(attrs["num_heads"])
        kv_heads = int(attrs["num_kv_heads"])
        head_dim = int(attrs["head_dim"])
        batch, q_len, _ = q.shape
        qh = q.reshape(batch, q_len, heads, head_dim)
        kh_new = k.reshape(batch, k.shape[1], kv_heads, head_dim)
        vh_new = v.reshape(batch, v.shape[1], kv_heads, head_dim)

        kh, vh = self.kv_cache.append(layer, kh_new, vh_new)

        repeat = heads // kv_heads
        if repeat > 1:
            kh_attn = np.repeat(kh, repeat, axis=2)
            vh_attn = np.repeat(vh, repeat, axis=2)
        else:
            kh_attn, vh_attn = kh, vh

        scores = np.einsum("bqhd,bkhd->bhqk", qh, kh_attn, optimize=True) / math_sqrt(head_dim)
        if mode == "prefill":
            k_len = kh_attn.shape[1]
            past_len = max(0, k_len - q_len)
            query_positions = past_len + np.arange(q_len, dtype=np.int64)[:, None]
            key_positions = np.arange(k_len, dtype=np.int64)[None, :]
            mask = key_positions > query_positions
            scores = np.where(mask[None, None, :, :], -1.0e30, scores)
        probs = self._softmax(scores, axis=-1)
        ctx = np.einsum("bhqk,bkhd->bqhd", probs, vh_attn, optimize=True)
        return ctx.reshape(batch, q_len, heads * head_dim)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)


def math_sqrt(value: int) -> float:
    return float(np.sqrt(float(value)))
