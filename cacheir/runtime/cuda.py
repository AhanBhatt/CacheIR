from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cacheir.quantization import PackedQuantizedTensor
from cacheir.ir import Graph, Node
from cacheir.runtime.artifact import CompileArtifact
from cacheir.runtime.cpu import WeightStore, _quantized_weight_inputs
from cacheir.runtime.kv_cache import KVPage, PrefixCache
from cacheir.runtime.tokenizer import TokenizerBridge


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def cuda_runtime_available() -> bool:
    torch = _try_import_torch()
    return bool(torch is not None and torch.cuda.is_available())


def _dtype_from_name(torch: Any, name: str | None) -> Any:
    if name in {None, "auto"}:
        return None
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported CUDA runtime dtype {name!r}") from exc


class CudaWeightStore:
    """Lazy CPU-to-GPU weight store used by the CacheIR CUDA runtime."""

    def __init__(
        self,
        base: WeightStore,
        *,
        torch: Any,
        device: Any,
        dtype: Any | None = None,
    ):
        self.base = base
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self._cache: dict[str, Any] = {}
        self._qcache: dict[str, CudaPackedQuantizedTensor] = {}

    def get(self, value_name: str) -> Any:
        if value_name in self._cache:
            return self._cache[value_name]
        array = np.ascontiguousarray(self.base.get(value_name))
        tensor = self.torch.from_numpy(array)
        dtype = self.dtype if array.dtype.kind == "f" and self.dtype is not None else None
        tensor = tensor.to(device=self.device, dtype=dtype, non_blocking=False)
        self._cache[value_name] = tensor.contiguous()
        return self._cache[value_name]

    def get_quantized(self, value_name: str) -> "CudaPackedQuantizedTensor":
        if value_name in self._qcache:
            return self._qcache[value_name]
        packed = self.base.get_quantized(value_name)
        values = self.torch.from_numpy(np.ascontiguousarray(packed.packed_values)).to(device=self.device, non_blocking=False)
        scales = self.torch.from_numpy(np.ascontiguousarray(packed.scales)).to(device=self.device, dtype=self.torch.float32, non_blocking=False)
        zero_points = self.torch.from_numpy(np.ascontiguousarray(packed.zero_points)).to(device=self.device, dtype=self.torch.float32, non_blocking=False)
        result = CudaPackedQuantizedTensor(
            packed_values=values.contiguous(),
            scales=scales.contiguous(),
            zero_points=zero_points.contiguous(),
            bits=packed.bits,
            shape=packed.shape,
            axis=packed.axis,
            group_size=packed.group_size,
        )
        self._qcache[value_name] = result
        return result


@dataclass
class CudaPackedQuantizedTensor:
    packed_values: Any
    scales: Any
    zero_points: Any
    bits: int
    shape: tuple[int, int]
    axis: int = 1
    group_size: int | None = None


@dataclass
class _TorchLayerKVCache:
    keys: Any | None = None
    values: Any | None = None
    pages: list[KVPage] = field(default_factory=list)

    @property
    def length(self) -> int:
        if self.keys is None:
            return 0
        return int(self.keys.shape[1])


class TorchKVPageAllocator:
    """Shared page-id allocator and persistent CUDA KV page pool.

    Forked request sessions share one page namespace and one set of layer-local
    K/V page tensors. Reference SDPA still keeps a contiguous per-session view,
    but fused decode kernels can now read the persistent page pool directly
    instead of repacking per-request K/V into temporary page tensors.
    """

    _owner_counter = itertools.count(1)

    def __init__(self, *, page_size: int = 16, max_resident_pages: int | None = None):
        self.page_size = max(1, int(page_size))
        self.max_resident_pages = max_resident_pages
        self._next_page_id = 0
        self.page_bytes: int | None = None
        self._resident: OrderedDict[int, dict[str, object]] = OrderedDict()
        self.spilled_pages: list[dict[str, object]] = []
        self.released_pages = 0
        self._pools: dict[tuple[int, str, str, int, int], dict[str, Any]] = {}
        self.pool_writes = 0
        self.pool_grows = 0

    def new_owner(self) -> str:
        return f"cuda-session-{next(self._owner_counter)}"

    def allocate(self, *, owner: str, layer: int, start: int, length: int) -> KVPage:
        page = KVPage(page_id=self._next_page_id, start=start, length=length)
        self._next_page_id += 1
        self._resident[page.page_id] = {
            "owner": owner,
            "layer": layer,
            "page_id": page.page_id,
            "start": start,
            "length": length,
            "device": "cuda",
        }
        self._enforce_budget()
        return page

    def observe_page_bytes(self, page_bytes: int) -> None:
        if page_bytes <= 0:
            return
        self.page_bytes = max(int(page_bytes), int(self.page_bytes or 0))

    def release_owner(self, owner: str) -> None:
        released = [page_id for page_id, marker in self._resident.items() if marker.get("owner") == owner]
        for page_id in released:
            self._resident.pop(page_id, None)
        self.released_pages += len(released)

    def write_page(self, *, layer: int, page: KVPage, keys: Any, values: Any) -> None:
        if keys.ndim != 3 or values.ndim != 3:
            return
        token_count = min(int(keys.shape[0]), int(values.shape[0]), int(page.length), self.page_size)
        if token_count <= 0:
            return
        kv_heads = int(keys.shape[1])
        head_dim = int(keys.shape[2])
        key = (int(layer), str(keys.device), str(keys.dtype), kv_heads, head_dim)
        pool = self._ensure_pool(key, keys, values, 1)
        page_slots = pool["page_slots"]
        if int(page.page_id) in page_slots:
            slot = int(page_slots[int(page.page_id)])
        else:
            slot = int(pool["next_slot"])
            page_slots[int(page.page_id)] = slot
            pool["next_slot"] = slot + 1
            pool = self._ensure_pool(key, keys, values, slot + 1)
        pool["keys"][slot].zero_()
        pool["values"][slot].zero_()
        pool["keys"][slot, :, :token_count, :] = keys[:token_count].permute(1, 0, 2).contiguous()
        pool["values"][slot, :, :token_count, :] = values[:token_count].permute(1, 0, 2).contiguous()
        marker = self._resident.get(page.page_id)
        if marker is not None:
            marker["pool_key"] = key
            marker["pool_slot"] = slot
            marker["stored_tokens"] = token_count
        self.pool_writes += 1

    def pool_for_layer(
        self,
        layer: int,
        *,
        device: Any | None = None,
        dtype: Any | None = None,
        kv_heads: int | None = None,
        head_dim: int | None = None,
    ) -> dict[str, Any] | None:
        device_name = str(device) if device is not None else None
        dtype_name = str(dtype) if dtype is not None else None
        candidates = []
        for key, pool in self._pools.items():
            pool_layer, pool_device, pool_dtype, pool_kv_heads, pool_head_dim = key
            if pool_layer != int(layer):
                continue
            if device_name is not None and pool_device != device_name:
                continue
            if dtype_name is not None and pool_dtype != dtype_name:
                continue
            if kv_heads is not None and pool_kv_heads != int(kv_heads):
                continue
            if head_dim is not None and pool_head_dim != int(head_dim):
                continue
            candidates.append(pool)
        if not candidates:
            return None
        return max(candidates, key=lambda item: int(item["keys"].shape[0]))

    def stats(self) -> dict[str, object]:
        return {
            "page_size": self.page_size,
            "page_bytes": self.page_bytes,
            "allocated_pages_total": self._next_page_id,
            "resident_pages": len(self._resident),
            "released_pages": self.released_pages,
            "max_resident_pages": self.max_resident_pages,
            "spilled_pages": list(self.spilled_pages),
            "pool_writes": self.pool_writes,
            "pool_grows": self.pool_grows,
            "persistent_pools": [
                {
                    "layer": key[0],
                    "device": key[1],
                    "dtype": key[2],
                    "kv_heads": key[3],
                    "head_dim": key[4],
                    "slots": int(pool.get("next_slot", pool["keys"].shape[0])),
                    "capacity": int(pool["keys"].shape[0]),
                    "bytes": int(pool["keys"].numel() * pool["keys"].element_size() + pool["values"].numel() * pool["values"].element_size()),
                }
                for key, pool in sorted(self._pools.items(), key=lambda item: item[0])
            ],
        }

    def _enforce_budget(self) -> None:
        if self.max_resident_pages is None:
            return
        while len(self._resident) > max(1, int(self.max_resident_pages)):
            _, marker = self._resident.popitem(last=False)
            marker = dict(marker)
            marker["target"] = "cpu"
            marker["policy"] = "oldest-page"
            self.spilled_pages.append(marker)

    def _ensure_pool(self, key: tuple[int, str, str, int, int], keys: Any, values: Any, capacity: int) -> dict[str, Any]:
        capacity = max(1, int(capacity))
        pool = self._pools.get(key)
        if pool is None:
            pool = {
                "keys": keys.new_zeros((capacity, key[3], self.page_size, key[4])),
                "values": values.new_zeros((capacity, key[3], self.page_size, key[4])),
                "page_slots": {},
                "next_slot": 0,
            }
            self._pools[key] = pool
            self.pool_grows += 1
            return pool
        current = int(pool["keys"].shape[0])
        if current >= capacity:
            return pool
        new_capacity = max(capacity, current * 2)
        new_keys = keys.new_zeros((new_capacity, key[3], self.page_size, key[4]))
        new_values = values.new_zeros((new_capacity, key[3], self.page_size, key[4]))
        new_keys[:current] = pool["keys"]
        new_values[:current] = pool["values"]
        pool["keys"] = new_keys
        pool["values"] = new_values
        self.pool_grows += 1
        return pool


class TorchPagedKVCache:
    """GPU-resident KV cache with the same page-table metadata as PagedKVCache."""

    def __init__(self, *, torch: Any, device: Any, page_size: int = 16, allocator: TorchKVPageAllocator | None = None):
        self.torch = torch
        self.device = device
        self.page_size = max(1, int(page_size))
        self.allocator = allocator or TorchKVPageAllocator(page_size=self.page_size)
        self.owner_id = self.allocator.new_owner()
        self.layers: dict[int, _TorchLayerKVCache] = {}
        self.page_bytes: int | None = None

    def clear(self) -> None:
        self.allocator.release_owner(self.owner_id)
        self.layers.clear()
        self.page_bytes = None

    def __contains__(self, layer: int) -> bool:
        return layer in self.layers and self.layers[layer].keys is not None

    def __getitem__(self, layer: int) -> tuple[Any, Any]:
        cache = self.layers[layer]
        if cache.keys is None or cache.values is None:
            raise KeyError(layer)
        return cache.keys, cache.values

    def length(self, layer: int) -> int:
        return self.layers.get(layer, _TorchLayerKVCache()).length

    def append(self, layer: int, keys: Any, values: Any) -> tuple[Any, Any]:
        cache = self.layers.setdefault(layer, _TorchLayerKVCache())
        start = cache.length
        self._observe_page_bytes(keys, values)
        if cache.keys is None:
            cache.keys = keys.contiguous()
            cache.values = values.contiguous()
        else:
            cache.keys = self.torch.cat([cache.keys, keys.contiguous()], dim=1)
            cache.values = self.torch.cat([cache.values, values.contiguous()], dim=1)
        self._extend_pages(layer, cache, start, int(keys.shape[1]))
        self._write_touched_pages(layer, cache, start, int(keys.shape[1]))
        return cache.keys, cache.values

    def snapshot_prefix(self, length: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for layer, cache in self.layers.items():
            if cache.keys is None or cache.values is None:
                continue
            keys = cache.keys[:, :length].detach().cpu().numpy().copy()
            values = cache.values[:, :length].detach().cpu().numpy().copy()
            result[layer] = (keys, values)
        return result

    def load_prefix(self, prefix: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        self.clear()
        for layer, (keys, values) in prefix.items():
            k_tensor = self.torch.from_numpy(np.ascontiguousarray(keys)).to(self.device)
            v_tensor = self.torch.from_numpy(np.ascontiguousarray(values)).to(self.device)
            cache = _TorchLayerKVCache(keys=k_tensor.contiguous(), values=v_tensor.contiguous())
            self.layers[layer] = cache
            self._extend_pages(layer, cache, 0, int(k_tensor.shape[1]))
            self._observe_page_bytes(k_tensor, v_tensor)
            self._write_touched_pages(layer, cache, 0, int(k_tensor.shape[1]))

    def stats(self) -> dict[str, object]:
        return {
            "page_size": self.page_size,
            "layers": {
                str(layer): {
                    "tokens": cache.length,
                    "pages": [page.to_dict() for page in cache.pages],
                }
                for layer, cache in sorted(self.layers.items())
            },
            "page_bytes": self.page_bytes,
            "device": str(self.device),
            "allocator": self.allocator.stats(),
        }

    def paged_view(self, layer: int) -> tuple[Any, Any, Any, int] | None:
        cache = self.layers.get(layer)
        if cache is None or cache.keys is None or cache.values is None or not cache.pages:
            return None
        kv_heads = int(cache.keys.shape[2])
        head_dim = int(cache.keys.shape[3])
        pool = self.allocator.pool_for_layer(
            layer,
            device=cache.keys.device,
            dtype=cache.keys.dtype,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        if pool is None:
            return None
        page_slots = pool.get("page_slots", {})
        try:
            page_ids = [int(page_slots[int(page.page_id)]) for page in cache.pages]
        except KeyError:
            return None
        page_table = self.torch.as_tensor(page_ids, device=self.device, dtype=self.torch.int32)
        return pool["keys"], pool["values"], page_table, cache.length

    def _extend_pages(self, layer: int, cache: _TorchLayerKVCache, start: int, length: int) -> None:
        remaining = length
        cursor = start
        while remaining > 0:
            page_offset = cursor % self.page_size
            take = min(remaining, self.page_size - page_offset)
            if page_offset == 0:
                cache.pages.append(self.allocator.allocate(owner=self.owner_id, layer=layer, start=cursor, length=take))
            else:
                cache.pages[-1].length += take
            cursor += take
            remaining -= take

    def _write_touched_pages(self, layer: int, cache: _TorchLayerKVCache, start: int, length: int) -> None:
        if cache.keys is None or cache.values is None or int(cache.keys.shape[0]) != 1:
            return
        end = int(start) + int(length)
        for page in cache.pages:
            page_start = int(page.start)
            page_end = page_start + int(page.length)
            if page_end <= start or page_start >= end:
                continue
            keys = cache.keys[0, page_start:page_end].contiguous()
            values = cache.values[0, page_start:page_end].contiguous()
            self.allocator.write_page(layer=layer, page=page, keys=keys, values=values)

    def _observe_page_bytes(self, keys: Any, values: Any) -> None:
        tokens = int(keys.shape[1]) if keys.ndim >= 2 else 0
        if tokens <= 0:
            return
        bytes_per_token = ((keys.numel() + values.numel()) * keys.element_size()) / tokens
        self.page_bytes = int(np.ceil(bytes_per_token * self.page_size))
        self.allocator.observe_page_bytes(self.page_bytes)


class CudaRuntime:
    """Execute compiled CacheIR graphs on CUDA tensors.

    This is an artifact-driven CacheIR runtime: it walks the lowered CacheIR IR,
    owns a GPU KV cache, and dispatches CUDA/Triton kernels where they are
    applicable. PyTorch is used as the CUDA tensor and cuBLAS substrate, while
    CacheIR still controls graph execution, KV state, and kernel selection.
    """

    supports_variable_prefill_batch = True

    def __init__(
        self,
        artifact: CompileArtifact | str | Path,
        *,
        device: str = "cuda",
        dtype: str | None = "auto",
        weights: CudaWeightStore | None = None,
        cpu_weights: WeightStore | None = None,
        tokenizer: TokenizerBridge | None = None,
        prefix_cache: PrefixCache | None = None,
        use_triton_elementwise: bool = True,
        use_triton_attention: bool = False,
        use_triton_matmul: bool = False,
        packed_weights: dict[tuple[str, ...], Any] | None = None,
        kv_allocator: TorchKVPageAllocator | None = None,
        memory_limit_mb: float | None = None,
    ):
        torch = _try_import_torch()
        if torch is None:
            raise RuntimeError("CudaRuntime requires PyTorch with CUDA support installed")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CudaRuntime requested CUDA, but torch.cuda.is_available() is false")
        self.torch = torch
        self.artifact = CompileArtifact.load(artifact) if isinstance(artifact, (str, Path)) else artifact
        self.device = torch.device(device)
        selected_dtype = _dtype_from_name(torch, dtype)
        if selected_dtype is None:
            selected_dtype = torch.float16 if str(self.artifact.config.dtype).lower() in {"float16", "fp16", "bfloat16", "bf16"} else torch.float32
        self.dtype = selected_dtype
        graph = self.artifact.graph("decode")
        self.cpu_weights = cpu_weights or WeightStore(self.artifact.model_path, graph.weights, self.artifact.quant)
        self.weights = weights or CudaWeightStore(self.cpu_weights, torch=torch, device=self.device, dtype=self.dtype)
        self.tokenizer = tokenizer or TokenizerBridge(self.artifact.model_path, self.artifact.config.vocab_size)
        page_size = int(graph.attrs.get("execution_hints", {}).get("page_size", 16))
        self.kv_allocator = kv_allocator or TorchKVPageAllocator(page_size=page_size)
        self.kv_cache = TorchPagedKVCache(torch=torch, device=self.device, page_size=page_size, allocator=self.kv_allocator)
        self.prefix_cache = prefix_cache or PrefixCache(capacity=8)
        self.use_triton_elementwise = use_triton_elementwise
        self.use_triton_attention = use_triton_attention
        self.use_triton_matmul = use_triton_matmul
        self.memory_limit_mb = memory_limit_mb
        self._packed_weights = packed_weights if packed_weights is not None else {}
        self.kernel_counts: OrderedDict[str, int] = OrderedDict()
        self.gemm_plans: OrderedDict[str, str] = OrderedDict()
        self.oom_recoveries = 0

    @classmethod
    def available(cls) -> bool:
        return cuda_runtime_available()

    def fork(self) -> "CudaRuntime":
        return CudaRuntime(
            self.artifact,
            device=str(self.device),
            dtype=self._dtype_name(),
            weights=self.weights,
            cpu_weights=self.cpu_weights,
            tokenizer=self.tokenizer,
            prefix_cache=self.prefix_cache,
            use_triton_elementwise=self.use_triton_elementwise,
            use_triton_attention=self.use_triton_attention,
            use_triton_matmul=self.use_triton_matmul,
            packed_weights=self._packed_weights,
            kv_allocator=self.kv_allocator,
            memory_limit_mb=self.memory_limit_mb,
        )

    def run(
        self,
        input_ids: Iterable[Iterable[int]] | np.ndarray,
        *,
        mode: str = "prefill",
        reset_kv: bool | None = None,
    ) -> Any:
        graph = self.artifact.graph(mode)
        if reset_kv is None:
            reset_kv = mode == "prefill"
        if mode == "prefill" and reset_kv:
            self.kv_cache.clear()
        ids = self.torch.as_tensor(input_ids, dtype=self.torch.long, device=self.device)
        if ids.ndim == 1:
            ids = ids[None, :]
        env: dict[str, Any] = {"input_ids": ids}
        quantized_weight_names = _quantized_weight_inputs(graph)
        for name in graph.weights:
            env[name] = self.weights.get_quantized(name) if name in quantized_weight_names else self.weights.get(name)
        for name, value in graph.constants.items():
            env[name] = self.torch.as_tensor(value, device=self.device)

        for node in graph.nodes:
            self._enforce_memory_limit()
            try:
                outputs = self._execute_node(graph, node, env, mode)
            except self.torch.cuda.OutOfMemoryError as exc:
                self._recover_from_oom()
                raise RuntimeError("CacheIR CUDA runtime recovered from an out-of-memory condition; retry with a smaller batch/context") from exc
            for name, value in zip(node.outputs, outputs):
                env[name] = value
        return env[graph.outputs[0]]

    def run_decode_batch(self, sessions: list["CudaRuntime"], token_ids: list[int] | tuple[int, ...]) -> Any:
        """Advance one decode token for multiple CUDA sessions in one graph walk.

        The sessions keep independent KV caches, but share the same artifact,
        weights, tokenizer, and packed weight cache. Matmul, norm, residual, and
        MLP nodes run with a real batch dimension. Attention nodes update and
        attend against each session's own KV cache because request lengths may
        differ.
        """

        if not sessions:
            raise ValueError("run_decode_batch requires at least one session")
        if len(sessions) != len(token_ids):
            raise ValueError("sessions and token_ids must have the same length")
        for session in sessions:
            if not isinstance(session, CudaRuntime):
                raise TypeError("run_decode_batch only accepts CudaRuntime sessions")
            if session.artifact is not self.artifact and session.artifact.to_dict() != self.artifact.to_dict():
                raise ValueError("all CUDA sessions must use the same CacheIR artifact")
            if session.device != self.device:
                raise ValueError("all CUDA sessions must use the same CUDA device")

        graph = self.artifact.graph("decode")
        ids = self.torch.as_tensor([[int(token)] for token in token_ids], dtype=self.torch.long, device=self.device)
        env: dict[str, Any] = {"input_ids": ids}
        quantized_weight_names = _quantized_weight_inputs(graph)
        for name in graph.weights:
            env[name] = self.weights.get_quantized(name) if name in quantized_weight_names else self.weights.get(name)
        for name, value in graph.constants.items():
            env[name] = self.torch.as_tensor(value, device=self.device)

        self._count_kernel("cacheir.cuda_decode_batch_graph")
        for node in graph.nodes:
            self._enforce_memory_limit()
            try:
                outputs = self._execute_batch_node(graph, node, env, sessions)
            except self.torch.cuda.OutOfMemoryError as exc:
                self._recover_from_oom()
                raise RuntimeError("CacheIR CUDA batch decode recovered from an out-of-memory condition; retry with fewer active requests") from exc
            for name, value in zip(node.outputs, outputs):
                env[name] = value
        return env[graph.outputs[0]]

    def run_prefill_batch(self, sessions: list["CudaRuntime"], token_id_batches: list[list[int]]) -> Any:
        """Run padded variable-length prefill for multiple CUDA sessions.

        Dense graph nodes execute over a padded batch. Attention/KV nodes slice
        each request back to its real sequence length before writing the request
        session's cache, so padding never becomes part of the KV state.
        """

        if not sessions:
            raise ValueError("run_prefill_batch requires at least one session")
        if len(sessions) != len(token_id_batches):
            raise ValueError("sessions and token_id_batches must have the same length")
        seq_lens = [len(tokens) for tokens in token_id_batches]
        if any(length <= 0 for length in seq_lens):
            raise ValueError("run_prefill_batch requires non-empty token batches")
        for session in sessions:
            if not isinstance(session, CudaRuntime):
                raise TypeError("run_prefill_batch only accepts CudaRuntime sessions")
            if session.artifact is not self.artifact and session.artifact.to_dict() != self.artifact.to_dict():
                raise ValueError("all CUDA sessions must use the same CacheIR artifact")
            if session.device != self.device:
                raise ValueError("all CUDA sessions must use the same CUDA device")
            session.kv_cache.clear()

        graph = self.artifact.graph("prefill")
        max_seq = max(seq_lens)
        padded = [list(tokens) + [0] * (max_seq - len(tokens)) for tokens in token_id_batches]
        ids = self.torch.as_tensor(padded, dtype=self.torch.long, device=self.device)
        env: dict[str, Any] = {"input_ids": ids}
        quantized_weight_names = _quantized_weight_inputs(graph)
        for name in graph.weights:
            env[name] = self.weights.get_quantized(name) if name in quantized_weight_names else self.weights.get(name)
        for name, value in graph.constants.items():
            env[name] = self.torch.as_tensor(value, device=self.device)

        self._count_kernel("cacheir.cuda_prefill_batch_graph")
        for node in graph.nodes:
            self._enforce_memory_limit()
            try:
                outputs = self._execute_batch_node(graph, node, env, sessions, seq_lens=seq_lens)
            except self.torch.cuda.OutOfMemoryError as exc:
                self._recover_from_oom()
                raise RuntimeError("CacheIR CUDA batched prefill recovered from an out-of-memory condition; retry with fewer or shorter prompts") from exc
            for name, value in zip(node.outputs, outputs):
                env[name] = value
        return env[graph.outputs[0]]

    def prefill_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        use_prefix_cache: bool = False,
        remember_prefix: bool = False,
    ) -> tuple[Any, tuple[int, ...]]:
        ids = [int(token) for token in token_ids]
        reused_prefix: tuple[int, ...] = ()
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
            next_id = int(self.torch.argmax(logits[0, -1]).item()) % self.artifact.config.vocab_size
            yield self.tokenizer.decode([next_id])
            logits = self.run([[next_id]], mode="decode")

    def remember_prefix(self, token_ids: list[int] | tuple[int, ...]) -> None:
        self.prefix_cache.put(token_ids, self.kv_cache.snapshot_prefix(len(token_ids)))

    def load_longest_prefix(self, token_ids: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        prefix, snapshot = self.prefix_cache.longest_prefix(token_ids)
        if snapshot is not None:
            self.kv_cache.load_prefix(snapshot)
        return prefix

    def cache_stats(self) -> dict[str, object]:
        stats = self.kv_cache.stats()
        stats["prefix_cache"] = self.prefix_cache.stats()
        stats["kernel_counts"] = dict(self.kernel_counts)
        stats["gemm_plans"] = dict(self.gemm_plans)
        stats["oom_recoveries"] = self.oom_recoveries
        stats["memory_limit_mb"] = self.memory_limit_mb
        return stats

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _enforce_memory_limit(self) -> None:
        if self.memory_limit_mb is None or self.device.type != "cuda":
            return
        allocated_mb = float(self.torch.cuda.memory_allocated(self.device)) / (1024.0 * 1024.0)
        if allocated_mb <= float(self.memory_limit_mb):
            return
        self._recover_from_oom()
        allocated_after_mb = float(self.torch.cuda.memory_allocated(self.device)) / (1024.0 * 1024.0)
        if allocated_after_mb > float(self.memory_limit_mb):
            raise RuntimeError(
                f"CacheIR CUDA memory limit exceeded: {allocated_after_mb:.1f} MB allocated after recovery, "
                f"limit is {float(self.memory_limit_mb):.1f} MB"
            )

    def _recover_from_oom(self) -> None:
        self.oom_recoveries += 1
        self.kv_cache.clear()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def _execute_node(self, graph: Graph, node: Node, env: dict[str, Any], mode: str) -> list[Any]:
        op = node.op
        if op == "token_embedding":
            return [env[node.inputs[1]][env[node.inputs[0]]]]
        if op == "rms_norm":
            return [self._rms_norm(env[node.inputs[0]], env[node.inputs[1]], float(node.attrs.get("eps", 1e-6)))]
        if op == "matmul":
            return [self._matmul(env[node.inputs[0]], env[node.inputs[1]])]
        if op == "quantized_matmul":
            return [self._quantized_matmul(env[node.inputs[0]], env[node.inputs[1]])]
        if op == "qkv_projection":
            x = env[node.inputs[0]]
            return [self._matmul(x, env[node.inputs[1]]), self._matmul(x, env[node.inputs[2]]), self._matmul(x, env[node.inputs[3]])]
        if op == "fused_rmsnorm_qkv_rope":
            x = self._rms_norm(env[node.inputs[0]], env[node.inputs[1]], float(node.attrs.get("eps", 1e-6)))
            q_weight = env[node.inputs[2]]
            k_weight = env[node.inputs[3]]
            v_weight = env[node.inputs[4]]
            packed = self._packed_weight(tuple(node.inputs[2:5]), [q_weight, k_weight, v_weight])
            qkv = self._matmul(x, packed, kernel_name="torch.matmul.qkv_packed")
            q, k, v = self.torch.split(qkv, [int(q_weight.shape[0]), int(k_weight.shape[0]), int(v_weight.shape[0])], dim=-1)
            layer = int(node.attrs.get("layer", 0))
            q, k = self._rope(q, k, node.attrs, self._position_offset(layer))
            return [q, k, v]
        if op == "quantized_fused_rmsnorm_qkv_rope":
            x = self._rms_norm(env[node.inputs[0]], env[node.inputs[1]], float(node.attrs.get("eps", 1e-6)))
            q = self._quantized_matmul(x, env[node.inputs[2]])
            k = self._quantized_matmul(x, env[node.inputs[3]])
            v = self._quantized_matmul(x, env[node.inputs[4]])
            layer = int(node.attrs.get("layer", 0))
            q, k = self._rope(q, k, node.attrs, self._position_offset(layer))
            return [q, k, v]
        if op == "rope":
            layer = int(node.attrs.get("layer", 0))
            q, k = self._rope(env[node.inputs[0]], env[node.inputs[1]], node.attrs, self._position_offset(layer))
            return [q, k]
        if op in {"paged_attention_prefill", "paged_attention_decode", "grouped_query_attention"}:
            return [self._attention(env[node.inputs[0]], env[node.inputs[1]], env[node.inputs[2]], node.attrs, mode)]
        if op == "add":
            return [env[node.inputs[0]] + env[node.inputs[1]]]
        if op == "silu":
            return [self.torch.nn.functional.silu(env[node.inputs[0]])]
        if op == "sigmoid":
            return [self.torch.sigmoid(env[node.inputs[0]])]
        if op == "elementwise_mul":
            return [env[node.inputs[0]] * env[node.inputs[1]]]
        if op == "fused_swiglu":
            x = env[node.inputs[0]]
            gate_weight = env[node.inputs[1]]
            up_weight = env[node.inputs[2]]
            packed = self._packed_weight(tuple(node.inputs[1:3]), [gate_weight, up_weight])
            gate_up = self._matmul(x, packed, kernel_name="torch.matmul.swiglu_packed")
            gate, up = self.torch.split(gate_up, [int(gate_weight.shape[0]), int(up_weight.shape[0])], dim=-1)
            return [self._silu_mul(gate, up)]
        if op == "quantized_fused_swiglu":
            x = env[node.inputs[0]]
            gate = self._quantized_matmul(x, env[node.inputs[1]])
            up = self._quantized_matmul(x, env[node.inputs[2]])
            return [self._silu_mul(gate, up)]
        if op == "softmax":
            return [self.torch.softmax(env[node.inputs[0]], dim=-1)]
        raise NotImplementedError(f"CUDA runtime does not implement op {op}")

    def _execute_batch_node(
        self,
        graph: Graph,
        node: Node,
        env: dict[str, Any],
        sessions: list["CudaRuntime"],
        *,
        seq_lens: list[int] | None = None,
    ) -> list[Any]:
        op = node.op
        if op == "token_embedding":
            return [env[node.inputs[1]][env[node.inputs[0]]]]
        if op == "rms_norm":
            return [self._rms_norm(env[node.inputs[0]], env[node.inputs[1]], float(node.attrs.get("eps", 1e-6)))]
        if op == "matmul":
            return [self._matmul(env[node.inputs[0]], env[node.inputs[1]])]
        if op == "quantized_matmul":
            return [self._quantized_matmul(env[node.inputs[0]], env[node.inputs[1]])]
        if op == "qkv_projection":
            x = env[node.inputs[0]]
            return [self._matmul(x, env[node.inputs[1]]), self._matmul(x, env[node.inputs[2]]), self._matmul(x, env[node.inputs[3]])]
        if op == "fused_rmsnorm_qkv_rope":
            x = self._rms_norm(env[node.inputs[0]], env[node.inputs[1]], float(node.attrs.get("eps", 1e-6)))
            q_weight = env[node.inputs[2]]
            k_weight = env[node.inputs[3]]
            v_weight = env[node.inputs[4]]
            packed = self._packed_weight(tuple(node.inputs[2:5]), [q_weight, k_weight, v_weight])
            qkv = self._matmul(x, packed, kernel_name="torch.matmul.qkv_packed.batch")
            q, k, v = self.torch.split(qkv, [int(q_weight.shape[0]), int(k_weight.shape[0]), int(v_weight.shape[0])], dim=-1)
            layer = int(node.attrs.get("layer", 0))
            offsets = [session._position_offset(layer) for session in sessions]
            q, k = self._rope_with_offsets(q, k, node.attrs, offsets)
            return [q, k, v]
        if op == "quantized_fused_rmsnorm_qkv_rope":
            x = self._rms_norm(env[node.inputs[0]], env[node.inputs[1]], float(node.attrs.get("eps", 1e-6)))
            q = self._quantized_matmul(x, env[node.inputs[2]])
            k = self._quantized_matmul(x, env[node.inputs[3]])
            v = self._quantized_matmul(x, env[node.inputs[4]])
            layer = int(node.attrs.get("layer", 0))
            offsets = [session._position_offset(layer) for session in sessions]
            q, k = self._rope_with_offsets(q, k, node.attrs, offsets)
            return [q, k, v]
        if op == "rope":
            layer = int(node.attrs.get("layer", 0))
            offsets = [session._position_offset(layer) for session in sessions]
            q, k = self._rope_with_offsets(env[node.inputs[0]], env[node.inputs[1]], node.attrs, offsets)
            return [q, k]
        if op in {"paged_attention_prefill", "paged_attention_decode", "grouped_query_attention"}:
            q = env[node.inputs[0]]
            k = env[node.inputs[1]]
            v = env[node.inputs[2]]
            mode = "prefill" if op == "paged_attention_prefill" else "decode"
            if mode == "decode":
                decoded = self._try_triton_decode_attention_batch(q, k, v, sessions, node.attrs)
                if decoded is not None:
                    return [decoded]
            outputs = []
            for idx, session in enumerate(sessions):
                if mode == "prefill" and seq_lens is not None:
                    length = int(seq_lens[idx])
                    q_slice = q[idx : idx + 1, :length]
                    k_slice = k[idx : idx + 1, :length]
                    v_slice = v[idx : idx + 1, :length]
                else:
                    q_slice = q[idx : idx + 1]
                    k_slice = k[idx : idx + 1]
                    v_slice = v[idx : idx + 1]
                outputs.append(session._attention(q_slice, k_slice, v_slice, node.attrs, mode))
            self._count_kernel(f"cacheir.cuda_{mode}_batch_attention_split")
            if mode == "prefill" and seq_lens is not None and len(set(seq_lens)) > 1:
                return [self._pad_sequence_outputs(outputs, max(seq_lens))]
            return [self.torch.cat(outputs, dim=0)]
        if op == "add":
            return [env[node.inputs[0]] + env[node.inputs[1]]]
        if op == "silu":
            return [self.torch.nn.functional.silu(env[node.inputs[0]])]
        if op == "sigmoid":
            return [self.torch.sigmoid(env[node.inputs[0]])]
        if op == "elementwise_mul":
            return [env[node.inputs[0]] * env[node.inputs[1]]]
        if op == "fused_swiglu":
            x = env[node.inputs[0]]
            gate_weight = env[node.inputs[1]]
            up_weight = env[node.inputs[2]]
            packed = self._packed_weight(tuple(node.inputs[1:3]), [gate_weight, up_weight])
            gate_up = self._matmul(x, packed, kernel_name="torch.matmul.swiglu_packed.batch")
            gate, up = self.torch.split(gate_up, [int(gate_weight.shape[0]), int(up_weight.shape[0])], dim=-1)
            return [self._silu_mul(gate, up)]
        if op == "quantized_fused_swiglu":
            x = env[node.inputs[0]]
            gate = self._quantized_matmul(x, env[node.inputs[1]])
            up = self._quantized_matmul(x, env[node.inputs[2]])
            return [self._silu_mul(gate, up)]
        if op == "softmax":
            return [self.torch.softmax(env[node.inputs[0]], dim=-1)]
        raise NotImplementedError(f"CUDA batch runtime does not implement op {op}")

    def _pad_sequence_outputs(self, tensors: list[Any], max_seq: int) -> Any:
        if not tensors:
            raise ValueError("Cannot pad an empty tensor list")
        padded = []
        for tensor in tensors:
            pad = int(max_seq) - int(tensor.shape[1])
            if pad > 0:
                shape = (int(tensor.shape[0]), pad, int(tensor.shape[2]))
                zeros = self.torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
                tensor = self.torch.cat([tensor, zeros], dim=1)
            padded.append(tensor)
        return self.torch.cat(padded, dim=0)

    def _rms_norm(self, x: Any, weight: Any, eps: float) -> Any:
        triton_result = self._try_triton_rms_norm(x, weight, eps)
        if triton_result is not None:
            return triton_result
        self._count_kernel("torch.rms_norm")
        variance = self.torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
        return (x * self.torch.rsqrt(variance + eps).to(x.dtype)) * weight

    def _try_triton_rms_norm(self, x: Any, weight: Any, eps: float) -> Any | None:
        if not self.use_triton_elementwise or not x.is_cuda:
            return None
        try:
            import cacheir.backends.triton_kernels as tk
        except Exception:
            return None
        if not tk.triton_available() or tk.rms_norm_kernel is None:
            return None
        hidden = int(x.shape[-1])
        if int(x.numel()) < 16384:
            return None
        block = int(tk.triton.next_power_of_2(hidden))
        if block > 8192:
            return None
        x_contig = x.contiguous()
        weight_contig = weight.contiguous()
        out = self.torch.empty_like(x_contig)
        rows = int(x_contig.numel() // hidden)
        tk.rms_norm_kernel[(rows,)](x_contig, weight_contig, out, hidden=hidden, eps=eps, BLOCK=block)
        self._count_kernel("triton.rms_norm")
        return out.reshape_as(x)

    def _matmul(self, x: Any, weight: Any, *, kernel_name: str = "torch.matmul") -> Any:
        if isinstance(weight, CudaPackedQuantizedTensor):
            return self._quantized_matmul(x, weight)
        plan = self._select_gemm_plan(x, weight, requested=kernel_name)
        if plan == "triton.matmul_f16":
            triton_result = self._try_triton_matmul(x, weight)
            if triton_result is not None:
                self._count_kernel(plan)
                return triton_result
        self._count_kernel(plan)
        return self.torch.matmul(x, weight.t())

    def _quantized_matmul(self, x: Any, weight: CudaPackedQuantizedTensor | Any) -> Any:
        if not isinstance(weight, CudaPackedQuantizedTensor):
            return self._matmul(x, weight)
        dense = self._dequantize_cuda_weight(weight).to(dtype=x.dtype if x.dtype in {self.torch.float16, self.torch.bfloat16, self.torch.float32} else self.dtype)
        plan = f"cacheir.qgemm.int{weight.bits}.dequant_matmul"
        self.gemm_plans[f"q{weight.bits}:{tuple(int(dim) for dim in x.shape)}x{weight.shape}"] = plan
        self._count_kernel(plan)
        return self.torch.matmul(x, dense.t())

    def _dequantize_cuda_weight(self, weight: CudaPackedQuantizedTensor) -> Any:
        rows, cols = weight.shape
        if weight.bits == 8:
            qvalues = weight.packed_values.to(self.torch.float32)
        elif weight.bits == 4:
            packed = weight.packed_values
            padded_cols = int(packed.shape[1]) * 2
            qvalues = self.torch.empty((rows, padded_cols), device=packed.device, dtype=self.torch.float32)
            qvalues[:, 0::2] = (packed & 0x0F).to(self.torch.float32)
            qvalues[:, 1::2] = ((packed >> 4) & 0x0F).to(self.torch.float32)
            qvalues = qvalues[:, :cols]
        else:
            raise ValueError(f"Unsupported CUDA packed quantization bit width {weight.bits}")
        return (qvalues - weight.zero_points[:, None]) * weight.scales[:, None]

    def _select_gemm_plan(self, x: Any, weight: Any, *, requested: str) -> str:
        if requested != "torch.matmul":
            self.gemm_plans[f"{requested}:{tuple(int(dim) for dim in x.shape)}x{tuple(int(dim) for dim in weight.shape)}"] = requested
            return requested
        if (
            self.use_triton_matmul
            and getattr(x, "is_cuda", False)
            and x.dtype is self.torch.float16
            and weight.dtype is self.torch.float16
            and int(x.shape[-1]) >= 16
            and int(weight.shape[0]) >= 16
        ):
            plan = "triton.matmul_f16"
        elif getattr(x, "is_cuda", False):
            plan = "cublaslt.torch.matmul"
        else:
            plan = "torch.matmul"
        self.gemm_plans[f"{tuple(int(dim) for dim in x.shape)}x{tuple(int(dim) for dim in weight.shape)}"] = plan
        return plan

    def _try_triton_matmul(self, x: Any, weight: Any) -> Any | None:
        try:
            import cacheir.backends.triton_kernels as tk
        except Exception:
            return None
        if not tk.triton_available() or tk.matmul_f16_kernel is None:
            return None
        if x.ndim < 2 or weight.ndim != 2:
            return None
        k_dim = int(x.shape[-1])
        out_features = int(weight.shape[0])
        flat = x.reshape(-1, k_dim).contiguous()
        if int(flat.shape[0]) == 0:
            return None
        if k_dim % 16 or out_features % 16:
            return None
        b = weight.t().contiguous()
        out = self.torch.empty((int(flat.shape[0]), out_features), device=x.device, dtype=self.torch.float32)
        block_m = 16 if int(flat.shape[0]) < 64 else 32
        block_n = 16 if out_features < 64 else 32
        block_k = 32 if k_dim >= 32 else 16
        grid = (int(tk.triton.cdiv(flat.shape[0], block_m)), int(tk.triton.cdiv(out_features, block_n)))
        tk.matmul_f16_kernel[grid](
            flat,
            b,
            out,
            m=int(flat.shape[0]),
            n=out_features,
            k=k_dim,
            stride_am=flat.stride(0),
            stride_ak=flat.stride(1),
            stride_bk=b.stride(0),
            stride_bn=b.stride(1),
            stride_om=out.stride(0),
            stride_on=out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        return out.reshape(*x.shape[:-1], out_features).to(dtype=x.dtype)

    def _packed_weight(self, key: tuple[str, ...], tensors: list[Any]) -> Any:
        cached = self._packed_weights.get(key)
        if cached is not None:
            return cached
        packed = self.torch.cat([tensor.contiguous() for tensor in tensors], dim=0).contiguous()
        self._packed_weights[key] = packed
        return packed

    def _silu_mul(self, gate: Any, up: Any) -> Any:
        triton_result = self._try_triton_silu_mul(gate, up)
        if triton_result is not None:
            return triton_result
        self._count_kernel("torch.silu_mul")
        return self.torch.nn.functional.silu(gate) * up

    def _try_triton_silu_mul(self, gate: Any, up: Any) -> Any | None:
        if not self.use_triton_elementwise or not gate.is_cuda:
            return None
        try:
            import cacheir.backends.triton_kernels as tk
        except Exception:
            return None
        if not tk.triton_available() or tk.silu_mul_kernel is None:
            return None
        if int(gate.numel()) < 16384:
            return None
        gate_contig = gate.contiguous()
        up_contig = up.contiguous()
        out = self.torch.empty_like(gate_contig)
        block = 1024
        grid = (int(tk.triton.cdiv(gate_contig.numel(), block)),)
        tk.silu_mul_kernel[grid](gate_contig, up_contig, out, total=gate_contig.numel(), BLOCK=block)
        self._count_kernel("triton.silu_mul")
        return out.reshape_as(gate)

    def _position_offset(self, layer: int) -> int:
        if layer in self.kv_cache:
            return self.kv_cache.length(layer)
        return 0

    def _rope(self, q: Any, k: Any, attrs: dict[str, object], offset: int) -> tuple[Any, Any]:
        head_dim = int(attrs["head_dim"])
        theta = float(attrs.get("rope_theta", 10000.0))
        return self._rope_one(q, head_dim, theta, offset), self._rope_one(k, head_dim, theta, offset)

    def _rope_with_offsets(self, q: Any, k: Any, attrs: dict[str, object], offsets: list[int]) -> tuple[Any, Any]:
        head_dim = int(attrs["head_dim"])
        theta = float(attrs.get("rope_theta", 10000.0))
        return self._rope_offsets_one(q, head_dim, theta, offsets), self._rope_offsets_one(k, head_dim, theta, offsets)

    def _rope_one(self, x: Any, head_dim: int, theta: float, offset: int) -> Any:
        batch, seq, flat = x.shape
        heads = flat // head_dim
        view = x.reshape(batch, seq, heads, head_dim)
        half = head_dim // 2
        freq = 1.0 / (theta ** (self.torch.arange(0, half, dtype=self.torch.float32, device=self.device) * 2.0 / head_dim))
        pos = (self.torch.arange(seq, dtype=self.torch.float32, device=self.device) + float(offset))[:, None]
        angles = pos * freq[None, :]
        cos = self.torch.cos(angles)[None, :, None, :].to(dtype=x.dtype)
        sin = self.torch.sin(angles)[None, :, None, :].to(dtype=x.dtype)
        even = view[..., 0::2]
        odd = view[..., 1::2]
        rotated = self.torch.empty_like(view)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos
        return rotated.reshape(batch, seq, flat)

    def _rope_offsets_one(self, x: Any, head_dim: int, theta: float, offsets: list[int]) -> Any:
        batch, seq, flat = x.shape
        if len(offsets) != batch:
            raise ValueError("RoPE offsets must match batch size")
        heads = flat // head_dim
        view = x.reshape(batch, seq, heads, head_dim)
        half = head_dim // 2
        freq = 1.0 / (theta ** (self.torch.arange(0, half, dtype=self.torch.float32, device=self.device) * 2.0 / head_dim))
        base = self.torch.as_tensor(offsets, dtype=self.torch.float32, device=self.device)[:, None]
        pos = base + self.torch.arange(seq, dtype=self.torch.float32, device=self.device)[None, :]
        angles = pos[:, :, None] * freq[None, None, :]
        cos = self.torch.cos(angles)[:, :, None, :].to(dtype=x.dtype)
        sin = self.torch.sin(angles)[:, :, None, :].to(dtype=x.dtype)
        even = view[..., 0::2]
        odd = view[..., 1::2]
        rotated = self.torch.empty_like(view)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos
        return rotated.reshape(batch, seq, flat)

    def _attention(self, q: Any, k: Any, v: Any, attrs: dict[str, object], mode: str) -> Any:
        layer = int(attrs.get("layer", 0))
        heads = int(attrs["num_heads"])
        kv_heads = int(attrs["num_kv_heads"])
        head_dim = int(attrs["head_dim"])
        batch, q_len, _ = q.shape
        qh = q.reshape(batch, q_len, heads, head_dim)
        kh_new = k.reshape(batch, k.shape[1], kv_heads, head_dim)
        vh_new = v.reshape(batch, v.shape[1], kv_heads, head_dim)
        kh, vh = self.kv_cache.append(layer, kh_new, vh_new)

        if self.use_triton_attention and mode == "decode" and batch == 1 and q_len == 1:
            decoded = self._try_triton_decode_attention(layer, qh, heads, kv_heads, head_dim)
            if decoded is not None:
                return decoded

        repeat = heads // kv_heads
        if repeat > 1:
            kh_attn = kh.repeat_interleave(repeat, dim=2)
            vh_attn = vh.repeat_interleave(repeat, dim=2)
        else:
            kh_attn, vh_attn = kh, vh

        q_sdpa = qh.transpose(1, 2)
        k_sdpa = kh_attn.transpose(1, 2)
        v_sdpa = vh_attn.transpose(1, 2)
        if mode == "prefill":
            k_len = int(kh_attn.shape[1])
            past_len = max(0, k_len - q_len)
            if past_len == 0:
                ctx = self.torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
            else:
                query_positions = past_len + self.torch.arange(q_len, dtype=self.torch.long, device=self.device)[:, None]
                key_positions = self.torch.arange(k_len, dtype=self.torch.long, device=self.device)[None, :]
                allowed = key_positions <= query_positions
                ctx = self.torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=allowed[None, None, :, :])
        else:
            ctx = self.torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        self._count_kernel("torch.sdpa_attention")
        return ctx.transpose(1, 2).reshape(batch, q_len, heads * head_dim)

    def _try_triton_decode_attention(self, layer: int, qh: Any, heads: int, kv_heads: int, head_dim: int) -> Any | None:
        try:
            import cacheir.backends.triton_kernels as tk
        except Exception:
            return None
        if not tk.triton_available() or tk.paged_attention_decode_batch_kernel is None:
            return None
        block = int(tk.triton.next_power_of_2(head_dim))
        if block > 256:
            return None
        view = self.kv_cache.paged_view(layer)
        if view is None:
            return None
        k_pages, v_pages, page_table_1d, seq_len = view
        max_pages = int(page_table_1d.numel())
        if max_pages <= 0:
            return None
        page_size = self.kv_cache.page_size
        page_table = page_table_1d.reshape(1, max_pages).contiguous()
        seq_lens = self.torch.tensor([seq_len], device=self.device, dtype=self.torch.int32)
        out = self.torch.empty((1, heads, head_dim), device=self.device, dtype=qh.dtype)
        tk.paged_attention_decode_batch_kernel[(heads, 1)](
            qh[:, -1].contiguous(),
            k_pages,
            v_pages,
            page_table,
            seq_lens,
            out,
            num_heads=heads,
            num_kv_heads=kv_heads,
            max_pages_per_seq=max_pages,
            page_size=page_size,
            head_dim=head_dim,
            BLOCK=block,
        )
        self._count_kernel("triton.paged_attention_decode_batch.persistent")
        return out.reshape(1, 1, heads * head_dim)

    def _try_triton_decode_attention_batch(
        self,
        q: Any,
        k: Any,
        v: Any,
        sessions: list["CudaRuntime"],
        attrs: dict[str, object],
    ) -> Any | None:
        if not self.use_triton_attention or int(q.shape[1]) != 1:
            return None
        try:
            import cacheir.backends.triton_kernels as tk
        except Exception:
            return None
        if not tk.triton_available() or tk.paged_attention_decode_batch_kernel is None:
            return None

        layer = int(attrs.get("layer", 0))
        heads = int(attrs["num_heads"])
        kv_heads = int(attrs["num_kv_heads"])
        head_dim = int(attrs["head_dim"])
        batch = int(q.shape[0])
        if len(sessions) != batch:
            return None
        block = int(tk.triton.next_power_of_2(head_dim))
        if block > 256:
            return None

        qh = q.reshape(batch, 1, heads, head_dim)[:, -1].contiguous()
        kh_new = k.reshape(batch, int(k.shape[1]), kv_heads, head_dim)
        vh_new = v.reshape(batch, int(v.shape[1]), kv_heads, head_dim)

        views = []
        seq_lens = []
        for idx, session in enumerate(sessions):
            session.kv_cache.append(layer, kh_new[idx : idx + 1], vh_new[idx : idx + 1])
            view = session.kv_cache.paged_view(layer)
            if view is None:
                return None
            views.append(view)
            seq_lens.append(int(view[3]))

        page_size = sessions[0].kv_cache.page_size
        k_pages = views[0][0]
        v_pages = views[0][1]
        if any(view[0] is not k_pages or view[1] is not v_pages for view in views):
            return None
        max_pages = max(1, max(int(view[2].numel()) for view in views))
        page_table = self.torch.zeros((batch, max_pages), device=self.device, dtype=self.torch.int32)
        for idx, view in enumerate(views):
            page_ids = view[2]
            page_table[idx, : int(page_ids.numel())] = page_ids

        seq_lens_tensor = self.torch.as_tensor(seq_lens, device=self.device, dtype=self.torch.int32)
        out = self.torch.empty((batch, heads, head_dim), device=self.device, dtype=q.dtype)
        tk.paged_attention_decode_batch_kernel[(heads, batch)](
            qh,
            k_pages,
            v_pages,
            page_table,
            seq_lens_tensor,
            out,
            num_heads=heads,
            num_kv_heads=kv_heads,
            max_pages_per_seq=max_pages,
            page_size=page_size,
            head_dim=head_dim,
            BLOCK=block,
        )
        self._count_kernel("triton.paged_attention_decode_batch.persistent_shared")
        return out.reshape(batch, 1, heads * head_dim)

    def _count_kernel(self, name: str) -> None:
        self.kernel_counts[name] = self.kernel_counts.get(name, 0) + 1

    def _dtype_name(self) -> str:
        if self.dtype is self.torch.float16:
            return "float16"
        if self.dtype is self.torch.bfloat16:
            return "bfloat16"
        return "float32"
