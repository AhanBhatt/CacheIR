from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class KVPage:
    page_id: int
    start: int
    length: int

    def to_dict(self) -> dict[str, int]:
        return {"page_id": self.page_id, "start": self.start, "length": self.length}


@dataclass
class LayerKVCache:
    keys: np.ndarray | None = None
    values: np.ndarray | None = None
    pages: list[KVPage] = field(default_factory=list)

    @property
    def length(self) -> int:
        if self.keys is None:
            return 0
        return int(self.keys.shape[1])


class PagedKVCache:
    """Reference paged KV cache.

    The NumPy runtime still stores contiguous arrays for simplicity, but it also
    maintains a page table. That lets the compiler/runtime expose and test the
    same policy objects future CUDA kernels will consume.
    """

    def __init__(self, page_size: int = 16, spillover_policy: "SpilloverPolicy | None" = None):
        self.page_size = max(1, int(page_size))
        self.layers: dict[int, LayerKVCache] = {}
        self._next_page_id = 0
        self.spillover_policy = spillover_policy
        self.spilled_pages: list[dict[str, object]] = []
        self.page_bytes: int | None = None

    def clear(self) -> None:
        self.layers.clear()
        self._next_page_id = 0
        self.spilled_pages.clear()
        self.page_bytes = None

    def __contains__(self, layer: int) -> bool:
        return layer in self.layers and self.layers[layer].keys is not None

    def __getitem__(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        cache = self.layers[layer]
        if cache.keys is None or cache.values is None:
            raise KeyError(layer)
        return cache.keys, cache.values

    def length(self, layer: int) -> int:
        return self.layers.get(layer, LayerKVCache()).length

    def append(self, layer: int, keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cache = self.layers.setdefault(layer, LayerKVCache())
        start = cache.length
        self._observe_page_bytes(keys, values)
        if cache.keys is None:
            cache.keys = keys.copy()
            cache.values = values.copy()
        else:
            cache.keys = np.concatenate([cache.keys, keys], axis=1)
            cache.values = np.concatenate([cache.values, values], axis=1)
        self._extend_pages(cache, start, int(keys.shape[1]))
        if self.spillover_policy:
            self.spilled_pages.extend(self.spillover_policy.apply(self))
        return cache.keys, cache.values

    def snapshot_prefix(self, length: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        result = {}
        for layer, cache in self.layers.items():
            if cache.keys is None or cache.values is None:
                continue
            result[layer] = (cache.keys[:, :length].copy(), cache.values[:, :length].copy())
        return result

    def load_prefix(self, prefix: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        self.clear()
        for layer, (keys, values) in prefix.items():
            self.layers[layer] = LayerKVCache(keys=keys.copy(), values=values.copy())
            self._extend_pages(self.layers[layer], 0, int(keys.shape[1]))

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
            "spilled_pages": self.spilled_pages,
            "page_bytes": self.estimated_page_bytes(),
            "spillover_policy": self.spillover_policy.to_dict() if self.spillover_policy else None,
        }

    def _extend_pages(self, cache: LayerKVCache, start: int, length: int) -> None:
        remaining = length
        cursor = start
        while remaining > 0:
            page_offset = cursor % self.page_size
            take = min(remaining, self.page_size - page_offset)
            if page_offset == 0:
                cache.pages.append(KVPage(page_id=self._next_page_id, start=cursor, length=take))
                self._next_page_id += 1
            else:
                cache.pages[-1].length += take
            cursor += take
            remaining -= take

    def total_pages(self) -> int:
        return sum(len(cache.pages) for cache in self.layers.values())

    def estimated_page_bytes(self) -> int | None:
        return self.page_bytes

    def _observe_page_bytes(self, keys: np.ndarray, values: np.ndarray) -> None:
        tokens = int(keys.shape[1]) if keys.ndim >= 2 else 0
        if tokens <= 0:
            return
        bytes_per_token = (keys.nbytes + values.nbytes) / tokens
        self.page_bytes = int(np.ceil(bytes_per_token * self.page_size))


@dataclass
class SpilloverCostModel:
    """Calibrated cost model for low-VRAM KV-cache residency decisions."""

    page_bytes: int
    gpu_free_memory_mb: float | None = None
    safety_margin_mb: float = 512.0
    pcie_bandwidth_gbps: float = 12.0
    cpu_read_bandwidth_gbps: float = 35.0
    gpu_read_bandwidth_gbps: float = 700.0
    spill_latency_us: float = 35.0
    prefetch_latency_us: float = 20.0

    @classmethod
    def from_hardware_profile(
        cls,
        profile: object,
        *,
        page_bytes: int,
        gpu_free_memory_mb: float | None = None,
        safety_margin_mb: float = 512.0,
    ) -> "SpilloverCostModel":
        if gpu_free_memory_mb is None:
            gpus = getattr(profile, "gpus", [])
            totals = [gpu.memory_total_mb for gpu in gpus if getattr(gpu, "memory_total_mb", None)]
            gpu_free_memory_mb = float(max(totals) * 0.35) if totals else None
        return cls(page_bytes=page_bytes, gpu_free_memory_mb=gpu_free_memory_mb, safety_margin_mb=safety_margin_mb)

    @classmethod
    def from_bandwidth_calibration(
        cls,
        calibration: object,
        *,
        page_bytes: int,
        gpu_free_memory_mb: float | None = None,
        safety_margin_mb: float = 512.0,
    ) -> "SpilloverCostModel":
        cpu_gbps = float(getattr(calibration, "cpu_copy_gbps", 35.0) or 35.0)
        cuda_gbps = getattr(calibration, "cuda_h2d_gbps", None)
        pcie_gbps = float(cuda_gbps or min(12.0, cpu_gbps))
        return cls(
            page_bytes=page_bytes,
            gpu_free_memory_mb=gpu_free_memory_mb,
            safety_margin_mb=safety_margin_mb,
            pcie_bandwidth_gbps=pcie_gbps,
            cpu_read_bandwidth_gbps=cpu_gbps,
        )

    def resident_page_budget(self, fallback: int | None = None) -> int:
        if self.gpu_free_memory_mb is None or self.page_bytes <= 0:
            return max(1, int(fallback or 1))
        usable_bytes = max(0.0, (self.gpu_free_memory_mb - self.safety_margin_mb) * 1024.0 * 1024.0)
        budget = int(usable_bytes // float(self.page_bytes))
        if fallback is not None:
            budget = min(int(fallback), budget)
        return max(1, budget)

    def transfer_ms(self, bytes_count: int, *, target: str = "cpu") -> float:
        bandwidth = self.pcie_bandwidth_gbps if target == "cpu" else min(self.pcie_bandwidth_gbps, self.cpu_read_bandwidth_gbps)
        latency = self.spill_latency_us if target == "cpu" else self.prefetch_latency_us
        return (latency / 1000.0) + (float(bytes_count) / (bandwidth * 1.0e9) * 1000.0)

    def eviction_score(self, *, layer: int, page: KVPage, target: str) -> float:
        transfer_penalty = self.transfer_ms(self.page_bytes, target=target)
        return float(page.start) + float(layer) * 0.001 + transfer_penalty * 0.01

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "page_bytes": self.page_bytes,
            "gpu_free_memory_mb": self.gpu_free_memory_mb,
            "safety_margin_mb": self.safety_margin_mb,
            "pcie_bandwidth_gbps": self.pcie_bandwidth_gbps,
            "cpu_read_bandwidth_gbps": self.cpu_read_bandwidth_gbps,
            "gpu_read_bandwidth_gbps": self.gpu_read_bandwidth_gbps,
            "spill_latency_us": self.spill_latency_us,
            "prefetch_latency_us": self.prefetch_latency_us,
        }


@dataclass
class SpilloverPolicy:
    max_resident_pages: int | None = None
    target: str = "cpu"
    cost_model: SpilloverCostModel | None = None

    def apply(self, cache: PagedKVCache) -> list[dict[str, object]]:
        max_resident = self._resident_budget(cache)
        overflow = cache.total_pages() - max_resident
        if overflow <= 0:
            return []
        spilled: list[dict[str, int | float | str]] = []
        already_spilled = {(int(marker["layer"]), int(marker["page_id"])) for marker in cache.spilled_pages}
        candidates: list[tuple[float, int, KVPage]] = []
        for layer, layer_cache in sorted(cache.layers.items()):
            for page in layer_cache.pages:
                if (layer, page.page_id) in already_spilled:
                    continue
                score = self.cost_model.eviction_score(layer=layer, page=page, target=self.target) if self.cost_model else float(page.start)
                candidates.append((score, layer, page))
        for _, layer, page in sorted(candidates, key=lambda item: (item[0], item[1], item[2].page_id)):
            if overflow <= 0:
                break
            marker: dict[str, int | float | str] = {"layer": layer, "page_id": page.page_id, "start": page.start, "length": page.length, "target": self.target}
            if self.cost_model:
                marker["estimated_transfer_ms"] = self.cost_model.transfer_ms(self.cost_model.page_bytes, target=self.target)
            spilled.append(marker)
            overflow -= 1
        return spilled

    def _resident_budget(self, cache: PagedKVCache) -> int:
        if self.cost_model is None:
            return max(1, int(self.max_resident_pages or 1))
        if self.cost_model.page_bytes <= 0 and cache.estimated_page_bytes():
            self.cost_model.page_bytes = int(cache.estimated_page_bytes() or 0)
        return self.cost_model.resident_page_budget(self.max_resident_pages)

    def to_dict(self) -> dict[str, object]:
        return {
            "max_resident_pages": self.max_resident_pages,
            "target": self.target,
            "cost_model": self.cost_model.to_dict() if self.cost_model else None,
        }


class PrefixCache:
    """Small LRU prefix-cache experiment for reusable KV snapshots."""

    def __init__(self, capacity: int = 8):
        self.capacity = max(1, int(capacity))
        self._entries: OrderedDict[tuple[int, ...], dict[int, tuple[np.ndarray, np.ndarray]]] = OrderedDict()

    def put(self, token_ids: list[int] | tuple[int, ...], snapshot: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        key = tuple(int(token) for token in token_ids)
        if key in self._entries:
            self._entries.pop(key)
        self._entries[key] = {
            layer: (keys.copy(), values.copy())
            for layer, (keys, values) in snapshot.items()
        }
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def longest_prefix(self, token_ids: list[int] | tuple[int, ...]) -> tuple[tuple[int, ...], dict[int, tuple[np.ndarray, np.ndarray]] | None]:
        query = tuple(int(token) for token in token_ids)
        best_key: tuple[int, ...] = ()
        best_value: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        for key, value in self._entries.items():
            if len(key) > len(best_key) and query[: len(key)] == key:
                best_key = key
                best_value = value
        if best_value is not None:
            self._entries.move_to_end(best_key)
            best_value = {layer: (keys.copy(), values.copy()) for layer, (keys, values) in best_value.items()}
        return best_key, best_value

    def stats(self) -> dict[str, int]:
        return {"capacity": self.capacity, "entries": len(self._entries)}
