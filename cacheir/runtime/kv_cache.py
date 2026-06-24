from __future__ import annotations

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

    def __init__(self, page_size: int = 16):
        self.page_size = max(1, int(page_size))
        self.layers: dict[int, LayerKVCache] = {}
        self._next_page_id = 0

    def clear(self) -> None:
        self.layers.clear()
        self._next_page_id = 0

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
        if cache.keys is None:
            cache.keys = keys.copy()
            cache.values = values.copy()
        else:
            cache.keys = np.concatenate([cache.keys, keys], axis=1)
            cache.values = np.concatenate([cache.values, values], axis=1)
        self._extend_pages(cache, start, int(keys.shape[1]))
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
