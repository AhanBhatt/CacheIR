from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_native() -> Any | None:
    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    candidates = []
    preferred_names = ("build-py-active", "build-py-gcc", "build-py", "build-native", "build")
    for root in roots:
        cpp = root / "cpp"
        candidates.extend(cpp / name for name in preferred_names)
        candidates.extend(sorted(cpp.glob("build*")))
    unique_candidates = []
    for path in candidates:
        if path not in unique_candidates:
            unique_candidates.append(path)
    for path in reversed(unique_candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
        if path.exists() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))
    if hasattr(os, "add_dll_directory"):
        for dll_dir in (Path("C:/Strawberry/c/bin"), Path("C:/msys64/mingw64/bin")):
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))
    try:
        return importlib.import_module("_cacheir_native")
    except ImportError:
        return None


_NATIVE = _load_native()


def available() -> bool:
    return _NATIVE is not None


def simd_backend() -> str:
    if _NATIVE is None:
        return "unavailable"
    return str(_NATIVE.simd_backend())


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    if _NATIVE is None:
        raise RuntimeError("CacheIR native extension is not available")
    return _NATIVE.rms_norm(np.asarray(x, dtype=np.float32), np.asarray(weight, dtype=np.float32), float(eps))


def matmul_out_in(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if _NATIVE is None:
        raise RuntimeError("CacheIR native extension is not available")
    return _NATIVE.matmul_out_in(np.asarray(x, dtype=np.float32), np.asarray(weight, dtype=np.float32))
