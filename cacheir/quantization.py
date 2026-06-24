from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantizedTensor:
    values: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray | None
    bits: int
    axis: int = 1


def quantize_weight(weight: np.ndarray, quant: str | None) -> QuantizedTensor | None:
    if not quant:
        return None
    name = quant.lower()
    if "int4" in name or "awq" in name or "gptq" in name:
        return _symmetric_quant(weight, bits=4)
    if "int8" in name or "8bit" in name:
        return _symmetric_quant(weight, bits=8)
    return None


def dequantize_weight(qweight: QuantizedTensor) -> np.ndarray:
    scale = np.expand_dims(qweight.scales, axis=qweight.axis)
    return qweight.values.astype(np.float32) * scale


def quantize_dequantize(weight: np.ndarray, quant: str | None) -> np.ndarray:
    qweight = quantize_weight(weight, quant)
    if qweight is None:
        return weight.astype(np.float32, copy=False)
    return dequantize_weight(qweight)


def _symmetric_quant(weight: np.ndarray, bits: int) -> QuantizedTensor:
    qmax = float((1 << (bits - 1)) - 1)
    max_abs = np.max(np.abs(weight), axis=1)
    scales = np.maximum(max_abs / qmax, 1.0e-8).astype(np.float32)
    values = np.round(weight / scales[:, None])
    values = np.clip(values, -qmax, qmax).astype(np.int8)
    return QuantizedTensor(values=values, scales=scales, zero_points=None, bits=bits)
