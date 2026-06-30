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


@dataclass
class PackedQuantizedTensor:
    packed_values: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray
    bits: int
    shape: tuple[int, int]
    axis: int = 1
    group_size: int | None = None

    @property
    def nbytes(self) -> int:
        return int(self.packed_values.nbytes + self.scales.nbytes + self.zero_points.nbytes)

    @property
    def compression_ratio(self) -> float:
        dense_bytes = int(np.prod(self.shape, dtype=np.int64)) * 4
        return dense_bytes / max(1, self.nbytes)


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
    packed = pack_quantized_weight(weight, quant)
    if packed is not None:
        return dequantize_packed_weight(packed)
    qweight = quantize_weight(weight, quant)
    if qweight is None:
        return weight.astype(np.float32, copy=False)
    return dequantize_weight(qweight)


def pack_quantized_weight(weight: np.ndarray, quant: str | None) -> PackedQuantizedTensor | None:
    if not quant:
        return None
    name = quant.lower()
    if "int4" in name or "awq" in name or "gptq" in name:
        return _asymmetric_pack(weight, bits=4)
    if "int8" in name or "8bit" in name:
        return _asymmetric_pack(weight, bits=8)
    return None


def dequantize_packed_weight(qweight: PackedQuantizedTensor) -> np.ndarray:
    rows, cols = qweight.shape
    if qweight.bits == 4:
        qvalues = _unpack_int4(qweight.packed_values, cols)
    elif qweight.bits == 8:
        qvalues = qweight.packed_values.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported packed quantization bit width {qweight.bits}")
    scale = qweight.scales.astype(np.float32)[:, None]
    zero = qweight.zero_points.astype(np.float32)[:, None]
    return (qvalues[:rows, :cols].astype(np.float32) - zero) * scale


def _symmetric_quant(weight: np.ndarray, bits: int) -> QuantizedTensor:
    qmax = float((1 << (bits - 1)) - 1)
    max_abs = np.max(np.abs(weight), axis=1)
    scales = np.maximum(max_abs / qmax, 1.0e-8).astype(np.float32)
    values = np.round(weight / scales[:, None])
    values = np.clip(values, -qmax, qmax).astype(np.int8)
    return QuantizedTensor(values=values, scales=scales, zero_points=None, bits=bits)


def _asymmetric_pack(weight: np.ndarray, bits: int) -> PackedQuantizedTensor:
    array = np.asarray(weight, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Packed CacheIR quantization currently expects 2D out_features x in_features weights")
    qmin = 0
    qmax = (1 << bits) - 1
    mins = np.min(array, axis=1)
    maxs = np.max(array, axis=1)
    scales = np.maximum((maxs - mins) / float(qmax - qmin), 1.0e-8).astype(np.float32)
    zero_points = np.round(qmin - mins / scales).astype(np.int32)
    qvalues = np.round(array / scales[:, None] + zero_points[:, None]).clip(qmin, qmax).astype(np.uint8)
    if bits == 4:
        packed = _pack_int4(qvalues)
    elif bits == 8:
        packed = qvalues
    else:
        raise ValueError(f"Unsupported CacheIR packed bit width {bits}")
    return PackedQuantizedTensor(
        packed_values=np.ascontiguousarray(packed),
        scales=np.ascontiguousarray(scales),
        zero_points=np.ascontiguousarray(zero_points),
        bits=bits,
        shape=(int(array.shape[0]), int(array.shape[1])),
        axis=1,
    )


def _pack_int4(qvalues: np.ndarray) -> np.ndarray:
    rows, cols = qvalues.shape
    padded_cols = cols + (cols % 2)
    padded = np.zeros((rows, padded_cols), dtype=np.uint8)
    padded[:, :cols] = qvalues & 0x0F
    low = padded[:, 0::2]
    high = padded[:, 1::2]
    return (low | (high << 4)).astype(np.uint8, copy=False)


def _unpack_int4(packed: np.ndarray, cols: int) -> np.ndarray:
    rows = int(packed.shape[0])
    padded_cols = int(packed.shape[1]) * 2
    out = np.empty((rows, padded_cols), dtype=np.float32)
    out[:, 0::2] = (packed & 0x0F).astype(np.float32)
    out[:, 1::2] = ((packed >> 4) & 0x0F).astype(np.float32)
    return out[:, :cols]
