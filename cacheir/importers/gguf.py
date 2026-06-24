from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


_GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}


try:
    import gguf as _gguf_reference
except ImportError:  # pragma: no cover - exercised when optional importer extra is absent
    _gguf_reference = None


_GGML_TYPE_INFO = {
    0: {"name": "F32", "kind": "dense", "dtype": np.float32, "itemsize": 4},
    1: {"name": "F16", "kind": "dense", "dtype": np.float16, "itemsize": 2},
    2: {"name": "Q4_0", "kind": "q4_0", "block_size": 32, "block_bytes": 18},
    3: {"name": "Q4_1", "kind": "q4_1", "block_size": 32, "block_bytes": 20},
    6: {"name": "Q5_0", "kind": "q5_0", "block_size": 32, "block_bytes": 22},
    7: {"name": "Q5_1", "kind": "q5_1", "block_size": 32, "block_bytes": 24},
    8: {"name": "Q8_0", "kind": "q8_0", "block_size": 32, "block_bytes": 34},
    9: {"name": "Q8_1", "kind": "q8_1", "block_size": 32, "block_bytes": 40},
    10: {"name": "Q2_K", "kind": "reference", "block_size": 256, "block_bytes": 84},
    11: {"name": "Q3_K", "kind": "reference", "block_size": 256, "block_bytes": 110},
    12: {"name": "Q4_K", "kind": "reference", "block_size": 256, "block_bytes": 144},
    13: {"name": "Q5_K", "kind": "reference", "block_size": 256, "block_bytes": 176},
    14: {"name": "Q6_K", "kind": "reference", "block_size": 256, "block_bytes": 210},
    15: {"name": "Q8_K", "kind": "unsupported_reference", "block_size": 256, "block_bytes": 292},
    16: {"name": "IQ2_XXS", "kind": "reference", "block_size": 256, "block_bytes": 66},
    17: {"name": "IQ2_XS", "kind": "reference", "block_size": 256, "block_bytes": 74},
    18: {"name": "IQ3_XXS", "kind": "reference", "block_size": 256, "block_bytes": 98},
    19: {"name": "IQ1_S", "kind": "reference", "block_size": 256, "block_bytes": 50},
    20: {"name": "IQ4_NL", "kind": "reference", "block_size": 32, "block_bytes": 18},
    21: {"name": "IQ3_S", "kind": "reference", "block_size": 256, "block_bytes": 110},
    22: {"name": "IQ2_S", "kind": "reference", "block_size": 256, "block_bytes": 82},
    23: {"name": "IQ4_XS", "kind": "reference", "block_size": 256, "block_bytes": 136},
    24: {"name": "I8", "kind": "dense", "dtype": np.int8, "itemsize": 1},
    25: {"name": "I16", "kind": "dense", "dtype": np.int16, "itemsize": 2},
    26: {"name": "I32", "kind": "dense", "dtype": np.int32, "itemsize": 4},
    27: {"name": "I64", "kind": "dense", "dtype": np.int64, "itemsize": 8},
    28: {"name": "F64", "kind": "dense", "dtype": np.float64, "itemsize": 8},
    29: {"name": "IQ1_M", "kind": "reference", "block_size": 256, "block_bytes": 56},
    30: {"name": "BF16", "kind": "bf16", "itemsize": 2},
    34: {"name": "TQ1_0", "kind": "reference", "block_size": 256, "block_bytes": 54},
    35: {"name": "TQ2_0", "kind": "reference", "block_size": 256, "block_bytes": 66},
    39: {"name": "MXFP4", "kind": "reference", "block_size": 32, "block_bytes": 17},
    40: {"name": "NVFP4", "kind": "reference", "block_size": 64, "block_bytes": 36},
    41: {"name": "Q1_0", "kind": "unsupported_reference", "block_size": 128, "block_bytes": 18},
}


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unexpected end of GGUF file")
    return data


def _u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _string(handle: BinaryIO) -> str:
    size = _u64(handle)
    return _read_exact(handle, size).decode("utf-8", errors="replace")


def _scalar(handle: BinaryIO, type_id: int) -> Any:
    if type_id == 0:
        return struct.unpack("<B", _read_exact(handle, 1))[0]
    if type_id == 1:
        return struct.unpack("<b", _read_exact(handle, 1))[0]
    if type_id == 2:
        return struct.unpack("<H", _read_exact(handle, 2))[0]
    if type_id == 3:
        return struct.unpack("<h", _read_exact(handle, 2))[0]
    if type_id == 4:
        return _u32(handle)
    if type_id == 5:
        return struct.unpack("<i", _read_exact(handle, 4))[0]
    if type_id == 6:
        return struct.unpack("<f", _read_exact(handle, 4))[0]
    if type_id == 7:
        return bool(struct.unpack("<?", _read_exact(handle, 1))[0])
    if type_id == 8:
        return _string(handle)
    if type_id == 10:
        return _u64(handle)
    if type_id == 11:
        return struct.unpack("<q", _read_exact(handle, 8))[0]
    if type_id == 12:
        return struct.unpack("<d", _read_exact(handle, 8))[0]
    raise ValueError(f"Unsupported GGUF metadata type {type_id}")


def _value(handle: BinaryIO) -> Any:
    type_id = _u32(handle)
    if type_id == 9:
        elem_type = _u32(handle)
        length = _u64(handle)
        return [_scalar(handle, elem_type) for _ in range(length)]
    return _scalar(handle, type_id)


def import_gguf_metadata(path: str | Path) -> dict[str, Any]:
    """Parse the GGUF header, metadata table, and tensor directory.

    The parser deliberately stops before tensor data. That is enough for CacheIR's
    importer to recover architecture fields, context length, quantization type, and
    tensor shapes without depending on llama.cpp.
    """
    file_path = Path(path)
    with file_path.open("rb") as handle:
        magic = _read_exact(handle, 4)
        if magic != b"GGUF":
            raise ValueError(f"{file_path} is not a GGUF file")
        version = _u32(handle)
        tensor_count = _u64(handle)
        kv_count = _u64(handle)

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _string(handle)
            metadata[key] = _value(handle)

        tensors = []
        for _ in range(tensor_count):
            name = _string(handle)
            ndim = _u32(handle)
            dims = tuple(_u64(handle) for _ in range(ndim))
            ggml_type = _u32(handle)
            offset = _u64(handle)
            tensors.append({"name": name, "shape": dims, "ggml_type": ggml_type, "offset": offset})
        alignment = int(metadata.get("general.alignment", 32))
        data_offset = _align(handle.tell(), alignment)

    return {
        "version": version,
        "metadata": metadata,
        "tensors": tensors,
        "data_offset": data_offset,
        "alignment": alignment,
        "value_types": _GGUF_VALUE_TYPES,
    }


def _align(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


class GGUFReader:
    """Read tensor data for a deliberately small GGUF subset.

    Supported tensor encodings include dense scalar tensors, the common classic
    GGML Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 block families, and K/IQ/TQ/NV/MX
    families when the optional reference ``gguf`` package exposes a dequantizer.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.info = import_gguf_metadata(self.path)
        self.tensors = {tensor["name"]: tensor for tensor in self.info["tensors"]}

    def tensor_names(self) -> list[str]:
        return sorted(self.tensors)

    def read_tensor(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise KeyError(f"GGUF tensor {name!r} not found")
        tensor = self.tensors[name]
        ggml_type = int(tensor["ggml_type"])
        if ggml_type not in _GGML_TYPE_INFO:
            type_name = f"GGML type {ggml_type}"
            raise NotImplementedError(f"Native GGUF execution does not yet support {type_name}")
        info = _GGML_TYPE_INFO[ggml_type]
        raw_shape = tuple(int(dim) for dim in tensor["shape"])
        logical_shape = tuple(reversed(raw_shape)) if len(raw_shape) > 1 else raw_shape
        count = int(np.prod(logical_shape, dtype=np.int64))
        byte_count = self._byte_count(count, info)
        offset = int(self.info["data_offset"]) + int(tensor["offset"])
        with self.path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(byte_count)
        if len(data) != byte_count:
            raise ValueError(f"GGUF tensor {name!r} ended early")
        return self._decode(data, count, info).reshape(logical_shape)

    @staticmethod
    def _byte_count(count: int, info: dict[str, Any]) -> int:
        if info["kind"] == "dense":
            return count * int(info["itemsize"])
        if info["kind"] == "bf16":
            return count * int(info["itemsize"])
        block_size = int(info["block_size"])
        if count % block_size != 0:
            raise ValueError(f"{info['name']} tensor element count must be divisible by {block_size}")
        return (count // block_size) * int(info["block_bytes"])

    @staticmethod
    def _decode(data: bytes, count: int, info: dict[str, Any]) -> np.ndarray:
        if info["kind"] == "dense":
            return np.frombuffer(data, dtype=info["dtype"]).astype(np.float32, copy=False)
        if info["kind"] == "bf16":
            return _decode_bf16(data, count)
        if info["kind"] == "q8_0":
            return _decode_q8_0(data, count)
        if info["kind"] == "q8_1":
            return _decode_q8_1(data, count)
        if info["kind"] == "q4_0":
            return _decode_q4_0(data, count)
        if info["kind"] == "q4_1":
            return _decode_q4_1(data, count)
        if info["kind"] == "q5_0":
            return _decode_q5_0(data, count)
        if info["kind"] == "q5_1":
            return _decode_q5_1(data, count)
        if info["kind"] == "reference":
            return _decode_with_reference_gguf(data, count, info)
        if info["kind"] == "unsupported_reference":
            raise NotImplementedError(f"Reference GGUF package does not yet expose dequantization for {info['name']}")
        raise NotImplementedError(f"Unsupported GGUF tensor kind {info['kind']}")


def _f16(data: bytes, cursor: int) -> float:
    return float(np.frombuffer(data[cursor : cursor + 2], dtype=np.float16, count=1).astype(np.float32)[0])


def _decode_bf16(data: bytes, count: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint16, count=count).astype(np.uint32)
    return (raw << 16).view(np.float32)


def _split_q4_block(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = (packed & 0x0F).astype(np.int16)
    high = (packed >> 4).astype(np.int16)
    return low, high


def _q5_values(packed: np.ndarray, high_bits: int) -> np.ndarray:
    low, high = _split_q4_block(packed)
    values = np.empty(32, dtype=np.int16)
    for idx in range(16):
        values[idx] = low[idx] | (((high_bits >> idx) & 1) << 4)
        values[idx + 16] = high[idx] | (((high_bits >> (idx + 16)) & 1) << 4)
    return values.astype(np.float32)


def _decode_q8_0(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.float32)
    cursor = 0
    for block in range(count // 32):
        scale = _f16(data, cursor)
        cursor += 2
        values = np.frombuffer(data[cursor : cursor + 32], dtype=np.int8, count=32).astype(np.float32)
        cursor += 32
        out[block * 32 : (block + 1) * 32] = values * scale
    return out


def _decode_q8_1(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.float32)
    cursor = 0
    for block in range(count // 32):
        scale = struct.unpack("<f", data[cursor : cursor + 4])[0]
        cursor += 8  # second fp32 stores the block sum and is not needed for dequantization
        values = np.frombuffer(data[cursor : cursor + 32], dtype=np.int8, count=32).astype(np.float32)
        cursor += 32
        out[block * 32 : (block + 1) * 32] = values * scale
    return out


def _decode_q4_0(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.float32)
    cursor = 0
    for block in range(count // 32):
        scale = _f16(data, cursor)
        cursor += 2
        packed = np.frombuffer(data[cursor : cursor + 16], dtype=np.uint8, count=16)
        cursor += 16
        low, high = _split_q4_block(packed)
        values = np.empty(32, dtype=np.float32)
        values[:16] = (low - 8).astype(np.float32)
        values[16:] = (high - 8).astype(np.float32)
        out[block * 32 : (block + 1) * 32] = values * scale
    return out


def _decode_q4_1(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.float32)
    cursor = 0
    for block in range(count // 32):
        scale = _f16(data, cursor)
        minimum = _f16(data, cursor + 2)
        cursor += 4
        packed = np.frombuffer(data[cursor : cursor + 16], dtype=np.uint8, count=16)
        cursor += 16
        low, high = _split_q4_block(packed)
        values = np.empty(32, dtype=np.float32)
        values[:16] = low.astype(np.float32)
        values[16:] = high.astype(np.float32)
        out[block * 32 : (block + 1) * 32] = values * scale + minimum
    return out


def _decode_q5_0(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.float32)
    cursor = 0
    for block in range(count // 32):
        scale = _f16(data, cursor)
        cursor += 2
        high_bits = struct.unpack("<I", data[cursor : cursor + 4])[0]
        cursor += 4
        packed = np.frombuffer(data[cursor : cursor + 16], dtype=np.uint8, count=16)
        cursor += 16
        values = _q5_values(packed, high_bits) - 16.0
        out[block * 32 : (block + 1) * 32] = values * scale
    return out


def _decode_q5_1(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.float32)
    cursor = 0
    for block in range(count // 32):
        scale = _f16(data, cursor)
        minimum = _f16(data, cursor + 2)
        cursor += 4
        high_bits = struct.unpack("<I", data[cursor : cursor + 4])[0]
        cursor += 4
        packed = np.frombuffer(data[cursor : cursor + 16], dtype=np.uint8, count=16)
        cursor += 16
        values = _q5_values(packed, high_bits)
        out[block * 32 : (block + 1) * 32] = values * scale + minimum
    return out


def _decode_with_reference_gguf(data: bytes, count: int, info: dict[str, Any]) -> np.ndarray:
    if _gguf_reference is None:
        raise RuntimeError(f"{info['name']} dequantization requires the optional 'gguf' package")
    qtype = getattr(_gguf_reference.GGMLQuantizationType, str(info["name"]))
    block_size = int(info["block_size"])
    block_bytes = int(info["block_bytes"])
    blocks = count // block_size
    byte_shape = (blocks, block_bytes)
    raw = np.frombuffer(data, dtype=np.uint8).reshape(byte_shape)
    return _gguf_reference.dequantize(raw, qtype).reshape(count).astype(np.float32, copy=False)
