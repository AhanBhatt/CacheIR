from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO


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

    return {
        "version": version,
        "metadata": metadata,
        "tensors": tensors,
        "value_types": _GGUF_VALUE_TYPES,
    }
