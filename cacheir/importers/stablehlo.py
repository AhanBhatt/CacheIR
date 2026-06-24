from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from cacheir.importers.hf import ModelConfig
from cacheir.ir import Graph, TensorType


_OP_MAP = {
    "stablehlo.abs": "abs",
    "stablehlo.dot_general": "matmul",
    "stablehlo.dot": "matmul",
    "stablehlo.add": "add",
    "stablehlo.broadcast_in_dim": "broadcast",
    "stablehlo.concatenate": "concat",
    "stablehlo.clamp": "clamp",
    "stablehlo.compare": "compare",
    "stablehlo.convert": "cast",
    "stablehlo.dynamic_slice": "dynamic_slice",
    "stablehlo.negate": "negate",
    "stablehlo.multiply": "elementwise_mul",
    "stablehlo.maximum": "maximum",
    "stablehlo.minimum": "minimum",
    "stablehlo.exponential": "exp",
    "stablehlo.divide": "divide",
    "stablehlo.log": "log",
    "stablehlo.power": "pow",
    "stablehlo.reduce": "reduce",
    "stablehlo.reshape": "reshape",
    "stablehlo.rsqrt": "rsqrt",
    "stablehlo.select": "select",
    "stablehlo.and": "logical_and",
    "stablehlo.or": "logical_or",
    "stablehlo.slice": "slice",
    "stablehlo.sqrt": "sqrt",
    "stablehlo.subtract": "subtract",
    "stablehlo.tanh": "tanh",
    "stablehlo.transpose": "transpose",
    "stablehlo.logistic": "sigmoid",
}

_DTYPE_MAP = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "ui8": "uint8",
}


def import_stablehlo_text(path: str | Path) -> tuple[ModelConfig, Graph]:
    """Experimental StableHLO textual importer.

    This is intentionally conservative. It recognizes SSA-style textual ops and
    maps arithmetic, shape, and reduction subsets to CacheIR nodes so StableHLO
    experiments can enter the same artifact/pass machinery without pulling in a
    full MLIR dependency.
    """
    text = Path(path).read_text(encoding="utf-8")
    graph = Graph(name=_function_name(text), mode="logical", target="generic")
    seen_outputs: list[str] = []

    for name, tensor_type in _function_args(text):
        graph.add_input(name, tensor_type)
    if not graph.inputs:
        graph.add_input("arg0", TensorType(("batch", "seq", "hidden"), "float32"))

    lines = text.splitlines()
    region_depth = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if region_depth > 0:
            region_depth += stripped.count("{") - stripped.count("}")
            continue
        returned = _return_values(stripped)
        if returned:
            graph.outputs = returned
            continue

        const = _constant(stripped)
        if const:
            name, value, tensor_type = const
            graph.add_constant(name, value, tensor_type)
            seen_outputs.append(name)
            continue

        match = _operation(stripped)
        if not match:
            continue
        outputs = [value.lstrip("%") for value in _ssa_values(match.group("outputs"))]
        stable_op = match.group("op")
        op = _OP_MAP.get(stable_op)
        if op is None:
            continue
        raw_inputs = match.group("paren") or match.group("plain") or ""
        inputs = [value.lstrip("%") for value in _ssa_values(raw_inputs) if value.lstrip("%") not in outputs]
        for inp in inputs:
            if inp not in graph.values:
                graph.add_input(inp, _first_known_type(graph, inputs) or TensorType(("batch", "seq", "hidden"), "float32"))
        type_source = _region_window(lines, idx) if stable_op == "stablehlo.reduce" else stripped
        output_types = _result_types(type_source, len(outputs)) or [_first_known_type(graph, inputs) or TensorType(("batch", "seq", "hidden"), "float32") for _ in outputs]
        attrs = {"stablehlo_op": stable_op}
        attrs.update(_stablehlo_attrs(type_source))
        if stable_op == "stablehlo.reduce":
            reducer = _region_reducer(lines, idx)
            if reducer:
                attrs["reducer"] = reducer
        graph.add_node(
            op,
            inputs,
            outputs,
            attrs=attrs,
            name=f"stablehlo.{idx}.{op}",
            output_types=output_types,
        )
        seen_outputs.extend(outputs)
        if "({" in stripped:
            region_depth += stripped.count("{") - stripped.count("}")

    if not graph.outputs:
        graph.outputs = seen_outputs[-1:] or [next(iter(graph.inputs))]
    hidden_size = _hidden_size(graph.inputs.values())
    config = ModelConfig(
        architecture="StableHLOExperiment",
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        num_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=1,
    )
    return config, graph


def _function_name(text: str) -> str:
    match = re.search(r"func\.func\s+@([\w\d_.$-]+)", text)
    return match.group(1) if match else "StableHLOExperiment"


def _function_args(text: str) -> list[tuple[str, TensorType]]:
    header = re.search(r"func\.func\s+@[\w\d_.$-]+\s*\((.*?)\)\s*(?:->|attributes|\{)", text, re.S)
    if not header:
        return []
    return [
        (match.group(1), _tensor_type(match.group(2)))
        for match in re.finditer(r"%([\w\d_]+)\s*:\s*(tensor<[^>]+>)", header.group(1))
    ]


def _tensor_type(text: str) -> TensorType:
    body_match = re.search(r"tensor<([^>]+)>", text)
    if not body_match:
        return TensorType(("batch", "seq", "hidden"), "float32")
    pieces = [piece.strip() for piece in body_match.group(1).split("x") if piece.strip()]
    if not pieces:
        return TensorType((), "float32")
    dtype = _DTYPE_MAP.get(pieces[-1], pieces[-1])
    shape: list[int | str] = []
    for dim in pieces[:-1]:
        shape.append(int(dim) if dim.isdigit() else dim.replace("?", "dyn"))
    return TensorType(tuple(shape), dtype)


def _result_types(line: str, expected: int) -> list[TensorType]:
    tensors = [_tensor_type(match.group(0)) for match in re.finditer(r"tensor<[^>]+>", line)]
    if not tensors:
        return []
    return tensors[-expected:] if len(tensors) >= expected else [tensors[-1] for _ in range(expected)]


def _constant(line: str) -> tuple[str, str, TensorType] | None:
    match = re.search(r"(%[\w\d_]+)\s*=\s*\"?stablehlo\.constant\"?.*?dense<([^>]*)>.*?:\s*(tensor<[^>]+>)", line)
    if not match:
        return None
    return match.group(1).lstrip("%"), match.group(2).strip(), _tensor_type(match.group(3))


def _operation(line: str) -> re.Match[str] | None:
    return re.search(
        r"(?P<outputs>%[\w\d_]+(?:\s*,\s*%[\w\d_]+)*)\s*=\s*\"?(?P<op>stablehlo\.[\w_]+)\"?\s*(?:\((?P<paren>[^)]*)\)|(?P<plain>.*?))(?:\s+attributes|\s*:\s*|$)",
        line,
    )


def _ssa_values(text: str) -> list[str]:
    return re.findall(r"%[\w\d_]+", text)


def _return_values(line: str) -> list[str]:
    if not (line.startswith("return ") or line.startswith("stablehlo.return ")):
        return []
    return [value.lstrip("%") for value in _ssa_values(line)]


def _stablehlo_attrs(line: str) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for key in ("dimensions", "broadcast_dimensions", "permutation", "slice_sizes", "start_indices", "limit_indices", "strides"):
        match = re.search(rf"{key}\s*=\s*\[([^\]]*)\]", line)
        if match:
            attrs[key] = [int(item.strip()) for item in match.group(1).split(",") if item.strip()]
    direction = re.search(r"comparison_direction\s*=\s*#stablehlo<comparison_direction\s+([A-Z]+)>", line)
    if direction:
        attrs["comparison_direction"] = direction.group(1)
    return attrs


def _region_window(lines: list[str], start: int) -> str:
    selected = []
    depth = 0
    for line in lines[start : min(len(lines), start + 16)]:
        selected.append(line.strip())
        depth += line.count("{")
        depth -= line.count("}")
        if selected and depth <= 0 and "}" in line:
            break
    return " ".join(selected)


def _region_reducer(lines: list[str], start: int) -> str | None:
    depth = 0
    for line in lines[start : min(len(lines), start + 16)]:
        depth += line.count("{")
        depth -= line.count("}")
        match = re.search(r"stablehlo\.(add|multiply|maximum|minimum|and|or)", line)
        if match:
            return _OP_MAP.get(f"stablehlo.{match.group(1)}", match.group(1))
        if depth <= 0 and line.strip().endswith("}"):
            break
    return None


def _first_known_type(graph: Graph, names: Iterable[str]) -> TensorType | None:
    for name in names:
        if name in graph.values:
            return graph.values[name]
    return None


def _hidden_size(types: Iterable[TensorType]) -> int:
    for tensor_type in types:
        if tensor_type.shape and isinstance(tensor_type.shape[-1], int):
            return int(tensor_type.shape[-1])
    return 1
