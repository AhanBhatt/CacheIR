from __future__ import annotations

import json
import re

from cacheir.ir import Graph, TensorType


def emit_cacheir_dialect(graph: Graph) -> str:
    """Emit an experimental MLIR-style CacheIR dialect module."""
    lines = ["module {", f"  cacheir.graph @{_sym(graph.name)} attributes {{mode = \"{graph.mode}\", target = \"{graph.target}\"}} {{"]
    for name, tensor_type in graph.inputs.items():
        lines.append(f"    cacheir.input %{_sym(name)} : !cacheir.tensor<{_shape(tensor_type.shape)}x{tensor_type.dtype}>")
    for name, spec in graph.weights.items():
        lines.append(
            f"    cacheir.weight %{_sym(name)} {{key = \"{spec.key}\"}} : !cacheir.tensor<{_shape(spec.tensor_type.shape)}x{spec.tensor_type.dtype}>"
        )
    for node in graph.nodes:
        attrs = json.dumps(node.attrs, sort_keys=True).replace('"', '\\"')
        inputs = ", ".join(f"%{_sym(value)}" for value in node.inputs)
        outputs = ", ".join(f"%{_sym(value)}" for value in node.outputs)
        lines.append(f"    {outputs} = cacheir.{node.op}({inputs}) {{attrs = \"{attrs}\"}}")
    returns = ", ".join(f"%{_sym(value)}" for value in graph.outputs)
    lines.append(f"    cacheir.return {returns}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _shape(shape: tuple[int | str, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _sym(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").replace("/", "_")


def parse_cacheir_dialect(text: str) -> Graph:
    """Parse the MLIR-style CacheIR dialect emitted by :func:`emit_cacheir_dialect`.

    This parser is intentionally scoped to CacheIR's own textual emitter. It is
    useful for round-trip tests, artifact inspection, and MLIR dialect
    experiments without requiring an MLIR runtime in the developer environment.
    """

    graph_match = re.search(
        r"cacheir\.graph\s+@(?P<name>[\w\d_.$-]+)\s+attributes\s+\{mode\s*=\s*\"(?P<mode>[^\"]+)\",\s*target\s*=\s*\"(?P<target>[^\"]+)\"\}",
        text,
    )
    if not graph_match:
        raise ValueError("CacheIR dialect text does not contain a cacheir.graph operation")
    graph = Graph(name=graph_match.group("name"), mode=graph_match.group("mode"), target=graph_match.group("target"))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        input_match = re.match(r"cacheir\.input\s+%([\w\d_.$-]+)\s*:\s*(!cacheir\.tensor<[^>]+>)", line)
        if input_match:
            graph.add_input(input_match.group(1), _parse_cacheir_tensor(input_match.group(2)))
            continue

        weight_match = re.match(
            r"cacheir\.weight\s+%([\w\d_.$-]+)\s+\{key\s*=\s*\"([^\"]+)\"\}\s*:\s*(!cacheir\.tensor<[^>]+>)",
            line,
        )
        if weight_match:
            graph.add_weight(weight_match.group(1), weight_match.group(2), _parse_cacheir_tensor(weight_match.group(3)))
            continue

        return_match = re.match(r"cacheir\.return\s+(.+)", line)
        if return_match:
            graph.outputs = [value.lstrip("%") for value in return_match.group(1).split(",") if value.strip()]
            continue

        op_match = re.match(
            r"(?P<outputs>%[\w\d_.$-]+(?:\s*,\s*%[\w\d_.$-]+)*)\s*=\s*cacheir\.(?P<op>[\w\d_]+)\((?P<inputs>[^)]*)\)(?:\s+\{attrs\s*=\s*\"(?P<attrs>.*)\"\})?",
            line,
        )
        if op_match:
            outputs = [value.strip().lstrip("%") for value in op_match.group("outputs").split(",")]
            inputs = [value.strip().lstrip("%") for value in op_match.group("inputs").split(",") if value.strip()]
            attrs = _decode_attrs(op_match.group("attrs"))
            known_type = next((graph.values[name] for name in inputs if name in graph.values), TensorType(("batch", "seq", "hidden"), "float32"))
            graph.add_node(op_match.group("op"), inputs, outputs, attrs=attrs, output_types=[known_type for _ in outputs])
    return graph


def verify_cacheir_dialect(text: str) -> list[str]:
    errors: list[str] = []
    try:
        graph = parse_cacheir_dialect(text)
    except ValueError as exc:
        return [str(exc)]
    if not graph.outputs:
        errors.append("cacheir.return is required")
    try:
        graph.validate()
    except ValueError as exc:
        errors.append(str(exc))
    for node in graph.nodes:
        if not node.outputs:
            errors.append(f"cacheir.{node.op} must produce at least one value")
    return errors


def _parse_cacheir_tensor(text: str) -> TensorType:
    body = re.search(r"!cacheir\.tensor<([^>]+)>", text)
    if not body:
        return TensorType(("batch", "seq", "hidden"), "float32")
    parts = [part.strip() for part in body.group(1).split("x") if part.strip()]
    if not parts:
        return TensorType((), "float32")
    dtype = parts[-1]
    shape = tuple(int(part) if part.isdigit() else part for part in parts[:-1])
    return TensorType(shape, dtype)


def _decode_attrs(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    try:
        return json.loads(raw.replace('\\"', '"'))
    except json.JSONDecodeError:
        return {}
