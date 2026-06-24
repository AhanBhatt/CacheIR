from __future__ import annotations

import copy
import difflib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable


DTYPE_SIZES = {
    "bool": 1,
    "int4": 0.5,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


ShapeDim = int | str


@dataclass(frozen=True)
class TensorType:
    shape: tuple[ShapeDim, ...]
    dtype: str = "float32"
    layout: str = "row_major"

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "layout": self.layout}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorType":
        return cls(tuple(data["shape"]), data.get("dtype", "float32"), data.get("layout", "row_major"))

    def numel(self, symbols: dict[str, int] | None = None) -> int | None:
        total = 1
        for dim in self.shape:
            if isinstance(dim, int):
                total *= dim
            elif symbols and dim in symbols:
                total *= symbols[dim]
            else:
                return None
        return total

    def nbytes(self, symbols: dict[str, int] | None = None) -> int | None:
        elems = self.numel(symbols)
        if elems is None:
            return None
        size = DTYPE_SIZES.get(self.dtype)
        if size is None:
            return None
        return int(math.ceil(elems * size))

    def with_layout(self, layout: str) -> "TensorType":
        return TensorType(self.shape, self.dtype, layout)

    def format(self) -> str:
        dims = ",".join(str(dim) for dim in self.shape)
        layout = "" if self.layout == "row_major" else f" layout={self.layout}"
        return f"{self.dtype}[{dims}]{layout}"


@dataclass
class WeightSpec:
    key: str
    tensor_type: TensorType
    quant: str | None = None
    file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tensor_type": self.tensor_type.to_dict(),
            "quant": self.quant,
            "file": self.file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeightSpec":
        return cls(
            key=data["key"],
            tensor_type=TensorType.from_dict(data["tensor_type"]),
            quant=data.get("quant"),
            file=data.get("file"),
        )


@dataclass
class Node:
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    side_effect: bool = False

    def clone(self) -> "Node":
        return copy.deepcopy(self)

    def label(self) -> str:
        return self.name or self.op

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attrs": self.attrs,
            "name": self.name,
            "side_effect": self.side_effect,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        return cls(
            op=data["op"],
            inputs=list(data.get("inputs", [])),
            outputs=list(data.get("outputs", [])),
            attrs=dict(data.get("attrs", {})),
            name=data.get("name"),
            side_effect=bool(data.get("side_effect", False)),
        )

    def format(self) -> str:
        outs = ", ".join(f"%{out}" for out in self.outputs)
        ins = ", ".join(f"%{inp}" for inp in self.inputs)
        attrs = ""
        if self.attrs:
            stable = json.dumps(self.attrs, sort_keys=True)
            attrs = f" {stable}"
        name = f" @{self.name}" if self.name else ""
        return f"  {outs} = {self.op}({ins}){attrs}{name}"


@dataclass
class Graph:
    name: str
    mode: str = "logical"
    target: str = "cpu"
    inputs: dict[str, TensorType] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    values: dict[str, TensorType] = field(default_factory=dict)
    weights: dict[str, WeightSpec] = field(default_factory=dict)
    constants: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    def clone(self, *, name: str | None = None, mode: str | None = None, target: str | None = None) -> "Graph":
        graph = copy.deepcopy(self)
        if name is not None:
            graph.name = name
        if mode is not None:
            graph.mode = mode
        if target is not None:
            graph.target = target
        return graph

    def add_input(self, name: str, tensor_type: TensorType) -> None:
        self.inputs[name] = tensor_type
        self.values[name] = tensor_type

    def add_weight(self, name: str, key: str, tensor_type: TensorType, quant: str | None = None, file: str | None = None) -> None:
        self.weights[name] = WeightSpec(key=key, tensor_type=tensor_type, quant=quant, file=file)
        self.values[name] = tensor_type

    def add_constant(self, name: str, value: Any, tensor_type: TensorType) -> None:
        self.constants[name] = value
        self.values[name] = tensor_type

    def add_node(
        self,
        op: str,
        inputs: Iterable[str],
        outputs: Iterable[str],
        *,
        attrs: dict[str, Any] | None = None,
        name: str | None = None,
        side_effect: bool = False,
        output_types: Iterable[TensorType] | None = None,
    ) -> Node:
        node = Node(op=op, inputs=list(inputs), outputs=list(outputs), attrs=attrs or {}, name=name, side_effect=side_effect)
        self.nodes.append(node)
        if output_types is not None:
            for output, tensor_type in zip(node.outputs, output_types):
                self.values[output] = tensor_type
        return node

    def producer_map(self) -> dict[str, Node]:
        producers: dict[str, Node] = {}
        for node in self.nodes:
            for output in node.outputs:
                producers[output] = node
        return producers

    def consumer_map(self) -> dict[str, list[Node]]:
        consumers: dict[str, list[Node]] = {}
        for node in self.nodes:
            for inp in node.inputs:
                consumers.setdefault(inp, []).append(node)
        return consumers

    def replace_all_uses(self, old: str, new: str) -> None:
        for node in self.nodes:
            node.inputs = [new if inp == old else inp for inp in node.inputs]
        self.outputs = [new if out == old else out for out in self.outputs]

    def remove_nodes(self, doomed: set[str]) -> None:
        self.nodes = [node for node in self.nodes if node.name not in doomed]

    def fresh(self, prefix: str) -> str:
        base = prefix.replace(".", "_")
        existing = set(self.values) | {out for node in self.nodes for out in node.outputs}
        idx = 0
        while f"{base}_{idx}" in existing:
            idx += 1
        return f"{base}_{idx}"

    def value_type(self, name: str) -> TensorType | None:
        return self.values.get(name)

    def validate(self) -> None:
        known = set(self.inputs) | set(self.weights) | set(self.constants)
        produced: set[str] = set()
        for node in self.nodes:
            for inp in node.inputs:
                if inp not in known and inp not in produced:
                    raise ValueError(f"Node {node.label()} reads unknown value %{inp}")
            for output in node.outputs:
                if output in known or output in produced:
                    raise ValueError(f"Value %{output} is produced more than once")
                produced.add(output)
        for output in self.outputs:
            if output not in known and output not in produced:
                raise ValueError(f"Graph output %{output} is unknown")

    def to_text(self) -> str:
        lines = [f"graph {self.name}(mode={self.mode}, target={self.target}) {{"]
        for name, tensor_type in self.inputs.items():
            lines.append(f"  input %{name}: {tensor_type.format()}")
        for name, spec in self.weights.items():
            quant = f" quant={spec.quant}" if spec.quant else ""
            lines.append(f"  weight %{name}: {spec.tensor_type.format()} <- {spec.key}{quant}")
        for name, value in self.constants.items():
            tensor_type = self.values.get(name)
            suffix = f": {tensor_type.format()}" if tensor_type else ""
            lines.append(f"  const %{name}{suffix} = {value}")
        for node in self.nodes:
            lines.append(node.format())
        lines.append("  return " + ", ".join(f"%{out}" for out in self.outputs))
        if self.attrs:
            lines.append("  attrs " + json.dumps(self.attrs, sort_keys=True))
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "target": self.target,
            "inputs": {name: ty.to_dict() for name, ty in self.inputs.items()},
            "outputs": self.outputs,
            "nodes": [node.to_dict() for node in self.nodes],
            "values": {name: ty.to_dict() for name, ty in self.values.items()},
            "weights": {name: spec.to_dict() for name, spec in self.weights.items()},
            "constants": self.constants,
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Graph":
        return cls(
            name=data["name"],
            mode=data.get("mode", "logical"),
            target=data.get("target", "cpu"),
            inputs={name: TensorType.from_dict(ty) for name, ty in data.get("inputs", {}).items()},
            outputs=list(data.get("outputs", [])),
            nodes=[Node.from_dict(node) for node in data.get("nodes", [])],
            values={name: TensorType.from_dict(ty) for name, ty in data.get("values", {}).items()},
            weights={name: WeightSpec.from_dict(spec) for name, spec in data.get("weights", {}).items()},
            constants=dict(data.get("constants", {})),
            attrs=dict(data.get("attrs", {})),
        )


def graph_text_diff(before: Graph, after: Graph) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.to_text().splitlines(),
            after.to_text().splitlines(),
            fromfile=f"{before.name}:before",
            tofile=f"{after.name}:after",
            lineterm="",
        )
    )
