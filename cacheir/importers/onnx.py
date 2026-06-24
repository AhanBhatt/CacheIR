from __future__ import annotations

from pathlib import Path
from typing import Any

from cacheir.importers.hf import ModelConfig
from cacheir.ir import Graph, TensorType


_ONNX_OPS = {
    "MatMul": "matmul",
    "Gemm": "matmul",
    "Add": "add",
    "Mul": "elementwise_mul",
    "Softmax": "softmax",
    "Sigmoid": "sigmoid",
    "Silu": "silu",
}


def _tensor_type(value_info: Any) -> TensorType:
    tensor = value_info.type.tensor_type
    dtype = {
        1: "float32",
        7: "int64",
        10: "float16",
        16: "bfloat16",
    }.get(tensor.elem_type, "float32")
    shape = []
    for dim in tensor.shape.dim:
        if dim.dim_value:
            shape.append(int(dim.dim_value))
        else:
            shape.append(dim.dim_param or "?")
    return TensorType(tuple(shape), dtype)


def import_onnx_graph(path: str | Path) -> tuple[ModelConfig, Graph]:
    """Import a generic ONNX graph into CacheIR.

    This keeps ONNX support honest without pretending every exported model has
    recoverable Llama metadata. Transformer-specific optimization still works best
    through the Hugging Face/GGUF paths.
    """
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX import requires the optional 'onnx' dependency") from exc

    model = onnx.load(Path(path))
    graph_proto = model.graph
    graph = Graph(name=graph_proto.name or "onnx_model", mode="logical", target="generic")

    initializer_names = {init.name for init in graph_proto.initializer}
    for value in graph_proto.input:
        if value.name not in initializer_names:
            graph.add_input(value.name, _tensor_type(value))
    for init in graph_proto.initializer:
        dims = tuple(int(dim) for dim in init.dims)
        dtype = {1: "float32", 7: "int64", 10: "float16", 16: "bfloat16"}.get(init.data_type, "float32")
        graph.add_weight(init.name, key=init.name, tensor_type=TensorType(dims, dtype))

    for idx, node in enumerate(graph_proto.node):
        op = _ONNX_OPS.get(node.op_type, node.op_type.lower())
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        graph.add_node(op, node.input, node.output, attrs=attrs, name=node.name or f"onnx.{idx}.{node.op_type}")

    graph.outputs = [out.name for out in graph_proto.output]
    config = ModelConfig(
        architecture="ONNXGraph",
        hidden_size=1,
        intermediate_size=1,
        num_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=1,
    )
    return config, graph
