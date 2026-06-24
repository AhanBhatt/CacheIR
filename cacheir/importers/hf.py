from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cacheir.ir import Graph, TensorType


@dataclass
class ModelConfig:
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dtype: str = "float32"
    tie_word_embeddings: bool = False
    quantization: str | None = None

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            architecture=data.get("architecture") or (data.get("architectures") or ["Unknown"])[0],
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data.get("intermediate_size", data.get("ffn_hidden_size", data["hidden_size"] * 4))),
            num_layers=int(data.get("num_layers", data.get("num_hidden_layers", data.get("n_layer", 0)))),
            num_attention_heads=int(data.get("num_attention_heads", data.get("n_head", 0))),
            num_key_value_heads=int(data.get("num_key_value_heads", data.get("num_attention_heads", data.get("n_head", 0)))),
            vocab_size=int(data["vocab_size"]),
            max_position_embeddings=int(data.get("max_position_embeddings", data.get("seq_length", 2048))),
            rope_theta=float(data.get("rope_theta", data.get("rope_freq_base", 10000.0))),
            rms_norm_eps=float(data.get("rms_norm_eps", data.get("layer_norm_epsilon", 1e-6))),
            dtype=str(data.get("torch_dtype", data.get("dtype", "float32"))).replace("torch.", ""),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            quantization=_quantization_name(data.get("quantization_config")),
        )


def _quantization_name(data: Any) -> str | None:
    if not data:
        return None
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        method = data.get("quant_method") or data.get("load_in_4bit") or data.get("bits")
        return str(method) if method else "quantized"
    return "quantized"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dtype_name(dtype: Any) -> str:
    text = str(dtype).replace("torch.", "")
    aliases = {
        "float": "float32",
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "int8": "int8",
        "uint8": "uint8",
    }
    return aliases.get(text, text)


def discover_weight_shapes(model_path: str | Path) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Discover tensor shapes without forcing callers through a framework runtime."""
    path = Path(model_path)
    shapes: dict[str, tuple[int, ...]] = {}
    files: dict[str, str] = {}

    npz_files = []
    if path.is_file() and path.suffix == ".npz":
        npz_files.append(path)
    elif path.is_dir():
        npz_files.extend(path.glob("*.npz"))
    if npz_files:
        import numpy as np

        for npz_path in npz_files:
            with np.load(npz_path) as data:
                for key in data.files:
                    shapes[key] = tuple(int(dim) for dim in data[key].shape)
                    files[key] = npz_path.name
        return shapes, files

    safetensor_files = []
    if path.is_file() and path.suffix == ".safetensors":
        safetensor_files.append(path)
    elif path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            weight_map = _read_json(index).get("weight_map", {})
            safetensor_files.extend(path / file for file in sorted(set(weight_map.values())))
        else:
            safetensor_files.extend(sorted(path.glob("*.safetensors")))

    if safetensor_files:
        try:
            from safetensors import safe_open
        except ImportError:
            return shapes, files

        for tensor_file in safetensor_files:
            with safe_open(tensor_file, framework="np") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    shapes[key] = tuple(int(dim) for dim in tensor.shape)
                    files[key] = tensor_file.name

    return shapes, files


def _shape_or_expected(
    weight_shapes: dict[str, tuple[int, ...]],
    key: str,
    expected: tuple[int, ...],
) -> tuple[int, ...]:
    return weight_shapes.get(key, expected)


def _add_weight(
    graph: Graph,
    value_name: str,
    key: str,
    expected_shape: tuple[int, ...],
    dtype: str,
    weight_shapes: dict[str, tuple[int, ...]],
    weight_files: dict[str, str],
    weight_key_map: dict[str, str],
    quant: str | None = None,
) -> None:
    actual_key = weight_key_map.get(key, key)
    shape = _shape_or_expected(weight_shapes, actual_key, _shape_or_expected(weight_shapes, key, expected_shape))
    graph.add_weight(
        value_name,
        key=actual_key,
        tensor_type=TensorType(shape, dtype),
        quant=quant,
        file=weight_files.get(actual_key, weight_files.get(key)),
    )


def build_decoder_graph(
    config: ModelConfig,
    *,
    weight_shapes: dict[str, tuple[int, ...]] | None = None,
    weight_files: dict[str, str] | None = None,
    weight_key_map: dict[str, str] | None = None,
) -> Graph:
    """Build a logical CacheIR graph for Llama/Mistral/Qwen-style decoder blocks."""
    weight_shapes = weight_shapes or {}
    weight_files = weight_files or {}
    weight_key_map = weight_key_map or {}
    graph = Graph(name=config.architecture, mode="logical", target="generic")
    graph.attrs["model_config"] = config.to_dict()
    graph.add_input("input_ids", TensorType(("batch", "seq"), "int64"))

    dtype = _dtype_name(config.dtype)
    h = config.hidden_size
    i = config.intermediate_size
    kv = config.kv_dim

    _add_weight(graph, "tok_embeddings", "model.embed_tokens.weight", (config.vocab_size, h), dtype, weight_shapes, weight_files, weight_key_map)
    _add_weight(graph, "final_norm_w", "model.norm.weight", (h,), dtype, weight_shapes, weight_files, weight_key_map)
    lm_key = "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    _add_weight(graph, "lm_head_w", lm_key, (config.vocab_size, h), dtype, weight_shapes, weight_files, weight_key_map)

    graph.add_node("token_embedding", ["input_ids", "tok_embeddings"], ["hidden_0"], name="embed")
    hidden = "hidden_0"

    for layer in range(config.num_layers):
        prefix = f"model.layers.{layer}"
        lname = f"layer{layer}"
        _add_weight(graph, f"{lname}_attn_norm_w", f"{prefix}.input_layernorm.weight", (h,), dtype, weight_shapes, weight_files, weight_key_map)
        _add_weight(graph, f"{lname}_q_w", f"{prefix}.self_attn.q_proj.weight", (h, h), dtype, weight_shapes, weight_files, weight_key_map)
        _add_weight(graph, f"{lname}_k_w", f"{prefix}.self_attn.k_proj.weight", (kv, h), dtype, weight_shapes, weight_files, weight_key_map)
        _add_weight(graph, f"{lname}_v_w", f"{prefix}.self_attn.v_proj.weight", (kv, h), dtype, weight_shapes, weight_files, weight_key_map)
        _add_weight(graph, f"{lname}_o_w", f"{prefix}.self_attn.o_proj.weight", (h, h), dtype, weight_shapes, weight_files, weight_key_map)
        _add_weight(graph, f"{lname}_mlp_norm_w", f"{prefix}.post_attention_layernorm.weight", (h,), dtype, weight_shapes, weight_files, weight_key_map)
        _add_weight(graph, f"{lname}_gate_w", f"{prefix}.mlp.gate_proj.weight", (i, h), dtype, weight_shapes, weight_files, weight_key_map)
        _add_weight(graph, f"{lname}_up_w", f"{prefix}.mlp.up_proj.weight", (i, h), dtype, weight_shapes, weight_files, weight_key_map)
        _add_weight(graph, f"{lname}_down_w", f"{prefix}.mlp.down_proj.weight", (h, i), dtype, weight_shapes, weight_files, weight_key_map)

        attn_norm = f"{lname}_attn_norm"
        q = f"{lname}_q"
        k = f"{lname}_k"
        v = f"{lname}_v"
        q_rope = f"{lname}_q_rope"
        k_rope = f"{lname}_k_rope"
        attn = f"{lname}_attn"
        attn_out = f"{lname}_attn_out"
        post_attn = f"{lname}_post_attn"

        graph.add_node("rms_norm", [hidden, f"{lname}_attn_norm_w"], [attn_norm], attrs={"eps": config.rms_norm_eps, "layer": layer}, name=f"{lname}.attn_norm")
        graph.add_node("matmul", [attn_norm, f"{lname}_q_w"], [q], attrs={"layer": layer, "projection": "q"}, name=f"{lname}.q_proj")
        graph.add_node("matmul", [attn_norm, f"{lname}_k_w"], [k], attrs={"layer": layer, "projection": "k"}, name=f"{lname}.k_proj")
        graph.add_node("matmul", [attn_norm, f"{lname}_v_w"], [v], attrs={"layer": layer, "projection": "v"}, name=f"{lname}.v_proj")
        graph.add_node(
            "rope",
            [q, k],
            [q_rope, k_rope],
            attrs={"layer": layer, "head_dim": config.head_dim, "rope_theta": config.rope_theta},
            name=f"{lname}.rope",
        )
        graph.add_node(
            "grouped_query_attention",
            [q_rope, k_rope, v],
            [attn],
            attrs={
                "layer": layer,
                "num_heads": config.num_attention_heads,
                "num_kv_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "kv_cache": "paged",
            },
            name=f"{lname}.attention",
        )
        graph.add_node("matmul", [attn, f"{lname}_o_w"], [attn_out], attrs={"layer": layer, "projection": "o"}, name=f"{lname}.o_proj")
        graph.add_node("add", [hidden, attn_out], [post_attn], attrs={"layer": layer, "kind": "residual"}, name=f"{lname}.attn_residual")

        mlp_norm = f"{lname}_mlp_norm"
        gate = f"{lname}_gate"
        up = f"{lname}_up"
        gate_act = f"{lname}_gate_act"
        gated = f"{lname}_gated"
        down = f"{lname}_down"
        post_mlp = f"{lname}_post_mlp"
        graph.add_node("rms_norm", [post_attn, f"{lname}_mlp_norm_w"], [mlp_norm], attrs={"eps": config.rms_norm_eps, "layer": layer}, name=f"{lname}.mlp_norm")
        graph.add_node("matmul", [mlp_norm, f"{lname}_gate_w"], [gate], attrs={"layer": layer, "projection": "gate"}, name=f"{lname}.gate_proj")
        graph.add_node("matmul", [mlp_norm, f"{lname}_up_w"], [up], attrs={"layer": layer, "projection": "up"}, name=f"{lname}.up_proj")
        graph.add_node("silu", [gate], [gate_act], attrs={"layer": layer}, name=f"{lname}.silu")
        graph.add_node("elementwise_mul", [gate_act, up], [gated], attrs={"layer": layer}, name=f"{lname}.swiglu_mul")
        graph.add_node("matmul", [gated, f"{lname}_down_w"], [down], attrs={"layer": layer, "projection": "down"}, name=f"{lname}.down_proj")
        graph.add_node("add", [post_attn, down], [post_mlp], attrs={"layer": layer, "kind": "residual"}, name=f"{lname}.mlp_residual")
        hidden = post_mlp

    graph.add_node("rms_norm", [hidden, "final_norm_w"], ["final_hidden"], attrs={"eps": config.rms_norm_eps}, name="final_norm")
    graph.add_node("matmul", ["final_hidden", "lm_head_w"], ["logits"], attrs={"projection": "lm_head"}, name="lm_head")
    graph.outputs = ["logits"]
    return graph


def import_hf_decoder(model_path: str | Path) -> tuple[ModelConfig, Graph]:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    if config_path.suffix != ".json":
        config_path = path / "config.json"
    config = ModelConfig.from_dict(_read_json(config_path))
    shapes, files = discover_weight_shapes(path)
    return config, build_decoder_graph(config, weight_shapes=shapes, weight_files=files)
