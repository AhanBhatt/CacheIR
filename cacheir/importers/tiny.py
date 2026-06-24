from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def create_tiny_model(
    path: str | Path,
    *,
    seed: int = 7,
    vocab_size: int = 64,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
) -> Path:
    """Create a tiny Llama-shaped model that exercises the compiler/runtime."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    config = {
        "architectures": ["CacheIRTinyForCausalLM"],
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": vocab_size,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "torch_dtype": "float32",
        "tie_word_embeddings": False,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def normal(shape: tuple[int, ...], scale: float = 0.02) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    weights: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": normal((vocab_size, hidden_size)),
        "model.norm.weight": np.ones((hidden_size,), dtype=np.float32),
        "lm_head.weight": normal((vocab_size, hidden_size)),
    }
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_key_value_heads * head_dim
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        weights[f"{prefix}.self_attn.q_proj.weight"] = normal((hidden_size, hidden_size))
        weights[f"{prefix}.self_attn.k_proj.weight"] = normal((kv_dim, hidden_size))
        weights[f"{prefix}.self_attn.v_proj.weight"] = normal((kv_dim, hidden_size))
        weights[f"{prefix}.self_attn.o_proj.weight"] = normal((hidden_size, hidden_size))
        weights[f"{prefix}.post_attention_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        weights[f"{prefix}.mlp.gate_proj.weight"] = normal((intermediate_size, hidden_size))
        weights[f"{prefix}.mlp.up_proj.weight"] = normal((intermediate_size, hidden_size))
        weights[f"{prefix}.mlp.down_proj.weight"] = normal((hidden_size, intermediate_size))

    np.savez(out / "weights.npz", **weights)
    tokenizer = {"type": "byte_mod", "vocab_size": vocab_size}
    (out / "tokenizer_config.json").write_text(json.dumps(tokenizer, indent=2), encoding="utf-8")
    return out
