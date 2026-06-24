from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cacheir.importers.gguf import import_gguf_metadata
from cacheir.importers.hf import ModelConfig, build_decoder_graph, import_hf_decoder
from cacheir.importers.onnx import import_onnx_graph
from cacheir.importers.stablehlo import import_stablehlo_text
from cacheir.hardware import profile_hardware
from cacheir.passes import (
    CompilerContext,
    ConstantFolding,
    DeadNodeElimination,
    ExecutionPlanning,
    HardwareAdaptivePlanning,
    KernelSelection,
    LayoutConversion,
    MemoryPlanning,
    PassManager,
    PrefillDecodeSpecialization,
    QKVProjectionFusion,
    QuantizationAwareLowering,
    RMSNormQKVRoPEFusion,
    ShapeInference,
    SwiGLUFusion,
)
from cacheir.runtime.artifact import CompileArtifact


@dataclass
class CompilerOptions:
    target: str = "cpu"
    quant: str | None = None
    mode: tuple[str, ...] = ("prefill", "decode")
    max_batch: int = 1
    max_seq: int = 128
    explain: bool = True


def _config_from_gguf(metadata: dict[str, object]) -> ModelConfig:
    meta = metadata["metadata"]  # type: ignore[index]
    assert isinstance(meta, dict)
    arch = str(meta.get("general.architecture", "llama"))
    prefix = arch
    hidden = int(meta.get(f"{prefix}.embedding_length", meta.get("llama.embedding_length", 1)))
    heads = int(meta.get(f"{prefix}.attention.head_count", meta.get("llama.attention.head_count", 1)))
    kv_heads = int(meta.get(f"{prefix}.attention.head_count_kv", meta.get("llama.attention.head_count_kv", heads)))
    return ModelConfig(
        architecture=f"GGUF:{arch}",
        hidden_size=hidden,
        intermediate_size=int(meta.get(f"{prefix}.feed_forward_length", meta.get("llama.feed_forward_length", hidden * 4))),
        num_layers=int(meta.get(f"{prefix}.block_count", meta.get("llama.block_count", 0))),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        vocab_size=int(meta.get(f"{prefix}.vocab_size", meta.get("tokenizer.ggml.tokens", [0]) and len(meta.get("tokenizer.ggml.tokens", [])) or 1)),
        max_position_embeddings=int(meta.get(f"{prefix}.context_length", meta.get("llama.context_length", 2048))),
        rope_theta=float(meta.get(f"{prefix}.rope.freq_base", meta.get("llama.rope.freq_base", 10000.0))),
        rms_norm_eps=float(meta.get(f"{prefix}.attention.layer_norm_rms_epsilon", meta.get("llama.attention.layer_norm_rms_epsilon", 1e-6))),
        dtype="float16",
        quantization="gguf",
    )


def _import_model(model_path: str | Path) -> tuple[ModelConfig, object]:
    path = Path(model_path)
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return import_onnx_graph(path)
    if suffix in {".stablehlo", ".mhlo"}:
        return import_stablehlo_text(path)
    if suffix == ".gguf":
        metadata = import_gguf_metadata(path)
        config = _config_from_gguf(metadata)
        weight_shapes = {
            tensor["name"]: _gguf_logical_shape(tuple(int(dim) for dim in tensor["shape"]))
            for tensor in metadata.get("tensors", [])  # type: ignore[index]
        }
        key_map = _gguf_key_map(config)
        return config, build_decoder_graph(config, weight_shapes=weight_shapes, weight_files={}, weight_key_map=key_map)
    return import_hf_decoder(path)


def _gguf_key_map(config: ModelConfig) -> dict[str, str]:
    mapping = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    for layer in range(config.num_layers):
        hf = f"model.layers.{layer}"
        gguf = f"blk.{layer}"
        mapping.update(
            {
                f"{hf}.input_layernorm.weight": f"{gguf}.attn_norm.weight",
                f"{hf}.self_attn.q_proj.weight": f"{gguf}.attn_q.weight",
                f"{hf}.self_attn.k_proj.weight": f"{gguf}.attn_k.weight",
                f"{hf}.self_attn.v_proj.weight": f"{gguf}.attn_v.weight",
                f"{hf}.self_attn.o_proj.weight": f"{gguf}.attn_output.weight",
                f"{hf}.post_attention_layernorm.weight": f"{gguf}.ffn_norm.weight",
                f"{hf}.mlp.gate_proj.weight": f"{gguf}.ffn_gate.weight",
                f"{hf}.mlp.up_proj.weight": f"{gguf}.ffn_up.weight",
                f"{hf}.mlp.down_proj.weight": f"{gguf}.ffn_down.weight",
            }
        )
    return mapping


def _gguf_logical_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(shape)) if len(shape) > 1 else shape


def _pipeline() -> PassManager:
    return PassManager(
        [
            ShapeInference(),
            ConstantFolding(),
            QKVProjectionFusion(),
            RMSNormQKVRoPEFusion(),
            SwiGLUFusion(),
            PrefillDecodeSpecialization(),
            LayoutConversion(),
            QuantizationAwareLowering(),
            ShapeInference(),
            DeadNodeElimination(),
            HardwareAdaptivePlanning(),
            KernelSelection(),
            ExecutionPlanning(),
            MemoryPlanning(),
        ]
    )


def compile_model(
    model_path: str | Path,
    *,
    target: str = "cpu",
    quant: str | None = None,
    mode: Iterable[str] = ("prefill", "decode"),
    max_batch: int = 1,
    max_seq: int = 128,
    output: str | Path | None = None,
) -> CompileArtifact:
    """Import, optimize, lower, and package a transformer graph."""
    modes = tuple(mode)
    config, logical_graph = _import_model(model_path)
    graphs = {}
    traces = {}
    manager = _pipeline()
    hardware = profile_hardware().to_dict()

    for execution_mode in modes:
        graph = logical_graph.clone(name=f"{logical_graph.name}_{execution_mode}", mode=execution_mode, target=target)
        context = CompilerContext(target=target, mode=execution_mode, quant=quant, max_batch=max_batch, max_seq=max_seq, hardware_profile=hardware)
        optimized, pass_traces = manager.run(graph, context)
        graphs[execution_mode] = optimized
        traces[execution_mode] = [trace.to_dict() for trace in pass_traces]

    artifact = CompileArtifact(
        target=target,
        quant=quant,
        model_path=str(Path(model_path)),
        config=config,
        graphs=graphs,
        pass_traces=traces,
        metadata={
            "compiler": "CacheIR",
            "modes": list(modes),
            "max_batch": max_batch,
            "max_seq": max_seq,
            "hardware_profile": hardware,
        },
    )
    if output is not None:
        out = Path(output)
        if out.suffix:
            artifact.save(out)
        else:
            artifact.save_bundle(out)
    return artifact
