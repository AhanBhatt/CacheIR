from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cacheir.ir import Graph, Node, TensorType
from cacheir.passes.base import CompilerContext, PassResult


def _same_input(a: Node, b: Node) -> bool:
    return bool(a.inputs and b.inputs and a.inputs[0] == b.inputs[0])


def _only_consumed_by(consumers: dict[str, list[Node]], value: str, allowed: list[Node]) -> bool:
    return all(any(consumer is candidate for candidate in allowed) for consumer in consumers.get(value, []))


@dataclass
class ShapeInference:
    name: str = "shape_inference"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        changed = False
        remarks: list[str] = []
        for node in graph.nodes:
            inferred = self._infer_node(graph, node)
            for output, tensor_type in zip(node.outputs, inferred):
                if tensor_type and graph.values.get(output) != tensor_type:
                    graph.values[output] = tensor_type
                    changed = True
        if changed:
            remarks.append("updated tensor shapes for graph values")
        return PassResult(changed, remarks)

    def _infer_node(self, graph: Graph, node: Node) -> list[TensorType | None]:
        values = graph.values
        op = node.op
        if op == "token_embedding":
            ids = values.get(node.inputs[0])
            emb = values.get(node.inputs[1])
            if ids and emb and len(emb.shape) == 2:
                return [TensorType(tuple(ids.shape) + (emb.shape[1],), emb.dtype)]
        if op == "rms_norm":
            return [values.get(node.inputs[0])]
        if op in {"matmul", "quantized_matmul"}:
            x = values.get(node.inputs[0])
            w = values.get(node.inputs[1])
            if x and w and len(w.shape) == 2:
                return [TensorType(tuple(x.shape[:-1]) + (w.shape[0],), x.dtype, x.layout)]
        if op == "qkv_projection":
            x = values.get(node.inputs[0])
            wq = values.get(node.inputs[1])
            wk = values.get(node.inputs[2])
            wv = values.get(node.inputs[3])
            if x and wq and wk and wv:
                base = tuple(x.shape[:-1])
                return [
                    TensorType(base + (wq.shape[0],), x.dtype),
                    TensorType(base + (wk.shape[0],), x.dtype),
                    TensorType(base + (wv.shape[0],), x.dtype),
                ]
        if op in {"fused_rmsnorm_qkv_rope", "quantized_fused_rmsnorm_qkv_rope"}:
            x = values.get(node.inputs[0])
            wq = values.get(node.inputs[2])
            wk = values.get(node.inputs[3])
            wv = values.get(node.inputs[4])
            if x and wq and wk and wv:
                base = tuple(x.shape[:-1])
                return [
                    TensorType(base + (wq.shape[0],), x.dtype),
                    TensorType(base + (wk.shape[0],), x.dtype),
                    TensorType(base + (wv.shape[0],), x.dtype),
                ]
        if op == "rope":
            return [values.get(node.inputs[0]), values.get(node.inputs[1])]
        if op in {"grouped_query_attention", "paged_attention_prefill", "paged_attention_decode"}:
            q = values.get(node.inputs[0])
            return [q]
        if op in {"add", "elementwise_mul", "silu", "sigmoid", "softmax"}:
            return [values.get(node.inputs[0])]
        if op in {"fused_swiglu", "quantized_fused_swiglu"}:
            x = values.get(node.inputs[0])
            gate_w = values.get(node.inputs[1])
            if x and gate_w:
                return [TensorType(tuple(x.shape[:-1]) + (gate_w.shape[0],), x.dtype)]
        return [None for _ in node.outputs]


@dataclass
class ConstantFolding:
    name: str = "constant_folding"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        changed = False
        remarks: list[str] = []
        kept: list[Node] = []
        for node in graph.nodes:
            if node.op not in {"add", "elementwise_mul", "silu", "sigmoid"}:
                kept.append(node)
                continue
            if not all(inp in graph.constants for inp in node.inputs):
                kept.append(node)
                continue
            arrays = [np.asarray(graph.constants[inp], dtype=np.float32) for inp in node.inputs]
            if node.op == "add":
                value = arrays[0] + arrays[1]
            elif node.op == "elementwise_mul":
                value = arrays[0] * arrays[1]
            elif node.op == "silu":
                value = arrays[0] / (1.0 + np.exp(-arrays[0]))
            else:
                value = 1.0 / (1.0 + np.exp(-arrays[0]))
            out = node.outputs[0]
            graph.constants[out] = value.tolist()
            graph.values[out] = TensorType(tuple(value.shape), "float32")
            changed = True
            remarks.append(f"folded {node.label()} into %{out}")
        graph.nodes = kept
        return PassResult(changed, remarks)


@dataclass
class QKVProjectionFusion:
    name: str = "qkv_projection_fusion"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        groups: dict[tuple[int, str], dict[str, Node]] = {}
        positions = {id(node): idx for idx, node in enumerate(graph.nodes)}
        for node in graph.nodes:
            if node.op != "matmul":
                continue
            projection = node.attrs.get("projection")
            if projection not in {"q", "k", "v"}:
                continue
            layer = int(node.attrs.get("layer", -1))
            groups.setdefault((layer, node.inputs[0]), {})[projection] = node

        replacements: list[tuple[int, set[str], Node]] = []
        for (layer, hidden), parts in groups.items():
            if set(parts) != {"q", "k", "v"}:
                continue
            q, k, v = parts["q"], parts["k"], parts["v"]
            first = min(positions[id(q)], positions[id(k)], positions[id(v)])
            fused = Node(
                op="qkv_projection",
                inputs=[hidden, q.inputs[1], k.inputs[1], v.inputs[1]],
                outputs=[q.outputs[0], k.outputs[0], v.outputs[0]],
                attrs={"layer": layer},
                name=f"layer{layer}.qkv_proj",
            )
            replacements.append((first, {q.name or "", k.name or "", v.name or ""}, fused))

        if not replacements:
            return PassResult(False, [])

        insert_at = {idx: fused for idx, _, fused in replacements}
        doomed = set().union(*(names for _, names, _ in replacements))
        new_nodes: list[Node] = []
        for idx, node in enumerate(graph.nodes):
            if idx in insert_at:
                new_nodes.append(insert_at[idx])
            if node.name not in doomed:
                new_nodes.append(node)
        graph.nodes = new_nodes
        return PassResult(True, [f"fused {len(replacements)} Q/K/V projection group(s)"])


@dataclass
class RMSNormQKVRoPEFusion:
    name: str = "rmsnorm_qkv_rope_fusion"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        producers = graph.producer_map()
        consumers = graph.consumer_map()
        positions = {id(node): idx for idx, node in enumerate(graph.nodes)}
        replacements: list[tuple[int, list[Node], Node]] = []

        for qkv in graph.nodes:
            if qkv.op != "qkv_projection":
                continue
            norm = producers.get(qkv.inputs[0])
            if not norm or norm.op != "rms_norm":
                continue
            q_out, k_out, v_out = qkv.outputs
            rope = None
            for maybe in consumers.get(q_out, []):
                if maybe.op == "rope" and maybe.inputs[:2] == [q_out, k_out]:
                    rope = maybe
                    break
            if rope is None:
                continue
            if len(consumers.get(qkv.inputs[0], [])) != 1:
                continue
            if not _only_consumed_by(consumers, q_out, [rope]) or not _only_consumed_by(consumers, k_out, [rope]):
                continue
            layer = qkv.attrs.get("layer", norm.attrs.get("layer"))
            fused = Node(
                op="fused_rmsnorm_qkv_rope",
                inputs=[norm.inputs[0], norm.inputs[1], qkv.inputs[1], qkv.inputs[2], qkv.inputs[3]],
                outputs=[rope.outputs[0], rope.outputs[1], v_out],
                attrs={
                    "layer": layer,
                    "eps": norm.attrs.get("eps", 1e-6),
                    "head_dim": rope.attrs.get("head_dim"),
                    "rope_theta": rope.attrs.get("rope_theta", 10000.0),
                },
                name=f"layer{layer}.fused_rmsnorm_qkv_rope",
            )
            first = min(positions[id(norm)], positions[id(qkv)], positions[id(rope)])
            replacements.append((first, [norm, qkv, rope], fused))

        if not replacements:
            return PassResult(False, [])

        insert_at = {idx: fused for idx, _, fused in replacements}
        doomed = {id(node) for _, nodes, _ in replacements for node in nodes}
        new_nodes: list[Node] = []
        for idx, node in enumerate(graph.nodes):
            if idx in insert_at:
                new_nodes.append(insert_at[idx])
            if id(node) not in doomed:
                new_nodes.append(node)
        graph.nodes = new_nodes
        return PassResult(True, [f"fused RMSNorm+QKV+RoPE in {len(replacements)} layer(s)"])


@dataclass
class SwiGLUFusion:
    name: str = "swiglu_fusion"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        producers = graph.producer_map()
        consumers = graph.consumer_map()
        positions = {id(node): idx for idx, node in enumerate(graph.nodes)}
        replacements: list[tuple[int, list[Node], Node]] = []

        for mul in graph.nodes:
            if mul.op != "elementwise_mul":
                continue
            left, right = mul.inputs
            silu = producers.get(left)
            up = producers.get(right)
            if not silu or silu.op != "silu":
                silu = producers.get(right)
                up = producers.get(left)
            if not silu or not up or silu.op != "silu" or up.op != "matmul":
                continue
            gate = producers.get(silu.inputs[0])
            if not gate or gate.op != "matmul":
                continue
            if gate.attrs.get("projection") != "gate" or up.attrs.get("projection") != "up":
                continue
            if not _same_input(gate, up):
                continue
            if len(consumers.get(gate.outputs[0], [])) != 1 or len(consumers.get(silu.outputs[0], [])) != 1:
                continue
            layer = gate.attrs.get("layer", up.attrs.get("layer"))
            fused = Node(
                op="fused_swiglu",
                inputs=[gate.inputs[0], gate.inputs[1], up.inputs[1]],
                outputs=mul.outputs,
                attrs={"layer": layer},
                name=f"layer{layer}.fused_swiglu",
            )
            first = min(positions[id(gate)], positions[id(up)], positions[id(silu)], positions[id(mul)])
            replacements.append((first, [gate, up, silu, mul], fused))

        if not replacements:
            return PassResult(False, [])
        insert_at = {idx: fused for idx, _, fused in replacements}
        doomed = {id(node) for _, nodes, _ in replacements for node in nodes}
        new_nodes: list[Node] = []
        for idx, node in enumerate(graph.nodes):
            if idx in insert_at:
                new_nodes.append(insert_at[idx])
            if id(node) not in doomed:
                new_nodes.append(node)
        graph.nodes = new_nodes
        return PassResult(True, [f"fused SwiGLU in {len(replacements)} layer(s)"])


@dataclass
class PrefillDecodeSpecialization:
    name: str = "prefill_decode_specialization"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        changed = False
        for node in graph.nodes:
            if node.op == "grouped_query_attention":
                node.op = "paged_attention_decode" if context.mode == "decode" else "paged_attention_prefill"
                node.attrs["mode"] = context.mode
                node.attrs["kv_cache_policy"] = "append_only_pages"
                changed = True
        graph.mode = context.mode
        graph.target = context.target
        graph.attrs["execution_mode"] = context.mode
        graph.attrs["kv_cache"] = {
            "policy": "paged",
            "compiled_with_graph": True,
            "decode_token_granularity": context.mode == "decode",
        }
        return PassResult(changed, [f"specialized graph for {context.mode}"])


@dataclass
class LayoutConversion:
    name: str = "layout_conversion"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        changed = False
        weight_layout = "row_major_out_in" if context.target == "cpu" else "tensorcore_interleaved"
        activation_layout = "bsh"
        for node in graph.nodes:
            if node.op in {
                "matmul",
                "quantized_matmul",
                "qkv_projection",
                "fused_rmsnorm_qkv_rope",
                "quantized_fused_rmsnorm_qkv_rope",
                "fused_swiglu",
                "quantized_fused_swiglu",
            }:
                if node.attrs.get("weight_layout") != weight_layout:
                    node.attrs["weight_layout"] = weight_layout
                    node.attrs["activation_layout"] = activation_layout
                    changed = True
        return PassResult(changed, [f"selected {weight_layout} weights for {context.target}"])


@dataclass
class QuantizationAwareLowering:
    name: str = "quantization_aware_lowering"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        if not context.quant:
            return PassResult(False, [])
        changed = False
        for node in graph.nodes:
            if node.op == "matmul":
                node.op = "quantized_matmul"
                node.attrs["quant"] = context.quant
                changed = True
            elif node.op == "fused_rmsnorm_qkv_rope":
                node.op = "quantized_fused_rmsnorm_qkv_rope"
                node.attrs["quant"] = context.quant
                changed = True
            elif node.op == "fused_swiglu":
                node.op = "quantized_fused_swiglu"
                node.attrs["quant"] = context.quant
                changed = True
        return PassResult(changed, [f"lowered matmul-family ops for {context.quant}"])


@dataclass
class KernelSelection:
    name: str = "kernel_selection"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        changed = False
        for node in graph.nodes:
            kernel = self._kernel_for(node, context)
            if node.attrs.get("kernel") != kernel:
                node.attrs["kernel"] = kernel
                changed = True
        graph.attrs["kernel_target"] = context.target
        return PassResult(changed, [f"selected kernels for {context.target}/{context.mode}"])

    def _kernel_for(self, node: Node, context: CompilerContext) -> str:
        prefix = "triton" if context.target in {"cuda", "triton"} else "cpu"
        table = {
            "token_embedding": "embedding_gather",
            "rms_norm": "rms_norm",
            "matmul": "matmul_avx" if prefix == "cpu" else "matmul_tensorcore",
            "quantized_matmul": "qmatmul_dequant" if prefix == "cpu" else "awq_int4_tensorcore",
            "fused_rmsnorm_qkv_rope": "fused_rmsnorm_qkv_rope",
            "quantized_fused_rmsnorm_qkv_rope": "awq_fused_rmsnorm_qkv_rope",
            "paged_attention_prefill": "paged_attention_prefill",
            "paged_attention_decode": "paged_attention_decode",
            "fused_swiglu": "fused_swiglu",
            "quantized_fused_swiglu": "awq_fused_swiglu",
            "add": "vector_add",
            "silu": "silu",
            "elementwise_mul": "elementwise_mul",
            "softmax": "softmax",
        }
        return f"{prefix}.{table.get(node.op, node.op)}"


@dataclass
class HardwareAdaptivePlanning:
    name: str = "hardware_adaptive_planning"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        profile = context.hardware_profile or {}
        gpus = profile.get("gpus") if isinstance(profile, dict) else []
        page_size = 32 if context.target in {"cuda", "triton"} and gpus else 16
        if context.mode == "decode":
            page_size = min(page_size, 16)

        hints = {
            "target": context.target,
            "mode": context.mode,
            "page_size": page_size,
            "prefer_fused_decode_attention": context.mode == "decode",
            "low_vram_mode": _is_low_vram(profile),
        }
        changed = graph.attrs.get("execution_hints") != hints
        graph.attrs["hardware_profile"] = profile
        graph.attrs["execution_hints"] = hints
        for node in graph.nodes:
            if node.op in {"paged_attention_decode", "paged_attention_prefill"}:
                node.attrs["page_size"] = page_size
                node.attrs["low_vram_mode"] = hints["low_vram_mode"]
            if node.op in {"matmul", "quantized_matmul"}:
                node.attrs["cost_class"] = "compute_bound"
            elif node.op in {"paged_attention_decode"}:
                node.attrs["cost_class"] = "kv_bandwidth_bound"
            elif node.op in {"paged_attention_prefill"}:
                node.attrs["cost_class"] = "mixed_attention"
            else:
                node.attrs.setdefault("cost_class", "elementwise_or_fused")
        return PassResult(changed, [f"planned hardware hints for {context.target}/{context.mode}"])


@dataclass
class ExecutionPlanning:
    name: str = "execution_planning"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        schedule = []
        for idx, node in enumerate(graph.nodes):
            node.attrs["schedule_index"] = idx
            schedule.append(
                {
                    "step": idx,
                    "name": node.name or f"{node.op}_{idx}",
                    "op": node.op,
                    "kernel": node.attrs.get("kernel", "unassigned"),
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "cost_class": node.attrs.get("cost_class", "unknown"),
                    "estimated_bytes": _node_estimated_bytes(graph, node, context),
                    "estimated_flops": _node_estimated_flops(graph, node, context),
                }
            )
        changed = graph.attrs.get("execution_schedule") != schedule
        graph.attrs["execution_schedule"] = schedule
        return PassResult(changed, [f"scheduled {len(schedule)} runtime call(s)"])


@dataclass
class DeadNodeElimination:
    name: str = "dead_node_elimination"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        needed = set(graph.outputs)
        kept_reversed: list[Node] = []
        removed = 0
        for node in reversed(graph.nodes):
            live = node.side_effect or any(out in needed for out in node.outputs)
            if not live:
                removed += 1
                continue
            kept_reversed.append(node)
            needed.update(node.inputs)
        graph.nodes = list(reversed(kept_reversed))
        return PassResult(removed > 0, [f"removed {removed} dead node(s)"] if removed else [])


@dataclass
class MemoryPlanning:
    name: str = "static_memory_planning"

    def run(self, graph: Graph, context: CompilerContext) -> PassResult:
        producer_idx: dict[str, int] = {}
        last_use: dict[str, int] = {}
        for idx, node in enumerate(graph.nodes):
            for output in node.outputs:
                producer_idx[output] = idx
                last_use[output] = idx
            for inp in node.inputs:
                last_use[inp] = idx
        final_idx = len(graph.nodes)
        for output in graph.outputs:
            last_use[output] = final_idx

        free: list[tuple[int, int]] = []
        active: list[tuple[int, int, int]] = []
        buffers: dict[str, dict[str, int]] = {}
        high_watermark = 0

        events = sorted(producer_idx.items(), key=lambda item: item[1])
        for value, start in events:
            if value in graph.outputs:
                continue
            tensor_type = graph.values.get(value)
            if tensor_type is None:
                continue
            size = tensor_type.nbytes(context.symbols)
            if size is None or size == 0:
                continue
            still_active = []
            for end, offset, block_size in active:
                if end < start:
                    free.append((offset, block_size))
                else:
                    still_active.append((end, offset, block_size))
            active = still_active

            offset = None
            free.sort(key=lambda item: item[1])
            for idx, (candidate_offset, candidate_size) in enumerate(free):
                if candidate_size >= size:
                    offset = candidate_offset
                    free.pop(idx)
                    break
            if offset is None:
                offset = high_watermark
                high_watermark += int(math.ceil(size / 64.0) * 64)
            buffers[value] = {"offset": int(offset), "size": int(size), "start": int(start), "end": int(last_use.get(value, start))}
            active.append((last_use.get(value, start), offset, size))

        plan = {
            "arena_bytes": int(high_watermark),
            "symbol_sizes": context.symbols,
            "buffers": buffers,
        }
        if graph.attrs.get("memory_plan") != plan:
            graph.attrs["memory_plan"] = plan
            return PassResult(True, [f"planned {high_watermark} byte activation arena"])
        return PassResult(False, [])


def _is_low_vram(profile: dict[str, object] | object) -> bool:
    if not isinstance(profile, dict):
        return False
    gpus = profile.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return False
    memories = []
    for gpu in gpus:
        if isinstance(gpu, dict) and isinstance(gpu.get("memory_total_mb"), int):
            memories.append(gpu["memory_total_mb"])
    return bool(memories and max(memories) < 8192)


def _node_estimated_bytes(graph: Graph, node: Node, context: CompilerContext) -> int:
    total = 0
    for value in node.inputs + node.outputs:
        tensor_type = graph.values.get(value)
        if tensor_type is None:
            continue
        nbytes = tensor_type.nbytes(context.symbols)
        if nbytes:
            total += nbytes
    return int(total)


def _node_estimated_flops(graph: Graph, node: Node, context: CompilerContext) -> int:
    if node.op not in {"matmul", "quantized_matmul", "qkv_projection", "fused_rmsnorm_qkv_rope", "quantized_fused_rmsnorm_qkv_rope"}:
        return 0
    x = graph.values.get(node.inputs[0])
    if x is None or len(x.shape) < 2:
        return 0
    batch_seq = 1
    for dim in x.shape[:-1]:
        if isinstance(dim, int):
            batch_seq *= dim
        else:
            batch_seq *= context.symbols.get(dim, 1)
    in_features = x.shape[-1] if isinstance(x.shape[-1], int) else context.symbols.get(x.shape[-1], 1)
    out_features = 0
    weight_inputs = node.inputs[1:] if node.op == "qkv_projection" else node.inputs[2:] if "qkv_rope" in node.op else node.inputs[1:2]
    for weight_name in weight_inputs:
        weight = graph.values.get(weight_name)
        if weight and len(weight.shape) == 2 and isinstance(weight.shape[0], int):
            out_features += weight.shape[0]
    return int(2 * batch_seq * int(in_features) * out_features)
