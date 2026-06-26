import struct
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

import cacheir.backends.native as native
import cacheir.backends.adapters as adapter_mod
import cacheir.backends.upstream as upstream_mod
from cacheir.backends.vllm_compat import install_no_uva_fallback
from cacheir.backends import native_available, probe_adapters, select_attention_backend, simd_backend
from cacheir.backends.cuda_graphs import plan_decode_cuda_graph
from cacheir.backends.upstream import (
    ExternalSystemStatus,
    compile_stablehlo_with_iree,
    probe_external_systems,
    run_iree_stablehlo_benchmark,
    run_tvm_vector_add_benchmark,
)
from cacheir.backends.triton_kernels import describe_kernels, triton_available
from cacheir.hardware import calibrate_bandwidth
from cacheir.importers.gguf import GGUFReader, import_gguf_metadata
from cacheir.importers.stablehlo import import_stablehlo_text
from cacheir.ir import Graph, TensorType
from cacheir.mlir import cpp_dialect_registration, emit_cacheir_dialect, parse_cacheir_dialect, verify_cacheir_dialect
from cacheir.runtime.kv_cache import PagedKVCache, PrefixCache, SpilloverCostModel, SpilloverPolicy


def _gguf_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def test_gguf_f32_tensor_reader(tmp_path):
    path = tmp_path / "tiny.gguf"
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", 1)
    header += struct.pack("<Q", 0)
    header += _gguf_string("token_embd.weight")
    header += struct.pack("<I", 2)
    header += struct.pack("<Q", 3)
    header += struct.pack("<Q", 2)
    header += struct.pack("<I", 0)
    header += struct.pack("<Q", 0)
    while len(header) % 32:
        header += b"\x00"
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    path.write_bytes(bytes(header) + data.tobytes())

    metadata = import_gguf_metadata(path)
    assert metadata["data_offset"] == len(header)
    tensor = GGUFReader(path).read_tensor("token_embd.weight")
    assert tensor.shape == (2, 3)
    np.testing.assert_allclose(tensor, data)


def test_gguf_quantized_tensor_readers(tmp_path):
    path = tmp_path / "quant.gguf"
    tensor_specs = [
        ("q8.weight", 8, 34),
        ("q4.weight", 2, 18),
        ("q4_1.weight", 3, 20),
        ("q5_0.weight", 6, 22),
        ("q5_1.weight", 7, 24),
        ("q8_1.weight", 9, 40),
    ]
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensor_specs))
    header += struct.pack("<Q", 0)
    cursor = 0
    for name, ggml_type, byte_count in tensor_specs:
        header += _gguf_string(name)
        header += struct.pack("<I", 1)
        header += struct.pack("<Q", 32)
        header += struct.pack("<I", ggml_type)
        header += struct.pack("<Q", cursor)
        cursor += byte_count
    while len(header) % 32:
        header += b"\x00"

    q8_values = np.arange(-16, 16, dtype=np.int8)
    q8_block = np.float16(0.5).tobytes() + q8_values.tobytes()

    low = np.arange(16, dtype=np.uint8) & 0x0F
    high = (15 - np.arange(16, dtype=np.uint8)) & 0x0F
    packed = low | (high << 4)
    q4_block = np.float16(0.25).tobytes() + packed.tobytes()

    q4_1_block = np.float16(0.125).tobytes() + np.float16(-1.5).tobytes() + packed.tobytes()

    q5_unsigned = ((np.arange(32, dtype=np.uint8) * 3) % 32).astype(np.uint8)
    q5_low = q5_unsigned[:16] & 0x0F
    q5_high = q5_unsigned[16:] & 0x0F
    q5_packed = q5_low | (q5_high << 4)
    q5_mask = 0
    for idx in range(16):
        q5_mask |= int((q5_unsigned[idx] >> 4) & 1) << idx
        q5_mask |= int((q5_unsigned[idx + 16] >> 4) & 1) << (idx + 16)
    q5_0_block = np.float16(0.375).tobytes() + struct.pack("<I", q5_mask) + q5_packed.tobytes()
    q5_1_block = (
        np.float16(0.0625).tobytes()
        + np.float16(2.0).tobytes()
        + struct.pack("<I", q5_mask)
        + q5_packed.tobytes()
    )

    q8_1_values = np.arange(31, -1, -1, dtype=np.int8) - 16
    q8_1_block = struct.pack("<f", 0.25) + struct.pack("<f", 0.0) + q8_1_values.tobytes()

    path.write_bytes(bytes(header) + q8_block + q4_block + q4_1_block + q5_0_block + q5_1_block + q8_1_block)

    reader = GGUFReader(path)
    np.testing.assert_allclose(reader.read_tensor("q8.weight"), q8_values.astype(np.float32) * 0.5)
    q4_expected = np.empty(32, dtype=np.float32)
    q4_expected[:16] = (low.astype(np.int16) - 8).astype(np.float32) * 0.25
    q4_expected[16:] = (high.astype(np.int16) - 8).astype(np.float32) * 0.25
    np.testing.assert_allclose(reader.read_tensor("q4.weight"), q4_expected)
    q4_1_expected = np.empty(32, dtype=np.float32)
    q4_1_expected[:16] = low.astype(np.float32) * 0.125 - 1.5
    q4_1_expected[16:] = high.astype(np.float32) * 0.125 - 1.5
    np.testing.assert_allclose(reader.read_tensor("q4_1.weight"), q4_1_expected)
    np.testing.assert_allclose(reader.read_tensor("q5_0.weight"), (q5_unsigned.astype(np.float32) - 16.0) * 0.375)
    np.testing.assert_allclose(reader.read_tensor("q5_1.weight"), q5_unsigned.astype(np.float32) * 0.0625 + 2.0)
    np.testing.assert_allclose(reader.read_tensor("q8_1.weight"), q8_1_values.astype(np.float32) * 0.25)


def test_gguf_reference_k_quant_tensor_readers(tmp_path):
    try:
        import gguf
    except ImportError:
        return

    path = tmp_path / "kquant.gguf"
    qtypes = [
        ("q4k.weight", gguf.GGMLQuantizationType.Q4_K),
        ("iq4nl.weight", gguf.GGMLQuantizationType.IQ4_NL),
    ]
    blocks = []
    cursor = 0
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(qtypes))
    header += struct.pack("<Q", 0)
    for name, qtype in qtypes:
        block_size, block_bytes = gguf.GGML_QUANT_SIZES[qtype]
        raw = np.zeros(block_bytes, dtype=np.uint8)
        raw[:] = np.arange(block_bytes, dtype=np.uint8) % 7
        expected = gguf.dequantize(raw.reshape(1, block_bytes), qtype).reshape(block_size)
        blocks.append((raw.tobytes(), expected))
        header += _gguf_string(name)
        header += struct.pack("<I", 1)
        header += struct.pack("<Q", block_size)
        header += struct.pack("<I", int(qtype))
        header += struct.pack("<Q", cursor)
        cursor += block_bytes
    while len(header) % 32:
        header += b"\x00"
    path.write_bytes(bytes(header) + b"".join(block for block, _ in blocks))

    reader = GGUFReader(path)
    for (name, _), (_, expected) in zip(qtypes, blocks):
        np.testing.assert_allclose(reader.read_tensor(name), expected, rtol=1e-6, atol=1e-6)


def test_prefix_cache_and_spillover_policy():
    cache = PagedKVCache(page_size=2, spillover_policy=SpilloverPolicy(max_resident_pages=1))
    keys = np.zeros((1, 3, 1, 2), dtype=np.float32)
    values = np.ones((1, 3, 1, 2), dtype=np.float32)
    cache.append(0, keys, values)
    stats = cache.stats()
    assert stats["layers"]["0"]["tokens"] == 3
    assert stats["spilled_pages"]

    prefix = PrefixCache(capacity=2)
    snapshot = cache.snapshot_prefix(2)
    prefix.put([1, 2], snapshot)
    key, restored = prefix.longest_prefix([1, 2, 3])
    assert key == (1, 2)
    assert restored is not None
    assert restored[0][0].shape[1] == 2


def test_calibrated_spillover_cost_model():
    model = SpilloverCostModel(page_bytes=1024, gpu_free_memory_mb=512.001, safety_margin_mb=512, pcie_bandwidth_gbps=16)
    assert model.resident_page_budget(fallback=128) == 1
    assert model.transfer_ms(1024, target="cpu") > 0

    cache = PagedKVCache(page_size=2, spillover_policy=SpilloverPolicy(max_resident_pages=4, cost_model=model))
    keys = np.zeros((1, 9, 1, 8), dtype=np.float32)
    values = np.ones((1, 9, 1, 8), dtype=np.float32)
    cache.append(0, keys, values)
    stats = cache.stats()
    assert stats["page_bytes"] == 128
    assert stats["spillover_policy"]["cost_model"]["gpu_free_memory_mb"] == 512.001
    assert len(stats["spilled_pages"]) == 4
    assert all(marker["target"] == "cpu" for marker in stats["spilled_pages"])


def test_stablehlo_import_and_mlir_emit(tmp_path):
    path = tmp_path / "toy.stablehlo"
    path.write_text(
        """
func.func @main(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  %c = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
  %1 = stablehlo.add(%arg0, %0) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %2 = stablehlo.reshape %1 : (tensor<1x4xf32>) -> tensor<4xf32>
  %3 = stablehlo.reshape %2 : (tensor<4xf32>) -> tensor<1x4xf32>
  %4 = stablehlo.multiply(%3, %arg0) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %4 : tensor<1x4xf32>
}
""",
        encoding="utf-8",
    )
    config, graph = import_stablehlo_text(path)
    assert config.hidden_size == 4
    assert graph.inputs["arg0"].shape == (1, 4)
    assert "c" in graph.constants
    assert [node.op for node in graph.nodes] == ["broadcast", "add", "reshape", "reshape", "elementwise_mul"]
    mlir = emit_cacheir_dialect(graph)
    assert "cacheir.graph" in mlir
    assert "cacheir.add" in mlir
    roundtrip = parse_cacheir_dialect(mlir)
    assert [node.op for node in roundtrip.nodes] == [node.op for node in graph.nodes]
    assert roundtrip.outputs == graph.outputs
    assert verify_cacheir_dialect(mlir) == []


def test_stablehlo_region_reduce_and_attrs(tmp_path):
    path = tmp_path / "reduce.stablehlo"
    path.write_text(
        """
func.func @reduce_main(%arg0: tensor<2x4xf32>, %init: tensor<f32>) -> tensor<2xf32> {
  %0 = "stablehlo.reduce"(%arg0, %init) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add(%lhs, %rhs) : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {dimensions = [1]} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
""",
        encoding="utf-8",
    )
    _, graph = import_stablehlo_text(path)
    assert [node.op for node in graph.nodes] == ["reduce"]
    assert graph.nodes[0].attrs["dimensions"] == [1]
    assert graph.nodes[0].attrs["reducer"] == "add"
    assert graph.values["0"].shape == (2,)


def test_optional_backend_metadata_imports():
    assert isinstance(native_available(), bool)
    assert isinstance(simd_backend(), str)
    assert isinstance(triton_available(), bool)
    assert "triton.rms_norm" in describe_kernels()
    assert "Triton fused" in describe_kernels()["triton.fused_rmsnorm_qkv_rope"]
    assert "triton.matmul_f16" in describe_kernels()
    assert "triton.paged_attention_decode_batch" in describe_kernels()
    adapters = probe_adapters()
    assert {"cutlass", "flash_attention", "flashinfer"} <= set(adapters)
    assert select_attention_backend(mode="decode", prefer_external=False) == "cacheir_triton"


def test_direct_adapter_wrappers_dispatch_with_fake_modules(monkeypatch):
    class FakeFlashAttention:
        @staticmethod
        def flash_attn_func(q, k, v, dropout_p, softmax_scale, causal):
            return {"q": q, "causal": causal, "dropout": dropout_p, "scale": softmax_scale}

    class FakeFlashInfer:
        @staticmethod
        def single_decode_with_kv_cache(q, k_cache, v_cache, *args, **kwargs):
            return {"q": q, "args": args, "kwargs": kwargs}

        class BatchDecodeWithPagedKVCacheWrapper:
            def __init__(self, workspace, *args, **kwargs):
                self.workspace = workspace
                self.plan_args = None

            def plan(self, **kwargs):
                self.plan_args = kwargs

            def run(self, q, k_cache, v_cache):
                return {"q": q, "workspace": self.workspace, "planned": self.plan_args}

    def fake_require(name):
        return FakeFlashAttention if name == "flash_attention" else FakeFlashInfer

    monkeypatch.setattr(adapter_mod, "require_adapter", fake_require)
    assert adapter_mod.flash_attention_prefill("q", "k", "v", causal=False)["causal"] is False
    assert adapter_mod.flashinfer_paged_decode("q", "k", "v", "page_table")["args"] == ("page_table",)
    batch = adapter_mod.flashinfer_batch_paged_decode("q", "k", "v", "workspace", plan_kwargs={"heads": 4})
    assert batch["workspace"] == "workspace"
    assert batch["planned"] == {"heads": 4}


def test_real_adapter_smoke_results_are_structured():
    smokes = adapter_mod.run_adapter_smokes()
    assert {"flash_attention", "flashinfer"} <= set(smokes)
    for result in smokes.values():
        payload = result.to_dict()
        assert payload["name"] in {"flash_attention", "flashinfer"}
        assert isinstance(payload["available"], bool)
        assert isinstance(payload["executed"], bool)
        if payload["executed"]:
            assert payload["output_shape"]
            assert payload["dtype"]
        else:
            assert payload["reason"]


def test_cuda_graph_capture_plan_and_bandwidth_calibration():
    graph = Graph(name="decode_test", mode="decode", target="cuda")
    graph.add_input("input_ids", TensorType((1, 1), "int64"))
    graph.add_node("paged_attention_decode", ["input_ids"], ["out"], output_types=[TensorType((1, 1, 8), "float32")])
    graph.outputs = ["out"]
    plan = plan_decode_cuda_graph(graph, batch_size=1, decode_tokens_per_replay=1)
    assert plan.mode == "decode"
    assert "paged_attention_decode" in plan.captured_ops
    assert "input_ids" in plan.static_inputs

    calibration = calibrate_bandwidth(sample_mb=1, repeats=1, include_cuda=False)
    assert calibration.cpu_copy_gbps > 0
    model = SpilloverCostModel.from_bandwidth_calibration(calibration, page_bytes=1024)
    assert model.cpu_read_bandwidth_gbps == calibration.cpu_copy_gbps


def test_upstream_iree_and_tvm_smoke_paths(tmp_path):
    systems = probe_external_systems()
    assert {"vllm", "llama.cpp", "iree", "tvm"} <= set(systems)

    stablehlo = """
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
    if systems["iree"].available:
        result = compile_stablehlo_with_iree(stablehlo, output_path=tmp_path / "add.vmfb")
        assert result.byte_count > 0
        assert (tmp_path / "add.vmfb").exists()
        bench = run_iree_stablehlo_benchmark(tmp_path)
        assert bench["available"] is True
        assert bench["returncode"] == 0

    if systems["tvm"].available:
        tvm_result = run_tvm_vector_add_benchmark(n=32, number=1, repeat=1)
        assert tvm_result["available"] is True
        assert tvm_result["checksum"] == 64.0


def test_upstream_model_benchmark_helpers_build_real_commands(monkeypatch, tmp_path):
    statuses = {
        "vllm": ExternalSystemStatus(name="vllm", available=True, module="vllm", command="vllm"),
        "llama.cpp": ExternalSystemStatus(name="llama.cpp", available=True, command="llama-bench"),
        "iree": ExternalSystemStatus(name="iree", available=False),
        "tvm": ExternalSystemStatus(name="tvm", available=False),
    }
    calls = []

    def fake_which(name):
        return {"vllm": "vllm", "llama-bench": "llama-bench"}.get(name)

    def fake_run(cmd, capture_output, text, check, timeout, env=None):
        calls.append((cmd, env))
        if "--output-json" in cmd:
            path = tmp_path / "vllm_latency.json"
            path.write_text('{"median_latency": 0.01}', encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout='[{"model_filename":"tiny.gguf","avg_ts":123.0}]', stderr="")

    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(upstream_mod, "probe_external_systems", lambda: statuses)
    monkeypatch.setattr(upstream_mod.shutil, "which", fake_which)
    monkeypatch.setattr(upstream_mod.subprocess, "run", fake_run)

    vllm = upstream_mod.run_vllm_latency_benchmark("hf-internal-testing/tiny-random-LlamaForCausalLM", tmp_path)
    llama = upstream_mod.run_llama_cpp_benchmark(model)
    assert vllm["parsed"] == {"median_latency": 0.01}
    assert llama["parsed"] == [{"model_filename": "tiny.gguf", "avg_ts": 123.0}]
    assert calls[0][0][:3] == ["vllm", "bench", "latency"]
    assert "hf-internal-testing/tiny-random-LlamaForCausalLM" in calls[0][0]
    assert calls[0][1]["CACHEIR_VLLM_NO_UVA_FALLBACK"] == "1"
    assert str(tmp_path / "cacheir_vllm_compat") in calls[0][1]["PYTHONPATH"]
    assert str(upstream_mod.Path(upstream_mod.__file__).resolve().parents[2]) in calls[0][1]["PYTHONPATH"]
    assert (tmp_path / "cacheir_vllm_compat" / "sitecustomize.py").exists()
    assert calls[1][0][:3] == ["llama-bench", "-m", str(model)]


def test_vllm_no_uva_fallback_patches_buffer_utils(monkeypatch):
    buffer_utils = ModuleType("vllm.v1.worker.gpu.buffer_utils")
    buffer_utils.is_uva_available = lambda: False
    buffer_utils._DEFAULT_MAX_CONCURRENCY = 2
    buffer_utils.UvaBuffer = object
    buffer_utils.UvaBufferPool = object
    buffer_utils.UvaBackedTensor = object
    parents = [
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
    ]
    for name in parents:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.buffer_utils", buffer_utils)

    result = install_no_uva_fallback(force=True)
    assert result["installed"] is True
    assert buffer_utils._CACHEIR_NO_UVA_FALLBACK is True
    assert buffer_utils.UvaBuffer.__name__ == "NoUvaBuffer"
    assert buffer_utils.UvaBufferPool.__name__ == "NoUvaBufferPool"


def test_mlir_cpp_dialect_registration_manifest():
    registration = cpp_dialect_registration()
    payload = registration.to_dict()
    assert payload["dialect_namespace"] == "cacheir"
    assert payload["cmake_option"] == "CACHEIR_BUILD_MLIR"
    assert payload["registration_function"] == "cacheir::mlir::registerCacheIRDialect"
    header = registration.header
    source = registration.source
    assert "cpp/mlir/include" in header.replace("\\", "/")
    assert "cpp/mlir/src" in source.replace("\\", "/")
    with open(header, encoding="utf-8") as handle:
        assert "class CacheIRDialect" in handle.read()
    with open(source, encoding="utf-8") as handle:
        content = handle.read()
    assert "registry.insert<CacheIRDialect>()" in content
    assert "allowUnknownOperations()" in content


def test_native_extension_matches_numpy_when_built():
    if not native.available():
        return
    x = (np.arange(24, dtype=np.float32).reshape(1, 3, 8) / 10.0).astype(np.float32)
    norm_weight = np.ones(8, dtype=np.float32)
    y = native.rms_norm(x, norm_weight, 1e-6)
    expected_y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(y, expected_y, rtol=1e-5, atol=1e-6)

    weight = (np.arange(32, dtype=np.float32).reshape(4, 8) / 100.0).astype(np.float32)
    z = native.matmul_out_in(x, weight)
    expected_z = np.einsum("...i,oi->...o", x, weight)
    np.testing.assert_allclose(z, expected_z, rtol=1e-5, atol=1e-6)


def test_triton_gpu_kernels_match_torch_when_available():
    if not triton_available():
        return
    try:
        import torch
        import cacheir.backends.triton_kernels as tk
    except Exception:
        return
    if not torch.cuda.is_available():
        return

    rows, hidden = 4, 8
    x = torch.arange(rows * hidden, device="cuda", dtype=torch.float32).reshape(rows, hidden) / 10
    weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    tk.rms_norm_kernel[(rows,)](x, weight, out, hidden=hidden, eps=1e-6, BLOCK=8)
    torch.cuda.synchronize()
    expected = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    assert float((out - expected).abs().max().item()) < 1e-5

    gate = torch.randn(128, device="cuda", dtype=torch.float32)
    up = torch.randn(128, device="cuda", dtype=torch.float32)
    swiglu = torch.empty_like(gate)
    tk.silu_mul_kernel[(1,)](gate, up, swiglu, total=gate.numel(), BLOCK=128)
    torch.cuda.synchronize()
    expected_swiglu = torch.nn.functional.silu(gate) * up
    assert float((swiglu - expected_swiglu).abs().max().item()) < 1e-5

    m, n, k_dim = 16, 16, 16
    a = torch.randn(m, k_dim, device="cuda", dtype=torch.float16)
    b = torch.randn(k_dim, n, device="cuda", dtype=torch.float16)
    matmul = torch.empty((m, n), device="cuda", dtype=torch.float32)
    tk.matmul_f16_kernel[(1, 1)](
        a,
        b,
        matmul,
        m=m,
        n=n,
        k=k_dim,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_om=matmul.stride(0),
        stride_on=matmul.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
    )
    torch.cuda.synchronize()
    expected_matmul = torch.matmul(a.float(), b.float())
    assert float((matmul - expected_matmul).abs().max().item()) < 1e-2

    seq_len, head_dim = 5, 8
    q = torch.randn(1, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32)
    attn = torch.empty_like(q)
    tk.paged_attention_decode_kernel[(1,)](q, k, v, attn, seq_len=seq_len, head_dim=head_dim, BLOCK=8)
    torch.cuda.synchronize()
    scores = (q[0][None, :] * k).sum(dim=-1) / (head_dim ** 0.5)
    expected_attn = (torch.softmax(scores, dim=-1)[:, None] * v).sum(dim=0)
    assert float((attn[0] - expected_attn).abs().max().item()) < 1e-5

    batch, num_heads, num_kv_heads, page_size, max_pages, head_dim = 2, 4, 2, 2, 3, 8
    q_batch = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float32)
    k_pages = torch.randn(5, num_kv_heads, page_size, head_dim, device="cuda", dtype=torch.float32)
    v_pages = torch.randn(5, num_kv_heads, page_size, head_dim, device="cuda", dtype=torch.float32)
    page_table = torch.tensor([[0, 1, 2], [3, 4, 0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([5, 3], device="cuda", dtype=torch.int32)
    out_batch = torch.empty_like(q_batch)
    tk.paged_attention_decode_batch_kernel[(num_heads, batch)](
        q_batch,
        k_pages,
        v_pages,
        page_table,
        seq_lens,
        out_batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_pages_per_seq=max_pages,
        page_size=page_size,
        head_dim=head_dim,
        BLOCK=8,
    )
    torch.cuda.synchronize()
    expected_batch = torch.empty_like(out_batch)
    repeat = num_heads // num_kv_heads
    for b in range(batch):
        seq = int(seq_lens[b].item())
        for h in range(num_heads):
            kv_head = h // repeat
            ks = []
            vs = []
            for pos in range(seq):
                page = int(page_table[b, pos // page_size].item())
                slot = pos % page_size
                ks.append(k_pages[page, kv_head, slot])
                vs.append(v_pages[page, kv_head, slot])
            k_ref = torch.stack(ks)
            v_ref = torch.stack(vs)
            scores = (q_batch[b, h][None, :] * k_ref).sum(dim=-1) / (head_dim ** 0.5)
            expected_batch[b, h] = (torch.softmax(scores, dim=-1)[:, None] * v_ref).sum(dim=0)
    assert float((out_batch - expected_batch).abs().max().item()) < 1e-4
