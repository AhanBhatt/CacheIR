import json

import numpy as np

from cacheir import Runtime, compile_model
from cacheir.benchmark import run_benchmark
from cacheir.hardware import profile_hardware
from cacheir.importers import create_tiny_model
from cacheir.visualize import export_graph, graph_to_dot, graph_to_html


def test_artifact_bundle_export_and_directory_load(tmp_path):
    model = create_tiny_model(tmp_path / "tiny")
    bundle = tmp_path / "bundle"
    artifact = compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=8, output=bundle)

    assert (bundle / "artifact.json").exists()
    assert (bundle / "graphs" / "decode.cir").exists()
    assert (bundle / "schedules" / "decode.json").exists()
    assert (bundle / "passes" / "decode.prefill_decode_specialization.diff").exists()

    runtime = Runtime(bundle)
    logits = runtime.run([[1, 2]], mode="prefill")
    assert logits.shape[-1] == artifact.config.vocab_size


def test_graph_export_html_and_dot(tmp_path):
    model = create_tiny_model(tmp_path / "tiny")
    artifact = compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=8)

    dot = graph_to_dot(artifact, "decode")
    html = graph_to_html(artifact, "decode")
    assert "paged_attention_decode" in dot
    assert "Execution Schedule" in html

    html_path = export_graph(artifact, tmp_path / "graph.html", mode="decode")
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_benchmark_and_hardware_profile(tmp_path):
    model = create_tiny_model(tmp_path / "tiny", vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1, num_attention_heads=4, num_key_value_heads=2)
    artifact = compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=8)

    result = run_benchmark(artifact, prompt="abc", decode_tokens=2, repeats=1)
    assert result.prompt_tokens == 3
    assert result.decode_tokens == 2
    assert result.prefill_tokens_per_s > 0
    assert result.decode_tokens_per_s > 0

    profile = profile_hardware().to_dict()
    assert profile["cpu_count"] >= 1
    json.dumps(profile)


def test_quantized_runtime_path(tmp_path):
    model = create_tiny_model(tmp_path / "tiny", vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1, num_attention_heads=4, num_key_value_heads=2)
    artifact = compile_model(model, target="cpu", quant="int4_awq", mode=["prefill", "decode"], max_seq=8)
    decode_ops = [node.op for node in artifact.graph("decode").nodes]
    assert "quantized_fused_rmsnorm_qkv_rope" in decode_ops
    assert "quantized_matmul" in decode_ops

    runtime = Runtime(artifact)
    logits = runtime.run([[1, 2, 3]], mode="prefill")
    assert logits.shape == (1, 3, 32)
    assert np.isfinite(logits).all()
