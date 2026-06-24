from cacheir import compile_model
from cacheir.importers import create_tiny_model
from cacheir.runtime.artifact import CompileArtifact


def test_compile_tiny_specializes_decode(tmp_path):
    model = create_tiny_model(tmp_path / "tiny")
    artifact = compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=16)

    decode = artifact.graph("decode")
    ops = [node.op for node in decode.nodes]
    assert "fused_rmsnorm_qkv_rope" in ops
    assert "fused_swiglu" in ops
    assert "paged_attention_decode" in ops
    assert "grouped_query_attention" not in ops
    assert decode.attrs["memory_plan"]["arena_bytes"] > 0
    assert all("kernel" in node.attrs for node in decode.nodes)


def test_artifact_roundtrip(tmp_path):
    model = create_tiny_model(tmp_path / "tiny")
    artifact = compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=8)
    path = tmp_path / "artifact.json"
    artifact.save(path)
    loaded = CompileArtifact.load(path)
    assert loaded.config.hidden_size == artifact.config.hidden_size
    assert loaded.graph("prefill").to_text() == artifact.graph("prefill").to_text()
