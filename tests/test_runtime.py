import numpy as np

from cacheir import Runtime, compile_model
from cacheir.importers import create_tiny_model


def test_runtime_prefill_and_decode(tmp_path):
    model = create_tiny_model(tmp_path / "tiny", vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1, num_attention_heads=4, num_key_value_heads=2)
    artifact = compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=8)
    runtime = Runtime(artifact)

    logits = runtime.run([[1, 2, 3]], mode="prefill")
    assert logits.shape == (1, 3, 32)
    assert np.isfinite(logits).all()
    assert 0 in runtime.kv_cache
    assert runtime.kv_cache[0][0].shape[1] == 3

    decode_logits = runtime.run([[4]], mode="decode")
    assert decode_logits.shape == (1, 1, 32)
    assert runtime.kv_cache[0][0].shape[1] == 4


def test_generate_streams_tokens(tmp_path):
    model = create_tiny_model(tmp_path / "tiny", vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1, num_attention_heads=4, num_key_value_heads=2)
    artifact = compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=8)
    runtime = Runtime(artifact)

    tokens = list(runtime.generate("abc", max_new_tokens=3))
    assert len(tokens) == 3
    assert all(isinstance(token, str) for token in tokens)
