import numpy as np

import cacheir.backends.native as native
from cacheir import CudaRuntime, Runtime, compile_model, create_runtime, cuda_runtime_available
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


def test_native_silu_mul_matches_numpy_when_available():
    if not native.available() or not hasattr(native, "silu_mul"):
        return
    gate = np.linspace(-3.0, 3.0, num=24, dtype=np.float32).reshape(2, 3, 4)
    up = np.linspace(0.5, 2.0, num=24, dtype=np.float32).reshape(2, 3, 4)
    actual = native.silu_mul(gate, up)
    expected = (gate / (1.0 + np.exp(-gate))) * up
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_create_runtime_auto_uses_cpu_for_cpu_artifact(tmp_path):
    model = create_tiny_model(tmp_path / "tiny", vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1, num_attention_heads=4, num_key_value_heads=2)
    artifact = compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=8)
    runtime = create_runtime(artifact, backend="auto")
    assert isinstance(runtime, Runtime)


def test_cuda_runtime_matches_cpu_when_available(tmp_path):
    if not cuda_runtime_available():
        return
    model = create_tiny_model(tmp_path / "tiny", vocab_size=48, hidden_size=32, intermediate_size=64, num_layers=1, num_attention_heads=4, num_key_value_heads=2)
    artifact = compile_model(model, target="cuda", mode=["prefill", "decode"], max_seq=8)
    cpu = Runtime(artifact)
    cuda = CudaRuntime(artifact, dtype="float32", use_triton_elementwise=False)

    cpu_logits = cpu.run([[1, 2, 3]], mode="prefill")
    cuda_logits = cuda.run([[1, 2, 3]], mode="prefill")
    cuda.synchronize()
    np.testing.assert_allclose(cuda_logits.detach().cpu().numpy(), cpu_logits, rtol=1e-4, atol=1e-4)

    cpu_decode = cpu.run([[4]], mode="decode")
    cuda_decode = cuda.run([[4]], mode="decode")
    cuda.synchronize()
    np.testing.assert_allclose(cuda_decode.detach().cpu().numpy(), cpu_decode, rtol=1e-4, atol=1e-4)

    assert len(list(cuda.generate("abc", max_new_tokens=2))) == 2
