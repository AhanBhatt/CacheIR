import numpy as np
import pytest

from cacheir import ContinuousBatchScheduler, CudaRuntime, Runtime, compile_model, cuda_runtime_available
from cacheir.importers import create_tiny_model


def _artifact(tmp_path):
    model = create_tiny_model(
        tmp_path / "tiny",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return compile_model(model, target="cpu", mode=["prefill", "decode"], max_seq=16)


def test_prefix_cache_prefill_matches_full_prefill_suffix(tmp_path):
    artifact = _artifact(tmp_path)
    cached = Runtime(artifact)
    full = Runtime(artifact)
    prompt = [1, 2, 3, 4, 5]
    prefix = prompt[:3]

    cached.run([prefix], mode="prefill")
    cached.remember_prefix(prefix)
    suffix_logits, reused = cached.prefill_tokens(prompt, use_prefix_cache=True, remember_prefix=True)
    full_logits = full.run([prompt], mode="prefill")

    assert reused == tuple(prefix)
    np.testing.assert_allclose(suffix_logits, full_logits[:, len(prefix) :], rtol=1e-4, atol=1e-4)
    assert cached.prefix_cache.stats()["hits"] == 1


def test_continuous_batch_scheduler_runs_requests_and_records_metrics(tmp_path):
    artifact = _artifact(tmp_path)
    scheduler = ContinuousBatchScheduler(artifact, max_batch_size=2, use_prefix_cache=True)
    results = scheduler.generate_batch(["CacheIR shared", "CacheIR shared suffix"], max_new_tokens=3)

    assert len(results) == 2
    assert all(result.finish_reason == "length" for result in results)
    assert all(len(result.generated_ids) == 3 for result in results)
    stats = scheduler.stats()
    assert stats["scheduler"]["completed_requests"] == 2
    assert stats["scheduler"]["generated_tokens"] == 6
    assert stats["scheduler"]["max_observed_batch"] == 2
    assert stats["prefix_cache"]["entries"] >= 1


def test_scheduler_priority_cancellation_and_queue_limit(tmp_path):
    artifact = _artifact(tmp_path)
    scheduler = ContinuousBatchScheduler(artifact, max_batch_size=1, use_prefix_cache=False, max_queue_size=2)
    low = scheduler.submit("low", max_new_tokens=1, priority=0)
    high = scheduler.submit("high", max_new_tokens=1, priority=10)
    assert scheduler.cancel(low)
    replacement = scheduler.submit("replacement", max_new_tokens=1, priority=1)
    with pytest.raises(RuntimeError):
        scheduler.submit("overflow", max_new_tokens=1)

    results = scheduler.run_until_complete()

    assert [result.request_id for result in results] == [high, replacement]
    stats = scheduler.stats()["scheduler"]
    assert stats["cancelled_requests"] == 1
    assert stats["rejected_requests"] == 1
    assert stats["completed_requests"] == 2


def test_server_health_metrics_and_batch_endpoint(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cacheir.runtime.server import create_app

    artifact = _artifact(tmp_path)
    app = create_app(artifact)
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post("/v1/cacheir/batch_completions", json={"prompts": ["a", "ab"], "max_tokens": 2})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["data"]) == 2
    assert payload["scheduler"]["scheduler"]["completed_requests"] == 2

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "cacheir_requests_total" in metrics.text
    assert "cacheir_prefix_cache_entries" in metrics.text
    assert "cacheir_scheduler_batched_decode_rounds" in metrics.text


def test_cuda_scheduler_uses_batched_decode_when_available(tmp_path):
    if not cuda_runtime_available():
        return
    model = create_tiny_model(
        tmp_path / "cuda_tiny",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    artifact = compile_model(model, target="cuda", mode=["prefill", "decode"], max_seq=16)
    prompts = ["abc", "wxyz"]

    batched = ContinuousBatchScheduler(
        CudaRuntime(artifact, dtype="float32", use_triton_elementwise=False),
        max_batch_size=2,
        use_prefix_cache=False,
    )
    sequential = ContinuousBatchScheduler(
        CudaRuntime(artifact, dtype="float32", use_triton_elementwise=False),
        max_batch_size=1,
        use_prefix_cache=False,
    )

    batched_results = batched.generate_batch(prompts, max_new_tokens=3)
    sequential_results = sequential.generate_batch(prompts, max_new_tokens=3)

    assert [result.generated_ids for result in batched_results] == [result.generated_ids for result in sequential_results]
    stats = batched.stats()["scheduler"]
    assert stats["batched_prefill_rounds"] == 1
    assert stats["batched_prefill_requests"] == 2
    assert stats["batched_prefill_tokens"] == 7
    assert stats["batched_prefill_padded_tokens"] == 1
    assert stats["batched_decode_rounds"] == 3
    assert stats["batched_decode_tokens"] == 6
    assert stats["max_observed_batch"] == 2


def test_cuda_runtime_forks_share_page_allocator(tmp_path):
    if not cuda_runtime_available():
        return
    model = create_tiny_model(
        tmp_path / "cuda_allocator_tiny",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    artifact = compile_model(model, target="cuda", mode=["prefill", "decode"], max_seq=16)
    template = CudaRuntime(artifact, dtype="float32", use_triton_elementwise=False)
    first = template.fork()
    second = template.fork()

    first.run([[1, 2]], mode="prefill")
    second.run([[3, 4, 5]], mode="prefill")
    first.synchronize()

    stats = template.kv_allocator.stats()
    assert stats["allocated_pages_total"] >= 2
    assert stats["resident_pages"] >= 2
    first_pages = first.cache_stats()["layers"]["0"]["pages"]
    second_pages = second.cache_stats()["layers"]["0"]["pages"]
    assert first_pages[0]["page_id"] != second_pages[0]["page_id"]
