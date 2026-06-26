from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from cacheir.runtime.cpu import Runtime


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 16
    request_id: str | None = None
    priority: int = 0


@dataclass
class GenerationResult:
    request_id: str
    prompt: str
    text: str
    token_ids: list[int]
    generated_ids: list[int]
    prefill_ms: float
    decode_ms: float
    reused_prefix_tokens: int
    finish_reason: str = "length"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SchedulerMetrics:
    submitted_requests: int = 0
    rejected_requests: int = 0
    cancelled_requests: int = 0
    completed_requests: int = 0
    generated_tokens: int = 0
    prefill_tokens: int = 0
    reused_prefix_tokens: int = 0
    scheduling_rounds: int = 0
    max_observed_batch: int = 0
    batched_prefill_rounds: int = 0
    batched_prefill_requests: int = 0
    batched_prefill_tokens: int = 0
    batched_prefill_padded_tokens: int = 0
    total_prefill_batch_ms: float = 0.0
    batched_decode_rounds: int = 0
    batched_decode_tokens: int = 0
    total_decode_batch_ms: float = 0.0
    total_prefill_ms: float = 0.0
    total_decode_ms: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)

    def to_dict(self) -> dict[str, object]:
        elapsed = max(1.0e-9, time.perf_counter() - self.started_at)
        return {
            **asdict(self),
            "elapsed_s": elapsed,
            "requests_per_s": self.completed_requests / elapsed,
            "generated_tokens_per_s": self.generated_tokens / elapsed,
        }


@dataclass
class _ActiveRequest:
    request: GenerationRequest
    runtime: object
    prompt_ids: list[int]
    generated_ids: list[int]
    next_id: int
    prefill_ms: float
    decode_ms: float
    reused_prefix_tokens: int


class ContinuousBatchScheduler:
    """Small continuous-batching scheduler for CacheIR runtime sessions.

    Each active request owns an independent KV cache, while weights, tokenizer,
    and the prefix cache are shared through ``Runtime.fork``. Decode work is
    advanced in scheduling rounds so the runtime can expose production-style
    queueing, batching, and prefix reuse semantics even on the reference CPU
    backend.
    """

    def __init__(
        self,
        artifact: Runtime | str | Path,
        *,
        max_batch_size: int = 4,
        use_prefix_cache: bool = True,
        max_queue_size: int | None = None,
    ):
        self.template = artifact if _looks_like_runtime(artifact) else Runtime(artifact)
        self.max_batch_size = max(1, int(max_batch_size))
        self.use_prefix_cache = use_prefix_cache
        self.max_queue_size = max_queue_size
        self._pending: list[GenerationRequest] = []
        self._cancelled: set[str] = set()
        self._counter = itertools.count(1)
        self.metrics = SchedulerMetrics()

    def submit(self, prompt: str, *, max_new_tokens: int = 16, request_id: str | None = None, priority: int = 0) -> str:
        self._check_capacity(1)
        return self._enqueue(GenerationRequest(prompt=prompt, max_new_tokens=max_new_tokens, request_id=request_id, priority=int(priority)))

    def cancel(self, request_id: str) -> bool:
        for idx, request in enumerate(self._pending):
            if request.request_id == request_id:
                self._pending.pop(idx)
                self.metrics.cancelled_requests += 1
                return True
        self._cancelled.add(request_id)
        return False

    def submit_many(self, requests: Iterable[GenerationRequest]) -> list[str]:
        request_list = list(requests)
        self._check_capacity(len(request_list))
        request_ids = []
        for request in request_list:
            request_ids.append(self._enqueue(request))
        return request_ids

    def run_until_complete(self) -> list[GenerationResult]:
        active: list[_ActiveRequest] = []
        results: list[GenerationResult] = []
        while self._pending or active:
            capacity = self.max_batch_size - len(active)
            if self._pending and capacity > 0:
                admitted = [self._pop_pending() for _ in range(min(capacity, len(self._pending)))]
                active.extend(self._prefill_admitted(admitted))
            self.metrics.max_observed_batch = max(self.metrics.max_observed_batch, len(active))
            if not active:
                continue
            self.metrics.scheduling_rounds += 1
            ready = [
                state
                for state in active
                if state.request.request_id not in self._cancelled and len(state.generated_ids) < state.request.max_new_tokens
            ]
            for state in active:
                if state.request.request_id in self._cancelled:
                    self._cancelled.remove(state.request.request_id or "")
                    self.metrics.cancelled_requests += 1
                    results.append(self._finish(state, finish_reason="cancelled"))
                elif len(state.generated_ids) >= state.request.max_new_tokens:
                    results.append(self._finish(state))
            if len(ready) > 1 and _batch_decode_available(ready):
                self._decode_batch(ready)
            else:
                for state in ready:
                    self._decode_one(state)
            survivors: list[_ActiveRequest] = []
            for state in ready:
                if state.request.request_id in self._cancelled:
                    self._cancelled.remove(state.request.request_id or "")
                    self.metrics.cancelled_requests += 1
                    results.append(self._finish(state, finish_reason="cancelled"))
                elif len(state.generated_ids) >= state.request.max_new_tokens:
                    results.append(self._finish(state))
                else:
                    survivors.append(state)
            active = survivors
        return results

    def generate_batch(self, prompts: list[str], *, max_new_tokens: int = 16) -> list[GenerationResult]:
        self.submit_many(GenerationRequest(prompt=prompt, max_new_tokens=max_new_tokens) for prompt in prompts)
        return self.run_until_complete()

    def stats(self) -> dict[str, object]:
        return {
            "scheduler": self.metrics.to_dict(),
            "prefix_cache": self.template.prefix_cache.stats(),
            "runtime_cache": _runtime_cache_stats(self.template),
            "max_batch_size": self.max_batch_size,
            "use_prefix_cache": self.use_prefix_cache,
            "max_queue_size": self.max_queue_size,
        }

    def _pop_pending(self) -> GenerationRequest:
        best_idx = max(range(len(self._pending)), key=lambda idx: (self._pending[idx].priority, -idx))
        return self._pending.pop(best_idx)

    def _check_capacity(self, count: int) -> None:
        count = max(0, int(count))
        if self.max_queue_size is not None and len(self._pending) + count > int(self.max_queue_size):
            self.metrics.rejected_requests += count
            raise RuntimeError("CacheIR scheduler queue is full")

    def _enqueue(self, request: GenerationRequest) -> str:
        assigned_id = request.request_id or f"cacheir-{next(self._counter)}"
        self._pending.append(
            GenerationRequest(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                request_id=assigned_id,
                priority=int(request.priority),
            )
        )
        self.metrics.submitted_requests += 1
        return assigned_id

    def _prefill(self, request: GenerationRequest) -> _ActiveRequest:
        runtime = self.template.fork()
        prompt_ids = runtime.tokenizer.encode(request.prompt)
        start = time.perf_counter()
        logits, reused_prefix = runtime.prefill_tokens(
            prompt_ids,
            use_prefix_cache=self.use_prefix_cache,
            remember_prefix=self.use_prefix_cache,
        )
        _synchronize(runtime)
        prefill_ms = (time.perf_counter() - start) * 1000.0
        self.metrics.prefill_tokens += len(prompt_ids)
        self.metrics.reused_prefix_tokens += len(reused_prefix)
        self.metrics.total_prefill_ms += prefill_ms
        next_id = _argmax_token(logits) % runtime.artifact.config.vocab_size
        return _ActiveRequest(
            request=request,
            runtime=runtime,
            prompt_ids=prompt_ids,
            generated_ids=[],
            next_id=next_id,
            prefill_ms=prefill_ms,
            decode_ms=0.0,
            reused_prefix_tokens=len(reused_prefix),
        )

    def _prefill_admitted(self, requests: list[GenerationRequest]) -> list[_ActiveRequest]:
        if len(requests) <= 1 or self.use_prefix_cache:
            return [self._prefill(request) for request in requests]
        runtimes = [self.template.fork() for _ in requests]
        if not _prefill_batch_available(runtimes):
            return [self._prefill_with_runtime(request, runtime) for request, runtime in zip(requests, runtimes)]

        encoded = [runtime.tokenizer.encode(request.prompt) for request, runtime in zip(requests, runtimes)]
        if _variable_prefill_batch_available(runtimes) and len(requests) > 1 and all(encoded):
            batch_prefill = getattr(runtimes[0], "run_prefill_batch")
            start = time.perf_counter()
            logits = batch_prefill(runtimes, encoded)
            _synchronize(runtimes[0])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            next_ids = _argmax_tokens_at_positions(logits, [len(tokens) - 1 for tokens in encoded])
            total_tokens = sum(len(tokens) for tokens in encoded)
            padded_tokens = len(encoded) * max(len(tokens) for tokens in encoded) - total_tokens
            self.metrics.batched_prefill_rounds += 1
            self.metrics.batched_prefill_requests += len(requests)
            self.metrics.batched_prefill_tokens += total_tokens
            self.metrics.batched_prefill_padded_tokens += padded_tokens
            self.metrics.total_prefill_batch_ms += elapsed_ms
            self.metrics.total_prefill_ms += elapsed_ms * len(requests)
            self.metrics.prefill_tokens += total_tokens
            states = []
            for request, runtime, prompt_ids, next_id in zip(requests, runtimes, encoded, next_ids):
                states.append(
                    _ActiveRequest(
                        request=request,
                        runtime=runtime,
                        prompt_ids=prompt_ids,
                        generated_ids=[],
                        next_id=int(next_id) % runtime.artifact.config.vocab_size,
                        prefill_ms=elapsed_ms,
                        decode_ms=0.0,
                        reused_prefix_tokens=0,
                    )
                )
            return states

        groups: dict[int, list[int]] = {}
        for idx, token_ids in enumerate(encoded):
            groups.setdefault(len(token_ids), []).append(idx)

        states: list[_ActiveRequest | None] = [None for _ in requests]
        for indexes in groups.values():
            if len(indexes) <= 1 or not encoded[indexes[0]]:
                for idx in indexes:
                    states[idx] = self._prefill_with_runtime(requests[idx], runtimes[idx], prompt_ids=encoded[idx])
                continue
            group_runtimes = [runtimes[idx] for idx in indexes]
            group_tokens = [encoded[idx] for idx in indexes]
            batch_prefill = getattr(group_runtimes[0], "run_prefill_batch")
            start = time.perf_counter()
            logits = batch_prefill(group_runtimes, group_tokens)
            _synchronize(group_runtimes[0])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            next_ids = _argmax_tokens(logits)
            self.metrics.batched_prefill_rounds += 1
            self.metrics.batched_prefill_requests += len(indexes)
            self.metrics.batched_prefill_tokens += sum(len(tokens) for tokens in group_tokens)
            self.metrics.batched_prefill_padded_tokens += 0
            self.metrics.total_prefill_batch_ms += elapsed_ms
            self.metrics.total_prefill_ms += elapsed_ms * len(indexes)
            self.metrics.prefill_tokens += sum(len(tokens) for tokens in group_tokens)
            for local_idx, request_idx in enumerate(indexes):
                runtime = runtimes[request_idx]
                states[request_idx] = _ActiveRequest(
                    request=requests[request_idx],
                    runtime=runtime,
                    prompt_ids=encoded[request_idx],
                    generated_ids=[],
                    next_id=int(next_ids[local_idx]) % runtime.artifact.config.vocab_size,
                    prefill_ms=elapsed_ms,
                    decode_ms=0.0,
                    reused_prefix_tokens=0,
                )
        return [state for state in states if state is not None]

    def _prefill_with_runtime(
        self,
        request: GenerationRequest,
        runtime: object,
        *,
        prompt_ids: list[int] | None = None,
    ) -> _ActiveRequest:
        prompt_ids = prompt_ids if prompt_ids is not None else runtime.tokenizer.encode(request.prompt)
        start = time.perf_counter()
        logits, reused_prefix = runtime.prefill_tokens(
            prompt_ids,
            use_prefix_cache=self.use_prefix_cache,
            remember_prefix=self.use_prefix_cache,
        )
        _synchronize(runtime)
        prefill_ms = (time.perf_counter() - start) * 1000.0
        self.metrics.prefill_tokens += len(prompt_ids)
        self.metrics.reused_prefix_tokens += len(reused_prefix)
        self.metrics.total_prefill_ms += prefill_ms
        next_id = _argmax_token(logits) % runtime.artifact.config.vocab_size
        return _ActiveRequest(
            request=request,
            runtime=runtime,
            prompt_ids=prompt_ids,
            generated_ids=[],
            next_id=next_id,
            prefill_ms=prefill_ms,
            decode_ms=0.0,
            reused_prefix_tokens=len(reused_prefix),
        )

    def _decode_one(self, state: _ActiveRequest) -> None:
        start = time.perf_counter()
        logits = state.runtime.run([[state.next_id]], mode="decode")
        _synchronize(state.runtime)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        state.decode_ms += elapsed_ms
        state.generated_ids.append(state.next_id)
        self.metrics.generated_tokens += 1
        self.metrics.total_decode_ms += elapsed_ms
        state.next_id = _argmax_token(logits) % state.runtime.artifact.config.vocab_size

    def _decode_batch(self, states: list[_ActiveRequest]) -> None:
        runtime = states[0].runtime
        batch_decode = getattr(runtime, "run_decode_batch", None)
        if not callable(batch_decode):
            for state in states:
                self._decode_one(state)
            return
        sessions = [state.runtime for state in states]
        token_ids = [state.next_id for state in states]
        start = time.perf_counter()
        logits = batch_decode(sessions, token_ids)
        _synchronize(runtime)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        next_ids = _argmax_tokens(logits)
        self.metrics.batched_decode_rounds += 1
        self.metrics.batched_decode_tokens += len(states)
        self.metrics.total_decode_batch_ms += elapsed_ms
        self.metrics.total_decode_ms += elapsed_ms * len(states)
        for state, next_id in zip(states, next_ids):
            state.decode_ms += elapsed_ms
            state.generated_ids.append(state.next_id)
            self.metrics.generated_tokens += 1
            state.next_id = int(next_id) % state.runtime.artifact.config.vocab_size

    def _finish(self, state: _ActiveRequest, *, finish_reason: str = "length") -> GenerationResult:
        self.metrics.completed_requests += 1
        text = state.runtime.tokenizer.decode(state.generated_ids)
        return GenerationResult(
            request_id=state.request.request_id or "",
            prompt=state.request.prompt,
            text=text,
            token_ids=state.prompt_ids,
            generated_ids=state.generated_ids,
            prefill_ms=state.prefill_ms,
            decode_ms=state.decode_ms,
            reused_prefix_tokens=state.reused_prefix_tokens,
            finish_reason=finish_reason,
        )


def _looks_like_runtime(value: object) -> bool:
    return all(hasattr(value, attr) for attr in ("run", "fork", "tokenizer", "artifact"))


def _synchronize(runtime: object) -> None:
    sync = getattr(runtime, "synchronize", None)
    if callable(sync):
        sync()


def _runtime_cache_stats(runtime: object) -> dict[str, object]:
    stats = getattr(runtime, "cache_stats", None)
    if not callable(stats):
        return {}
    try:
        data = stats()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _argmax_token(logits: object) -> int:
    value = logits[0, -1].argmax()  # type: ignore[index, union-attr]
    item = getattr(value, "item", None)
    return int(item() if callable(item) else value)


def _argmax_tokens(logits: object) -> list[int]:
    slice_ = logits[:, -1]  # type: ignore[index]
    try:
        argmax = slice_.argmax(dim=-1)  # type: ignore[call-arg, union-attr]
    except TypeError:
        argmax = slice_.argmax(axis=-1)  # type: ignore[call-arg, union-attr]
    tolist = getattr(argmax, "tolist", None)
    values = tolist() if callable(tolist) else list(argmax)
    return [int(value) for value in values]


def _argmax_tokens_at_positions(logits: object, positions: list[int]) -> list[int]:
    values = []
    for row, position in enumerate(positions):
        slice_ = logits[row, int(position)]  # type: ignore[index]
        try:
            argmax = slice_.argmax(dim=-1)  # type: ignore[call-arg, union-attr]
        except TypeError:
            argmax = slice_.argmax(axis=-1)  # type: ignore[call-arg, union-attr]
        item = getattr(argmax, "item", None)
        values.append(int(item() if callable(item) else argmax))
    return values


def _batch_decode_available(states: list[_ActiveRequest]) -> bool:
    if not states:
        return False
    runtime = states[0].runtime
    if not callable(getattr(runtime, "run_decode_batch", None)):
        return False
    return all(callable(getattr(state.runtime, "run_decode_batch", None)) for state in states)


def _prefill_batch_available(runtimes: list[object]) -> bool:
    if not runtimes:
        return False
    return all(callable(getattr(runtime, "run_prefill_batch", None)) for runtime in runtimes)


def _variable_prefill_batch_available(runtimes: list[object]) -> bool:
    if not _prefill_batch_available(runtimes):
        return False
    return all(bool(getattr(runtime, "supports_variable_prefill_batch", False)) for runtime in runtimes)
