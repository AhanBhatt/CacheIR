from __future__ import annotations

from typing import Any


def _try_import_triton() -> tuple[Any | None, Any | None]:
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None, None
    return triton, tl


triton, tl = _try_import_triton()


def triton_available() -> bool:
    return triton is not None and tl is not None


def require_triton() -> None:
    if not triton_available():
        raise RuntimeError("Triton kernels require the optional 'triton' package and a CUDA-capable environment")


def describe_kernels() -> dict[str, str]:
    return {
        "triton.rms_norm": "Triton RMSNorm reference kernel",
        "triton.silu_mul": "Triton SwiGLU activation multiply kernel",
        "triton.matmul_f16": "Triton FP16 tiled matmul kernel using tl.dot/Tensor Core lowering when available",
        "triton.fused_rmsnorm_qkv_rope": "Triton fused RMSNorm/QKV/RoPE projection kernel",
        "triton.paged_attention_decode": "Triton single-query paged decode attention kernel",
        "triton.paged_attention_decode_batch": "Triton multi-batch page-table decode attention kernel with GQA mapping",
    }


if triton_available():

    @triton.jit
    def rms_norm_kernel(x, weight, out, hidden: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < hidden
        values = tl.load(x + row * hidden + offsets, mask=mask, other=0.0)
        w = tl.load(weight + offsets, mask=mask, other=0.0)
        variance = tl.sum(values * values, axis=0) / hidden
        normalized = values * tl.rsqrt(variance + eps) * w
        tl.store(out + row * hidden + offsets, normalized, mask=mask)

    @triton.jit
    def silu_mul_kernel(gate, up, out, total: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        g = tl.load(gate + offsets, mask=mask, other=0.0)
        u = tl.load(up + offsets, mask=mask, other=0.0)
        silu = g / (1.0 + tl.exp(-g))
        tl.store(out + offsets, silu * u, mask=mask)

    @triton.jit
    def matmul_f16_kernel(
        a,
        b,
        out,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start_k in range(0, k, BLOCK_K):
            a_ptrs = a + offs_m[:, None] * stride_am + (start_k + offs_k[None, :]) * stride_ak
            b_ptrs = b + (start_k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            a_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & ((start_k + offs_k[None, :]) < k), other=0.0)
            b_tile = tl.load(b_ptrs, mask=((start_k + offs_k[:, None]) < k) & (offs_n[None, :] < n), other=0.0)
            acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
        out_ptrs = out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def fused_rmsnorm_qkv_rope_kernel(
        x,
        norm_weight,
        q_weight,
        k_weight,
        v_weight,
        q_out,
        k_out,
        v_out,
        hidden: tl.constexpr,
        q_out_dim: tl.constexpr,
        kv_out_dim: tl.constexpr,
        head_dim: tl.constexpr,
        kind: tl.constexpr,
        position_offset: tl.constexpr,
        eps: tl.constexpr,
        rope_theta: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        col = tl.program_id(0)
        row = tl.program_id(1)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < hidden
        values = tl.load(x + row * hidden + offsets, mask=mask, other=0.0)
        norm = tl.load(norm_weight + offsets, mask=mask, other=0.0)
        variance = tl.sum(values * values, axis=0) / hidden
        normalized = values * tl.rsqrt(variance + eps) * norm

        if kind == 2:
            if col < kv_out_dim:
                weights = tl.load(v_weight + col * hidden + offsets, mask=mask, other=0.0)
                value = tl.sum(normalized * weights, axis=0)
                tl.store(v_out + row * kv_out_dim + col, value)
        else:
            out_dim = q_out_dim if kind == 0 else kv_out_dim
            if col < out_dim and (col % head_dim) % 2 == 0 and col + 1 < out_dim:
                weight_base = q_weight if kind == 0 else k_weight
                w0 = tl.load(weight_base + col * hidden + offsets, mask=mask, other=0.0)
                w1 = tl.load(weight_base + (col + 1) * hidden + offsets, mask=mask, other=0.0)
                raw0 = tl.sum(normalized * w0, axis=0)
                raw1 = tl.sum(normalized * w1, axis=0)
                pair = (col % head_dim) // 2
                freq = tl.exp((-2.0 * pair / head_dim) * tl.log(tl.full((), rope_theta, tl.float32)))
                angle = (position_offset + row) * freq
                c = tl.cos(angle)
                s = tl.sin(angle)
                out = q_out if kind == 0 else k_out
                tl.store(out + row * out_dim + col, raw0 * c - raw1 * s)
                tl.store(out + row * out_dim + col + 1, raw0 * s + raw1 * c)

    @triton.jit
    def paged_attention_decode_kernel(
        q,
        k_cache,
        v_cache,
        out,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        head = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        dim_mask = offsets < head_dim
        qv = tl.load(q + head * head_dim + offsets, mask=dim_mask, other=0.0)
        max_score = -3.4028234663852886e38
        denom = 0.0
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for pos in range(0, seq_len):
            kv = tl.load(k_cache + (pos * head_dim) + offsets, mask=dim_mask, other=0.0)
            score = tl.sum(qv * kv, axis=0) * tl.rsqrt(tl.full((), head_dim, tl.float32))
            new_max = tl.maximum(max_score, score)
            old_scale = tl.exp(max_score - new_max)
            new_scale = tl.exp(score - new_max)
            vv = tl.load(v_cache + (pos * head_dim) + offsets, mask=dim_mask, other=0.0)
            acc = acc * old_scale + vv * new_scale
            denom = denom * old_scale + new_scale
            max_score = new_max
        tl.store(out + head * head_dim + offsets, acc / denom, mask=dim_mask)

    @triton.jit
    def paged_attention_decode_batch_kernel(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        out,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        max_pages_per_seq: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        head = tl.program_id(0)
        batch = tl.program_id(1)
        offsets = tl.arange(0, BLOCK)
        dim_mask = offsets < head_dim
        group_size = num_heads // num_kv_heads
        kv_head = head // group_size
        qv = tl.load(q + (batch * num_heads + head) * head_dim + offsets, mask=dim_mask, other=0.0)
        seq_len = tl.load(seq_lens + batch)
        scale = tl.rsqrt(tl.full((), head_dim, tl.float32))
        max_score = -3.4028234663852886e38
        denom = 0.0
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for page_idx in range(0, max_pages_per_seq):
            page_id = tl.load(page_table + batch * max_pages_per_seq + page_idx)
            for slot in range(0, page_size):
                pos = page_idx * page_size + slot
                valid = pos < seq_len
                kv_base = ((page_id * num_kv_heads + kv_head) * page_size + slot) * head_dim
                kv = tl.load(k_cache + kv_base + offsets, mask=dim_mask & valid, other=0.0)
                score = tl.sum(qv * kv, axis=0) * scale
                score = tl.where(valid, score, -3.4028234663852886e38)
                new_max = tl.maximum(max_score, score)
                old_scale = tl.exp(max_score - new_max)
                new_scale = tl.exp(score - new_max)
                vv = tl.load(v_cache + kv_base + offsets, mask=dim_mask & valid, other=0.0)
                acc = acc * old_scale + vv * new_scale
                denom = denom * old_scale + new_scale
                max_score = new_max
        tl.store(out + (batch * num_heads + head) * head_dim + offsets, acc / denom, mask=dim_mask)

else:
    rms_norm_kernel = None
    silu_mul_kernel = None
    matmul_f16_kernel = None
    fused_rmsnorm_qkv_rope_kernel = None
    paged_attention_decode_kernel = None
    paged_attention_decode_batch_kernel = None
