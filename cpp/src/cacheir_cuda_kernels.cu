#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <math.h>
#include <stddef.h>

namespace {

using namespace nvcuda;

__device__ float silu(float value) {
  return value / (1.0F + expf(-value));
}

__device__ float project_value(const float* x,
                               const float* norm_weight,
                               const float* weight,
                               int row,
                               int hidden,
                               int col,
                               float eps) {
  const float* row_x = x + static_cast<size_t>(row) * hidden;
  float sum = 0.0F;
  for (int i = 0; i < hidden; ++i) {
    sum += row_x[i] * row_x[i];
  }
  const float scale = rsqrtf(sum / static_cast<float>(hidden) + eps);
  float acc = 0.0F;
  const float* row_w = weight + static_cast<size_t>(col) * hidden;
  for (int i = 0; i < hidden; ++i) {
    acc += row_x[i] * scale * norm_weight[i] * row_w[i];
  }
  return acc;
}

__global__ void rms_norm_kernel(const float* x,
                                const float* weight,
                                float* out,
                                int rows,
                                int hidden,
                                float eps) {
  const int row = blockIdx.x;
  const int col = threadIdx.x + blockIdx.y * blockDim.x;
  if (row >= rows || col >= hidden) {
    return;
  }
  const float* row_x = x + static_cast<size_t>(row) * hidden;
  float sum = 0.0F;
  for (int i = 0; i < hidden; ++i) {
    sum += row_x[i] * row_x[i];
  }
  const float scale = rsqrtf(sum / static_cast<float>(hidden) + eps);
  out[static_cast<size_t>(row) * hidden + col] = row_x[col] * scale * weight[col];
}

__global__ void fused_swiglu_kernel(const float* gate, const float* up, float* out, size_t total) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = silu(gate[idx]) * up[idx];
  }
}

__global__ void matmul_f16_scalar_kernel(const half* a, const half* b, float* out, int m, int n, int k) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }
  float acc = 0.0F;
  for (int idx = 0; idx < k; ++idx) {
    acc += __half2float(a[static_cast<size_t>(row) * k + idx]) *
           __half2float(b[static_cast<size_t>(idx) * n + col]);
  }
  out[static_cast<size_t>(row) * n + col] = acc;
}

__global__ void matmul_f16_tensorcore_kernel(const half* a, const half* b, float* out, int m, int n, int k) {
  const int tile_n = blockIdx.x;
  const int tile_m = blockIdx.y;
  if ((tile_m + 1) * 16 > m || (tile_n + 1) * 16 > n) {
    return;
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0F);

  for (int tile_k = 0; tile_k < k; tile_k += 16) {
    const half* a_tile = a + static_cast<size_t>(tile_m) * 16 * k + tile_k;
    const half* b_tile = b + static_cast<size_t>(tile_k) * n + tile_n * 16;
    wmma::load_matrix_sync(a_frag, a_tile, k);
    wmma::load_matrix_sync(b_frag, b_tile, n);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }
  wmma::store_matrix_sync(out + static_cast<size_t>(tile_m) * 16 * n + tile_n * 16, acc_frag, n, wmma::mem_row_major);
}

__global__ void fused_rmsnorm_qkv_rope_kernel(const float* x,
                                              const float* norm_weight,
                                              const float* q_weight,
                                              const float* k_weight,
                                              const float* v_weight,
                                              float* q_out,
                                              float* k_out,
                                              float* v_out,
                                              int rows,
                                              int hidden,
                                              int q_out_dim,
                                              int kv_out_dim,
                                              int head_dim,
                                              int position_offset,
                                              float eps,
                                              float rope_theta) {
  const int col = blockIdx.x;
  const int row = blockIdx.y;
  const int kind = blockIdx.z;
  if (row >= rows) {
    return;
  }

  if (kind == 2) {
    if (col >= kv_out_dim) {
      return;
    }
    v_out[static_cast<size_t>(row) * kv_out_dim + col] =
        project_value(x, norm_weight, v_weight, row, hidden, col, eps);
    return;
  }

  const int out_dim = kind == 0 ? q_out_dim : kv_out_dim;
  if (col >= out_dim) {
    return;
  }
  if (head_dim <= 0 || (col % head_dim) % 2 != 0 || col + 1 >= out_dim) {
    return;
  }

  const float* weight = kind == 0 ? q_weight : k_weight;
  float* out = kind == 0 ? q_out : k_out;
  const float raw0 = project_value(x, norm_weight, weight, row, hidden, col, eps);
  const float raw1 = project_value(x, norm_weight, weight, row, hidden, col + 1, eps);
  const int pair = (col % head_dim) / 2;
  const float freq = expf((-2.0F * static_cast<float>(pair) / static_cast<float>(head_dim)) * logf(rope_theta));
  const float angle = static_cast<float>(position_offset + row) * freq;
  const float c = cosf(angle);
  const float s = sinf(angle);
  out[static_cast<size_t>(row) * out_dim + col] = raw0 * c - raw1 * s;
  out[static_cast<size_t>(row) * out_dim + col + 1] = raw0 * s + raw1 * c;
}

__global__ void paged_attention_decode_batch_kernel(const float* q,
                                                    const float* k_cache,
                                                    const float* v_cache,
                                                    const int* page_table,
                                                    const int* seq_lens,
                                                    float* out,
                                                    int batch_size,
                                                    int num_heads,
                                                    int num_kv_heads,
                                                    int max_pages_per_seq,
                                                    int page_size,
                                                    int head_dim) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int dim = blockIdx.z;
  if (batch >= batch_size || head >= num_heads || dim >= head_dim || num_kv_heads <= 0) {
    return;
  }
  const int group_size = num_heads / num_kv_heads;
  const int kv_head = head / group_size;
  const int seq_len = seq_lens[batch];
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float max_score = -3.4028234663852886e38F;
  float denom = 0.0F;
  float acc = 0.0F;

  for (int page_idx = 0; page_idx < max_pages_per_seq; ++page_idx) {
    const int page_id = page_table[batch * max_pages_per_seq + page_idx];
    for (int slot = 0; slot < page_size; ++slot) {
      const int pos = page_idx * page_size + slot;
      if (pos >= seq_len) {
        continue;
      }
      const size_t q_base = (static_cast<size_t>(batch) * num_heads + head) * head_dim;
      const size_t kv_base =
          ((static_cast<size_t>(page_id) * num_kv_heads + kv_head) * page_size + slot) * head_dim;
      float score = 0.0F;
      for (int d = 0; d < head_dim; ++d) {
        score += q[q_base + d] * k_cache[kv_base + d];
      }
      score *= scale;
      const float new_max = fmaxf(max_score, score);
      const float old_scale = expf(max_score - new_max);
      const float new_scale = expf(score - new_max);
      acc = acc * old_scale + v_cache[kv_base + dim] * new_scale;
      denom = denom * old_scale + new_scale;
      max_score = new_max;
    }
  }
  out[(static_cast<size_t>(batch) * num_heads + head) * head_dim + dim] = acc / denom;
}

__global__ void paged_attention_decode_batch_reduced_kernel(const float* q,
                                                            const float* k_cache,
                                                            const float* v_cache,
                                                            const int* page_table,
                                                            const int* seq_lens,
                                                            float* out,
                                                            int batch_size,
                                                            int num_heads,
                                                            int num_kv_heads,
                                                            int max_pages_per_seq,
                                                            int page_size,
                                                            int head_dim) {
  extern __shared__ float scratch[];
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (batch >= batch_size || head >= num_heads || num_kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int group_size = num_heads / num_kv_heads;
  const int kv_head = head / group_size;
  const int seq_len = seq_lens[batch];
  const size_t q_base = (static_cast<size_t>(batch) * num_heads + head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float max_score = -3.4028234663852886e38F;
  float denom = 0.0F;
  float acc = 0.0F;

  for (int page_idx = 0; page_idx < max_pages_per_seq; ++page_idx) {
    const int page_id = page_table[batch * max_pages_per_seq + page_idx];
    for (int slot = 0; slot < page_size; ++slot) {
      const int pos = page_idx * page_size + slot;
      if (pos >= seq_len) {
        continue;
      }
      const size_t kv_base =
          ((static_cast<size_t>(page_id) * num_kv_heads + kv_head) * page_size + slot) * head_dim;
      float partial = 0.0F;
      if (tid < head_dim) {
        partial = q[q_base + tid] * k_cache[kv_base + tid];
      }
      scratch[tid] = partial;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
      }
      const float score = scratch[0] * scale;
      const float new_max = fmaxf(max_score, score);
      const float old_scale = expf(max_score - new_max);
      const float new_scale = expf(score - new_max);
      if (tid < head_dim) {
        acc = acc * old_scale + v_cache[kv_base + tid] * new_scale;
      }
      denom = denom * old_scale + new_scale;
      max_score = new_max;
      __syncthreads();
    }
  }
  if (tid < head_dim) {
    out[(static_cast<size_t>(batch) * num_heads + head) * head_dim + tid] = acc / denom;
  }
}

}  // namespace

extern "C" cudaError_t cacheir_cuda_rms_norm(const float* x,
                                             const float* weight,
                                             float* out,
                                             int rows,
                                             int hidden,
                                             float eps,
                                             cudaStream_t stream) {
  const dim3 block(256);
  const dim3 grid(rows, (hidden + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));
  rms_norm_kernel<<<grid, block, 0, stream>>>(x, weight, out, rows, hidden, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t cacheir_cuda_fused_swiglu(const float* gate,
                                                 const float* up,
                                                 float* out,
                                                 size_t total,
                                                 cudaStream_t stream) {
  const dim3 block(256);
  const dim3 grid((total + block.x - 1) / block.x);
  fused_swiglu_kernel<<<grid, block, 0, stream>>>(gate, up, out, total);
  return cudaGetLastError();
}

extern "C" cudaError_t cacheir_cuda_matmul_f16(const half* a,
                                               const half* b,
                                               float* out,
                                               int m,
                                               int n,
                                               int k,
                                               cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return cudaSuccess;
  }
  if ((m % 16) == 0 && (n % 16) == 0 && (k % 16) == 0) {
    const dim3 grid(n / 16, m / 16);
    matmul_f16_tensorcore_kernel<<<grid, 32, 0, stream>>>(a, b, out, m, n, k);
    return cudaGetLastError();
  }
  const dim3 block(16, 16);
  const dim3 grid((n + static_cast<int>(block.x) - 1) / static_cast<int>(block.x),
                  (m + static_cast<int>(block.y) - 1) / static_cast<int>(block.y));
  matmul_f16_scalar_kernel<<<grid, block, 0, stream>>>(a, b, out, m, n, k);
  return cudaGetLastError();
}

extern "C" cudaError_t cacheir_cuda_fused_rmsnorm_qkv_rope(const float* x,
                                                           const float* norm_weight,
                                                           const float* q_weight,
                                                           const float* k_weight,
                                                           const float* v_weight,
                                                           float* q_out,
                                                           float* k_out,
                                                           float* v_out,
                                                           int rows,
                                                           int hidden,
                                                           int q_out_dim,
                                                           int kv_out_dim,
                                                           int head_dim,
                                                           int position_offset,
                                                           float eps,
                                                           float rope_theta,
                                                           cudaStream_t stream) {
  const int max_out = q_out_dim > kv_out_dim ? q_out_dim : kv_out_dim;
  const dim3 grid(max_out, rows, 3);
  fused_rmsnorm_qkv_rope_kernel<<<grid, 1, 0, stream>>>(x,
                                                       norm_weight,
                                                       q_weight,
                                                       k_weight,
                                                       v_weight,
                                                       q_out,
                                                       k_out,
                                                       v_out,
                                                       rows,
                                                       hidden,
                                                       q_out_dim,
                                                       kv_out_dim,
                                                       head_dim,
                                                       position_offset,
                                                       eps,
                                                       rope_theta);
  return cudaGetLastError();
}

extern "C" cudaError_t cacheir_cuda_paged_attention_decode_batch(const float* q,
                                                                  const float* k_cache,
                                                                  const float* v_cache,
                                                                  const int* page_table,
                                                                  const int* seq_lens,
                                                                  float* out,
                                                                  int batch_size,
                                                                  int num_heads,
                                                                  int num_kv_heads,
                                                                  int max_pages_per_seq,
                                                                  int page_size,
                                                                  int head_dim,
                                                                  cudaStream_t stream) {
  if (batch_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return cudaSuccess;
  }
  if (head_dim <= 1024) {
    int block_size = 1;
    while (block_size < head_dim) {
      block_size <<= 1;
    }
    if (block_size < 32) {
      block_size = 32;
    }
    const dim3 grid(num_heads, batch_size);
    paged_attention_decode_batch_reduced_kernel<<<grid, block_size, static_cast<size_t>(block_size) * sizeof(float), stream>>>(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        out,
        batch_size,
        num_heads,
        num_kv_heads,
        max_pages_per_seq,
        page_size,
        head_dim);
    return cudaGetLastError();
  }
  const dim3 grid(num_heads, batch_size, head_dim);
  paged_attention_decode_batch_kernel<<<grid, 1, 0, stream>>>(q,
                                                              k_cache,
                                                              v_cache,
                                                              page_table,
                                                              seq_lens,
                                                              out,
                                                              batch_size,
                                                              num_heads,
                                                              num_kv_heads,
                                                              max_pages_per_seq,
                                                              page_size,
                                                              head_dim);
  return cudaGetLastError();
}
