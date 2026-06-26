#include "cacheir/cacheir_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#define CACHEIR_GNU_X86 1
#endif

namespace cacheir {
namespace {

std::size_t checked_rank(ConstTensorView view, std::size_t rank, const char* name) {
  if (view.shape.size() != rank) {
    throw std::invalid_argument(name);
  }
  return view.shape[rank - 1];
}

float dot_scalar(const float* lhs, const float* rhs, std::size_t count) {
  float acc = 0.0F;
  for (std::size_t i = 0; i < count; ++i) {
    acc += lhs[i] * rhs[i];
  }
  return acc;
}

#if CACHEIR_GNU_X86
__attribute__((target("avx2,fma")))
float dot_avx2(const float* lhs, const float* rhs, std::size_t count) {
  std::size_t i = 0;
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= count; i += 8) {
    const __m256 a = _mm256_loadu_ps(lhs + i);
    const __m256 b = _mm256_loadu_ps(rhs + i);
    acc = _mm256_fmadd_ps(a, b, acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < count; ++i) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

__attribute__((target("avx512f")))
float dot_avx512(const float* lhs, const float* rhs, std::size_t count) {
  std::size_t i = 0;
  __m512 acc = _mm512_setzero_ps();
  for (; i + 16 <= count; i += 16) {
    const __m512 a = _mm512_loadu_ps(lhs + i);
    const __m512 b = _mm512_loadu_ps(rhs + i);
    acc = _mm512_add_ps(acc, _mm512_mul_ps(a, b));
  }
  float sum = _mm512_reduce_add_ps(acc);
  for (; i < count; ++i) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

bool has_avx2() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

bool has_avx512() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f");
}
#endif

float dot_product(const float* lhs, const float* rhs, std::size_t count) {
#if CACHEIR_GNU_X86
  if (has_avx512()) {
    return dot_avx512(lhs, rhs, count);
  }
  if (has_avx2()) {
    return dot_avx2(lhs, rhs, count);
  }
#endif
  return dot_scalar(lhs, rhs, count);
}

}  // namespace

void rms_norm(ConstTensorView x, ConstTensorView weight, TensorView out, float eps) {
  const auto hidden = checked_rank(x, 3, "rms_norm expects x rank 3");
  if (weight.shape.size() != 1 || weight.shape[0] != hidden) {
    throw std::invalid_argument("rms_norm weight shape mismatch");
  }
  const std::size_t rows = x.shape[0] * x.shape[1];

#if CACHEIR_USE_OPENMP
#pragma omp parallel for
#endif
  for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(rows); ++row) {
    const auto base = static_cast<std::size_t>(row) * hidden;
    const float sum = dot_product(x.data + base, x.data + base, hidden);
    const float scale = 1.0F / std::sqrt(sum / static_cast<float>(hidden) + eps);
    for (std::size_t i = 0; i < hidden; ++i) {
      out.data[base + i] = x.data[base + i] * scale * weight.data[i];
    }
  }
}

void silu(ConstTensorView x, TensorView out) {
  std::size_t total = 1;
  for (const auto dim : x.shape) {
    total *= dim;
  }
#if CACHEIR_USE_OPENMP
#pragma omp parallel for
#endif
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(total); ++i) {
    const float value = x.data[i];
    out.data[i] = value / (1.0F + std::exp(-value));
  }
}

void silu_mul(ConstTensorView gate, ConstTensorView up, TensorView out) {
  if (gate.shape.size() != up.shape.size() || gate.shape.size() != out.shape.size()) {
    throw std::invalid_argument("silu_mul expects matching ranks");
  }
  std::size_t total = 1;
  for (std::size_t i = 0; i < gate.shape.size(); ++i) {
    if (gate.shape[i] != up.shape[i] || gate.shape[i] != out.shape[i]) {
      throw std::invalid_argument("silu_mul shape mismatch");
    }
    total *= gate.shape[i];
  }
#if CACHEIR_USE_OPENMP
#pragma omp parallel for
#endif
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(total); ++i) {
    const float value = gate.data[i];
    out.data[i] = (value / (1.0F + std::exp(-value))) * up.data[i];
  }
}

void matmul_out_in(ConstTensorView x, ConstTensorView weight, TensorView out) {
  const auto in_features = checked_rank(x, 3, "matmul expects x rank 3");
  if (weight.shape.size() != 2 || weight.shape[1] != in_features) {
    throw std::invalid_argument("matmul weight shape mismatch");
  }
  const auto rows = x.shape[0] * x.shape[1];
  const auto out_features = weight.shape[0];

#if CACHEIR_USE_OPENMP
#pragma omp parallel for
#endif
  for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(rows); ++row) {
    for (std::size_t out_col = 0; out_col < out_features; ++out_col) {
      float acc = 0.0F;
      acc = dot_product(x.data + static_cast<std::size_t>(row) * in_features,
                        weight.data + out_col * in_features,
                        in_features);
      out.data[static_cast<std::size_t>(row) * out_features + out_col] = acc;
    }
  }
}

const char* simd_backend() {
#if CACHEIR_GNU_X86
  if (has_avx512()) {
    return "avx512";
  }
  if (has_avx2()) {
    return "avx2";
  }
#endif
  return "scalar";
}

}  // namespace cacheir
