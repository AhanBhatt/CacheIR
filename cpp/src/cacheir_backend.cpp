#include "cacheir/cacheir_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cacheir {
namespace {

std::size_t checked_rank(ConstTensorView view, std::size_t rank, const char* name) {
  if (view.shape.size() != rank) {
    throw std::invalid_argument(name);
  }
  return view.shape[rank - 1];
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
    float sum = 0.0F;
    for (std::size_t i = 0; i < hidden; ++i) {
      const float value = x.data[base + i];
      sum += value * value;
    }
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
      for (std::size_t in_col = 0; in_col < in_features; ++in_col) {
        acc += x.data[static_cast<std::size_t>(row) * in_features + in_col] *
               weight.data[out_col * in_features + in_col];
      }
      out.data[static_cast<std::size_t>(row) * out_features + out_col] = acc;
    }
  }
}

}  // namespace cacheir
