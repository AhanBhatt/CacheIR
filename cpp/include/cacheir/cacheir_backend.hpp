#pragma once

#include <cstddef>
#include <span>

namespace cacheir {

struct TensorView {
  float* data;
  std::span<const std::size_t> shape;
};

struct ConstTensorView {
  const float* data;
  std::span<const std::size_t> shape;
};

void rms_norm(ConstTensorView x, ConstTensorView weight, TensorView out, float eps);
void silu(ConstTensorView x, TensorView out);
void matmul_out_in(ConstTensorView x, ConstTensorView weight, TensorView out);
const char* simd_backend();

}  // namespace cacheir
