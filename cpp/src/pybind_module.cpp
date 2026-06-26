#include "cacheir/cacheir_backend.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<std::size_t> shape_of(const py::array& array) {
  std::vector<std::size_t> shape;
  shape.reserve(static_cast<std::size_t>(array.ndim()));
  for (py::ssize_t i = 0; i < array.ndim(); ++i) {
    shape.push_back(static_cast<std::size_t>(array.shape(i)));
  }
  return shape;
}

}  // namespace

PYBIND11_MODULE(_cacheir_native, m) {
  m.doc() = "Optional CacheIR native CPU kernels";
  m.def("simd_backend", []() { return cacheir::simd_backend(); });

  m.def("rms_norm", [](py::array_t<float, py::array::c_style | py::array::forcecast> x,
                       py::array_t<float, py::array::c_style | py::array::forcecast> weight,
                       float eps) {
    const auto shape = shape_of(x);
    const auto weight_shape = shape_of(weight);
    py::array_t<float> out(shape);
    cacheir::rms_norm(
        cacheir::ConstTensorView{x.data(), std::span<const std::size_t>(shape.data(), shape.size())},
        cacheir::ConstTensorView{weight.data(), std::span<const std::size_t>(weight_shape.data(), weight_shape.size())},
        cacheir::TensorView{out.mutable_data(), std::span<const std::size_t>(shape.data(), shape.size())},
        eps);
    return out;
  });

  m.def("matmul_out_in", [](py::array_t<float, py::array::c_style | py::array::forcecast> x,
                            py::array_t<float, py::array::c_style | py::array::forcecast> weight) {
    const auto x_shape = shape_of(x);
    const auto weight_shape = shape_of(weight);
    if (x_shape.size() != 3 || weight_shape.size() != 2) {
      throw py::value_error("matmul_out_in expects x rank 3 and weight rank 2");
    }
    std::vector<std::size_t> out_shape{x_shape[0], x_shape[1], weight_shape[0]};
    py::array_t<float> out(out_shape);
    cacheir::matmul_out_in(
        cacheir::ConstTensorView{x.data(), std::span<const std::size_t>(x_shape.data(), x_shape.size())},
        cacheir::ConstTensorView{weight.data(), std::span<const std::size_t>(weight_shape.data(), weight_shape.size())},
        cacheir::TensorView{out.mutable_data(), std::span<const std::size_t>(out_shape.data(), out_shape.size())});
    return out;
  });

  m.def("silu_mul", [](py::array_t<float, py::array::c_style | py::array::forcecast> gate,
                       py::array_t<float, py::array::c_style | py::array::forcecast> up) {
    const auto shape = shape_of(gate);
    const auto up_shape = shape_of(up);
    py::array_t<float> out(shape);
    cacheir::silu_mul(
        cacheir::ConstTensorView{gate.data(), std::span<const std::size_t>(shape.data(), shape.size())},
        cacheir::ConstTensorView{up.data(), std::span<const std::size_t>(up_shape.data(), up_shape.size())},
        cacheir::TensorView{out.mutable_data(), std::span<const std::size_t>(shape.data(), shape.size())});
    return out;
  });
}
