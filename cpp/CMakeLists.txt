cmake_minimum_required(VERSION 3.20)
project(cacheir_backend LANGUAGES CXX)
include(CheckLanguage)

option(CACHEIR_USE_OPENMP "Build CacheIR CPU kernels with OpenMP" ON)
option(CACHEIR_BUILD_PYTHON "Build optional pybind11 Python extension" OFF)
option(CACHEIR_BUILD_CUDA "Build optional CUDA fused kernels" OFF)
option(CACHEIR_BUILD_MLIR "Build optional upstream MLIR CacheIR dialect registration library" OFF)

add_library(cacheir_backend STATIC
  src/cacheir_backend.cpp
)
target_include_directories(cacheir_backend PUBLIC include)
target_compile_features(cacheir_backend PUBLIC cxx_std_20)

if(CACHEIR_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cacheir_backend PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(cacheir_backend PUBLIC CACHEIR_USE_OPENMP=1)
  endif()
endif()

if(CACHEIR_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE CACHEIR_PYBIND11_CMAKEDIR
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CACHEIR_PYBIND11_CMAKEDIR)
      list(APPEND CMAKE_PREFIX_PATH "${CACHEIR_PYBIND11_CMAKEDIR}")
    endif()
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "from pathlib import Path; import importlib.util; spec=importlib.util.find_spec('torch'); print(Path(spec.origin).parent / 'include' if spec else '')"
      OUTPUT_VARIABLE CACHEIR_TORCH_INCLUDE
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(_cacheir_native src/pybind_module.cpp)
    target_link_libraries(_cacheir_native PRIVATE cacheir_backend)
    target_compile_features(_cacheir_native PRIVATE cxx_std_20)
  elseif(Python3_Development.Module_FOUND AND CACHEIR_TORCH_INCLUDE)
    find_path(CACHEIR_PYBIND11_INCLUDE_DIR
      NAMES pybind11/pybind11.h
      PATHS "${CACHEIR_TORCH_INCLUDE}"
      NO_DEFAULT_PATH
    )
    if(CACHEIR_PYBIND11_INCLUDE_DIR)
      add_library(_cacheir_native MODULE src/pybind_module.cpp)
      target_include_directories(_cacheir_native PRIVATE "${CACHEIR_PYBIND11_INCLUDE_DIR}")
      target_link_libraries(_cacheir_native PRIVATE cacheir_backend Python3::Module)
      target_compile_features(_cacheir_native PRIVATE cxx_std_20)
      set_target_properties(_cacheir_native PROPERTIES PREFIX "")
      if(Python3_EXTENSION_SUFFIX)
        set_target_properties(_cacheir_native PROPERTIES SUFFIX "${Python3_EXTENSION_SUFFIX}")
      else()
        set_target_properties(_cacheir_native PROPERTIES SUFFIX ".pyd")
      endif()
    else()
      message(WARNING "CACHEIR_BUILD_PYTHON=ON but pybind11 headers were not found")
    endif()
  else()
    message(WARNING "CACHEIR_BUILD_PYTHON=ON but pybind11 was not found")
  endif()
endif()

if(CACHEIR_BUILD_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_library(cacheir_cuda_kernels STATIC src/cacheir_cuda_kernels.cu)
    target_include_directories(cacheir_cuda_kernels PUBLIC include)
    set_target_properties(cacheir_cuda_kernels PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON
    )
  else()
    message(WARNING "CACHEIR_BUILD_CUDA=ON but no CUDA compiler was found")
  endif()
endif()

if(CACHEIR_BUILD_MLIR)
  find_package(MLIR CONFIG QUIET)
  if(MLIR_FOUND)
    message(STATUS "Building CacheIR MLIR dialect registration against ${MLIR_DIR}")
    add_library(cacheir_mlir_dialect STATIC
      mlir/src/CacheIRDialect.cpp
    )
    target_include_directories(cacheir_mlir_dialect
      PUBLIC
        mlir/include
        ${MLIR_INCLUDE_DIRS}
    )
    target_compile_features(cacheir_mlir_dialect PUBLIC cxx_std_20)
    target_compile_definitions(cacheir_mlir_dialect PUBLIC ${MLIR_DEFINITIONS})
    target_link_libraries(cacheir_mlir_dialect PUBLIC MLIRIR)
  else()
    message(WARNING "CACHEIR_BUILD_MLIR=ON but upstream MLIR was not found; set MLIR_DIR to an MLIRConfig.cmake directory")
  endif()
endif()

if(MSVC)
  target_compile_options(cacheir_backend PRIVATE /W4)
else()
  target_compile_options(cacheir_backend PRIVATE -Wall -Wextra -Wpedantic)
endif()
