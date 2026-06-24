cmake_minimum_required(VERSION 3.20)
project(cacheir_backend LANGUAGES CXX)

option(CACHEIR_USE_OPENMP "Build CacheIR CPU kernels with OpenMP" ON)

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

if(MSVC)
  target_compile_options(cacheir_backend PRIVATE /W4)
else()
  target_compile_options(cacheir_backend PRIVATE -Wall -Wextra -Wpedantic)
endif()
