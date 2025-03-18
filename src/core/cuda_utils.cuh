#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

namespace ears {

static inline void cuda_check(const cudaError_t err, const char *const file, const int line,
                              const char *const func) {
  if (err != cudaSuccess) {
    std::cerr << "[CUDA ERROR] " << file << ":" << line << ": " << func << ": "
              << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace ears

#define CUDA_CHECK(func) ears::cuda_check(func, __FILE__, __LINE__, #func)
