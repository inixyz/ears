#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdexcept>
#include <string>

namespace ears {

static inline void cuda_check(const cudaError_t err, const std::string &file, const int line,
                              const std::string &func) {
  if (err != cudaSuccess) {
    const std::string err_pos = file + ":" + std::to_string(line) + ": " + func + ": ";
    throw std::runtime_error("[CUDA ERROR] " + err_pos + cudaGetErrorString(err));
  }
}

} // namespace ears

#define CUDA_CHECK(func) ears::cuda_check(func, __FILE__, __LINE__, #func)
