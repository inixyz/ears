#pragma once

#include <iostream>

static inline void cuda_check(const cudaError_t err, const char *const file, const int line,
                              const char *const func) {
  if (err != cudaSuccess) {
    std::cerr << "[CUDA ERROR] " << file << ":" << line << ": " << func << ": "
              << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(func) cuda_check(func, __FILE__, __LINE__, #func)
