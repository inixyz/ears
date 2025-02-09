#include "world.hpp"

#include <cstdint>

World::World(const Vec3<int> &size)
    : size(size), slice_size(size.x * size.y), data_size(slice_size * size.z) {

  materials = new uint8_t[data_size]();
  data_t0 = new float[data_size]();
  data_t1 = new float[data_size]();
  data_t2 = new float[data_size]();
}

World::~World() {
  if (on_gpu) {
    cudaFree(materials);
    cudaFree(data_t0);
    cudaFree(data_t1);
    cudaFree(data_t2);
  } else {
    delete[] materials;
    delete[] data_t0;
    delete[] data_t1;
    delete[] data_t2;
  }
}

void World::to_gpu() {
  if (on_gpu)
    return;

  uint8_t *gpu_materials = nullptr;
  float *gpu_data_t0 = nullptr, *gpu_data_t1 = nullptr, *gpu_data_t2 = nullptr;

  cudaMalloc(&gpu_materials, data_size * sizeof(uint8_t));
  cudaMalloc(&gpu_data_t0, data_size * sizeof(float));
  cudaMalloc(&gpu_data_t1, data_size * sizeof(float));
  cudaMalloc(&gpu_data_t2, data_size * sizeof(float));

  cudaMemcpy(gpu_materials, materials, data_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_data_t0, data_t0, data_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_data_t1, data_t1, data_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_data_t2, data_t2, data_size * sizeof(float), cudaMemcpyHostToDevice);

  delete[] materials;
  delete[] data_t0;
  delete[] data_t1;
  delete[] data_t2;

  materials = gpu_materials;
  data_t0 = gpu_data_t0;
  data_t1 = gpu_data_t1;
  data_t2 = gpu_data_t2;

  on_gpu = true;
}
