#include "world.hpp"

#include <cstdint>

World::World(const Device device, const int size_x, const int size_y, const int size_z) {
  this->device = device;

  this->size_x = size_x;
  this->size_y = size_y;
  this->size_z = size_z;
  this->size_slice = size_x * size_y;
  this->size_data = this->size_slice * size_z;

  if (device == CPU) {
    courants = new float[NO_MATERIALS]();
    courants_squared = new float[NO_MATERIALS]();
    acoustic_impedances_doubled = new float[NO_MATERIALS]();

    material_ids = new uint8_t[size_data]();
    data_t0 = new float[size_data]();
    data_t1 = new float[size_data]();
    data_t2 = new float[size_data]();
  } else if (device == GPU) {
    const size_t material_attribute_bytes = NO_MATERIALS * sizeof(float);

    cudaMalloc(&courants, material_attribute_bytes);
    cudaMalloc(&courants_squared, material_attribute_bytes);
    cudaMalloc(&acoustic_impedances_doubled, material_attribute_bytes);

    cudaMalloc(&material_ids, size_data * sizeof(uint8_t));

    const size_t data_bytes = size_data * sizeof(float);
    cudaMalloc(&data_t0, data_bytes);
    cudaMalloc(&data_t1, data_bytes);
    cudaMalloc(&data_t2, data_bytes);
  }
}

World::~World() {
  if (device == CPU) {
    delete[] courants;
    delete[] courants_squared;
    delete[] acoustic_impedances_doubled;

    delete[] material_ids;

    delete[] data_t0;
    delete[] data_t1;
    delete[] data_t2;
  } else if (device == GPU) {
    cudaFree(courants);
    cudaFree(courants_squared);
    cudaFree(acoustic_impedances_doubled);

    cudaFree(material_ids);

    cudaFree(data_t0);
    cudaFree(data_t1);
    cudaFree(data_t2);
  }
}

/*
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
*/
