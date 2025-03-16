#include "world.cuh"

#include "fdtd.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <iostream>

inline void cuda_check(const cudaError_t err, const char *const file, const int line,
                       const char *const func) {
  if (err != cudaSuccess) {
    std::cerr << "[CUDA ERROR] " << file << ":" << line << ": " << func << ": "
              << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(func) cuda_check(func, __FILE__, __LINE__, #func)

namespace ears {

World::World(const Vec3i &size) : size(size), size_xy(size.x * size.y), size_xyz(size_xy * size.z) {
  const size_t no_bytes = size_xyz * sizeof(float);

  CUDA_CHECK(cudaMalloc(&t0, no_bytes));
  CUDA_CHECK(cudaMalloc(&t1, no_bytes));
  CUDA_CHECK(cudaMalloc(&t2, no_bytes));

  CUDA_CHECK(cudaMemset(t0, 0, no_bytes));
  CUDA_CHECK(cudaMemset(t1, 0, no_bytes));
  CUDA_CHECK(cudaMemset(t2, 0, no_bytes));
}

World::~World() {
  CUDA_CHECK(cudaFree(t0));
  CUDA_CHECK(cudaFree(t1));
  CUDA_CHECK(cudaFree(t2));
}

} // namespace ears

const Vec3i &World::get_size() const {
  return size;
}

float World::get_spacing_distance() const {
  return spacing_distance;
}

float World::get_spacing_temporal() const {
  return spacing_temporal;
}

#define GENERATE_WORLD_GET(grid_member, dtype)                                                     \
  dtype World::get_##grid_member(const Vec3i &pos) const {                                         \
    const int i = pos.x + pos.y * size.x + pos.z * size_slice;                                     \
    dtype val;                                                                                     \
    cudaMemcpy(&val, grid.grid_member + i, sizeof(dtype), cudaMemcpyDeviceToHost);                 \
    return val;                                                                                    \
  }

GENERATE_WORLD_GET(material_id, uint8_t)
GENERATE_WORLD_GET(t0, float)
GENERATE_WORLD_GET(t1, float)
GENERATE_WORLD_GET(t2, float)

#define GENERATE_WORLD_SET(grid_member, dtype)                                                     \
  void World::set_##grid_member(const Vec3i &pos, const dtype val) const {                         \
    const int i = pos.x + pos.y * size.x + pos.z * size_slice;                                     \
    cudaMemcpy(grid.grid_member + i, &val, sizeof(dtype), cudaMemcpyHostToDevice);                 \
  }

GENERATE_WORLD_SET(material_id, uint8_t)
GENERATE_WORLD_SET(t0, float)
GENERATE_WORLD_SET(t1, float)
GENERATE_WORLD_SET(t2, float)

void World::compute_material_attributes() {
  // compute max sound speed found in materials
  auto cmp_sound_speed = [](const Material lhs, const Material rhs) {
    return lhs.sound_speed < rhs.sound_speed;
  };
  const float max_sound_speed =
      std::max_element(materials.begin(), materials.end(), cmp_sound_speed)->sound_speed;

  // adjust dt to satisfy CFL stability condition
  spacing_temporal = spacing_distance / (max_sound_speed * std::sqrt(3));

  // pre-compute material attributes
  for (int i = 0; i < NO_MATERIALS; i++) {
    const float courant = materials[i].sound_speed * spacing_temporal / spacing_distance;
    const float courant_squared = courant * courant;
    const float acoustic_impedance_doubled = 2 * materials[i].acoustic_impedance;

    if (i == 0)
      std::cout << courant << " " << courant_squared << std::endl;

    cudaMemcpy(material_attributes.courants + i, &courant, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(material_attributes.courants_squared + i, &courant_squared, sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(material_attributes.acoustic_impedances_doubled + i, &acoustic_impedance_doubled,
               sizeof(float), cudaMemcpyHostToDevice);
  }
}

void World::step() {
  // advance time axis
  std::swap(grid.t1, grid.t0);
  std::swap(grid.t2, grid.t0);

  fdtd_step<<<dim_grid, dim_block>>>(size, size_slice, material_attributes, grid);
  cudaDeviceSynchronize();
}

void World::step(const int no_iterations) {
  for (int i = 0; i < no_iterations; i++)
    step();
}
