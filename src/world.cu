#include "world.cuh"

#include "fdtd.cuh"
#include "vec.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

World::World(const Vec3i &size, const float grid_spacing_distance)
    : size(size), size_slice(size.x * size.y), size_grid(size_slice * size.z),
      grid_spacing_distance(grid_spacing_distance),
      dim_grid(dim3(size.x / dim_block.x, size.y / dim_block.y, size.z / dim_block.z)) {

  // alloc material attributes
  const size_t no_bytes_material_attributes = NO_MATERIALS * sizeof(float);

  cudaMalloc(&material_attributes.courants, no_bytes_material_attributes);
  cudaMalloc(&material_attributes.courants_squared, no_bytes_material_attributes);
  cudaMalloc(&material_attributes.acoustic_impedances_doubled, no_bytes_material_attributes);

  cudaMemset(material_attributes.courants, 0, no_bytes_material_attributes);
  cudaMemset(material_attributes.courants_squared, 0, no_bytes_material_attributes);
  cudaMemset(material_attributes.acoustic_impedances_doubled, 0, no_bytes_material_attributes);

  // alloc grid
  cudaMalloc(&grid.material_id, size_grid * sizeof(uint8_t));

  const size_t no_bytes_grid = size_grid * sizeof(float);

  cudaMalloc(&grid.t0, no_bytes_grid);
  cudaMalloc(&grid.t1, no_bytes_grid);
  cudaMalloc(&grid.t2, no_bytes_grid);

  cudaMemset(grid.t0, 0, no_bytes_grid);
  cudaMemset(grid.t1, 0, no_bytes_grid);
  cudaMemset(grid.t2, 0, no_bytes_grid);
}

World::~World() {
  // free material attributes
  cudaFree(material_attributes.courants);
  cudaFree(material_attributes.courants_squared);
  cudaFree(material_attributes.acoustic_impedances_doubled);

  // free grid
  cudaFree(grid.material_id);
  cudaFree(grid.t0);
  cudaFree(grid.t1);
  cudaFree(grid.t2);
}

const Vec3i &World::get_size() const {
  return size;
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

void World::compute_material_attributes() const {
  // compute max sound speed found in materials
  auto cmp_sound_speed = [](const Material lhs, const Material rhs) {
    return lhs.sound_speed < rhs.sound_speed;
  };
  const float max_sound_speed =
      std::max_element(materials.begin(), materials.end(), cmp_sound_speed)->sound_speed;

  const float grid_spacing_temporal = grid_spacing_distance / (max_sound_speed * std::sqrt(3));

  for (int i = 0; i < NO_MATERIALS; i++) {
    const float courant = materials[i].sound_speed * grid_spacing_temporal / grid_spacing_distance;
    const float courant_squared = courant * courant;
    const float acoustic_impedance_doubled = 2 * materials[i].acoustic_impedance;

    cudaMemcpy(material_attributes.courants + i, &courant, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(material_attributes.courants_squared + i, &courant_squared, sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(material_attributes.acoustic_impedances_doubled + i, &acoustic_impedance_doubled,
               sizeof(float), cudaMemcpyHostToDevice);
  }
}

void World::step(const int no_iterations) {
  for (int i = 0; i < no_iterations; i++) {
    // advance time axis
    std::swap(grid.t1, grid.t0);
    std::swap(grid.t2, grid.t0);

    fdtd_step<<<dim_grid, dim_block>>>(size, size_slice, material_attributes, grid);
    cudaDeviceSynchronize();
  }
}
