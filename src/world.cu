#include "world.cuh"

#include "vec.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <driver_types.h>

World::World(const Vec3<int> &size, const float grid_spacing_distance)
    : size(size), size_slice(size.x * size.y), size_grid(size_slice * size.z),
      grid_spacing_distance(grid_spacing_distance) {

  // alloc material attributes
  const size_t no_bytes_material_attributes = NO_MATERIALS * sizeof(float);

  cudaMalloc(&material_attributes.courants, no_bytes_material_attributes);
  cudaMalloc(&material_attributes.courants_squared, no_bytes_material_attributes);
  cudaMalloc(&material_attributes.acoustic_impedances_doubled, no_bytes_material_attributes);

  cudaMemset(material_attributes.courants, 0, no_bytes_material_attributes);
  cudaMemset(material_attributes.courants_squared, 0, no_bytes_material_attributes);
  cudaMemset(material_attributes.acoustic_impedances_doubled, 0, no_bytes_material_attributes);

  // alloc grid
  cudaMalloc(&grid.material_ids, size_grid * sizeof(uint8_t));

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
  cudaFree(grid.material_ids);
  cudaFree(grid.t0);
  cudaFree(grid.t1);
  cudaFree(grid.t2);
}

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
