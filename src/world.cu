#include "world.cuh"

#include "vec.hpp"
#include <cstdint>

World::World(const Vec3<int> &size)
    : size(size), size_slice(size.x * size.y), size_grid(size_slice * size.z) {

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
