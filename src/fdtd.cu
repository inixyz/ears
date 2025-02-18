#include "fdtd.cuh"

#include "vec.hpp"
#include "world.cuh"
#include <cstdint>

__global__ void fdtd_step(const Vec3<int> size, const int size_slice,
                          const World::MaterialAttributes material_attributes,
                          const World::Grid grid) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int pos = x + y * size.x + z * size_slice;

  int nr_neighbours = 0;
  float sum_neighbours = 0;

  if (x > 0) {
    nr_neighbours++;
    sum_neighbours += grid.t1[pos - 1];
  }
  if (y > 0) {
    nr_neighbours++;
    sum_neighbours += grid.t1[pos - size.x];
  }
  if (z > 0) {
    nr_neighbours++;
    sum_neighbours += grid.t1[pos - size_slice];
  }
  if (x < size.x - 1) {
    nr_neighbours++;
    sum_neighbours += grid.t1[pos + 1];
  }
  if (y < size.y - 1) {
    nr_neighbours++;
    sum_neighbours += grid.t1[pos + size.x];
  }
  if (z < size.z - 1) {
    nr_neighbours++;
    sum_neighbours += grid.t1[pos + size_slice];
  }

  const uint8_t material_id = grid.material_id[pos];
  const float courant = material_attributes.courants[material_id];
  const float courant_squared = material_attributes.courants_squared[material_id];
  const float beta =
      (6 - nr_neighbours) / material_attributes.acoustic_impedances_doubled[material_id];

  grid.t0[pos] =
      (courant_squared * sum_neighbours + (2 - nr_neighbours * courant_squared) * grid.t1[pos] +
       (courant * beta - 1) * grid.t2[pos]) /
      (1 + courant * beta);
}
