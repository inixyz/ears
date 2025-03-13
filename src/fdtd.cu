#include "fdtd.cuh"

#include "vec.cuh"
#include "world.cuh"
#include <cstdint>

__global__ void fdtd_step_old(const Vec3i size, const int size_slice,
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
  const float courant_squared = material_attributes.courants_squared[material_id];
  const float courant_beta = material_attributes.courants[material_id] * (6 - nr_neighbours) /
                             material_attributes.acoustic_impedances_doubled[material_id];

  grid.t0[pos] =
      (courant_squared * sum_neighbours + (2 - nr_neighbours * courant_squared) * grid.t1[pos] +
       (courant_beta - 1) * grid.t2[pos]) /
      (1 + courant_beta);
}

__global__ void fdtd_step_that_works(const Vec3i size, const int size_slice,
                                     const World::MaterialAttributes material_attributes,
                                     const World::Grid grid) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int pos = x + y * size.x + z * size_slice;

  float sum_neighbours = 0;

  if (x > 0)
    sum_neighbours += grid.t1[pos - 1];
  if (y > 0)
    sum_neighbours += grid.t1[pos - size.x];
  if (z > 0)
    sum_neighbours += grid.t1[pos - size_slice];
  if (x < size.x - 1)
    sum_neighbours += grid.t1[pos + 1];
  if (y < size.y - 1)
    sum_neighbours += grid.t1[pos + size.x];
  if (z < size.z - 1)
    sum_neighbours += grid.t1[pos + size_slice];

  grid.t0[pos] = 0.33 * (sum_neighbours - 6 * grid.t1[pos]) + 2 * grid.t1[pos] - grid.t2[pos];
}

__global__ void fdtd_step_crazy(const Vec3i size, const int size_slice,
                                const World::MaterialAttributes material_attributes,
                                const World::Grid grid) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  // Bounds check
  if (x >= size.x || y >= size.y || z >= size.z)
    return;

  const int pos = x + y * size.x + z * size_slice;
  const uint8_t material_id = grid.material_id[pos];

  // Material properties
  const float S2 = material_attributes.courants_squared[material_id];

  float sum_neighbours = 0.0f;

  // Stencil: 6-point Laplacian (handling boundaries)
  sum_neighbours +=
      (x > 0) ? grid.t1[pos - 1]
              : material_attributes.acoustic_impedances_doubled[material_id] * 0.5f * grid.t1[pos];
  sum_neighbours +=
      (y > 0) ? grid.t1[pos - size.x]
              : material_attributes.acoustic_impedances_doubled[material_id] * 0.5f * grid.t1[pos];
  sum_neighbours +=
      (z > 0) ? grid.t1[pos - size_slice]
              : material_attributes.acoustic_impedances_doubled[material_id] * 0.5f * grid.t1[pos];
  sum_neighbours +=
      (x < size.x - 1)
          ? grid.t1[pos + 1]
          : material_attributes.acoustic_impedances_doubled[material_id] * 0.5f * grid.t1[pos];
  sum_neighbours +=
      (y < size.y - 1)
          ? grid.t1[pos + size.x]
          : material_attributes.acoustic_impedances_doubled[material_id] * 0.5f * grid.t1[pos];
  sum_neighbours +=
      (z < size.z - 1)
          ? grid.t1[pos + size_slice]
          : material_attributes.acoustic_impedances_doubled[material_id] * 0.5f * grid.t1[pos];

  // FDTD update equation
  grid.t0[pos] = S2 * (sum_neighbours - 6 * grid.t1[pos]) + 2 * grid.t1[pos] - grid.t2[pos];
}

__global__ void fdtd_step(const Vec3i size, const int size_slice,
                          const World::MaterialAttributes material_attributes,
                          const World::Grid grid) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int pos = x + y * size.x + z * size_slice;

  float sum_neighbours = 0;

  if (x > 0) {
    sum_neighbours += grid.t1[pos - 1];
  }
  if (y > 0) {
    sum_neighbours += grid.t1[pos - size.x];
  }
  if (z > 0) {
    sum_neighbours += grid.t1[pos - size_slice];
  }
  if (x < size.x - 1) {
    sum_neighbours += grid.t1[pos + 1];
  }
  if (y < size.y - 1) {
    sum_neighbours += grid.t1[pos + size.x];
  }
  if (z < size.z - 1) {
    sum_neighbours += grid.t1[pos + size_slice];
  }

  const uint8_t material_id = grid.material_id[pos];
  const float courant_squared = material_attributes.courants_squared[material_id];

  grid.t0[pos] = courant_squared * sum_neighbours + 2 * (1 - 3 * courant_squared) * grid.t1[pos] -
                 grid.t2[pos];
}
