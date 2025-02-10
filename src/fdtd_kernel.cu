#include "fdtd_kernel.cuh"

#include <cstdint>

__global__ void fdtd_step(const int size_x, const int size_y, const int size_z,
                          const int size_slice, const float *const courants,
                          const float *const courants_squared,
                          const float *const acoustic_impedances_doubled,
                          const uint8_t *const material_ids, float *const data_t0,
                          const float *const data_t1, const float *const data_t2) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int pos = x + y * size_x + z * size_slice;

  int nr_neighbours = 0;
  float sum_neighbours = 0;

  if (x > 0) {
    nr_neighbours++;
    sum_neighbours += data_t1[pos - 1];
  }
  if (y > 0) {
    nr_neighbours++;
    sum_neighbours += data_t1[pos - size_x];
  }
  if (z > 0) {
    nr_neighbours++;
    sum_neighbours += data_t1[pos - size_slice];
  }
  if (x < size_x - 1) {
    nr_neighbours++;
    sum_neighbours += data_t1[pos + 1];
  }
  if (y < size_y - 1) {
    nr_neighbours++;
    sum_neighbours += data_t1[pos + size_x];
  }
  if (z < size_z - 1) {
    nr_neighbours++;
    sum_neighbours += data_t1[pos + size_slice];
  }

  const uint8_t material_id = material_ids[pos];
  const float courant = courants[material_id];
  const float courant_squared = courants_squared[material_id];
  const float beta = (6 - nr_neighbours) / acoustic_impedances_doubled[material_id];

  data_t0[pos] =
      (courant_squared * sum_neighbours + (2 - nr_neighbours * courant_squared) * data_t1[pos] +
       (courant * beta - 1) * data_t2[pos]) /
      (1 + courant * beta);
}
