#include "world.cuh"

#include "cuda_utils.cuh"
#include <algorithm>

namespace ears {

World::World(const Vec3i &size, const float courant, const dim3 dim_grid, const dim3 dim_block)
    : size(size), size_xy(size.x * size.y), size_xyz(size_xy * size.z), courant(courant),
      dim_grid(dim_grid), dim_block(dim_block) {

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
  t0 = nullptr;
  CUDA_CHECK(cudaFree(t1));
  t1 = nullptr;
  CUDA_CHECK(cudaFree(t2));
  t2 = nullptr;
}

const Vec3i &World::get_size() const {
  return size;
}

float World::get_courant() const {
  return courant;
}

#define GENERATE_WORLD_GET(time, dtype)                                                            \
  dtype World::get_##time(const Vec3i &pos) const {                                                \
    const int i = pos.x + pos.y * size.x + pos.z * size_xy;                                        \
    dtype val;                                                                                     \
    CUDA_CHECK(cudaMemcpy(&val, time + i, sizeof(dtype), cudaMemcpyDeviceToHost));                 \
    return val;                                                                                    \
  }

GENERATE_WORLD_GET(t0, float)
GENERATE_WORLD_GET(t1, float)
GENERATE_WORLD_GET(t2, float)

#define GENERATE_WORLD_SET(time, dtype)                                                            \
  void World::set_##time(const Vec3i &pos, const dtype val) const {                                \
    const int i = pos.x + pos.y * size.x + pos.z * size_xy;                                        \
    CUDA_CHECK(cudaMemcpy(time + i, &val, sizeof(dtype), cudaMemcpyHostToDevice));                 \
  }

GENERATE_WORLD_SET(t0, float)
GENERATE_WORLD_SET(t1, float)
GENERATE_WORLD_SET(t2, float)

__global__ void fdtd(const Vec3i size, const int size_xy, const float courant, float *const t0,
                     const float *const t1, const float *const t2) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int pos = x + y * size.x + z * size_xy;

  float sum_neighbours = 0;

  if (x > 0)
    sum_neighbours += t1[pos - 1];
  if (y > 0)
    sum_neighbours += t1[pos - size.x];
  if (z > 0)
    sum_neighbours += t1[pos - size_xy];
  if (x < size.x - 1)
    sum_neighbours += t1[pos + 1];
  if (y < size.y - 1)
    sum_neighbours += t1[pos + size.x];
  if (z < size.z - 1)
    sum_neighbours += t1[pos + size_xy];

  t0[pos] = courant * courant * (sum_neighbours - 6 * t1[pos]) + 2 * t1[pos] - t2[pos];
}

void World::step() {
  std::swap(t1, t0);
  std::swap(t2, t0);

  fdtd<<<dim_grid, dim_block>>>(size, size_xy, courant, t0, t1, t2);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void World::step(const int no_iter) {
  for (int i = 0; i < no_iter; i++)
    step();
}

} // namespace ears
