#include "world.cuh"

#include "cuda_utils.cuh"
#include "vec.cuh"
#include <algorithm>

namespace ears {

World::World(const Vec3i &size, const float courant, const Vec3i &dim_grid, const Vec3i &dim_block)
    : size(size), size_xy(size.x * size.y), size_xyz(size_xy * size.z), courant(courant),
      dim_grid(dim_grid), dim_block(dim_block) {

  const size_t no_bytes = size_xyz * sizeof(float);

  CUDA_CHECK(cudaMalloc(&imp, no_bytes));
  CUDA_CHECK(cudaMalloc(&t0, no_bytes));
  CUDA_CHECK(cudaMalloc(&t1, no_bytes));
  CUDA_CHECK(cudaMalloc(&t2, no_bytes));

  CUDA_CHECK(cudaMemset(imp, 0, no_bytes));
  CUDA_CHECK(cudaMemset(t0, 0, no_bytes));
  CUDA_CHECK(cudaMemset(t1, 0, no_bytes));
  CUDA_CHECK(cudaMemset(t2, 0, no_bytes));
}

World::~World() {
  CUDA_CHECK(cudaFree(imp));
  imp = nullptr;
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

#define GENERATE_WORLD_GET(var, dtype)                                                             \
  dtype World::get_##var(const Vec3i &pos) const {                                                 \
    const int i = pos.x + pos.y * size.x + pos.z * size_xy;                                        \
    dtype val;                                                                                     \
    CUDA_CHECK(cudaMemcpy(&val, var + i, sizeof(dtype), cudaMemcpyDeviceToHost));                  \
    return val;                                                                                    \
  }

GENERATE_WORLD_GET(imp, float)
GENERATE_WORLD_GET(t0, float)
GENERATE_WORLD_GET(t1, float)
GENERATE_WORLD_GET(t2, float)

#define GENERATE_WORLD_SET(var, dtype)                                                             \
  void World::set_##var(const Vec3i &pos, const dtype val) const {                                 \
    const int i = pos.x + pos.y * size.x + pos.z * size_xy;                                        \
    CUDA_CHECK(cudaMemcpy(var + i, &val, sizeof(dtype), cudaMemcpyHostToDevice));                  \
  }

GENERATE_WORLD_SET(imp, float)
GENERATE_WORLD_SET(t0, float)
GENERATE_WORLD_SET(t1, float)
GENERATE_WORLD_SET(t2, float)

template <typename T>
__global__ void kernel_fill(const Vec3i size, const int size_xy, T *const data, const T val) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  data[x + y * size.x + z * size_xy] = val;
}

#define GENERATE_WORLD_FILL(var, dtype)                                                            \
  void World::fill_##var(const dtype val) const {                                                  \
    kernel_fill<dtype><<<dim_grid, dim_block>>>(size, size_xy, var, val);                          \
    CUDA_CHECK(cudaDeviceSynchronize());                                                           \
  }

GENERATE_WORLD_FILL(imp, float)

__global__ void kernel_fdtd(const Vec3i size, const int size_xy, const float courant,
                            const float *const imp, float *const t0, const float *const t1,
                            const float *const t2) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int pos = x + y * size.x + z * size_xy;

  const float imp_val = imp[pos];
  if (!imp_val)
    return;

  int nr_neighbours = 0;
  float sum_neighbours = 0;

  if (x > 0) {
    nr_neighbours++;
    sum_neighbours += t1[pos - 1];
  }
  if (y > 0) {
    nr_neighbours++;
    sum_neighbours += t1[pos - size.x];
  }
  if (z > 0) {
    nr_neighbours++;
    sum_neighbours += t1[pos - size_xy];
  }
  if (x < size.x - 1) {
    nr_neighbours++;
    sum_neighbours += t1[pos + 1];
  }
  if (y < size.y - 1) {
    nr_neighbours++;
    sum_neighbours += t1[pos + size.x];
  }
  if (z < size.z - 1) {
    nr_neighbours++;
    sum_neighbours += t1[pos + size_xy];
  }

  const float courant_squared = courant * courant;
  const float courant_beta = courant * ((6 - nr_neighbours) / (2 * imp_val));

  t0[pos] = (courant_squared * sum_neighbours + (2 - nr_neighbours * courant_squared) * t1[pos] +
             (courant_beta - 1) * t2[pos]) /
            (1 + courant_beta);
}

void World::step() {
  std::swap(t1, t0);
  std::swap(t2, t0);

  kernel_fdtd<<<dim_grid, dim_block>>>(size, size_xy, courant, imp, t0, t1, t2);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void World::step(const int no_iter) {
  for (int i = 0; i < no_iter; i++)
    step();
}

} // namespace ears
