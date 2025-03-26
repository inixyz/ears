#include "world.cuh"

#include "cuda_utils.cuh"
#include "vec.cuh"
#include <algorithm>

namespace ears {

World::World(const Vec3i &size, const float courant, const Vec3i &dim_grid, const Vec3i &dim_block)
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

__global__ void fdtd_old(const Vec3i size, const int size_xy, const float courant, float *const t0,
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

__global__ void fdtd_old2(const Vec3i size, const int size_xy, const float courant, float *const t0,
                          const float *const t1, const float *const t2) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x >= size.x || y >= size.y || z >= size.z)
    return;

  const int pos = x + y * size.x + z * size_xy;

  float sum_neighbours = 0.0f;

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

  // PML config
  const int pml_thickness = 16;
  const float max_sigma = 0.5;

  // Damping in x-direction
  float damping_x = 0.0f;
  if (x < pml_thickness)
    damping_x = max_sigma * powf((float)(pml_thickness - x) / pml_thickness, 2);
  else if (x >= size.x - pml_thickness)
    damping_x = max_sigma * powf((float)(x - (size.x - pml_thickness - 1)) / pml_thickness, 2);

  // Damping in y-direction (SKIP y == 0 only)
  float damping_y = 0.0f;
  if (y >= size.y - pml_thickness)
    damping_y = max_sigma * powf((float)(y - (size.y - pml_thickness - 1)) / pml_thickness, 2);

  // Damping in z-direction
  float damping_z = 0.0f;
  if (z < pml_thickness)
    damping_z = max_sigma * powf((float)(pml_thickness - z) / pml_thickness, 2);
  else if (z >= size.z - pml_thickness)
    damping_z = max_sigma * powf((float)(z - (size.z - pml_thickness - 1)) / pml_thickness, 2);

  float damping = damping_x + damping_y + damping_z;

  // --- AIR LOSS ---
  // Simple exponential decay factor for air absorption
  const float air_loss = 0.002;            // Adjust to control how fast sound decays in the air
  const float air_decay = expf(-air_loss); // decay per timestep

  // Compute update
  float update = courant * courant * (sum_neighbours - 6 * t1[pos]) + 2 * t1[pos] - t2[pos];

  // Apply PML and air loss
  t0[pos] = update * expf(-damping) * air_decay;
}

__global__ void fdtd_old3(const Vec3i size, const int size_xy, const float courant, float *const t0,
                          const float *const t1, const float *const t2) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x >= size.x || y >= size.y || z >= size.z)
    return;

  const int pos = x + y * size.x + z * size_xy;
  const float center = t1[pos];

  float sum = 0.0f;

  if (x > 0)
    sum += t1[pos - 1];
  if (x < size.x - 1)
    sum += t1[pos + 1];
  if (y > 0)
    sum += t1[pos - size.x];
  if (y < size.y - 1)
    sum += t1[pos + size.x];
  if (z > 0)
    sum += t1[pos - size_xy];
  if (z < size.z - 1)
    sum += t1[pos + size_xy];

  // ✅ Smarter damping layer
  float margin = 6.0f; // smaller = higher freq survives
  float damping = 1.0f;

  float dx =
      fminf(fminf(x, size.x - 1 - x), fminf(fminf(y, size.y - 1 - y), fminf(z, size.z - 1 - z)));

  if (dx < margin) {
    float taper = dx / margin;
    float sigma = 1.0f - taper * taper; // quadratic damping ramp
    damping = 1.0f - 0.1f * sigma;      // mild damping only
  }

  float laplacian = sum - 6.0f * center;
  t0[pos] = damping * (courant * courant * laplacian + 2.0f * center - t2[pos]);
}

__global__ void fdtd(const Vec3i size, const int size_xy, const float courant, float *const t0,
                     const float *const t1, const float *const t2) {

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int pos = x + y * size.x + z * size_xy;

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
  float acoustic_imp = 1;
  if (y == 0)
    acoustic_imp = 0.000000000001;

  const float courant_beta = courant * ((6 - nr_neighbours) / (2 * acoustic_imp));

  t0[pos] = (courant_squared * sum_neighbours + (2 - nr_neighbours * courant_squared) * t1[pos] +
             (courant_beta - 1) * t2[pos]) /
            (1 + courant_beta);
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
