#include <cassert>
#include <chrono>
#include <cmath>
#include <driver_types.h>
#include <fstream>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *data_t0, *data_t1, *data_t2;
  uint8_t *material_ids;
  int size_x, size_y, size_z, size_slice, size;
} World;

#define LAYER_0(world, x, y, z)                                                \
  (world->data_t0[x + y * world->size_x + z * world->size_slice])
#define LAYER_1(world, x, y, z)                                                \
  (world->data_t1[x + y * world->size_x + z * world->size_slice])
#define LAYER_2(world, x, y, z)                                                \
  (world->data_t2[x + y * world->size_x + z * world->size_slice])
#define MATERIAL_ID(world, x, y, z)                                            \
  (world->material_ids[x + y * world->size_x + z * world->size_slice])

typedef enum { WORLD_MEM_HOST, WORLD_MEM_DEVICE } WorldMemType;

World *world_alloc(const int size_x, const int size_y, const int size_z,
                   const WorldMemType type) {
  World *world;

  switch (type) {
  case WORLD_MEM_HOST: {
    world = (World *)malloc(sizeof(World));

    world->size_x = size_x;
    world->size_y = size_y;
    world->size_z = size_z;
    world->size_slice = world->size_x * world->size_y;
    world->size = world->size_x * world->size_y * world->size_z;

    printf("[INFO] Allocating %lu bytes.\n",
           world->size * (3 * sizeof(float) + sizeof(uint8_t)));

    world->data_t0 = (float *)malloc(world->size * sizeof(float));
    world->data_t1 = (float *)malloc(world->size * sizeof(float));
    world->data_t2 = (float *)malloc(world->size * sizeof(float));
    world->material_ids = (uint8_t *)malloc(world->size * sizeof(uint8_t));
  } break;
  case WORLD_MEM_DEVICE: {
    cudaMalloc(&world, sizeof(World));

    int world_size_slice = size_x * size_y,
        world_size = size_x * size_y * size_z;

    cudaMemcpy(&world->size_x, &size_x, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&world->size_y, &size_y, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&world->size_z, &size_z, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&world->size_slice, &world_size_slice, sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&world->size, &world_size, sizeof(int), cudaMemcpyHostToDevice);

    printf("[INFO] Allocating %lu bytes on GPU.\n",
           world_size * (3 * sizeof(float) + sizeof(uint8_t)));

    float *data_t0, *data_t1, *data_t2;
    cudaMalloc(&data_t0, world_size * sizeof(float));
    cudaMalloc(&data_t1, world_size * sizeof(float));
    cudaMalloc(&data_t2, world_size * sizeof(float));

    cudaMemcpy(&world->data_t0, &data_t0, sizeof(float *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&world->data_t1, &data_t1, sizeof(float *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&world->data_t2, &data_t2, sizeof(float *),
               cudaMemcpyHostToDevice);

    uint8_t *material_ids;
    cudaMalloc(&material_ids, world_size * sizeof(uint8_t));

    cudaMemcpy(&world->material_ids, &material_ids, sizeof(uint8_t *),
               cudaMemcpyHostToDevice);
  } break;
  default: {
    printf("[ERROR] Invalid WorldAllocType.\n");
    exit(EXIT_FAILURE);
  }
  }

  return world;
}

void world_free(World *const world, const WorldMemType type) {
  switch (type) {
  case WORLD_MEM_HOST: {
    free(world->data_t0);
    free(world->data_t1);
    free(world->data_t2);
    free(world->material_ids);
    free(world);
  } break;
  case WORLD_MEM_DEVICE: {
    float *data_t0, *data_t1, *data_t2;
    uint8_t *material_ids;

    cudaMemcpy(&data_t0, &world->data_t0, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&data_t1, &world->data_t1, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&data_t2, &world->data_t2, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&material_ids, &world->material_ids, sizeof(uint8_t *),
               cudaMemcpyDeviceToHost);

    cudaFree(data_t0);
    cudaFree(data_t1);
    cudaFree(data_t2);
    cudaFree(material_ids);
    cudaFree(world);
  } break;
  default: {
    printf("[ERROR] Invalid WorldMemType in world_free().\n");
    exit(EXIT_FAILURE);
  }
  }
}

typedef enum {
  WORLD_CPY_HOST_TO_DEVICE,
  WORLD_CPY_DEVICE_TO_HOST
} WorldCpyType;

World *world_cpy(World *const dst, const World *const src,
                 const WorldCpyType type) {

  float *data_t0, *data_t1, *data_t2;
  uint8_t *material_ids;

  switch (type) {
  case WORLD_CPY_HOST_TO_DEVICE: {
    cudaMemcpy(&data_t0, &dst->data_t0, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&data_t1, &dst->data_t1, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&data_t2, &dst->data_t2, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&material_ids, &dst->material_ids, sizeof(uint8_t *),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(data_t0, src->data_t0, src->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(data_t1, src->data_t1, src->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(data_t2, src->data_t2, src->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(material_ids, src->material_ids, src->size * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
  } break;
  case WORLD_CPY_DEVICE_TO_HOST: {
    cudaMemcpy(&data_t0, &src->data_t0, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&data_t1, &src->data_t1, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&data_t2, &src->data_t2, sizeof(float *),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&material_ids, &src->material_ids, sizeof(uint8_t *),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(dst->data_t0, data_t0, dst->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->data_t1, data_t1, dst->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->data_t2, data_t2, dst->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->material_ids, material_ids, dst->size * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
  } break;
  default: {
    printf("[ERROR] Invalid WorldCpyType.\n");
    exit(EXIT_FAILURE);
  }
  }

  return dst;
}

typedef struct {
  float sound_speed, acoustic_impedance;
} Material;

typedef struct {
  float courant, courant_squared, acoustic_impedance_doubled;
} MaterialComputed;

float max_material_sound_speed(const Material *const materials,
                               const uint8_t materials_len) {

  if (!materials_len) {
    printf("[ERROR] Cannot get the maximum sound speed of the provided "
           "materials. Material array length is 0.\n");
    exit(EXIT_FAILURE);
  }

  float max_sound_speed = materials[0].sound_speed;
  for (uint8_t i = 1; i < materials_len; i++) {
    const float sound_speed = materials[i].sound_speed;
    if (max_sound_speed > sound_speed)
      max_sound_speed = sound_speed;
  }

  return max_sound_speed;
}

void world_pre_compute(World *const world, const float dx,
                       const Material *const materials,
                       MaterialComputed *const materials_computed,
                       const uint8_t materials_len) {

  const float max_sound_speed =
      max_material_sound_speed(materials, materials_len);

  const float dt = dx / (max_sound_speed * sqrt(3));

  for (int i = 0; i < materials_len; i++) {
    materials_computed[i].courant = materials[i].sound_speed * dt / dx;
    materials_computed[i].courant_squared =
        materials_computed[i].courant * materials_computed[i].courant;
    materials_computed[i].acoustic_impedance_doubled =
        2 * materials[i].acoustic_impedance;
  }

  for (int x = 0; x < world->size_x; x++) {
    for (int y = 0; y < world->size_y; y++) {
      for (int z = 0; z < world->size_z; z++) {
        LAYER_0(world, x, y, z) = LAYER_1(world, x, y, z) =
            LAYER_2(world, x, y, z) = 0;
        MATERIAL_ID(world, x, y, z) = 0;
      }
    }
  }
}

__global__ void world_update(const World *const world,
                             MaterialComputed *const materials_computed,
                             const uint8_t materials_len,
                             const int chunk_size_x, const int chunk_size_y,
                             const int chunk_size_z) {

  const int start_x = (threadIdx.x + blockIdx.x * blockDim.x) * chunk_size_x;
  const int end_x = start_x + chunk_size_x;

  const int start_y = (threadIdx.y + blockIdx.y * blockDim.y) * chunk_size_y;
  const int end_y = start_y + chunk_size_y;

  const int start_z = (threadIdx.z + blockIdx.z * blockDim.z) * chunk_size_z;
  const int end_z = start_z + chunk_size_z;

  if (end_x >= world->size_x || end_y >= world->size_y ||
      end_z >= world->size_z) {
    return;
  }

  float neighbours, beta, courant, courant_squared;
  int k;

  for (int z = start_z; z < end_z; z++)
    for (int y = start_y; y < end_y; y++)
      for (int x = start_x; x < end_x; x++) {
        neighbours = k = 0;

        if (x != 0) {
          neighbours += LAYER_1(world, (int)(x - 1), (int)y, (int)z);
          k++;
        }
        if (y != 0) {
          neighbours += LAYER_1(world, (int)x, (int)(y - 1), (int)z);
          k++;
        }
        if (z != 0) {
          neighbours += LAYER_1(world, (int)x, (int)y, (int)(z - 1));
          k++;
        }
        if (x != world->size_x - 1) {
          neighbours += LAYER_1(world, (int)(x + 1), (int)y, (int)z);
          k++;
        }
        if (y != world->size_y - 1) {
          neighbours += LAYER_1(world, (int)x, (int)(y + 1), (int)z);
          k++;
        }
        if (z != world->size_z - 1) {
          neighbours += LAYER_1(world, (int)x, (int)y, (int)(z + 1));
          k++;
        }

        beta = (6 - k) / materials_computed[MATERIAL_ID(world, x, y, z)]
                             .acoustic_impedance_doubled;

        courant = materials_computed[MATERIAL_ID(world, x, y, z)].courant;
        courant_squared =
            materials_computed[MATERIAL_ID(world, x, y, z)].courant_squared;

        LAYER_0(world, x, y, z) = courant_squared * neighbours;

        LAYER_0(world, x, y, z) +=
            (2 - k * courant_squared) * LAYER_1(world, x, y, z);

        LAYER_0(world, x, y, z) +=
            (materials_computed[MATERIAL_ID(world, x, y, z)].courant * beta -
             1) *
            LAYER_2(world, x, y, z);

        LAYER_0(world, x, y, z) /= 1 + courant * beta;
      }
}

int main() {
  const int num_iterations = 1000;

  const int world_size_x = 500, world_size_y = 500, world_size_z = 500;

  const int block_size_x = 10, block_size_y = 10, block_size_z = 10;
  dim3 block_size(block_size_x, block_size_y, block_size_z);

  const int grid_size_x = 50, grid_size_y = 50, grid_size_z = 50;
  dim3 grid_size(grid_size_x, grid_size_y, grid_size_z);

  const int thread_size_x = grid_size_x * block_size_x,
            thread_size_y = grid_size_y * block_size_y,
            thread_size_z = grid_size_z * block_size_z;

  const int chunk_size_x = world_size_x / thread_size_x,
            chunk_size_y = world_size_y / thread_size_y,
            chunk_size_z = world_size_z / thread_size_z;

  printf("[INFO] Started pre-processing.\n");
  auto pre_processing_start_time = std::chrono::high_resolution_clock::now();

  uint8_t materials_len = 1;
  auto materials_host = (Material *)malloc(materials_len * sizeof(Material));
  materials_host[0] = {.sound_speed = 343, .acoustic_impedance = 1};
  auto materials_computed_host =
      (MaterialComputed *)malloc(materials_len * sizeof(MaterialComputed));

  World *world_host =
      world_alloc(world_size_x, world_size_y, world_size_z, WORLD_MEM_HOST);

  world_pre_compute(world_host, 0.1, materials_host, materials_computed_host,
                    materials_len);

  World *world_device =
      world_alloc(world_size_x, world_size_y, world_size_z, WORLD_MEM_DEVICE);

  // initial impulse
  LAYER_0(world_host, world_size_x / 2, world_size_y / 2, world_size_z / 2) = 1;

  world_cpy(world_device, world_host, WORLD_CPY_HOST_TO_DEVICE);
  world_free(world_host, WORLD_MEM_HOST);

  MaterialComputed *materials_computed_device;
  cudaMalloc(&materials_computed_device,
             materials_len * sizeof(MaterialComputed));
  cudaMemcpy(materials_computed_device, materials_computed_host,
             materials_len * sizeof(MaterialComputed), cudaMemcpyHostToDevice);

  float **world_data_swap;
  cudaMalloc(&world_data_swap, sizeof(float *));

  auto pre_processing_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> pre_processing_total_time =
      pre_processing_end_time - pre_processing_start_time;
  printf("[INFO] Pre processing took %lf seconds.\n",
         pre_processing_total_time.count());

  printf("[INFO] Started processing.\n");
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; i++) {
    // Advance time axis
    cudaMemcpy(world_data_swap, &world_device->data_t2, sizeof(float *),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(&world_device->data_t2, &world_device->data_t1, sizeof(float *),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(&world_device->data_t1, &world_device->data_t0, sizeof(float *),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(&world_device->data_t0, world_data_swap, sizeof(float *),
               cudaMemcpyDeviceToDevice);

    // simulate
    world_update<<<grid_size, block_size>>>(
        world_device, materials_computed_device, materials_len, chunk_size_x,
        chunk_size_y, chunk_size_z);
    cudaDeviceSynchronize();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_time = end_time - start_time;
  printf("[INFO] Processing took %lf seconds.\n", total_time.count());

  // move world back to host
  world_host =
      world_alloc(world_size_x, world_size_y, world_size_z, WORLD_MEM_HOST);
  world_cpy(world_host, world_device, WORLD_CPY_DEVICE_TO_HOST);
  world_free(world_device, WORLD_MEM_DEVICE);
  cudaFree(materials_computed_device);

  // save slice to file
  int z_slice = world_size_z / 2;
  std::ofstream outfile("z_slice_output.txt");

  if (outfile.is_open()) {
    for (int y = 0; y < world_size_y; y++) {
      for (int x = 0; x < world_size_x; x++) {
        outfile << LAYER_0(world_host, x, y, z_slice);
        if (x < world_size_x - 1)
          outfile << ",";
      }
      outfile << "\n";
    }
    outfile.close();
    printf("[INFO] Z-slice %d saved to z_slice_output.txt.\n", z_slice);
  } else {
    printf("[ERROR] Unable to open file for writing.\n");
  }

  world_free(world_host, WORLD_MEM_HOST);
}
