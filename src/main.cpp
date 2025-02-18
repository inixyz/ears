#include "vec.hpp"
#include "world.cuh"

#include <chrono>
#include <iostream>

int main() {
  World world(Vec3i(8, 8, 8), 0.1);

  world.materials[0] = {343, 0.5};
  world.compute_material_attributes();

  world.set_t0(Vec3i(3, 3, 3), 1);

  auto start_time = std::chrono::high_resolution_clock::now();

  world.step(1000);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_time = end_time - start_time;
  printf("[INFO] Processing took %lf seconds.\n", total_time.count());

  for (int z = 0; z < world.get_size().z; z++) {
    std::cout << " z=" << z << std::endl;
    for (int y = 0; y < world.get_size().y; y++) {
      for (int x = 0; x < world.get_size().x; x++)
        printf("%.1f ", world.get_t0(Vec3i(x, y, z)));
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
