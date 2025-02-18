#include "vec.hpp"
#include "world.cuh"

#include <iostream>

int main() {
  World world(Vec3<int>(8, 8, 8), 0.1);

  world.compute_material_attributes();
  int test;
  std::cin >> test;
}
