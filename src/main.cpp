#include "world.hpp"

#include "vec.hpp"

int main() {
  World world(Vec3<int>(10, 10, 10));
  world.to_gpu();
}
