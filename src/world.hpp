#pragma once

#include "vec.hpp"
#include <cstdint>

class World {
public:
  World(const Vec3<int> &size);
  ~World();
  void to_gpu();

private:
  const Vec3<int> size;
  const int slice_size, data_size;
  bool on_gpu = false;
  uint8_t *materials = nullptr;
  float *data_t0 = nullptr, *data_t1 = nullptr, *data_t2 = nullptr;
};
