#pragma once

#include "vec.cuh"

namespace ears {

class World {
public:
  World(const Vec3i &size, const float courant, const dim3 dim_grid, const dim3 dim_block);
  ~World();

  const Vec3i &get_size() const;
  float get_courant() const;
  float get_t0(const Vec3i &pos) const;
  float get_t1(const Vec3i &pos) const;
  float get_t2(const Vec3i &pos) const;

  void set_t0(const Vec3i &pos, const float val) const;
  void set_t1(const Vec3i &pos, const float val) const;
  void set_t2(const Vec3i &pos, const float val) const;

  void step();
  void step(const int no_iter);

private:
  const Vec3i size;
  const int size_xy, size_xyz;
  const float courant;
  const dim3 dim_grid, dim_block;
  float *t0 = nullptr, *t1 = nullptr, *t2 = nullptr;
};

} // namespace ears
