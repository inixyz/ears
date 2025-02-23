#pragma once

#include "vec.cuh"
#include <array>
#include <cstdint>

class World {
public:
  struct Material {
    float sound_speed{}, acoustic_impedance{};
  };

  struct MaterialAttributes {
    float *courants{}, *courants_squared{}, *acoustic_impedances_doubled{};
  };

  struct Grid {
    uint8_t *material_id{};
    float *t0{}, *t1{}, *t2{};
  };

  static constexpr int NO_MATERIALS = 256;
  std::array<Material, NO_MATERIALS> materials;

  World(const Vec3i &size, const float spacing_distance, const dim3 dim_grid, const dim3 dim_block);
  ~World();

  const Vec3i &get_size() const;

  uint8_t get_material_id(const Vec3<int> &pos) const;
  float get_t0(const Vec3i &pos) const;
  float get_t1(const Vec3i &pos) const;
  float get_t2(const Vec3i &pos) const;

  void set_material_id(const Vec3<int> &pos, const uint8_t val) const;
  void set_t0(const Vec3i &pos, const float val) const;
  void set_t1(const Vec3i &pos, const float val) const;
  void set_t2(const Vec3i &pos, const float val) const;

  void compute_material_attributes();
  void step();
  void step(const int no_iterations);

private:
  // data on host
  const Vec3i size;
  const int size_slice, size_grid;

  const float spacing_distance;
  float spacing_temporal;

  const dim3 dim_grid, dim_block;

  // data on device
  MaterialAttributes material_attributes;
  Grid grid;
};
