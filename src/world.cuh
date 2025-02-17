#pragma once

#include "vec.hpp"
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
    uint8_t *material_ids{};
    float *t0{}, *t1{}, *t2{};
  };

  static constexpr int NO_MATERIALS = 256;
  std::array<Material, NO_MATERIALS> materials;

  World(const Vec3<int> &size, const float grid_spacing_distance);
  ~World();
  void compute_material_attributes() const;

private:
  // data on host
  const Vec3<int> size;
  const int size_slice, size_grid;
  const float grid_spacing_distance;

  // data on device
  MaterialAttributes material_attributes;
  Grid grid;
};
