#pragma once

#include <array>
#include <cstdint>

class World {
public:
  enum Device { CPU, GPU };

  struct Material {
    float sound_speed{}, acoustic_impedance{};
  };

  static constexpr int NO_MATERIALS = 256;
  std::array<Material, NO_MATERIALS> materials;

  World(const Device device, const int size_x, const int size_y, const int size_z);
  ~World();

private:
  Device device;
  int size_x, size_y, size_z, size_slice, size_data;

  float *courants, *courants_squared, *acoustic_impedances_doubled;
  uint8_t *material_ids;
  float *data_t0, *data_t1, *data_t2;
};
