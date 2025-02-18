#pragma once

#include "vec.hpp"
#include "world.cuh"

__global__ void fdtd_step(const Vec3i size, const int size_slice,
                          const World::MaterialAttributes material_attributes,
                          const World::Grid grid);
