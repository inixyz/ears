#pragma once

#include <cstdint>

__global__ void fdtd_step(const int size_x, const int size_y, const int size_z,
                          const int size_slice, const uint8_t *const material_ids,
                          float *const data_t0, const float *const data_t1,
                          const float *const data_t2, const float *const courants,
                          const float *const courants_squared,
                          const float *const acoustic_impedances_doubled);
