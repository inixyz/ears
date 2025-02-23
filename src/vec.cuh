#pragma once

#include <vector_types.h>

template <typename T>
struct Vec3 {
  T x{}, y{}, z{};

  Vec3(const T &x, const T &y, const T &z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  operator dim3() const {
    return dim3(x, y, z);
  }
};

using Vec3i = Vec3<int>;
