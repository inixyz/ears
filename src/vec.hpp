#pragma once

template <typename T>
struct Vec3 {
  T x{}, y{}, z{};

  Vec3(const T &x, const T &y, const T &z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }
};
