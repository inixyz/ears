#pragma once

template <typename T>
class Vec3 {
public:
  T x, y, z;

  Vec3(const T &x, const T &y, const T &z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }
};
