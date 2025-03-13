#include "world.cuh"
#include <algorithm>
#include <raylib.h>

Color map_val_to_color(const float val, const float min, const float max) {
  const float clamped_val = std::clamp(val, min, max);

  if (clamped_val >= 0)
    return Color{0, static_cast<unsigned char>(clamped_val * 255), 0, 255};
  else {
    return Color{0, 0, static_cast<unsigned char>(-clamped_val * 255), 255};
  }
}

const int PIXEL_SIZE = 32;

void draw_world(const World &world) {
  const Vec3i size = world.get_size();

  for (int x = 0; x < size.x; x++)
    for (int y = 0; y < size.y; y++) {
      const Color color = map_val_to_color(world.get_t0(Vec3i(x, y, 50)), -1, 1);
      DrawRectangle(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, color);
    }
}

int main() {
  // initial conditions
  World world(Vec3i(100, 50, 100), 0.1, Vec3i(10, 5, 10), Vec3i(10, 10, 10));

  world.materials[0] = {343, 1};
  // world.materials[1] = {3500, 1};
  world.compute_material_attributes();

  // for (int x = 0; x < world.get_size().x; x++)
  //   for (int z = 0; z < world.get_size().z; z++)
  //     world.set_material_id(Vec3i(x, 0, z), 1);

  world.set_t0(Vec3i(50, 25, 50), 100);

  // raylib start
  SetTraceLogLevel(LOG_WARNING);
  InitWindow(world.get_size().x * PIXEL_SIZE, world.get_size().y * PIXEL_SIZE, "app");
  SetTargetFPS(60);

  while (!WindowShouldClose()) {
    world.step();

    BeginDrawing();
    ClearBackground(GRAY);

    draw_world(world);
    DrawFPS(50, 50);
    EndDrawing();
  }

  CloseWindow();
}
