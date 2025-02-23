#include "imgui.h"
#include "rlImGui.h"
#include "world.cuh"
#include <raylib.h>

void menu(World &world) {
  ImGui::Begin("boss");
  if (ImGui::Button("Step"))
    world.step();
  if (ImGui::Button("ZERO"))
    world.set_t0(Vec3i(16, 16, 16), 0);
  ImGui::End();
}

void draw_world(World &world) {
  const Vec3i size = world.get_size();
  const float spacing_distance = world.get_spacing_distance();
  const Vector3 cube_size = {spacing_distance, spacing_distance, spacing_distance};

  int y = 16;
  for (int x = 0; x < size.x; x++)
    // for (int y = 0; y < size.y; y++)
    for (int z = 0; z < size.z; z++) {
      const Vector3 pos = {x * spacing_distance, y * spacing_distance, z * spacing_distance};
      const float val = world.get_t0(Vec3i(x, y, z));
      if (val == 0)
        continue;
      const Color color = {0, static_cast<unsigned char>(val * 127 + 127), 0, 255};
      DrawCubeV(pos, cube_size, color);
    }
}

int main() {
  SetTraceLogLevel(LOG_WARNING);
  InitWindow(2000, 2000, "app");

  rlImGuiSetup(false);
  constexpr float menu_scale_factor = 1.5;
  ImGui::GetIO().FontGlobalScale = menu_scale_factor;
  ImGui::GetStyle().ScaleAllSizes(menu_scale_factor);

  Camera3D camera{};
  camera.position = {10, 10, 10};
  camera.target = {0, 0, 0};
  camera.up = {0, 1, 0};
  camera.fovy = 60;
  camera.projection = CAMERA_PERSPECTIVE;

  // initial state
  World world(Vec3i(32, 32, 32), 0.1, Vec3i(4, 4, 4), Vec3i(8, 8, 8));
  world.materials[0] = {343, 0.5};

  world.compute_material_attributes();
  world.set_t0(Vec3i(16, 16, 16), 1);

  while (!WindowShouldClose()) {
    if (IsWindowFocused() && IsMouseButtonDown(MOUSE_LEFT_BUTTON))
      UpdateCamera(&camera, CAMERA_FREE);

    BeginDrawing();
    ClearBackground(GRAY);

    BeginMode3D(camera);
    draw_world(world);
    EndMode3D();

    rlImGuiBegin();
    menu(world);
    rlImGuiEnd();

    DrawFPS(50, 50);
    EndDrawing();
  }

  rlImGuiShutdown();
  CloseWindow();
}
