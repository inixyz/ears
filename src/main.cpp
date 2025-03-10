#include "imgui.h"
#include "implot.h"
#include "rlImGui.h"
#include "world.cuh"
#include <algorithm>
#include <functional>
#include <raylib.h>
#include <vector>

std::vector<float> saved_signal;

void heatmap(const char *const label, const World &world, const int pos_y) {
  const int size_x = world.get_size().x;
  const int size_z = world.get_size().z;

  float grid[size_x][size_z];

  for (int x = 0; x < size_x; x++)
    for (int z = 0; z < size_z; z++)
      grid[x][z] = world.get_t0(Vec3i(x, pos_y, z));

  static ImPlotAxisFlags axes_flags =
      ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;

  ImGui::Begin(label);

  static char format[20] = "%g";
  ImGui::InputText("format:", format, 20);

  ImPlot::PushColormap(ImPlotColormap_Viridis);
  if (ImPlot::BeginPlot(label, ImVec2(-1, -1))) {
    ImPlot::PlotHeatmap(label, grid[0], size_x, size_z, -1, 1, format);
    ImPlot::EndPlot();
  }
  ImPlot::PopColormap();
  ImGui::End();
}

void menu(World &world) {
  ImGui::Begin("control");
  if (ImGui::Button("step")) {
    world.step();
    saved_signal.push_back(world.get_t0(Vec3i(10, 10, 25)));
  }
  ImGui::End();

  // heatmap("y=16", world, 25);

  ImGui::Begin("saved signal");
  if (ImPlot::BeginPlot("saved signal")) {
    ImPlot::PlotLine("##saved_signal", saved_signal.data(), saved_signal.size());
    ImPlot::EndPlot();
  }
  ImGui::End();
}

Color map_val_to_color(const float val, const float min, const float max) {
  const float normalized_val = (std::clamp(val, min, max) - min) / (max - min);
  return ColorFromHSV(360 * normalized_val, 1, 1);
}

enum SliceAxis { X, Y, Z };

void draw_world_section(const World &world, const SliceAxis slice_axis, const int slice_idx) {
  auto t0_slice_x = [&](const int i, const int j, const int slice_idx) {
    return world.get_t0(Vec3i(slice_idx, i, j));
  };
  auto t0_slice_y = [&](const int i, const int j, const int slice_idx) {
    return world.get_t0(Vec3i(i, slice_idx, j));
  };
  auto t0_slice_z = [&](const int i, const int j, const int slice_idx) {
    return world.get_t0(Vec3i(i, j, slice_idx));
  };

  std::function<float(int, int, int)> t0_slice;
  int size_i, size_j;

  switch (slice_axis) {
  case SliceAxis::X:
    t0_slice = t0_slice_x;
    size_i = world.get_size().y;
    size_j = world.get_size().z;
    break;
  case SliceAxis::Y:
    t0_slice = t0_slice_y;
    size_i = world.get_size().x;
    size_j = world.get_size().z;
    break;
  case SliceAxis::Z:
    t0_slice = t0_slice_z;
    size_i = world.get_size().x;
    size_j = world.get_size().y;
    break;
  }

  for (int i = 0; i < size_i; i++)
    for (int j = 0; j < size_j; j++) {
      const Color color = map_val_to_color(t0_slice(i, j, slice_idx), -1, 1);
      DrawRectangle(i * 4, j * 4, 4, 4, color);
      // DrawPixel(i, j, color);
    }
}

int main() {
  // init raylib and window
  SetTraceLogLevel(LOG_WARNING);
  InitWindow(1000, 800, "app");

  // init dear imgui
  rlImGuiSetup(true);
  ImPlot::CreateContext();
  constexpr float menu_scale_factor = 1;
  ImGui::GetIO().FontGlobalScale = menu_scale_factor;
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui::GetStyle().ScaleAllSizes(menu_scale_factor);

  // initial state
  World world(Vec3i(50, 50, 50), 0.1, Vec3i(5, 5, 5), Vec3i(10, 10, 10));
  world.materials[0] = {343, 15};

  world.compute_material_attributes();
  world.set_t0(Vec3i(25, 25, 25), 10);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(GRAY);

    rlImGuiBegin();
    menu(world);
    rlImGuiEnd();

    draw_world_section(world, SliceAxis::Y, 25);
    DrawFPS(50, 50);
    EndDrawing();
  }

  ImPlot::DestroyContext();
  rlImGuiShutdown();
  CloseWindow();
}
