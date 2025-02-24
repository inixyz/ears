#include "imgui.h"
#include "implot.h"
#include "rlImGui.h"
#include "world.cuh"
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
    world.step(1000000);
    saved_signal.push_back(world.get_t0(Vec3i(10, 10, 25)));
  }
  ImGui::End();

  heatmap("y=16", world, 25);

  ImGui::Begin("saved signal");
  if (ImPlot::BeginPlot("saved signal")) {
    ImPlot::PlotLine("##saved_signal", saved_signal.data(), saved_signal.size());
    ImPlot::EndPlot();
  }
  ImGui::End();
}

int main() {
  // init raylib and window
  SetTraceLogLevel(LOG_WARNING);
  InitWindow(1000, 1000, "app");

  // init dear imgui
  rlImGuiSetup(true);
  ImPlot::CreateContext();
  constexpr float menu_scale_factor = 1;
  ImGui::GetIO().FontGlobalScale = menu_scale_factor;
  ImGui::GetStyle().ScaleAllSizes(menu_scale_factor);

  // initial state
  World world(Vec3i(50, 50, 50), 0.1, Vec3i(5, 5, 5), Vec3i(10, 10, 10));
  world.materials[0] = {343, 1};

  world.compute_material_attributes();
  world.set_t0(Vec3i(25, 25, 25), 100);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(GRAY);

    rlImGuiBegin();
    menu(world);
    rlImGuiEnd();

    DrawFPS(50, 50);
    EndDrawing();
  }

  ImPlot::DestroyContext();
  rlImGuiShutdown();
  CloseWindow();
}
