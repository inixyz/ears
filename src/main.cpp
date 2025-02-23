#include "imgui.h"
#include "rlImGui.h"
#include <raylib.h>

void menu() {
  ImGui::Begin("boss");
  ImGui::Text("Hello, World!");
  ImGui::End();
}

int main() {
  SetTraceLogLevel(LOG_WARNING);
  InitWindow(2000, 2000, "app");
  rlImGuiSetup(false);
  constexpr float menu_scale_factor = 1.5;
  ImGui::GetIO().FontGlobalScale = menu_scale_factor;
  ImGui::GetStyle().ScaleAllSizes(menu_scale_factor);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    rlImGuiBegin();
    menu();
    rlImGuiEnd();

    DrawFPS(50, 50);
    EndDrawing();
  }

  rlImGuiShutdown();
  CloseWindow();
}
