#include "world.cuh"
#include <raylib.h>

void draw_world(const World &world) {
}

int main() {
  SetTraceLogLevel(LOG_WARNING);
  InitWindow(1000, 1000, "app");
  SetTargetFPS(60);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(GRAY);

    DrawFPS(50, 50);
    EndDrawing();
  }

  CloseWindow();
}
