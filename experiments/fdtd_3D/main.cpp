#include <raylib.h>
#include <stdint.h>

const int WORLD_SIZE_X = 5, WORLD_SIZE_Y = 5, WORLD_SIZE_Z = 5;
uint8_t world[WORLD_SIZE_X][WORLD_SIZE_Y][WORLD_SIZE_Z];

void draw_world() {
  for (int x = 0; x < WORLD_SIZE_X; x++) {
    for (int y = 0; y < WORLD_SIZE_Y; y++) {
      for (int z = 0; z < WORLD_SIZE_Z; z++) {
        DrawCubeWires({x, y, z}, 1, 1, 1, RED);
      }
    }
  }
}

const int WINDOW_SIZE_X = 1600, WINDOW_SIZE_Y = 900;
int main() {
  InitWindow(WINDOW_SIZE_X, WINDOW_SIZE_Y, "FDTD 3D");
  SetTargetFPS(60);
  // DisableCursor();

  Camera camera = {0};
  camera.position = {10, 10, 10};
  camera.target = {0, 0, 0};
  camera.up = {0, 1, 0};
  camera.fovy = 60;
  camera.projection = CAMERA_PERSPECTIVE;

  while (!WindowShouldClose()) {
    UpdateCamera(&camera, CAMERA_FREE);

    BeginDrawing();
    ClearBackground(BLACK);

    BeginMode3D(camera);
    DrawGrid(20, 5);
    draw_world();
    EndMode3D();

    DrawFPS(30, 30);

    EndDrawing();
  }
}
