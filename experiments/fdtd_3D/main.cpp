#include <cmath>
#include <memory.h>
#include <raylib.h>
#include <raymath.h>

const int WINDOW_SIZE_X = 800, WINDOW_SIZE_Y = 500;

const int WORLD_SIZE_X = 32, WORLD_SIZE_Y = 32, WORLD_SIZE_Z = 32;
float world[3][WORLD_SIZE_X][WORLD_SIZE_Y][WORLD_SIZE_Z];

void draw_world() {
  for (int x = 0; x < WORLD_SIZE_X; x++) {
    for (int y = 0; y < WORLD_SIZE_Y; y++) {
      for (int z = 0; z < WORLD_SIZE_Z; z++) {
        const float amplitude = world[0][x][y][z];
        const unsigned char alpha = Remap(std::abs(amplitude), -1, 1, 0, 255);
        Color color = {0, 0, 0, 0};
        if (amplitude > 0)
          color = {0, 255, 0, alpha};
        else if (amplitude < 0)
          color = {255, 0, 0, alpha};

        DrawCube({(float)x, (float)y, (float)z}, 1, 1, 1, color);
      }
    }
  }

  DrawCubeWires({WORLD_SIZE_X / 2, WORLD_SIZE_Y / 2, WORLD_SIZE_Z / 2},
                WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z, WHITE);
}

// void rigid_update(int x, int y, int z) {
//   world[0][x][y][z] = 0.33 * (world[1][x + 1][y][z] + world[1][x - 1][y][z] +
//                               world[1][x][y + 1][z] + world[1][x][y - 1][z] +
//                               world[1][x][y][z + 1] + world[1][x][y][z - 1])
//                               -
//                       world[2][x][y][z];
// }

void simulate() {
  static const int world_size =
      WORLD_SIZE_X * WORLD_SIZE_Y * WORLD_SIZE_Z * sizeof(float);

  for (int time_iter = 2; time_iter > 0; time_iter--) {
    memcpy(world[time_iter], world[time_iter - 1], world_size);
  }

  for (int x = 1; x < WORLD_SIZE_X - 1; x++) {
    for (int y = 1; y < WORLD_SIZE_Y - 1; y++) {
      for (int z = 1; z < WORLD_SIZE_Z - 1; z++) {
        // rigid_update(x, y, z);
        // PLACE UPDATE CODE HERE MAYBE
      }
    }
  }
}

int main() {
  InitWindow(WINDOW_SIZE_X, WINDOW_SIZE_Y, "FDTD 3D");
  SetTargetFPS(60);
  // DisableCursor();

  Camera camera = {0};
  camera.position = {16, 40, 16};
  camera.target = {0, 0, 0};
  camera.up = {0, 1, 0};
  camera.fovy = 60;
  camera.projection = CAMERA_PERSPECTIVE;

  int x_start = 16, y_start = 16, z_start = 16;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        world[0][x + x_start][y + y_start][z + z_start] = 1;
      }
    }
  }

  while (!WindowShouldClose()) {
    UpdateCamera(&camera, CAMERA_FREE);
    // check for alt + enter
    if (IsKeyPressed(KEY_ENTER) &&
        (IsKeyDown(KEY_LEFT_ALT) || IsKeyDown(KEY_RIGHT_ALT))) {
      ToggleFullscreen();
    }

    simulate();

    BeginDrawing();
    ClearBackground(BLACK);

    BeginMode3D(camera);
    // DrawGrid(20, 5);
    draw_world();
    EndMode3D();

    DrawFPS(30, 30);

    EndDrawing();
  }
}
