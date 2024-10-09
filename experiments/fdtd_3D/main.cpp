#include <cmath>
#include <memory.h>
#include <raylib.h>
#include <raymath.h>

const int WINDOW_SIZE_X = 800, WINDOW_SIZE_Y = 500;
const int WORLD_SIZE_X = 32, WORLD_SIZE_Y = 32, WORLD_SIZE_Z = 32;
float world[3][WORLD_SIZE_X][WORLD_SIZE_Y][WORLD_SIZE_Z];

// Simulation parameters
const float c = 343.0f;               // Speed of sound in air (m/s)
const float dx = 0.01f;               // Grid spacing (meters)
const float dt = 0.5f * dx / c;       // Time step (based on CFL condition)
const float air_loss_factor = 0.001f; // Damping due to air losses
const float reflection_coefficient =
    0.1f; // Reflection coefficient for walls (0 < R <= 1)

// Draw the 3D world (wave propagation visualization)
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

// FDTD 3D wave propagation simulation with air losses and wall reflections
void simulate() {
  static const int world_size =
      WORLD_SIZE_X * WORLD_SIZE_Y * WORLD_SIZE_Z * sizeof(float);

  // Shift pressure values through time (t -> t-1, t-1 -> t-2)
  for (int time_iter = 2; time_iter > 0; time_iter--) {
    memcpy(world[time_iter], world[time_iter - 1], world_size);
  }

  // FDTD update for each interior point in the 3D grid with air losses
  for (int x = 1; x < WORLD_SIZE_X - 1; x++) {
    for (int y = 1; y < WORLD_SIZE_Y - 1; y++) {
      for (int z = 1; z < WORLD_SIZE_Z - 1; z++) {
        world[0][x][y][z] = (2.0f - air_loss_factor) * world[1][x][y][z] -
                            (1.0f - air_loss_factor / 2) * world[2][x][y][z] +
                            (c * c * dt * dt / (dx * dx)) *
                                (world[1][x + 1][y][z] + world[1][x - 1][y][z] +
                                 world[1][x][y + 1][z] + world[1][x][y - 1][z] +
                                 world[1][x][y][z + 1] + world[1][x][y][z - 1] -
                                 6.0f * world[1][x][y][z]);
      }
    }
  }

  // Reflective boundaries (simulate partial reflection with absorption)
  for (int x = 0; x < WORLD_SIZE_X; x++) {
    for (int y = 0; y < WORLD_SIZE_Y; y++) {
      // Z boundaries
      world[0][x][y][0] =
          reflection_coefficient * world[1][x][y][1]; // Front boundary
      world[0][x][y][WORLD_SIZE_Z - 1] =
          reflection_coefficient *
          world[1][x][y][WORLD_SIZE_Z - 2]; // Back boundary
    }
  }

  for (int x = 0; x < WORLD_SIZE_X; x++) {
    for (int z = 0; z < WORLD_SIZE_Z; z++) {
      // Y boundaries
      world[0][x][0][z] =
          reflection_coefficient * world[1][x][1][z]; // Bottom boundary
      world[0][x][WORLD_SIZE_Y - 1][z] =
          reflection_coefficient *
          world[1][x][WORLD_SIZE_Y - 2][z]; // Top boundary
    }
  }

  for (int y = 0; y < WORLD_SIZE_Y; y++) {
    for (int z = 0; z < WORLD_SIZE_Z; z++) {
      // X boundaries
      world[0][0][y][z] =
          reflection_coefficient * world[1][1][y][z]; // Left boundary
      world[0][WORLD_SIZE_X - 1][y][z] =
          reflection_coefficient *
          world[1][WORLD_SIZE_X - 2][y][z]; // Right boundary
    }
  }
}

int main() {
  InitWindow(WINDOW_SIZE_X, WINDOW_SIZE_Y,
             "FDTD 3D Wave Simulation with Air Losses and Wall Reflections");
  SetTargetFPS(60);

  Camera camera = {0};
  camera.position = {16, 40, 16};
  camera.target = {0, 0, 0};
  camera.up = {0, 1, 0};
  camera.fovy = 60;
  camera.projection = CAMERA_PERSPECTIVE;

  // Initialize source in the center of the world
  int x_start = 16, y_start = 16, z_start = 16;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        world[0][x + x_start][y + y_start][z + z_start] = 1.0f;
      }
    }
  }

  while (!WindowShouldClose()) {
    UpdateCamera(&camera, CAMERA_FREE);
    // Check for alt + enter to toggle fullscreen
    if (IsKeyPressed(KEY_ENTER) &&
        (IsKeyDown(KEY_LEFT_ALT) || IsKeyDown(KEY_RIGHT_ALT))) {
      ToggleFullscreen();
    }

    simulate(); // Run the FDTD wave simulation

    BeginDrawing();
    ClearBackground(BLACK);

    BeginMode3D(camera);
    draw_world(); // Render the 3D wave field
    EndMode3D();

    DrawFPS(30, 30);

    EndDrawing();
  }

  CloseWindow(); // Close the window and OpenGL context
}
