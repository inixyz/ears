#include "raylib.h"
#include "world.cuh"
#include <cmath>
#include <iostream>
#include <vector>

const int WINDOW_SIZE = 600;                     // Window dimensions
const int GRID_SIZE = 100;                       // FDTD grid resolution
const int PIXEL_SCALE = WINDOW_SIZE / GRID_SIZE; // Scale factor for visualization
const int SIM_STEPS_PER_FRAME = 5; // Run multiple steps per frame to make waves visible

void drawWave(const std::vector<float> &data, int width, int height) {
  float max_val = 0.1f; // Prevent tiny values from being invisible
  for (float val : data) {
    if (fabs(val) > max_val)
      max_val = fabs(val); // Find max amplitude
  }

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      float value = data[x * height + y] / max_val; // Normalize
      int colorVal = (int)(fabs(value) * 255);
      colorVal = colorVal > 255 ? 255 : colorVal; // Clamp values

      // Use a blue-red colormap for better contrast
      Color color = (value > 0) ? Color{colorVal, 0, 0, 255} : Color{0, 0, colorVal, 255};

      // Draw the pixel
      DrawRectangle(x * PIXEL_SCALE, y * PIXEL_SCALE, PIXEL_SCALE, PIXEL_SCALE, color);
    }
  }
}

int main() {
  // Simulation setup
  Vec3i size(GRID_SIZE, GRID_SIZE, GRID_SIZE);
  float courant = 0.5f;
  dim3 dim_grid(10, 10, 10);
  dim3 dim_block(10, 10, 10);

  // Create FDTD simulation world
  ears::World world(size, courant, dim_grid, dim_block);

  // Define impulse position and inject wave
  Vec3i impulse_pos(size.x / 2, size.y / 2, size.z / 2);
  world.set_t0(impulse_pos, 1.0f); // Initial impulse

  // Initialize Raylib
  InitWindow(WINDOW_SIZE, WINDOW_SIZE, "FDTD Acoustic Simulation");
  SetTargetFPS(60);

  std::vector<float> visualization(size.x * size.y, 0.0f); // Flattened 2D array

  // Main simulation loop
  while (!WindowShouldClose()) {
    // Run multiple steps per frame for better visualization
    for (int i = 0; i < SIM_STEPS_PER_FRAME; i++) {
      world.step();
    }

    // Extract 2D slice at z_slice = middle of 3D space
    int z_slice = size.z / 2;
    for (int x = 0; x < size.x; ++x) {
      for (int y = 0; y < size.y; ++y) {
        visualization[x * size.y + y] = world.get_t0(Vec3i(x, y, z_slice));
      }
    }

    // Render the wave propagation
    BeginDrawing();
    ClearBackground(BLACK);
    drawWave(visualization, size.x, size.y);
    DrawText("FDTD Wave Propagation", 10, 10, 20, WHITE);
    EndDrawing();
  }

  // Cleanup
  CloseWindow();
  return 0;
}
