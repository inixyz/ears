#include "world.cuh"
#include <algorithm>
#include <iostream>
#include <raylib.h>
#include <sndfile.h>
#include <vector>

Color map_val_to_color(const float val, const float min, const float max) {
  const float clamped_val = std::clamp(val, min, max);

  unsigned char r, g, b;

  if (clamped_val < 0) {
    // Interpolate from blue (-1) to white (0)
    float t = (clamped_val + 1.0f) / 1.0f; // Normalize to [0,1]
    r = static_cast<unsigned char>(t * 255);
    g = static_cast<unsigned char>(t * 255);
    b = 255;
  } else {
    // Interpolate from white (0) to green (1)
    float t = clamped_val / 1.0f; // Normalize to [0,1]
    r = static_cast<unsigned char>((1.0f - t) * 255);
    g = 255;
    b = static_cast<unsigned char>((1.0f - t) * 255);
  }

  return Color{r, g, b, 255};
}

const int PIXEL_SIZE = 3;

void draw_world(const World &world) {
  const Vec3i size = world.get_size();

  for (int x = 0; x < size.x; x++)
    for (int y = 0; y < size.y; y++) {
      const Color color = map_val_to_color(world.get_t0(Vec3i(x, y, 250)), -1, 1);
      DrawRectangle(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, color);
    }
}

void save_wav(const char *filename, const std::vector<float> &signal, int sample_rate) {
  SF_INFO sfinfo;
  sfinfo.samplerate = sample_rate;
  sfinfo.channels = 1;
  sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

  SNDFILE *outfile = sf_open(filename, SFM_WRITE, &sfinfo);
  if (!outfile) {
    std::cerr << "Error: Could not open WAV file for writing.\n";
    return;
  }

  sf_write_float(outfile, signal.data(), signal.size());
  sf_close(outfile);
  std::cout << "Saved WAV file: " << filename << std::endl;
}

std::vector<float> saved_signal;

int main() {
  // initial conditions
  World world(Vec3i(500, 100, 500), 0.01, Vec3i(50, 10, 50), Vec3i(10, 10, 10));

  world.materials[0] = {343, 1};
  world.materials[1] = {3500, 2500};
  world.compute_material_attributes();

  for (int x = 0; x < world.get_size().x; x++)
    for (int z = 0; z < world.get_size().z; z++)
      world.set_material_id(Vec3i(x, 0, z), 1);

  world.set_t0(Vec3i(125, 50, 250), 200);

  // raylib start
  // SetTraceLogLevel(LOG_WARNING);
  // InitWindow(world.get_size().x * PIXEL_SIZE, world.get_size().y * PIXEL_SIZE, "app");
  // SetTargetFPS(60);
  //
  // while (!WindowShouldClose()) {
  //   world.step();
  //   saved_signal.push_back(world.get_t0(Vec3i(250, 50, 250)));
  //
  //   BeginDrawing();
  //   ClearBackground(GRAY);
  //
  //   draw_world(world);
  //   DrawFPS(50, 50);
  //   EndDrawing();
  // }

  for (int i = 0; i < 10000; i++) {
    world.step();
    saved_signal.push_back(world.get_t0(Vec3i(250, 50, 250)));
  }

  std::cout << "Simulation stopped after: " << saved_signal.size() << " steps." << std::endl;
  save_wav("samples/old_formula_with_reflecting_floor.wav", saved_signal,
           int(1.0f / world.get_spacing_temporal()));

  // CloseWindow();
}
