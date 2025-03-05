#include "world.cuh"
#include <iostream>
#include <sndfile.h> // libsndfile for saving as WAV
#include <vector>

std::vector<float> saved_signal;

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

int main() {
  // Define simulation parameters
  Vec3i world_size(200, 50, 200);
  World world(world_size, 0.05, Vec3i(20, 5, 20), Vec3i(10, 10, 10));
  world.materials[0] = {343, 1};  // Air properties
  world.materials[1] = {3430, 1}; // Air properties
  world.compute_material_attributes();

  for (int x = 0; x < world.get_size().x; x++)
    for (int z = 0; z < world.get_size().z; z++)
      world.set_material_id(Vec3i(x, 0, z), 1);

  // Define impulse source and receiver (mirrored across the floor)
  Vec3i source_pos(150, 25, 25);
  Vec3i receiver_pos(80, 25, 25);
  world.set_t0(source_pos, 1); // Set impulse

  std::cout << "Simulating...\n";

  int num_steps = 30000;
  for (int step = 0; step < num_steps; step++) {
    world.step();

    // for (int x = 0; x < world.get_size().x; x++)
    //   for (int z = 0; z < world.get_size().z; z++)
    //     world.set_t0(Vec3i(x, 0, z), 0);

    saved_signal.push_back(world.get_t0(receiver_pos));

    if (step % 1000 == 0) {
      std::cout << "Step " << step << " / " << num_steps << std::endl;
    }
  }

  std::cout << "Simulation complete.\n";
  save_wav("samples/RIR_sim.wav", saved_signal, int(1.0f / world.get_spacing_temporal()));

  return 0;
}
