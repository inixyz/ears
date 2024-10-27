#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NX 100          // Number of grid points in the x direction
#define NY 100          // Number of grid points in the y direction
#define NZ 100          // Number of grid points in the z direction
#define TIME_STEPS 2000 // Number of time steps in the simulation
#define SOUND_SPEED 343 // Speed of sound in m/s
#define DX 0.01         // Grid spacing in meters
#define DT (DX / SOUND_SPEED / sqrt(3)) // Time step size (CFL condition)

// Define a 3D grid to store pressure values at each time step
double pressure[NX][NY][NZ];
double velocity_x[NX][NY][NZ];
double velocity_y[NX][NY][NZ];
double velocity_z[NX][NY][NZ];

// Function to initialize the grid
void initialize() {
  for (int i = 0; i < NX; i++) {
    for (int j = 0; j < NY; j++) {
      for (int k = 0; k < NZ; k++) {
        pressure[i][j][k] = 0.0;
        velocity_x[i][j][k] = 0.0;
        velocity_y[i][j][k] = 0.0;
        velocity_z[i][j][k] = 0.0;
      }
    }
  }
}

// Apply the source signal at a specific point in the grid
void apply_source(int t) {
  int x0 = NX / 2;
  int y0 = NY / 2;
  int z0 = NZ / 2;
  // Example: simple sinusoidal source
  pressure[x0][y0][z0] += sin(2.0 * M_PI * 440 * t * DT); // 440 Hz tone
}

// Update the FDTD grid for one time step
void update_grid() {
  // Update velocity
  for (int i = 1; i < NX - 1; i++) {
    for (int j = 1; j < NY - 1; j++) {
      for (int k = 1; k < NZ - 1; k++) {
        velocity_x[i][j][k] +=
            DT / DX * (pressure[i + 1][j][k] - pressure[i][j][k]);
        velocity_y[i][j][k] +=
            DT / DX * (pressure[i][j + 1][k] - pressure[i][j][k]);
        velocity_z[i][j][k] +=
            DT / DX * (pressure[i][j][k + 1] - pressure[i][j][k]);
      }
    }
  }

  // Update pressure
  for (int i = 1; i < NX - 1; i++) {
    for (int j = 1; j < NY - 1; j++) {
      for (int k = 1; k < NZ - 1; k++) {
        pressure[i][j][k] -=
            DT / DX *
            (velocity_x[i][j][k] + velocity_y[i][j][k] + velocity_z[i][j][k]);
      }
    }
  }
}

// Main FDTD loop
int main(int argc, char *argv[]) {
  initialize();

  // Output file for the received signal
  FILE *output_file = fopen("received_signal.txt", "w");

  // Run the simulation
  for (int t = 0; t < TIME_STEPS; t++) {
    apply_source(t);
    update_grid();

    // Record the signal at a receiver point (e.g., at the corner of the room)
    fprintf(output_file, "%lf\n", pressure[1][1][1]);
  }

  fclose(output_file);
  printf("Simulation complete. Output saved to received_signal.txt.\n");
  return 0;
}
