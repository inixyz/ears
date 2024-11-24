#include <cmath>
#include <cstring>
#include <iostream>
#include <raylib.h>
#include <raymath.h>
#include <vector>

struct Cell {
  float p, courant;
};
// Global constants
const float WALL_IMPEDANCE = 200000; // rayl
// const float WALL_IMPEDANCE = 4'000;
// const float WALL_IMPEDANCE = 1'570'000;
const float WALL_DENSITY = 510;
const float AIR_SPEED = 343, MAT_SPEED = 1000;
const float EW = WALL_IMPEDANCE / (WALL_DENSITY * AIR_SPEED);
// const float EW = 1;
const float reflection = (EW - 1) / (EW + 1);

const float dx = 0.01;
const float dt = dx / (std::max(AIR_SPEED, MAT_SPEED) * sqrt(2));
// const float dx = dt * AIR_SPEED * sqrt(2);

const float AIR_COURANT = (dt * AIR_SPEED) / dx;
const float MAT_COURANT = (dt * MAT_SPEED) / dx;

const int world_width = 100, world_height = 100;
const int scale = 14;
Cell world[3][world_width][world_height];

// Signal types
const int source_type = 1;
const float impulse_value = 5;
const int source_x = world_width / 4, source_y = world_height / 2;
const int sinewave_freq = 4000;

// Receiver position and signal buffer
const int receiver_x = world_width * 3 / 4, receiver_y = world_height / 2;
const int max_signal_samples = 500;
std::vector<float> receiver_signal;

// State variables
bool is_paused = false;
long long timestep = 0;

// Graph dimensions
const int graph_width = 1500;
const int graph_height = world_height * scale;
const int graph_x_offset = world_width * scale;
const int graph_y_offset = (world_height * scale - graph_height) / 2;

// Function to draw the world
void draw_world() {
  Color color;
  const float color_intensify = 1;

  for (int x = 0; x < world_width; x++) {
    for (int y = 0; y < world_height; y++) {
      float p_norm = Remap(world[0][x][y].p, -1 * color_intensify,
                           1 * color_intensify, -128, 127);
      p_norm = fmax(fmin(p_norm, 127), -128);

      if (world[0][x][y].courant == AIR_COURANT) {
        color = Color{0, static_cast<unsigned char>(128 + p_norm),
                      static_cast<unsigned char>(128 - p_norm), 255};
      } else if (world[0][x][y].courant == MAT_COURANT) {
        color = Color{static_cast<unsigned char>(128 + p_norm), 0,
                      static_cast<unsigned char>(128 - p_norm), 255};
      }

      DrawRectangle(x * scale, y * scale, scale, scale, color);
    }
  }

  // Highlight the receiver position
  DrawRectangle(receiver_x * scale, receiver_y * scale, scale, scale, RED);
  DrawRectangle(source_x * scale, source_y * scale, scale, scale, RED);
}

// Function to draw the receiver signal graph with gradation lines
void draw_graph() {
  const float gain = 3;
  const float thickness = 3;

  // Draw graph background
  DrawRectangle(graph_x_offset, graph_y_offset, graph_width, graph_height,
                GRAY);

  // Draw X and Y axes
  DrawLine(graph_x_offset, graph_y_offset + graph_height / 2,
           graph_x_offset + graph_width, graph_y_offset + graph_height / 2,
           WHITE);
  DrawLine(graph_x_offset, graph_y_offset, graph_x_offset,
           graph_y_offset + graph_height, WHITE);

  // Gradation lines and labels
  int num_x_lines = 10; // Number of vertical lines (time divisions)
  int num_y_lines = 10; // Number of horizontal lines (amplitude divisions)
  float x_step = (float)graph_width / num_x_lines;
  float y_step = (float)graph_height / num_y_lines;

  // Y-axis gradations
  for (int i = 0; i <= num_y_lines; ++i) {
    float y = graph_y_offset + i * y_step;
    DrawLine(graph_x_offset, y, graph_x_offset + graph_width, y, DARKGRAY);
    float label =
        (0.5f - (float)i / num_y_lines) * 2 / gain; // Normalize amplitude
    DrawText(TextFormat("%.2f", label), graph_x_offset - 50, y - 10, 20, WHITE);
  }

  // X-axis gradations
  for (int i = 0; i <= num_x_lines; ++i) {
    float x = graph_x_offset + i * x_step;
    DrawLine(x, graph_y_offset, x, graph_y_offset + graph_height, DARKGRAY);

    // Calculate the timestep for this position
    int start_timestep =
        (timestep > max_signal_samples) ? (timestep - max_signal_samples) : 0;
    float label =
        (start_timestep + i * (max_signal_samples / num_x_lines)) * dt;
    DrawText(TextFormat("%.4f", label), x - 20,
             graph_y_offset + graph_height / 2 + 5, 20, WHITE);
  }

  // Plot receiver signal
  if (!receiver_signal.empty()) {
    for (size_t i = 1; i < receiver_signal.size(); ++i) {
      float x1 =
          graph_x_offset + (i - 1) * (graph_width / (float)max_signal_samples);
      float y1 = graph_y_offset + graph_height / 2 -
                 gain * receiver_signal[i - 1] * (graph_height / 2);
      float x2 = graph_x_offset + i * (graph_width / (float)max_signal_samples);
      float y2 = graph_y_offset + graph_height / 2 -
                 gain * receiver_signal[i] * (graph_height / 2);
      DrawLineEx(Vector2{x1, y1}, Vector2{x2, y2}, thickness, RED);
    }
  }
}

// Initialize the world
void init_world() {
  for (int x = 0; x < world_width; x++) {
    for (int y = 0; y < world_height; y++) {
      for (int t = 0; t < 3; t++) {
        world[t][x][y].p = 0;
        if (x >= world_width / 2) {
          world[t][x][y].courant = MAT_COURANT;
        } else {
          world[t][x][y].courant = AIR_COURANT;
        }
      }
    }
  }
}

void update_inner() {
  float N;

  for (int x = 1; x < world_width - 1; x++) {
    for (int y = 1; y < world_height - 1; y++) {
      N = world[1][x - 1][y].p + world[1][x + 1][y].p + world[1][x][y - 1].p +
          world[1][x][y + 1].p - 4 * world[1][x][y].p;
      world[0][x][y].p = world[1][x][y].courant * world[1][x][y].courant * N +
                         2 * world[1][x][y].p - world[2][x][y].p;
      world[0][x][y].p *= 1 - dx;
    }
  }
}

void update_right() {
  const int x = world_width - 1;
  for (int y = 1; y < world_height - 1; y++) {
    world[0][x][y].p = (world[1][x][y].courant * world[1][x][y].courant *
                            (world[2][x][y].p / (world[1][x][y].courant * EW) +
                             2 * world[1][x - 1][y].p + world[1][x][y + 1].p +
                             world[1][x][y - 1].p - 4 * world[1][x][y].p) +
                        2 * world[1][x][y].p - world[2][x][y].p) /
                       (1 + world[1][x][y].courant / EW);
  }
}

void update_left() {
  const int x = 0;
  for (int y = 1; y < world_height - 1; y++) {
    world[0][x][y].p = (world[1][x][y].courant * world[1][x][y].courant *
                            (world[2][x][y].p / (world[1][x][y].courant * EW) +
                             2 * world[1][x + 1][y].p + world[1][x][y + 1].p +
                             world[1][x][y - 1].p - 4 * world[1][x][y].p) +
                        2 * world[1][x][y].p - world[2][x][y].p) /
                       (1 + world[1][x][y].courant / EW);
  }
}

void update_top() {
  const int y = 0;
  for (int x = 1; x < world_width - 1; x++) {
    world[0][x][y].p = (world[1][x][y].courant * world[1][x][y].courant *
                            (world[2][x][y].p / (world[1][x][y].courant * EW) +
                             2 * world[1][x][y + 1].p + world[1][x + 1][y].p +
                             world[1][x - 1][y].p - 4 * world[1][x][y].p) +
                        2 * world[1][x][y].p - world[2][x][y].p) /
                       (1 + world[1][x][y].courant / EW);
  }
}

void update_bottom() {
  const int y = world_height - 1;
  for (int x = 1; x < world_width - 1; x++) {
    world[0][x][y].p = (world[1][x][y].courant * world[1][x][y].courant *
                            (world[2][x][y].p / (world[1][x][y].courant * EW) +
                             2 * world[1][x][y - 1].p + world[1][x + 1][y].p +
                             world[1][x - 1][y].p - 4 * world[1][x][y].p) +
                        2 * world[1][x][y].p - world[2][x][y].p) /
                       (1 + world[1][x][y].courant / EW);
  }
}

void update_top_right() {
  const int y = 0, x = world_width - 1;

  world[0][x][y].p =
      (world[1][x][y].courant * world[1][x][y].courant *
           (2 * world[2][x][y].p / (world[1][x][y].courant * EW) +
            2 * world[1][x - 1][y].p + 2 * world[1][x][y + 1].p -
            4 * world[1][x][y].p) +
       2 * world[1][x][y].p - world[2][x][y].p) /
      (1 + 2 * world[1][x][y].courant / EW);
}

void update_top_left() {
  const int y = 0, x = 0;

  world[0][x][y].p =
      (world[1][x][y].courant * world[1][x][y].courant *
           (2 * world[2][x][y].p / (world[1][x][y].courant * EW) +
            2 * world[1][x + 1][y].p + 2 * world[1][x][y + 1].p -
            4 * world[1][x][y].p) +
       2 * world[1][x][y].p - world[2][x][y].p) /
      (1 + 2 * world[1][x][y].courant / EW);
}

void update_bottom_left() {
  const int y = world_height - 1, x = 0;

  world[0][x][y].p =
      (world[1][x][y].courant * world[1][x][y].courant *
           (2 * world[2][x][y].p / (world[1][x][y].courant * EW) +
            2 * world[1][x + 1][y].p + 2 * world[1][x][y - 1].p -
            4 * world[1][x][y].p) +
       2 * world[1][x][y].p - world[2][x][y].p) /
      (1 + 2 * world[1][x][y].courant / EW);
}

void update_bottom_right() {
  const int y = world_height - 1, x = world_width - 1;

  world[0][x][y].p =
      (world[1][x][y].courant * world[1][x][y].courant *
           (2 * world[2][x][y].p / (world[1][x][y].courant * EW) +
            2 * world[1][x - 1][y].p + 2 * world[1][x][y - 1].p -
            4 * world[1][x][y].p) +
       2 * world[1][x][y].p - world[2][x][y].p) /
      (1 + 2 * world[1][x][y].courant / EW);
}

// Update world with source and capture receiver signal
void update_world() {
  memcpy(world[2], world[1], sizeof(Cell) * world_width * world_height);
  memcpy(world[1], world[0], sizeof(Cell) * world_width * world_height);

  // Update propagation
  update_inner();
  update_right();
  update_left();
  update_top();
  update_bottom();
  update_top_right();
  update_top_left();
  update_bottom_left();
  update_bottom_right();

  // Source signal
  if (source_type == 0) {
    if (timestep == 0)
      world[0][source_x][source_y].p = impulse_value;
    else if (timestep == 1)
      world[0][source_x][source_y].p = 0;
  } else if (source_type == 1) {
    world[0][source_x][source_y].p =
        impulse_value * sin(2 * M_PI * sinewave_freq * timestep * dt);
  }

  // Capture receiver signal
  // if (timestep % 2 == 0) {
  receiver_signal.push_back(world[0][receiver_x][receiver_y].p);
  if (receiver_signal.size() > max_signal_samples) {
    receiver_signal.erase(receiver_signal.begin());
  }
  // }

  timestep++;
}

int main() {
  // Window size doubled in width
  InitWindow(world_width * scale + graph_width, world_height * scale,
             "Sound Propagation with Receiver Graph and Gradations");
  SetTargetFPS(60);

  std::string dx_string = "dx: " + std::to_string(dx) + " [m]";
  std::string dt_string = "dt: " + std::to_string(dx) + " [s]";
  std::string ew_string = "Ew: " + std::to_string(EW);
  std::string reflection_string = "refl: " + std::to_string(reflection);
  std::string size_string = "size: " + std::to_string(world_width * dx) +
                            " [m] x " + std::to_string(world_height * dx) +
                            " [m]     (" + std::to_string(world_width) + " x " +
                            std::to_string(world_height) + " cells)";
  init_world();

  while (!WindowShouldClose()) {
    // Pause/unpause simulation
    if (IsKeyPressed(KEY_SPACE)) {
      is_paused = !is_paused;
    }

    if (!is_paused) {
      update_world();
    }

    // Draw the simulation and graph
    BeginDrawing();
    ClearBackground(BLACK);
    draw_world();
    draw_graph();
    DrawFPS(10, 10);

    if (is_paused) {
      DrawText("PAUSED", graph_x_offset + graph_width / 2 - 50, 20, 30, YELLOW);
    }

    DrawText(dx_string.c_str(), 20, 20, 30, RAYWHITE);
    DrawText(dt_string.c_str(), 20, 50, 30, RAYWHITE);
    DrawText(ew_string.c_str(), 20, 80, 30, RAYWHITE);
    DrawText(reflection_string.c_str(), 20, 110, 30, RAYWHITE);
    DrawText(size_string.c_str(), 20, 140, 30, RAYWHITE);
    DrawText(("t: " + std::to_string(timestep * dt) + " [s]").c_str(), 20, 170,
             30, RAYWHITE);

    EndDrawing();
  }

  CloseWindow();
}
