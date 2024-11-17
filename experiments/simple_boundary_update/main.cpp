#include <cmath>
#include <cstring>
#include <iostream>
#include <raylib.h>
#include <raymath.h>

struct Cell {
  float p, courant;
};

// const float WALL_IMPEDANCE = 200000; // rayl
const float WALL_IMPEDANCE = 1'575'000; // rayl
const float WALL_DENSITY = 510;
const float AIR_SPEED = 343;
const float EW = WALL_IMPEDANCE / (WALL_DENSITY * AIR_SPEED);

const float dx = 0.1;
const float dt = dx / (AIR_SPEED * sqrt(2));

const int world_width = 100, world_height = 100;
const int scale = 14;
Cell world[3][world_width][world_height];

// signal types
// 0 = impulse
// 1 = sinwave
const int source_type = 0;
const int source_x = world_width / 2, source_y = world_height / 2;
const int sinewave_freq = 500;

void draw_world() {
  Color color;
  float p_norm;

  const float color_intensify = 0.3;

  for (int x = 0; x < world_width; x++) {
    for (int y = 0; y < world_height; y++) {
      p_norm = Remap(world[0][x][y].p, -1 * color_intensify,
                     1 * color_intensify, -128, 127);
      p_norm = p_norm > 127 ? 127 : p_norm;
      p_norm = p_norm < -128 ? -128 : p_norm;
      p_norm = (int)p_norm;
      color = Color{0, (unsigned char)(128 + p_norm),
                    (unsigned char)(128 - p_norm), 255};

      DrawRectangle(x * scale, y * scale, scale, scale, color);
    }
  }
}

void init_world() {
  for (int x = 0; x < world_width; x++) {
    for (int y = 0; y < world_height; y++) {
      for (int t = 0; t < 3; t++) {
        world[t][x][y].p = 0;
        world[t][x][y].courant = (dt * AIR_SPEED) / dx;
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

long long timestep = 0;
void update_world() {
  memcpy(world[2], world[1], sizeof(Cell) * world_width * world_height);
  memcpy(world[1], world[0], sizeof(Cell) * world_width * world_height);

  update_inner();

  update_right();
  update_left();
  update_top();
  update_bottom();

  update_top_right();
  update_top_left();
  update_bottom_left();
  update_bottom_right();

  if (source_type == 0) {
    if (timestep == 0)
      world[0][source_x][source_y].p = 1;
  } else if (source_type == 1) {
    world[0][source_x][source_y].p =
        sin(2 * M_PI * sinewave_freq * timestep * dt);
  }

  timestep++;
}

int main() {
  InitWindow(world_width * scale, world_height * scale,
             "simple_boundary_update");
  SetTargetFPS(10);

  std::string dx_string = "dx: " + std::to_string(dx) + " [m]";
  std::string dt_string = "dt: " + std::to_string(dx) + " [s]";
  std::string ew_string = "Ew: " + std::to_string(EW);
  std::string size_string = "size: " + std::to_string(world_width * dx) +
                            " [m] x " + std::to_string(world_height * dx) +
                            " [m]     (" + std::to_string(world_width) + " x " +
                            std::to_string(world_height) + " cells)";

  init_world();

  while (!WindowShouldClose()) {
    update_world();
    // std::cout << world[0][world_width / 2][world_height / 2].p << std::endl;

    BeginDrawing();
    ClearBackground(BLACK);
    draw_world();
    DrawFPS(0, 0);
    DrawText(dx_string.c_str(), 20, 20, 30, RAYWHITE);
    DrawText(dt_string.c_str(), 20, 50, 30, RAYWHITE);
    DrawText(ew_string.c_str(), 20, 80, 30, RAYWHITE);
    DrawText(size_string.c_str(), 20, 110, 30, RAYWHITE);
    DrawText(("t: " + std::to_string(timestep * dt) + " [s]").c_str(), 20, 140,
             30, RAYWHITE);
    EndDrawing();
  }

  CloseWindow();
}
