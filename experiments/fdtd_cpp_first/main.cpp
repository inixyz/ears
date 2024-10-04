#include <algorithm>
#include <iostream>
#include <raylib.h>
#include <raymath.h>
#include <stdint.h>
#include <string.h>
#include <thread>

const Vector2 WINDOW_SIZE = {2300, 1080};
const Vector2 CELL_SIZE = {4, 4};

const int WORLD_TIME_ITER = 3;
constexpr Vector2 WORLD_SIZE = {500, 200};
const float SOUND_SPEED = 343;
const float WALL_REFLECTION_FACTOR = 0.5;
const float TIME_SCALE = 0.001, SIZE_SCALE = 0.1;

const float TIME_STEP = (SOUND_SPEED * TIME_SCALE) / (2 * SIZE_SCALE);
const float EMITTANCE =
    (1 - WALL_REFLECTION_FACTOR) / (1 + WALL_REFLECTION_FACTOR);

float world[WORLD_TIME_ITER][(int)WORLD_SIZE.x][(int)WORLD_SIZE.y];
bool is_solid[(int)WORLD_SIZE.x][(int)WORLD_SIZE.y];
int nr_neighbours[(int)WORLD_SIZE.x][(int)WORLD_SIZE.y];

bool simulation_paused = true;
float time_factor = 100;
const float TIME_FACTOR_STEP = 10;

const Vector2 canvas_size = Vector2Multiply(WORLD_SIZE, CELL_SIZE);
Vector2 offset = Vector2Scale(Vector2Subtract(WINDOW_SIZE, canvas_size), 0.5);

void draw_world() {
  // draw outline
  const int OUTLINE_THICKNESS = 10;
  DrawRectangleV(Vector2SubtractValue(offset, OUTLINE_THICKNESS),
                 Vector2AddValue(canvas_size, 2 * OUTLINE_THICKNESS), BLACK);

  for (int x = 0; x < WORLD_SIZE.x; x++) {
    for (int y = 0; y < WORLD_SIZE.y; y++) {
      const Vector2 pos = Vector2Multiply({(float)x, (float)y}, CELL_SIZE);
      const float amplitude = world[0][x][y];

      Color color = RAYWHITE; // color for amplitude 0
      if (is_solid[x][y])
        color = BLACK;
      else {
        int color_amount = Remap(amplitude, -1, 1, -255, 255);
        if (color_amount > 0)
          color = Color{255, (unsigned char)(255 - color_amount), 255, 255};
        else if (color_amount < 0)
          color = Color{(unsigned char)(255 + color_amount), 255, 255, 255};
      }

      DrawRectangleV(Vector2Add(pos, offset), CELL_SIZE, color);
    }
  }
}

void rigid_update(int x, int y) {
  world[0][x][y] = (2 - 0.5 * nr_neighbours[x][y]) * world[1][x][y] +
                   0.5 * (world[1][x + 1][y] + world[1][x - 1][y] +
                          world[1][x][y + 1] + world[1][x][y - 1]) -
                   world[2][x][y];
}

void apply_loss(int x, int y) {
  world[0][x][y] = (world[0][x][y] +
                    EMITTANCE * (4 - nr_neighbours[x][y]) * world[2][x][y]) /
                   (1 + EMITTANCE * (4 - nr_neighbours[x][y]));
}

void simulate() {
  // move timeline
  for (int time_iter = 2; time_iter > 0; time_iter--)
    memcpy(world[time_iter], world[time_iter - 1],
           WORLD_SIZE.x * WORLD_SIZE.y * sizeof(float));

  // compute new time step
  for (int x = 0; x < WORLD_SIZE.x; x++) {
    for (int y = 0; y < WORLD_SIZE.y; y++) {
      if (is_solid[x][y])
        continue;
      rigid_update(x, y);
      apply_loss(x, y);
    }
  }
}

void set_solid_rect(const Vector2 pos, const Vector2 size) {
  const Vector2 end_pos = Vector2Add(pos, size);

  for (int x = pos.x; x < end_pos.x; x++)
    for (int y = pos.y; y < end_pos.y; y++)
      is_solid[x][y] = true;
}

void set_solid_circle(int x0, int y0, int radius, bool value) {
  int x = radius, y = 0;
  int err = 0;

  while (x >= y) {
    // plot the circle points
    is_solid[x0 + x][y0 + y] = value;

    is_solid[x0 + y][y0 + x] = value;
    is_solid[x0 - y][y0 + x] = value;
    is_solid[x0 - x][y0 + y] = value;
    is_solid[x0 - x][y0 - y] = value;
    is_solid[x0 - y][y0 - x] = value;
    is_solid[x0 + y][y0 - x] = value;
    is_solid[x0 + x][y0 - y] = value;

    // update the error term and coordinates
    if (err <= 0) {
      y++;
      err += 2 * y + 1;
    }
    if (err > 0) {
      x--;
      err -= 2 * x + 1;
    }
  }
}

void set_solid_fill(int x, int y, bool value) {
  if (is_solid[x][y] == value)
    return;

  is_solid[x][y] = value;

  set_solid_fill(x - 1, y, value);
  set_solid_fill(x + 1, y, value);
  set_solid_fill(x, y - 1, value);
  set_solid_fill(x, y + 1, value);
}
void init_world() {
  const int CAMERA_ID = 0;
  // bounding box
  set_solid_rect({0, 0}, {WORLD_SIZE.x, 1});
  set_solid_rect({WORLD_SIZE.x - 1, 0}, {1, WORLD_SIZE.y});
  set_solid_rect({0, WORLD_SIZE.y - 1}, {WORLD_SIZE.x, 1});
  set_solid_rect({0, 0}, {1, WORLD_SIZE.y});

  if (CAMERA_ID == 1) {
    set_solid_rect({0, 0}, {WORLD_SIZE.x, WORLD_SIZE.y * 0.4});
    set_solid_rect({0, 0}, {WORLD_SIZE.x * 0.15, WORLD_SIZE.y});
    set_solid_rect({WORLD_SIZE.x * 0.85, 0},
                   {WORLD_SIZE.x * 0.15, WORLD_SIZE.y});
    set_solid_circle(WORLD_SIZE.x / 2, WORLD_SIZE.y * 0.4 / 2 + 50,
                     WORLD_SIZE.y * 0.7 / 2, false);
    set_solid_fill(WORLD_SIZE.x / 2, WORLD_SIZE.y * 0.4 / 2, false);
  }
  // compute nr_neighbours
  for (int x = 0; x < WORLD_SIZE.x; x++) {
    for (int y = 0; y < WORLD_SIZE.y; y++) {
      if (is_solid[x][y])
        continue;
      nr_neighbours[x][y] = !is_solid[x + 1][y] + !is_solid[x - 1][y] +
                            !is_solid[x][y + 1] + !is_solid[x][y - 1];
    }
  }
}

void apply_impulse(const Vector2 pos, bool positive) {
  const float START_AMPLITUDE = 4;
  const int IMPULSE_RADIUS = 2;

  for (int x = pos.x - IMPULSE_RADIUS; x < pos.x + IMPULSE_RADIUS; x++)
    for (int y = pos.y - IMPULSE_RADIUS; y < pos.y + IMPULSE_RADIUS; y++)
      if (positive)
        world[0][x][y] = START_AMPLITUDE;
      else
        world[0][x][y] = -START_AMPLITUDE;
}

void processing_thread() {
  while (true) {
    if (!simulation_paused) {
      simulate();
      WaitTime(TIME_STEP * 1 / time_factor);
    }
  }
}

int main() {
  InitWindow(WINDOW_SIZE.x, WINDOW_SIZE.y, "fdtd_cpp_first");
  SetTargetFPS(60);

  init_world();

  std::thread simulation_thread(processing_thread);

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE))
      simulation_paused = !simulation_paused;
    else if (IsKeyPressed(KEY_Q)) {
      time_factor -= TIME_FACTOR_STEP;
      time_factor = std::max(time_factor, (float)0);
    } else if (IsKeyPressed(KEY_E))
      time_factor += TIME_FACTOR_STEP;

    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
      Vector2 mpos =
          Vector2Divide(Vector2Subtract(GetMousePosition(), offset), CELL_SIZE);
      if (CheckCollisionPointRec(mpos,
                                 Rectangle{0, 0, WORLD_SIZE.x, WORLD_SIZE.y})) {
        apply_impulse(mpos, true);
      }
    } else if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
      Vector2 mpos =
          Vector2Divide(Vector2Subtract(GetMousePosition(), offset), CELL_SIZE);
      if (CheckCollisionPointRec(mpos,
                                 Rectangle{0, 0, WORLD_SIZE.x, WORLD_SIZE.y})) {
        apply_impulse(mpos, false);
      }
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);
    draw_world();
    DrawFPS(10, 10);
    DrawText(TextFormat("TIME_SCALE = %f [s/tick], TIME_STEP = %f [s], "
                        "time_factor = %f",
                        TIME_SCALE, TIME_STEP, time_factor),
             10, 30, 30, RED);
    DrawText(TextFormat("SIZE_SCALE = %f [m/cell]", SIZE_SCALE), 10, 60, 30,
             RED);
    EndDrawing();
  }
}
