#include <raylib.h>
#include <unordered_map>

const int WINDOW_WIDTH = 1600, WINDOW_HEIGHT = 900;

struct World {
  struct Cell {
    enum Material { AIR, WOOD } material;
    float value;
  };

  static const int WIDTH = 500, HEIGHT = 500;
  Cell data[3][WIDTH][HEIGHT];
};

void init_world(World &world) {
  for (int x = 0; x < world.WIDTH; x++) {
    for (int y = 0; y < world.HEIGHT; y++) {
      if (x == 0 || x == world.WIDTH - 1 || y == 0 ||
          world.HEIGHT == world.HEIGHT - 1)
        world.data[0][x][y].material = World::Cell::WOOD;
    }
  }
}

void draw_world(const World &world) {
  std::unordered_map<World::Cell::Material, Color> MATERIAL_COLORS = {
      {World::Cell::AIR, {173, 216, 230, 255}},
      {World::Cell::WOOD, {150, 111, 51, 255}}};

  for (int x = 0; x < world.WIDTH; x++) {
    for (int y = 0; y < world.HEIGHT; y++) {
      DrawPixel(x, y, MATERIAL_COLORS[world.data[0][x][y].material]);
    }
  }
}

int main() {
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "fdtd_best");

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);
    EndDrawing();
  }

  CloseWindow();
}
