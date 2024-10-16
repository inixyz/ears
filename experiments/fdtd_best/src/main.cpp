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

void update_camera(Camera2D &camera) {
  Vector2 dir;
  dir.x = IsKeyDown(KEY_D) - IsKeyDown(KEY_A);
  dir.y = IsKeyDown(KEY_S) - IsKeyDown(KEY_W);

  const float CAMERA_SPEED = 500;

  float delta = GetFrameTime();
  Vector2 velocity = {dir.x * CAMERA_SPEED * delta,
                      dir.y * CAMERA_SPEED * delta};

  camera.target.x += velocity.x;
  camera.target.y += velocity.y;

  // Camera zoom controls
  camera.zoom += ((float)GetMouseWheelMove() * 0.05f);

  if (camera.zoom > 3.0f)
    camera.zoom = 3.0f;
  else if (camera.zoom < 0.1f)
    camera.zoom = 0.1f;

  // Camera reset (zoom and rotation)
  if (IsKeyPressed(KEY_R)) {
    camera.zoom = 1.0f;
    camera.rotation = 0.0f;
  }
}

int main() {
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "fdtd_best");
  SetTargetFPS(60);

  World world;
  init_world(world);

  Camera2D camera = {0};
  camera.target = {(float)world.WIDTH / 2, (float)world.HEIGHT / 2};
  camera.offset = {WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2};
  camera.rotation = 0;
  camera.zoom = 1;

  while (!WindowShouldClose()) {
    update_camera(camera);

    BeginDrawing();
    ClearBackground(BLACK);

    BeginMode2D(camera);
    draw_world(world);
    EndMode2D();

    EndDrawing();
  }

  CloseWindow();
}
