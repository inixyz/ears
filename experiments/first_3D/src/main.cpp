// #include "glad/glad.h"
// #include <GLFW/glfw3.h>
// #include <iostream>
//
// void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
//   glViewport(0, 0, width, height);
// }
//
// void process_input(GLFWwindow *window) {
//   if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
//     glfwSetWindowShouldClose(window, true);
//   }
// }
//
// int main() {
//   glfwInit();
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//
//   GLFWwindow *window = glfwCreateWindow(1600, 900, "first_3D", NULL, NULL);
//   if (window == NULL) {
//     std::cout << "failed to create GLFW window" << std::endl;
//     glfwTerminate();
//     return -1;
//   }
//   glfwMakeContextCurrent(window);
//
//   if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
//     std::cout << "failed to initialize GLAD" << std::endl;
//     return -1;
//   }
//
//   glViewport(0, 0, 1600, 900);
//   glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
//
//   while (!glfwWindowShouldClose(window)) {
//     glfwPollEvents();
//     process_input(window);
//
//     glClearColor(0.2, 0.3, 0.3, 1);
//     glClear(GL_COLOR_BUFFER_BIT);
//
//     glfwSwapBuffers(window);
//   }
//
//   glfwTerminate();
// }

#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Create a perspective matrix equivalent to gluPerspective
void set_perspective(float fov, float aspect_ratio, float near, float far) {
  float top = near * tan(fov * M_PI / 360.0f);
  float bottom = -top;
  float right = top * aspect_ratio;
  float left = -right;

  glFrustum(left, right, bottom, top, near, far);
}

// Initialize OpenGL
void init_opengl() {
  glEnable(GL_DEPTH_TEST); // Enable depth testing for proper 3D rendering
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  set_perspective(45.0, 800.0 / 600.0, 0.1,
                  100.0); // Replace gluPerspective with custom perspective
}

// Render a single cube (voxel) at position x, y, z
void render_voxel(float x, float y, float z) {
  glPushMatrix();
  glTranslatef(x, y, z);

  // Use glBegin/glEnd to manually specify each face of the cube
  glBegin(GL_QUADS);
  // Front face (z = 0.5)
  glVertex3f(-0.5f, -0.5f, 0.5f);
  glVertex3f(0.5f, -0.5f, 0.5f);
  glVertex3f(0.5f, 0.5f, 0.5f);
  glVertex3f(-0.5f, 0.5f, 0.5f);

  // Back face (z = -0.5)
  glVertex3f(-0.5f, -0.5f, -0.5f);
  glVertex3f(-0.5f, 0.5f, -0.5f);
  glVertex3f(0.5f, 0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, -0.5f);

  // Left face (x = -0.5)
  glVertex3f(-0.5f, -0.5f, -0.5f);
  glVertex3f(-0.5f, -0.5f, 0.5f);
  glVertex3f(-0.5f, 0.5f, 0.5f);
  glVertex3f(-0.5f, 0.5f, -0.5f);

  // Right face (x = 0.5)
  glVertex3f(0.5f, -0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, 0.5f);
  glVertex3f(0.5f, 0.5f, 0.5f);
  glVertex3f(0.5f, 0.5f, -0.5f);

  // Top face (y = 0.5)
  glVertex3f(-0.5f, 0.5f, -0.5f);
  glVertex3f(-0.5f, 0.5f, 0.5f);
  glVertex3f(0.5f, 0.5f, 0.5f);
  glVertex3f(0.5f, 0.5f, -0.5f);

  // Bottom face (y = -0.5)
  glVertex3f(-0.5f, -0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, 0.5f);
  glVertex3f(-0.5f, -0.5f, 0.5f);
  glEnd();

  glPopMatrix();
}

// Main render loop
void render_scene() {
  for (int x = -10; x <= 10; x++) {
    for (int y = -10; y <= 10; y++) {
      for (int z = -10; z <= 10; z++) {
        render_voxel(x, y, z); // Render a 10x10x10 grid of voxels
      }
    }
  }
}

// Keyboard input callback
void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, 1);
}

int main() {
  // Initialize GLFW
  if (!glfwInit()) {
    printf("Failed to initialize GLFW\n");
    return -1;
  }

  // Create a windowed mode window and OpenGL context
  GLFWwindow *window = glfwCreateWindow(800, 600, "Voxel Engine", NULL, NULL);
  if (!window) {
    printf("Failed to create GLFW window\n");
    glfwTerminate();
    return -1;
  }

  // Make the window's context current
  glfwMakeContextCurrent(window);

  init_opengl();

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    // Input
    process_input(window);

    // Render here
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up camera
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -50.0f); // Move the "camera" back

    render_scene(); // Render the voxel grid

    // Swap front and back buffers
    glfwSwapBuffers(window);

    // Poll for and process events
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}
