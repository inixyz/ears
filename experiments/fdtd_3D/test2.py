import numpy as np
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 200  # Size of the grid
NUM_RAYS = 100  # Number of rays to simulate
WAVE_LENGTH = 20  # Length of the sound wave
DIRECTIONS = np.linspace(0, 2 * np.pi, NUM_RAYS)  # Direction of rays

# Initialize the grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Source of sound (can be changed to any point)
source = (GRID_SIZE // 2, GRID_SIZE // 2)


def trace_ray(angle):
    """Trace a single ray from the source in the given angle."""
    x, y = source
    dx = np.cos(angle)
    dy = np.sin(angle)

    while 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        # Mark the grid to indicate sound wave presence
        grid[int(y), int(x)] += 1

        # Move the ray forward
        x += dx
        y += dy


# Trace all rays
for angle in DIRECTIONS:
    trace_ray(angle)

# Visualize the results
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap="hot", interpolation="nearest")
plt.colorbar(label="Sound Wave Intensity")
plt.title("2D Sound Wave Propagation")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.scatter(*source[::-1], color="blue", label="Sound Source")  # Mark the source
plt.legend()
plt.show()
