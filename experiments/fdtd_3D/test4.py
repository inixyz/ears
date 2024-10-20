import pygame
import numpy as np

# Simulation parameters
grid_size = 100  # 3D grid size (NxNxN)
time_steps = 1000  # Number of time steps
c = 1  # Wave speed
dt = 0.1  # Time step
dx = 1.0  # Spatial step

# CFL condition for stability
CFL = (c * dt / dx) ** 2

# Pygame setup
pygame.init()
screen_size = 600
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("3D FDTD Wave Simulation (2D Slice Visualization)")


# Define color map for visualization
def get_color(val, min_val, max_val):
    normalized = int(255 * (val - min_val) / (max_val - min_val))
    return (normalized, normalized, 255 - normalized)  # Blue shades


# Initialize wave grids for current, previous, and next time steps
u = np.zeros((grid_size, grid_size, grid_size))  # Current time step
u_prev = np.zeros_like(u)  # Previous time step
u_next = np.zeros_like(u)  # Next time step

# Source position (center of the grid)
source_x, source_y, source_z = grid_size // 2, grid_size // 2, grid_size // 2
u[source_x, source_y, source_z] = 1.0  # Initial wave perturbation


# Main FDTD function to update wave at each time step
def update_wave(u, u_prev, u_next, CFL):
    # Compute the wave equation using central differences in all 3 dimensions
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            for k in range(1, grid_size - 1):
                laplacian = (
                    u[i + 1, j, k]
                    + u[i - 1, j, k]
                    + u[i, j + 1, k]
                    + u[i, j - 1, k]
                    + u[i, j, k + 1]
                    + u[i, j, k - 1]
                    - 6 * u[i, j, k]
                )

                # Update the wave at each grid point using the wave equation
                u_next[i, j, k] = 2 * u[i, j, k] - u_prev[i, j, k] + CFL * laplacian


# Function to draw a 2D slice of the 3D wave grid
def draw_2d_slice(screen, u_slice):
    # Normalize the wave values for better visualization
    min_val, max_val = np.min(u_slice), np.max(u_slice)
    slice_scaled = (u_slice - min_val) / (max_val - min_val) * 255

    # Scale the slice to the screen size and render
    scale = screen_size // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            color = get_color(slice_scaled[i, j], 0, 255)
            pygame.draw.rect(screen, color, (i * scale, j * scale, scale, scale))


# Simulation loop
def main():
    global u, u_prev, u_next

    running = True
    clock = pygame.time.Clock()

    z_slice = grid_size // 2  # We will visualize this z slice

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update the wave equation
        update_wave(u, u_prev, u_next, CFL)

        # Swap grids for the next time step
        u_prev, u, u_next = u, u_next, u_prev

        # Draw the 2D slice (we are visualizing the z_slice of the 3D grid)
        screen.fill((0, 0, 0))  # Clear the screen
        draw_2d_slice(screen, u[:, :, z_slice])

        # Update display
        pygame.display.flip()
        clock.tick(60)  # Limit the simulation to 60 FPS

    pygame.quit()


if __name__ == "__main__":
    main()
