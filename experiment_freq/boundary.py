import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
nx, ny = 100, 100  # Grid dimensions
dx = 0.1  # Spatial step
dt = 0.0001  # Time step
c = 343  # Speed of sound in air (m/s)
Z_air = 400  # Impedance of air (example value)
Z_wall = 2000  # Impedance of wall (example value)

# Initialize pressure and velocity fields
p = np.zeros((nx, ny))  # Pressure field
vx = np.zeros((nx, ny))  # Velocity field in x direction
vy = np.zeros((nx, ny))  # Velocity field in y direction
p_prev = np.zeros((nx, ny))  # Previous time step pressure field

# Source position
source_x, source_y = nx // 2, ny // 2

# Boundary impedance ratio for transmission (simple model)
alpha = Z_air / Z_wall

# Time-stepping loop
n_steps = 1000
for n in range(n_steps):
    # Add a pressure pulse at the source
    p[source_x, source_y] += (
        np.sin(2 * np.pi * 100 * n * dt) * 1000
    )  # Example sinusoidal source

    # Update velocity fields based on pressure gradient
    vx[1:, :] -= (dt / dx) * (p[1:, :] - p[:-1, :]) / Z_air
    vy[:, 1:] -= (dt / dx) * (p[:, 1:] - p[:, :-1]) / Z_air

    # Update pressure field based on velocity divergence
    p[:-1, :-1] -= (
        (dt / dx) * (vx[1:, :-1] - vx[:-1, :-1] + vy[:-1, 1:] - vy[:-1, :-1]) * Z_air
    )

    # Apply Impedance Boundary Condition (IBC) at edges
    # Left and right boundaries
    p[0, :] = (1 - alpha) * p[1, :] + alpha * p_prev[0, :]  # Left boundary
    p[-1, :] = (1 - alpha) * p[-2, :] + alpha * p_prev[-1, :]  # Right boundary

    # Top and bottom boundaries
    p[:, 0] = (1 - alpha) * p[:, 1] + alpha * p_prev[:, 0]  # Bottom boundary
    p[:, -1] = (1 - alpha) * p[:, -2] + alpha * p_prev[:, -1]  # Top boundary

    # Update the previous pressure field for the boundary condition
    p_prev[:, :] = p[:, :]

    # Visualization (optional)
    if n % 10 == 0:
        plt.imshow(p, cmap="RdBu", vmin=-1000, vmax=1000)
        plt.colorbar()
        plt.title(f"Time Step: {n}")
        plt.pause(0.1)
        plt.clf()

plt.show()
