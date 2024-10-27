import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# FDTD parameters
nx, ny, nz = 400, 400, 100  # Grid size
dx, dy, dz = 0.01, 0.01, 0.01  # Grid spacing (in meters)
dt = 0.00001  # Time step (in seconds)
c = 343  # Speed of sound in air (m/s)
steps = 1000  # Number of time steps

# Courant number check for stability: (c * dt) <= 1/sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
courant_limit = 1 / np.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2)
if c * dt > courant_limit:
    raise ValueError(
        f"Courant condition not met. Reduce dt or increase dx, dy, dz. Max dt: {courant_limit / c}"
    )

# Initialize the pressure fields for current, previous, and next time steps
p = np.zeros((nx, ny, nz))  # Pressure at the current time step
p_prev = np.zeros((nx, ny, nz))  # Pressure at the previous time step
p_next = np.zeros((nx, ny, nz))  # Pressure at the next time step

# Visualization parameters
fig, ax = plt.subplots()
slice_index = nz // 2  # Visualize a slice of the 3D field
im = ax.imshow(p[:, :, slice_index], cmap="RdBu", vmin=-0.01, vmax=0.01, animated=True)


# Function to add a pulse (initial wave) at the start of the simulation
def add_wave(p, center, amplitude=1.0, width=1.0):
    cx, cy, cz = center
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                p[i, j, k] += amplitude * np.exp(-(r**2) / (2 * width**2))


# Initialize a wave pulse in the middle of the grid
add_wave(p, center=(nx // 2, ny // 2, nz // 2), amplitude=1.0, width=5.0)


# Function to update the pressure fields (FDTD core loop)
def fdtd_step(p, p_prev, p_next, dx, dy, dz, dt, c):
    # Compute the Laplacian of p (second spatial derivatives)
    laplacian_p = (
        (p[2:, 1:-1, 1:-1] - 2 * p[1:-1, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) / dx**2
        + (p[1:-1, 2:, 1:-1] - 2 * p[1:-1, 1:-1, 1:-1] + p[1:-1, :-2, 1:-1]) / dy**2
        + (p[1:-1, 1:-1, 2:] - 2 * p[1:-1, 1:-1, 1:-1] + p[1:-1, 1:-1, :-2]) / dz**2
    )

    # Update the pressure field using the wave equation finite-difference formulation
    p_next[1:-1, 1:-1, 1:-1] = (
        2 * p[1:-1, 1:-1, 1:-1] - p_prev[1:-1, 1:-1, 1:-1] + (c * dt) ** 2 * laplacian_p
    )


# Animation update function
def update(frame):
    global p, p_prev, p_next

    # Update the pressure fields using the FDTD method
    fdtd_step(p, p_prev, p_next, dx, dy, dz, dt, c)

    # Swap references: shift the time steps forward
    p_prev, p, p_next = p, p_next, p_prev

    # Update the visualization
    im.set_array(p[:, :, slice_index])
    return [im]


# Set up the animation
ani = animation.FuncAnimation(fig, update, frames=steps, interval=10, blit=True)

# Display the simulation
plt.colorbar(im)
plt.title("3D Acoustic Wave Simulation (Slice)")
plt.show()
