import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

# Simulation parameters
nx, ny = 150, 150  # Grid size (reduced for speed)
dx = dy = 1.0  # Spatial step
dt = 0.1  # Time step
c1 = 1.0  # Sound speed in material 1
c2 = 0.5  # Sound speed in material 2
n_steps = 400  # Number of time steps to simulate

# Define speed of sound grid (materials)
c = np.ones((nx, ny)) * c1
c[:, ny // 2 :] = c2  # Right half of the grid has slower sound speed (material 2)

# Courant number (local)
courant_sq = (c * dt / dx) ** 2

# Initialize pressure fields
p = np.zeros((nx, ny))  # Current time step
p_prev = np.zeros((nx, ny))  # Previous time step
p_next = np.zeros((nx, ny))  # Next time step

# Initial conditions: place a Gaussian pulse in material 1
x0, y0 = nx // 4, ny // 2
p[x0, y0] = 1.0


# Numba-accelerated FDTD update function
@njit
def fdtd_update(p, p_prev, p_next, courant_sq):
    nx, ny = p.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            p_next[i, j] = (
                2 * p[i, j]
                - p_prev[i, j]
                + courant_sq[i, j]
                * (p[i + 1, j] + p[i - 1, j] + p[i, j + 1] + p[i, j - 1] - 4 * p[i, j])
            )


# Setup the plot for visualization
fig, ax = plt.subplots()
img = ax.imshow(p, vmin=-0.1, vmax=0.1, cmap="seismic", origin="lower")
ax.set_title("2D FDTD Acoustic Wave Simulation")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(img, ax=ax, label="Pressure")


# Animation update function
def update(frame):
    global p, p_prev, p_next
    fdtd_update(p, p_prev, p_next, courant_sq)

    # Update plot data
    img.set_data(p)

    # Rotate variables for next time step
    p_prev, p, p_next = p, p_next, p_prev  # Rotate variables

    return [img]


# Run the animation
ani = animation.FuncAnimation(fig, update, frames=n_steps, blit=True)
plt.show()
