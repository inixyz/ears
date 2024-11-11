import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numba import njit, prange

# Parameters
size_x = size_y = 1000
dx = 0.1  # meters
c0 = 343  # speed of sound in air (m/s)

# Initialize fields and speed of sound
p = np.zeros((3, size_x, size_y), dtype=np.float32)
c = np.full((size_x, size_y), c0, dtype=np.float32)

# Time step based on stability condition
dt = dx / (np.max(c) * np.sqrt(2))

# Source properties
frequency = 100  # Frequency of the sine wave (Hz)
source_amplitude = 10
omega = 2 * np.pi * frequency  # Angular frequency

# Sponge layer settings
sponge_thickness = 150  # Thickness of the damping layer in grid points
damping_factor = 0.015  # Base damping factor

# Define the damping layer with exponential decay towards the edges
damping = np.ones((size_x, size_y), dtype=np.float32)
for i in range(sponge_thickness):
    factor = np.exp(-damping_factor * ((sponge_thickness - i) / sponge_thickness) ** 2)
    damping[i, :] *= factor
    damping[-i - 1, :] *= factor
    damping[:, i] *= factor
    damping[:, -i - 1] *= factor

# Precompute constant terms
c_squared_dt_dx2 = (c**2) * (dt**2 / dx**2)


# JIT-compiled update function with parallelism
@njit(parallel=True)
def update(p, c_squared_dt_dx2, damping, dt, time_step):
    # Shift layers
    p[2] = p[1]
    p[1] = p[0]

    # Compute the Laplacian in parallel
    laplace = np.zeros_like(p[1])
    for x in prange(1, size_x - 1):
        for y in range(1, size_y - 1):
            laplace[x, y] = (
                p[1, x - 1, y]
                + p[1, x + 1, y]
                + p[1, x, y - 1]
                + p[1, x, y + 1]
                - 4 * p[1, x, y]
            )

    # Perform the main update in parallel, applying sponge layer damping
    for x in prange(1, size_x - 1):
        for y in range(1, size_y - 1):
            p[0, x, y] = (
                c_squared_dt_dx2[x, y] * laplace[x, y] + 2 * p[1, x, y] - p[2, x, y]
            ) * damping[x, y]  # Apply damping

    # Add the sinusoidal source at the center of the grid
    source_value = source_amplitude * np.sin(omega * time_step * dt)
    p[0, size_x // 2, size_y // 2] += source_value


# Visualization setup
fig, ax = plt.subplots()
cax = ax.imshow(
    p[0],
    cmap="viridis",
    interpolation="nearest",
    norm=colors.Normalize(vmin=-1, vmax=1),
)
fig.colorbar(cax)

# Annotation for parameters
text = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes,
    color="white",
    fontsize=12,
    bbox=dict(facecolor="black", alpha=0.5),
)


def update_plot(time_step):
    # Only update the plot every 10 steps to reduce rendering overhead
    if time_step % 10 == 0:
        cax.set_data(p[0])
        text.set_text(
            f"Time Step: {time_step}\n"
            f"dx: {dx} m\n"
            f"dt: {dt:.5f} s\n"
            f"Size: {size_x} x {size_y}"
        )
        plt.draw()
        plt.pause(0.01)


# Infinite simulation until plot window is closed
time_step = 0
print("Close the plot window to stop the simulation.")
while plt.fignum_exists(fig.number):  # Run until the plot window is closed
    update(p, c_squared_dt_dx2, damping, dt, time_step)
    update_plot(time_step)
    time_step += 1

plt.close()
