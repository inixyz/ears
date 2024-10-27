import numpy as np
import matplotlib.pyplot as plt

# Room dimensions and discretization parameters
dx = 0.05  # grid spacing in meters
dy = 0.05
dt = 0.0001  # time step
room_width = 10  # meters
room_height = 10
c_air = 343  # speed of sound in air (m/s)
max_time = 0.1  # simulation time in seconds
num_steps = int(max_time / dt)

# Material properties
materials = {
    "air": {"c": 343, "rho": 1.21, "loss": 0.001},
    "wood": {"c": 1500, "rho": 700, "loss": 0.01},
}

# Initialize room as a grid of air
nx = int(room_width / dx)
ny = int(room_height / dy)
room = np.full((nx, ny), "air")

# Add a square wooden object somewhere in the room
wood_start = (int(nx * 0.4), int(ny * 0.4))
wood_size = int(nx * 0.2)
room[
    wood_start[0] : wood_start[0] + wood_size, wood_start[1] : wood_start[1] + wood_size
] = "wood"

# Pressure and velocity fields
p = np.zeros((nx, ny))  # pressure field at the current time step
p_prev = np.zeros_like(p)  # pressure field at the previous time step
p_next = np.zeros_like(p)  # pressure field at the next time step

# Source and receiver positions
source_x, source_y = int(nx * 0.1), int(ny * 0.1)
receiver_x, receiver_y = int(nx * 0.9), int(ny * 0.9)

# Input signal (simple sine wave pulse)
f = 1000  # frequency of source (Hz)
source_signal = np.sin(2 * np.pi * f * np.arange(num_steps) * dt)

# Storage for output signal at the receiver
receiver_signal = np.zeros(num_steps)


# Helper function to compute wave speed and material losses
def get_material_properties(i, j):
    mat = room[i, j]
    return materials[mat]["c"], materials[mat]["rho"], materials[mat]["loss"]


# FDTD loop
for step in range(num_steps):
    # Set the source
    p[source_x, source_y] = source_signal[step]

    # Update pressure field using the wave equation with material properties
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            c, rho, loss = get_material_properties(i, j)
            p_next[i, j] = (
                2 * p[i, j]
                - p_prev[i, j]
                + (dt**2 / rho)
                * (
                    (c**2 / dx**2) * (p[i + 1, j] + p[i - 1, j] - 2 * p[i, j])
                    + (c**2 / dy**2) * (p[i, j + 1] + p[i, j - 1] - 2 * p[i, j])
                )
            )
            # Apply loss to the wave propagation
            p_next[i, j] *= 1 - loss

    # Apply boundary conditions (simple absorbing boundaries)
    p_next[0, :] = p_next[1, :]  # top boundary
    p_next[-1, :] = p_next[-2, :]  # bottom boundary
    p_next[:, 0] = p_next[:, 1]  # left boundary
    p_next[:, -1] = p_next[:, -2]  # right boundary

    # Record the signal at the receiver
    receiver_signal[step] = p[receiver_x, receiver_y]

    # Shift time steps
    p_prev = np.copy(p)
    p = np.copy(p_next)

    # Visualize the pressure field every 100 steps
    if step % 100 == 0:
        plt.imshow(
            p.T,
            origin="lower",
            cmap="jet",
            aspect="auto",
            extent=[0, room_width, 0, room_height],
        )
        plt.colorbar(label="Acoustic Pressure")
        plt.title(f"Step {step}")
        plt.pause(0.01)

# Plot the source and receiver signals
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.arange(num_steps) * dt, source_signal)
plt.title("Source Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(np.arange(num_steps) * dt, receiver_signal)
plt.title("Receiver Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
