import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
Nx, Ny, Nz = 50, 50, 50  # Reduced grid size for faster 3D visualization
dx = 0.01  # Spatial step (m)
c = 343  # Speed of sound (m/s)
dt = 0.99 * dx / (np.sqrt(3) * c)  # Time step based on CFL condition (s)
Tmax = 0.01  # Simulation time (s)
Nt = int(Tmax / dt)  # Number of time steps

# Initialize pressure field and velocity fields
p = np.zeros((Nx, Ny, Nz))  # Pressure
vx = np.zeros((Nx, Ny, Nz))  # Velocity in x direction
vy = np.zeros((Nx, Ny, Nz))  # Velocity in y direction
vz = np.zeros((Nx, Ny, Nz))  # Velocity in z direction

# Source properties
src_x, src_y, src_z = Nx // 2, Ny // 2, Nz // 2  # Source position (center of the grid)
src_freq = 1000  # Source frequency (Hz)
src_amplitude = 1.0  # Source amplitude

# Precompute constants for update equations
C1 = c**2 * dt / dx
C2 = dt / dx

# Set up 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Time loop for FDTD simulation
for n in range(Nt):
    # Update velocity fields (staggered grid)
    vx[1:, :, :] -= C2 * (p[1:, :, :] - p[:-1, :, :])
    vy[:, 1:, :] -= C2 * (p[:, 1:, :] - p[:, :-1, :])
    vz[:, :, 1:] -= C2 * (p[:, :, 1:] - p[:, :, :-1])

    # Update pressure field
    p[:-1, :, :] += C1 * (vx[1:, :, :] - vx[:-1, :, :])
    p[:, :-1, :] += C1 * (vy[:, 1:, :] - vy[:, :-1, :])
    p[:, :, :-1] += C1 * (vz[:, :, 1:] - vz[:, :, :-1])

    # Inject source signal (simple sine wave)
    p[src_x, src_y, src_z] += src_amplitude * np.sin(2 * np.pi * src_freq * n * dt)

    # Visualization (every 10th time step)
    if n % 10 == 0:
        ax.clear()

        # Select points above a certain pressure threshold for visualization
        threshold = 0.01  # Choose a threshold to highlight the wavefront
        x, y, z = np.where(np.abs(p) > threshold)
        values = p[x, y, z]

        # Create a 3D scatter plot
        scatter = ax.scatter(x, y, z, c=values, cmap="RdBu", marker="o")

        # Set plot limits
        ax.set_xlim(0, Nx)
        ax.set_ylim(0, Ny)
        ax.set_zlim(0, Nz)

        # Add color bar and title
        plt.colorbar(scatter, ax=ax, shrink=0.5)
        ax.set_title(f"3D Acoustic Wave Propagation at Time step {n}")

        plt.pause(0.01)

plt.show()
