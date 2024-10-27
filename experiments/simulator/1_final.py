import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm
import sys

# Constants for air and wood
c_air = 343  # Speed of sound in air (m/s)
c_wood = 3300  # Speed of sound in wood (m/s)
rho_air = 1.225  # Air density (kg/m^3)
rho_wood = 700  # Wood density (kg/m^3)

# Grid size and time parameters
nx, ny, nz = 100, 100, 100  # Grid dimensions
dx = 0.1  # Grid spacing (meters)
dt = dx / (np.sqrt(3) * c_wood)  # Time step satisfying CFL condition
steps = 5000  # Number of time steps

# Initialize fields
pressure_current = np.zeros((nx, ny, nz))
pressure_previous = np.zeros((nx, ny, nz))
velocity_x = np.zeros((nx, ny, nz))
velocity_y = np.zeros((nx, ny, nz))
velocity_z = np.zeros((nx, ny, nz))

# Material properties (0: air, 1: wood)
material = np.zeros((nx, ny, nz))
material[:, :, :50] = 1  # The left half is wood, the rest is air


# Get material properties function
def get_material_properties(material):
    c = np.where(material == 1, c_wood, c_air)
    rho = np.where(material == 1, rho_wood, rho_air)
    return c, rho


# Apply propagation losses (air and wood have different losses)
def apply_losses(pressure, material):
    loss_air = 0.999  # Slight loss for air
    loss_wood = 0.998  # Slightly more loss for wood
    return np.where(material == 1, pressure * loss_wood, pressure * loss_air)


# Inject a point source signal
def inject_source(pressure, signal, t, source_pos):
    pressure[source_pos[0], source_pos[1], source_pos[2]] += signal[t]
    return pressure


# Clamp pressure to avoid overflow
def clamp_pressure(pressure, min_val=-1e5, max_val=1e5):
    return np.clip(pressure, min_val, max_val)


# Load source signal from .wav file if provided
if len(sys.argv) > 1 and sys.argv[1].endswith(".wav"):
    sample_rate, source_signal = wavfile.read(sys.argv[1])
    source_signal = source_signal / np.max(np.abs(source_signal))  # Normalize signal
    signal_length = len(source_signal)
else:
    # Default signal is a simple sinusoidal wave (if no .wav file provided)
    t = np.linspace(0, 1, 1000)
    source_signal = np.sin(2 * np.pi * 100 * t)
    signal_length = len(source_signal)

# Set source and receiver positions
source_pos = (50, 50, 50)  # Center of the room
receiver_pos = (80, 80, 80)

# Initialize recorded signal array
recorded_signal = np.zeros(steps)

# Set mode from command line argument (SIM or VISUALIZE)
if len(sys.argv) > 2:
    mode = sys.argv[2]
else:
    mode = "SIM"

# Visualization setup for 'VISUALIZE' mode
if mode == "VISUALIZE":
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((nx, ny)), cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(img)

# Time-stepping loop
for t in tqdm(range(steps), desc="Simulation progress"):
    # Get current material properties
    c, rho = get_material_properties(material)

    # Update velocities using central difference for pressure
    velocity_x[1:-1, 1:-1, 1:-1] += (dt / (rho[1:-1, 1:-1, 1:-1] * dx)) * (
        pressure_current[2:, 1:-1, 1:-1] - pressure_current[1:-1, 1:-1, 1:-1]
    )
    velocity_y[1:-1, 1:-1, 1:-1] += (dt / (rho[1:-1, 1:-1, 1:-1] * dx)) * (
        pressure_current[1:-1, 2:, 1:-1] - pressure_current[1:-1, 1:-1, 1:-1]
    )
    velocity_z[1:-1, 1:-1, 1:-1] += (dt / (rho[1:-1, 1:-1, 1:-1] * dx)) * (
        pressure_current[1:-1, 1:-1, 2:] - pressure_current[1:-1, 1:-1, 1:-1]
    )

    # Update pressures using central difference for velocities
    pressure_next = (
        2 * pressure_current[1:-1, 1:-1, 1:-1]
        - pressure_previous[1:-1, 1:-1, 1:-1]
        + (dt**2 / (rho[1:-1, 1:-1, 1:-1] * dx**2))
        * (
            (
                pressure_current[2:, 1:-1, 1:-1]
                - 2 * pressure_current[1:-1, 1:-1, 1:-1]
                + pressure_current[:-2, 1:-1, 1:-1]
            )
            + (
                pressure_current[1:-1, 2:, 1:-1]
                - 2 * pressure_current[1:-1, 1:-1, 1:-1]
                + pressure_current[1:-1, :-2, 1:-1]
            )
            + (
                pressure_current[1:-1, 1:-1, 2:]
                - 2 * pressure_current[1:-1, 1:-1, 1:-1]
                + pressure_current[1:-1, 1:-1, :-2]
            )
        )
    )

    # Apply losses
    pressure_next = apply_losses(pressure_next, material)

    # Inject source signal
    if t < signal_length:
        pressure_next = inject_source(pressure_next, source_signal, t, source_pos)

    # Record signal at receiver
    recorded_signal[t] = pressure_current[
        receiver_pos[0], receiver_pos[1], receiver_pos[2]
    ]

    # Clamp pressure to avoid numerical overflow
    pressure_next = clamp_pressure(pressure_next)

    # Update pressure fields (advance in time)
    pressure_previous = pressure_current.copy()
    pressure_current[1:-1, 1:-1, 1:-1] = pressure_next

    # Real-time visualization in 'VISUALIZE' mode
    if mode == "VISUALIZE" and t % 10 == 0:
        img.set_data(pressure_current[:, :, 50])  # Horizontal slice at z=50
        plt.draw()
        plt.pause(0.01)

# Plot the recorded signal at the receiver
plt.figure()
plt.plot(recorded_signal)
plt.title("Recorded Signal at Receiver")
plt.xlabel("Time Steps")
plt.ylabel("Pressure")
plt.show()
