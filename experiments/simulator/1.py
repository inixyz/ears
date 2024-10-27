import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys
from scipy.io.wavfile import read as wavread
from tqdm import tqdm

# Constants for air and wood
c_air = 343  # Speed of sound in air (m/s)
c_wood = 3300  # Speed of sound in wood (m/s)
rho_air = 1.21  # Density of air (kg/m^3)
rho_wood = 600  # Density of wood (kg/m^3)

# Simulation parameters
dx = 0.1  # Spatial step (m)
c_max = max(c_air, c_wood)
dt = dx / (np.sqrt(3) * c_max)  # Adjust dt based on CFL condition
size_x, size_y, size_z = 100, 100, 100  # Grid dimensions
max_time = 2000  # Number of time steps

# Create 3D arrays for current and previous acoustic pressure, velocity components, and material properties
pressure_current = np.zeros(
    (size_x, size_y, size_z)
)  # Acoustic pressure at current time step
pressure_previous = np.zeros_like(
    pressure_current
)  # Acoustic pressure at previous time step
velocity_x = np.zeros_like(pressure_current)  # Velocity components
velocity_y = np.zeros_like(pressure_current)
velocity_z = np.zeros_like(pressure_current)

# Material grid (1=air, 2=wood)
material = np.ones((size_x, size_y, size_z))  # Initially air everywhere
material[40:60, 40:60, 40:60] = 2  # Example wood block in the center


# Get sound speed and density based on the material type
def get_material_properties(material):
    c = np.where(material == 1, c_air, c_wood)
    rho = np.where(material == 1, rho_air, rho_wood)
    return c, rho


# Source and receiver positions
source_pos = (50, 50, 50)
receiver_pos = (20, 20, 20)


# Function to initialize the source signal (sinusoidal or from .wav file)
def initialize_source_signal(
    filename=None, frequency=440, duration=0.02, sampling_rate=44100
):
    if filename:
        # Load from .wav file
        data, sr = sf.read(filename)
        if len(data.shape) > 1:
            data = data[:, 0]  # Take only one channel if stereo
        return data, len(data), sr
    else:
        # Default: Sinusoidal signal
        t = np.arange(0, duration, 1 / sampling_rate)
        source_signal = np.sin(2 * np.pi * frequency * t)
        return source_signal, len(source_signal), sampling_rate


# Initialize the source signal based on command line argument
if len(sys.argv) > 1 and sys.argv[1].lower() == "sim":
    mode = "SIM"
elif len(sys.argv) > 1 and sys.argv[1].lower() == "visualize":
    mode = "VISUALIZE"
else:
    print("Please provide 'SIM' or 'VISUALIZE' as the mode.")
    sys.exit(1)

if len(sys.argv) > 2:
    source_signal, signal_length, sampling_rate = initialize_source_signal(sys.argv[2])
else:
    source_signal, signal_length, sampling_rate = initialize_source_signal()


# Calculate pressure losses due to propagation (air, wood)
def apply_losses(pressure, material):
    # Simple propagation loss (frequency-dependent losses can be modeled separately)
    loss_air = 0.9999  # Air has small losses
    loss_wood = 0.98  # Wood has larger losses
    # Ensure the dimensions match
    pressure *= np.where(material[2:-2, 2:-2, 2:-2] == 1, loss_air, loss_wood)
    return pressure


# Array to store the recorded signal at the receiver
recorded_signal = np.zeros(max_time)

# Visualization setup for 'VISUALIZE' mode
if mode == "VISUALIZE":
    fig, ax = plt.subplots()
    img = ax.imshow(pressure_current[:, :, 50], cmap="jet", vmin=-0.01, vmax=0.01)
    plt.colorbar(img)
    plt.title("Acoustic Pressure - Horizontal Slice")
    plt.xlabel("X")
    plt.ylabel("Y")

# FDTD time-stepping loop
time_range = range(1, max_time - 1)
if mode == "SIM":
    time_range = tqdm(time_range)  # Add loading bar with tqdm


# Function to clamp pressure values to prevent overflow
def clamp_pressure(pressure, min_val=-1e5, max_val=1e5):
    return np.clip(pressure, min_val, max_val)


# Recalculate the time-stepping loop
for t in time_range:
    # Get current material properties (speed of sound and density)
    c, rho = get_material_properties(material)

    # Compute velocity updates
    velocity_x[2:-2, 2:-2, 2:-2] += (dt / (rho[2:-2, 2:-2, 2:-2] * dx)) * (
        pressure_current[3:-1, 2:-2, 2:-2] - pressure_current[2:-2, 2:-2, 2:-2]
    )
    velocity_y[2:-2, 2:-2, 2:-2] += (dt / (rho[2:-2, 2:-2, 2:-2] * dx)) * (
        pressure_current[2:-2, 3:-1, 2:-2] - pressure_current[2:-2, 2:-2, 2:-2]
    )
    velocity_z[2:-2, 2:-2, 2:-2] += (dt / (rho[2:-2, 2:-2, 2:-2] * dx)) * (
        pressure_current[2:-2, 2:-2, 3:-1] - pressure_current[2:-2, 2:-2, 2:-2]
    )

    # Compute pressure updates using both current and previous time steps
    pressure_next = (
        2 * pressure_current[2:-2, 2:-2, 2:-2]
        - pressure_previous[2:-2, 2:-2, 2:-2]
        + (rho[2:-2, 2:-2, 2:-2] * c[2:-2, 2:-2, 2:-2] ** 2)
        * dt**2
        * (
            (pressure_current[3:-1, 2:-2, 2:-2] - pressure_current[2:-2, 2:-2, 2:-2])
            / dx
            + (pressure_current[2:-2, 3:-1, 2:-2] - pressure_current[2:-2, 2:-2, 2:-2])
            / dx
            + (pressure_current[2:-2, 2:-2, 3:-1] - pressure_current[2:-2, 2:-2, 2:-2])
            / dx
        )
    )

    # Apply propagation losses
    pressure_next = apply_losses(pressure_next, material)

    # Inject source signal at the source position
    if t < signal_length:
        pressure_next[source_pos[0], source_pos[1], source_pos[2]] += source_signal[t]

    # Record the signal at the receiver position
    recorded_signal[t] = pressure_current[
        receiver_pos[0], receiver_pos[1], receiver_pos[2]
    ]

    # Clamp the pressure values to avoid overflow
    pressure_next = clamp_pressure(pressure_next)

    # Update pressure fields (advance in time)
    pressure_previous = pressure_current.copy()
    pressure_current[2:-2, 2:-2, 2:-2] = pressure_next.copy()

    # Real-time visualization for 'VISUALIZE' mode
    if mode == "VISUALIZE" and t % 10 == 0:  # Update every 10 iterations
        img.set_data(pressure_current[:, :, 50])  # Horizontal slice at z = 50
        plt.draw()
        plt.pause(0.01)

# Plot source signal and recorded signal
time = np.arange(max_time) * dt

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(source_signal)
plt.title("Source Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(time, recorded_signal)
plt.title("Recorded Signal at Receiver")
plt.xlabel("Time (s)")
plt.ylabel("Acoustic Pressure")

plt.tight_layout()
plt.show()
