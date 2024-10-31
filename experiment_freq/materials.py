import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm
from scipy.io import wavfile
from numba import jit, prange

c = 343  # speed of sound in air (m/s)
dx = 0.1  # spatial step (m)
dt = dx / (c * math.sqrt(2))  # time step, satisfying CFL condition

# Material properties
materials_dict = {"air": 0, "wood": 1, "absorber": 2}
reflection_coefficients = [0.5, 0.2, 0.1]  # reflection values for air, wood, absorber
speed_factors = [1.0, 0.7, 0.5]  # wave speed factor for air, wood, absorber
attenuation_factors = [0.999, 0.995, 0.99]  # propagation loss for air, wood, absorber

dim_x, dim_y = 500, 500
u = np.zeros((3, dim_x, dim_y))
k = np.zeros((dim_x, dim_y))

# Material grid (integers for each material type)
materials = np.full(
    (dim_x, dim_y), materials_dict["air"], dtype=np.int32
)  # Default to air
materials[20:100, 40:60] = materials_dict["wood"]  # Wood zone
materials[:, :10] = materials_dict["absorber"]  # Left boundary absorber (PML)
materials[:, -10:] = materials_dict["absorber"]  # Right boundary absorber (PML)
materials[:10, :] = materials_dict["absorber"]  # Top boundary absorber (PML)
materials[-10:, :] = materials_dict["absorber"]  # Bottom boundary absorber (PML)

# Precompute gamma and common terms for each material
gamma_values = [(1 - r) / (1 + r) for r in reflection_coefficients]
common_terms = [
    gamma * ((c * dt * speed_factors[i]) / (2 * dx))
    for i, gamma in enumerate(gamma_values)
]

# Source and recording positions
source_x, source_y = 60, 60
record_x, record_y = 400, 400

# Material color map
material_colors = ["lightblue", "tan", "grey"]
cmap = mcolors.ListedColormap(material_colors)
bounds = [0, 1, 2, 3]
norm = mcolors.BoundaryNorm(bounds, cmap.N)


def compute_neighbours():
    k[1:-1, 1:-1] = 4  # Inner cells have 4 neighbors
    k[0, :] = k[-1, :] = k[:, 0] = k[:, -1] = 3  # Edge cells
    k[0, 0] = k[0, -1] = k[-1, 0] = k[-1, -1] = 2  # Corner cells


@jit(nopython=True, parallel=True)
def step(u, k, materials, common_terms, attenuation_factors):
    for x in prange(1, dim_x - 1):
        for y in range(1, dim_y - 1):
            mat_idx = materials[x, y]  # Get the material index
            neighbours = (
                u[1, x - 1, y] + u[1, x + 1, y] + u[1, x, y - 1] + u[1, x, y + 1]
            )
            # Rigid update with propagation loss
            u[0, x, y] = (
                (2 - 0.5 * k[x, y]) * u[1, x, y] + 0.5 * neighbours - u[2, x, y]
            ) * attenuation_factors[mat_idx]

            # Apply material-specific reflection/transmission loss
            local_common_term = common_terms[mat_idx] * (4 - k[x, y])
            numerator = u[0, x, y] + local_common_term * u[2, x, y]
            denominator = 1 + local_common_term
            u[0, x, y] = numerator / denominator


def display(fig, ax, im_wave, im_material, t):
    """
    Update the plot with new wave data for a given time step t.
    """
    im_wave.set_array(u[0])  # Update wave plot with new values
    ax.set_title(f"Wave Propagation - Time Step {t}")
    plt.pause(0.05)  # Pause to update animation


def main():
    # Load source signal from a .wav file
    sample_rate, input_signal = wavfile.read("samples/1.wav")
    input_signal = input_signal / np.max(np.abs(input_signal))  # Normalize

    # Calculate the sample interval based on dt and the input signal's sample rate
    input_interval = int(sample_rate * dt)
    num_sim_steps = int(len(input_signal) * input_interval)

    # Prepare to record signal at a specific point
    recorded_signal = []

    # Compute neighbors once
    compute_neighbours()

    # Initialize the display figure and axis
    fig, ax = plt.subplots()

    # Plot material layer (static) with full opacity
    im_material = ax.imshow(materials, cmap=cmap, norm=norm, alpha=1.0)
    plt.colorbar(
        im_material, ax=ax, boundaries=bounds, ticks=[0, 1, 2], label="Materials"
    )

    # Plot wave layer (dynamic) with partial transparency
    im_wave = ax.imshow(
        u[0], cmap="viridis", interpolation="nearest", vmin=-1, vmax=1, alpha=0.6
    )
    plt.colorbar(im_wave, ax=ax, label="Wave Amplitude")
    ax.set_title("Wave Propagation with Material Overlay")

    # Time loop with tqdm for progress indication
    for t in tqdm.tqdm(range(num_sim_steps), desc="Simulating"):
        # Set source signal at specified location if it's within bounds of input signal
        if t // input_interval < len(input_signal):
            u[0, source_x, source_y] = input_signal[t // input_interval]

        u[2], u[1] = u[1], u[0]  # Rotate arrays for next step
        step(u, k, materials, common_terms, attenuation_factors)

        # Record the signal at the recording location
        recorded_signal.append(u[0, record_x, record_y])

        # Update visualization every 100 steps
        if t % 100 == 0:
            display(fig, ax, im_wave, im_material, t)

    # Save recorded signal as a .wav file
    recorded_signal = np.array(recorded_signal) * 32767  # Scale for int16 range
    recorded_signal = recorded_signal.astype(np.int16)  # Convert to int16 format
    wavfile.write("samples/1_recorded.wav", sample_rate, recorded_signal)

    plt.show()  # Show the final plot


if __name__ == "__main__":
    main()
