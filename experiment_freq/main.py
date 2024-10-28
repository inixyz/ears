import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm
from scipy.io import wavfile
from scipy.signal import resample_poly
from numba import jit, prange
from matplotlib.colors import ListedColormap

# Parameters
c = [343, 344]
dx = 0.01
dt = dx / (max(c) * math.sqrt(2))
print(f"dx: {dx}, dt: {dt}")

dim_x, dim_y = 500, 500
u = np.zeros((3, dim_x, dim_y))
k = np.zeros((dim_x, dim_y))
m = np.zeros((dim_x, dim_y))

# Define material region
m[100:300, 100:300] = 1  # Material with different properties

# Source and recording positions
source_x, source_y = 200, 350
record_x, record_y = 70, 70

# Define colormaps for each material type
cmap_air = plt.cm.viridis  # Colormap for air (m == 0)
cmap_material = plt.cm.plasma  # Colormap for material (m == 1)


def compute_neighbours():
    k[1:-1, 1:-1] = 4  # Inner cells have 4 neighbors
    k[0, :] = k[-1, :] = k[:, 0] = k[:, -1] = 3  # Edge cells
    k[0, 0] = k[0, -1] = k[-1, 0] = k[-1, -1] = 2  # Corner cells


@jit(nopython=True, parallel=True)
def step(u, k, c, m):
    for x in prange(1, dim_x - 1):
        for y in range(1, dim_y - 1):
            neighbours = (
                u[1, x - 1, y] + u[1, x + 1, y] + u[1, x, y - 1] + u[1, x, y + 1]
            )
            c_mat = c[int(m[x, y])]
            u[0, x, y] = (c_mat**2 * dt**2 / dx**2) * neighbours - u[2, x, y]


def display(fig, ax, im_air, im_material, t):
    """
    Update the plot with new wave data for a given time step t.
    """
    # Mask for different materials
    air_wave = np.ma.masked_where(m == 1, u[0])  # Air wave data (m == 0)
    material_wave = np.ma.masked_where(m == 0, u[0])  # Material wave data (m == 1)

    # Update each plot for respective material areas
    im_air.set_array(air_wave)
    im_material.set_array(material_wave)

    ax.set_title(f"Wave Propagation - Time Step {t}")
    plt.pause(0.05)


def main():
    sample_rate, input_signal = wavfile.read("samples/1.wav")
    input_signal = input_signal / np.max(np.abs(input_signal))  # Normalize

    target_sample_rate = int(1 / dt)
    gcd = np.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd

    # Resample the signal
    input_signal = resample_poly(input_signal, up, down)

    time_steps = len(input_signal)
    print(f"target_sample_rate: {target_sample_rate}, time_steps: {time_steps}")

    recorded_signal = []

    compute_neighbours()

    fig, ax = plt.subplots()
    # Plot separate images for air and material, using different colormaps
    im_air = ax.imshow(u[0], cmap=cmap_air, interpolation="nearest", vmin=-1, vmax=1)
    im_material = ax.imshow(
        u[0], cmap=cmap_material, interpolation="nearest", vmin=-1, vmax=1, alpha=0.7
    )
    plt.colorbar(im_air, ax=ax)
    ax.set_title("Wave Propagation")

    for t in tqdm.tqdm(range(time_steps), desc="Simulating"):
        u[0, source_x, source_y] = input_signal[t]
        u[2], u[1] = u[1], u[0]
        step(u, k, c, m)

        recorded_signal.append(u[0, record_x, record_y])

        if t % 100 == 0:
            display(fig, ax, im_air, im_material, t)

    recorded_signal = np.array(recorded_signal) * 32767
    recorded_signal = recorded_signal.astype(np.int16)
    wavfile.write("samples/1_recorded.wav", target_sample_rate, recorded_signal)

    plt.show()


if __name__ == "__main__":
    main()
