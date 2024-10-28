import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm
from scipy.io import wavfile
from scipy.signal import resample_poly
from numba import jit, prange

c = [343, 1000]
dx = 0.01
dt = dx / (max(c) * math.sqrt(2))
print(f"dx: {dx}, dt: {dt}")

mat_const = [(c_mat**2 * dt**2 / dx**2) for c_mat in c]

dim_x, dim_y = 500, 500
u = np.zeros((3, dim_x, dim_y))
m = np.zeros((dim_x, dim_y))

# world data
m[100:300, 100:300] = 1
source_x, source_y = 200, 350
record_x, record_y = 70, 70


@jit(nopython=True, parallel=True)
def step(u, mat_const):
    for x in prange(1, dim_x - 1):
        for y in range(1, dim_y - 1):
            neighbours = (
                u[1, x - 1, y]
                + u[1, x + 1, y]
                + u[1, x, y - 1]
                + u[1, x, y + 1]
                - 4 * u[1, x, y]
            )
            u[0, x, y] = (
                mat_const[int(m[x, y])] * neighbours + 2 * u[1, x, y] - u[2, x, y]
            )


def display(fig, ax, im_air, im_material, t):
    air_wave = np.ma.masked_where(m == 1, u[0])
    material_wave = np.ma.masked_where(m == 0, u[0])

    im_air.set_array(air_wave)
    im_material.set_array(material_wave)

    ax.set_title(f"Wave Propagation - Time Step {t}")
    plt.pause(0.05)


def scale_back(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    scaled_arr = 2 * (arr - arr_min) / (arr_max - arr_min) - 1
    return scaled_arr


def main():
    sample_rate, input_signal = wavfile.read("samples/impulse.wav")
    input_signal = input_signal / np.max(np.abs(input_signal))  # normalize

    # resample
    target_sample_rate = int(1 / dt)
    gcd = np.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    input_signal = resample_poly(input_signal, up, down)

    time_steps = len(input_signal)
    print(f"target_sample_rate: {target_sample_rate}, time_steps: {time_steps}")

    recorded_signal = []

    fig, ax = plt.subplots()
    im_air = ax.imshow(
        u[0], cmap=plt.cm.viridis, interpolation="nearest", vmin=-1, vmax=1
    )
    im_material = ax.imshow(
        u[0], cmap=plt.cm.plasma, interpolation="nearest", vmin=-1, vmax=1, alpha=0.7
    )
    plt.colorbar(im_air, ax=ax)
    ax.set_title("Wave Propagation")

    for t in tqdm.tqdm(range(time_steps), desc="Simulating"):
        u[0, source_x, source_y] = input_signal[t]
        u[2], u[1] = u[1], u[0]
        step(u, mat_const)

        recorded_signal.append(u[0, record_x, record_y])

        if t % 100 == 0:
            display(fig, ax, im_air, im_material, t)

    recorded_signal = scale_back(recorded_signal)
    recorded_signal = np.array(recorded_signal) * 32767
    recorded_signal = recorded_signal.astype(np.int16)
    wavfile.write("samples/1_recorded.wav", target_sample_rate, recorded_signal)

    plt.show()


if __name__ == "__main__":
    main()
