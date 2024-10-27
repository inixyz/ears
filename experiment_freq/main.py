import numpy as np
import math
import tqdm
from scipy.io import wavfile
from numba import jit, prange

c = 343  # speed of sound in air (m/s)
dx = 0.1  # spatial step (m)
dt = dx / (c * math.sqrt(2))  # time step, satisfying CFL condition
reflection = 0.5
gama = (1 - reflection) / (1 + reflection)

dim_x, dim_y = 1000, 1000
u = np.zeros((3, dim_x, dim_y))
k = np.zeros((dim_x, dim_y))

# Source and recording positions
source_x, source_y = 50, 50
record_x, record_y = 70, 70

# Precompute the constant term
common_term = gama * ((c * dt) / (2 * dx))


def compute_neighbours():
    k[1:-1, 1:-1] = 4  # Inner cells have 4 neighbors
    k[0, :] = k[-1, :] = k[:, 0] = k[:, -1] = 3  # Edge cells
    k[0, 0] = k[0, -1] = k[-1, 0] = k[-1, -1] = 2  # Corner cells


@jit(nopython=True, parallel=True)
def step(u, k):
    for x in prange(1, dim_x - 1):
        for y in range(1, dim_y - 1):
            neighbours = (
                u[1, x - 1, y] + u[1, x + 1, y] + u[1, x, y - 1] + u[1, x, y + 1]
            )
            # Rigid update
            u[0, x, y] = (
                (2 - 0.5 * k[x, y]) * u[1, x, y] + 0.5 * neighbours - u[2, x, y]
            )
            # Apply loss
            local_common_term = common_term * (4 - k[x, y])
            numerator = u[0, x, y] + local_common_term * u[2, x, y]
            denominator = 1 + local_common_term
            u[0, x, y] = numerator / denominator


def main():
    # Load source signal from a .wav file
    sample_rate, input_signal = wavfile.read("samples/1.wav")
    input_signal = input_signal / np.max(np.abs(input_signal))  # Normalize
    time_steps = len(input_signal)

    # Prepare to record signal at a specific point
    recorded_signal = []

    # Compute neighbors once
    compute_neighbours()

    # Time loop with tqdm for progress indication
    for t in tqdm.tqdm(range(time_steps), desc="Simulating"):
        # Set source signal at specified location
        u[0, source_x, source_y] = input_signal[t]

        u[2], u[1] = u[1], u[0]  # Rotate arrays for next step
        step(u, k)

        # Record the signal at the recording location
        recorded_signal.append(u[0, record_x, record_y])

    # Save recorded signal as a .wav file
    recorded_signal = np.array(recorded_signal) * 32767  # Scale for int16 range
    recorded_signal = recorded_signal.astype(np.int16)  # Convert to int16 format
    wavfile.write("samples/1_recorded.wav", sample_rate, recorded_signal)


if __name__ == "__main__":
    main()
