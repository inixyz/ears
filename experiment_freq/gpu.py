import cupy as cp
import math
import tqdm
from scipy.io import wavfile

# Constants
c = 343  # speed of sound in air (m/s)
dx = 0.1  # spatial step (m)
dt = dx / (c * math.sqrt(2))  # time step, satisfying CFL condition
reflection = 0.5
gama = (1 - reflection) / (1 + reflection)

# Grid dimensions
dim_x, dim_y = 1000, 1000
u = cp.zeros((3, dim_x, dim_y), dtype=cp.float32)
k = cp.zeros((dim_x, dim_y), dtype=cp.float32)

# Source and recording positions
source_x, source_y = 50, 50
record_x, record_y = 70, 70

# Precompute the constant term
common_term = gama * ((c * dt) / (2 * dx))


def compute_neighbours():
    k[1:-1, 1:-1] = 4  # Inner cells have 4 neighbors
    k[0, :] = k[-1, :] = k[:, 0] = k[:, -1] = 3  # Edge cells
    k[0, 0] = k[0, -1] = k[-1, 0] = k[-1, -1] = 2  # Corner cells


def step(u, k):
    neighbours = (
        cp.roll(u[1], shift=1, axis=0)
        + cp.roll(u[1], shift=-1, axis=0)
        + cp.roll(u[1], shift=1, axis=1)
        + cp.roll(u[1], shift=-1, axis=1)
    )
    # Rigid update
    u[0] = (2 - 0.5 * k) * u[1] + 0.5 * neighbours - u[2]

    # Apply loss
    local_common_term = common_term * (4 - k)
    numerator = u[0] + local_common_term * u[2]
    denominator = 1 + local_common_term
    u[0] = numerator / denominator


def main():
    # Load source signal from a .wav file
    sample_rate, input_signal = wavfile.read("samples/1.wav")
    input_signal = input_signal / cp.max(cp.abs(input_signal))  # Normalize
    time_steps = len(input_signal)

    # Prepare to record signal at a specific point
    recorded_signal = []

    # Compute neighbors once (on CPU, as k is simple)
    compute_neighbours()

    # Time loop with tqdm for progress indication
    for t in tqdm.tqdm(range(time_steps), desc="Simulating"):
        # Set source signal at specified location
        u[0, source_x, source_y] = input_signal[t]

        u[2], u[1] = u[1], u[0]  # Rotate arrays for next step
        step(u, k)

        # Record the signal at the recording location
        recorded_signal.append(u[0, record_x, record_y].get())

    # Save recorded signal as a .wav file
    recorded_signal = cp.array(recorded_signal) * 32767  # Scale for int16 range
    recorded_signal = recorded_signal.astype(
        cp.int16
    ).get()  # Convert to int16 and move to CPU
    wavfile.write("samples/1_recorded.wav", sample_rate, recorded_signal)


if __name__ == "__main__":
    main()
