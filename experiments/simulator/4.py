import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dim_x = dim_y = 100
u = np.zeros((3, dim_x, dim_y))


def initialize_source_signal(frequency=440, duration=0.02, sampling_rate=44100):
    t = np.arange(0, duration, 1 / sampling_rate)
    source_signal = np.sin(2 * np.pi * frequency * t)
    return source_signal, len(source_signal)


source_pos_x = int(dim_x / 2)
source_pos_y = int(dim_y / 2)
source_signal, source_signal_len = initialize_source_signal()


def update(frame):
    # advance time
    u[2] = u[1]
    u[1] = u[0]

    # FDTD
    for x in range(1, dim_x - 1):
        for y in range(1, dim_y - 1):
            u[0, x, y] = (
                0.5
                * (u[1, x + 1, y] + u[1, x - 1, y] + u[1, x, y + 1] + u[1, x, y - 1])
                - u[2, x, y]
            )

    # apply_source
    if frame < source_signal_len:
        u[0, source_pos_x, source_pos_y] = source_signal[frame]

    # drawing
    plt.clf()
    plt.imshow(u[0], cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"Frame {frame}")


def main():
    fig, _ = plt.subplots()
    anim = FuncAnimation(fig, update, frames=100, interval=1)
    plt.show()


if __name__ == "__main__":
    main()
