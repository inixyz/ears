import build.ears as ears
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.io.wavfile import write


def gaussian_pulse(t, center=20, sigma=5, amplitude=20):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * sigma**2))


def world_get_slice_z(world, slice_idx):
    return np.array(
        [
            [
                world.get_t0(ears.Vec3i(x, y, slice_idx))
                for x in range(world.get_size().x)
            ]
            for y in range(world.get_size().y)
        ]
    )


def main():
    dx = 0.02  # m
    # dt = dx / (343 * math.sqrt(3))
    dt = 1 / 44100
    # sample_rate = int(1 / dt)  # Compute correct sample rate
    sample_rate = 44100

    world = ears.World(
        ears.Vec3i(11 * 10 * 5, 6 * 10 * 5, 3 * 10 * 5),
        343 * dt / dx,  # Courant number
        ears.Vec3i(55, 30, 15),
        ears.Vec3i(10, 10, 10),
    )

    slice_z = int(1.5 * 10 * 5)
    num_iter = 1200
    source_pos = ears.Vec3i(3 * 10 * 5, 3 * 10 * 5, slice_z)
    source_iter_duration = 40
    receiver_pos = ears.Vec3i(8 * 10 * 5, 3 * 10 * 5, slice_z)  # Receiver position

    receiver_signal = []

    for t in tqdm(range(num_iter)):
        if t < source_iter_duration:
            world.set_t0(source_pos, gaussian_pulse(t))
        world.step()
        receiver_signal.append(world.get_t0(receiver_pos))  # Record pressure values

    # fig, ax = plt.subplots()
    # imgs = []
    # for t in tqdm(range(num_iter)):
    #     if t == 0:
    #         world.set_t0(source_pos, 1)
    #
    #     world.step()
    #     receiver_signal.append(world.get_t0(receiver_pos))  # Record pressure values
    #
    #     slice_data = world_get_slice_z(world, slice_z)
    #     img = ax.imshow(slice_data, animated=True)
    #     imgs.append([img])
    #
    # # Save animation
    # anim = animation.ArtistAnimation(
    #     fig, imgs, interval=50, blit=True, repeat_delay=1000
    # )
    # anim.save("room.mp4")

    # Normalize and save receiver signal as WAV file
    receiver_signal = np.array(receiver_signal, dtype=np.float32)
    write("receiver_output.wav", sample_rate, receiver_signal)
    print(f"Receiver signal saved to receiver_output.wav at {sample_rate} Hz")


if __name__ == "__main__":
    main()
