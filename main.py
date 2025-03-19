import build.ears as ears
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


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
    # x = 11m
    # y = 6m
    # z = 3m

    dx = 0.02  # m
    dt = dx / (343 * math.sqrt(3))
    courant = 343 * dt / dx

    world = ears.World(
        ears.Vec3i(11 * 10 * 5, 6 * 10 * 5, 3 * 10 * 5),
        courant,
        ears.Vec3i(55, 30, 15),
        ears.Vec3i(10, 10, 10),
    )

    slice_z = int(1.5 * 10 * 5)
    num_iter = 300

    source_pos = ears.Vec3i(3 * 10 * 5, 3 * 10 * 5, int(1.5 * 10 * 5))
    source_iter_duration = 40

    fig, ax = plt.subplots()
    imgs = []
    for t in tqdm(range(num_iter)):
        if t < source_iter_duration:
            world.set_t0(source_pos, gaussian_pulse(t))
        world.step()
        slice_data = world_get_slice_z(world, slice_z)
        img = ax.imshow(slice_data, animated=True)
        imgs.append([img])

    anim = animation.ArtistAnimation(
        fig, imgs, interval=50, blit=True, repeat_delay=1000
    )
    anim.save("room.mp4")


if __name__ == "__main__":
    main()
