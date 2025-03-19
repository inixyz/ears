import build.ears as ears
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
    world = ears.World(
        ears.Vec3i(500, 100, 100), 0.5, ears.Vec3i(50, 10, 10), ears.Vec3i(10, 10, 10)
    )

    slice_z = 50
    num_iter = 100

    source_pos = ears.Vec3i(250, 50, 50)
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
    anim.save("movie.mp4")


if __name__ == "__main__":
    main()
