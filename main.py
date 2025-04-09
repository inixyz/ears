import build.ears as ears
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    dx = 0.01
    sound_speed = 343
    dt = dx / (sound_speed * math.sqrt(3))
    courant = sound_speed * dt / dx
    print(f"{courant=}")

    if sound_speed * dt <= 1 / math.sqrt(3 / dx**2):
        print("stability condition satisfied")

    world = ears.World((1100, 300, 600), courant, (110, 30, 60), (10, 10, 10))

    slice_z = 298
    source_pos = (290, 150, slice_z)
    num_iter = 2000

    source_steps = 10
    input_signal = np.zeros(source_steps)
    input_signal[0] = 10

    fig, ax = plt.subplots()
    imgs = []

    for t in tqdm(range(num_iter)):
        if t < source_steps:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        sx = source_pos[0] + dx
                        sy = source_pos[1] + dy
                        sz = source_pos[2] + dz
                        amplitude = world.get_t0((sx, sy, sz))
                        amplitude += input_signal[t]
                        world.set_t0((sx, sy, sz), amplitude)

        world.step()
        slice_data = world_get_slice_z(world, slice_z)

        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        title_text = f"Time step {t} - Min: {min_val:.3f}, Max: {max_val:.3f}"

        img = ax.imshow(slice_data, animated=True)
        title = ax.text(
            0.5,
            1.01,
            title_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            animated=True,
        )

        imgs.append([img, title])

    anim = animation.ArtistAnimation(
        fig, imgs, interval=50, blit=True, repeat_delay=1000
    )
    anim.save("samples/rir_sim_3x3x3.mp4")


if __name__ == "__main__":
    main()
