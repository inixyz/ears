import build.ears as ears
import math
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
from scipy.signal import resample  # for resampling


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

    world = ears.World((1100, 300, 600), courant, (110, 30, 60), (10, 10, 10))

    slice_z = 298
    source_pos = (290, 150, slice_z)
    receiver_pos = (809, 150, slice_z)
    source_amplitude = 200
    num_iter = 10000

    receiver_signal = []

    for t in tqdm(range(num_iter)):
        if t == 0:
            world.set_t0(source_pos, source_amplitude)

        world.step()
        receiver_signal.append(world.get_t0(receiver_pos))

    receiver_signal = np.array(receiver_signal, dtype=np.float32)

    original_sr = int(1 / dt)
    target_sr = 44100

    # Resample the signal
    num_samples = int(len(receiver_signal) * target_sr / original_sr)
    receiver_signal_resampled = resample(receiver_signal, num_samples)

    out_file = "samples/rir_simulated_lrs_test.wav"
    write(out_file, target_sr, receiver_signal_resampled.astype(np.float32))
    print(f"Receiver signal saved to {out_file} at {target_sr} Hz")


if __name__ == "__main__":
    main()
