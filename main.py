import build.ears as ears
import math
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt


def apply_lowpass_filter(signal, fs, cutoff_freq=4000, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff_freq / nyq
    b, a = butter(order, norm_cutoff, btype="low")
    return filtfilt(b, a, signal)


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
    receiver_pos = (809, 150, slice_z)
    num_iter = 10000

    receiver_signal = []
    source_signal = []

    source_steps = 10
    input_signal = np.zeros(source_steps)
    input_signal[0] = 10

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
        receiver_signal.append(world.get_t0(receiver_pos))
        source_signal.append(world.get_t0(source_pos))

    receiver_signal = np.array(receiver_signal, dtype=np.float32)
    original_sr = int(1 / dt)

    # Apply low-pass filter
    filtered_signal = apply_lowpass_filter(receiver_signal, original_sr)

    # Save filtered signal
    out_receiver = "samples/rir_sim_3x3x3_filtered_lowpass_4KHz.wav"
    write(out_receiver, original_sr, filtered_signal.astype(np.float32))
    print(f"Filtered receiver signal saved to {out_receiver} at {original_sr} Hz")


if __name__ == "__main__":
    main()
