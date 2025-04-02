import build.ears as ears
import math
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
from scipy.signal import resample


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


def generate_band_limited_sinc(fs, duration, f_low, f_high, amplitude=1.0):
    t = np.arange(-duration / 2, duration / 2, 1 / fs)
    # Avoid division by zero at t = 0
    sinc_high = np.sinc(2 * f_high * t)
    sinc_low = np.sinc(2 * f_low * t)
    band_limited = sinc_high - sinc_low
    window = np.hanning(len(band_limited))
    return amplitude * band_limited * window


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

    fs = int(1 / dt)  # Sampling rate in Hz
    sinc_duration = 0.02  # in seconds (~20ms)
    f_low = 50  # Hz
    f_high = 2000  # Hz

    # Generate the band-limited sinc
    sinc_signal = generate_band_limited_sinc(
        fs, sinc_duration, f_low, f_high, amplitude=source_amplitude
    )
    sinc_len = len(sinc_signal)

    receiver_signal = []
    source_signal = []

    for t in tqdm(range(num_iter)):
        if t < sinc_len:
            world.set_t0(source_pos, sinc_signal[t])

        world.step()
        receiver_signal.append(world.get_t0(receiver_pos))
        source_signal.append(world.get_t0(source_pos))

    receiver_signal = np.array(receiver_signal, dtype=np.float32)
    source_signal = np.array(source_signal, dtype=np.float32)

    original_sr = int(1 / dt)
    target_sr = 44100

    # Resample the signals
    num_samples = int(len(receiver_signal) * target_sr / original_sr)
    receiver_signal_resampled = resample(receiver_signal, num_samples)
    source_signal_resampled = resample(source_signal, num_samples)

    # Save both signals
    out_receiver = "samples/rir_simulated_lrs_sinc.wav"
    out_source = "samples/source_signal_lrs_sinc.wav"
    write(out_receiver, target_sr, receiver_signal_resampled.astype(np.float32))
    write(out_source, target_sr, source_signal_resampled.astype(np.float32))

    print(f"Receiver signal saved to {out_receiver} at {target_sr} Hz")
    print(f"Source signal saved to {out_source} at {target_sr} Hz")


if __name__ == "__main__":
    main()
