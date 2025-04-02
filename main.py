import build.ears as ears
import math
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
from scipy.signal import resample, fftconvolve


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


def generate_exponential_sweep(f_start, f_end, duration, fs, amplitude=1.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    K = duration * math.log(f_end / f_start)
    sweep = amplitude * np.sin(
        2
        * np.pi
        * f_start
        * K
        * (np.exp(t / duration * math.log(f_end / f_start)) - 1)
        / math.log(f_end / f_start)
    )
    return sweep


def generate_inverse_filter(sweep, f_start, f_end, fs):
    # Generate inverse filter by time-reversing the sweep and applying exponential correction
    duration = len(sweep) / fs
    t = np.linspace(0, duration, len(sweep), endpoint=False)
    w = np.exp(t * math.log(f_end / f_start) / duration)
    return (sweep[::-1] / w).astype(np.float32)


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
    source_amplitude = 1.0
    num_iter = 10000

    fs = int(1 / dt)
    sweep_duration = 2.0  # seconds
    f_start = 50  # Hz
    f_end = 2000  # Hz

    sweep = generate_exponential_sweep(
        f_start, f_end, sweep_duration, fs, amplitude=source_amplitude
    )
    inverse_filter = generate_inverse_filter(sweep, f_start, f_end, fs)

    sweep_len = len(sweep)
    receiver_signal = []
    source_signal = []

    for t in tqdm(range(num_iter)):
        if t < sweep_len:
            world.set_t0(source_pos, sweep[t])

        world.step()
        receiver_signal.append(world.get_t0(receiver_pos))
        source_signal.append(world.get_t0(source_pos))

    receiver_signal = np.array(receiver_signal, dtype=np.float32)
    source_signal = np.array(source_signal, dtype=np.float32)

    # Convolve with inverse filter to get impulse response
    ir = fftconvolve(receiver_signal, inverse_filter, mode="full")

    # Resample everything to 44.1 kHz
    original_sr = fs
    target_sr = 44100
    num_samples = int(len(receiver_signal) * target_sr / original_sr)
    num_samples_ir = int(len(ir) * target_sr / original_sr)

    receiver_resampled = resample(receiver_signal, num_samples)
    sweep_resampled = resample(sweep, int(len(sweep) * target_sr / fs))
    ir_resampled = resample(ir, num_samples_ir)

    # Save to disk
    write(
        "samples/exponential_sweep.wav", target_sr, sweep_resampled.astype(np.float32)
    )
    write("samples/rir_exponential_ir.wav", target_sr, ir_resampled.astype(np.float32))

    print("Sweep saved to samples/exponential_sweep.wav")
    print("Impulse response saved to samples/rir_exponential_ir.wav")


if __name__ == "__main__":
    main()
