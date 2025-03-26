import argparse
from scipy.io import wavfile
from scipy.signal import resample_poly
import numpy as np
import matplotlib.pyplot as plt
import math


def load_and_resample(file, target_sr=None):
    original_sr, audio = wavfile.read(file)
    print(f"sample_rate = {original_sr}")

    # normalize audio if it's in int format
    if audio.dtype.kind in "iu":
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val

    if target_sr and target_sr != original_sr:
        # compute resample ratio
        gcd = math.gcd(original_sr, target_sr)
        up = target_sr // gcd
        down = original_sr // gcd

        audio = resample_poly(audio, up, down)
        print(f"new_sr = {target_sr}")
        return target_sr, audio

    return original_sr, audio


def plot_amplitude_time(file, target_sr=None):
    sr, audio = load_and_resample(file, target_sr)
    time = np.linspace(0, len(audio) / sr, len(audio))

    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.grid()
    plt.plot(time, audio)
    plt.show()


def plot_freq_spectrum(file, target_sr=None):
    sr, audio = load_and_resample(file, target_sr)

    x, y = np.fft.rfftfreq(len(audio), d=1 / sr), np.fft.rfft(audio)
    y = 20 * np.log10(np.abs(y) + 1e-10)  # move magnitude to dBFS

    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude [dB]")
    plt.grid()
    plt.plot(x, y)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str)
    parser.add_argument("type", type=str, choices=["amplitude_time", "freq_spectrum"])
    parser.add_argument("--sample_rate", type=int)
    args = parser.parse_args()

    match args.type:
        case "amplitude_time":
            plot_amplitude_time(args.in_path, args.sample_rate)
        case "freq_spectrum":
            plot_freq_spectrum(args.in_path, args.sample_rate)


if __name__ == "__main__":
    main()
