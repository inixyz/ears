import argparse
from scipy.io import wavfile
from scipy.signal import resample_poly
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def load_and_resample(file, target_sr=None):
    original_sr, audio = wavfile.read(file)
    print(f"{file}: sample_rate = {original_sr}")

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
        print(f"{file}: new_sr = {target_sr}")
        return target_sr, audio

    return original_sr, audio


def plot_amplitude_time(files, target_sr=None):
    plt.figure()
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.grid()

    for file in files:
        sr, audio = load_and_resample(file, target_sr)
        time = np.linspace(0, len(audio) / sr, len(audio))
        plt.plot(time, audio, label=os.path.basename(file))

    plt.legend()
    plt.title("amplitude_time")
    plt.show()


def plot_freq_spectrum(files, target_sr=None):
    plt.figure()
    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude [dB]")
    plt.grid()

    for file in files:
        sr, audio = load_and_resample(file, target_sr)
        x = np.fft.rfftfreq(len(audio), d=1 / sr)
        y = 20 * np.log10(np.abs(np.fft.rfft(audio)) + 1e-10)
        plt.plot(x, y, label=os.path.basename(file))

    plt.legend()
    plt.title("frequency_spectrum")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=["amplitude_time", "freq_spectrum"])
    parser.add_argument("files", nargs="+", help="Input WAV files")
    parser.add_argument("--sample_rate", type=int, help="Optional target sample rate")
    args = parser.parse_args()

    if args.type == "amplitude_time":
        plot_amplitude_time(args.files, args.sample_rate)
    elif args.type == "freq_spectrum":
        plot_freq_spectrum(args.files, args.sample_rate)


if __name__ == "__main__":
    main()
