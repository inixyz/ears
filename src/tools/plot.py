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

    if audio.dtype.kind in "iu":
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val

    if target_sr and target_sr != original_sr:
        gcd = math.gcd(original_sr, target_sr)
        up = target_sr // gcd
        down = original_sr // gcd
        audio = resample_poly(audio, up, down)
        print(f"{file}: new_sr = {target_sr}")
        return target_sr, audio

    return original_sr, audio


def plot_amplitude_time(files, target_sr=None, subplot=False):
    audios = []
    times = []
    labels = []

    max_time = 0
    min_amp, max_amp = float("inf"), float("-inf")

    for file in files:
        sr, audio = load_and_resample(file, target_sr)
        time = np.linspace(0, len(audio) / sr, len(audio))
        audios.append(audio)
        times.append(time)
        labels.append(os.path.basename(file))
        max_time = max(max_time, time[-1])
        min_amp = min(min_amp, np.min(audio))
        max_amp = max(max_amp, np.max(audio))

    if subplot:
        fig, axs = plt.subplots(
            len(files), 1, sharex=True, sharey=True, figsize=(10, 3 * len(files))
        )
        if len(files) == 1:
            axs = [axs]
        for i, ax in enumerate(axs):
            ax.plot(times[i], audios[i])
            ax.set_title(labels[i])
            ax.set_xlim(0, max_time)
            ax.set_ylim(min_amp, max_amp)
            ax.set_ylabel("amplitude")
            ax.grid()
        axs[-1].set_xlabel("time [s]")
        fig.suptitle("amplitude_time")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        for time, audio, label in zip(times, audios, labels):
            plt.plot(time, audio, label=label)
        plt.xlabel("time [s]")
        plt.ylabel("amplitude")
        plt.grid()
        plt.legend()
        plt.title("amplitude_time")
        plt.show()


def plot_freq_spectrum(files, target_sr=None, subplot=False):
    freqs = []
    mags = []
    labels = []

    max_freq = 0
    min_db, max_db = float("inf"), float("-inf")

    for file in files:
        sr, audio = load_and_resample(file, target_sr)
        x = np.fft.rfftfreq(len(audio), d=1 / sr)
        y = 20 * np.log10(np.abs(np.fft.rfft(audio)) + 1e-10)
        freqs.append(x)
        mags.append(y)
        labels.append(os.path.basename(file))
        max_freq = max(max_freq, x[-1])
        min_db = min(min_db, np.min(y))
        max_db = max(max_db, np.max(y))

    if subplot:
        fig, axs = plt.subplots(
            len(files), 1, sharex=True, sharey=True, figsize=(10, 3 * len(files))
        )
        if len(files) == 1:
            axs = [axs]
        for i, ax in enumerate(axs):
            ax.plot(freqs[i], mags[i])
            ax.set_title(labels[i])
            ax.set_xlim(0, max_freq)
            ax.set_ylim(min_db, max_db)
            ax.set_ylabel("magnitude [dB]")
            ax.grid()
        axs[-1].set_xlabel("frequency [Hz]")
        fig.suptitle("frequency_spectrum")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        for x, y, label in zip(freqs, mags, labels):
            plt.plot(x, y, label=label)
        plt.xlabel("frequency [Hz]")
        plt.ylabel("magnitude [dB]")
        plt.grid()
        plt.legend()
        plt.title("frequency_spectrum")
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=["amplitude_time", "freq_spectrum"])
    parser.add_argument("files", nargs="+", help="Input WAV files")
    parser.add_argument("--sample_rate", type=int, help="Optional target sample rate")
    parser.add_argument(
        "--subplot", action="store_true", help="Show each file in a separate subplot"
    )
    args = parser.parse_args()

    if args.type == "amplitude_time":
        plot_amplitude_time(args.files, args.sample_rate, subplot=args.subplot)
    elif args.type == "freq_spectrum":
        plot_freq_spectrum(args.files, args.sample_rate, subplot=args.subplot)


if __name__ == "__main__":
    main()
