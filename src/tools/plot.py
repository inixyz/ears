import argparse
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def plot_amplitude_time(file):
    sr, audio = wavfile.read(file)
    time = np.linspace(0, len(audio) / sr, len(audio))

    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.grid()
    plt.plot(time, audio)
    plt.show()


def plot_freq_spectrum(file):
    sr, audio = wavfile.read(file)

    x, y = np.fft.rfftfreq(len(audio), d=1 / sr), np.fft.rfft(audio)
    y = 20 * np.log10(np.abs(y) + 1e-10)  # move magnitude to dbFS

    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude [dB]")
    plt.grid()
    plt.plot(x, y)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("type")
    args = parser.parse_args()

    match args.type:
        case "amplitude_time":
            plot_amplitude_time(args.in_path)
        case "freq_spectrum":
            plot_freq_spectrum(args.in_path)


if __name__ == "__main__":
    main()
