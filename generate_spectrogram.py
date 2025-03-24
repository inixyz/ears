import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


def generate_spectrum(input_path, output_path):
    # Read audio file
    sample_rate, data = wavfile.read(input_path)

    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize to float32 (optional but good practice)
    if data.dtype != np.float32:
        data = data / np.max(np.abs(data))

    # Compute FFT
    N = len(data)
    fft_result = np.fft.rfft(data)
    fft_freq = np.fft.rfftfreq(N, d=1 / sample_rate)
    magnitude = np.abs(fft_result)

    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Plot spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(fft_freq, magnitude_db)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Frequency Spectrum")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Spectrum saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save a frequency spectrum from an audio file."
    )
    parser.add_argument("input_path", type=str, help="Path to input .wav file")
    parser.add_argument(
        "output_path", type=str, help="Path to save spectrum image (e.g. output.png)"
    )
    args = parser.parse_args()

    generate_spectrum(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
