import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


# Define the filter function
def apply_lowpass_filter(signal, fs, cutoff_freq=4000, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff_freq / nyq
    b, a = butter(order, norm_cutoff, btype="low")
    return filtfilt(b, a, signal)


# Main program
def main():
    parser = argparse.ArgumentParser(
        description="Apply a low-pass filter to a WAV file."
    )
    parser.add_argument("input_wav", help="Path to the input WAV file")
    parser.add_argument("output_wav", help="Path to save the filtered WAV file")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=4000.0,
        help="Cutoff frequency in Hz (default: 4000)",
    )
    parser.add_argument(
        "--order", type=int, default=4, help="Order of the filter (default: 4)"
    )

    args = parser.parse_args()

    # Read input WAV file
    fs, data = wavfile.read(args.input_wav)

    # Handle stereo by filtering each channel
    if data.ndim == 2:
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = apply_lowpass_filter(
                data[:, ch], fs, args.cutoff, args.order
            )
    else:
        filtered_data = apply_lowpass_filter(data, fs, args.cutoff, args.order)

    # Clip and convert to original dtype
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        filtered_data = np.clip(filtered_data, info.min, info.max)
        filtered_data = filtered_data.astype(data.dtype)
    else:
        # If float (e.g., float32), just cast directly
        filtered_data = filtered_data.astype(data.dtype)

    # Write to output WAV file
    wavfile.write(args.output_wav, fs, filtered_data)


if __name__ == "__main__":
    main()
