import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve


def load_audio(file_path, target_sr=None):
    """Loads an audio file (MP3 or WAV) and resamples if needed."""
    audio, sr = librosa.load(
        file_path, sr=target_sr, mono=False
    )  # Keep stereo if present
    return audio.T, sr  # Transpose to match (samples, channels) format


def apply_impulse_response(sound_file, impulse_response_file, output_file):
    # Load the sound file (MP3/WAV) and keep its sample rate
    sound, sr_sound = load_audio(sound_file)

    # Load the impulse response and resample if needed
    ir, sr_ir = load_audio(impulse_response_file, target_sr=sr_sound)

    # If impulse response is stereo, take only the first channel
    if len(ir.shape) > 1:
        ir = ir[:, 0]

    # If input sound is stereo, apply convolution per channel
    if len(sound.shape) > 1:
        processed = np.zeros((sound.shape[0] + len(ir) - 1, sound.shape[1]))
        for i in range(sound.shape[1]):
            processed[:, i] = fftconvolve(sound[:, i], ir, mode="full")
    else:
        processed = fftconvolve(sound, ir, mode="full")

    # Normalize to avoid clipping
    processed = processed / np.max(np.abs(processed))

    # Save output as WAV
    sf.write(output_file, processed, sr_sound)
    print(f"Processed audio saved as {output_file}")


# Example usage
apply_impulse_response(
    "input.mp3", "bras/scene1_RIR_Absorbing_LS1_MP1.wav", "samples/output_bras.wav"
)
# apply_impulse_response(
#     "input.mp3", "bras/scene1_RIR_Absorbing_LS1_MP1.wav", "output.wav"
# )
