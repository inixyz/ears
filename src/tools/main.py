import build.ears as ears
import math
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
from scipy.signal import firwin, lfilter


def main():
    dx = 0.01
    sound_speed = 343
    dt = dx / (sound_speed * math.sqrt(3))
    courant = sound_speed * dt / dx
    print(f"{courant=}")

    if sound_speed * dt <= 1 / math.sqrt(3 / dx**2):
        print("stability condition satisfied")

    world = ears.World((1100, 300, 600), (110, 30, 60), (10, 10, 10))

    print(f"center: {world.get_imp((200, 200, 200))}")
    print(f"floor: {world.get_imp((200, 0, 200))}")

    world.fill_imp(1)
    world.fill_courant(courant)

    world.rect_imp((0, 0, 0), (1100, 1, 600), 100000)

    print(f"center: {world.get_imp((200, 200, 200))}")
    print(f"floor: {world.get_imp((200, 0, 200))}")

    slice_z = 298
    source_pos = (400, 260, slice_z)
    receiver_pos = (700, 260, slice_z)
    num_iter = 5000

    receiver_signal = []
    source_signal = []

    # === FIR Low-pass Filtered Impulse Setup ===
    original_sr = int(1 / dt)
    nyquist = original_sr / 2
    cutoff_freq = 3400  # Hz
    normalized_cutoff = cutoff_freq / nyquist
    numtaps = 101  # Length of the FIR filter

    fir_filter = firwin(numtaps, normalized_cutoff)
    impulse = np.zeros(numtaps)
    impulse[0] = 1  # Dirac impulse
    delay = (numtaps - 1) // 2
    input_signal = lfilter(fir_filter, 1.0, impulse) * 10  # Scaled for amplitude

    source_steps = len(input_signal)

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
    receiver_signal = receiver_signal[delay:]

    # === File Naming ===
    out_receiver = "./test_courant.wav"

    # Save the signals
    write(out_receiver, original_sr, receiver_signal)

    print(f"Filtered receiver signal saved to {out_receiver} at {original_sr} Hz")


if __name__ == "__main__":
    main()
