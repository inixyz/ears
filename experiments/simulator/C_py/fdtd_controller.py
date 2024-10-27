import subprocess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Define the number of time steps (should match the C program)
TIME_STEPS = 2000


# Define the source signal (example: 440 Hz sine wave)
def generate_source_signal():
    frequency = 440  # Hz
    sampling_rate = 1 / (0.01 / 343)  # Time step size
    t = np.arange(0, TIME_STEPS) * (1 / sampling_rate)
    signal = np.sin(2.0 * np.pi * frequency * t)
    return signal


# Run the C program for the FDTD simulation
def run_simulation():
    print("Running the FDTD C simulation...")
    start_time = time.time()
    process = subprocess.Popen(["./fdtd_room_acoustics"], stdout=subprocess.PIPE)
    process.wait()
    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")


# Read the received signal from the output file
def read_received_signal():
    with open("received_signal.txt", "r") as f:
        signal = np.array([float(line.strip()) for line in f])
    return signal


# Main function to control the simulation
def main():
    # Generate the source signal
    source_signal = generate_source_signal()

    # Display the progress with tqdm
    with tqdm(total=TIME_STEPS, desc="Simulation Progress") as pbar:
        run_simulation()
        pbar.update(TIME_STEPS)

    # Read the received signal
    received_signal = read_received_signal()

    # Plot the source and received signals
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(source_signal[: len(received_signal)])
    plt.title("Source Signal")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(received_signal)
    plt.title("Received Signal")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
