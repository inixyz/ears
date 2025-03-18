import build.ears as ears
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def gaussian_pulse(t, t0=20, sigma=5, amplitude=20):
    """Generate a Gaussian pulse centered at t0 with standard deviation sigma."""
    return amplitude * np.exp(-((t - t0) ** 2) / (2 * sigma**2))


def main():
    # Set world size to 25x25x25
    world_size = ears.Vec3i(100, 50, 100)
    dx = 0.5  # Grid spacing
    source_position = ears.Vec3i(50, 25, 50)  # Source centered in the smaller grid
    world = ears.World(world_size, dx, ears.Vec3i(10, 5, 10), ears.Vec3i(10, 10, 10))

    # Visualization setup
    fig, ax = plt.subplots()
    slice_z = 50  # Z-plane slice to visualize (middle of 25)
    initial_data = np.zeros((100, 50))  # Initialize empty data array
    image = ax.imshow(initial_data, cmap="RdBu", animated=True)

    ax.set_title("Acoustic Wave Propagation (Z-Slice)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add colorbar (but only update limits dynamically)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    t = 0  # Time counter for Gaussian pulse application

    def update(frame):
        nonlocal t

        # Apply Gaussian pulse at the source for the first few steps
        if t < 40:
            world.set_t0(
                source_position, gaussian_pulse(t, t0=20, sigma=5, amplitude=20)
            )

        world.step()  # Step the simulation by 1 timestep
        t += 1  # Increase time

        # Extract 2D slice at z = 12 using vectorized lookup
        x_indices, y_indices = np.meshgrid(np.arange(100), np.arange(50), indexing="ij")
        get_t0_vec = np.vectorize(
            lambda x, y: world.get_t0(ears.Vec3i(int(x), int(y), slice_z))
        )
        slice_data = get_t0_vec(x_indices, y_indices)

        # Compute min/max dynamically for current world step
        max_abs_value = max(abs(slice_data.min()), abs(slice_data.max()))

        # Update image and color limits
        image.set_data(slice_data)
        image.set_clim(-max_abs_value, max_abs_value)

        return (image,)

    # Faster animation with interval optimization
    ani = animation.FuncAnimation(fig, update, frames=500, interval=1, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
