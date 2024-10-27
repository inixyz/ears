import numpy as np
import pygame
import sys

# Constants
dim_x = dim_y = 200
u = np.zeros((3, dim_x, dim_y))


def initialize_source_signal(frequency=440, duration=0.02, sampling_rate=44100):
    t = np.arange(0, duration, 1 / sampling_rate)
    source_signal = np.sin(2 * np.pi * frequency * t)
    return source_signal, len(source_signal)


source_pos_x = int(dim_x / 2)
source_pos_y = int(dim_y / 2)
source_signal, source_signal_len = initialize_source_signal()


# Colors and pygame setup
def map_value_to_color(value, vmin=-1, vmax=1):
    """Map a value to a color (blue to yellow gradient)."""
    norm_value = (value - vmin) / (vmax - vmin)  # Normalize to [0, 1]
    color = pygame.Color(0)  # Initialize as black
    color.hsva = (240 - 240 * norm_value, 100, 100)  # blue to yellow
    return color


def update_wave(frame):
    # Advance time
    u[2] = u[1]
    u[1] = u[0]

    # FDTD
    for x in range(1, dim_x - 1):
        for y in range(1, dim_y - 1):
            u[0, x, y] = (
                0.5
                * (u[1, x + 1, y] + u[1, x - 1, y] + u[1, x, y + 1] + u[1, x, y - 1])
                - u[2, x, y]
            )

    # Apply source
    if frame < source_signal_len:
        u[0, source_pos_x, source_pos_y] = source_signal[frame]


def draw_grid(screen, scale=5):
    """Draws the 2D wave simulation on the pygame screen."""
    for x in range(dim_x):
        for y in range(dim_y):
            color = map_value_to_color(u[0, x, y])
            pygame.draw.rect(
                screen, color, pygame.Rect(x * scale, y * scale, scale, scale)
            )


def main():
    pygame.init()
    scale = 5  # Scale each pixel by this factor
    screen = pygame.display.set_mode((dim_x * scale, dim_y * scale))
    frame = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        update_wave(frame)

        screen.fill((0, 0, 0))  # Clear screen with black
        draw_grid(screen, scale)
        pygame.display.flip()  # Update the display

        frame += 1

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
