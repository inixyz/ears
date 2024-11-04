import numpy as np
import pygame
import sys

# Parameters
width, height = 800, 400  # Window size
array_length = 200  # Length of the 1D array

# Initialize the wave array with zeros
wave_array = np.zeros((3, array_length))
m = np.zeros(array_length)

m[: array_length // 2] = 343
m[array_length // 2 :] = 3490
dx = 0.1
dt = dx / max(m)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("1D Wave Visualization")
clock = pygame.time.Clock()

# Scaling factors
x_scale = width / array_length
y_scale = height / 2


def draw_wave(screen, wave_array):
    screen.fill((0, 0, 0))  # Clear screen with black background

    # Draw the wave
    for i in range(1, array_length):
        x1, y1 = (
            int((i - 1) * x_scale),
            int(height / 2 - wave_array[0, i - 1] * y_scale),
        )
        x2, y2 = int(i * x_scale), int(height / 2 - wave_array[0, i] * y_scale)

        # Ensure coordinates are within the display bounds
        x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
        x2, y2 = max(0, min(width - 1, x2)), max(0, min(height - 1, y2))

        pygame.draw.line(screen, (0, 255, 255), (x1, y1), (x2, y2), 2)

    pygame.display.flip()  # Update display


def update():
    wave_array[2, :] = wave_array[1, :]
    wave_array[1, :] = wave_array[0, :]
    for i in range(1, array_length - 1):
        wave_array[0, i] = (
            (m[i] ** 2 * dt**2 / dx**2)
            * (wave_array[1, i - 1] - 2 * wave_array[1, i] + wave_array[1, i + 1])
            + 2 * wave_array[1, i]
            - wave_array[2, i]
        )


# Main loop
wave_array[0, 70] = 0.9
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    update()

    draw_wave(screen, wave_array)  # Draw the updated wave
    clock.tick(60)  # Limit to 60 FPS

pygame.quit()
sys.exit()
