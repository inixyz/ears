import numpy as np
import pygame
import math

# Simulation parameters
dim_x, dim_y = 21, 21  # Smaller grid for readability
u = np.zeros((3, dim_x, dim_y))

c = 343  # speed of sound in air (m/s)
dx = 0.1  # spatial step (m)
dt = dx / (c * math.sqrt(2))  # time step, satisfying CFL condition

# Initial wave source
u[0, dim_x // 2, dim_y // 2] = 1.0  # Initial pulse at the center

# Pygame setup
pygame.init()
screen_scale = 80  # Larger scale for readability
screen = pygame.display.set_mode((dim_x * screen_scale, dim_y * screen_scale))
pygame.display.set_caption("Wave Propagation with Amplitude Values")
font = pygame.font.Font(None, 20)  # Font for rendering amplitude values


def step():
    for x in range(1, dim_x - 1):
        for y in range(1, dim_y - 1):
            u[0, x, y] = (
                0.5
                * (u[1, x + 1, y] + u[1, x - 1, y] + u[1, x, y + 1] + u[1, x, y - 1])
                - u[2, x, y]
            )


def update_display():
    screen.fill((0, 0, 0))  # Clear screen

    for x in range(dim_x):
        for y in range(dim_y):
            # Map u[0] values to color intensity for visualization
            intensity = int((u[0, x, y] + 1) * 127.5)  # scale to 0-255
            color = (intensity, intensity, 255)  # blue tint

            # Draw the cell background
            pygame.draw.rect(
                screen,
                color,
                (x * screen_scale, y * screen_scale, screen_scale, screen_scale),
            )

            # Render and draw amplitude value as text on each cell
            amplitude_text = font.render(f"{u[0, x, y]:.2f}", True, (255, 255, 255))
            text_rect = amplitude_text.get_rect(
                center=(
                    x * screen_scale + screen_scale / 2,
                    y * screen_scale + screen_scale / 2,
                )
            )
            screen.blit(amplitude_text, text_rect)

    pygame.display.flip()  # Update the display


def main():
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Execute one step when spacebar is pressed
                    u[2], u[1] = u[1], u[0]  # Rotate arrays for next step
                    step()
                    update_display()  # Update the display after each step

        clock.tick(30)  # Control the frame rate for event loop

    pygame.quit()


# Run the main function
main()
