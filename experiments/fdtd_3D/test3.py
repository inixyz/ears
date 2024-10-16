import pygame
import math

# Initialize Pygame
pygame.init()

# Set up display
width, height = 2000, 1600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2D Ray Tracing Simulator")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)


# Define Ray class
class Ray:
    def __init__(self, x, y, angle, speed=4):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.dx = math.cos(angle) * speed
        self.dy = math.sin(angle) * speed

    def move(self):
        self.x += self.dx
        self.y += self.dy

    def reflect(self, normal_angle):
        # Incident vector (ray's current direction)
        incident_vector = pygame.Vector2(self.dx, self.dy)

        # Normal vector from the boundary
        normal_vector = pygame.Vector2(math.cos(normal_angle), math.sin(normal_angle))

        # Reflect the incident vector around the normal vector
        reflected_vector = incident_vector.reflect(normal_vector)

        # Update ray's direction based on the reflection
        self.dx = reflected_vector.x
        self.dy = reflected_vector.y
        self.angle = math.atan2(self.dy, self.dx)

    def draw(self, screen):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), 2)


# Define Boundary class
class Boundary:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, screen):
        pygame.draw.line(screen, RED, (self.x1, self.y1), (self.x2, self.y2), 2)

    def get_normal(self):
        # Calculate normal (perpendicular) of the boundary
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        boundary_angle = math.atan2(dy, dx)
        normal_angle = (
            boundary_angle + math.pi / 2
        )  # Normal is 90 degrees to the boundary
        return normal_angle

    def distance_to_ray(self, ray):
        # Calculate perpendicular distance from the ray to the boundary line segment
        x0, y0 = ray.x, ray.y
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2

        # Line segment's direction
        line_length = math.hypot(x2 - x1, y2 - y1)
        if line_length == 0:
            return float("inf")

        # Projection factor 't' along the line segment
        t = max(
            0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_length**2)
        )

        # Find the nearest point on the boundary
        nearest_x = x1 + t * (x2 - x1)
        nearest_y = y1 + t * (y2 - y1)

        # Calculate the distance between the ray and the nearest point on the boundary
        distance = math.hypot(x0 - nearest_x, y0 - nearest_y)

        return distance

    def check_collision(self, ray, threshold=5):
        # Check if the ray is within the threshold distance from the boundary
        distance = self.distance_to_ray(ray)
        return distance < threshold


# Main simulation loop
def main():
    running = True
    clock = pygame.time.Clock()
    fullscreen = False  # To track fullscreen state

    # Create boundaries (walls)
    boundaries = [
        Boundary(50, 50, 1950, 50),  # Top wall
        Boundary(50, 50, 50, 1550),  # Left wall
        Boundary(1950, 50, 1950, 1550),  # Right wall
        Boundary(50, 1550, 1950, 1550),  # Bottom wall
        Boundary(1600, 400, 1700, 900),  # Middle Wall
    ]

    # Create rays originating from the center
    num_rays = 720
    rays = []
    source_x, source_y = width // 2, height // 2
    for i in range(num_rays):
        angle = i * (2 * math.pi / num_rays)
        rays.append(Ray(source_x, source_y, angle))

    global screen
    while running:
        screen.fill(BLACK)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Fullscreen toggle with Alt + Enter
            if event.type == pygame.KEYDOWN:
                if (
                    event.key == pygame.K_RETURN
                    and pygame.key.get_mods() & pygame.KMOD_ALT
                ):
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode(
                            (width, height), pygame.FULLSCREEN
                        )
                    else:
                        screen = pygame.display.set_mode((width, height))

        # Move and draw rays
        for ray in rays:
            ray.move()

            # Check for collisions with boundaries
            for boundary in boundaries:
                if boundary.check_collision(ray):
                    normal_angle = boundary.get_normal()
                    ray.reflect(normal_angle)

            ray.draw(screen)

        # Draw boundaries
        for boundary in boundaries:
            boundary.draw(screen)

        # Draw the source point
        pygame.draw.circle(screen, YELLOW, (source_x, source_y), 5)

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
