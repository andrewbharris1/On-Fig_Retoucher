#!/usr/bin/env python3
"""
Create synthetic test images for pipeline testing.

Generates:
- A fake "model" image with a simple shape
- A matching R1 file with red circles
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def create_test_images(output_dir: str = "./test_input"):
    """Create a pair of test images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a 6000x9000 image (scaled down for testing)
    # Using 600x900 for faster testing, scale up for real tests
    width, height = 600, 900

    # Create "original" image - white background with gray "model" shape
    original = Image.new("RGB", (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(original)

    # Draw a simple "person" shape (rectangle for body, circle for head)
    body_left = width // 3
    body_right = 2 * width // 3
    body_top = height // 5
    body_bottom = 4 * height // 5

    # Body
    draw.rectangle([body_left, body_top + 50, body_right, body_bottom],
                   fill=(100, 100, 150))

    # Head
    head_center = (width // 2, body_top)
    head_radius = 40
    draw.ellipse([head_center[0] - head_radius, head_center[1] - head_radius,
                  head_center[0] + head_radius, head_center[1] + head_radius],
                 fill=(200, 180, 160))

    # Shadow on floor
    shadow_top = body_bottom - 20
    draw.ellipse([body_left - 30, shadow_top, body_right + 50, body_bottom + 30],
                 fill=(150, 150, 150))

    # Save original
    original_path = output_dir / "TEST01_001_a.jpg"
    original.save(original_path, quality=95)
    print(f"Created: {original_path}")

    # Create R1 file (copy of original with red circles)
    r1_image = original.copy()
    r1_draw = ImageDraw.Draw(r1_image)

    # Draw red circles indicating areas to retouch
    red = (255, 0, 0)

    # Circle on body (fabric area)
    r1_draw.ellipse([body_left + 20, body_top + 100,
                     body_left + 80, body_top + 160],
                    outline=red, width=5)

    # Another circle lower on body
    r1_draw.ellipse([body_right - 80, body_top + 200,
                     body_right - 20, body_top + 260],
                    outline=red, width=5)

    # Save R1
    r1_path = output_dir / "TEST01_001_a_R1.jpg"
    r1_image.save(r1_path, quality=95)
    print(f"Created: {r1_path}")

    # Create a second pair
    original2 = original.copy()
    original2_path = output_dir / "TEST01_002_a.jpg"
    original2.save(original2_path, quality=95)
    print(f"Created: {original2_path}")

    r1_image2 = r1_image.copy()
    r1_path2 = output_dir / "TEST01_002_a_R1.jpg"
    r1_image2.save(r1_path2, quality=95)
    print(f"Created: {r1_path2}")

    print(f"\nTest images created in: {output_dir}")
    print("Run: python main.py validate ./test_input")


if __name__ == "__main__":
    create_test_images()
