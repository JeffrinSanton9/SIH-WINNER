"""
generate_figure8_beacon.py

Generates a monochromatic video where a bright white beacon
moves along a figure-8 (Lissajous) path with slight jitter.

Requirements:
    pip install opencv-python numpy

Usage:
    python generate_figure8_beacon.py
"""

import cv2
import numpy as np
import random
import math

# ----------------- Parameters -----------------
WIDTH, HEIGHT = 640, 480       # Video resolution
FPS = 30                       # Frames per second
DURATION_SEC = 10              # Video length in seconds
OUTPUT_FILE = "figure8_beacon.mp4"

# Figure-8 motion parameters
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
AMP_X = 200                    # Horizontal amplitude (pixels)
AMP_Y = 100                    # Vertical amplitude (pixels)
FREQ = 1.0                     # Full figure-8 cycles per second

# Beacon appearance
SPOT_RADIUS = 5                # Beacon radius in pixels

# Jitter (random position noise)
JITTER_X = 2                   # Max jitter in pixels (X)
JITTER_Y = 2                   # Max jitter in pixels (Y)

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
# -----------------------------------------------


def main():
    total_frames = FPS * DURATION_SEC
    out = cv2.VideoWriter(OUTPUT_FILE, FOURCC, FPS, (WIDTH, HEIGHT), isColor=False)

    if not out.isOpened():
        raise RuntimeError("Could not open VideoWriter. Check codec/permissions.")

    print(f"Generating {total_frames} frames -> {OUTPUT_FILE}")

    for i in range(total_frames):
        t = i / FPS  # Time in seconds

        # Figure-8 (Lissajous) path:
        # x = A*sin(2*pi*f*t)
        # y = B*sin(4*pi*f*t)   (double frequency => figure-8)
        bx = CENTER_X + AMP_X * math.sin(2 * math.pi * FREQ * t)
        by = CENTER_Y + AMP_Y * math.sin(4 * math.pi * FREQ * t)

        # Add jitter
        jx = random.randint(-JITTER_X, JITTER_X)
        jy = random.randint(-JITTER_Y, JITTER_Y)
        cx, cy = int(bx + jx), int(by + jy)

        # Create black grayscale frame
        frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

        # Draw bright white beacon
        cv2.circle(frame, (cx, cy), SPOT_RADIUS, 255, -1)

        # Optional: soften edges slightly (more optical-like)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        out.write(frame)

    out.release()
    print("Done.")


if __name__ == "__main__":
    main()