"""
Real-Time Color-Segmentation Masking Application
------------------------------------------------
Detects red cloth in the webcam feed and replaces it with a pre-captured
background so the cloth appears "invisible".

Usage:
    pip install opencv-python numpy
    python red_cloak.py

Controls:
    b  - re-capture the background (step out of frame first!)
    q  - quit
"""

import time

import cv2
import numpy as np

# ---------- Configuration ----------
CAMERA_INDEX = 0
BACKGROUND_FRAMES = 60  # frames averaged/kept when capturing background

# Red wraps around the hue axis in HSV (OpenCV hue range is 0-179),
# so we need two ranges: one near 0 and one near 179.
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

KERNEL = np.ones((3, 3), np.uint8)


def capture_background(cap, num_frames=BACKGROUND_FRAMES):
    """Capture a static background. Person/cloth should be out of frame."""
    print("Capturing background in 3 seconds... step out of the frame!")
    time.sleep(3)

    # Discard warm-up/stale frames (auto-exposure often gives dark or black
    # frames right after the camera starts, which causes black spots later)
    for _ in range(30):
        cap.read()

    frames = []
    for _ in range(num_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError("Could not read from camera.")

    # Median across frames removes noise and any momentary dark/black frames
    background = np.median(np.stack(frames), axis=0).astype(np.uint8)

    # Mirror so it matches the flipped live feed
    background = cv2.flip(background, 1)
    print("Background captured.")
    return background


def build_red_mask(frame):
    """Return a cleaned binary mask where red pixels are white (255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Remove speckle noise, then fill small holes and slightly expand edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.dilate(mask, KERNEL, iterations=1)
    return mask


def apply_cloak(frame, background, mask):
    """Replace masked (red) pixels in frame with the background."""
    inverse_mask = cv2.bitwise_not(mask)

    # Keep everything that is NOT red from the live frame
    current_part = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    # Keep only the red region from the stored background
    background_part = cv2.bitwise_and(background, background, mask=mask)

    return cv2.addWeighted(current_part, 1, background_part, 1, 0)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    background = capture_background(cap)

    print("Running. Press 'b' to re-capture background, 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror view

        # Make sure the background matches the frame size
        if background.shape != frame.shape:
            background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

        mask = build_red_mask(frame)
        output = apply_cloak(frame, background, mask)

        cv2.imshow("Red Cloak", output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            background = capture_background(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()