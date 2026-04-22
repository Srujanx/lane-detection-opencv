"""
preprocessor.py
---------------
Module 1 — Image Preprocessing Pipeline

Prepares a raw dashcam frame for lane detection by combining:
  - Grayscale conversion
  - Gaussian Blur        (reduces noise)
  - HSV Color Masking    (isolates white and yellow lane markings)
  - Canny Edge Detection (finds strong gradients / lane edges)

All tunable thresholds are in CONFIG — no magic numbers in functions.
"""

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "blur_kernel": 5,
    "canny_low": 30,
    "canny_high": 100,
    "white_s_max": 80,
    "white_v_min": 120,
    "yellow_h_min": 15,
    "yellow_h_max": 35,
    "yellow_s_min": 60,
    "yellow_v_min": 80,
    # CLAHE settings — boosts contrast on low-light/wet roads
    "clahe_clip": 3.0,
    "clahe_grid": 8,
}
# ──────────────────────────────────────────────────────────────────────────────


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Args:
        img: BGR image as numpy array.

    Returns:
        Single-channel grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_blur(gray: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian Blur to reduce noise before edge detection.

    Args:
        gray: Grayscale image.

    Returns:
        Blurred grayscale image.
    """
    k = CONFIG["blur_kernel"]
    return cv2.GaussianBlur(gray, (k, k), 0)


def apply_hsv_mask(img: np.ndarray) -> np.ndarray:
    """
    Isolate white and yellow lane markings using HSV color masking.
    Combining both colors makes detection robust across road conditions.

    Args:
        img: BGR image.

    Returns:
        Binary mask where white/yellow pixels = 255, everything else = 0.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # White mask: low saturation + high brightness
    white_lower = np.array([0,   0,                   CONFIG["white_v_min"]])
    white_upper = np.array([180, CONFIG["white_s_max"], 255])
    white_mask  = cv2.inRange(hsv, white_lower, white_upper)

    # Yellow mask: specific hue range
    yellow_lower = np.array([CONFIG["yellow_h_min"], CONFIG["yellow_s_min"], CONFIG["yellow_v_min"]])
    yellow_upper = np.array([CONFIG["yellow_h_max"], 255, 255])
    yellow_mask  = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Combine both masks
    combined = cv2.bitwise_or(white_mask, yellow_mask)
    return combined


def apply_canny(img: np.ndarray, color_mask: np.ndarray) -> np.ndarray:
    """
    Apply Canny edge detection on the combination of blurred grayscale
    and the HSV color mask, giving stronger lane edges.

    Args:
        img:        BGR image (used for color masking).
        color_mask: Binary mask from apply_hsv_mask().

    Returns:
        Binary edge image.
    """
    gray    = to_grayscale(img)
    blurred = apply_blur(gray)

    # Mask the grayscale with the color mask to focus Canny on lane colors
    masked_gray = cv2.bitwise_and(blurred, blurred, mask=color_mask)

    edges = cv2.Canny(
        masked_gray,
        CONFIG["canny_low"],
        CONFIG["canny_high"]
    )
    return edges


def preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Step 1 — CLAHE to boost contrast (critical for winter/wet roads)
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CONFIG["clahe_clip"],
        tileGridSize=(CONFIG["clahe_grid"], CONFIG["clahe_grid"])
    )
    l     = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Step 2 — Color mask on enhanced image
    color_mask = apply_hsv_mask(enhanced)

    # Step 3 — Canny on enhanced image
    edges = apply_canny(enhanced, color_mask)
    return edges, color_mask
