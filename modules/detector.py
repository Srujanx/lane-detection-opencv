"""
detector.py
-----------
Module 2 — Lane Line Detection

Steps:
  1. Apply a trapezoidal Region of Interest (ROI) mask to remove sky / hood.
  2. Run Probabilistic Hough Line Transform to find line segments.
  3. Classify segments into LEFT lane and RIGHT lane by slope sign.
  4. Average + extrapolate each side into one clean lane line.

All ROI points are expressed as fractions of frame dimensions so the
pipeline works at any resolution.
"""

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "hough_rho": 1,
    "hough_theta": np.pi / 180,
    "hough_threshold": 15,
    "hough_min_line_len": 20,
    "hough_max_line_gap": 150,
    "slope_min_abs": 0.3,

    # Adjusted ROI for forward dashcam
    "roi_bottom_left_x":  0.05,
    "roi_bottom_right_x": 0.95,
    "roi_top_left_x":     0.40,
    "roi_top_right_x":    0.60,
    "roi_top_y":          0.60,
}
# ──────────────────────────────────────────────────────────────────────────────


def apply_roi_mask(edges: np.ndarray) -> np.ndarray:
    """
    Zero out everything outside a trapezoidal road region.

    Args:
        edges: Binary edge image from preprocessor.

    Returns:
        Edge image with only the road region retained.
    """
    H, W = edges.shape[:2]
    mask = np.zeros_like(edges)

    # Build trapezoid vertices from CONFIG ratios
    pts = np.array([[
        (int(CONFIG["roi_bottom_left_x"]  * W), H),
        (int(CONFIG["roi_bottom_right_x"] * W), H),
        (int(CONFIG["roi_top_right_x"]    * W), int(CONFIG["roi_top_y"] * H)),
        (int(CONFIG["roi_top_left_x"]     * W), int(CONFIG["roi_top_y"] * H)),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(edges, mask)


def run_hough(masked_edges: np.ndarray) -> list | None:
    """
    Detect line segments using Probabilistic Hough Line Transform.

    Args:
        masked_edges: ROI-masked edge image.

    Returns:
        Array of line segments [[x1,y1,x2,y2], ...], or None if none found.
    """
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=CONFIG["hough_rho"],
        theta=CONFIG["hough_theta"],
        threshold=CONFIG["hough_threshold"],
        minLineLength=CONFIG["hough_min_line_len"],
        maxLineGap=CONFIG["hough_max_line_gap"],
    )
    return lines


def _slope_intercept(x1, y1, x2, y2) -> tuple[float, float] | None:
    """
    Compute slope and y-intercept of a line segment.
    Returns None if the line is vertical (dx == 0).
    """
    dx = x2 - x1
    if dx == 0:
        return None
    slope = (y2 - y1) / dx
    intercept = y1 - slope * x1
    return slope, intercept


def classify_lines(lines, frame_width: int) -> tuple[list, list]:
    """
    Split Hough line segments into left-lane and right-lane groups
    based on BOTH slope sign AND x-position.

    Args:
        lines:       Output of run_hough().
        frame_width: Width of the frame in pixels.

    Returns:
        (left_lines, right_lines) — each is a list of (slope, intercept) tuples.
    """
    left_lines  = []
    right_lines = []

    if lines is None:
        return left_lines, right_lines

    midpoint = frame_width / 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        result = _slope_intercept(x1, y1, x2, y2)
        if result is None:
            continue
        slope, intercept = result

        # Discard near-horizontal lines
        if abs(slope) < CONFIG["slope_min_abs"]:
            continue

        # Line centre x must also be on the correct side
        x_center = (x1 + x2) / 2

        if slope < 0 and x_center < midpoint:
            left_lines.append((slope, intercept))
        elif slope > 0 and x_center > midpoint:
            right_lines.append((slope, intercept))

    return left_lines, right_lines

def average_lines(
    left_lines: list,
    right_lines: list,
    frame_shape: tuple
) -> tuple:
    """
    Average slope+intercept across all segments on each side,
    then extrapolate to produce one (x1,y1,x2,y2) line per lane.

    The line runs from the bottom of the frame up to the top of the ROI.

    Args:
        left_lines:  List of (slope, intercept) for left candidates.
        right_lines: List of (slope, intercept) for right candidates.
        frame_shape: (H, W) of the frame.

    Returns:
        (left_line, right_line) where each is (x1,y1,x2,y2) or None.
    """
    H, W = frame_shape[:2]
    y_bottom = H
    y_top    = int(CONFIG["roi_top_y"] * H)

    def make_line(side_lines):
        if not side_lines:
            return None
        avg_slope     = np.mean([s for s, _ in side_lines])
        avg_intercept = np.mean([b for _, b in side_lines])
        # x = (y - b) / m
        x_bottom = int((y_bottom - avg_intercept) / avg_slope)
        x_top    = int((y_top    - avg_intercept) / avg_slope)
        return (x_bottom, y_bottom, x_top, y_top)

    return make_line(left_lines), make_line(right_lines)


def detect_lanes(edges: np.ndarray, frame_shape: tuple) -> tuple:
    """
    Full lane detection pipeline: ROI → Hough → classify → average.

    Args:
        edges:       Preprocessed edge image.
        frame_shape: Shape of the original frame (H, W, C).

    Returns:
        (left_line, right_line) — each is (x1,y1,x2,y2) or None if not found.
    """
    masked  = apply_roi_mask(edges)
    lines   = run_hough(masked)
    left_l, right_l = classify_lines(lines, frame_shape[1])
    left_line, right_line = average_lines(left_l, right_l, frame_shape)
    return left_line, right_line
