"""
annotator.py
------------
Module 3 — Frame Annotation & Visualization

Draws all visual overlays onto the output frame:
  - Semi-transparent green fill between the two detected lanes
  - Left lane line (RED) and right lane line (BLUE)
  - Ground-truth dots (YELLOW) for TuSimple comparison
  - HUD text: lane departure status, vehicle offset, FPS
"""

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "lane_fill_alpha":  0.25,          # transparency of the green lane fill
    "lane_fill_color":  (0, 255, 0),   # BGR green
    "left_line_color":  (0, 0, 255),   # BGR red
    "right_line_color": (255, 0, 0),   # BGR blue
    "line_thickness":   5,
    "gt_dot_color":     (0, 255, 255), # BGR yellow — ground truth dots
    "gt_dot_radius":    4,

    # HUD text
    "font":             cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale":       0.7,
    "font_thickness":   2,
    "text_color":       (255, 255, 255),

    # Departure warning banner
    "warning_color":    (0, 0, 220),   # red
    "ok_color":         (0, 180, 0),   # green
    "banner_height":    40,

    # Offset scale: metres per pixel (standard TuSimple — 3.7m lane / ~700px)
    "xm_per_pix": 3.7 / 700,
}
# ──────────────────────────────────────────────────────────────────────────────


def draw_lane_fill(
    frame: np.ndarray,
    left_line: tuple | None,
    right_line: tuple | None
) -> np.ndarray:
    """
    Draw a semi-transparent filled polygon between left and right lanes.

    Args:
        frame:       BGR frame.
        left_line:   (x1,y1,x2,y2) for the left lane, or None.
        right_line:  (x1,y1,x2,y2) for the right lane, or None.

    Returns:
        Annotated frame (modified copy).
    """
    if left_line is None or right_line is None:
        return frame

    overlay = frame.copy()
    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line

    # Polygon: bottom-left → top-left → top-right → bottom-right
    pts = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], CONFIG["lane_fill_color"])

    alpha  = CONFIG["lane_fill_alpha"]
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return result


def draw_lane_lines(
    frame: np.ndarray,
    left_line: tuple | None,
    right_line: tuple | None
) -> np.ndarray:
    """
    Draw the detected left (RED) and right (BLUE) lane lines.

    Args:
        frame:      BGR frame.
        left_line:  (x1,y1,x2,y2) or None.
        right_line: (x1,y1,x2,y2) or None.

    Returns:
        Frame with lane lines drawn.
    """
    t = CONFIG["line_thickness"]
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(frame, (x1, y1), (x2, y2), CONFIG["left_line_color"], t)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(frame, (x1, y1), (x2, y2), CONFIG["right_line_color"], t)
    return frame


def draw_ground_truth(
    frame: np.ndarray,
    lanes: list,
    h_samples: list
) -> np.ndarray:
    """
    Draw TuSimple ground-truth lane points as yellow dots.
    Useful for visual comparison between GT and prediction.

    Args:
        frame:     BGR frame.
        lanes:     List of lane x-coordinate lists from TuSimple JSON.
        h_samples: List of y-coordinates from TuSimple JSON.

    Returns:
        Frame with yellow GT dots drawn.
    """
    for lane in lanes:
        for x, y in zip(lane, h_samples):
            if x == -2:   # -2 means no lane at this row
                continue
            cv2.circle(
                frame, (int(x), int(y)),
                CONFIG["gt_dot_radius"],
                CONFIG["gt_dot_color"], -1
            )
    return frame


def compute_offset(
    left_line: tuple | None,
    right_line: tuple | None,
    frame_width: int
) -> float | None:
    """
    Calculate the vehicle's lateral offset from lane centre (in metres).
    Positive = vehicle is right of centre; negative = left of centre.

    Args:
        left_line:   Detected left lane (x1,y1,x2,y2) or None.
        right_line:  Detected right lane (x1,y1,x2,y2) or None.
        frame_width: Width of the frame in pixels.

    Returns:
        Offset in metres, or None if either lane is missing.
    """
    if left_line is None or right_line is None:
        return None
    # Use bottom x-coordinates (x1) which are at frame height
    lane_center   = (left_line[0] + right_line[0]) / 2
    vehicle_center = frame_width / 2
    offset_px     = vehicle_center - lane_center
    return offset_px * CONFIG["xm_per_pix"]


def draw_hud(
    frame: np.ndarray,
    left_line: tuple | None,
    right_line: tuple | None,
    fps: float
) -> np.ndarray:
    """
    Draw the heads-up display: lane status banner + info text.

    Args:
        frame:      BGR frame.
        left_line:  Detected left lane or None.
        right_line: Detected right lane or None.
        fps:        Current frames-per-second.

    Returns:
        Frame with HUD overlay.
    """
    H, W = frame.shape[:2]
    offset = compute_offset(left_line, right_line, W)

    # ── Departure warning banner ───────────────────────────────────────────
    bh = CONFIG["banner_height"]
    if offset is None:
        # No lanes detected
        banner_color = (80, 80, 80)
        status_text  = "No Lane Detected"
    elif abs(offset) > 0.3:
        banner_color = CONFIG["warning_color"]
        side = "RIGHT" if offset > 0 else "LEFT"
        status_text  = f"⚠ LANE DEPARTURE — Drifting {side}"
    else:
        banner_color = CONFIG["ok_color"]
        status_text  = "✓ Lane Centred"

    cv2.rectangle(frame, (0, 0), (W, bh), banner_color, -1)
    cv2.putText(
        frame, status_text,
        (10, bh - 10),
        CONFIG["font"], CONFIG["font_scale"],
        CONFIG["text_color"], CONFIG["font_thickness"]
    )

    # ── Info text bottom-left ─────────────────────────────────────────────
    offset_str = f"{offset:.2f}m" if offset is not None else "N/A"
    info = f"Offset: {offset_str}  |  FPS: {fps:.1f}"
    cv2.putText(
        frame, info,
        (10, H - 15),
        CONFIG["font"], CONFIG["font_scale"],
        CONFIG["text_color"], CONFIG["font_thickness"]
    )

    return frame


def annotate(
    frame: np.ndarray,
    left_line: tuple | None,
    right_line: tuple | None,
    fps: float = 0.0,
    gt_lanes: list | None = None,
    h_samples: list | None = None
) -> np.ndarray:
    """
    Full annotation pipeline — call this once per frame.

    Args:
        frame:      BGR input frame.
        left_line:  Detected left lane (x1,y1,x2,y2) or None.
        right_line: Detected right lane (x1,y1,x2,y2) or None.
        fps:        Current FPS for display.
        gt_lanes:   Optional TuSimple ground-truth lanes for overlay.
        h_samples:  Optional TuSimple h_samples for GT overlay.

    Returns:
        Fully annotated BGR frame.
    """
    out = frame.copy()

    # Layer 1 — lane fill polygon
    out = draw_lane_fill(out, left_line, right_line)

    # Layer 2 — ground truth dots (if provided)
    if gt_lanes is not None and h_samples is not None:
        out = draw_ground_truth(out, gt_lanes, h_samples)

    # Layer 3 — detected lane lines
    out = draw_lane_lines(out, left_line, right_line)

    # Layer 4 — HUD
    out = draw_hud(out, left_line, right_line, fps)

    return out
