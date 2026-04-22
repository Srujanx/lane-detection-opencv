"""
accuracy.py
-----------
TuSimple Official Evaluation Metric

The TuSimple benchmark measures accuracy by checking whether each predicted
lane x-coordinate falls within ±20 pixels of the ground-truth x-coordinate
at every h_sample row.

  Accuracy = (# correct predictions) / (# total annotated points) × 100

A predicted x is "correct" if:  |pred_x - gt_x| ≤ 20  (and gt_x ≠ -2)
"""

import os
import csv
import numpy as np

from utils.tusimple_loader import parse_json, load_frame
from modules.preprocessor  import preprocess
from modules.detector       import detect_lanes

# Width tolerance for a prediction to count as correct (TuSimple standard)
PIXEL_THRESHOLD = 20


def lanes_to_points(
    left_line: tuple | None,
    right_line: tuple | None,
    h_samples: list
) -> list[list[int]]:
    """
    Convert our (x1,y1,x2,y2) lane lines to TuSimple point format:
    a list of x-coordinates — one per h_sample row — for each detected lane.
    Returns -2 at any row where the lane is not available.

    Args:
        left_line:  (x1,y1,x2,y2) for left lane or None.
        right_line: (x1,y1,x2,y2) for right lane or None.
        h_samples:  List of y-values from TuSimple JSON.

    Returns:
        List of two lanes (left, right), each a list of x-coords.
    """
    def interpolate(line, y):
        """Given a line (x1,y1,x2,y2), find x at a given y using linear interp."""
        if line is None:
            return -2
        x1, y1, x2, y2 = line
        if y2 == y1:
            return -2
        # Parametric: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        return int(x)

    left_pts  = [interpolate(left_line,  y) for y in h_samples]
    right_pts = [interpolate(right_line, y) for y in h_samples]
    return [left_pts, right_pts]


def tusimple_accuracy(
    pred_lanes: list[list[int]],
    gt_lanes: list[list[int]],
    h_samples: list[int]
) -> dict:
    """
    Compute TuSimple accuracy for a single frame.

    Args:
        pred_lanes: List of predicted lanes (each: list of x per h_sample).
        gt_lanes:   List of GT lanes from JSON (each: list of x per h_sample).
        h_samples:  Y-coordinates.

    Returns:
        dict with keys: correct (int), total (int), accuracy (float 0-100).
    """
    correct = 0
    total   = 0

    for gt_lane in gt_lanes:
        # Find the closest predicted lane for this GT lane
        best_correct = 0
        for pred_lane in pred_lanes:
            lane_correct = 0
            for gt_x, pred_x in zip(gt_lane, pred_lane):
                if gt_x == -2:           # no GT annotation at this row
                    continue
                total += 1
                if pred_x != -2 and abs(pred_x - gt_x) <= PIXEL_THRESHOLD:
                    lane_correct += 1
            best_correct = max(best_correct, lane_correct)
        correct += best_correct

    acc = (correct / total * 100) if total > 0 else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


def run_evaluation(
    json_path: str,
    base_dir: str,
    save_csv: str | None = None
) -> float:
    """
    Run the full TuSimple evaluation loop over all annotated frames.

    For each frame:
      1. Load the image
      2. Run the classical CV pipeline (preprocess → detect)
      3. Convert detected lines to TuSimple point format
      4. Compute accuracy vs ground truth

    Args:
        json_path: Path to label_data_*.json.
        base_dir:  TuSimple dataset root directory.
        save_csv:  Optional path to save per-frame results as CSV.

    Returns:
        Mean accuracy (%) across all evaluated frames.
    """
    records = parse_json(json_path)
    results = []

    print(f"\n[Evaluation] {len(records)} frames found in {os.path.basename(json_path)}")
    print(f"[Evaluation] Pixel threshold: ±{PIXEL_THRESHOLD}px\n")

    for i, rec in enumerate(records):
        img = load_frame(base_dir, rec["raw_file"])
        if img is None:
            continue

        # Run pipeline
        edges, _ = preprocess(img)
        left_line, right_line = detect_lanes(edges, img.shape)

        # Convert to TuSimple point format
        pred_lanes = lanes_to_points(left_line, right_line, rec["h_samples"])

        # Compute accuracy
        metrics = tusimple_accuracy(pred_lanes, rec["lanes"], rec["h_samples"])
        metrics["frame"] = rec["raw_file"]
        results.append(metrics)

        if (i + 1) % 100 == 0:
            running_mean = np.mean([r["accuracy"] for r in results])
            print(f"  [{i+1}/{len(records)}] Running accuracy: {running_mean:.2f}%")

    mean_accuracy = float(np.mean([r["accuracy"] for r in results])) if results else 0.0
    print(f"\n{'='*50}")
    print(f"  FINAL ACCURACY: {mean_accuracy:.2f}%")
    print(f"  Frames evaluated: {len(results)}")
    print(f"{'='*50}\n")

    # Optionally save per-frame CSV
    if save_csv and results:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "correct", "total", "accuracy"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[Evaluation] Results saved to {save_csv}")

    return mean_accuracy
