import argparse
import os
import sys
import time
import numpy as np
from collections import deque
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from modules.preprocessor import preprocess
from modules.detector import detect_lanes
from modules.annotator import annotate


def smooth_line(history):
    """Average last N detected lines to reduce jitter."""
    if not history:
        return None
    return tuple(int(np.mean([h[i] for h in history])) for i in range(4))


def run_on_video(video_path, save_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return

    print("[Video] Running... Press 'q' to quit, SPACE to pause.")

    writer        = None
    paused        = False
    prev_time     = time.time()
    left_history  = deque(maxlen=8)
    right_history = deque(maxlen=8)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Video] Done.")
                break

            frame = cv2.resize(frame, (1280, 720))

            # Run CV pipeline
            edges, _              = preprocess(frame)
            left_line, right_line = detect_lanes(edges, frame.shape)

            # Update history only when detection succeeds
            if left_line is not None:
                left_history.append(left_line)
            if right_line is not None:
                right_history.append(right_line)

            # Use smoothed lines (holds last good detection when lanes disappear)
            left_smooth  = smooth_line(left_history)
            right_smooth = smooth_line(right_history)

            # FPS
            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Annotate and display
            out_frame = annotate(frame, left_smooth, right_smooth, fps=fps)

            if save_path and writer is None:
                H, W   = out_frame.shape[:2]
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H)
                )
            if writer:
                writer.write(out_frame)

            cv2.imshow("Lane Detection", out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("[Video] Paused." if paused else "[Video] Resumed.")

    cap.release()
    if writer:
        writer.release()
        print(f"[Video] Saved to {save_path}")
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to dashcam video")
parser.add_argument("--save", default=None, help="Save output to this .mp4 path")
args = parser.parse_args()

run_on_video(args.video, args.save)