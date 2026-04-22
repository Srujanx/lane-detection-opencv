"""
tusimple_loader.py
------------------
Loads frames and ground-truth annotations from the TuSimple Lane Detection Dataset.

TuSimple JSON format (one dict per line):
  {
    "raw_file":  "clips/0313-1/20.jpg",   <- path to the 20th (annotated) frame
    "lanes":     [[x1, x2, ...], ...],    <- x-coordinate per lane at each h_sample row
    "h_samples": [160, 170, 180, ...]     <- y-coordinates where lanes are sampled
  }
  x = -2 means no lane exists at that height row.
"""

import json
import os
import cv2


def parse_json(json_path: str) -> list[dict]:
    """
    Parse a TuSimple label JSON file into a list of annotation dicts.

    Args:
        json_path: Full path to a label_data_*.json file.

    Returns:
        List of dicts, each with keys: 'raw_file', 'lanes', 'h_samples'.
    """
    records = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_frame(base_dir: str, raw_file: str):
    """
    Load a single BGR image from disk using the TuSimple relative path.

    Args:
        base_dir: Root directory of the TuSimple dataset.
        raw_file: Relative path as stored in the JSON (e.g. 'clips/0313-1/20.jpg').

    Returns:
        BGR image (numpy array), or None if the file is not found.
    """
    full_path = os.path.join(base_dir, raw_file)
    img = cv2.imread(full_path)
    if img is None:
        print(f"[WARNING] Could not load image: {full_path}")
    return img


def get_clip_frames(base_dir: str, raw_file: str) -> list:
    """
    Load all 20 sequential frames for a TuSimple clip.
    The annotated frame is always frame 20; frames 1-19 have no labels.

    Args:
        base_dir: Root directory of the TuSimple dataset.
        raw_file: Path to the 20th frame (e.g. 'clips/0313-1/20.jpg').

    Returns:
        List of BGR images (frames 1 through 20), skipping missing files.
    """
    clip_dir = os.path.dirname(os.path.join(base_dir, raw_file))
    frames = []
    for i in range(1, 21):
        path = os.path.join(clip_dir, f"{i}.jpg")
        img = cv2.imread(path)
        if img is not None:
            frames.append(img)
    return frames
