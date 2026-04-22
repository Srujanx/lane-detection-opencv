# Lane Detection System — TuSimple Dataset
### Classical Computer Vision | OpenCV Only | No Deep Learning

---

## Pipeline Overview

```
Raw Frame (TuSimple JPG)
        │
        ▼
┌─────────────────────┐
│  1. Preprocessor    │  HSV Color Mask (white + yellow)
│                     │  Gaussian Blur → Canny Edges
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Detector        │  ROI Trapezoidal Mask
│                     │  Hough Line Transform
│                     │  Slope Classification (left/right)
│                     │  Line Averaging + Extrapolation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Annotator       │  Lane Fill Polygon (green, semi-transparent)
│                     │  Left Lane (RED) + Right Lane (BLUE)
│                     │  GT Dots (YELLOW, optional)
│                     │  HUD: Offset | Departure Warning | FPS
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Evaluator       │  TuSimple ±20px Accuracy Metric
│                     │  Per-frame CSV log
└─────────────────────┘
```

---

## Classical Techniques Used
- Gaussian Blur
- HSV Color Space Masking (White + Yellow)
- Canny Edge Detection
- Trapezoidal ROI Masking
- Probabilistic Hough Line Transform (cv2.HoughLinesP)
- Slope-based Line Classification
- Line Averaging via np.polyfit (degree 1)
- Lane Departure Detection (offset from centre)

---

## Dataset — TuSimple

**Download from Kaggle:**
```bash
kaggle datasets download manideep1108/tusimple
unzip tusimple.zip -d data/
```

**Expected folder structure after download:**
```
data/
├── clips/
│   ├── 0313-1/
│   │   ├── 1.jpg
│   │   ├── ...
│   │   └── 20.jpg       ← annotated frame
│   └── ...
├── label_data_0313.json
├── label_data_0531.json
└── label_data_0601.json
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### Demo — view live detection on annotated frames
```bash
python main.py --json data/label_data_0313.json --base_dir data/
```

### Demo with ground-truth overlay (yellow dots = GT, coloured lines = prediction)
```bash
python main.py --json data/label_data_0313.json --base_dir data/ --gt
```

### Save output as .mp4 video
```bash
python main.py --json data/label_data_0313.json --base_dir data/ --save output/result.mp4
```

### Run TuSimple accuracy evaluation
```bash
python main.py --json data/label_data_0313.json --base_dir data/ --eval
```
Results saved to `output/evaluation_log.csv`

### Single-frame test (debugging)
```bash
python main.py --json data/label_data_0313.json --base_dir data/ --test
```

### Controls (during demo)
| Key   | Action        |
|-------|---------------|
| `q`   | Quit          |
| SPACE | Pause/Resume  |

---

## Project Structure
```
lane_detection/
├── main.py                       ← Entry point
├── modules/
│   ├── preprocessor.py           ← Module 1: Preprocessing
│   ├── detector.py               ← Module 2: Lane detection
│   └── annotator.py              ← Module 3: Visualization
├── utils/
│   └── tusimple_loader.py        ← Dataset loader
├── evaluation/
│   └── accuracy.py               ← TuSimple accuracy metric
├── output/                       ← Saved results go here
├── assets/                       ← Drop sample images here
├── requirements.txt
└── README.md
```
