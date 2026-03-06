# WhisperSign — Data Collection & Usage Guide

Complete guide to collecting, processing, and using sign language data for training WhisperSign.

---

## Table of Contents

1. [Data Format Overview](#data-format-overview)
2. [Option A: Download Existing Datasets](#option-a-download-existing-datasets)
3. [Option B: Record Your Own Data](#option-b-record-your-own-data)
4. [Processing Pipeline](#processing-pipeline)
5. [Using Processed Data for Training](#using-processed-data-for-training)
6. [Quality Checklist](#quality-checklist)
7. [Troubleshooting](#troubleshooting)

---

## Data Format Overview

WhisperSign accepts **3D hand skeleton sequences**, not raw video or images. Every video is converted to a NumPy array before training.

### Input Tensor: `(T, 42, 7)` per sample

| Dimension | Meaning |
|-----------|---------|
| **T** | Number of frames (variable, typically 60–300 at 60 FPS) |
| **42** | Hand joints: 21 left + 21 right (MediaPipe convention) |
| **7** | Features per joint (see table below) |

### 7 Features Per Joint

| Index | Feature | Description | Reliability |
|-------|---------|-------------|-------------|
| 0 | x | Horizontal position (normalized 0–1) | ✅ High |
| 1 | y | Vertical position (normalized 0–1) | ✅ High |
| 2 | z | Depth (relative to wrist) | ⚠️ Approximate |
| 3 | vx | Velocity in x (computed from position) | ✅ High |
| 4 | vy | Velocity in y | ✅ High |
| 5 | vz | Velocity in z | ⚠️ Approximate |
| 6 | confidence | Hand detection confidence score | ✅ High |

### Joint Layout (42 joints)

```
Left Hand (indices 0–20):
  0: Wrist
  1-4: Thumb (CMC → MCP → IP → TIP)
  5-8: Index finger (MCP → PIP → DIP → TIP)
  9-12: Middle finger (MCP → PIP → DIP → TIP)
  13-16: Ring finger (MCP → PIP → DIP → TIP)
  17-20: Pinky (MCP → PIP → DIP → TIP)

Right Hand (indices 21–41):
  Same layout, offset by 21
```

### File Structure After Processing

```
data/processed/
├── train/
│   ├── features/
│   │   ├── sample_00000.npy    ← shape (T, 42, 7), float32
│   │   ├── sample_00001.npy
│   │   └── ...
│   └── labels/
│       ├── sample_00000.npy    ← shape (num_glosses,), int64
│       ├── sample_00001.npy
│       └── ...
├── val/
│   ├── features/
│   └── labels/
├── test/
│   ├── features/
│   └── labels/
└── label_map.json              ← {"<blank>": 0, "xin_chao": 1, "cam_on": 2, ...}
```

---

## Option A: Download Existing Datasets

### A1. Kaggle VSL — Vietnamese Sign Language ⭐ Recommended

**What it is:** Video recordings of isolated Vietnamese sign language glosses with labels.

**Step 1 — Download:**

1. Go to: https://www.kaggle.com/datasets/phamminhhoang/vsl-vietnamese-sign-languages
2. Click "Download" (requires Kaggle account)
3. Extract into your project:

```
data/raw/kaggle_vsl/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── label.csv          ← columns: ID, VIDEO, LABEL
```

**Step 2 — Install dependencies:**

```bash
pip install opencv-python mediapipe pandas scipy scikit-learn
```

**Step 3 — Process:**

```bash
python scripts/prepare_vsl_data.py \
  --source kaggle \
  --data_dir data/raw/kaggle_vsl \
  --output_dir data/processed \
  --target_fps 60
```

This script automatically:
- Reads each video, runs **MediaPipe Hands** to extract 21 landmarks per hand
- Computes velocities (vx, vy, vz) using `np.gradient`
- Resamples all sequences to 60 FPS using cubic spline interpolation
- Splits data into **80% train / 10% val / 10% test**
- Creates `label_map.json` mapping each gloss name to an integer ID

**Step 4 — Update config:**

Open `configs/config.yaml` and change `vocab_size` to match your label map:

```yaml
model:
  decoder:
    vocab_size: <count entries in label_map.json>
```

---

### A2. HuggingFace VOYA_VSL — Pre-extracted Keypoints

**What it is:** Already-extracted MediaPipe keypoints from Vietnamese sign language. No video processing needed.

```bash
pip install datasets
python scripts/prepare_vsl_data.py --source huggingface --output_dir data/processed
```

> **Note:** This dataset is small (~161 samples). Good for testing the pipeline, not for production training.

---

### A3. Other Sign Language Datasets

You can use any sign language dataset that provides either videos or skeletal keypoints. Some options:

| Dataset | Language | Type | Size | Link |
|---------|----------|------|------|------|
| Kaggle VSL | Vietnamese | Videos | ~500+ | [Kaggle](https://www.kaggle.com/datasets/phamminhhoang/vsl-vietnamese-sign-languages) |
| VOYA_VSL | Vietnamese | Keypoints | ~161 | [HuggingFace](https://huggingface.co/datasets/VOYA/VOYA_VSL) |
| MS-ASL | American (ASL) | Videos | ~25,000 | [Microsoft](https://www.microsoft.com/en-us/research/project/ms-asl/) |
| LSA64 | Argentine | Videos | ~3,200 | [Paper](https://core.ac.uk/download/pdf/76495887.pdf) |
| RWTH Phoenix | German (DGS) | Videos | ~1,000+ | [RWTH](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |

For video datasets, use the custom video pipeline (see Option B, Step 3).

---

## Option B: Record Your Own Data

If no existing dataset covers your target sign language or vocabulary.

### Step 1 — Plan Your Vocabulary

Before recording, decide:

| Decision | Recommendation |
|----------|----------------|
| **Which signs?** | Start with 20–50 most common glosses |
| **How many recordings per sign?** | Minimum 20, ideally 50+ |
| **How many signers?** | 3+ for diversity (different hand sizes, skin tones, speeds) |
| **Clip duration?** | 1–5 seconds per sign |
| **Total minimum dataset** | 20 signs × 20 recordings = 400 samples |

### Step 2 — Recording Setup

**Equipment:**
- Webcam or phone camera: 720p+, 30 FPS minimum (60 FPS ideal)
- Tripod or stable surface (avoid shaky camera)
- Good lighting: even, front-facing light, minimal backlight

**Environment:**
- Plain background (solid color, no clutter)
- Camera positioned at chest/torso height
- Signer distance: ~50–80 cm from camera
- Both hands clearly visible from wrist to fingertips

**Recording tips:**
- Perform each sign naturally at comfortable speed
- Include a 0.5-second neutral pose (hands resting) before and after each sign
- Vary signing speed across recordings (slow / normal / fast)
- Avoid jewelry, long sleeves, or gloves that could occlude the hand

### Step 3 — Recording Script

Save this as `scripts/record_data.py` and run it:

```python
import cv2
import os
import csv
import time

# ============ CONFIG ============
OUTPUT_DIR = "data/raw/my_recordings"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
LABEL_CSV = os.path.join(OUTPUT_DIR, "labels.csv")
CAMERA_ID = 0          # 0 = default webcam
FPS = 30               # Recording FPS
RESOLUTION = (640, 480)
# ================================

os.makedirs(VIDEO_DIR, exist_ok=True)

# Load existing labels or start fresh
existing_labels = []
if os.path.exists(LABEL_CSV):
    with open(LABEL_CSV, 'r') as f:
        reader = csv.DictReader(f)
        existing_labels = list(reader)
    print(f"Found {len(existing_labels)} existing recordings")

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

print("\n" + "="*50)
print("WhisperSign Data Recorder")
print("="*50)
print("Controls:")
print("  SPACE  = Start/Stop recording")
print("  Q      = Quit")
print("  N      = Next sign (enter new gloss name)")
print("="*50)

gloss_name = input("\nEnter first sign gloss name (e.g., xin_chao): ").strip()
recording_count = len([l for l in existing_labels if l.get("label") == gloss_name])
recording = False
writer = None
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    if recording:
        writer.write(frame)
        frames.append(1)
        elapsed = len(frames) / FPS
        cv2.putText(display, f"REC [{gloss_name}] {elapsed:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.circle(display, (RESOLUTION[0]-30, 30), 12, (0, 0, 255), -1)
    else:
        cv2.putText(display, f"Ready: '{gloss_name}' (#{recording_count+1})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=Record | N=Next sign | Q=Quit",
                    (10, RESOLUTION[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("WhisperSign Recorder", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if not recording:
            # Start recording
            filename = f"{gloss_name}_{recording_count:03d}.mp4"
            filepath = os.path.join(VIDEO_DIR, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, FPS, RESOLUTION)
            frames = []
            recording = True
            print(f"  Recording: {filename}...")
        else:
            # Stop recording
            writer.release()
            writer = None
            recording = False
            recording_count += 1
            duration = len(frames) / FPS

            filename = f"{gloss_name}_{recording_count-1:03d}.mp4"
            existing_labels.append({"filename": filename, "label": gloss_name})
            print(f"  Saved: {filename} ({duration:.1f}s, {len(frames)} frames)")

    elif key == ord('n'):
        if recording:
            writer.release()
            recording = False
        gloss_name = input("\nEnter next sign gloss name: ").strip()
        recording_count = len([l for l in existing_labels if l.get("label") == gloss_name])
        print(f"  Switched to '{gloss_name}' (existing: {recording_count})")

    elif key == ord('q'):
        if recording and writer:
            writer.release()
        break

cap.release()
cv2.destroyAllWindows()

# Save labels CSV
with open(LABEL_CSV, 'w', newline='') as f:
    writer_csv = csv.DictWriter(f, fieldnames=["filename", "label"])
    writer_csv.writeheader()
    writer_csv.writerows(existing_labels)

print(f"\n{'='*50}")
print(f"Total recordings: {len(existing_labels)}")
print(f"Labels saved to: {LABEL_CSV}")
print(f"Videos saved to: {VIDEO_DIR}")

# Count per gloss
from collections import Counter
counts = Counter(l["label"] for l in existing_labels)
print(f"\nPer-sign counts:")
for gloss, count in sorted(counts.items()):
    status = "✅" if count >= 20 else "⚠️ need more"
    print(f"  {gloss}: {count} recordings {status}")
```

### Step 4 — Process Your Recordings

After recording, run the preparation script:

```bash
python scripts/prepare_vsl_data.py \
  --source video \
  --data_dir data/raw/my_recordings/videos \
  --label_csv data/raw/my_recordings/labels.csv \
  --output_dir data/processed \
  --target_fps 60
```

This converts your videos into the `(T, 42, 7)` format automatically.

---

## Processing Pipeline

When you run `prepare_vsl_data.py`, here's exactly what happens to each video:

### Stage 1: MediaPipe Extraction (`mediapipe_extract.py`)

```
Video frame → MediaPipe Hands → 21 landmarks per hand (x, y, z)
```

- MediaPipe runs with `max_num_hands=2`, `min_detection_confidence=0.5`
- Each landmark gives normalized (x, y) in [0,1] and relative z
- If a hand is not visible, its 21 joints are set to **all zeros**

### Stage 2: Velocity Computation

```
Positions (T, 42, 3) → np.gradient(dt=1/fps) → Velocities (T, 42, 3)
```

- Velocities are added as features 3, 4, 5 (vx, vy, vz)
- Feature 6 (confidence) is the hand classification score from MediaPipe

### Stage 3: Resampling (`preprocessing.py`)

```
Variable FPS → Cubic spline interpolation → Fixed 60 FPS
```

- Input video may be 24, 30, or 60 FPS — resampling standardizes this
- Uses `scipy.interpolate.CubicSpline` for smooth interpolation

### Stage 4: Normalization (applied at training time by `dataset.py`)

```
1. SpatialNormalizer:  P̂ᵢ = Pᵢ - P_palm         (hand-centric coordinates)
2. ScaleNormalizer:    P̂ᵢ = P̂ᵢ / L_ref          (bone-length scale invariance)
```

- **SpatialNormalizer**: Subtracts the wrist position from all joints of each hand, making coordinates relative to the hand
- **ScaleNormalizer**: Divides by the total middle finger bone chain length (Wrist→MCP→PIP→DIP→TIP), making small and large hands produce identical normalized values

### Stage 5: Augmentation (applied during training only)

| Augmentation | What It Does | Default |
|-------------|--------------|---------|
| **GestureMasking** | Randomly zeroes out joints or time segments | 15% joints, 10% temporal |
| **TemporalJitter** | Shifts the sequence by ±2 frames | max_shift=2 |
| **NoiseInjection** | Adds Gaussian noise to all features | std=0.005 |

---

## Using Processed Data for Training

### On Your Local Machine

```bash
# Full 3-stage pipeline
python scripts/train.py --config configs/config.yaml --data_dir data/processed --device cuda
```

### On Kaggle

Use the provided `scripts/kaggle_notebook.py`:

1. Create new Kaggle Notebook → enable GPU
2. Copy the script content into cells
3. Set `USE_GITHUB = True` and `USE_DUMMY_DATA = False`
4. Upload your `data/processed/` folder as a Kaggle Dataset
5. Update `DATA_DIR` to point to your Kaggle dataset input path

### On Google Colab

See `COLAB_TRAINING_GUIDE.md` for complete Colab instructions.

### Bringing Your Own Pre-extracted Keypoints

If you already have skeletal data from another source (Leap Motion, depth camera, another pose estimator), you can skip MediaPipe entirely. Just format your data as:

```python
import numpy as np

# Your keypoints: shape (T, 42, 7) or at minimum (T, 42, 3)
keypoints = np.load("my_keypoints.npy")  # (T, num_joints, num_features)

# If only 3 features (x,y,z), pad to 7:
if keypoints.shape[2] < 7:
    padded = np.zeros((keypoints.shape[0], 42, 7), dtype=np.float32)
    padded[:, :keypoints.shape[1], :keypoints.shape[2]] = keypoints
    # Compute velocities
    dt = 1.0 / 60  # your FPS
    padded[:, :, 3:6] = np.gradient(padded[:, :, :3], dt, axis=0)
    padded[:, :, 6] = 1.0  # confidence = 1 for known-good data
    keypoints = padded

# Save
np.save("data/processed/train/features/sample_00000.npy", keypoints.astype(np.float32))

# Label: array of gloss IDs that appear in this sample
labels = np.array([1], dtype=np.int64)  # single isolated sign with ID=1
np.save("data/processed/train/labels/sample_00000.npy", labels)
```

---

## Quality Checklist

Before training, verify your data quality:

### ✅ Data Verification Commands

```python
import numpy as np
import os

data_dir = "data/processed/train/features"
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])

print(f"Total samples: {len(files)}")

# Check shapes and stats
for f in files[:5]:
    data = np.load(os.path.join(data_dir, f))
    print(f"  {f}: shape={data.shape}, "
          f"min={data.min():.3f}, max={data.max():.3f}, "
          f"has_nan={np.any(np.isnan(data))}, "
          f"all_zero_frames={np.sum(np.all(data == 0, axis=(1,2)))}/{data.shape[0]}")
```

### ✅ Quality Criteria

| Check | Expected | Problem If Failed |
|-------|----------|-------------------|
| Shape | `(T, 42, 7)` with T > 10 | Wrong extraction or corrupt file |
| No NaN values | `has_nan = False` | Corrupt source video |
| Not all zeros | Most frames should have non-zero joints | MediaPipe failed to detect hands |
| All-zero frames < 50% | Less than half of frames are empty | Poor video quality or hands not visible |
| Reasonable range | Values roughly in [-5, 5] after normalization | Data not normalized or extreme outliers |
| Min 20 samples per class | Count per label ≥ 20 | Need more recordings of that sign |
| Train/val/test split exists | All 3 directories have data | Re-run prepare script |

### ✅ Label Verification

```python
import json

with open("data/processed/label_map.json") as f:
    label_map = json.load(f)

print(f"Vocabulary: {len(label_map)} classes (including <blank>)")
for name, idx in sorted(label_map.items(), key=lambda x: x[1])[:10]:
    print(f"  {idx}: {name}")
```

Make sure `configs/config.yaml` → `decoder.vocab_size` matches the number of entries in `label_map.json`.

---

## Troubleshooting

### MediaPipe doesn't detect hands

- **Cause:** Poor lighting, hands too far/close, hands moving too fast
- **Fix:** Improve lighting, position camera 50–80cm from hands, sign slower
- **Check:** Run a single video through `mediapipe_extract.py` and count non-zero frames

### Too few frames extracted

- **Cause:** Video is too short or low FPS
- **Fix:** Record at least 1 second per sign at 30+ FPS
- **Check:** `keypoints.shape[0]` should be > 30

### Different number of joints than 42

- **Cause:** Single-hand dataset, or different pose estimator
- **Fix:** The preparation script auto-pads to 42 joints. For single-hand data, only indices 0–20 will be non-zero

### Labels don't match

- **Cause:** CSV column names don't match expected format
- **Fix:** The script auto-detects columns named `filename/file/video/path` and `label/gloss/class/sign`. If your columns have unusual names, rename them.

### Out of memory during MediaPipe extraction

- **Cause:** Very long videos (>30 seconds)
- **Fix:** Trim videos to individual sign clips (1–5 seconds each) before processing
