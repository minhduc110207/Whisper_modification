# WhisperSign — Complete Instructions (Start to Finish)

Everything you need to go from zero to a working sign language recognition model, covering every potential problem along the way.

---

## Phase 0: Prerequisites

### What You Need

| Requirement | Why | How to Check |
|-------------|-----|--------------|
| **Python 3.8+** | Required by PyTorch and MediaPipe | `python --version` |
| **pip** | Package installer | `pip --version` |
| **Git** | Clone the project | `git --version` |
| **Kaggle account** | Free GPU for training | https://www.kaggle.com |
| **~2 GB disk space** | For raw + processed data | — |

### Optional (for inference only)

| Requirement | Why |
|-------------|-----|
| **Webcam** | Record your own sign language data |
| **Leap Motion Controller** | Higher-accuracy hand tracking for inference |
| **NVIDIA GPU (local)** | Faster local inference (not needed if using Kaggle for training) |

---

## Phase 1: Get the Code

### Step 1.1 — Clone the Repository

```bash
git clone https://github.com/minhduc110207/Whisper_modification.git
cd Whisper_modification
```

### Step 1.2 — Create a Virtual Environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate
```

> **⚠️ Problem: `python` not found**
> - Windows: Install from https://www.python.org, check "Add to PATH" during install
> - Linux: `sudo apt install python3 python3-venv python3-pip`

### Step 1.3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `torch`, `mediapipe`, `scipy`, `scikit-learn`, `numpy`, `pandas`, `pyyaml`, `tqdm`, `tensorboard`, `matplotlib`

> **⚠️ Problem: PyTorch CUDA not detected**
> If you have a GPU but `torch.cuda.is_available()` returns False, install the CUDA-specific PyTorch:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> Visit https://pytorch.org for your exact command.

> **⚠️ Problem: `mediapipe` fails to install on Python 3.13+**
> MediaPipe may not support the latest Python. Use Python 3.10 or 3.11 instead.

### Step 1.4 — Verify Installation

```bash
python scripts/smoke_test.py
```

This tests that the model can be created, data can flow through it, and loss can be computed. If all components show `OK`, you're ready.

> **⚠️ Problem: `ModuleNotFoundError: No module named 'src'`**
> You must run from the project root directory (`Whisper_modification/`). Make sure you `cd Whisper_modification` first.

---

## Phase 2: Collect or Download Data

You need sign language videos with labels. Choose **one** of these options:

### Option A: Kaggle VSL Dataset ⭐ Easiest

**Step 2A.1** — Go to https://www.kaggle.com/datasets/phamminhhoang/vsl-vietnamese-sign-languages

**Step 2A.2** — Click **Download** (requires free Kaggle account)

**Step 2A.3** — Extract to the project:

```
Whisper_modification/
└── data/
    └── raw/
        └── kaggle_vsl/
            ├── videos/
            │   ├── video_001.mp4
            │   └── ...
            └── label.csv
```

> **⚠️ Problem: CSV columns have different names**
> The script auto-detects common column names (VIDEO, LABEL, filename, sign, etc). If your CSV has unusual names, rename the columns to `VIDEO` and `LABEL`.

> **⚠️ Problem: Some videos are corrupt / unreadable**
> The script will print `[SKIP]` for bad videos and continue. This is normal — a few skipped videos won't affect training.

---

### Option B: HuggingFace VOYA_VSL (Pre-extracted, No Videos Needed)

```bash
pip install datasets
python scripts/prepare_vsl_data.py --source huggingface --output_dir data/processed
```

> **⚠️ This dataset is small (~161 samples).** Good for testing the pipeline, but not enough for production training.

---

### Option C: Record Your Own Data

See `DATA_COLLECTION_GUIDE.md` for a full recording script. Key rules:

- **Minimum 20 recordings per sign** (50+ is better)
- **1–5 seconds per clip**
- **Good lighting, plain background, hands clearly visible**
- **Multiple signers** for diversity (different hand sizes)

After recording, create a CSV:
```csv
filename,label
video_001.mp4,xin_chao
video_002.mp4,cam_on
```

---

## Phase 3: Process Data

This converts videos → skeleton data (`.npy` files).

### Step 3.1 — Install Processing Dependencies

```bash
pip install opencv-python mediapipe pandas scipy scikit-learn
```

### Step 3.2 — Run the Preparation Script

**From Kaggle VSL videos:**
```bash
python scripts/prepare_vsl_data.py \
  --source kaggle \
  --data_dir data/raw/kaggle_vsl \
  --output_dir data/processed \
  --target_fps 60
```

**From your own videos:**
```bash
python scripts/prepare_vsl_data.py \
  --source video \
  --data_dir path/to/your/videos \
  --label_csv path/to/your/labels.csv \
  --output_dir data/processed \
  --target_fps 60
```

**From HuggingFace:**
```bash
python scripts/prepare_vsl_data.py \
  --source huggingface \
  --output_dir data/processed
```

This script:
1. Reads each video → runs **MediaPipe Hands** → extracts 21 landmarks per hand (x, y, z)
2. Computes velocities (vx, vy, vz)
3. Resamples to 60 FPS
4. Splits into **80% train / 10% val / 10% test**
5. Creates `label_map.json`

> **⚠️ Problem: Processing is very slow**
> MediaPipe processes ~10-30 FPS. A 1000-video dataset may take 1-3 hours. This is normal. You only need to do this once.

> **⚠️ Problem: `No hands detected` for many videos**
> Causes: poor lighting, hands not in frame, hands moving too fast, low resolution video. Try improving video quality or increasing `min_detection_confidence` in the script.

> **⚠️ Problem: Out of memory during processing**
> Long videos (>30 sec) use a lot of RAM. Trim videos to individual sign clips (1-5 sec each) before processing.

### Step 3.3 — Verify Processed Data

```bash
python -c "
import os, json, numpy as np
d = 'data/processed'

# Check files exist
for split in ['train', 'val', 'test']:
    feat = os.path.join(d, split, 'features')
    if os.path.exists(feat):
        files = [f for f in os.listdir(feat) if f.endswith('.npy')]
        sample = np.load(os.path.join(feat, files[0]))
        print(f'{split}: {len(files)} samples, shape={sample.shape}')
    else:
        print(f'{split}: MISSING!')

# Check label map
with open(os.path.join(d, 'label_map.json')) as f:
    lm = json.load(f)
print(f'Vocab size: {len(lm)} (including <blank>)')
print(f'First 5 labels: {dict(list(lm.items())[:5])}')
"
```

**Expected output:**
```
train: 400 samples, shape=(180, 42, 7)
val: 50 samples, shape=(120, 42, 7)
test: 50 samples, shape=(150, 42, 7)
Vocab size: 51 (including <blank>)
```

> **⚠️ Keys to check:**
> - Shape must be `(T, 42, 7)` — not `(T, 21, 3)` or anything else
> - Each split must have samples — if val/test is empty, re-run with more data
> - T should be > 10 for each sample

---

## Phase 4: Upload to Kaggle

Kaggle provides **free GPU** (T4 or P100), but you need to upload your code and data as **Kaggle Datasets**.

### Step 4.1 — Create Code Dataset

**What to include** (from your project folder):
```
whispersign-code/
├── configs/
│   └── config.yaml
├── src/                    ← entire src folder
│   ├── __init__.py
│   ├── data/
│   ├── model/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── train.py
│   ├── smoke_test.py
│   ├── kaggle_notebook.py
│   └── prepare_vsl_data.py
├── requirements.txt
└── setup.py
```

**What NOT to include:**
- ❌ `.venv/` (virtual environment — huge, not needed)
- ❌ `.git/` (git history — large, not needed)
- ❌ `data/` (uploaded separately)
- ❌ `__pycache__/` (Python bytecode cache)
- ❌ `tests/` (not needed for training)
- ❌ `.md` files (documentation, not needed for training)
- ❌ `logs/`, `checkpoints/` (will be created during training)

**How to upload:**
1. Go to https://www.kaggle.com → **Datasets** → **New Dataset**
2. Name it `whispersign-code`
3. Drag and drop the files/folders listed above
4. Click **Create**

### Step 4.2 — Create Data Dataset

Upload your processed data:
```
whispersign-data/
├── train/
│   ├── features/          ← all .npy feature files
│   └── labels/            ← all .npy label files
├── val/
│   ├── features/
│   └── labels/
├── test/
│   ├── features/
│   └── labels/
└── label_map.json
```

**How to upload:**
1. **Datasets** → **New Dataset**
2. Name it `whispersign-data`
3. Upload the entire `data/processed/` folder contents
4. Click **Create**

> **⚠️ Problem: Upload is slow / fails**
> Kaggle has a 100 GB limit per dataset. For large datasets, compress to ZIP first:
> ```bash
> cd data/processed
> zip -r ../../whispersign-data.zip .
> ```
> Then upload the ZIP — Kaggle auto-extracts it.

> **⚠️ Problem: Folder structure is wrong after upload**
> After creating the dataset, click on it and verify the folder structure matches. Kaggle sometimes adds an extra nesting level. If you see `whispersign-data/processed/train/` instead of `whispersign-data/train/`, adjust the `DATA` path in your notebook.

---

## Phase 5: Train on Kaggle

### Step 5.1 — Create a New Notebook

1. Go to https://www.kaggle.com → **Code** → **New Notebook**

### Step 5.2 — Enable GPU

1. Click **Settings** (gear icon on right sidebar)
2. **Accelerator** → Select **GPU T4 x2** (or **P100**)
3. **Persistence** → Turn **ON** (saves output files)

> **⚠️ Problem: No GPU available**
> Kaggle gives 30 GPU hours/week. If you've exceeded the limit, wait until the weekly reset (usually Monday). Alternatively, use Google Colab (see `COLAB_TRAINING_GUIDE.md`).

### Step 5.3 — Add Your Datasets

1. Click **Add Data** (plus icon on right sidebar)
2. Search for your `whispersign-code` dataset → **Add**
3. Search for your `whispersign-data` dataset → **Add**

After adding, your data will be at:
```
/kaggle/input/whispersign-code/    ← your code (READ-ONLY)
/kaggle/input/whispersign-data/    ← your data (READ-ONLY)
```

### Step 5.4 — Write the Notebook Cells

**Cell 1: Setup**
```python
import os, sys, subprocess, shutil

# Copy code to writable directory (Kaggle input is read-only)
CODE_SRC = "/kaggle/input/whispersign-code"
PROJECT  = "/kaggle/working/WhisperSign"
DATA     = "/kaggle/input/whispersign-data"

if not os.path.exists(PROJECT):
    shutil.copytree(CODE_SRC, PROJECT)
    print(f"Copied code to {PROJECT}")

os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# Install missing dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "scipy", "scikit-learn", "pyyaml", "tqdm", "tensorboard"], check=True)

# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

> **⚠️ Problem: `ModuleNotFoundError` for scipy, sklearn, etc.**
> This means the pip install line didn't run. Make sure Cell 1 runs before any other cell.

> **⚠️ Problem: Path structure is different**
> Check the actual path by adding: `print(os.listdir("/kaggle/input/"))` and adjust paths accordingly. Kaggle sometimes lowercases or hyphenates dataset names.

**Cell 2: Verify data and configure**
```python
import json, yaml, numpy as np

# Check data exists
for split in ["train", "val", "test"]:
    feat_dir = os.path.join(DATA, split, "features")
    if os.path.exists(feat_dir):
        n = len([f for f in os.listdir(feat_dir) if f.endswith(".npy")])
        print(f"{split}: {n} samples")
    else:
        # Maybe nested one level deeper — check
        for subdir in os.listdir(DATA):
            alt = os.path.join(DATA, subdir, split, "features")
            if os.path.exists(alt):
                print(f"FOUND at {alt} — update DATA path!")

# Load vocab size
label_map_path = os.path.join(DATA, "label_map.json")
if not os.path.exists(label_map_path):
    # Search for it
    for root, dirs, files in os.walk(DATA):
        if "label_map.json" in files:
            label_map_path = os.path.join(root, "label_map.json")
            print(f"Found label_map at: {label_map_path}")
            break

with open(label_map_path) as f:
    label_map = json.load(f)
vocab_size = len(label_map)
print(f"Vocab size: {vocab_size}")

# Load and modify config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

config["model"]["decoder"]["vocab_size"] = vocab_size
config["training"]["num_workers"] = 2          # Kaggle limit
config["training"]["save_dir"] = "/kaggle/working/checkpoints"
config["training"]["log_dir"] = "/kaggle/working/logs"

# For small datasets (<500 samples), use smaller model to prevent overfitting
train_dir = os.path.join(DATA, "train", "features")
n_train = len([f for f in os.listdir(train_dir) if f.endswith(".npy")])
if n_train < 500:
    print(f"Small dataset ({n_train} samples) — using d_model=256")
    config["model"]["frontend"]["d_model"] = 256
    config["model"]["frontend"]["dropout"] = 0.2
    config["model"]["encoder"]["d_model"] = 256
    config["model"]["encoder"]["num_heads"] = 4
    config["model"]["encoder"]["num_layers"] = 4
    config["model"]["encoder"]["d_ff"] = 1024
    config["model"]["encoder"]["dropout"] = 0.2

print("\nConfig ready!")
```

**Cell 3: Train Stage 1 — Frontend Warm-up**
```python
import torch
import numpy as np
from src.model.whisper_sign import WhisperSignModel
from src.data.dataset import create_dataloaders
from src.training.trainer import WhisperSignTrainer

torch.manual_seed(42)
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperSignModel(config["model"])
print(f"Parameters: {model.get_num_params(False):,}")

train_loader, val_loader, _ = create_dataloaders(
    data_dir=DATA, config=config["data"],
    batch_size=config["training"]["stage1"]["batch_size"],
    num_workers=2)

print(f"Train: {len(train_loader.dataset)} samples")
print(f"Val: {len(val_loader.dataset)} samples")

trainer = WhisperSignTrainer(
    model=model, config=config["training"], device=device,
    save_dir="/kaggle/working/checkpoints",
    log_dir="/kaggle/working/logs")

print("\n🚀 Stage 1: Frontend Warm-up (Encoder+Decoder frozen)")
trainer.train_stage1(train_loader, val_loader)
print("✅ Stage 1 complete!")
```

> **⚠️ Problem: `CUDA out of memory`**
> Reduce batch size: change `config["training"]["stage1"]["batch_size"] = 16` (or 8).
> Or reduce max sequence length: `config["data"]["max_seq_length"] = 750`

> **⚠️ Problem: Loss is NaN**
> - Data might have NaN values. Add: `print(np.any(np.isnan(np.load("sample.npy"))))`
> - Reduce learning rate: `config["training"]["stage1"]["lr"] = 5e-4`

**Cell 4: Train Stage 2 — Joint Training**
```python
train_loader2, val_loader2, _ = create_dataloaders(
    data_dir=DATA, config=config["data"],
    batch_size=config["training"]["stage2"]["batch_size"],
    num_workers=2)

print("🚀 Stage 2: Joint Training (all layers, hybrid loss)")
trainer.train_stage2(train_loader2, val_loader2)
print("✅ Stage 2 complete!")
```

> **⚠️ Problem: Kaggle session disconnects mid-training**
> This is common — Kaggle has a 12-hour limit. To resume:
> ```python
> # In a NEW notebook session, after re-running Cell 1:
> model, ckpt = WhisperSignModel.load_checkpoint(
>     "/kaggle/working/checkpoints/best_stage1.pt", device)
> # Then continue with Stage 2
> ```
> **Tip:** Save checkpoints to `/kaggle/working/` — files here persist in the Output tab even after disconnection.

**Cell 5: Train Stage 3 — Real-time Optimization**
```python
print("🚀 Stage 3: Real-time Optimization (fine-tuning)")
trainer.train_stage3(train_loader2, val_loader2)
print("✅ Stage 3 complete!")

# Save final model
import shutil
src = "/kaggle/working/checkpoints"
for f in os.listdir(src):
    print(f"  Saved: {f} ({os.path.getsize(os.path.join(src, f))/1e6:.1f} MB)")
```

**Cell 6: Quick Evaluation**
```python
model.eval()
_, _, test_loader = create_dataloaders(
    data_dir=DATA, config=config["data"], batch_size=16, num_workers=2)

with torch.no_grad():
    for batch in test_loader:
        features = batch["features"].to(device)
        lengths = batch["feature_lengths"].to(device)
        preds = model.decode(features, lengths)
        for i, p in enumerate(preds[:5]):
            gloss_names = [list(label_map.keys())[list(label_map.values()).index(id)]
                          if id in label_map.values() else f"UNK_{id}" for id in p]
            print(f"  Sample {i}: {gloss_names}")
        break  # just show first batch
```

### Step 5.5 — Download Results

After training completes:
1. Click the **Output** tab (left sidebar) in your notebook
2. You'll see `/kaggle/working/checkpoints/` containing:
   - `final_stage1.pt`, `final_stage2.pt`, `final_model.pt`
3. Click **Download** on each checkpoint file
4. Save them to `Whisper_modification/checkpoints/` on your local machine

---

## Phase 6: Use the Trained Model (Inference)

### Option A: From a Video File (MediaPipe)

```python
import torch, json
from src.model.whisper_sign import WhisperSignModel
from src.utils.mediapipe_extract import extract_hand_keypoints
from src.data.normalization import SpatialNormalizer, ScaleNormalizer

# 1. Load model
model, _ = WhisperSignModel.load_checkpoint("checkpoints/final_model.pt")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 2. Load label map
with open("data/processed/label_map.json") as f:
    label_map = json.load(f)
id_to_gloss = {v: k for k, v in label_map.items()}

# 3. Extract keypoints from video
keypoints, timestamps = extract_hand_keypoints("my_sign_video.mp4")
print(f"Extracted {len(keypoints)} frames")

# 4. Normalize (same pipeline as training!)
keypoints = SpatialNormalizer().normalize(keypoints)
keypoints = ScaleNormalizer().normalize(keypoints)

# 5. Inference
x = torch.from_numpy(keypoints).float().unsqueeze(0).to(device)
lengths = torch.tensor([keypoints.shape[0]], device=device)

with torch.no_grad():
    predictions = model.decode(x, lengths)

# 6. Convert IDs to text
glosses = [id_to_gloss.get(p, f"?{p}") for p in predictions[0]]
print(f"Recognized: {' '.join(glosses)}")
```

> **⚠️ Problem: Model outputs empty list or only blanks**
> - Model not trained enough — train for more epochs
> - Video too short — need at least ~1 second of signing
> - Hands not detected — check MediaPipe can see the hands (good lighting required)

> **⚠️ Problem: Wrong predictions**
> - Make sure you use the **same label_map.json** from training
> - Make sure normalization is applied (Spatial + Scale) — without it, the model sees completely different input

---

### Option B: From Leap Motion Sensor

```python
import numpy as np
from src.utils.leap_motion_extract import LeapMotionAdapter

adapter = LeapMotionAdapter(fps=120.0)

# From raw numpy: (T, 21, 3) array in millimeters
leap_data = np.load("my_leap_recording.npy")  # your Leap Motion export
keypoints = adapter.from_numpy(leap_data, hand_type="right", input_format="joints_xyz")

# From CSV export:
# keypoints = adapter.from_csv("recording.csv", hand_type="right")

# Then normalize + infer (same as MediaPipe)
keypoints = SpatialNormalizer().normalize(keypoints)
keypoints = ScaleNormalizer().normalize(keypoints)

x = torch.from_numpy(keypoints).float().unsqueeze(0).to(device)
lengths = torch.tensor([keypoints.shape[0]], device=device)

with torch.no_grad():
    predictions = model.decode(x, lengths)

glosses = [id_to_gloss.get(p, f"?{p}") for p in predictions[0]]
print(f"Recognized: {' '.join(glosses)}")
```

> **⚠️ Important: Training data vs Inference data mismatch**
> If you train with **MediaPipe** data but infer with **Leap Motion**, there will be a domain gap because:
> - MediaPipe: normalized [0,1] coordinates from 2D camera (monocular depth for z)
> - Leap Motion: absolute mm coordinates from IR stereo (true 3D depth)
>
> The `LeapMotionAdapter` normalizes Leap Motion data to [0,1] to minimize this gap, but some accuracy loss is expected. For best results, **fine-tune** the model on a small amount of Leap Motion data.

---

### Option C: Real-Time Streaming (Continuous Recognition)

```python
from src.utils.sliding_window import SlidingWindowInference
from src.utils.smoothing import MovingAverageSmoothing

smoother = MovingAverageSmoothing(window_size=5)
slider = SlidingWindowInference(
    model,
    window_duration=3.0,   # 3-second windows
    overlap=0.5,           # 50% overlap
    sample_rate=60,
    device=device)

# Get continuous data from your sensor/camera
keypoints = ...  # (T, 42, 7) - many seconds of recording
keypoints = SpatialNormalizer().normalize(keypoints)
keypoints = ScaleNormalizer().normalize(keypoints)
keypoints = smoother.smooth(keypoints)

predictions = slider(keypoints)
glosses = [id_to_gloss.get(p, f"?{p}") for p in predictions]
print(f"Continuous recognition: {' '.join(glosses)}")
```

---

## Quick Reference

### All Commands in Order

```bash
# --- SETUP ---
git clone https://github.com/minhduc110207/Whisper_modification.git
cd Whisper_modification
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/smoke_test.py

# --- DATA ---
# Download Kaggle VSL to data/raw/kaggle_vsl/
python scripts/prepare_vsl_data.py --source kaggle \
  --data_dir data/raw/kaggle_vsl --output_dir data/processed

# --- TRAIN (local, if you have GPU) ---
python scripts/train.py --config configs/config.yaml \
  --data_dir data/processed --device cuda

# --- OR TRAIN ON KAGGLE ---
# Upload src/, configs/, scripts/, requirements.txt as "whispersign-code"
# Upload data/processed/ as "whispersign-data"
# Create notebook, enable GPU, add both datasets, paste cells above

# --- INFER ---
python -c "
import torch, json
from src.model.whisper_sign import WhisperSignModel
from src.utils.mediapipe_extract import extract_hand_keypoints
from src.data.normalization import SpatialNormalizer, ScaleNormalizer

model, _ = WhisperSignModel.load_checkpoint('checkpoints/final_model.pt')
model.eval()

kp, _ = extract_hand_keypoints('my_video.mp4')
kp = SpatialNormalizer().normalize(kp)
kp = ScaleNormalizer().normalize(kp)
x = torch.from_numpy(kp).float().unsqueeze(0)

with open('data/processed/label_map.json') as f:
    lm = json.load(f)
id2g = {v:k for k,v in lm.items()}

preds = model.decode(x, torch.tensor([len(kp)]))[0]
print([id2g.get(p,'?') for p in preds])
"
```

### Files Map

| File | Purpose | When You Need It |
|------|---------|-----------------|
| `configs/config.yaml` | Model + training config | Always |
| `src/data/normalization.py` | Spatial + Scale normalization | Training + Inference |
| `src/data/preprocessing.py` | Resampling, padding | Data processing |
| `src/data/augmentation.py` | GestureMask, Jitter, Noise | Training only |
| `src/data/dataset.py` | PyTorch DataLoader | Training only |
| `src/model/frontend.py` | Skeleton → embeddings | Always |
| `src/model/encoder.py` | Spatio-temporal Transformer | Always |
| `src/model/decoder.py` | CTC + Attention decoder | Always |
| `src/model/whisper_sign.py` | Main model class | Always |
| `src/training/trainer.py` | 3-stage training loop | Training only |
| `src/training/losses.py` | Hybrid CTC+Attention loss | Training only |
| `src/utils/mediapipe_extract.py` | Video → keypoints | Data processing + MediaPipe inference |
| `src/utils/leap_motion_extract.py` | Leap Motion → keypoints | Leap Motion inference |
| `src/utils/sliding_window.py` | Real-time streaming | Real-time inference |
| `scripts/prepare_vsl_data.py` | Dataset preparation | Data processing (once) |
| `scripts/train.py` | Training entry point | Training (local only) |
| `scripts/kaggle_notebook.py` | Kaggle training notebook | Training (Kaggle only) |
| `scripts/smoke_test.py` | Verify installation | Setup (once) |
