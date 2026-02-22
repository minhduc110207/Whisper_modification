# WhisperSign Training Guide for Google Colab

## Introduction

This document provides a step-by-step guide to train the **WhisperSign** model (a modified Whisper model for sign language recognition) on Google Colab with a free GPU.

## Step 1: Prepare Project on Google Drive

### 1.1 Upload Project to Google Drive

1. Compress the entire `D:\Whisper_modification` folder into a ZIP file
2. Upload the ZIP file to Google Drive at `My Drive/WhisperSign/`
3. Or use the commands below directly in Colab

### 1.2 Prepare Data

Your data should follow this structure:

```
data/processed/
  train/
    features/   -> .npy files, each with shape (T, 42, 7)
    labels/     -> .npy files, each containing label index arrays
  val/
    features/
    labels/
  test/
    features/
    labels/
```

**How to create data from video:**
- Use MediaPipe to extract 42 hand joints from video
- The script `src/utils/mediapipe_extract.py` supports this
- Or download existing datasets such as RWTH-PHOENIX, MS-ASL, LSA64

---

## Step 2: Google Colab Notebook

Copy the entire content below into a new Colab notebook file.

### Cell 1: Environment Setup

```python
# ============================================
# CELL 1: Mount Drive and install dependencies
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# Copy project from Drive
!cp -r "/content/drive/MyDrive/WhisperSign/Whisper_modification" /content/WhisperSign
%cd /content/WhisperSign

# Install dependencies
!pip install -q torch torchaudio
!pip install -q mediapipe scipy scikit-learn pyyaml tqdm tensorboard matplotlib pandas

# Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### Cell 2: Run Smoke Test

```python
# ============================================
# CELL 2: Verify the model runs correctly
# ============================================
!python scripts/smoke_test.py
```

### Cell 3: Extract Keypoints from Video (if needed)

```python
# ============================================
# CELL 3: Extract keypoints from video dataset
# (Only run this if you have videos but no .npy files)
# ============================================
import sys
sys.path.insert(0, '/content/WhisperSign')

from src.utils.mediapipe_extract import extract_from_dataset

# Change these paths to match your setup
VIDEO_DIR = "/content/drive/MyDrive/WhisperSign/videos/train"
OUTPUT_DIR = "/content/WhisperSign/data/processed/train/features"

extract_from_dataset(VIDEO_DIR, OUTPUT_DIR, target_fps=60)
```

### Cell 4: Create Demo Data (if no real data available)

```python
# ============================================
# CELL 4: Create dummy data to test the training pipeline
# (Delete this cell when you have real data)
# ============================================
import numpy as np
import os

def create_dummy_dataset(base_dir, num_samples=100, num_classes=50):
    """Create synthetic data to test the pipeline."""
    for split in ['train', 'val', 'test']:
        n = num_samples if split == 'train' else num_samples // 5
        feat_dir = os.path.join(base_dir, split, 'features')
        label_dir = os.path.join(base_dir, split, 'labels')
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for i in range(n):
            # Random sequence length 60-300 frames (1-5 seconds at 60Hz)
            T = np.random.randint(60, 300)
            features = np.random.randn(T, 42, 7).astype(np.float32)

            # Random labels (3-8 sign glosses per sequence)
            num_labels = np.random.randint(3, 8)
            labels = np.random.randint(1, num_classes, size=num_labels).astype(np.int64)

            np.save(os.path.join(feat_dir, f'sample_{i:04d}.npy'), features)
            np.save(os.path.join(label_dir, f'sample_{i:04d}.npy'), labels)

    print(f"Created dummy dataset at {base_dir}")

create_dummy_dataset('/content/WhisperSign/data/processed', num_samples=200, num_classes=50)
```

### Cell 5: Training Configuration

```python
# ============================================
# CELL 5: Configure model and training
# ============================================
import yaml

config = {
    "model": {
        "frontend": {
            "num_joints": 42,
            "num_features": 7,
            "patch_size": 4,
            "d_model": 512,      # Use 512 for full model
            "dropout": 0.1,
            "spatial_dropout": 0.15,
        },
        "encoder": {
            "num_heads": 8,
            "num_layers": 6,
            "d_model": 512,
            "d_ff": 2048,
            "dropout": 0.1,
        },
        "decoder": {
            "vocab_size": 1296,  # Adjust based on your sign gloss vocabulary
            "blank_id": 0,
        },
    },
    "data": {
        "sample_rate": 60,
        "max_seq_length": 1500,
        "num_left_joints": 21,
        "num_right_joints": 21,
        "augmentation": {
            "gesture_masking": {
                "enabled": True,
                "joint_mask_prob": 0.15,
                "temporal_mask_prob": 0.1,
                "max_temporal_mask": 10,
            },
            "noise": {"enabled": True, "std": 0.005},
            "temporal_jitter": {"enabled": True, "max_shift": 2},
        },
    },
    "training": {
        "stage1": {
            "epochs": 30,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "batch_size": 32,
            "freeze_encoder": True,
            "freeze_decoder": True,
        },
        "stage2": {
            "epochs": 100,
            "lr": 5.0e-5,
            "weight_decay": 1.0e-4,
            "batch_size": 16,
            "alpha": 0.3,
            "freeze_decoder": False,
        },
        "stage3": {
            "epochs": 30,
            "lr": 1.0e-5,
            "weight_decay": 1.0e-5,
            "batch_size": 16,
            "alpha": 0.3,
        },
        "warmup_steps": 500,
        "grad_clip": 1.0,
        "seed": 42,
        "num_workers": 2,   # Colab only supports 2 workers
        "save_dir": "/content/WhisperSign/checkpoints",
        "log_dir": "/content/WhisperSign/logs",
    },
}

# Save config
with open('/content/WhisperSign/configs/config_colab.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config saved to configs/config_colab.yaml")

# Print model info
from src.model.whisper_sign import WhisperSignModel
model = WhisperSignModel(config["model"])
total = model.get_num_params(trainable_only=False)
print(f"\nTotal model parameters: {total:,}")
print(f"Estimated model size: {total * 4 / 1e6:.1f} MB (float32)")
```

### Cell 6: Stage 1 — Warm-up Frontend

```python
# ============================================
# CELL 6: STAGE 1 - Warm-up Frontend
# Only trains the Frontend; Encoder + Decoder are frozen
# ============================================
import sys
sys.path.insert(0, '/content/WhisperSign')

import yaml
import torch
import numpy as np
from src.model.whisper_sign import WhisperSignModel
from src.data.dataset import create_dataloaders
from src.training.trainer import WhisperSignTrainer

# Load config
with open('/content/WhisperSign/configs/config_colab.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create model
model = WhisperSignModel(config["model"])
print(f"Total params: {model.get_num_params(False):,}")

# Create dataloaders
data_cfg = config["data"]
train_cfg = config["training"]

train_loader, val_loader, _ = create_dataloaders(
    data_dir="/content/WhisperSign/data/processed",
    config=data_cfg,
    batch_size=train_cfg["stage1"]["batch_size"],
    num_workers=train_cfg["num_workers"],
)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Create trainer
trainer = WhisperSignTrainer(
    model=model, config=train_cfg, device=device,
    save_dir=train_cfg["save_dir"],
    log_dir=train_cfg["log_dir"],
)

# Train Stage 1
trainer.train_stage1(train_loader, val_loader)

# Save to Drive
import shutil
shutil.copy(
    f"{train_cfg['save_dir']}/final_stage1.pt",
    "/content/drive/MyDrive/WhisperSign/checkpoints/final_stage1.pt"
)
print("Stage 1 checkpoint saved to Google Drive!")
```

### Cell 7: Stage 2 — Joint Training

```python
# ============================================
# CELL 7: STAGE 2 - Joint Training (Hybrid Loss)
# Unfreezes Encoder, trains with CTC + Attention loss
# ============================================

# (If Colab disconnected, reload from checkpoint)
# model, ckpt = WhisperSignModel.load_checkpoint(
#     "/content/drive/MyDrive/WhisperSign/checkpoints/final_stage1.pt", device
# )

# Create new dataloaders with stage2 batch size
train_loader2, val_loader2, _ = create_dataloaders(
    data_dir="/content/WhisperSign/data/processed",
    config=data_cfg,
    batch_size=train_cfg["stage2"]["batch_size"],
    num_workers=train_cfg["num_workers"],
)

# Train Stage 2
trainer.train_stage2(train_loader2, val_loader2)

# Save to Drive
shutil.copy(
    f"{train_cfg['save_dir']}/final_stage2.pt",
    "/content/drive/MyDrive/WhisperSign/checkpoints/final_stage2.pt"
)
print("Stage 2 checkpoint saved to Google Drive!")
```

### Cell 8: Stage 3 — Real-time Optimization

```python
# ============================================
# CELL 8: STAGE 3 - Real-time Optimization
# Fine-tune with Sliding Window + Gesture Masking
# ============================================

# Train Stage 3
trainer.train_stage3(train_loader2, val_loader2)

# Save final model to Drive
shutil.copy(
    f"{train_cfg['save_dir']}/final_model.pt",
    "/content/drive/MyDrive/WhisperSign/checkpoints/final_model.pt"
)
print("Final model saved to Google Drive!")
```

### Cell 9: Evaluate Model

```python
# ============================================
# CELL 9: Evaluate model on test set
# ============================================
_, _, test_loader = create_dataloaders(
    data_dir="/content/WhisperSign/data/processed",
    config=data_cfg,
    batch_size=16,
    num_workers=train_cfg["num_workers"],
)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        features = batch["features"].to(device)
        lengths = batch["feature_lengths"].to(device)

        preds = model.decode(features, lengths)
        all_predictions.extend(preds)

print(f"Total test samples: {len(all_predictions)}")
for i in range(min(5, len(all_predictions))):
    print(f"  Sample {i}: {all_predictions[i]}")
```

### Cell 10: TensorBoard

```python
# ============================================
# CELL 10: View training logs with TensorBoard
# ============================================
%load_ext tensorboard
%tensorboard --logdir /content/WhisperSign/logs
```

---

## Step 3: Resume Training After Colab Disconnects

Colab sessions can be interrupted at any time. To resume:

```python
import yaml
import torch
from src.model.whisper_sign import WhisperSignModel
from src.data.dataset import create_dataloaders
from src.training.trainer import WhisperSignTrainer

# Load config
with open('/content/WhisperSign/configs/config_colab.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint from Drive
CHECKPOINT = "/content/drive/MyDrive/WhisperSign/checkpoints/final_stage1.pt"
model, ckpt = WhisperSignModel.load_checkpoint(CHECKPOINT, device)
print(f"Resumed from epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")

# Continue training stage 2
train_loader, val_loader, _ = create_dataloaders(
    data_dir="/content/WhisperSign/data/processed",
    config=config["data"],
    batch_size=config["training"]["stage2"]["batch_size"],
    num_workers=2,
)

trainer = WhisperSignTrainer(
    model=model, config=config["training"], device=device,
    save_dir=config["training"]["save_dir"],
    log_dir=config["training"]["log_dir"],
)

trainer.train_stage2(train_loader, val_loader)
```

---

## Step 4: Using the Trained Model

```python
import torch
from src.model.whisper_sign import WhisperSignModel

# Load model
model, _ = WhisperSignModel.load_checkpoint("checkpoints/final_model.pt")
model.eval()

# Inference
import numpy as np
data = np.load("test_sample.npy")  # (T, 42, 7)
x = torch.from_numpy(data).float().unsqueeze(0)  # (1, T, 42, 7)
lengths = torch.tensor([data.shape[0]])

predictions = model.decode(x, lengths)
print(f"Predicted sign glosses: {predictions[0]}")
```

---

## Important Notes

1. **GPU**: Free Colab provides a T4 (16GB VRAM). This is sufficient for the base model (d_model=512)
2. **Training Time**: Stage 1 is fast (30 epochs), Stage 2 takes the longest (100 epochs)
3. **Save checkpoints frequently** to avoid losing progress when Colab disconnects
4. **Start with dummy data** to ensure the pipeline runs correctly, then switch to real data
5. **Adjust vocab_size** to match the number of sign glosses in your dataset
6. **Adjust d_model** based on available GPU memory:
   - `d_model=256`: lightweight, suitable for weaker GPUs
   - `d_model=512`: balanced (recommended)
   - `d_model=768`: powerful, requires GPU >= 16GB VRAM
