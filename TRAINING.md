# WhisperSign Training Guide

Complete guide to training WhisperSign with Vietnamese Sign Language data.

---

## Table of Contents

1. [Training Process Overview](#training-process-overview)
2. [Data Preparation](#data-preparation)
3. [Configuration](#configuration)
4. [Running Training](#running-training)
5. [Monitoring and Evaluation](#monitoring-and-evaluation)
6. [Troubleshooting](#troubleshooting)

---

## Training Process Overview

WhisperSign uses a **3-stage progressive training strategy**. Each stage unfreezes more of the model, allowing the network to learn incrementally from simple pattern recognition to complex sequence modeling.

### Why 3 Stages?

The model replaces Whisper's audio frontend with a skeletal data frontend. If we train everything from scratch simultaneously, the randomly initialized frontend produces garbage embeddings, making it impossible for the pre-trained encoder to learn useful representations. The 3-stage approach solves this:

### Stage 1: Frontend Warm-up (30 epochs)

```
Frontend: TRAINABLE    <-- Only this learns
Encoder:  FROZEN
Decoder:  FROZEN
Loss:     CTC only
LR:       1e-3 (high, fast convergence)
```

**What happens:** The frontend learns to convert raw skeletal data `(T, 42, 7)` into meaningful embeddings `(T/P, d_model)` that the frozen encoder can process. The CTC head provides a simple gradient signal — just align input frames to output glosses monotonically.

**What to expect:**
- Loss drops sharply in first 5 epochs (frontend learns basic spatial patterns)
- Convergence around epoch 15-20
- CTC loss should reach ~1.0-2.0 depending on vocabulary size
- Training speed: ~2-5 min/epoch on a single GPU with small dataset

### Stage 2: Joint Training (100 epochs)

```
Frontend: TRAINABLE
Encoder:  TRAINABLE    <-- Now unfrozen
Decoder:  TRAINABLE    <-- Now unfrozen
Loss:     0.3 * CTC + 0.7 * Attention (hybrid)
LR:       5e-5 (low, preserve encoder knowledge)
```

**What happens:** The encoder adapts its attention patterns from audio to skeletal features. The attention decoder learns to generate gloss sequences using cross-attention on encoder output. The hybrid loss balances CTC's monotonic alignment with the attention decoder's flexible reordering.

**What to expect:**
- Loss decreases slowly but steadily
- The attention loss dominates (weight 0.7) and drives most of the learning
- Overfitting may appear after epoch 50-60 on small datasets — watch validation loss
- Training speed: ~3-8 min/epoch (more parameters updating)

### Stage 3: Real-time Optimization (30 epochs)

```
Frontend: TRAINABLE
Encoder:  TRAINABLE
Decoder:  TRAINABLE
Loss:     0.3 * CTC + 0.7 * Attention
LR:       1e-5 (very low, fine-tuning)
Augmentation: Sliding window enabled
```

**What happens:** The model is fine-tuned on sliding window segments, simulating real-time streaming input. This teaches the model to handle partial gestures and transitions between signs, which is critical for deployment.

**What to expect:**
- Small but consistent improvements
- Model becomes more robust to input timing variations
- Validation accuracy should stabilize or improve slightly

### Loss Function

The hybrid CTC-Attention loss combines two objectives:

```
L_total = alpha * L_CTC + (1 - alpha) * L_attention

where:
  L_CTC       = CTC loss (monotonic alignment between input and output)
  L_attention  = Cross-entropy loss from the attention decoder
  alpha        = 0.3 (default, configurable)
```

**CTC Loss** forces the model to learn a monotonic input-to-output alignment without requiring explicit frame-level labels. It handles variable-length inputs and outputs naturally.

**Attention Loss** allows the model to learn more flexible sequence-to-sequence mappings through cross-attention, producing higher-quality predictions.

### Optimizer and Scheduler

- **AdamW** optimizer with weight decay (prevents overfitting)
- **Cosine Warmup Scheduler**: Linear warmup for `warmup_steps` steps, then cosine annealing decay
- **Gradient clipping** at 1.0 (prevents exploding gradients)

```
LR
 ^
 |     /\
 |    /  \
 |   /    \___
 |  /         \___
 | /              \___
 |/                   \___
 +--------------------------> steps
   warmup    cosine decay
```

---

## Data Preparation

### Vietnamese Sign Language Datasets

Three publicly available datasets are supported:

#### Option 1: Kaggle VSL (Recommended for beginners)

The VSL-Vietnamese Sign Languages dataset contains video recordings with Vietnamese gloss labels.

```bash
# 1. Download from Kaggle
#    https://www.kaggle.com/datasets/phamminhhoang/vsl-vietnamese-sign-languages
#    Extract to data/raw/kaggle_vsl/

# 2. Expected structure:
#    data/raw/kaggle_vsl/
#      videos/        -> .mp4 files
#      label.csv      -> ID, VIDEO, LABEL

# 3. Run preparation
python scripts/prepare_vsl_data.py \
  --source kaggle \
  --data_dir data/raw/kaggle_vsl \
  --output_dir data/processed \
  --target_fps 60
```

This will:
- Extract hand keypoints from each video using MediaPipe
- Resample to 60 FPS
- Split into train/val/test (80/10/10)
- Generate `label_map.json` with Vietnamese gloss-to-ID mapping

#### Option 2: HuggingFace VOYA_VSL (Pre-extracted keypoints)

Already contains MediaPipe keypoints — no video processing needed.

```bash
# Requires: pip install datasets
python scripts/prepare_vsl_data.py \
  --source huggingface \
  --output_dir data/processed
```

#### Option 3: Custom Video Collection

If you have your own Vietnamese sign language videos:

```bash
# 1. Create a CSV file mapping filenames to labels:
#    filename,label
#    video_001.mp4,xin_chao
#    video_002.mp4,cam_on
#    ...

# 2. Run preparation
python scripts/prepare_vsl_data.py \
  --source video \
  --data_dir path/to/your/videos \
  --label_csv path/to/labels.csv \
  --output_dir data/processed
```

### Output Format

After preparation, your data directory looks like:

```
data/processed/
  train/
    features/
      sample_00000.npy    # shape (T, 42, 7) float32
      sample_00001.npy
      ...
    labels/
      sample_00000.npy    # shape (num_glosses,) int64
      sample_00001.npy
      ...
  val/
    features/
    labels/
  test/
    features/
    labels/
  label_map.json          # {"<blank>": 0, "xin_chao": 1, "cam_on": 2, ...}
```

Each feature file contains:
- **T** frames at 60 FPS
- **42** joints (21 left hand + 21 right hand, MediaPipe convention)
- **7** features per joint: x, y, z, velocity_x, velocity_y, velocity_z, confidence

### Updating Config

After preparing data, update `configs/config.yaml` to match your dataset:

```yaml
model:
  decoder:
    # Set this to the number of glosses + 1 (for CTC blank token)
    vocab_size: <NUMBER_FROM_label_map.json>

data:
  sample_rate: 60
```

---

## Configuration

Key parameters in `configs/config.yaml`:

### Model Size

For small datasets (<500 samples), reduce model capacity to prevent overfitting:

```yaml
model:
  frontend:
    d_model: 256           # Reduced from 512
    dropout: 0.2           # Increased dropout
    spatial_dropout: 0.2
  encoder:
    num_heads: 4           # Reduced from 8
    num_layers: 4           # Reduced from 6
    d_model: 256
    d_ff: 1024
    dropout: 0.2
```

For large datasets (>10,000 samples), use default or increase:

```yaml
model:
  frontend:
    d_model: 512
  encoder:
    num_heads: 8
    num_layers: 6
    d_model: 512
    d_ff: 2048
```

### Training Hyperparameters

```yaml
training:
  stage1:
    epochs: 30             # Increase if loss hasn't converged
    lr: 1.0e-3             # High LR for fast frontend warm-up
    batch_size: 32         # Reduce if GPU OOM

  stage2:
    epochs: 100            # Main training, most epochs here
    lr: 5.0e-5             # Low LR to preserve encoder knowledge
    batch_size: 16         # Smaller batch for stability
    alpha: 0.3             # CTC weight (0.3 = 30% CTC, 70% Attention)

  stage3:
    epochs: 30             # Fine-tuning
    lr: 1.0e-5
    batch_size: 16
    alpha: 0.3

  warmup_steps: 500        # LR warmup steps (reduce for small datasets)
  grad_clip: 1.0           # Max gradient norm
```

---

## Running Training

### Full Pipeline

```bash
# Train all 3 stages sequentially
python scripts/train.py \
  --config configs/config.yaml \
  --data_dir data/processed \
  --device cuda
```

### Stage by Stage

```bash
# Stage 1: Frontend warm-up
python scripts/train.py --config configs/config.yaml --stage 1 --data_dir data/processed

# Stage 2: Joint training (resume from Stage 1 checkpoint)
python scripts/train.py --config configs/config.yaml --stage 2 \
  --resume checkpoints/best_stage1.pt --data_dir data/processed

# Stage 3: Real-time optimization (resume from Stage 2)
python scripts/train.py --config configs/config.yaml --stage 3 \
  --resume checkpoints/best_stage2.pt --data_dir data/processed
```

### Resume Interrupted Training

```bash
python scripts/train.py --config configs/config.yaml \
  --resume checkpoints/latest_checkpoint.pt --data_dir data/processed
```

### Google Colab

See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for complete Colab setup with GPU.

---

## Monitoring and Evaluation

### TensorBoard

```bash
tensorboard --logdir logs/
```

Key metrics to watch:
- `loss/total` — Should decrease across all stages
- `loss/ctc` — CTC component, should converge in Stage 1
- `loss/attention` — Attention component, main driver in Stage 2+
- `lr` — Learning rate schedule (verify warmup + decay)

### Expected Training Curves

```
Loss
 ^
 4 |*.
   | *.          Stage 1           Stage 2              Stage 3
 3 |  *..
   |    *..
 2 |      **..
   |         ***...
 1 |              ******.....
   |                       ********......
 0 +--|---------|-------|----------|---------|---------> Epoch
   0  10       30      60         100      130       160
```

### Inference Test

```python
import torch
from src.model.whisper_sign import WhisperSignModel

model, config = WhisperSignModel.load_checkpoint("checkpoints/best_stage3.pt")
model.eval()

# Test with random data
x = torch.randn(1, 120, 42, 7)
lengths = torch.tensor([120])

with torch.no_grad():
    predictions = model.decode(x, lengths)

print(f"Predictions: {predictions}")
```

---

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or model size:
```yaml
training:
  stage2:
    batch_size: 8     # Halve the batch size
```

Or reduce sequence length:
```yaml
data:
  max_seq_length: 750  # Halve from 1500
```

### Loss Not Decreasing in Stage 1

- Verify data is loaded correctly: run `python scripts/smoke_test.py`
- Check that features are normalized (not raw pixel coordinates)
- Increase learning rate to 5e-3
- Increase epochs to 50

### Loss Exploding (NaN)

- Reduce learning rate
- Enable gradient clipping (should be on by default)
- Check for corrupted data files (NaN values in .npy files)

### Overfitting (Train Loss Low, Val Loss High)

- Increase dropout: set all dropout values to 0.2-0.3
- Reduce model size (d_model, num_layers)
- Enable all augmentations in config
- Collect more training data

### Slow Training

- Use GPU: `--device cuda`
- Increase `num_workers` in config (4-8 for most systems)
- Reduce `max_seq_length` if sequences are very long
- Use mixed precision: add `torch.cuda.amp` in trainer (not yet built-in)
