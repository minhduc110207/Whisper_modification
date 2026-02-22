<p align="center">
  <h1 align="center">🤟 WhisperSign</h1>
  <p align="center">
    <strong>Modified OpenAI Whisper for Real-Time Sign Language Recognition</strong><br>
    <em>From Audio Spectrograms to 3D Skeletal Data</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
    <img src="https://img.shields.io/badge/Tests-112%20passed-brightgreen" alt="Tests">
  </p>
</p>

---

## 📖 Overview

**WhisperSign** reimagines OpenAI's Whisper architecture for sign language recognition. Instead of processing audio via Log-Mel Spectrograms, it accepts **3D skeletal hand data** `(T × 42 × 7)` from Leap Motion or MediaPipe — 42 hand joints (21 per hand) with 7 features each (x, y, z, velocity_x, velocity_y, velocity_z, confidence).

The model outputs **sign glosses** — semantic labels for individual signs — enabling real-time translation of hand gestures into text.

### Why Modify Whisper?

| Challenge | Whisper's Strength | Our Adaptation |
|-----------|-------------------|----------------|
| Temporal sequence modeling | Proven on variable-length audio | Applied to variable-length gesture sequences |
| Noisy real-world input | Robust to audio noise | Robust to skeletal tracking noise |
| Multi-scale pattern detection | Phoneme → word → sentence | Finger config → hand shape → sign phrase |
| Real-time streaming | Efficient attention mechanism | Sliding window inference for live translation |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WhisperSign Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Leap Motion / MediaPipe                                    │
│       │                                                     │
│       ▼                                                     │
│  Raw Skeletal Data (T, 42, 7)                               │
│       │                                                     │
│       ▼                                                     │
│  ┌───────────────────────────────────┐                      │
│  │  FRONTEND (replaces Mel Spectrogram)                     │
│  │  ┌─────────────────────────┐                             │
│  │  │ Temporal Patch Embedding│  Groups P frames → patches  │
│  │  │ (T,42,7) → (T/P, d)    │  Reduces sequence T → T/P  │
│  │  └────────┬────────────────┘                             │
│  │           ▼                                              │
│  │  ┌─────────────────────────┐                             │
│  │  │ ConvSPE                 │  Learns spatial-positional  │
│  │  │ Depthwise + Pointwise   │  relationships dynamically  │
│  │  └────────┬────────────────┘                             │
│  │           ▼                                              │
│  │  BatchNorm → SpatialDropout1D                            │
│  └───────────┬───────────────────┘                          │
│              ▼                                              │
│  ┌───────────────────────────────────┐                      │
│  │  ENCODER (Spatio-Temporal Blocks × N)                    │
│  │  ┌─────────────────────────┐                             │
│  │  │ S-MHSA                  │  Spatial: handshape at t    │
│  │  │ (Spatial Self-Attention)│  "What fingers are where?"  │
│  │  └────────┬────────────────┘                             │
│  │           ▼                                              │
│  │  ┌─────────────────────────┐                             │
│  │  │ T-MHSA + RPE            │  Temporal: motion over time│
│  │  │ (Temporal Self-Attention)│  "How does the hand move?" │
│  │  └────────┬────────────────┘                             │
│  │           ▼                                              │
│  │  ┌─────────────────────────┐                             │
│  │  │ Feed-Forward (Pre-Norm) │  GELU activation            │
│  │  └────────┬────────────────┘                             │
│  └───────────┼───────────────────┘                          │
│              ▼                                              │
│  ┌───────────────────────────────────┐                      │
│  │  DECODER (Two-Pass)                                      │
│  │                                                          │
│  │  Pass 1: CTC Head ──────► Fast monotonic alignment       │
│  │          (Linear → LogSoftmax → Greedy Decode)           │
│  │                                                          │
│  │  Pass 2: Attention Decoder ─► Rescoring for accuracy     │
│  │          (Transformer Decoder with causal mask)          │
│  └───────────┬───────────────────┘                          │
│              ▼                                              │
│       Sign Glosses: ["hello", "thank_you", "please"]        │
└─────────────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

| Component | Design Choice | Rationale |
|-----------|--------------|-----------|
| **Frontend** | Temporal Patch Embedding (not Conv2D) | Skeletal data is structured (joints × features), not pixels. Patches group P consecutive frames. |
| **Positional Encoding** | ConvSPE (learned, convolutional) | Hand skeleton has fixed topology — learned spatial relationships outperform sinusoidal PE. |
| **Encoder Attention** | Dual S-MHSA + T-MHSA (not unified) | Separating spatial and temporal attention allows the model to independently reason about *what the hand looks like* vs *how it moves*. |
| **Temporal Attention** | Relative Positional Encoding (RPE) | Sign dynamics depend on *relative* timing (how long between movements), not absolute position. |
| **Decoder** | CTC + Attention (two-pass) | CTC provides fast, monotonic alignment; Attention rescoring improves accuracy. Combined hybrid loss: `L = α·CTC + (1-α)·Attention`. |
| **Normalization** | Pre-Norm (LayerNorm before attention) | More stable training with deeper networks, better gradient flow. |

---

## ✨ Features

### Data Pipeline
- **Spline Interpolation Resampling** — Converts variable frame rates to fixed 60 Hz using cubic spline interpolation
- **Hand-Centric Spatial Normalization** — Translates coordinates so palm joint is at origin (left and right hands independently)
- **Scale Normalization** — Normalizes by metacarpal bone length for hand-size invariance
- **Feature Scaling** — StandardScaler or MinMaxScaler applied across the dataset
- **Gesture Masking Augmentation** — Randomly masks joints or temporal segments (like SpecAugment for audio)
- **Temporal Jitter** — Random frame shifting for temporal robustness
- **Noise Injection** — Gaussian noise to simulate sensor inaccuracy

### Model
- **4.2M parameters** (base config, d_model=512) — lightweight enough for real-time inference
- **Configurable depth** — Scale from tiny (d_model=128) to large (d_model=768)
- **Freeze/Unfreeze API** — Selective component training for transfer learning
- **Checkpoint save/load** — Full state persistence including epoch, loss, and config

### Training
- **3-Stage Progressive Training**
  - Stage 1: Frontend warm-up (encoder + decoder frozen)
  - Stage 2: Joint training with hybrid CTC-Attention loss
  - Stage 3: Real-time optimization with sliding window augmentation
- **Hybrid CTC-Attention Loss** with configurable weight α
- **Cosine Warmup Scheduler** — Linear warmup followed by cosine annealing
- **Gradient Clipping** — Prevents training instability
- **TensorBoard Logging** — Real-time loss and metric visualization

### Inference
- **Sliding Window Inference** — Process continuous streams in real-time with configurable overlap
- **Moving Average Smoothing** — Reduces sensor noise in live data
- **MediaPipe Integration** — Extract hand keypoints directly from video

---

## 📊 Technical Specifications

### Model Configurations

| Config | d_model | Layers | Heads | Params | GPU Memory | Use Case |
|--------|---------|--------|-------|--------|------------|----------|
| Tiny | 128 | 2 | 4 | ~1.1M | ~2 GB | Prototyping, edge devices |
| Base | 256 | 4 | 4 | ~4.2M | ~4 GB | Balanced performance |
| **Default** | **512** | **6** | **8** | **~18M** | **~8 GB** | **Recommended** |
| Large | 768 | 8 | 12 | ~45M | ~16 GB | Maximum accuracy |

### Input Format

| Feature | Index | Description |
|---------|-------|-------------|
| x, y, z | 0-2 | 3D joint coordinates (meters) |
| vx, vy, vz | 3-5 | Joint velocities (m/s) |
| confidence | 6 | Tracking confidence [0, 1] |

**Joint Layout:** 21 joints per hand × 2 hands = 42 joints total (following MediaPipe hand landmark convention)

### Training Pipeline

```
Stage 1: Frontend Warm-up          Stage 2: Joint Training          Stage 3: Real-time Opt.
┌──────────────────────┐     ┌──────────────────────────┐     ┌────────────────────────┐
│ Frontend: TRAINABLE  │     │ Frontend: TRAINABLE      │     │ Frontend: TRAINABLE    │
│ Encoder:  FROZEN     │ ──► │ Encoder:  TRAINABLE      │ ──► │ Encoder:  TRAINABLE    │
│ Decoder:  FROZEN     │     │ Decoder:  TRAINABLE      │     │ Decoder:  TRAINABLE    │
│ LR: 1e-3             │     │ LR: 5e-5                 │     │ LR: 1e-5              │
│ Loss: CTC only       │     │ Loss: 0.3·CTC + 0.7·ATT │     │ Loss: 0.3·CTC+0.7·ATT │
│ Epochs: 30           │     │ Epochs: 100              │     │ Epochs: 30             │
└──────────────────────┘     └──────────────────────────┘     └────────────────────────┘
```

---

## 🧪 Verification

The model has been verified with **112 tests** across two test suites:

### Structural Tests (68/68 passed)
- Tensor shape propagation through all components
- Gradient flow from loss to every trainable parameter
- Hybrid loss formula correctness (`L = α·CTC + (1-α)·Attention`)
- Checkpoint save/load round-trip (weights identical)
- Numerical stability with extreme inputs (×100, ×0.001)
- Edge cases: batch=1, minimum sequence length, all-zero input

### Functional Tests (44/44 passed)
- **CTC Decoding**: Blank removal, deduplication, alternating patterns all correct
- **Causal Mask**: Verified no future information leakage in attention decoder
- **Encoder Masking**: Padded positions properly ignored (cosine similarity = 0.975)
- **RPE**: Shift-invariant, distance-differentiating relative position encoding
- **Memorization**: Loss 3.678 → 0.006 in 80 steps, 4/4 samples decoded correctly
- **Gradient Health**: No vanishing/exploding, frontend↔encoder ratio = 1.3×
- **End-to-End**: Full numpy → normalize → preprocess → model → decode pipeline

Run the tests yourself:
```bash
python scripts/smoke_test.py       # Quick sanity check (~10s)
python scripts/deep_test.py        # Structural tests (~30s)
python scripts/functional_test.py  # Functional tests (~60s)
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Whisper_modification.git
cd Whisper_modification
pip install -r requirements.txt
```

### Quick Verification

```bash
python scripts/smoke_test.py
```

### Training

```bash
# Train all 3 stages sequentially
python scripts/train.py --config configs/config.yaml --data_dir data/processed

# Train individual stages
python scripts/train.py --config configs/config.yaml --stage 1
python scripts/train.py --config configs/config.yaml --stage 2 --resume checkpoints/best_stage1.pt
python scripts/train.py --config configs/config.yaml --stage 3 --resume checkpoints/best_stage2.pt

# Specify device
python scripts/train.py --config configs/config.yaml --device cuda
```

### Inference

```python
import torch
from src.model.whisper_sign import WhisperSignModel

# Load trained model
model, _ = WhisperSignModel.load_checkpoint("checkpoints/final_model.pt")
model.eval()

# Run inference on skeletal data
data = torch.randn(1, 120, 42, 7)  # (batch, frames, joints, features)
lengths = torch.tensor([120])
predictions = model.decode(data, lengths)
print(f"Predicted signs: {predictions[0]}")
```

### Real-Time Streaming

```python
from src.utils.sliding_window import SlidingWindowInference
from src.utils.smoothing import MovingAverageSmoothing

# Setup
model, _ = WhisperSignModel.load_checkpoint("checkpoints/final_model.pt")
smoother = MovingAverageSmoothing(window_size=5)
slider = SlidingWindowInference(model, window_duration=1.0, overlap=0.5)

# Process live stream
stream_data = get_leap_motion_stream()  # Your data source
smoothed = smoother.smooth(stream_data)
predictions = slider(smoothed)
```

### Training on Google Colab

See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for a complete step-by-step guide with ready-to-run notebook cells.

---

## 📁 Project Structure

```
Whisper_modification/
├── configs/
│   └── config.yaml                 # Model & training hyperparameters
├── src/
│   ├── data/
│   │   ├── preprocessing.py        # Resampling, windowing, padding
│   │   ├── normalization.py        # Spatial, scale, feature normalization
│   │   ├── augmentation.py         # Masking, jitter, noise injection
│   │   └── dataset.py              # PyTorch Dataset & DataLoader
│   ├── model/
│   │   ├── frontend.py             # Patch Embedding + ConvSPE + Dropout
│   │   ├── positional.py           # RPE + Sinusoidal PE
│   │   ├── encoder.py              # S-MHSA + T-MHSA Transformer blocks
│   │   ├── decoder.py              # CTC + Attention two-pass decoder
│   │   └── whisper_sign.py         # Main model class
│   ├── training/
│   │   ├── losses.py               # Hybrid CTC-Attention loss
│   │   ├── scheduler.py            # Cosine warmup scheduler
│   │   └── trainer.py              # 3-stage training orchestrator
│   └── utils/
│       ├── sliding_window.py       # Real-time sliding window inference
│       ├── smoothing.py            # Moving average noise filter
│       └── mediapipe_extract.py    # Video → hand keypoints extraction
├── scripts/
│   ├── train.py                    # CLI training entry point
│   ├── smoke_test.py               # Quick sanity check
│   ├── deep_test.py                # 68 structural tests
│   └── functional_test.py          # 44 functional tests
├── tests/
│   └── test_model.py               # Pytest unit tests
├── COLAB_TRAINING_GUIDE.md         # Google Colab training guide
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🔧 Configuration

All hyperparameters are managed through `configs/config.yaml`:

```yaml
model:
  frontend:
    num_joints: 42          # 21 left + 21 right hand joints
    num_features: 7         # x, y, z, vx, vy, vz, confidence
    patch_size: 4           # Temporal grouping factor
    d_model: 512            # Hidden dimension
  encoder:
    num_heads: 8            # Multi-head attention heads
    num_layers: 6           # Transformer blocks
    d_ff: 2048              # Feed-forward dimension
  decoder:
    vocab_size: 1296        # Number of sign glosses
    blank_id: 0             # CTC blank token

training:
  stage1: { epochs: 30,  lr: 1e-3, freeze_encoder: true }
  stage2: { epochs: 100, lr: 5e-5, alpha: 0.3 }
  stage3: { epochs: 30,  lr: 1e-5, alpha: 0.3 }
```

---

## 📚 Data Format

### Input Data Structure

```
data/processed/
├── train/
│   ├── features/          # .npy files, shape (T, 42, 7)
│   └── labels/            # .npy files, integer arrays
├── val/
│   ├── features/
│   └── labels/
└── test/
    ├── features/
    └── labels/
```

### Supported Data Sources

| Source | Joints | FPS | Notes |
|--------|--------|-----|-------|
| **Leap Motion** | 42 (2×21) | 120 Hz | Highest accuracy, requires hardware |
| **MediaPipe** | 42 (2×21) | 30-60 Hz | Camera-based, use `mediapipe_extract.py` |
| **Custom** | Any | Any | Resample to 60 Hz using `preprocessing.py` |

---

## 📝 References

- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Radford et al., 2022
- [CTC: Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — Graves et al., 2006
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) — Google, 2020

---

## 📄 License

This project is licensed under the MIT License.
