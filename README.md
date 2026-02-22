# WhisperSign

**Modified Whisper OpenAI for Sign Language Recognition using Leap Motion 3D Data**

WhisperSign transforms the Whisper model from speech recognition to sign language recognition. Instead of receiving audio (Log-Mel Spectrogram), the model accepts 3D tensors `(T x 42 x F)` containing coordinates of 42 hand joints from Leap Motion or MediaPipe.

## Architecture

```
Skeletal Data (T, 42, 7)
  -> Frontend (Temporal Patch Embedding + ConvSPE + BatchNorm)
  -> Encoder (Spatio-temporal Blocks: S-MHSA + T-MHSA)
  -> Decoder (CTC Pass 1 + Attention Rescoring Pass 2)
  -> Sign Glosses
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Verification

```bash
python scripts/smoke_test.py
```

## Training

```bash
# Train all 3 stages
python scripts/train.py --config configs/config.yaml --data_dir data/processed

# Train individual stages
python scripts/train.py --config configs/config.yaml --stage 1
python scripts/train.py --config configs/config.yaml --stage 2 --resume checkpoints/best_stage1.pt
python scripts/train.py --config configs/config.yaml --stage 3 --resume checkpoints/best_stage2.pt
```

## Training on Google Colab

See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for detailed instructions.

## Project Structure

```
├── configs/config.yaml          # Model & training configuration
├── src/
│   ├── data/                    # Preprocessing, normalization, augmentation
│   ├── model/                   # Frontend, Encoder, Decoder, WhisperSign
│   ├── training/                # Trainer, Loss, Scheduler
│   └── utils/                   # Sliding window, smoothing, MediaPipe
├── scripts/                     # Train, evaluate, smoke test
├── tests/                       # Unit tests
└── COLAB_TRAINING_GUIDE.md      # Colab training guide
```
