"""
WhisperSign — Kaggle Training Notebook
=======================================
Copy-paste this entire script into a single Kaggle notebook cell,
or split at the # %% markers into multiple cells.

Requirements:
  - Kaggle Notebook with GPU accelerator enabled (P100/T4)
  - Upload your WhisperSign project as a Kaggle Dataset, OR
    clone from GitHub

How to use:
  1. Create a new Kaggle Notebook
  2. Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100)
  3. Add your project as a dataset, OR set USE_GITHUB = True below
  4. Copy-paste this script and run
"""

# ============================================================
# CONFIG — Edit these values before running
# ============================================================
USE_GITHUB = True                        # True = clone from GitHub, False = use Kaggle dataset
GITHUB_REPO = "minhduc110207/Whisper_modification"  # Your GitHub repo
KAGGLE_DATASET_SLUG = None               # e.g. "yourname/whispersign" if USE_GITHUB=False
USE_DUMMY_DATA = True                    # True = create synthetic data for testing pipeline
NUM_DUMMY_SAMPLES = 200                  # Number of dummy training samples
NUM_DUMMY_CLASSES = 50                   # Number of dummy sign classes
SMALL_MODEL = True                       # True = d_model=256 (faster), False = d_model=512 (full)

# %% [markdown]
# ## Cell 1: Setup Environment

# %%
import os
import sys
import subprocess

WORK_DIR = "/kaggle/working"
PROJECT_DIR = os.path.join(WORK_DIR, "WhisperSign")

# --- Get the project code ---
if USE_GITHUB:
    if not os.path.exists(PROJECT_DIR):
        print(f"Cloning from GitHub: {GITHUB_REPO}")
        subprocess.run([
            "git", "clone", f"https://github.com/{GITHUB_REPO}.git", PROJECT_DIR
        ], check=True)
    else:
        print(f"Project already exists at {PROJECT_DIR}")
elif KAGGLE_DATASET_SLUG:
    # Copy from Kaggle dataset input
    dataset_path = f"/kaggle/input/{KAGGLE_DATASET_SLUG.split('/')[-1]}"
    if os.path.exists(dataset_path):
        subprocess.run(["cp", "-r", dataset_path, PROJECT_DIR], check=True)
        print(f"Copied dataset from {dataset_path}")
    else:
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Available inputs:", os.listdir("/kaggle/input/"))
        raise FileNotFoundError(f"Dataset {KAGGLE_DATASET_SLUG} not found")

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Install dependencies
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "scipy", "scikit-learn", "pyyaml", "tqdm", "tensorboard", "matplotlib"
], check=True)

# Check GPU
import torch
print(f"\n{'='*50}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"{'='*50}")

# %% [markdown]
# ## Cell 2: Smoke Test

# %%
print("Running smoke test...")
exec(open(os.path.join(PROJECT_DIR, "scripts", "smoke_test.py")).read())
print("\n✅ Smoke test passed!")

# %% [markdown]
# ## Cell 3: Create or Load Data

# %%
import numpy as np

DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

if USE_DUMMY_DATA:
    print("Creating synthetic training data...")
    for split in ['train', 'val', 'test']:
        n = NUM_DUMMY_SAMPLES if split == 'train' else NUM_DUMMY_SAMPLES // 5
        feat_dir = os.path.join(DATA_DIR, split, 'features')
        label_dir = os.path.join(DATA_DIR, split, 'labels')
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for i in range(n):
            T = np.random.randint(60, 300)
            features = np.random.randn(T, 42, 7).astype(np.float32) * 0.1
            num_labels = np.random.randint(1, 5)
            labels = np.random.randint(1, NUM_DUMMY_CLASSES, size=num_labels).astype(np.int64)

            np.save(os.path.join(feat_dir, f'sample_{i:05d}.npy'), features)
            np.save(os.path.join(label_dir, f'sample_{i:05d}.npy'), labels)

        print(f"  {split}: {n} samples")

    # Save label map
    import json
    label_map = {"<blank>": 0}
    label_map.update({f"sign_{i}": i for i in range(1, NUM_DUMMY_CLASSES)})
    with open(os.path.join(DATA_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Vocab size: {len(label_map)}")
else:
    # Check if data already exists
    train_feat = os.path.join(DATA_DIR, "train", "features")
    if os.path.exists(train_feat):
        n = len([f for f in os.listdir(train_feat) if f.endswith('.npy')])
        print(f"Found existing data: {n} training samples")
    else:
        print("ERROR: No data found! Set USE_DUMMY_DATA=True or provide data.")
        print(f"Expected data at: {DATA_DIR}")

print("\n✅ Data ready!")

# %% [markdown]
# ## Cell 4: Configure Model & Training

# %%
import yaml

d_model = 256 if SMALL_MODEL else 512
num_heads = 4 if SMALL_MODEL else 8
num_layers = 4 if SMALL_MODEL else 6
d_ff = 1024 if SMALL_MODEL else 2048
vocab_size = NUM_DUMMY_CLASSES if USE_DUMMY_DATA else 1296

config = {
    "model": {
        "frontend": {
            "num_joints": 42,
            "num_features": 7,
            "patch_size": 4,
            "d_model": d_model,
            "dropout": 0.15,
            "spatial_dropout": 0.15,
        },
        "encoder": {
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_model": d_model,
            "d_ff": d_ff,
            "dropout": 0.15,
        },
        "decoder": {
            "vocab_size": vocab_size,
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
        "num_workers": 2,
        "save_dir": os.path.join(PROJECT_DIR, "checkpoints"),
        "log_dir": os.path.join(PROJECT_DIR, "logs"),
    },
}

# Save config
config_path = os.path.join(PROJECT_DIR, "configs", "config_kaggle.yaml")
os.makedirs(os.path.dirname(config_path), exist_ok=True)
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Print model info
from src.model.whisper_sign import WhisperSignModel

model = WhisperSignModel(config["model"])
total = model.get_num_params(trainable_only=False)
print(f"Model: d_model={d_model}, layers={num_layers}, heads={num_heads}")
print(f"Parameters: {total:,} ({total * 4 / 1e6:.1f} MB)")
print(f"Vocab size: {vocab_size}")
print(f"\n✅ Config saved to {config_path}")

# %% [markdown]
# ## Cell 5: Stage 1 — Frontend Warm-up

# %%
import torch
import numpy as np
from src.model.whisper_sign import WhisperSignModel
from src.data.dataset import create_dataloaders
from src.training.trainer import WhisperSignTrainer

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperSignModel(config["model"]).to(device)

data_cfg = config["data"]
train_cfg = config["training"]

train_loader, val_loader, _ = create_dataloaders(
    data_dir=DATA_DIR,
    config=data_cfg,
    batch_size=train_cfg["stage1"]["batch_size"],
    num_workers=train_cfg["num_workers"],
)
print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
print(f"Val: {len(val_loader.dataset)} samples")

trainer = WhisperSignTrainer(
    model=model, config=train_cfg, device=device,
    save_dir=train_cfg["save_dir"],
    log_dir=train_cfg["log_dir"],
)

print("\n🚀 Starting Stage 1: Frontend Warm-up")
print("   Frontend: TRAINABLE | Encoder: FROZEN | Decoder: FROZEN")
print(f"   Epochs: {train_cfg['stage1']['epochs']} | LR: {train_cfg['stage1']['lr']}")
print("-" * 60)

trainer.train_stage1(train_loader, val_loader)

# Save checkpoint to /kaggle/working (persists after notebook ends)
import shutil
CKPT_OUTPUT = "/kaggle/working/checkpoints"
os.makedirs(CKPT_OUTPUT, exist_ok=True)
stage1_src = os.path.join(train_cfg["save_dir"], "final_stage1.pt")
stage1_dst = os.path.join(CKPT_OUTPUT, "final_stage1.pt")
if os.path.exists(stage1_src):
    shutil.copy(stage1_src, stage1_dst)
    print(f"\n✅ Stage 1 checkpoint saved to {stage1_dst}")

# %% [markdown]
# ## Cell 6: Stage 2 — Joint Training

# %%
# If Kaggle session restarted, uncomment these lines:
# model, ckpt = WhisperSignModel.load_checkpoint("/kaggle/working/checkpoints/final_stage1.pt", device)
# print(f"Resumed from epoch {ckpt['epoch']}")

train_loader2, val_loader2, _ = create_dataloaders(
    data_dir=DATA_DIR,
    config=data_cfg,
    batch_size=train_cfg["stage2"]["batch_size"],
    num_workers=train_cfg["num_workers"],
)

print("🚀 Starting Stage 2: Joint Training")
print("   Frontend: TRAINABLE | Encoder: TRAINABLE | Decoder: TRAINABLE")
print(f"   Epochs: {train_cfg['stage2']['epochs']} | LR: {train_cfg['stage2']['lr']}")
print(f"   Loss: {train_cfg['stage2']['alpha']}×CTC + {1-train_cfg['stage2']['alpha']}×Attention")
print("-" * 60)

trainer.train_stage2(train_loader2, val_loader2)

stage2_src = os.path.join(train_cfg["save_dir"], "final_stage2.pt")
stage2_dst = os.path.join(CKPT_OUTPUT, "final_stage2.pt")
if os.path.exists(stage2_src):
    shutil.copy(stage2_src, stage2_dst)
    print(f"\n✅ Stage 2 checkpoint saved to {stage2_dst}")

# %% [markdown]
# ## Cell 7: Stage 3 — Real-time Optimization

# %%
print("🚀 Starting Stage 3: Real-time Optimization")
print("   All layers TRAINABLE | Sliding Window enabled")
print(f"   Epochs: {train_cfg['stage3']['epochs']} | LR: {train_cfg['stage3']['lr']}")
print("-" * 60)

trainer.train_stage3(train_loader2, val_loader2)

final_src = os.path.join(train_cfg["save_dir"], "final_model.pt")
final_dst = os.path.join(CKPT_OUTPUT, "final_model.pt")
if os.path.exists(final_src):
    shutil.copy(final_src, final_dst)
    print(f"\n✅ Final model saved to {final_dst}")
else:
    # Save manually if trainer didn't
    model.save_checkpoint(final_dst, epoch=999, loss=0.0)
    print(f"\n✅ Final model saved to {final_dst}")

# %% [markdown]
# ## Cell 8: Evaluate

# %%
model.eval()
_, _, test_loader = create_dataloaders(
    data_dir=DATA_DIR,
    config=data_cfg,
    batch_size=16,
    num_workers=2,
)

all_preds = []
with torch.no_grad():
    for batch in test_loader:
        features = batch["features"].to(device)
        lengths = batch["feature_lengths"].to(device)
        preds = model.decode(features, lengths)
        all_preds.extend(preds)

print(f"Test samples: {len(all_preds)}")
print(f"\nSample predictions:")
for i in range(min(10, len(all_preds))):
    print(f"  Sample {i}: {all_preds[i]}")

# %% [markdown]
# ## Cell 9: Download Checkpoints
# After the notebook finishes, all files in `/kaggle/working/checkpoints/`
# will be available in the **Output** tab for download.

# %%
print("\n📦 Output files (available in 'Output' tab after notebook completes):")
if os.path.exists(CKPT_OUTPUT):
    for f in sorted(os.listdir(CKPT_OUTPUT)):
        size = os.path.getsize(os.path.join(CKPT_OUTPUT, f)) / 1e6
        print(f"  {f} ({size:.1f} MB)")
print("\nDone! 🎉")

# %% [markdown]
# ## [Optional] Cell 10: TensorBoard
# Run this cell to see training loss/LR curves.

# %%
# Uncomment and run in Kaggle notebook:
# %load_ext tensorboard
# %tensorboard --logdir /kaggle/working/WhisperSign/logs
print(f"TensorBoard logs are at: {train_cfg['log_dir']}")

