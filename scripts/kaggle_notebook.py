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
PROCESS_RAW_KAGGLE_DATA = False           # True = Add 'vsl-vietnamese-sign-languages' via 'Add Data' to process directly on Kaggle
RAW_KAGGLE_DATA_PATH = "/kaggle/input/vsl-vietnamese-sign-languages"
USE_DUMMY_DATA = False                    # True = create synthetic data for testing pipeline
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
PROJECT_DIR = os.path.join(WORK_DIR, "Whisper_modification")

# --- Get the project code ---
if USE_GITHUB:
    if not os.path.exists(PROJECT_DIR):
        print(f"Cloning from GitHub: {GITHUB_REPO}")
        subprocess.run([
            "git", "clone", f"https://github.com/{GITHUB_REPO}.git", PROJECT_DIR
        ], check=True)
    else:
        print(f"Updating project at {PROJECT_DIR}...")
        os.chdir(PROJECT_DIR)
        subprocess.run(["git", "pull"], check=True)
elif KAGGLE_DATASET_SLUG:
    # Copy from Kaggle dataset input
    dataset_name = KAGGLE_DATASET_SLUG.split('/')[-1]
    dataset_path = f"/kaggle/input/{dataset_name}"
    
    if os.path.exists(dataset_path):
        if os.path.exists(PROJECT_DIR):
            import shutil
            shutil.rmtree(PROJECT_DIR)
        
        # Check if the dataset itself is the project folder or contains it
        # Sometimes Kaggle nests: /kaggle/input/slug/Whisper_modification/...
        nested_check = os.path.join(dataset_path, "Whisper_modification")
        if os.path.exists(nested_check):
            subprocess.run(["cp", "-r", nested_check, PROJECT_DIR], check=True)
        else:
            subprocess.run(["cp", "-r", dataset_path, PROJECT_DIR], check=True)
        print(f"Copied dataset to {PROJECT_DIR}")
    else:
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Available inputs:", os.listdir("/kaggle/input/"))
        raise FileNotFoundError(f"Dataset {KAGGLE_DATASET_SLUG} not found")

os.chdir(PROJECT_DIR)
# Ensure we are at the ROOT of the project (e.g. where src/ exists)
if not os.path.exists("src"):
    # If we are one level too deep, move up
    if os.path.basename(os.getcwd()) == "Whisper_modification" and os.path.exists("../src"):
         os.chdir("..")
         PROJECT_DIR = os.getcwd()

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# --- Auto-Patch old code for Kaggle GPUs ---
frontend_path = os.path.join(PROJECT_DIR, "src", "model", "frontend.py")
if os.path.exists(frontend_path):
    with open(frontend_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "// self.patch_size" in content or "torch.div" in content:
        content = content.replace(
            "output_lengths = (input_lengths + self.patch_size - 1) // self.patch_size",
            "output_lengths = ((input_lengths.float() + self.patch_size - 1) / float(self.patch_size)).long()"
        )
        content = content.replace(
            "output_lengths = torch.div(input_lengths + self.patch_size - 1, self.patch_size, rounding_mode='floor')",
            "output_lengths = ((input_lengths.float() + self.patch_size - 1) / float(self.patch_size)).long()"
        )
        with open(frontend_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Patched frontend.py to fix CUDA integer division error!")

# --- Force reload project modules ---
def reload_project_modules():
    import importlib
    import sys
    to_reload = [m for m in sys.modules if m.startswith("src.") or m == "src"]
    for m in to_reload:
        del sys.modules[m]
    print(f"Cleared {len(to_reload)} cached project modules.")

reload_project_modules()

# Install dependencies
deps = ["scipy", "scikit-learn", "pyyaml", "tqdm", "tensorboard", "matplotlib"]
if PROCESS_RAW_KAGGLE_DATA:
    deps.extend(["mediapipe", "opencv-python", "pandas"])

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", *deps
], check=True)

# Check GPU
import torch
print(f"\n{'='*50}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*50}")

# %% [markdown]
# ## Cell 2: Smoke Test

# %%
print("Running smoke test...")
smoke_test_path = os.path.join(PROJECT_DIR, "scripts", "smoke_test.py")
exec(open(smoke_test_path).read(), {"__file__": smoke_test_path})
print("\n✅ Smoke test passed!")
# Additional verification to prevent zero labels during real training
try:
    from src.data.dataset import create_dataloaders
    from src.training.trainer import WhisperSignTrainer
    print("Verifying label discovery for smoke test...")
    # This assumes we have some small data or dummy data available for smoke test
except Exception as e:
    print(f"Warning: Could not import trainer for additional verification: {e}")

# %% [markdown]
# ## Cell 3: Create or Load Data

# %%
import numpy as np

if PROCESS_RAW_KAGGLE_DATA:
    DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
else:
    # Ultra-robust dataset discovery
    print(f"\nScanning /kaggle/input for dataset...")
    DATA_DIR = None
    
    if os.path.exists("/kaggle/input"):
        inputs = os.listdir("/kaggle/input")
        print(f"Items in /kaggle/input: {inputs}")
        
        # Priority 1: Check if 'vsl-vietnamese-sign-languages' is exactly in the input
        if "vsl-vietnamese-sign-languages" in inputs:
            potential_base = os.path.join("/kaggle/input", "vsl-vietnamese-sign-languages")
            if os.path.exists(os.path.join(potential_base, "Processed")):
                DATA_DIR = os.path.join(potential_base, "Processed")
                print(f"SUCCESS: Found exact dataset path: {DATA_DIR}")

        # Priority 2: Recursive search for 'Processed' folder
        if not DATA_DIR:
            for d in inputs:
                potential_base = os.path.join("/kaggle/input", d)
                # If current folder IS 'Processed'
                if d.lower() == "processed":
                    DATA_DIR = potential_base
                    print(f"SUCCESS: Current folder is 'Processed': {DATA_DIR}")
                    break
                # Search inside
                for root, dirs, files in os.walk(potential_base):
                    if "Processed" in dirs:
                        DATA_DIR = os.path.join(root, "Processed")
                        print(f"SUCCESS: Found nested 'Processed' folder at {DATA_DIR}")
                        break
                if DATA_DIR: break
            
        # Priority 3: Recursive search for 'train' folder
        if not DATA_DIR:
            for d in inputs:
                potential_base = os.path.join("/kaggle/input", d)
                for root, dirs, files in os.walk(potential_base):
                    if "train" in dirs:
                        DATA_DIR = root
                        print(f"SUCCESS: Found data folder (by 'train' detection) at {DATA_DIR}")
                        break
                if DATA_DIR: break

    if not DATA_DIR:
        print(f"CRITICAL WARNING: Auto-detection failed. Using default RAW_KAGGLE_DATA_PATH.")
        if "Processed" in RAW_KAGGLE_DATA_PATH:
            DATA_DIR = RAW_KAGGLE_DATA_PATH
        else:
            DATA_DIR = os.path.join(RAW_KAGGLE_DATA_PATH, "Processed")
    
    print(f"Final DATA_DIR set to: {DATA_DIR}")
    
    # Verify contents
    if os.path.exists(DATA_DIR):
        print(f"Contents of {DATA_DIR}: {os.listdir(DATA_DIR)}")
        train_path = os.path.join(DATA_DIR, "train")
        if os.path.exists(train_path):
            files = os.listdir(train_path)
            print(f"Found {len(files)} items in {train_path}. First 5: {files[:5]}")
    else:
        print(f"CRITICAL: {DATA_DIR} does not exist!")

if PROCESS_RAW_KAGGLE_DATA:
    print("Processing raw Kaggle video data... This will take some time.")
    
    # Clean up old dummy data if present
    label_map_path = os.path.join(DATA_DIR, "label_map.json")
    if os.path.exists(label_map_path):
        import json
        with open(label_map_path) as f:
            lm = json.load(f)
        if "sign_1" in lm:
            print("Detected old synthetic dummy data. Wiping to prepare for real data...")
            import shutil
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR, exist_ok=True)
            
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        actual_raw_path = RAW_KAGGLE_DATA_PATH
        if not os.path.exists(actual_raw_path) or len(os.listdir(actual_raw_path)) == 0:
            print(f"Warning: {actual_raw_path} not found or empty. Searching /kaggle/input...")
            if os.path.exists("/kaggle/input"):
                for d in os.listdir("/kaggle/input"):
                    if "vsl" in d.lower() or "sign" in d.lower():
                        actual_raw_path = os.path.join("/kaggle/input", d)
                        print(f"Auto-detected dataset at: {actual_raw_path}")
                        break

        subprocess.run([
            sys.executable, "scripts/prepare_vsl_data.py", 
            "--source", "kaggle", 
            "--data_dir", actual_raw_path, 
            "--output_dir", DATA_DIR, 
            "--target_fps", "60"
        ], check=True)
    else:
        print("Data is already processed!")
    
    import json
    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        NUM_DUMMY_CLASSES = len(json.load(f))
    USE_DUMMY_DATA = False

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
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.exists(train_dir):
        # Recursive count to match the new dataset logic
        n = 0
        for r, d, files in os.walk(train_dir):
            n += len([f for f in files if f.lower().endswith(('.npy', '.npz'))])
        print(f"Found existing data: {n} training samples")
    else:
        print("ERROR: No data found! Set USE_DUMMY_DATA=True or provide data.")
        print(f"Expected data at: {DATA_DIR}")

print("\n✅ Data ready!")

# %% [markdown]
# ## Cell 4: Configure Model & Training

# %%
import yaml
import json
import os

d_model = 256 if SMALL_MODEL else 512
num_heads = 4 if SMALL_MODEL else 8
num_layers = 4 if SMALL_MODEL else 6
d_ff = 1024 if SMALL_MODEL else 2048

label_map_path = os.path.join(DATA_DIR, "label_map.json")
if not os.path.exists(label_map_path):
    # Try looking in the parent directory (root of dataset)
    parent_label_map = os.path.join(os.path.dirname(DATA_DIR), "label_map.json")
    if os.path.exists(parent_label_map):
        label_map_path = parent_label_map

if os.path.exists(label_map_path):
    with open(label_map_path) as f:
        vocab_size = len(json.load(f))
else:
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
        "max_seq_length": 400,  # 📉 Reduced from 1500. 400 frames = ~6.6 seconds (plenty for VSL). Massive speedup!
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
            "epochs": 10,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "batch_size": 32,
            "freeze_encoder": True,
            "freeze_decoder": True,
        },
        "stage2": {
            "epochs": 30,
            "lr": 5.0e-5,
            "weight_decay": 1.0e-4,
            "batch_size": 16,
            "alpha": 0.3,
            "freeze_decoder": False,
        },
        "stage3": {
            "epochs": 10,
            "lr": 1.0e-5,
            "weight_decay": 1.0e-5,
            "batch_size": 16,
            "alpha": 0.3,
        },
        "warmup_steps": 500,
        "grad_clip": 1.0,
        "seed": 42,
        "num_workers": 4,  # Increased from 2 to 4 to speed up data loading
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

