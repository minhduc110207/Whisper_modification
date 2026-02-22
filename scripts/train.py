"""
Main training script for WhisperSign model.
Orchestrates the 3-stage training pipeline.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --stage 2 --resume checkpoints/best_stage1.pt
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.whisper_sign import WhisperSignModel
from src.data.dataset import create_dataloaders
from src.training.trainer import WhisperSignTrainer


def main():
    parser = argparse.ArgumentParser(description="Train WhisperSign Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Path to processed data")
    parser.add_argument("--stage", type=int, default=0,
                       help="Training stage (0=all, 1/2/3=specific stage)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto/cuda/cpu)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create model
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model, checkpoint = WhisperSignModel.load_checkpoint(args.resume, device)
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f}")
    else:
        model = WhisperSignModel(config.get("model", {}))

    total_params = model.get_num_params(trainable_only=False)
    print(f"Model parameters: {total_params:,}")

    # Create data loaders
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    batch_size = training_config.get(f"stage{max(args.stage, 1)}", {}).get("batch_size", 16)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        config=data_config,
        batch_size=batch_size,
        num_workers=training_config.get("num_workers", 4),
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create trainer
    trainer = WhisperSignTrainer(
        model=model,
        config=training_config,
        device=device,
        save_dir=training_config.get("save_dir", "checkpoints"),
        log_dir=training_config.get("log_dir", "logs"),
    )

    # Train
    if args.stage == 0:
        trainer.train_all_stages(train_loader, val_loader)
    elif args.stage == 1:
        trainer.train_stage1(train_loader, val_loader)
    elif args.stage == 2:
        trainer.train_stage2(train_loader, val_loader)
    elif args.stage == 3:
        trainer.train_stage3(train_loader, val_loader)
    else:
        print(f"Unknown stage: {args.stage}")

    print("Done!")


if __name__ == "__main__":
    main()
