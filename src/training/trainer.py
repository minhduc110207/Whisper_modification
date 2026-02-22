"""
Three-stage training pipeline for WhisperSign.

Stage 1: Warm-up Frontend (freeze Encoder + Decoder)
Stage 2: Joint Training (unfreeze Encoder, Hybrid Loss)
Stage 3: Real-time Optimization (Sliding Window + Gesture Masking)
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
from tqdm import tqdm

from ..model.whisper_sign import WhisperSignModel
from .losses import HybridCTCAttentionLoss
from .scheduler import CosineWarmupScheduler


class WhisperSignTrainer:
    """
    Trainer implementing the 3-stage training pipeline.
    """

    def __init__(
        self,
        model: WhisperSignModel,
        config: dict,
        device: str = "cuda",
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def _create_optimizer(self, lr: float, weight_decay: float):
        """Create AdamW optimizer for trainable parameters only."""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[CosineWarmupScheduler],
        loss_fn: HybridCTCAttentionLoss,
        epoch: int,
        stage: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Stage {stage} | Epoch {epoch}")
        for batch in pbar:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)

            # Forward pass
            outputs = self.model(features, feature_lengths)

            # Compute loss
            loss_dict = loss_fn(
                ctc_log_probs=outputs["ctc_log_probs"],
                att_logits=outputs.get("att_logits"),
                labels=labels,
                output_lengths=outputs["output_lengths"],
                label_lengths=label_lengths,
            )

            loss = loss_dict["total"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip = self.config.get("grad_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            self.writer.add_scalar("train/total_loss", loss.item(), self.global_step)
            self.writer.add_scalar("train/ctc_loss", loss_dict["ctc"].item(), self.global_step)
            if "attention" in loss_dict:
                self.writer.add_scalar("train/att_loss", loss_dict["attention"].item(), self.global_step)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ctc": f"{loss_dict['ctc'].item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        loss_fn: HybridCTCAttentionLoss,
    ) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)

            outputs = self.model(features, feature_lengths)

            loss_dict = loss_fn(
                ctc_log_probs=outputs["ctc_log_probs"],
                att_logits=outputs.get("att_logits"),
                labels=labels,
                output_lengths=outputs["output_lengths"],
                label_lengths=label_lengths,
            )

            total_loss += loss_dict["total"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar("val/loss", avg_loss, self.global_step)
        return avg_loss

    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Stage 1: Warm-up Frontend.
        Freeze Encoder + Decoder, train only Frontend.
        """
        print("=" * 60)
        print("STAGE 1: Warm-up Frontend")
        print("=" * 60)

        cfg = self.config.get("stage1", {})

        # Freeze
        self.model.freeze_encoder()
        self.model.freeze_decoder()

        trainable = self.model.get_num_params(trainable_only=True)
        total = self.model.get_num_params(trainable_only=False)
        print(f"Trainable params: {trainable:,} / {total:,}")

        optimizer = self._create_optimizer(
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        epochs = cfg.get("epochs", 30)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=self.config.get("warmup_steps", 500),
            total_steps=epochs * len(train_loader),
        )
        loss_fn = HybridCTCAttentionLoss(alpha=1.0, blank_id=0)  # CTC only

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(
                train_loader, optimizer, scheduler, loss_fn, epoch, stage=1
            )
            print(f"  Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss = self._validate(val_loader, loss_fn)
                print(f"  Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_checkpoint(
                        os.path.join(self.save_dir, "best_stage1.pt"),
                        optimizer, epoch, val_loss,
                    )

        self.model.save_checkpoint(
            os.path.join(self.save_dir, "final_stage1.pt"),
            optimizer, epochs, train_loss,
        )

    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Stage 2: Joint Training.
        Unfreeze Encoder, train with Hybrid Loss.
        """
        print("=" * 60)
        print("STAGE 2: Joint Training (Hybrid Loss)")
        print("=" * 60)

        cfg = self.config.get("stage2", {})

        # Unfreeze encoder
        self.model.unfreeze_encoder()
        if not cfg.get("freeze_decoder", False):
            self.model.unfreeze_decoder()

        trainable = self.model.get_num_params(trainable_only=True)
        total = self.model.get_num_params(trainable_only=False)
        print(f"Trainable params: {trainable:,} / {total:,}")

        alpha = cfg.get("alpha", 0.3)
        optimizer = self._create_optimizer(
            lr=cfg.get("lr", 5e-5),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        epochs = cfg.get("epochs", 100)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=self.config.get("warmup_steps", 500),
            total_steps=epochs * len(train_loader),
        )
        loss_fn = HybridCTCAttentionLoss(alpha=alpha, blank_id=0)

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(
                train_loader, optimizer, scheduler, loss_fn, epoch, stage=2
            )
            print(f"  Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss = self._validate(val_loader, loss_fn)
                print(f"  Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_checkpoint(
                        os.path.join(self.save_dir, "best_stage2.pt"),
                        optimizer, epoch, val_loss,
                    )

            # Save every 10 epochs
            if epoch % 10 == 0:
                self.model.save_checkpoint(
                    os.path.join(self.save_dir, f"stage2_epoch{epoch}.pt"),
                    optimizer, epoch, train_loss,
                )

        self.model.save_checkpoint(
            os.path.join(self.save_dir, "final_stage2.pt"),
            optimizer, epochs, train_loss,
        )

    def train_stage3(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Stage 3: Real-time Optimization.
        Fine-tune with Sliding Window + Gesture Masking.
        """
        print("=" * 60)
        print("STAGE 3: Real-time Optimization")
        print("=" * 60)

        cfg = self.config.get("stage3", {})

        alpha = cfg.get("alpha", 0.3)
        optimizer = self._create_optimizer(
            lr=cfg.get("lr", 1e-5),
            weight_decay=cfg.get("weight_decay", 1e-5),
        )
        epochs = cfg.get("epochs", 30)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=self.config.get("warmup_steps", 200),
            total_steps=epochs * len(train_loader),
        )
        loss_fn = HybridCTCAttentionLoss(alpha=alpha, blank_id=0)

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(
                train_loader, optimizer, scheduler, loss_fn, epoch, stage=3
            )
            print(f"  Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss = self._validate(val_loader, loss_fn)
                print(f"  Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_checkpoint(
                        os.path.join(self.save_dir, "best_stage3.pt"),
                        optimizer, epoch, val_loss,
                    )

        self.model.save_checkpoint(
            os.path.join(self.save_dir, "final_model.pt"),
            optimizer, epochs, train_loss,
        )
        print("Training complete!")

    def train_all_stages(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Run all 3 training stages sequentially."""
        self.train_stage1(train_loader, val_loader)
        self.train_stage2(train_loader, val_loader)
        self.train_stage3(train_loader, val_loader)
