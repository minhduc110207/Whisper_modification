"""
PyTorch Dataset and DataLoader for sign language skeletal data.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from .preprocessing import preprocess_sequence, compute_sequence_mask
from .normalization import SpatialNormalizer, ScaleNormalizer, FeatureScaler
from .augmentation import GestureMasking, TemporalJitter, NoiseInjection, ComposeAugmentations


class SignLanguageDataset(Dataset):
    """
    PyTorch Dataset for sign language recognition.

    Each sample is a dictionary with:
        - 'features': Tensor of shape (T, 42, F) - skeletal coordinates
        - 'labels': Tensor of label indices (variable length)
        - 'feature_length': int - actual sequence length before padding
        - 'label_length': int - length of label sequence
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_length: int = 1500,
        sample_rate: int = 60,
        num_joints: int = 42,
        num_features: int = 7,
        augment: bool = True,
        augmentation_config: Optional[dict] = None,
    ):
        """
        Args:
            data_dir: Path to processed data directory
            split: "train", "val", or "test"
            max_seq_length: Maximum sequence length (frames)
            sample_rate: Target sampling rate
            num_joints: Number of joints (42 = 21L + 21R)
            num_features: Features per joint (x,y,z,vx,vy,vz,confidence)
            augment: Whether to apply augmentations
            augmentation_config: Config dict for augmentations
        """
        self.data_dir = data_dir
        self.split = split
        self.max_seq_length = max_seq_length
        self.sample_rate = sample_rate
        self.num_joints = num_joints
        self.num_features = num_features

        # Normalizers
        self.spatial_norm = SpatialNormalizer()
        self.scale_norm = ScaleNormalizer()

        # Augmentations (only for training)
        self.augment = augment and (split == "train")
        if self.augment:
            cfg = augmentation_config or {}
            transforms = []
            if cfg.get("gesture_masking", {}).get("enabled", True):
                gm_cfg = cfg.get("gesture_masking", {})
                transforms.append(GestureMasking(
                    joint_mask_prob=gm_cfg.get("joint_mask_prob", 0.15),
                    temporal_mask_prob=gm_cfg.get("temporal_mask_prob", 0.1),
                    max_temporal_mask=gm_cfg.get("max_temporal_mask", 10),
                ))
            if cfg.get("temporal_jitter", {}).get("enabled", True):
                tj_cfg = cfg.get("temporal_jitter", {})
                transforms.append(TemporalJitter(
                    max_shift=tj_cfg.get("max_shift", 2),
                ))
            if cfg.get("noise", {}).get("enabled", True):
                n_cfg = cfg.get("noise", {})
                transforms.append(NoiseInjection(
                    std=n_cfg.get("std", 0.005),
                ))
            self.augmentor = ComposeAugmentations(transforms) if transforms else None
        else:
            self.augmentor = None

        # Load data index
        self.samples = self._load_index()

    def _load_index(self) -> List[Dict]:
        """
        Load data index from directory.
        Expects files organized as:
            data_dir/{split}/
                features/  -> .npy files with shape (T, 42, F)
                labels/    -> .npy files with label indices
        """
        split_dir = os.path.join(self.data_dir, self.split)
        features_dir = os.path.join(split_dir, "features")
        labels_dir = os.path.join(split_dir, "labels")

        samples = []

        if not os.path.exists(features_dir):
            # Return empty dataset if directory doesn't exist
            return samples

        for fname in sorted(os.listdir(features_dir)):
            if fname.endswith(".npy"):
                sample_id = fname.replace(".npy", "")
                label_path = os.path.join(labels_dir, fname)

                samples.append({
                    "id": sample_id,
                    "feature_path": os.path.join(features_dir, fname),
                    "label_path": label_path if os.path.exists(label_path) else None,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load features
        features = np.load(sample["feature_path"])  # (T_raw, 42, F)
        actual_length = min(features.shape[0], self.max_seq_length)

        # Normalize
        features = self.spatial_norm.normalize(features)
        features = self.scale_norm.normalize(features)

        # Preprocess (pad/truncate)
        features = preprocess_sequence(
            features,
            target_rate=self.sample_rate,
            max_seq_length=self.max_seq_length,
        )

        # Augment
        if self.augmentor is not None:
            features = self.augmentor(features)

        # Load labels
        if sample["label_path"] is not None:
            labels = np.load(sample["label_path"]).astype(np.int64)
        else:
            labels = np.array([], dtype=np.int64)

        return {
            "features": torch.from_numpy(features).float(),
            "labels": torch.from_numpy(labels).long(),
            "feature_length": torch.tensor(actual_length, dtype=torch.long),
            "label_length": torch.tensor(len(labels), dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length labels.

    Returns:
        Dictionary with padded features, concatenated labels,
        and length tensors for CTC loss.
    """
    features = torch.stack([s["features"] for s in batch])
    feature_lengths = torch.stack([s["feature_length"] for s in batch])
    label_lengths = torch.stack([s["label_length"] for s in batch])

    # Concatenate labels (CTC loss expects flat labels)
    labels = torch.cat([s["labels"] for s in batch])

    return {
        "features": features,
        "labels": labels,
        "feature_lengths": feature_lengths,
        "label_lengths": label_lengths,
    }


def create_dataloaders(
    data_dir: str,
    config: dict,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_dir: Path to processed data
        config: Data configuration dictionary
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    common_args = {
        "data_dir": data_dir,
        "max_seq_length": config.get("max_seq_length", 1500),
        "sample_rate": config.get("sample_rate", 60),
        "num_joints": config.get("num_left_joints", 21) + config.get("num_right_joints", 21),
        "num_features": 7,
    }

    train_ds = SignLanguageDataset(
        split="train",
        augment=True,
        augmentation_config=config.get("augmentation", {}),
        **common_args,
    )
    val_ds = SignLanguageDataset(split="val", augment=False, **common_args)
    test_ds = SignLanguageDataset(split="test", augment=False, **common_args)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
