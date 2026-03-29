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

        # Load label map if it exists
        self.label_map = None
        label_map_path = os.path.join(self.data_dir, "label_map.json")
        if os.path.exists(label_map_path):
            import json
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)

        # Load data index
        self.samples = self._load_index()

    def _load_index(self) -> List[Dict]:
        """
        Ultra-robust data discovery using recursive os.walk.
        Supports:
            - split/features/*.npy
            - split/class_name/*.npy
            - split/any/nested/path/*.npy or .npz
        """
        import unicodedata
        def normalize_vn(s):
            if not s: return ""
            # Normalize to NFC and strip spaces/lowercasing for fuzzy match
            s = unicodedata.normalize('NFC', s).strip().lower()
            return s

        split_dir = os.path.join(self.data_dir, self.split)
        samples = []

        if not os.path.exists(split_dir):
            print(f"[{self.split}] ERROR: Split folder not found at {split_dir}")
            return samples

        # Build a normalized label map for fuzzy matching
        norm_label_map = {}
        if self.label_map:
            for k, v in self.label_map.items():
                norm_label_map[normalize_vn(k)] = v

        print(f"[{self.split}] Scanning recursively in: {split_dir}")
        for root, dirs, files in os.walk(split_dir):
            rel_dir = os.path.relpath(root, split_dir)
            
            # Skip standard features/labels folders if they are empty
            if rel_dir in ["features", "labels"] and not files:
                continue

            for fname in sorted(files):
                if not (fname.lower().endswith(".npy") or fname.lower().endswith(".npz")):
                    continue
                # Skip files that look like dedicated label files (unless in standard structure)
                if rel_dir != "labels" and ("_lab" in fname.lower() or "_label" in fname.lower()):
                    continue

                fpath = os.path.join(root, fname)
                label_path = None
                class_id = None

                if rel_dir == "features":
                    # Standard structure: features/*.npy paired with labels/*.npy
                    possible_label = os.path.join(split_dir, "labels", fname)
                    if os.path.exists(possible_label):
                        label_path = possible_label
                elif rel_dir != ".":
                    # Class-Folder structure: /class_name/sample.npy
                    # We use the folder name AS the class name
                    class_name = os.path.basename(root)
                    class_id = norm_label_map.get(normalize_vn(class_name))

                samples.append({
                    "id": f"{rel_dir}_{fname}".replace(os.sep, "_"),
                    "feature_path": fpath,
                    "label_path": label_path,
                    "class_id": class_id,
                })

        num_path_labels = sum(1 for s in samples if s["label_path"])
        num_class_labels = sum(1 for s in samples if s["class_id"] is not None)
        num_unlabeled = sum(1 for s in samples if s["label_path"] is None and s["class_id"] is None)

        print(f"[{self.split}] Summary:")
        print(f"  - Total samples: {len(samples)}")
        print(f"  - Labeled via path: {num_path_labels}")
        print(f"  - Labeled via class_id: {num_class_labels}")
        print(f"  - UNLABELED: {num_unlabeled}")

        if num_unlabeled == len(samples) and len(samples) > 0:
            print(f"[{self.split}] CRITICAL ERROR: 100% of samples are unlabeled! Check folder names vs label_map.json.")
        
        if samples and all(s["class_id"] is None and s["label_path"] is None for s in samples[:100]):
            print(f"[{self.split}] WARNING: No labels found for first 100 samples. Check folder names vs label_map.json.")
            if self.label_map:
                example_class = os.path.basename(os.path.dirname(samples[0]["feature_path"]))
                print(f"[{self.split}] Example folder: '{example_class}' (normalized: '{normalize_vn(example_class)}')")
                print(f"[{self.split}] First few keys in label_map: {list(self.label_map.keys())[:5]}")

        return samples

    def _load_numpy(self, path: str) -> np.ndarray:
        """Safely load .npy or .npz file."""
        data = np.load(path, allow_pickle=True)
        if path.lower().endswith(".npz"):
            # If it's an NpzFile, extract the first array
            if hasattr(data, 'files') and len(data.files) > 0:
                return data[data.files[0]]
            else:
                raise ValueError(f"Empty or invalid .npz file: {path}")
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load features
        try:
            features = self._load_numpy(sample["feature_path"])  # Expected (T_raw, J, F)
            if not isinstance(features, np.ndarray):
                raise ValueError(f"Loaded data is not a numpy array: {type(features)}")
            
            # Robust self-repair for skeletal structures
            if features.ndim == 2:
                T_raw, F_total = features.shape
                
                # Check known common structures and slice to hands if needed
                found = False
                
                # AGGRESSIVE DEBUG: Print the shape and num_joints to Kaggle logs
                # print(f"DEBUG: features.shape={features.shape}, num_joints={self.num_joints}")
                
                # F_p candidates: 7 (full), 3 (coordinates only)
                for f_p in [7, 3]:
                    if F_total % f_p == 0:
                        j_p = F_total // f_p
                        
                        if j_p == self.num_joints:
                            features = features.reshape(T_raw, j_p, f_p)
                            found = True
                        elif j_p == 67 and self.num_joints == 42:
                            # Standard MediaPipe Pose(25) + Hands(42) -> Slice hands
                            # Shape (T, 67, 3) -> 201 features
                            features = features.reshape(T_raw, 67, f_p)[:, 25:, :]
                            found = True
                        elif j_p == 75 and self.num_joints == 42:
                            # Standard MediaPipe Pose(33) + Hands(42) -> Slice hands
                            features = features.reshape(T_raw, 75, f_p)[:, 33:, :]
                            found = True
                        elif j_p == 543 and self.num_joints == 42:
                            # MediaPipe Holistic: Base(33) + Face(468) + Hands(42)
                            features = features.reshape(T_raw, 543, f_p)[:, 501:, :]
                            found = True
                        elif F_total == 201 and self.num_joints == 42:
                            # Explicit handle for (T, 201) mapping to 42 joints
                            # Assuming 25 pose (3 features each) + 42 hands
                            features = features.reshape(T_raw, 67, 3)[:, 25:, :]
                            found = True
                            
                        if found: break
                
                if not found:
                    # Specific error message to match user experience
                    expected_294 = self.num_joints * 7
                    expected_126 = self.num_joints * 3
                    # Add more info to the error
                    msg = (f"Unexpected 2D feature shape {features.shape}. "
                           f"Expected second dim to be {expected_294} or {expected_126}. "
                           f"Detected j_p={F_total//3 if F_total%3==0 else 'N/A'} for f_p=3.")
                    raise ValueError(msg)

            # Ensure final feature count (F) matches self.num_features (usually 7)
            if features.ndim == 3:
                T, J, F = features.shape
                if J != self.num_joints:
                     raise ValueError(f"Joint count mismatch: Got {J}, expected {self.num_joints}")
                
                if F != self.num_features:
                    # Pad or truncate features to match expected count
                    temp = np.zeros((T, J, self.num_features), dtype=np.float32)
                    copy_f = min(F, self.num_features)
                    temp[:, :, :copy_f] = features[:, :, :copy_f]
                    if copy_f <= 3 and self.num_features >= 7:
                        temp[:, :, 6] = 1.0 # Set default confidence if missing
                    features = temp
            else:
                raise ValueError(f"Invalid feature dimensions: {features.ndim}")

        except Exception as e:
            print(f"Error loading features from {sample['feature_path']}: {e}")
            # Fallback to zero features to avoid crashing the whole training
            features = np.zeros((10, self.num_joints, self.num_features), dtype=np.float32)

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
            try:
                labels = self._load_numpy(sample["label_path"]).astype(np.int64)
            except Exception as e:
                print(f"Error loading labels from {sample['label_path']}: {e}")
                labels = np.array([], dtype=np.int64)
        elif sample.get("class_id") is not None:
            # Single gloss per video from Class-Folder structure
            labels = np.array([sample["class_id"]], dtype=np.int64)
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
        padded labels for attention decoder, and length tensors.
    """
    features = torch.stack([s["features"] for s in batch])
    feature_lengths = torch.stack([s["feature_length"] for s in batch])
    label_lengths = torch.stack([s["label_length"] for s in batch])

    # Concatenate labels (CTC loss expects flat labels)
    labels = torch.cat([s["labels"] for s in batch])

    # Padded labels for attention decoder teacher-forcing (B, max_label_len)
    # Padding value -1 is ignored by nn.CrossEntropyLoss(ignore_index=-1)
    max_label_len = max(s["labels"].shape[0] for s in batch)
    max_label_len = max(max_label_len, 1)  # Ensure at least length 1
    padded_labels = torch.full((len(batch), max_label_len), -1, dtype=torch.long)
    for i, s in enumerate(batch):
        length = s["labels"].shape[0]
        if length > 0:
            padded_labels[i, :length] = s["labels"]

    return {
        "features": features,
        "labels": labels,
        "padded_labels": padded_labels,
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
