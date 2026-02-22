"""
Vietnamese Sign Language Dataset Preparation
=============================================
Converts publicly available Vietnamese Sign Language datasets
into the format required by WhisperSign for training.

Supported datasets:
  1. Kaggle VSL (videos + label.csv) -> MediaPipe extraction -> .npy
  2. VOYA_VSL from HuggingFace (pre-extracted .npz keypoints)
  3. Custom video folder with label mapping

Output format:
  data/processed/{split}/features/*.npy   shape (T, 42, 7)
  data/processed/{split}/labels/*.npy     shape (num_glosses,)
  data/processed/label_map.json           {gloss_name: int_id}

Usage:
  python scripts/prepare_vsl_data.py --source kaggle --data_dir path/to/kaggle_vsl
  python scripts/prepare_vsl_data.py --source huggingface
  python scripts/prepare_vsl_data.py --source video --data_dir path/to/videos --label_csv labels.csv
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_from_kaggle_vsl(data_dir: str, output_dir: str, target_fps: int = 60):
    """
    Prepare data from Kaggle VSL-Vietnamese Sign Languages dataset.

    Expected input structure:
      data_dir/
        videos/         -> .mp4 video files
        label.csv       -> columns: ID, VIDEO, LABEL
    """
    import pandas as pd
    from src.utils.mediapipe_extract import extract_hand_keypoints
    from src.data.preprocessing import resample_to_fixed_rate

    print("Loading Kaggle VSL dataset...")
    label_csv = os.path.join(data_dir, "label.csv")
    if not os.path.exists(label_csv):
        # Try alternative locations
        for alt in ["labels.csv", "metadata.csv"]:
            alt_path = os.path.join(data_dir, alt)
            if os.path.exists(alt_path):
                label_csv = alt_path
                break

    df = pd.read_csv(label_csv)
    print(f"  Found {len(df)} samples")

    # Build label mapping
    unique_labels = sorted(df["LABEL"].unique())
    label_map = {label: idx + 1 for idx, label in enumerate(unique_labels)}  # 0 = CTC blank
    label_map["<blank>"] = 0
    print(f"  Vocabulary size: {len(unique_labels)} glosses")

    # Determine splits (80/10/10)
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    n_train = int(0.8 * len(df))
    n_val = int(0.1 * len(df))

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for split_name, split_indices in splits.items():
        feat_dir = os.path.join(output_dir, split_name, "features")
        lab_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)

        success = 0
        for i, idx in enumerate(split_indices):
            row = df.iloc[idx]
            video_path = os.path.join(data_dir, "videos", row["VIDEO"])

            if not os.path.exists(video_path):
                video_path = os.path.join(data_dir, row["VIDEO"])

            if not os.path.exists(video_path):
                print(f"  [SKIP] Video not found: {row['VIDEO']}")
                continue

            try:
                keypoints, timestamps = extract_hand_keypoints(video_path)

                if len(keypoints) < 4:
                    print(f"  [SKIP] Too few frames ({len(keypoints)}): {row['VIDEO']}")
                    continue

                # Resample to target FPS
                if len(timestamps) > 1:
                    resampled, _ = resample_to_fixed_rate(keypoints, timestamps, target_fps)
                else:
                    resampled = keypoints

                # Save features
                np.save(os.path.join(feat_dir, f"sample_{success:05d}.npy"),
                        resampled.astype(np.float32))

                # Save label (single gloss per video)
                label_id = label_map[row["LABEL"]]
                np.save(os.path.join(lab_dir, f"sample_{success:05d}.npy"),
                        np.array([label_id], dtype=np.int64))

                success += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{split_name}] Processed {i+1}/{len(split_indices)} "
                          f"({success} success)")

            except Exception as e:
                print(f"  [ERROR] {row['VIDEO']}: {e}")

        print(f"  {split_name}: {success} samples saved")

    # Save label map
    save_label_map(label_map, output_dir)
    return label_map


def prepare_from_huggingface(output_dir: str):
    """
    Prepare data from VOYA_VSL HuggingFace dataset.
    Pre-extracted MediaPipe keypoints in .npz format.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

    print("Loading VOYA_VSL from HuggingFace...")
    dataset = load_dataset("VOYA/VOYA_VSL")

    # Load label mapping
    label_map = {}
    label_counter = 1  # 0 = CTC blank

    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            alt_name = "val" if split_name == "validation" else split_name
            if alt_name not in dataset:
                print(f"  [SKIP] Split '{split_name}' not found")
                continue
            split_data = dataset[alt_name]
        else:
            split_data = dataset[split_name]

        out_split = "val" if split_name == "validation" else split_name
        feat_dir = os.path.join(output_dir, out_split, "features")
        lab_dir = os.path.join(output_dir, out_split, "labels")
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)

        success = 0
        for i, sample in enumerate(split_data):
            try:
                # Extract keypoints - format depends on dataset structure
                if "keypoints" in sample:
                    kp = np.array(sample["keypoints"], dtype=np.float32)
                elif "sequence" in sample:
                    kp = np.array(sample["sequence"], dtype=np.float32)
                else:
                    # Try loading from file path
                    kp_data = np.load(sample["file"], allow_pickle=True)
                    kp = kp_data["keypoints"] if "keypoints" in kp_data else kp_data["arr_0"]

                # Reshape to (T, 42, 7) if needed
                if kp.ndim == 2:
                    # Might be (T, 42*3) -> reshape
                    T = kp.shape[0]
                    if kp.shape[1] == 42 * 3:
                        kp_reshaped = np.zeros((T, 42, 7), dtype=np.float32)
                        kp_reshaped[:, :, :3] = kp.reshape(T, 42, 3)
                    elif kp.shape[1] == 21 * 3:
                        # Single hand -> pad to 42 joints
                        kp_reshaped = np.zeros((T, 42, 7), dtype=np.float32)
                        kp_reshaped[:, :21, :3] = kp.reshape(T, 21, 3)
                    else:
                        kp_reshaped = np.zeros((T, 42, 7), dtype=np.float32)
                        kp_reshaped[:, :, :min(kp.shape[1]//42, 7)] = kp.reshape(T, 42, -1)[:, :, :7]
                    kp = kp_reshaped
                elif kp.ndim == 3 and kp.shape[2] < 7:
                    # Pad features to 7 (add zero velocity and confidence)
                    T, J = kp.shape[0], kp.shape[1]
                    kp_padded = np.zeros((T, J, 7), dtype=np.float32)
                    kp_padded[:, :, :kp.shape[2]] = kp
                    if kp.shape[2] <= 3:
                        kp_padded[:, :, 6] = 1.0  # Set confidence to 1
                    kp = kp_padded

                # Ensure 42 joints
                if kp.shape[1] < 42:
                    padded = np.zeros((kp.shape[0], 42, 7), dtype=np.float32)
                    padded[:, :kp.shape[1]] = kp
                    kp = padded

                if kp.shape[0] < 4:
                    continue

                np.save(os.path.join(feat_dir, f"sample_{success:05d}.npy"), kp.astype(np.float32))

                # Extract label
                label_name = sample.get("label", sample.get("gloss", sample.get("class", str(i))))
                if isinstance(label_name, int):
                    label_id = label_name + 1  # Shift for CTC blank
                else:
                    if label_name not in label_map:
                        label_map[label_name] = label_counter
                        label_counter += 1
                    label_id = label_map[label_name]

                np.save(os.path.join(lab_dir, f"sample_{success:05d}.npy"),
                        np.array([label_id], dtype=np.int64))
                success += 1

            except Exception as e:
                print(f"  [ERROR] Sample {i}: {e}")

        print(f"  {out_split}: {success} samples saved")

    label_map["<blank>"] = 0
    save_label_map(label_map, output_dir)
    return label_map


def prepare_from_videos(data_dir: str, output_dir: str, label_csv: str,
                        target_fps: int = 60, test_ratio: float = 0.1,
                        val_ratio: float = 0.1):
    """
    Prepare data from a folder of videos with a label CSV file.

    Expected CSV format: filename,label (header row)
    """
    import pandas as pd
    from src.utils.mediapipe_extract import extract_hand_keypoints
    from src.data.preprocessing import resample_to_fixed_rate

    print(f"Loading videos from {data_dir}...")
    df = pd.read_csv(label_csv)

    # Auto-detect column names
    filename_col = None
    label_col = None
    for col in df.columns:
        if col.lower() in ["filename", "file", "video", "path", "video_path"]:
            filename_col = col
        if col.lower() in ["label", "gloss", "class", "sign", "meaning"]:
            label_col = col
    if filename_col is None:
        filename_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    print(f"  Using columns: filename='{filename_col}', label='{label_col}'")
    print(f"  Found {len(df)} entries")

    # Build label map
    unique_labels = sorted(df[label_col].unique())
    label_map = {"<blank>": 0}
    label_map.update({label: idx + 1 for idx, label in enumerate(unique_labels)})

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    n_test = int(test_ratio * len(df))
    n_val = int(val_ratio * len(df))

    splits = {
        "train": indices[:-(n_test + n_val)] if n_test + n_val > 0 else indices,
        "val": indices[-(n_test + n_val):-n_test] if n_test > 0 else indices[-n_val:],
        "test": indices[-n_test:] if n_test > 0 else np.array([], dtype=int),
    }

    for split_name, split_indices in splits.items():
        if len(split_indices) == 0:
            continue

        feat_dir = os.path.join(output_dir, split_name, "features")
        lab_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)

        success = 0
        for i, idx in enumerate(split_indices):
            row = df.iloc[idx]
            video_path = os.path.join(data_dir, str(row[filename_col]))

            if not os.path.exists(video_path):
                continue

            try:
                keypoints, timestamps = extract_hand_keypoints(video_path)
                if len(keypoints) < 4:
                    continue

                if len(timestamps) > 1:
                    resampled, _ = resample_to_fixed_rate(keypoints, timestamps, target_fps)
                else:
                    resampled = keypoints

                np.save(os.path.join(feat_dir, f"sample_{success:05d}.npy"),
                        resampled.astype(np.float32))

                label_id = label_map[row[label_col]]
                np.save(os.path.join(lab_dir, f"sample_{success:05d}.npy"),
                        np.array([label_id], dtype=np.int64))
                success += 1

                if (i + 1) % 50 == 0:
                    print(f"  [{split_name}] {i+1}/{len(split_indices)} ({success} ok)")

            except Exception as e:
                print(f"  [ERROR] {row[filename_col]}: {e}")

        print(f"  {split_name}: {success} samples")

    save_label_map(label_map, output_dir)
    return label_map


def save_label_map(label_map: dict, output_dir: str):
    """Save label mapping to JSON."""
    path = os.path.join(output_dir, "label_map.json")
    # Sort by ID for readability
    sorted_map = dict(sorted(label_map.items(), key=lambda x: x[1]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_map, f, ensure_ascii=False, indent=2)
    print(f"\nLabel map saved to {path}")
    print(f"Vocabulary size: {len(label_map)} (including <blank>)")


def print_dataset_stats(output_dir: str):
    """Print statistics about the prepared dataset."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    for split in ["train", "val", "test"]:
        feat_dir = os.path.join(output_dir, split, "features")
        if not os.path.exists(feat_dir):
            continue

        files = sorted(Path(feat_dir).glob("*.npy"))
        if not files:
            continue

        lengths = []
        shapes = []
        for f in files:
            data = np.load(f)
            lengths.append(data.shape[0])
            shapes.append(data.shape)

        print(f"\n{split}:")
        print(f"  Samples: {len(files)}")
        print(f"  Shape: {shapes[0]}")
        print(f"  Sequence length: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")
        print(f"  Total frames: {sum(lengths):,}")

    # Label distribution
    label_map_path = os.path.join(output_dir, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"\nVocabulary: {len(label_map)} classes (including <blank>)")


def main():
    parser = argparse.ArgumentParser(description="Prepare Vietnamese Sign Language data for WhisperSign")
    parser.add_argument("--source", choices=["kaggle", "huggingface", "video"],
                        required=True, help="Data source type")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to source data directory")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="Path to label CSV (for video source)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--target_fps", type=int, default=60,
                        help="Target frame rate for resampling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.source == "kaggle":
        if not args.data_dir:
            print("Error: --data_dir is required for kaggle source")
            sys.exit(1)
        prepare_from_kaggle_vsl(args.data_dir, args.output_dir, args.target_fps)

    elif args.source == "huggingface":
        prepare_from_huggingface(args.output_dir)

    elif args.source == "video":
        if not args.data_dir or not args.label_csv:
            print("Error: --data_dir and --label_csv are required for video source")
            sys.exit(1)
        prepare_from_videos(args.data_dir, args.output_dir, args.label_csv, args.target_fps)

    print_dataset_stats(args.output_dir)
    print("\nDone! You can now train with:")
    print(f"  python scripts/train.py --config configs/config.yaml --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
