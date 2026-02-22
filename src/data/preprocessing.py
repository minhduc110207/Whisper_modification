"""
Preprocessing pipeline for Leap Motion / MediaPipe skeletal data.
Handles resampling, windowing, and tensor assembly.
"""
import numpy as np
from scipy import interpolate
from typing import Tuple, Optional


def resample_to_fixed_rate(
    data: np.ndarray,
    timestamps: np.ndarray,
    target_rate: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample variable-rate skeletal data to a fixed sampling rate
    using cubic spline interpolation.

    Args:
        data: Raw skeletal data of shape (T_raw, num_joints, num_features)
        timestamps: Original timestamps in seconds, shape (T_raw,)
        target_rate: Target sampling rate in Hz (default 60)

    Returns:
        resampled_data: Shape (T_new, num_joints, num_features)
        new_timestamps: Shape (T_new,)
    """
    t_start = timestamps[0]
    t_end = timestamps[-1]
    duration = t_end - t_start

    num_new_frames = int(duration * target_rate) + 1
    new_timestamps = np.linspace(t_start, t_end, num_new_frames)

    T_raw, num_joints, num_features = data.shape
    resampled = np.zeros((num_new_frames, num_joints, num_features))

    for j in range(num_joints):
        for f in range(num_features):
            # Cubic spline interpolation for smooth resampling
            cs = interpolate.CubicSpline(timestamps, data[:, j, f])
            resampled[:, j, f] = cs(new_timestamps)

    return resampled, new_timestamps


def create_sliding_windows(
    data: np.ndarray,
    window_duration: float = 3.0,
    overlap: float = 0.5,
    sample_rate: int = 60,
) -> list:
    """
    Split data into overlapping sliding windows.

    Args:
        data: Skeletal data of shape (T, num_joints, num_features)
        window_duration: Window size in seconds
        overlap: Overlap ratio (0.0 to 1.0)
        sample_rate: Sampling rate in Hz

    Returns:
        List of windows, each of shape (window_frames, num_joints, num_features)
    """
    window_frames = int(window_duration * sample_rate)
    step_frames = int(window_frames * (1 - overlap))
    T = data.shape[0]

    windows = []
    start = 0
    while start + window_frames <= T:
        window = data[start: start + window_frames]
        windows.append(window)
        start += step_frames

    # Handle last window (pad if necessary)
    if start < T:
        last_window = data[start:]
        pad_length = window_frames - last_window.shape[0]
        if pad_length > 0:
            padding = np.zeros((pad_length,) + last_window.shape[1:])
            last_window = np.concatenate([last_window, padding], axis=0)
        windows.append(last_window)

    return windows


def preprocess_sequence(
    data: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    target_rate: int = 60,
    max_seq_length: int = 1500,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single sequence.

    Args:
        data: Raw data (T_raw, num_joints, num_features)
        timestamps: Optional timestamps; if None, assumes fixed rate
        target_rate: Target sampling rate
        max_seq_length: Maximum sequence length (pad/truncate)

    Returns:
        Preprocessed tensor of shape (max_seq_length, num_joints, num_features)
    """
    # Step 1: Resample if timestamps provided
    if timestamps is not None:
        data, _ = resample_to_fixed_rate(data, timestamps, target_rate)

    T, num_joints, num_features = data.shape

    # Step 2: Truncate or pad to max_seq_length
    if T > max_seq_length:
        data = data[:max_seq_length]
    elif T < max_seq_length:
        pad_length = max_seq_length - T
        padding = np.zeros((pad_length, num_joints, num_features))
        data = np.concatenate([data, padding], axis=0)

    return data.astype(np.float32)


def compute_sequence_mask(
    seq_length: int,
    max_length: int,
) -> np.ndarray:
    """
    Create a boolean mask for valid (non-padded) positions.

    Args:
        seq_length: Actual sequence length before padding
        max_length: Total padded length

    Returns:
        Boolean mask of shape (max_length,), True for valid positions
    """
    mask = np.zeros(max_length, dtype=bool)
    mask[:seq_length] = True
    return mask
