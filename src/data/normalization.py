"""
Normalization modules for skeletal data.
Implements Spatial Invariance, Scale Invariance, and Feature Scaling.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional


class SpatialNormalizer:
    """
    Dual Hand-centric Normalization (Spatial Invariance).
    Converts absolute coordinates to relative coordinates
    centered on each hand's palm position.

    Formula: P_hat_i = P_i - P_palm
    - Left hand joints (0-20): relative to palm index 0
    - Right hand joints (21-41): relative to palm index 21
    """

    def __init__(self, num_left_joints: int = 21, num_right_joints: int = 21):
        self.num_left = num_left_joints
        self.num_right = num_right_joints
        self.left_palm_idx = 0
        self.right_palm_idx = num_left_joints  # = 21

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply hand-centric normalization.

        Args:
            data: Shape (T, 42, F) where F includes x, y, z coordinates

        Returns:
            Normalized data of same shape
        """
        result = data.copy()
        T = data.shape[0]

        # Left hand: subtract left palm position (index 0)
        left_palm = data[:, self.left_palm_idx: self.left_palm_idx + 1, :3]
        result[:, :self.num_left, :3] -= left_palm

        # Right hand: subtract right palm position (index 21)
        right_palm = data[:, self.right_palm_idx: self.right_palm_idx + 1, :3]
        result[:, self.num_left: self.num_left + self.num_right, :3] -= right_palm

        return result


class ScaleNormalizer:
    """
    Bone-Length-Based Scale Invariance normalization.

    Normalizes joint positions by dividing by the total bone chain length
    of the middle finger (Wrist→MCP→PIP→DIP→TIP), providing anatomically
    correct scale invariance across different hand sizes.

    Formula: P_hat_i = P_i / L_ref
    where L_ref = sum of 4 bone lengths along the middle finger chain.

    After normalization, a fully extended fingertip position ≈ 1.0,
    regardless of whether the hand belongs to a child or an adult.
    """

    # MediaPipe middle finger bone chain (per-hand, 0-indexed)
    # Wrist(0)→MCP(9)→PIP(10)→DIP(11)→TIP(12)
    MIDDLE_FINGER_BONES = [(0, 9), (9, 10), (10, 11), (11, 12)]

    def __init__(self, num_left_joints: int = 21, num_right_joints: int = 21):
        self.num_left = num_left_joints
        self.num_right = num_right_joints

    def _compute_hand_scale(self, hand_data: np.ndarray, palm_idx: int = 0) -> float:
        """
        Compute the reference bone chain length for scaling.

        Sums the Euclidean lengths of all 4 bones in the middle finger
        chain (Metacarpal + Proximal + Intermediate + Distal) and averages
        across all frames for temporal stability.

        Args:
            hand_data: Shape (T, 21, F) - one hand's data
            palm_idx: Index of palm/wrist joint (always 0 within the hand)

        Returns:
            Average total bone chain length (scalar)
        """
        T = hand_data.shape[0]
        total_bone_length = np.zeros(T)

        for start_idx, end_idx in self.MIDDLE_FINGER_BONES:
            bone_vec = hand_data[:, end_idx, :3] - hand_data[:, start_idx, :3]
            total_bone_length += np.linalg.norm(bone_vec, axis=1)

        # Average over time for stability against per-frame jitter
        avg_length = np.mean(total_bone_length)

        return max(avg_length, 1e-6)  # Avoid division by zero

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bone-length-based scale normalization.

        Divides both position (x, y, z) and velocity (vx, vy, vz) features
        by the reference bone chain length. Velocity must be scaled because
        it is derived from position (v = dP/dt) and shares the same units.

        Args:
            data: Shape (T, 42, F), should be spatially normalized first

        Returns:
            Scale-normalized data
        """
        result = data.copy()

        # Left hand scale
        left_data = data[:, :self.num_left, :]
        left_scale = self._compute_hand_scale(left_data)
        result[:, :self.num_left, :3] /= left_scale    # Position: x, y, z
        result[:, :self.num_left, 3:6] /= left_scale   # Velocity: vx, vy, vz

        # Right hand scale
        right_data = data[:, self.num_left:self.num_left + self.num_right, :]
        right_scale = self._compute_hand_scale(right_data)
        result[:, self.num_left:self.num_left + self.num_right, :3] /= right_scale
        result[:, self.num_left:self.num_left + self.num_right, 3:6] /= right_scale

        return result


class FeatureScaler:
    """
    Feature-wise standardization (zero mean, unit variance)
    or MinMax scaling to prevent exploding gradients.
    """

    def __init__(self, method: str = "standard"):
        """
        Args:
            method: "standard" for StandardScaler, "minmax" for MinMaxScaler
        """
        self.method = method
        self.scaler = None
        self.fitted = False

    def fit(self, data: np.ndarray) -> 'FeatureScaler':
        """
        Fit the scaler on training data.

        Args:
            data: Shape (N, T, 42, F) - batch of sequences

        Returns:
            self
        """
        # Reshape to (N*T*42, F) for fitting
        original_shape = data.shape
        flat = data.reshape(-1, original_shape[-1])

        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.scaler.fit(flat)
        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            data: Shape (T, 42, F) or (N, T, 42, F)

        Returns:
            Scaled data of same shape
        """
        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform")

        original_shape = data.shape
        flat = data.reshape(-1, original_shape[-1])
        scaled = self.scaler.transform(flat)
        return scaled.reshape(original_shape).astype(np.float32)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
