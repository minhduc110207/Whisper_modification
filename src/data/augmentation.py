"""
Data augmentation for skeletal sign language data.
Includes Gesture Masking, Temporal Jitter, and Noise Injection.
"""
import numpy as np
from typing import Optional


class GestureMasking:
    """
    Randomly mask joints or temporal segments to simulate occlusion.
    Helps model robustness when fingers are hidden from Leap Motion sensor.
    """

    def __init__(
        self,
        joint_mask_prob: float = 0.15,
        temporal_mask_prob: float = 0.1,
        max_temporal_mask: int = 10,
    ):
        self.joint_mask_prob = joint_mask_prob
        self.temporal_mask_prob = temporal_mask_prob
        self.max_temporal_mask = max_temporal_mask

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply gesture masking.

        Args:
            data: Shape (T, 42, F)

        Returns:
            Masked data of same shape
        """
        result = data.copy()
        T, J, F = result.shape

        # Joint masking: randomly zero-out entire joints
        joint_mask = np.random.random(J) < self.joint_mask_prob
        result[:, joint_mask, :] = 0.0

        # Temporal masking: randomly zero-out contiguous time segments
        if np.random.random() < self.temporal_mask_prob:
            mask_len = np.random.randint(1, self.max_temporal_mask + 1)
            start = np.random.randint(0, max(1, T - mask_len))
            result[start: start + mask_len, :, :] = 0.0

        return result


class TemporalJitter:
    """
    Apply small random temporal shifts to simulate timing variations.
    """

    def __init__(self, max_shift: int = 2):
        self.max_shift = max_shift

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply temporal jittering.

        Args:
            data: Shape (T, 42, F)

        Returns:
            Jittered data of same shape
        """
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift == 0:
            return data.copy()

        result = np.zeros_like(data)
        if shift > 0:
            result[shift:] = data[:-shift]
            result[:shift] = data[0]  # Repeat first frame
        else:
            result[:shift] = data[-shift:]
            result[shift:] = data[-1]  # Repeat last frame

        return result


class NoiseInjection:
    """
    Add Gaussian noise to simulate sensor measurement errors
    from Leap Motion's infrared sensors.
    """

    def __init__(self, std: float = 0.005):
        self.std = std

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise.

        Args:
            data: Shape (T, 42, F)

        Returns:
            Noisy data of same shape
        """
        noise = np.random.normal(0, self.std, size=data.shape)
        return (data + noise).astype(np.float32)


class ComposeAugmentations:
    """Compose multiple augmentation transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            data = t(data)
        return data
