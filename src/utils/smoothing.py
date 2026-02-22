"""
Moving Average Smoothing for sensor data.
Reduces noise and jitter from Leap Motion's infrared sensors.
"""
import numpy as np


class MovingAverageSmoothing:
    """
    Apply moving average filter to smooth skeletal data.
    The average of n nearest frames prevents the model from
    misinterpreting natural hand tremor as different gestures.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def smooth(self, data: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing.

        Args:
            data: (T, 42, F) skeletal data

        Returns:
            Smoothed data of same shape
        """
        T, J, F = data.shape
        result = np.zeros_like(data)

        half_w = self.window_size // 2
        for t in range(T):
            start = max(0, t - half_w)
            end = min(T, t + half_w + 1)
            result[t] = np.mean(data[start:end], axis=0)

        return result
