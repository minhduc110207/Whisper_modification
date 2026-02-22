"""
Sliding Window Inference for real-time sign language recognition.
"""
import torch
import numpy as np
from typing import List, Optional


class SlidingWindowInference:
    """
    Performs inference using sliding windows for real-time processing.

    Instead of waiting for 30s of data (like original Whisper),
    processes 2-3s windows with 50% overlap for low latency.
    """

    def __init__(
        self,
        model,
        window_duration: float = 3.0,
        overlap: float = 0.5,
        sample_rate: int = 60,
        device: str = "cuda",
    ):
        self.model = model
        self.window_frames = int(window_duration * sample_rate)
        self.step_frames = int(self.window_frames * (1 - overlap))
        self.device = device

    @torch.no_grad()
    def __call__(self, data: np.ndarray) -> List[int]:
        """
        Run sliding window inference on a full sequence.

        Args:
            data: (T, 42, F) numpy array

        Returns:
            Predicted sign gloss sequence
        """
        self.model.eval()
        T = data.shape[0]
        all_predictions = []

        start = 0
        while start < T:
            end = min(start + self.window_frames, T)
            window = data[start:end]

            # Pad if needed
            if window.shape[0] < self.window_frames:
                pad = np.zeros(
                    (self.window_frames - window.shape[0],) + window.shape[1:],
                    dtype=np.float32,
                )
                window = np.concatenate([window, pad], axis=0)

            # Convert to tensor
            x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
            lengths = torch.tensor([end - start], device=self.device)

            # Decode
            predictions = self.model.decode(x, lengths)
            if predictions and predictions[0]:
                all_predictions.extend(predictions[0])

            start += self.step_frames

        # Remove consecutive duplicates from overlapping windows
        deduped = []
        prev = -1
        for token in all_predictions:
            if token != prev:
                deduped.append(token)
            prev = token

        return deduped
