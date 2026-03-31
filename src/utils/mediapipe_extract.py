"""
MediaPipe keypoint extraction from video.
Extracts 21 left + 21 right hand landmarks (42 joints)
and outputs tensor of shape (T, 42, F).
"""
import numpy as np
from typing import Optional, Tuple


def extract_hand_keypoints(
    video_path: str,
    target_fps: int = 60,
    num_features: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hand keypoints from a video file using MediaPipe.

    Args:
        video_path: Path to video file
        target_fps: Target frame rate for resampling
        num_features: Number of features per joint
            (x, y, z, vx, vy, vz, confidence) = 7

    Returns:
        keypoints: (T, 42, F) numpy array
        timestamps: (T,) timestamps in seconds
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "Please install opencv-python and mediapipe: "
            "pip install opencv-python mediapipe"
        )

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_data = []
    timestamps = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        frame_keypoints = np.zeros((42, num_features), dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                confidence = handedness.classification[0].score

                # Determine offset: Left=0, Right=21
                offset = 0 if label == "Left" else 21

                for i, lm in enumerate(hand_landmarks.landmark):
                    idx = offset + i
                    frame_keypoints[idx, 0] = lm.x  # Normalized x
                    frame_keypoints[idx, 1] = lm.y  # Normalized y
                    frame_keypoints[idx, 2] = lm.z  # Normalized z
                    # Velocity will be computed later
                    frame_keypoints[idx, 6] = confidence

        frames_data.append(frame_keypoints)
        timestamps.append(frame_idx / fps)
        frame_idx += 1

    cap.release()
    hands.close()

    if not frames_data:
        return np.zeros((0, 42, num_features)), np.array([])

    keypoints = np.stack(frames_data)  # (T, 42, F)
    timestamps = np.array(timestamps)

    # Compute velocities (features 3, 4, 5)
    if keypoints.shape[0] > 1:
        dt = 1.0 / fps
        velocity = np.gradient(keypoints[:, :, :3], dt, axis=0)
        keypoints[:, :, 3:6] = velocity

    return keypoints, timestamps


def extract_from_dataset(
    video_dir: str,
    output_dir: str,
    target_fps: int = 60,
):
    """
    Batch extract keypoints from a directory of videos.

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save .npy keypoint files
        target_fps: Target frame rate
    """
    import os
    from ..data.preprocessing import resample_to_fixed_rate
    from ..data.normalization import SpatialNormalizer, ScaleNormalizer

    os.makedirs(output_dir, exist_ok=True)

    spatial_norm = SpatialNormalizer()
    scale_norm = ScaleNormalizer()

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    for fname in sorted(os.listdir(video_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in video_extensions:
            continue

        video_path = os.path.join(video_dir, fname)
        sample_id = os.path.splitext(fname)[0]

        print(f"Processing: {fname}")

        keypoints, timestamps = extract_hand_keypoints(
            video_path, target_fps
        )

        if keypoints.shape[0] == 0:
            print(f"  WARNING: No frames extracted from {fname}")
            continue

        # Resample to fixed rate
        keypoints, _ = resample_to_fixed_rate(
            keypoints, timestamps, target_fps
        )

        # Normalize
        keypoints = spatial_norm.normalize(keypoints)
        keypoints = scale_norm.normalize(keypoints)

        # Save
        output_path = os.path.join(output_dir, f"{sample_id}.npy")
        np.save(output_path, keypoints.astype(np.float32))
        print(f"  Saved: {output_path} | Shape: {keypoints.shape}")


class OneEuroFilter:
    """
    Adaptive low-pass filter for noisy signal smoothing.
    Balances between low-speed jitter reduction and high-speed lag reduction.
    """

    def __init__(self, t0: float, x0: np.ndarray, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        dt = t - self.t_prev
        if dt <= 0:
            return x

        # Filter the derivative (velocity) to get the adaptive cutoff
        d_alpha = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = d_alpha * dx + (1 - d_alpha) * self.dx_prev

        # Compute adaptive cutoff for the signal
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(cutoff, dt)

        # Filter the signal
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class MediaPipeAdapter:
    """
    Adapter for real-time MediaPipe hand tracking results.
    Accumulates frames, computes velocities, and formats for WhisperSign (T, 42, 7).
    Includes temporal smoothing (OneEuroFilter) and confidence-based gating.
    """

    def __init__(
        self,
        fps: float = 30.0,
        target_fps: float = 60.0,
        use_smoothing: bool = True,
        min_cutoff: float = 0.5,
        beta: float = 0.01,
        min_confidence: float = 0.4
    ):
        """
        Args:
            fps: Expected frame rate of the input stream (e.g. webcam).
            target_fps: Target frame rate expected by the model (e.g. 60).
            use_smoothing: Enable OneEuroFilter.
            min_cutoff: Smoothing cutoff at zero velocity (higher = less smooth).
            beta: Speed coefficient for adaptive smoothing (higher = less lag).
            min_confidence: Landmark confidence threshold.
        """
        self.fps = fps
        self.target_fps = target_fps
        self.use_smoothing = use_smoothing
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.min_confidence = min_confidence
        
        self._frame_buffer = []
        self._timestamps = []
        self._start_time = None
        self._filters = None  # Lazy init on first frame
        self._last_valid_frame = None
        self._frame_count = 0

    def process_results(self, results) -> np.ndarray:
        """
        Convert MediaPipe Hand results object to a single (42, 7) frame.
        Applies confidence gating: holds previous values if detection is poor.
        """
        frame = np.zeros((42, 7), dtype=np.float32)
        has_hands = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            has_hands = True
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                # Confidence gate: skip if too low
                if confidence < self.min_confidence:
                    continue
                
                # Left=0, Right=21
                offset = 0 if label == "Left" else 21
                
                for i, lm in enumerate(hand_landmarks.landmark):
                    idx = offset + i
                    frame[idx, 0] = lm.x
                    frame[idx, 1] = lm.y
                    frame[idx, 2] = lm.z
                    frame[idx, 6] = confidence

        # Self-repair: if no hands detected, but we have a recent valid frame,
        # we keep positions but set confidence to 0 (velocity will be 0)
        # to prevent hallucinations from sudden jumps to (0,0,0).
        if not has_hands and self._last_valid_frame is not None:
            frame[:, :3] = self._last_valid_frame[:, :3]
            frame[:, 6] = 0.0  # Zero confidence
        
        if has_hands:
            self._last_valid_frame = frame.copy()
            
        return frame

    def add_frame(self, results):
        """
        Add a MediaPipe results object to the internal buffer with smoothing.
        """
        frame = self.process_results(results)
        
        if self.use_smoothing:
            t = self._frame_count / self.fps
            
            # Initialize filters on first valid frame
            if self._filters is None:
                self._filters = [
                    OneEuroFilter(t, frame[j, :3], self.min_cutoff, self.beta)
                    for j in range(42)
                ]
            else:
                for j in range(42):
                    # Only smooth position (features 0,1,2)
                    frame[j, :3] = self._filters[j](t, frame[j, :3])
        
        import time
        if self._start_time is None:
            self._start_time = time.time()
        self._timestamps.append(time.time() - self._start_time)
        
        self._frame_buffer.append(frame)
        self._frame_count += 1

    def get_sequence(self, clear_buffer: bool = True) -> np.ndarray:
        """
        Get the accumulated sequence with velocities computed.
        """
        if not self._frame_buffer:
            return np.zeros((0, 42, 7), dtype=np.float32)

        sequence = np.stack(self._frame_buffer)
        
        # Resample to target_fps if needed
        import math
        if self.target_fps and not math.isclose(self.fps, self.target_fps) and len(self._frame_buffer) > 1:
            from ..data.preprocessing import resample_to_fixed_rate
            timestamp_arr = np.array(self._timestamps)
            try:
                sequence, _ = resample_to_fixed_rate(
                    sequence, timestamp_arr, target_rate=int(self.target_fps)
                )
            except Exception as e:
                print(f"Warning: Interpolation failed, falling back to raw sequence: {e}")
        
        # Compute velocities (vx, vy, vz) in features 3, 4, 5
        if sequence.shape[0] > 1:
            dt = 1.0 / (self.target_fps if self.target_fps else self.fps)
            velocity = np.gradient(sequence[:, :, :3], dt, axis=0)
            sequence[:, :, 3:6] = velocity
            
        if clear_buffer:
            self._frame_buffer = []
            self._timestamps = []
            
        return sequence

