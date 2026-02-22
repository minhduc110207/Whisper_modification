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
