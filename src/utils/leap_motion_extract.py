"""
Leap Motion (Ultraleap) data adapter for WhisperSign.

Converts Leap Motion hand tracking data into the same (T, 42, 7) format
used by the MediaPipe pipeline, enabling inference with a trained model.

Leap Motion SDK differences from MediaPipe:
  - Units: millimeters (not normalized 0-1)
  - Coordinate system: right-handed, origin at sensor center
  - Joint order: per-bone (4 bones per finger), not per-landmark
  - Provides palm, wrist, and 20 finger joints (5 fingers × 4 joints)

This module handles:
  1. Joint reordering: Leap Motion bone order → MediaPipe landmark order
  2. Coordinate normalization: mm → normalized [0,1] range
  3. Velocity computation: using np.gradient
  4. Real-time streaming: accumulate frames from Leap Motion callback

Usage:
  adapter = LeapMotionAdapter()

  # From saved Leap Motion CSV/JSON data
  keypoints = adapter.from_csv("leap_recording.csv")

  # From a list of frame dicts (real-time)
  adapter.add_frame(leap_frame_dict)
  keypoints = adapter.get_sequence()

  # Then use with the model
  model.decode(keypoints, lengths)
"""
import numpy as np
from typing import List, Dict, Optional, Tuple


# =================================================================
# Leap Motion → MediaPipe joint mapping
# =================================================================
# MediaPipe hand landmarks (21 per hand):
#   0: Wrist
#   1: Thumb_CMC, 2: Thumb_MCP, 3: Thumb_IP, 4: Thumb_TIP
#   5: Index_MCP, 6: Index_PIP, 7: Index_DIP, 8: Index_TIP
#   9: Middle_MCP, 10: Middle_PIP, 11: Middle_DIP, 12: Middle_TIP
#   13: Ring_MCP, 14: Ring_PIP, 15: Ring_DIP, 16: Ring_TIP
#   17: Pinky_MCP, 18: Pinky_PIP, 19: Pinky_DIP, 20: Pinky_TIP
#
# Leap Motion finger bones (per finger):
#   Bone 0: Metacarpal (base), Bone 1: Proximal,
#   Bone 2: Intermediate (or IP for thumb), Bone 3: Distal
#   Each bone has a prev_joint (start) and next_joint (end)
# =================================================================

# Maps (finger_name, joint_index) to MediaPipe landmark index
# Leap Motion fingers: "thumb"=0, "index"=1, "middle"=2, "ring"=3, "pinky"=4
# For each finger, we use the bone start/end joints to reconstruct
# the MediaPipe landmark positions.
LEAP_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# MediaPipe landmark indices for each finger's joints
# Format: {finger_index: [MCP/CMC, PIP/MCP, DIP/IP, TIP]}
MEDIAPIPE_FINGER_LANDMARKS = {
    0: [1, 2, 3, 4],      # Thumb: CMC, MCP, IP, TIP
    1: [5, 6, 7, 8],      # Index: MCP, PIP, DIP, TIP
    2: [9, 10, 11, 12],   # Middle
    3: [13, 14, 15, 16],  # Ring
    4: [17, 18, 19, 20],  # Pinky
}


class LeapMotionAdapter:
    """
    Converts Leap Motion hand tracking data to WhisperSign (T, 42, 7) format.

    Handles coordinate normalization, joint reordering, velocity computation,
    and frame accumulation for both batch and real-time use cases.
    """

    def __init__(
        self,
        normalize_coords: bool = True,
        norm_range: Tuple[float, float] = (0.0, 1.0),
        sensor_range_mm: float = 500.0,
        fps: float = 120.0,
    ):
        """
        Args:
            normalize_coords: If True, normalize mm coordinates to [0, 1]
            norm_range: Target range for normalization
            sensor_range_mm: Approximate interaction range in mm (for normalization)
                Leap Motion effective range is roughly ±250mm in x/z, 50-600mm in y
            fps: Frame rate of the Leap Motion data (typically 120 Hz)
        """
        self.normalize_coords = normalize_coords
        self.norm_range = norm_range
        self.sensor_range_mm = sensor_range_mm
        self.fps = fps

        # Buffer for real-time frame accumulation
        self._frame_buffer: List[np.ndarray] = []

    def convert_frame(
        self,
        hands: List[Dict],
    ) -> np.ndarray:
        """
        Convert a single Leap Motion frame to WhisperSign format.

        Args:
            hands: List of hand dictionaries from Leap Motion.
                Each hand dict should contain:
                - "type": "left" or "right"
                - "palm_position": [x, y, z] in mm
                - "wrist_position": [x, y, z] in mm (if available)
                - "confidence": float (0-1)
                - "fingers": list of 5 finger dicts, each containing:
                    - "bones": list of 4 bone dicts, each containing:
                        - "prev_joint": [x, y, z] (start of bone)
                        - "next_joint": [x, y, z] (end of bone)

        Returns:
            frame_keypoints: (42, 7) numpy array
        """
        frame = np.zeros((42, 7), dtype=np.float32)

        for hand in hands:
            hand_type = hand.get("type", "right").lower()
            offset = 0 if hand_type == "left" else 21
            confidence = hand.get("confidence", 1.0)

            # Wrist (MediaPipe index 0)
            wrist_pos = hand.get("wrist_position")
            if wrist_pos is None:
                # Fall back to palm position if wrist not available
                wrist_pos = hand.get("palm_position", [0, 0, 0])
            frame[offset + 0, :3] = self._normalize_point(wrist_pos)
            frame[offset + 0, 6] = confidence

            # Fingers
            fingers = hand.get("fingers", [])
            for finger_idx, finger in enumerate(fingers):
                if finger_idx >= 5:
                    break
                mp_indices = MEDIAPIPE_FINGER_LANDMARKS[finger_idx]
                bones = finger.get("bones", [])

                for bone_idx, bone in enumerate(bones):
                    if bone_idx >= 4:
                        break
                    mp_joint_idx = mp_indices[bone_idx]

                    if bone_idx < 3:
                        # For first 3 joints: use the start of the bone
                        pos = bone.get("prev_joint", [0, 0, 0])
                    else:
                        # For TIP (last joint): use the end of the last bone
                        pos = bone.get("next_joint", [0, 0, 0])

                    frame[offset + mp_joint_idx, :3] = self._normalize_point(pos)
                    frame[offset + mp_joint_idx, 6] = confidence

        return frame

    def _normalize_point(self, point: list) -> np.ndarray:
        """
        Normalize a 3D point from Leap Motion mm coordinates.

        Leap Motion coordinate system:
          - Origin: center of the sensor
          - X: left-right (mm)
          - Y: up from sensor (mm), typically 50-600mm
          - Z: toward/away from screen (mm)

        MediaPipe coordinate system:
          - X, Y: normalized to [0, 1] relative to image
          - Z: relative depth

        We map Leap coords to [0, 1] for compatibility.
        """
        pt = np.array(point, dtype=np.float32)

        if self.normalize_coords:
            # Normalize to [0, 1] range
            # X: center at 0, range roughly ±sensor_range/2
            pt[0] = (pt[0] / self.sensor_range_mm) + 0.5
            # Y: starts at ~50mm above sensor, normalize
            pt[1] = pt[1] / self.sensor_range_mm
            # Z: center at 0, range roughly ±sensor_range/2
            pt[2] = (pt[2] / self.sensor_range_mm) + 0.5

            # Clamp to [0, 1]
            pt = np.clip(pt, 0.0, 1.0)

        return pt

    def add_frame(self, hands: List[Dict]):
        """
        Add a frame to the internal buffer (for real-time streaming).

        Args:
            hands: List of hand dicts (same format as convert_frame)
        """
        frame = self.convert_frame(hands)
        self._frame_buffer.append(frame)

    def get_sequence(self, clear_buffer: bool = True) -> np.ndarray:
        """
        Get the accumulated sequence with velocities computed.

        Args:
            clear_buffer: If True, clear the buffer after retrieving

        Returns:
            keypoints: (T, 42, 7) numpy array with velocities computed
        """
        if not self._frame_buffer:
            return np.zeros((0, 42, 7), dtype=np.float32)

        sequence = np.stack(self._frame_buffer)  # (T, 42, 7)
        sequence = self._compute_velocities(sequence)

        if clear_buffer:
            self._frame_buffer.clear()

        return sequence

    def clear_buffer(self):
        """Clear the frame buffer."""
        self._frame_buffer.clear()

    def _compute_velocities(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Compute velocity features (vx, vy, vz) from position data.

        Args:
            keypoints: (T, 42, 7) array with positions in [:, :, :3]

        Returns:
            keypoints with velocities filled in [:, :, 3:6]
        """
        if keypoints.shape[0] > 1:
            dt = 1.0 / self.fps
            velocity = np.gradient(keypoints[:, :, :3], dt, axis=0)
            keypoints[:, :, 3:6] = velocity
        return keypoints

    # =================================================================
    # Convenience: Load from common Leap Motion export formats
    # =================================================================

    def from_csv(
        self,
        csv_path: str,
        hand_type: str = "right",
    ) -> np.ndarray:
        """
        Load Leap Motion data from a CSV file.

        Expects columns for each joint's x, y, z coordinates.
        Common CSV format from Leap Motion recordings:
          frame_id, timestamp, hand_type,
          wrist_x, wrist_y, wrist_z,
          thumb_cmc_x, thumb_cmc_y, thumb_cmc_z,
          thumb_mcp_x, ..., pinky_tip_z

        Args:
            csv_path: Path to CSV file
            hand_type: Default hand type if not in CSV

        Returns:
            keypoints: (T, 42, 7) numpy array
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        T = len(df)

        keypoints = np.zeros((T, 42, 7), dtype=np.float32)
        offset = 0 if hand_type == "left" else 21

        # Try to find coordinate columns
        # Pattern 1: wrist_x, wrist_y, wrist_z, thumb_cmc_x, ...
        joint_names = [
            "wrist",
            "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
        ]

        for joint_idx, joint_name in enumerate(joint_names):
            for axis_idx, axis in enumerate(["x", "y", "z"]):
                col_name = f"{joint_name}_{axis}"
                if col_name in df.columns:
                    values = df[col_name].values.astype(np.float32)
                    if self.normalize_coords:
                        if axis == "x" or axis == "z":
                            values = (values / self.sensor_range_mm) + 0.5
                        else:  # y
                            values = values / self.sensor_range_mm
                        values = np.clip(values, 0, 1)
                    keypoints[:, offset + joint_idx, axis_idx] = values
                    keypoints[:, offset + joint_idx, 6] = 1.0  # confidence

        keypoints = self._compute_velocities(keypoints)
        return keypoints

    def from_json_frames(
        self,
        frames: List[Dict],
    ) -> np.ndarray:
        """
        Convert a list of Leap Motion frame dictionaries.

        Args:
            frames: List of frame dicts, each containing a "hands" key
                with the standard Leap Motion hand format.

        Returns:
            keypoints: (T, 42, 7) numpy array
        """
        self._frame_buffer.clear()
        for frame in frames:
            hands = frame.get("hands", [])
            self.add_frame(hands)
        return self.get_sequence(clear_buffer=True)

    def from_numpy(
        self,
        data: np.ndarray,
        hand_type: str = "right",
        input_format: str = "joints_xyz",
    ) -> np.ndarray:
        """
        Convert raw Leap Motion numpy data.

        Supports common export formats:
          - "joints_xyz": (T, 21, 3) — 21 joints with x,y,z
          - "flat": (T, 63) — flattened 21×3
          - "bones": (T, 20, 6) — 20 bones with start_xyz + end_xyz

        Args:
            data: Raw numpy array from Leap Motion
            hand_type: "left" or "right"
            input_format: One of "joints_xyz", "flat", "bones"

        Returns:
            keypoints: (T, 42, 7) numpy array
        """
        offset = 0 if hand_type == "left" else 21
        T = data.shape[0]
        keypoints = np.zeros((T, 42, 7), dtype=np.float32)

        if input_format == "joints_xyz":
            # (T, 21, 3) → directly map
            n_joints = min(data.shape[1], 21)
            coords = data[:, :n_joints, :3].copy()
            if self.normalize_coords:
                coords[:, :, 0] = (coords[:, :, 0] / self.sensor_range_mm) + 0.5
                coords[:, :, 1] = coords[:, :, 1] / self.sensor_range_mm
                coords[:, :, 2] = (coords[:, :, 2] / self.sensor_range_mm) + 0.5
                coords = np.clip(coords, 0, 1)
            keypoints[:, offset:offset + n_joints, :3] = coords
            keypoints[:, offset:offset + n_joints, 6] = 1.0

        elif input_format == "flat":
            # (T, 63) → reshape to (T, 21, 3)
            reshaped = data.reshape(T, -1, 3)
            return self.from_numpy(reshaped, hand_type, "joints_xyz")

        elif input_format == "bones":
            # (T, 20, 6) → each bone has [start_x,y,z, end_x,y,z]
            # Use bone starts for first 3 joints of each finger, bone end for TIP
            for finger_idx in range(5):
                mp_indices = MEDIAPIPE_FINGER_LANDMARKS[finger_idx]
                for bone_idx in range(4):
                    bone_data_idx = finger_idx * 4 + bone_idx
                    if bone_data_idx >= data.shape[1]:
                        break
                    mp_joint = mp_indices[bone_idx]
                    if bone_idx < 3:
                        pos = data[:, bone_data_idx, :3]  # prev_joint
                    else:
                        pos = data[:, bone_data_idx, 3:6]  # next_joint (TIP)
                    coords = pos.copy()
                    if self.normalize_coords:
                        coords[:, 0] = (coords[:, 0] / self.sensor_range_mm) + 0.5
                        coords[:, 1] = coords[:, 1] / self.sensor_range_mm
                        coords[:, 2] = (coords[:, 2] / self.sensor_range_mm) + 0.5
                        coords = np.clip(coords, 0, 1)
                    keypoints[:, offset + mp_joint, :3] = coords
                    keypoints[:, offset + mp_joint, 6] = 1.0

        keypoints = self._compute_velocities(keypoints)
        return keypoints
