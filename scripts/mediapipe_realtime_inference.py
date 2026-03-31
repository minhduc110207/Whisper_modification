import os
import sys
import time
import torch
import cv2
import mediapipe as mp
import numpy as np
import argparse
import yaml
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.whisper_sign import WhisperSignModel
from src.utils.mediapipe_extract import MediaPipeAdapter
from src.utils.sliding_window import SlidingWindowInference
from src.data.normalization import SpatialNormalizer, ScaleNormalizer

def run_inference(args):
    # 1. Load Config & Label Map
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    label_map_path = args.label_map or os.path.join("data/processed", "label_map.json")
    if not os.path.exists(label_map_path):
        print(f"Error: Label map not found at {label_map_path}")
        return

    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    id_to_gloss = {v: k for k, v in label_map.items()}

    # 2. Load Model
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    try:
        model, _ = WhisperSignModel.load_checkpoint(args.checkpoint, device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Setup Processing Pipeline
    adapter = MediaPipeAdapter(
        fps=args.fps,
        use_smoothing=args.smoothing,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        min_confidence=args.min_confidence
    )
    spatial_norm = SpatialNormalizer()
    scale_norm = ScaleNormalizer()
    
    slider = SlidingWindowInference(
        model,
        window_duration=args.window_duration,
        overlap=args.overlap,
        sample_rate=args.fps,
        device=device
    )

    # 4. Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence
    )

    # 5. Open Camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return

    print("\nStarting Real-time MediaPipe Inference. Press 'q' to stop.")
    print("-" * 50)
    
    window_limit = int(args.window_duration * args.fps)
    last_pred = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb)

            # Draw landmarks for visual feedback
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Add to adapter
            adapter.add_frame(results)
            
            # Check if we have enough frames for a window
            if len(adapter._frame_buffer) >= window_limit:
                # Get sequence and compute velocities
                keypoints = adapter.get_sequence(clear_buffer=False)
                
                # Slicing the buffer to keep sliding (FIFO style)
                keep_size = int(window_limit * args.overlap)
                adapter._frame_buffer = adapter._frame_buffer[-keep_size:]
                
                # Normalize
                keypoints = spatial_norm.normalize(keypoints)
                keypoints = scale_norm.normalize(keypoints)
                
                # Movement check: check if hands are visible and actually moving
                # (Prevents hallucinations and frozen frames when user is idle)
                avg_conf = np.mean(keypoints[:, :, 6])
                pos_std = np.mean(np.std(keypoints[:, :, :3], axis=0))
                
                is_visible = avg_conf >= 0.1
                is_moving = pos_std >= args.motion_threshold
                
                if not is_visible or not is_moving:
                    last_pred = [] # Clear last prediction if nothing moving
                else:
                    # Infer using hybrid CTC-Attention decode
                    predictions = slider.model.decode(
                        torch.from_numpy(keypoints).float().unsqueeze(0).to(device),
                        torch.tensor([len(keypoints)], device=device),
                        ctc_weight=args.ctc_weight
                    )
                    
                    if predictions and predictions[0]:
                        gloss_ids = predictions[0]
                        glosses = [id_to_gloss.get(gid, f"?{gid}") for gid in gloss_ids]
                        
                        # Filter out CTC blanks if any leaked through
                        glosses = [g for g in glosses if g != "<blank>"]
                        
                        if glosses != last_pred and len(glosses) > 0:
                            print(f"[{time.strftime('%H:%M:%S')}] Recognized: {' '.join(glosses)}")
                            last_pred = glosses

            # UI Overlay
            status_text = f"Recognized: {' '.join(last_pred)}" if last_pred else "Listening..."
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('WhisperSign Real-time (MediaPipe)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_detector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperSign Real-time MediaPipe Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--label_map", type=str, help="Path to label_map.json")
    parser.add_argument("--fps", type=int, default=30, help="Processing frame rate (camera dependent)")
    parser.add_argument("--window_duration", type=float, default=3.0, help="Sliding window size in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu")
    
    # MediaPipe Optimization Args
    parser.add_argument("--smoothing", type=bool, default=True, help="Enable OneEuroFilter")
    parser.add_argument("--min_cutoff", type=float, default=0.5, help="Smoothing min cutoff (lower = smoother)")
    parser.add_argument("--beta", type=float, default=0.01, help="Smoothing speed coefficient (lower = less lag)")
    parser.add_argument("--min_confidence", type=float, default=0.4, help="Landmark confidence threshold")
    parser.add_argument("--motion_threshold", type=float, default=0.005, help="Minimum motion variance to trigger inference")
    parser.add_argument("--ctc_weight", type=float, default=0.7, help="CTC weight for stable decoding (0.0-1.0)")
    
    args = parser.parse_args()
    run_inference(args)
