"""
WhisperSign — Real-time Leap Motion Inference
=============================================
Streams hand tracking data from Leap Motion, normalizes it,
and runs continuous sign recognition using a sliding window.

Uses hybrid CTC-Attention decoding (each segment decoded independently,
condition_on_previous_text=False) to prevent hallucination loops.

Requirements:
  - Leap Motion Controller + UltraLeap Gemini Software installed
  - pip install leap-hand-tracking (or use provided mock for testing)
"""
import os
import sys
import time
import torch
import numpy as np
import argparse
import yaml
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.whisper_sign import WhisperSignModel
from src.utils.leap_motion_extract import LeapMotionAdapter
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

    # 3. Inference settings
    inference_cfg = config.get("training", {}).get("inference", {})
    ctc_weight = args.ctc_weight or inference_cfg.get("ctc_weight", 0.5)
    max_decode_length = inference_cfg.get("max_decode_length", 100)
    print(f"  Hybrid decode: ctc_weight={ctc_weight}, max_len={max_decode_length}")
    print(f"  Context: condition_on_previous_text={not args.no_context}")

    # 4. Setup Processing Pipeline
    adapter = LeapMotionAdapter(fps=args.fps)
    spatial_norm = SpatialNormalizer()
    scale_norm = ScaleNormalizer()
    
    slider = SlidingWindowInference(
        model,
        window_duration=args.window_duration,
        overlap=args.overlap,
        sample_rate=args.fps, # Standardize to sensor FPS
        device=device
    )

    # 5. Leap Motion Connection
    print("\nConnecting to Leap Motion sensor...")
    
    if args.mock:
        print("Using MOCK Leap Motion data source for testing.")
        # Simulated hand data generator
        def mock_leap_generator():
            while True:
                # 42 joints, 3 coordinates (mm scale roughly)
                hands = [{
                    "type": "right",
                    "confidence": 1.0,
                    "palm_position": [0, 200, 0],
                    "fingers": [{"bones": [{"prev_joint": [0,0,0], "next_joint": [0,0,0]} for _ in range(4)]} for _ in range(5)]
                }]
                yield hands
                time.sleep(1.0 / args.fps)
        source = mock_leap_generator()
    else:
        try:
            import leap
            # Open connection
            connection = leap.Connection()
            connection.connect()
            print("Successfully connected to Leap Motion Service.")
            
            def leap_generator():
                while True:
                    frame = connection.poll()
                    if frame and frame.hands:
                        # Convert leap-hand-tracking format to adapter-friendly dict
                        hands = []
                        for hand in frame.hands:
                            h_dict = {
                                "type": "left" if hand.type == leap.HandType.Left else "right",
                                "confidence": hand.confidence,
                                "palm_position": [hand.palm.position.x, hand.palm.position.y, hand.palm.position.z],
                                "wrist_position": [hand.arm.next_joint.x, hand.arm.next_joint.y, hand.arm.next_joint.z],
                                "fingers": []
                            }
                            for finger in hand.fingers:
                                f_dict = {"bones": []}
                                for bone in finger.bones:
                                    f_dict["bones"].append({
                                        "prev_joint": [bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z],
                                        "next_joint": [bone.next_joint.x, bone.next_joint.y, bone.next_joint.z]
                                    })
                                h_dict["fingers"].append(f_dict)
                            hands.append(h_dict)
                        yield hands
                    else:
                        yield []
                    time.sleep(1.0 / args.fps)
            source = leap_generator()
            
        except ImportError:
            print("Error: 'leap' package not found. Install it with: pip install leap-hand-tracking")
            print("Falling back to mock mode...")
            return
        except Exception as e:
            print(f"Error connecting to Leap Motion: {e}")
            return

    # 6. Main Inference Loop
    # Each window is decoded INDEPENDENTLY (condition_on_previous_text=False)
    # No decoder state is carried between windows to prevent hallucination chains
    print("\nStarting Real-time Inference. Press Ctrl+C to stop.")
    print("-" * 50)
    
    silence_counter = 0
    silence_threshold = int(args.fps * 0.5)  # 0.5s silence triggers end of action
    min_sequence_length = int(args.fps * 0.5) # Minimum 0.5s sequence
    is_recording = False
    last_pred = []
    
    try:
        for hands in source:
            # Add frame to adapter buffer
            adapter.add_frame(hands)
            
            # Action-end based inference logic
            if len(adapter._frame_buffer) >= 5:
                # Check recent activity to see if moving
                recent_frames = np.stack(adapter._frame_buffer[-5:])
                # Check confidence and standard deviation of position for motion
                avg_conf = np.mean(recent_frames[:, :, 6])
                pos_std = np.mean(np.std(recent_frames[:, :, :3], axis=0))
                
                is_visible = avg_conf >= 0.1
                is_moving = pos_std >= args.motion_threshold
                
                if is_visible and is_moving:
                    silence_counter = 0
                    if not is_recording:
                        is_recording = True
                        last_pred = []
                else:
                    if is_recording:
                        silence_counter += 1
                        if silence_counter > silence_threshold:
                            # Action is Complete: End of movement detected
                            is_recording = False
                            
                            keypoints = adapter.get_sequence(clear_buffer=True)
                            
                            # Only predict if sequence is long enough to be an actual sign
                            if len(keypoints) >= min_sequence_length:
                                # Normalize
                                keypoints = spatial_norm.normalize(keypoints)
                                keypoints = scale_norm.normalize(keypoints)
                                
                                # Infer using hybrid CTC-Attention decode
                                predictions = model.decode(
                                    torch.from_numpy(keypoints).float().unsqueeze(0).to(device),
                                    torch.tensor([len(keypoints)], device=device),
                                    ctc_weight=ctc_weight,
                                    max_decode_length=max_decode_length,
                                )
                                
                                if predictions and predictions[0]:
                                    gloss_ids = predictions[0]
                                    glosses = [id_to_gloss.get(gid, f"?{gid}") for gid in gloss_ids]
                                    glosses = [g for g in glosses if g != "<blank>"]
                                    
                                    if len(glosses) > 0:
                                        print(f"[{time.strftime('%H:%M:%S')}] Action Finished. Recognized: {' '.join(glosses)}")
                                        last_pred = glosses
                    else:
                        # Not recording, maintain a small buffer (5 frames) for start of next action
                        if len(adapter._frame_buffer) > 5:
                            adapter._frame_buffer = adapter._frame_buffer[-5:]

    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        if not args.mock:
            # Cleanup if needed
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperSign Real-time Leap Motion Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--label_map", type=str, help="Path to label_map.json")
    parser.add_argument("--fps", type=int, default=60, help="Processing frame rate")
    parser.add_argument("--window_duration", type=float, default=3.0, help="Sliding window size in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--ctc_weight", type=float, default=None, help="CTC weight for hybrid decode (0-1)")
    parser.add_argument("--no_context", action="store_true", default=True,
                        help="Decode each segment independently (default: True, prevents hallucination)")
    parser.add_argument("--motion_threshold", type=float, default=0.005, help="Minimum motion variance to trigger inference")
    
    args = parser.parse_args()
    run_inference(args)

