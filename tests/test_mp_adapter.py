import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.mediapipe_extract import MediaPipeAdapter

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

class MockClassification:
    def __init__(self, label, score):
        self.label = label
        self.score = score

class MockHandedness:
    def __init__(self, classification):
        self.classification = [classification]

class MockResults:
    def __init__(self, multi_hand_landmarks, multi_handedness):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness

def test_mediapipe_adapter():
    # Test with 60 FPS
    adapter = MediaPipeAdapter(fps=60)
    
    # Create 5 mock results (5 frames)
    for i in range(5):
        # Move right hand wrist (index 21) from x=0.1 to 0.5
        x_val = 0.1 + (i * 0.1)
        wrist = MockLandmark(x_val, 0.2, 0.3)
        landmarks = [wrist] + [MockLandmark(0,0,0) for _ in range(20)]
        
        results = MockResults(
            multi_hand_landmarks=[MockHandLandmarks(landmarks)],
            multi_handedness=[MockHandedness(MockClassification("Right", 0.95))]
        )
        adapter.add_frame(results)
    
    # Get sequence
    sequence = adapter.get_sequence(clear_buffer=True)
    
    # Check shape: (5, 42, 7)
    print(f"Sequence shape: {sequence.shape}")
    assert sequence.shape == (5, 42, 7), f"Expected (5, 42, 7), got {sequence.shape}"
    
    # Check values for right hand wrist (index 21)
    np.testing.assert_almost_equal(sequence[:, 21, 0], [0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Check velocities vx (index 3)
    # dx = 0.1, dt = 1/60, vx = 0.1 / (1/60) = 6.0
    print(f"Wrist VX velocities: {sequence[:, 21, 3]}")
    np.testing.assert_almost_equal(sequence[:, 21, 3], [6.0, 6.0, 6.0, 6.0, 6.0], decimal=5)
    
    print(">>> MediaPipeAdapter synthetic test PASSED <<<")

if __name__ == "__main__":
    try:
        test_mediapipe_adapter()
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
