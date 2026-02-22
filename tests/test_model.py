"""
Tests for the WhisperSign model - end-to-end verification.
Run with: python -m pytest tests/ -v
"""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================
# Test Configuration
# =========================================
TEST_CONFIG = {
    "frontend": {
        "num_joints": 42,
        "num_features": 7,
        "patch_size": 4,
        "d_model": 256,  # Smaller for testing
        "dropout": 0.1,
        "spatial_dropout": 0.15,
    },
    "encoder": {
        "num_heads": 4,
        "num_layers": 2,  # Fewer layers for testing
        "d_model": 256,
        "d_ff": 512,
        "dropout": 0.1,
    },
    "decoder": {
        "vocab_size": 100,
        "blank_id": 0,
    },
}

BATCH_SIZE = 2
SEQ_LEN = 120   # 2 seconds at 60Hz
NUM_JOINTS = 42
NUM_FEATURES = 7


# =========================================
# Preprocessing Tests
# =========================================
class TestPreprocessing:

    def test_resample_to_fixed_rate(self):
        from src.data.preprocessing import resample_to_fixed_rate

        T_raw = 100
        data = np.random.randn(T_raw, NUM_JOINTS, NUM_FEATURES).astype(np.float32)
        timestamps = np.linspace(0, 2.0, T_raw)  # 2 seconds, variable rate

        resampled, new_ts = resample_to_fixed_rate(data, timestamps, target_rate=60)

        # Should have ~120 frames for 2 seconds at 60Hz
        assert resampled.shape[0] == 121  # linspace inclusive
        assert resampled.shape[1] == NUM_JOINTS
        assert resampled.shape[2] == NUM_FEATURES

    def test_preprocess_sequence_pad(self):
        from src.data.preprocessing import preprocess_sequence

        data = np.random.randn(50, NUM_JOINTS, NUM_FEATURES).astype(np.float32)
        result = preprocess_sequence(data, max_seq_length=200)

        assert result.shape == (200, NUM_JOINTS, NUM_FEATURES)
        # Check padding is zeros
        assert np.allclose(result[50:], 0.0)

    def test_preprocess_sequence_truncate(self):
        from src.data.preprocessing import preprocess_sequence

        data = np.random.randn(300, NUM_JOINTS, NUM_FEATURES).astype(np.float32)
        result = preprocess_sequence(data, max_seq_length=200)

        assert result.shape == (200, NUM_JOINTS, NUM_FEATURES)

    def test_sequence_mask(self):
        from src.data.preprocessing import compute_sequence_mask

        mask = compute_sequence_mask(50, 200)
        assert mask.sum() == 50
        assert mask[:50].all()
        assert not mask[50:].any()


# =========================================
# Normalization Tests
# =========================================
class TestNormalization:

    def test_spatial_normalization(self):
        from src.data.normalization import SpatialNormalizer

        norm = SpatialNormalizer()
        data = np.random.randn(SEQ_LEN, NUM_JOINTS, NUM_FEATURES).astype(np.float32)

        result = norm.normalize(data)

        # Palm positions should be zero after normalization (for xyz)
        assert np.allclose(result[:, 0, :3], 0.0, atol=1e-6)  # Left palm
        assert np.allclose(result[:, 21, :3], 0.0, atol=1e-6)  # Right palm

    def test_scale_normalization(self):
        from src.data.normalization import ScaleNormalizer

        norm = ScaleNormalizer()
        data = np.random.randn(SEQ_LEN, NUM_JOINTS, NUM_FEATURES).astype(np.float32)

        result = norm.normalize(data)
        assert result.shape == data.shape


# =========================================
# Augmentation Tests
# =========================================
class TestAugmentation:

    def test_gesture_masking(self):
        from src.data.augmentation import GestureMasking

        masker = GestureMasking(joint_mask_prob=0.5, temporal_mask_prob=1.0)
        data = np.ones((SEQ_LEN, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)

        result = masker(data)
        # Some values should be zeroed out
        assert (result == 0).any()

    def test_noise_injection(self):
        from src.data.augmentation import NoiseInjection

        noiser = NoiseInjection(std=0.01)
        data = np.zeros((SEQ_LEN, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)

        result = noiser(data)
        assert not np.allclose(result, 0.0)


# =========================================
# Frontend Tests
# =========================================
class TestFrontend:

    def test_forward_shape(self):
        from src.model.frontend import SignLanguageFrontend

        frontend = SignLanguageFrontend(
            num_joints=42, num_features=7,
            patch_size=4, d_model=256,
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, NUM_FEATURES)
        lengths = torch.tensor([SEQ_LEN, 100])

        out, out_lengths = frontend(x, lengths)

        expected_T = SEQ_LEN // 4  # patch_size = 4
        assert out.shape == (BATCH_SIZE, expected_T, 256)
        assert out_lengths is not None

    def test_patch_embedding(self):
        from src.model.frontend import TemporalPatchEmbedding

        embed = TemporalPatchEmbedding(
            num_joints=42, num_features=7,
            patch_size=4, d_model=256,
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, NUM_FEATURES)
        out = embed(x)

        assert out.shape == (BATCH_SIZE, SEQ_LEN // 4, 256)


# =========================================
# Encoder Tests
# =========================================
class TestEncoder:

    def test_spatio_temporal_block(self):
        from src.model.encoder import SpatioTemporalBlock

        block = SpatioTemporalBlock(d_model=256, num_heads=4, d_ff=512)
        x = torch.randn(BATCH_SIZE, 30, 256)

        out = block(x)
        assert out.shape == x.shape

    def test_full_encoder(self):
        from src.model.encoder import SpatioTemporalEncoder

        encoder = SpatioTemporalEncoder(
            num_layers=2, d_model=256, num_heads=4, d_ff=512,
        )
        x = torch.randn(BATCH_SIZE, 30, 256)
        out = encoder(x)

        assert out.shape == x.shape


# =========================================
# Decoder Tests
# =========================================
class TestDecoder:

    def test_ctc_decoder(self):
        from src.model.decoder import CTCDecoder

        decoder = CTCDecoder(d_model=256, vocab_size=100)
        x = torch.randn(BATCH_SIZE, 30, 256)

        log_probs = decoder(x)
        assert log_probs.shape == (BATCH_SIZE, 30, 100)

        # Check valid log probabilities
        probs = log_probs.exp()
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_ctc_greedy_decode(self):
        from src.model.decoder import CTCDecoder

        decoder = CTCDecoder(d_model=256, vocab_size=100)
        x = torch.randn(BATCH_SIZE, 30, 256)
        log_probs = decoder(x)

        decoded = decoder.greedy_decode(log_probs)
        assert len(decoded) == BATCH_SIZE
        assert all(isinstance(seq, list) for seq in decoded)

    def test_two_pass_decoder(self):
        from src.model.decoder import TwoPassDecoder

        decoder = TwoPassDecoder(
            d_model=256, vocab_size=100,
            num_heads=4, num_decoder_layers=2, d_ff=512,
        )
        encoder_output = torch.randn(BATCH_SIZE, 30, 256)
        target_tokens = torch.randint(0, 100, (BATCH_SIZE, 10))

        ctc_probs, att_logits = decoder(encoder_output, target_tokens)

        assert ctc_probs.shape == (BATCH_SIZE, 30, 100)
        assert att_logits.shape == (BATCH_SIZE, 10, 100)


# =========================================
# Full Model Tests
# =========================================
class TestWhisperSignModel:

    def test_forward_pass(self):
        from src.model.whisper_sign import WhisperSignModel

        model = WhisperSignModel(TEST_CONFIG)
        features = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, NUM_FEATURES)
        lengths = torch.tensor([SEQ_LEN, 100])

        outputs = model(features, lengths)

        assert "ctc_log_probs" in outputs
        assert "encoder_output" in outputs
        assert "output_lengths" in outputs
        assert outputs["ctc_log_probs"].shape[0] == BATCH_SIZE

    def test_decode(self):
        from src.model.whisper_sign import WhisperSignModel

        model = WhisperSignModel(TEST_CONFIG)
        features = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, NUM_FEATURES)
        lengths = torch.tensor([SEQ_LEN, 100])

        decoded = model.decode(features, lengths)

        assert len(decoded) == BATCH_SIZE

    def test_freeze_unfreeze(self):
        from src.model.whisper_sign import WhisperSignModel

        model = WhisperSignModel(TEST_CONFIG)

        total_before = model.get_num_params(trainable_only=True)

        model.freeze_encoder()
        model.freeze_decoder()
        trainable_frozen = model.get_num_params(trainable_only=True)

        # Should have fewer trainable params
        assert trainable_frozen < total_before

        model.unfreeze_encoder()
        model.unfreeze_decoder()
        trainable_unfrozen = model.get_num_params(trainable_only=True)

        assert trainable_unfrozen == total_before

    def test_gradient_flow(self):
        from src.model.whisper_sign import WhisperSignModel

        model = WhisperSignModel(TEST_CONFIG)
        features = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_JOINTS, NUM_FEATURES)
        lengths = torch.tensor([SEQ_LEN, 100])

        outputs = model(features, lengths)
        loss = outputs["ctc_log_probs"].sum()
        loss.backward()

        # Check gradients exist for frontend
        for name, param in model.frontend.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for frontend.{name}"

    def test_checkpoint_save_load(self, tmp_path):
        from src.model.whisper_sign import WhisperSignModel

        model = WhisperSignModel(TEST_CONFIG)
        ckpt_path = str(tmp_path / "test_ckpt.pt")

        model.save_checkpoint(ckpt_path, epoch=5, loss=0.5)

        loaded_model, checkpoint = WhisperSignModel.load_checkpoint(ckpt_path)

        assert checkpoint["epoch"] == 5
        assert checkpoint["loss"] == 0.5

        # Check weights are identical
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)


# =========================================
# Loss Tests
# =========================================
class TestLoss:

    def test_hybrid_loss_ctc_only(self):
        from src.training.losses import HybridCTCAttentionLoss

        loss_fn = HybridCTCAttentionLoss(alpha=1.0)

        ctc_log_probs = torch.randn(BATCH_SIZE, 30, 100).log_softmax(dim=-1)
        labels = torch.randint(1, 100, (10,))  # Avoid blank=0
        output_lengths = torch.tensor([30, 30])
        label_lengths = torch.tensor([5, 5])

        result = loss_fn(ctc_log_probs, None, labels, output_lengths, label_lengths)

        assert "total" in result
        assert "ctc" in result
        assert result["total"].item() > 0

    def test_hybrid_loss_combined(self):
        from src.training.losses import HybridCTCAttentionLoss

        loss_fn = HybridCTCAttentionLoss(alpha=0.3)

        ctc_log_probs = torch.randn(BATCH_SIZE, 30, 100).log_softmax(dim=-1)
        att_logits = torch.randn(BATCH_SIZE, 10, 100)
        labels = torch.randint(1, 100, (10,))
        output_lengths = torch.tensor([30, 30])
        label_lengths = torch.tensor([5, 5])
        att_targets = torch.randint(0, 100, (BATCH_SIZE, 10))

        result = loss_fn(
            ctc_log_probs, att_logits, labels,
            output_lengths, label_lengths, att_targets,
        )

        assert result["total"].item() > 0
        assert result["ctc"].item() > 0
        assert result["attention"].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
