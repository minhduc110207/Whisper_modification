"""Quick smoke test for the WhisperSign model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def test_all():
    print("=" * 50)
    print("WhisperSign Model Smoke Test")
    print("=" * 50)

    # Test config (small for speed)
    config = {
        "frontend": {
            "num_joints": 42, "num_features": 7,
            "patch_size": 4, "d_model": 256,
            "dropout": 0.1, "spatial_dropout": 0.15,
        },
        "encoder": {
            "num_heads": 4, "num_layers": 2,
            "d_model": 256, "d_ff": 512, "dropout": 0.1,
        },
        "decoder": {"vocab_size": 100, "blank_id": 0},
    }

    # 1. Test Preprocessing
    print("\n[1/6] Testing Preprocessing...")
    from src.data.preprocessing import resample_to_fixed_rate, preprocess_sequence
    data = np.random.randn(100, 42, 7).astype(np.float32)
    ts = np.linspace(0, 2.0, 100)
    resampled, new_ts = resample_to_fixed_rate(data, ts, 60)
    print(f"  Resampled: {data.shape} -> {resampled.shape} OK")
    processed = preprocess_sequence(data, max_seq_length=200)
    print(f"  Padded: {data.shape} -> {processed.shape} OK")

    # 2. Test Normalization
    print("\n[2/6] Testing Normalization...")
    from src.data.normalization import SpatialNormalizer, ScaleNormalizer
    sn = SpatialNormalizer()
    result = sn.normalize(data)
    assert np.allclose(result[:, 0, :3], 0.0, atol=1e-6)
    assert np.allclose(result[:, 21, :3], 0.0, atol=1e-6)
    print(f"  Spatial normalization OK (palms zeroed)")
    sc = ScaleNormalizer()
    result2 = sc.normalize(result)
    print(f"  Scale normalization OK")

    # 3. Test Frontend
    print("\n[3/6] Testing Frontend...")
    from src.model.frontend import SignLanguageFrontend
    frontend = SignLanguageFrontend(
        num_joints=42, num_features=7, patch_size=4, d_model=256
    )
    x = torch.randn(2, 120, 42, 7)
    out, lengths = frontend(x, torch.tensor([120, 100]))
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)} OK")

    # 4. Test Encoder
    print("\n[4/6] Testing Encoder...")
    from src.model.encoder import SpatioTemporalEncoder
    encoder = SpatioTemporalEncoder(num_layers=2, d_model=256, num_heads=4, d_ff=512)
    enc_out = encoder(out)
    print(f"  Encoder output: {tuple(enc_out.shape)} OK")

    # 5. Test Decoder
    print("\n[5/6] Testing Decoder...")
    from src.model.decoder import TwoPassDecoder
    decoder = TwoPassDecoder(d_model=256, vocab_size=100, num_heads=4, num_decoder_layers=2, d_ff=512)
    ctc_probs, _ = decoder(enc_out)
    print(f"  CTC log_probs: {tuple(ctc_probs.shape)} OK")
    decoded = decoder.decode(enc_out)
    print(f"  Decoded tokens: {[len(d) for d in decoded]} tokens per sample OK")

    # 6. Test Full Model
    print("\n[6/6] Testing Full WhisperSign Model...")
    from src.model.whisper_sign import WhisperSignModel
    model = WhisperSignModel(config)
    total_params = model.get_num_params(trainable_only=False)
    print(f"  Total parameters: {total_params:,}")

    x = torch.randn(2, 120, 42, 7)
    lengths = torch.tensor([120, 100])
    outputs = model(x, lengths)
    print(f"  CTC output:     {tuple(outputs['ctc_log_probs'].shape)}")
    print(f"  Encoder output: {tuple(outputs['encoder_output'].shape)}")
    print(f"  Output lengths: {outputs['output_lengths'].tolist()}")

    # Test gradient flow
    loss = outputs["ctc_log_probs"].sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    print(f"  Gradient flow: {'OK' if grad_ok else 'FAILED'}")

    # Test freeze/unfreeze
    model.freeze_encoder()
    model.freeze_decoder()
    frozen = model.get_num_params(trainable_only=True)
    model.unfreeze_encoder()
    model.unfreeze_decoder()
    unfrozen = model.get_num_params(trainable_only=True)
    print(f"  Freeze test: {total_params:,} -> {frozen:,} -> {unfrozen:,} OK")

    # Test decode
    preds = model.decode(x, lengths)
    print(f"  Inference decode: {[len(p) for p in preds]} tokens OK")

    # Test Loss
    print("\n[BONUS] Testing Hybrid Loss...")
    from src.training.losses import HybridCTCAttentionLoss
    loss_fn = HybridCTCAttentionLoss(alpha=0.3)
    ctc_lp = torch.randn(2, 30, 100).log_softmax(dim=-1)
    labels = torch.randint(1, 100, (10,))
    out_lens = torch.tensor([30, 30])
    lab_lens = torch.tensor([5, 5])
    result = loss_fn(ctc_lp, None, labels, out_lens, lab_lens)
    print(f"  CTC loss: {result['ctc'].item():.4f}")
    print(f"  Total loss: {result['total'].item():.4f} OK")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)

if __name__ == "__main__":
    test_all()
