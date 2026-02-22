"""
WhisperSign - Deep Functional Inspection
============================================
Tests that go BEYOND shape/gradient checks to verify the model
actually FUNCTIONS correctly as a sign language recognizer.

Sections:
  F1. CTC Decoding Logic (blank removal, dedup, known patterns)
  F2. Attention Causal Mask (no future information leakage)
  F3. Encoder Masking (padded positions ignored)
  F4. RPE Correctness (relative positions computed correctly)
  F5. Deterministic Inference (same input = same output)
  F6. Training Convergence on Synthetic Data (memorization test)
  F7. Two-Pass Decoder Interaction (CTC + Attention agreement)
  F8. Full End-to-End Pipeline (raw numpy -> prediction)
  F9. Attention Weight Distribution (not degenerate)
  F10. Gradient Magnitude Analysis (no vanishing/exploding)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

PASS = 0
FAIL = 0

def report(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
    else:
        FAIL += 1
    pad = 62 - len(name)
    tag = "[OK]" if passed else "[!!]"
    suffix = f" ({detail})" if detail else ""
    print(f"  {tag} {name}{'.' * max(pad, 2)} {'PASS' if passed else 'FAIL'}{suffix}")


def run_all():
    print("=" * 72)
    print("  WhisperSign - Deep Functional Inspection")
    print("=" * 72)

    config = {
        "frontend": {
            "num_joints": 42, "num_features": 7,
            "patch_size": 4, "d_model": 128,
            "dropout": 0.0, "spatial_dropout": 0.0,
        },
        "encoder": {
            "num_heads": 4, "num_layers": 2,
            "d_model": 128, "d_ff": 256, "dropout": 0.0,
        },
        "decoder": {"vocab_size": 20, "blank_id": 0},
    }

    # =============================================
    # F1. CTC Decoding Logic
    # =============================================
    print("\n--- F1: CTC Decoding Logic ---")
    from src.model.decoder import CTCDecoder

    ctc = CTCDecoder(d_model=128, vocab_size=20)

    # Test 1a: Handcrafted log-probs with known correct decode
    # Sequence: blank, 3, 3, blank, 5, blank, 7, 7, 7, blank
    # Expected after dedup+blank removal: [3, 5, 7]
    T = 10
    log_probs = torch.full((1, T, 20), -100.0)  # Very low prob for all
    pattern = [0, 3, 3, 0, 5, 0, 7, 7, 7, 0]
    for t, token in enumerate(pattern):
        log_probs[0, t, token] = 0.0  # log(1) = 0

    log_probs = log_probs.log_softmax(dim=-1)
    decoded = ctc.greedy_decode(log_probs)
    report("CTC blank+dedup removal", decoded[0] == [3, 5, 7], str(decoded[0]))

    # Test 1b: All blanks -> empty sequence
    all_blank = torch.full((1, 10, 20), -100.0)
    all_blank[0, :, 0] = 0.0
    all_blank = all_blank.log_softmax(dim=-1)
    decoded_blank = ctc.greedy_decode(all_blank)
    report("All blanks -> empty", decoded_blank[0] == [], str(decoded_blank[0]))

    # Test 1c: Single repeated token -> single token
    single_tok = torch.full((1, 10, 20), -100.0)
    single_tok[0, :, 5] = 0.0
    single_tok = single_tok.log_softmax(dim=-1)
    decoded_single = ctc.greedy_decode(single_tok)
    report("Repeated single token -> [5]", decoded_single[0] == [5], str(decoded_single[0]))

    # Test 1d: Alternating same token with blanks -> multiple of same token
    # Pattern: 3, blank, 3, blank, 3 -> [3, 3, 3]
    alt = torch.full((1, 5, 20), -100.0)
    alt_pattern = [3, 0, 3, 0, 3]
    for t, tok in enumerate(alt_pattern):
        alt[0, t, tok] = 0.0
    alt = alt.log_softmax(dim=-1)
    decoded_alt = ctc.greedy_decode(alt)
    report("3,blank,3,blank,3 -> [3,3,3]", decoded_alt[0] == [3, 3, 3], str(decoded_alt[0]))

    # =============================================
    # F2. Attention Causal Mask (No Future Leakage)
    # =============================================
    print("\n--- F2: Attention Causal Mask ---")
    from src.model.decoder import AttentionDecoder

    att_dec = AttentionDecoder(vocab_size=20, d_model=128, num_heads=4, num_layers=2, d_ff=256)
    att_dec.eval()

    # If causal mask works, changing future tokens should NOT affect past outputs
    enc_out = torch.randn(1, 20, 128)
    tgt_a = torch.tensor([[1, 2, 3, 4, 5]])
    tgt_b = torch.tensor([[1, 2, 3, 9, 9]])  # Changed tokens 4,5 -> 9,9
    mask = torch.ones(1, 20, dtype=torch.bool)

    with torch.no_grad():
        out_a = att_dec(enc_out, tgt_a, mask)
        out_b = att_dec(enc_out, tgt_b, mask)

    # Positions 0,1,2 should be identical (causal mask prevents seeing 3,4)
    pos0_match = torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-5)
    pos1_match = torch.allclose(out_a[0, 1], out_b[0, 1], atol=1e-5)
    pos2_match = torch.allclose(out_a[0, 2], out_b[0, 2], atol=1e-5)
    # Position 3 should differ (it can see position 3 which changed)
    pos3_differs = not torch.allclose(out_a[0, 3], out_b[0, 3], atol=1e-3)

    report("Causal: position 0 unaffected", pos0_match)
    report("Causal: position 1 unaffected", pos1_match)
    report("Causal: position 2 unaffected", pos2_match)
    report("Causal: position 3 changes", pos3_differs)

    # =============================================
    # F3. Encoder Masking (Padded Positions Ignored)
    # =============================================
    print("\n--- F3: Encoder Masking ---")
    from src.model.whisper_sign import WhisperSignModel

    model = WhisperSignModel(config)
    model.eval()

    # Two inputs: same content but different padding
    base = torch.randn(1, 60, 42, 7)
    # input A: 60 valid frames
    # input B: same 60 frames + 60 garbage frames, but length=60
    padded = torch.cat([base, torch.randn(1, 60, 42, 7) * 999], dim=1)

    with torch.no_grad():
        out_a = model(base, torch.tensor([60]))
        out_b = model(padded, torch.tensor([60]))

    # Output lengths should match
    report("Mask: output lengths match", out_a["output_lengths"].item() == out_b["output_lengths"].item())

    # CTC outputs for valid region should be similar
    valid_len = out_a["output_lengths"].item()
    ctc_a = out_a["ctc_log_probs"][0, :valid_len]
    ctc_b = out_b["ctc_log_probs"][0, :valid_len]
    similarity = torch.cosine_similarity(ctc_a.flatten(), ctc_b.flatten(), dim=0)
    # Not exact due to BatchNorm stats, but should be correlated
    report("Mask: valid region outputs correlated", similarity > 0.5, f"cosine={similarity:.4f}")

    # =============================================
    # F4. RPE Correctness
    # =============================================
    print("\n--- F4: Relative Positional Encoding ---")
    from src.model.positional import RelativePositionalEncoding

    rpe = RelativePositionalEncoding(d_model=64, max_len=100)

    # Test symmetry: RPE(i, j) structure should be consistent
    pe_10 = rpe(10)  # (10, 10, 64)
    report("RPE output shape", pe_10.shape == (10, 10, 64))

    # Diagonal should all be the same (distance=0)
    diag_vals = pe_10[torch.arange(10), torch.arange(10)]  # (10, 64)
    report("RPE diagonal consistent (dist=0)", torch.allclose(diag_vals[0], diag_vals[5], atol=1e-6))

    # Symmetric distances should have same encoding
    # RPE[i, j] should equal RPE[i+1, j+1] (same relative distance)
    report("RPE shift invariance", torch.allclose(pe_10[0, 3], pe_10[1, 4], atol=1e-6))
    report("RPE shift invariance (2)", torch.allclose(pe_10[2, 5], pe_10[4, 7], atol=1e-6))

    # Different distances should differ
    report("RPE different distances differ", not torch.allclose(pe_10[0, 0], pe_10[0, 5], atol=1e-3))

    # =============================================
    # F5. Deterministic Inference
    # =============================================
    print("\n--- F5: Deterministic Inference ---")
    model_det = WhisperSignModel(config)
    model_det.eval()

    x_det = torch.randn(2, 80, 42, 7)
    lens_det = torch.tensor([80, 60])

    with torch.no_grad():
        out1 = model_det(x_det, lens_det)
        out2 = model_det(x_det, lens_det)

    report("Deterministic: CTC output identical", torch.allclose(out1["ctc_log_probs"], out2["ctc_log_probs"], atol=1e-6))
    report("Deterministic: encoder output identical", torch.allclose(out1["encoder_output"], out2["encoder_output"], atol=1e-6))

    dec1 = model_det.decode(x_det, lens_det)
    dec2 = model_det.decode(x_det, lens_det)
    report("Deterministic: decode results identical", dec1 == dec2)

    # =============================================
    # F6. Training Convergence (Memorization Test)
    # =============================================
    print("\n--- F6: Training Convergence (Memorization Test) ---")
    from src.training.losses import HybridCTCAttentionLoss

    # Create a tiny model that should memorize 4 samples
    tiny_config = {
        "frontend": {
            "num_joints": 42, "num_features": 7,
            "patch_size": 4, "d_model": 128,
            "dropout": 0.0, "spatial_dropout": 0.0,
        },
        "encoder": {
            "num_heads": 4, "num_layers": 2,
            "d_model": 128, "d_ff": 256, "dropout": 0.0,
        },
        "decoder": {"vocab_size": 10, "blank_id": 0},
    }

    model_mem = WhisperSignModel(tiny_config)
    model_mem.train()
    optimizer = torch.optim.Adam(model_mem.parameters(), lr=5e-4)
    loss_fn = HybridCTCAttentionLoss(alpha=1.0, blank_id=0)

    # Fixed training data: 4 samples, each with 2 label tokens
    torch.manual_seed(42)
    train_x = torch.randn(4, 40, 42, 7)  # 4 samples, 40 frames
    train_lengths = torch.tensor([40, 40, 40, 40])
    train_labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])  # 2 labels each
    train_label_lens = torch.tensor([2, 2, 2, 2])

    losses_mem = []
    for step in range(80):
        optimizer.zero_grad()
        outputs = model_mem(train_x, train_lengths)
        loss_dict = loss_fn(
            outputs["ctc_log_probs"], None,
            train_labels, outputs["output_lengths"], train_label_lens,
        )
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model_mem.parameters(), 1.0)
        optimizer.step()
        losses_mem.append(loss_dict["total"].item())

    initial_loss = np.mean(losses_mem[:5])
    final_loss = np.mean(losses_mem[-5:])
    report("Memorization: loss decreases significantly", final_loss < initial_loss * 0.3,
           f"{initial_loss:.3f} -> {final_loss:.3f}")
    report("Memorization: final loss < 1.0", final_loss < 1.0, f"final={final_loss:.4f}")

    # After memorization, decode should produce non-empty sequences
    model_mem.eval()
    with torch.no_grad():
        decoded_mem = model_mem.decode(train_x, train_lengths)
    nonempty = sum(1 for d in decoded_mem if len(d) > 0)
    report("Memorization: decoded non-empty results", nonempty >= 3, f"{nonempty}/4 non-empty")

    # =============================================
    # F7. Two-Pass Decoder Interaction
    # =============================================
    print("\n--- F7: Two-Pass Decoder ---")
    from src.model.decoder import TwoPassDecoder

    tp = TwoPassDecoder(d_model=128, vocab_size=20, num_heads=4, num_decoder_layers=2, d_ff=256)
    tp.eval()

    enc_out_tp = torch.randn(2, 20, 128)

    # CTC-only (no target tokens)
    with torch.no_grad():
        ctc_only, att_only = tp(enc_out_tp)
    report("Two-pass: CTC works without targets", ctc_only is not None and ctc_only.shape == (2, 20, 20))
    report("Two-pass: Attention is None without targets", att_only is None)

    # Both passes (with target tokens)
    tgt_tp = torch.randint(0, 20, (2, 8))
    with torch.no_grad():
        ctc_both, att_both = tp(enc_out_tp, tgt_tp)
    report("Two-pass: CTC shape with targets", ctc_both.shape == (2, 20, 20))
    report("Two-pass: Attention shape with targets", att_both.shape == (2, 8, 20))

    # CTC output should be the same regardless of target presence
    report("Two-pass: CTC invariant to targets", torch.allclose(ctc_only, ctc_both, atol=1e-6))

    # =============================================
    # F8. Full End-to-End Pipeline
    # =============================================
    print("\n--- F8: End-to-End Pipeline (numpy -> prediction) ---")
    from src.data.preprocessing import preprocess_sequence
    from src.data.normalization import SpatialNormalizer, ScaleNormalizer

    # Simulate raw Leap Motion data
    np.random.seed(42)
    raw_data = np.random.randn(200, 42, 7).astype(np.float32)

    # Step 1: Normalize
    spatial_norm = SpatialNormalizer()
    scale_norm = ScaleNormalizer()
    normed = spatial_norm.normalize(raw_data)
    normed = scale_norm.normalize(normed)
    report("E2E: normalization produces valid data", not np.any(np.isnan(normed)))

    # Step 2: Preprocess (pad to model input size)
    processed = preprocess_sequence(normed, max_seq_length=400)
    report("E2E: preprocessing shape correct", processed.shape == (400, 42, 7))

    # Step 3: To tensor and run model
    x_e2e = torch.from_numpy(processed).float().unsqueeze(0)  # (1, 400, 42, 7)
    lengths_e2e = torch.tensor([200])  # Original length before padding

    model_e2e = WhisperSignModel(config)
    model_e2e.eval()
    with torch.no_grad():
        out_e2e = model_e2e(x_e2e, lengths_e2e)

    report("E2E: model output is finite", torch.isfinite(out_e2e["ctc_log_probs"]).all())
    report("E2E: output length correct", out_e2e["output_lengths"].item() == 50, 
           f"200/4={out_e2e['output_lengths'].item()}")

    predictions_e2e = model_e2e.decode(x_e2e, lengths_e2e)
    report("E2E: decode returns list", isinstance(predictions_e2e, list) and len(predictions_e2e) == 1)

    # Step 4: Sliding window inference
    from src.utils.sliding_window import SlidingWindowInference

    slider = SlidingWindowInference(model_e2e, window_duration=1.0, overlap=0.5, sample_rate=60, device="cpu")
    sw_preds = slider(normed)
    report("E2E: sliding window returns list", isinstance(sw_preds, list))

    # =============================================
    # F9. Attention Weight Analysis
    # =============================================
    print("\n--- F9: Attention Weight Distribution ---")
    from src.model.encoder import SpatialAttention, TemporalAttention

    # Hook into attention to capture weights
    sa = SpatialAttention(d_model=128, num_heads=4)
    sa.eval()

    # Manually compute attention weights
    x_attn = torch.randn(1, 20, 128)
    with torch.no_grad():
        B, T, D = x_attn.shape
        qkv = sa.qkv(x_attn).reshape(B, T, 3, sa.num_heads, sa.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_weights = (q @ k.transpose(-2, -1)) * sa.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

    # Check attention properties
    report("Attn: weights sum to 1", torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-5))
    report("Attn: weights all >= 0", (attn_weights >= 0).all())
    report("Attn: weights all <= 1", (attn_weights <= 1 + 1e-6).all())

    # Check it's not degenerate (not all attention on one position)
    entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1).mean()
    max_entropy = np.log(20)
    report("Attn: not degenerate (entropy > 0)", entropy.item() > 0.5, f"entropy={entropy:.3f}/{max_entropy:.3f}")

    # Different heads should attend differently
    head_0 = attn_weights[0, 0]  # (20, 20)
    head_1 = attn_weights[0, 1]
    heads_differ = not torch.allclose(head_0, head_1, atol=0.1)
    report("Attn: different heads differ", heads_differ)

    # =============================================
    # F10. Gradient Magnitude Analysis
    # =============================================
    print("\n--- F10: Gradient Magnitude Analysis ---")
    model_ga = WhisperSignModel(config)
    model_ga.train()

    x_ga = torch.randn(2, 80, 42, 7)
    lens_ga = torch.tensor([80, 80])
    labels_ga = torch.randint(1, 20, (6,))
    lab_lens_ga = torch.tensor([3, 3])

    out_ga = model_ga(x_ga, lens_ga)
    loss_fn_ga = HybridCTCAttentionLoss(alpha=1.0, blank_id=0)
    loss_ga = loss_fn_ga(out_ga["ctc_log_probs"], None, labels_ga, out_ga["output_lengths"], lab_lens_ga)
    loss_ga["total"].backward()

    # Collect gradient stats per component
    components = {
        "Frontend": model_ga.frontend,
        "Encoder": model_ga.encoder,
        "CTC Decoder": model_ga.decoder.ctc_decoder,
    }

    for name, module in components.items():
        grads = [p.grad.abs().mean().item() for p in module.parameters() if p.grad is not None]
        if grads:
            mean_grad = np.mean(grads)
            max_grad = np.max(grads)
            min_grad = np.min(grads)

            # Check for vanishing (mean < 1e-8) or exploding (max > 100)
            not_vanishing = mean_grad > 1e-8
            not_exploding = max_grad < 100
            report(f"Grad {name}: not vanishing", not_vanishing,
                   f"mean={mean_grad:.2e}")
            report(f"Grad {name}: not exploding", not_exploding,
                   f"max={max_grad:.2e}")

    # Check gradient ratios between layers (should not be extreme)
    frontend_grad = np.mean([p.grad.abs().mean().item() for p in model_ga.frontend.parameters() if p.grad is not None])
    encoder_grad = np.mean([p.grad.abs().mean().item() for p in model_ga.encoder.parameters() if p.grad is not None])
    ratio = max(frontend_grad, encoder_grad) / (min(frontend_grad, encoder_grad) + 1e-12)
    report("Grad ratio frontend/encoder reasonable", ratio < 100, f"ratio={ratio:.1f}x")

    # =============================================
    # SUMMARY
    # =============================================
    print(f"\n{'=' * 72}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'=' * 72}")

    if FAIL == 0:
        print("  >>> ALL FUNCTIONAL TESTS PASSED <<<")
    else:
        print("  >>> SOME TESTS FAILED <<<")

    return FAIL == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
