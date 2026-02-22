"""
Deep Technical Verification for WhisperSign
Tests: tensor shapes, gradient flow, loss convergence, checkpoint round-trip,
       training step simulation, edge cases, memory, numerical stability
"""
import sys, os, time, tempfile, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

PASS = 0
FAIL = 0

def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    pad = 60 - len(name)
    print(f"  {'[OK]' if passed else '[!!]'} {name}{'.' * max(pad, 2)} {status}" + (f" ({detail})" if detail else ""))


def run_all():
    print("=" * 70)
    print("  WhisperSign - Deep Technical Verification Suite")
    print("=" * 70)

    config = {
        "frontend": {
            "num_joints": 42, "num_features": 7,
            "patch_size": 4, "d_model": 256,
            "dropout": 0.0, "spatial_dropout": 0.0,  # Disable for deterministic tests
        },
        "encoder": {
            "num_heads": 4, "num_layers": 2,
            "d_model": 256, "d_ff": 512, "dropout": 0.0,
        },
        "decoder": {"vocab_size": 100, "blank_id": 0},
    }

    # =============================================
    # SECTION 1: Tensor Shape Propagation
    # =============================================
    print("\n--- SECTION 1: Tensor Shape Propagation ---")

    from src.model.frontend import TemporalPatchEmbedding, ConvSPE, SignLanguageFrontend
    from src.model.encoder import SpatialAttention, TemporalAttention, SpatioTemporalBlock, SpatioTemporalEncoder
    from src.model.decoder import CTCDecoder, AttentionDecoder, TwoPassDecoder
    from src.model.whisper_sign import WhisperSignModel

    B, T, J, F = 2, 120, 42, 7
    d_model = 256

    # 1a. Patch Embedding with exact divisibility
    pe = TemporalPatchEmbedding(num_joints=42, num_features=7, patch_size=4, d_model=d_model)
    x = torch.randn(B, T, J, F)
    out = pe(x)
    report("PatchEmbed exact div (120/4=30)", out.shape == (B, 30, d_model), str(out.shape))

    # 1b. Patch Embedding with non-divisible T (should auto-pad)
    x_odd = torch.randn(B, 123, J, F)
    out_odd = pe(x_odd)
    expected_T = 124 // 4  # padded to 124
    report("PatchEmbed non-div pad (123->124/4=31)", out_odd.shape == (B, expected_T, d_model), str(out_odd.shape))

    # 1c. ConvSPE preserves shape
    cspe = ConvSPE(d_model=d_model)
    x_spe = torch.randn(B, 30, d_model)
    out_spe = cspe(x_spe)
    report("ConvSPE shape preservation", out_spe.shape == x_spe.shape)

    # 1d. ConvSPE is residual
    cspe.eval()
    x_test = torch.randn(B, 30, d_model)
    out_res = cspe(x_test)
    # Should not be identical (conv adds something) but same shape
    report("ConvSPE residual connection", not torch.allclose(out_res, x_test, atol=1e-3))

    # 1e. Full Frontend shape
    frontend = SignLanguageFrontend(num_joints=42, num_features=7, patch_size=4, d_model=d_model, dropout=0.0, spatial_dropout=0.0)
    frontend.eval()
    x_in = torch.randn(B, T, J, F)
    fe_out, fe_lens = frontend(x_in, torch.tensor([120, 80]))
    report("Frontend output shape", fe_out.shape == (B, 30, d_model), str(fe_out.shape))
    report("Frontend output lengths", fe_lens is not None and fe_lens.tolist() == [30, 20], str(fe_lens.tolist()))

    # 1f. Spatial Attention
    sa = SpatialAttention(d_model, num_heads=4)
    sa_out = sa(torch.randn(B, 30, d_model))
    report("SpatialAttention shape", sa_out.shape == (B, 30, d_model))

    # 1g. Temporal Attention
    ta = TemporalAttention(d_model, num_heads=4)
    ta_out = ta(torch.randn(B, 30, d_model))
    report("TemporalAttention shape", ta_out.shape == (B, 30, d_model))

    # 1h. SpatioTemporalBlock
    stb = SpatioTemporalBlock(d_model=d_model, num_heads=4, d_ff=512, dropout=0.0)
    stb_out = stb(torch.randn(B, 30, d_model))
    report("SpatioTemporalBlock shape", stb_out.shape == (B, 30, d_model))

    # 1i. Full Encoder
    enc = SpatioTemporalEncoder(num_layers=2, d_model=d_model, num_heads=4, d_ff=512, dropout=0.0)
    enc_out = enc(torch.randn(B, 30, d_model))
    report("Encoder shape", enc_out.shape == (B, 30, d_model))

    # 1j. CTC Decoder
    ctc = CTCDecoder(d_model=d_model, vocab_size=100)
    ctc_out = ctc(enc_out)
    report("CTCDecoder shape", ctc_out.shape == (B, 30, 100))

    # 1k. Attention Decoder
    ad = AttentionDecoder(vocab_size=100, d_model=d_model, num_heads=4, num_layers=2, d_ff=512)
    tgt = torch.randint(0, 100, (B, 10))
    mask = torch.ones(B, 30, dtype=torch.bool)
    ad_out = ad(enc_out, tgt, mask)
    report("AttentionDecoder shape", ad_out.shape == (B, 10, 100))

    # 1l. Full model end-to-end
    model = WhisperSignModel(config)
    model.eval()
    full_out = model(torch.randn(B, T, J, F), torch.tensor([T, 80]))
    report("Full model CTC shape", full_out["ctc_log_probs"].shape[:2] == (B, 30))
    report("Full model encoder shape", full_out["encoder_output"].shape == (B, 30, d_model))

    # =============================================
    # SECTION 2: Gradient Flow & Backward Pass
    # =============================================
    print("\n--- SECTION 2: Gradient Flow & Backward Pass ---")

    model_grad = WhisperSignModel(config)
    model_grad.train()
    x_g = torch.randn(B, T, J, F, requires_grad=True)
    out_g = model_grad(x_g, torch.tensor([T, T]))

    # 2a. CTC loss backward through full model
    loss_ctc = out_g["ctc_log_probs"].sum()
    loss_ctc.backward()

    frontend_grads = sum(1 for p in model_grad.frontend.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    encoder_grads = sum(1 for p in model_grad.encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    ctc_grads = sum(1 for p in model_grad.decoder.ctc_decoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)

    report("Gradients reach Frontend", frontend_grads > 0, f"{frontend_grads} params with grad")
    report("Gradients reach Encoder", encoder_grads > 0, f"{encoder_grads} params with grad")
    report("Gradients reach CTC Decoder", ctc_grads > 0, f"{ctc_grads} params with grad")
    report("Input tensor has gradient", x_g.grad is not None and x_g.grad.abs().sum() > 0)

    # 2b. No NaN in gradients
    all_finite = all(
        torch.isfinite(p.grad).all() for p in model_grad.parameters()
        if p.grad is not None
    )
    report("All gradients are finite (no NaN/Inf)", all_finite)

    # 2c. Freeze encoder - no gradients
    model_freeze = WhisperSignModel(config)
    model_freeze.freeze_encoder()
    x_f = torch.randn(B, T, J, F)
    out_f = model_freeze(x_f, torch.tensor([T, T]))
    out_f["ctc_log_probs"].sum().backward()

    enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_freeze.encoder.parameters())
    fe_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_freeze.frontend.parameters())
    report("Frozen encoder: no gradients", not enc_has_grad)
    report("Frozen encoder: frontend still has grads", fe_has_grad)

    # =============================================
    # SECTION 3: CTC Loss Computation
    # =============================================
    print("\n--- SECTION 3: Loss Computation ---")

    from src.training.losses import HybridCTCAttentionLoss

    # 3a. CTC-only loss
    loss_fn = HybridCTCAttentionLoss(alpha=1.0, blank_id=0)
    ctc_lp = torch.randn(B, 30, 100).log_softmax(dim=-1)
    labels = torch.randint(1, 100, (8,))  # Avoid blank
    out_lens = torch.tensor([30, 30])
    lab_lens = torch.tensor([4, 4])
    result = loss_fn(ctc_lp, None, labels, out_lens, lab_lens)
    report("CTC loss is positive", result["ctc"].item() > 0)
    report("CTC loss is finite", torch.isfinite(result["ctc"]))
    report("Total = CTC when alpha=1.0", torch.allclose(result["total"], result["ctc"]))

    # 3b. Hybrid loss
    loss_fn2 = HybridCTCAttentionLoss(alpha=0.3, blank_id=0)
    att_logits = torch.randn(B, 10, 100)
    att_targets = torch.randint(0, 100, (B, 10))
    result2 = loss_fn2(ctc_lp, att_logits, labels, out_lens, lab_lens, att_targets)
    report("Hybrid loss has all components", all(k in result2 for k in ["total", "ctc", "attention"]))
    expected_total = 0.3 * result2["ctc"] + 0.7 * result2["attention"]
    report("Hybrid loss formula L=0.3*CTC+0.7*ATT", torch.allclose(result2["total"], expected_total, atol=1e-5))

    # 3c. CTC loss backward through a linear layer
    linear = nn.Linear(256, 100)
    inp = torch.randn(B, 30, 256)
    ctc_lp_g = linear(inp).log_softmax(dim=-1)
    result3 = loss_fn(ctc_lp_g, None, labels, out_lens, lab_lens)
    result3["total"].backward()
    has_grad = any(p.grad is not None for p in linear.parameters())
    report("CTC loss backward succeeds", has_grad)

    # =============================================
    # SECTION 4: Checkpoint Round-Trip
    # =============================================
    print("\n--- SECTION 4: Checkpoint Round-Trip ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")

        model_save = WhisperSignModel(config)
        # Set specific weights
        with torch.no_grad():
            for p in model_save.parameters():
                p.fill_(0.42)

        model_save.save_checkpoint(ckpt_path, epoch=10, loss=1.234)

        # Load and compare
        model_load, ckpt_info = WhisperSignModel.load_checkpoint(ckpt_path)

        report("Checkpoint epoch preserved", ckpt_info["epoch"] == 10)
        report("Checkpoint loss preserved", abs(ckpt_info["loss"] - 1.234) < 1e-6)

        # Compare all weights
        weights_match = all(
            torch.allclose(p1, p2) for (_, p1), (_, p2)
            in zip(model_save.named_parameters(), model_load.named_parameters())
        )
        report("All weights identical after load", weights_match)

        # Verify loaded model produces same output
        model_save.eval()
        model_load.eval()
        x_ck = torch.randn(1, 60, 42, 7)
        out_save = model_save(x_ck, torch.tensor([60]))
        out_load = model_load(x_ck, torch.tensor([60]))
        report("Same output after checkpoint load", torch.allclose(
            out_save["ctc_log_probs"], out_load["ctc_log_probs"], atol=1e-5
        ))

    # =============================================
    # SECTION 5: Simulated Training Step
    # =============================================
    print("\n--- SECTION 5: Simulated Training Step ---")

    model_train = WhisperSignModel(config)
    model_train.train()
    optimizer = torch.optim.AdamW(model_train.parameters(), lr=1e-3)
    loss_fn_train = HybridCTCAttentionLoss(alpha=1.0, blank_id=0)

    x_t = torch.randn(B, T, J, F)
    lengths_t = torch.tensor([T, T])
    labels_t = torch.randint(1, 100, (8,))
    label_lens_t = torch.tensor([4, 4])

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        outputs = model_train(x_t, lengths_t)
        loss_dict = loss_fn_train(
            outputs["ctc_log_probs"], None, labels_t,
            outputs["output_lengths"], label_lens_t,
        )
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
        optimizer.step()
        losses.append(loss_dict["total"].item())

    report("5 training steps completed", len(losses) == 5)
    report("All training losses finite", all(np.isfinite(l) for l in losses))
    report("Loss decreased over 5 steps", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    # =============================================
    # SECTION 6: Edge Cases
    # =============================================
    print("\n--- SECTION 6: Edge Cases ---")

    model_edge = WhisperSignModel(config)
    model_edge.eval()

    # 6a. Batch size 1
    out_b1 = model_edge(torch.randn(1, 60, 42, 7), torch.tensor([60]))
    report("Batch size 1", out_b1["ctc_log_probs"].shape[0] == 1)

    # 6b. Minimum sequence length (1 patch = 4 frames)
    out_min = model_edge(torch.randn(1, 4, 42, 7), torch.tensor([4]))
    report("Minimum seq len (4 frames = 1 patch)", out_min["ctc_log_probs"].shape[1] == 1)

    # 6c. All-zero input
    out_zero = model_edge(torch.zeros(1, 120, 42, 7), torch.tensor([120]))
    report("All-zero input doesn't crash", out_zero["ctc_log_probs"].shape[0] == 1)
    report("All-zero output is finite", torch.isfinite(out_zero["ctc_log_probs"]).all())

    # 6d. Very long sequence
    try:
        out_long = model_edge(torch.randn(1, 600, 42, 7), torch.tensor([600]))
        report("Long sequence (600 frames)", out_long["ctc_log_probs"].shape == (1, 150, 100))
    except Exception as e:
        report("Long sequence (600 frames)", False, str(e))

    # 6e. Greedy decode returns valid tokens
    preds = model_edge.decode(torch.randn(B, T, J, F), torch.tensor([T, T]))
    all_valid = all(all(0 < t < 100 for t in seq) for seq in preds if seq)
    report("Decoded tokens in valid range", all_valid)

    # =============================================
    # SECTION 7: Normalization & Preprocessing
    # =============================================
    print("\n--- SECTION 7: Normalization & Preprocessing ---")

    from src.data.normalization import SpatialNormalizer, ScaleNormalizer, FeatureScaler
    from src.data.preprocessing import resample_to_fixed_rate, create_sliding_windows, compute_sequence_mask

    # 7a. Spatial normalization idempotent on palm
    sn = SpatialNormalizer()
    data = np.random.randn(100, 42, 7).astype(np.float32)
    normed = sn.normalize(data)
    normed2 = sn.normalize(normed)
    # After first norm, palms are at 0; second norm should not change much
    report("Spatial norm palm is zero", np.allclose(normed[:, 0, :3], 0, atol=1e-6))
    report("Spatial norm right palm zero", np.allclose(normed[:, 21, :3], 0, atol=1e-6))

    # 7b. Scale normalization doesn't produce NaN
    scn = ScaleNormalizer()
    scaled = scn.normalize(normed)
    report("Scale norm no NaN", not np.any(np.isnan(scaled)))
    report("Scale norm no Inf", not np.any(np.isinf(scaled)))

    # 7c. FeatureScaler round trip
    fs = FeatureScaler(method="standard")
    batch_data = np.random.randn(10, 50, 42, 7).astype(np.float32)
    scaled = fs.fit_transform(batch_data)
    # After standardization, mean should be ~0, std ~1
    flat = scaled.reshape(-1, 7)
    report("FeatureScaler mean ~0", np.allclose(flat.mean(axis=0), 0, atol=0.1))
    report("FeatureScaler std ~1", np.allclose(flat.std(axis=0), 1, atol=0.1))

    # 7d. Sliding windows
    data_sw = np.random.randn(300, 42, 7).astype(np.float32)
    windows = create_sliding_windows(data_sw, window_duration=1.0, overlap=0.5, sample_rate=60)
    report("Sliding windows created", len(windows) > 0, f"{len(windows)} windows")
    report("Window shape correct", all(w.shape == (60, 42, 7) for w in windows))

    # 7e. Resampling preserves endpoints
    ts_orig = np.linspace(0, 2, 100)
    data_rs = np.random.randn(100, 42, 7).astype(np.float32)
    resampled, new_ts = resample_to_fixed_rate(data_rs, ts_orig, 30)
    report("Resample preserves start time", abs(new_ts[0] - ts_orig[0]) < 1e-6)
    report("Resample preserves end time", abs(new_ts[-1] - ts_orig[-1]) < 1e-6)

    # 7f. Mask computation
    mask = compute_sequence_mask(50, 200)
    report("Mask sum equals seq_len", mask.sum() == 50)
    report("Mask valid region is True", mask[:50].all() and not mask[50:].any())

    # =============================================
    # SECTION 8: Augmentation
    # =============================================
    print("\n--- SECTION 8: Augmentation ---")

    from src.data.augmentation import GestureMasking, TemporalJitter, NoiseInjection, ComposeAugmentations

    data_aug = np.ones((100, 42, 7), dtype=np.float32)

    gm = GestureMasking(joint_mask_prob=1.0, temporal_mask_prob=1.0, max_temporal_mask=5)
    masked = gm(data_aug)
    report("GestureMasking masks joints", (masked == 0).any())

    tj = TemporalJitter(max_shift=3)
    jittered = tj(data_aug)
    report("TemporalJitter preserves shape", jittered.shape == data_aug.shape)

    ni = NoiseInjection(std=0.01)
    noised = ni(np.zeros_like(data_aug))
    report("NoiseInjection adds noise", not np.allclose(noised, 0))
    report("NoiseInjection magnitude reasonable", np.abs(noised).max() < 0.1)

    composed = ComposeAugmentations([gm, tj, ni])
    composed_out = composed(data_aug)
    report("ComposeAugmentations runs", composed_out.shape == data_aug.shape)

    # =============================================
    # SECTION 9: Numerical Stability
    # =============================================
    print("\n--- SECTION 9: Numerical Stability ---")

    model_ns = WhisperSignModel(config)
    model_ns.eval()

    # 9a. Large magnitude input
    x_large = torch.randn(1, 60, 42, 7) * 100
    out_large = model_ns(x_large, torch.tensor([60]))
    report("Large input (x100) no NaN", torch.isfinite(out_large["ctc_log_probs"]).all())

    # 9b. Small magnitude input
    x_small = torch.randn(1, 60, 42, 7) * 0.001
    out_small = model_ns(x_small, torch.tensor([60]))
    report("Small input (x0.001) no NaN", torch.isfinite(out_small["ctc_log_probs"]).all())

    # 9c. CTC log_probs are valid log probabilities
    probs = out_large["ctc_log_probs"].exp()
    sums = probs.sum(dim=-1)
    report("CTC outputs are valid log-probs", torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    # =============================================
    # SECTION 10: Scheduler
    # =============================================
    print("\n--- SECTION 10: Learning Rate Scheduler ---")

    from src.training.scheduler import CosineWarmupScheduler

    dummy_model = nn.Linear(10, 10)
    opt = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)
    sched = CosineWarmupScheduler(opt, warmup_steps=100, total_steps=1000, min_lr=1e-7)

    lrs = []
    for i in range(1000):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    report("LR starts near 0", lrs[0] < 1e-4)
    report("LR peaks around warmup end", max(lrs[90:110]) > 0.9e-3)
    report("LR decays after warmup", lrs[500] < lrs[100])
    report("LR above min_lr", min(lrs) >= 1e-7)

    # =============================================
    # SECTION 11: Parameter Count Consistency
    # =============================================
    print("\n--- SECTION 11: Parameter Counts ---")

    model_pc = WhisperSignModel(config)
    total = model_pc.get_num_params(trainable_only=False)
    trainable = model_pc.get_num_params(trainable_only=True)
    report("All params trainable initially", total == trainable)

    model_pc.freeze_encoder()
    frozen_trainable = model_pc.get_num_params(trainable_only=True)
    report("Freeze reduces trainable count", frozen_trainable < total)

    model_pc.freeze_decoder()
    double_frozen = model_pc.get_num_params(trainable_only=True)
    report("Double freeze further reduces", double_frozen < frozen_trainable)

    model_pc.unfreeze_encoder()
    model_pc.unfreeze_decoder()
    restored = model_pc.get_num_params(trainable_only=True)
    report("Unfreeze restores all params", restored == total)

    print(f"\n  Total params: {total:,}")
    print(f"  After freeze encoder: {frozen_trainable:,} (frontend+decoder)")
    print(f"  After freeze both: {double_frozen:,} (frontend only)")

    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 70)

    if FAIL == 0:
        print("  >>> ALL TESTS PASSED <<<")
    else:
        print("  >>> SOME TESTS FAILED <<<")

    return FAIL == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
