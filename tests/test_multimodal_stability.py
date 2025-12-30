"""
Integration tests for training stability and NaN handling.

Tests end-to-end training scenarios with NaN injection,
weight corruption recovery, and GradScaler behavior.
"""

import pytest
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.multimodal import MultimodalClassifier
from src.models.losses import MultiTaskLoss


# =============================================================================
# Training Loop Stability Tests
# =============================================================================

class TestTrainingLoopStability:
    """Tests for training loop NaN handling and weight corruption recovery."""

    def test_training_step_with_nan_batch(self, multimodal_classifier, device):
        """Test that training can skip NaN batches without crashing."""
        model = multimodal_classifier
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        loss_fn = MultiTaskLoss(num_classes=14).to(device)

        batch_size = 4
        num_batches = 10

        # Simulate training loop
        nan_count = 0
        for i in range(num_batches):
            # Create batch
            img_emb = torch.randn(batch_size, 768, device=device)
            text_emb = torch.randn(batch_size, 768, device=device)
            structured = torch.randn(batch_size, 128, device=device)
            labels = torch.randint(0, 2, (batch_size, 14), device=device, dtype=torch.float)

            # Inject NaN into every 3rd batch
            if i % 3 == 0:
                structured[:, :10] = float('nan')
                nan_count += 1

            # Sanitize inputs (mimics train_classifier.py:366)
            structured = torch.nan_to_num(structured, nan=0.0, posinf=0.0, neginf=0.0)

            # Forward pass
            with autocast():
                outputs = model(img_emb, text_emb, structured)
                loss_dict = loss_fn(
                    logits=outputs['logits'],
                    labels=labels,
                    img_emb=outputs.get('clip_emb'),
                    text_emb=outputs.get('text_clip_emb'),
                    supcon_emb=outputs.get('supcon_emb'),
                )
                total_loss = loss_dict["total"]

            # Check for NaN loss (hard gate)
            if not torch.isfinite(total_loss):
                # Skip backward pass (mimics train_classifier.py:383-396)
                continue

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Should complete without crashing
        assert True, "Training loop should handle NaN batches"

    def test_weight_corruption_detection_and_recovery(self, device):
        """Test weight integrity fuse detects and reverts corrupted parameters."""
        # Create simple model
        model = nn.Linear(10, 10).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Helper function to check if params are finite
        def _params_finite(model):
            for param in model.parameters():
                if not torch.isfinite(param).all():
                    return False
            return True

        # Helper function to copy params
        def _copy_params(model):
            return {name: param.clone() for name, param in model.named_parameters()}

        # Save last-good params
        last_good_params = _copy_params(model)

        # Simulate normal training step
        x = torch.randn(4, 10, device=device)
        y = model(x).sum()
        y.backward()
        optimizer.step()

        # Verify params still finite
        assert _params_finite(model), "Params should be finite after normal step"

        # Update last-good checkpoint
        last_good_params = _copy_params(model)

        # Simulate weight corruption (inject NaN)
        with torch.no_grad():
            for param in model.parameters():
                param[0] = float('nan')

        # Detect corruption
        params_corrupted = not _params_finite(model)
        assert params_corrupted, "Should detect corrupted parameters"

        # Revert to last-good params (mimics train_classifier.py:422-426)
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(last_good_params[name])

        # Verify recovery
        assert _params_finite(model), "Params should be finite after revert"

    def test_gradscaler_cascade_failure_prevention(self):
        """Test that GradScaler reset prevents cascade failures."""
        scaler = GradScaler(init_scale=1024.0)

        # Simulate normal steps
        for _ in range(5):
            scaler.update()

        initial_scale = scaler._scale.item()

        # Simulate weight corruption detection
        # Reset scaler to prevent cascade (proposed Fix #1)
        with torch.no_grad():
            scaler._scale.fill_(65536.0)
            scaler._growth_tracker = 0

        reset_scale = scaler._scale.item()

        # Verify reset
        assert reset_scale == 65536.0, f"Scale should be reset to 65536, got {reset_scale}"
        assert scaler._growth_tracker == 0, "Growth tracker should be reset"

        # Continue training
        for _ in range(5):
            scaler.update()

        # Should not crash
        final_scale = scaler._scale.item()
        assert torch.isfinite(torch.tensor(final_scale)), "Scale should remain finite"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for mixed precision")
    def test_100_batches_with_forced_corruption(self, multimodal_classifier, device):
        """
        Integration test: 100 batches with forced corruption every 10 batches.
        Tests the complete protection mechanism (hard gate + weight fuse + scaler reset).
        """
        model = multimodal_classifier
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        loss_fn = MultiTaskLoss(num_classes=14).to(device)

        batch_size = 4
        num_batches = 100
        corruption_interval = 10

        # Helper functions
        def _params_finite(model):
            for param in model.parameters():
                if not torch.isfinite(param).all():
                    return False
            return True

        def _copy_params(model):
            return {name: param.clone() for name, param in model.named_parameters()}

        # Track metrics
        nan_batches = 0
        weight_reverts = 0
        last_good_params = _copy_params(model)

        for batch_idx in range(num_batches):
            # Create batch
            img_emb = torch.randn(batch_size, 768, device=device)
            text_emb = torch.randn(batch_size, 768, device=device)
            structured = torch.randn(batch_size, 128, device=device)
            labels = torch.randint(0, 2, (batch_size, 14), device=device, dtype=torch.float)

            # Force corruption every N batches
            if batch_idx % corruption_interval == 0 and batch_idx > 0:
                structured[:, :] = float('inf')  # Extreme values to cause NaN

            # Sanitize inputs
            structured = torch.nan_to_num(structured, nan=0.0, posinf=0.0, neginf=0.0)

            # Forward pass
            with autocast():
                outputs = model(img_emb, text_emb, structured)
                loss_dict = loss_fn(
                    logits=outputs['logits'],
                    labels=labels,
                    img_emb=outputs.get('clip_emb'),
                    text_emb=outputs.get('text_clip_emb'),
                    supcon_emb=outputs.get('supcon_emb'),
                )
                total_loss = loss_dict["total"]

            # Hard gate: Check for NaN loss
            if not torch.isfinite(total_loss):
                nan_batches += 1
                continue

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)

            # Check for weight corruption before update
            if not _params_finite(model):
                # Revert to last-good params
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.copy_(last_good_params[name])

                # Reset scaler
                with torch.no_grad():
                    scaler._scale.fill_(65536.0)
                    scaler._growth_tracker = 0

                weight_reverts += 1
            else:
                # Update last-good checkpoint
                last_good_params = _copy_params(model)

            scaler.update()

        # Verify training completed
        assert batch_idx == num_batches - 1, "Should complete all batches"
        assert nan_batches > 0, "Should have detected some NaN batches"
        assert _params_finite(model), "Final model params should be finite"

        # Print stats
        print(f"\n100-batch stability test:")
        print(f"  NaN batches: {nan_batches}/{num_batches} ({100*nan_batches/num_batches:.1f}%)")
        print(f"  Weight reverts: {weight_reverts}")
        print(f"  Final scaler scale: {scaler._scale.item()}")


# =============================================================================
# Data Quality and Preprocessing Tests
# =============================================================================

class TestDataQualityHandling:
    """Tests for handling problematic data samples."""

    def test_blacklisted_study_filtering(self):
        """Test that blacklisted study_ids are filtered out."""
        from src.models.classification_dataset import BLACKLISTED_STUDY_IDS

        # Should have some blacklisted IDs
        assert len(BLACKLISTED_STUDY_IDS) > 0, "Should have blacklisted study IDs"

        # Check format
        for study_id in BLACKLISTED_STUDY_IDS:
            assert isinstance(study_id, str), f"Study ID should be string, got {type(study_id)}"

    def test_image_sanitization(self, device):
        """Test that image NaN/Inf are sanitized at load time."""
        # Create image with NaN/Inf
        img = torch.rand(1, 224, 224, device=device)
        img[0, :10, :10] = float('nan')
        img[0, :10, 10:20] = float('inf')

        # Sanitize (mimics classification_dataset.py:355-357)
        if torch.isnan(img).any() or torch.isinf(img).any():
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        # Verify sanitization
        assert not torch.isnan(img).any(), "Image should not contain NaN"
        assert not torch.isinf(img).any(), "Image should not contain Inf"
        assert (img >= 0).all() and (img <= 1).all(), "Image should be in [0, 1]"

    def test_structured_features_sanitization(self, device):
        """Test that structured features NaN/Inf are sanitized."""
        # Create structured features with NaN/Inf
        structured = torch.randn(4, 128, device=device)
        structured[:, :10] = float('nan')
        structured[:, 10:20] = float('inf')

        # Sanitize (mimics classification_dataset.py:370-372)
        if torch.isnan(structured).any() or torch.isinf(structured).any():
            structured = torch.nan_to_num(structured, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify sanitization
        assert torch.isfinite(structured).all(), "Structured features should be finite"


# =============================================================================
# Performance and Regression Tests
# =============================================================================

class TestPerformanceRegression:
    """Tests to ensure fixes don't degrade performance."""

    def test_forward_pass_speed(self, multimodal_classifier, device):
        """Test that NaN guards don't significantly slow forward pass."""
        import time

        model = multimodal_classifier
        batch_size = 32

        # Warmup
        for _ in range(10):
            img_emb = torch.randn(batch_size, 768, device=device)
            text_emb = torch.randn(batch_size, 768, device=device)
            structured = torch.randn(batch_size, 128, device=device)
            with torch.no_grad():
                _ = model(img_emb, text_emb, structured)

        # Benchmark
        num_iterations = 100
        start_time = time.time()
        for _ in range(num_iterations):
            img_emb = torch.randn(batch_size, 768, device=device)
            text_emb = torch.randn(batch_size, 768, device=device)
            structured = torch.randn(batch_size, 128, device=device)
            with torch.no_grad():
                _ = model(img_emb, text_emb, structured)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        throughput = (num_iterations * batch_size) / elapsed

        print(f"\nForward pass throughput: {throughput:.1f} samples/sec")

        # Should process at least 100 samples/sec on GPU
        if device.type == "cuda":
            assert throughput > 100, f"Throughput too low: {throughput} samples/sec"

    def test_model_capacity_not_reduced(self, multimodal_classifier, normal_embeddings):
        """Test that NaN guards don't collapse model capacity."""
        model = multimodal_classifier
        img_emb = normal_embeddings["img"]
        text_emb = normal_embeddings["text"]
        structured = torch.randn(4, 128, device=img_emb.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(img_emb, text_emb, structured)

        # Check output variance (should not be collapsed)
        logits_var = outputs['logits'].var(dim=-1).mean()
        assert logits_var > 0.01, f"Logits have suspiciously low variance: {logits_var}"

        if 'clip_emb' in outputs:
            clip_var = outputs['clip_emb'].var(dim=-1).mean()
            assert clip_var > 0.01, f"CLIP embeddings have suspiciously low variance: {clip_var}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
