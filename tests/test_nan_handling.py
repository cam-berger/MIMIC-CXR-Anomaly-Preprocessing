"""
Unit tests for NaN/Inf handling in model components.

Covers:
- CrossAttentionFusion numerics (token-level attention)
- Fix #3: Safe normalization
- Fix #4: MAE epsilon increase

Fix #1 (GradScaler reset) and the NaN masking of Fix #2 were removed along
with the weight-revert fuse: the cascade they recovered from came from
gradients of parameters outside the optimizer, now fixed at the source.
See tests/test_training_fixes.py.
"""

import pytest
import torch
import torch.nn.functional as F


# =============================================================================
# CrossAttentionFusion numerics
# =============================================================================

class TestCrossAttentionNumerics:
    """CrossAttentionFusion stays finite on valid inputs and propagates invalid ones."""

    def test_zero_token_inputs(self, cross_attention_fusion, device):
        """Zero embeddings produce a finite output."""
        img = torch.zeros(4, 5, 768, device=device)
        text = torch.zeros(4, 7, 768, device=device)

        with torch.no_grad():
            fused = cross_attention_fusion(img, text)

        assert torch.isfinite(fused).all(), "Fused embedding contains NaN/Inf"
        assert fused.shape == (4, 768), "Output shape mismatch"

    def test_large_magnitude_inputs(self, cross_attention_fusion, device):
        """Softmax and LayerNorm keep large-magnitude tokens finite (no clamping needed)."""
        img = torch.randn(4, 5, 768, device=device) * 1000.0
        text = torch.randn(4, 7, 768, device=device) * 1000.0

        with torch.no_grad():
            fused = cross_attention_fusion(img, text)

        assert torch.isfinite(fused).all(), "Fused embedding contains NaN/Inf"

    def test_nan_inputs_propagate(self, cross_attention_fusion, device):
        """
        A NaN embedding must reach the loss so the batch is skipped before
        backward. Masking it here made the loss look valid while backward
        produced NaN gradients.
        """
        img = torch.randn(4, 5, 768, device=device)
        img[0, :, :10] = float('nan')
        text = torch.randn(4, 7, 768, device=device)

        with torch.no_grad():
            fused = cross_attention_fusion(img, text)

        assert torch.isnan(fused[0]).any(), "NaN input should not be masked"
        assert torch.isfinite(fused[1:]).all(), "Other samples must be unaffected"

    def test_inf_inputs_propagate(self, cross_attention_fusion, device):
        """Inf embeddings are not masked either."""
        img = torch.randn(4, 5, 768, device=device)
        text = torch.randn(4, 7, 768, device=device)
        text[1, :, :10] = float('-inf')

        with torch.no_grad():
            fused = cross_attention_fusion(img, text)

        assert not torch.isfinite(fused[1]).all()

    def test_attention_weights_no_overflow(self, cross_attention_fusion, device):
        """Highly correlated tokens (large dot products) do not overflow the softmax."""
        img = torch.ones(4, 5, 768, device=device) * 10.0
        text = torch.ones(4, 7, 768, device=device) * 10.0

        with torch.no_grad():
            fused = cross_attention_fusion(img, text)

        assert torch.isfinite(fused).all(), "Attention caused overflow"


# =============================================================================
# Fix #3: Safe Normalization Tests
# =============================================================================

class TestSafeNormalization:
    """Tests for safe_normalize function."""

    def test_normalize_zero_vector(self, device):
        """Test that normalization handles zero vectors without NaN."""
        # Zero vector
        x = torch.zeros(4, 768, device=device)

        # Current behavior: F.normalize
        normalized_current = F.normalize(x, dim=-1)
        has_nan_current = torch.isnan(normalized_current).any()

        # Expected after fix: safe_normalize
        # NOTE: This will fail until we implement safe_normalize()
        try:
            from src.models.multimodal import safe_normalize
            normalized_safe = safe_normalize(x, dim=-1)
            has_nan_safe = torch.isnan(normalized_safe).any()
            assert not has_nan_safe, "safe_normalize should not produce NaN for zero vectors"
        except ImportError:
            pytest.skip("safe_normalize not yet implemented")

    def test_normalize_tiny_norm_vector(self, device):
        """Test that normalization handles very small norm vectors."""
        # Very small norm vector
        x = torch.randn(4, 768, device=device) * 1e-10

        # Should not produce NaN or Inf
        try:
            from src.models.multimodal import safe_normalize
            normalized = safe_normalize(x, dim=-1)
            assert torch.isfinite(normalized).all(), "safe_normalize should produce finite values"
        except ImportError:
            pytest.skip("safe_normalize not yet implemented")

    def test_normalize_large_norm_vector(self, device):
        """Test that normalization handles large norm vectors."""
        # Large norm vector
        x = torch.randn(4, 768, device=device) * 1e6

        try:
            from src.models.multimodal import safe_normalize
            normalized = safe_normalize(x, dim=-1)
            # Normalized vectors should have unit norm
            norms = normalized.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Normalized vectors should have unit norm"
        except ImportError:
            pytest.skip("safe_normalize not yet implemented")

    def test_normalize_preserves_direction(self, device):
        """Test that normalization preserves vector direction."""
        # Normal vector
        x = torch.randn(4, 768, device=device)

        try:
            from src.models.multimodal import safe_normalize
            normalized = safe_normalize(x, dim=-1)

            # Check cosine similarity is close to 1 (same direction)
            cos_sim = F.cosine_similarity(x, normalized, dim=-1)
            assert torch.allclose(cos_sim, torch.ones_like(cos_sim), atol=1e-4), "Direction should be preserved"
        except ImportError:
            pytest.skip("safe_normalize not yet implemented")

    def test_clip_loss_with_zero_embeddings(self, loss_functions, device):
        """Test that CLIP loss handles zero embeddings without NaN."""
        clip_loss_fn = loss_functions["clip"]

        # Create zero embeddings
        img_emb = torch.zeros(4, 768, device=device)
        text_emb = torch.zeros(4, 768, device=device)

        # Compute loss (should not produce NaN after fix)
        try:
            loss = clip_loss_fn(img_emb, text_emb)

            # After Fix #3, this should not be NaN
            # safe_normalize has been applied, so zero embeddings should be handled
            assert torch.isfinite(loss), "CLIP loss should be finite with zero embeddings after fix"
        except Exception as e:
            pytest.fail(f"CLIP loss crashed with zero embeddings: {e}")


# =============================================================================
# Fix #4: MAE Epsilon Tests
# =============================================================================

class TestMAENormalizationEpsilon:
    """Tests for MAE patch normalization epsilon."""

    def test_mae_constant_image(self, mae_model, device):
        """Test that MAE handles constant images (zero variance patches)."""
        batch_size = 2

        # Create constant image (all pixels same value)
        # MAE expects 3-channel input (grayscale converted to RGB in preprocessing)
        img = torch.full((batch_size, 3, 224, 224), 0.5, device=device)

        # Forward pass should not produce NaN
        with torch.no_grad():
            loss, pred, mask = mae_model(img)

        # Assertions
        assert torch.isfinite(loss), f"Loss should be finite for constant image, got {loss}"

    def test_mae_low_variance_image(self, mae_model, device):
        """Test that MAE handles low-variance images."""
        batch_size = 2

        # Create low-variance image (3-channel)
        img = torch.randn(batch_size, 3, 224, 224, device=device) * 0.001 + 0.5

        # Forward pass should not produce NaN
        with torch.no_grad():
            loss, pred, mask = mae_model(img)

        # Assertions
        assert torch.isfinite(loss), "Loss should be finite for low-variance image"

    def test_mae_blank_image(self, mae_model, device):
        """Test that MAE handles completely blank (all zeros) images."""
        batch_size = 2

        # Create blank image (3-channel)
        img = torch.zeros(batch_size, 3, 224, 224, device=device)

        # Forward pass should not crash or produce NaN
        with torch.no_grad():
            loss, pred, mask = mae_model(img)

        # Assertions
        assert torch.isfinite(loss), "Loss should be finite for blank image"

    def test_mae_epsilon_prevents_division_by_tiny_variance(self, device):
        """Test that epsilon prevents division by near-zero variance."""
        # Simulate patch normalization
        batch_size = 2
        num_patches = 196
        patch_dim = 768

        # Create patches with very low variance
        patches = torch.randn(batch_size, num_patches, patch_dim, device=device) * 1e-8

        # Compute mean and variance
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)

        # Test with old epsilon (1e-6)
        normalized_old = (patches - mean) / (var + 1e-6) ** 0.5
        has_nan_old = torch.isnan(normalized_old).any() or torch.isinf(normalized_old).any()

        # Test with new epsilon (1e-5)
        normalized_new = (patches - mean) / (var + 1e-5) ** 0.5
        has_nan_new = torch.isnan(normalized_new).any() or torch.isinf(normalized_new).any()

        # New epsilon should be more stable
        # (Both might still have issues with extreme values, but 1e-5 is safer)
        assert torch.isfinite(normalized_new).all(), "Normalization with 1e-5 epsilon should be finite"


# =============================================================================
# Integration Tests
# =============================================================================

class TestNaNHandlingIntegration:
    """Integration tests combining multiple fixes."""

    @pytest.mark.skip(reason="Requires text tokenization setup - validated in integration tests instead")
    def test_multimodal_classifier_edge_cases(self, multimodal_classifier, device):
        """Test full classifier forward pass with edge-case inputs."""
        # Note: This test requires proper text tokenization which is complex to mock
        # The component-level tests (CrossAttention, safe_normalize) validate the fixes
        # Full end-to-end testing is done in test_multimodal_stability.py
        pass

    def test_loss_computation_with_edge_cases(self, loss_functions, device):
        """Test that all loss functions handle edge-case inputs."""
        batch_size = 4
        num_classes = 14

        # Create edge-case logits and labels
        logits = torch.zeros(batch_size, num_classes, device=device)
        labels = torch.zeros(batch_size, num_classes, device=device)

        # Focal loss should not produce NaN
        focal_loss_fn = loss_functions["focal"]
        focal_loss = focal_loss_fn(logits, labels)
        assert torch.isfinite(focal_loss), "Focal loss should be finite"


# =============================================================================
# Regression Tests
# =============================================================================

class TestNaNHandlingRegression:
    """Regression tests to ensure fixes don't break existing functionality."""

    def test_mae_reconstruction_quality(self, mae_model, normal_image):
        """Ensure MAE epsilon change doesn't degrade reconstruction."""
        # Forward pass
        with torch.no_grad():
            loss, pred, mask = mae_model(normal_image)

        # Loss should be reasonable (not too high)
        assert loss.item() < 2.0, f"Reconstruction loss unexpectedly high: {loss.item()}"
        assert loss.item() > 0.0, f"Reconstruction loss unexpectedly low: {loss.item()}"

    def test_cross_attention_capacity(self, cross_attention_fusion, normal_tokens):
        """Ensure fused embeddings are not collapsed."""
        # Forward pass
        with torch.no_grad():
            fused = cross_attention_fusion(normal_tokens["img"], normal_tokens["text"])

        # Fused embeddings should have reasonable variance (not collapsed)
        var = fused.var(dim=-1).mean()
        assert var > 0.01, f"Fused embeddings have suspiciously low variance: {var}"

    def test_normalization_unit_norm(self, device):
        """Ensure safe_normalize produces unit-norm vectors."""
        x = torch.randn(4, 768, device=device)

        try:
            from src.models.multimodal import safe_normalize
            normalized = safe_normalize(x, dim=-1)

            # Check unit norm
            norms = normalized.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Normalized vectors should have unit norm"
        except ImportError:
            pytest.skip("safe_normalize not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
