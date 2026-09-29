"""
Integration tests for training stability and NaN handling.

Runs the real training loop (train_classifier.train_epoch and
src.training.backward_and_step) with injected NaN batches and non-finite
gradients: bad batches must be skipped without ever corrupting weights.
"""

import time

import h5py
import numpy as np
import pytest
import torch
from torch.cuda.amp import GradScaler

import train_classifier as tc
from src.models.classification_dataset import MultimodalClassificationDataset
from src.training import backward_and_step
from tests.utils import make_batch, make_classifier, make_training


# =============================================================================
# Training Loop Stability Tests
# =============================================================================

class TestTrainingLoopStability:
    """Tests for skipping non-finite batches in the real training loop."""

    def test_training_step_with_nan_batch(self):
        """Batches whose forward pass produces NaN are skipped before backward."""
        model = make_classifier(freeze_mae_epochs=0)
        loader = [make_batch() for _ in range(9)]
        for batch in loader[::3]:
            batch["image"][0, 0, 0, 0] = float("nan")
        config, loss_fn, optimizer, scheduler = make_training(model, len(loader))

        metrics = tc.train_epoch(
            model, loader, optimizer, scheduler, GradScaler(enabled=False),
            loss_fn, config, "cpu", epoch=0,
        )

        assert metrics["skipped_loss"] == 3
        assert metrics["steps"] == 6
        assert all(torch.isfinite(p).all() for p in model.parameters())

    def test_nonfinite_gradients_are_never_applied(self):
        """Every other step gets an Inf gradient: those steps change nothing."""
        model = make_classifier()
        loader = [make_batch() for _ in range(8)]
        config, loss_fn, optimizer, scheduler = make_training(model, len(loader))

        weight = model.classifier.weight
        calls = {"n": 0}

        def poison_every_other(grad):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                grad = grad.clone()
                grad[0, 0] = float("inf")
            return grad

        handle = weight.register_hook(poison_every_other)
        try:
            metrics = tc.train_epoch(
                model, loader, optimizer, scheduler, GradScaler(enabled=False),
                loss_fn, config, "cpu", epoch=0,
            )
        finally:
            handle.remove()

        assert metrics["skipped_grad"] == 4
        assert metrics["steps"] == 4
        assert all(torch.isfinite(p).all() for p in model.parameters())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for mixed precision")
    def test_amp_overflow_skips_step_and_backs_off_scale(self):
        """Under AMP an overflow skips the update and halves the loss scale."""
        layer = torch.nn.Linear(4, 1).cuda()
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        scaler = GradScaler(init_scale=2.0 ** 16)
        before = [p.detach().clone() for p in layer.parameters()]

        with torch.autocast("cuda"):
            loss = layer(torch.full((2, 4), 6e4, device="cuda")).float().sum() * 1e4
        stepped, _ = backward_and_step(loss, optimizer, scaler, max_grad_norm=1.0)

        assert not stepped
        assert scaler.get_scale() == 2.0 ** 15
        for p, b in zip(layer.parameters(), before):
            assert torch.equal(p, b)

    def test_forced_corruption_every_10_batches(self, device):
        """
        30 batches with a NaN batch every 10: the loop skips each one and keeps
        training (previously a single bad gradient cascaded into weight
        reverts on every later step).
        """
        model = make_classifier(freeze_mae_epochs=0).to(device)
        loader = [make_batch() for _ in range(30)]
        for i in range(10, 30, 10):
            loader[i]["structured"][:] = float("inf")
            loader[i]["image"][:] = float("nan")
        config, loss_fn, optimizer, scheduler = make_training(model, len(loader))
        loss_fn.to(device)
        scaler = GradScaler(enabled=device.type == "cuda")

        metrics = tc.train_epoch(
            model, loader, optimizer, scheduler, scaler, loss_fn, config, str(device), epoch=0,
        )

        assert metrics["skipped_loss"] == 2
        assert metrics["steps"] + metrics["skipped_grad"] == 28
        assert all(torch.isfinite(p).all() for p in model.parameters())


# =============================================================================
# Data Quality and Preprocessing Tests
# =============================================================================

class TestDataQualityHandling:
    """Tests for handling problematic data samples."""

    def make_dataset(self, preprocessed_dir):
        return MultimodalClassificationDataset(
            preprocessed_dir, preprocessed_dir / "chexpert.csv",
            target_size=(32, 32), augment=False,
        )

    def test_image_sanitization(self, preprocessed_dir):
        """NaN/Inf pixels in a stored image are sanitized at load time."""
        with h5py.File(preprocessed_dir / "images.h5", "r+") as f:
            image = f["images/0"][:]
            image[0, :10, :10] = np.nan
            image[0, 10:20, :10] = np.inf
            f["images/0"][...] = image

        sample = self.make_dataset(preprocessed_dir)[0]
        assert torch.isfinite(sample["image"]).all(), "Image should not contain NaN/Inf"

    def test_structured_features_with_missing_values(self, preprocessed_dir):
        """Missing/Inf structured values reach the model as NaN and encode to finite features."""
        dataset = self.make_dataset(preprocessed_dir)
        model = make_classifier()
        model.struct_encoder = type(model.struct_encoder)(
            input_dim=len(dataset.STRUCTURED_FEATURES), hidden_dim=16, output_dim=16,
            log_transform=dataset.structured_log_transform_flags(),
        )
        model.struct_encoder.fit_normalization(dataset.structured_matrix())

        batch = torch.stack([dataset[i]["structured"] for i in range(len(dataset))])
        assert torch.isnan(batch).any(), "Missing values should stay NaN until the model"
        with torch.no_grad():
            embedding = model.struct_encoder(batch)
        assert torch.isfinite(embedding).all(), "Structured embedding should be finite"


# =============================================================================
# Performance and Regression Tests
# =============================================================================

class TestPerformanceRegression:
    """Tests to ensure fixes don't degrade performance."""

    def test_forward_pass_speed(self, multimodal_classifier, device):
        """Forward pass throughput with the real inputs."""
        model = multimodal_classifier
        batch = make_batch(batch_size=8)
        batch["structured"] = torch.randn(8, model.struct_encoder.input_dim)
        inputs = [batch[k].to(device) for k in ("image", "text_tokens", "structured", "attention_mask")]

        with torch.no_grad():
            model(*inputs)  # warmup
            start_time = time.time()
            num_iterations = 5
            for _ in range(num_iterations):
                model(*inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        throughput = (num_iterations * 8) / elapsed

        print(f"\nForward pass throughput: {throughput:.1f} samples/sec")

        # Should process at least 100 samples/sec on GPU
        if device.type == "cuda":
            assert throughput > 100, f"Throughput too low: {throughput} samples/sec"

    def test_model_capacity_not_reduced(self, multimodal_classifier, device):
        """Outputs are not collapsed across samples or classes."""
        batch = make_batch(batch_size=4)
        batch["structured"] = torch.randn(4, multimodal_classifier.struct_encoder.input_dim)
        inputs = [batch[k].to(device) for k in ("image", "text_tokens", "structured", "attention_mask")]

        with torch.no_grad():
            logits, clip_emb, _ = multimodal_classifier(*inputs)

        logits_var = logits.var(dim=-1).mean()
        assert logits_var > 0.01, f"Logits have suspiciously low variance: {logits_var}"
        assert clip_emb.std(dim=0).mean() > 1e-3, "CLIP embeddings collapsed across samples"
