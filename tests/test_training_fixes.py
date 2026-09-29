"""
Regression tests for the training-pipeline fixes.

Each test exercises the project's real code (MultimodalClassifier, MultiTaskLoss,
train_classifier.train_epoch, src.training) rather than re-implementing it.
The classifier is built at img_size=32 (4 patches) so the real ViT-B + BERT
stack runs in seconds on CPU.
"""

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.cuda.amp import GradScaler

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import train_classifier as tc
import train_mae
from src.models.config import MAEConfig, get_classifier_config
from src.models.losses import MultiTaskLoss
from src.models.mae import MaskedAutoencoder
from src.models.multimodal import CrossAttentionFusion, MultimodalClassifier, StructuredEncoder
from src.training import (
    ConsecutiveSkipGuard,
    backward_and_step,
    build_warmup_cosine_scheduler,
    check_trainable_params_in_optimizer,
    is_cuda_device,
    load_checkpoint_states,
    save_checkpoint_files,
    warmup_cosine_factor,
)
from tests.utils import IMG, make_batch, make_classifier, make_training


@pytest.fixture(scope="module")
def classifier() -> MultimodalClassifier:
    return make_classifier()


# =============================================================================
# Parameters outside the optimizer (the gradient-clipping cascade)
# =============================================================================

class TestOptimizerCoverage:
    """Every trainable parameter must be in the optimizer, frozen or unfrozen."""

    @pytest.mark.parametrize("epoch", [0, 5])
    def test_all_trainable_params_in_optimizer(self, classifier, epoch):
        _, loss_fn, optimizer, _ = make_training(classifier, steps_per_epoch=4)
        classifier.set_epoch(epoch)
        check_trainable_params_in_optimizer([classifier, loss_fn], optimizer)

    def test_text_clip_proj_and_logit_scale_are_optimized(self, classifier):
        _, loss_fn, optimizer, _ = make_training(classifier, steps_per_epoch=4)
        in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert all(id(p) in in_opt for p in classifier.text_clip_proj.parameters())
        assert id(loss_fn.clip_loss.logit_scale) in in_opt
        loss_group = next(g for g in optimizer.param_groups if g["name"] == "loss")
        assert loss_group["weight_decay"] == 0.0

    def test_unfreeze_leaves_decoder_and_pos_embed_frozen(self, classifier):
        classifier.set_epoch(100)
        mae = classifier.image_encoder
        assert not mae.pos_embed.requires_grad
        assert not mae.mask_token.requires_grad
        assert not any(p.requires_grad for p in mae.decoder_blocks.parameters())
        assert mae.cls_token.requires_grad
        classifier.set_epoch(0)
        assert not mae.cls_token.requires_grad

    def test_check_flags_orphan_parameter(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        optimizer = torch.optim.AdamW(model[0].parameters())
        with pytest.raises(RuntimeError, match="not in the optimizer"):
            check_trainable_params_in_optimizer([model], optimizer)


class TestNonFiniteGradientRecovery:
    """One overflow event must cost one skipped step, not the rest of the run."""

    def test_inf_gradient_skips_one_step_then_training_continues(self):
        model = make_classifier(freeze_mae_epochs=100)
        loader = [make_batch() for _ in range(6)]
        config, loss_fn, optimizer, scheduler = make_training(model, len(loader))
        scaler = GradScaler(enabled=False)

        # One overflow event on the 3rd backward pass, in a head parameter
        calls = {"n": 0}
        weight = model.text_clip_proj[0].weight

        def poison_once(grad):
            calls["n"] += 1
            if calls["n"] == 3:
                grad = grad.clone()
                grad[0, 0] = float("inf")
            return grad

        handle = weight.register_hook(poison_once)
        try:
            metrics = tc.train_epoch(
                model, loader, optimizer, scheduler, scaler, loss_fn, config, "cpu", epoch=0
            )
        finally:
            handle.remove()

        assert metrics["skipped_grad"] == 1
        assert metrics["steps"] == len(loader) - 1
        assert all(torch.isfinite(p).all() for p in model.parameters())
        # Gradients are cleared after every step, so nothing non-finite lingers
        assert weight.grad is None

    def test_backward_and_step_never_applies_nonfinite_update(self):
        layer = torch.nn.Linear(3, 1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        before = [p.detach().clone() for p in layer.parameters()]
        loss = layer(torch.tensor([[float("inf"), 1.0, 1.0]])).sum() * 0.0
        stepped, grad_norm = backward_and_step(loss, optimizer, GradScaler(enabled=False), 1.0)
        assert not stepped and not math.isfinite(grad_norm)
        for p, b in zip(layer.parameters(), before):
            assert torch.equal(p, b)
            assert p.grad is None

    def test_consecutive_skip_guard_fails_loudly(self):
        guard = ConsecutiveSkipGuard(limit=3)
        guard.update(False)
        guard.update(True)  # progress resets the count
        guard.update(False)
        guard.update(False)
        with pytest.raises(RuntimeError, match="3 consecutive batches"):
            guard.update(False)

    def test_mae_epoch_skips_nonfinite_loss(self):
        torch.manual_seed(0)
        model = MaskedAutoencoder(img_size=IMG, embed_dim=32, depth=1, num_heads=2,
                                  decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2)
        config = MAEConfig(epochs=1, warmup_epochs=0, log_interval=10_000)
        images = [torch.rand(2, 3, IMG, IMG) for _ in range(3)]
        images[1][0, 0, 0, 0] = float("nan")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_warmup_cosine_scheduler(optimizer, 0, 3, 0.1)
        metrics = train_mae.train_epoch(
            model, images, optimizer, scheduler, GradScaler(enabled=False), config, "cpu", 0
        )
        assert metrics["skipped"] == 1
        assert math.isfinite(metrics["loss"])
        assert all(torch.isfinite(p).all() for p in model.parameters())


# =============================================================================
# Learning-rate schedule and layer-wise decay
# =============================================================================

class TestSchedule:

    def test_epochs_equal_warmup_does_not_divide_by_zero(self):
        # previously: ZeroDivisionError on the final scheduler step
        assert warmup_cosine_factor(10, warmup_steps=10, total_steps=10, min_ratio=0.1) == 1.0
        assert warmup_cosine_factor(15, warmup_steps=10, total_steps=10, min_ratio=0.1) == pytest.approx(0.1)

    def test_warmup_then_cosine(self):
        assert warmup_cosine_factor(0, 10, 110, 0.0) == 0.0
        assert warmup_cosine_factor(5, 10, 110, 0.0) == 0.5
        assert warmup_cosine_factor(60, 10, 110, 0.0) == pytest.approx(0.5)
        assert warmup_cosine_factor(110, 10, 110, 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_encoder_lr_rewarms_after_unfreeze(self, classifier):
        steps_per_epoch = 10
        config, _, optimizer, scheduler = make_training(
            classifier, steps_per_epoch, freeze_mae_epochs=1, unfreeze_warmup_epochs=1
        )
        groups = {g["name"]: i for i, g in enumerate(optimizer.param_groups)}
        lrs = []
        for _ in range(3 * steps_per_epoch):
            lrs.append(scheduler.get_last_lr())
            scheduler.step()
        block = groups["mae.block11"]
        head = groups["head"]
        # Frozen epoch: encoder LR held at 0 while the head warms up
        assert all(step_lrs[block] == 0.0 for step_lrs in lrs[:steps_per_epoch])
        # First step after unfreeze is a small fraction of the head-relative LR
        ratio = config.lr_decay ** 1
        first = lrs[steps_per_epoch][block] / (lrs[steps_per_epoch][head] * ratio)
        assert first == pytest.approx(1 / steps_per_epoch)
        # Fully ramped one epoch later
        later = lrs[2 * steps_per_epoch][block] / (lrs[2 * steps_per_epoch][head] * ratio)
        assert later == pytest.approx(1.0)

    def test_per_block_layer_decay_matches_docs(self, classifier):
        config, _, optimizer, _ = make_training(classifier, steps_per_epoch=4)
        lr = {g["name"]: g["initial_lr"] for g in optimizer.param_groups}
        # docs/MODEL_TRAINING_RESEARCH.md: layer i gets decay ** (12 - i)
        assert lr["mae.block0"] == pytest.approx(config.base_lr * 0.9 ** 12)
        assert lr["mae.block11"] == pytest.approx(config.base_lr * 0.9)
        assert lr["mae.embed"] == pytest.approx(config.base_lr * 0.9 ** 13)
        assert lr["head"] == pytest.approx(config.base_lr)


# =============================================================================
# Text branch
# =============================================================================

class TestTextBranch:

    def test_text_encoder_matches_preprocessing_tokenizer(self, classifier):
        from src.config.settings import PreprocessingConfig
        assert classifier.text_encoder.model_name == PreprocessingConfig().tokenizer_model

    def test_frozen_bert_stays_in_eval_mode(self, classifier):
        classifier.train()
        assert not classifier.text_encoder.bert.training
        ids = torch.tensor([[101, 2000, 3000, 102]])
        with torch.no_grad():
            a = classifier.text_encoder(ids)
            b = classifier.text_encoder(ids)
        assert torch.equal(a, b), "dropout is active in the frozen text encoder"
        classifier.eval()


# =============================================================================
# Cross-attention fusion
# =============================================================================

class TestCrossAttentionFusion:

    @pytest.fixture
    def fusion(self):
        torch.manual_seed(0)
        return CrossAttentionFusion(embed_dim=32, num_heads=4, dropout=0.0)

    def test_query_and_key_projections_learn(self, fusion):
        img = torch.randn(2, 5, 32)
        text = torch.randn(2, 7, 32)
        fusion(img, text, torch.ones(2, 7)).sum().backward()
        for attn in (fusion.img_to_text_attn, fusion.text_to_img_attn):
            q_k_grad = attn.in_proj_weight.grad[:64]  # query and key rows
            assert q_k_grad.abs().sum() > 0

    def test_output_depends_on_text_tokens_beyond_cls(self, fusion):
        img = torch.randn(1, 5, 32)
        text = torch.randn(1, 7, 32)
        other = text.clone()
        other[:, 3:] = torch.randn(1, 4, 32)
        with torch.no_grad():
            assert not torch.allclose(fusion(img, text), fusion(img, other))

    def test_padding_is_ignored_and_empty_text_is_finite(self, fusion):
        img = torch.randn(2, 5, 32)
        text = torch.randn(2, 7, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)
        changed = text.clone()
        changed[:, 3:] = 100.0  # only padded positions change
        with torch.no_grad():
            out = fusion(img, text, mask)
            assert torch.isfinite(out).all()
            assert torch.allclose(out[0], fusion(img, changed, mask)[0], atol=1e-6)

    def test_nan_input_reaches_the_loss_gate(self, fusion):
        # NaN must propagate so the training loop skips the batch before backward
        img = torch.randn(1, 5, 32)
        img[0, 2, 0] = float("nan")
        with torch.no_grad():
            assert torch.isnan(fusion(img, torch.randn(1, 7, 32))).any()


# =============================================================================
# Structured features
# =============================================================================

class TestStructuredNormalization:

    def make_encoder(self):
        torch.manual_seed(0)
        encoder = StructuredEncoder(input_dim=3, hidden_dim=8, output_dim=8,
                                    log_transform=[True, False, False])
        rng = np.random.default_rng(0)
        train = np.stack([
            rng.lognormal(7, 1.5, 500),  # NT-proBNP-like
            rng.normal(140, 4, 500),     # sodium-like
            rng.normal(80, 15, 500),     # heart rate
        ], axis=1).astype(np.float32)
        train[:50, 2] = np.nan           # some missing
        encoder.fit_normalization(torch.from_numpy(train))
        return encoder, torch.from_numpy(train)

    def test_extreme_lab_values_stay_fp16_safe(self):
        encoder, _ = self.make_encoder()
        raw = torch.tensor([[70000.0, 140.0, 80.0]])
        assert torch.isinf(raw.half()).any(), "raw value is not representable in fp16"
        normalized = encoder.normalize(raw)
        assert normalized.abs().max() <= encoder.clip_value
        assert torch.isfinite(encoder.encoder[0](normalized).half()).all()

    def test_standardized_on_training_data(self):
        encoder, train = self.make_encoder()
        z = encoder.normalize(train)[:, :3]
        observed = torch.isfinite(train)
        for j in range(3):
            values = z[observed[:, j], j]
            assert values.mean().abs() < 0.05
            assert (values.std() - 1).abs() < 0.1

    def test_missing_values_get_indicator_not_a_fake_value(self):
        encoder, _ = self.make_encoder()
        out = encoder.normalize(torch.tensor([[float("nan"), 140.0, float("inf")]]))
        assert torch.equal(out[0, 3:], torch.tensor([1.0, 0.0, 1.0]))
        assert out[0, 0] == 0 and out[0, 2] == 0

    def test_normalization_is_saved_with_the_model(self):
        encoder, _ = self.make_encoder()
        fresh = StructuredEncoder(input_dim=3, hidden_dim=8, output_dim=8)
        fresh.load_state_dict(encoder.state_dict())
        assert torch.equal(fresh.feature_mean, encoder.feature_mean)
        assert bool(fresh.normalization_fitted)

    def test_censored_lab_results_use_reporting_limit(self):
        from src.preprocessing.structured import fill_censored_lab_values
        valuenum = pd.Series([np.nan, 5.0, np.nan, np.nan, np.nan], dtype="float32")
        value = pd.Series(["GREATER THAN 70000", "5", ">300", "<0.01", "NEG"])
        filled = fill_censored_lab_values(valuenum, value)
        assert filled.iloc[:4].tolist() == pytest.approx([70000, 5, 300, 0.01])
        assert np.isnan(filled.iloc[4])

    def test_procalcitonin_feature_removed(self):
        from src.datasets.mimic_iv import MIMICIVLoader
        from src.models.classification_dataset import MultimodalClassificationDataset as D
        assert 50976 not in {i for ids in MIMICIVLoader.PRIORITY_LAB_IDS.values() for i in ids}
        assert not any("procalcitonin" in name for name in D.STRUCTURED_FEATURES)


# =============================================================================
# Preprocessed dataset on disk
# =============================================================================

class TestClassificationDataset:

    def make(self, preprocessed_dir, **kwargs):
        from src.models.classification_dataset import MultimodalClassificationDataset
        return MultimodalClassificationDataset(
            preprocessed_dir, preprocessed_dir / "chexpert.csv",
            target_size=(IMG, IMG), augment=False, **kwargs,
        )

    def test_missing_and_infinite_structured_values_are_nan(self, preprocessed_dir):
        ds = self.make(preprocessed_dir)
        bnp = ds.STRUCTURED_FEATURES.index("lab_bnp_mean")
        acuity = ds.STRUCTURED_FEATURES.index("triage_acuity")
        matrix = ds.structured_matrix()
        assert matrix.shape == (3, len(ds.STRUCTURED_FEATURES))
        assert matrix[0, bnp] == 70000.0
        assert torch.isnan(matrix[1, bnp]) and torch.isnan(matrix[2, acuity])
        assert torch.equal(ds[1]["structured"].isnan(), matrix[1].isnan())

    def test_token_vocabulary_check(self, preprocessed_dir):
        ds = self.make(preprocessed_dir)
        ds.validate_text_tokens(cls_token_id=101, vocab_size=28996)  # Bio_ClinicalBERT
        with pytest.raises(ValueError, match="do not match the text encoder"):
            ds.validate_text_tokens(cls_token_id=2, vocab_size=30522)  # PubMedBERT
        with pytest.raises(ValueError, match="out-of-vocabulary"):
            ds.validate_text_tokens(cls_token_id=101, vocab_size=2600)

    def test_resize_keeps_the_whole_radiograph(self, preprocessed_dir):
        resized = self.make(preprocessed_dir)[0]["image"]
        cropped = self.make(preprocessed_dir, image_mode="center_crop")[0]["image"]
        # The corner marker survives a full-view resize but not a center crop
        assert resized[0, :2, :2].mean() > resized[0, -2:, -2:].mean()
        assert torch.allclose(cropped[0, :2, :2], cropped[0, -2:, -2:])


# =============================================================================
# Inference preprocessing and view selection
# =============================================================================

class TestInferenceAndViews:

    def test_detect_anomalies_matches_training_transform(self, tmp_path):
        from PIL import Image
        import detect_anomalies
        from src.models.dataset import build_image_transform
        from src.preprocessing.images import load_and_process_image

        pixels = (np.random.default_rng(0).random((300, 250)) * 255).astype(np.uint8)
        path = tmp_path / "xray.jpg"
        Image.fromarray(pixels).save(path)

        for mode in ("resize", "center_crop"):
            got = detect_anomalies.preprocess_image(path, img_size=64, image_mode=mode)
            arr = load_and_process_image(path, "minmax")
            expected = build_image_transform((64, 64), mode=mode)(torch.from_numpy(arr[0]))
            assert got.shape == (1, 3, 64, 64)
            assert torch.allclose(got[0], expected)

    def test_frontal_view_selected_from_metadata(self):
        from src.preprocessing.images import select_frontal_dicoms
        metadata = pd.DataFrame({
            "study_id": [1, 1, 1, 2, 2, 3],
            "dicom_id": ["c-lat", "b-ap", "a-pa", "z-ap", "y-lat", "x-ll"],
            "ViewPosition": pd.Categorical(["LATERAL", "AP", "PA", "AP", "LATERAL", "LL"]),
        })
        assert select_frontal_dicoms(metadata) == {1: "a-pa", 2: "z-ap"}


# =============================================================================
# Resume, CLI overrides, device handling
# =============================================================================

class TestResumeAndCli:

    def test_config_round_trips_through_checkpoint(self, tmp_path):
        config = get_classifier_config("fast")
        config.train_dir = tmp_path / "train"
        config.chexpert_csv = tmp_path / "labels.csv"
        config.classifier.clip_weight = 0.0
        restored = tc.config_from_dict(tc.config_to_dict(config))
        assert restored.classifier == config.classifier
        assert restored.train_dir == config.train_dir
        assert restored.chexpert_csv == config.chexpert_csv

    def test_zero_valued_overrides_are_applied(self, tmp_path, monkeypatch):
        import argparse
        monkeypatch.chdir(tmp_path)  # the MAE TrainingConfig creates output dirs in CWD
        config = get_classifier_config("base")
        args = argparse.Namespace(**{k: None for k in tc.CLI_OVERRIDES})
        args.clip_weight = 0.0
        args.supcon_weight = 0.0
        tc.apply_overrides(config, args, resuming=False)
        assert config.classifier.clip_weight == 0.0
        assert config.classifier.supcon_weight == 0.0

        mae_config = train_mae.get_base_config()
        mae_args = argparse.Namespace(**{k: None for k in train_mae.CLI_OVERRIDES})
        mae_args.num_workers = 0
        train_mae.apply_overrides(mae_config, mae_args, resuming=False)
        assert mae_config.mae.num_workers == 0

    def test_resume_rejects_schedule_changes(self):
        import argparse
        config = get_classifier_config("base")
        args = argparse.Namespace(**{k: None for k in tc.CLI_OVERRIDES})
        args.epochs = config.classifier.epochs + 10
        with pytest.raises(ValueError, match="differs from the checkpoint"):
            tc.apply_overrides(config, args, resuming=True)
        args.epochs = None
        args.num_workers = 2  # hardware settings may change
        tc.apply_overrides(config, args, resuming=True)

    def test_checkpoint_restores_epoch_best_history_and_loss_state(self, tmp_path):
        model = torch.nn.Linear(2, 2)
        loss_fn = MultiTaskLoss()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()))
        scheduler = build_warmup_cosine_scheduler(optimizer, 2, 10, 0.1)
        for _ in range(3):
            scheduler.step()
        with torch.no_grad():
            loss_fn.clip_loss.logit_scale.fill_(3.0)
        checkpoint = {
            "epoch": 4,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "best_auroc": 0.8,
            "history": {"auroc_mean": [0.8, 0.7]},
        }
        path = save_checkpoint_files(checkpoint, tmp_path, "classifier", 4, best_path=tmp_path / "best.pt")
        assert (tmp_path / "classifier_latest.pt").exists() and (tmp_path / "best.pt").exists()

        loaded = torch.load(path)
        new_loss = MultiTaskLoss()
        new_opt = torch.optim.AdamW(list(model.parameters()) + list(new_loss.parameters()))
        new_sched = build_warmup_cosine_scheduler(new_opt, 2, 10, 0.1)
        load_checkpoint_states(loaded, model, new_opt, new_sched,
                               scaler=GradScaler(enabled=False),
                               extra_modules={"loss_fn": new_loss})
        assert new_loss.clip_loss.logit_scale.item() == 3.0
        assert new_sched.last_epoch == 3
        # train() resumes after the completed epoch with the saved best metric
        assert loaded["epoch"] + 1 == 5 and loaded["best_auroc"] == 0.8

    def test_legacy_mae_checkpoint_keeps_center_crop(self):
        legacy = {"config": {"mae": {"img_size": 1024, "epochs": 40}}}
        assert train_mae.config_from_checkpoint(legacy).mae.image_mode == "center_crop"
        current = {"config": {"mae": {"img_size": 1024, "image_mode": "resize"}}}
        assert train_mae.config_from_checkpoint(current).mae.image_mode == "resize"

    def test_any_cuda_device_string_enables_amp(self):
        assert is_cuda_device("cuda") and is_cuda_device("cuda:1")
        assert not is_cuda_device("cpu")


# =============================================================================
# End to end: the real CLIs on a tiny cohort
# =============================================================================

class TestEndToEnd:

    def test_classifier_trains_then_resumes_from_checkpoint(self, preprocessed_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out, ckpt = tmp_path / "models", tmp_path / "checkpoints"
        try:
            self._train_then_resume(preprocessed_dir, out, ckpt, monkeypatch)
        finally:
            # Each checkpoint carries the full ViT-B + BERT weights (~1 GB)
            shutil.rmtree(out, ignore_errors=True)
            shutil.rmtree(ckpt, ignore_errors=True)

    def _train_then_resume(self, preprocessed_dir, out, ckpt, monkeypatch):
        import json
        from src.models.classification_dataset import MultimodalClassificationDataset as D

        monkeypatch.setattr(sys, "argv", [
            "train_classifier.py", "--config", "debug",
            "--train-dir", str(preprocessed_dir), "--val-dir", str(preprocessed_dir),
            "--chexpert-csv", str(preprocessed_dir / "chexpert.csv"),
            "--img-size", str(IMG), "--batch-size", "2", "--epochs", "2",
            "--output-dir", str(out), "--checkpoint-dir", str(ckpt),
            "--device", "cpu", "--num-workers", "0",
        ])
        tc.main()

        history = json.loads((out / "classifier_history.json").read_text())
        assert history["epoch"] == [0, 1]
        saved = json.loads((out / "classifier_config.json").read_text())
        assert saved["classifier"]["struct_input_dim"] == len(D.STRUCTURED_FEATURES)

        # The documented resume command: no data paths or preset needed
        monkeypatch.setattr(sys, "argv", [
            "train_classifier.py", "--resume", str(ckpt / "classifier_epoch_0000.pt"), "--device", "cpu",
        ])
        tc.main()
        history = json.loads((out / "classifier_history.json").read_text())
        assert history["epoch"] == [0, 1], "resume must continue after the saved epoch"
        assert len(history["auroc_mean"]) == 2

    def test_mae_trains_and_detect_anomalies_reads_its_hdf5(self, preprocessed_dir, tmp_path, monkeypatch):
        import json
        import detect_anomalies

        monkeypatch.chdir(tmp_path)
        out = tmp_path / "mae"
        monkeypatch.setattr(sys, "argv", [
            "train_mae.py", "--config", "debug", "--train-dir", str(preprocessed_dir),
            "--img-size", str(IMG), "--batch-size", "2", "--epochs", "1",
            "--output-dir", str(out), "--checkpoint-dir", str(out),
            "--device", "cpu", "--num-workers", "0", "--skip-anomaly",
        ])
        train_mae.main()
        assert torch.load(out / "mae_final.pt")["config"]["image_mode"] == "resize"

        results = tmp_path / "results.json"
        monkeypatch.setattr(sys, "argv", [
            "detect_anomalies.py", "--hdf5", str(preprocessed_dir / "images.h5"),
            "--model", str(out / "mae_final.pt"), "--device", "cpu", "--output", str(results),
        ])
        detect_anomalies.main()
        assert len(json.loads(results.read_text())) == 3
