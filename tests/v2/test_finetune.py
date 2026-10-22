"""M3: optimizer groups, schedule, loss, and a fine-tune smoke run with resume."""

import json
import math

import numpy as np
import pytest
import torch

from v2 import finetune
from v2.common import LABELS
from v2.encoders import load_encoder
from v2.finetune import Classifier, EpochShuffle, masked_bce, param_groups, warmup_cosine


def _groups(model, **kw):
    args = dict(encoder_lr=2e-5, head_lr=1e-3, weight_decay=0.05, llrd=0.75) | kw
    return param_groups(model, **args)


def _lr_of(groups, name):
    return next(g for g in groups if name in g["names"])


def test_layerwise_lr_decay_and_no_decay_sets():
    model = Classifier(load_encoder("tiny-dinov2"))
    groups = _groups(model)
    n_layers = model.encoder.backbone.config.num_hidden_layers  # 2
    emb = _lr_of(groups, "embeddings.patch_embeddings.projection.weight")
    top = _lr_of(groups, f"encoder.layer.{n_layers - 1}.mlp.fc1.weight")  # attention names differ across versions
    norm = _lr_of(groups, "layernorm.weight")
    head = _lr_of(groups, "head.weight")
    assert emb["lr"] == pytest.approx(2e-5 * 0.75 ** (n_layers + 1))
    assert top["lr"] == pytest.approx(2e-5 * 0.75)
    assert norm["lr"] == pytest.approx(2e-5) and norm["weight_decay"] == 0.0
    assert head["lr"] == pytest.approx(1e-3) and head["weight_decay"] == 0.05
    assert _lr_of(groups, "embeddings.position_embeddings")["weight_decay"] == 0.0
    assert _lr_of(groups, "embeddings.cls_token")["weight_decay"] == 0.0
    assert _lr_of(groups, "head.bias")["weight_decay"] == 0.0
    assert top["weight_decay"] == 0.05
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == len([p for p in model.parameters() if p.requires_grad])


def test_densenet_has_no_layerwise_decay():
    model = Classifier(load_encoder("densenet121", size=32, pretrained=False))
    lrs = {g["lr"] for g in _groups(model, encoder_lr=1e-4)}
    assert lrs == {1e-4, 1e-3}


def test_warmup_cosine_schedule():
    fn = warmup_cosine(10, 110)
    assert fn(0) == pytest.approx(0.1)
    assert fn(9) == pytest.approx(1.0)
    assert fn(60) == pytest.approx(0.5)
    assert fn(110) == pytest.approx(0.0, abs=1e-12)


def test_masked_bce_ignores_masked_entries():
    logits = torch.tensor([[10.0, -10.0], [0.0, 0.0]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    mask = torch.tensor([[False, False], [True, True]])
    assert masked_bce(logits, targets, mask).item() == pytest.approx(math.log(2), rel=1e-5)


def test_head_bias_starts_at_label_priors():
    model = Classifier(load_encoder("tiny-dinov2"), np.full(len(LABELS), 0.2))
    assert torch.allclose(torch.sigmoid(model.head.bias), torch.full((len(LABELS),), 0.2))


def test_epoch_shuffle_is_deterministic_per_epoch():
    s = EpochShuffle(50, seed=3)
    s.set_epoch(1)
    a = list(s)
    s.set_epoch(2)
    b = list(s)
    s.set_epoch(1)
    assert list(s) == a and a != b and sorted(a) == list(range(50))


def test_finetune_writes_checkpoints_predictions_and_resumes(prepared):
    args = ["--model", "tiny-dinov2", "--out", str(prepared), "--epochs", "2", "--batch-size", "8",
            "--accum", "2", "--workers", "0", "--steps-per-epoch", "2", "--run", "ft_test", "--save-embeddings"]
    run = finetune.main(args)
    run_dir = prepared / "runs" / run
    assert (run_dir / "last.pt").exists() and (run_dir / "best.pt").exists()
    assert (prepared / "predictions" / f"{run}.parquet").exists()
    assert (prepared / "embeddings" / f"{run}.npz").exists()
    meta = json.loads((prepared / "predictions" / f"{run}.json").read_text())
    assert meta["kind"] == "finetune" and meta["encoder"] == "tiny-dinov2"
    assert all(np.isfinite(h["train_loss"]) for h in meta["history"])

    args[args.index("--epochs") + 1] = "3"
    finetune.main([*args, "--resume"])
    history = json.loads((run_dir / "history.json").read_text())
    assert [h["epoch"] for h in history] == [0, 1, 2]
    assert history[-1]["steps"] == 6  # 2 optimizer steps per epoch, continued from step 4


def test_attention_classifier_starts_at_the_priors():
    """--head attention (§6 M5b amendment): M4b's h0 head on the encoder's tokens, bias at the label priors."""
    enc = load_encoder("tiny-dinov2")
    g = enc.token_grid(12)
    p = np.linspace(0.05, 0.6, len(LABELS))
    model = finetune.AttentionClassifier(enc, 12, np.zeros(enc.token_dim, np.float32),
                                         np.ones(enc.token_dim, np.float32), p)
    torch.testing.assert_close(model.head.bias.detach(), torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float32))
    x = torch.randn(2, 3, enc.size, enc.size)
    assert model.features(x).shape == (2, 1 + g * g, enc.token_dim) and model(x).shape == (2, len(LABELS))
    groups = _groups(model)
    assert _lr_of(groups, "head.queries")["lr"] == pytest.approx(1e-3)
    assert _lr_of(groups, "head.bias")["weight_decay"] == 0.0


def test_attention_head_finetune_writes_its_own_run(prepared, tmp_path):
    import shutil

    out = tmp_path / "v2"
    shutil.copytree(prepared, out, ignore=shutil.ignore_patterns("embeddings", "predictions", "reports", "runs"))
    args = ["--model", "tiny-dinov2", "--head", "attention", "--out", str(out), "--epochs", "2", "--batch-size", "8",
            "--accum", "1", "--workers", "0", "--steps-per-epoch", "3"]
    run = finetune.main(args)
    assert run.endswith("_ft_attn_s0")
    meta = json.loads((out / "predictions" / f"{run}.json").read_text())
    assert meta["kind"] == "finetune" and meta["head"] == "attention" and meta["config"]["grid"] == 12
    state = torch.load(out / "runs" / run / "best.pt", map_location="cpu", weights_only=False)["model"]
    assert "head.queries" in state and float(state["head.token_std"].min()) > 0  # normalization from the encoder
    assert float(state["head.token_mean"].abs().sum()) > 0
    assert not (out / "embeddings").exists() or not list((out / "embeddings").glob(f"{run}*"))


def test_the_attention_head_has_no_embeddings_to_save():
    with pytest.raises(SystemExit):
        finetune.build_args(["--head", "attention", "--save-embeddings"])
