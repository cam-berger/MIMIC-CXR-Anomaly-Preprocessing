"""M4b part 1: attention heads over cached tokens, the steering mechanism, and gate G4b end to end."""

import json

import numpy as np
import pytest
import torch

from v2 import attend, embed, evaluate, probe
from v2.attend import Inputs, Recipe, build_head, fit_head, predict_head
from v2.common import LABELS, TEST, TRAIN, VAL, write_predictions
from v2.evaluate import auroc

SMALL = Recipe(width=32, heads=2, steer_hidden=16, lr=3e-3, batch_size=64, max_epochs=60, patience=15)


def steering_inputs(n=2400, cells=6, seed=0) -> Inputs:
    """
    Each study has `cells` candidate image cells, each bright or not at random. The text names one cell, and
    every label is whether *that* cell is bright. Text alone says nothing, the image alone only a little:
    solving it takes reading the image where the text points.
    """
    rng = np.random.default_rng(seed)
    bright = rng.random((n, cells)) < 0.5
    named = rng.integers(0, cells, n)
    tokens = rng.normal(0, 1, (n, 1 + cells + 2, 4))
    tokens[:, 1:1 + cells, 0] = bright + rng.normal(0, 0.1, (n, cells))
    text = np.eye(cells)[named]
    text = ((text - text.mean(0)) / text.std(0)).astype(np.float32)
    y = bright[np.arange(n), named].astype(float)
    split = np.array([TRAIN] * (n // 2) + [VAL] * (n // 4) + [TEST] * (n - n // 2 - n // 4))
    raw = np.repeat(y[:, None], len(LABELS), axis=1)
    mean, std = tokens.reshape(-1, 4).mean(0), tokens.reshape(-1, 4).std(0)
    return Inputs(np.arange(n), tokens.astype(np.float16), split, raw, text, {}, mean.astype(np.float32),
                  std.astype(np.float32), {})


def test_h2_starts_as_exactly_h1():
    inputs = steering_inputs(n=64)
    h1, h2 = build_head("h1", inputs, SMALL, seed=3), build_head("h2", inputs, SMALL, seed=3)
    rows = np.arange(16)
    torch.testing.assert_close(torch.as_tensor(predict_head(h1, inputs, rows, "cpu")),
                               torch.as_tensor(predict_head(h2, inputs, rows, "cpu")))
    assert sum(p.numel() for p in h2.parameters()) > sum(p.numel() for p in h1.parameters())


def test_text_steered_attention_reads_the_region_the_text_names():
    inputs = steering_inputs()
    test = inputs.rows(TEST)
    scores = {}
    for arm in ("h1", "h2"):
        model, info = fit_head(arm, inputs, 0, SMALL, "cpu")
        scores[arm] = auroc(inputs.raw[test, 0], predict_head(model, inputs, test, "cpu")[:, 0])
        if arm == "h2":
            shift = attend.attention_shift(model, inputs, test, "cpu")
    assert scores["h2"] > 0.95, scores
    assert scores["h1"] < 0.85 and scores["h2"] - scores["h1"] > 0.1, scores  # late fusion can't do this
    assert shift.mean() > 0.3  # and the text really moves h2's attention


def test_masked_labels_do_not_train_the_head():
    inputs = steering_inputs(n=400)
    inputs.raw[:, 1:] = -1.0  # uncertain: ignored by the v2 policy
    model, info = fit_head("h0", inputs, 0, Recipe(width=16, heads=2, max_epochs=2, patience=1, batch_size=64), "cpu")
    assert info["epochs_run"] >= 1 and np.isfinite(info["best_val_macro_auroc"])


@pytest.fixture(scope="module")
def pixel_tokens(prepared, pixel_embeddings):
    path = embed.tokens_path(prepared, "test-pixels", 48, 12)
    if not path.with_suffix(".json").exists():
        embed.main(["--model", "test-pixels", "--tokens", "12", "--out", str(prepared), "--workers", "0"])
    if not (prepared / "predictions" / "test-pixels_48_probe.parquet").exists():
        probe.main(["--embeddings", str(pixel_embeddings), "--out", str(prepared), "--jobs", "1"])
    return path


def test_attend_end_to_end(bigger):
    tokens = embed.tokens_path(bigger, "test-pixels", 48, 12)
    # A few hundred training studies need many more steps per study than the real recipe gives.
    out = attend.main(["--tokens", str(tokens), "--out", str(bigger), "--seeds", "0,1", "--max-epochs", "40",
                       "--patience", "40", "--batch-size", "16", "--lr", "3e-3", "--n-boot", "20", "--device", "cpu"])
    for arm in attend.ARMS:
        meta = json.loads((bigger / "predictions" / f"test-pixels_48_attn_{arm}.json").read_text())
        assert meta["kind"] == "attend" and meta["arm"] == arm and len(meta["per_seed"]) == 2
    h0 = evaluate._get(evaluate.evaluate_run(bigger, "test-pixels_48_attn_h0", n_boot=20), "test_clean", "v2")
    assert h0["macro_auroc"] >= 0.85, f"the attention head missed the planted signal: {h0['macro_auroc']:.3f}"
    gate = out["gate"]
    assert gate["check_ok"], gate["check"]
    assert gate["cmp"] is not None and gate["verdict"].split(":")[0] in ("PASS", "FAIL")
    report = out["report"].read_text()
    for heading in ("Head validity check", "**G4b**", "Recipe differs", "Reported contrasts", "Per class",
                    "How far the text moves"):
        assert heading in report, heading
    assert out["shift"] is not None and out["shift"].shape == (len(LABELS),)
    assert "**G4b**" in evaluate.summarize(bigger, n_boot=20)


def test_tiny_val_withholds_g4b(prepared, synth, pixel_tokens):
    """With too few val studies to score, the head check can't run, so G4b is withheld rather than read."""
    from v2 import context

    context.main(["--out", str(prepared), "--reports", str(synth.reports), "--mimic-iv-ed", str(synth.mimic_iv_ed),
                  "--mimic-iv", str(synth.mimic_iv), "--mimic-cxr-jpg", str(synth.mimic_root)])
    out = attend.main(["--tokens", str(pixel_tokens), "--out", str(prepared), "--seeds", "0", "--max-epochs", "3",
                       "--n-boot", "20", "--device", "cpu"])
    assert not out["gate"]["check_ok"] and out["gate"]["verdict"].startswith("WITHHELD")
    assert "held back" in out["report"].read_text()


def test_finished_seeds_are_reused_and_stale_ones_retrained(bigger, monkeypatch):
    tokens = embed.tokens_path(bigger, "test-pixels", 48, 12)
    args = ["--tokens", str(tokens), "--out", str(bigger), "--arms", "h0", "--seeds", "0", "--max-epochs", "2",
            "--n-boot", "20", "--device", "cpu"]
    attend.main(args)
    first = (bigger / "predictions" / "test-pixels_48_attn_h0.parquet").read_bytes()
    calls = []
    real = attend.fit_head
    monkeypatch.setattr(attend, "fit_head", lambda *a, **k: calls.append(1) or real(*a, **k))
    attend.main(args)                                   # same inputs and recipe: nothing retrains
    assert calls == [] and (bigger / "predictions" / "test-pixels_48_attn_h0.parquet").read_bytes() == first
    attend.main([*args, "--max-epochs", "3"])          # another recipe: retrains
    assert calls == [1]


def test_tokens_must_match_the_embeddings(prepared, pixel_tokens, pixel_embeddings, tmp_path):
    inputs = attend.load_inputs(prepared, pixel_tokens, None)
    assert attend.check_alignment(inputs, pixel_embeddings) < 1e-2
    ids, emb, meta = embed.load_npz(pixel_embeddings)
    shuffled = tmp_path / "shuffled.npz"
    embed.save_npz(shuffled, ids, emb[np.random.default_rng(0).permutation(len(ids))], meta)
    with pytest.raises(SystemExit, match="same images"):
        attend.check_alignment(inputs, shuffled)


def test_g4b_is_withheld_when_the_head_fails_its_check(prepared, pixel_tokens):
    from v2.common import load_manifest, read_predictions

    good, _ = read_predictions(prepared / "predictions" / "test-pixels_48_probe.parquet")
    rng = np.random.default_rng(0)
    for name, scores in (("probe", good[list(LABELS)].to_numpy()),
                         *((f"attn_{a}", rng.normal(size=(len(good), len(LABELS)))) for a in attend.ARMS)):
        write_predictions(prepared, f"broken_{name}", good["study_id"], good["split"], scores, {"kind": "test"})
    gate = evaluate.g4b(prepared, "broken", n_boot=20)
    assert not gate["check_ok"] and gate["verdict"].startswith("WITHHELD") and gate["cmp"] is None
    for path in (prepared / "predictions").glob("broken_*"):
        path.unlink()
    for path in (prepared / "reports").glob("broken_*"):
        path.unlink()
    assert load_manifest(prepared) is not None
