"""M4b part 3: the per-class combiner fit on val (cross-fitted val scores, fallbacks, end to end)."""

import json

import numpy as np
import pytest

from v2 import evaluate, fuse, stack
from v2.common import LABELS
from v2.evaluate import auroc


def arrays(n_val_patients=80, per_patient=4, n_test=400, seed=0):
    rng = np.random.default_rng(seed)
    n_val = n_val_patients * per_patient
    n = n_val + n_test
    y = (rng.random((n, len(LABELS))) < 0.4).astype(float)
    S_img = rng.normal(size=(n, len(LABELS)))
    S_txt = rng.normal(size=(n, len(LABELS)))
    S_img[:, 0] += 2.5 * y[:, 0]   # class 0: only the image knows
    S_txt[:, 1] += 2.5 * y[:, 1]   # class 1: only the text knows
    groups = np.concatenate([np.repeat(np.arange(n_val_patients), per_patient), 10_000 + np.arange(n_test)])
    is_val = np.arange(n) < n_val
    return S_img, S_txt, y, is_val, groups


def test_stack_uses_whichever_score_is_informative():
    S_img, S_txt, y, is_val, groups = arrays()
    scores, info = stack.stack_scores(S_img, S_txt, y, is_val, groups)
    test = ~is_val
    assert auroc(y[test, 0], scores[test, 0]) > 0.9 and auroc(y[test, 1], scores[test, 1]) > 0.9
    assert info[LABELS[0]]["image_weight"] > 3 * abs(info[LABELS[0]]["text_weight"])
    assert info[LABELS[1]]["text_weight"] > 3 * abs(info[LABELS[1]]["image_weight"])


def test_val_scores_are_cross_fitted():
    """A val study's own label never trains the model that scores it: flipping it leaves its score alone."""
    S_img, S_txt, y, is_val, groups = arrays()
    base, _ = stack.stack_scores(S_img, S_txt, y, is_val, groups)
    flipped = y.copy()
    flipped[5, 0] = 1 - flipped[5, 0]
    again, _ = stack.stack_scores(S_img, S_txt, flipped, is_val, groups)
    assert again[5, 0] == base[5, 0]
    assert not np.allclose(again[is_val, 0], base[is_val, 0])   # other folds did see it
    assert not np.allclose(again[~is_val, 0], base[~is_val, 0])  # and so did the fit that scores test


def test_a_class_without_both_labels_in_val_keeps_the_image_score():
    S_img, S_txt, y, is_val, groups = arrays()
    y[is_val, 2] = 0.0
    scores, info = stack.stack_scores(S_img, S_txt, y, is_val, groups)
    assert info[LABELS[2]]["kept_image_score"]
    np.testing.assert_array_equal(scores[:, 2], S_img[:, 2])


def test_stack_end_to_end(bigger):
    emb = bigger / "embeddings" / "test-pixels_48.npz"
    fuse.main(["--embeddings", str(emb), "--out", str(bigger), "--arms", "image,image_text,text", "--n-boot", "20",
               "--jobs", "1"])
    out = stack.main(["--pair", "test-pixels_48_probe", "test-pixels_48_ctx_text", "--out", str(bigger),
                      "--n-boot", "20"])
    assert out["runs"] == ["test-pixels_48_probe_stack_text"]
    meta = json.loads((bigger / "predictions" / "test-pixels_48_probe_stack_text.json").read_text())
    assert meta["kind"] == "stack" and meta["image_run"] == "test-pixels_48_probe" and meta["n_val"] > 0
    report = out["report"].read_text()
    assert "`test-pixels_48_probe_stack_text` minus `test-pixels_48_ctx_image_text`" in report
    rep = evaluate.evaluate_run(bigger, "test-pixels_48_probe_stack_text", n_boot=20)
    assert evaluate._get(rep, "test_clean", "v2")["macro_auroc"] > 0.85


def test_stack_needs_existing_runs(bigger):
    with pytest.raises(SystemExit, match="not found"):
        stack.main(["--pair", "test-pixels_48_probe", "no_such_run", "--out", str(bigger)])
