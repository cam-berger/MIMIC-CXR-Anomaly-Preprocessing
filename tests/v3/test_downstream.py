"""v3.5: Jev's labels as training targets for v2's probe (synthetic data, fake Jev, no network)."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from v2.common import LABELS
from v3 import downstream, jev, label, smoke


def client_for(fake):
    return jev.JevClient(fake.model, api_key="test", transport=fake, sleep=lambda s: None)


def test_soft_targets_of_0_and_1_reproduce_the_ordinary_fit():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 5))
    y = (Z[:, 0] + rng.normal(size=300) > 0).astype(float)
    fit = downstream._fit_soft_label("x", Z, y, Z, y, np.ones(300, bool), [1.0], 2000)
    hard = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    assert np.allclose(fit.coef, hard.coef_[0], atol=1e-3) and abs(fit.intercept - hard.intercept_[0]) < 1e-3


def test_soft_targets_move_the_fit_toward_uncertainty():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(400, 3))
    y = (Z[:, 0] > 0).astype(float)
    sharp = downstream._fit_soft_label("x", Z, y, Z, y, np.ones(400, bool), [10.0], 2000)
    soft = downstream._fit_soft_label("x", Z, 0.25 + 0.5 * y, Z, y, np.ones(400, bool), [10.0], 2000)
    assert 0 < soft.coef[0] < sharp.coef[0]


def test_jev_targets_use_the_labeler_coding_and_the_v3_3_score():
    cache = {7: {"choices": ["present", "uncertain", "absent"] + ["absent"] * 9,
                 "probs": [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.0, 0.1, 0.9]] + [[0.0, 0.0, 1.0]] * 9}}
    raw, soft = downstream.jev_targets(cache, np.array([7]))
    assert list(raw[0, :3]) == [1.0, -1.0, 0.0]
    assert np.allclose(soft[0, :3], [0.85, 0.5, 0.05])


def test_too_many_unlabeled_studies_stop_the_run():
    with pytest.raises(SystemExit, match="no Jev label"):
        downstream.answered({1: {}}, np.arange(1, 11), "training")
    assert list(downstream.answered({i: {} for i in range(1, 200)}, np.arange(1, 201), "training")) == list(range(1, 200))


def test_the_three_probes_train_on_the_same_studies_and_are_scored(v3out, v3_base, synth_sources):
    reports, labels = synth_sources
    fake = smoke.FakeJev()
    label.run(v3out, "train", client_for(fake), reports, None, limit=60)
    label.run(v3out, "val", client_for(fake), reports, None)
    emb = Path(json.loads((v3out / "v3_setup.json").read_text())["bench"]) / "embeddings" / "test-pixels_48.npz"
    out = downstream.main(["--model", fake.model, "--labels", str(labels), "--embeddings", str(emb),
                           "--n-train", "60", "--n-jobs", "1", "--n-boot", "20", "--out", str(v3out)])
    assert out["runs"] == list(downstream.ARMS)  # no rad-dino reference on synthetic data
    assert out["facts"]["n_train"] == 60 and out["facts"]["unanswered"] == {"train": 0, "val": 0}
    for run in downstream.ARMS:
        meta = json.loads((v3out / "predictions" / f"{run}.json").read_text())
        assert meta["n_train"] == 60 and meta["milestone"] == "v3.5" and set(meta["per_label"]) == set(LABELS)
    text = out["report"].read_text()
    assert "Gate L2" in text and "Against the radiologist" in text and "favours `probe20k_chexpert`" in text
