"""Label policy, measurable classes, paths and the prediction-file format."""

import numpy as np
import pytest

from v2.common import (
    LABELS,
    label_counts,
    load_dotenv,
    measurable,
    mimic_csv,
    read_predictions,
    targets_and_mask,
    write_predictions,
)


def test_labels_match_v1_order():
    from src.datasets.mimic_cxr import MIMICCXRLoader

    assert list(LABELS) == MIMICCXRLoader.PATHOLOGY_LABELS


def test_v2_policy_counts_blank_as_negative_and_ignores_uncertain():
    raw = np.array([[1.0, 0.0, -1.0, np.nan]])
    y, m = targets_and_mask(raw, "v2")
    assert y.tolist() == [[1, 0, 0, 0]]
    assert m.tolist() == [[True, True, False, True]]


def test_v1_policy_ignores_blank_and_uncertain():
    raw = np.array([[1.0, 0.0, -1.0, np.nan]])
    y, m = targets_and_mask(raw, "v1")
    assert y.tolist() == [[1, 0, 0, 0]]
    assert m.tolist() == [[True, True, False, False]]


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        targets_and_mask(np.zeros((1, 1)), "v3")


def test_measurable_needs_min_positives_and_negatives():
    raw = np.full((100, len(LABELS)), np.nan)
    raw[:30, 0] = 1.0           # 30 pos, 70 neg (blank) -> measurable
    raw[:29, 1] = 1.0           # 29 pos -> not measurable
    raw[:, 2] = 1.0             # 100 pos, 0 neg -> not measurable
    raw[:40, 3] = 1.0
    raw[40:, 3] = -1.0          # 40 pos, 0 neg (uncertain ignored) -> not measurable
    assert measurable(raw, "v2", 30) == ["Atelectasis"]
    counts = label_counts(raw, "v2").set_index("label")
    assert counts.loc["Edema", "ignored"] == 60


def test_predictions_roundtrip(tmp_path):
    ids = np.array([5, 3, 9])
    scores = np.random.default_rng(0).normal(size=(3, len(LABELS)))
    path = write_predictions(tmp_path, "r", ids, ["val", "test", "test"], scores, {"kind": "probe"})
    wide, meta = read_predictions(path)
    assert meta["kind"] == "probe" and meta["n_studies"] == 3
    wide = wide.set_index("study_id")
    for i, sid in enumerate(ids):
        np.testing.assert_allclose(wide.loc[sid, list(LABELS)].to_numpy(float), scores[i])
    assert wide.loc[5, "split"] == "val"


@pytest.mark.parametrize("bad", ["shape", "nan", "duplicate"])
def test_predictions_are_validated(tmp_path, bad):
    ids, scores = np.array([1, 2]), np.zeros((2, len(LABELS)))
    if bad == "shape":
        scores = np.zeros((2, 3))
    elif bad == "nan":
        scores[0, 0] = np.nan
    else:
        ids = np.array([1, 1])
    with pytest.raises(ValueError):
        write_predictions(tmp_path, "r", ids, ["val", "val"], scores)


def test_mimic_csv_prefers_gz_and_falls_back_to_csv(tmp_path):
    (tmp_path / "mimic-cxr-2.0.0-split.csv").write_text("x")
    assert mimic_csv(tmp_path, "split").name == "mimic-cxr-2.0.0-split.csv"
    (tmp_path / "mimic-cxr-2.0.0-split.csv.gz").write_text("x")
    assert mimic_csv(tmp_path, "split").name == "mimic-cxr-2.0.0-split.csv.gz"


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nV2_TEST_A=from_file\nexport V2_TEST_B='quoted'\n")
    monkeypatch.setenv("V2_TEST_A", "from_env")
    monkeypatch.delenv("V2_TEST_B", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["V2_TEST_A"] == "from_env"
    assert os.environ["V2_TEST_B"] == "quoted"
    monkeypatch.delenv("V2_TEST_B")


def test_mention_only_labels_bias_text_gains_toward_zero():
    """
    §6 M4b amendment. A pre-imaging question about X makes the report address X either way. Under the v1
    policy (mentioned findings only) that selection makes an informative question look useless or worse; under
    v2 (unmentioned = negative) it cancels, and the question scores what it is worth against the truth.
    """
    from v2.evaluate import auroc

    rng = np.random.default_rng(0)
    n = 200_000
    disease = rng.random(n) < 0.2
    asked = np.where(disease, rng.random(n) < 0.45, rng.random(n) < 0.25)     # real but modest suspicion
    explicit_negative = ~disease & (rng.random(n) < np.where(asked, 0.9, 0.1))  # asked -> "no X"
    raw = np.where(disease, 1.0, np.where(explicit_negative, 0.0, np.nan))[:, None]
    score = asked + rng.normal(0, 1e-6, n)
    truth = auroc(disease, score)
    got = {}
    for policy in ("v2", "v1"):
        y, m = targets_and_mask(raw, policy)
        got[policy] = auroc(y[m[:, 0], 0], score[m[:, 0]])
    assert abs(got["v2"] - truth) < 0.01
    assert got["v1"] < 0.45 < truth  # informative text looks worse than useless under v1
