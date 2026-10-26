"""v3.4: the hierarchy rule and cross-fitted cut-offs for Jev's report labels (synthetic data, no network)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from v2.common import LABELS
from v3 import bridge, smoke

LO, AT, ECM, CM = (LABELS.index(c) for c in ("Lung Opacity", "Atelectasis", "Enlarged Cardiomediastinum", "Cardiomegaly"))


def frame(rows):
    return pd.DataFrame([{label: row.get(label, np.nan) for label in LABELS} for row in rows])


def test_hierarchy_rates_count_parents_among_child_positives():
    df = frame([{"Atelectasis": 1, "Lung Opacity": 1}, {"Edema": 1, "Lung Opacity": 1}, {"Edema": 1},
                {"Lung Opacity": 1}, {"Cardiomegaly": 1, "Enlarged Cardiomediastinum": 0}])
    rates = bridge.hierarchy_rates(df)
    assert rates["Lung Opacity"]["n_child_positive"] == 3
    assert rates["Lung Opacity"]["parent_given_child"] == 2 / 3
    assert rates["Lung Opacity"]["per_child"]["Atelectasis"] == 1.0 and rates["Lung Opacity"]["per_child"]["Edema"] == 0.5
    assert rates["Enlarged Cardiomediastinum"]["parent_given_child"] == 0.0


def test_the_rule_raises_a_parent_to_its_highest_child_only():
    s = np.zeros((2, 12))
    s[0, AT], s[1, LO], s[1, CM] = 1.0, 0.5, 0.5
    out = bridge.apply_hierarchy(s, ["Lung Opacity"])
    assert out[0, LO] == 1.0 and out[1, LO] == 0.5 and out[1, ECM] == 0.0  # ECM not listed: untouched
    assert (s[0, LO] == 0.0) and np.array_equal(np.delete(out, LO, axis=1), np.delete(s, LO, axis=1))


def test_cut_offs_separate_a_separable_score():
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1])
    lo, hi = bridge.fit_cuts(s, y)
    assert set(bridge.three_level(s[:3], lo, hi)) <= {0.0, 0.5} and set(bridge.three_level(s[3:], lo, hi)) <= {0.5, 1.0}
    assert bridge.three_level(s, lo, hi)[:3].max() < bridge.three_level(s, lo, hi)[3:].min()
    assert bridge.fit_cuts(s, np.ones(6)) is None


def test_each_fold_is_labeled_by_cut_offs_fit_on_the_other_folds():
    rng = np.random.default_rng(0)
    n = 200
    groups = np.repeat(np.arange(50), 4)
    Y = (rng.random((n, 12)) < 0.3).astype(float)
    S = np.clip(Y * 0.4 + rng.random((n, 12)) * 0.6, 0, 1)
    M = np.ones((n, 12), dtype=bool)
    fallback = np.full((n, 12), 0.5)
    out, info = bridge.crossfit_cuts(S, fallback, Y, M, groups)
    train, held = next(GroupKFold(n_splits=bridge.FOLDS).split(S, groups=groups))
    lo, hi = bridge.fit_cuts(S[train, 0], Y[train, 0])
    assert np.array_equal(out[held, 0], bridge.three_level(S[held, 0], lo, hi))
    assert info["folds"] == bridge.FOLDS and set(np.unique(out)) <= {0.0, 0.5, 1.0}


def test_the_bridge_runs_on_the_synthetic_labeler_and_reports(v3out, v3_base, synth_sources):
    assert smoke.check_labels(v3out, v3_base.parent, smoke.FakeJev()) == []
    out = bridge.main(["--out", str(v3out), "--labels", str(synth_sources[1]), "--n-boot", "20"])
    expected_parents = [p for p, r in out["train_rates"].items() if r["parent_given_child"] >= bridge.HIERARCHY_MIN]
    assert out["parents"] == expected_parents
    assert out["new_runs"] == (["labeler_jev_cut", "labeler_jev_hier", "labeler_jev_cut_hier"] if expected_parents
                               else ["labeler_jev_cut"])
    for run in out["new_runs"]:
        assert (v3out / "predictions" / f"{run}.parquet").exists()
    text = out["report"].read_text()
    assert "The hierarchy, decided on the training split" in text and "Lung Opacity: Jev's answer" in text
