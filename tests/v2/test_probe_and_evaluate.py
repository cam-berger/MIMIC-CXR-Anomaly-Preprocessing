"""M1 probes and the evaluation protocol: AUROC, patient bootstrap, paired comparisons, reports."""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from v2 import evaluate, probe
from v2.common import LABELS, TEST, VAL, label_matrix, load_manifest, targets_and_mask, write_predictions
from v2.evaluate import auroc, bootstrap_indices, compare_runs, evaluate_run


# --- metrics ---------------------------------------------------------------

def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.random(200) < 0.3
        s = rng.integers(0, 10, 200).astype(float)  # many ties
        assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auroc_known_answers():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert auroc(y, np.array([0.4, 0.3, 0.2, 0.1])) == 0.0
    assert auroc(y, np.zeros(4)) == 0.5
    assert np.isnan(auroc(np.ones(4), np.arange(4.0)))


def test_bootstrap_resamples_whole_patients():
    groups = np.array([1, 1, 1, 2, 3, 3])
    boots = bootstrap_indices(groups, 200, seed=0)
    for idx in boots:
        picked = groups[idx]
        for g, n in zip(*np.unique(picked, return_counts=True)):
            assert n % (groups == g).sum() == 0  # a patient enters with all their studies
    again = bootstrap_indices(groups, 200, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(boots, again))


def test_ci_narrows_with_more_patients():
    rng = np.random.default_rng(1)

    def width(n):
        y = rng.random(n) < 0.4
        s = y + rng.normal(0, 1.0, n)
        vals = [auroc(y[i], s[i]) for i in bootstrap_indices(np.arange(n), 300, 0)]
        lo, hi = evaluate.ci(np.array(vals))
        assert lo <= auroc(y, s) <= hi
        return hi - lo

    assert width(2000) < width(100)


# --- probe -------------------------------------------------------------------

def test_probe_recovers_a_planted_signal():
    rng = np.random.default_rng(0)
    n, d = 600, 20
    raw = np.where(rng.random((n, len(LABELS))) < 0.3, 1.0, np.nan)
    X = rng.normal(size=(n, d))
    X[:, : len(LABELS)] += 2.0 * (raw == 1.0)
    p = probe.fit_probe(X[:400], raw[:400], X[400:500], raw[400:500], n_jobs=1)
    scores = p.predict(X[500:])
    y, m = targets_and_mask(raw[500:])
    aucs = [auroc(y[m[:, k], k], scores[m[:, k], k]) for k in range(len(LABELS))]
    assert min(aucs) > 0.85
    assert all(f.C in probe.DEFAULT_CS for f in p.fits)


def test_probe_handles_a_label_with_one_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    raw = np.where(rng.random((100, len(LABELS))) < 0.5, 1.0, 0.0)
    raw[:, 5] = 0.0  # Fracture: no positives
    p = probe.fit_probe(X[:80], raw[:80], X[80:], raw[80:], n_jobs=1)
    assert "single class" in p.fits[5].note
    assert np.all(p.predict(X)[:, 5] == 0)


def test_split_xy_aligns_shuffled_embeddings(prepared):
    m = load_manifest(prepared)
    ids = m["study_id"].to_numpy()
    emb = ids[:, None].astype(np.float32) * np.ones((1, 3), np.float32)
    perm = np.random.default_rng(0).permutation(len(ids))
    parts = probe.split_xy(m, ids[perm], emb[perm])
    for rows, X in parts.values():
        np.testing.assert_array_equal(X[:, 0], rows["study_id"].to_numpy(np.float32))


# --- end to end on the synthetic tree ---------------------------------------------

def test_probe_and_report_on_planted_signal(prepared, pixel_embeddings):
    probe.main(["--embeddings", str(pixel_embeddings), "--out", str(prepared), "--jobs", "1"])
    rep = evaluate_run(prepared, "test-pixels_48_probe", n_boot=100)
    by_set = {(r["test_set"], r["metric"]): r for r in rep["results"]}
    assert by_set[("test_clean", "v2")]["macro_auroc"] > 0.9
    assert by_set[("test_v1", "v1")]["macro_auroc"] > 0.9
    res = by_set[("test_clean", "v2")]
    lo, hi = res["macro_auroc_ci"]
    assert lo <= res["macro_auroc"] <= hi
    frozen = json.loads((prepared / "measurable_classes.json").read_text())
    assert res["classes"] == frozen["test_clean:v2"]
    assert (prepared / "reports" / "test-pixels_48_probe.md").exists()


def _fake_run(prepared, name, noise, seed=0):
    m = load_manifest(prepared)
    rows = m[m["split"].isin([VAL, TEST])]
    y, _ = targets_and_mask(label_matrix(rows))
    s = y + np.random.default_rng(seed).normal(0, noise, y.shape)
    write_predictions(prepared, name, rows["study_id"], rows["split"], s, {"kind": "probe"})


def test_paired_comparison(prepared):
    _fake_run(prepared, "good", 0.3)
    _fake_run(prepared, "bad", 3.0)
    same = compare_runs(prepared, "good", "good", n_boot=100)
    assert same["diff"] == 0 and same["diff_ci"] == [0.0, 0.0] and not same["a_better"]
    res = compare_runs(prepared, "good", "bad", n_boot=100)
    assert res["diff"] > 0 and res["a_better"] and not res["b_better"]


def test_v1_metric_uses_fewer_negatives(prepared):
    _fake_run(prepared, "any", 1.0)
    rep = evaluate_run(prepared, "any", n_boot=20)
    v2 = next(r for r in rep["results"] if r["test_set"] == "test_v1" and r["metric"] == "v2")
    v1 = next(r for r in rep["results"] if r["test_set"] == "test_v1" and r["metric"] == "v1")
    assert sum(c["n_neg"] for c in v1["per_class"]) < sum(c["n_neg"] for c in v2["per_class"])
    assert [c["n_pos"] for c in v1["per_class"]] == [c["n_pos"] for c in v2["per_class"]]


def test_missing_predictions_are_counted(prepared):
    m = load_manifest(prepared)
    rows = m[m["split"] == TEST].iloc[:-5]
    write_predictions(prepared, "partial", rows["study_id"], rows["split"],
                      np.zeros((len(rows), len(LABELS))), {"kind": "probe"})
    rep = evaluate_run(prepared, "partial", n_boot=10)
    total_missing = sum(r["n_missing_predictions"] for r in rep["results"] if r["test_set"] != "val")
    assert total_missing >= 5


def test_summary_has_gates_and_log_rows(prepared):
    _fake_run(prepared, "ft_like", 0.2)
    pred = prepared / "predictions" / "ft_like.json"
    meta = json.loads(pred.read_text())
    pred.write_text(json.dumps({**meta, "kind": "finetune"}))
    text = evaluate.summarize(prepared, n_boot=50)
    assert "**G1**" in text and "**G2**" in text and "| M3 |" in text


def test_edge_note_flags_grid_edges_but_not_ties():
    Cs = probe.DEFAULT_CS
    falling = {c: 0.8 - 0.01 * i for i, c in enumerate(Cs)}
    rising = {c: 0.7 + 0.01 * i for i, c in enumerate(Cs)}
    flat = {c: 0.75 for c in Cs}
    assert "smallest" in probe.edge_note(Cs, falling, Cs[0])
    assert "largest" in probe.edge_note(Cs, rising, Cs[-1])
    assert probe.edge_note(Cs, flat, Cs[0]) == ""  # a tie picks the smallest C by design
    assert probe.edge_note(Cs, falling, Cs[2]) == ""


def test_gates_ignore_flagged_encoders_and_exposure_check_is_reported(prepared, monkeypatch):
    monkeypatch.setattr(evaluate, "MIN_COUNT_VAL", 2)  # the small synthetic val split needs a lower bar
    m = load_manifest(prepared)
    rows = m[m["split"].isin([VAL, TEST])]
    y, _ = targets_and_mask(label_matrix(rows))
    rng = np.random.default_rng(5)
    for run, noise, encoder in (("flagged_best", 0.1, "medsiglip"), ("clean_probe", 0.8, "rad-dino")):
        write_predictions(prepared, run, rows["study_id"], rows["split"], y + rng.normal(0, noise, y.shape),
                          {"kind": "probe", "encoder": encoder})
    text = evaluate.summarize(prepared, n_boot=30, runs=["flagged_best", "clean_probe"])
    g1 = next(line for line in text.splitlines() if line.startswith("- **G1**"))
    assert "`clean_probe`" in g1 and "flagged encoders excluded" in g1
    assert " †" in text and "Exposure check" in text
    rep = json.loads((prepared / "reports" / "clean_probe.json").read_text())
    assert rep["exposure_check"]["classes"] == json.loads((prepared / "measurable_classes.json").read_text())["test_clean:v2"]
