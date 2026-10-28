"""v3.8: Jev's labels on the whole training split for the main fine-tune (synthetic data, a tiny encoder; no network)."""

import json

import numpy as np
import pandas as pd
import pytest

from v2.common import LABELS, TEST, TRAIN, VAL, label_matrix, load_manifest
from v3 import label, retrain, smoke

from .test_chexpert import client_for

LO, AT = LABELS.index("Lung Opacity"), LABELS.index("Atelectasis")
TINY = smoke.TINY_FT


@pytest.fixture
def labeled(v3out, synth_sources):
    """Every training study and every val study labeled by the fake Jev, and the targets file built from them."""
    reports, _ = synth_sources
    fake = smoke.FakeJev()
    label.run(v3out, "train", client_for(fake), reports, None, every=True)
    label.run(v3out, "val", client_for(fake), reports, None)
    return retrain.build_targets(v3out, fake.model), fake.model


def test_soft_values_pass_the_label_policy_and_the_rest_follows_v2():
    raw = np.array([[0.37, 1.0, 0.0, np.nan, -1.0, 0.99]])
    y, m = retrain.soft_targets_and_mask(raw)
    assert np.allclose(y, [[0.37, 1.0, 0.0, 0.0, 0.0, 0.99]]) and m.tolist() == [[True, True, True, True, False, True]]
    with pytest.raises(ValueError):
        retrain.soft_targets_and_mask(raw, "u0")


def test_the_whole_training_split_is_labeled_and_needs_all_or_limit(v3out, synth_sources, labeled):
    facts, model = labeled
    n_train = int((load_manifest(v3out).drop_duplicates("study_id")["split"] == TRAIN).sum())
    assert (facts[TRAIN]["studies"], facts[TRAIN]["answered"], facts[TRAIN]["unanswered"]) == (n_train, n_train, 0)
    assert (v3out / "reports" / "v3_label_train_all.md").read_text().startswith("# v3.8 labels")
    fake = smoke.FakeJev()
    for kwargs in ({}, {"limit": 10, "every": True}):
        with pytest.raises(SystemExit, match="--limit"):
            label.run(v3out, "train", client_for(fake), synth_sources[0], None, **kwargs)
    with pytest.raises(SystemExit, match="--all is for"):
        label.main(["run", "--split", "val", "--all", "--model", model, "--out", str(v3out),
                    "--reports", str(synth_sources[0])])


def test_too_many_unanswered_studies_stop_the_targets(v3out, labeled):
    _, model = labeled
    path = label.cache_path(v3out, model)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[: len(lines) // 2]) + "\n")
    with pytest.raises(SystemExit, match="no Jev label"):
        retrain.build_targets(v3out, model)


def test_each_arm_gets_its_own_targets_and_v2_finetune_sees_them(v3out, labeled):
    targets = retrain.load_targets(v3out)
    manifest = load_manifest(v3out)
    t = targets.set_index("study_id")
    for arm in retrain.ARMS:
        m = retrain.arm_manifest(manifest, targets, arm)
        assert np.array_equal(label_matrix(m[m["split"] == TEST]), label_matrix(manifest[manifest["split"] == TEST]),
                              equal_nan=True)  # test labels are never touched
        tr, va = m[m["split"] == TRAIN], m[m["split"] == VAL]
        assert set(tr["study_id"]) == set(t.index[t["split"] == TRAIN]) and len(va) == int((t["split"] == VAL).sum())
        if arm == "jev_soft":
            assert np.allclose(label_matrix(tr), t.loc[tr["study_id"], retrain._cols("soft")].to_numpy())
            assert np.array_equal(label_matrix(va), t.loc[va["study_id"], retrain._cols("jev")].to_numpy())
        elif arm == "chexpert_hier":
            raw = label_matrix(tr)
            assert (raw[raw[:, AT] == 1.0, LO] == 1.0).all()
    from v2 import finetune as ft

    m = retrain.arm_manifest(manifest, targets, "jev_soft")
    before = ft.load_manifest, ft.targets_and_mask
    with retrain.arm_targets(ft, m):
        tr = ft.load_manifest(v3out)
        tr = tr[tr["split"] == TRAIN]
        y, mask = ft.targets_and_mask(label_matrix(tr), "v2")
        assert mask.all() and np.allclose(y, t.loc[tr["study_id"], retrain._cols("soft")].to_numpy())
    assert (ft.load_manifest, ft.targets_and_mask) == before  # v2 is left as it was


def test_six_fine_tunes_are_applied_to_the_external_set_and_reported(v3out, tmp_path, labeled, synth_sources):
    retrain.gpu(v3out, extra=TINY)
    for run in retrain.all_runs():
        side = json.loads((v3out / "runs" / run / "v3_targets.json").read_text())
        assert side["targets_sha256"] == json.loads(retrain.targets_path(v3out).with_suffix(".json").read_text())["sha256"]
        meta = json.loads((v3out / "predictions" / f"{run}.json").read_text())
        assert meta["milestone"] == "v3.8" and meta["config"]["run"] == run
    assert (v3out / "predictions" / "ft213k_jev_soft_s0_smoke.parquet").exists()
    assert retrain.gpu(v3out, extra=TINY) == []  # every run already has predictions
    ext = smoke.stand_in_external(v3out, tmp_path)
    assert sorted(retrain.apply(v3out, ext, workers=0, device="cpu")) == sorted(retrain.all_runs())
    chex = v3out / "chexpert"
    assert (chex / "images").is_symlink() and not (ext / "predictions").exists()  # nothing written into EXT_OUT
    out = retrain.main(["report", "--ext", str(ext), "--labels", str(synth_sources[1]), "--n-boot", "20",
                        "--out", str(v3out)])
    assert out["seeds"] == [0, 1] and set(out["gates"]) == {"L4", "L5"}
    assert all(any(g.startswith(v) for v in ("PASS", "FAIL", "no detectable", "n/a")) for g in out["gates"].values())
    assert any(f.startswith("RECIPE") for f in out["flags"])  # the tiny encoder isn't M5b's recipe
    assert not any(f.startswith(("TARGETS", "COVERAGE")) for f in out["flags"])
    assert all(r["correlation"] >= 0.99 for r in out["reload"].values())
    text = out["report"].read_text()
    assert "**Gate L4:**" in text and "**Gate L5:**" in text and "**Reading:**" in text and "## Runs" in text


def test_the_report_waits_for_both_seeds(v3out, tmp_path, labeled, synth_sources):
    retrain.gpu(v3out, seeds=(0,), extra=TINY)
    ext = smoke.stand_in_external(v3out, tmp_path)
    retrain.apply(v3out, ext, workers=0, device="cpu")
    out = retrain.main(["report", "--ext", str(ext), "--labels", str(synth_sources[1]), "--n-boot", "20",
                        "--out", str(v3out)])
    assert all(g.startswith("PENDING") for g in out["gates"].values()) and out["reading"].startswith("pending")


def test_the_cli_keeps_the_effective_batch():
    for args in (["--batch-size", "8"], ["--batch-size", "8", "--accum", "2"]):
        with pytest.raises(SystemExit):
            retrain.main(["train", "--arm", "jev_soft", "--seed", "0", *args])


def test_the_reading_follows_the_gates():
    assert retrain.reading("PASS", "PASS").startswith("Jev's labels make a better")
    assert retrain.reading("PASS", "FAIL").startswith("Jev's edge is in matching")
    assert retrain.reading("no detectable difference", "PASS").startswith("Jev's labels improve")
    assert retrain.reading("FAIL", "no detectable difference").startswith("v3.7's edge doesn't hold")
    assert retrain._gate([{"diff": 0.01, "a_better": True, "b_better": False}] * 2, 2) == "PASS"
    assert retrain._gate([{"diff": 0.01, "a_better": True, "b_better": False},
                          {"diff": 0.0, "a_better": False, "b_better": False}], 2).startswith("no detectable")


def test_targets_report_holds_aggregates_only(v3out, labeled):
    facts, _ = labeled
    text = retrain.write_targets_report(v3out, facts).read_text()
    assert "Mean training target per finding" in text
    assert not any(str(s) in text for s in pd.read_parquet(retrain.targets_path(v3out))["study_id"].head(50))
