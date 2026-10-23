"""M7: external validation. A radiologist's labels on the official test split (M7a); the CheXpert test set (M7b)."""

import json
import shutil

import pandas as pd
import pytest

from v2 import evaluate, external, finetune
from v2.common import LABELS, TEST, load_manifest, load_measurable, read_predictions, write_predictions
from v2.synth import make_chexpert_test

P = "test-pixels_48"
FT = ("tiny-dinov2_32_ft_s0", "tiny-dinov2_32_ft_attn_s0")


@pytest.fixture(scope="module")
def m7(bench, tmp_path_factory):
    """A copy of M5's benchmark directory with two tiny fine-tunes (both heads), as M5b's GPU stage makes them."""
    out = tmp_path_factory.mktemp("v2m7") / "bench"
    shutil.copytree(bench, out)
    args = ["--model", "tiny-dinov2", "--epochs", "2", "--batch-size", "16", "--accum", "1", "--encoder-lr", "1e-3",
            "--head-lr", "1e-2", "--llrd", "1.0", "--steps-per-epoch", "4", "--workers", "0", "--device", "cpu",
            "--out", str(out)]
    finetune.main(args)
    finetune.main([*args, "--head", "attention"])
    return out


@pytest.fixture(scope="module")
def chexpert(tmp_path_factory):
    return make_chexpert_test(tmp_path_factory.mktemp("chexpert"), n_patients=150, seed=7)


def test_compare_labels_is_zero_for_the_same_labels_and_sees_a_flip(bench):
    pred, _ = read_predictions(bench / "predictions" / f"{P}_probe.parquet")
    m, frozen = load_manifest(bench), load_measurable(bench)
    same = evaluate.compare_labels(pred, m, m, frozen, n_boot=20)
    assert same["diff"] == 0 and same["diff_ci"] == [0.0, 0.0] and same["classes"]
    k = same["classes"][0]
    flipped = m.assign(**{k: (m[k] != 1.0).astype(float)})  # positives and the rest swap places
    c = evaluate.compare_labels(pred, m, flipped, {"test_clean:u0": [k]}, metric="u0", n_boot=20)
    a = next(p for p in c["per_class"] if p["label"] == k)
    assert a["b"] == pytest.approx(1 - a["a"]) and c["diff"] > 0  # under positive vs rest, the AUROC inverts


def test_radiologist_report(m7, synth):
    labels = synth.mimic_root / external.RADIOLOGIST_FILE
    res = external.radiologist_report(m7, labels, P, [f"{P}_probe", f"{P}_attn_h0"], f"{P}_attn_h0_stack_indication",
                                      f"{P}_attn_h0", n_boot=20, min_count=5)  # a few dozen synthetic studies
    text = res["path"].read_text()
    assert res["path"] == m7 / "reports" / "m7a_radiologist.md"
    assert res["renamed"] == ["Airspace Opacity"] and "read as Lung Opacity" in text
    assert res["info"]["in_official_test"] > 0 and res["primary"] in (f"{P}_probe", f"{P}_attn_h0")
    for heading in ("The primary image model", "minus the labeler's", "The text gain", "Per class",
                    "Every benchmark run", "Power"):
        assert heading in text, heading
    assert {run for run, _ in res["label_effect"]} == {res["primary"], f"{P}_attn_h0_stack_indication"}
    assert {which for which, _ in res["text_gain"]} == {"labeler", "radiologist"}
    assert all(set(v) <= set(LABELS) for k, v in res["frozen"].items())
    # The labeler's side uses the benchmark's own labels on the same studies.
    m = load_manifest(m7)
    labeler, reader, _ = external.radiologist_manifests(m7, external.read_radiologist(labels)[0])
    cols = list(LABELS)
    assert labeler[cols].equals(m[cols]) and labeler["in_test_clean"].sum() == reader["in_test_clean"].sum()
    assert not reader.loc[reader["in_test_clean"], cols].equals(labeler.loc[labeler["in_test_clean"], cols])


def test_read_radiologist_refuses_files_it_cant_read(tmp_path):
    bad = tmp_path / "labels.csv"
    pd.DataFrame({"study_id": [1], "Cardiomegaly": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit, match="has no column"):
        external.read_radiologist(bad)
    pd.DataFrame({"study_id": [1], **{c: [2.0] for c in LABELS}}).to_csv(bad, index=False)
    with pytest.raises(SystemExit, match="label values"):
        external.read_radiologist(bad)


def test_chexpert_prepare(m7, chexpert, synth, tmp_path):
    out = tmp_path / "ext"
    audit = external.prepare_chexpert(chexpert, out, m7, str(synth.rad_dino_list), workers=0)
    m = load_manifest(out)
    gt = external.read_chexpert_labels(chexpert / "groundtruth.csv")
    assert len(m) == 149 and audit["drops"]["no frontal image"] == 1       # one study has only a lateral image
    assert (m["split"] == TEST).all() and m["in_test_clean"].all() and not m["in_test_v1"].any()
    assert m.loc[m["study_id"] == external.chexpert_study_id(64742, 1), "dicom_id"].item() == "view2_frontal"
    assert m["jpg"].str.contains("/test/").all()                            # the validation images are ignored
    joined = m.merge(gt, on="study_id", suffixes=("", "_gt"))
    assert all((joined[c] == joined[f"{c}_gt"]).all() for c in LABELS)
    assert audit["split_scheme"] == external.EXTERNAL and evaluate.split_scheme(out) == external.EXTERNAL
    assert audit["image_size"] == 48 and all((out / p).exists() for p in m["image"])
    assert audit["rad_dino"]["chexpert_images"] == 5 and audit["rad_dino"]["test_images_in_list"] == 0
    assert set(load_measurable(out)) >= {"test_clean:v2", "test_clean:u0"}
    assert "must be 0" in (out / "audit.md").read_text()


def test_chexpert_prepare_stops_on_rad_dino_overlap(m7, chexpert, synth, tmp_path):
    rad = pd.read_csv(synth.rad_dino_list)
    rad.loc[len(rad)] = ["image-x", "Chexpert/p64745_s1_view1_frontal.png"]  # a test patient
    listed = tmp_path / "training_images.csv"
    rad.to_csv(listed, index=False)
    with pytest.raises(SystemExit, match="RED FLAG: RAD-DINO's training list holds 1 images of 1"):
        external.prepare_chexpert(chexpert, tmp_path / "ext", m7, str(listed), workers=0)


def test_chexpert_prepare_stops_when_the_label_files_disagree_or_images_are_missing(m7, chexpert, synth, tmp_path):
    root = tmp_path / "chexpert"
    shutil.copytree(chexpert, root)
    gt = pd.read_csv(root / "groundtruth.csv")
    gt.loc[0, "Edema"] = 1 - gt.loc[0, "Edema"]
    gt.to_csv(root / "test_labels.csv", index=False)
    with pytest.raises(SystemExit, match="disagree"):
        external.prepare_chexpert(root, tmp_path / "ext", m7, str(synth.rad_dino_list), workers=0)
    labels_only = tmp_path / "labels_only"
    labels_only.mkdir()
    shutil.copy(chexpert / "groundtruth.csv", labels_only)
    with pytest.raises(SystemExit, match="no CheXpert test images"):
        external.prepare_chexpert(labels_only, tmp_path / "ext2", m7, str(synth.rad_dino_list), workers=0)


def test_chexpert_output_needs_its_own_directory(m7, bench):
    with pytest.raises(SystemExit, match="must differ"):
        external.guard_external(m7, m7)
    with pytest.raises(SystemExit, match="give the CheXpert test set its own"):
        external.guard_external(bench, m7)  # another v2 directory: never overwritten


def test_chexpert_end_to_end(m7, chexpert, synth, tmp_path):
    out = tmp_path / "ext"
    runs = [f"{P}_probe", *FT]
    res = external.main(["chexpert", "--chexpert", str(chexpert), "--bench", str(m7), "--out", str(out),
                         "--candidates", ",".join(runs), "--rad-dino-list", str(synth.rad_dino_list),
                         "--workers", "0", "--device", "cpu", "--jobs", "1", "--n-boot", "20"])
    m = load_manifest(out)
    for run in runs:
        pred, meta = read_predictions(out / "predictions" / f"{run}.parquet")
        assert len(pred) == len(m) and meta["milestone"] == "M7" and meta["source_run"] == run
    assert read_predictions(out / "predictions" / f"{P}_probe.parquet")[1]["reproduced_max_diff"] <= 1e-6
    assert all(read_predictions(out / "predictions" / f"{r}.parquet")[1]["reload_check"]["correlation"] >= 0.99
               for r in FT)
    report = res["report"]
    text = report["path"].read_text()
    assert report["primary"] in runs and report["gate"]["verdict"].split(":")[0] in ("PASS", "FAIL")
    assert "**G7**" in text and "CheXzero" in text and "DAM 0.931" in text and "transfer gap" in text
    probe_five = report["results"][f"{P}_probe"]["five"]
    assert probe_five["classes"] == list(external.FIVE) and probe_five["macro_auroc"] > 0.9  # the signal transfers
    assert any(f"`{P}_probe`'s five-finding mean" in f for f in report["flags"])  # a perfect score is a red flag
    summary = evaluate.summarize(out, n_boot=10).splitlines()
    assert any("| M7 |" in line and "CheXpert test" in line and f"| {P}_probe |" in line for line in summary)
    assert any("external test set" in line for line in summary)


def test_the_probe_refit_must_reproduce(m7, chexpert, synth, tmp_path):
    bench = tmp_path / "bench"
    shutil.copytree(m7, bench, ignore=shutil.ignore_patterns("images", "*_tokens*"))
    shutil.copytree(m7 / "images", bench / "images")
    pred, meta = read_predictions(bench / "predictions" / f"{P}_probe.parquet")
    write_predictions(bench, f"{P}_probe", pred["study_id"], pred["split"], pred[list(LABELS)].to_numpy() + 0.01, meta)
    out = tmp_path / "ext"
    external.prepare_chexpert(chexpert, out, bench, str(synth.rad_dino_list), workers=0)
    with pytest.raises(SystemExit, match="RED FLAG: the refit"):
        external.apply_probe(bench, out, f"{P}_probe", workers=0, device="cpu", n_jobs=1)


def test_pick_primary_takes_the_best_val_score(bench):
    best, scores = external.pick_primary(bench, [f"{P}_probe", f"{P}_attn_h0", "missing_run"])
    assert set(scores) == {f"{P}_probe", f"{P}_attn_h0"} and best == max(scores, key=scores.get)
    assert json.dumps(scores)  # plain numbers, for the report
