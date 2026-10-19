"""M0: frontal-view selection, splits, labels, images, drops and audit checks."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from v2 import prepare_data
from v2.common import LABELS, TEST, TRAIN, VAL, load_manifest
from v2.prepare_data import rad_dino_overlap, select_frontal, study_datetime


def _meta(rows):
    return pd.DataFrame(rows, columns=["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime"])


def test_select_frontal_prefers_pa_then_lowest_dicom_and_never_lateral():
    meta = _meta([
        ("c", 1, 10, "LATERAL", 21500101, 120000.0),
        ("b", 1, 10, "AP", 21500101, 120000.0),
        ("z", 1, 10, "PA", 21500101, 120000.0),   # PA wins over AP even with a higher dicom_id
        ("y", 2, 20, "AP", 21500101, 120000.0),
        ("x", 2, 20, "AP", 21500101, 120000.0),   # tie between APs: lowest dicom_id
        ("l", 3, 30, "LATERAL", 21500101, 120000.0),
        ("m", 3, 30, "LL", 21500101, 120000.0),  # no frontal: study absent
        ("n", 4, 40, None, 21500101, 120000.0),  # missing view: not frontal
        ("o", 5, 50, "AP AXIAL", 21500101, 120000.0),
    ])
    chosen = select_frontal(meta).set_index("study_id")
    assert chosen.loc[10, "dicom_id"] == "z" and chosen.loc[10, "view"] == "PA"
    assert chosen.loc[20, "dicom_id"] == "x"
    assert set(chosen.index) == {10, 20}


def test_study_datetime_restores_leading_zeros():
    dt = study_datetime(pd.Series([21500101, 21500102, 21500103, 21500104]),
                        pd.Series([5014.531, 213014.0, np.nan, 84512.937]))
    assert dt[0] == pd.Timestamp("2150-01-01 00:50:14")
    assert dt[1] == pd.Timestamp("2150-01-02 21:30:14")
    assert pd.isna(dt[2])  # a missing time is not midnight
    assert dt[3] == pd.Timestamp("2150-01-04 08:45:12")


def test_manifest_splits_are_patient_disjoint_and_follow_the_definitions(prepared, synth):
    m = load_manifest(prepared)
    patients = {s: set(m.loc[m["split"] == s, "subject_id"]) for s in (TRAIN, VAL, TEST)}
    assert not patients[TRAIN] & patients[VAL]
    assert not patients[TRAIN] & patients[TEST]
    assert not patients[VAL] & patients[TEST]

    official = m["subject_id"].map(synth.official)
    assert (m["in_test_clean"] == official.isin(["validate", "test"])).all()
    v1_val = set(pd.read_parquet(synth.cohort_dir / "anomalous_val.parquet")["study_id"])
    assert (m["in_test_v1"] == m["study_id"].isin(v1_val)).all()
    assert (m.loc[m["in_test_clean"] | m["in_test_v1"], "split"] == TEST).all()
    assert set(m["view"]) <= {"PA", "AP"}
    assert len(patients[VAL]) > 0 and len(patients[TRAIN]) > len(patients[VAL])


def test_manifest_labels_equal_the_chexpert_file(prepared, synth):
    m = load_manifest(prepared).set_index("study_id")
    chex = pd.read_csv(synth.mimic_root / "mimic-cxr-2.0.0-chexpert.csv.gz").set_index("study_id")
    common = m.index.intersection(chex.index)
    assert len(common) == len(m)
    a = m.loc[common, list(LABELS)].to_numpy(float)
    b = chex.loc[common, list(LABELS)].to_numpy(float)
    assert np.array_equal(np.nan_to_num(a, nan=9), np.nan_to_num(b, nan=9))


def test_images_are_grayscale_pngs_with_the_requested_shorter_side(prepared):
    m = load_manifest(prepared)
    for rel in m["image"].head(10):
        with Image.open(prepared / rel) as img:
            assert img.mode == "L" and min(img.size) == 48


def test_audit_counts_drops_and_rad_dino_overlap(prepared):
    audit = json.loads((prepared / "audit.json").read_text())
    assert audit["drops"]["no frontal (PA/AP) image"] > 0
    assert audit["drops"]["image missing or unreadable"] == 1
    assert len(pd.read_csv(prepared / "unreadable_images.csv")) == 1  # details stay in a local file
    assert audit["patient_overlap"] == {"train/val": 0, "train/test": 0, "val/test": 0}
    rad = audit["rad_dino"]
    assert rad["test_clean"]["images_in_list"] == 0 and rad["test_clean"]["patients_in_list"] == 0
    assert rad["train"]["images_in_list"] > 0  # the synthetic list holds every official-train image
    assert (prepared / "audit.md").read_text().count("Gate G0") == 1
    frozen = json.loads((prepared / "measurable_classes.json").read_text())
    assert {"test_clean:v2", "test_v1:v2", "test_v1:v1", "test_clean:v1"} <= set(frozen)


def test_rad_dino_overlap_flags_a_test_image_in_the_list(prepared):
    m = load_manifest(prepared)
    leaked = m[m["in_test_clean"]].iloc[0]
    rad = pd.DataFrame({"dicom_id": [leaked["dicom_id"]], "subject_id": [leaked["subject_id"]]})
    res = rad_dino_overlap(m, rad)
    assert res["test_clean"]["images_in_list"] == 1 and res["test_clean"]["patients_in_list"] == 1


def _prepare(synth, out, *extra):
    prepare_data.main(["--mimic-cxr-jpg", str(synth.mimic_root), "--out", str(out), "--image-size", "32",
                       "--workers", "0", "--min-count", "3", "--rad-dino-list", "skip", *extra])
    return load_manifest(out)


def test_extend_test_clean_adds_only_official_test_studies(synth, tmp_path):
    base = _prepare(synth, tmp_path / "a", "--cohort-dir", str(synth.cohort_dir), "--skip-images")
    ext = _prepare(synth, tmp_path / "b", "--cohort-dir", str(synth.cohort_dir), "--skip-images",
                   "--extend-test-clean")
    added = ext[~ext["study_id"].isin(base["study_id"])]
    assert len(added) > 0
    assert (added["source"] == "test_clean extension").all()
    assert added["in_test_clean"].all() and (added["split"] == TEST).all() and not added["in_test_v1"].any()


def test_without_cohort_uses_all_abnormal_studies_and_test_v1_is_empty(synth, tmp_path):
    m = _prepare(synth, tmp_path, "--skip-images")
    assert not m["in_test_v1"].any()
    assert ((m[list(LABELS)] == 1.0).any(axis=1)).all()


def test_limit_keeps_whole_patients(synth, tmp_path):
    m = _prepare(synth, tmp_path, "--skip-images", "--limit", "30")
    assert 0 < len(m) <= 30


def test_rerun_keeps_existing_images_and_rejects_a_new_size(synth, tmp_path):
    _prepare(synth, tmp_path, "--limit", "20")
    first = sorted((tmp_path / "images").glob("*.png"))[0]
    mtime = first.stat().st_mtime_ns
    _prepare(synth, tmp_path, "--limit", "20")
    assert first.stat().st_mtime_ns == mtime
    with pytest.raises(SystemExit):
        prepare_data.main(["--mimic-cxr-jpg", str(synth.mimic_root), "--out", str(tmp_path), "--image-size", "64",
                           "--workers", "0", "--rad-dino-list", "skip", "--limit", "20"])


def test_v1_lateral_share(tmp_path):
    h5py = pytest.importorskip("h5py")
    meta = pd.DataFrame({"dicom_id": ["a", "b", "c"], "ViewPosition": ["PA", "LATERAL", "AP"]})
    with h5py.File(tmp_path / "images.h5", "w") as f:
        g = f.create_group("metadata")
        for i, d in enumerate(["a", "b", "c"]):
            g.create_dataset(str(i), data=json.dumps({"study_id": i, "image_path": f"/x/s{i}/{d}.jpg"}))
    res = prepare_data.v1_lateral_share(tmp_path, meta)
    assert res["images"] == 3 and res["non_frontal_share"] == pytest.approx(1 / 3, abs=1e-3)
