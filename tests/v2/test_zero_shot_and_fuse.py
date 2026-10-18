"""M2 zero-shot scoring and M4 vitals features."""

import json

import numpy as np
import pandas as pd
import pytest

from v2 import embed, fuse, zero_shot
from v2.zero_shot import TEMPLATES, zero_shot_scores


def test_zero_shot_score_is_the_cosine_difference():
    img = np.array([[1.0, 0.0], [0.0, 2.0]])
    pos = np.array([[1.0, 0.0]] * 12)
    neg = np.array([[0.0, 1.0]] * 12)
    s = zero_shot_scores(img, pos, neg)
    assert s.shape == (2, 12)
    assert np.allclose(s[0], 1.0) and np.allclose(s[1], -1.0)


def test_zero_shot_end_to_end(prepared):
    embed.main(["--model", "tiny-siglip", "--out", str(prepared), "--workers", "0"])
    zero_shot.main(["--model", "tiny-siglip", "--out", str(prepared)])
    meta = json.loads((prepared / "predictions" / "tiny-siglip_32_zeroshot.json").read_text())
    assert meta["kind"] == "zero_shot" and meta["template"]["name"] in TEMPLATES
    assert set(meta["val_macro_auroc_by_template"]) == set(TEMPLATES)


def test_zero_shot_rejects_embeddings_from_another_model(prepared, pixel_embeddings):
    with pytest.raises(SystemExit):
        zero_shot.main(["--model", "tiny-siglip", "--out", str(prepared), "--embeddings", str(pixel_embeddings)])


def test_vitals_after_the_xray_or_out_of_range_are_missing():
    t = pd.Timestamp("2150-01-01 12:00")
    manifest = pd.DataFrame({"study_id": [1, 2, 3, 4], "study_datetime": [t] * 4})
    cohort = pd.DataFrame({
        "study_id": [1, 2, 3],
        "anchor_age": [50, 60, 70],
        "gender": ["M", "F", "M"],
        "intime": [t - pd.Timedelta(hours=2), t + pd.Timedelta(hours=1), t - pd.Timedelta(hours=1)],
        **{col: [100.0, 100.0, 100.0] for col in fuse.VITALS},
    })
    cohort["triage_temperature"] = [101.0, 101.0, 500.0]  # 500 F is a data-entry error
    feats, stats = fuse.vitals_features(manifest, cohort)
    feats = feats.set_index("study_id")
    assert feats.loc[1, "triage_temperature"] == 101.0
    assert np.isnan(feats.loc[2, "triage_temperature"])  # ED arrival after the X-ray
    assert np.isnan(feats.loc[3, "triage_temperature"])  # out of range
    assert np.isnan(feats.loc[4, "triage_heartrate"])    # not in the cohort
    assert feats.loc[2, "age"] == 60 and feats.loc[2, "sex_male"] == 0.0  # demographics are pre-imaging
    assert stats["ed_arrival_before_xray"] == 2


def test_imputation_uses_train_medians_and_adds_indicators():
    feats = pd.DataFrame({"study_id": [1, 2, 3], "a": [1.0, np.nan, 100.0]})
    out = fuse.impute_with_indicators(feats, np.array([True, True, False]))
    assert out[:, 0].tolist() == [1.0, 1.0, 100.0]  # median of train rows only
    assert out[:, 1].tolist() == [0.0, 1.0, 0.0]


def test_fuse_end_to_end(prepared, synth):
    embed.main(["--model", "tiny-siglip", "--out", str(prepared), "--workers", "0"])
    fuse.main(["--embeddings", str(prepared / "embeddings" / "tiny-siglip_32.npz"),
               "--cohort-dir", str(synth.cohort_dir), "--out", str(prepared), "--n-boot", "50", "--jobs", "1"])
    for name in ("imageonly", "vitals"):
        assert (prepared / "predictions" / f"tiny-siglip_32_fuse_{name}.parquet").exists()
    assert list((prepared / "reports").glob("compare_tiny-siglip_32_fuse_vitals__*"))
