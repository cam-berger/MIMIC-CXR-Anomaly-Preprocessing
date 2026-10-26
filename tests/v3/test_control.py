"""v3.7: the hierarchy control (synthetic data with a stand-in external set; no network)."""

import numpy as np

from v2.common import LABELS
from v3 import chexpert, control

from .test_chexpert import make_external, trained  # noqa: F401  (the fixture)

LO, AT, PN, ECM, CM = (LABELS.index(c) for c in ("Lung Opacity", "Atelectasis", "Pneumonia",
                                                 "Enlarged Cardiomediastinum", "Cardiomegaly"))


def test_the_hierarchy_sets_parents_positive_and_leaves_the_rest():
    raw = np.full((4, 12), np.nan)
    raw[0, AT] = 1.0                       # a child positive: Lung Opacity becomes 1
    raw[1, PN], raw[1, LO] = -1.0, 0.0     # an uncertain child: Lung Opacity stays 0
    raw[2, CM], raw[2, ECM] = 1.0, -1.0    # Cardiomegaly: an uncertain parent becomes 1
    raw[3, LO] = -1.0                      # no child: uncertain stays
    out = control.hierarchical(raw)
    assert out[0, LO] == 1.0 and out[1, LO] == 0.0 and out[2, ECM] == 1.0 and out[3, LO] == -1.0
    assert np.isnan(out[0, ECM]) and np.array_equal(np.isnan(out[:, AT]), np.isnan(raw[:, AT]))


def test_the_control_trains_on_v3_5_s_studies_and_is_scored_on_both_judges(v3out, tmp_path, trained,
                                                                            synth_sources):  # noqa: F811
    emb, model = trained
    ext = make_external(v3out, tmp_path)
    args = ["--model", model, "--ext", str(ext), "--embeddings", str(emb), "--ext-embeddings", str(emb),
            "--n-train", "60", "--n-jobs", "1", "--n-boot", "20", "--out", str(v3out)]
    chexpert.main(args)
    out = control.main([*args, "--labels", str(synth_sources[1])])
    assert out["fitted"]["n_train"] == 60
    assert all(v == 1.0 for v in out["fitted"]["train_rate_after_rule"].values() if np.isfinite(v))
    for where in (v3out, v3out / "chexpert"):
        assert (where / "predictions" / f"{control.RUN}.parquet").exists()
    assert set(out["views"]) == {"CheXpert test, measurable classes", "CheXpert test, five standard findings",
                                 "MIMIC-CXR, M7a's radiologist"}
    assert "**Primary:**" in out["report"].read_text()
