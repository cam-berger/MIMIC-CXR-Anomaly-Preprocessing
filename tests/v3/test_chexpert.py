"""v3.6: the v3.5 probes on an external test set (here a stand-in built from the synthetic benchmark's test split)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from v2.common import TEST, label_matrix, load_manifest, measurable
from v3 import chexpert, downstream, jev, label, smoke


def client_for(fake):
    return jev.JevClient(fake.model, api_key="test", transport=fake, sleep=lambda s: None)


def make_external(v3out: Path, root: Path) -> Path:
    """A stand-in for M7b's EXT_OUT: the synthetic benchmark's test studies, with their own frozen classes."""
    ext = root / "ext"
    ext.mkdir()
    rows = load_manifest(v3out)
    rows = rows[rows["split"] == TEST].drop_duplicates("study_id")
    rows.to_parquet(ext / "manifest.parquet", index=False)
    classes = measurable(label_matrix(rows), "v2", 3)
    (ext / "measurable_classes.json").write_text(json.dumps({"test_clean:v2": classes, "test_clean:u0": classes}))
    (ext / "audit.json").write_text(json.dumps({"split_scheme": "external", "test_set_name": "synthetic"}))
    return ext


@pytest.fixture
def trained(v3out, synth_sources):
    reports, labels = synth_sources
    fake = smoke.FakeJev()
    label.run(v3out, "train", client_for(fake), reports, None, limit=60)
    label.run(v3out, "val", client_for(fake), reports, None)
    emb = Path(json.loads((v3out / "v3_setup.json").read_text())["bench"]) / "embeddings" / "test-pixels_48.npz"
    downstream.main(["--model", fake.model, "--labels", str(labels), "--embeddings", str(emb), "--n-train", "60",
                     "--n-jobs", "1", "--n-boot", "20", "--out", str(v3out)])
    return emb, fake.model


def test_the_probes_are_refit_checked_and_scored_on_the_external_set(v3out, tmp_path, trained):
    emb, model = trained
    ext = make_external(v3out, tmp_path)
    out = chexpert.main(["--model", model, "--ext", str(ext), "--embeddings", str(emb), "--ext-embeddings", str(emb),
                         "--n-train", "60", "--n-jobs", "1", "--n-boot", "20", "--out", str(v3out)])
    assert max(out["applied"]["reproduced_max_diff"].values()) <= chexpert.REPRODUCE_TOL
    chex = v3out / "chexpert"
    assert (chex / "manifest.parquet").is_symlink() and not (ext / "predictions").exists()  # nothing written to EXT_OUT
    for run in downstream.ARMS:
        pred = pd.read_parquet(chex / "predictions" / f"{run}.parquet")
        assert pred["study_id"].nunique() == out["n_studies"]
    text = out["report"].read_text()
    assert "Gate L3" in text and "Read with L2" in text and "Five standard findings" in text


def test_a_refit_that_doesn_t_reproduce_stops_the_run(v3out, tmp_path, trained):
    emb, model = trained
    path = v3out / "predictions" / "probe20k_jev_soft.parquet"
    df = pd.read_parquet(path)
    df["score"] += 1e-3
    df.to_parquet(path, index=False)
    with pytest.raises(SystemExit, match="RED FLAG"):
        chexpert.main(["--model", model, "--ext", str(make_external(v3out, tmp_path)), "--embeddings", str(emb),
                       "--ext-embeddings", str(emb), "--n-train", "60", "--n-jobs", "1", "--out", str(v3out)])
