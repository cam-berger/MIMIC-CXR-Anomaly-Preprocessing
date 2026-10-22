"""M5: the standard benchmark (all studies, official split), positive-vs-rest scoring and the lineup report."""

import json

import numpy as np
import pandas as pd
import pytest

from v2 import evaluate, lineup, prepare_data, probe
from v2.common import LABELS, TEST, TRAIN, VAL, load_manifest, targets_and_mask


def test_u0_counts_everything_but_positives_as_negative():
    raw = np.array([[1.0], [0.0], [-1.0], [np.nan]])
    y, m = targets_and_mask(raw, "u0")
    assert y[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0] and m.all()


def test_official_scheme_follows_the_official_split(bench, synth):
    m = load_manifest(bench)
    official = pd.Series(synth.official)
    want = official.reindex(m["subject_id"]).map({"train": TRAIN, "validate": VAL, "test": TEST}).to_numpy()
    assert (m["split"].to_numpy() == want).all()
    assert (m["in_test_clean"] == (m["split"] == TEST)).all() and not m["in_test_v1"].any()
    assert not (m[list(LABELS)] == 1.0).any(axis=1).all()  # every study, normal ones included
    assert m["image"].str.endswith(".jpg").all() and all((bench / p).exists() for p in m["image"].head(20))
    audit = json.loads((bench / "audit.json").read_text())
    assert audit["split_scheme"] == "official" and audit["population"] == "all studies"
    assert audit["rad_dino"]["test_clean"]["images_in_list"] == 0
    assert "test_clean:u0" in json.loads((bench / "measurable_classes.json").read_text())


def test_the_benchmark_never_replaces_the_v2_manifest(prepared, synth):
    before = (prepared / "manifest.parquet").read_bytes()
    with pytest.raises(SystemExit, match="Use another --out"):
        prepare_data.main(["--mimic-cxr-jpg", str(synth.mimic_root), "--out", str(prepared), "--population", "all",
                           "--split-scheme", "official", "--skip-images", "--rad-dino-list", "skip"])
    assert (prepared / "manifest.parquet").read_bytes() == before


def test_u0_is_scored_only_where_its_classes_were_frozen(bench, prepared, pixel_embeddings):
    sets = lambda rep: {(r["test_set"], r["metric"]) for r in rep["results"]}  # noqa: E731
    assert ("test_clean", "u0") in sets(evaluate.evaluate_run(bench, "test-pixels_48_probe", n_boot=10))
    if not (prepared / "predictions" / "test-pixels_48_probe.parquet").exists():
        probe.main(["--embeddings", str(pixel_embeddings), "--out", str(prepared), "--jobs", "1"])
    frozen = json.loads((prepared / "measurable_classes.json").read_text())
    want = ("test_clean", "u0") in sets(evaluate.evaluate_run(prepared, "test-pixels_48_probe", n_boot=10))
    assert want == ("test_clean:u0" in frozen)


def test_lineup_report(bench):
    res = lineup.main(["--out", str(bench), "--n-boot", "50"])
    text = res["path"].read_text()
    for heading in ("Red flags", "Our runs", "Like for like", "Per class", "Published results"):
        assert heading in text, heading
    pairs = {(c["run"], c["published"]) for c in res["compared"]}
    assert ("test-pixels_48_probe", "ResNet-50") in pairs
    assert ("test-pixels_48_ctx_image_indication", "MMBT") in pairs
    assert ("test-pixels_48_ctx_indication", "BERT") in pairs
    assert ("test-pixels_48_attn_h0", "ResNet-50") in pairs                       # M5 amendment
    assert not any(run == "test-pixels_48_attn_h1" for run, _ in pairs)           # reported, not like for like
    assert ("test-pixels_48_attn_h0_stack_indication", "MMBT") in pairs           # M5b amendment: val combiner
    assert not any(run.endswith("_stack_text_prior_labels") for run, _ in pairs)  # all text: not like for like
    attn = (bench / "reports" / "m4b_attention_test-pixels_48.md").read_text()
    assert attn.startswith("# M5: attention heads on the benchmark") and "**G4b**" not in attn
    assert all(c["published"] not in ("CheXclusion DenseNet-121", "CaMCheX") for c in res["compared"])  # other protocols
    summary = evaluate.summarize(bench, n_boot=20)
    assert "M5's benchmark" in summary and "**G1**" not in summary and "| M5 |" in summary


def test_benchmark_reports_have_no_v2_gates(bench):
    fused = (bench / "reports" / "m4_test-pixels_48.md").read_text()
    assert fused.startswith("# M5:") and "**G4**" not in fused and "No gate here" in fused
    assert "largest score difference" in fused                                    # the probe check stays
    prior = (bench / "reports" / "m4b_test-pixels_48.md").read_text()
    assert prior.startswith("# M5:") and "test_v1" not in prior and "positive vs rest" in prior
    stacked = (bench / "reports" / "m5_stack.md").read_text()
    assert stacked.startswith("# M5: combiners fit on val") and "test_v1" not in stacked
    for text in ("indication", "text_prior_labels"):  # the check, against the matching concatenation
        assert f"`test-pixels_48_probe_stack_{text}` minus `test-pixels_48_ctx_image_{text}`" in stacked
    meta = json.loads((bench / "predictions" / "test-pixels_48_attn_h0_stack_indication.json").read_text())
    assert (meta["image_kind"], meta["image_arm"], meta["text_arm"]) == ("attend", "h0", "indication")


def test_roles_follow_the_inputs():
    role = lineup.role
    assert role({"kind": "finetune", "head": "attention"}) == "image"
    assert role({"kind": "fuse", "arm": "image_indication", "image_features": "frozen"}) == "image + indication"
    assert role({"kind": "fuse", "arm": "image_indication", "image_features": "fine-tuned"}) is None  # exploratory
    assert role({"kind": "stack", "image_kind": "finetune", "text_arm": "indication"}) == "image + indication"
    assert role({"kind": "stack", "image_kind": "attend", "image_arm": "h1", "text_arm": "indication"}) is None
    assert role({"kind": "stack", "image_kind": "probe", "text_arm": "text_prior_labels"}) is None


def test_lineup_refuses_the_v2_cohort(prepared):
    with pytest.raises(SystemExit, match="benchmark directory"):
        lineup.main(["--out", str(prepared)])


def test_verdict_rule():
    assert lineup.verdict(0.70, [0.75, 0.80]) == "below ours"
    assert lineup.verdict(0.77, [0.75, 0.80]) == "level"
    assert lineup.verdict(0.81, [0.75, 0.80]).strip("*") == "above ours"
    assert lineup.verdict(None, [0.75, 0.80]) == "-"


@pytest.mark.parametrize("name, no_finding, support_devices, paper_macro", [
    ("ResNet-50", 0.779, 0.859, 0.738), ("Attentive", 0.809, 0.905, 0.766), ("MMBT", 0.831, 0.922, 0.806)])
def test_transcribed_tables_reproduce_the_papers_macros(name, no_finding, support_devices, paper_macro):
    """With the two classes v2 leaves out, MMBT's Table 3 must average to its Table 1 macro (a transcription check)."""
    ref = next(r for r in lineup.PUBLISHED if r.name == name)
    assert ref.macro == paper_macro and len(ref.per_class) == len(LABELS)
    assert np.mean([*ref.per_class.values(), no_finding, support_devices]) == pytest.approx(paper_macro, abs=6e-4)


def test_published_means_use_only_the_classes_we_measure():
    ref = next(r for r in lineup.PUBLISHED if r.name == "MMBT")
    assert lineup.published_mean(ref, ["Edema", "Pleural Effusion"]) == pytest.approx((0.843 + 0.886) / 2)
    assert lineup.published_mean(next(r for r in lineup.PUBLISHED if r.name == "MBRANet"), ["Edema"]) is None
