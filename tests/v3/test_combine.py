"""v3.2: Jev's answers as inputs to v2's val combiner, on the synthetic benchmark with a fake Jev."""

import numpy as np
import pytest
from scipy.stats import rankdata

from v2.common import LABELS, read_predictions
from v3 import combine, jev, prompt, report, smoke


def client_for(fake):
    return jev.JevClient(fake.model, api_key="test", transport=fake, sleep=lambda s: None)


def test_combine_needs_the_jev_runs_on_test(v3out):
    with pytest.raises(SystemExit, match="format"):
        combine.build(v3out, n_boot=20)


def test_logodds_are_clipped_and_keep_each_finding_s_ranks(v3out):
    c = client_for(smoke.FakeJev())
    smoke.run_order(v3out, c)
    run = prompt.run_name("jev_text", report.choose_format(v3out)["chosen"])
    name = combine.to_logodds(v3out, run)
    p, _ = read_predictions(v3out / "predictions" / f"{run}.parquet")
    z, meta = read_predictions(v3out / "predictions" / f"{name}.parquet")
    lim = np.log((1 - combine.CLIP) / combine.CLIP)
    assert np.abs(z[list(LABELS)].to_numpy()).max() <= lim + 1e-9
    for label in LABELS:
        assert (rankdata(p[label]) == rankdata(z[label])).all()
    assert meta["stack_tag"] == run and meta["source_run"] == run


def test_the_five_stacks_are_fit_and_reported(v3out):
    c = client_for(smoke.FakeJev())
    chosen = smoke.run_order(v3out, c)
    smoke.run_v3_1(v3out, c, chosen)
    out = combine.main(["--out", str(v3out), "--n-boot", "20"])
    assert [row["label"] for row in out["rows"]] == list("ABCDE")
    setup = report.load_setup(v3out)
    assert out["rows"][0]["run"] == f"{setup['image_run']}_stack_{prompt.run_name('jev_text', chosen)}"
    assert out["rows"][2]["base"] == setup["stack_text"]
    for row in out["rows"]:
        assert (v3out / "predictions" / f"{row['run']}.parquet").exists() and row["test"] is not None
    text = out["report"].read_text()
    assert text.count("| A |") == 1 and "| E |" in text and "Read against" in text
