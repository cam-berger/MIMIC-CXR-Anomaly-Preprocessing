"""v3.3: Jev as a report labeler, on the synthetic benchmark's reports and radiologist's labels with a fake Jev."""

import numpy as np
import pytest

from v2.common import LABELS, read_predictions
from v3 import jev, label, prompt, smoke


def client_for(fake):
    return jev.JevClient(fake.model, api_key="test", transport=fake, sleep=lambda s: None)


def test_questions_are_12_three_way_choices_in_the_labeler_s_terms():
    qs = label.questions()
    assert list(qs) == [prompt.question_name(lab) for lab in LABELS]
    assert all(q["type"] == "choice" and tuple(q["criteria"]) == label.OPTIONS for q in qs.values())
    assert label.LEVEL == {"present": 1.0, "uncertain": 0.5, "absent": 0.0}
    assert label.render_state("  ").endswith(prompt.MISSING) and label.render_state("x").startswith(label.PREAMBLE)


def test_val_is_for_pilots_only(v3out, synth_sources):
    reports, labels = synth_sources
    with pytest.raises(SystemExit, match="pilots only"):
        label.run(v3out, "val", client_for(smoke.FakeJev()), reports, labels)


def test_a_pilot_writes_its_report_and_no_predictions(v3out, synth_sources):
    reports, labels = synth_sources
    stats = label.run(v3out, "val", client_for(smoke.FakeJev()), reports, labels, limit=8)
    assert stats["n_answered"] == 8 and stats["n_failed"] == 0 and set(stats["choice_shares"]) == set(label.OPTIONS)
    assert (v3out / "reports" / "v3_label_pilot.md").exists()
    assert not (v3out / "predictions" / "labeler_jev.parquet").exists()


def test_the_radiologist_s_studies_are_labeled_from_their_own_reports_and_scored(v3out, v3_base, synth_sources):
    fake = smoke.FakeJev()
    assert smoke.check_labels(v3out, v3_base.parent, fake) == []
    rows = label.target_rows(v3out, "test", synth_sources[1])
    for run in label.RUNS:
        wide, meta = read_predictions(v3out / "predictions" / f"{run}.parquet")
        assert set(wide["study_id"]) == set(rows["study_id"]) and meta["kind"] == "labeler"
    prob, _ = read_predictions(v3out / "predictions" / "labeler_jev_prob.parquet")
    assert np.allclose(np.unique(prob[list(LABELS)].to_numpy()), [0.075, 0.925])  # 0.9 + 0.5 * 0.05, 0.05 + 0.025
    text = (v3out / "reports" / "v3_label.md").read_text()
    assert "Gate L1" in text and "Per finding" in text
    before = fake.calls
    label.run(v3out, "test", client_for(fake), *synth_sources)
    assert fake.calls == before  # cached: nothing asked twice


def test_the_report_needs_the_three_runs(v3out, synth_sources):
    with pytest.raises(SystemExit, match="v3.label run"):
        label.build_report(v3out, synth_sources[1], n_boot=20)
