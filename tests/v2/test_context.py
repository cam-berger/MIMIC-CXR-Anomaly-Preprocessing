"""M4 inputs: report sections, the time rules for ED stays and earlier studies, and the leak traps."""

import re
import zipfile

import numpy as np
import pandas as pd
import pytest

from v2 import context
from v2.context import Reports, indication_text, link_ed_stays, link_priors, prior_text

STANDARD = """                                 FINAL REPORT
 EXAMINATION:  CHEST (PA AND LAT)

 INDICATION:  ___F with new onset ascites  // eval for infection

 TECHNIQUE:  Chest PA and lateral

 COMPARISON:  None.

 FINDINGS:

 LEAK no focal consolidation.

 IMPRESSION:

 LEAK no acute process.
"""


@pytest.mark.parametrize("report, expected", [
    (STANDARD, "___F with new onset ascites // eval for infection"),
    (" WET READ: ___ ___ 10:12 PM\n   LEAK opacity\n ____\n FINAL REPORT\n INDICATION:  Fever.\n\n FINDINGS:  LEAK.\n",
     "Fever."),
    (" CHEST RADIOGRAPH ON ___  COMPARISON: ___. CLINICAL HISTORY: 75-year-old with cough. FINDINGS: LEAK. "
     "IMPRESSION: LEAK.", "75-year-old with cough."),
    (" FINAL REPORT\n INDICATION:  Shortness of breath.\n\n LEAK The lungs are clear.\n", "Shortness of breath."),
    (" INDICATION:\n ___M with chest pain\n and dyspnea\n\n FINDINGS: LEAK\n", "___M with chest pain and dyspnea"),
    (" HISTORY:  SOB.\n REASON FOR EXAMINATION:  eval for chf\n IMPRESSION:  LEAK edema\n", "SOB. eval for chf"),
    (" FINAL REPORT\n EXAMINATION: CHEST\n FINDINGS: LEAK\n IMPRESSION: LEAK\n", ""),
    (" INDICATION:  cough, see findings: LEAK\n", "cough, see"),
])
def test_indication_keeps_only_text_written_before_the_image(report, expected):
    text, _ = indication_text(report)
    assert text == expected
    assert "LEAK" not in text


def test_indication_cuts_are_counted():
    _, cuts = indication_text(" INDICATION:  SOB.\n\n LEAK lungs clear.\n")
    assert cuts["paragraph"] and not cuts["guard"]
    _, cuts = indication_text(" INDICATION:  cough, findings: LEAK\n")
    assert cuts["guard"]
    text, cuts = indication_text(" INDICATION:  " + "x" * 500 + "\n")
    assert cuts["length"] and len(text) == context.INDICATION_MAX_CHARS


def test_prior_text_is_findings_and_impression():
    assert prior_text(STANDARD) == "LEAK no focal consolidation. LEAK no acute process."
    assert prior_text("No headers at all.") == "No headers at all."
    assert prior_text("Lungs clear. ADDENDUM: written days later.") == "Lungs clear."


def test_reports_from_zip_or_folder(tmp_path):
    name = "files/p10/p10000032/s50414267.txt"
    with zipfile.ZipFile(tmp_path / "r.zip", "w") as zf:
        zf.writestr("mimic-cxr-reports/" + name, STANDARD)
    folder = tmp_path / "unzipped"
    (folder / name).parent.mkdir(parents=True)
    (folder / name).write_text(STANDARD)
    for source in (tmp_path / "r.zip", folder, folder / "files"):
        reports = Reports(source)
        assert reports.get(10000032, 50414267) == STANDARD
        assert reports.get(10000032, 1) is None
        reports.close()


def test_ed_stay_must_begin_at_or_before_the_xray_and_within_24h():
    t = pd.Timestamp("2150-01-01 12:00")
    studies = pd.DataFrame({"study_id": [1, 2, 3], "subject_id": [7, 7, 8],
                            "study_datetime": [t, t + pd.Timedelta(days=30), t]})
    stays = pd.DataFrame({
        "subject_id": [7, 7, 7, 7, 8],
        "stay_id": [100, 101, 102, 103, 200],
        "intime": [t - pd.Timedelta(hours=2), t - pd.Timedelta(hours=1), t + pd.Timedelta(hours=1),
                   t + pd.Timedelta(days=30) - pd.Timedelta(hours=25), t + pd.Timedelta(hours=3)],
    })
    linked, stats = link_ed_stays(studies, stays)
    linked = linked.set_index("study_id")
    assert linked.loc[1, "stay_id"] == 101 and linked.loc[1, "ed_hours_before"] == 1.0  # the latest eligible stay
    assert 2 not in linked.index                                                          # that stay began 25 h earlier
    assert 3 not in linked.index                                                          # arrival after the X-ray
    assert stats["only_stays_after_the_xray"] == 1 and stats["only_stays_over_24h_before"] == 1


def test_prior_report_comes_from_a_study_at_least_24h_earlier():
    t = pd.Timestamp("2150-06-01 08:00")
    studies = pd.DataFrame({"study_id": [10, 20], "subject_id": [7, 8], "study_datetime": [t, t]})
    everything = pd.DataFrame({
        "prior_study_id": [1, 2, 3, 10, 4, 20],
        "subject_id": [7, 7, 7, 7, 8, 8],
        "prior_datetime": [t - pd.Timedelta(days=10), t - pd.Timedelta(hours=24), t - pd.Timedelta(hours=2), t,
                           t - pd.Timedelta(hours=5), t],
    })
    linked = link_priors(studies, everything).set_index("study_id")
    assert linked.loc[10, "prior_study_id"] == 2 and linked.loc[10, "prior_days"] == 1.0  # exactly 24 h counts
    assert 20 not in linked.index                                                         # only 5 h before


@pytest.fixture(scope="module")
def context_table(prepared, synth):
    context.main(["--out", str(prepared), "--reports", str(synth.reports), "--mimic-iv-ed", str(synth.mimic_iv_ed),
                  "--mimic-iv", str(synth.mimic_iv), "--mimic-cxr-jpg", str(synth.mimic_root)])
    return pd.read_parquet(prepared / "context.parquet")


def test_no_study_sees_its_own_findings(context_table):
    ctx = context_table
    for col in ("indication", "chief_complaint", "prior_report"):
        own = [f"leak{s}" in text for s, text in zip(ctx["study_id"], ctx[col])]
        assert not any(own), f"{sum(own)} studies' {col} contains their own findings"
    assert ctx["indication"].ne("").mean() > 0.8
    # Prior reports are another study's findings; they must exist and come from that study.
    has_prior = ctx["prior_study_id"].notna()
    assert has_prior.any()
    assert all(f"leak{p}" in text for p, text in zip(ctx.loc[has_prior, "prior_study_id"],
                                                     ctx.loc[has_prior, "prior_report"]) if text)


def test_decoy_ed_stays_are_never_linked(context_table):
    linked = set(context_table["chief_complaint"])
    assert "later visit" not in linked and "stale visit" not in linked
    with_cc = context_table["chief_complaint"] != ""
    assert with_cc.any()
    assert (context_table.loc[with_cc, "ed_hours_before"].between(0, 24)).all()
    assert context_table.loc[with_cc, "temperature"].notna().all()
    assert context_table["age"].notna().any() and context_table["sex_male"].notna().any()


def test_audit_holds_counts_only(context_table, prepared):
    text = (prepared / "context_audit.md").read_text() + (prepared / "context_audit.json").read_text()
    assert not re.search(r"\b5\d{7}\b", text), "a study id reached the audit"
    assert "leak" not in text
    audit = pd.read_json(prepared / "context_audit.json", typ="series")
    assert set(audit["coverage"]) == {"train", "val", "test_clean", "test_v1"}


def test_missing_sources_leave_inputs_empty(prepared, tmp_path, monkeypatch):
    for env in ("MIMIC_CXR_REPORTS_PATH", "MIMIC_IV_ED_PATH", "MIMIC_IV_PATH", "MIMIC_CXR_JPG_PATH"):
        monkeypatch.setenv(env, "")  # empty, so a local .env cannot fill them in
    out = tmp_path / "v2"
    out.mkdir()
    (out / "manifest.parquet").write_bytes((prepared / "manifest.parquet").read_bytes())
    context.main(["--out", str(out)])
    ctx = pd.read_parquet(out / "context.parquet")
    assert (ctx[list(context.TEXT_SOURCES)] == "").all().all()
    assert np.isnan(ctx["temperature"]).all()


def test_prior_labels_come_from_the_prior_study(context_table, synth):
    from v2.common import CHEXPERT_COLUMNS, mimic_csv
    from v2.context import PRIOR_LABEL_COLUMNS

    ctx = context_table
    chexpert = pd.read_csv(mimic_csv(synth.mimic_root, "chexpert")).set_index("study_id")
    has = ctx["prior_labels_found"].to_numpy(bool)
    assert has.any() and (ctx["prior_study_id"].notna() | ~has).all()
    rows = ctx[has]
    want = chexpert.loc[rows["prior_study_id"].astype(int), list(CHEXPERT_COLUMNS)].to_numpy(np.float64)
    got = rows[list(PRIOR_LABEL_COLUMNS)].to_numpy(np.float64)
    assert np.array_equal(np.nan_to_num(got, nan=9.0), np.nan_to_num(want, nan=9.0))
    # ...and never from the study itself.
    own_ids = rows["study_id"][rows["study_id"].isin(chexpert.index)]
    own = chexpert.loc[own_ids, list(CHEXPERT_COLUMNS)].to_numpy(np.float64)
    same = (np.nan_to_num(own, nan=9.0) == np.nan_to_num(got[rows["study_id"].isin(chexpert.index).to_numpy()], nan=9.0)).all(axis=1)
    assert same.mean() < 0.5
    assert np.isnan(ctx.loc[~has, list(PRIOR_LABEL_COLUMNS)].to_numpy(np.float64)).all()
