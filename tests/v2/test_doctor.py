"""Setup check: data and cohort checks (the Hugging Face checks need the network and are not run here)."""

import shutil

from v2.doctor import Report, check_cohort, check_mimic, check_output


def test_mimic_and_cohort_checks_pass_on_a_complete_tree(synth, tmp_path):
    r = Report()
    check_mimic(r, synth.mimic_root, 50, synth.cohort_dir)
    check_cohort(r, synth.cohort_dir)
    check_output(r, tmp_path / "v2")
    assert r.failures == 0


def test_missing_cohort_images_fail(synth, tmp_path):
    root = tmp_path / "mimic"
    shutil.copytree(synth.mimic_root, root)
    for path in list((root / "files").rglob("*.jpg"))[::2]:  # lose half the images
        path.unlink()
    r = Report()
    check_mimic(r, root, 50, synth.cohort_dir)
    assert r.failures == 1


def test_missing_drive_fails(tmp_path):
    r = Report()
    check_mimic(r, tmp_path / "not-mounted", 10)
    assert r.failures == 1
