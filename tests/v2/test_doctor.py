"""Setup check: data and cohort checks (the Hugging Face checks need the network and are not run here)."""

import shutil

from v2.doctor import Report, check_cohort, check_m4b, check_m7, check_mimic, check_output
from v2.synth import make_chexpert_test


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


def test_m4b_check_reads_only_the_context_schema(bigger, synth):
    r = Report()
    check_m4b(r, bigger, synth.mimic_root)
    assert r.failures == 0 and r.warnings == 0  # labels found, context rebuilt, tokens cached


def test_m7_check_counts_the_inputs(synth, tmp_path, monkeypatch):
    chex = make_chexpert_test(tmp_path / "chexpert", n_patients=10, seed=1)
    bench = tmp_path / "bench"
    for run in ("rad-dino_518_ft_s0", "rad-dino_518_ft_attn_s0"):
        (bench / "runs" / run).mkdir(parents=True)
        (bench / "runs" / run / "best.pt").write_bytes(b"")
    monkeypatch.setenv("CHEXPERT_TEST_PATH", str(chex))
    monkeypatch.setenv("BENCH_OUT", str(bench))
    r = Report()
    check_m7(r, synth.mimic_root)
    assert r.failures == 0 and r.warnings == 0
    labels_only = tmp_path / "labels_only"
    labels_only.mkdir()
    shutil.copy(chex / "groundtruth.csv", labels_only)
    monkeypatch.setenv("CHEXPERT_TEST_PATH", str(labels_only))  # Redivis's copy: labels, no images
    r = Report()
    check_m7(r, synth.mimic_root)
    assert r.failures == 0 and r.warnings == 1
