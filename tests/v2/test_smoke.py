"""The full wiring check (python -m v2.smoke) must pass."""

from v2.smoke import run_checks


def test_wiring_check_passes(tmp_path):
    results = run_checks(tmp_path, patients=300)
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert not failed, failed
    assert len(results) == 12
