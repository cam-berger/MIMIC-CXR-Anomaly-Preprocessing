"""Shell scripts: syntax, usage and argument handling (the stages themselves are covered by v2.smoke)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "v2" / "scripts"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.mark.parametrize("name", ["setup.sh", "run.sh"])
def test_scripts_parse(name):
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


def _run(args, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MIMIC_CXR_JPG_PATH"}
    env.update({"PYTHON": sys.executable, "V2_OUT": str(tmp_path / "v2")})
    return subprocess.run(["bash", str(SCRIPTS / "run.sh"), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


def test_run_without_stages_prints_usage(tmp_path):
    res = _run([], tmp_path)
    assert res.returncode == 2 and "Stages:" in res.stdout


def test_run_rejects_unknown_stage(tmp_path):
    res = _run(["m9"], tmp_path)
    assert res.returncode == 2 and "Unknown stage" in res.stderr


def test_m0_requires_the_data_path(tmp_path):
    env_file = ROOT / ".env"
    if env_file.exists() and "MIMIC_CXR_JPG_PATH" in env_file.read_text():
        pytest.skip("a local .env sets MIMIC_CXR_JPG_PATH")
    res = _run(["m0"], tmp_path)
    assert res.returncode != 0 and "MIMIC_CXR_JPG_PATH" in res.stderr
