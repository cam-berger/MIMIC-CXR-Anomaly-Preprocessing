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


def _run(args, tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if k != "MIMIC_CXR_JPG_PATH"}
    env.update({"PYTHON": sys.executable, "V2_OUT": str(tmp_path / "v2"), "ALLOW_UNPINNED": "1", **extra})
    return subprocess.run(["bash", str(SCRIPTS / "run.sh"), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


def test_run_without_stages_prints_usage(tmp_path):
    res = _run([], tmp_path)
    assert res.returncode == 2 and "Stages:" in res.stdout
    assert "WORKERS" in res.stdout and "set -euo" not in res.stdout  # the whole header, nothing after it


def test_run_rejects_unknown_stage(tmp_path):
    res = _run(["m9"], tmp_path)
    assert res.returncode == 2 and "Unknown stage" in res.stderr


def test_bench_stack_needs_the_bench_stage_first(tmp_path):
    res = _run(["bench_stack"], tmp_path, BENCH_OUT=str(tmp_path / "bench"))
    assert res.returncode != 0 and "needs the bench stage's text arms" in res.stderr
    assert "bench stages: RAD-DINO at 518 px" in res.stdout


def test_m0_requires_the_data_path(tmp_path):
    env_file = ROOT / ".env"
    if env_file.exists() and "MIMIC_CXR_JPG_PATH" in env_file.read_text():
        pytest.skip("a local .env sets MIMIC_CXR_JPG_PATH")
    res = _run(["m0"], tmp_path)
    assert res.returncode != 0 and "MIMIC_CXR_JPG_PATH" in res.stderr


def test_unpinned_transformers_is_refused(tmp_path):
    import transformers

    env = {k: v for k, v in os.environ.items() if k not in ("MIMIC_CXR_JPG_PATH", "ALLOW_UNPINNED")}
    env.update({"PYTHON": sys.executable, "V2_OUT": str(tmp_path / "v2")})
    pinned = next(line.split("==")[1].split()[0] for line in (ROOT / "v2" / "requirements.txt").read_text().splitlines()
                  if line.startswith("transformers=="))
    res = subprocess.run(["bash", str(SCRIPTS / "run.sh"), "eval"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    if transformers.__version__ == pinned:
        assert "v2 pins" not in res.stderr
    else:
        assert res.returncode == 1 and "v2 pins" in res.stderr
