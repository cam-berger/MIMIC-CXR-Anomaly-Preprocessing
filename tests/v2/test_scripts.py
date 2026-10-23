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
    env_file = ROOT / ".env"
    if env_file.exists() and "BENCH_OUT" in env_file.read_text():  # run.sh would read the real benchmark
        pytest.skip("a local .env sets BENCH_OUT")
    res = _run(["bench_stack"], tmp_path, BENCH_OUT=str(tmp_path / "bench"))
    assert res.returncode != 0 and "needs the bench stage's text arms" in res.stderr
    assert "bench stages: RAD-DINO at 518 px" in res.stdout


def test_bench_text_pairs_each_encoder_with_the_image_runs(tmp_path):
    env_file = ROOT / ".env"
    if env_file.exists() and "BENCH_OUT" in env_file.read_text():
        pytest.skip("a local .env sets BENCH_OUT")
    pred, p = tmp_path / "bench" / "predictions", "rad-dino_518"
    pred.mkdir(parents=True)
    done = [f"{p}_ctx_indication", f"{p}_probe", f"{p}_ft_attn_s0", "bert_ft_indication_s0",
            *(f"{p}_ctx_{arm}_{enc}" for arm in ("indication", "text_prior_labels") for enc in ("bert", "biomedbert"))]
    for run in done:
        (pred / f"{run}.parquet").touch()
    # A stand-in python that records the v2 commands instead of running them.
    calls, fake = tmp_path / "calls.txt", tmp_path / "python"
    fake.write_text(f'#!/usr/bin/env bash\nif [ "$1" = -c ]; then exec "{sys.executable}" "$@"; fi\necho "$*" >> "{calls}"\n')
    fake.chmod(0o755)
    res = _run(["bench_text"], tmp_path, BENCH_OUT=str(tmp_path / "bench"), PYTHON=str(fake),
               TEXT_ENCODERS="bert biomedbert")
    assert res.returncode == 0, res.stderr
    lines = calls.read_text().splitlines()
    assert [line.split()[3] for line in lines if line.startswith("-m v2.textenc")] == ["bert", "biomedbert"]
    assert [line.split()[3] for line in lines if line.startswith("-m v2.textft --encoder")] == ["biomedbert"]  # bert's kept
    words = next(line for line in lines if line.startswith("-m v2.stack")).split()
    pairs = {(words[i + 1], words[i + 2]) for i, w in enumerate(words) if w == "--pair"}
    m6 = {(f"{p}_{img}", f"{p}_ctx_{arm}_{enc}") for img in ("probe", "ft_attn_s0")
          for arm in ("indication", "text_prior_labels") for enc in ("bert", "biomedbert")}
    m5b = {(f"{p}_{img}", f"{p}_ctx_{arm}") for img in ("probe", "ft_attn_s0") for arm in ("indication", "text_prior_labels")}
    assert pairs == m5b | m6 | {(f"{p}_ft_attn_s0", "bert_ft_indication_s0")}  # biomedbert's fine-tune never ran
    assert lines[-3:] == [f"-m v2.textft --report --out {tmp_path / 'bench'}", f"-m v2.lineup --out {tmp_path / 'bench'}",
                          f"-m v2.evaluate --all --out {tmp_path / 'bench'}"]

def test_m7_stages_run_the_external_module(tmp_path):
    env_file = ROOT / ".env"
    if env_file.exists() and any(k in env_file.read_text() for k in ("BENCH_OUT", "EXT_OUT", "CHEXPERT_TEST_PATH")):
        pytest.skip("a local .env sets M7's directories")
    calls, fake = tmp_path / "calls.txt", tmp_path / "python"
    fake.write_text(f'#!/usr/bin/env bash\nif [ "$1" = -c ]; then exec "{sys.executable}" "$@"; fi\necho "$*" >> "{calls}"\n')
    fake.chmod(0o755)
    bench, ext, chex = tmp_path / "bench", tmp_path / "ext", tmp_path / "chexpert"
    paths = {"BENCH_OUT": str(bench), "EXT_OUT": str(ext), "CHEXPERT_TEST_PATH": str(chex),
             "MIMIC_CXR_JPG_PATH": str(tmp_path / "mimic")}
    res = _run(["m7a", "m7b"], tmp_path, PYTHON=str(fake), **paths)
    assert res.returncode == 0, res.stderr
    lines = calls.read_text().splitlines()
    assert lines[0] == f"-m v2.external radiologist --bench {bench}"
    assert lines[1].startswith(f"-m v2.external chexpert --chexpert {chex} --bench {bench} --out {ext} --workers ")
    assert lines[2:] == [f"-m v2.evaluate --all --out {ext}"]
    res = _run(["m7b"], tmp_path, PYTHON=str(fake), **{**paths, "EXT_OUT": str(bench)})
    assert res.returncode != 0 and "must differ" in res.stderr


def test_m8_stages_run_seed_1_the_long_run_and_the_report(tmp_path):
    env_file = ROOT / ".env"
    if env_file.exists() and any(k in env_file.read_text() for k in ("BENCH_OUT", "EXT_OUT")):
        pytest.skip("a local .env sets M8's directories")
    calls, fake = tmp_path / "calls.txt", tmp_path / "python"
    fake.write_text(f'#!/usr/bin/env bash\nif [ "$1" = -c ]; then exec "{sys.executable}" "$@"; fi\necho "$*" >> "{calls}"\n')
    fake.chmod(0o755)
    bench, ext, p = tmp_path / "bench", tmp_path / "ext", "rad-dino_518"
    (bench / "predictions").mkdir(parents=True)
    (bench / "manifest.parquet").touch()
    paths = {"BENCH_OUT": str(bench), "EXT_OUT": str(ext), "PYTHON": str(fake)}
    res = _run(["m8"], tmp_path, **paths)
    assert res.returncode != 0 and "m8 needs m8_gpu's runs" in res.stderr
    res = _run(["m8_gpu"], tmp_path, **paths)
    assert res.returncode == 0, res.stderr
    ft = calls.read_text().splitlines()
    assert all(line.startswith("-m v2.finetune --model rad-dino --size 518 ") and "--milestone M8" in line for line in ft)
    assert ["--smoke" in line for line in ft] == [True, True, False, False, False]
    assert "--head attention --seed 1 --epochs 3 --max-hours 3 --resume" in ft[2] and "--save-embeddings" not in ft[2]
    assert "--head mean --seed 1 --epochs 3 --max-hours 3 --resume --save-embeddings" in ft[3]
    assert f"--head attention --seed 0 --epochs 6 --run {p}_ft_attn_e6_s0 --max-hours 6 --resume" in ft[4]

    calls.unlink()  # the Mac part, once the runs are home
    new = [f"{p}_ft_attn_s1", f"{p}_ft_s1", f"{p}_ft_attn_e6_s0"]
    texts = [f"{p}_ctx_indication", f"{p}_ctx_text_prior_labels", f"{p}_ctx_indication_bert"]
    for run in new + texts:
        (bench / "predictions" / f"{run}.parquet").touch()
    ext.mkdir()
    (ext / "manifest.parquet").touch()
    res = _run(["m8"], tmp_path, **paths)
    assert res.returncode == 0, res.stderr
    lines = calls.read_text().splitlines()
    words = lines[0].split()
    assert words[:2] == ["-m", "v2.stack"] and words[-2:] == ["--report", "m8_stack.md"]  # M5b's report stays whole
    pairs = {(words[i + 1], words[i + 2]) for i, w in enumerate(words) if w == "--pair"}
    assert pairs == {(img, text) for img in new for text in texts if (img, text) != (new[1], texts[2])}
    assert lines[1].startswith(f"-m v2.external chexpert --steps apply --bench {bench} --out {ext} "
                               f"--candidates {','.join(new)} --workers ")
    assert lines[2:] == [f"-m v2.replicate --bench {bench} --ext {ext}", f"-m v2.evaluate --all --out {bench}",
                         f"-m v2.evaluate --all --out {ext}"]

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
