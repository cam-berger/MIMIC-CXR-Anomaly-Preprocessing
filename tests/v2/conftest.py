"""Shared fixtures for the v2 tests: a synthetic MIMIC-CXR-JPG tree and its prepared outputs."""

import os
from pathlib import Path

import pytest

from v2 import prepare_data
from v2.synth import make_dataset

# A local .env may set COHORT_DIR; the tests choose their cohort files explicitly.
os.environ["COHORT_DIR"] = ""


def pytest_configure(config):
    config.addinivalue_line("markers", "network: downloads real models (set V2_TEST_NETWORK=1 to run)")


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("synth"), n_patients=200, seed=3)


@pytest.fixture(scope="session")
def prepared(synth, tmp_path_factory) -> Path:
    """A v2 output dir built from the synthetic tree (images at 48 px)."""
    out = tmp_path_factory.mktemp("v2out")
    prepare_data.main([
        "--mimic-cxr-jpg", str(synth.mimic_root), "--cohort-dir", str(synth.cohort_dir), "--out", str(out),
        "--image-size", "48", "--workers", "0", "--min-count", "3", "--rad-dino-list", str(synth.rad_dino_list),
    ])
    return out


@pytest.fixture(scope="session")
def pixel_embeddings(prepared) -> Path:
    from v2 import embed

    embed.main(["--model", "test-pixels", "--out", str(prepared), "--workers", "0"])
    return prepared / "embeddings" / "test-pixels_48.npz"


def network_enabled() -> bool:
    return os.environ.get("V2_TEST_NETWORK") == "1"
