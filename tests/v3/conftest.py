"""Shared fixtures for the v3 tests: a synthetic benchmark and v3's output dir on it (no MIMIC data, no network)."""

import os
import shutil

import pytest

# A local .env may set COHORT_DIR; the synthetic benchmark chooses its population explicitly.
os.environ["COHORT_DIR"] = ""


@pytest.fixture(scope="session")
def v3_base(tmp_path_factory):
    from v3 import smoke

    return smoke.setup_synthetic(tmp_path_factory.mktemp("v3"), n_patients=300)


@pytest.fixture
def v3out(v3_base, tmp_path):
    """A private copy of the set-up v3 dir (its links still point at the shared synthetic benchmark)."""
    out = tmp_path / "v3"
    shutil.copytree(v3_base, out, symlinks=True)
    return out
